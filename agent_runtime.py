#!/usr/bin/env python3
"""
SO-101 Agent Runtime: audio (Vosk) -> instruction router -> policy backend -> arm,
with full per-chunk Splunk telemetry.

Architecture (v1 core — option B):
  - Audio thread: Vosk wake-word listener; emits audio_event telemetry; on
    wake+instruction, stamps a trigger_id and dispatches to the router.
  - Instruction router: maps instruction text to a policy backend.
      * ScriptedPolicy — hardcoded choreography (e.g. "wave"); the only backend
        that works before fine-tuning exists.
      * GrootPolicyBackend — loads a LeRobot-format fine-tuned checkpoint
        (optional; enabled with --policy-path once one exists).
  - Arm layer: SO101Follower with clamped smooth interpolation.
  - Telemetry: per-chunk policy_step while the policy drives the arm, audio_event
    on wake/instruction, system_stats heartbeat 0.5s. All events carry trigger_id
    when part of a voice chain.

Usage:
  env -u PYTHONPATH SPLUNK_HEC_TOKEN=<token> \
    /home/vlad/miniconda3/envs/lerobot/bin/python agent_runtime.py --dry-run
  ... --test-instruction "wave"      # inject an instruction without audio
  ... --policy-path /path/to/finetuned_ckpt   # enable the GR00T backend
"""

import argparse
import json
import logging
import math
import os
import signal
import socket
import subprocess
import sys
import threading
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

logging.getLogger("lerobot").setLevel(logging.ERROR)
logging.getLogger().setLevel(logging.ERROR)

_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from lerobot_splunk import (  # noqa: E402
    SplunkHECSender,
    TelemetryEvent,
    RobotObservation,
    JetsonSystemStats,
    AudioEvent,
    PolicyStepStats,
)

MOTOR_NAMES = [
    "shoulder_pan", "shoulder_lift", "elbow_flex",
    "wrist_flex", "wrist_roll", "gripper",
]

# Hardware normalization range (use_degrees=False -> RANGE_M100_100 body, 0..100 gripper)
SAFE_RANGES = {
    "shoulder_pan":  (-100.0, 100.0),
    "shoulder_lift": (-100.0, 100.0),
    "elbow_flex":    (-100.0, 100.0),
    "wrist_flex":    (-100.0, 100.0),
    "wrist_roll":    (-100.0, 100.0),
    "gripper":       (0.0,    100.0),
}


def clamp_pose(pose: dict) -> dict:
    out = {}
    for motor, val in pose.items():
        lo, hi = SAFE_RANGES.get(motor, (-100.0, 100.0))
        out[motor] = max(lo, min(hi, float(val)))
    return out


# ── Arm ──────────────────────────────────────────────────────────────────
class ArmController:
    """Smooth interpolated control of the SO101 follower."""

    def __init__(self, robot, rate_hz: float = 30.0):
        self.robot = robot
        self.rate = rate_hz
        self.current: dict | None = None

    def read(self) -> dict:
        obs = self.robot.get_observation()
        pose = {m: float(obs[f"{m}.pos"]) for m in MOTOR_NAMES if f"{m}.pos" in obs}
        self.current = pose
        return pose

    def send(self, pose: dict):
        pose = clamp_pose(pose)
        self.current = pose
        self.robot.send_action({f"{m}.pos": v for m, v in pose.items()})

    def move_to(self, target: dict, duration_s: float):
        if self.robot is None or self.current is None:
            return
        start = dict(self.current)
        steps = max(1, int(duration_s * self.rate))
        for i in range(1, steps + 1):
            a = i / steps
            pose = {m: start[m] + (target.get(m, start[m]) - start[m]) * a
                    for m in MOTOR_NAMES if m in start}
            self.send(pose)
            time.sleep(1.0 / self.rate)

    def wave(self, base_pose: dict, amplitude: float = 18.0,
             cycles: float = 3.0, freq_hz: float = 2.2):
        """Claw wave: gripper opens, wrist_roll oscillates (sign-agnostic)."""
        if self.robot is None:
            return
        duration = cycles / freq_hz
        steps = max(1, int(duration * self.rate))
        roll_amp = max(amplitude, 18.0)
        grip_open = min(100.0, base_pose.get("gripper", 0.0) + 45.0)
        for i in range(steps):
            t = i / self.rate
            phase = 2 * math.pi * freq_hz * t
            envelope = min(1.0, t / 0.4, (duration - t) / 0.4)
            pose = dict(base_pose)
            grip_env = min(1.0, t / 0.4, max(0.0, (duration - t)) / 0.4)
            pose["gripper"] = base_pose.get("gripper", 0.0) + \
                (grip_open - base_pose.get("gripper", 0.0)) * grip_env
            pose["wrist_roll"] = base_pose.get("wrist_roll", 0.0) + \
                roll_amp * envelope * math.sin(phase)
            self.send(pose)
            time.sleep(1.0 / self.rate)
        self.move_to(base_pose, duration_s=0.4)


# ── Policy backends ──────────────────────────────────────────────────────
@dataclass
class PolicyResult:
    action_vector: list          # commanded values (post-processed)
    raw_action: list             # policy output before post-processing
    backend: str
    chunk_index: int
    chunk_step: int


class ScriptedPolicy:
    """Hardcoded choreography. The only backend that drives the SO101 before
    a fine-tuned checkpoint exists."""

    name = "scripted"

    def __init__(self, arm: ArmController | None, look_up_pose: dict | None):
        self.arm = arm
        self.base_pose = look_up_pose or {m: 0.0 for m in MOTOR_NAMES}
        self.chunk_index = 0

    def handles(self, instruction: str) -> bool:
        return instruction.strip().lower() in ("wave", "wave back", "say hi", "hello")

    def select_action(self, instruction: str) -> PolicyResult:
        """Executes the full scripted gesture, returns its final pose as the
        'action' (this backend drives the arm directly, chunk = gesture)."""
        self.chunk_index += 1
        if self.arm is not None and self.base_pose:
            self.arm.wave(self.base_pose)
        return PolicyResult(
            action_vector=[self.base_pose.get(m, 0.0) for m in MOTOR_NAMES],
            raw_action=[],
            backend=self.name,
            chunk_index=self.chunk_index,
            chunk_step=0,
        )


class GrootPolicyBackend:
    """GR00T N1.7 fine-tuned checkpoint (LeRobot format) driving the SO101.

    Requires --policy-path pointing at a checkpoint fine-tuned on SO101 data
    (embodiment new_embodiment). Zero-shot GR00T cannot drive this arm.
    """

    name = "gr00t_n1_7"

    def __init__(self, checkpoint: str, device: str = "cuda",
                 camera_indices: dict | None = None, logger=print):
        import torch  # noqa: F401
        self.checkpoint = checkpoint
        self.device = device
        self.logger = logger
        self.camera_indices = camera_indices or {"up": 2, "wrist": 0}
        self.policy = None
        self.pre = None
        self.post = None
        self.captures = {}
        self.chunk_index = 0
        self._queue_step = 0

    def _load(self):
        import torch
        import cv2
        from lerobot.configs.policies import PreTrainedConfig
        from lerobot.policies.factory import make_policy, make_pre_post_processors

        cfg = PreTrainedConfig.from_pretrained(self.checkpoint)
        cfg.device = self.device
        if getattr(cfg, "base_model_path", None) and \
                not Path(str(cfg.base_model_path)).is_dir():
            self.logger("checkpoint references a missing base model path — "
                        "set policy.base_model_path to the local GR00T snapshot")
            raise FileNotFoundError(cfg.base_model_path)

        self.policy = make_policy(cfg)
        self.policy.eval()
        self.pre, self.post = make_pre_post_processors(
            cfg, pretrained_path=self.checkpoint)

        for name, idx in self.camera_indices.items():
            cap = cv2.VideoCapture(idx, cv2.CAP_V4L2)
            cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            ok, frame = cap.read()
            if not ok:
                cap.release()
                raise RuntimeError(f"camera {idx} ({name}) failed to start")
            self.captures[name] = cap
            self.logger(f"camera '{name}' (/dev/video{idx}) OK {frame.shape[1]}x{frame.shape[0]}")

    def handles(self, instruction: str) -> bool:
        return self.policy is not None  # GR00T handles any instruction it was trained on

    def select_action(self, instruction: str) -> PolicyResult:
        import torch
        if self.policy is None:
            self._load()
        # Observation from live cameras + current joint state
        batch = {"task": [instruction.strip()]}
        for name, cap in self.captures.items():
            ok, frame = cap.read()
            if not ok or frame is None:
                raise RuntimeError(f"camera '{name}' returned no frame")
            import cv2
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            batch[f"observation.images.{name}"] = \
                torch.from_numpy(rgb).permute(2, 0, 1).float().unsqueeze(0)
        # joints fed by the runtime via set_state()
        batch["observation.state"] = torch.tensor(
            np.array([self._state]), dtype=torch.float32).unsqueeze(0)

        t0 = time.time()
        proc = self.pre(batch)
        out = self.policy.select_action(proc)
        act = self.post(out)
        latency_ms = (time.time() - t0) * 1000.0

        if not torch.is_tensor(act):
            act = next(iter(act.values()))
        raw = np.asarray(act.cpu()).squeeze().astype(np.float32).reshape(-1)
        self._queue_step = (self._queue_step + 1) % getattr(self.policy.config, "chunk_size", 1)
        return PolicyResult(
            action_vector=raw[:6].tolist(),
            raw_action=raw.tolist(),
            backend=self.name,
            chunk_index=self.chunk_index,
            chunk_step=self._queue_step,
        )

    _state = [0.0] * 6

    def set_state(self, joints: dict):
        self._state = [float(joints.get(m, 0.0)) for m in MOTOR_NAMES]


# ── Audio (Vosk wake word) ───────────────────────────────────────────────
class WakeWordAudio:
    """Vosk wake-word listener with instruction extraction. Falls back to a
    sound-level trigger if vosk/model unavailable."""

    def __init__(self, device: str, wake_word: str, model_dir: str | None,
                 emit, logger=print, refractory_s: float = 6.0):
        self.device = device
        self.wake_word = wake_word.lower().strip()
        self.keyword = self.wake_word.split()[-1]
        self.model_dir = model_dir
        self.emit = emit
        self.logger = logger
        self.refractory_s = refractory_s
        self.rate = 16000
        self.chunk_bytes = self.rate * 2 * 2 // 10  # 100ms stereo s16
        self.threshold = 120.0
        self.ambient_rms = 30.0
        self.speech = threading.Event()
        self.instruction = ""
        self.trigger_id = ""
        self.rms = 0.0
        self._proc = None
        self._running = False

    def _is_gated_ok(self) -> bool:
        return self.model_dir and Path(self.model_dir).exists()

    def start(self):
        self._running = True
        self._proc = subprocess.Popen(
            ["arecord", "-D", self.device, "-f", "S16_LE", "-r", str(self.rate),
             "-c", "2", "-t", "raw", "-q"],
            stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
        mode = "vosk" if self.model_dir else "sound-level"
        self._thread = threading.Thread(target=self._loop_vosk if self.model_dir
                                        else self._loop_rms, daemon=True,
                                        name=f"audio-{mode}")
        self._thread.start()
        self.logger(f"[audio] {mode} mode, wake word '{self.wake_word}'")

    def _loop_vosk(self):
        from vosk import Model, KaldiRecognizer
        import difflib
        model = Model(self.model_dir)
        rec = KaldiRecognizer(model, self.rate)
        last_hit = 0.0
        while self._running:
            data = self._proc.stdout.read(self.chunk_bytes)
            if not data or len(data) < self.chunk_bytes:
                time.sleep(0.01)
                continue
            samples = np.frombuffer(data, dtype=np.int16).astype(np.float32) \
                .reshape(-1, 2).mean(axis=1)
            self.rms = float(np.sqrt((samples ** 2).mean()))
            if rec.AcceptWaveform(samples.tobytes()):
                text = json.loads(rec.Result()).get("text", "")
            else:
                text = json.loads(rec.PartialResult()).get("partial", "")
            if not text:
                continue
            now = time.monotonic()
            words = text.lower().split()
            hit = any(difflib.SequenceMatcher(None, w, self.keyword).ratio() >= 0.75
                      for w in words)
            if hit and now - last_hit >= self.refractory_s:
                last_hit = now
                # instruction = words after the wake keyword
                kw_idx = max((i for i, w in enumerate(words)
                              if difflib.SequenceMatcher(None, w, self.keyword).ratio() >= 0.75),
                             default=-1)
                instruction = " ".join(words[kw_idx + 1:]).strip(" ,.!?")
                self.trigger_id = str(uuid.uuid4())
                self.instruction = instruction
                self.emit("audio_event", AudioEvent(
                    stage="wake_word", transcript=text,
                    matched_phrase=self.wake_word, match_ratio=0.75,
                    rms=self.rms, threshold=self.threshold,
                    device=self.device), self.trigger_id)
                self.speech.set()

    def _loop_rms(self):
        streak = 0
        last_hit = 0.0
        while self._running:
            data = self._proc.stdout.read(self.chunk_bytes)
            if not data or len(data) < self.chunk_bytes:
                time.sleep(0.01)
                continue
            samples = np.frombuffer(data, dtype=np.int16).astype(np.float32) \
                .reshape(-1, 2).mean(axis=1)
            self.rms = float(np.sqrt((samples ** 2).mean()))
            now = time.monotonic()
            if self.rms >= self.threshold:
                streak += 1
                if streak >= 2 and now - last_hit >= self.refractory_s:
                    last_hit = now
                    streak = 0
                    self.trigger_id = str(uuid.uuid4())
                    self.instruction = ""  # sound-level has no text
                    self.emit("audio_event", AudioEvent(
                        stage="wake_word", transcript="(sound trigger)",
                        matched_phrase="(level)", rms=self.rms,
                        threshold=self.threshold, device=self.device),
                        self.trigger_id)
                    self.speech.set()
            else:
                streak = 0

    def stop(self):
        self._running = False
        if self._proc:
            self._proc.terminate()


# ── Runtime ──────────────────────────────────────────────────────────────
class AgentRuntime:
    def __init__(self, args):
        self.args = args
        self.running = True
        self.session_id = f"agent_{int(time.time())}"
        self.host = socket.gethostname()
        self.trigger_id = None

        self.sender = SplunkHECSender()
        self._emit_lock = threading.Lock()

        self.robot = None
        self.arm = None
        self.rest_pose = None
        self.look_up_pose = None

        self.backends = []
        self.groot = None

    # -- telemetry helpers -------------------------------------------------
    def _emit(self, event_type: str, *, audio: AudioEvent | None = None,
              policy: PolicyStepStats | None = None,
              robot: RobotObservation | None = None,
              system: JetsonSystemStats | None = None,
              raw_message: str | None = None):
        now = datetime.now(timezone.utc)
        ev = TelemetryEvent(
            timestamp_utc=now.isoformat(),
            epoch_ms=int(now.timestamp() * 1000),
            event_type=event_type,
            session_id=self.session_id,
            host=self.host,
            audio=audio,
            policy=policy,
            robot=robot,
            system=system,
            raw_message=raw_message,
            trigger_id=self.trigger_id,
        )
        self.sender.enqueue(ev)

    def _system_stats_loop(self):
        import psutil
        while self.running:
            try:
                mem = psutil.virtual_memory()
                proc = psutil.Process(os.getpid())
                sys_stats = JetsonSystemStats(
                    cpu_percent=psutil.cpu_percent(interval=None),
                    cpu_per_core=psutil.cpu_percent(interval=None, percpu=True),
                    ram_used_mb=round(mem.used / 1024 ** 2, 1),
                    ram_total_mb=round(mem.total / 1024 ** 2, 1),
                    ram_percent=round(mem.percent, 1),
                    swap_used_mb=round(psutil.swap_memory().used / 1024 ** 2, 1),
                    gpu_util_percent=None, gpu_freq_mhz=None,
                    gpu_mem_used_mb=None, gpu_mem_total_mb=None,
                    cpu_temp_c=None, gpu_temp_c=None, soc_temp_c=None,
                    tj_temp_c=None, power_cpu_gpu_mw=None, power_soc_mw=None,
                    power_total_mw=None, fan_percent=None, jetson_clocks=None,
                    nvp_model=None, emc_util_percent=None, cpu_freq_mhz=None,
                    process_cpu_percent=proc.cpu_percent(interval=None),
                    process_mem_mb=round(proc.memory_info().rss / 1024 ** 2, 1),
                )
                self._emit("system_stats", system=sys_stats)
            except Exception as e:
                print(f"[stats] error: {e}")
            time.sleep(0.5)

    # -- setup --------------------------------------------------------------
    def setup(self):
        args = self.args

        if not args.dry_run:
            from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig
            cfg = SO101FollowerConfig(
                port=args.port, id=args.robot_id, cameras={},
                use_degrees=False, max_relative_target=10.0,
                disable_torque_on_disconnect=True,
            )
            self.robot = SO101Follower(cfg)
            self.robot.connect(calibrate=False)
            self.arm = ArmController(self.robot)
            self.rest_pose = self.arm.read()
            print("[arm] connected, rest pose: "
                  + ", ".join(f"{m}={self.rest_pose[m]:.1f}" for m in MOTOR_NAMES
                              if m in self.rest_pose))
            # Look-up pose: absolute targets from the user-posed reference
            self.look_up_pose = clamp_pose({
                **self.rest_pose,
                "shoulder_lift": args.look_up_lift_abs,
                "elbow_flex": args.look_up_elbow_abs,
            })
            self.arm.move_to(self.look_up_pose, duration_s=2.0)
            self._emit("look_up", raw_message="arm at look-up pose")
        else:
            print("[dry-run] arm disabled")
            self.look_up_pose = {m: 0.0 for m in MOTOR_NAMES}

        # Backends: scripted always; GR00T only when a checkpoint is configured
        self.backends.append(ScriptedPolicy(self.arm, self.look_up_pose))
        if args.policy_path:
            try:
                self.groot = GrootPolicyBackend(
                    args.policy_path, device="cuda",
                    camera_indices={"up": args.camera_up, "wrist": args.camera_wrist},
                    logger=lambda m: print(f"[gr00t] {m}"))
                self.backends.append(self.groot)
                print("[gr00t] backend enabled (loads on first instruction)")
            except Exception as e:
                print(f"[gr00t] backend unavailable: {e}")

        threading.Thread(target=self._system_stats_loop, daemon=True,
                         name="system-stats").start()

    # -- instruction handling ------------------------------------------------
    def handle_instruction(self, instruction: str, audio_event: AudioEvent | None):
        """Route instruction -> backend -> execute -> telemetry."""
        instruction = (instruction or "").strip()
        chosen = None
        for backend in reversed(self.backends):  # GR00T (if loaded) wins over scripted
            if backend.handles(instruction):
                chosen = backend
                break

        self.trigger_id = self.trigger_id or str(uuid.uuid4())
        self._emit("instruction", audio=audio_event,
                   raw_message=f"instruction='{instruction}' backend="
                               f"{chosen.name if chosen else 'none'}")

        if chosen is None:
            print(f"[router] no backend for '{instruction}' — ignoring (decision 1)")
            self.trigger_id = None
            return

        print(f"[router] '{instruction}' -> {chosen.name}")
        if isinstance(chosen, GrootPolicyBackend):
            chosen.set_state(self.arm.read() if self.arm else {})

        t0 = time.time()
        result = chosen.select_action(instruction)
        latency_ms = round((time.time() - t0) * 1000.0, 2)

        # Per-chunk telemetry with robot observation
        joints = self.arm.read() if self.arm else {}
        robot = RobotObservation(
            episode=0, step=result.chunk_index,
            joint_positions={m: round(joints.get(m, 0.0), 4) for m in MOTOR_NAMES if m in joints},
            gripper_position=joints.get("gripper"),
            action_vector=result.action_vector,
            inference_latency_ms=latency_ms,
            policy_fps=round(1000.0 / latency_ms, 2) if latency_ms > 0 else None,
            joint_violations={},
        )
        self._emit("policy_step",
                   policy=PolicyStepStats(
                       policy_backend=result.backend,
                       checkpoint=self.args.policy_path or "scripted",
                       embodiment_tag="new_embodiment" if result.backend == "gr00t_n1_7" else "",
                       task_instruction=instruction,
                       chunk_index=result.chunk_index,
                       chunk_step=result.chunk_step,
                       raw_action=result.raw_action,
                       commanded_action=result.action_vector),
                   robot=robot)
        self.trigger_id = None  # chain complete

    # -- main loop ------------------------------------------------------------
    def run(self):
        args = self.args

        audio = None
        if not args.no_audio:
            audio = WakeWordAudio(
                device=args.mic_device, wake_word=args.wake_word,
                model_dir=(str(_SCRIPT_DIR / "models" / "vosk-model-small-en-us-0.15")
                           if not args.no_vosk else None),
                emit=lambda etype, aevt, tid: (setattr(self, "trigger_id", tid),
                                               self._emit(etype, audio=aevt,
                                                          raw_message=aevt.stage)),
                logger=print)
            audio.start()

        def shutdown(signum, _):
            self.running = False

        signal.signal(signal.SIGINT, shutdown)
        signal.signal(signal.SIGTERM, shutdown)

        print(f"[runtime] live — say '{args.wake_word} <instruction>' (Ctrl+C to stop)")
        try:
            while self.running:
                if audio and audio.speech.is_set():
                    audio.speech.clear()
                    self.trigger_id = audio.trigger_id
                    aevt = AudioEvent(stage="instruction", transcript=audio.instruction,
                                      matched_phrase=args.wake_word,
                                      rms=audio.rms, threshold=audio.threshold,
                                      device=args.mic_device)
                    self.handle_instruction(audio.instruction, aevt)
                time.sleep(0.05)
        finally:
            print("[runtime] shutting down ...")
            if audio:
                audio.stop()
            if self.arm is not None and self.rest_pose is not None:
                try:
                    self.arm.move_to(self.rest_pose, duration_s=2.0)
                except Exception as e:
                    print(f"[arm] return-home error: {e}")
                try:
                    self.robot.disconnect()
                    print("[arm] disconnected")
                except Exception as e:
                    print(f"[arm] disconnect error: {e}")
            print(f"[splunk] stats: {self.sender.stats}")


def parse_args():
    ap = argparse.ArgumentParser(description="SO-101 agent runtime (audio -> policy -> arm)")
    ap.add_argument("--port", default="/dev/ttyACM0")
    ap.add_argument("--robot-id", default="vlad-lerobot")
    ap.add_argument("--dry-run", action="store_true", help="no arm connection")
    ap.add_argument("--no-audio", action="store_true", help="disable mic listener")
    ap.add_argument("--no-vosk", action="store_true", help="force sound-level trigger")
    ap.add_argument("--wake-word", default="hey max")
    ap.add_argument("--mic-device", default="hw:CARD=BRIO,DEV=0")
    ap.add_argument("--test-instruction", default=None,
                    help="inject one instruction at startup (no audio needed)")
    ap.add_argument("--policy-path", default=None,
                    help="fine-tuned GR00T checkpoint dir (enables GR00T backend)")
    ap.add_argument("--camera-up", type=int, default=2)
    ap.add_argument("--camera-wrist", type=int, default=0)
    ap.add_argument("--look-up-lift-abs", type=float, default=-95.0)
    ap.add_argument("--look-up-elbow-abs", type=float, default=95.0)
    return ap.parse_args()


def main():
    args = parse_args()
    runtime = AgentRuntime(args)
    runtime.setup()

    if args.test_instruction:
        threading.Thread(target=lambda: (time.sleep(2.0),
                                         runtime.handle_instruction(
                                             args.test_instruction, None)),
                         daemon=True).start()

    runtime.run()


if __name__ == "__main__":
    main()
