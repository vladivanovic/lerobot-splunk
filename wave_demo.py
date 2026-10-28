#!/usr/bin/env python3
"""
SO-101 "look up & wave back" demo.

Behavior loop:
  1. Connect SO101 follower (LeRobot API), read current pose.
  2. Smoothly move to a "look up at you" pose (relative deltas, unit-safe).
  3. Watch the camera: detect a human wave (oscillating motion blob).
  4. On detection: perform a scripted wave, then return to look-up pose.
  5. Repeat until Ctrl+C. On exit: ease back to the startup pose.

Run with the conda env that has lerobot + torch:
  /home/vlad/miniconda3/envs/lerobot/bin/python wave_demo.py

Dry run (camera + detection only, no robot):
  /home/vlad/miniconda3/envs/lerobot/bin/python wave_demo.py --dry-run
"""

import argparse
import json
import logging
import math
import signal
import socket
import subprocess
import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

# Silence LeRobot's per-command clamp warnings (they flood the console during waves)
logging.getLogger("lerobot").setLevel(logging.ERROR)
logging.getLogger().setLevel(logging.ERROR)

# ── LeRobot imports (available in the conda `lerobot` env) ─────────────
from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig

# ── SO-101 motor setup ──────────────────────────────────────────────────
MOTOR_NAMES = [
    "shoulder_pan", "shoulder_lift", "elbow_flex",
    "wrist_flex", "wrist_roll", "gripper",
]

# Normalized hardware range (use_degrees=False -> RANGE_M100_100 body, 0..100 gripper).
# Clamping here only prevents out-of-range commands; per-command jumps are limited
# separately via max_relative_target in the robot config.
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


# ── Wave detector ────────────────────────────────────────────────────────
@dataclass
class WaveDetector:
    """Detects a human wave: an oscillating motion blob in the upper ROI.

    Frame differencing -> threshold -> largest blob in ROI -> track centroid x.
    A wave = enough horizontal direction reversals of the blob within a window.

    Robustness notes:
    - Track is NOT cleared on a single low-energy frame (a waving hand decelerates
      at each swing extreme, producing energy dips). We only reset after a
      sustained quiet period (grace_sec).
    - min_motion_energy is auto-calibrated from observed idle energy at startup
      (set_idle_baseline) — scenes differ wildly in brightness.
    """
    roi_top_frac: float = 0.0     # top of ROI (fraction of height)
    roi_bottom_frac: float = 0.66 # bottom of ROI
    diff_threshold: int = 18      # pixel-intensity delta to count as motion
    min_blob_area: int = 150      # ignore tiny specks (px at 640-wide)
    window_sec: float = 2.5       # reversal window
    min_reversals: int = 3        # direction changes needed
    min_amplitude_px: float = 22  # total horizontal travel needed
    min_motion_energy: float = 2.0  # overridable via set_idle_baseline()
    grace_sec: float = 0.5        # low-energy grace before track reset
    cooldown_sec: float = 4.0

    _prev_gray: np.ndarray | None = field(default=None, repr=False)
    _track: list = field(default_factory=list, repr=False)  # (t, x, energy)
    _last_trigger: float = field(default=-1e9, repr=False)
    _last_motion: float = field(default=0.0, repr=False)    # last frame with motion
    last_energy: float = 0.0
    last_reversals: int = 0

    def set_idle_baseline(self, idle_mean_energy: float):
        """Call after sampling ~1.5s of stillness: set the motion gate relative
        to this scene's idle noise floor."""
        self.min_motion_energy = max(1.5, idle_mean_energy * 2.5)
        print(f"[det] idle energy={idle_mean_energy:.2f} -> "
              f"motion gate={self.min_motion_energy:.2f}")

    def process(self, frame_bgr: np.ndarray) -> tuple[bool, float]:
        """Returns (wave_detected, motion_energy)."""
        now = time.monotonic()
        h, w = frame_bgr.shape[:2]
        y0, y1 = int(h * self.roi_top_frac), int(h * self.roi_bottom_frac)
        x0, x1 = int(w * 0.05), int(w * 0.95)

        roi = cv2.cvtColor(frame_bgr[y0:y1, x0:x1], cv2.COLOR_BGR2GRAY)
        roi = cv2.GaussianBlur(roi, (5, 5), 0)

        if self._prev_gray is None or self._prev_gray.shape != roi.shape:
            self._prev_gray = roi
            return False, 0.0

        diff = cv2.absdiff(roi, self._prev_gray)
        self._prev_gray = roi
        energy = float(diff.mean())
        self.last_energy = energy

        if energy < self.min_motion_energy:
            # Grace period: only reset tracking after sustained stillness,
            # so deceleration dips mid-wave don't wipe the reversal count.
            if now - self._last_motion > self.grace_sec:
                self._track.clear()
                self.last_reversals = 0
            return False, energy
        self._last_motion = now

        _, mask = cv2.threshold(diff, self.diff_threshold, 255, cv2.THRESH_BINARY)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        blob_x = None
        if contours:
            biggest = max(contours, key=cv2.contourArea)
            area = cv2.contourArea(biggest)
            if area >= self.min_blob_area:
                m = cv2.moments(biggest)
                if m["m00"] > 0:
                    blob_x = m["m10"] / m["m00"]

        if blob_x is None:
            return False, energy

        self._track.append((now, blob_x, energy))
        horizon = now - self.window_sec
        self._track = [t for t in self._track if t[0] >= horizon]

        # Cooldown check first (after a trigger, ignore new ones for a while)
        if now - self._last_trigger < self.cooldown_sec:
            return False, energy

        pts = self._track
        if len(pts) >= 6:
            xs = [p[1] for p in pts]
            span = max(xs) - min(xs)
            reversals = 0
            for i in range(2, len(xs)):
                d1, d2 = xs[i - 1] - xs[i - 2], xs[i] - xs[i - 1]
                if d1 * d2 < 0 and abs(d1) > 2 and abs(d2) > 2:
                    reversals += 1
            self.last_reversals = reversals
            if reversals >= self.min_reversals and span >= self.min_amplitude_px:
                self._last_trigger = now
                self._track.clear()
                self.last_reversals = 0
                return True, energy

        return False, energy


# ── Robot motion helpers ────────────────────────────────────────────────
class ArmController:
    """Thin wrapper: smooth relative moves with clamping, at loop rate."""

    def __init__(self, robot: SO101Follower | None, rate_hz: float = 30.0):
        self.robot = robot
        self.rate = rate_hz
        self.current: dict | None = None  # last commanded/observed pose

    def read(self) -> dict:
        obs = self.robot.get_observation()
        pose = {m: float(obs[f"{m}.pos"]) for m in MOTOR_NAMES if f"{m}.pos" in obs}
        self.current = pose
        return pose

    def send(self, pose: dict):
        pose = clamp_pose(pose)
        self.current = pose
        self.robot.send_action({f"{m}.pos": v for m, v in pose.items()})

    def move_to(self, target: dict, duration_s: float, on_frame=None):
        """Linear interpolation from current pose to target over duration."""
        if self.robot is None or self.current is None:
            return
        start = dict(self.current)
        steps = max(1, int(duration_s * self.rate))
        for i in range(1, steps + 1):
            a = i / steps
            pose = {m: start[m] + (target.get(m, start[m]) - start[m]) * a
                    for m in MOTOR_NAMES if m in start}
            self.send(pose)
            if on_frame:
                on_frame()
            time.sleep(1.0 / self.rate)

    def wave(self, base_pose: dict, amplitude: float = 18.0,
             cycles: float = 3.0, freq_hz: float = 2.2, on_frame=None,
             style: str = "claw"):
        """Waving gesture.

        style="claw": open the gripper and oscillate wrist_roll (palm turning
        side to side) — reads as a hand wave, no body shake.
        style="classic": wrist_flex sine with a gentle shoulder_pan sway.
        """
        if self.robot is None:
            return
        duration = cycles / freq_hz
        steps = max(1, int(duration * self.rate))
        roll_amp = max(amplitude, 18.0)
        grip_open = min(100.0, base_pose.get("gripper", 0.0) + 45.0)

        for i in range(steps):
            t = i / self.rate
            phase = 2 * math.pi * freq_hz * t
            envelope = min(1.0, t / 0.4, (duration - t) / 0.4)  # ease in/out
            pose = dict(base_pose)
            if style == "claw":
                # Grip opens during the ease-in, closes during ease-out
                grip_env = min(1.0, t / 0.4, max(0.0, (duration - t)) / 0.4)
                pose["gripper"] = base_pose.get("gripper", 0.0) + \
                    (grip_open - base_pose.get("gripper", 0.0)) * grip_env
                pose["wrist_roll"] = base_pose.get("wrist_roll", 0.0) + \
                    roll_amp * envelope * math.sin(phase)
            else:  # classic
                pose["wrist_flex"] = base_pose.get("wrist_flex", 0.0) + \
                    amplitude * envelope * math.sin(phase)
                pose["shoulder_pan"] = base_pose.get("shoulder_pan", 0.0) + \
                    0.45 * amplitude * envelope * math.sin(phase)
            self.send(pose)
            if on_frame:
                on_frame()
            time.sleep(1.0 / self.rate)

        # Ensure gripper returns to base after the gesture
        self.move_to(base_pose, duration_s=0.4, on_frame=on_frame)


# ── Camera ───────────────────────────────────────────────────────────────
class Camera:
    def __init__(self, index: int, width: int = 640, height: int = 360, fps: int = 30,
                 exposure_time: int | None = None, brightness: int | None = None,
                 auto_exposure: float | None = None):
        """exposure_time: driver-level exposure_time_absolute via v4l2-ctl (the
        OpenCV prop maps to different units on this camera and saturates — use
        this instead for the Arducam, 30-60 works indoors with IR)."""
        if exposure_time is not None:
            import subprocess
            subprocess.run(
                ["v4l2-ctl", "-d", f"/dev/video{index}",
                 "--set-ctrl", "auto_exposure=1",
                 "--set-ctrl", f"exposure_time_absolute={int(exposure_time)}"],
                check=False, capture_output=True)
        self.cap = cv2.VideoCapture(index, cv2.CAP_V4L2)
        self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.cap.set(cv2.CAP_PROP_FPS, fps)
        if auto_exposure is not None:
            self.cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, auto_exposure)
        if brightness is not None:
            self.cap.set(cv2.CAP_PROP_BRIGHTNESS, brightness)
        ok, frame = self.cap.read()
        if not ok or frame is None:
            raise RuntimeError(f"Camera {index} failed to deliver frames")
        print(f"[cam] /dev/video{index} OK {frame.shape[1]}x{frame.shape[0]} "
              f"brightness={frame.mean():.0f}")

    def read(self):
        ok, frame = self.cap.read()
        return frame if ok else None

    def release(self):
        self.cap.release()


# ── Microphone voice-activity detection ─────────────────────────────────
class MicVAD:
    """Background thread reading a USB mic via arecord; flags speech onsets.

    Auto-calibrates an ambient noise floor at start(); speech = RMS sustained
    over threshold for a few chunks, with a refractory period after each hit.
    """

    def __init__(self, device: str = "hw:CARD=BRIO,DEV=0", rate: int = 16000,
                 chunk_ms: int = 100, factor: float = 3.0,
                 min_rms: int = 120, refractory_s: float = 4.0):
        self.device = device
        self.rate = rate
        self.chunk_bytes = rate * 2 * 2 * chunk_ms // 1000  # stereo s16
        self.factor = factor
        self.min_rms = min_rms          # absolute floor (device-dependent)
        self.refractory_s = refractory_s
        self.ambient_rms = 30.0
        self.threshold = float(min_rms)  # set properly in start(); init early to avoid race
        self.speech_event = threading.Event()
        self.rms = 0.0
        self._proc = None
        self._thread = None
        self._running = False

    def start(self):
        self._running = True
        self._proc = subprocess.Popen(
            ["arecord", "-D", self.device, "-f", "S16_LE", "-r", str(self.rate),
             "-c", "2", "-t", "raw", "-q"],
            stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
        self._thread = threading.Thread(target=self._loop, daemon=True,
                                        name="mic-vad")
        self._thread.start()
        # Calibrate ambient noise floor from the first ~1s
        samples = []
        t_end = time.monotonic() + 1.0
        last = None
        while time.monotonic() < t_end:
            if last is not None and last != self.rms:
                samples.append(last)
            last = self.rms
            time.sleep(0.05)
        if samples:
            self.ambient_rms = float(np.median(samples))
        self.threshold = max(self.min_rms, self.ambient_rms * self.factor)
        print(f"[mic] ambient={self.ambient_rms:.0f} -> speech threshold={self.threshold:.0f}")

    def _loop(self):
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
                    self.speech_event.set()
            else:
                streak = 0

    def stop(self):
        self._running = False
        if self._proc:
            self._proc.terminate()


# ── Wake-word listener (Vosk offline speech recognition) ────────────────
class WakeWordListener:
    """Listens for a spoken wake phrase (e.g. "Hey Max") using Vosk.

    Streams the mic via arecord, downmixes stereo->mono, and fuzzy-matches the
    phrase in Vosk's partial transcripts ("max"/"macks"/"hey max" all count).
    Falls back gracefully if vosk or the model are unavailable (returns False
    from setup(), caller keeps the plain RMS VAD instead).
    """

    def __init__(self, model_dir: str, wake_word: str = "hey max",
                 device: str = "hw:CARD=BRIO,DEV=0", rate: int = 16000,
                 refractory_s: float = 6.0, match_ratio: float = 0.75):
        from vosk import Model, KaldiRecognizer  # noqa: PLC0415
        self.kaldi_cls = KaldiRecognizer
        self.model = Model(model_dir)
        self.wake_word = wake_word.lower().strip()
        self.wake_tokens = self.wake_word.split()
        self.keyword = self.wake_tokens[-1]  # "max"
        self.device = device
        self.rate = rate
        self.refractory_s = refractory_s
        self.match_ratio = match_ratio
        self.speech_event = threading.Event()
        self.rms = 0.0
        self.last_text = ""
        self._proc = None
        self._thread = None
        self._running = False

    def setup(self) -> bool:
        """Verify model + mic; prints status. Call before start()."""
        try:
            rec = self.kaldi_cls(self.model, self.rate)
            rec.Reset()
        except Exception as e:
            print(f"[wake] vosk setup failed: {e}")
            return False
        print(f"[wake] listening for '{self.wake_word}' "
              f"(fuzzy match ≥{self.match_ratio:.2f})")
        return True

    @staticmethod
    def _fuzzy(a: str, b: str) -> float:
        import difflib  # noqa: PLC0415
        return difflib.SequenceMatcher(None, a, b).ratio()

    def _matched(self, text: str) -> bool:
        text = text.lower().strip()
        if not text:
            return False
        if self._fuzzy(text, self.wake_word) >= max(0.55, self.match_ratio - 0.2):
            return True
        for token in text.split():
            if self._fuzzy(token, self.keyword) >= self.match_ratio:
                return True
        return False

    def start(self):
        self._running = True
        self._proc = subprocess.Popen(
            ["arecord", "-D", self.device, "-f", "S16_LE", "-r", str(self.rate),
             "-c", "2", "-t", "raw", "-q"],
            stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
        self._thread = threading.Thread(target=self._loop, daemon=True,
                                        name="wake-word")
        self._thread.start()

    def _loop(self):
        rec = self.kaldi_cls(self.model, self.rate)
        last_hit = 0.0
        chunk = self.rate * 2 * 2 // 10  # 100ms stereo s16
        while self._running:
            data = self._proc.stdout.read(chunk)
            if not data or len(data) < chunk:
                time.sleep(0.01)
                continue
            samples = np.frombuffer(data, dtype=np.int16).astype(np.float32) \
                .reshape(-1, 2).mean(axis=1)
            self.rms = float(np.sqrt((samples ** 2).mean()))
            if rec.AcceptWaveform(samples.tobytes()):
                text = (json.loads(rec.Result()).get("text", "") or "")
            else:
                text = (json.loads(rec.PartialResult()).get("partial", "") or "")
            if text:
                self.last_text = text
                now = time.monotonic()
                if now - last_hit >= self.refractory_s and self._matched(text):
                    last_hit = now
                    print(f"[wake] heard: '{text}'")
                    self.speech_event.set()

    def stop(self):
        self._running = False
        if self._proc:
            self._proc.terminate()


# ── Face finding (YuNet DNN) ────────────────────────────────────────────
class FaceFinder:
    """Wraps cv2.FaceDetectorYN and steers shoulder_pan until a face is centered."""

    def __init__(self, model_path: str, cam_width: int = 640):
        self.detector = cv2.FaceDetectorYN_create(model_path, "", (cam_width, 360))
        self.detector.setScoreThreshold(0.55)
        self.center_x = cam_width / 2.0

    def find(self, frame_bgr) -> tuple[float, float] | None:
        """Returns (face_center_x, face_score) of the best face, or None."""
        h, w = frame_bgr.shape[:2]
        self.detector.setInputSize((w, h))
        n, faces = self.detector.detect(frame_bgr)
        if faces is None or len(faces) == 0:
            return None
        best = max(faces, key=lambda f: f[-1])  # highest score
        x = float(best[0]) + float(best[2]) / 2.0
        return x, float(best[-1])

    def aim(self, arm: "ArmController", base_pose: dict, cam,
            sweep_deg: float = 35.0, max_scan_s: float = 6.0) -> bool:
        """Sweep pan while looking for a face; steer toward it. Returns True if found."""
        start_pan = base_pose.get("shoulder_pan", 0.0)
        search = [(start_pan + d) for d in
                  (0, -sweep_deg / 2, sweep_deg / 2, -sweep_deg, sweep_deg, 0)]
        deadline = time.monotonic() + max_scan_s
        tries = 0
        for target_pan in search:
            if time.monotonic() > deadline:
                break
            pose = {**base_pose, "shoulder_pan": target_pan}
            arm.move_to(pose, duration_s=0.5)
            # hold briefly, sampling frames for faces
            t_hold = time.monotonic() + 0.7
            while time.monotonic() < t_hold:
                frame = cam.read()
                if frame is None:
                    continue
                hit = self.find(frame)
                if hit is not None:
                    fx, score = hit
                    tries = 0
                    # proportional steer toward center
                    for _ in range(4):
                        frame = cam.read()
                        if frame is None:
                            continue
                        hit = self.find(frame)
                        if hit is None:
                            break
                        fx, _ = hit
                        offset = fx - self.center_x
                        if abs(offset) < 55:
                            print(f"[find] face locked (offset={offset:.0f}px)")
                            return True
                        step = max(-8.0, min(8.0, -offset / 40.0))
                        cur = arm.current.get("shoulder_pan", target_pan)
                        arm.move_to({**base_pose, "shoulder_pan": cur + step},
                                    duration_s=0.25)
                    print(f"[find] face seen (score={score:.2f}) — tracking")
                    return True
        print("[find] no face found — holding look-up pose")
        return False


# ── Main ─────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser(description="SO-101 look-up & wave-back demo")
    ap.add_argument("--port", default="/dev/ttyACM0")
    ap.add_argument("--robot-id", default="vlad-lerobot",
                    help="must match the calibration file name "
                         "(~/.cache/huggingface/lerobot/calibration/robots/so_follower/<id>.json)")
    ap.add_argument("--camera-index", type=int, default=0,
                    help="0=Arducam UC-A53 wristcam (default — faces you when wrist is up), "
                         "2=Logicool BRIO top view")
    ap.add_argument("--exposure-time", type=int, default=None,
                    help="Arducam manual exposure_time_absolute via v4l2-ctl "
                         "(30-60 works indoors with IR; omit for driver auto)")
    ap.add_argument("--auto-exposure", type=float, default=None,
                    help="1=manual, 3=auto (V4L2)")
    ap.add_argument("--brightness", type=int, default=None,
                    help="camera brightness 0..255 (Arducam nightvision tuning)")
    ap.add_argument("--dry-run", action="store_true",
                    help="camera + detection only, no robot connection")
    ap.add_argument("--lift-delta", type=float, default=0.0,
                    help="shoulder_lift delta applied to the startup pose. "
                         "Default 0: the arm holds whatever pose you set before "
                         "launching (pose it up first). Sign depends on calibration.")
    ap.add_argument("--elbow-delta", type=float, default=0.0,
                    help="elbow_flex delta (same sign caveat as --lift-delta)")
    ap.add_argument("--pan-delta", type=float, default=0.0,
                    help="shoulder_pan delta for look-up pose")
    ap.add_argument("--wave-amplitude", type=float, default=18.0)
    ap.add_argument("--wave-cycles", type=float, default=3.0)
    ap.add_argument("--wave-style", default="claw", choices=["claw", "classic"],
                    help="claw = gripper opens + wrist_roll palm turn (no body shake); "
                         "classic = old wrist_flex + pan sway")
    ap.add_argument("--sensitivity", type=float, default=1.0,
                    help="wave detector sensitivity multiplier (>1 = more sensitive)")
    ap.add_argument("--verbose", action="store_true",
                    help="print motion energy every ~2s to help tune --sensitivity")
    ap.add_argument("--with-splunk", action="store_true",
                    help="also emit telemetry events via lerobot_splunk.SplunkHECSender")
    ap.add_argument("--no-voice", action="store_true",
                    help="disable the mic trigger (arm waits in look-up pose instead "
                         "of resting until it hears you)")
    ap.add_argument("--wake-word", default="hey max",
                    help="spoken trigger phrase (fuzzy-matched; default 'hey max'). "
                         "Used when vosk + the model are available; otherwise any "
                         "sound above the noise floor triggers.")
    ap.add_argument("--mic-device", default="hw:CARD=BRIO,DEV=0",
                    help="ALSA capture device for the voice trigger (BRIO mic default)")
    ap.add_argument("--look-up-lift-abs", type=float, default=-95.0,
                    help="absolute shoulder_lift target when looking up "
                         "(from your verified wrist-up calibration)")
    ap.add_argument("--look-up-elbow-abs", type=float, default=95.0,
                    help="absolute elbow_flex target when looking up")
    ap.add_argument("--face-model", default=None,
                    help="YuNet ONNX path; default: models/ next to this script")
    args = ap.parse_args()

    splunk = None
    if args.with_splunk:
        from lerobot_splunk import SplunkHECSender, TelemetryEvent
        from datetime import datetime, timezone
        splunk = SplunkHECSender()

        def emit(event_type: str, message: str):
            now = datetime.now(timezone.utc)
            splunk.enqueue(TelemetryEvent(
                timestamp_utc=now.isoformat(),
                epoch_ms=int(now.timestamp() * 1000),
                event_type=event_type,
                session_id=f"wave_demo_{int(now.timestamp())}",
                host=socket.gethostname(),
                raw_message=message,
            ))
    else:
        def emit(event_type: str, message: str):
            print(f"[event] {event_type}: {message}")

    robot = None
    arm = None
    startup_pose = None
    look_up_pose = None

    cam = Camera(args.camera_index,
                 exposure_time=args.exposure_time,
                 auto_exposure=args.auto_exposure,
                 brightness=args.brightness)
    det = WaveDetector(min_amplitude_px=28.0 / max(args.sensitivity, 0.05),
                       min_motion_energy=3.0 / max(args.sensitivity, 0.05),
                       min_blob_area=int(220 / max(args.sensitivity, 0.05) ** 0.5))

    voice_mode = not args.no_voice and not args.dry_run
    mic = None
    face_finder = None
    if voice_mode:
        face_model = args.face_model or str(
            Path(__file__).resolve().parent / "models" / "face_detection_yunet_2023mar.onnx")
        if Path(face_model).exists():
            try:
                face_finder = FaceFinder(face_model)
                print(f"[find] YuNet face detector loaded: {face_model}")
            except Exception as e:
                print(f"[find] face detector unavailable ({e}) — scan disabled")
        else:
            print(f"[find] face model not found at {face_model} — scan disabled")

    if not args.dry_run:
        print(f"[robot] connecting {args.robot_id} on {args.port} ...")
        cfg = SO101FollowerConfig(
            port=args.port,
            id=args.robot_id,
            cameras={},                 # camera handled directly by OpenCV here
            use_degrees=False,          # -100..100 normalized (matches prior setup)
            max_relative_target=10.0,   # per-command safety clamp
            disable_torque_on_disconnect=True,
        )
        robot = SO101Follower(cfg)
        robot.connect(calibrate=False)  # calibration file already exists for this id
        arm = ArmController(robot)
        startup_pose = arm.read()
        print(f"[robot] connected. startup pose: "
              + ", ".join(f"{m}={startup_pose[m]:.1f}" for m in MOTOR_NAMES if m in startup_pose))

        def build_look_up_pose() -> dict:
            """Look-up pose: absolute lift/elbow targets (verified wrist-up
            calibration), keeping current pan/wrist/roll/gripper."""
            cur = arm.read()
            return clamp_pose({
                **cur,
                "shoulder_lift": args.look_up_lift_abs,
                "elbow_flex": args.look_up_elbow_abs,
            })

        if voice_mode:
            # Prefer the wake-word listener; fall back to plain RMS VAD.
            listener = None
            wake_model = (Path(__file__).resolve().parent / "models"
                          / "vosk-model-small-en-us-0.15")
            try:
                import vosk  # noqa: F401
                if wake_model.exists():
                    listener = WakeWordListener(str(wake_model),
                                                wake_word=args.wake_word,
                                                device=args.mic_device)
                    if not listener.setup():
                        listener = None
            except ImportError:
                print("[wake] vosk not installed — falling back to sound-level trigger")
            mic = listener or MicVAD(device=args.mic_device)
            mic.start()
            hint = (f"say '{args.wake_word}'" if listener
                    else "make a sound (wake word unavailable — using sound level)")
            print(f"[demo] resting ... {hint} and I'll look up and find you")
            mic.speech_event.wait()
            mic.speech_event.clear()
            print("[demo] *** VOICE HEARD *** -> looking up")
            emit("voice_heard", "speech trigger")
            look_up_pose = build_look_up_pose()
            arm.move_to(look_up_pose, duration_s=2.0)
            emit("look_up", "reached look-up pose")
            if face_finder is not None:
                print("[find] scanning the room for a face ...")
                face_finder.aim(arm, look_up_pose, cam)
                emit("face_scan_done", "find-you scan complete")
        else:
            look_up_pose = clamp_pose({
                **startup_pose,
                "shoulder_lift": startup_pose.get("shoulder_lift", 0.0) + args.lift_delta,
                "elbow_flex": startup_pose.get("elbow_flex", 0.0) + args.elbow_delta,
                "shoulder_pan": startup_pose.get("shoulder_pan", 0.0) + args.pan_delta,
            })
            print("[robot] moving to look-up pose ...")
            arm.move_to(look_up_pose, duration_s=2.0)
            emit("look_up", "reached look-up pose")
    else:
        print("[dry-run] robot disabled; camera + detector only")
        look_up_pose = {m: 0.0 for m in MOTOR_NAMES}

    running = True
    waves_triggered = 0

    # ── Idle-baseline calibration: sample the scene while everything is still ──
    print("[det] calibrating idle motion baseline (keep still ~1.5s) ...")
    idle_samples = []
    t_end = time.monotonic() + 1.5
    while time.monotonic() < t_end:
        frame = cam.read()
        if frame is None:
            continue
        _, energy = det.process(frame)
        idle_samples.append(energy)
        time.sleep(1.0 / 30.0)
    if idle_samples:
        det.set_idle_baseline(float(np.mean(idle_samples)))

    # ── Attention move: one short greeting wave so you know it's watching ──
    print("[demo] attention wave ...")
    emit("attention_wave", "greeting wave to attract attention")
    if arm is not None:
        arm.wave(look_up_pose, amplitude=args.wave_amplitude * 0.7,
                 cycles=2.0, freq_hz=2.2, style=args.wave_style)
        arm.move_to(look_up_pose, duration_s=0.8)
    time.sleep(1.0)
    print("[demo] now wave at the camera!")

    def shutdown(signum, _frame):
        nonlocal running
        running = False

    signal.signal(signal.SIGINT, shutdown)
    signal.signal(signal.SIGTERM, shutdown)

    print("[demo] watching for a wave ... (Ctrl+C to stop)")
    last_verbose = time.monotonic()
    try:
        while running:
            frame = cam.read()
            if frame is None:
                time.sleep(0.01)
                continue

            # Voice heard while watching -> re-find the speaker
            if mic is not None and mic.speech_event.is_set():
                mic.speech_event.clear()
                print("[demo] *** VOICE HEARD *** -> re-finding you")
                emit("voice_heard", "re-scan trigger")
                if face_finder is not None:
                    face_finder.aim(arm, look_up_pose, cam)

            detected, energy = det.process(frame)

            if args.verbose and time.monotonic() - last_verbose >= 1.0:
                last_verbose = time.monotonic()
                print(f"[det] energy={energy:5.2f} (gate={det.min_motion_energy:.2f}) "
                      f"reversals={det.last_reversals}/{det.min_reversals} "
                      f"waves={waves_triggered}")

            if detected:
                waves_triggered += 1
                print(f"[demo] *** WAVE DETECTED (#{waves_triggered}) -> waving back ***")
                emit("wave_detected", f"human wave #{waves_triggered}")

                if arm is not None:
                    arm.wave(look_up_pose,
                             amplitude=args.wave_amplitude,
                             cycles=args.wave_cycles, style=args.wave_style)
                    arm.move_to(look_up_pose, duration_s=0.8)
                    emit("wave_done", f"completed wave #{waves_triggered}")

                time.sleep(1.0)  # settle before watching again
            else:
                time.sleep(1.0 / 30.0)
    finally:
        print(f"[demo] stopping ... waves triggered: {waves_triggered}")
        if mic is not None:
            mic.stop()
            print("[mic] stopped")
        if arm is not None and startup_pose is not None:
            print("[robot] easing back to startup pose ...")
            try:
                arm.move_to(startup_pose, duration_s=2.0)
            except Exception as e:
                print(f"[robot] return-home error: {e}")
            try:
                robot.disconnect()
                print("[robot] disconnected cleanly")
            except Exception as e:
                print(f"[robot] disconnect error: {e}")
        cam.release()
        if splunk is not None:
            print(f"[splunk] stats: {splunk.stats}")


if __name__ == "__main__":
    main()
