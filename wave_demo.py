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
import math
import signal
import socket
import sys
import time
from dataclasses import dataclass, field

import cv2
import numpy as np

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
    """
    roi_top_frac: float = 0.0     # top of ROI (fraction of height)
    roi_bottom_frac: float = 0.66 # bottom of ROI
    diff_threshold: int = 22      # pixel-intensity delta to count as motion
    min_blob_area: int = 220      # ignore tiny specks (px at 640-wide)
    window_sec: float = 2.2       # reversal window
    min_reversals: int = 3        # direction changes needed
    min_amplitude_px: float = 28  # total horizontal travel needed
    min_motion_energy: float = 3.0  # mean diff-brightness, rejects static scenes
    cooldown_sec: float = 4.0

    _prev_gray: np.ndarray | None = field(default=None, repr=False)
    _track: list = field(default_factory=list, repr=False)  # (t, x, energy)
    _last_trigger: float = field(default=-1e9, repr=False)
    last_energy: float = 0.0

    def process(self, frame_bgr: np.ndarray) -> tuple[bool, float]:
        """Returns (wave_detected, motion_energy)."""
        now = time.monotonic()
        h, w = frame_bgr.shape[:2]
        y0, y1 = int(h * self.roi_top_frac), int(h * self.roi_bottom_frac)
        x0, x1 = int(w * 0.10), int(w * 0.90)

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
            self._track.clear()
            return False, energy

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
            if reversals >= self.min_reversals and span >= self.min_amplitude_px:
                self._last_trigger = now
                self._track.clear()
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
             cycles: float = 3.0, freq_hz: float = 2.2, on_frame=None):
        """Sine wave on wrist_flex with a gentle shoulder_pan sway."""
        if self.robot is None:
            return
        duration = cycles / freq_hz
        steps = max(1, int(duration * self.rate))
        for i in range(steps):
            t = i / self.rate
            phase = 2 * math.pi * freq_hz * t
            envelope = min(1.0, t / 0.4, (duration - t) / 0.4)  # ease in/out
            pose = dict(base_pose)
            pose["wrist_flex"] = base_pose.get("wrist_flex", 0.0) + \
                amplitude * envelope * math.sin(phase)
            pose["shoulder_pan"] = base_pose.get("shoulder_pan", 0.0) + \
                0.45 * amplitude * envelope * math.sin(phase)
            self.send(pose)
            if on_frame:
                on_frame()
            time.sleep(1.0 / self.rate)


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
    ap.add_argument("--sensitivity", type=float, default=1.0,
                    help="wave detector sensitivity multiplier (>1 = more sensitive)")
    ap.add_argument("--verbose", action="store_true",
                    help="print motion energy every ~2s to help tune --sensitivity")
    ap.add_argument("--with-splunk", action="store_true",
                    help="also emit telemetry events via lerobot_splunk.SplunkHECSender")
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

        look_up_pose = clamp_pose({
            **startup_pose,
            "shoulder_lift": startup_pose.get("shoulder_lift", 0.0) + args.lift_delta,
            "elbow_flex": startup_pose.get("elbow_flex", 0.0) + args.elbow_delta,
            "shoulder_pan": startup_pose.get("shoulder_pan", 0.0) + args.pan_delta,
        })
        print("[robot] moving to look-up pose ...")
        arm.move_to(look_up_pose, duration_s=2.0)
        emit("look_up", "reached look-up pose; watching for waves")
    else:
        print("[dry-run] robot disabled; camera + detector only")
        look_up_pose = {m: 0.0 for m in MOTOR_NAMES}

    running = True
    waves_triggered = 0

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

            detected, energy = det.process(frame)

            if args.verbose and time.monotonic() - last_verbose >= 2.0:
                last_verbose = time.monotonic()
                print(f"[det] motion_energy={energy:5.2f} "
                      f"(trigger needs reversals in window; waves={waves_triggered})")

            if detected:
                waves_triggered += 1
                print(f"[demo] *** WAVE DETECTED (#{waves_triggered}) -> waving back ***")
                emit("wave_detected", f"human wave #{waves_triggered}")

                if arm is not None:
                    arm.wave(look_up_pose,
                             amplitude=args.wave_amplitude,
                             cycles=args.wave_cycles)
                    arm.move_to(look_up_pose, duration_s=0.8)
                    emit("wave_done", f"completed wave #{waves_triggered}")

                time.sleep(1.0)  # settle before watching again
            else:
                time.sleep(1.0 / 30.0)
    finally:
        print(f"[demo] stopping ... waves triggered: {waves_triggered}")
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
