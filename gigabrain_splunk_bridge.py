#!/usr/bin/env python3
"""
GigaBrain-0 ROS2 → Splunk Telemetry Bridge

Integrates a GigaBrain VLA (Vision-Language-Action) model with
LeRobot + ROS2 + Splunk HEC pipeline.

Key Features:
- GigaBrain inference on AGX Orin (~16GB VRAM, Orin-optimized)
- Text command → Action vector pipeline
- Real-time joint state telemetry to Splunk HEC
- ROS2 native: subscribers to /joint_states, /inference_text
- Compatible with existing lerobot_splunk.py telemetry structure

Usage:
    # Pass model path via CLI flag:
    python3 gigabrain_splunk_bridge.py --model-path /home/vlad/models/GigaBrain-0-3.5B-Base

    # Or via ROS2 parameter:
    python3 gigabrain_splunk_bridge.py --ros-args -p model_path:=/home/vlad/models/GigaBrain-0-3.5B-Base

    # Or via environment variable:
    GIGABRAIN_MODEL_PATH=/path/to/model python3 gigabrain_splunk_bridge.py

    # All Splunk / logging settings are ROS2 parameters, e.g.:
    python3 gigabrain_splunk_bridge.py --model-path ... \
        --ros-args -p splunk_hec_url:=http://my-splunk:8088/services/collector/event \
                   -p splunk_hec_token:=<TOKEN> \
                   -p session_name:=gigabrain_test
"""

import argparse
import json
import os
import signal
import socket
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch  # noqa: F401  (used by policy backend when weights load)
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy

from sensor_msgs.msg import JointState
from std_msgs.msg import String

# Import telemetry classes from lerobot_splunk (same dir as this script)
_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from lerobot_splunk import (  # noqa: E402
    SplunkHECSender,
    TelemetryEvent,
    RobotObservation,
    JetsonSystemStats,
    FileLogger,
)


class GigaBrainPolicy:
    """
    Wrapper for a GigaBrain VLA model checkpoint directory.

    Expects a local directory containing:
      - config.json                              (GigaBrain0Policy config)
      - diffusion_pytorch_model*.safetensors     (weights, 4 shards)
      - diffusion_pytorch_model.safetensors.index.json
    """

    def __init__(self, model_path: str, logger=None, device: str = "cuda"):
        self.model_path = Path(model_path).expanduser().resolve()
        self.device = device
        self._log = logger or (lambda msg: print(f"[GigaBrain] {msg}"))

        if not self.model_path.is_dir():
            raise FileNotFoundError(f"Model directory not found: {self.model_path}")

        config_file = self.model_path / "config.json"
        if not config_file.exists():
            raise FileNotFoundError(f"config.json not found in {self.model_path}")

        with open(config_file) as f:
            self.config = json.load(f)

        # Param count from the safetensors index (avoids guessing from config)
        index_file = self.model_path / "diffusion_pytorch_model.safetensors.index.json"
        self.num_tensors = None
        if index_file.exists():
            with open(index_file) as f:
                self.num_tensors = len(json.load(f)["weight_map"])

        self._log(f"Model loaded from: {self.model_path}")
        self._log(f"Class: {self.config.get('_class_name', '?')} | "
                  f"action_dim: {self.config.get('max_action_dim', '?')} | "
                  f"tensors: {self.num_tensors or 'unknown'}")

        self.policy = None
        self._initialized = False

    def maybe_initialize(self):
        """Lazy initialization - load weights when available."""
        if self._initialized:
            return

        weight_index = self.model_path / "diffusion_pytorch_model.safetensors.index.json"
        shards = sorted(self.model_path.glob("diffusion_pytorch_model-*.safetensors"))

        if not shards:
            self._log("WARNING: no weight shards found - placeholder mode "
                      "(telemetry only, actions will be zeros)")
        else:
            self._log(f"Found {len(shards)} weight shard(s); "
                      f"full inference wiring pending (GigaBrain0Policy runtime)")
        self._initialized = True

    def select_action(self, observation: dict, task: str = "") -> np.ndarray:
        """
        Select action given observation and text task.

        Args:
            observation: dict with 'image' (HWC uint8/float) and 'state' (joint positions)
            task: text command, e.g. "Pick up the cube"

        Returns:
            np.ndarray of action values (SO-101: 6 dims)
        """
        if not self._initialized:
            self.maybe_initialize()

        # Full implementation would run:
        #   policy(images, text=task, states) -> action_vector
        # Placeholder until the GigaBrain0Policy runtime is wired in.
        return np.zeros(6, dtype=np.float32)


class GigaBrainSplunkBridge(Node):
    """
    ROS2 Node integrating a GigaBrain VLA with LeRobot telemetry to Splunk HEC.

    Subscriptions:
      - <joint_topic>     (sensor_msgs/msg/JointState) — robot joint positions/velocities
      - <inference_topic> (std_msgs/msg/String)       — text command for GigaBrain

    Publishes: none (forwards all data to Splunk HEC)
    """

    def __init__(self, model_path: str, log_path: str | None = None):
        super().__init__('gigabrain_splunk_bridge')

        # ── Parameters (all overridable via --ros-args -p name:=value) ──
        self.declare_parameter('model_path', model_path or '')
        self.declare_parameter('splunk_hec_url', 'http://192.168.128.224:8088/services/collector/event')
        self.declare_parameter('splunk_hec_token', '<YOUR_HEC_TOKEN>')
        self.declare_parameter('splunk_index', 'lerobot_telemetry')
        self.declare_parameter('splunk_source', 'lerobot:jetson')
        self.declare_parameter('splunk_sourcetype', 'lerobot:telemetry')
        self.declare_parameter('session_name', 'gigabrain_so101')
        self.declare_parameter('camera_topic', '/camera/image_raw')
        self.declare_parameter('joint_topic', '/joint_states')
        self.declare_parameter('inference_topic', '/inference_text')
        self.declare_parameter('log_path', log_path or str(_SCRIPT_DIR / 'gigabrain_telemetry.log'))
        self.declare_parameter('use_file_logger', True)

        # ── Read parameters ─────────────────────────────────────────
        # model_path precedence: ROS2 param > --model-path CLI > GIGABRAIN_MODEL_PATH env
        self._model_path = (
            self.get_parameter('model_path').value
            or model_path
            or os.environ.get('GIGABRAIN_MODEL_PATH', '')
        ).strip()

        if not self._model_path:
            self.get_logger().error(
                "No model path provided. Pass one of:\n"
                "  --model-path /path/to/model\n"
                "  --ros-args -p model_path:=/path/to/model\n"
                "  GIGABRAIN_MODEL_PATH=/path/to/model"
            )
            raise SystemExit(1)

        self._splunk_url       = self.get_parameter('splunk_hec_url').value
        self._splunk_token     = self.get_parameter('splunk_hec_token').value
        self._splunk_index     = self.get_parameter('splunk_index').value
        self._splunk_source    = self.get_parameter('splunk_source').value
        self._splunk_sourcetype = self.get_parameter('splunk_sourcetype').value
        self._session_id       = self.get_parameter('session_name').value
        self._camera_topic     = self.get_parameter('camera_topic').value
        self._joint_topic      = self.get_parameter('joint_topic').value
        self._inference_topic  = self.get_parameter('inference_topic').value
        self._log_path         = Path(self.get_parameter('log_path').value)
        self._use_file_logger  = self.get_parameter('use_file_logger').value

        self.get_logger().info(f"Session: {self._session_id}")
        self.get_logger().info(f"Splunk HEC: {self._splunk_url} (index={self._splunk_index})")
        self.get_logger().info(f"Model path: {self._model_path}")

        # ── Initialize GigaBrain policy ─────────────────────────────
        self._policy = GigaBrainPolicy(
            model_path=self._model_path,
            logger=lambda msg: self.get_logger().info(msg),
            device="cuda",
        )
        self._policy.maybe_initialize()

        # ── Initialize Splunk sender ────────────────────────────────
        self._splunk_sender = SplunkHECSender()

        # ── Optional file logger ────────────────────────────────────
        self._file_logger = None
        if self._use_file_logger:
            self._file_logger = FileLogger(self._log_path)
            self.get_logger().info(f"File logger: {self._log_path}")

        # ── ROS2 subscriptions ──────────────────────────────────────
        qos = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=10,
        )

        self._joint_sub = self.create_subscription(
            JointState, self._joint_topic, self._joint_state_callback, qos)
        self._inference_sub = self.create_subscription(
            String, self._inference_topic, self._inference_text_callback, qos)

        # Periodic system stats (0.5s interval, matching lerobot_splunk.py)
        self._stats_timer = self.create_timer(0.2, self._collect_and_send_system_stats)

        # ── Episode tracking ────────────────────────────────────────
        self._episode_count = 0
        self._step_count = 0
        self._running = True
        self._last_text_command = ""

        # Motor safe ranges (mirroring lerobot_splunk.py MOTOR_SAFE_RANGES)
        self._motor_safe_ranges = {
            "shoulder_pan": (-80.0, 80.0),
            "shoulder_lift": (-80.0, 80.0),
            "elbow_flex": (-80.0, 80.0),
            "wrist_flex": (-80.0, 80.0),
            "wrist_roll": (-100.0, 100.0),
            "gripper": (0.0, 100.0),
        }

        self.get_logger().info("GigaBrain Splunk bridge initialized successfully")

    # ── Callbacks ───────────────────────────────────────────────────

    def _joint_state_callback(self, msg: JointState):
        """ROS2 joint states → RobotObservation → Splunk"""
        try:
            joint_positions, joint_velocities = {}, {}

            names = list(msg.name) if msg.name else [
                'shoulder_pan', 'shoulder_lift', 'elbow_flex',
                'wrist_flex', 'wrist_roll', 'gripper',
            ]
            positions = list(msg.position) if msg.position else [0.0] * 6
            velocities = list(msg.velocity) if msg.velocity else [0.0] * 6

            for i, name in enumerate(names[:6]):
                joint_positions[name] = round(float(positions[i]), 4) if i < len(positions) else 0.0
                joint_velocities[name] = round(float(velocities[i]), 4) if i < len(velocities) else 0.0

            # Safety violation detection
            violations = {
                motor: {"position": pos, "min": lo, "max": hi}
                for motor, pos in joint_positions.items()
                for lo, hi in [self._motor_safe_ranges.get(motor, (-100.0, 100.0))]
                if pos < lo or pos > hi
            }

            # Action from GigaBrain inference on last text command (fallback: zeros)
            action_list = [0.0] * 6
            if self._last_text_command:
                try:
                    action_list = list(self._policy.select_action(
                        observation={'state': joint_positions},
                        task=self._last_text_command,
                    ))
                except Exception as e:
                    self.get_logger().warn(f"GigaBrain inference failed, zeros used: {e}")

            obs = RobotObservation(
                episode=self._episode_count,
                step=self._step_count,
                joint_positions=joint_positions,
                joint_velocities=joint_velocities,
                gripper_state=None,
                gripper_position=joint_positions.get('gripper', 0.0),
                action_vector=action_list,
                inference_latency_ms=None,
                policy_fps=None,
                reward=None,
                done=None,
                joint_violations=violations,
            )

            event = self._to_telemetry_event(obs, "robot_step")
            self._splunk_sender.enqueue(event)
            if self._file_logger:
                self._file_logger.write(event)

            self._step_count += 1

        except Exception as e:
            self.get_logger().error(f"Joint state callback error: {e}")

    def _inference_text_callback(self, msg: String):
        """Store latest text command for the next joint-state tick."""
        self._last_text_command = msg.data
        self.get_logger().info(f"GigaBrain text command: '{self._last_text_command}'")

    def _collect_and_send_system_stats(self):
        """Periodic system stats → Splunk"""
        import psutil

        try:
            mem = psutil.virtual_memory()
            swap = psutil.swap_memory()

            proc_cpu, proc_mem = 0.0, 0.0
            try:
                proc = psutil.Process(os.getpid())
                proc_cpu = proc.cpu_percent(interval=None)
                proc_mem = round(proc.memory_info().rss / 1024 ** 2, 1)
            except psutil.NoSuchProcess:
                pass

            sys_stats = JetsonSystemStats(
                cpu_percent=psutil.cpu_percent(interval=None),
                cpu_per_core=psutil.cpu_percent(interval=None, percpu=True),
                ram_used_mb=round(mem.used / 1024 ** 2, 1),
                ram_total_mb=round(mem.total / 1024 ** 2, 1),
                ram_percent=round(mem.percent, 1),
                swap_used_mb=round(swap.used / 1024 ** 2, 1),
                gpu_util_percent=None,   # jtop fields — wired when jtop integration lands
                gpu_freq_mhz=None,
                gpu_mem_used_mb=None,
                gpu_mem_total_mb=None,
                cpu_temp_c=None,
                gpu_temp_c=None,
                soc_temp_c=None,
                tj_temp_c=None,
                power_cpu_gpu_mw=None,
                power_soc_mw=None,
                power_total_mw=None,
                fan_percent=None,
                jetson_clocks=None,
                nvp_model=None,
                emc_util_percent=None,
                cpu_freq_mhz=None,
                process_cpu_percent=proc_cpu,
                process_mem_mb=proc_mem,
            )

            now = datetime.now(timezone.utc)
            event = TelemetryEvent(
                timestamp_utc=now.isoformat(),
                epoch_ms=int(now.timestamp() * 1000),
                event_type='system_stats',
                session_id=self._session_id,
                host=socket.gethostname(),
                system=sys_stats,
            )
            self._splunk_sender.enqueue(event)
            if self._file_logger:
                self._file_logger.write(event)

        except Exception as e:
            self.get_logger().error(f"System stats error: {e}")

    # ── Helpers ─────────────────────────────────────────────────────

    def _to_telemetry_event(self, obs: RobotObservation, event_type: str) -> TelemetryEvent:
        now = datetime.now(timezone.utc)
        return TelemetryEvent(
            timestamp_utc=now.isoformat(),
            epoch_ms=int(now.timestamp() * 1000),
            event_type=event_type,
            session_id=self._session_id,
            host=socket.gethostname(),
            robot=obs,
            system=None,
            raw_message=None,
        )


def parse_args():
    parser = argparse.ArgumentParser(
        description="GigaBrain VLA → ROS2 → Splunk telemetry bridge")
    parser.add_argument(
        '--model-path', '-m',
        default=os.environ.get('GIGABRAIN_MODEL_PATH', ''),
        help='Path to GigaBrain model directory (config.json + safetensors shards). '
             'Falls back to GIGABRAIN_MODEL_PATH env var.')
    parser.add_argument(
        '--log-path', '-l',
        default=None,
        help='Telemetry JSONL log file path (default: alongside this script).')
    # Let rclpy consume --ros-args from the same command line
    known, remaining = parser.parse_known_args()
    return known, remaining


def main(args=None):
    cli_args, ros_args = parse_args()

    rclpy.init(args=args + ros_args if args else ros_args)

    bridge = None
    try:
        bridge = GigaBrainSplunkBridge(
            model_path=cli_args.model_path,
            log_path=cli_args.log_path,
        )
    except SystemExit:
        rclpy.shutdown()
        sys.exit(1)
    except FileNotFoundError as e:
        print(f"[Bridge] Fatal: {e}", file=sys.stderr)
        rclpy.shutdown()
        sys.exit(1)

    # Spin in a daemon thread so main thread can handle signals
    spin_thread = threading.Thread(target=rclpy.spin, args=(bridge,), daemon=True)
    spin_thread.start()

    def signal_handler(signum, frame):
        bridge.get_logger().info(f"Signal {signum} — shutting down...")
        bridge._running = False
        time.sleep(0.5)  # allow Splunk queue drain
        rclpy.shutdown()
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    try:
        while rclpy.ok():
            time.sleep(0.1)
    except KeyboardInterrupt:
        bridge._running = False
        rclpy.shutdown()
        sys.exit(0)


if __name__ == '__main__':
    main()
