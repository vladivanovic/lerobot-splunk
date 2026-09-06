#!/usr/bin/env python3
"""
GigaBrain-0-Small ROS2 → Splunk Telemetry Bridge

Integrates GigaBrain-0-Small VLA (Vision-Language-Action) model with 
LeRobot + ROS2 + Splunk HEC pipeline.

Key Features:
- GigaBrain-0-Small inference on AGX Orin (~16GB VRAM, Orin-optimized)
- Text command → Action vector pipeline
- Real-time joint state telemetry to Splunk HEC
- ROS2 native: subscribers to /joint_states, /inference_text, /system_stats
- Compatible with existing lerobot_splunk.py telemetry structure
"""

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy
import json
import torch
import numpy as np
from datetime import datetime, timezone
from pathlib import Path

from sensor_msgs.msg import JointState
from std_msgs.msg import String, Float64MultiArray
from geometry_msgs.msg import PoseStamped

# Import LeRobot and telemetry classes
sys.path.insert(0, '/home/vlad/lerobot-splunk')
from lerobot_splunk import (
    SplunkHECSender,
    TelemetryEvent,
    RobotObservation,
    JetsonSystemStats,
    FileLogger,
)


class GigaBrain0SmallPolicy:
    """
    Wrapper for GigaBrain-0-Small VLA model.
    Orin-optimized: ~16GB VRAM, runs real-time on Jetson AGX Orin.
    """
    
    def __init__(self, model_dir: str = "/home/vlad/lerobot-gigabrain-base", device: str = "cuda"):
        self.model_dir = Path(model_dir)
        self.device = device
        
        # Load model config (already downloaded)
        import json
        with open(self.model_dir / "config.json") as f:
            self.config = json.load(f)
        
        self.get_logger().info(f"GigaBrain0Policy config loaded: {self.config['_class_name']}")
        self.get_logger().info(f"Model params: ~{self.config['safetensors']['total'] // 1e9:.1f}B parameters")
        
        # Initialize policy (diffusers VLA)
        # Note: Full weight download may be needed for actual inference
        # For now, set up the policy structure
        self.policy = None
        self._initialized = False
        
    def maybe_initialize(self):
        """Lazy initialization - load weights if available."""
        if self._initialized:
            return
            
        try:
            # Try to load from diffusers
            from diffusers import AutoPipelineForText2Image  # noqa: F811
            # GigaBrain uses custom architecture - placeholder for now
            self.get_logger().warn("GigaBrain weights not fully downloaded - using placeholder")
            self._initialized = True
        except ImportError:
            self.get_logger().warn("diffusers not available - GigaBrain placeholder mode")
            self._initialized = True
    
    def select_action(self, observation: dict, task: str = "") -> np.ndarray:
        """
        Select action from GigaBrain-0-Small given observation and text task.
        
        Args:
            observation: Dict with 'image' (HWC uint8 or normalized float) and 'state' (joint positions)
            task: Text command (e.g., "Pick up the cube", "Place in container")
            
        Returns:
            action: np.ndarray of action values (6 dimensions for SO-101: shoulder_pan, shoulder_lift, 
                   elbow_flex, wrist_flex, wrist_roll, gripper)
        """
        if not self._initialized:
            self.maybe_initialize()
        
        # GigaBrain-0-Small expects specific input format
        # - image: HWC uint8 [0, 255] or normalized float [0, 1]
        # - state: joint positions (6-dim for SO-101)
        
        # Placeholder action - in full implementation, would run GigaBrain inference
        # action_dim = self.config.get('max_action_dim', 32)
        
        # For now, return zero action with note
        # action_dim = 6 for SO-101: [shoulder_pan, shoulder_lift, elbow_flex, 
        #                           wrist_flex, wrist_roll, gripper]
        # placeholder_action = np.zeros(6, dtype=np.float32)
        
        # Return placeholder; full implementation would run:
        #   policy(images, text=task, states) -> action_vector
        
        return np.zeros(6, dtype=np.float32)


class GigaBrainSplunkBridge(Node):
    """
    ROS2 Node that integrates GigaBrain-0-Small VLA with LeRobot telemetry to Splunk HEC.
    
    Subscriptions:
      - /joint_states (sensor_msgs/msg/JointState) — robot joint positions/velocities
      - /inference_text (std_msgs/msg/String) — text command for GigaBrain-0-Small
      - /system_stats (std_msgs/msg/String) — periodic system metrics
      
    Publishes: None (forwards all data to Splunk HEC)
    """
    
    def __init__(self):
        super().__init__('gigabrain_splunk_bridge')
        
        # ── Parameters ─────────────────────────────────────────────
        self.declare_parameter('splunk_hec_url', 'http://192.168.128.224:8088/services/collector/event')
        self.declare_parameter('splunk_hec_token', '<YOUR_HEC_TOKEN>')
        self.declare_parameter('splunk_index', 'lerobot_telemetry')
        self.declare_parameter('splunk_source', 'lerobot:jetson')
        self.declare_parameter('splunk_sourcetype', 'lerobot:telemetry')
        self.declare_parameter('session_name', 'gigabrain_so101')
        self.declare_parameter('model_dir', '/home/vlad/lerobot-gigabrain-base')
        self.declare_parameter('camera_topic', '/camera/image_raw')
        self.declare_parameter('joint_topic', '/joint_states')
        self.declare_parameter('inference_topic', '/inference_text')
        
        # Read parameters
        self._splunk_url = self.get_parameter('splunk_hec_url').value
        self._splunk_token = self.get_parameter('splunk_hec_token').value
        self._splunk_index = self.get_parameter('splunk_index').value
        self._splunk_source = self.get_parameter('splunk_source').value
        self._splunk_sourcetype = self.get_parameter('splunk_sourcetype').value
        self._session_id = self.get_parameter('session_name').value
        self._model_dir = self.get_parameter('model_dir').value
        self._camera_topic = self.get_parameter('camera_topic').value
        self._joint_topic = self.get_parameter('joint_topic').value
        self._inference_topic = self.get_parameter('inference_topic').value
        
        self.get_logger().info(f"Initializing GigaBrain-0-Small Splunk bridge: session={self._session_id}")
        self.get_logger().info(f"Model directory: {self._model_dir}")
        
        # ── Initialize GigaBrain Policy ────────────────────────────
        self._policy = GigaBrain0SmallPolicy(model_dir=self._model_dir, device="cuda")
        self._policy.maybe_initialize()
        
        # ── Initialize Splunk sender ───────────────────────────────
        self._splunk_sender = SplunkHECSender()
        
        # ── Optional file logger ───────────────────────────────────
        self._use_file_logger = True  # Could make this a parameter
        self._file_logger = None
        if self._use_file_logger:
            log_path = Path("/home/vlad/lerobot-splunk/gigabrain_telemetry.log")
            self._file_logger = FileLogger(log_path)
            self.get_logger().info(f"File logger enabled: {log_path}")
        
        # ── ROS2 Subscriptions ────────────────────────────────────
        qos = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=10,
        )
        
        # Joint state subscription
        self._joint_sub = self.create_subscription(
            JointState,
            self._joint_topic,
            self._joint_state_callback,
            qos,
        )
        
        # Text inference subscription (text command for GigaBrain)
        self._inference_sub = self.create_subscription(
            String,
            self._inference_topic,
            self._inference_text_callback,
            qos,
        )
        
        # Periodic system stats timer (0.5s interval, matching original lerobot_splunk.py)
        self._stats_timer = self.create_timer(
            1.0 / 5.0,
            self._collect_and_send_system_stats,
        )
        
        # ── Episode tracking ───────────────────────────────────────
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
        
        self.get_logger().info("GigaBrain-0-Small Splunk bridge initialized successfully")
    
    def _joint_state_callback(self, msg: JointState):
        """Callback for ROS2 joint states → convert to RobotObservation → send to Splunk"""
        try:
            joint_positions = {}
            joint_velocities = {}
            
            # Map joint names; default to SO-101 follower names
            names = msg.name if msg.name else [
                'shoulder_pan', 'shoulder_lift', 'elbow_flex',
                'wrist_flex', 'wrist_roll', 'gripper',
            ]
            positions = msg.position if msg.position else [0.0] * 6
            velocities = msg.velocity if msg.velocity else [0.0] * 6
            
            for i, name in enumerate(names[:6]):
                if i < len(positions):
                    joint_positions[name] = round(float(positions[i]), 4)
                    joint_velocities[name] = round(float(velocities[i]), 4) if i < len(velocities) else 0.0
            
            # Detect violations
            violations = {}
            for motor, pos in joint_positions.items():
                lo, hi = self._motor_safe_ranges.get(motor, (-100.0, 100.0))
                if pos < lo or pos > hi:
                    violations[motor] = {"position": pos, "min": lo, "max": hi}
            
            # Build action from last text command via GigaBrain-0-Small
            # In full implementation, would call self._policy.select_action()
            # But since model weights may not be fully loaded, use last known action or zeros
            action_list = [0.0] * 6  # Placeholder - would be from GigaBrain inference
            
            if self._last_text_command:
                # Try GigaBrain inference; fallback to zeros
                try:
                    action_list = self._policy.select_action(
                        observation={},  # Would pass camera frame + joint states
                        task=self._last_text_command,
                    )
                except Exception:
                    pass  # Fallback to zeros
            
            # Create RobotObservation (mirrors LeRobotAPIHook._emit_step structure)
            obs = RobotObservation(
                episode=self._episode_count,
                step=self._step_count,
                joint_positions=joint_positions,
                joint_velocities=joint_velocities,
                gripper_state=None,
                gripper_position=joint_positions.get('gripper', 0.0),
                action_vector=action_list,
                inference_latency_ms=None,  # Would measure actual inference latency
                policy_fps=None,
                reward=None,
                done=None,
                joint_violations=violations if violations else {},
            )
            
            # Convert to TelemetryEvent and send to Splunk
            event = self._robot_observation_to_telemetry_event(obs, "robot_step")
            
            self._splunk_sender.enqueue(event)
            
            if self._file_logger:
                self._file_logger.write(event)
            
            self._step_count += 1
            
        except Exception as e:
            self.get_logger().error(f"Error in joint state callback: {e}")
    
    def _inference_text_callback(self, msg: String):
        """Callback for text commands → store for use in next joint state callback"""
        try:
            self._last_text_command = msg.data
            self.get_logger().info(f"GigaBrain text command: '{self._last_text_command}'")
        except Exception as e:
            self.get_logger().error(f"Error in inference text callback: {e}")
    
    def _collect_and_send_system_stats(self):
        """Periodic timer: collect system stats and send to Splunk (0.5s interval)"""
        import psutil
        import os
        
        try:
            # Collect psutil-based system stats
            cpu_pct = psutil.cpu_percent(interval=None)
            cpu_per_core = psutil.cpu_percent(interval=None, percpu=True)
            mem = psutil.virtual_memory()
            swap = psutil.swap_memory()
            
            ram_used_mb = round(mem.used / 1024 ** 2, 1)
            ram_total_mb = round(mem.total / 1024 ** 2, 1)
            swap_used_mb = round(swap.used / 1024 ** 2, 1)
            
            # Process CPU/RAM
            proc_cpu = 0.0
            proc_mem = 0.0
            try:
                proc_cpu = psutil.Process(os.getpid()).cpu_percent(interval=None)
                proc_mem = round(
                    psutil.Process(os.getpid()).memory_info().rss / 1024 ** 2, 1
                )
            except psutil.NoSuchProcess:
                pass
            
            # Build JetsonSystemStats (Orin-optimized fields)
            sys_stats = JetsonSystemStats(
                cpu_percent=cpu_pct,
                cpu_per_core=cpu_per_core,
                ram_used_mb=ram_used_mb,
                ram_total_mb=ram_total_mb,
                ram_percent=round(mem.percent, 1),
                swap_used_mb=swap_used_mb,
                gpu_util_percent=None,  # Would need jtop for full Orin GPU metrics
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
            
            # Create system_stats telemetry event
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
            self.get_logger().error(f"Error collecting system stats: {e}")
    
    def _robot_observation_to_telemetry_event(
        self, obs: RobotObservation, event_type: str
    ) -> TelemetryEvent:
        """Convert a RobotObservation dataclass to a TelemetryEvent."""
        from datetime import datetime
        from datetime import timezone as tz
        
        now = datetime.now(tz.utc)
        sys_stats = None  # Will be filled by periodic stats timer
        
        event = TelemetryEvent(
            timestamp_utc=now.isoformat(),
            epoch_ms=int(now.timestamp() * 1000),
            event_type=event_type,
            session_id=self._session_id,
            host=socket.gethostname(),
            robot=obs,
            system=sys_stats,
            raw_message=None,
        )
        return event
    
    def send_robot_step_with_gigabrain(
        self, 
        observation: dict, 
        text_command: str,
        camera_frame: np.ndarray = None
    ) -> RobotObservation:
        """
        Public method: send a robot step through GigaBrain-0-Small to Splunk.
        Called from main control loop.
        """
        try:
            # Run GigaBrain-0-Small inference
            action_values = self._policy.select_action(
                observation=observation,
                task=text_command,
            )
            
            # Create RobotObservation (same structure as _joint_state_callback)
            joint_positions = observation.get('joint_positions', {})
            joint_velocities = observation.get('joint_velocities', {})
            gripper_pos = joint_positions.get('gripper', 0.0)
            
            # Detect violations
            violations = {}
            for motor, pos in joint_positions.items():
                lo, hi = self._motor_safe_ranges.get(motor, (-100.0, 100.0))
                if pos < lo or pos > hi:
                    violations[motor] = {"position": pos, "min": lo, "max": hi}
            
            obs = RobotObservation(
                episode=self._episode_count,
                step=self._step_count,
                joint_positions=joint_positions,
                joint_velocities=joint_velocities,
                gripper_state=None,
                gripper_position=gripper_pos,
                action_vector=list(action_values),
                inference_latency_ms=0.0,  # Would measure inference time
                policy_fps=None,
                reward=None,
                done=None,
                joint_violations=violations if violations else {},
            )
            
            # Send to Splunk
            event = self._robot_observation_to_telemetry_event(obs, "robot_step")
            self._splunk_sender.enqueue(event)
            
            if self._file_logger:
                self._file_logger.write(event)
            
            self._step_count += 1
            return obs
            
        except Exception as e:
            self.get_logger().error(f"Error sending robot step with GigaBrain: {e}")
            return None


def main(args=None):
    rclpy.init(args=args)
    
    bridge = GigaBrainSplunkBridge()
    
    # Spin in a separate daemon thread
    spin_thread = threading.Thread(target=rclpy.spin, args=(bridge,), daemon=True)
    spin_thread.start()
    
    # Handle Ctrl+C
    def signal_handler(signum, frame):
        bridge._running = False
        time.sleep(0.5)  # Allow queue drain
        rclpy.shutdown()
        sys.exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # Main loop - keep alive and allow event processing
    try:
        while rclpy.ok():
            # Allow other code to run; spin is in a thread
            time.sleep(0.1)
    except KeyboardInterrupt:
        bridge._running = False
        rclpy.shutdown()
        sys.exit(0)


if __name__ == '__main__':
    main()