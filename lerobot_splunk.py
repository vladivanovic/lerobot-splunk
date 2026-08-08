#!/usr/bin/env python3
"""
LeRobot Direct API → Cisco Splunk HEC Telemetry Bridge
Captures robot telemetry by hooking directly into LeRobot's Python API,
giving full joint positions, action vectors, and inference timing.

Platform: NVIDIA Jetson Orin Nano Super
LeRobot Version: 0.4.3
Robot: SO101 Follower Arm
"""

import threading
import time
import json
import queue
import requests
import os
import psutil
import signal
import sys
import socket
import torch

from datetime import datetime, timezone
from dataclasses import dataclass, asdict, field
from typing import Optional
from pathlib import Path

# ─────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────

SPLUNK_HEC_URL    = "http://192.168.128.224:8088/services/collector/event"
SPLUNK_HEC_TOKEN  = "<YOUR_HEC_TOKEN>"
SPLUNK_INDEX      = "lerobot_telemetry"
SPLUNK_SOURCE     = "lerobot:jetson"
SPLUNK_SOURCETYPE = "lerobot:telemetry"

SPLUNK_BATCH_SIZE  = 10
SPLUNK_TIMEOUT_SEC = 5

TELEMETRY_INTERVAL_SEC = 0.5

LOG_TO_FILE   = True
LOG_FILE_PATH = Path("lerobot_telemetry.log")

# ── Robot config — mirrors your working CLI command ───────────
ROBOT_CONFIG = {
    "type": "so101_follower",
    "id":   "follower_arm",
    "port": "/dev/ttyACM0",
    "cameras": {
        "front": {
            "type":          "opencv",
            "index_or_path": 2,
            "width":         640,
            "height":        360,
            "fps":           30,
            "fourcc":        "MJPG",
        }
    },
}

POLICY_PATH    = "outputs/train/lego-block-front-only/checkpoints/last/pretrained_model"
NUM_EPISODES   = 5
EPISODE_TIME_S = 15
RESET_TIME_S   = 15
FPS            = 30
SINGLE_TASK    = "Put lego brick into the plate box"

# ── Motor safe ranges for violation detection ──────────────────
# Values are normalized units: -100 to 100 for body joints, 0 to 100 for gripper
# Adjust these after watching a few runs to match your actual safe operating range
MOTOR_SAFE_RANGES = {
    "shoulder_pan":  (-80.0,  80.0),
    "shoulder_lift": (-80.0,  80.0),
    "elbow_flex":    (-80.0,  80.0),
    "wrist_flex":    (-80.0,  80.0),
    "wrist_roll":    (-100.0, 100.0),
    "gripper":       (0.0,    100.0),
}


# ─────────────────────────────────────────────────────────────
# DATA STRUCTURES
# ─────────────────────────────────────────────────────────────

@dataclass
class JetsonSystemStats:
    cpu_percent:          float
    cpu_per_core:         list
    ram_used_mb:          float
    ram_total_mb:         float
    ram_percent:          float
    swap_used_mb:         float
    gpu_util_percent:     Optional[float]
    gpu_freq_mhz:         Optional[float]
    gpu_mem_used_mb:      Optional[float]
    gpu_mem_total_mb:     Optional[float]
    cpu_temp_c:           Optional[float]
    gpu_temp_c:           Optional[float]
    soc_temp_c:           Optional[float]
    tj_temp_c:            Optional[float]
    power_cpu_gpu_mw:     Optional[int]
    power_soc_mw:         Optional[int]
    power_total_mw:       Optional[int]
    fan_percent:          Optional[float]
    jetson_clocks:        Optional[str]
    nvp_model:            Optional[str]
    emc_util_percent:     Optional[float]
    cpu_freq_mhz:         Optional[float]
    process_cpu_percent:  float
    process_mem_mb:       float


@dataclass
class RobotObservation:
    episode:               Optional[int]   = None
    step:                  Optional[int]   = None
    # Keys: "shoulder_pan", "shoulder_lift", "elbow_flex",
    #       "wrist_flex", "wrist_roll", "gripper"
    joint_positions:       dict            = field(default_factory=dict)
    joint_velocities:      dict            = field(default_factory=dict)
    gripper_state:         Optional[str]   = None
    gripper_position:      Optional[float] = None
    # Ordered list matching motor order — for timeseries charts
    action_vector:         list            = field(default_factory=list)
    inference_latency_ms:  Optional[float] = None
    policy_fps:            Optional[float] = None
    reward:                Optional[float] = None
    done:                  Optional[bool]  = None
    # Dict of motors that exceeded MOTOR_SAFE_RANGES
    joint_violations:      dict            = field(default_factory=dict)


@dataclass
class TelemetryEvent:
    timestamp_utc:   str
    epoch_ms:        int
    event_type:      str
    session_id:      str
    host:            str
    platform:        str          = "jetson_orin_nano_super"
    system:          Optional[JetsonSystemStats] = None
    robot:           Optional[RobotObservation]  = None
    raw_message:     Optional[str]               = None


# ─────────────────────────────────────────────────────────────
# JETSON STATS COLLECTOR
# ─────────────────────────────────────────────────────────────

class JetsonStatsCollector:
    """
    Collects CPU, RAM, GPU, thermal, and power metrics from the Jetson.
    Confirmed API structure from jtop_inspect.py on Jetson Orin Nano Super:

      j.gpu["gpu"]["status"]["load"]             -> float (e.g. 0.7 = 70%)
      j.gpu["gpu"]["freq"]["cur"]                -> int Hz
      j.memory["RAM"]["tot/used/free"]           -> int KB
      j.temperature["cpu/gpu/soc0/tj"]["temp"]   -> float C
      j.power["rail"]["VDD_CPU_GPU_CV"]["power"] -> int mW
      j.power["tot"]["power"]                    -> int mW
      j.stats["Fan pwmfan0"]                     -> float %
      j.stats["jetson_clocks"]                   -> str
      j.stats["nvp model"]                       -> str
      j.stats["EMC"]                             -> float %
      j.cpu["cpu"]                               -> list of per-core dicts

    IMPORTANT: jtop is read on a dedicated background thread only.
    collect() reads from a cache — this ensures zero blocking on the
    30fps inference loop.
    """

    def __init__(self):
        self.jtop_available          = False
        self._jtop_instance          = None
        self._lerobot_process: Optional[psutil.Process] = None

        # Cached jtop result — updated by background thread every 0.5s
        self._cached_jtop            = self._empty_jtop_result()
        self._cache_lock             = threading.Lock()
        self._jtop_thread: Optional[threading.Thread] = None
        self._jtop_running           = False

        self._try_init_jtop()

    def _try_init_jtop(self):
        try:
            from jtop import jtop as JTop
            self._jtop_instance = JTop(interval=0.5)
            self._jtop_instance.start()
            timeout = 3.0
            start   = time.time()
            while not self._jtop_instance.ok():
                if time.time() - start > timeout:
                    raise RuntimeError("jtop did not become ready within 3s")
                time.sleep(0.1)
            self.jtop_available = True
            print("[Stats] jtop (jetson-stats) detected — GPU metrics enabled.")

            self._jtop_running = True
            self._jtop_thread  = threading.Thread(
                target = self._jtop_poll_loop,
                daemon = True,
                name   = "jtop-poller"
            )
            self._jtop_thread.start()

        except Exception as e:
            print(f"[Stats] jtop init failed: {e}")
            print("[Stats] Falling back to sysfs for GPU stats.")
            self._jtop_instance = None
            self.jtop_available = False

    def _jtop_poll_loop(self):
        """Dedicated background thread — the ONLY place jtop is read."""
        print("[Stats] jtop poll thread started.")
        while self._jtop_running:
            try:
                result = self._read_jtop_now()
                if any(v is not None for v in result[:4]):
                    with self._cache_lock:
                        self._cached_jtop = result
            except Exception as e:
                print(f"[Stats] jtop poll error: {e}")
            time.sleep(0.5)
        print("[Stats] jtop poll thread stopped.")

    def _read_jtop_now(self):
        """Single jtop read — called only from _jtop_poll_loop."""
        gpu_util = gpu_freq_mhz = gpu_mem_used_mb = gpu_mem_tot_mb = None
        ram_free_mb = cpu_temp = gpu_temp = soc_temp = tj_temp = None
        power_cpu_gpu_mw = power_soc_mw = power_total_mw = None
        fan_pct = jetson_clocks = nvp_model = emc_util = cpu_freq_mhz = None
        per_core_load = []

        j = self._jtop_instance
        if not j.ok():
            return self._empty_jtop_result()

        # GPU
        try:
            gpu_entry = j.gpu["gpu"]
            raw_load  = gpu_entry.get("status", {}).get("load", None)
            if isinstance(raw_load, (int, float)):
                gpu_util = round(float(raw_load), 1)
            cur_hz = gpu_entry.get("freq", {}).get("cur", None)
            if isinstance(cur_hz, (int, float)):
                gpu_freq_mhz = round(cur_hz / 1000.0, 1)
        except Exception:
            raw = j.stats.get("GPU", None)
            if isinstance(raw, (int, float)):
                gpu_util = round(float(raw) * 100.0, 1)

        # EMC
        try:
            emc_raw = j.stats.get("EMC", None)
            if isinstance(emc_raw, (int, float)):
                emc_util = float(emc_raw)
        except Exception:
            pass

        # Memory (KB)
        try:
            ram_entry = j.memory["RAM"]
            tot_kb    = ram_entry.get("tot",  None)
            used_kb   = ram_entry.get("used", None)
            free_kb   = ram_entry.get("free", None)
            if isinstance(tot_kb,  (int, float)):
                gpu_mem_tot_mb  = round(tot_kb  / 1024.0, 1)
            if isinstance(used_kb, (int, float)):
                gpu_mem_used_mb = round(used_kb / 1024.0, 1)
            if isinstance(free_kb, (int, float)):
                ram_free_mb     = round(free_kb / 1024.0, 1)
        except Exception as e:
            print(f"[Stats] Memory read failed: {e}")

        # Temperatures
        try:
            temp_data = j.temperature
            def safe_temp(key):
                entry = temp_data.get(key, {})
                if entry.get("online", False):
                    return round(float(entry["temp"]), 2)
                return None
            cpu_temp = safe_temp("cpu")
            gpu_temp = safe_temp("gpu")
            soc_temp = safe_temp("soc0")
            tj_temp  = safe_temp("tj")
        except Exception as e:
            print(f"[Stats] Temperature read failed: {e}")

        # Power (mW)
        try:
            rails        = j.power.get("rail", {})
            cpu_gpu_rail = rails.get("VDD_CPU_GPU_CV", {})
            if cpu_gpu_rail.get("online", False):
                power_cpu_gpu_mw = int(cpu_gpu_rail.get("power", 0))
            soc_rail = rails.get("VDD_SOC", {})
            if soc_rail.get("online", False):
                power_soc_mw = int(soc_rail.get("power", 0))
            tot_rail = j.power.get("tot", {})
            if tot_rail.get("online", False):
                power_total_mw = int(tot_rail.get("power", 0))
        except Exception as e:
            print(f"[Stats] Power read failed: {e}")

        # Per-core CPU
        try:
            cpu_cores     = j.cpu.get("cpu", [])
            per_core_load = [
                round(c.get("user", 0.0) + c.get("system", 0.0), 1)
                for c in cpu_cores
            ]
            if cpu_cores:
                cur_hz = cpu_cores[0].get("freq", {}).get("cur", None)
                if isinstance(cur_hz, (int, float)):
                    cpu_freq_mhz = round(cur_hz / 1000.0, 1)
        except Exception as e:
            print(f"[Stats] CPU detail read failed: {e}")

        # Fan + board state
        try:
            stats_flat    = j.stats
            fan_raw       = stats_flat.get("Fan pwmfan0", None)
            if isinstance(fan_raw, (int, float)):
                fan_pct = round(float(fan_raw), 1)
            jetson_clocks = stats_flat.get("jetson_clocks", None)
            nvp_model     = stats_flat.get("nvp model",     None)
        except Exception as e:
            print(f"[Stats] Fan/board state read failed: {e}")

        return (
            gpu_util, gpu_freq_mhz,
            gpu_mem_used_mb, gpu_mem_tot_mb, ram_free_mb,
            cpu_temp, gpu_temp, soc_temp, tj_temp,
            power_cpu_gpu_mw, power_soc_mw, power_total_mw,
            fan_pct, jetson_clocks, nvp_model,
            emc_util, cpu_freq_mhz, per_core_load
        )

    def _empty_jtop_result(self):
        return (None, None, None, None, None,
                None, None, None, None,
                None, None, None,
                None, None, None,
                None, None, [])

    def stop(self):
        """Call on shutdown to cleanly stop the jtop poll thread and close jtop."""
        self._jtop_running = False
        if self._jtop_thread:
            self._jtop_thread.join(timeout=2)
        if self._jtop_instance and self.jtop_available:
            try:
                self._jtop_instance.close()
                print("[Stats] jtop closed cleanly.")
            except Exception:
                pass

    def set_lerobot_pid(self, pid: int):
        """Register the PID to track for per-process CPU and RAM stats."""
        try:
            self._lerobot_process = psutil.Process(pid)
            print(f"[Stats] Tracking process PID: {pid}")
        except psutil.NoSuchProcess:
            self._lerobot_process = None

    def _read_sysfs_temp(self, zone_keyword: str) -> Optional[float]:
        thermal_base = Path("/sys/class/thermal")
        try:
            for zone_path in thermal_base.iterdir():
                type_file = zone_path / "type"
                temp_file = zone_path / "temp"
                if type_file.exists() and temp_file.exists():
                    if zone_keyword.lower() in type_file.read_text().strip().lower():
                        return round(int(temp_file.read_text().strip()) / 1000.0, 2)
        except Exception:
            pass
        return None

    def _read_gpu_sysfs(self) -> Optional[float]:
        load_path = Path("/sys/devices/gpu.0/load")
        if load_path.exists():
            try:
                return round(int(load_path.read_text().strip()) / 10.0, 1)
            except Exception:
                pass
        return None

    def collect(self) -> JetsonSystemStats:
        """Main collection method — psutil is instant, jtop comes from cache."""
        cpu_pct      = psutil.cpu_percent(interval=None)
        cpu_per_core = psutil.cpu_percent(interval=None, percpu=True)
        mem          = psutil.virtual_memory()
        swap         = psutil.swap_memory()
        ram_used_mb  = round(mem.used  / 1024**2, 1)
        ram_total_mb = round(mem.total / 1024**2, 1)
        swap_used_mb = round(swap.used / 1024**2, 1)

        proc_cpu = 0.0
        proc_mem = 0.0
        if self._lerobot_process:
            try:
                proc_cpu = self._lerobot_process.cpu_percent(interval=None)
                proc_mem = round(
                    self._lerobot_process.memory_info().rss / 1024**2, 1
                )
            except psutil.NoSuchProcess:
                self._lerobot_process = None

        if self.jtop_available:
            with self._cache_lock:
                cached = self._cached_jtop
            (gpu_util, gpu_freq_mhz,
             gpu_mem_used_mb, gpu_mem_tot_mb, _,
             cpu_temp, gpu_temp, soc_temp, tj_temp,
             pwr_cpu_gpu, pwr_soc, pwr_total,
             fan, jclocks, nvp,
             emc, cpu_freq, _) = cached
        else:
            gpu_util        = self._read_gpu_sysfs()
            gpu_freq_mhz    = None
            gpu_mem_used_mb = None
            gpu_mem_tot_mb  = None
            cpu_temp        = self._read_sysfs_temp("cpu")
            gpu_temp        = self._read_sysfs_temp("gpu")
            soc_temp = tj_temp = pwr_cpu_gpu = pwr_soc = pwr_total = None
            fan = jclocks = nvp = emc = cpu_freq = None

        return JetsonSystemStats(
            cpu_percent         = cpu_pct,
            cpu_per_core        = cpu_per_core,
            ram_used_mb         = ram_used_mb,
            ram_total_mb        = ram_total_mb,
            ram_percent         = round(mem.percent, 1),
            swap_used_mb        = swap_used_mb,
            gpu_util_percent    = gpu_util,
            gpu_freq_mhz        = gpu_freq_mhz,
            gpu_mem_used_mb     = gpu_mem_used_mb,
            gpu_mem_total_mb    = gpu_mem_tot_mb,
            cpu_temp_c          = cpu_temp,
            gpu_temp_c          = gpu_temp,
            soc_temp_c          = soc_temp,
            tj_temp_c           = tj_temp,
            power_cpu_gpu_mw    = pwr_cpu_gpu,
            power_soc_mw        = pwr_soc,
            power_total_mw      = pwr_total,
            fan_percent         = fan,
            jetson_clocks       = jclocks,
            nvp_model           = nvp,
            emc_util_percent    = emc,
            cpu_freq_mhz        = cpu_freq,
            process_cpu_percent = proc_cpu,
            process_mem_mb      = proc_mem,
        )


# ─────────────────────────────────────────────────────────────
# SPLUNK HEC SENDER
# ─────────────────────────────────────────────────────────────

class SplunkHECSender:
    def __init__(self):
        self._session = requests.Session()
        self._session.headers.update({
            "Authorization": f"Splunk {SPLUNK_HEC_TOKEN}",
            "Content-Type":  "application/json",
        })
        self._queue: queue.Queue = queue.Queue(maxsize=1000)
        self._send_thread = threading.Thread(
            target=self._send_loop, daemon=True, name="splunk-sender"
        )
        self._send_thread.start()
        self._sent_count   = 0
        self._failed_count = 0

    def enqueue(self, event: TelemetryEvent):
        try:
            self._queue.put_nowait(event)
        except queue.Full:
            try:
                self._queue.get_nowait()
                self._queue.put_nowait(event)
            except queue.Empty:
                pass

    def _build_hec_payload(self, event: TelemetryEvent) -> dict:
        event_dict = {k: v for k, v in asdict(event).items() if v is not None}
        return {
            "time":       event.epoch_ms / 1000.0,
            "host":       event.host,
            "source":     SPLUNK_SOURCE,
            "sourcetype": SPLUNK_SOURCETYPE,
            "index":      SPLUNK_INDEX,
            "event":      event_dict,
        }

    def _send_loop(self):
        batch = []
        while True:
            try:
                deadline = time.monotonic() + 1.0
                while len(batch) < SPLUNK_BATCH_SIZE:
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        break
                    try:
                        evt = self._queue.get(timeout=remaining)
                        batch.append(evt)
                    except queue.Empty:
                        break
                if batch:
                    self._post_batch(batch)
                    batch = []
            except Exception as e:
                print(f"[Splunk] Send loop error: {e}")
                time.sleep(1)

    def _post_batch(self, events: list):
        payloads = [self._build_hec_payload(e) for e in events]
        body     = "\n".join(json.dumps(p) for p in payloads)
        for attempt in range(3):
            try:
                resp = self._session.post(
                    SPLUNK_HEC_URL, data=body,
                    timeout=SPLUNK_TIMEOUT_SEC, verify=False,
                )
                if resp.status_code == 200:
                    self._sent_count += len(events)
                    return
                else:
                    print(f"[Splunk] HEC HTTP {resp.status_code}: {resp.text[:200]}")
            except requests.RequestException as e:
                print(f"[Splunk] POST attempt {attempt+1} failed: {e}")
                time.sleep(0.5 * (attempt + 1))
        self._failed_count += len(events)
        print(f"[Splunk] Dropped {len(events)} events after 3 retries. "
              f"Total failed: {self._failed_count}")

    @property
    def stats(self):
        return {
            "sent":   self._sent_count,
            "failed": self._failed_count,
            "queued": self._queue.qsize(),
        }


# ─────────────────────────────────────────────────────────────
# FILE LOGGER
# ─────────────────────────────────────────────────────────────

class FileLogger:
    def __init__(self, path: Path):
        self._path = path
        self._lock = threading.Lock()
        print(f"[Logger] Writing telemetry to {path}")

    def write(self, event: TelemetryEvent):
        with self._lock:
            with self._path.open("a") as f:
                f.write(json.dumps(asdict(event)) + "\n")


# ─────────────────────────────────────────────────────────────
# LEROBOT API HOOK
# ─────────────────────────────────────────────────────────────

class LeRobotAPIHook:
    """
    Drives LeRobot 0.4.3 inference directly via Python API.
    Confirmed module paths and real motor names from source inspection:

      lerobot.robots.make_robot_from_config
      lerobot.robots.so_follower.config_so_follower.SOFollowerRobotConfig
      lerobot.policies.factory.make_policy, make_pre_post_processors
      lerobot.utils.control_utils.predict_action
      lerobot.processor.make_default_processors
      lerobot.configs.policies.PreTrainedConfig
      lerobot.datasets.lerobot_dataset.LeRobotDatasetMetadata

    Real motor names: shoulder_pan, shoulder_lift, elbow_flex,
                       wrist_flex, wrist_roll, gripper

    Observation format from robot.get_observation():
      {"shoulder_pan.pos": float, ..., "front": np.array (H,W,3)}
    """

    MOTOR_NAMES = [
        "shoulder_pan",
        "shoulder_lift",
        "elbow_flex",
        "wrist_flex",
        "wrist_roll",
        "gripper",
    ]

    def __init__(
        self,
        splunk_sender:   SplunkHECSender,
        stats_collector: JetsonStatsCollector,
        session_id:      str,
        hostname:        str,
        file_logger:     Optional[FileLogger] = None,
    ):
        self._splunk           = splunk_sender
        self._stats            = stats_collector
        self._session_id       = session_id
        self._hostname         = hostname
        self._logger           = file_logger
        self._robot            = None
        self._policy           = None
        self._pre              = None
        self._post             = None
        self._robot_action_proc = None
        self._robot_obs_proc    = None
        self._episode           = 0
        self._step               = 0
        self.running             = False

    def setup(self):
        from lerobot.robots                                import make_robot_from_config
        from lerobot.robots.so_follower.config_so_follower  import SOFollowerRobotConfig
        from lerobot.cameras.opencv.configuration_opencv    import OpenCVCameraConfig
        from lerobot.policies.factory                       import make_policy, make_pre_post_processors
        from lerobot.configs.policies                       import PreTrainedConfig
        from lerobot.processor                               import make_default_processors

        # ── Build robot config — id must match existing calibration ────
        camera_cfg = OpenCVCameraConfig(
            index_or_path = ROBOT_CONFIG["cameras"]["front"]["index_or_path"],
            width         = ROBOT_CONFIG["cameras"]["front"]["width"],
            height        = ROBOT_CONFIG["cameras"]["front"]["height"],
            fps           = ROBOT_CONFIG["cameras"]["front"]["fps"],
            fourcc        = ROBOT_CONFIG["cameras"]["front"]["fourcc"],
        )

        robot_cfg = SOFollowerRobotConfig(
            id      = ROBOT_CONFIG["id"],      # loads existing calibration file
            port    = ROBOT_CONFIG["port"],
            cameras = {"front": camera_cfg},
            use_degrees = False,
        )

        print(f"[API] Connecting to robot | id={robot_cfg.id} port={robot_cfg.port}")
        self._robot = make_robot_from_config(robot_cfg)
        self._robot.connect()
        print(f"[API] Robot connected — motors: {list(self._robot.bus.motors.keys())}")

        # ── Load policy config ──────────────────────────────────────
        print(f"[API] Loading policy config from {POLICY_PATH} ...")
        policy_cfg                 = PreTrainedConfig.from_pretrained(POLICY_PATH)
        policy_cfg.pretrained_path = POLICY_PATH
        policy_cfg.device          = "cuda"

        # ── Build default processors ────────────────────────────────
        _, self._robot_action_proc, self._robot_obs_proc = make_default_processors()

        # ── Load dataset metadata for normalization stats ────────────
        print("[API] Loading dataset metadata for normalization stats...")
        ds_meta = self._load_dataset_meta(policy_cfg)

        # ── Load policy weights ───────────────────────────────────────
        print("[API] Loading policy weights...")
        self._policy = make_policy(policy_cfg, ds_meta=ds_meta)
        self._policy.eval()
        print(f"[API] Policy ready — type: {policy_cfg.type}, device: {policy_cfg.device}")

        # ── Load pre/post processors (normalization from checkpoint) ──
        print("[API] Loading pre/post processors from checkpoint...")
        self._pre, self._post = make_pre_post_processors(
            policy_cfg      = policy_cfg,
            pretrained_path = POLICY_PATH,
            dataset_stats   = None,
            preprocessor_overrides = {
                "device_processor": {"device": policy_cfg.device},
            },
        )
        print("[API] Setup complete.")

    def _load_dataset_meta(self, policy_cfg):
        """
        Loads dataset metadata needed by make_policy for normalization stats.
        Strategy 1: HuggingFace Hub metadata using repo_id from train_config.json
        Strategy 2: Local cached dataset
        Strategy 3: Minimal skeleton built from config.json feature shapes
        """
        import json as _json

        train_cfg_path = Path(POLICY_PATH) / "train_config.json"

        # Strategy 1 — Hub metadata
        try:
            if train_cfg_path.exists():
                with open(train_cfg_path) as f:
                    train_cfg = _json.load(f)
                dataset_repo_id = (
                    train_cfg.get("dataset", {}).get("repo_id")
                    or train_cfg.get("repo_id")
                )
                if dataset_repo_id:
                    print(f"[API] Loading dataset metadata from hub: {dataset_repo_id}")
                    from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata
                    meta = LeRobotDatasetMetadata(repo_id=dataset_repo_id, root=None)
                    print(f"[API] Metadata loaded from hub.")
                    return meta
        except Exception as e:
            print(f"[API] Hub metadata load failed: {e}")

        # Strategy 2 — Local cache
        try:
            local_cache = Path.home() / ".cache/huggingface/lerobot"
            possible_paths = [
                local_cache / "vladivanovic" / "eval_lego-block-front-only",
                local_cache / "vladivanovic" / "lego-block-front-only",
            ]
            for local_path in possible_paths:
                if local_path.exists():
                    print(f"[API] Loading dataset metadata from local cache: {local_path}")
                    from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata
                    meta = LeRobotDatasetMetadata(
                        repo_id = "vladivanovic/eval_lego-block-front-only",
                        root    = local_path.parent.parent,
                    )
                    return meta
        except Exception as e:
            print(f"[API] Local metadata load failed: {e}")

        # Strategy 3 — Minimal skeleton
        try:
            print("[API] Building minimal metadata from checkpoint config...")
            return self._build_minimal_meta()
        except Exception as e:
            print(f"[API] Minimal metadata build failed: {e}")
            raise RuntimeError("Could not load dataset metadata via any strategy.")

    def _build_minimal_meta(self):
        import json as _json
        config_path = Path(POLICY_PATH) / "config.json"
        with open(config_path) as f:
            cfg_data = _json.load(f)

        stats = {}
        input_features  = cfg_data.get("input_features",  {})
        output_features = cfg_data.get("output_features", {})
        for feat_key, feat_info in {**input_features, **output_features}.items():
            shape = tuple(feat_info["shape"])
            stats[feat_key] = {
                "mean": torch.zeros(shape),
                "std":  torch.ones(shape),
                "min":  torch.zeros(shape),
                "max":  torch.ones(shape),
            }

        class MinimalMeta:
            def __init__(self, stats, robot_type="so101_follower"):
                self.stats      = stats
                self.robot_type = robot_type
                self.features   = {}

        return MinimalMeta(stats=stats, robot_type="so101_follower")

    def teardown(self):
        if self._robot and self._robot.is_connected:
            try:
                self._robot.disconnect()
                print("[API] Robot disconnected.")
            except Exception as e:
                print(f"[API] Disconnect error: {e}")

    def run(self):
        from lerobot.utils.control_utils        import predict_action
        from lerobot.policies.utils             import make_robot_action
        from lerobot.utils.robot_utils          import precise_sleep
        from lerobot.utils.utils                import get_safe_torch_device
        from lerobot.datasets.utils             import build_dataset_frame, combine_feature_dicts
        from lerobot.datasets.pipeline_features import (
            aggregate_pipeline_dataset_features,
            create_initial_features,
        )
        from lerobot.utils.constants            import OBS_STR

        device = get_safe_torch_device(self._policy.config.device)

        # ── Build features dict exactly as lerobot_record.py does ─────
        features = combine_feature_dicts(
            aggregate_pipeline_dataset_features(
                pipeline         = self._robot_action_proc,
                initial_features = create_initial_features(
                    action = self._robot.action_features
                ),
                use_videos = True,
            ),
            aggregate_pipeline_dataset_features(
                pipeline         = self._robot_obs_proc,
                initial_features = create_initial_features(
                    observation = self._robot.observation_features
                ),
                use_videos = True,
            ),
        )
        print(f"[API] Features built — keys: {list(features.keys())}")
        for k, v in features.items():
            print(f"[API]   {k}: dtype={v.get('dtype','?')} shape={v.get('shape','?')}")

        self.running       = True
        steps_per_episode  = int(EPISODE_TIME_S * FPS)

        for ep in range(NUM_EPISODES):
            if not self.running:
                break

            self._episode = ep
            self._step    = 0
            print(f"\n[API] ── Episode {ep + 1}/{NUM_EPISODES} ──")
            self._emit_event(
                "episode_start",
                raw_message=f"Episode {ep} started | task: {SINGLE_TASK}"
            )

            self._policy.reset()
            self._pre.reset()
            self._post.reset()

            for step in range(steps_per_episode):
                if not self.running:
                    break

                step_start = time.perf_counter()
                self._step = step

                try:
                    raw_obs = self._robot.get_observation()
                    obs_processed = self._robot_obs_proc(raw_obs)
                    obs_frame = build_dataset_frame(features, obs_processed, prefix=OBS_STR)

                    inf_start = time.perf_counter()
                    action_values = predict_action(
                        observation   = obs_frame,
                        policy        = self._policy,
                        device        = device,
                        preprocessor  = self._pre,
                        postprocessor = self._post,
                        use_amp       = self._policy.config.use_amp,
                        task          = SINGLE_TASK,
                        robot_type    = self._robot.name,
                    )
                    latency_ms = round((time.perf_counter() - inf_start) * 1000, 2)

                    robot_action   = make_robot_action(action_values, features)
                    action_to_send = self._robot_action_proc((robot_action, obs_processed))
                    self._robot.send_action(action_to_send)

                    self._emit_step(
                        raw_obs    = raw_obs,
                        action     = robot_action,
                        latency_ms = latency_ms,
                        step       = step,
                    )

                except Exception as e:
                    import traceback
                    print(f"[API] Step error ep={ep} step={step}:\n{traceback.format_exc()}")
                    self._emit_event("error", raw_message=str(e))

                precise_sleep(max(0.0, (1.0 / FPS) - (time.perf_counter() - step_start)))

            self._emit_event(
                "episode_end",
                raw_message=f"Episode {ep} complete | steps: {self._step + 1}"
            )
            print(f"[API] Episode {ep + 1} done — {self._step + 1} steps")

            if ep < NUM_EPISODES - 1 and self.running:
                print(f"[API] Reset window: {RESET_TIME_S}s")
                time.sleep(RESET_TIME_S)

        self.running = False

    def _emit_step(self, raw_obs: dict, action: dict, latency_ms: float, step: int):
        joint_positions = {}
        for key, val in raw_obs.items():
            if key.endswith(".pos"):
                motor = key.removesuffix(".pos")
                joint_positions[motor] = round(float(val), 4)

        action_dict = {}
        for key, val in action.items():
            if key.endswith(".pos"):
                motor = key.removesuffix(".pos")
                try:
                    action_dict[motor] = round(float(val), 4)
                except Exception:
                    pass

        action_list = [action_dict.get(m, None) for m in self.MOTOR_NAMES]

        gripper_pos   = joint_positions.get("gripper", None)
        gripper_state = None
        if gripper_pos is not None:
            gripper_state = (
                "open"    if gripper_pos > 80 else
                "closed"  if gripper_pos < 20 else
                "partial"
            )

        violations = {}
        for motor, pos in joint_positions.items():
            lo, hi = MOTOR_SAFE_RANGES.get(motor, (-100, 100))
            if pos < lo or pos > hi:
                violations[motor] = {"position": pos, "min": lo, "max": hi}

        if violations:
            print(f"[API] Joint violation ep={self._episode} step={step}: {violations}")

        obs = RobotObservation(
            episode              = self._episode,
            step                 = step,
            joint_positions      = joint_positions,
            gripper_state        = gripper_state,
            gripper_position     = gripper_pos,
            action_vector        = action_list,
            inference_latency_ms = latency_ms,
            policy_fps           = round(1000.0 / latency_ms, 1) if latency_ms > 0 else None,
            joint_violations     = violations,
        )

        sys_stats = self._stats.collect()
        self._emit_event("robot_step", robot=obs, system=sys_stats)

    def _emit_event(
        self,
        event_type:  str,
        robot:       Optional[RobotObservation]  = None,
        system:      Optional[JetsonSystemStats] = None,
        raw_message: Optional[str]               = None,
    ):
        now   = datetime.now(timezone.utc)
        event = TelemetryEvent(
            timestamp_utc = now.isoformat(),
            epoch_ms      = int(now.timestamp() * 1000),
            event_type    = event_type,
            session_id    = self._session_id,
            host          = self._hostname,
            system        = system,
            robot         = robot,
            raw_message   = raw_message,
        )
        self._splunk.enqueue(event)
        if self._logger:
            self._logger.write(event)


# ─────────────────────────────────────────────────────────────
# MAIN BRIDGE
# ─────────────────────────────────────────────────────────────

class LeRobotSplunkBridge:

    def __init__(self):
        self._session_id = f"lerobot_{int(time.time())}"
        self._hostname   = socket.gethostname()

        self._stats_collector = JetsonStatsCollector()
        self._splunk          = SplunkHECSender()
        self._file_logger     = FileLogger(LOG_FILE_PATH) if LOG_TO_FILE else None

        self._running = False
        self._hook: Optional[LeRobotAPIHook] = None

        signal.signal(signal.SIGINT,  self._handle_shutdown)
        signal.signal(signal.SIGTERM, self._handle_shutdown)

    def _make_event(self, event_type: str, raw_message: str = None) -> TelemetryEvent:
        now = datetime.now(timezone.utc)
        return TelemetryEvent(
            timestamp_utc = now.isoformat(),
            epoch_ms      = int(now.timestamp() * 1000),
            event_type    = event_type,
            session_id    = self._session_id,
            host          = self._hostname,
            raw_message   = raw_message,
        )

    def _emit(self, event: TelemetryEvent):
        self._splunk.enqueue(event)
        if self._file_logger:
            self._file_logger.write(event)

    def _system_stats_loop(self):
        print(f"[Stats] Collecting system metrics every {TELEMETRY_INTERVAL_SEC}s")
        while self._running:
            try:
                stats = self._stats_collector.collect()
                now   = datetime.now(timezone.utc)
                event = TelemetryEvent(
                    timestamp_utc = now.isoformat(),
                    epoch_ms      = int(now.timestamp() * 1000),
                    event_type    = "system_stats",
                    session_id    = self._session_id,
                    host          = self._hostname,
                    system        = stats,
                )
                self._emit(event)
            except Exception as e:
                print(f"[Stats] Collection error: {e}")
            time.sleep(TELEMETRY_INTERVAL_SEC)

    def run(self):
        print("=" * 60)
        print(" LeRobot API → Splunk Telemetry Bridge")
        print(f" Session  : {self._session_id}")
        print(f" Host     : {self._hostname}")
        print(f" Splunk   : {SPLUNK_HEC_URL}")
        print(f" Index    : {SPLUNK_INDEX}")
        print(f" Policy   : {POLICY_PATH}")
        print(f" Episodes : {NUM_EPISODES} x {EPISODE_TIME_S}s @ {FPS}fps")
        print("=" * 60)

        self._emit(self._make_event("session_start",
                   raw_message=f"Session started | task: {SINGLE_TASK}"))

        # Track this Python process for CPU/RAM stats
        self._stats_collector.set_lerobot_pid(os.getpid())

        self._running = True
        stats_thread  = threading.Thread(
            target=self._system_stats_loop,
            daemon=True, name="jetson-stats"
        )
        stats_thread.start()

        self._hook = LeRobotAPIHook(
            splunk_sender   = self._splunk,
            stats_collector = self._stats_collector,
            session_id      = self._session_id,
            hostname        = self._hostname,
            file_logger     = self._file_logger,
        )

        try:
            self._hook.setup()
            self._hook.run()
        except Exception as e:
            print(f"[Main] Fatal error: {e}")
            self._emit(self._make_event("error", raw_message=str(e)))
        finally:
            self._hook.teardown()
            self._running = False

        self._emit(self._make_event("session_end", raw_message="Session complete"))
        time.sleep(2)

        print("\n" + "=" * 60)
        print(f" Splunk events | {self._splunk.stats}")
        print("=" * 60)

    def _handle_shutdown(self, signum, frame):
        print(f"\n[Main] Signal {signum} — shutting down...")
        self._running = False
        if self._hook:
            self._hook.running = False
        self._stats_collector.stop()
        time.sleep(1)
        sys.exit(0)


# ─────────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    bridge = LeRobotSplunkBridge()
    bridge.run()
