"""Centralized constants for SS_PROJECT_VCR_FRESH.

All pin assignments, speed boundaries, sensor thresholds, audio/video parameters,
and standard model paths are declared here as immutable, authoritative single sources of truth.
"""
from __future__ import annotations

from enum import Enum
from pathlib import Path


# ==============================================================================
# BCM GPIO Pin Mappings (Raspberry Pi 4 Physical Header Standard)
# ==============================================================================
# Left Motors (L298N Dual H-Bridge Channel A)
PIN_LEFT_ENA: int = 18    # PWM Speed Control (Physical Pin 12)
PIN_LEFT_IN1: int = 17    # Direction Bit 1    (Physical Pin 11)
PIN_LEFT_IN2: int = 27    # Direction Bit 2    (Physical Pin 13)

# Right Motors (L298N Dual H-Bridge Channel B)
PIN_RIGHT_ENB: int = 13   # PWM Speed Control (Physical Pin 33)
PIN_RIGHT_IN3: int = 5    # Direction Bit 3    (Physical Pin 29) — Inverted (5/22) for mirrored 4WD chassis polarity
PIN_RIGHT_IN4: int = 22   # Direction Bit 4    (Physical Pin 15) — Inverted (5/22) for mirrored 4WD chassis polarity

# Forward Ultrasonic Sensor (HC-SR04)
PIN_ULTRASONIC_TRIG: int = 23  # Pi Output Trigger (Physical Pin 16)
PIN_ULTRASONIC_ECHO: int = 24  # Pi Input Echo via 5V->3.3V Divider (Physical Pin 18)

# All GPIO Pins used by this robot
ALL_BCM_PINS = (
    PIN_LEFT_ENA,
    PIN_LEFT_IN1,
    PIN_LEFT_IN2,
    PIN_RIGHT_ENB,
    PIN_RIGHT_IN3,
    PIN_RIGHT_IN4,
    PIN_ULTRASONIC_TRIG,
    PIN_ULTRASONIC_ECHO,
)

# ==============================================================================
# Motor Controller & Motion Bounds
# ==============================================================================
PWM_FREQUENCY_HZ: int = 100       # Standardized L298N switching frequency
DEFAULT_MANUAL_SPEED: float = 0.75 # Manual teleoperation speed (0.0 to 1.0)
TURN_SPEED: float = 0.40          # Sector search & orientation turn speed (0.0 to 1.0)
APPROACH_SPEED: float = 0.40      # Target approach forward pulse speed (0.0 to 1.0)
SEARCH_SPEED: float = 0.30        # Slow in-place scan speed (0.0 to 1.0)
SPEED_RAMP_STEP: float = 0.15     # Acceleration soft-start step to prevent brownouts
MAX_SPEED: float = 1.00           # Absolute upper speed clamp
MIN_SPEED: float = 0.10           # Minimum effective motor stall threshold

# Motion Timing Bounds
DEFAULT_MANUAL_PULSE_SEC: float = 0.75    # 750 ms floor manual rollout pulse
DEFAULT_APPROACH_PULSE_SEC: float = 0.50  # 500 ms forward pulse (~5.0 cm at 40% PWM)
DEFAULT_SEARCH_TURN_PULSE_SEC: float = 1.00 # 1000 ms floor calibrated sector turn pulse (30 deg at 40% PWM)
DEFAULT_CENTERING_TURN_PULSE_SEC: float = 0.25 # 250 ms optical alignment micro turn pulse (~6-8 deg at 40% PWM)
DEFAULT_SETTLING_DELAY_SEC: float = 0.08  # 80 ms pause between pulses
DEFAULT_SEARCH_SCAN_SEC: float = 12.0     # Default open-loop 360 scan duration
DEFAULT_MISSION_TIMEOUT_SEC: float = 60.0 # 60 s maximum search & approach duration
DEFAULT_SENSOR_STALENESS_SEC: float = 0.80 # 800 ms max age for valid ultrasonic reading
DEFAULT_MIN_CONFIRMATIONS: int = 2        # 2 consecutive frames for target confirmation

# ==============================================================================
# Ultrasonic Distance Thresholds (Centimeters)
# ==============================================================================
MIN_ULTRASONIC_DISTANCE_CM: float = 2.0    # Physical sensor blind zone floor (<2cm is blind)
TARGET_STOP_DISTANCE_CM: float = 25.0     # Calibrated target standoff in 20-25 cm band
CRITICAL_OBSTACLE_DISTANCE_CM: float = 15.0 # Emergency stop threshold (all motion blocked)
BLOCKED_OBSTACLE_DISTANCE_CM: float = 25.0  # Forward motion blocked threshold
WARNING_OBSTACLE_DISTANCE_CM: float = 50.0  # Speed throttling warning threshold
MAX_ULTRASONIC_DISTANCE_CM: float = 400.0   # Maximum measurable acoustic range

# ==============================================================================
# Camera & Computer Vision Specifications
# ==============================================================================
CAMERA_DEVICE_INDEX: int = 0
CAMERA_WIDTH: int = 640
CAMERA_HEIGHT: int = 480
CAMERA_FPS: int = 15

YOLO_INPUT_WIDTH: int = 320
YOLO_INPUT_HEIGHT: int = 320
YOLO_CONFIDENCE_THRESHOLD: float = 0.20
YOLO_NMS_THRESHOLD: float = 0.45

DEADBAND_PIXELS: int = 40                 # Centering deadband (+/- 40 px around x=320)
TARGET_REACHED_AREA: int = 35000          # Telemetry only; never used as a physical distance criterion

# ==============================================================================
# Audio & Speech Recognition Specifications
# ==============================================================================
AUDIO_SAMPLE_RATE: int = 16000            # Vosk small model requirement (16 kHz)
AUDIO_CHANNELS: int = 1                   # Mono PCM
AUDIO_CHUNK_SIZE: int = 1024              # 1024 samples per buffer (64 ms at 16 kHz)
AUDIO_RECORD_SECONDS: float = 3.0
AUDIO_ENERGY_THRESHOLD: float = 0.02
AUDIO_CONFIDENCE_THRESHOLD: float = 0.60

# ==============================================================================
# Project Relative Asset Paths (Resolved dynamically by config)
# ==============================================================================
REL_PATH_PRIMARY_YOLO_ONNX = "models/object_detection/best_v2_target.onnx"
REL_PATH_FALLBACK_YOLO_ONNX = "models/object_detection/best_v1.onnx"
REL_PATH_VOSK_MODEL_DIR = "models/voice/vosk-model"
REL_PATH_CLASS_METADATA = "models/metadata/yolo_classes.json"
REL_PATH_DEFAULT_CONFIG = "config/default_config.yaml"

# ==============================================================================
# Fault Codes & Execution Modes
# ==============================================================================
class ExecutionMode(str, Enum):
    """Operational mode of the robot application."""
    REAL_HARDWARE = "REAL_HARDWARE"  # Physical Raspberry Pi with real GPIO, camera, mic
    SIMULATION = "SIMULATION"        # Hardware-free test / CI mode with deterministic mocks


class FaultCode(str, Enum):
    """Structured fault codes for system health reporting."""
    OK = "OK"
    CONFIG_INVALID = "FAULT_CONFIG_INVALID"
    ASSET_MISSING = "FAULT_ASSET_MISSING"
    HARDWARE_INIT_FAILED = "FAULT_HARDWARE_INIT_FAILED"
    SENSOR_TIMEOUT = "FAULT_SENSOR_TIMEOUT"
    SENSOR_BLIND_ZONE = "FAULT_SENSOR_BLIND_ZONE"
    CAMERA_FRAME_DROP = "FAULT_CAMERA_FRAME_DROP"
    AUDIO_STREAM_ERROR = "FAULT_AUDIO_STREAM_ERROR"
    EMERGENCY_STOP_LATCHED = "FAULT_EMERGENCY_STOP_LATCHED"
    WATCHDOG_TIMEOUT = "FAULT_WATCHDOG_TIMEOUT"
    TARGET_LOST = "FAULT_TARGET_LOST"


# ==============================================================================
# Robot Finite State Machine (FSM) States
# ==============================================================================
class RobotState(str, Enum):
    """Finite State Machine operational states for robot missions and navigation."""
    INITIALIZING = "INITIALIZING"
    IDLE_LISTENING = "IDLE_LISTENING"
    MANUAL_MOVEMENT = "MANUAL_MOVEMENT"
    SEARCHING = "SEARCHING"
    TARGET_CONFIRMED = "TARGET_CONFIRMED"
    CENTERING = "CENTERING"
    APPROACHING = "APPROACHING"
    TARGET_REACHED = "TARGET_REACHED"
    TARGET_NOT_FOUND = "TARGET_NOT_FOUND"
    CANCELLED = "CANCELLED"
    BLOCKED = "BLOCKED"
    FAULT = "FAULT"
    EMERGENCY_STOP = "EMERGENCY_STOP"

