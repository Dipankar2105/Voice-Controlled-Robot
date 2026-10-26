"""Typed, validated configuration engine for SS_PROJECT_VCR_FRESH.

Features:
- Dynamically resolves project root independently of the current shell working directory.
- Enforces explicit ExecutionMode (REAL_HARDWARE vs SIMULATION) with zero silent fallbacks.
- Verifies GPIO pin uniqueness, electrical bounds, sensor threshold monotonicity, and speeds.
- Validates model assets and class metadata consistency.
"""
from __future__ import annotations

import json
import os
import platform
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Union
import yaml

from app.constants import (
    ALL_BCM_PINS,
    AUDIO_CHANNELS,
    AUDIO_CHUNK_SIZE,
    AUDIO_CONFIDENCE_THRESHOLD,
    AUDIO_ENERGY_THRESHOLD,
    AUDIO_RECORD_SECONDS,
    AUDIO_SAMPLE_RATE,
    BLOCKED_OBSTACLE_DISTANCE_CM,
    CAMERA_DEVICE_INDEX,
    CAMERA_FPS,
    CAMERA_HEIGHT,
    CAMERA_WIDTH,
    CRITICAL_OBSTACLE_DISTANCE_CM,
    DEADBAND_PIXELS,
    DEFAULT_MANUAL_SPEED,
    DEFAULT_MANUAL_PULSE_SEC,
    ExecutionMode,
    MAX_SPEED,
    MAX_ULTRASONIC_DISTANCE_CM,
    MIN_ULTRASONIC_DISTANCE_CM,
    PIN_LEFT_ENA,
    PIN_LEFT_IN1,
    PIN_LEFT_IN2,
    PIN_RIGHT_ENB,
    PIN_RIGHT_IN3,
    PIN_RIGHT_IN4,
    PIN_ULTRASONIC_ECHO,
    PIN_ULTRASONIC_TRIG,
    PWM_FREQUENCY_HZ,
    REL_PATH_CLASS_METADATA,
    REL_PATH_DEFAULT_CONFIG,
    REL_PATH_FALLBACK_YOLO_ONNX,
    REL_PATH_PRIMARY_YOLO_ONNX,
    REL_PATH_VOSK_MODEL_DIR,
    SEARCH_SPEED,
    SPEED_RAMP_STEP,
    TARGET_REACHED_AREA,
    TARGET_STOP_DISTANCE_CM,
    TURN_SPEED,
    WARNING_OBSTACLE_DISTANCE_CM,
    YOLO_CONFIDENCE_THRESHOLD,
    YOLO_INPUT_HEIGHT,
    YOLO_INPUT_WIDTH,
    YOLO_NMS_THRESHOLD,
)


class ConfigurationError(ValueError):
    """Raised when configuration violates physical, electrical, or architectural constraints."""


def get_project_root() -> Path:
    """Return the absolute path to SS_Project_VCR_FRESH root independently of current working directory."""
    return Path(__file__).resolve().parent.parent


@dataclass(frozen=True)
class MotorDriverConfig:
    driver_type: str = "L298N"
    pwm_frequency_hz: int = PWM_FREQUENCY_HZ
    left_ena_pin: int = PIN_LEFT_ENA
    left_in1_pin: int = PIN_LEFT_IN1
    left_in2_pin: int = PIN_LEFT_IN2
    right_enb_pin: int = PIN_RIGHT_ENB
    right_in3_pin: int = PIN_RIGHT_IN3
    right_in4_pin: int = PIN_RIGHT_IN4

    @property
    def all_pins(self) -> List[int]:
        return [
            self.left_ena_pin,
            self.left_in1_pin,
            self.left_in2_pin,
            self.right_enb_pin,
            self.right_in3_pin,
            self.right_in4_pin,
        ]


@dataclass(frozen=True)
class UltrasonicConfig:
    sensor_type: str = "HC-SR04"
    trig_pin: int = PIN_ULTRASONIC_TRIG
    echo_pin: int = PIN_ULTRASONIC_ECHO
    max_distance_cm: float = MAX_ULTRASONIC_DISTANCE_CM
    min_distance_cm: float = MIN_ULTRASONIC_DISTANCE_CM
    timeout_sec: float = 0.06

    @property
    def all_pins(self) -> List[int]:
        return [self.trig_pin, self.echo_pin]


@dataclass(frozen=True)
class MotionConfig:
    manual_speed: float = DEFAULT_MANUAL_SPEED
    turn_speed: float = TURN_SPEED
    approach_speed: float = TURN_SPEED
    search_speed: float = SEARCH_SPEED
    speed_ramp_step: float = SPEED_RAMP_STEP
    max_speed: float = MAX_SPEED
    search_scan_duration_sec: float = 12.0
    calibrated_360_duration_sec: Optional[float] = None
    manual_pulse_sec: float = DEFAULT_MANUAL_PULSE_SEC


@dataclass(frozen=True)
class SafetyConfig:
    obstacle_warning_cm: float = WARNING_OBSTACLE_DISTANCE_CM
    obstacle_blocked_cm: float = BLOCKED_OBSTACLE_DISTANCE_CM
    obstacle_critical_cm: float = CRITICAL_OBSTACLE_DISTANCE_CM
    target_stop_distance_cm: float = TARGET_STOP_DISTANCE_CM
    watchdog_timeout_sec: float = 2.0
    max_consecutive_target_loss: int = 3


@dataclass(frozen=True)
class CameraConfig:
    device_index: int = CAMERA_DEVICE_INDEX
    width: int = CAMERA_WIDTH
    height: int = CAMERA_HEIGHT
    fps: int = CAMERA_FPS


@dataclass(frozen=True)
class VisionConfig:
    model_path: str = REL_PATH_PRIMARY_YOLO_ONNX
    fallback_model_path: Optional[str] = REL_PATH_FALLBACK_YOLO_ONNX
    input_width: int = YOLO_INPUT_WIDTH
    input_height: int = YOLO_INPUT_HEIGHT
    confidence_threshold: float = YOLO_CONFIDENCE_THRESHOLD
    nms_threshold: float = YOLO_NMS_THRESHOLD
    deadband_pixels: int = DEADBAND_PIXELS
    target_reached_area: int = TARGET_REACHED_AREA
    detect_frequency_hz: int = 6
    class_metadata_path: str = REL_PATH_CLASS_METADATA


@dataclass(frozen=True)
class AudioConfig:
    model_dir: str = REL_PATH_VOSK_MODEL_DIR
    sample_rate: int = AUDIO_SAMPLE_RATE
    channels: int = AUDIO_CHANNELS
    chunk_size: int = AUDIO_CHUNK_SIZE
    record_seconds: float = AUDIO_RECORD_SECONDS
    energy_threshold: float = AUDIO_ENERGY_THRESHOLD
    confidence_threshold: float = AUDIO_CONFIDENCE_THRESHOLD


@dataclass(frozen=True)
class CommunicationConfig:
    host: str = "0.0.0.0"
    port: int = 8000
    ws_port: int = 8001
    enable_cors: bool = False


@dataclass(frozen=True)
class AppConfig:
    robot_name: str = "VoiceRobot4WD"
    version: str = "2.0.0"
    mode: ExecutionMode = ExecutionMode.SIMULATION
    motor: MotorDriverConfig = field(default_factory=MotorDriverConfig)
    ultrasonic: UltrasonicConfig = field(default_factory=UltrasonicConfig)
    motion: MotionConfig = field(default_factory=MotionConfig)
    safety: SafetyConfig = field(default_factory=SafetyConfig)
    camera: CameraConfig = field(default_factory=CameraConfig)
    vision: VisionConfig = field(default_factory=VisionConfig)
    audio: AudioConfig = field(default_factory=AudioConfig)
    communication: CommunicationConfig = field(default_factory=CommunicationConfig)

    def validate(self) -> None:
        """Enforce all physical, electrical, architectural, and safety rules."""
        # 1. Execution Mode Rules (No silent mock fallback in REAL_HARDWARE mode)
        if self.mode == ExecutionMode.REAL_HARDWARE:
            is_linux = sys.platform.startswith("linux")
            machine = platform.machine().lower()
            is_arm = any(arch in machine for arch in ("arm", "aarch64"))
            if not (is_linux and is_arm):
                raise ConfigurationError(
                    f"ExecutionMode.REAL_HARDWARE was explicitly requested, but this host is "
                    f"'{sys.platform} ({platform.machine()})', not a Raspberry Pi ARM Linux board. "
                    "Silent fallback to simulation is prohibited. Configure mode: SIMULATION."
                )

        # 2. Pin Uniqueness & Physical Range
        all_pins = self.motor.all_pins + self.ultrasonic.all_pins
        if len(all_pins) != len(set(all_pins)):
            seen = set()
            duplicates = set()
            for p in all_pins:
                if p in seen:
                    duplicates.add(p)
                seen.add(p)
            raise ConfigurationError(
                f"Duplicate GPIO pin assignment detected: {sorted(list(duplicates))}. "
                "Each pin must serve exactly one hardware function."
            )

        for pin in all_pins:
            if not isinstance(pin, int) or not (0 <= pin <= 40):
                raise ConfigurationError(
                    f"Invalid BCM GPIO pin number: {pin}. Must be an integer between 0 and 40."
                )

        # 3. PWM Frequency Range
        if not (10 <= self.motor.pwm_frequency_hz <= 20000):
            raise ConfigurationError(
                f"PWM frequency {self.motor.pwm_frequency_hz} Hz is outside safe operating range (10-20000 Hz)."
            )

        # 4. Motion Speeds
        speeds = {
            "manual_speed": self.motion.manual_speed,
            "turn_speed": self.motion.turn_speed,
            "approach_speed": self.motion.approach_speed,
            "search_speed": self.motion.search_speed,
            "max_speed": self.motion.max_speed,
        }
        for name, val in speeds.items():
            if not (0.0 < val <= 1.0):
                raise ConfigurationError(
                    f"Speed parameter '{name}'={val} violates valid range (0.0, 1.0]."
                )
        if self.motion.manual_speed > self.motion.max_speed:
            raise ConfigurationError(
                f"manual_speed ({self.motion.manual_speed}) cannot exceed max_speed ({self.motion.max_speed})."
            )
        if not (0.01 <= self.motion.manual_pulse_sec <= 2.0):
            raise ConfigurationError(
                f"manual_pulse_sec ({self.motion.manual_pulse_sec}) must be within [0.01, 2.0] seconds."
            )

        # 5. Ultrasonic Safety Threshold Monotonicity
        if not (
            self.ultrasonic.min_distance_cm
            <= self.safety.obstacle_critical_cm
            < self.safety.obstacle_blocked_cm
            < self.safety.target_stop_distance_cm
            < self.safety.obstacle_warning_cm
            < self.ultrasonic.max_distance_cm
        ):
            raise ConfigurationError(
                "Obstacle safety thresholds violate mandatory monotonic ordering: "
                f"min({self.ultrasonic.min_distance_cm}) <= "
                f"critical({self.safety.obstacle_critical_cm}) < "
                f"blocked({self.safety.obstacle_blocked_cm}) < "
                f"target_stop({self.safety.target_stop_distance_cm}) < "
                f"warning({self.safety.obstacle_warning_cm}) < "
                f"max({self.ultrasonic.max_distance_cm}). Target stop must be outside the blocked zone."
            )

        # 6. Optical & Vision Specifications
        if self.camera.width <= 0 or self.camera.height <= 0 or self.camera.fps <= 0:
            raise ConfigurationError(
                f"Invalid camera dimensions or framerate: {self.camera.width}x{self.camera.height} @ {self.camera.fps} FPS"
            )

        if self.vision.input_width != 320 or self.vision.input_height != 320:
            raise ConfigurationError(
                f"Vision input dimensions ({self.vision.input_width}x{self.vision.input_height}) "
                "must match YOLO ONNX contract 320x320."
            )

        if not (0.0 < self.vision.confidence_threshold < 1.0):
            raise ConfigurationError(
                f"Vision confidence threshold {self.vision.confidence_threshold} out of bounds (0.0, 1.0)."
            )

        if not (0.0 < self.vision.nms_threshold < 1.0):
            raise ConfigurationError(
                f"Vision NMS threshold {self.vision.nms_threshold} out of bounds (0.0, 1.0)."
            )

        # 7. Audio Specification
        if self.audio.sample_rate != 16000 or self.audio.channels != 1:
            raise ConfigurationError(
                f"Vosk speech model strictly requires 16000 Hz mono audio; got {self.audio.sample_rate} Hz, {self.audio.channels} ch."
            )

        # 8. Communication Ports
        for name, port in (("port", self.communication.port), ("ws_port", self.communication.ws_port)):
            if not (1024 <= port <= 65535):
                raise ConfigurationError(
                    f"Communication port {name}={port} is outside unprivileged TCP range (1024-65535)."
                )

        # 9. Asset Files & Class Metadata Validation
        root = get_project_root()
        missing_assets = self.validate_asset_files(root)
        if missing_assets:
            if self.mode == ExecutionMode.REAL_HARDWARE:
                raise ConfigurationError(
                    f"Missing required model asset(s) in REAL_HARDWARE mode: {missing_assets}"
                )

        # 10. Class Metadata Consistency
        meta_file = root / self.vision.class_metadata_path
        if meta_file.is_file():
            try:
                with open(meta_file, "r", encoding="utf-8") as f:
                    meta = json.load(f)
                v2_meta = meta.get("models", {}).get("best_v2_target.onnx", {})
                classes = v2_meta.get("classes", [])
                if len(classes) != 26:
                    raise ConfigurationError(
                        f"Class metadata in {self.vision.class_metadata_path} defines {len(classes)} classes; "
                        "best_v2_target.onnx strictly requires 26 classes."
                    )
                if len(set(classes)) != len(classes):
                    raise ConfigurationError("Class metadata contains duplicate class names.")
            except json.JSONDecodeError as exc:
                raise ConfigurationError(f"Corrupt class metadata JSON file: {exc}")

    def validate_asset_files(self, base_dir: Path) -> List[str]:
        """Verify presence and non-zero size of required model assets."""
        missing = []

        # ONNX Model
        model_path = base_dir / self.vision.model_path
        if not (model_path.is_file() and model_path.stat().st_size > 0):
            missing.append(f"Primary ONNX model missing or empty: {model_path}")

        # Vosk language model files
        vosk_dir = base_dir / self.audio.model_dir
        required_vosk = [
            "am/final.mdl",
            "conf/model.conf",
            "graph/Gr.fst",
            "graph/HCLr.fst",
            "ivector/final.ie",
        ]
        for rel in required_vosk:
            vf = vosk_dir / rel
            if not (vf.is_file() and vf.stat().st_size > 0):
                missing.append(f"Vosk component missing or empty: {vf}")

        return missing


def load_config_from_dict(d: Dict[str, Any]) -> AppConfig:
    """Build and validate AppConfig from a nested dictionary."""
    robot = d.get("robot", {})
    hw = d.get("hardware", {})
    md = hw.get("motor_driver", {})
    lm = md.get("left_motor", {})
    rm = md.get("right_motor", {})
    us = hw.get("ultrasonic_sensor", {})
    motion = d.get("motion", {})
    safety = d.get("safety", {})
    cam = d.get("camera", {})
    vis = d.get("vision", {})
    audio = d.get("audio", {})
    comm = d.get("communication", {})

    mode_str = str(robot.get("mode", "SIMULATION" if robot.get("simulated", False) else "REAL_HARDWARE")).upper()
    try:
        mode = ExecutionMode(mode_str)
    except ValueError:
        raise ConfigurationError(
            f"Invalid execution mode '{mode_str}'. Must be REAL_HARDWARE or SIMULATION."
        )

    motor_cfg = MotorDriverConfig(
        driver_type=str(md.get("driver_type", "L298N")),
        pwm_frequency_hz=int(md.get("pwm_frequency_hz", PWM_FREQUENCY_HZ)),
        left_ena_pin=int(lm.get("ena_pin", PIN_LEFT_ENA)),
        left_in1_pin=int(lm.get("in1_pin", PIN_LEFT_IN1)),
        left_in2_pin=int(lm.get("in2_pin", PIN_LEFT_IN2)),
        right_enb_pin=int(rm.get("enb_pin", PIN_RIGHT_ENB)),
        right_in3_pin=int(rm.get("in3_pin", PIN_RIGHT_IN3)),
        right_in4_pin=int(rm.get("in4_pin", PIN_RIGHT_IN4)),
    )

    ultrasonic_cfg = UltrasonicConfig(
        sensor_type=str(us.get("sensor_type", "HC-SR04")),
        trig_pin=int(us.get("trig_pin", PIN_ULTRASONIC_TRIG)),
        echo_pin=int(us.get("echo_pin", PIN_ULTRASONIC_ECHO)),
        max_distance_cm=float(us.get("max_distance_cm", MAX_ULTRASONIC_DISTANCE_CM)),
        min_distance_cm=float(us.get("min_distance_cm", MIN_ULTRASONIC_DISTANCE_CM)),
        timeout_sec=float(us.get("timeout_sec", 0.06)),
    )

    motion_cfg = MotionConfig(
        manual_speed=float(motion.get("manual_speed", DEFAULT_MANUAL_SPEED)),
        turn_speed=float(motion.get("turn_speed", TURN_SPEED)),
        approach_speed=float(motion.get("approach_speed", TURN_SPEED)),
        search_speed=float(motion.get("search_speed", SEARCH_SPEED)),
        speed_ramp_step=float(motion.get("speed_ramp_step", SPEED_RAMP_STEP)),
        max_speed=float(motion.get("max_speed", MAX_SPEED)),
        search_scan_duration_sec=float(motion.get("search_scan_duration_sec", 12.0)),
        calibrated_360_duration_sec=float(motion["calibrated_360_duration_sec"])
        if motion.get("calibrated_360_duration_sec") is not None
        else None,
        manual_pulse_sec=float(motion.get("manual_pulse_sec", DEFAULT_MANUAL_PULSE_SEC)),
    )

    safety_cfg = SafetyConfig(
        obstacle_warning_cm=float(safety.get("obstacle_warning_cm", WARNING_OBSTACLE_DISTANCE_CM)),
        obstacle_blocked_cm=float(safety.get("obstacle_blocked_cm", BLOCKED_OBSTACLE_DISTANCE_CM)),
        obstacle_critical_cm=float(safety.get("obstacle_critical_cm", CRITICAL_OBSTACLE_DISTANCE_CM)),
        target_stop_distance_cm=float(safety.get("target_stop_distance_cm", TARGET_STOP_DISTANCE_CM)),
        watchdog_timeout_sec=float(safety.get("watchdog_timeout_sec", 2.0)),
        max_consecutive_target_loss=int(safety.get("max_consecutive_target_loss", 3)),
    )

    cam_cfg = CameraConfig(
        device_index=int(cam.get("device_index", CAMERA_DEVICE_INDEX)),
        width=int(cam.get("width", CAMERA_WIDTH)),
        height=int(cam.get("height", CAMERA_HEIGHT)),
        fps=int(cam.get("fps", CAMERA_FPS)),
    )

    vis_cfg = VisionConfig(
        model_path=str(vis.get("model_path", REL_PATH_PRIMARY_YOLO_ONNX)),
        fallback_model_path=str(vis["fallback_model_path"]) if vis.get("fallback_model_path") else None,
        input_width=int(vis.get("input_width", YOLO_INPUT_WIDTH)),
        input_height=int(vis.get("input_height", YOLO_INPUT_HEIGHT)),
        confidence_threshold=float(vis.get("confidence_threshold", YOLO_CONFIDENCE_THRESHOLD)),
        nms_threshold=float(vis.get("nms_threshold", YOLO_NMS_THRESHOLD)),
        deadband_pixels=int(vis.get("deadband_pixels", DEADBAND_PIXELS)),
        target_reached_area=int(vis.get("target_reached_area", TARGET_REACHED_AREA)),
        detect_frequency_hz=int(vis.get("detect_frequency_hz", 6)),
        class_metadata_path=str(vis.get("class_metadata_path", REL_PATH_CLASS_METADATA)),
    )

    audio_cfg = AudioConfig(
        model_dir=str(audio.get("model_dir", REL_PATH_VOSK_MODEL_DIR)),
        sample_rate=int(audio.get("sample_rate", AUDIO_SAMPLE_RATE)),
        channels=int(audio.get("channels", AUDIO_CHANNELS)),
        chunk_size=int(audio.get("chunk_size", AUDIO_CHUNK_SIZE)),
        record_seconds=float(audio.get("record_seconds", AUDIO_RECORD_SECONDS)),
        energy_threshold=float(audio.get("energy_threshold", AUDIO_ENERGY_THRESHOLD)),
        confidence_threshold=float(audio.get("confidence_threshold", AUDIO_CONFIDENCE_THRESHOLD)),
    )

    comm_cfg = CommunicationConfig(
        host=str(comm.get("host", "0.0.0.0")),
        port=int(comm.get("port", 8000)),
        ws_port=int(comm.get("ws_port", 8001)),
        enable_cors=bool(comm.get("enable_cors", True)),
    )

    app_cfg = AppConfig(
        robot_name=str(robot.get("name", "VoiceRobot4WD")),
        version=str(robot.get("version", "2.0.0")),
        mode=mode,
        motor=motor_cfg,
        ultrasonic=ultrasonic_cfg,
        motion=motion_cfg,
        safety=safety_cfg,
        camera=cam_cfg,
        vision=vis_cfg,
        audio=audio_cfg,
        communication=comm_cfg,
    )
    app_cfg.validate()
    return app_cfg


def load_config(config_path: Optional[Union[str, Path]] = None) -> AppConfig:
    """Load configuration from a YAML file, defaulting to config/default_config.yaml."""
    root = get_project_root()
    if config_path is None:
        target_path = root / REL_PATH_DEFAULT_CONFIG
    else:
        p = Path(config_path)
        target_path = p if p.is_absolute() else (root / p)

    if not target_path.is_file():
        raise FileNotFoundError(f"Configuration file not found: {target_path}")

    with open(target_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}

    return load_config_from_dict(data)
