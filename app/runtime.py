"""Runtime Lifecycle and Dependency Injection Manager for SS_PROJECT_VCR_FRESH.

Coordinates subsystem initialization, starting, stopping, and resource cleanup.
Guarantees:
- Single authoritative runtime with one owner per physical resource.
- Motors default to stopped and disabled.
- Full idempotency for start(), stop(), and cleanup().
- Complete dependency injection for hardware and perceptual adapters.
- Real hardware mode fails closed without silent fallback.
- Read-only telemetry reporting without triggering sensing or movement.
- Safe handling of client disconnects and concurrent command rejection.
"""
from __future__ import annotations

import math
import threading
import time
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

from app.command.command_parser import (
    CommandAction,
    CommandIntent,
    CommandParser,
    RobotCommand,
)
from app.config import AppConfig, ConfigurationError, get_project_root, load_config
from app.constants import (
    DEFAULT_MANUAL_SPEED,
    ExecutionMode,
    FaultCode,
    RobotState,
)
from app.hardware.interfaces import (
    AudioSourceInterface,
    CameraInterface,
    DetectorInterface,
    DistanceSensorInterface,
    MotorControllerInterface,
    ObstacleState,
    SpeechRecognizerInterface,
    UltrasonicReading,
)
from app.hardware.mock_hardware import (
    MockAudioSource,
    MockCamera,
    MockDetector,
    MockDistanceSensor,
    MockMotorController,
    MockSpeechRecognizer,
)
from app.logger import get_logger, log_fault
from app.navigation.state_machine import MissionContext, RobotStateMachine
from app.status import SubsystemStatus, SystemHealth

logger = get_logger("runtime")


class LifecycleState(str, Enum):
    """Runtime lifecycle states."""
    CREATED = "CREATED"
    INITIALIZED = "INITIALIZED"
    RUNNING = "RUNNING"
    STOPPED = "STOPPED"
    CLEANED_UP = "CLEANED_UP"


def _sanitize_float(val: Optional[float]) -> Optional[float]:
    """Ensure float value is strictly compliant with JSON specification (no NaN or Inf)."""
    if val is None:
        return None
    try:
        f = float(val)
        if math.isnan(f) or math.isinf(f):
            return None
        return round(f, 2)
    except (TypeError, ValueError):
        return None


class RobotRuntime:
    """Authoritative lifecycle and subsystem manager for the robot application."""

    def __init__(
        self,
        config: Optional[AppConfig] = None,
        motor_controller: Optional[MotorControllerInterface] = None,
        distance_sensor: Optional[DistanceSensorInterface] = None,
        camera: Optional[CameraInterface] = None,
        audio_source: Optional[AudioSourceInterface] = None,
        speech_recognizer: Optional[SpeechRecognizerInterface] = None,
        detector: Optional[DetectorInterface] = None,
    ):
        self.config = config or load_config()
        self._state = LifecycleState.CREATED
        self._start_time: Optional[float] = None
        self._active_fault: FaultCode = FaultCode.OK

        # Subsystems (Injected or constructed during initialize)
        self.motors: Optional[MotorControllerInterface] = motor_controller
        self.distance_sensor: Optional[DistanceSensorInterface] = distance_sensor
        self.camera: Optional[CameraInterface] = camera
        self.audio_source: Optional[AudioSourceInterface] = audio_source
        self.speech_recognizer: Optional[SpeechRecognizerInterface] = speech_recognizer
        self.detector: Optional[DetectorInterface] = detector

        # State Machine & Command Parser
        self.fsm: Optional[RobotStateMachine] = None
        self.command_parser: Optional[CommandParser] = None

        self._subsystem_statuses: dict[str, SubsystemStatus] = {
            "motors": SubsystemStatus.UNINITIALIZED,
            "sensor": SubsystemStatus.UNINITIALIZED,
            "camera": SubsystemStatus.UNINITIALIZED,
            "audio": SubsystemStatus.UNINITIALIZED,
            "speech": SubsystemStatus.UNINITIALIZED,
            "detector": SubsystemStatus.UNINITIALIZED,
        }

        # Background threads & concurrency control
        self._stop_event = threading.Event()
        self._command_lock = threading.Lock()
        self._is_executing_command = False
        self._speech_thread: Optional[threading.Thread] = None
        self._mission_thread: Optional[threading.Thread] = None
        self._distance_thread: Optional[threading.Thread] = None
        self._vision_thread: Optional[threading.Thread] = None

        # Cached telemetry state for read-only status reporting
        self._cached_distance_reading: Optional[UltrasonicReading] = None
        self._latest_detections: list = []
        self._latest_detections_time: float = 0.0
        self._last_raw_transcript: str = ""
        self._current_partial_transcript: str = ""
        self._last_parsed_command: Optional[RobotCommand] = None
        self.ultrasonic_protection_enabled: bool = False

    def set_ultrasonic_protection(self, enabled: bool) -> None:
        """Enable or disable ultrasonic obstacle protection."""
        self.ultrasonic_protection_enabled = bool(enabled)
        if self.fsm:
            self.fsm.ultrasonic_protection_enabled = bool(enabled)
        logger.warning(
            "Ultrasonic obstacle protection %s by operator.",
            "ENABLED" if self.ultrasonic_protection_enabled else "DISABLED",
        )

    @property
    def lifecycle_state(self) -> LifecycleState:
        return self._state

    def initialize(self) -> None:
        """Validate configuration, wire dependencies, and prepare subsystems."""
        if self._state in (LifecycleState.INITIALIZED, LifecycleState.RUNNING):
            logger.warning("initialize() called but runtime is already %s (idempotent)", self._state)
            return

        logger.info("Initializing RobotRuntime in mode: %s", self.config.mode.value)

        # 1. Validate configuration and asset files
        try:
            self.config.validate()
            root = get_project_root()
            missing = self.config.validate_asset_files(root)
            if missing:
                raise ConfigurationError(f"Missing required model asset files: {missing}")

            # 2. Initialize subsystems according to mode
            if self.config.mode == ExecutionMode.SIMULATION:
                self._init_simulation_subsystems()
            elif self.config.mode == ExecutionMode.REAL_HARDWARE:
                self._init_real_hardware_subsystems()

            # 3. Instantiate State Machine & Command Parser
            self.fsm = RobotStateMachine(
                config=self.config,
                motors=self.motors,
                distance_sensor=self.distance_sensor,
                camera=self.camera,
                detector=self.detector,
            )
            self.fsm.ultrasonic_protection_enabled = self.ultrasonic_protection_enabled

            supported_classes = (
                set(self.detector.class_names)
                if (self.detector and hasattr(self.detector, "class_names"))
                else None
            )
            self.command_parser = CommandParser(supported_classes=supported_classes)

            # Invariant: Motors default to disabled and stopped
            if self.motors:
                self.motors.stop()
                self.motors.enable(False)

            self._state = LifecycleState.INITIALIZED
            logger.info("RobotRuntime initialized successfully. Motors remain disabled and stopped.")
        except Exception as exc:
            logger.error("Subsystem initialization failed: %s. Cleaning up partial resources.", exc)
            self._cleanup_partial()
            if isinstance(exc, ConfigurationError):
                self._active_fault = (
                    FaultCode.HARDWARE_INIT_FAILED
                    if any(marker in str(exc) for marker in ("ExecutionMode", "refuses mock", "physical GPIO is required"))
                    else FaultCode.CONFIG_INVALID
                )
            else:
                self._active_fault = FaultCode.HARDWARE_INIT_FAILED
            raise

    def _cleanup_partial(self) -> None:
        """Clean up partially allocated resources if initialization fails midway."""
        logger.warning("Executing partial initialization cleanup...")
        if self.motors:
            try:
                self.motors.stop()
                self.motors.cleanup()
            except Exception:
                pass
        if self.distance_sensor:
            try:
                self.distance_sensor.stop()
            except Exception:
                pass
        if self.camera:
            try:
                self.camera.release()
            except Exception:
                pass
        if self.audio_source:
            try:
                self.audio_source.close()
            except Exception:
                pass
        if self.detector and hasattr(self.detector, "close"):
            try:
                self.detector.close()
            except Exception:
                pass
        for k in self._subsystem_statuses:
            self._subsystem_statuses[k] = SubsystemStatus.FAULT
        self._state = LifecycleState.STOPPED

    def _init_simulation_subsystems(self) -> None:
        """Instantiate deterministic mock subsystems if not already injected."""
        if self.motors is None:
            self.motors = MockMotorController(initially_enabled=False)
        self._subsystem_statuses["motors"] = SubsystemStatus.READY

        if self.distance_sensor is None:
            self.distance_sensor = MockDistanceSensor(
                warning_cm=self.config.safety.obstacle_warning_cm,
                blocked_cm=self.config.safety.obstacle_blocked_cm,
                critical_cm=self.config.safety.obstacle_critical_cm,
            )
        self._subsystem_statuses["sensor"] = SubsystemStatus.READY

        if self.camera is None:
            self.camera = MockCamera(
                width=self.config.camera.width,
                height=self.config.camera.height,
            )
        self._subsystem_statuses["camera"] = SubsystemStatus.READY

        if self.audio_source is None:
            self.audio_source = MockAudioSource(
                chunk_size=self.config.audio.chunk_size,
                sample_rate=self.config.audio.sample_rate,
            )
        self._subsystem_statuses["audio"] = SubsystemStatus.READY

        if self.speech_recognizer is None:
            self.speech_recognizer = MockSpeechRecognizer()
        self._subsystem_statuses["speech"] = SubsystemStatus.READY

        if self.detector is None:
            self.detector = MockDetector()
        self._subsystem_statuses["detector"] = SubsystemStatus.READY

    def _init_real_hardware_subsystems(self) -> None:
        """Construct real Pi drivers for missing dependencies; never substitute mocks."""
        from app.hardware.gpio_adapter import FakeGPIOAdapter, get_gpio_adapter
        from app.hardware.motors import L298NMotorController
        from app.hardware.ultrasonic import HCSR04UltrasonicSensor
        from app.vision.camera import CameraOwner
        from app.audio.audio_source import MicrophoneAudioSource
        from app.audio.vosk_recognizer import VoskSpeechRecognizer
        from app.vision.yolo_detector import ONNXYOLODetector

        # Real mode must never accept a simulation implementation through injection.
        for name, device in (("motors", self.motors), ("sensor", self.distance_sensor),
                             ("camera", self.camera), ("audio", self.audio_source),
                             ("speech", self.speech_recognizer), ("detector", self.detector)):
            if device is not None and type(device).__name__.startswith("Mock"):
                raise ConfigurationError(f"REAL_HARDWARE mode refuses mock {name} implementation ({type(device).__name__}).")

        # Motor and ultrasonic drivers must share one physical GPIO adapter.
        gpio = getattr(self.motors, "gpio", None) or getattr(self.distance_sensor, "gpio", None)
        if gpio is None and (self.motors is None or self.distance_sensor is None):
            gpio = get_gpio_adapter(ExecutionMode.REAL_HARDWARE)
        if isinstance(gpio, FakeGPIOAdapter):
            raise ConfigurationError("REAL_HARDWARE mode refuses FakeGPIOAdapter; physical GPIO is required.")

        if self.motors is None:
            self.motors = L298NMotorController(gpio=gpio, config=self.config.motor, motion_config=self.config.motion)
            self.motors.initialize()
        elif hasattr(self.motors, "initialize"):
            self.motors.initialize()
        self._subsystem_statuses["motors"] = SubsystemStatus.READY

        if self.distance_sensor is None:
            self.distance_sensor = HCSR04UltrasonicSensor(
                gpio=gpio, config=self.config.ultrasonic, safety_config=self.config.safety
            )
            self.distance_sensor.initialize()
        elif hasattr(self.distance_sensor, "initialize"):
            self.distance_sensor.initialize()
        self._subsystem_statuses["sensor"] = SubsystemStatus.READY

        if self.camera is None:
            self.camera = CameraOwner(config=self.config.camera)
        self._subsystem_statuses["camera"] = SubsystemStatus.INITIALIZING

        if self.audio_source is None:
            self.audio_source = MicrophoneAudioSource(config=self.config.audio)
        self._subsystem_statuses["audio"] = SubsystemStatus.INITIALIZING

        if self.speech_recognizer is None:
            self.speech_recognizer = VoskSpeechRecognizer(config=self.config.audio)
        if hasattr(self.speech_recognizer, "load_model"):
            self.speech_recognizer.load_model()
        self._subsystem_statuses["speech"] = SubsystemStatus.READY

        if self.detector is None:
            self.detector = ONNXYOLODetector(config=self.config.vision)
        if hasattr(self.detector, "load_model"):
            self.detector.load_model()
        self._subsystem_statuses["detector"] = SubsystemStatus.READY

    def start(self) -> None:
        """Start all sources with readiness checks; motors enable only after preflight succeeds."""
        if self._state == LifecycleState.RUNNING:
            logger.debug("start() called while already RUNNING (no-op)")
            return
        if self._state != LifecycleState.INITIALIZED:
            self.initialize()

        logger.info("Starting RobotRuntime...")
        self._start_time = time.time()
        self._stop_event.clear()
        try:
            if self.distance_sensor:
                self.distance_sensor.start()
                deadline = time.monotonic() + (3.0 if self.config.mode == ExecutionMode.REAL_HARDWARE else 0.1)
                reading = None
                while time.monotonic() < deadline:
                    try:
                        reading = self.distance_sensor.get_reading()
                    except Exception:
                        reading = None
                    if reading is not None and (reading.is_valid or reading.state == ObstacleState.OUT_OF_RANGE):
                        break
                    time.sleep(0.05)
                self._cached_distance_reading = reading
                if self.config.mode == ExecutionMode.REAL_HARDWARE and (
                    reading is None or reading.state == ObstacleState.FAULT
                ):
                    raise RuntimeError("Ultrasonic sensor did not produce a valid fresh reading; motors remain disabled.")

            if self.audio_source:
                self.audio_source.start_stream()
                self._subsystem_statuses["audio"] = SubsystemStatus.READY

            if self.camera and hasattr(self.camera, "start") and callable(getattr(self.camera, "start")):
                camera_started = self.camera.start()
                if camera_started is False:
                    raise RuntimeError("Camera failed to start; motors remain disabled.")
                # A successful thread start is not proof that frames are arriving.
                camera_deadline = time.monotonic() + 4.0
                first_frame_ready = False
                while time.monotonic() < camera_deadline:
                    try:
                        frame = self.camera.capture_frame()
                        if frame is not None:
                            first_frame_ready = True
                            break
                    except Exception:
                        pass
                    time.sleep(0.02)
                if not first_frame_ready:
                    raise RuntimeError("Camera produced no fresh valid frame; motors remain disabled.")
                self._subsystem_statuses["camera"] = SubsystemStatus.READY

            # Do not enable motors until sensor/audio/camera startup has succeeded.
            if self.motors:
                self.motors.stop()
                self.motors.enable(True)
                self.motors.stop()

            self._state = LifecycleState.RUNNING
            self._distance_thread = threading.Thread(target=self._distance_worker, daemon=True, name="DistanceWorker")
            self._mission_thread = threading.Thread(target=self._mission_worker, daemon=True, name="MissionWorker")
            self._speech_thread = threading.Thread(target=self._speech_worker, daemon=True, name="SpeechWorker")
            self._vision_thread = threading.Thread(target=self._vision_worker, daemon=True, name="VisionWorker")
            for thread in (self._distance_thread, self._mission_thread, self._speech_thread, self._vision_thread):
                thread.start()
            logger.info("RobotRuntime is now RUNNING. All required sources started; motors are stopped.")
        except Exception:
            logger.exception("Runtime startup failed; stopping and releasing partial startup resources.")
            self._stop_event.set()
            if self.motors:
                try:
                    self.motors.stop()
                    self.motors.enable(False)
                except Exception:
                    pass
            if self.audio_source:
                try:
                    self.audio_source.stop_stream()
                except Exception:
                    pass
            if self.distance_sensor:
                try:
                    self.distance_sensor.stop()
                except Exception:
                    pass
            if self.camera and hasattr(self.camera, "stop"):
                try:
                    self.camera.stop()
                except Exception:
                    pass
            self._state = LifecycleState.STOPPED
            raise

    def _distance_worker(self) -> None:
        """Continuously update cached ultrasonic reading to ensure read-only status calls."""
        while not self._stop_event.is_set():
            if self.distance_sensor and self._state == LifecycleState.RUNNING:
                try:
                    reading = self.distance_sensor.get_reading()
                    self._cached_distance_reading = reading
                except Exception as exc:
                    logger.error("Distance worker sample error; invalidating cached reading: %s", exc)
                    self._cached_distance_reading = UltrasonicReading.create_fault()
            time.sleep(0.08)  # ~12.5 Hz refresh

    def _mission_worker(self) -> None:
        """Periodically advance active search/centering/approach navigation state machine."""
        while not self._stop_event.is_set():
            if self.fsm and self.fsm.state in (
                RobotState.SEARCHING,
                RobotState.TARGET_CONFIRMED,
                RobotState.CENTERING,
                RobotState.APPROACHING,
            ):
                try:
                    self.fsm.step_mission()
                except Exception as exc:
                    logger.error("Error stepping mission: %s", exc)
            time.sleep(0.05)  # 20 Hz tick

    def _speech_worker(self) -> None:
        """Continuously drain audio chunks, decode voice transcripts, and dispatch commands."""
        while not self._stop_event.is_set():
            if not self.audio_source or not self.speech_recognizer:
                time.sleep(0.05)
                continue

            try:
                chunk = self.audio_source.read_chunk()
                if not chunk:
                    time.sleep(0.01)
                    continue

                accepted = False
                if hasattr(self.speech_recognizer, "accept_waveform"):
                    accepted = self.speech_recognizer.accept_waveform(chunk)
                elif hasattr(self.speech_recognizer, "accept_audio"):
                    accepted = self.speech_recognizer.accept_audio(chunk)

                if accepted:
                    raw_text = self.speech_recognizer.get_result()
                    if raw_text and raw_text.strip():
                        logger.info("Vosk final recognized transcript: '%s'", raw_text)
                        self._current_partial_transcript = ""
                        self.dispatch_text_command(raw_text)
                else:
                    if hasattr(self.speech_recognizer, "get_partial_result"):
                        partial = self.speech_recognizer.get_partial_result()
                        if partial and partial.strip():
                            self._current_partial_transcript = partial.strip()
            except Exception as exc:
                logger.error("Error in speech worker: %s", exc)
                time.sleep(0.05)

    def _vision_worker(self) -> None:
        """Background optical perception thread running periodic YOLO inference (~1-2 Hz)."""
        while not self._stop_event.is_set():
            if (
                self.camera
                and self.detector
                and self._state == LifecycleState.RUNNING
                and self._subsystem_statuses.get("camera") == SubsystemStatus.READY
                and self._subsystem_statuses.get("detector") == SubsystemStatus.READY
            ):
                try:
                    frame = None
                    if hasattr(self.camera, "get_latest_frame_with_timestamp"):
                        frame, _ = self.camera.get_latest_frame_with_timestamp()
                    elif hasattr(self.camera, "capture_frame"):
                        frame = self.camera.capture_frame()

                    if frame is not None and hasattr(self.detector, "detect"):
                        results = self.detector.detect(frame)
                        self._latest_detections = results
                        self._latest_detections_time = time.time()
                except Exception as exc:
                    logger.debug("Vision worker error: %s", exc)
            time.sleep(0.35)

    def dispatch_text_command(self, raw_text: str) -> Tuple[bool, Optional[RobotCommand], str]:
        """Parse and execute a text command through the authoritative state machine.
        
        Enforces concurrency rejection: if a non-emergency command is already executing,
        subsequent commands are rejected until the previous command completes.
        Emergency Stop and Stop commands always preempt ongoing commands.
        """
        if not self.command_parser:
            return False, None, "Command parser not initialized"

        cmd = self.command_parser.parse(raw_text)
        self._last_raw_transcript = raw_text
        self._current_partial_transcript = ""
        self._last_parsed_command = cmd

        # Emergency Stop and Stop override EVERYTHING immediately
        if cmd.action in (CommandAction.EMERGENCY_STOP, CommandAction.STOP):
            if self.fsm:
                success = self.fsm.handle_command(cmd)
                return success, cmd, "Emergency/Stop command executed"
            return False, cmd, "State machine not initialized"

        # Concurrency protection for non-emergency commands
        if self._is_executing_command:
            logger.warning("Rejecting command '%s': another command is currently executing.", raw_text)
            return False, cmd, "Concurrent command rejected: another command is currently executing"

        with self._command_lock:
            if not cmd.is_executable:
                logger.info("Command '%s' is not executable: %s", raw_text, cmd.error_reason)
                return False, cmd, cmd.error_reason or "Command is not executable"

            if not self.fsm:
                return False, cmd, "State machine not initialized"

            self._is_executing_command = True
            try:
                success = self.fsm.handle_command(cmd)
                status_msg = "Command accepted" if success else "Command rejected by state machine"
                return success, cmd, status_msg
            finally:
                self._is_executing_command = False

    def stop(self) -> None:
        """Immediately zero motor speeds, cancel active missions, and pause sampling workers."""
        if self._state == LifecycleState.STOPPED:
            logger.debug("stop() called while already STOPPED (no-op)")
            return

        logger.info("Stopping RobotRuntime...")
        self._stop_event.set()

        # 1. Stop motors first (highest safety priority)
        if self.motors:
            try:
                self.motors.stop()
                self.motors.enable(False)
            except Exception as exc:
                logger.error("Error stopping/disabling motors: %s", exc)

        # 2. Cancel active mission on FSM
        if self.fsm:
            try:
                self.fsm.cancel_mission("Runtime stopped")
            except Exception as exc:
                logger.error("Error cancelling FSM mission: %s", exc)

        # 3. Stop audio streaming
        if self.audio_source:
            try:
                self.audio_source.stop_stream()
            except Exception as exc:
                logger.error("Error stopping audio: %s", exc)

        # 4. Stop distance sampling
        if self.distance_sensor:
            try:
                self.distance_sensor.stop()
            except Exception as exc:
                logger.error("Error stopping distance sensor: %s", exc)

        # 5. Stop camera worker if applicable
        if hasattr(self.camera, "stop") and callable(getattr(self.camera, "stop")):
            try:
                self.camera.stop()
            except Exception as exc:
                logger.error("Error stopping camera: %s", exc)

        # 6. Join worker threads with bounded timeout
        for t in (self._distance_thread, self._mission_thread, self._speech_thread, self._vision_thread):
            if t and t.is_alive():
                t.join(timeout=0.3)

        for k in self._subsystem_statuses:
            self._subsystem_statuses[k] = SubsystemStatus.STOPPED

        self._state = LifecycleState.STOPPED
        logger.info("RobotRuntime stopped cleanly. All actuation zeroed.")

    def cleanup(self) -> None:
        """Release all hardware resources permanently. Safe and idempotent."""
        if self._state == LifecycleState.CLEANED_UP:
            logger.debug("cleanup() called while already CLEANED_UP (no-op)")
            return

        logger.info("Cleaning up RobotRuntime resources...")
        self.stop()

        if self.motors:
            try:
                self.motors.cleanup()
            except Exception as exc:
                logger.error("Error cleaning up motors: %s", exc)

        if self.camera:
            try:
                self.camera.release()
            except Exception as exc:
                logger.error("Error releasing camera: %s", exc)

        if self.audio_source:
            try:
                self.audio_source.close()
            except Exception as exc:
                logger.error("Error closing audio: %s", exc)

        self._state = LifecycleState.CLEANED_UP
        logger.info("RobotRuntime cleanup complete. Resources released.")

    def get_telemetry(self) -> Dict[str, Any]:
        """Generate a strictly read-only, non-blocking telemetry snapshot.
        
        Guarantees:
        - Never triggers ultrasonic acoustic pulses or hardware I/O.
        - Never actuates motors or moves the robot.
        - All floating point values are strictly JSON compliant (no NaN or Inf).
        - Accurately reports high-level clear messages:
          TARGET_FOUND, TARGET_NOT_FOUND, OBSTACLE_BLOCKED, CAMERA_FAILURE, DETECTOR_FAILURE, SENSOR_FAILURE.
        """
        now = time.time()
        uptime = (now - self._start_time) if (self._start_time and self._state == LifecycleState.RUNNING) else 0.0

        # Subsystems & Motor status
        speeds = self.motors.current_speeds if self.motors else (0.0, 0.0)
        motors_enabled = self.motors.is_enabled if self.motors else False
        left_spd = _sanitize_float(speeds[0]) or 0.0
        right_spd = _sanitize_float(speeds[1]) or 0.0

        if left_spd > 0 and right_spd > 0:
            cmd_state = "FORWARD"
        elif left_spd < 0 and right_spd < 0:
            cmd_state = "BACKWARD"
        elif left_spd != 0 or right_spd != 0:
            cmd_state = "TURNING"
        else:
            cmd_state = "STOPPED"

        # Ultrasonic distance (read cached value only!)
        reading = self._cached_distance_reading
        if reading is None and self.distance_sensor and hasattr(self.distance_sensor, "_current_reading"):
            # Mock-only cache access; never call a sensor method from telemetry.
            reading = getattr(self.distance_sensor, "_current_reading")
        reading_age = (now - reading.timestamp) if (reading and reading.timestamp is not None) else 999.0
        reading_fresh = bool(reading and 0.0 <= reading_age <= 0.50)
        sensor_valid = bool(reading_fresh and reading and reading.is_valid)
        dist_val = _sanitize_float(reading.distance_cm) if sensor_valid else None
        raw_dist_val = _sanitize_float(getattr(reading, "raw_distance_cm", None))
        if raw_dist_val is None and sensor_valid:
            raw_dist_val = dist_val
        filtered_dist_val = dist_val
        obs_state = reading.state.value if (reading_fresh and reading) else ObstacleState.FAULT.value
        sensor_healthy = bool(sensor_valid and reading and reading.state not in (ObstacleState.FAULT, ObstacleState.OUT_OF_RANGE))

        # Categorical 4-state indicator: CLEAR, OBSTACLE, INVALID, STALE
        if hasattr(self.distance_sensor, "get_sensor_state_label"):
            sensor_state_label = self.distance_sensor.get_sensor_state_label()
        else:
            if not reading_fresh:
                sensor_state_label = "STALE"
            elif not sensor_valid or obs_state in ("FAULT", "OUT_OF_RANGE"):
                sensor_state_label = "INVALID"
            elif obs_state in ("BLOCKED", "CRITICAL"):
                sensor_state_label = "OBSTACLE"
            else:
                sensor_state_label = "CLEAR"

        sample_count = getattr(self.distance_sensor, "valid_sample_count", 1 if sensor_valid else 0)

        # FSM and Mission context
        fsm_state_str = self.fsm.state.value if self.fsm else "UNINITIALIZED"
        estop_latched = self.fsm.is_emergency_stop_latched if self.fsm else False
        active_fault = self.fsm.active_fault.value if self.fsm else self._active_fault.value

        context: Optional[MissionContext] = self.fsm.context if self.fsm else None
        req_target = context.target_class if context else None
        target_alias = context.target_alias if context else None
        conf_count = context.target_confirmed_count if context else 0
        terminal_msg = context.terminal_message if context else None
        sectors_scanned = context.sectors_scanned if context else 0
        total_turn_sec = _sanitize_float(context.total_turn_duration_sec) if context else 0.0
        is_calibrated = context.is_360_calibrated if context else (self.config.motion.calibrated_360_duration_sec is not None)

        cur_det = None
        all_dets = []
        if context and context.target_detection:
            d = context.target_detection
            label = getattr(d, "class_name", getattr(d, "label", "unknown"))
            conf = getattr(d, "confidence", 0.0)
            box = getattr(d, "box_xyxy", getattr(d, "box", (0, 0, 0, 0)))
            box_xywh = getattr(d, "box_xywh", None)
            if box_xywh is not None:
                is_centered = abs(box_xywh[0] - (self.config.camera.width / 2.0)) <= self.config.vision.deadband_pixels
            elif hasattr(d, "is_centered"):
                is_centered = d.is_centered(self.config.camera.width, self.config.vision.deadband_pixels)
            else:
                is_centered = False

            cur_det = {
                "label": label,
                "confidence": _sanitize_float(conf),
                "box": [round(float(c), 1) for c in box],
                "is_centered": is_centered,
            }
            all_dets.append(cur_det)
        elif self._latest_detections and (time.time() - self._latest_detections_time) <= 3.0:
            for d in self._latest_detections:
                box = getattr(d, "box_xyxy", getattr(d, "box", (0, 0, 0, 0)))
                box_xywh = getattr(d, "box_xywh", None)
                if box_xywh is not None:
                    is_centered = abs(box_xywh[0] - (self.config.camera.width / 2.0)) <= self.config.vision.deadband_pixels
                else:
                    is_centered = False
                all_dets.append({
                    "label": getattr(d, "class_name", getattr(d, "label", "unknown")),
                    "confidence": _sanitize_float(getattr(d, "confidence", 0.0)),
                    "box": [round(float(c), 1) for c in box],
                    "is_centered": is_centered,
                })
            if all_dets:
                cur_det = max(all_dets, key=lambda x: x["confidence"])

        # Safety block reason
        block_reason = None
        if estop_latched:
            block_reason = "Emergency stop latched"
        elif self.ultrasonic_protection_enabled and (not sensor_valid or obs_state in ("FAULT", "OUT_OF_RANGE")):
            block_reason = "Distance sensor unavailable/invalid; movement must remain inhibited"
        elif self.ultrasonic_protection_enabled and (fsm_state_str == "BLOCKED" or obs_state in ("BLOCKED", "CRITICAL")):
            block_reason = f"Forward obstacle detected ({dist_val} cm, {obs_state})"
        elif not motors_enabled:
            block_reason = "Motors software disabled"

        # Audio / Vosk listening state
        mic_active = self._subsystem_statuses.get("audio") == SubsystemStatus.READY
        vosk_active = self._subsystem_statuses.get("speech") == SubsystemStatus.READY
        if self._state == LifecycleState.RUNNING and mic_active and vosk_active:
            listening_state = "LISTENING"
        else:
            listening_state = "STOPPED"

        # Evaluate Clear Status Messages
        messages: List[str] = []
        if fsm_state_str == "TARGET_REACHED" or (
            fsm_state_str in ("TARGET_CONFIRMED", "APPROACHING") and cur_det is not None
        ):
            messages.append("TARGET_FOUND")

        if fsm_state_str == "TARGET_NOT_FOUND":
            messages.append("TARGET_NOT_FOUND")

        if self.ultrasonic_protection_enabled and (fsm_state_str == "BLOCKED" or obs_state in ("BLOCKED", "CRITICAL")):
            messages.append("OBSTACLE_BLOCKED")

        if not self.ultrasonic_protection_enabled:
            messages.append("ULTRASONIC_PROTECTION_DISABLED")

        cam_op = getattr(self.camera, "is_operational", True) if self.camera else False
        if not cam_op or active_fault == FaultCode.CAMERA_FRAME_DROP.value:
            messages.append("CAMERA_FAILURE")

        det_ready = self._subsystem_statuses.get("detector") == SubsystemStatus.READY
        if not det_ready:
            messages.append("DETECTOR_FAILURE")

        if not sensor_healthy or active_fault in (
            FaultCode.SENSOR_TIMEOUT.value,
            FaultCode.SENSOR_BLIND_ZONE.value,
        ):
            messages.append("SENSOR_FAILURE")

        # Serialized command snapshot
        last_cmd_dict = None
        if self._last_parsed_command:
            c = self._last_parsed_command
            last_cmd_dict = {
                "action": c.action.value,
                "intent": c.intent.value,
                "target_class": c.target_class,
                "target_alias": c.target_alias,
                "speed": _sanitize_float(c.speed),
                "is_executable": c.is_executable,
                "error_reason": c.error_reason,
            }

        is_healthy = (
            active_fault == FaultCode.OK.value
            and not estop_latched
            and self._state in (LifecycleState.INITIALIZED, LifecycleState.RUNNING)
            and sensor_healthy
            and obs_state not in (ObstacleState.CRITICAL.value, ObstacleState.FAULT.value, ObstacleState.OUT_OF_RANGE.value)
        )

        return {
            "health": {
                "is_healthy": is_healthy,
                "mode": self.config.mode.value,
                "lifecycle_state": self._state.value,
                "uptime_sec": _sanitize_float(uptime),
                "active_fault": active_fault,
            },
            "speech": {
                "listening_state": listening_state,
                "mic_active": mic_active,
                "vosk_active": vosk_active,
                "last_transcript": getattr(self, "_current_partial_transcript", "") or self._last_raw_transcript,
                "is_partial": bool(getattr(self, "_current_partial_transcript", "")),
                "last_command": last_cmd_dict,
            },
            "fsm": {
                "state": fsm_state_str,
                "emergency_stop_latched": estop_latched,
                "last_terminal_result": terminal_msg,
            },
            "target": {
                "requested_target": req_target,
                "target_alias": target_alias,
                "confirmed_count": conf_count,
                "current_detection": cur_det,
                "detections": all_dets,
            },
            "search": {
                "sectors_scanned": sectors_scanned,
                "search_progress": f"Sector {sectors_scanned}" if sectors_scanned > 0 else "Idle",
                "total_turn_duration_sec": total_turn_sec,
                "is_360_calibrated": is_calibrated,
                "rotation_calibration_status": "CALIBRATED" if is_calibrated else "UNCALIBRATED",
            },
            "ultrasonic": {
                "distance_cm": dist_val,
                "raw_distance_cm": raw_dist_val,
                "filtered_distance_cm": filtered_dist_val,
                "is_valid": sensor_valid,
                "obstacle_state": obs_state,
                "sensor_healthy": sensor_healthy,
                "sensor_state": sensor_state_label,
                "protection_enabled": self.ultrasonic_protection_enabled,
                "reading_age_sec": _sanitize_float(reading_age),
                "sample_count": sample_count,
            },
            "motor": {
                "left_speed": left_spd,
                "right_speed": right_spd,
                "enabled": motors_enabled,
                "command_state": cmd_state,
                "safety_block_reason": block_reason,
            },
            "messages": messages,
            "subsystems": {k: v.value for k, v in self._subsystem_statuses.items()},
        }

    def get_health(self) -> SystemHealth:
        """Generate an immutable snapshot of current system health and telemetry.
        
        Preserves backwards-compatibility with Phase 2 test suite.
        """
        uptime = (time.time() - self._start_time) if self._start_time and self._state == LifecycleState.RUNNING else 0.0
        speeds = self.motors.current_speeds if self.motors else (0.0, 0.0)
        motors_enabled = self.motors.is_enabled if self.motors else False

        # Strictly read the cache only: health/status calls must not trigger a
        # new acoustic pulse or touch hardware.
        reading = self._cached_distance_reading
        reading_fresh = bool(reading and 0.0 <= (time.time() - reading.timestamp) <= 0.50)
        dist_val = reading.distance_cm if (reading_fresh and reading and reading.is_valid) else None
        obs_state = reading.state if (reading_fresh and reading) else ObstacleState.FAULT

        is_healthy = (
            self._active_fault == FaultCode.OK
            and self._state in (LifecycleState.INITIALIZED, LifecycleState.RUNNING)
            and reading_fresh and reading is not None and reading.is_valid
            and obs_state not in (ObstacleState.CRITICAL, ObstacleState.FAULT, ObstacleState.OUT_OF_RANGE)
        )

        return SystemHealth(
            mode=self.config.mode,
            is_healthy=is_healthy,
            active_fault=self._active_fault,
            subsystems=dict(self._subsystem_statuses),
            motors_enabled=motors_enabled,
            current_speeds=speeds,
            last_distance_cm=dist_val,
            obstacle_state=obs_state,
            uptime_sec=uptime,
        )
