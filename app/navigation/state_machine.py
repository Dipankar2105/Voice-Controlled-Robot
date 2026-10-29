"""Robot Finite State Machine (FSM) and Autonomous Navigation Engine for SS_PROJECT_VCR_FRESH.

Features:
- Deterministic state machine managing missions, manual teleoperation, safety stops, and autonomous search.
- Integrates existing CameraOwner, L298NMotorController, HCSR04DistanceSensor, and ONNXYOLODetector
  without duplicating hardware ownership.
- Multi-frame target confirmation before rotation to eliminate unneeded spins.
- Bounded sector search with explicit tracking of calibrated vs uncalibrated 360 coverage.
- Bounded turn pulses for optical centering with configurable deadband.
- Forward approach gated by ultrasonic proximity: stops on BLOCKED/CRITICAL obstacles, sensor fault, or target reached.
- Never infers physical distance solely from bounding box area.
- Latched EMERGENCY_STOP requiring explicit reset.
- Responsive stop and cancellation overriding ongoing motion or navigation.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import math
import time
from typing import Any, Dict, List, Optional, Tuple
import numpy as np

from app.command.command_parser import (
    CommandAction,
    CommandIntent,
    RobotCommand,
)
from app.config import AppConfig, load_config
from app.constants import (
    APPROACH_SPEED,
    CAMERA_WIDTH,
    DEADBAND_PIXELS,
    DEFAULT_APPROACH_PULSE_SEC,
    DEFAULT_CENTERING_TURN_PULSE_SEC,
    DEFAULT_MANUAL_SPEED,
    DEFAULT_MIN_CONFIRMATIONS,
    DEFAULT_MISSION_TIMEOUT_SEC,
    DEFAULT_SEARCH_SCAN_SEC,
    DEFAULT_SEARCH_TURN_PULSE_SEC,
    DEFAULT_SENSOR_STALENESS_SEC,
    DEFAULT_SETTLING_DELAY_SEC,
    FaultCode,
    RobotState,
    SEARCH_SPEED,
    TURN_SPEED,
)
from app.hardware.interfaces import (
    CameraInterface,
    DetectionResult,
    DetectorInterface,
    DistanceSensorInterface,
    MotorControllerInterface,
    ObstacleState,
    UltrasonicReading,
)
from app.logger import get_logger, log_fault

logger = get_logger("fsm")


@dataclass
class MissionContext:
    """State tracking context for an active search and approach mission."""
    target_class: Optional[str] = None
    target_alias: Optional[str] = None
    target_detection: Optional[DetectionResult] = None
    target_confirmed_count: int = 0
    consecutive_target_loss: int = 0
    sectors_scanned: int = 0
    total_turn_duration_sec: float = 0.0
    is_360_calibrated: bool = False
    last_distance_cm: Optional[float] = None
    start_time: float = field(default_factory=time.time)
    terminal_message: Optional[str] = None


class RobotStateMachine:
    """Authoritative finite state machine coordinating robot perception, safety, and motion."""

    def __init__(
        self,
        config: Optional[AppConfig] = None,
        motors: Optional[MotorControllerInterface] = None,
        distance_sensor: Optional[DistanceSensorInterface] = None,
        camera: Optional[CameraInterface] = None,
        detector: Optional[DetectorInterface] = None,
    ):
        self.config = config or load_config()
        self.motors = motors
        self.distance_sensor = distance_sensor
        self.camera = camera
        self.detector = detector

        self._state = RobotState.INITIALIZING
        self._active_fault = FaultCode.OK
        self._emergency_stop_latched = False
        self._context: Optional[MissionContext] = None
        self._last_state_change_time = time.time()
        self._min_confirmations = 2
        self._consecutive_camera_drops = 0

        # Parameters from config
        self.deadband_pixels = self.config.vision.deadband_pixels
        self.camera_width = self.config.camera.width
        self.target_stop_distance_cm = self.config.safety.target_stop_distance_cm
        self.max_consecutive_target_loss = self.config.safety.max_consecutive_target_loss
        self.search_scan_sec = self.config.motion.search_scan_duration_sec
        self.calibrated_360_duration_sec = self.config.motion.calibrated_360_duration_sec
        self.ultrasonic_protection_enabled: bool = True

        # Motion & safety pulse parameters from config (no magic numbers)
        self.search_turn_pulse_sec = getattr(self.config.motion, "search_turn_pulse_sec", DEFAULT_SEARCH_TURN_PULSE_SEC)
        self.centering_turn_pulse_sec = getattr(self.config.motion, "centering_turn_pulse_sec", DEFAULT_CENTERING_TURN_PULSE_SEC)
        self.approach_pulse_sec = getattr(self.config.motion, "approach_pulse_sec", DEFAULT_APPROACH_PULSE_SEC)
        self.settling_delay_sec = getattr(self.config.motion, "settling_delay_sec", DEFAULT_SETTLING_DELAY_SEC)
        self.mission_timeout_sec = getattr(self.config.safety, "mission_timeout_sec", DEFAULT_MISSION_TIMEOUT_SEC)
        self.sensor_staleness_sec = getattr(self.config.safety, "sensor_staleness_sec", DEFAULT_SENSOR_STALENESS_SEC)
        self._min_confirmations = getattr(self.config.vision, "min_confirmations", DEFAULT_MIN_CONFIRMATIONS)
        self.turn_speed = getattr(self.config.motion, "turn_speed", TURN_SPEED)
        self.approach_speed = getattr(self.config.motion, "approach_speed", APPROACH_SPEED)

        # Bounded scan sector count
        max_scan_sec = self.calibrated_360_duration_sec or self.search_scan_sec
        self.max_scan_sectors = max(4, int(math.ceil(max_scan_sec / max(0.05, self.search_turn_pulse_sec))))

        # Transition to IDLE_LISTENING upon successful construction
        self._transition_to(RobotState.IDLE_LISTENING, "System initialized and ready.")

    @property
    def state(self) -> RobotState:
        """Current FSM state."""
        return self._state

    @property
    def active_fault(self) -> FaultCode:
        """Current active fault code."""
        return self._active_fault

    @property
    def is_emergency_stop_latched(self) -> bool:
        """True if emergency stop is latched and requires explicit clearing."""
        return self._emergency_stop_latched

    @property
    def context(self) -> Optional[MissionContext]:
        """Telemetry snapshot of current mission context."""
        return self._context

    def _transition_to(self, new_state: RobotState, reason: str = "") -> None:
        """Safely transition to a new FSM state with logging and invariant enforcement."""
        old_state = self._state
        self._state = new_state
        self._last_state_change_time = time.time()
        logger.info("FSM State Transition: %s -> %s (%s)", old_state.value, new_state.value, reason)

        # Invariant: If entering any non-motion state, stop motors immediately
        if new_state in (
            RobotState.IDLE_LISTENING,
            RobotState.TARGET_REACHED,
            RobotState.TARGET_NOT_FOUND,
            RobotState.CANCELLED,
            RobotState.BLOCKED,
            RobotState.FAULT,
            RobotState.EMERGENCY_STOP,
        ):
            if self.motors:
                try:
                    self.motors.stop()
                except Exception as exc:
                    logger.error("Error stopping motors during state transition: %s", exc)

    def trigger_emergency_stop(self, reason: str = "Emergency stop commanded") -> None:
        """Latch an emergency stop condition immediately halting all motor actuation."""
        self._emergency_stop_latched = True
        self._active_fault = FaultCode.EMERGENCY_STOP_LATCHED
        if self.motors:
            try:
                self.motors.stop()
                self.motors.enable(False)
            except Exception as exc:
                logger.error("Error in emergency stop motor shutdown: %s", exc)

        self._transition_to(RobotState.EMERGENCY_STOP, reason)

    def clear_emergency_stop(self) -> bool:
        """Explicitly clear the emergency stop latch and return to IDLE_LISTENING."""
        if not self._emergency_stop_latched:
            return True

        # Clearing the latch is not permission to re-enable motion if the
        # forward proximity sensor is missing, invalid, or still blocked, UNLESS
        # ultrasonic protection is explicitly disabled by operator.
        if self.ultrasonic_protection_enabled:
            if not self.distance_sensor:
                logger.warning("E-stop clear refused: no distance sensor is attached.")
                return False
            try:
                reading = self.distance_sensor.get_reading()
            except Exception as exc:
                logger.warning("E-stop clear refused: sensor read failed: %s", exc)
                return False
            if (reading is None or not reading.is_valid
                    or reading.state in (ObstacleState.BLOCKED, ObstacleState.CRITICAL)
                    or (reading.distance_cm is not None and reading.distance_cm <= self.config.safety.obstacle_blocked_cm)):
                logger.warning("E-stop clear refused until sensor is valid and the blocked zone is clear.")
                return False
        else:
            logger.warning("E-stop clear check: ultrasonic protection is DISABLED (operator override).")

        logger.info("Clearing emergency stop latch after sensor safety precheck...")
        self._emergency_stop_latched = False
        self._active_fault = FaultCode.OK
        if self.motors:
            try:
                self.motors.stop()
                self.motors.enable(True)
                self.motors.stop()
            except Exception as exc:
                logger.error("Error re-enabling motors after estop: %s", exc)
                self._emergency_stop_latched = True
                self._active_fault = FaultCode.EMERGENCY_STOP_LATCHED
                return False

        self._transition_to(RobotState.IDLE_LISTENING, "Emergency stop cleared by operator after sensor precheck.")
        return True

    def cancel_mission(self, reason: str = "Mission cancelled by operator") -> None:
        """Cancel ongoing search or navigation mission and safely stop motors."""
        if self._state in (
            RobotState.SEARCHING,
            RobotState.TARGET_CONFIRMED,
            RobotState.CENTERING,
            RobotState.APPROACHING,
            RobotState.MANUAL_MOVEMENT,
        ):
            if self._context:
                self._context.terminal_message = reason
            self._transition_to(RobotState.CANCELLED, reason)

    def handle_command(self, cmd: RobotCommand) -> bool:
        """Process a parsed voice or API command according to FSM rules.
        
        Returns True if command was accepted and acted upon; False if rejected.
        """
        if not cmd.is_executable:
            logger.debug("Ignoring unexecutable command: %s (%s)", cmd.raw_transcript, cmd.error_reason)
            return False

        # 1. Emergency Stop overrides EVERYTHING
        if cmd.action == CommandAction.EMERGENCY_STOP:
            self.trigger_emergency_stop(f"Voice E-Stop: '{cmd.raw_transcript}'")
            return True

        # Invariant: While emergency stop is latched, refuse all non-clear commands
        if self._emergency_stop_latched:
            logger.warning("Command '%s' rejected: Emergency stop is LATCHED.", cmd.raw_transcript)
            return False

        # 2. Immediate Active Stop
        if cmd.action == CommandAction.STOP:
            if self._state != RobotState.IDLE_LISTENING:
                self._transition_to(RobotState.IDLE_LISTENING, f"Stop commanded: '{cmd.raw_transcript}'")
            return True

        # 3. Cancellation Commands
        if cmd.action in (CommandAction.CANCEL_SEARCH, CommandAction.STOP_SCANNING):
            self.cancel_mission(f"Cancelled via command: '{cmd.raw_transcript}'")
            return True

        # 4. Resume Listening
        if cmd.action == CommandAction.RESUME_LISTENING:
            if self._state in (
                RobotState.CANCELLED,
                RobotState.TARGET_REACHED,
                RobotState.TARGET_NOT_FOUND,
                RobotState.BLOCKED,
                RobotState.FAULT,
            ):
                if self._state == RobotState.FAULT:
                    if self._active_fault == FaultCode.SENSOR_TIMEOUT and self.ultrasonic_protection_enabled:
                        if self.distance_sensor:
                            try:
                                r = self.distance_sensor.get_reading()
                                if r is None or not r.is_valid:
                                    logger.warning("Resume listening rejected: distance sensor still failing.")
                                    return False
                            except Exception:
                                logger.warning("Resume listening rejected: distance sensor read failed.")
                                return False
                    elif self._active_fault == FaultCode.CAMERA_FRAME_DROP:
                        if self.camera and hasattr(self.camera, "is_operational") and not self.camera.is_operational:
                            logger.warning("Resume listening rejected: camera hardware still degraded.")
                            return False

                self._active_fault = FaultCode.OK
                self._transition_to(RobotState.IDLE_LISTENING, "Resumed listening.")
                return True
            return True

        # 5. Manual Directional Movement Commands
        if cmd.intent == CommandIntent.MOVEMENT:
            # If robot is in FAULT due to TARGET_LOST, clear the optical fault and accept manual intervention
            if self._state == RobotState.FAULT and self._active_fault in (FaultCode.TARGET_LOST, FaultCode.OK):
                self._active_fault = FaultCode.OK
                self._transition_to(RobotState.IDLE_LISTENING, "Recovered from TARGET_LOST via manual command.")

            if self._state not in (
                RobotState.IDLE_LISTENING,
                RobotState.MANUAL_MOVEMENT,
                RobotState.BLOCKED,
                RobotState.CANCELLED,
                RobotState.TARGET_REACHED,
                RobotState.TARGET_NOT_FOUND,
            ):
                logger.warning("Manual movement rejected: Robot is in state %s", self._state.value)
                return False
            return self._execute_manual_movement(cmd.action, cmd.speed or DEFAULT_MANUAL_SPEED)

        # 6. Autonomous Search Command
        if cmd.intent == CommandIntent.SEARCH and cmd.action == CommandAction.SEARCH_TARGET:
            # If robot is in FAULT due to TARGET_LOST, clear the optical fault and accept new search
            if self._state == RobotState.FAULT and self._active_fault in (FaultCode.TARGET_LOST, FaultCode.OK):
                self._active_fault = FaultCode.OK
                self._transition_to(RobotState.IDLE_LISTENING, "Recovered from TARGET_LOST via new search.")

            if self._state not in (
                RobotState.IDLE_LISTENING,
                RobotState.CANCELLED,
                RobotState.TARGET_REACHED,
                RobotState.TARGET_NOT_FOUND,
                RobotState.BLOCKED,
            ):
                logger.warning("Search command rejected: Robot is currently in state %s", self._state.value)
                return False
            return self.start_search_mission(cmd.target_class, cmd.target_alias)

        return True

    def _execute_manual_movement(self, action: CommandAction, speed: float) -> bool:
        """Execute a short bounded manual movement pulse subject to ultrasonic safety."""
        # Safety gate forward movement
        if action == CommandAction.MOVE_FORWARD:
            safe = self._is_forward_path_safe()
            if not safe:
                self._transition_to(RobotState.BLOCKED, "Forward motion blocked by ultrasonic obstacle.")
                return False

        self._transition_to(RobotState.MANUAL_MOVEMENT, f"Manual motion: {action.value}")

        if not self.motors:
            self._transition_to(RobotState.IDLE_LISTENING, "Mock movement completed.")
            return True

        if hasattr(self.motors, "is_enabled") and not self.motors.is_enabled:
            try:
                self.motors.enable(True)
            except Exception as exc:
                logger.error("Failed to enable motors: %s", exc)

        raw_pulse = getattr(self.config.motion, "manual_pulse_sec", 1.25) if hasattr(self.config, "motion") else 1.25
        pulse_sec = max(0.05, min(2.00, float(raw_pulse)))
        turn_spd = float(getattr(self.config.motion, "turn_speed", 0.40))
        is_turn = action in (CommandAction.TURN_LEFT, CommandAction.TURN_RIGHT)
        motion_duration = 2.50 if is_turn else pulse_sec
        try:
            if action == CommandAction.MOVE_FORWARD:
                self.motors.set_speeds(speed, speed)
            elif action == CommandAction.MOVE_BACKWARD:
                self.motors.set_speeds(-speed, -speed)
            elif action == CommandAction.TURN_LEFT:
                self.motors.set_speeds(-turn_spd, turn_spd)
            elif action == CommandAction.TURN_RIGHT:
                self.motors.set_speeds(turn_spd, -turn_spd)

            # Responsive sleep checking for emergency stop / active stop preemption / dynamic obstacle
            t_end = time.time() + motion_duration
            last_obs_check = time.time()
            while time.time() < t_end:
                if self._emergency_stop_latched or self._state != RobotState.MANUAL_MOVEMENT:
                    break
                now_check = time.time()
                if (
                    action == CommandAction.MOVE_FORWARD
                    and self.ultrasonic_protection_enabled
                    and (now_check - last_obs_check >= 0.05)
                ):
                    last_obs_check = now_check
                    if not self._is_forward_path_safe():
                        self._transition_to(RobotState.BLOCKED, "Forward motion blocked by ultrasonic obstacle.")
                        break
                time.sleep(0.02)
        finally:
            if self.motors:
                try:
                    self.motors.stop()
                except Exception as exc:
                    logger.error("Error stopping motors after manual pulse: %s", exc)
        # STOP / EMERGENCY_STOP may preempt this blocking pulse from another
        # thread. Never overwrite that newer state when the pulse timer expires.
        if self._emergency_stop_latched or self._state != RobotState.MANUAL_MOVEMENT:
            return False
        self._transition_to(RobotState.IDLE_LISTENING, "Manual pulse completed.")
        return True

    def _is_forward_path_safe(self) -> bool:
        """Check forward ultrasonic sensor before forward movement."""
        if not self.ultrasonic_protection_enabled:
            logger.warning("Ultrasonic obstacle protection is DISABLED: forward path check bypassed.")
            return True

        if not self.distance_sensor:
            logger.error("Forward motion refused: no distance sensor is attached.")
            self._active_fault = FaultCode.SENSOR_TIMEOUT
            return False

        try:
            reading: UltrasonicReading = self.distance_sensor.get_reading()
        except Exception as exc:
            logger.error("Failed to read ultrasonic sensor: %s", exc)
            self._active_fault = FaultCode.SENSOR_TIMEOUT
            return False

        if reading is None or not reading.is_valid:
            logger.warning("Ultrasonic reading invalid: state=%s", reading.state if reading else None)
            return False

        ts = getattr(reading, "timestamp", 0.0)
        if ts > 0:
            age = time.time() - ts
            if age > self.sensor_staleness_sec:
                logger.warning("Forward motion refused: ultrasonic reading is stale (age=%.3fs > %.3fs)", age, self.sensor_staleness_sec)
                self._active_fault = FaultCode.SENSOR_TIMEOUT
                return False

        # Fail closed on BLOCKED/CRITICAL state or a numeric reading inside the blocked zone.
        if (
            reading.state in (ObstacleState.BLOCKED, ObstacleState.CRITICAL)
            or (reading.distance_cm is not None and reading.distance_cm <= self.config.safety.obstacle_blocked_cm)
        ):
            logger.warning("Forward motion refused: obstacle at %.1f cm (%s)", reading.distance_cm or 0.0, reading.state.value)
            return False

        return True

    def start_search_mission(self, target_class: Optional[str], target_alias: Optional[str] = None) -> bool:
        """Initialize and launch an autonomous search mission for the requested class."""
        if not target_class:
            logger.error("Cannot start search mission without a target class.")
            return False

        # Validate that detector supports this target
        if self.detector and hasattr(self.detector, "has_class"):
            if not self.detector.has_class(target_class):
                logger.error("Target class '%s' is not supported by current detector model.", target_class)
                return False

        if self.motors:
            try:
                self.motors.stop()
            except Exception as exc:
                logger.error("Error stopping motors on search mission start: %s", exc)

        self._context = MissionContext(
            target_class=target_class,
            target_alias=target_alias,
            is_360_calibrated=(self.calibrated_360_duration_sec is not None),
        )

        logger.info(
            "Starting search mission for target '%s' (alias '%s', 360 calibrated=%s)...",
            target_class,
            target_alias,
            self._context.is_360_calibrated,
        )

        self._transition_to(RobotState.SEARCHING, f"Commencing search for '{target_class}'")
        return True

    def step_mission(self) -> RobotState:
        """Execute one iteration of the autonomous search, centering, and approach mission loop.
        
        Returns current FSM state after the step.
        """
        if self._emergency_stop_latched:
            return RobotState.EMERGENCY_STOP

        if self._context is None:
            return self._state

        target_class = self._context.target_class
        if not target_class:
            self._transition_to(RobotState.FAULT, "Missing target class in context.")
            return self._state

        # Invariant: Mission timeout check
        elapsed = time.time() - self._context.start_time
        if elapsed >= self.mission_timeout_sec:
            self._context.terminal_message = (
                f"Mission timeout ({elapsed:.1f}s >= {self.mission_timeout_sec:.1f}s) "
                f"exceeded without completing approach."
            )
            self._transition_to(RobotState.TARGET_NOT_FOUND, self._context.terminal_message)
            return self._state

        # ===================================================================
        # STATE: SEARCHING (In-Place Check & Bounded Sector Turns)
        # ===================================================================
        if self._state == RobotState.SEARCHING:
            # Settle before capture to avoid motion blur
            time.sleep(self.settling_delay_sec)

            det = self._detect_target_in_frame(target_class)
            if det is not None:
                self._context.target_confirmed_count += 1
                logger.info("Candidate target '%s' detected (count %d/%d).", target_class, self._context.target_confirmed_count, self._min_confirmations)
                if self._context.target_confirmed_count >= self._min_confirmations:
                    self._context.target_detection = det
                    self._transition_to(RobotState.TARGET_CONFIRMED, f"Target '{target_class}' confirmed over {self._context.target_confirmed_count} frames.")
                    return self._state
                return self._state

            # Target not seen in this sector: reset confirmation counter
            self._context.target_confirmed_count = 0

            # Check scan completion (both duration and sector count bounded)
            scan_limit_sec = self.calibrated_360_duration_sec or self.search_scan_sec
            if (
                self._context.total_turn_duration_sec >= scan_limit_sec
                or self._context.sectors_scanned >= self.max_scan_sectors
            ):
                calib_note = "calibrated 360" if self._context.is_360_calibrated else "uncalibrated sector scan limit"
                self._context.terminal_message = (
                    f"Full {calib_note} scan completed ({self._context.sectors_scanned}/{self.max_scan_sectors} pulses, "
                    f"{self._context.total_turn_duration_sec:.1f}s) without finding '{target_class}'."
                )
                self._transition_to(RobotState.TARGET_NOT_FOUND, self._context.terminal_message)
                return self._state

            # Search in-place rotation: ultrasonic clearance is not required to observe or rotate.
            if self.distance_sensor:
                try:
                    turn_reading = self.distance_sensor.get_reading()
                    if turn_reading and turn_reading.is_valid:
                        self._context.last_distance_cm = turn_reading.distance_cm
                except Exception:
                    pass

            # Execute short bounded turn pulse into next sector
            turn_pulse_sec = self.search_turn_pulse_sec
            sector_idx = self._context.sectors_scanned + 1
            logger.info(
                "Search pulse %d/%d active: LEFT @ %.0f%% PWM for %.2fs (cumulative: %.2fs/%.2fs)",
                sector_idx,
                self.max_scan_sectors,
                self.turn_speed * 100.0,
                turn_pulse_sec,
                self._context.total_turn_duration_sec + turn_pulse_sec,
                scan_limit_sec,
            )
            self._turn_pulse(self.turn_speed, turn_pulse_sec, direction="left")
            self._context.total_turn_duration_sec += turn_pulse_sec
            self._context.sectors_scanned += 1
            return self._state

        # ===================================================================
        # STATE: TARGET_CONFIRMED -> Transition to CENTERING
        # ===================================================================
        if self._state == RobotState.TARGET_CONFIRMED:
            self._context.consecutive_target_loss = 0
            self._transition_to(RobotState.CENTERING, f"Centering optical alignment on '{target_class}'.")
            return self._state

        # ===================================================================
        # STATE: CENTERING (Align bounding-box center with frame center)
        # ===================================================================
        if self._state == RobotState.CENTERING:
            if self.distance_sensor:
                try:
                    centering_reading = self.distance_sensor.get_reading()
                    if centering_reading and centering_reading.is_valid:
                        self._context.last_distance_cm = centering_reading.distance_cm
                except Exception:
                    pass

            time.sleep(self.settling_delay_sec)
            det = self._detect_target_in_frame(target_class)

            if det is None:
                self._context.consecutive_target_loss += 1
                logger.warning(
                    "Target '%s' not detected during centering (consecutive loss: %d/%d).",
                    target_class,
                    self._context.consecutive_target_loss,
                    self.max_consecutive_target_loss,
                )
                if self._context.consecutive_target_loss >= self.max_consecutive_target_loss:
                    self._active_fault = FaultCode.TARGET_LOST
                    self._transition_to(RobotState.FAULT, f"Target '{target_class}' lost during centering.")
                return self._state

            self._context.consecutive_target_loss = 0
            self._context.target_detection = det

            frame_center_x = self.camera_width / 2.0  # 320 for 640x480
            target_cx = det.box_xywh[0]
            offset = target_cx - frame_center_x

            # Check deadband
            if abs(offset) <= self.deadband_pixels:
                logger.info("Target centered within deadband (cx=%.1f, offset=%.1fpx).", target_cx, offset)
                self._transition_to(RobotState.APPROACHING, f"Target '{target_class}' centered; commencing approach.")
                return self._state

            # Pulse turn to correct optical error (micro-pulse from config)
            turn_pulse_sec = self.centering_turn_pulse_sec
            direction = "left" if offset < 0 else "right"
            logger.info(
                "Target not centered (cx=%.1f, offset=%.1fpx, deadband=±%dpx). Commanding %s centering turn (%.2fs @ %.0f%% PWM).",
                target_cx,
                offset,
                self.deadband_pixels,
                direction.upper(),
                turn_pulse_sec,
                self.turn_speed * 100.0,
            )
            self._turn_pulse(self.turn_speed, turn_pulse_sec, direction=direction)
            return self._state

        # ===================================================================
        # STATE: APPROACHING (Ultrasonic-Gated Forward Pulses)
        # ===================================================================
        if self._state == RobotState.APPROACHING:
            # 1. Forward Ultrasonic Safety Gate (Mandatory for forward motion)
            distance_cm = None
            if self.ultrasonic_protection_enabled:
                if not self.distance_sensor:
                    if self.motors:
                        self.motors.stop()
                    self._active_fault = FaultCode.SENSOR_TIMEOUT
                    self._transition_to(RobotState.FAULT, "No distance sensor is attached during approach.")
                    return self._state

                try:
                    reading = self.distance_sensor.get_reading()
                except Exception as exc:
                    logger.error("Distance sensor error during approach: %s", exc)
                    if self.motors:
                        self.motors.stop()
                    self._active_fault = FaultCode.SENSOR_TIMEOUT
                    self._transition_to(RobotState.FAULT, "Ultrasonic timeout during approach.")
                    return self._state

                if reading is None or not reading.is_valid:
                    logger.warning("Invalid distance reading during approach.")
                    if self.motors:
                        self.motors.stop()
                    self._active_fault = FaultCode.SENSOR_TIMEOUT
                    self._transition_to(RobotState.FAULT, "Invalid ultrasonic measurement during approach.")
                    return self._state

                ts = getattr(reading, "timestamp", 0.0)
                if ts > 0:
                    age = time.time() - ts
                    if age > self.sensor_staleness_sec:
                        if self.motors:
                            self.motors.stop()
                        self._active_fault = FaultCode.SENSOR_TIMEOUT
                        self._transition_to(RobotState.FAULT, f"Stale distance reading (age {age:.2f}s > {self.sensor_staleness_sec:.2f}s) during approach.")
                        return self._state

                distance_cm = reading.distance_cm
                self._context.last_distance_cm = distance_cm
            elif self.distance_sensor:
                try:
                    reading = self.distance_sensor.get_reading()
                    if reading and reading.is_valid:
                        distance_cm = reading.distance_cm
                        self._context.last_distance_cm = distance_cm
                except Exception:
                    pass

            if distance_cm is None:
                if self.motors:
                    self.motors.stop()
                self._active_fault = FaultCode.SENSOR_TIMEOUT
                self._transition_to(RobotState.FAULT, "Cannot approach target: ultrasonic distance reading is None.")
                return self._state

            # 2. Critical obstacle safety floor (< 15 cm)
            if distance_cm <= self.config.safety.obstacle_critical_cm:
                if self.motors:
                    self.motors.stop()
                self._transition_to(RobotState.BLOCKED, f"Obstacle inside critical zone at {distance_cm:.1f} cm.")
                return self._state

            # 3. Observe fresh camera frame before deciding motion or arrival
            time.sleep(self.settling_delay_sec)
            det = self._detect_target_in_frame(target_class)

            if det is None:
                self._context.consecutive_target_loss += 1
                if self.motors:
                    self.motors.stop()
                logger.warning(
                    "Target '%s' not detected on approach (reacquisition %d/%d); robot stationary.",
                    target_class,
                    self._context.consecutive_target_loss,
                    self.max_consecutive_target_loss,
                )
                if distance_cm <= self.config.safety.obstacle_blocked_cm:
                    self._transition_to(RobotState.BLOCKED, f"Obstacle detected ahead at {distance_cm:.1f} cm (target absent).")
                    return self._state
                if self._context.consecutive_target_loss >= self.max_consecutive_target_loss:
                    self._active_fault = FaultCode.TARGET_LOST
                    self._transition_to(RobotState.FAULT, f"Target '{target_class}' lost during approach.")
                return self._state

            self._context.consecutive_target_loss = 0
            self._context.target_detection = det

            # 4. Check Optical Alignment (Centering deadband)
            frame_center_x = self.camera_width / 2.0
            offset = det.box_xywh[0] - frame_center_x
            if abs(offset) > self.deadband_pixels:
                if self.motors:
                    self.motors.stop()
                logger.info("Target drifted out of centering deadband (offset=%.1fpx > %dpx); re-centering.", offset, self.deadband_pixels)
                self._transition_to(RobotState.CENTERING, f"Target '{target_class}' drifted out of centering deadband.")
                return self._state

            # 5. Refresh ultrasonic reading right before motion/arrival decision
            if self.ultrasonic_protection_enabled:
                try:
                    fresh_reading = self.distance_sensor.get_reading()
                    if fresh_reading is not None and fresh_reading.is_valid:
                        distance_cm = fresh_reading.distance_cm
                        self._context.last_distance_cm = distance_cm
                    else:
                        if self.motors:
                            self.motors.stop()
                        self._active_fault = FaultCode.SENSOR_TIMEOUT
                        self._transition_to(RobotState.FAULT, "Ultrasonic reading invalid before approach actuation.")
                        return self._state
                except Exception as exc:
                    if self.motors:
                        self.motors.stop()
                    self._active_fault = FaultCode.SENSOR_TIMEOUT
                    self._transition_to(RobotState.FAULT, f"Ultrasonic read error before actuation: {exc}")
                    return self._state

            # 6. Target standoff arrival & obstacle safety policy
            blocked_threshold = self.config.safety.obstacle_blocked_cm
            standoff_threshold = self.target_stop_distance_cm

            # A distance inside the blocked zone is always a safety block.
            # When standoff > blocked, reaching the blocked zone is a block, not arrival.
            # When standoff == blocked, only strictly below blocked is a block.
            is_blocked = (distance_cm < blocked_threshold) or (
                standoff_threshold > blocked_threshold and distance_cm <= blocked_threshold
            )

            if is_blocked:
                if self.motors:
                    self.motors.stop()
                self._transition_to(RobotState.BLOCKED, f"Obstacle detected ahead at {distance_cm:.1f} cm.")
                return self._state

            if distance_cm <= standoff_threshold:
                if self.motors:
                    self.motors.stop()
                self._context.terminal_message = (
                    f"Requested target centred; ultrasonic distance {distance_cm:.1f} cm "
                    f"is within safe stop threshold {standoff_threshold:.1f} cm."
                )
                self._transition_to(RobotState.TARGET_REACHED, self._context.terminal_message)
                return self._state

            # 7. Hard forward-motion safety threshold at 25 cm
            # At or below 25 cm, no new forward pulse may start!
            if distance_cm <= 25.0:
                if self.motors:
                    self.motors.stop()
                return self._state

            # 8. Clear path (> 25 cm and > standoff): execute one bounded forward pulse
            forward_pulse_sec = self.approach_pulse_sec
            self._forward_pulse(self.approach_speed, forward_pulse_sec)
            return self._state

        return self._state

    def _is_valid_detection(self, det: Any) -> bool:
        """Validate detection result structure and bounding box invariants."""
        if det is None:
            return False
        box = getattr(det, "box_xywh", None)
        if not isinstance(box, (tuple, list)) or len(box) < 4:
            return False
        try:
            cx, cy, w, h = box[0], box[1], box[2], box[3]
            if not all(isinstance(v, (int, float)) and math.isfinite(v) for v in (cx, cy, w, h)):
                return False
            if w <= 0 or h <= 0:
                return False
            if cx < -100 or cx > self.camera_width + 100:
                return False
        except Exception:
            return False
        return True

    def _detect_target_in_frame(self, target_class: str) -> Optional[DetectionResult]:
        """Capture fresh optical frame and run YOLO inference for the target class."""
        if not self.camera or not self.detector:
            return None

        if hasattr(self.camera, "is_operational") and not self.camera.is_operational:
            logger.error("Camera is degraded or not operational.")
            self._active_fault = FaultCode.CAMERA_FRAME_DROP
            self._transition_to(RobotState.FAULT, "Camera hardware is degraded/unoperational.")
            return None

        frame = None
        try:
            frame = self.camera.capture_frame()
            if frame is None and hasattr(self.camera, "is_operational") and self.camera.is_operational:
                # If camera is operational, allow background thread a brief moment to produce fresh frame
                time.sleep(0.06)
                frame = self.camera.capture_frame()
        except Exception as exc:
            logger.error("Camera capture failure: %s", exc)
            self._active_fault = FaultCode.CAMERA_FRAME_DROP
            self._transition_to(RobotState.FAULT, "Camera hardware capture error.")
            return None

        if frame is None:
            self._consecutive_camera_drops += 1
            logger.warning("Camera returned dropped or stale frame (drop count: %d).", self._consecutive_camera_drops)
            if self._consecutive_camera_drops >= 3:
                self._active_fault = FaultCode.CAMERA_FRAME_DROP
                self._transition_to(RobotState.FAULT, "Camera frame capture dropped repeatedly (3 consecutive drops).")
            return None

        self._consecutive_camera_drops = 0

        try:
            target_det = None
            if hasattr(self.detector, "detect_target"):
                target_det = self.detector.detect_target(frame, target_class)
            elif hasattr(self.detector, "detect"):
                results = self.detector.detect(frame)
                if isinstance(results, (list, tuple)):
                    for r in results:
                        if getattr(r, "class_name", None) == target_class:
                            target_det = r
                            break

            if target_det is not None and not self._is_valid_detection(target_det):
                logger.warning("Detector returned malformed detection for '%s': %s", target_class, target_det)
                return None

            return target_det
        except Exception as exc:
            logger.error("Inference failure in detector: %s", exc)
            self._active_fault = FaultCode.WATCHDOG_TIMEOUT
            self._transition_to(RobotState.FAULT, "Vision detector inference crashed.")
            return None

    def _turn_pulse(self, speed: float, duration_sec: float, direction: str = "left") -> None:
        """Execute a short bounded turning pulse with responsive preemption."""
        if not self.motors:
            return

        active_states = (RobotState.SEARCHING, RobotState.CENTERING)
        if self._emergency_stop_latched or self._state not in active_states:
            return

        try:
            if direction == "left":
                self.motors.set_speeds(-speed, speed)
            else:
                self.motors.set_speeds(speed, -speed)
            logger.info("Turn pulse active: %s @ %.0f%% PWM for %.2fs", direction.upper(), speed * 100.0, duration_sec)
            t_end = time.time() + duration_sec
            while time.time() < t_end:
                if self._emergency_stop_latched or self._state not in active_states:
                    break
                time.sleep(0.02)
        except Exception as exc:
            logger.error("Motor turn pulse error: %s", exc)
        finally:
            if self._state != RobotState.MANUAL_MOVEMENT and self.motors:
                try:
                    self.motors.stop()
                except Exception as exc:
                    logger.error("Error stopping motors after turn pulse: %s", exc)

    def _forward_pulse(self, speed: float, duration_sec: float) -> None:
        """Execute a short bounded forward movement pulse with responsive preemption and ultrasonic gating."""
        if not self.motors:
            return

        if self._emergency_stop_latched or self._state != RobotState.APPROACHING:
            return

        # Hard forward-motion safety threshold at 25 cm:
        # At or below 25 cm, no new forward pulse may start!
        if self.ultrasonic_protection_enabled and self.distance_sensor:
            try:
                pre_chk = self.distance_sensor.get_reading()
                if pre_chk is not None and pre_chk.distance_cm is not None and pre_chk.distance_cm <= 25.0:
                    self.motors.stop()
                    return
            except Exception:
                self.motors.stop()
                return

        try:
            self.motors.set_speeds(speed, speed)
            logger.info("Forward pulse active: speed=%.0f%% PWM for %.2fs", speed * 100.0, duration_sec)
            t_end = time.time() + duration_sec
            last_chk = time.time()
            while time.time() < t_end:
                if self._emergency_stop_latched or self._state != RobotState.APPROACHING:
                    break
                now = time.time()
                if self.ultrasonic_protection_enabled and self.distance_sensor and (now - last_chk >= 0.05):
                    last_chk = now
                    try:
                        cur_rd = self.distance_sensor.get_reading()
                        if cur_rd and cur_rd.distance_cm is not None and cur_rd.distance_cm <= 25.0:
                            break
                    except Exception:
                        break
                time.sleep(0.02)
        except Exception as exc:
            logger.error("Motor forward pulse error: %s", exc)
        finally:
            if self._state != RobotState.MANUAL_MOVEMENT and self.motors:
                try:
                    self.motors.stop()
                except Exception as exc:
                    logger.error("Error stopping motors after forward pulse: %s", exc)
