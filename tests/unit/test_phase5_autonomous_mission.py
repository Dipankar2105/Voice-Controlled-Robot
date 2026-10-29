"""Phase 5 unit tests: Autonomous Target Search & Approach Mission.

Comprehensive validation covering:
1. Search-intent parsing without accidental mission start (including partial rejection).
2. Mission start, cancellation, timeout, and bounded scan count.
3. Target detection stability (multi-frame confirmation) and loss handling.
4. Target centering decisions (deadband alignment, left/right micro-pulses).
5. Short approach pulses and stand-off stopping outside the blocked zone.
6. Ultrasonic blocked, invalid, and stale readings.
7. Camera and detector failures failing closed.
8. Emergency-stop preemption at every stage.
9. Voice, joystick, and autonomous concurrency arbitration.
10. Motor zeroing on all error, exit, and shutdown paths.
"""
from __future__ import annotations

import time
import threading
from unittest.mock import MagicMock
import pytest

from app.command.command_parser import (
    CommandAction,
    CommandIntent,
    CommandParser,
    RobotCommand,
)
from app.config import (
    AppConfig,
    MotionConfig,
    SafetyConfig,
    VisionConfig,
)
from app.constants import (
    FaultCode,
    RobotState,
)
from app.hardware.interfaces import (
    DetectionResult,
    ObstacleState,
    UltrasonicReading,
)
from app.hardware.mock_hardware import (
    MockCamera,
    MockDetector,
    MockDistanceSensor,
    MockMotorController,
)
from app.navigation.state_machine import MissionContext, RobotStateMachine
from app.runtime import RobotRuntime, LifecycleState


@pytest.fixture
def phase5_config() -> AppConfig:
    return AppConfig(
        safety=SafetyConfig(
            obstacle_warning_cm=50.0,
            obstacle_blocked_cm=25.0,
            obstacle_critical_cm=15.0,
            target_stop_distance_cm=30.0,
            watchdog_timeout_sec=2.0,
            mission_timeout_sec=2.0,  # Fast 2.0s timeout for test efficiency
            sensor_staleness_sec=0.50,
            max_consecutive_target_loss=3,
        ),
        vision=VisionConfig(
            deadband_pixels=40,
            confidence_threshold=0.25,
            min_confirmations=2,
        ),
        motion=MotionConfig(
            search_speed=0.30,
            turn_speed=0.40,
            approach_speed=0.40,
            search_scan_duration_sec=1.0,
            calibrated_360_duration_sec=None,
            search_turn_pulse_sec=0.10,
            centering_turn_pulse_sec=0.05,
            approach_pulse_sec=0.10,
            settling_delay_sec=0.01,  # Short settle delay for fast tests
        ),
    )


@pytest.fixture
def fsm(phase5_config: AppConfig) -> RobotStateMachine:
    motors = MockMotorController(initially_enabled=True)
    sensor = MockDistanceSensor(
        warning_cm=50.0,
        blocked_cm=25.0,
        critical_cm=15.0,
        initial_distance_cm=100.0,
    )
    camera = MockCamera(width=640, height=480)
    detector = MockDetector()

    return RobotStateMachine(
        config=phase5_config,
        motors=motors,
        distance_sensor=sensor,
        camera=camera,
        detector=detector,
    )


# ==============================================================================
# 1. Search-intent parsing without accidental mission start
# ==============================================================================

def test_search_intent_parsing_and_partial_rejection():
    """Verify search intent parsing, article stripping, and partial rejection."""
    parser = CommandParser(supported_classes={"bottle", "chair", "person", "scissors"})

    # Complete search command parses correctly
    cmd = parser.parse("find bottle")
    assert cmd.intent == CommandIntent.SEARCH
    assert cmd.action == CommandAction.SEARCH_TARGET
    assert cmd.target_class == "bottle"
    assert cmd.is_executable is True

    # "search for the chair" -> strips "the" and resolves "chair"
    cmd2 = parser.parse("search for the chair")
    assert cmd2.intent == CommandIntent.SEARCH
    assert cmd2.action == CommandAction.SEARCH_TARGET
    assert cmd2.target_class == "chair"

    # "locate person"
    cmd3 = parser.parse("locate person")
    assert cmd3.intent == CommandIntent.SEARCH
    assert cmd3.action == CommandAction.SEARCH_TARGET
    assert cmd3.target_class == "person"

    # Invariant: Partial transcript MUST NOT trigger autonomous search
    partial_cmd = parser.parse("find bottle", is_partial=True)
    assert partial_cmd.intent == CommandIntent.UNKNOWN
    assert partial_cmd.action == CommandAction.UNKNOWN
    assert partial_cmd.is_executable is False
    assert "Partial transcript discarded" in (partial_cmd.error_reason or "")

    # Unsupported class rejects cleanly
    unsupported_cmd = parser.parse("find elephant")
    assert unsupported_cmd.intent == CommandIntent.UNKNOWN
    assert unsupported_cmd.action == CommandAction.UNKNOWN
    assert unsupported_cmd.is_executable is False


# ==============================================================================
# 2. Mission start, cancellation, timeout, and bounded scan count
# ==============================================================================

def test_mission_start_and_explicit_cancellation(fsm: RobotStateMachine):
    """Verify mission start and explicit cancellation ('cancel search', 'stop scanning')."""
    fsm.detector.has_class = MagicMock(return_value=True)

    # Start search
    assert fsm.start_search_mission("bottle") is True
    assert fsm.state == RobotState.SEARCHING
    assert fsm.context is not None
    assert fsm.context.target_class == "bottle"

    # Spoken cancellation
    cancel_cmd = RobotCommand(
        raw_transcript="cancel search",
        normalized_text="cancel search",
        intent=CommandIntent.CONTROL,
        action=CommandAction.CANCEL_SEARCH,
    )
    assert fsm.handle_command(cancel_cmd) is True
    assert fsm.state == RobotState.CANCELLED
    assert fsm.motors.current_speeds == (0.0, 0.0)


def test_mission_overall_timeout(fsm: RobotStateMachine):
    """Verify that exceeding mission_timeout_sec safely halts the robot."""
    fsm.detector.has_class = MagicMock(return_value=True)
    fsm.detector.detect_target = MagicMock(return_value=None)
    fsm.start_search_mission("bottle")

    # Artificially age mission start time past configured timeout (2.0s)
    fsm.context.start_time = time.time() - 3.5

    fsm.step_mission()
    assert fsm.state == RobotState.TARGET_NOT_FOUND
    assert "Mission timeout" in (fsm.context.terminal_message or "")
    assert fsm.motors.current_speeds == (0.0, 0.0)


def test_bounded_scan_count_and_sector_exhaustion(fsm: RobotStateMachine):
    """Verify search stops in TARGET_NOT_FOUND after scan duration or max sectors."""
    fsm.detector.has_class = MagicMock(return_value=True)
    fsm.detector.detect_target = MagicMock(return_value=None)
    fsm.start_search_mission("bottle")

    # Step repeatedly until scan limit is reached
    for _ in range(15):
        if fsm.state == RobotState.TARGET_NOT_FOUND:
            break
        fsm.step_mission()

    assert fsm.state == RobotState.TARGET_NOT_FOUND
    assert fsm.motors.current_speeds == (0.0, 0.0)
    assert fsm.context.sectors_scanned > 0


# ==============================================================================
# 3. Target detection stability and loss handling
# ==============================================================================

def test_target_detection_requires_multi_frame_confirmation(fsm: RobotStateMachine):
    """Single detection does not trigger TARGET_CONFIRMED; requires min_confirmations."""
    det = DetectionResult(
        class_id=12,
        class_name="bottle",
        confidence=0.80,
        box_xywh=(320.0, 240.0, 50.0, 100.0),
        box_xyxy=(295.0, 190.0, 345.0, 290.0),
    )
    fsm.detector.has_class = MagicMock(return_value=True)
    fsm.detector.detect_target = MagicMock(return_value=det)

    fsm.start_search_mission("bottle")

    # Frame 1: Candidate detected (count = 1/2) -> Still SEARCHING
    fsm.step_mission()
    assert fsm.state == RobotState.SEARCHING
    assert fsm.context.target_confirmed_count == 1

    # Frame 2: Confirmed (count = 2/2) -> TARGET_CONFIRMED
    fsm.step_mission()
    assert fsm.state == RobotState.TARGET_CONFIRMED
    assert fsm.context.target_confirmed_count == 2


def test_target_loss_during_centering_fails_closed(fsm: RobotStateMachine):
    """Target disappearance during CENTERING triggers FAULT (TARGET_LOST)."""
    fsm.detector.has_class = MagicMock(return_value=True)
    fsm.detector.detect_target = MagicMock(return_value=None)

    fsm.start_search_mission("bottle")
    fsm._transition_to(RobotState.CENTERING, "Test centering loss")

    # Consecutive losses equal to max_consecutive_target_loss (3)
    fsm.step_mission()
    assert fsm.state == RobotState.CENTERING
    assert fsm.context.consecutive_target_loss == 1

    fsm.step_mission()
    assert fsm.state == RobotState.CENTERING
    assert fsm.context.consecutive_target_loss == 2

    fsm.step_mission()
    assert fsm.state == RobotState.FAULT
    assert fsm.active_fault == FaultCode.TARGET_LOST
    assert fsm.motors.current_speeds == (0.0, 0.0)


# ==============================================================================
# 4. Target centering decisions (Deadband alignment & micro-pulses)
# ==============================================================================

def test_target_centering_decisions(fsm: RobotStateMachine):
    """Verify left micro-pulse, right micro-pulse, and deadband transition to approach."""
    fsm.detector.has_class = MagicMock(return_value=True)

    # 1. Target on left (cx = 200 < 320 - 40)
    det_left = DetectionResult(
        class_id=12, class_name="bottle", confidence=0.85,
        box_xywh=(200.0, 240.0, 50.0, 100.0), box_xyxy=(175.0, 190.0, 225.0, 290.0),
    )
    # 2. Target on right (cx = 450 > 320 + 40)
    det_right = DetectionResult(
        class_id=12, class_name="bottle", confidence=0.85,
        box_xywh=(450.0, 240.0, 50.0, 100.0), box_xyxy=(425.0, 190.0, 475.0, 290.0),
    )
    # 3. Target centered (cx = 325 within 320 +/- 40)
    det_centered = DetectionResult(
        class_id=12, class_name="bottle", confidence=0.85,
        box_xywh=(325.0, 240.0, 50.0, 100.0), box_xyxy=(300.0, 190.0, 350.0, 290.0),
    )

    fsm.detector.detect_target = MagicMock(side_effect=[det_left, det_right, det_centered])
    fsm.start_search_mission("bottle")
    fsm._transition_to(RobotState.CENTERING, "Test centering decisions")

    # Step 1: Detects left, turns left, stays in CENTERING
    fsm.step_mission()
    assert fsm.state == RobotState.CENTERING

    # Step 2: Detects right, turns right, stays in CENTERING
    fsm.step_mission()
    assert fsm.state == RobotState.CENTERING

    # Step 3: Aligned within deadband, transitions to APPROACHING
    fsm.step_mission()
    assert fsm.state == RobotState.APPROACHING


# ==============================================================================
# 5. Short approach pulses and stand-off stopping
# ==============================================================================

def test_approach_pulses_and_standoff_arrival(fsm: RobotStateMachine):
    """Approach executes forward pulses when clear and stops when distance <= 30cm."""
    fsm.detector.has_class = MagicMock(return_value=True)
    det = DetectionResult(
        class_id=12, class_name="bottle", confidence=0.90,
        box_xywh=(320.0, 240.0, 80.0, 120.0), box_xyxy=(280.0, 180.0, 360.0, 300.0),
    )
    fsm.detector.detect_target = MagicMock(return_value=det)

    fsm.start_search_mission("bottle")
    fsm._transition_to(RobotState.APPROACHING, "Test approach pulses")

    # Clear at 60 cm -> pulses forward, remains APPROACHING
    fsm.distance_sensor.set_distance(60.0)
    fsm.step_mission()
    assert fsm.state == RobotState.APPROACHING

    # Reaches 29 cm (within 30 cm stop distance and outside 25 cm blocked zone)
    fsm.distance_sensor.set_distance(29.0)
    fsm.step_mission()
    assert fsm.state == RobotState.TARGET_REACHED
    assert fsm.motors.current_speeds == (0.0, 0.0)


# ==============================================================================
# 6. Ultrasonic blocked, invalid, and stale readings
# ==============================================================================

def test_ultrasonic_blocked_stops_approach(fsm: RobotStateMachine):
    """Obstacle inside blocked zone (<= 25cm) forces state to BLOCKED."""
    fsm.detector.has_class = MagicMock(return_value=True)
    det = DetectionResult(
        class_id=12, class_name="bottle", confidence=0.90,
        box_xywh=(320.0, 240.0, 80.0, 120.0), box_xyxy=(280.0, 180.0, 360.0, 300.0),
    )
    fsm.detector.detect_target = MagicMock(return_value=det)
    fsm.start_search_mission("bottle")
    fsm._transition_to(RobotState.APPROACHING, "Test blocked approach")

    # Obstacle at 20 cm
    fsm.distance_sensor.set_distance(20.0)
    fsm.step_mission()
    assert fsm.state == RobotState.BLOCKED
    assert fsm.motors.current_speeds == (0.0, 0.0)


def test_ultrasonic_stale_reading_fails_closed(fsm: RobotStateMachine):
    """Sensor reading older than sensor_staleness_sec transitions to FAULT."""
    fsm.detector.has_class = MagicMock(return_value=True)
    fsm.start_search_mission("bottle")
    fsm._transition_to(RobotState.APPROACHING, "Test stale reading")

    # Stale reading: timestamp 2.0s ago (> 0.50s threshold)
    stale_reading = UltrasonicReading(
        distance_cm=45.0,
        state=ObstacleState.CLEAR,
        timestamp=time.time() - 2.0,
    )
    fsm.distance_sensor.get_reading = MagicMock(return_value=stale_reading)

    fsm.step_mission()
    assert fsm.state == RobotState.FAULT
    assert fsm.active_fault == FaultCode.SENSOR_TIMEOUT
    assert fsm.motors.current_speeds == (0.0, 0.0)


def test_ultrasonic_invalid_reading_fails_closed(fsm: RobotStateMachine):
    """Sensor fault reading transitions FSM to FAULT."""
    fsm.detector.has_class = MagicMock(return_value=True)
    fsm.start_search_mission("bottle")
    fsm._transition_to(RobotState.APPROACHING, "Test invalid reading")

    fault_reading = UltrasonicReading.create_fault()
    fsm.distance_sensor.get_reading = MagicMock(return_value=fault_reading)

    fsm.step_mission()
    assert fsm.state == RobotState.FAULT
    assert fsm.active_fault == FaultCode.SENSOR_TIMEOUT
    assert fsm.motors.current_speeds == (0.0, 0.0)


# ==============================================================================
# 7. Camera and detector failures fail closed
# ==============================================================================

def test_camera_capture_failure_halts_mission(fsm: RobotStateMachine):
    """Camera capture exception transitions FSM to FAULT and stops motors."""
    fsm.camera.capture_frame = MagicMock(side_effect=RuntimeError("Camera device unplugged"))
    fsm.start_search_mission("bottle")

    fsm.step_mission()
    assert fsm.state == RobotState.FAULT
    assert fsm.active_fault == FaultCode.CAMERA_FRAME_DROP
    assert fsm.motors.current_speeds == (0.0, 0.0)


def test_detector_inference_failure_halts_mission(fsm: RobotStateMachine):
    """Detector inference crash transitions FSM to FAULT and stops motors."""
    fsm.detector.detect_target = MagicMock(side_effect=RuntimeError("ONNX engine crash"))
    fsm.start_search_mission("bottle")

    fsm.step_mission()
    assert fsm.state == RobotState.FAULT
    assert fsm.active_fault == FaultCode.WATCHDOG_TIMEOUT
    assert fsm.motors.current_speeds == (0.0, 0.0)


# ==============================================================================
# 8. Emergency-stop preemption at every stage
# ==============================================================================

@pytest.mark.parametrize("initial_state", [
    RobotState.SEARCHING,
    RobotState.TARGET_CONFIRMED,
    RobotState.CENTERING,
    RobotState.APPROACHING,
])
def test_emergency_stop_preempts_all_mission_states(fsm: RobotStateMachine, initial_state: RobotState):
    """Emergency stop immediately latches, zeroes motors, and disables actuation across all mission states."""
    fsm.detector.has_class = MagicMock(return_value=True)
    fsm.start_search_mission("bottle")
    fsm._transition_to(initial_state, f"Direct test from {initial_state.value}")

    estop_cmd = RobotCommand(
        raw_transcript="emergency stop",
        normalized_text="emergency stop",
        intent=CommandIntent.SAFETY,
        action=CommandAction.EMERGENCY_STOP,
    )
    assert fsm.handle_command(estop_cmd) is True
    assert fsm.state == RobotState.EMERGENCY_STOP
    assert fsm.is_emergency_stop_latched is True
    assert fsm.motors.current_speeds == (0.0, 0.0)
    assert fsm.motors.is_enabled is False


# ==============================================================================
# 9. Voice, joystick, and autonomous concurrency arbitration
# ==============================================================================

def test_voice_manual_movement_rejected_during_autonomous_mission(fsm: RobotStateMachine):
    """Manual directional movement voice commands are rejected while autonomous mission is active."""
    fsm.detector.has_class = MagicMock(return_value=True)
    fsm.start_search_mission("bottle")
    assert fsm.state == RobotState.SEARCHING

    move_cmd = RobotCommand(
        raw_transcript="forward",
        normalized_text="forward",
        intent=CommandIntent.MOVEMENT,
        action=CommandAction.MOVE_FORWARD,
    )
    accepted = fsm.handle_command(move_cmd)
    assert accepted is False
    assert fsm.state == RobotState.SEARCHING


def test_concurrent_search_command_rejected(fsm: RobotStateMachine):
    """A new search command is rejected if a mission is already running."""
    fsm.detector.has_class = MagicMock(return_value=True)
    assert fsm.start_search_mission("bottle") is True

    search_cmd = RobotCommand(
        raw_transcript="find chair",
        normalized_text="find chair",
        intent=CommandIntent.SEARCH,
        action=CommandAction.SEARCH_TARGET,
        target_class="chair",
    )
    accepted = fsm.handle_command(search_cmd)
    assert accepted is False
    assert fsm.context.target_class == "bottle"


def test_joystick_preempts_and_cancels_mission(phase5_config: AppConfig):
    """Manual joystick input cancels an ongoing autonomous mission."""
    motors = MockMotorController(initially_enabled=True)
    sensor = MockDistanceSensor(initial_distance_cm=100.0)
    camera = MockCamera(width=640, height=480)
    detector = MockDetector()

    runtime = RobotRuntime(
        config=phase5_config,
        motor_controller=motors,
        distance_sensor=sensor,
        camera=camera,
        detector=detector,
    )
    runtime.initialize()

    # Start autonomous search
    runtime.fsm.detector.has_class = MagicMock(return_value=True)
    runtime.fsm.start_search_mission("bottle")
    assert runtime.fsm.state == RobotState.SEARCHING

    # Issue joystick action
    ok, msg = runtime.dispatch_joystick("forward")
    assert ok is True
    # Mission is cancelled and robot shifts to joystick control
    assert runtime.fsm.state == RobotState.MANUAL_MOVEMENT
    assert runtime._joystick_active is True

    # Voice command is rejected while joystick is actively held
    accepted, _, reason = runtime.dispatch_text_command("find chair")
    assert accepted is False
    assert "joystick is currently active" in reason

    # Release joystick
    runtime.dispatch_joystick("stop")
    assert runtime._joystick_active is False
    assert runtime.fsm.state == RobotState.IDLE_LISTENING


# ==============================================================================
# 10. Motor stop on all error and shutdown paths
# ==============================================================================

def test_motor_zeroed_on_runtime_stop_and_cleanup(phase5_config: AppConfig):
    """Runtime stop and cleanup de-energize and disable motors unconditionally."""
    motors = MockMotorController(initially_enabled=True)
    runtime = RobotRuntime(
        config=phase5_config,
        motor_controller=motors,
        distance_sensor=MockDistanceSensor(),
        camera=MockCamera(),
        detector=MockDetector(),
    )
    runtime.initialize()
    runtime.motors.set_speeds(0.5, 0.5)

    runtime.stop()
    assert runtime.motors.current_speeds == (0.0, 0.0)
    assert runtime.motors.is_enabled is False
    assert runtime.lifecycle_state == LifecycleState.STOPPED

    runtime.cleanup()
    assert runtime.lifecycle_state == LifecycleState.CLEANED_UP


# ==============================================================================
# 11. Safety Audit Specific Tests
# ==============================================================================

def test_camera_unoperational_fails_closed(fsm: RobotStateMachine):
    """When camera.is_operational is False, FSM transitions to FAULT immediately."""
    fsm.camera._is_opened = False
    assert fsm.camera.is_operational is False
    fsm.start_search_mission("bottle")

    fsm.step_mission()
    assert fsm.state == RobotState.FAULT
    assert fsm.active_fault == FaultCode.CAMERA_FRAME_DROP
    assert fsm.motors.current_speeds == (0.0, 0.0)


def test_approach_refused_when_distance_is_none(fsm: RobotStateMachine):
    """When distance is None, approach refuses forward pulse and transitions to FAULT."""
    fsm.detector.has_class = MagicMock(return_value=True)
    det = DetectionResult(
        class_id=12, class_name="bottle", confidence=0.90,
        box_xywh=(320.0, 240.0, 80.0, 120.0), box_xyxy=(280.0, 180.0, 360.0, 300.0),
    )
    fsm.detector.detect_target = MagicMock(return_value=det)
    fsm.start_search_mission("bottle")
    fsm._transition_to(RobotState.APPROACHING, "Test none distance")

    fsm.distance_sensor.get_reading = MagicMock(return_value=UltrasonicReading(
        distance_cm=None,
        state=ObstacleState.CLEAR,
        timestamp=time.time(),
    ))

    fsm.step_mission()
    assert fsm.state == RobotState.FAULT
    assert fsm.active_fault == FaultCode.SENSOR_TIMEOUT
    assert fsm.motors.current_speeds == (0.0, 0.0)


def test_cancel_search_immediate_preemption_in_runtime(phase5_config: AppConfig):
    """'cancel search' and 'stop scanning' immediately preempt without command lock delay."""
    motors = MockMotorController(initially_enabled=True)
    runtime = RobotRuntime(
        config=phase5_config,
        motor_controller=motors,
        distance_sensor=MockDistanceSensor(),
        camera=MockCamera(),
        detector=MockDetector(),
    )
    runtime.initialize()
    runtime.fsm.detector.has_class = MagicMock(return_value=True)
    runtime.fsm.start_search_mission("bottle")
    assert runtime.fsm.state == RobotState.SEARCHING

    # Issue "cancel search"
    ok, cmd, msg = runtime.dispatch_text_command("cancel search")
    assert ok is True
    assert runtime.fsm.state == RobotState.CANCELLED
    assert runtime.motors.current_speeds == (0.0, 0.0)


def test_configuration_bounds_validation(phase5_config: AppConfig):
    """Invalid pulse durations, negative timeouts, or bad ordering raise ConfigurationError."""
    from app.config import ConfigurationError
    from dataclasses import replace

    # Negative pulse
    bad_motion = replace(phase5_config.motion, search_turn_pulse_sec=-0.5)
    bad_cfg = replace(phase5_config, motion=bad_motion)
    with pytest.raises(ConfigurationError, match="search_turn_pulse_sec"):
        bad_cfg.validate()

    # Zero timeout
    bad_safety = replace(phase5_config.safety, mission_timeout_sec=0.0)
    bad_cfg2 = replace(phase5_config, safety=bad_safety)
    with pytest.raises(ConfigurationError, match="mission_timeout_sec"):
        bad_cfg2.validate()

    # Invalid calibrated 360 duration above 60.0s
    with pytest.raises(ConfigurationError, match="calibrated_360_duration_sec"):
        bad_cfg4 = replace(phase5_config, motion=replace(phase5_config.motion, calibrated_360_duration_sec=75.0))
        bad_cfg4.validate()


def test_motor_failure_during_search_and_approach_fails_safe(fsm: RobotStateMachine):
    """If motor actuation raises an exception during pulses, motors must immediately fail safe to stop."""
    fsm.detector.has_class = MagicMock(return_value=True)
    fsm.motors.stop = MagicMock(wraps=fsm.motors.stop)
    fsm.motors.set_speeds = MagicMock(side_effect=RuntimeError("Simulated motor hardware disconnect"))

    # 1. During SEARCHING rotation pulse
    fsm.start_search_mission("bottle")
    fsm.step_mission()
    assert fsm.motors.stop.called

    # 2. During APPROACHING forward pulse
    fsm.motors.stop.reset_mock()
    fsm._transition_to(RobotState.APPROACHING, "Direct approach test")
    fsm._detect_target_in_frame = MagicMock(return_value=DetectionResult(class_name="bottle", confidence=0.9, box_xywh=(320, 240, 100, 200), box_xyxy=(270, 140, 370, 340), class_id=12))
    fsm.distance_sensor.get_reading = MagicMock(return_value=UltrasonicReading(distance_cm=80.0, state=ObstacleState.CLEAR, timestamp=time.time()))
    fsm.step_mission()
    assert fsm.motors.stop.called


# ==============================================================================
# 12. Phase 5 Integration & Deterministic Safety Tests
# ==============================================================================

def test_new_target_search_replaces_active_mission_in_runtime(phase5_config: AppConfig):
    """A new target search command replaces the currently active search mission cleanly."""
    motors = MockMotorController(initially_enabled=True)
    sensor = MockDistanceSensor(initial_distance_cm=100.0)
    camera = MockCamera(width=640, height=480)
    detector = MockDetector(class_names=["bottle", "chair", "person"])

    runtime = RobotRuntime(
        config=phase5_config,
        motor_controller=motors,
        distance_sensor=sensor,
        camera=camera,
        detector=detector,
    )
    runtime.initialize()

    # Launch initial search for bottle
    ok1, cmd1, msg1 = runtime.dispatch_text_command("find bottle")
    assert ok1 is True
    assert runtime.fsm.state == RobotState.SEARCHING
    assert runtime.fsm.context.target_class == "bottle"

    # Mid-search, user commands a new search for chair
    ok2, cmd2, msg2 = runtime.dispatch_text_command("find chair")
    assert ok2 is True
    assert runtime.fsm.state == RobotState.SEARCHING
    assert runtime.fsm.context.target_class == "chair"
    assert runtime.fsm.motors.current_speeds == (0.0, 0.0)


def test_malformed_detector_output_handled_gracefully(fsm: RobotStateMachine):
    """Detector returning malformed bounding boxes (NaNs, negative dims) is safely rejected."""
    fsm.detector.has_class = MagicMock(return_value=True)

    # Malformed with NaN in bounding box
    malformed_nan = DetectionResult(
        class_id=12,
        class_name="bottle",
        confidence=0.85,
        box_xywh=(float("nan"), 240.0, 50.0, 100.0),
        box_xyxy=(0.0, 0.0, 0.0, 0.0),
    )
    fsm.detector.detect_target = MagicMock(return_value=malformed_nan)
    fsm.start_search_mission("bottle")

    # Step mission: malformed detection should be rejected as invalid/unconfirmed
    fsm.step_mission()
    assert fsm.state == RobotState.SEARCHING
    assert fsm.context.target_confirmed_count == 0

    # Malformed with negative width
    malformed_neg = DetectionResult(
        class_id=12,
        class_name="bottle",
        confidence=0.85,
        box_xywh=(320.0, 240.0, -50.0, 100.0),
        box_xyxy=(0.0, 0.0, 0.0, 0.0),
    )
    fsm.detector.detect_target = MagicMock(return_value=malformed_neg)
    fsm.step_mission()
    assert fsm.state == RobotState.SEARCHING
    assert fsm.context.target_confirmed_count == 0


def test_repeated_camera_frame_drops_transition_to_fault(fsm: RobotStateMachine):
    """Repeated camera frame drops (3 consecutive) safely halt search and enter FAULT."""
    fsm.detector.has_class = MagicMock(return_value=True)
    fsm.camera.capture_frame = MagicMock(return_value=None)
    fsm.start_search_mission("bottle")

    # Drops 1 and 2: still SEARCHING
    fsm.step_mission()
    assert fsm.state == RobotState.SEARCHING
    fsm.step_mission()
    assert fsm.state == RobotState.SEARCHING

    # Drop 3: fails closed to FAULT
    fsm.step_mission()
    assert fsm.state == RobotState.FAULT
    assert fsm.active_fault == FaultCode.CAMERA_FRAME_DROP
    assert fsm.motors.current_speeds == (0.0, 0.0)


def test_dynamic_obstacle_during_manual_forward_pulse_halts_motors(fsm: RobotStateMachine):
    """An obstacle appearing in the blocked zone during manual forward movement immediately halts motors."""
    clear_reading = UltrasonicReading(distance_cm=80.0, state=ObstacleState.CLEAR, timestamp=time.time())
    blocked_reading = UltrasonicReading(distance_cm=18.0, state=ObstacleState.BLOCKED, timestamp=time.time())

    fsm.distance_sensor.get_reading = MagicMock(side_effect=[clear_reading, clear_reading, blocked_reading])

    fwd_cmd = RobotCommand(
        raw_transcript="forward",
        normalized_text="forward",
        intent=CommandIntent.MOVEMENT,
        action=CommandAction.MOVE_FORWARD,
        speed=0.5,
    )

    accepted = fsm.handle_command(fwd_cmd)
    assert fsm.state == RobotState.BLOCKED
    assert fsm.motors.current_speeds == (0.0, 0.0)


def test_resume_listening_refused_while_hardware_fault_persists(fsm: RobotStateMachine):
    """RESUME_LISTENING cannot exit FAULT if the hardware sensor fault persists."""
    fsm.start_search_mission("bottle")
    fsm._active_fault = FaultCode.SENSOR_TIMEOUT
    fsm._transition_to(RobotState.FAULT, "Simulated sensor failure")

    # Distance sensor is still producing faulty readings
    fsm.distance_sensor.get_reading = MagicMock(return_value=UltrasonicReading.create_fault())

    resume_cmd = RobotCommand(
        raw_transcript="resume listening",
        normalized_text="resume listening",
        intent=CommandIntent.CONTROL,
        action=CommandAction.RESUME_LISTENING,
    )
    accepted = fsm.handle_command(resume_cmd)
    assert accepted is False
    assert fsm.state == RobotState.FAULT

    # Now sensor recovers and produces valid reading
    valid_reading = UltrasonicReading(distance_cm=90.0, state=ObstacleState.CLEAR, timestamp=time.time())
    fsm.distance_sensor.get_reading = MagicMock(return_value=valid_reading)

    accepted2 = fsm.handle_command(resume_cmd)
    assert accepted2 is True
    assert fsm.state == RobotState.IDLE_LISTENING
    assert fsm.active_fault == FaultCode.OK


def test_no_direct_gpio_in_navigation_modules():
    """Ensure app/navigation/ has zero direct imports or calls to RPi.GPIO."""
    import inspect
    import app.navigation.state_machine as sm_mod

    source = inspect.getsource(sm_mod)
    assert "RPi.GPIO" not in source
    assert "import RPi" not in source
    assert "GPIO.output" not in source
    assert "GPIO.setup" not in source

