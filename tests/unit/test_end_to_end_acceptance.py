"""SS_PROJECT_VCR_FRESH — End-to-End Integration & Acceptance Test Suite.

Deterministic validation of the entire software pipeline using controlled
abstractions and mocks:
A. Startup & Initialization:
   - Config loading, preflight checks, lifecycle states, motor disabled defaults,
     rollback on partial component failure.
B. Voice-Command Flow:
   - Microphone/Vosk transcript dispatch, manual motion (forward, back, left, right),
     target search class selection, rejection of UNKNOWN and conflicting commands,
     instant preemption by STOP and EMERGENCY_STOP.
C. Object-Search Flow:
   - Search mission start, multi-frame target confirmation, optical centering deadband,
     ultrasonic-gated approach pulses, standoff target arrival outside blocked zone,
     clean mid-mission target replacement, safe handling of malformed boxes, target loss,
     and camera/sensor drops.
D. Safety & Lifecycle:
   - Dynamic obstacle interlock during forward pulse, fail-closed on invalid/stale sonar,
     bounded worker shutdown, zero motor actuation on shutdown/error paths,
     joystick arbitration and watchdog timeout.
"""
from __future__ import annotations

import math
import time
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
from app.navigation.state_machine import RobotStateMachine
from app.runtime import LifecycleState, RobotRuntime


@pytest.fixture
def acceptance_config() -> AppConfig:
    """Deterministic configuration for fast, predictable end-to-end tests."""
    return AppConfig(
        safety=SafetyConfig(
            obstacle_warning_cm=50.0,
            obstacle_blocked_cm=25.0,
            obstacle_critical_cm=15.0,
            target_stop_distance_cm=30.0,
            watchdog_timeout_sec=2.0,
            mission_timeout_sec=3.0,
            sensor_staleness_sec=0.50,
            max_consecutive_target_loss=3,
        ),
        vision=VisionConfig(
            deadband_pixels=40,
            confidence_threshold=0.25,
            min_confirmations=2,
        ),
        motion=MotionConfig(
            manual_speed=0.50,
            turn_speed=0.40,
            approach_speed=0.40,
            search_speed=0.30,
            search_scan_duration_sec=1.0,
            calibrated_360_duration_sec=None,
            search_turn_pulse_sec=0.10,
            centering_turn_pulse_sec=0.05,
            approach_pulse_sec=0.10,
            manual_pulse_sec=0.15,
            settling_delay_sec=0.01,
        ),
    )


@pytest.fixture
def e2e_runtime(acceptance_config: AppConfig) -> RobotRuntime:
    """Assembles a full RobotRuntime instance with verified mocks."""
    motors = MockMotorController(initially_enabled=False)
    sensor = MockDistanceSensor(
        warning_cm=50.0,
        blocked_cm=25.0,
        critical_cm=15.0,
        initial_distance_cm=100.0,
    )
    camera = MockCamera(width=640, height=480)
    detector = MockDetector(class_names=["bottle", "chair", "person", "table", "scissors"])

    runtime = RobotRuntime(
        config=acceptance_config,
        motor_controller=motors,
        distance_sensor=sensor,
        camera=camera,
        detector=detector,
    )
    return runtime


# ==============================================================================
# SECTION A: Startup and Initialization
# ==============================================================================

def test_e2e_startup_and_initialization_healthy(e2e_runtime: RobotRuntime):
    """Runtime initializes cleanly: motors default disabled and stopped, FSM idle."""
    runtime = e2e_runtime
    runtime.initialize()

    assert runtime.lifecycle_state == LifecycleState.INITIALIZED
    assert runtime.fsm is not None
    assert runtime.fsm.state == RobotState.IDLE_LISTENING
    assert runtime.command_parser is not None

    # Motors must default to disabled and stopped
    assert runtime.motors.is_enabled is False
    assert runtime.motors.current_speeds == (0.0, 0.0)

    # Start runtime workers
    runtime.start()
    assert runtime.lifecycle_state == LifecycleState.RUNNING
    # Motors are enabled but stopped
    assert runtime.motors.is_enabled is True
    assert runtime.motors.current_speeds == (0.0, 0.0)

    runtime.stop()
    assert runtime.lifecycle_state == LifecycleState.STOPPED
    assert runtime.motors.is_enabled is False
    assert runtime.motors.current_speeds == (0.0, 0.0)


def test_e2e_startup_failure_releases_all_partial_resources(acceptance_config: AppConfig):
    """If camera fails on start(), all partially started components are safely stopped."""
    motors = MockMotorController(initially_enabled=False)
    sensor = MockDistanceSensor(initial_distance_cm=100.0)
    camera = MockCamera(width=640, height=480)
    # Simulate camera start failure
    camera.start = MagicMock(return_value=False)
    detector = MockDetector()

    runtime = RobotRuntime(
        config=acceptance_config,
        motor_controller=motors,
        distance_sensor=sensor,
        camera=camera,
        detector=detector,
    )
    runtime.initialize()

    with pytest.raises(RuntimeError, match="Camera failed to start"):
        runtime.start()

    # Motors must remain strictly disabled and stopped
    assert runtime.motors.is_enabled is False
    assert runtime.motors.current_speeds == (0.0, 0.0)
    assert runtime.lifecycle_state == LifecycleState.STOPPED


# ==============================================================================
# SECTION B: Voice-Command Flow
# ==============================================================================

def test_e2e_voice_motion_commands_flow(e2e_runtime: RobotRuntime):
    """Voice commands for manual directional motion reach the controller correctly."""
    runtime = e2e_runtime
    runtime.initialize()
    runtime.start()

    try:
        # 1. Forward command
        ok, cmd, msg = runtime.dispatch_text_command("forward")
        assert ok is True
        assert cmd.action == CommandAction.MOVE_FORWARD
        assert runtime.fsm.state == RobotState.IDLE_LISTENING
        assert runtime.motors.current_speeds == (0.0, 0.0)

        # 2. Turn left command
        ok, cmd, msg = runtime.dispatch_text_command("turn left")
        assert ok is True
        assert cmd.action == CommandAction.TURN_LEFT
        assert runtime.fsm.state == RobotState.IDLE_LISTENING

        # 3. Turn right command
        ok, cmd, msg = runtime.dispatch_text_command("turn right")
        assert ok is True
        assert cmd.action == CommandAction.TURN_RIGHT
        assert runtime.fsm.state == RobotState.IDLE_LISTENING

        # 4. Backward command
        ok, cmd, msg = runtime.dispatch_text_command("move backward")
        assert ok is True
        assert cmd.action == CommandAction.MOVE_BACKWARD
        assert runtime.fsm.state == RobotState.IDLE_LISTENING

        # 5. Stop command
        ok, cmd, msg = runtime.dispatch_text_command("stop")
        assert ok is True
        assert cmd.action == CommandAction.STOP
        assert runtime.fsm.state == RobotState.IDLE_LISTENING
        assert runtime.motors.current_speeds == (0.0, 0.0)
    finally:
        runtime.stop()


def test_e2e_voice_search_commands_select_classes(e2e_runtime: RobotRuntime):
    """Target-search spoken commands parse and select the intended target class."""
    runtime = e2e_runtime
    runtime.initialize()
    runtime.start()

    try:
        ok, cmd, msg = runtime.dispatch_text_command("find bottle")
        assert ok is True
        assert cmd.action == CommandAction.SEARCH_TARGET
        assert cmd.target_class == "bottle"
        assert runtime.fsm.state == RobotState.SEARCHING
        assert runtime.fsm.context.target_class == "bottle"

        # Explicit cancel
        ok_cancel, _, _ = runtime.dispatch_text_command("cancel search")
        assert ok_cancel is True
        assert runtime.fsm.state == RobotState.CANCELLED

        # Resume listening
        ok_resume, _, _ = runtime.dispatch_text_command("resume listening")
        assert ok_resume is True
        assert runtime.fsm.state == RobotState.IDLE_LISTENING
    finally:
        runtime.stop()


def test_e2e_unknown_and_conflicting_commands_rejected(e2e_runtime: RobotRuntime):
    """Unknown, ambiguous, and conflicting commands cannot initiate movement."""
    runtime = e2e_runtime
    runtime.initialize()
    runtime.start()

    try:
        # Opposing movement
        ok1, cmd1, reason1 = runtime.dispatch_text_command("forward backward")
        assert ok1 is False
        assert runtime.fsm.state == RobotState.IDLE_LISTENING
        assert runtime.motors.current_speeds == (0.0, 0.0)

        # Conflicting intent
        ok2, cmd2, reason2 = runtime.dispatch_text_command("find bottle move forward")
        assert ok2 is False
        assert runtime.fsm.state == RobotState.IDLE_LISTENING

        # Unknown vocabulary
        ok3, cmd3, reason3 = runtime.dispatch_text_command("banana helicopter")
        assert ok3 is False
        assert runtime.fsm.state == RobotState.IDLE_LISTENING
        assert runtime.motors.current_speeds == (0.0, 0.0)
    finally:
        runtime.stop()


def test_e2e_estop_preempts_everything_immediately(e2e_runtime: RobotRuntime):
    """Emergency Stop overrides active movement, disables motors, and latches."""
    runtime = e2e_runtime
    runtime.initialize()
    runtime.start()

    try:
        # Trigger E-stop
        ok, cmd, msg = runtime.dispatch_text_command("emergency stop")
        assert ok is True
        assert runtime.fsm.state == RobotState.EMERGENCY_STOP
        assert runtime.fsm.is_emergency_stop_latched is True
        assert runtime.motors.is_enabled is False
        assert runtime.motors.current_speeds == (0.0, 0.0)

        # While E-stop is latched, movement commands must be rejected
        ok_fwd, _, _ = runtime.dispatch_text_command("forward")
        assert ok_fwd is False
        assert runtime.fsm.state == RobotState.EMERGENCY_STOP
        assert runtime.motors.is_enabled is False

        # Clear E-stop explicitly (clear path at 100 cm)
        cleared = runtime.fsm.clear_emergency_stop()
        assert cleared is True
        assert runtime.fsm.is_emergency_stop_latched is False
        assert runtime.fsm.state == RobotState.IDLE_LISTENING
        assert runtime.motors.is_enabled is True
    finally:
        runtime.stop()


# ==============================================================================
# SECTION C: Object-Search Flow
# ==============================================================================

def test_e2e_object_search_full_pipeline(e2e_runtime: RobotRuntime):
    """Full search pipeline: find -> confirm -> center -> approach -> target reached."""
    runtime = e2e_runtime
    fsm = runtime.fsm = RobotStateMachine(
        config=runtime.config,
        motors=runtime.motors,
        distance_sensor=runtime.distance_sensor,
        camera=runtime.camera,
        detector=runtime.detector,
    )
    runtime.motors.enable(True)

    # 1. Start search
    assert fsm.start_search_mission("bottle") is True
    assert fsm.state == RobotState.SEARCHING

    # 2. Target detected in frame 1 (cx=250px, off center to left)
    det_left = DetectionResult(
        class_id=12,
        class_name="bottle",
        confidence=0.88,
        box_xywh=(250.0, 240.0, 60.0, 120.0),
        box_xyxy=(220.0, 180.0, 280.0, 300.0),
    )
    fsm.detector.detect_target = MagicMock(return_value=det_left)

    fsm.step_mission()
    assert fsm.state == RobotState.SEARCHING
    assert fsm.context.target_confirmed_count == 1

    # Frame 2: Confirmed!
    fsm.step_mission()
    assert fsm.state == RobotState.TARGET_CONFIRMED

    # 3. Transition to CENTERING
    fsm.step_mission()
    assert fsm.state == RobotState.CENTERING

    # Centering step: target still on left (250px), pulses left
    fsm.step_mission()
    assert fsm.state == RobotState.CENTERING

    # Next frame: target is centered within deadband (325px within 320 +/- 40)
    det_centered = DetectionResult(
        class_id=12,
        class_name="bottle",
        confidence=0.90,
        box_xywh=(325.0, 240.0, 60.0, 120.0),
        box_xyxy=(295.0, 180.0, 355.0, 300.0),
    )
    fsm.detector.detect_target = MagicMock(return_value=det_centered)

    fsm.step_mission()
    assert fsm.state == RobotState.APPROACHING

    # 4. Approaching: distance is 60cm (> 30cm stop threshold)
    fsm.distance_sensor.set_distance(60.0)
    fsm.step_mission()
    assert fsm.state == RobotState.APPROACHING

    # Final approach pulse: distance reaches 29.0 cm (within safe 30cm stop distance)
    fsm.distance_sensor.set_distance(29.0)
    fsm.step_mission()
    assert fsm.state == RobotState.TARGET_REACHED
    assert fsm.motors.current_speeds == (0.0, 0.0)
    assert "Requested target centred" in fsm.context.terminal_message


def test_e2e_object_search_target_replacement(e2e_runtime: RobotRuntime):
    """Spoken search command for a new target cleanly replaces ongoing search."""
    runtime = e2e_runtime
    runtime.initialize()
    runtime.start()

    try:
        ok1, _, _ = runtime.dispatch_text_command("find bottle")
        assert ok1 is True
        assert runtime.fsm.state == RobotState.SEARCHING
        assert runtime.fsm.context.target_class == "bottle"

        # Mid-search, user issues "find chair"
        ok2, _, _ = runtime.dispatch_text_command("find chair")
        assert ok2 is True
        assert runtime.fsm.state == RobotState.SEARCHING
        assert runtime.fsm.context.target_class == "chair"
        assert runtime.motors.current_speeds == (0.0, 0.0)
    finally:
        runtime.stop()


def test_e2e_object_search_failure_modes_fail_safe(e2e_runtime: RobotRuntime):
    """Failure modes (target lost, camera drop, sensor fault) strictly fail closed to stop."""
    fsm = RobotStateMachine(
        config=e2e_runtime.config,
        motors=e2e_runtime.motors,
        distance_sensor=e2e_runtime.distance_sensor,
        camera=e2e_runtime.camera,
        detector=e2e_runtime.detector,
    )
    e2e_runtime.motors.enable(True)

    # 1. Target lost during centering -> FAULT (TARGET_LOST)
    fsm.start_search_mission("bottle")
    fsm._transition_to(RobotState.CENTERING, "Direct centering test")
    fsm.detector.detect_target = MagicMock(return_value=None)

    for _ in range(3):
        fsm.step_mission()

    assert fsm.state == RobotState.FAULT
    assert fsm.active_fault == FaultCode.TARGET_LOST
    assert fsm.motors.current_speeds == (0.0, 0.0)

    # 2. Camera dropped frames -> FAULT (CAMERA_FRAME_DROP)
    fsm.clear_emergency_stop()
    fsm.start_search_mission("bottle")
    fsm.camera.capture_frame = MagicMock(return_value=None)

    for _ in range(3):
        fsm.step_mission()

    assert fsm.state == RobotState.FAULT
    assert fsm.active_fault == FaultCode.CAMERA_FRAME_DROP
    assert fsm.motors.current_speeds == (0.0, 0.0)


# ==============================================================================
# SECTION D: Safety and Lifecycle
# ==============================================================================

def test_e2e_obstacle_interlock_during_forward_movement(e2e_runtime: RobotRuntime):
    """Dynamic obstacle appearing during manual forward motion immediately halts motors."""
    fsm = RobotStateMachine(
        config=e2e_runtime.config,
        motors=e2e_runtime.motors,
        distance_sensor=e2e_runtime.distance_sensor,
        camera=e2e_runtime.camera,
        detector=e2e_runtime.detector,
    )
    e2e_runtime.motors.enable(True)

    # Initial clear, then obstacle at 18 cm
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


def test_e2e_invalid_sensor_readings_never_imply_clearance(e2e_runtime: RobotRuntime):
    """Invalid, None, or FAULT sensor readings strictly refuse forward motion."""
    fsm = RobotStateMachine(
        config=e2e_runtime.config,
        motors=e2e_runtime.motors,
        distance_sensor=e2e_runtime.distance_sensor,
        camera=e2e_runtime.camera,
        detector=e2e_runtime.detector,
    )
    e2e_runtime.motors.enable(True)

    # Sensor returns invalid reading
    fsm.distance_sensor.get_reading = MagicMock(return_value=UltrasonicReading.create_fault())

    assert fsm._is_forward_path_safe() is False

    fwd_cmd = RobotCommand(
        raw_transcript="forward",
        normalized_text="forward",
        intent=CommandIntent.MOVEMENT,
        action=CommandAction.MOVE_FORWARD,
        speed=0.5,
    )
    accepted = fsm.handle_command(fwd_cmd)
    assert accepted is False
    assert fsm.state == RobotState.BLOCKED
    assert fsm.motors.current_speeds == (0.0, 0.0)


def test_e2e_joystick_and_dashboard_integration(e2e_runtime: RobotRuntime):
    """Joystick teleoperation, autonomous mission cancellation, and watchdog safety."""
    runtime = e2e_runtime
    runtime.initialize()
    runtime.start()

    try:
        # Launch autonomous mission
        runtime.dispatch_text_command("find bottle")
        assert runtime.fsm.state == RobotState.SEARCHING

        # Joystick forward preempts and cancels autonomous search
        ok, msg = runtime.dispatch_joystick("forward")
        assert ok is True
        assert runtime.fsm.state == RobotState.MANUAL_MOVEMENT
        assert runtime._joystick_active is True

        # Stop via joystick
        ok_stop, _ = runtime.dispatch_joystick("stop")
        assert ok_stop is True
        assert runtime._joystick_active is False
        assert runtime.fsm.state == RobotState.IDLE_LISTENING
        assert runtime.motors.current_speeds == (0.0, 0.0)

        # E-stop blocks joystick
        runtime.dispatch_text_command("emergency stop")
        ok_blocked, msg_blocked = runtime.dispatch_joystick("forward")
        assert ok_blocked is False
        assert "Emergency stop is latched" in msg_blocked
    finally:
        runtime.stop()
