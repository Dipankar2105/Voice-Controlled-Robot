"""Phase 7 unit tests: Robot Finite State Machine (FSM), autonomous search,
centering, approach gating, obstacle handling, camera/sensor faults, cancellation,
and emergency-stop latching.
"""
from __future__ import annotations

import time
import threading
from dataclasses import replace
from typing import Optional
from unittest.mock import MagicMock
import numpy as np
import pytest

from app.command.command_parser import (
    CommandAction,
    CommandIntent,
    RobotCommand,
)
from app.config import AppConfig, MotionConfig, SafetyConfig, VisionConfig
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


@pytest.fixture
def test_config() -> AppConfig:
    return AppConfig(
        safety=SafetyConfig(
            obstacle_warning_cm=50.0,
            obstacle_blocked_cm=25.0,
            obstacle_critical_cm=15.0,
            target_stop_distance_cm=30.0,
            max_consecutive_target_loss=3,
        ),
        vision=VisionConfig(
            deadband_pixels=40,
            confidence_threshold=0.25,
        ),
        motion=MotionConfig(
            search_scan_duration_sec=1.0,  # Fast timeout for tests
            calibrated_360_duration_sec=None,  # Uncalibrated baseline
        ),
    )


@pytest.fixture
def fsm(test_config: AppConfig) -> RobotStateMachine:
    motors = MockMotorController()
    motors.enable(True)
    sensor = MockDistanceSensor(warning_cm=50.0, blocked_cm=25.0, critical_cm=15.0)
    camera = MockCamera(width=640, height=480)
    detector = MockDetector()

    return RobotStateMachine(
        config=test_config,
        motors=motors,
        distance_sensor=sensor,
        camera=camera,
        detector=detector,
    )


# ---------------------------------------------------------------------------
# 1. Target Already Visible Skips Full Scan
# ---------------------------------------------------------------------------

def test_target_already_visible_skips_full_scan(fsm: RobotStateMachine):
    """If requested target is immediately detected in frame, skip rotational search."""
    # Program detector to return target at center
    det = DetectionResult(
        class_id=12,
        class_name="bottle",
        confidence=0.85,
        box_xywh=(320.0, 240.0, 80.0, 150.0),
        box_xyxy=(280.0, 165.0, 360.0, 315.0),
    )
    fsm.detector.detect_target = MagicMock(return_value=det)
    fsm.detector.has_class = MagicMock(return_value=True)

    # Launch search
    started = fsm.start_search_mission("bottle")
    assert started is True
    assert fsm.state == RobotState.SEARCHING

    # Step 1: Confirmation frame 1
    fsm.step_mission()
    assert fsm.state == RobotState.SEARCHING
    assert fsm.context.target_confirmed_count == 1

    # Step 2: Confirmation frame 2 -> Targets Confirmed!
    fsm.step_mission()
    assert fsm.state == RobotState.TARGET_CONFIRMED
    assert fsm.context.sectors_scanned == 0  # Zero rotational pulses!

    # Step 3: Proceeds to CENTERING
    fsm.step_mission()
    assert fsm.state == RobotState.CENTERING


# ---------------------------------------------------------------------------
# 2. Target Found in Later Sector
# ---------------------------------------------------------------------------

def test_target_found_in_later_sector(fsm: RobotStateMachine):
    """Target not visible initially; robot turns and finds target in subsequent sector."""
    fsm.detector.has_class = MagicMock(return_value=True)

    det = DetectionResult(
        class_id=10,
        class_name="backpack",
        confidence=0.75,
        box_xywh=(320.0, 240.0, 100.0, 100.0),
        box_xyxy=(270.0, 190.0, 370.0, 290.0),
    )
    # Sector 1: None; Sector 2: Detection
    fsm.detector.detect_target = MagicMock(side_effect=[None, det, det])

    fsm.start_search_mission("backpack")

    # Step in sector 1 (no target) -> triggers turn pulse
    fsm.step_mission()
    assert fsm.state == RobotState.SEARCHING
    assert fsm.context.sectors_scanned == 1

    # Step in sector 2 (detected frame 1)
    fsm.step_mission()
    assert fsm.state == RobotState.SEARCHING

    # Step in sector 2 (detected frame 2 -> confirmed)
    fsm.step_mission()
    assert fsm.state == RobotState.TARGET_CONFIRMED
    assert fsm.context.sectors_scanned == 1


# ---------------------------------------------------------------------------
# 3. Target Absent and Full Scan Exhausted
# ---------------------------------------------------------------------------

def test_target_absent_full_scan_exhausted(fsm: RobotStateMachine):
    """When target is never found, FSM terminates in TARGET_NOT_FOUND without spinning forever."""
    fsm.detector.has_class = MagicMock(return_value=True)
    fsm.detector.detect_target = MagicMock(return_value=None)

    fsm.start_search_mission("person")

    # Execute search steps until scan limit reached
    for _ in range(10):
        if fsm.state == RobotState.TARGET_NOT_FOUND:
            break
        fsm.step_mission()

    assert fsm.state == RobotState.TARGET_NOT_FOUND
    assert "uncalibrated sector scan limit" in fsm.context.terminal_message
    assert fsm.motors.current_speeds == (0.0, 0.0)


# ---------------------------------------------------------------------------
# 4. Centering Correction Left and Right
# ---------------------------------------------------------------------------

def test_centering_left_and_right(fsm: RobotStateMachine):
    """Target offset triggers bounded turn pulses until aligned within deadband."""
    fsm.detector.has_class = MagicMock(return_value=True)

    # 1. Target on left (cx = 150 < 320 - 40)
    det_left = DetectionResult(
        class_id=12, class_name="bottle", confidence=0.8,
        box_xywh=(150.0, 240.0, 50.0, 100.0), box_xyxy=(125.0, 190.0, 175.0, 290.0),
    )
    # 2. Target centered (cx = 320)
    det_centered = DetectionResult(
        class_id=12, class_name="bottle", confidence=0.8,
        box_xywh=(320.0, 240.0, 50.0, 100.0), box_xyxy=(295.0, 190.0, 345.0, 290.0),
    )
    fsm.detector.detect_target = MagicMock(side_effect=[det_left, det_centered])

    fsm.start_search_mission("bottle")
    fsm._transition_to(RobotState.CENTERING, "Direct centering test")

    # Step 1: Detects left -> turns left, remains CENTERING
    fsm.step_mission()
    assert fsm.state == RobotState.CENTERING

    # Step 2: Detects centered -> transitions to APPROACHING
    fsm.step_mission()
    assert fsm.state == RobotState.APPROACHING


# ---------------------------------------------------------------------------
# 5. Approach Gated by Ultrasonic and Target Reached
# ---------------------------------------------------------------------------

def test_approach_gated_by_ultrasonic_and_reaches_target(fsm: RobotStateMachine):
    """Approach moves forward in pulses and stops when target distance threshold reached."""
    fsm.detector.has_class = MagicMock(return_value=True)

    det = DetectionResult(
        class_id=12, class_name="bottle", confidence=0.85,
        box_xywh=(320.0, 240.0, 60.0, 120.0), box_xyxy=(290.0, 180.0, 350.0, 300.0),
    )
    fsm.detector.detect_target = MagicMock(return_value=det)

    fsm.start_search_mission("bottle")
    fsm._transition_to(RobotState.APPROACHING, "Direct approach test")

    # 1. Sensor reads 45 cm (CLEAR, above the 30 cm target stop distance) -> pulses forward
    fsm.distance_sensor.set_distance(45.0)
    fsm.step_mission()
    assert fsm.state == RobotState.APPROACHING

    # 2. Sensor reads 28 cm (WARNING band, within 30 cm stop distance) -> Target Reached.
    fsm.distance_sensor.set_distance(28.0)
    fsm.step_mission()
    assert fsm.state == RobotState.TARGET_REACHED
    assert fsm.motors.current_speeds == (0.0, 0.0)


# ---------------------------------------------------------------------------
# 6. Approach Blocked by Obstacle
# ---------------------------------------------------------------------------

def test_large_bounding_box_alone_never_completes_approach(fsm: RobotStateMachine):
    """Pixel area is not a physical distance and must not complete a mission."""
    fsm.detector.has_class = MagicMock(return_value=True)
    large_det = DetectionResult(
        class_id=12, class_name="bottle", confidence=0.99,
        box_xywh=(320.0, 240.0, 500.0, 400.0), box_xyxy=(70.0, 40.0, 570.0, 440.0),
    )
    fsm.detector.detect_target = MagicMock(return_value=large_det)
    fsm.start_search_mission("bottle")
    fsm._transition_to(RobotState.APPROACHING, "Direct approach test")
    fsm.distance_sensor.set_distance(80.0)
    fsm.step_mission()
    assert fsm.state == RobotState.APPROACHING


def test_blocked_distance_is_never_reported_as_target_reached(fsm: RobotStateMachine):
    """A close sonar return from any object is a safety block, not proof of target arrival."""
    fsm.detector.has_class = MagicMock(return_value=True)
    det = DetectionResult(
        class_id=12, class_name="bottle", confidence=0.85,
        box_xywh=(320.0, 240.0, 60.0, 120.0), box_xyxy=(290.0, 180.0, 350.0, 300.0),
    )
    fsm.detector.detect_target = MagicMock(return_value=det)
    fsm.start_search_mission("bottle")
    fsm._transition_to(RobotState.APPROACHING, "Direct approach test")
    fsm.distance_sensor.set_distance(18.0)
    fsm.step_mission()
    assert fsm.state == RobotState.BLOCKED
    assert fsm.motors.current_speeds == (0.0, 0.0)


def test_offcenter_target_inside_stop_zone_does_not_trigger_blind_turn(fsm: RobotStateMachine):
    """Close off-centre targets stop instead of turning without side-clearance sensing."""
    fsm.detector.has_class = MagicMock(return_value=True)
    offcenter = DetectionResult(
        class_id=12, class_name="bottle", confidence=0.9,
        box_xywh=(180.0, 240.0, 80.0, 120.0), box_xyxy=(140.0, 180.0, 220.0, 300.0),
    )
    fsm.detector.detect_target = MagicMock(return_value=offcenter)
    fsm.start_search_mission("bottle")
    fsm._transition_to(RobotState.CENTERING, "Direct centering test")
    fsm.distance_sensor.set_distance(28.0)
    fsm.step_mission()
    assert fsm.state == RobotState.BLOCKED
    assert fsm.motors.current_speeds == (0.0, 0.0)


def test_target_must_still_be_visible_before_arrival(fsm: RobotStateMachine):
    """Distance alone cannot complete a mission when the requested target disappears."""
    fsm.detector.has_class = MagicMock(return_value=True)
    fsm.detector.detect_target = MagicMock(return_value=None)
    fsm.start_search_mission("bottle")
    fsm._transition_to(RobotState.APPROACHING, "Direct approach test")
    fsm.distance_sensor.set_distance(28.0)
    fsm.step_mission()
    assert fsm.state == RobotState.APPROACHING
    assert fsm.motors.current_speeds == (0.0, 0.0)


def test_approach_blocked_by_obstacle(fsm: RobotStateMachine):
    """Critical or blocked obstacle stops forward approach and transitions to BLOCKED."""
    fsm.detector.has_class = MagicMock(return_value=True)

    det = DetectionResult(
        class_id=12, class_name="bottle", confidence=0.85,
        box_xywh=(320.0, 240.0, 60.0, 120.0), box_xyxy=(290.0, 180.0, 350.0, 300.0),
    )
    fsm.detector.detect_target = MagicMock(return_value=det)

    fsm.start_search_mission("bottle")
    fsm._transition_to(RobotState.APPROACHING, "Direct approach test")

    # Simulate sudden obstacle at 23 cm (BLOCKED, even though below target stop distance).
    fsm.distance_sensor.set_distance(23.0)
    fsm.step_mission()

    assert fsm.state == RobotState.BLOCKED
    assert fsm.motors.current_speeds == (0.0, 0.0)


# ---------------------------------------------------------------------------
# 7. Sensor Timeout & Stale Readings Fail Closed
# ---------------------------------------------------------------------------

def test_sensor_timeout_fails_closed(fsm: RobotStateMachine):
    """Sensor read failure or timeout transitions FSM to FAULT state."""
    fsm.detector.has_class = MagicMock(return_value=True)
    fsm.start_search_mission("bottle")
    fsm._transition_to(RobotState.APPROACHING, "Direct approach test")

    # Simulate sensor exception / timeout
    fsm.distance_sensor.get_reading = MagicMock(side_effect=RuntimeError("Sensor hardware read timeout"))
    fsm.step_mission()

    assert fsm.state == RobotState.FAULT
    assert fsm.active_fault == FaultCode.SENSOR_TIMEOUT
    assert fsm.motors.current_speeds == (0.0, 0.0)


# ---------------------------------------------------------------------------
# 8. Target Loss Detection
# ---------------------------------------------------------------------------

def test_target_loss_during_approach(fsm: RobotStateMachine):
    """Consecutive frames without target transitions FSM to FAULT (TARGET_LOST)."""
    fsm.detector.has_class = MagicMock(return_value=True)
    fsm.detector.detect_target = MagicMock(return_value=None)  # Target disappeared

    fsm.start_search_mission("bottle")
    fsm._transition_to(RobotState.APPROACHING, "Direct approach test")
    fsm.distance_sensor.set_distance(40.0)

    # 3 consecutive losses
    fsm.step_mission()
    fsm.step_mission()
    fsm.step_mission()

    assert fsm.state == RobotState.FAULT
    assert fsm.active_fault == FaultCode.TARGET_LOST
    assert fsm.motors.current_speeds == (0.0, 0.0)


# ---------------------------------------------------------------------------
# 9. Camera and Inference Failures
# ---------------------------------------------------------------------------

def test_camera_failure_fails_closed(fsm: RobotStateMachine):
    """Camera capture exception transitions FSM to FAULT without crashing."""
    fsm.camera.capture_frame = MagicMock(side_effect=RuntimeError("V4L2 device disconnected"))

    fsm.start_search_mission("bottle")
    fsm.step_mission()

    assert fsm.state == RobotState.FAULT
    assert fsm.active_fault == FaultCode.CAMERA_FRAME_DROP
    assert fsm.motors.current_speeds == (0.0, 0.0)


def test_inference_failure_fails_closed(fsm: RobotStateMachine):
    """Detector inference crash transitions FSM to FAULT."""
    fsm.detector.detect_target = MagicMock(side_effect=RuntimeError("ONNX engine crash"))

    fsm.start_search_mission("bottle")
    fsm.step_mission()

    assert fsm.state == RobotState.FAULT
    assert fsm.active_fault == FaultCode.WATCHDOG_TIMEOUT
    assert fsm.motors.current_speeds == (0.0, 0.0)


# ---------------------------------------------------------------------------
# 10. Cancellation During Missions
# ---------------------------------------------------------------------------

def test_cancellation_during_rotation(fsm: RobotStateMachine):
    """Cancellation during search immediately stops robot and sets CANCELLED."""
    fsm.detector.has_class = MagicMock(return_value=True)
    fsm.start_search_mission("bottle")
    assert fsm.state == RobotState.SEARCHING

    cancel_cmd = RobotCommand(
        raw_transcript="stop scanning",
        normalized_text="stop scanning",
        intent=CommandIntent.CONTROL,
        action=CommandAction.STOP_SCANNING,
    )
    handled = fsm.handle_command(cancel_cmd)
    assert handled is True
    assert fsm.state == RobotState.CANCELLED
    assert fsm.motors.current_speeds == (0.0, 0.0)


def test_cancellation_during_approach(fsm: RobotStateMachine):
    """Cancellation during approach stops robot and sets CANCELLED."""
    fsm.detector.has_class = MagicMock(return_value=True)
    fsm.start_search_mission("bottle")
    fsm._transition_to(RobotState.APPROACHING, "Test approach")

    cancel_cmd = RobotCommand(
        raw_transcript="cancel search",
        normalized_text="cancel search",
        intent=CommandIntent.CONTROL,
        action=CommandAction.CANCEL_SEARCH,
    )
    handled = fsm.handle_command(cancel_cmd)
    assert handled is True
    assert fsm.state == RobotState.CANCELLED
    assert fsm.motors.current_speeds == (0.0, 0.0)


# ---------------------------------------------------------------------------
# 11. Emergency Stop Priority and Latching
# ---------------------------------------------------------------------------

def test_estop_clear_is_refused_while_sensor_reports_blocked(fsm: RobotStateMachine):
    """E-stop cannot be cleared while the forward path is still blocked."""
    fsm.trigger_emergency_stop("test")
    fsm.distance_sensor.set_distance(20.0)
    assert fsm.clear_emergency_stop() is False
    assert fsm.state == RobotState.EMERGENCY_STOP
    assert fsm.is_emergency_stop_latched is True
    assert fsm.motors.is_enabled is False
    fsm.distance_sensor.set_distance(100.0)
    assert fsm.clear_emergency_stop() is True
    assert fsm.state == RobotState.IDLE_LISTENING


def test_emergency_stop_during_manual_pulse_keeps_latched_state(fsm: RobotStateMachine):
    """A pulse finishing after E-stop must not overwrite the emergency state."""
    cmd = RobotCommand(
        raw_transcript="forward", normalized_text="forward",
        intent=CommandIntent.MOVEMENT, action=CommandAction.MOVE_FORWARD,
        speed=0.4, timestamp=time.time(),
    )
    worker = threading.Thread(target=lambda: fsm.handle_command(cmd), daemon=True)
    worker.start()
    deadline = time.monotonic() + 0.15
    while fsm.state != RobotState.MANUAL_MOVEMENT and time.monotonic() < deadline:
        time.sleep(0.005)
    assert fsm.state == RobotState.MANUAL_MOVEMENT
    fsm.trigger_emergency_stop("test preemption")
    worker.join(timeout=0.5)
    assert not worker.is_alive()
    assert fsm.state == RobotState.EMERGENCY_STOP
    assert fsm.is_emergency_stop_latched is True
    assert fsm.motors.current_speeds == (0.0, 0.0)


def test_emergency_stop_priority_and_latching(fsm: RobotStateMachine):
    """Emergency stop latches, overrides all motion, and requires explicit clearing."""
    fsm.detector.has_class = MagicMock(return_value=True)
    fsm.start_search_mission("bottle")
    assert fsm.state == RobotState.SEARCHING

    # Trigger E-Stop
    estop_cmd = RobotCommand(
        raw_transcript="emergency stop",
        normalized_text="emergency stop",
        intent=CommandIntent.SAFETY,
        action=CommandAction.EMERGENCY_STOP,
    )
    fsm.handle_command(estop_cmd)
    assert fsm.state == RobotState.EMERGENCY_STOP
    assert fsm.is_emergency_stop_latched is True
    assert fsm.motors.current_speeds == (0.0, 0.0)

    # Invariant: Further commands must be rejected while latched
    move_cmd = RobotCommand(
        raw_transcript="forward",
        normalized_text="forward",
        intent=CommandIntent.MOVEMENT,
        action=CommandAction.MOVE_FORWARD,
    )
    assert fsm.handle_command(move_cmd) is False
    assert fsm.state == RobotState.EMERGENCY_STOP

    # Ordinary resume does NOT clear the latch
    resume_cmd = RobotCommand(
        raw_transcript="resume listening",
        normalized_text="resume listening",
        intent=CommandIntent.CONTROL,
        action=CommandAction.RESUME_LISTENING,
    )
    fsm.handle_command(resume_cmd)
    assert fsm.state == RobotState.EMERGENCY_STOP
    assert fsm.is_emergency_stop_latched is True

    # Explicit clear required
    fsm.clear_emergency_stop()
    assert fsm.is_emergency_stop_latched is False
    assert fsm.state == RobotState.IDLE_LISTENING


def test_active_stop_during_manual_pulse_preempts_immediately(fsm: RobotStateMachine):
    """An active STOP command must interrupt a manual motion pulse and halt motors."""
    fsm.distance_sensor.set_distance(100.0)
    cmd = RobotCommand(
        raw_transcript="forward", normalized_text="forward",
        intent=CommandIntent.MOVEMENT, action=CommandAction.MOVE_FORWARD,
        speed=0.5, timestamp=time.time(),
    )
    worker = threading.Thread(target=lambda: fsm.handle_command(cmd), daemon=True)
    worker.start()

    deadline = time.monotonic() + 0.15
    while fsm.state != RobotState.MANUAL_MOVEMENT and time.monotonic() < deadline:
        time.sleep(0.005)
    assert fsm.state == RobotState.MANUAL_MOVEMENT

    stop_cmd = RobotCommand(
        raw_transcript="stop", normalized_text="stop",
        intent=CommandIntent.SAFETY, action=CommandAction.STOP,
        timestamp=time.time(),
    )
    assert fsm.handle_command(stop_cmd) is True
    worker.join(timeout=0.3)
    assert not worker.is_alive()
    assert fsm.state == RobotState.IDLE_LISTENING
    assert fsm.motors.current_speeds == (0.0, 0.0)


def test_manual_forward_refused_when_obstacle_present(fsm: RobotStateMachine):
    """Forward motion must transition to BLOCKED and refuse actuation when obstacle <= 25cm."""
    fsm.distance_sensor.set_distance(20.0)  # Inside blocked zone (20 cm < 25 cm)
    cmd = RobotCommand(
        raw_transcript="forward", normalized_text="forward",
        intent=CommandIntent.MOVEMENT, action=CommandAction.MOVE_FORWARD,
        speed=0.5, timestamp=time.time(),
    )
    accepted = fsm.handle_command(cmd)
    assert accepted is False
    assert fsm.state == RobotState.BLOCKED
    assert fsm.motors.current_speeds == (0.0, 0.0)


def test_turn_pulse_preemption_on_stop(fsm: RobotStateMachine):
    """A search/centering turn pulse must abort and stop motors when STOP is received."""
    fsm._state = RobotState.SEARCHING
    worker = threading.Thread(
        target=lambda: fsm._turn_pulse(0.4, 0.75, direction="left"), daemon=True
    )
    worker.start()
    deadline = time.monotonic() + 0.15
    while fsm.motors.current_speeds == (0.0, 0.0) and time.monotonic() < deadline:
        time.sleep(0.005)
    assert fsm.motors.current_speeds == (-0.4, 0.4)

    # Issue STOP
    fsm._transition_to(RobotState.IDLE_LISTENING, "Test stop")
    worker.join(timeout=0.2)
    assert not worker.is_alive()
    assert fsm.motors.current_speeds == (0.0, 0.0)


def test_forward_pulse_preemption_on_estop(fsm: RobotStateMachine):
    """An approach forward pulse must abort and stop motors when EMERGENCY_STOP is latched."""
    fsm._state = RobotState.APPROACHING
    worker = threading.Thread(
        target=lambda: fsm._forward_pulse(0.4, 0.75), daemon=True
    )
    worker.start()
    deadline = time.monotonic() + 0.15
    while fsm.motors.current_speeds == (0.0, 0.0) and time.monotonic() < deadline:
        time.sleep(0.005)
    assert fsm.motors.current_speeds == (0.4, 0.4)

    # Issue Emergency Stop
    fsm.trigger_emergency_stop("Test estop preemption")
    worker.join(timeout=0.2)
    assert not worker.is_alive()
    assert fsm.motors.current_speeds == (0.0, 0.0)
    assert fsm.is_emergency_stop_latched is True


def test_manual_backward_allowed_when_forward_obstacle_present(fsm: RobotStateMachine):
    """Backward movement must succeed even if forward ultrasonic sensor detects an obstacle."""
    fsm.distance_sensor.set_distance(15.0)  # Close obstacle directly ahead
    fsm.config = replace(fsm.config, motion=replace(fsm.config.motion, manual_pulse_sec=0.05))

    cmd = RobotCommand(
        raw_transcript="backward", normalized_text="backward",
        intent=CommandIntent.MOVEMENT, action=CommandAction.MOVE_BACKWARD,
        speed=0.5, timestamp=time.time(),
    )
    accepted = fsm.handle_command(cmd)
    assert accepted is True
    # Invariant: Pulse finishes, motors are zeroed, FSM returns to IDLE_LISTENING
    assert fsm.state == RobotState.IDLE_LISTENING
    assert fsm.motors.current_speeds == (0.0, 0.0)


def test_manual_forward_pulse_completes_and_stops_motors(fsm: RobotStateMachine):
    """Forward motion pulse completes bounded duration and automatically zeros motors."""
    fsm.distance_sensor.set_distance(100.0)  # Clear path ahead
    fsm.config = replace(fsm.config, motion=replace(fsm.config.motion, manual_pulse_sec=0.05))

    cmd = RobotCommand(
        raw_transcript="forward", normalized_text="forward",
        intent=CommandIntent.MOVEMENT, action=CommandAction.MOVE_FORWARD,
        speed=0.5, timestamp=time.time(),
    )
    accepted = fsm.handle_command(cmd)
    assert accepted is True
    assert fsm.state == RobotState.IDLE_LISTENING
    assert fsm.motors.current_speeds == (0.0, 0.0)

