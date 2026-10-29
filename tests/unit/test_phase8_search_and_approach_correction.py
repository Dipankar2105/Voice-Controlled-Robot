"""Unit tests for SS_Project_VCR Phase 8: 360-degree search and 20-25cm target approach correction.

Verifies:
1. SEARCHING continues bounded in-place rotation when ultrasonic reports 20 cm, 25 cm, 28 cm.
2. SEARCHING inspects successive camera frames and successfully detects the target when it enters the frame.
3. Target found during rotation stops rotation immediately before entering centering.
4. Target not found results in bounded 360-degree/timeout completion without crashing or spinning indefinitely.
5. Emergency stop and cancellation stop rotation immediately.
6. Camera failure during rotation fails safely without motor movement.
7. APPROACHING with target ahead at 40 cm requests a bounded forward pulse.
8. APPROACHING with target ahead at or below 25 cm never starts or continues forward movement.
9. Invalid, stale, missing, or faulted ultrasonic readings during APPROACHING prevent forward movement.
10. Target drifting out of the corridor during approach stops forward motion and triggers re-centering.
11. Transitions cleanly to TARGET_REACHED when the standoff policy is satisfied without overshoot.
"""
from __future__ import annotations

import time
from unittest.mock import MagicMock

import pytest

from app.command.command_parser import CommandAction, CommandIntent, RobotCommand
from app.config import AppConfig, MotionConfig, SafetyConfig, VisionConfig
from app.constants import FaultCode, RobotState
from app.hardware.ultrasonic import ObstacleState, UltrasonicReading
from app.navigation.state_machine import RobotStateMachine
from app.vision.yolo_detector import DetectionResult
from tests.unit.test_phase5_autonomous_mission import (
    MockCamera,
    MockDetector,
    MockDistanceSensor,
    MockMotorController,
)


@pytest.fixture
def fsm() -> RobotStateMachine:
    cfg = AppConfig(
        safety=SafetyConfig(
            obstacle_warning_cm=50.0,
            obstacle_blocked_cm=20.0,
            obstacle_critical_cm=15.0,
            target_stop_distance_cm=25.0,
            sensor_staleness_sec=0.50,
            max_consecutive_target_loss=3,
        ),
        vision=VisionConfig(
            deadband_pixels=40,
            confidence_threshold=0.25,
            min_confirmations=2,
        ),
        motion=MotionConfig(
            search_scan_duration_sec=2.0,
            search_turn_pulse_sec=0.10,
            centering_turn_pulse_sec=0.05,
            approach_pulse_sec=0.10,
            settling_delay_sec=0.0,
        ),
    )
    motors = MockMotorController()
    motors.enable(True)
    sensor = MockDistanceSensor(warning_cm=50.0, blocked_cm=20.0, critical_cm=15.0)
    camera = MockCamera(width=640, height=480)
    detector = MockDetector(class_names=["bottle", "chair", "person"])

    machine = RobotStateMachine(
        config=cfg,
        motors=motors,
        distance_sensor=sensor,
        camera=camera,
        detector=detector,
    )
    return machine


# ==============================================================================
# 1. SEARCHING continues rotation despite close ultrasonic readings
# ==============================================================================

def test_1_searching_continues_rotation_when_ultrasonic_reports_close(fsm: RobotStateMachine):
    """SEARCHING continues bounded in-place rotation when ultrasonic reports 20, 25, 28 cm."""
    fsm.detector.has_class = MagicMock(return_value=True)
    fsm.detector.detect_target = MagicMock(return_value=None)  # Target not visible yet

    fsm.start_search_mission("bottle")
    assert fsm.state == RobotState.SEARCHING

    # Test sector rotation with ultrasonic at 20 cm, 25 cm, 28 cm
    for dist in (20.0, 25.0, 28.0):
        fsm.distance_sensor.set_distance(dist)
        sectors_before = fsm.context.sectors_scanned
        fsm.step_mission()
        assert fsm.state == RobotState.SEARCHING, f"Failed at distance {dist} cm: expected SEARCHING, got {fsm.state}"
        assert fsm.context.sectors_scanned == sectors_before + 1
        assert fsm.motors.current_speeds == (0.0, 0.0)  # Bounded pulse completed


# ==============================================================================
# 2. SEARCHING inspects frames and detects target
# ==============================================================================

def test_2_searching_inspects_frames_and_detects_target(fsm: RobotStateMachine):
    """SEARCHING inspects successive frames and successfully detects the target when it enters."""
    fsm.detector.has_class = MagicMock(return_value=True)
    det = DetectionResult(
        class_id=12, class_name="bottle", confidence=0.88,
        box_xywh=(250.0, 240.0, 60.0, 120.0), box_xyxy=(220.0, 180.0, 280.0, 300.0),
    )

    # Frame 1: absent; Frame 2: present; Frame 3: present -> confirmed
    fsm.detector.detect_target = MagicMock(side_effect=[None, det, det])
    fsm.start_search_mission("bottle")

    # Step 1: target absent -> sector rotated
    fsm.step_mission()
    assert fsm.state == RobotState.SEARCHING
    assert fsm.context.target_confirmed_count == 0

    # Step 2: target candidate seen (1/2) -> held position
    fsm.step_mission()
    assert fsm.state == RobotState.SEARCHING
    assert fsm.context.target_confirmed_count == 1

    # Step 3: target confirmed (2/2) -> transitions to TARGET_CONFIRMED
    fsm.step_mission()
    assert fsm.state == RobotState.TARGET_CONFIRMED


# ==============================================================================
# 3. Target found stops rotation immediately before centering
# ==============================================================================

def test_3_target_found_stops_rotation_immediately(fsm: RobotStateMachine):
    """Target found during rotation stops rotation immediately before entering centering."""
    fsm.detector.has_class = MagicMock(return_value=True)
    det = DetectionResult(
        class_id=12, class_name="bottle", confidence=0.90,
        box_xywh=(250.0, 240.0, 60.0, 120.0), box_xyxy=(220.0, 180.0, 280.0, 300.0),
    )
    fsm.detector.detect_target = MagicMock(return_value=det)
    fsm.start_search_mission("bottle")

    # Two frames to confirm
    fsm.step_mission()
    fsm.step_mission()

    assert fsm.state == RobotState.TARGET_CONFIRMED
    assert fsm.motors.current_speeds == (0.0, 0.0)

    # Next step enters CENTERING with motors stopped
    fsm.step_mission()
    assert fsm.state == RobotState.CENTERING


# ==============================================================================
# 4. Target not found results in bounded 360/timeout completion
# ==============================================================================

def test_4_target_not_found_bounded_sweep_completion(fsm: RobotStateMachine):
    """Target not found results in bounded 360-degree/timeout completion without infinite spin."""
    fsm.detector.has_class = MagicMock(return_value=True)
    fsm.detector.detect_target = MagicMock(return_value=None)
    fsm.start_search_mission("bottle")

    for _ in range(50):
        if fsm.state == RobotState.TARGET_NOT_FOUND:
            break
        fsm.step_mission()

    assert fsm.state == RobotState.TARGET_NOT_FOUND
    assert fsm.motors.current_speeds == (0.0, 0.0)


# ==============================================================================
# 5. Emergency stop and cancellation stop rotation immediately
# ==============================================================================

def test_5_emergency_stop_and_cancellation_stop_rotation(fsm: RobotStateMachine):
    """Emergency stop and cancellation stop rotation immediately."""
    fsm.detector.has_class = MagicMock(return_value=True)
    fsm.detector.detect_target = MagicMock(return_value=None)

    # Test Emergency Stop
    fsm.start_search_mission("bottle")
    assert fsm.state == RobotState.SEARCHING
    estop_cmd = RobotCommand(
        raw_transcript="emergency stop",
        normalized_text="emergency stop",
        intent=CommandIntent.SAFETY,
        action=CommandAction.EMERGENCY_STOP,
    )
    fsm.handle_command(estop_cmd)
    assert fsm.state == RobotState.EMERGENCY_STOP
    assert fsm.motors.current_speeds == (0.0, 0.0)

    # Reset and test cancellation
    fsm.clear_emergency_stop()
    fsm.start_search_mission("bottle")
    assert fsm.state == RobotState.SEARCHING
    cancel_cmd = RobotCommand(
        raw_transcript="stop",
        normalized_text="stop",
        intent=CommandIntent.SAFETY,
        action=CommandAction.STOP,
    )
    fsm.handle_command(cancel_cmd)
    assert fsm.state == RobotState.IDLE_LISTENING
    assert fsm.motors.current_speeds == (0.0, 0.0)


# ==============================================================================
# 6. Camera failure during rotation fails safely
# ==============================================================================

def test_6_camera_failure_during_rotation_fails_safely(fsm: RobotStateMachine):
    """Camera failure during rotation fails safely without motor movement."""
    fsm.detector.has_class = MagicMock(return_value=True)
    fsm.start_search_mission("bottle")
    assert fsm.state == RobotState.SEARCHING

    # Make camera fail on next capture
    fsm.camera.capture_frame = MagicMock(side_effect=RuntimeError("Camera disconnected"))
    fsm.step_mission()

    assert fsm.state == RobotState.FAULT
    assert fsm.active_fault == FaultCode.CAMERA_FRAME_DROP
    assert fsm.motors.current_speeds == (0.0, 0.0)


# ==============================================================================
# 7. APPROACHING at 40 cm requests bounded forward pulse
# ==============================================================================

def test_7_approaching_at_40cm_requests_bounded_forward_pulse(fsm: RobotStateMachine):
    """APPROACHING with target ahead at 40 cm requests a bounded forward pulse."""
    fsm.detector.has_class = MagicMock(return_value=True)
    centered = DetectionResult(
        class_id=12, class_name="bottle", confidence=0.92,
        box_xywh=(320.0, 240.0, 60.0, 120.0), box_xyxy=(290.0, 180.0, 350.0, 300.0),
    )
    fsm.detector.detect_target = MagicMock(return_value=centered)

    fsm.start_search_mission("bottle")
    fsm._transition_to(RobotState.APPROACHING, "Direct approach test")
    fsm.distance_sensor.set_distance(40.0)

    fsm.motors.set_speeds = MagicMock(wraps=fsm.motors.set_speeds)
    fsm.step_mission()

    assert fsm.state == RobotState.APPROACHING
    # Verified forward speeds requested (speed > 0, speed > 0)
    assert any(call.args[0] > 0 and call.args[1] > 0 for call in fsm.motors.set_speeds.call_args_list)
    assert fsm.motors.current_speeds == (0.0, 0.0)  # Motor stopped cleanly after pulse


# ==============================================================================
# 8. APPROACHING at or below 25 cm never starts forward movement
# ==============================================================================

def test_8_approaching_at_or_below_25cm_never_starts_forward_movement(fsm: RobotStateMachine):
    """APPROACHING with target ahead at or below 25 cm never starts or continues forward movement."""
    fsm.detector.has_class = MagicMock(return_value=True)
    centered = DetectionResult(
        class_id=12, class_name="bottle", confidence=0.92,
        box_xywh=(320.0, 240.0, 60.0, 120.0), box_xyxy=(290.0, 180.0, 350.0, 300.0),
    )
    fsm.detector.detect_target = MagicMock(return_value=centered)

    for test_dist in (25.0, 24.0, 23.0, 21.0):
        fsm.start_search_mission("bottle")
        fsm._transition_to(RobotState.APPROACHING, "Direct approach test")
        fsm.distance_sensor.set_distance(test_dist)

        fsm.motors.set_speeds = MagicMock(wraps=fsm.motors.set_speeds)
        fsm.step_mission()

        # No forward pulse may be initiated
        forward_calls = [
            call for call in fsm.motors.set_speeds.call_args_list
            if len(call.args) >= 2 and call.args[0] > 0 and call.args[1] > 0
        ]
        assert len(forward_calls) == 0, f"Forward movement was initiated at {test_dist} cm!"
        assert fsm.motors.current_speeds == (0.0, 0.0)


# ==============================================================================
# 9. Invalid/stale/missing/faulted ultrasonic prevents forward movement
# ==============================================================================

def test_9_invalid_stale_missing_ultrasonic_prevents_forward_movement(fsm: RobotStateMachine):
    """Invalid, stale, missing, or faulted ultrasonic readings during APPROACHING prevent forward movement."""
    fsm.detector.has_class = MagicMock(return_value=True)
    det = DetectionResult(
        class_id=12, class_name="bottle", confidence=0.92,
        box_xywh=(320.0, 240.0, 60.0, 120.0), box_xyxy=(290.0, 180.0, 350.0, 300.0),
    )
    fsm.detector.detect_target = MagicMock(return_value=det)

    # 1. Invalid reading
    fsm.start_search_mission("bottle")
    fsm._transition_to(RobotState.APPROACHING, "Direct approach test")
    invalid_rd = UltrasonicReading(distance_cm=None, state=ObstacleState.CRITICAL, timestamp=time.time())
    fsm.distance_sensor.get_reading = MagicMock(return_value=invalid_rd)
    fsm.step_mission()
    assert fsm.state == RobotState.FAULT
    assert fsm.motors.current_speeds == (0.0, 0.0)

    # 2. Stale reading
    fsm.clear_emergency_stop()
    fsm.start_search_mission("bottle")
    fsm._transition_to(RobotState.APPROACHING, "Direct approach test")
    stale_rd = UltrasonicReading(distance_cm=45.0, state=ObstacleState.CLEAR, timestamp=time.time() - 2.0)
    fsm.distance_sensor.get_reading = MagicMock(return_value=stale_rd)
    fsm.step_mission()
    assert fsm.state == RobotState.FAULT
    assert fsm.motors.current_speeds == (0.0, 0.0)

    # 3. Faulted sensor (exception)
    fsm.clear_emergency_stop()
    fsm.start_search_mission("bottle")
    fsm._transition_to(RobotState.APPROACHING, "Direct approach test")
    fsm.distance_sensor.get_reading = MagicMock(side_effect=TimeoutError("Ultrasonic echo timed out"))
    fsm.step_mission()
    assert fsm.state == RobotState.FAULT
    assert fsm.motors.current_speeds == (0.0, 0.0)


# ==============================================================================
# 10. Target drifting out of corridor stops and re-centers
# ==============================================================================

def test_10_target_drifting_out_of_corridor_triggers_recentering(fsm: RobotStateMachine):
    """Target drifting out of the corridor during approach stops forward motion and triggers re-centering."""
    fsm.detector.has_class = MagicMock(return_value=True)
    drifted = DetectionResult(
        class_id=12, class_name="bottle", confidence=0.89,
        box_xywh=(200.0, 240.0, 60.0, 120.0), box_xyxy=(170.0, 180.0, 230.0, 300.0),  # cx=200, offset=-120px > 40px
    )
    fsm.detector.detect_target = MagicMock(return_value=drifted)

    fsm.start_search_mission("bottle")
    fsm._transition_to(RobotState.APPROACHING, "Direct approach test")
    fsm.distance_sensor.set_distance(40.0)

    fsm.step_mission()

    assert fsm.state == RobotState.CENTERING
    assert fsm.motors.current_speeds == (0.0, 0.0)


# ==============================================================================
# 11. Transitions cleanly to TARGET_REACHED when standoff policy satisfied
# ==============================================================================

def test_11_transitions_cleanly_to_target_reached_without_overshoot(fsm: RobotStateMachine):
    """Transitions cleanly to TARGET_REACHED when the standoff policy is satisfied without overshoot."""
    fsm.detector.has_class = MagicMock(return_value=True)
    centered = DetectionResult(
        class_id=12, class_name="bottle", confidence=0.92,
        box_xywh=(320.0, 240.0, 60.0, 120.0), box_xyxy=(290.0, 180.0, 350.0, 300.0),
    )
    fsm.detector.detect_target = MagicMock(return_value=centered)

    fsm.start_search_mission("bottle")
    fsm._transition_to(RobotState.APPROACHING, "Direct approach test")

    # Set distance inside the 20-25 cm standoff arrival band
    fsm.distance_sensor.set_distance(23.0)
    fsm.step_mission()

    assert fsm.state == RobotState.TARGET_REACHED
    assert fsm.motors.current_speeds == (0.0, 0.0)
    assert "centred" in fsm.context.terminal_message
    assert "23.0 cm" in fsm.context.terminal_message


# ==============================================================================
# 12. Centering commands turning pulse toward target offset
# ==============================================================================

def test_12_centering_requests_turning_pulse_towards_target_offset(fsm: RobotStateMachine):
    """Centering with offset > 40px requests turning pulse toward target."""
    fsm.detector.has_class = MagicMock(return_value=True)

    # 1. Target on the right (cx=487.5px, offset=+167.5px > +40px) -> turn RIGHT (left > 0, right < 0)
    right_det = DetectionResult(
        class_id=12, class_name="bottle", confidence=0.88,
        box_xywh=(487.5, 240.0, 60.0, 120.0), box_xyxy=(457.5, 180.0, 517.5, 300.0),
    )
    fsm.detector.detect_target = MagicMock(return_value=right_det)
    fsm.start_search_mission("bottle")
    fsm._transition_to(RobotState.CENTERING, "Direct centering test")

    fsm.motors.set_speeds = MagicMock(wraps=fsm.motors.set_speeds)
    fsm.step_mission()

    # Verify right turn requested (speed > 0 on left, speed < 0 on right)
    right_turn_calls = [
        c for c in fsm.motors.set_speeds.call_args_list
        if len(c.args) >= 2 and c.args[0] > 0 and c.args[1] < 0
    ]
    assert len(right_turn_calls) >= 1, "Expected RIGHT turn pulse for offset > +40px"
    assert fsm.state == RobotState.CENTERING
    assert fsm.motors.current_speeds == (0.0, 0.0)

    # 2. Target on the left (cx=150.0px, offset=-170px < -40px) -> turn LEFT (left < 0, right > 0)
    left_det = DetectionResult(
        class_id=12, class_name="bottle", confidence=0.88,
        box_xywh=(150.0, 240.0, 60.0, 120.0), box_xyxy=(120.0, 180.0, 180.0, 300.0),
    )
    fsm.detector.detect_target = MagicMock(return_value=left_det)
    fsm.motors.set_speeds.reset_mock()
    fsm.step_mission()

    left_turn_calls = [
        c for c in fsm.motors.set_speeds.call_args_list
        if len(c.args) >= 2 and c.args[0] < 0 and c.args[1] > 0
    ]
    assert len(left_turn_calls) >= 1, "Expected LEFT turn pulse for offset < -40px"
    assert fsm.state == RobotState.CENTERING
    assert fsm.motors.current_speeds == (0.0, 0.0)


# ==============================================================================
# 13. Standoff arrival when blocked_cm == target_stop_cm
# ==============================================================================

def test_13_standoff_arrival_with_equal_blocked_and_standoff_thresholds(fsm: RobotStateMachine):
    """When blocked_cm == target_stop_cm == 25.0 cm, approaching at 25.0 cm enters TARGET_REACHED."""
    # Update safety config to match default_config.yaml (blocked == standoff == 25.0 cm)
    fsm.config = AppConfig(
        safety=SafetyConfig(
            obstacle_warning_cm=50.0,
            obstacle_blocked_cm=25.0,
            obstacle_critical_cm=15.0,
            target_stop_distance_cm=25.0,
        ),
        vision=fsm.config.vision,
        motion=fsm.config.motion,
    )
    fsm.target_stop_distance_cm = 25.0

    fsm.detector.has_class = MagicMock(return_value=True)
    centered = DetectionResult(
        class_id=12, class_name="bottle", confidence=0.92,
        box_xywh=(320.0, 240.0, 60.0, 120.0), box_xyxy=(290.0, 180.0, 350.0, 300.0),
    )
    fsm.detector.detect_target = MagicMock(return_value=centered)

    fsm.start_search_mission("bottle")
    fsm._transition_to(RobotState.APPROACHING, "Direct approach test")
    fsm.distance_sensor.set_distance(25.0)

    fsm.step_mission()

    assert fsm.state == RobotState.TARGET_REACHED, f"Expected TARGET_REACHED at 25.0 cm, got {fsm.state}"
    assert fsm.motors.current_speeds == (0.0, 0.0)


# ==============================================================================
# 14. Complete search pulse accounting and termination message
# ==============================================================================

def test_14_complete_search_pulse_accounting_and_termination_message(fsm: RobotStateMachine):
    """Search accounts for every pulse and includes sector count in termination message."""
    fsm.config = AppConfig(
        motion=MotionConfig(
            calibrated_360_duration_sec=1.6,
            search_turn_pulse_sec=0.10,
            settling_delay_sec=0.0,
        ),
        safety=fsm.config.safety,
        vision=fsm.config.vision,
    )
    fsm.calibrated_360_duration_sec = 1.6
    fsm.search_turn_pulse_sec = 0.10
    fsm.max_scan_sectors = 16

    fsm.detector.has_class = MagicMock(return_value=True)
    fsm.detector.detect_target = MagicMock(return_value=None)
    fsm.start_search_mission("bottle")

    for _ in range(20):
        if fsm.state == RobotState.TARGET_NOT_FOUND:
            break
        fsm.step_mission()

    assert fsm.state == RobotState.TARGET_NOT_FOUND
    assert fsm.context.sectors_scanned == 16
    assert "16/16 pulses" in fsm.context.terminal_message
    assert fsm.motors.current_speeds == (0.0, 0.0)


# ==============================================================================
# 15. Approach stationary reacquisition then clean recovery
# ==============================================================================

def test_15_approach_stationary_reacquisition_then_clean_recovery(fsm: RobotStateMachine):
    """Approach keeps motors stationary when target is missed, then recovers upon redetection."""
    fsm.detector.has_class = MagicMock(return_value=True)
    centered = DetectionResult(
        class_id=12, class_name="bottle", confidence=0.88,
        box_xywh=(320.0, 240.0, 60.0, 120.0), box_xyxy=(290.0, 180.0, 350.0, 300.0),
    )

    fsm.start_search_mission("bottle")
    fsm._transition_to(RobotState.APPROACHING, "Direct approach test")
    fsm.distance_sensor.set_distance(60.0)

    # Frame 1: Target missed (optical drop) -> motors stay stationary (0,0)
    fsm.detector.detect_target = MagicMock(return_value=None)
    fsm.step_mission()
    assert fsm.state == RobotState.APPROACHING
    assert fsm.context.consecutive_target_loss == 1
    assert fsm.motors.current_speeds == (0.0, 0.0)

    # Frame 2: Target re-acquired -> consecutive_target_loss resets to 0, forward pulse executes
    fsm.detector.detect_target = MagicMock(return_value=centered)
    fsm.motors.set_speeds = MagicMock(wraps=fsm.motors.set_speeds)
    fsm.step_mission()

    assert fsm.state == RobotState.APPROACHING
    assert fsm.context.consecutive_target_loss == 0
    # Forward pulse called: speed > 0 on both channels
    forward_calls = [
        c for c in fsm.motors.set_speeds.call_args_list
        if len(c.args) >= 2 and c.args[0] > 0 and c.args[1] > 0
    ]
    assert len(forward_calls) >= 1
    assert fsm.motors.current_speeds == (0.0, 0.0)

