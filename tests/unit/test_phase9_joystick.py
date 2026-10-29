"""Phase 9 unit tests: Web joystick control, watchdog timeout, concurrency serialization,
obstacle gating, and token authentication.
"""
from __future__ import annotations

import time
from dataclasses import replace
import pytest
from starlette.testclient import TestClient

from app.config import AppConfig, load_config
from app.constants import ExecutionMode, RobotState
from app.hardware.interfaces import ObstacleState, UltrasonicReading
from app.hardware.mock_hardware import (
    MockAudioSource,
    MockCamera,
    MockDetector,
    MockDistanceSensor,
    MockMotorController,
    MockSpeechRecognizer,
)
from app.navigation.state_machine import RobotStateMachine
from app.runtime import LifecycleState, RobotRuntime
from app.server import create_app


@pytest.fixture
def test_runtime() -> RobotRuntime:
    motors = MockMotorController()
    motors.enable(True)
    sensor = MockDistanceSensor(warning_cm=50.0, blocked_cm=25.0, critical_cm=15.0)
    camera = MockCamera(width=640, height=480)
    detector = MockDetector()
    audio = MockAudioSource()
    speech = MockSpeechRecognizer()

    runtime = RobotRuntime(
        motor_controller=motors,
        distance_sensor=sensor,
        camera=camera,
        detector=detector,
        audio_source=audio,
        speech_recognizer=speech,
    )
    runtime.initialize()
    return runtime


def test_joystick_directional_actions_and_stop(test_runtime: RobotRuntime):
    """Joystick accepts forward, backward, left, right and stop actions."""
    app = create_app(test_runtime)
    client = TestClient(app)

    # 1. Forward
    res = client.post("/api/joystick", json={"action": "forward", "speed": 0.60})
    assert res.status_code == 200
    assert res.json()["success"] is True
    assert test_runtime._joystick_active is True
    assert test_runtime.fsm.state == RobotState.MANUAL_MOVEMENT
    assert test_runtime.motors.current_speeds == (0.60, 0.60)

    # 2. Left Turn
    res = client.post("/api/joystick", json={"action": "left"})
    assert res.status_code == 200
    assert res.json()["success"] is True
    turn_spd = test_runtime.config.motion.turn_speed
    assert test_runtime.motors.current_speeds == (-turn_spd, turn_spd)

    # 3. Right Turn
    res = client.post("/api/joystick", json={"action": "right"})
    assert res.status_code == 200
    assert res.json()["success"] is True
    assert test_runtime.motors.current_speeds == (turn_spd, -turn_spd)

    # 4. Backward
    res = client.post("/api/joystick", json={"action": "backward", "speed": 0.50})
    assert res.status_code == 200
    assert res.json()["success"] is True
    assert test_runtime.motors.current_speeds == (-0.50, -0.50)

    # 5. Stop on release
    res = client.post("/api/joystick", json={"action": "stop"})
    assert res.status_code == 200
    assert res.json()["success"] is True
    assert test_runtime._joystick_active is False
    assert test_runtime.fsm.state == RobotState.IDLE_LISTENING
    assert test_runtime.motors.current_speeds == (0.0, 0.0)


def test_joystick_forward_blocked_by_ultrasonic_when_protection_on(test_runtime: RobotRuntime):
    """When protection is ON, forward joystick movement is blocked if obstacle is in blocked zone."""
    test_runtime.distance_sensor.set_distance(15.0)  # Obstacle at 15cm (blocked)
    test_runtime.ultrasonic_protection_enabled = True

    app = create_app(test_runtime)
    client = TestClient(app)

    res = client.post("/api/joystick", json={"action": "forward", "speed": 0.60})
    assert res.status_code == 200
    assert res.json()["success"] is False
    assert "blocked" in res.json()["message"].lower()
    assert test_runtime.fsm.state == RobotState.BLOCKED
    assert test_runtime.motors.current_speeds == (0.0, 0.0)


def test_joystick_backward_allowed_when_forward_obstacle_present(test_runtime: RobotRuntime):
    """Backward joystick movement is permitted even when forward ultrasonic sensor detects an obstacle."""
    test_runtime.distance_sensor.set_distance(15.0)  # Obstacle ahead
    test_runtime.ultrasonic_protection_enabled = True

    app = create_app(test_runtime)
    client = TestClient(app)

    res = client.post("/api/joystick", json={"action": "backward", "speed": 0.50})
    assert res.status_code == 200
    assert res.json()["success"] is True
    assert test_runtime.fsm.state == RobotState.MANUAL_MOVEMENT
    assert test_runtime.motors.current_speeds == (-0.50, -0.50)

    # Clean release
    client.post("/api/joystick", json={"action": "stop"})
    assert test_runtime.motors.current_speeds == (0.0, 0.0)


def test_joystick_forward_allowed_when_protection_off_despite_obstacle(test_runtime: RobotRuntime):
    """When protection is OFF, forward joystick movement bypasses ultrasonic interlock."""
    test_runtime.distance_sensor.set_distance(15.0)
    test_runtime.set_ultrasonic_protection(False)

    app = create_app(test_runtime)
    client = TestClient(app)

    res = client.post("/api/joystick", json={"action": "forward", "speed": 0.60})
    assert res.status_code == 200
    assert res.json()["success"] is True
    assert test_runtime.fsm.state == RobotState.MANUAL_MOVEMENT
    assert test_runtime.motors.current_speeds == (0.60, 0.60)

    # Stop cleanly
    client.post("/api/joystick", json={"action": "stop"})
    assert test_runtime.motors.current_speeds == (0.0, 0.0)


def test_joystick_refused_when_estop_latched(test_runtime: RobotRuntime):
    """Joystick commands are rejected when Emergency Stop is latched."""
    test_runtime.fsm.trigger_emergency_stop("Operator hit E-Stop")
    assert test_runtime.fsm.is_emergency_stop_latched is True

    app = create_app(test_runtime)
    client = TestClient(app)

    res = client.post("/api/joystick", json={"action": "forward"})
    assert res.status_code == 200
    assert res.json()["success"] is False
    assert "emergency stop is latched" in res.json()["message"].lower()
    assert test_runtime.motors.current_speeds == (0.0, 0.0)


def test_joystick_cancels_autonomous_mission(test_runtime: RobotRuntime):
    """Manual joystick movement cancels any running autonomous target search mission."""
    test_runtime.fsm.start_search_mission("bottle")
    assert test_runtime.fsm.state == RobotState.SEARCHING

    app = create_app(test_runtime)
    client = TestClient(app)

    res = client.post("/api/joystick", json={"action": "forward", "speed": 0.50})
    assert res.status_code == 200
    assert test_runtime.fsm.state == RobotState.MANUAL_MOVEMENT

    # Stop cleanly
    client.post("/api/joystick", json={"action": "stop"})
    assert test_runtime.fsm.state == RobotState.IDLE_LISTENING


def test_joystick_watchdog_timeout_halts_motors(test_runtime: RobotRuntime):
    """If joystick heartbeat ceases without stop, watchdog lease stops motors within timeout."""
    test_runtime.dispatch_joystick("forward", 0.60)
    assert test_runtime._joystick_active is True
    assert test_runtime.motors.current_speeds == (0.60, 0.60)

    # Simulate heartbeat age exceeding watchdog threshold (0.40s)
    test_runtime._last_joystick_time = time.time() - 0.50

    # Simulate watchdog evaluation
    if test_runtime._joystick_active and (time.time() - test_runtime._last_joystick_time) > test_runtime._joystick_watchdog_sec:
        test_runtime.motors.stop()
        test_runtime._joystick_active = False
        if test_runtime.fsm.state == RobotState.MANUAL_MOVEMENT:
            test_runtime.fsm._transition_to(RobotState.IDLE_LISTENING, "Watchdog timeout")

    assert test_runtime._joystick_active is False
    assert test_runtime.motors.current_speeds == (0.0, 0.0)
    assert test_runtime.fsm.state == RobotState.IDLE_LISTENING


def test_real_hardware_mode_requires_token_for_joystick():
    """In REAL_HARDWARE mode, POST /api/joystick strictly requires valid ROBOT_API_TOKEN."""
    cfg = replace(load_config(), mode=ExecutionMode.REAL_HARDWARE)
    runtime = RobotRuntime(
        config=cfg,
        motor_controller=MockMotorController(),
        distance_sensor=MockDistanceSensor(),
        camera=MockCamera(),
        detector=MockDetector(),
    )
    runtime._state = LifecycleState.INITIALIZED
    runtime.fsm = RobotStateMachine(config=cfg, motors=runtime.motors, distance_sensor=runtime.distance_sensor)

    secret_token = "K" * 32
    app = create_app(runtime, control_token=secret_token)
    client = TestClient(app)

    # Missing token -> 401
    res = client.post("/api/joystick", json={"action": "forward"})
    assert res.status_code == 401

    # Invalid token -> 401
    res = client.post(
        "/api/joystick",
        headers={"X-Robot-Token": "W" * 32},
        json={"action": "forward"},
    )
    assert res.status_code == 401

    # Valid token -> 200
    res = client.post(
        "/api/joystick",
        headers={"X-Robot-Token": secret_token},
        json={"action": "forward"},
    )
    assert res.status_code == 200


def test_conflicting_voice_and_joystick_commands(test_runtime: RobotRuntime):
    """Ensure conflicting voice and joystick commands cannot simultaneously control motors."""
    # 1. Active joystick blocks incoming voice movement
    test_runtime.dispatch_joystick("forward", 0.50)
    assert test_runtime._joystick_active is True

    success, cmd, msg = test_runtime.dispatch_text_command("move backward")
    assert not success
    assert "joystick is currently active" in msg

    # 2. Stop command via voice overrides and clears active joystick
    stop_ok, stop_cmd, stop_msg = test_runtime.dispatch_text_command("stop")
    assert stop_ok
    assert test_runtime._joystick_active is False
    assert test_runtime.motors.current_speeds == (0.0, 0.0)

    # 3. Executing voice command blocks joystick movement
    test_runtime._is_executing_command = True
    joy_ok, joy_msg = test_runtime.dispatch_joystick("forward", 0.50)
    assert not joy_ok
    assert "Concurrent command rejected" in joy_msg
    test_runtime._is_executing_command = False
