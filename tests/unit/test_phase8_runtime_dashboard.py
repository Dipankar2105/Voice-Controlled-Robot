"""Phase 8 verification tests: Unified Runtime, Dashboard, Concurrency, and Safety.

Tests:
- Single authoritative runtime with one owner per physical resource.
- Startup failure and partial initialization cleanup.
- Repeated shutdown idempotency.
- Concurrent command rejection and emergency stop preemption.
- Strictly read-only status reporting without side effects.
- Clear status messages: TARGET_FOUND, TARGET_NOT_FOUND, OBSTACLE_BLOCKED,
  CAMERA_FAILURE, DETECTOR_FAILURE, SENSOR_FAILURE.
- Mission cancellation and safe stopping.
- Dashboard HTTP and MJPEG streaming endpoints with client disconnect handling.
"""
from __future__ import annotations

import json
from dataclasses import replace
import time
from typing import Optional
import pytest
from fastapi.testclient import TestClient

from app.command.command_parser import CommandAction
from app.config import AppConfig, ConfigurationError, load_config
from app.constants import ExecutionMode, FaultCode, RobotState
from app.hardware.interfaces import DetectionResult, ObstacleState, UltrasonicReading
from app.hardware.mock_hardware import (
    MockAudioSource,
    MockCamera,
    MockDetector,
    MockDistanceSensor,
    MockMotorController,
    MockSpeechRecognizer,
)
from app.runtime import LifecycleState, RobotRuntime
from app.server import create_app
from app.status import SubsystemStatus


@pytest.fixture
def mock_subsystems():
    """Create independent mock subsystems for testing."""
    motors = MockMotorController(initially_enabled=False)
    sensor = MockDistanceSensor(initial_distance_cm=55.0)
    camera = MockCamera(width=640, height=480)
    audio = MockAudioSource()
    speech = MockSpeechRecognizer()
    detector = MockDetector(class_names=["cup", "bottle", "backpack", "person"])
    return {
        "motors": motors,
        "sensor": sensor,
        "camera": camera,
        "audio": audio,
        "speech": speech,
        "detector": detector,
    }


def test_startup_and_resource_initialization_once(mock_subsystems):
    """Runtime initializes resources once and does not report READY prematurely."""
    cfg = load_config()
    runtime = RobotRuntime(
        config=cfg,
        motor_controller=mock_subsystems["motors"],
        distance_sensor=mock_subsystems["sensor"],
        camera=mock_subsystems["camera"],
        audio_source=mock_subsystems["audio"],
        speech_recognizer=mock_subsystems["speech"],
        detector=mock_subsystems["detector"],
    )

    # Before initialize: UNINITIALIZED
    assert runtime.lifecycle_state == LifecycleState.CREATED
    for status in runtime._subsystem_statuses.values():
        assert status == SubsystemStatus.UNINITIALIZED

    runtime.initialize()

    # After initialize: INITIALIZED and subsystems READY
    assert runtime.lifecycle_state == LifecycleState.INITIALIZED
    assert runtime.motors.current_speeds == (0.0, 0.0)
    assert not runtime.motors.is_enabled
    for status in runtime._subsystem_statuses.values():
        assert status == SubsystemStatus.READY

    # Start runtime
    runtime.start()
    assert runtime.lifecycle_state == LifecycleState.RUNNING
    assert runtime.motors.is_enabled
    assert runtime.motors.current_speeds == (0.0, 0.0)

    runtime.cleanup()


def test_startup_failure_and_partial_initialization_cleanup():
    """Failure during initialize() triggers partial cleanup and releases open resources."""
    mock_motors = MockMotorController()
    mock_cam = MockCamera()

    # Subclass mock camera to simulate failure after motors were initialized
    class FaultyDetector(MockDetector):
        def __init__(self):
            super().__init__()
            raise RuntimeError("Synthetic detector device allocation failure")

    runtime = RobotRuntime(
        motor_controller=mock_motors,
        camera=mock_cam,
    )
    # Inject faulty detector
    runtime.detector = None  # Will fail when detector initialization fails or forced
    
    # We force an exception during initialize by providing invalid config
    cfg = AppConfig(mode=ExecutionMode.REAL_HARDWARE)
    runtime.config = cfg

    with pytest.raises(ConfigurationError):
        runtime.initialize()

    # Verify that partial cleanup was executed and fault is recorded
    assert runtime.lifecycle_state == LifecycleState.STOPPED
    assert runtime._active_fault == FaultCode.HARDWARE_INIT_FAILED
    for status in runtime._subsystem_statuses.values():
        assert status == SubsystemStatus.FAULT


def test_repeated_shutdown_idempotency(mock_subsystems):
    """Calling stop() and cleanup() repeatedly never crashes and remains safe."""
    runtime = RobotRuntime(
        motor_controller=mock_subsystems["motors"],
        distance_sensor=mock_subsystems["sensor"],
        camera=mock_subsystems["camera"],
        audio_source=mock_subsystems["audio"],
        speech_recognizer=mock_subsystems["speech"],
        detector=mock_subsystems["detector"],
    )
    runtime.start()
    assert runtime.lifecycle_state == LifecycleState.RUNNING

    # First stop
    runtime.stop()
    assert runtime.lifecycle_state == LifecycleState.STOPPED

    # Repeated stop
    runtime.stop()
    assert runtime.lifecycle_state == LifecycleState.STOPPED

    # First cleanup
    runtime.cleanup()
    assert runtime.lifecycle_state == LifecycleState.CLEANED_UP

    # Repeated cleanup
    runtime.cleanup()
    assert runtime.lifecycle_state == LifecycleState.CLEANED_UP


def test_concurrent_command_rejection_and_emergency_override(mock_subsystems):
    """Concurrent non-emergency commands are rejected; emergency stops always preempt."""
    runtime = RobotRuntime(
        motor_controller=mock_subsystems["motors"],
        distance_sensor=mock_subsystems["sensor"],
        camera=mock_subsystems["camera"],
        audio_source=mock_subsystems["audio"],
        speech_recognizer=mock_subsystems["speech"],
        detector=mock_subsystems["detector"],
    )
    runtime.start()

    # Simulate an actively executing command lock
    runtime._is_executing_command = True

    # Attempt to dispatch a new movement command
    success, cmd, msg = runtime.dispatch_text_command("move forward")
    assert not success
    assert "Concurrent command rejected" in msg

    # Emergency stop must OVERRIDE the lock immediately
    estop_success, estop_cmd, estop_msg = runtime.dispatch_text_command("emergency stop")
    assert estop_success
    assert runtime.fsm.is_emergency_stop_latched
    assert runtime.fsm.state == RobotState.EMERGENCY_STOP

    # Reset executing flag
    runtime._is_executing_command = False
    runtime.cleanup()


def test_status_reporting_is_strictly_read_only(mock_subsystems):
    """get_telemetry() must read only cached state and never trigger sensor pulses or movement."""
    sensor = mock_subsystems["sensor"]
    runtime = RobotRuntime(
        motor_controller=mock_subsystems["motors"],
        distance_sensor=sensor,
        camera=mock_subsystems["camera"],
        audio_source=mock_subsystems["audio"],
        speech_recognizer=mock_subsystems["speech"],
        detector=mock_subsystems["detector"],
    )
    runtime.initialize()
    runtime._cached_distance_reading = UltrasonicReading(
        distance_cm=42.5,
        state=ObstacleState.CLEAR,
        timestamp=time.time(),
    )

    # Monkeypatch sensor sample method to verify it is NOT called during get_telemetry
    sample_called = False

    def fail_if_sampled():
        nonlocal sample_called
        sample_called = True
        raise AssertionError("Hardware sampling triggered during read-only status call!")

    sensor.get_reading = fail_if_sampled

    # Request telemetry
    telemetry = runtime.get_telemetry()

    assert not sample_called, "get_telemetry() must NEVER trigger active sensor sampling"
    assert telemetry["ultrasonic"]["distance_cm"] == 42.5
    assert telemetry["ultrasonic"]["obstacle_state"] == "CLEAR"
    assert telemetry["health"]["mode"] == "SIMULATION"
    assert telemetry["motor"]["command_state"] == "STOPPED"

    # Verify JSON serializability (no NaN or Inf)
    serialized = json.dumps(telemetry)
    assert "NaN" not in serialized
    assert "Infinity" not in serialized

    runtime.cleanup()


def test_clear_status_messages_evaluation(mock_subsystems):
    """Verify distinct activation of all six required status messages."""
    runtime = RobotRuntime(
        motor_controller=mock_subsystems["motors"],
        distance_sensor=mock_subsystems["sensor"],
        camera=mock_subsystems["camera"],
        audio_source=mock_subsystems["audio"],
        speech_recognizer=mock_subsystems["speech"],
        detector=mock_subsystems["detector"],
    )
    runtime.initialize()

    # 1. OBSTACLE_BLOCKED
    runtime.fsm._transition_to(RobotState.BLOCKED, "Obstacle in path")
    runtime._cached_distance_reading = UltrasonicReading(12.0, ObstacleState.BLOCKED, time.time())
    t = runtime.get_telemetry()
    assert "OBSTACLE_BLOCKED" in t["messages"]

    # 2. TARGET_FOUND
    runtime.fsm._transition_to(RobotState.TARGET_REACHED, "Arrived at target")
    t = runtime.get_telemetry()
    assert "TARGET_FOUND" in t["messages"]

    # 3. TARGET_NOT_FOUND
    runtime.fsm._transition_to(RobotState.TARGET_NOT_FOUND, "Exhausted 360 scan")
    t = runtime.get_telemetry()
    assert "TARGET_NOT_FOUND" in t["messages"]

    # 4. CAMERA_FAILURE
    runtime.camera._is_opened = False
    t = runtime.get_telemetry()
    assert "CAMERA_FAILURE" in t["messages"]
    runtime.camera._is_opened = True

    # 5. DETECTOR_FAILURE
    runtime._subsystem_statuses["detector"] = SubsystemStatus.FAULT
    t = runtime.get_telemetry()
    assert "DETECTOR_FAILURE" in t["messages"]
    runtime._subsystem_statuses["detector"] = SubsystemStatus.READY

    # 6. SENSOR_FAILURE
    runtime._cached_distance_reading = UltrasonicReading.create_fault()
    t = runtime.get_telemetry()
    assert "SENSOR_FAILURE" in t["messages"]

    runtime.cleanup()


def test_mission_cancellation_and_safe_stopping(mock_subsystems):
    """Cancelling a search mission halts motors immediately and updates context."""
    runtime = RobotRuntime(
        motor_controller=mock_subsystems["motors"],
        distance_sensor=mock_subsystems["sensor"],
        camera=mock_subsystems["camera"],
        audio_source=mock_subsystems["audio"],
        speech_recognizer=mock_subsystems["speech"],
        detector=mock_subsystems["detector"],
    )
    runtime.start()

    # Start search mission
    success, cmd, msg = runtime.dispatch_text_command("find cup")
    assert success
    assert runtime.fsm.state == RobotState.SEARCHING

    # Cancel search mission
    c_success, c_cmd, c_msg = runtime.dispatch_text_command("cancel search")
    assert c_success
    assert runtime.fsm.state == RobotState.CANCELLED
    assert runtime.motors.current_speeds == (0.0, 0.0)
    assert runtime.fsm.context.terminal_message is not None

    runtime.cleanup()


def test_dashboard_api_endpoints_and_telemetry(mock_subsystems):
    """Test FastAPI dashboard endpoints via TestClient without opening real network sockets."""
    runtime = RobotRuntime(
        motor_controller=mock_subsystems["motors"],
        distance_sensor=mock_subsystems["sensor"],
        camera=mock_subsystems["camera"],
        audio_source=mock_subsystems["audio"],
        speech_recognizer=mock_subsystems["speech"],
        detector=mock_subsystems["detector"],
    )
    runtime.start()

    app = create_app(runtime)
    client = TestClient(app)

    # 1. GET / (Dashboard HTML)
    res = client.get("/")
    assert res.status_code == 200
    assert "text/html" in res.headers["content-type"]
    assert "SS_PROJECT_VCR // AUTONOMOUS RUNTIME" in res.text
    assert "OPTICAL FRAME & TRACKING" in res.text
    assert "ULTRASONIC PROXIMITY" in res.text

    # 2. GET /api/status (Read-only status)
    res = client.get("/api/status")
    assert res.status_code == 200
    data = res.json()
    assert "health" in data
    assert "speech" in data
    assert "fsm" in data
    assert "ultrasonic" in data
    assert "motor" in data
    assert data["health"]["mode"] == "SIMULATION"

    # 3. POST /api/command (Valid command)
    res = client.post("/api/command", json={"command": "status"})
    assert res.status_code == 200
    assert res.json()["success"] is True

    # 4. POST /api/command (Empty command -> 400)
    res = client.post("/api/command", json={"command": "   "})
    assert res.status_code == 400

    # 5. POST /api/stop
    res = client.post("/api/stop")
    assert res.status_code == 200
    assert res.json()["success"] is True

    # 6. POST /api/emergency-stop and clear-estop
    res = client.post("/api/emergency-stop")
    assert res.status_code == 200
    assert runtime.fsm.is_emergency_stop_latched

    res = client.post("/api/clear-estop")
    assert res.status_code == 200
    assert not runtime.fsm.is_emergency_stop_latched

    runtime.cleanup()


def test_dashboard_stream_disconnect_handling(mock_subsystems):
    """Disconnecting from /api/camera/stream terminates cleanly without degrading runtime safety."""
    runtime = RobotRuntime(
        motor_controller=mock_subsystems["motors"],
        distance_sensor=mock_subsystems["sensor"],
        camera=mock_subsystems["camera"],
        audio_source=mock_subsystems["audio"],
        speech_recognizer=mock_subsystems["speech"],
        detector=mock_subsystems["detector"],
    )
    runtime.start()

    app = create_app(runtime)
    client = TestClient(app)

    # Bounded stream request (2 frames) verifies MJPEG headers and graceful termination
    res = client.get("/api/camera/stream?max_frames=2")
    assert res.status_code == 200
    assert b"--frame" in res.content
    assert b"Content-Type: image/jpeg" in res.content

    # Verify runtime is still fully intact and healthy after client stream terminates
    assert runtime.lifecycle_state == LifecycleState.RUNNING
    assert runtime.get_telemetry()["health"]["is_healthy"] is True

    runtime.cleanup()


def test_dashboard_concurrent_command_rejection_http_409(mock_subsystems):
    """POST /api/command returns HTTP 409 Conflict when a command is rejected due to concurrency."""
    runtime = RobotRuntime(
        motor_controller=mock_subsystems["motors"],
        distance_sensor=mock_subsystems["sensor"],
        camera=mock_subsystems["camera"],
        audio_source=mock_subsystems["audio"],
        speech_recognizer=mock_subsystems["speech"],
        detector=mock_subsystems["detector"],
    )
    runtime.start()

    app = create_app(runtime)
    client = TestClient(app)

    # Simulate an active command lock
    runtime._is_executing_command = True

    res = client.post("/api/command", json={"command": "move forward"})
    assert res.status_code == 409
    assert "Concurrent command rejected" in res.json()["detail"]

    runtime._is_executing_command = False
    runtime.cleanup()


def test_real_control_routes_require_operator_token(mock_subsystems):
    """When configured, command/control routes reject unauthenticated requests."""
    cfg = load_config()
    runtime = RobotRuntime(
        config=cfg,
        motor_controller=mock_subsystems["motors"],
        distance_sensor=mock_subsystems["sensor"],
        camera=mock_subsystems["camera"],
        audio_source=mock_subsystems["audio"],
        speech_recognizer=mock_subsystems["speech"],
        detector=mock_subsystems["detector"],
    )
    runtime.initialize()
    token = "test-operator-token-with-more-than-24-chars"
    client = TestClient(create_app(runtime, control_token=token))
    try:
        assert client.post("/api/command", json={"command": "find cup"}).status_code == 401
        accepted = client.post(
            "/api/command", json={"command": "status"}, headers={"X-Robot-Token": token}
        )
        assert accepted.status_code == 200
        assert client.post("/api/auth-check", headers={"X-Robot-Token": "wrong-token"}).status_code == 401
        # A stop is intentionally unauthenticated because it only removes actuation.
        assert client.post("/api/stop").status_code == 200
        html = client.get("/").text
        assert "CONTROL_TOKEN_REQUIRED = true" in html
        assert token not in html
    finally:
        runtime.cleanup()


def test_real_hardware_dashboard_refuses_missing_control_token(mock_subsystems):
    """The dashboard factory must not expose unauthenticated real motor controls."""
    cfg = replace(load_config(), mode=ExecutionMode.REAL_HARDWARE)
    runtime = RobotRuntime(config=cfg)
    with pytest.raises(ValueError, match="ROBOT_API_TOKEN"):
        create_app(runtime)


def test_stale_cached_distance_is_reported_as_fault_not_clear(mock_subsystems):
    """A stale last-good reading cannot be reported as healthy/clear telemetry."""
    cfg = load_config()
    runtime = RobotRuntime(
        config=cfg,
        motor_controller=mock_subsystems["motors"],
        distance_sensor=mock_subsystems["sensor"],
        camera=mock_subsystems["camera"],
        audio_source=mock_subsystems["audio"],
        speech_recognizer=mock_subsystems["speech"],
        detector=mock_subsystems["detector"],
    )
    runtime.initialize()
    runtime._cached_distance_reading = UltrasonicReading(
        distance_cm=100.0, state=ObstacleState.CLEAR, timestamp=time.time() - 2.0
    )
    telemetry = runtime.get_telemetry()
    assert telemetry["ultrasonic"]["is_valid"] is False
    assert telemetry["ultrasonic"]["obstacle_state"] == "FAULT"
    assert telemetry["ultrasonic"]["sensor_healthy"] is False
    assert telemetry["motor"]["safety_block_reason"]
    runtime.cleanup()


def test_health_endpoint_uses_cached_sensor_data_only(mock_subsystems):
    """Health reporting must never trigger a physical ultrasonic read."""
    from unittest.mock import MagicMock

    cfg = load_config()
    runtime = RobotRuntime(
        config=cfg,
        motor_controller=mock_subsystems["motors"],
        distance_sensor=mock_subsystems["sensor"],
        camera=mock_subsystems["camera"],
        audio_source=mock_subsystems["audio"],
        speech_recognizer=mock_subsystems["speech"],
        detector=mock_subsystems["detector"],
    )
    runtime.initialize()
    runtime._cached_distance_reading = UltrasonicReading(
        distance_cm=55.0, state=ObstacleState.WARNING, timestamp=time.time()
    )
    mock_subsystems["sensor"].get_reading = MagicMock(side_effect=AssertionError("status triggered sensing"))
    health = runtime.get_health()
    assert health.obstacle_state == ObstacleState.WARNING
    mock_subsystems["sensor"].get_reading.assert_not_called()
    runtime.cleanup()


def test_real_hardware_start_is_refused_without_operator_token(monkeypatch):
    """Missing network-control token must fail before constructing/initializing runtime."""
    from app.main import run_server
    monkeypatch.delenv("ROBOT_API_TOKEN", raising=False)
    assert run_server(mode_override="REAL_HARDWARE") == 2
