"""Unit tests for HC-SR04 ultrasonic reliability fixes and the Ultrasonic Obstacle Protection toggle."""
from __future__ import annotations

from dataclasses import replace
import time
import pytest
from starlette.testclient import TestClient

from app.command.command_parser import CommandAction, CommandIntent, CommandParser, RobotCommand
from app.config import AppConfig, UltrasonicConfig, load_config
from app.constants import ExecutionMode, FaultCode, RobotState
from app.hardware.gpio_adapter import FakeGPIOAdapter, HIGH, LOW
from app.hardware.interfaces import ObstacleState, UltrasonicReading
from app.hardware.mock_hardware import (
    MockAudioSource,
    MockCamera,
    MockDetector,
    MockDistanceSensor,
    MockMotorController,
    MockSpeechRecognizer,
)
from app.hardware.ultrasonic import HCSR04UltrasonicSensor
from app.navigation.state_machine import RobotStateMachine
from app.runtime import LifecycleState, RobotRuntime
from app.server import create_app


class GlitchSimulatingGPIO(FakeGPIOAdapter):
    """Simulates a short sub-millisecond noise pulse on ECHO pin."""

    def __init__(self, pulse_duration_sec: float = 0.000050):
        super().__init__()
        self.pulse_duration_sec = pulse_duration_sec
        self._trig_fired = False
        self._t_echo_start = None

    def output(self, pin: int, value: int) -> None:
        super().output(pin, value)
        if pin == 23 and value == HIGH:
            self._trig_fired = True

    def input(self, pin: int) -> int:
        if pin == 24:  # ECHO pin
            if not self._trig_fired:
                return LOW  # Settle check passes
            if self._t_echo_start is None:
                self._t_echo_start = time.perf_counter()
                return HIGH
            if (time.perf_counter() - self._t_echo_start) < self.pulse_duration_sec:
                return HIGH
            return LOW
        return super().input(pin)


def test_ultrasonic_glitch_rejection_submillisecond():
    """A pulse duration < 115 us (< 2 cm physical limit) must be rejected as FAULT, not CRITICAL."""
    gpio = GlitchSimulatingGPIO(pulse_duration_sec=0.000050)
    sensor = HCSR04UltrasonicSensor(gpio=gpio, config=UltrasonicConfig(timeout_sec=0.05))
    sensor.initialize()

    reading = sensor.sample_distance_cm()
    assert reading.state == ObstacleState.FAULT
    assert reading.distance_cm is None
    assert not reading.is_valid
    assert not reading.allows_forward_motion
    assert reading.raw_pulse_sec is not None
    assert reading.raw_pulse_sec < 0.000115


def test_ultrasonic_pretrigger_settle_stuck_high():
    """If ECHO pin is stuck HIGH before trigger, sensor must fail closed as FAULT."""
    fake_gpio = FakeGPIOAdapter()
    cfg = UltrasonicConfig(timeout_sec=0.05)
    sensor = HCSR04UltrasonicSensor(gpio=fake_gpio, config=cfg)
    sensor.initialize()

    # Force ECHO pin to remain HIGH
    fake_gpio.pin_values[cfg.echo_pin] = HIGH

    reading = sensor.sample_distance_cm()
    assert reading.state == ObstacleState.FAULT
    assert reading.distance_cm is None
    assert not reading.is_valid


def test_ultrasonic_rolling_median_filter_and_sample_count():
    """Valid samples must be median filtered; invalid samples must not corrupt the window."""
    fake_gpio = FakeGPIOAdapter()
    sensor = HCSR04UltrasonicSensor(gpio=fake_gpio, filter_window_size=5, min_valid_samples=2)
    sensor.initialize()
    sensor._is_running = True  # Prevent synchronous resample from empty mock

    now = time.time()
    # Sample 1: First valid sample (100 cm). Window has 1 sample < min_valid_samples (2).
    r1 = UltrasonicReading(distance_cm=100.0, state=ObstacleState.CLEAR, timestamp=now)
    sensor._update_reading(r1)
    assert sensor.valid_sample_count == 1
    # Until min_valid_samples (2) are accumulated, fail closed
    assert sensor.get_reading().state == ObstacleState.FAULT

    # Sample 2: Second valid sample (104 cm). Now 2 samples >= min_valid_samples (2).
    r2 = UltrasonicReading(distance_cm=104.0, state=ObstacleState.CLEAR, timestamp=now + 0.07)
    sensor._update_reading(r2)
    assert sensor.valid_sample_count == 2
    filtered = sensor.get_reading()
    assert filtered.is_valid
    assert filtered.state == ObstacleState.CLEAR
    assert filtered.distance_cm == 104.0  # sorted: [100.0, 104.0], index 2//2 = 1 -> 104.0

    # Sample 3: Glitch/spike sample (e.g. spurious 30 cm) followed by 102 cm
    r3 = UltrasonicReading(distance_cm=30.0, state=ObstacleState.WARNING, timestamp=now + 0.14)
    sensor._update_reading(r3)
    # Sorted: [30.0, 100.0, 104.0] -> median is 100.0 (spike suppressed!)
    assert sensor.get_reading().distance_cm == 100.0
    assert sensor.get_reading().state == ObstacleState.CLEAR


def test_ultrasonic_state_labels():
    """Driver must distinguish CLEAR, OBSTACLE, INVALID, and STALE states."""
    fake_gpio = FakeGPIOAdapter()
    sensor = HCSR04UltrasonicSensor(gpio=fake_gpio, stale_threshold_sec=0.20, min_valid_samples=1)
    sensor.initialize()
    sensor._is_running = True

    now = time.time()
    # CLEAR
    sensor._sample_history.clear()
    sensor._update_reading(UltrasonicReading(distance_cm=80.0, state=ObstacleState.CLEAR, timestamp=now))
    assert sensor.get_sensor_state_label() == "CLEAR"

    # OBSTACLE
    sensor._sample_history.clear()
    sensor._update_reading(UltrasonicReading(distance_cm=15.0, state=ObstacleState.CRITICAL, timestamp=now))
    assert sensor.get_sensor_state_label() == "OBSTACLE"

    # INVALID (fault / timeout)
    sensor._update_reading(UltrasonicReading.create_fault(now))
    assert sensor.get_sensor_state_label() == "INVALID"

    # STALE (older than stale_threshold_sec)
    sensor._sample_history.clear()
    sensor._update_reading(UltrasonicReading(distance_cm=80.0, state=ObstacleState.CLEAR, timestamp=now - 0.50))
    assert sensor.get_sensor_state_label() == "STALE"


def test_protection_defaults_to_off_for_demo():
    """Ultrasonic obstacle protection must default to False for demonstration launch."""
    runtime = RobotRuntime()
    assert runtime.ultrasonic_protection_enabled is False

    runtime.initialize()
    assert runtime.fsm is not None
    assert runtime.fsm.ultrasonic_protection_enabled is False


def test_forward_motion_blocked_when_protection_on_with_obstacle():
    """When protection is ON, forward motion is blocked by forward obstacle."""
    motors = MockMotorController()
    sensor = MockDistanceSensor()
    sensor.set_distance(12.0)  # Inside blocked zone (12 cm < 25 cm)

    fsm = RobotStateMachine(motors=motors, distance_sensor=sensor)
    fsm.ultrasonic_protection_enabled = True

    parser = CommandParser()
    cmd = parser.parse("move forward")
    accepted = fsm.handle_command(cmd)

    assert accepted is False
    assert fsm.state == RobotState.BLOCKED
    assert motors.current_speeds == (0.0, 0.0)


def test_forward_motion_blocked_when_protection_on_with_invalid_sensor():
    """When protection is ON, forward motion is blocked if sensor is invalid or timed out."""
    motors = MockMotorController()
    sensor = MockDistanceSensor()
    sensor.set_distance(None)  # Sensor FAULT / missing

    fsm = RobotStateMachine(motors=motors, distance_sensor=sensor)
    fsm.ultrasonic_protection_enabled = True
    assert fsm.ultrasonic_protection_enabled is True

    parser = CommandParser()
    cmd = parser.parse("move forward")
    accepted = fsm.handle_command(cmd)

    assert accepted is False
    assert fsm.state == RobotState.BLOCKED
    assert motors.current_speeds == (0.0, 0.0)


def test_forward_motion_allowed_when_protection_off_despite_obstacle():
    """When protection is OFF, ultrasonic interlock is bypassed while allowing forward testing."""
    motors = MockMotorController()
    sensor = MockDistanceSensor()
    sensor.set_distance(10.0)  # Normally blocked

    fsm = RobotStateMachine(motors=motors, distance_sensor=sensor)
    fsm.ultrasonic_protection_enabled = False

    parser = CommandParser()
    cmd = parser.parse("move forward")
    accepted = fsm.handle_command(cmd)

    assert accepted is True
    # In mock mode, short pulse completes and returns to IDLE_LISTENING
    assert fsm.state == RobotState.IDLE_LISTENING


def test_other_safety_remains_active_when_protection_off():
    """Turning OFF ultrasonic protection must NEVER disable E-Stop, active stop, or motor stop."""
    motors = MockMotorController()
    sensor = MockDistanceSensor()
    sensor.set_distance(10.0)

    fsm = RobotStateMachine(motors=motors, distance_sensor=sensor)
    fsm.ultrasonic_protection_enabled = False

    # 1. Emergency stop still latches immediately
    fsm.trigger_emergency_stop("Operator hit E-Stop")
    assert fsm.is_emergency_stop_latched is True
    assert fsm.state == RobotState.EMERGENCY_STOP
    assert motors.is_enabled is False

    # 2. Movement commands are strictly rejected while E-stop is latched
    parser = CommandParser()
    cmd = parser.parse("move forward")
    assert fsm.handle_command(cmd) is False

    # 3. Active STOP command still stops motors
    fsm.clear_emergency_stop()
    assert fsm.state == RobotState.IDLE_LISTENING
    stop_cmd = parser.parse("stop")
    assert fsm.handle_command(stop_cmd) is True


def test_api_ultrasonic_protection_toggle_and_telemetry():
    """API endpoint POST /api/ultrasonic/protection toggles protection and updates telemetry."""
    runtime = RobotRuntime()
    runtime.initialize()

    app = create_app(runtime)
    client = TestClient(app)

    # 1. Initial status shows protection OFF by default for demo
    res = client.get("/api/status")
    assert res.status_code == 200
    telemetry = res.json()
    assert telemetry["ultrasonic"]["protection_enabled"] is False
    assert "ULTRASONIC_PROTECTION_DISABLED" in telemetry["messages"]

    # 2. Toggle protection ON via API
    res = client.post("/api/ultrasonic/protection", json={"enabled": True})
    assert res.status_code == 200
    assert res.json()["success"] is True
    assert res.json()["enabled"] is True
    assert runtime.ultrasonic_protection_enabled is True
    assert runtime.fsm.ultrasonic_protection_enabled is True

    # 3. Status now reflects ON and removes warning notice
    res = client.get("/api/status")
    telemetry = res.json()
    assert telemetry["ultrasonic"]["protection_enabled"] is True
    assert "ULTRASONIC_PROTECTION_DISABLED" not in telemetry["messages"]

    # 4. Input validation: invalid payload rejected with 422
    res = client.post("/api/ultrasonic/protection", json={"invalid_field": 123})
    assert res.status_code == 422

    # 5. Toggle protection back OFF
    res = client.post("/api/ultrasonic/protection", json={"enabled": False})
    assert res.status_code == 200
    assert res.json()["enabled"] is False
    assert runtime.ultrasonic_protection_enabled is False
    assert runtime.fsm.ultrasonic_protection_enabled is False
    assert runtime.fsm.ultrasonic_protection_enabled is True


def test_real_hardware_mode_requires_token_to_toggle_protection():
    """In REAL_HARDWARE mode, POST /api/ultrasonic/protection requires valid ROBOT_API_TOKEN."""
    cfg = replace(load_config(), mode=ExecutionMode.REAL_HARDWARE)
    runtime = RobotRuntime(
        config=cfg,
        motor_controller=MockMotorController(),
        distance_sensor=MockDistanceSensor(),
        camera=MockCamera(),
        audio_source=MockAudioSource(),
        speech_recognizer=MockSpeechRecognizer(),
        detector=MockDetector(),
    )
    # Bypass mock checks in real mode for this isolated API test
    runtime._state = LifecycleState.INITIALIZED
    runtime.fsm = RobotStateMachine(config=cfg, motors=runtime.motors, distance_sensor=runtime.distance_sensor)

    secret_token = "A" * 32
    app = create_app(runtime, control_token=secret_token)
    client = TestClient(app)

    # Missing token -> 401
    res = client.post("/api/ultrasonic/protection", json={"enabled": False})
    assert res.status_code == 401

    # Invalid token -> 401
    res = client.post(
        "/api/ultrasonic/protection",
        headers={"X-Robot-Token": "B" * 32},
        json={"enabled": False},
    )
    assert res.status_code == 401

    # Valid token -> 200
    res = client.post(
        "/api/ultrasonic/protection",
        headers={"X-Robot-Token": secret_token},
        json={"enabled": False},
    )
    assert res.status_code == 200
    assert res.json()["enabled"] is False
