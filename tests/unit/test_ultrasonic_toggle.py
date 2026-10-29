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


def test_protection_defaults_to_on_at_startup():
    """Ultrasonic obstacle protection must default to True at startup."""
    runtime = RobotRuntime()
    assert runtime.ultrasonic_protection_enabled is True

    runtime.initialize()
    assert runtime.fsm is not None
    assert runtime.fsm.ultrasonic_protection_enabled is True


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

    # 1. Initial status shows protection ON by default
    res = client.get("/api/status")
    assert res.status_code == 200
    telemetry = res.json()
    assert telemetry["ultrasonic"]["protection_enabled"] is True
    assert "ULTRASONIC_PROTECTION_DISABLED" not in telemetry["messages"]

    # 2. Toggle protection OFF via API
    res = client.post("/api/ultrasonic/protection", json={"enabled": False})
    assert res.status_code == 200
    assert res.json()["success"] is True
    assert res.json()["enabled"] is False
    assert runtime.ultrasonic_protection_enabled is False
    assert runtime.fsm.ultrasonic_protection_enabled is False

    # 3. Status now reflects OFF and adds warning notice
    res = client.get("/api/status")
    telemetry = res.json()
    assert telemetry["ultrasonic"]["protection_enabled"] is False
    assert "ULTRASONIC_PROTECTION_DISABLED" in telemetry["messages"]

    # 4. Input validation: invalid payload rejected with 422
    res = client.post("/api/ultrasonic/protection", json={"invalid_field": 123})
    assert res.status_code == 422

    # 5. Toggle protection back ON
    res = client.post("/api/ultrasonic/protection", json={"enabled": True})
    assert res.status_code == 200
    assert res.json()["enabled"] is True
    assert runtime.ultrasonic_protection_enabled is True
    assert runtime.fsm.ultrasonic_protection_enabled is True

    # 6. Status now reflects ON and removes warning notice
    res = client.get("/api/status")
    telemetry = res.json()
    assert telemetry["ultrasonic"]["protection_enabled"] is True
    assert "ULTRASONIC_PROTECTION_DISABLED" not in telemetry["messages"]


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


class SimulatedEchoGPIO(FakeGPIOAdapter):
    """Accurately simulates ultrasonic echo pulses with custom rise delay and duration."""

    def __init__(
        self,
        pulse_duration_sec: float = 0.001,
        rise_delay_sec: float = 0.0001,
        stuck_high_after_rise: bool = False,
    ):
        super().__init__()
        self.pulse_duration_sec = pulse_duration_sec
        self.rise_delay_sec = rise_delay_sec
        self.stuck_high_after_rise = stuck_high_after_rise
        self._trig_time = None

    def output(self, pin: int, value: int) -> None:
        super().output(pin, value)
        if pin == 23 and value == HIGH:
            self._trig_time = time.perf_counter()

    def input(self, pin: int) -> int:
        if pin == 24:
            if self._trig_time is None:
                return LOW
            elapsed = time.perf_counter() - self._trig_time
            if elapsed < self.rise_delay_sec:
                return LOW
            if self.stuck_high_after_rise:
                return HIGH
            if elapsed < (self.rise_delay_sec + self.pulse_duration_sec):
                return HIGH
            return LOW
        return super().input(pin)


def test_ultrasonic_valid_near_and_far_readings():
    """Verify accurate measurement and classification for near (5cm) and far (250cm) physical targets."""
    # Near: 5 cm -> pulse = 5.0 * 2 / 34300 ~ 0.0002915 s
    pulse_near = (5.0 * 2.0) / 34300.0
    gpio_near = SimulatedEchoGPIO(pulse_duration_sec=pulse_near, rise_delay_sec=0.00005)
    sensor_near = HCSR04UltrasonicSensor(gpio=gpio_near, config=UltrasonicConfig(timeout_sec=0.05))
    sensor_near.initialize()
    r_near = sensor_near.sample_distance_cm()
    assert r_near.is_valid
    assert r_near.state == ObstacleState.CRITICAL
    assert 3.5 <= r_near.distance_cm <= 6.5

    # Far: 250 cm -> pulse = 250.0 * 2 / 34300 ~ 0.014577 s
    pulse_far = (250.0 * 2.0) / 34300.0
    gpio_far = SimulatedEchoGPIO(pulse_duration_sec=pulse_far, rise_delay_sec=0.00005)
    sensor_far = HCSR04UltrasonicSensor(gpio=gpio_far, config=UltrasonicConfig(timeout_sec=0.05))
    sensor_far.initialize()
    r_far = sensor_far.sample_distance_cm()
    assert r_far.is_valid
    assert r_far.state == ObstacleState.CLEAR
    assert 245.0 <= r_far.distance_cm <= 255.0


def test_ultrasonic_no_echo_rising_edge():
    """When ECHO line never rises (pin stays LOW), driver returns fail-closed FAULT within bounded timeout."""
    fake_gpio = FakeGPIOAdapter()
    fake_gpio.pin_values[24] = LOW  # Stays LOW
    cfg = UltrasonicConfig(timeout_sec=0.01)
    sensor = HCSR04UltrasonicSensor(gpio=fake_gpio, config=cfg)
    sensor.initialize()

    t0 = time.perf_counter()
    reading = sensor.sample_distance_cm()
    elapsed = time.perf_counter() - t0

    assert reading.state == ObstacleState.FAULT
    assert not reading.is_valid
    assert not reading.allows_forward_motion
    assert reading.distance_cm is None
    assert elapsed < 0.05  # Bound check


def test_ultrasonic_no_echo_falling_edge():
    """When ECHO rises but never falls (stuck HIGH after trigger), driver returns FAULT within bounded timeout."""
    gpio = SimulatedEchoGPIO(rise_delay_sec=0.0001, stuck_high_after_rise=True)
    cfg = UltrasonicConfig(timeout_sec=0.01)
    sensor = HCSR04UltrasonicSensor(gpio=gpio, config=cfg)
    sensor.initialize()

    t0 = time.perf_counter()
    reading = sensor.sample_distance_cm()
    elapsed = time.perf_counter() - t0

    assert reading.state == ObstacleState.FAULT
    assert not reading.is_valid
    assert not reading.allows_forward_motion
    assert reading.distance_cm is None
    assert elapsed < 0.05


def test_ultrasonic_zero_duration_and_impossible_pulse():
    """Zero duration or unphysical pulse must fail closed as FAULT."""
    fake_gpio = FakeGPIOAdapter()
    sensor = HCSR04UltrasonicSensor(gpio=fake_gpio)
    sensor.initialize()

    # Direct test of classification on zero and negative pulse / distance
    r_zero = sensor._classify_distance(0.0, 0.0, time.time())
    assert r_zero.state == ObstacleState.FAULT
    assert not r_zero.is_valid

    r_neg = sensor._classify_distance(-15.0, -0.001, time.time())
    assert r_neg.state == ObstacleState.FAULT
    assert not r_neg.is_valid


def test_ultrasonic_gpio_exceptions_handled_gracefully():
    """Driver must catch GPIO exceptions during trigger, settle, echo wait and cleanup, returning FAULT."""
    class FailingGPIO(FakeGPIOAdapter):
        def __init__(self, fail_on_op: str):
            super().__init__()
            self.fail_on_op = fail_on_op

        def output(self, pin: int, value: int) -> None:
            if self.fail_on_op == "output":
                raise IOError("Simulated GPIO output bus error")
            super().output(pin, value)

        def input(self, pin: int) -> int:
            if self.fail_on_op == "input":
                raise IOError("Simulated GPIO input line error")
            return super().input(pin)

        def cleanup(self, pins=None) -> None:
            if self.fail_on_op == "cleanup":
                raise IOError("Simulated GPIO cleanup error")
            super().cleanup(pins)

    # Output failure on trigger
    sensor_out = HCSR04UltrasonicSensor(gpio=FailingGPIO("output"))
    sensor_out.initialize()
    assert sensor_out.sample_distance_cm().state == ObstacleState.FAULT

    # Input failure on echo read
    sensor_in = HCSR04UltrasonicSensor(gpio=FailingGPIO("input"))
    sensor_in.initialize()
    assert sensor_in.sample_distance_cm().state == ObstacleState.FAULT

    # Cleanup exception on stop
    sensor_clean = HCSR04UltrasonicSensor(gpio=FailingGPIO("cleanup"))
    sensor_clean.initialize()
    sensor_clean.stop()  # Must not raise
    assert not sensor_clean._is_initialized


def test_ultrasonic_initialization_failure_and_rollback():
    """If GPIO setup fails during initialize(), sensor marks itself uninitialized and cleans up partial pins."""
    class SetupFailingGPIO(FakeGPIOAdapter):
        def setup(self, pin: int, mode: int, pull_up_down: int = 20, initial: int = 0) -> None:
            if pin == 24:  # Fail on ECHO setup
                raise RuntimeError("Simulated ECHO pin hardware reservation failure")
            super().setup(pin, mode, pull_up_down, initial)

    failing_gpio = SetupFailingGPIO()
    sensor = HCSR04UltrasonicSensor(gpio=failing_gpio)

    with pytest.raises(RuntimeError):
        sensor.initialize()

    assert not sensor._is_initialized

    # Calling sample_distance_cm on a failed-init sensor returns FAULT cleanly
    reading = sensor.sample_distance_cm()
    assert reading.state == ObstacleState.FAULT
    assert not reading.is_valid


def test_ultrasonic_stop_clears_history_and_allows_reinitialization():
    """Calling stop() clears median filter history and consecutive faults, and allows clean reinit."""
    fake_gpio = FakeGPIOAdapter()
    sensor = HCSR04UltrasonicSensor(gpio=fake_gpio, filter_window_size=5, min_valid_samples=2)
    sensor.initialize()
    sensor._is_running = True

    now = time.time()
    sensor._update_reading(UltrasonicReading(distance_cm=60.0, state=ObstacleState.CLEAR, timestamp=now))
    assert sensor.valid_sample_count == 1

    sensor.stop()
    assert not sensor._is_running
    assert not sensor._is_initialized
    assert sensor.valid_sample_count == 0

    # Re-initialize and verify clean state
    sensor.initialize()
    assert sensor._is_initialized
    assert sensor.valid_sample_count == 0
    sensor.stop()


def test_navigation_behavior_when_distance_sensor_unavailable_or_faulty():
    """RobotStateMachine must reject forward motion when distance sensor is unavailable or returning FAULT."""
    fsm = RobotStateMachine(
        config=load_config(),
        motors=MockMotorController(),
        distance_sensor=None,  # No distance sensor
    )
    fsm.ultrasonic_protection_enabled = True

    # When sensor is None:
    assert fsm._is_forward_path_safe() is False

    # When sensor returns FAULT:
    class FaultyDistanceSensor:
        def get_reading(self):
            return UltrasonicReading.create_fault()

    fsm.distance_sensor = FaultyDistanceSensor()
    assert fsm._is_forward_path_safe() is False

    # When sensor throws an exception:
    class CrashingDistanceSensor:
        def get_reading(self):
            raise RuntimeError("Sensor I/O timeout")

    fsm.distance_sensor = CrashingDistanceSensor()
    assert fsm._is_forward_path_safe() is False
