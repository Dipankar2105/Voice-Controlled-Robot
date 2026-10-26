"""Phase 3 verification tests: L298N motor driver, HC-SR04 ultrasonic driver, and pinout safety."""
from __future__ import annotations

import time
import pytest

from app.config import MotionConfig, MotorDriverConfig, SafetyConfig, UltrasonicConfig
from app.constants import (
    PIN_LEFT_ENA,
    PIN_LEFT_IN1,
    PIN_LEFT_IN2,
    PIN_RIGHT_ENB,
    PIN_RIGHT_IN3,
    PIN_RIGHT_IN4,
    PIN_ULTRASONIC_ECHO,
    PIN_ULTRASONIC_TRIG,
)
from app.hardware.gpio_adapter import FakeGPIOAdapter, HIGH, LOW
from app.hardware.interfaces import ObstacleState, UltrasonicReading
from app.hardware.motors import L298NMotorController
from app.hardware.ultrasonic import HCSR04UltrasonicSensor


# ==============================================================================
# Motor Driver Pin Mapping & Actuation Tests
# ==============================================================================

def test_motor_direction_to_pin_mapping():
    """Verify exact BCM pin states for forward, backward, turn left, turn right, and stop."""
    fake_gpio = FakeGPIOAdapter()
    motors = L298NMotorController(gpio=fake_gpio)
    motors.initialize()
    motors.enable(True)

    # 1. FORWARD
    motors.forward(speed=0.50)
    assert fake_gpio.pin_values[PIN_LEFT_IN1] == HIGH
    assert fake_gpio.pin_values[PIN_LEFT_IN2] == LOW
    assert fake_gpio.pin_values[PIN_RIGHT_IN3] == HIGH
    assert fake_gpio.pin_values[PIN_RIGHT_IN4] == LOW
    assert fake_gpio.pwm_channels[PIN_LEFT_ENA].duty_cycle == 50.0
    assert fake_gpio.pwm_channels[PIN_RIGHT_ENB].duty_cycle == 50.0

    # 2. BACKWARD
    motors.backward(speed=0.50)
    assert fake_gpio.pin_values[PIN_LEFT_IN1] == LOW
    assert fake_gpio.pin_values[PIN_LEFT_IN2] == HIGH
    assert fake_gpio.pin_values[PIN_RIGHT_IN3] == LOW
    assert fake_gpio.pin_values[PIN_RIGHT_IN4] == HIGH
    assert fake_gpio.pwm_channels[PIN_LEFT_ENA].duty_cycle == 50.0
    assert fake_gpio.pwm_channels[PIN_RIGHT_ENB].duty_cycle == 50.0

    # 3. TURN LEFT (in-place CCW: left reverse, right forward)
    motors.turn_left(speed=0.40)
    assert fake_gpio.pin_values[PIN_LEFT_IN1] == LOW
    assert fake_gpio.pin_values[PIN_LEFT_IN2] == HIGH
    assert fake_gpio.pin_values[PIN_RIGHT_IN3] == HIGH
    assert fake_gpio.pin_values[PIN_RIGHT_IN4] == LOW
    assert fake_gpio.pwm_channels[PIN_LEFT_ENA].duty_cycle == 40.0
    assert fake_gpio.pwm_channels[PIN_RIGHT_ENB].duty_cycle == 40.0

    # 4. TURN RIGHT (in-place CW: left forward, right reverse)
    motors.turn_right(speed=0.40)
    assert fake_gpio.pin_values[PIN_LEFT_IN1] == HIGH
    assert fake_gpio.pin_values[PIN_LEFT_IN2] == LOW
    assert fake_gpio.pin_values[PIN_RIGHT_IN3] == LOW
    assert fake_gpio.pin_values[PIN_RIGHT_IN4] == HIGH
    assert fake_gpio.pwm_channels[PIN_LEFT_ENA].duty_cycle == 40.0
    assert fake_gpio.pwm_channels[PIN_RIGHT_ENB].duty_cycle == 40.0

    # 5. STOP (all directional pins LOW, PWM 0%)
    motors.stop()
    assert fake_gpio.pin_values[PIN_LEFT_IN1] == LOW
    assert fake_gpio.pin_values[PIN_LEFT_IN2] == LOW
    assert fake_gpio.pin_values[PIN_RIGHT_IN3] == LOW
    assert fake_gpio.pin_values[PIN_RIGHT_IN4] == LOW
    assert fake_gpio.pwm_channels[PIN_LEFT_ENA].duty_cycle == 0.0
    assert fake_gpio.pwm_channels[PIN_RIGHT_ENB].duty_cycle == 0.0


def test_motor_pwm_bounds_and_clamping():
    """Motor speeds must be strictly clamped to motion_config.max_speed."""
    fake_gpio = FakeGPIOAdapter()
    motion_cfg = MotionConfig(max_speed=0.80)
    motors = L298NMotorController(gpio=fake_gpio, motion_config=motion_cfg)
    motors.initialize()
    motors.enable(True)

    # Command speeds exceeding max_speed
    motors.set_speeds(1.50, -2.00)
    assert motors.current_speeds == (0.80, -0.80)
    assert fake_gpio.pwm_channels[PIN_LEFT_ENA].duty_cycle == 80.0
    assert fake_gpio.pwm_channels[PIN_RIGHT_ENB].duty_cycle == 80.0


def test_motor_stop_overrides_all_commands():
    """Calling stop() immediately zeroing pins even when movement was active."""
    fake_gpio = FakeGPIOAdapter()
    motors = L298NMotorController(gpio=fake_gpio)
    motors.initialize()
    motors.enable(True)

    motors.forward(0.75)
    assert motors.current_speeds == (0.75, 0.75)

    motors.stop()
    assert motors.current_speeds == (0.0, 0.0)
    assert fake_gpio.pwm_channels[PIN_LEFT_ENA].duty_cycle == 0.0
    assert fake_gpio.pwm_channels[PIN_RIGHT_ENB].duty_cycle == 0.0
    assert fake_gpio.pin_values[PIN_LEFT_IN1] == LOW
    assert fake_gpio.pin_values[PIN_RIGHT_IN3] == LOW


def test_motor_cleanup_and_exception_handling():
    """cleanup() must safely stop motors and reset pin states."""
    fake_gpio = FakeGPIOAdapter()
    motors = L298NMotorController(gpio=fake_gpio)
    motors.initialize()
    motors.enable(True)
    motors.forward(0.50)

    motors.cleanup()
    assert not motors.is_enabled
    assert motors.current_speeds == (0.0, 0.0)


# ==============================================================================
# Ultrasonic Distance Sensor Tests
# ==============================================================================

def test_ultrasonic_distance_calculation_and_categories():
    """Verify flight-time calculation and state classification."""
    fake_gpio = FakeGPIOAdapter()
    sensor = HCSR04UltrasonicSensor(gpio=fake_gpio)
    sensor.initialize()

    # 1. Clear zone (60.0 cm -> pulse = 60 * 2 / 34300 ~ 0.003498s)
    dist = 60.0
    pulse_sec = (dist * 2.0) / 34300.0
    reading = sensor._classify_distance(dist, pulse_sec, time.time())
    assert reading.state == ObstacleState.CLEAR
    assert reading.distance_cm == 60.0
    assert reading.allows_forward_motion

    # 2. Warning zone (35.0 cm)
    dist = 35.0
    pulse_sec = (dist * 2.0) / 34300.0
    reading = sensor._classify_distance(dist, pulse_sec, time.time())
    assert reading.state == ObstacleState.WARNING
    assert reading.allows_forward_motion

    # 3. Blocked zone (20.0 cm)
    dist = 20.0
    pulse_sec = (dist * 2.0) / 34300.0
    reading = sensor._classify_distance(dist, pulse_sec, time.time())
    assert reading.state == ObstacleState.BLOCKED
    assert not reading.allows_forward_motion

    # 4. Critical zone (10.0 cm)
    dist = 10.0
    pulse_sec = (dist * 2.0) / 34300.0
    reading = sensor._classify_distance(dist, pulse_sec, time.time())
    assert reading.state == ObstacleState.CRITICAL
    assert not reading.allows_forward_motion

    # 5. Acoustic Blind Zone (< 2.0 cm) -> Fail-closed CRITICAL
    dist = 1.2
    pulse_sec = (dist * 2.0) / 34300.0
    reading = sensor._classify_distance(dist, pulse_sec, time.time())
    assert reading.state == ObstacleState.CRITICAL
    assert not reading.allows_forward_motion

    # 6. Out of Range (> 400.0 cm)
    dist = 450.0
    pulse_sec = (dist * 2.0) / 34300.0
    reading = sensor._classify_distance(dist, pulse_sec, time.time())
    assert reading.state == ObstacleState.OUT_OF_RANGE
    assert not reading.allows_forward_motion


def test_ultrasonic_timeout_returns_fault():
    """If echo pulse never arrives, sensor returns fail-closed FAULT reading."""
    fake_gpio = FakeGPIOAdapter()
    fake_gpio.echo_timeout = True  # Echo pin stays LOW

    cfg = UltrasonicConfig(timeout_sec=0.01)  # 10 ms test timeout
    sensor = HCSR04UltrasonicSensor(gpio=fake_gpio, config=cfg)
    sensor.initialize()

    reading = sensor.sample_distance_cm()
    assert reading.state == ObstacleState.FAULT
    assert not reading.is_valid
    assert not reading.allows_forward_motion


def test_ultrasonic_stale_reading_detection():
    """Readings older than stale_threshold_sec must be rejected as FAULT."""
    fake_gpio = FakeGPIOAdapter()
    sensor = HCSR04UltrasonicSensor(gpio=fake_gpio, stale_threshold_sec=0.10)
    sensor.initialize()

    # Manually inject an old valid reading (0.50s old)
    old_timestamp = time.time() - 0.50
    old_reading = UltrasonicReading(
        distance_cm=80.0,
        state=ObstacleState.CLEAR,
        timestamp=old_timestamp,
    )
    sensor._latest_reading = old_reading
    sensor._is_running = True  # Prevent sample_distance_cm() from overwriting

    reading = sensor.get_reading()
    assert reading.state == ObstacleState.FAULT
    assert not reading.allows_forward_motion


def test_forward_motion_refused_for_unknown_or_unsafe_distance():
    """allows_forward_motion must be strictly False for any non-clear state."""
    for state in (ObstacleState.CRITICAL, ObstacleState.BLOCKED, ObstacleState.FAULT, ObstacleState.OUT_OF_RANGE):
        r = UltrasonicReading(distance_cm=10.0 if state != ObstacleState.FAULT else None, state=state, timestamp=time.time())
        assert not r.allows_forward_motion, f"State {state} unexpectedly allowed forward motion!"


def test_hardware_free_imports_have_no_gpio_side_effects():
    """Importing motors, ultrasonic, and gpio_adapter must never trigger RPi.GPIO calls."""
    import app.hardware.gpio_adapter
    import app.hardware.motors
    import app.hardware.ultrasonic

    fake = FakeGPIOAdapter()
    assert fake.pin_values == {}
    assert fake.pwm_channels == {}


def test_uninitialized_ultrasonic_read_does_not_deadlock_and_fails_closed():
    """Direct get_reading on a fresh sensor must not deadlock its initialization lock."""
    fake_gpio = FakeGPIOAdapter()
    fake_gpio.echo_timeout = True
    sensor = HCSR04UltrasonicSensor(
        gpio=fake_gpio,
        config=UltrasonicConfig(timeout_sec=0.002),
        safety_config=SafetyConfig(),
    )
    reading = sensor.get_reading()
    assert reading.state == ObstacleState.FAULT
    assert reading.is_valid is False
    sensor.stop()


def test_nonfinite_ultrasonic_measurement_fails_closed():
    """NaN/Inf readings must never be classified as CLEAR."""
    sensor = HCSR04UltrasonicSensor(gpio=FakeGPIOAdapter())
    for value in (float("nan"), float("inf"), -1.0, 0.0):
        reading = sensor._classify_distance(value, 0.0, time.time())
        assert reading.state == ObstacleState.FAULT
        assert reading.is_valid is False
