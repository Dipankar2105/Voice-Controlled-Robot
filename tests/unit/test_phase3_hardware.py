"""Phase 3 verification tests: L298N motor driver, HC-SR04 ultrasonic driver, and pinout safety."""
from __future__ import annotations

import threading
import time
import pytest

from app.config import MotionConfig, MotorDriverConfig, SafetyConfig, UltrasonicConfig, load_config
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


def test_isolated_channel_direction_zeroing():
    """Verify that when one channel is actuated, the zero-speed channel sets both direction pins strictly LOW."""
    fake_gpio = FakeGPIOAdapter()
    motors = L298NMotorController(gpio=fake_gpio)
    motors.initialize()
    motors.enable(True)

    # 1. Left only forward (left 0.50, right 0.0)
    motors.set_speeds(0.50, 0.0)
    assert fake_gpio.pin_values[PIN_LEFT_IN1] == HIGH
    assert fake_gpio.pin_values[PIN_LEFT_IN2] == LOW
    assert fake_gpio.pin_values[PIN_RIGHT_IN3] == LOW
    assert fake_gpio.pin_values[PIN_RIGHT_IN4] == LOW
    assert fake_gpio.pwm_channels[PIN_LEFT_ENA].duty_cycle == 50.0
    assert fake_gpio.pwm_channels[PIN_RIGHT_ENB].duty_cycle == 0.0

    # 2. Right only forward (left 0.0, right 0.50)
    motors.set_speeds(0.0, 0.50)
    assert fake_gpio.pin_values[PIN_LEFT_IN1] == LOW
    assert fake_gpio.pin_values[PIN_LEFT_IN2] == LOW
    assert fake_gpio.pin_values[PIN_RIGHT_IN3] == HIGH
    assert fake_gpio.pin_values[PIN_RIGHT_IN4] == LOW
    assert fake_gpio.pwm_channels[PIN_LEFT_ENA].duty_cycle == 0.0
    assert fake_gpio.pwm_channels[PIN_RIGHT_ENB].duty_cycle == 50.0

    # 3. Left only reverse (left -0.50, right 0.0)
    motors.set_speeds(-0.50, 0.0)
    assert fake_gpio.pin_values[PIN_LEFT_IN1] == LOW
    assert fake_gpio.pin_values[PIN_LEFT_IN2] == HIGH
    assert fake_gpio.pin_values[PIN_RIGHT_IN3] == LOW
    assert fake_gpio.pin_values[PIN_RIGHT_IN4] == LOW

    # 4. Right only reverse (left 0.0, right -0.50)
    motors.set_speeds(0.0, -0.50)
    assert fake_gpio.pin_values[PIN_LEFT_IN1] == LOW
    assert fake_gpio.pin_values[PIN_LEFT_IN2] == LOW
    assert fake_gpio.pin_values[PIN_RIGHT_IN3] == LOW
    assert fake_gpio.pin_values[PIN_RIGHT_IN4] == HIGH

    motors.stop()


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


def test_uninitialized_motor_stop_does_not_attempt_gpio_output():
    """Calling stop() before initialize() must be safe and must not attempt GPIO outputs."""
    fake_gpio = FakeGPIOAdapter()
    motors = L298NMotorController(gpio=fake_gpio)
    assert not motors._is_initialized

    # Calling stop() before initialize()
    motors.stop()
    assert motors.current_speeds == (0.0, 0.0)
    assert motors.last_command == "STOP"
    # Ensure no GPIO output calls were attempted on unconfigured pins
    assert len(fake_gpio.history) == 0

    # Repeated stop() must be idempotent
    motors.stop()
    assert len(fake_gpio.history) == 0


def test_repeated_cleanup_and_stop_calls_are_safe_and_idempotent():
    """cleanup() and subsequent stop() calls must remain completely safe and idempotent."""
    fake_gpio = FakeGPIOAdapter()
    motors = L298NMotorController(gpio=fake_gpio)
    motors.initialize()
    motors.enable(True)
    motors.forward(0.50)

    # First cleanup
    motors.cleanup()
    assert not motors._is_initialized
    assert not motors.is_enabled
    assert motors.current_speeds == (0.0, 0.0)

    # Second cleanup
    motors.cleanup()
    assert not motors._is_initialized
    assert not motors.is_enabled

    # Calling stop() after cleanup must not attempt GPIO outputs on released pins
    output_count_before = len([h for h in fake_gpio.history if h[1] == "OUTPUT"])
    motors.stop()
    output_count_after = len([h for h in fake_gpio.history if h[1] == "OUTPUT"])
    assert output_count_before == output_count_after


def test_uninitialized_and_disabled_set_speeds_cannot_energize_outputs():
    """Uninitialized or disabled controller must reject speed commands and never energize outputs."""
    fake_gpio = FakeGPIOAdapter()
    motors = L298NMotorController(gpio=fake_gpio)

    # 1. Uninitialized
    motors.set_speeds(0.60, 0.60)
    assert motors.current_speeds == (0.0, 0.0)
    assert len(fake_gpio.history) == 0

    # 2. Initialized but disabled
    motors.initialize()
    motors.enable(False)
    motors.set_speeds(0.60, 0.60)
    assert motors.current_speeds == (0.0, 0.0)
    assert fake_gpio.pwm_channels[PIN_LEFT_ENA].duty_cycle == 0.0
    assert fake_gpio.pwm_channels[PIN_RIGHT_ENB].duty_cycle == 0.0


def test_fsm_initialization_with_uninitialized_motors_is_safe():
    """State machine construction with uninitialized motors must not throw or attempt unconfigured GPIO."""
    fake_gpio = FakeGPIOAdapter()
    motors = L298NMotorController(gpio=fake_gpio)
    assert not motors._is_initialized

    # FSM constructor transitions to IDLE_LISTENING, which calls motors.stop()
    from app.navigation.state_machine import RobotStateMachine
    fsm = RobotStateMachine(motors=motors)
    assert fsm.state.value == "IDLE_LISTENING"
    assert motors.current_speeds == (0.0, 0.0)
    # fake_gpio must have zero output calls from the uninitialized motor controller
    motor_outputs = [h for h in fake_gpio.history if h[1] == "OUTPUT" and h[2] in (PIN_LEFT_IN1, PIN_LEFT_IN2, PIN_RIGHT_IN3, PIN_RIGHT_IN4)]
    assert len(motor_outputs) == 0


# ==============================================================================
# YAML Config Pin Mapping Consistency Tests
# ==============================================================================

def test_yaml_config_pin_mapping_matches_constants():
    """Verify that default_config.yaml pin values match constants.py after the polarity alignment."""
    cfg = load_config()
    assert cfg.motor.left_ena_pin == PIN_LEFT_ENA, "YAML left ENA pin must match constants.py"
    assert cfg.motor.left_in1_pin == PIN_LEFT_IN1, "YAML left IN1 pin must match constants.py"
    assert cfg.motor.left_in2_pin == PIN_LEFT_IN2, "YAML left IN2 pin must match constants.py"
    assert cfg.motor.right_enb_pin == PIN_RIGHT_ENB, "YAML right ENB pin must match constants.py"
    assert cfg.motor.right_in3_pin == PIN_RIGHT_IN3, "YAML right IN3 pin must match constants.py"
    assert cfg.motor.right_in4_pin == PIN_RIGHT_IN4, "YAML right IN4 pin must match constants.py"


def test_motor_direction_with_yaml_config():
    """Verify motor direction pin states using the actual YAML-loaded configuration."""
    cfg = load_config()
    fake_gpio = FakeGPIOAdapter()
    motors = L298NMotorController(gpio=fake_gpio, config=cfg.motor, motion_config=cfg.motion)
    motors.initialize()
    motors.enable(True)

    in3 = cfg.motor.right_in3_pin
    in4 = cfg.motor.right_in4_pin

    # FORWARD: IN3 HIGH, IN4 LOW
    motors.forward(speed=0.50)
    assert fake_gpio.pin_values[cfg.motor.left_in1_pin] == HIGH
    assert fake_gpio.pin_values[cfg.motor.left_in2_pin] == LOW
    assert fake_gpio.pin_values[in3] == HIGH
    assert fake_gpio.pin_values[in4] == LOW
    assert fake_gpio.pwm_channels[cfg.motor.left_ena_pin].duty_cycle == 50.0
    assert fake_gpio.pwm_channels[cfg.motor.right_enb_pin].duty_cycle == 50.0

    # BACKWARD: IN3 LOW, IN4 HIGH
    motors.backward(speed=0.50)
    assert fake_gpio.pin_values[cfg.motor.left_in1_pin] == LOW
    assert fake_gpio.pin_values[cfg.motor.left_in2_pin] == HIGH
    assert fake_gpio.pin_values[in3] == LOW
    assert fake_gpio.pin_values[in4] == HIGH

    # STOP: all LOW, PWM 0%
    motors.stop()
    assert fake_gpio.pin_values[cfg.motor.left_in1_pin] == LOW
    assert fake_gpio.pin_values[cfg.motor.left_in2_pin] == LOW
    assert fake_gpio.pin_values[in3] == LOW
    assert fake_gpio.pin_values[in4] == LOW
    assert fake_gpio.pwm_channels[cfg.motor.left_ena_pin].duty_cycle == 0.0
    assert fake_gpio.pwm_channels[cfg.motor.right_enb_pin].duty_cycle == 0.0

    motors.cleanup()


def test_motor_initialize_all_pins_low():
    """Verify that initialize() sets ALL direction and enable pins to LOW before any command."""
    fake_gpio = FakeGPIOAdapter()
    motors = L298NMotorController(gpio=fake_gpio)
    motors.initialize()

    for pin in [motors.config.left_in1_pin, motors.config.left_in2_pin,
                motors.config.right_in3_pin, motors.config.right_in4_pin]:
        assert fake_gpio.pin_values[pin] == LOW, f"Pin {pin} must be LOW after initialize()"

    assert fake_gpio.pwm_channels[motors.config.left_ena_pin].duty_cycle == 0.0
    assert fake_gpio.pwm_channels[motors.config.right_enb_pin].duty_cycle == 0.0
    assert not motors.is_enabled
    motors.cleanup()


def test_motor_disabled_rejects_and_stops():
    """Verify that set_speeds() on a disabled controller calls _apply_hardware_stop()."""
    fake_gpio = FakeGPIOAdapter()
    motors = L298NMotorController(gpio=fake_gpio)
    motors.initialize()

    motors.set_speeds(0.5, 0.5)
    assert motors.current_speeds == (0.0, 0.0)
    assert fake_gpio.pwm_channels[motors.config.left_ena_pin].duty_cycle == 0.0
    assert fake_gpio.pwm_channels[motors.config.right_enb_pin].duty_cycle == 0.0
    motors.cleanup()


# ==============================================================================
# Phase 2: Direction, Balance & Calibration Verification Tests
# ==============================================================================

def test_differential_turning_pin_polarities_and_speeds():
    """Verify differential turn pin states, opposite polarities, and configurable turn speed."""
    fake_gpio = FakeGPIOAdapter()
    motion_cfg = MotionConfig(turn_speed=0.35, max_speed=1.0)
    motors = L298NMotorController(gpio=fake_gpio, motion_config=motion_cfg)
    motors.initialize()
    motors.enable(True)

    in3 = motors.config.right_in3_pin
    in4 = motors.config.right_in4_pin

    # 1. Turn Left (in-place CCW: Left Reverse, Right Forward)
    motors.turn_left()
    assert motors.current_speeds == (-0.35, 0.35)
    assert fake_gpio.pin_values[motors.config.left_in1_pin] == LOW
    assert fake_gpio.pin_values[motors.config.left_in2_pin] == HIGH
    assert fake_gpio.pin_values[in3] == HIGH
    assert fake_gpio.pin_values[in4] == LOW
    assert fake_gpio.pwm_channels[motors.config.left_ena_pin].duty_cycle == 35.0
    assert fake_gpio.pwm_channels[motors.config.right_enb_pin].duty_cycle == 35.0

    # 2. Turn Right (in-place CW: Left Forward, Right Reverse)
    motors.turn_right()
    assert motors.current_speeds == (0.35, -0.35)
    assert fake_gpio.pin_values[motors.config.left_in1_pin] == HIGH
    assert fake_gpio.pin_values[motors.config.left_in2_pin] == LOW
    assert fake_gpio.pin_values[in3] == LOW
    assert fake_gpio.pin_values[in4] == HIGH
    assert fake_gpio.pwm_channels[motors.config.left_ena_pin].duty_cycle == 35.0
    assert fake_gpio.pwm_channels[motors.config.right_enb_pin].duty_cycle == 35.0

    # 3. Explicit override speed
    motors.turn_left(speed=0.55)
    assert motors.current_speeds == (-0.55, 0.55)
    assert fake_gpio.pwm_channels[motors.config.left_ena_pin].duty_cycle == pytest.approx(55.0)
    assert fake_gpio.pwm_channels[motors.config.right_enb_pin].duty_cycle == pytest.approx(55.0)

    motors.stop()
    assert motors.current_speeds == (0.0, 0.0)
    motors.cleanup()


def test_speed_compensation_trim_application():
    """Verify left_trim and right_trim scale PWM duty cycles accurately across all motion primitives."""
    fake_gpio = FakeGPIOAdapter()
    motion_cfg = MotionConfig(left_trim=0.90, right_trim=0.80, max_speed=1.0)
    motors = L298NMotorController(gpio=fake_gpio, motion_config=motion_cfg)
    motors.initialize()
    motors.enable(True)

    # 1. Forward at 0.50
    motors.set_speeds(0.50, 0.50)
    assert fake_gpio.pwm_channels[motors.config.left_ena_pin].duty_cycle == pytest.approx(45.0, 0.01)  # 50 * 0.90
    assert fake_gpio.pwm_channels[motors.config.right_enb_pin].duty_cycle == pytest.approx(40.0, 0.01)  # 50 * 0.80

    # 2. Backward at 0.50
    motors.backward(0.50)
    assert fake_gpio.pwm_channels[motors.config.left_ena_pin].duty_cycle == pytest.approx(45.0, 0.01)
    assert fake_gpio.pwm_channels[motors.config.right_enb_pin].duty_cycle == pytest.approx(40.0, 0.01)

    # 3. Turn Left (in-place CCW) at 0.40
    motors.turn_left(0.40)
    assert fake_gpio.pwm_channels[motors.config.left_ena_pin].duty_cycle == pytest.approx(36.0, 0.01)  # 40 * 0.90
    assert fake_gpio.pwm_channels[motors.config.right_enb_pin].duty_cycle == pytest.approx(32.0, 0.01)  # 40 * 0.80

    # 4. Turn Right (in-place CW) at 0.40
    motors.turn_right(0.40)
    assert fake_gpio.pwm_channels[motors.config.left_ena_pin].duty_cycle == pytest.approx(36.0, 0.01)  # 40 * 0.90
    assert fake_gpio.pwm_channels[motors.config.right_enb_pin].duty_cycle == pytest.approx(32.0, 0.01)  # 40 * 0.80

    # 5. Duty cycle upper clamping at 100% even if trim exceeds 1.0
    motors.stop()
    motors.cleanup()


def test_guaranteed_motor_stop_after_pulse_and_exception():
    """move_pulse must strictly de-energize PWM and direction pins upon completion or error."""
    fake_gpio = FakeGPIOAdapter()
    motors = L298NMotorController(gpio=fake_gpio)
    motors.initialize()
    motors.enable(True)

    # 1. Normal pulse completes, returns True, and leaves hardware in full stop
    pulse_res = motors.move_pulse(0.40, 0.40, duration_sec=0.05)
    assert pulse_res is True
    assert motors.current_speeds == (0.0, 0.0)
    assert fake_gpio.pwm_channels[motors.config.left_ena_pin].duty_cycle == 0.0
    assert fake_gpio.pwm_channels[motors.config.right_enb_pin].duty_cycle == 0.0
    assert fake_gpio.pin_values[motors.config.left_in1_pin] == LOW
    assert fake_gpio.pin_values[motors.config.left_in2_pin] == LOW
    assert fake_gpio.pin_values[motors.config.right_in3_pin] == LOW
    assert fake_gpio.pin_values[motors.config.right_in4_pin] == LOW

    # 2. Non-finite speed raises exception and guarantees stop
    with pytest.raises(ValueError, match="finite numbers"):
        motors.set_speeds(float("nan"), 0.50)
    assert motors.current_speeds == (0.0, 0.0)
    assert fake_gpio.pwm_channels[motors.config.left_ena_pin].duty_cycle == 0.0
    assert fake_gpio.pwm_channels[motors.config.right_enb_pin].duty_cycle == 0.0

    # 3. Exception during pulse execution ensures stop via finally block
    class InterruptedSleep:
        def __call__(self, duration):
            raise KeyboardInterrupt("Simulated emergency interrupt")

    import time
    orig_sleep = time.sleep
    try:
        time.sleep = InterruptedSleep()
        with pytest.raises(KeyboardInterrupt):
            motors.move_pulse(0.50, 0.50, 0.10)
        assert motors.current_speeds == (0.0, 0.0)
        assert fake_gpio.pwm_channels[motors.config.left_ena_pin].duty_cycle == 0.0
        assert fake_gpio.pwm_channels[motors.config.right_enb_pin].duty_cycle == 0.0
        assert fake_gpio.pin_values[motors.config.left_in1_pin] == LOW
        assert fake_gpio.pin_values[motors.config.right_in3_pin] == LOW
    finally:
        time.sleep = orig_sleep

    motors.cleanup()


def test_move_pulse_acceptance_semantics_and_rejection():
    """Verify that move_pulse() returns True ONLY for accepted, executed pulses."""
    fake_gpio = FakeGPIOAdapter()
    motors = L298NMotorController(gpio=fake_gpio)

    # 1. Uninitialized controller rejects pulse and returns False
    assert motors.move_pulse(0.40, 0.40, 0.05) is False
    assert len(fake_gpio.history) == 0

    # 2. Initialized but disabled controller rejects pulse and returns False
    motors.initialize()
    motors.enable(False)
    assert motors.move_pulse(0.40, 0.40, 0.05) is False
    assert motors.current_speeds == (0.0, 0.0)

    # 3. Enabled controller with deadband or non-positive duration requests
    motors.enable(True)

    # Zero speeds (deadband)
    assert motors.move_pulse(0.0, 0.0, 0.05) is False
    assert motors.current_speeds == (0.0, 0.0)

    # Sub-deadband speeds (< 1e-4)
    assert motors.move_pulse(0.00005, -0.00005, 0.05) is False
    assert motors.current_speeds == (0.0, 0.0)

    # Zero duration
    assert motors.move_pulse(0.40, 0.40, 0.0) is False
    assert motors.current_speeds == (0.0, 0.0)

    # Negative duration
    assert motors.move_pulse(0.40, 0.40, -0.10) is False
    assert motors.current_speeds == (0.0, 0.0)

    # Non-finite duration (NaN)
    assert motors.move_pulse(0.40, 0.40, float("nan")) is False
    assert motors.current_speeds == (0.0, 0.0)

    # 4. Non-finite speed raises ValueError according to set_speeds() semantics
    # while still guaranteeing full hardware stop via finally block
    with pytest.raises(ValueError, match="finite numbers"):
        motors.move_pulse(float("nan"), 0.40, 0.05)
    assert motors.current_speeds == (0.0, 0.0)
    assert fake_gpio.pwm_channels[motors.config.left_ena_pin].duty_cycle == 0.0
    assert fake_gpio.pwm_channels[motors.config.right_enb_pin].duty_cycle == 0.0

    # 5. Accepted command returns True and leaves outputs in full stop
    pulse_ok = motors.move_pulse(0.35, 0.35, 0.05)
    assert pulse_ok is True
    assert motors.current_speeds == (0.0, 0.0)
    assert fake_gpio.pwm_channels[motors.config.left_ena_pin].duty_cycle == 0.0
    assert fake_gpio.pwm_channels[motors.config.right_enb_pin].duty_cycle == 0.0
    assert fake_gpio.pin_values[motors.config.left_in1_pin] == LOW
    assert fake_gpio.pin_values[motors.config.left_in2_pin] == LOW
    assert fake_gpio.pin_values[motors.config.right_in3_pin] == LOW
    assert fake_gpio.pin_values[motors.config.right_in4_pin] == LOW

    motors.cleanup()


def test_configuration_loading_for_turns_and_trim():
    """AppConfig correctly loads and validates turn speed, calibration duration, and trim values."""
    from app.config import AppConfig, ConfigurationError, load_config_from_dict

    valid_dict = {
        "motion": {
            "turn_speed": 0.45,
            "calibrated_360_duration_sec": 8.5,
            "left_trim": 0.95,
            "right_trim": 0.90,
        }
    }
    cfg = load_config_from_dict(valid_dict)
    assert cfg.motion.turn_speed == 0.45
    assert cfg.motion.calibrated_360_duration_sec == 8.5
    assert cfg.motion.left_trim == 0.95
    assert cfg.motion.right_trim == 0.90

    # Invalid trim below 0.10
    with pytest.raises(ConfigurationError, match="Speed trim 'left_trim'"):
        load_config_from_dict({"motion": {"left_trim": 0.05}})

    # Invalid trim above 1.00
    with pytest.raises(ConfigurationError, match="Speed trim 'right_trim'"):
        load_config_from_dict({"motion": {"right_trim": 1.05}})

    # Invalid turn speed <= 0
    with pytest.raises(ConfigurationError, match="turn_speed"):
        load_config_from_dict({"motion": {"turn_speed": 0.0}})

    # Invalid turn speed > 1.0
    with pytest.raises(ConfigurationError, match="turn_speed"):
        load_config_from_dict({"motion": {"turn_speed": 1.2}})

    # Invalid calibrated 360 duration below 1.0s
    with pytest.raises(ConfigurationError, match="calibrated_360_duration_sec"):
        load_config_from_dict({"motion": {"calibrated_360_duration_sec": 0.5}})

    # Invalid calibrated 360 duration above 60.0s
    with pytest.raises(ConfigurationError, match="calibrated_360_duration_sec"):
        load_config_from_dict({"motion": {"calibrated_360_duration_sec": 75.0}})


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


def test_motor_pulse_cooperative_stop_interruption():
    """Verify that calling stop() while move_pulse() is active immediately terminates the pulse."""
    fake_gpio = FakeGPIOAdapter()
    motors = L298NMotorController(gpio=fake_gpio)
    motors.initialize()
    motors.enable(True)

    pulse_completed = []

    def pulse_worker():
        ok = motors.move_pulse(0.40, 0.40, 0.50)
        pulse_completed.append(ok)

    t = threading.Thread(target=pulse_worker)
    t.start()
    time.sleep(0.03)  # Allow pulse to start

    # Stop mid-pulse
    motors.stop()
    t.join(timeout=0.20)

    assert not t.is_alive(), "move_pulse thread did not terminate promptly after stop()"
    assert pulse_completed == [False], "move_pulse should return False when interrupted by stop()"
    assert motors.current_speeds == (0.0, 0.0)
    motors.cleanup()


def test_motor_initialization_failure_rollback():
    """Verify that an exception during initialize() cleans up partial pins and marks controller uninitialized and faulted."""
    class FailingGPIOAdapter(FakeGPIOAdapter):
        def setup(self, pin: int, mode: int, pull_up_down: int = 20, initial: int = 0) -> None:
            if pin == 27:  # Fail on IN2
                raise IOError("Simulated GPIO setup failure")
            super().setup(pin, mode, pull_up_down, initial)

    failing_gpio = FailingGPIOAdapter()
    motors = L298NMotorController(gpio=failing_gpio)

    with pytest.raises(IOError, match="Simulated GPIO setup failure"):
        motors.initialize()

    assert motors.is_initialized is False
    assert motors.has_fault is True
    assert motors.is_enabled is False


def test_motor_reinitialization_after_cleanup():
    """Verify that after cleanup(), a controller can be successfully re-initialized and operated."""
    fake_gpio = FakeGPIOAdapter()
    motors = L298NMotorController(gpio=fake_gpio)
    motors.initialize()
    motors.enable(True)
    motors.forward(0.40)
    assert motors.current_speeds == (0.40, 0.40)

    motors.cleanup()
    assert motors.is_initialized is False
    assert motors.is_enabled is False

    # Re-initialize
    motors.initialize()
    assert motors.is_initialized is True
    assert motors.is_enabled is False

    motors.enable(True)
    motors.backward(0.30)
    assert motors.current_speeds == (-0.30, -0.30)
    motors.cleanup()


def test_motor_hardware_output_exception_reporting_and_fault_state():
    """Verify that hardware exceptions in set_speeds() trigger fault state and emergency zeroing."""
    class FailingOutputGPIO(FakeGPIOAdapter):
        def output(self, pin: int, value: int) -> None:
            if pin == 17:
                raise RuntimeError("Simulated bus write error on IN1")
            super().output(pin, value)

    failing_gpio = FailingOutputGPIO()
    motors = L298NMotorController(gpio=failing_gpio)
    motors.initialize()
    motors.enable(True)

    with pytest.raises(RuntimeError, match="Simulated bus write error"):
        motors.forward(0.40)

    assert motors.has_fault is True
    assert "FAULT" in motors.last_command
    motors.cleanup()


def test_motor_concurrent_stop_and_motion():
    """Verify thread safety when stop() races with set_speeds()."""
    fake_gpio = FakeGPIOAdapter()
    motors = L298NMotorController(gpio=fake_gpio)
    motors.initialize()
    motors.enable(True)

    def speed_worker():
        for i in range(50):
            try:
                motors.set_speeds(0.30, 0.30)
            except Exception:
                pass
            time.sleep(0.001)

    def stop_worker():
        for i in range(50):
            motors.stop()
            time.sleep(0.001)

    t1 = threading.Thread(target=speed_worker)
    t2 = threading.Thread(target=stop_worker)
    t1.start()
    t2.start()
    t1.join(timeout=1.0)
    t2.join(timeout=1.0)

    # After both finish, calling stop() leaves outputs strictly zeroed
    motors.stop()
    assert motors.current_speeds == (0.0, 0.0)
    motors.cleanup()
