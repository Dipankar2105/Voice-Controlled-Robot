"""L298N Dual H-Bridge Motor Controller for SS_PROJECT_VCR_FRESH.

Controls 4WD chassis via:
- Left motors: ENA (BCM 18 / PWM), IN1 (BCM 17), IN2 (BCM 27)
- Right motors: ENB (BCM 13 / PWM), IN3 (BCM 22), IN4 (BCM 5)

Safety Invariants:
- Outputs default to stopped (PWM 0%, IN1..4 LOW).
- Never energizes outputs during construction or imports.
- Stop overrides all active commands.
- Fails closed on any exception.
- Reports commanded electrical state, not physical wheel displacement.
"""
from __future__ import annotations

import threading
import math
import time
from typing import Optional, Tuple

from app.config import MotionConfig, MotorDriverConfig
from app.constants import (
    DEFAULT_MANUAL_SPEED,
    MAX_SPEED,
    PIN_LEFT_ENA,
    PIN_LEFT_IN1,
    PIN_LEFT_IN2,
    PIN_RIGHT_ENB,
    PIN_RIGHT_IN3,
    PIN_RIGHT_IN4,
    PWM_FREQUENCY_HZ,
    SEARCH_SPEED,
    TURN_SPEED,
)
from app.hardware.gpio_adapter import (
    BCM,
    GPIOAdapter,
    HIGH,
    LOW,
    OUT,
    PWMChannel,
    FakeGPIOAdapter,
)
from app.hardware.interfaces import MotorControllerInterface
from app.logger import get_logger

logger = get_logger("motors")


class L298NMotorController(MotorControllerInterface):
    """Authoritative motor driver for 4WD DC chassis via L298N dual H-bridge."""

    def __init__(
        self,
        gpio: Optional[GPIOAdapter] = None,
        config: Optional[MotorDriverConfig] = None,
        motion_config: Optional[MotionConfig] = None,
    ):
        self.config = config or MotorDriverConfig()
        self.motion_config = motion_config or MotionConfig()
        self.gpio: GPIOAdapter = gpio or FakeGPIOAdapter()

        self._is_enabled: bool = False
        self._is_initialized: bool = False
        self._pwm_left: Optional[PWMChannel] = None
        self._pwm_right: Optional[PWMChannel] = None

        self._current_left: float = 0.0
        self._current_right: float = 0.0
        self._last_command: str = "STOP"
        self._lock = threading.Lock()

    def initialize(self) -> None:
        """Configure BCM pin modes and initialize PWM channels without energizing outputs."""
        with self._lock:
            if self._is_initialized:
                return

            self.gpio.setmode(BCM)

            # Configure directional pins as OUT, initially LOW
            direction_pins = [
                self.config.left_in1_pin,
                self.config.left_in2_pin,
                self.config.right_in3_pin,
                self.config.right_in4_pin,
            ]
            for pin in direction_pins:
                self.gpio.setup(pin, OUT, initial=LOW)

            # Configure PWM speed pins as OUT, initially LOW
            self.gpio.setup(self.config.left_ena_pin, OUT, initial=LOW)
            self.gpio.setup(self.config.right_enb_pin, OUT, initial=LOW)

            # Start PWM channels at 0% duty cycle (completely stopped)
            self._pwm_left = self.gpio.PWM(self.config.left_ena_pin, self.config.pwm_frequency_hz)
            self._pwm_right = self.gpio.PWM(self.config.right_enb_pin, self.config.pwm_frequency_hz)
            self._pwm_left.start(0.0)
            self._pwm_right.start(0.0)

            self._is_initialized = True
            self._is_enabled = False
            logger.info("L298N motor controller initialized. Outputs are strictly LOW (0%% PWM).")

    def enable(self, enabled: bool = True) -> None:
        """Enable or disable software output to motors."""
        with self._lock:
            self._is_enabled = enabled
            if not enabled:
                self._apply_hardware_stop()
            logger.info("Motor driver enable state: %s", self._is_enabled)

    @property
    def is_enabled(self) -> bool:
        return self._is_enabled

    @property
    def current_speeds(self) -> Tuple[float, float]:
        """Return commanded (left, right) speeds in [-1.0, 1.0]."""
        return (self._current_left, self._current_right)

    @property
    def last_command(self) -> str:
        return self._last_command

    def stop(self) -> None:
        """Immediately pull all directional pins LOW and zero both PWM channels."""
        with self._lock:
            self._apply_hardware_stop()

    def _apply_hardware_stop(self) -> None:
        """Internal zeroing routine executed with lock held."""
        self._current_left = 0.0
        self._current_right = 0.0
        self._last_command = "STOP"

        try:
            # 1. Zero PWM duty cycles first
            if self._pwm_left:
                self._pwm_left.ChangeDutyCycle(0.0)
            if self._pwm_right:
                self._pwm_right.ChangeDutyCycle(0.0)

            # 2. Pull all directional pins LOW
            self.gpio.output(self.config.left_in1_pin, LOW)
            self.gpio.output(self.config.left_in2_pin, LOW)
            self.gpio.output(self.config.right_in3_pin, LOW)
            self.gpio.output(self.config.right_in4_pin, LOW)
        except Exception as exc:
            logger.error("Exception during hardware stop: %s", exc)

    def set_speeds(self, left: float, right: float) -> None:
        """Set normalized speeds for left and right channels in [-1.0, 1.0].
        Clamps to motion_config.max_speed. Fails closed if not enabled or uninitialized.
        """
        with self._lock:
            if not self._is_initialized:
                logger.warning("set_speeds() rejected: controller not initialized.")
                return

            if not self._is_enabled:
                logger.debug("set_speeds() ignored: controller disabled.")
                self._apply_hardware_stop()
                return

            try:
                raw_left = float(left)
                raw_right = float(right)
                if not math.isfinite(raw_left) or not math.isfinite(raw_right):
                    raise ValueError("Motor speed commands must be finite numbers")
                # Clamp within [-max_speed, max_speed]
                max_spd = self.motion_config.max_speed
                clamped_left = max(-max_spd, min(max_spd, raw_left))
                clamped_right = max(-max_spd, min(max_spd, raw_right))

                if abs(clamped_left) < 1e-4 and abs(clamped_right) < 1e-4:
                    self._apply_hardware_stop()
                    return

                # Left Channel Direction
                if clamped_left > 0:
                    self.gpio.output(self.config.left_in1_pin, HIGH)
                    self.gpio.output(self.config.left_in2_pin, LOW)
                else:
                    self.gpio.output(self.config.left_in1_pin, LOW)
                    self.gpio.output(self.config.left_in2_pin, HIGH)

                # Right Channel Direction
                if clamped_right > 0:
                    self.gpio.output(self.config.right_in3_pin, HIGH)
                    self.gpio.output(self.config.right_in4_pin, LOW)
                else:
                    self.gpio.output(self.config.right_in3_pin, LOW)
                    self.gpio.output(self.config.right_in4_pin, HIGH)

                # Apply PWM duty cycles (0.0 to 100.0)
                duty_left = abs(clamped_left) * 100.0
                duty_right = abs(clamped_right) * 100.0

                if self._pwm_left:
                    self._pwm_left.ChangeDutyCycle(duty_left)
                if self._pwm_right:
                    self._pwm_right.ChangeDutyCycle(duty_right)

                self._current_left = clamped_left
                self._current_right = clamped_right
                self._last_command = f"SPEEDS({clamped_left:.2f}, {clamped_right:.2f})"
            except Exception as exc:
                logger.error("Exception in set_speeds, executing emergency stop: %s", exc)
                self._apply_hardware_stop()
                raise

    def forward(self, speed: Optional[float] = None) -> None:
        """Command nominal forward motion at bounded speed."""
        spd = self.motion_config.manual_speed if speed is None else float(speed)
        self.set_speeds(spd, spd)
        self._last_command = f"FORWARD({spd:.2f})"

    def backward(self, speed: Optional[float] = None) -> None:
        """Command nominal reverse motion at bounded speed."""
        spd = self.motion_config.manual_speed if speed is None else float(speed)
        self.set_speeds(-spd, -spd)
        self._last_command = f"BACKWARD({spd:.2f})"

    def turn_left(self, speed: Optional[float] = None) -> None:
        """Command in-place counter-clockwise rotation (left reverse, right forward)."""
        spd = self.motion_config.turn_speed if speed is None else float(speed)
        self.set_speeds(-spd, spd)
        self._last_command = f"TURN_LEFT({spd:.2f})"

    def turn_right(self, speed: Optional[float] = None) -> None:
        """Command in-place clockwise rotation (left forward, right reverse)."""
        spd = self.motion_config.turn_speed if speed is None else float(speed)
        self.set_speeds(spd, -spd)
        self._last_command = f"TURN_RIGHT({spd:.2f})"

    def move_pulse(self, left: float, right: float, duration_sec: float) -> None:
        """Execute a short, bounded movement pulse and immediately stop.
        Clamped to a maximum safe pulse duration of 0.50 seconds.
        """
        clamped_duration = max(0.01, min(0.50, float(duration_sec)))
        try:
            self.set_speeds(left, right)
            time.sleep(clamped_duration)
        finally:
            self.stop()

    def cleanup(self) -> None:
        """De-energize motors, stop PWM channels, and release GPIO pins."""
        with self._lock:
            self._apply_hardware_stop()
            self._is_enabled = False

            try:
                if self._pwm_left:
                    self._pwm_left.stop()
                    self._pwm_left = None
                if self._pwm_right:
                    self._pwm_right.stop()
                    self._pwm_right = None

                all_pins = self.config.all_pins
                self.gpio.cleanup(all_pins)
                self._is_initialized = False
                logger.info("L298N motor controller cleaned up. Pins released.")
            except Exception as exc:
                logger.error("Exception during motor cleanup: %s", exc)
