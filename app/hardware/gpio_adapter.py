"""GPIO Hardware Abstraction Adapter for SS_PROJECT_VCR_FRESH.

Isolates Raspberry Pi physical GPIO calls from application and business logic.
Provides:
- GPIOAdapter: Formal interface for digital I/O and hardware/software PWM.
- FakeGPIOAdapter: Complete in-memory simulation tracking pin states, PWM duty cycles,
  and simulating echo pulses for automated tests on any host OS without hardware side effects.
- RPiGPIOAdapter: Concrete wrapper around RPi.GPIO, loaded strictly when running on physical Linux ARM.
"""
from __future__ import annotations

import sys
import time
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

from app.constants import ExecutionMode
from app.logger import get_logger

logger = get_logger("gpio")


# BCM Pin Logic Constants
BCM = 11
OUT = 0
IN = 1
HIGH = 1
LOW = 0
PUD_OFF = 20
PUD_DOWN = 21
PUD_UP = 22


class PWMChannel(ABC):
    """Abstract interface for a PWM channel."""

    @abstractmethod
    def start(self, duty_cycle: float) -> None:
        """Start PWM generation with initial duty cycle (0.0 to 100.0)."""

    @abstractmethod
    def ChangeDutyCycle(self, duty_cycle: float) -> None:
        """Change duty cycle (0.0 to 100.0)."""

    @abstractmethod
    def ChangeFrequency(self, frequency_hz: float) -> None:
        """Change PWM frequency in Hz."""

    @abstractmethod
    def stop(self) -> None:
        """Stop PWM generation."""


class FakePWMChannel(PWMChannel):
    """In-memory simulated PWM channel tracking duty cycles."""

    def __init__(self, pin: int, frequency_hz: float):
        self.pin = pin
        self.frequency_hz = max(1.0, float(frequency_hz))
        self.duty_cycle: float = 0.0
        self.is_running: bool = False

    def start(self, duty_cycle: float) -> None:
        self.duty_cycle = max(0.0, min(100.0, float(duty_cycle)))
        self.is_running = True

    def ChangeDutyCycle(self, duty_cycle: float) -> None:
        self.duty_cycle = max(0.0, min(100.0, float(duty_cycle)))

    def ChangeFrequency(self, frequency_hz: float) -> None:
        self.frequency_hz = max(1.0, float(frequency_hz))

    def stop(self) -> None:
        self.duty_cycle = 0.0
        self.is_running = False


class GPIOAdapter(ABC):
    """Abstract interface for Raspberry Pi GPIO operations."""

    @abstractmethod
    def setmode(self, mode: int) -> None:
        """Set pin numbering scheme (BCM)."""

    @abstractmethod
    def setup(self, pin: int, mode: int, pull_up_down: int = PUD_OFF, initial: int = LOW) -> None:
        """Configure pin mode and initial state."""

    @abstractmethod
    def output(self, pin: int, value: int) -> None:
        """Set digital output state (HIGH or LOW)."""

    @abstractmethod
    def input(self, pin: int) -> int:
        """Read digital input state (HIGH or LOW)."""

    @abstractmethod
    def PWM(self, pin: int, frequency_hz: float) -> PWMChannel:
        """Create a PWM instance on the given pin."""

    @abstractmethod
    def cleanup(self, pins: Optional[Union[Sequence[int], int]] = None) -> None:
        """Reset pin modes and release GPIO resources."""


class FakeGPIOAdapter(GPIOAdapter):
    """In-memory simulation of Raspberry Pi GPIO for automated testing."""

    def __init__(self):
        self.mode: Optional[int] = None
        self.pin_modes: Dict[int, int] = {}
        self.pin_values: Dict[int, int] = {}
        self.pwm_channels: Dict[int, FakePWMChannel] = {}
        self.history: List[Tuple[float, str, int, int]] = []  # (timestamp, action, pin, value)
        self.echo_pulse_duration_sec: Optional[float] = None
        self.echo_timeout: bool = False
        self._trigger_start_time: Optional[float] = None

    def setmode(self, mode: int) -> None:
        self.mode = mode

    def setup(self, pin: int, mode: int, pull_up_down: int = PUD_OFF, initial: int = LOW) -> None:
        self.pin_modes[pin] = mode
        self.pin_values[pin] = initial
        self.history.append((time.time(), "SETUP", pin, mode))

    def output(self, pin: int, value: int) -> None:
        val = HIGH if value else LOW
        self.pin_values[pin] = val
        self.history.append((time.time(), "OUTPUT", pin, val))

        # Check if TRIG pin was set HIGH to start echo pulse simulation
        if val == HIGH and self.echo_pulse_duration_sec is not None:
            self._trigger_start_time = time.time()

    def input(self, pin: int) -> int:
        # If simulating echo pulse for HC-SR04:
        if self.echo_timeout:
            return LOW
        if self.echo_pulse_duration_sec is not None and self._trigger_start_time is not None:
            elapsed = time.time() - self._trigger_start_time
            if elapsed < self.echo_pulse_duration_sec:
                return HIGH
            return LOW
        return self.pin_values.get(pin, LOW)

    def PWM(self, pin: int, frequency_hz: float) -> PWMChannel:
        pwm = FakePWMChannel(pin, frequency_hz)
        self.pwm_channels[pin] = pwm
        return pwm

    def cleanup(self, pins: Optional[Union[Sequence[int], int]] = None) -> None:
        if pins is not None:
            pin_list = [pins] if isinstance(pins, int) else list(pins)
            for p in pin_list:
                self.pin_values[p] = LOW
                self.pin_modes.pop(p, None)
                if p in self.pwm_channels:
                    self.pwm_channels[p].stop()
                    self.pwm_channels.pop(p, None)
        else:
            for p in list(self.pwm_channels.values()):
                p.stop()
            self.pwm_channels.clear()
            self.pin_values.clear()
            self.pin_modes.clear()
            self.mode = None


class RPiGPIOAdapter(GPIOAdapter):
    """Production wrapper for physical RPi.GPIO on Raspberry Pi Linux."""

    def __init__(self):
        try:
            import RPi.GPIO as real_gpio
            self._gpio = real_gpio
            self._gpio.setwarnings(False)
            self._gpio.setmode(self._gpio.BCM)
            logger.info("RPiGPIOAdapter initialized on physical BCM GPIO.")
        except ImportError as exc:
            raise RuntimeError(
                f"RPi.GPIO is not available on this platform ({sys.platform}). "
                "Ensure you are running on Raspberry Pi OS Linux or use FakeGPIOAdapter."
            ) from exc

    def setmode(self, mode: int) -> None:
        if mode != BCM and mode != self._gpio.BCM:
            logger.warning("RPiGPIOAdapter requires BCM mode; requested %s, enforcing BCM.", mode)
        self._gpio.setmode(self._gpio.BCM)

    def setup(self, pin: int, mode: int, pull_up_down: int = PUD_OFF, initial: int = LOW) -> None:
        rpi_mode = self._gpio.OUT if mode == OUT else self._gpio.IN
        rpi_pud = self._gpio.PUD_OFF
        if pull_up_down == PUD_DOWN:
            rpi_pud = self._gpio.PUD_DOWN
        elif pull_up_down == PUD_UP:
            rpi_pud = self._gpio.PUD_UP

        if mode == OUT:
            self._gpio.setup(pin, rpi_mode, initial=initial)
        else:
            self._gpio.setup(pin, rpi_mode, pull_up_down=rpi_pud)

    def output(self, pin: int, value: int) -> None:
        self._gpio.output(pin, self._gpio.HIGH if value else self._gpio.LOW)

    def input(self, pin: int) -> int:
        return HIGH if self._gpio.input(pin) else LOW

    def PWM(self, pin: int, frequency_hz: float) -> PWMChannel:
        class RPiPWMChannel(PWMChannel):
            def __init__(self, raw_pwm):
                self._raw = raw_pwm
                self._is_stopped = False

            def start(self, duty_cycle: float) -> None:
                clamped = max(0.0, min(100.0, float(duty_cycle)))
                try:
                    self._raw.start(clamped)
                    self._is_stopped = False
                except Exception as exc:
                    logger.error("Error starting PWM: %s", exc)
                    raise

            def ChangeDutyCycle(self, duty_cycle: float) -> None:
                clamped = max(0.0, min(100.0, float(duty_cycle)))
                try:
                    self._raw.ChangeDutyCycle(clamped)
                except Exception as exc:
                    logger.error("Error setting PWM duty cycle (%.1f%%): %s", clamped, exc)
                    raise

            def ChangeFrequency(self, frequency_hz: float) -> None:
                freq = max(1.0, float(frequency_hz))
                try:
                    self._raw.ChangeFrequency(freq)
                except Exception as exc:
                    logger.error("Error changing PWM frequency (%.1f Hz): %s", freq, exc)
                    raise

            def stop(self) -> None:
                if self._is_stopped:
                    return
                try:
                    self._raw.stop()
                except Exception as exc:
                    logger.debug("Non-fatal exception stopping PWM channel: %s", exc)
                finally:
                    self._is_stopped = True

        raw = self._gpio.PWM(pin, max(1.0, float(frequency_hz)))
        return RPiPWMChannel(raw)

    def cleanup(self, pins: Optional[Union[Sequence[int], int]] = None) -> None:
        try:
            if pins is not None:
                pin_list = [pins] if isinstance(pins, int) else list(pins)
                self._gpio.cleanup(pin_list)
            else:
                self._gpio.cleanup()
        except Exception as exc:
            logger.warning("Error during RPi.GPIO cleanup: %s", exc)


def get_gpio_adapter(mode: ExecutionMode = ExecutionMode.SIMULATION) -> GPIOAdapter:
    """Factory creating the appropriate GPIO adapter without side effects in simulation."""
    if mode == ExecutionMode.REAL_HARDWARE:
        return RPiGPIOAdapter()
    return FakeGPIOAdapter()
