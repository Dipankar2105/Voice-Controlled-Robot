"""HC-SR04 Ultrasonic Distance Sensor Driver for SS_PROJECT_VCR_FRESH.

Acoustic & Physical Constraints:
- Transducer Blind Zone: Distances < 2.0 cm suffer acoustic ring-down distortion;
  evaluated as CRITICAL fail-closed.
- Coverage Limitation: Single forward-facing sensor (15°-30° cone) provides NO
  lateral or rear coverage. Never assume clearance while reversing or rotating.
- Fail-Closed Semantics: Sensor timeouts, out-of-range readings, or stale data
  strictly evaluate to allows_forward_motion = False.
- Noise Rejection: Sub-millisecond acoustic / electrical glitch durations (< 115 us)
  cannot represent physical obstacle reflection; rejected as FAULT.
"""
from __future__ import annotations

import collections
import math
import threading
import time
from typing import Deque, List, Optional

from app.config import SafetyConfig, UltrasonicConfig
from app.constants import (
    BLOCKED_OBSTACLE_DISTANCE_CM,
    CRITICAL_OBSTACLE_DISTANCE_CM,
    MAX_ULTRASONIC_DISTANCE_CM,
    MIN_ULTRASONIC_DISTANCE_CM,
    PIN_ULTRASONIC_ECHO,
    PIN_ULTRASONIC_TRIG,
    WARNING_OBSTACLE_DISTANCE_CM,
)
from app.hardware.gpio_adapter import (
    BCM,
    GPIOAdapter,
    HIGH,
    IN,
    LOW,
    OUT,
    PUD_DOWN,
    FakeGPIOAdapter,
)
from app.hardware.interfaces import DistanceSensorInterface, ObstacleState, UltrasonicReading
from app.logger import get_logger

logger = get_logger("ultrasonic")


class HCSR04UltrasonicSensor(DistanceSensorInterface):
    """Authoritative forward ultrasonic distance sensor driver with bounded sampling."""

    def __init__(
        self,
        gpio: Optional[GPIOAdapter] = None,
        config: Optional[UltrasonicConfig] = None,
        safety_config: Optional[SafetyConfig] = None,
        filter_window_size: int = 5,
        stale_threshold_sec: float = 0.50,
        min_valid_samples: int = 2,
    ):
        self.config = config or UltrasonicConfig()
        self.safety_config = safety_config or SafetyConfig()
        self.gpio: GPIOAdapter = gpio or FakeGPIOAdapter()

        self.filter_window_size = max(1, int(filter_window_size))
        self.stale_threshold_sec = float(stale_threshold_sec)
        self.min_valid_samples = max(1, int(min_valid_samples))

        self._is_initialized = False
        self._is_running = False
        self._latest_reading: UltrasonicReading = UltrasonicReading.create_fault()
        self._last_raw_reading: UltrasonicReading = UltrasonicReading.create_fault()
        self._sample_history: Deque[float] = collections.deque(maxlen=self.filter_window_size)
        self._consecutive_faults: int = 0

        self._lock = threading.RLock()
        self._worker_thread: Optional[threading.Thread] = None

    @property
    def latest_raw_reading(self) -> UltrasonicReading:
        """Latest unfiltered single-shot measurement."""
        with self._lock:
            return self._last_raw_reading

    @property
    def valid_sample_count(self) -> int:
        """Number of valid samples in the median filter window."""
        with self._lock:
            return len(self._sample_history)

    def initialize(self) -> None:
        """Setup TRIG as OUT (LOW) and ECHO as IN with pull-down."""
        with self._lock:
            if self._is_initialized:
                return

            self.gpio.setmode(BCM)
            self.gpio.setup(self.config.trig_pin, OUT, initial=LOW)
            self.gpio.setup(self.config.echo_pin, IN, pull_up_down=PUD_DOWN)

            # Settle sensor
            time.sleep(0.02)
            self._is_initialized = True
            logger.info(
                "HC-SR04 initialized (TRIG: %d, ECHO: %d). Pull-down active on ECHO.",
                self.config.trig_pin,
                self.config.echo_pin,
            )

    def sample_distance_cm(self) -> UltrasonicReading:
        """Execute a single acoustic trigger-echo flight time measurement."""
        now = time.time()
        if not self._is_initialized:
            self.initialize()

        # 1. Pre-trigger settle check:
        # If ECHO pin is stuck HIGH from ringing or a previous echo, wait up to 15 ms to settle LOW.
        # If still HIGH, fail closed without sending trigger.
        t_settle_start = time.perf_counter()
        while self.gpio.input(self.config.echo_pin) == HIGH:
            if (time.perf_counter() - t_settle_start) > 0.015:
                logger.debug("Ultrasonic settle failed: ECHO stuck HIGH before trigger.")
                return UltrasonicReading(
                    distance_cm=None,
                    state=ObstacleState.FAULT,
                    timestamp=time.time(),
                    raw_pulse_sec=None,
                    raw_distance_cm=None,
                )

        # 2. Send 10-12 microsecond trigger pulse
        try:
            self.gpio.output(self.config.trig_pin, LOW)
            time.sleep(0.000004)  # 4 us settle
            self.gpio.output(self.config.trig_pin, HIGH)
            time.sleep(0.000012)  # 12 us pulse
            self.gpio.output(self.config.trig_pin, LOW)
        except Exception as exc:
            logger.error("Failed to pulse TRIG pin: %s", exc)
            return UltrasonicReading.create_fault(time.time())

        # 3. Wait for ECHO to rise to HIGH with timeout
        t_wait_rise = time.perf_counter()
        rise_deadline = t_wait_rise + self.config.timeout_sec

        while self.gpio.input(self.config.echo_pin) == LOW:
            if time.perf_counter() > rise_deadline:
                logger.debug("Ultrasonic timeout: ECHO failed to rise HIGH.")
                return UltrasonicReading(
                    distance_cm=None,
                    state=ObstacleState.FAULT,
                    timestamp=time.time(),
                    raw_pulse_sec=None,
                    raw_distance_cm=None,
                )

        pulse_start = time.perf_counter()
        fall_deadline = pulse_start + self.config.timeout_sec

        # 4. Wait for ECHO to fall to LOW with timeout
        while self.gpio.input(self.config.echo_pin) == HIGH:
            if time.perf_counter() > fall_deadline:
                logger.debug("Ultrasonic timeout: ECHO failed to fall LOW.")
                return UltrasonicReading(
                    distance_cm=None,
                    state=ObstacleState.FAULT,
                    timestamp=time.time(),
                    raw_pulse_sec=None,
                    raw_distance_cm=None,
                )

        pulse_end = time.perf_counter()
        pulse_duration = pulse_end - pulse_start

        # 5. Acoustic flight time to distance conversion
        # Speed of sound in air ~ 343 m/s = 34300 cm/s
        raw_distance = (pulse_duration * 34300.0) / 2.0

        # Physical noise rejection:
        # Pulses shorter than 115 us (< 2.0 cm) cannot be real acoustic reflections
        # from an obstacle because the 8-cycle 40 kHz burst takes 200 us.
        # Sub-millisecond spikes are electrical noise or transducer ringing; reject as FAULT.
        if pulse_duration < 0.000115 or raw_distance < self.config.min_distance_cm:
            logger.debug(
                "Acoustic sub-millisecond noise rejected (duration=%.1fus, dist=%.2fcm < %.1fcm).",
                pulse_duration * 1e6,
                raw_distance,
                self.config.min_distance_cm,
            )
            return UltrasonicReading(
                distance_cm=None,
                state=ObstacleState.FAULT,
                timestamp=time.time(),
                raw_pulse_sec=pulse_duration,
                raw_distance_cm=round(raw_distance, 1),
            )

        return self._classify_distance(raw_distance, pulse_duration, time.time())

    def _classify_distance(self, distance_cm: float, pulse_sec: float, timestamp: float) -> UltrasonicReading:
        """Map raw distance reading into validated ObstacleState categories."""
        # NaN/Inf/negative values must never fall through to CLEAR.
        if not isinstance(distance_cm, (int, float)) or not math.isfinite(float(distance_cm)) or distance_cm <= 0:
            return UltrasonicReading(distance_cm=None, state=ObstacleState.FAULT, timestamp=timestamp, raw_pulse_sec=pulse_sec, raw_distance_cm=None)

        # 1. Physical Blind Zone (< 2.0 cm)
        if distance_cm < self.config.min_distance_cm:
            logger.warning(
                "Acoustic blind-zone underflow (%.1f cm < %.1f cm). Fail-closed CRITICAL.",
                distance_cm,
                self.config.min_distance_cm,
            )
            return UltrasonicReading(
                distance_cm=distance_cm,
                state=ObstacleState.CRITICAL,
                timestamp=timestamp,
                raw_pulse_sec=pulse_sec,
                raw_distance_cm=round(distance_cm, 1),
            )

        # 2. Out of Range (> max_distance_cm)
        if distance_cm > self.config.max_distance_cm:
            return UltrasonicReading(
                distance_cm=distance_cm,
                state=ObstacleState.OUT_OF_RANGE,
                timestamp=timestamp,
                raw_pulse_sec=pulse_sec,
                raw_distance_cm=round(distance_cm, 1),
            )

        # 3. Categorical Obstacle Proximity Thresholds
        if distance_cm < self.safety_config.obstacle_critical_cm:
            state = ObstacleState.CRITICAL
        elif distance_cm < self.safety_config.obstacle_blocked_cm:
            state = ObstacleState.BLOCKED
        elif distance_cm < self.safety_config.obstacle_warning_cm:
            state = ObstacleState.WARNING
        else:
            state = ObstacleState.CLEAR

        return UltrasonicReading(
            distance_cm=round(distance_cm, 1),
            state=state,
            timestamp=timestamp,
            raw_pulse_sec=pulse_sec,
            raw_distance_cm=round(distance_cm, 1),
        )

    def get_reading(self) -> UltrasonicReading:
        """Return the latest synchronized distance measurement, verifying bounded freshness."""
        with self._lock:
            # If background worker is not running, take a direct synchronous sample
            if not self._is_running:
                reading = self.sample_distance_cm()
                self._update_reading(reading)

            # Check for stale data (> stale_threshold_sec old)
            reading = self._latest_reading
            now = time.time()
            if (now - reading.timestamp) > self.stale_threshold_sec:
                logger.warning(
                    "Stale ultrasonic data detected (age %.3fs > threshold %.3fs). Rejecting reading.",
                    now - reading.timestamp,
                    self.stale_threshold_sec,
                )
                return UltrasonicReading(
                    distance_cm=None,
                    state=ObstacleState.FAULT,
                    timestamp=reading.timestamp,
                    raw_pulse_sec=reading.raw_pulse_sec,
                    raw_distance_cm=reading.raw_distance_cm,
                )
            return reading

    def get_sensor_state_label(self) -> str:
        """Distinguish CLEAR, OBSTACLE, INVALID, and STALE sensor states."""
        with self._lock:
            reading = self._latest_reading
            now = time.time()
            if reading is None or (now - reading.timestamp) > self.stale_threshold_sec:
                return "STALE"
            if not reading.is_valid or reading.state in (ObstacleState.FAULT, ObstacleState.OUT_OF_RANGE):
                return "INVALID"
            if reading.state in (ObstacleState.BLOCKED, ObstacleState.CRITICAL):
                return "OBSTACLE"
            return "CLEAR"

    def _update_reading(self, reading: UltrasonicReading) -> None:
        """Store reading and apply median filtering across valid historical window."""
        self._last_raw_reading = reading

        if reading.is_valid and reading.distance_cm is not None:
            self._consecutive_faults = 0
            self._sample_history.append(reading.distance_cm)

            if len(self._sample_history) >= self.min_valid_samples:
                # Compute median over history window
                sorted_samples = sorted(self._sample_history)
                median_dist = sorted_samples[len(sorted_samples) // 2]
                filtered_reading = self._classify_distance(
                    median_dist, reading.raw_pulse_sec or 0.0, reading.timestamp
                )
                self._latest_reading = UltrasonicReading(
                    distance_cm=filtered_reading.distance_cm,
                    state=filtered_reading.state,
                    timestamp=filtered_reading.timestamp,
                    raw_pulse_sec=filtered_reading.raw_pulse_sec,
                    raw_distance_cm=reading.distance_cm,
                )
            else:
                # Accumulating initial window of valid samples; fail closed until confirmed
                self._latest_reading = UltrasonicReading(
                    distance_cm=None,
                    state=ObstacleState.FAULT,
                    timestamp=reading.timestamp,
                    raw_pulse_sec=reading.raw_pulse_sec,
                    raw_distance_cm=reading.distance_cm,
                )
        else:
            self._consecutive_faults += 1
            if self._consecutive_faults >= 3:
                self._sample_history.clear()
            self._latest_reading = reading

    def start(self) -> None:
        """Start asynchronous background sampling thread (~14 Hz)."""
        with self._lock:
            if self._is_running:
                return
            if not self._is_initialized:
                self.initialize()
            self._is_running = True
            self._worker_thread = threading.Thread(target=self._sampling_worker, daemon=True, name="UltrasonicWorker")
            self._worker_thread.start()
            logger.info("Ultrasonic background sampling worker started (~14 Hz).")

    def _sampling_worker(self) -> None:
        """Continuous background sampling loop with ~70 ms cycle period (>60ms holdoff)."""
        while self._is_running:
            try:
                reading = self.sample_distance_cm()
                with self._lock:
                    self._update_reading(reading)
            except Exception as exc:
                logger.error("Exception in ultrasonic background worker: %s", exc)
            time.sleep(0.07)  # 70 ms loop period (~14 Hz)

    def stop(self) -> None:
        """Stop background sampling thread and release GPIO pins."""
        with self._lock:
            self._is_running = False

        if self._worker_thread and self._worker_thread.is_alive():
            self._worker_thread.join(timeout=0.5)

        with self._lock:
            try:
                pins = [self.config.trig_pin, self.config.echo_pin]
                self.gpio.cleanup(pins)
                self._is_initialized = False
                logger.info("Ultrasonic sensor stopped and pins released.")
            except Exception as exc:
                logger.error("Exception during ultrasonic cleanup: %s", exc)
