"""Hardware-free mock implementations for unit testing and CI simulation."""
from __future__ import annotations

import time
from typing import Callable, List, Optional, Tuple, Union
import numpy as np

from app.hardware.interfaces import (
    AudioSourceInterface,
    CameraInterface,
    DetectionResult,
    DetectorInterface,
    DistanceSensorInterface,
    MotorControllerInterface,
    ObstacleState,
    SpeechRecognizerInterface,
    UltrasonicReading,
)


class MockMotorController(MotorControllerInterface):
    """Deterministic, hardware-free motor controller that records all actuations."""

    def __init__(self, initially_enabled: bool = False):
        self._is_enabled = initially_enabled
        self._left_speed: float = 0.0
        self._right_speed: float = 0.0
        self.history: List[Tuple[float, float, float]] = []  # (timestamp, left, right)
        self.stop_count: int = 0
        self.is_cleaned_up: bool = False

    def enable(self, enabled: bool = True) -> None:
        self._is_enabled = enabled
        if not enabled:
            self.stop()

    @property
    def is_enabled(self) -> bool:
        return self._is_enabled

    @property
    def current_speeds(self) -> Tuple[float, float]:
        return (self._left_speed, self._right_speed)

    def set_speeds(self, left: float, right: float) -> None:
        if not self._is_enabled:
            self._left_speed = 0.0
            self._right_speed = 0.0
            return

        # Clamp between -1.0 and 1.0
        clamped_left = max(-1.0, min(1.0, float(left)))
        clamped_right = max(-1.0, min(1.0, float(right)))

        self._left_speed = clamped_left
        self._right_speed = clamped_right
        self.history.append((time.time(), clamped_left, clamped_right))

    def stop(self) -> None:
        self._left_speed = 0.0
        self._right_speed = 0.0
        self.stop_count += 1
        self.history.append((time.time(), 0.0, 0.0))

    def cleanup(self) -> None:
        self.stop()
        self._is_enabled = False
        self.is_cleaned_up = True


class MockDistanceSensor(DistanceSensorInterface):
    """Programmable distance sensor for safety testing and obstacle scenarios."""

    def __init__(
        self,
        initial_distance_cm: Optional[float] = 100.0,
        warning_cm: float = 50.0,
        blocked_cm: float = 25.0,
        critical_cm: float = 15.0,
        min_cm: float = 2.0,
        max_cm: float = 400.0,
    ):
        self.warning_cm = warning_cm
        self.blocked_cm = blocked_cm
        self.critical_cm = critical_cm
        self.min_cm = min_cm
        self.max_cm = max_cm

        self._current_reading: UltrasonicReading = self._classify(initial_distance_cm)
        self._reading_sequence: List[Optional[float]] = []
        self._custom_sampler: Optional[Callable[[], UltrasonicReading]] = None
        self._is_running: bool = False

    def _classify(self, dist: Optional[float]) -> UltrasonicReading:
        now = time.time()
        if dist is None or dist < 0:
            return UltrasonicReading(distance_cm=None, state=ObstacleState.FAULT, timestamp=now)
        if dist > self.max_cm:
            return UltrasonicReading(distance_cm=dist, state=ObstacleState.OUT_OF_RANGE, timestamp=now)
        if dist < self.critical_cm or dist < self.min_cm:
            return UltrasonicReading(distance_cm=dist, state=ObstacleState.CRITICAL, timestamp=now)
        if dist < self.blocked_cm:
            return UltrasonicReading(distance_cm=dist, state=ObstacleState.BLOCKED, timestamp=now)
        if dist < self.warning_cm:
            return UltrasonicReading(distance_cm=dist, state=ObstacleState.WARNING, timestamp=now)
        return UltrasonicReading(distance_cm=dist, state=ObstacleState.CLEAR, timestamp=now)

    def set_distance(self, distance_cm: Optional[float]) -> None:
        """Program a fixed distance measurement."""
        self._current_reading = self._classify(distance_cm)

    def set_sequence(self, sequence: List[Optional[float]]) -> None:
        """Program a queue of distances returned in sequential calls."""
        self._reading_sequence = list(sequence)

    def set_custom_sampler(self, sampler: Callable[[], UltrasonicReading]) -> None:
        """Inject a dynamic distance calculation callback."""
        self._custom_sampler = sampler

    def get_reading(self) -> UltrasonicReading:
        if self._custom_sampler is not None:
            return self._custom_sampler()
        if self._reading_sequence:
            next_dist = self._reading_sequence.pop(0)
            self._current_reading = self._classify(next_dist)
        else:
            # A fixed mock distance represents a new simulated sample on every
            # poll, so its timestamp remains fresh like the physical worker.
            self._current_reading = self._classify(self._current_reading.distance_cm)
        return self._current_reading

    def get_sensor_state_label(self) -> str:
        reading = self._current_reading
        now = time.time()
        if reading is None or (now - reading.timestamp) > 0.50:
            return "STALE"
        if not reading.is_valid or reading.state in (ObstacleState.FAULT, ObstacleState.OUT_OF_RANGE):
            return "INVALID"
        if reading.state in (ObstacleState.BLOCKED, ObstacleState.CRITICAL):
            return "OBSTACLE"
        return "CLEAR"

    def start(self) -> None:
        self._is_running = True

    def stop(self) -> None:
        self._is_running = False


_DEFAULT_FRAME = object()


class MockCamera(CameraInterface):
    """Generates synthetic optical frames or serves injected frames."""

    def __init__(self, width: int = 640, height: int = 480):
        self.width = width
        self.height = height
        self._is_opened = True
        self._injected_frame: Union[np.ndarray, None, object] = _DEFAULT_FRAME
        self.frame_count: int = 0

    def inject_frame(self, frame: Optional[np.ndarray]) -> None:
        """Provide a specific frame or None to simulate dropped frames."""
        self._injected_frame = frame

    def clear_injected_frame(self) -> None:
        """Clear custom injection and revert to default synthetic frames."""
        self._injected_frame = _DEFAULT_FRAME

    def capture_frame(self) -> Optional[np.ndarray]:
        if not self._is_opened:
            return None
        self.frame_count += 1
        if self._injected_frame is not _DEFAULT_FRAME:
            return self._injected_frame.copy() if self._injected_frame is not None else None
        # Default blank dark gray frame [H, W, 3] uint8
        return np.full((self.height, self.width, 3), 40, dtype=np.uint8)

    def is_opened(self) -> bool:
        return self._is_opened

    @property
    def is_operational(self) -> bool:
        return self._is_opened

    def get_latest_frame_with_timestamp(self) -> Tuple[Optional[np.ndarray], float]:
        frame = self.capture_frame()
        return frame, time.time() if frame is not None else 0.0

    def start(self) -> bool:
        self._is_opened = True
        return True

    def stop(self) -> None:
        pass

    def release(self) -> None:
        self._is_opened = False


class MockAudioSource(AudioSourceInterface):
    """Provides synthetic 16 kHz 16-bit mono PCM audio chunks."""

    def __init__(self, chunk_size: int = 1024, sample_rate: int = 16000):
        self.chunk_size = chunk_size
        self.sample_rate = sample_rate
        self._is_streaming = False
        self._is_closed = False
        self._audio_queue: List[bytes] = []

    def queue_pcm_data(self, data: bytes) -> None:
        """Enqueue PCM bytes to be read chunk by chunk."""
        for i in range(0, len(data), self.chunk_size * 2):  # 2 bytes per sample
            chunk = data[i : i + self.chunk_size * 2]
            if len(chunk) < self.chunk_size * 2:
                chunk = chunk + b"\x00" * (self.chunk_size * 2 - len(chunk))
            self._audio_queue.append(chunk)

    def read_chunk(self) -> Optional[bytes]:
        if not self._is_streaming or self._is_closed:
            return None
        if self._audio_queue:
            return self._audio_queue.pop(0)
        # Default silence chunk (all zeros)
        return b"\x00" * (self.chunk_size * 2)

    def start_stream(self) -> None:
        self._is_streaming = True

    def stop_stream(self) -> None:
        self._is_streaming = False

    def close(self) -> None:
        self._is_streaming = False
        self._is_closed = True


class MockSpeechRecognizer(SpeechRecognizerInterface):
    """Mock speech recognizer that returns pre-queued text transcripts."""

    def __init__(self, initial_result: str = ""):
        self._queued_results: List[str] = [initial_result] if initial_result else []
        self._partial: str = ""
        self.accept_call_count: int = 0

    def queue_utterance(self, text: str) -> None:
        """Enqueue an utterance string to be returned by get_result()."""
        self._queued_results.append(text)

    def set_partial(self, partial: str) -> None:
        """Set in-progress partial transcription string."""
        self._partial = partial

    def accept_waveform(self, pcm_data: bytes) -> bool:
        self.accept_call_count += 1
        return len(self._queued_results) > 0

    def get_result(self) -> str:
        if self._queued_results:
            return self._queued_results.pop(0)
        return ""

    def get_partial_result(self) -> str:
        return self._partial

    def reset(self) -> None:
        self._partial = ""
        self._queued_results.clear()


class MockDetector(DetectorInterface):
    """Mock detector returning pre-programmed object detections."""

    def __init__(self, class_names: Optional[List[str]] = None):
        self._class_names = class_names or [
            "person", "chair", "table", "door", "RED_TARGET", "BLUE_TARGET",
            "laptop", "monitor", "keyboard", "mouse", "backpack", "book",
            "bottle", "trash_can", "whiteboard", "projector", "clock", "fan",
            "fire_extinguisher", "bookshelf", "cup", "cell_phone", "remote_control",
            "potted_plant", "window", "scissors"
        ]
        self._injected_detections: List[DetectionResult] = []
        self.detect_call_count: int = 0
        self._is_ready: bool = True

    def inject_detections(self, detections: List[DetectionResult]) -> None:
        """Inject specific detections to return on next detect() call."""
        self._injected_detections = list(detections)

    def detect(self, frame: np.ndarray) -> List[DetectionResult]:
        self.detect_call_count += 1
        return list(self._injected_detections)

    @property
    def is_ready(self) -> bool:
        return self._is_ready

    @property
    def class_names(self) -> List[str]:
        return list(self._class_names)

