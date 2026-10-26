"""Abstract Hardware Interfaces and Domain Data Types for SS_PROJECT_VCR_FRESH."""
from __future__ import annotations

import time
import math
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Optional, Tuple
import numpy as np


class ObstacleState(Enum):
    """Categorical classification of forward obstacle proximity."""
    CLEAR = "CLEAR"              # distance >= warning_cm
    WARNING = "WARNING"          # blocked_cm <= distance < warning_cm
    BLOCKED = "BLOCKED"          # critical_cm <= distance < blocked_cm
    CRITICAL = "CRITICAL"        # distance < critical_cm (or < min_distance_cm)
    OUT_OF_RANGE = "OUT_OF_RANGE"  # distance > max_distance_cm
    FAULT = "FAULT"              # Timeout, physical disconnection, or unphysical reading


@dataclass(frozen=True)
class UltrasonicReading:
    """Immutable, strongly-typed distance measurement with timestamp and status."""
    distance_cm: Optional[float]
    state: ObstacleState
    timestamp: float
    raw_pulse_sec: Optional[float] = None
    raw_distance_cm: Optional[float] = None

    @property
    def is_valid(self) -> bool:
        """Returns True if measurement represents a physically valid obstacle distance."""
        return (
            self.state in (ObstacleState.CLEAR, ObstacleState.WARNING, ObstacleState.BLOCKED, ObstacleState.CRITICAL)
            and self.distance_cm is not None
            and math.isfinite(float(self.distance_cm))
            and float(self.distance_cm) > 0.0
        )

    @property
    def allows_forward_motion(self) -> bool:
        """Fail-closed gate: Only CLEAR and WARNING permit forward motor pulses."""
        return self.is_valid and self.state in (ObstacleState.CLEAR, ObstacleState.WARNING)

    @classmethod
    def create_fault(cls, timestamp: Optional[float] = None) -> UltrasonicReading:
        """Factory for fail-closed sensor fault readings."""
        return cls(
            distance_cm=None,
            state=ObstacleState.FAULT,
            timestamp=timestamp if timestamp is not None else time.time(),
        )

    @classmethod
    def create_out_of_range(cls, timestamp: Optional[float] = None) -> UltrasonicReading:
        """Factory for echoes beyond maximum measurable sensor range."""
        return cls(
            distance_cm=None,
            state=ObstacleState.OUT_OF_RANGE,
            timestamp=timestamp if timestamp is not None else time.time(),
        )


class MotorControllerInterface(ABC):
    """Authoritative interface for dual-channel differential drive motor control."""

    @abstractmethod
    def set_speeds(self, left: float, right: float) -> None:
        """Set normalized speeds for left and right channels in [-1.0, 1.0].
        Positive is forward, negative is reverse, 0.0 is stopped.
        """

    @abstractmethod
    def stop(self) -> None:
        """Immediately de-energize both channels and pull directional pins LOW."""

    @abstractmethod
    def enable(self, enabled: bool = True) -> None:
        """Enable or disable software output to motors."""

    @property
    @abstractmethod
    def is_enabled(self) -> bool:
        """Return software motor enable state."""

    @property
    @abstractmethod
    def current_speeds(self) -> Tuple[float, float]:
        """Return currently commanded (left, right) speeds."""

    @abstractmethod
    def cleanup(self) -> None:
        """Release any hardware timers, PWM channels, or GPIO configurations."""


class DistanceSensorInterface(ABC):
    """Authoritative interface for forward distance measurement."""

    @abstractmethod
    def get_reading(self) -> UltrasonicReading:
        """Sample or return the latest synchronized distance measurement."""

    @abstractmethod
    def start(self) -> None:
        """Start asynchronous distance sampling background worker if applicable."""

    @abstractmethod
    def stop(self) -> None:
        """Stop sampling and release sensor pins."""


class CameraInterface(ABC):
    """Authoritative interface for optical frame acquisition."""

    @abstractmethod
    def capture_frame(self) -> Optional[np.ndarray]:
        """Acquire latest fresh BGR optical frame as uint8 numpy array [H, W, 3].
        Returns None on frame drop or hardware disconnection.
        """

    @abstractmethod
    def is_opened(self) -> bool:
        """Check if camera device stream is actively connected and opened."""

    @abstractmethod
    def release(self) -> None:
        """Release underlying camera capture device resources."""


class AudioSourceInterface(ABC):
    """Authoritative interface for microphone audio PCM streaming."""

    @abstractmethod
    def read_chunk(self) -> Optional[bytes]:
        """Read a single raw 16-bit 16 kHz mono PCM chunk of configured size."""

    @abstractmethod
    def start_stream(self) -> None:
        """Open and start microphone audio stream."""

    @abstractmethod
    def stop_stream(self) -> None:
        """Stop and pause microphone audio stream."""

    @abstractmethod
    def close(self) -> None:
        """Permanently close and release audio hardware device."""


@dataclass(frozen=True)
class DetectionResult:
    """Bounding box detection result from computer vision model."""
    class_id: int
    class_name: str
    confidence: float
    box_xywh: Tuple[float, float, float, float]  # (center_x, center_y, width, height) in pixels
    box_xyxy: Tuple[float, float, float, float]  # (x1, y1, x2, y2) in pixels

    @property
    def area(self) -> float:
        """Bounding box pixel area."""
        return self.box_xywh[2] * self.box_xywh[3]


class DetectorInterface(ABC):
    """Authoritative interface for real-time object detection inference."""

    @abstractmethod
    def detect(self, frame: np.ndarray) -> list[DetectionResult]:
        """Perform object detection inference on a BGR optical frame [H, W, 3]."""

    @property
    @abstractmethod
    def is_ready(self) -> bool:
        """Check if detector model weights are loaded and ready for inference."""

    @property
    @abstractmethod
    def class_names(self) -> list[str]:
        """Return ordered list of classes supported by the loaded detector model."""


class SpeechRecognizerInterface(ABC):
    """Authoritative interface for offline speech recognition."""

    @abstractmethod
    def accept_waveform(self, pcm_data: bytes) -> bool:
        """Feed a PCM chunk into recognizer. Returns True if a complete utterance boundary is detected."""

    @abstractmethod
    def get_result(self) -> str:
        """Retrieve the final transcription text for the completed utterance."""

    @abstractmethod
    def get_partial_result(self) -> str:
        """Retrieve the latest streaming in-progress partial transcription."""

    @abstractmethod
    def reset(self) -> None:
        """Reset internal recognizer decoding state."""

