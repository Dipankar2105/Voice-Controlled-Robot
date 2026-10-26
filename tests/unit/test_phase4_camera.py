"""Phase 4 unit tests: CameraOwner, frame freshness, dimension validation, and single owner pattern."""
from __future__ import annotations

import time
from typing import Optional, Tuple
import numpy as np
import pytest

from app.config import CameraConfig
from app.vision.camera import CameraOwner


class MockCameraBackend:
    """Mock backend that delivers programmed frames or simulated drops."""

    def __init__(self, width: int = 640, height: int = 480):
        self.width = width
        self.height = height
        self._is_opened = False
        self._next_frame: Optional[np.ndarray] = None
        self.open_call_count = 0
        self.release_call_count = 0
        self.read_call_count = 0

    def open(self) -> bool:
        self.open_call_count += 1
        self._is_opened = True
        return True

    def is_opened(self) -> bool:
        return self._is_opened

    def set_next_frame(self, frame: Optional[np.ndarray]) -> None:
        self._next_frame = frame

    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        self.read_call_count += 1
        if not self._is_opened:
            return False, None
        if self._next_frame is not None:
            return True, self._next_frame.copy()
        # Default frame (480, 640, 3)
        return True, np.zeros((self.height, self.width, 3), dtype=np.uint8)

    def release(self) -> None:
        self.release_call_count += 1
        self._is_opened = False


def test_camera_owner_single_instance_and_freshness():
    """CameraOwner provides shared, fresh frames without stale buffer lag."""
    mock_backend = MockCameraBackend(width=640, height=480)
    owner = CameraOwner(backend=mock_backend, max_staleness_sec=0.50)

    # Inactive before start()
    assert not owner.is_operational
    assert owner.capture_frame() is None

    # Start background capture
    started = owner.start()
    assert started is True
    assert mock_backend.open_call_count == 1

    # Operational status is true only after the first valid frame arrives.
    deadline = time.monotonic() + 0.5
    while not owner.is_operational and time.monotonic() < deadline:
        time.sleep(0.01)
    assert owner.is_operational

    # Settle for 1 frame
    time.sleep(0.08)
    frame, ts = owner.get_latest_frame_with_timestamp()
    assert frame is not None
    assert frame.shape == (480, 640, 3)
    assert ts > 0.0
    assert (time.time() - ts) < 0.50  # Fresh

    owner.release()
    assert mock_backend.release_call_count == 1
    assert not owner.is_operational
    assert owner.capture_frame() is None


def test_camera_owner_frame_dimension_validation():
    """Frames that do not match configured dimensions (640x480) must be rejected."""
    # Test startup with wrong dimension stream (e.g. 320x240)
    mock_backend_wrong = MockCameraBackend(width=320, height=240)
    owner = CameraOwner(backend=mock_backend_wrong, max_consecutive_drops=2)
    owner.start()

    time.sleep(0.10)
    # Never accepted, capture_frame() returns None
    assert owner.capture_frame() is None
    owner.release()

    # Test dynamic injection of bad dimensions causing degradation
    mock_backend_ok = MockCameraBackend(width=640, height=480)
    owner2 = CameraOwner(backend=mock_backend_ok, max_consecutive_drops=2)
    owner2.start()
    time.sleep(0.08)
    assert owner2.capture_frame() is not None

    # Inject mismatched dimensions
    wrong_frame = np.zeros((240, 320, 3), dtype=np.uint8)
    mock_backend_ok.set_next_frame(wrong_frame)
    time.sleep(0.25)  # Wait for 2+ consecutive drops at 15 FPS

    # Frame should be rejected and camera flagged degraded
    assert not owner2.is_operational
    assert owner2.capture_frame() is None
    owner2.release()


def test_camera_owner_drop_detection_and_degradation():
    """Consecutive capture failures must flag camera as not operational."""
    mock_backend = MockCameraBackend(width=640, height=480)
    owner = CameraOwner(backend=mock_backend, max_consecutive_drops=3)
    owner.start()
    deadline = time.monotonic() + 0.5
    while not owner.is_operational and time.monotonic() < deadline:
        time.sleep(0.01)
    assert owner.is_operational

    # Simulate hardware disconnection or read failures
    class BrokenBackend(MockCameraBackend):
        def read(self):
            return False, None

    owner.backend = BrokenBackend(width=640, height=480)
    owner.backend._is_opened = True

    # Allow worker to encounter consecutive drops
    time.sleep(0.25)
    assert not owner.is_operational
    assert owner.capture_frame() is None

    owner.release()
