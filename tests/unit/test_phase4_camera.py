"""Phase 4 unit tests: CameraOwner, frame freshness, dimension validation, and single owner pattern."""
from __future__ import annotations

import time
from typing import Optional, Tuple
from unittest.mock import MagicMock, patch
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


def test_camera_backend_initial_open_success_and_failure():
    """Verify backend open handles failure gracefully and releases partial handles."""
    class FlakyBackend(MockCameraBackend):
        def __init__(self, should_fail=False):
            super().__init__()
            self.should_fail = should_fail

        def open(self) -> bool:
            self.open_call_count += 1
            if self.should_fail:
                self._is_opened = False
                return False
            self._is_opened = True
            return True

    # 1. Open failure does not launch worker or mark operational
    failing_backend = FlakyBackend(should_fail=True)
    owner = CameraOwner(backend=failing_backend)
    assert owner.start() is False
    assert owner.is_operational is False
    assert owner.capture_frame() is None

    # 2. Open success works
    ok_backend = FlakyBackend(should_fail=False)
    owner_ok = CameraOwner(backend=ok_backend)
    assert owner_ok.start() is True
    owner_ok.release()


def test_camera_owner_recovery_exhausts_retry_budget_and_sets_terminal_fault():
    """Verify that repeated failures trigger recovery attempts and enter terminal fault when exhausted."""
    class FailingOnReadBackend(MockCameraBackend):
        def open(self) -> bool:
            self.open_call_count += 1
            self._is_opened = True
            return True

        def read(self) -> Tuple[bool, Optional[np.ndarray]]:
            self.read_call_count += 1
            return False, None

    backend = FailingOnReadBackend()
    owner = CameraOwner(
        backend=backend,
        max_consecutive_drops=2,
        max_recovery_attempts=2,
        recovery_backoff_sec=0.01,
    )
    owner.start()

    # Wait for drops and recovery attempts to exhaust budget
    deadline = time.monotonic() + 1.0
    while not owner.has_terminal_fault and time.monotonic() < deadline:
        time.sleep(0.02)

    assert owner.is_operational is False
    assert owner.has_terminal_fault is True
    assert owner.capture_frame() is None
    owner.release()


def test_camera_owner_repeated_start_stop_idempotency():
    """Verify start(), stop(), and release() are safe to call repeatedly."""
    backend = MockCameraBackend()
    owner = CameraOwner(backend=backend)

    # Calling stop() on unstarted owner
    owner.stop()
    assert owner.is_operational is False

    # Start and stop repeatedly
    for _ in range(3):
        assert owner.start() is True
        owner.stop()
        assert owner.is_operational is False

    owner.release()
    owner.release()


def test_camera_owner_concurrent_read_requests():
    """Verify concurrent calls to capture_frame() return safely and independently."""
    import threading

    backend = MockCameraBackend()
    owner = CameraOwner(backend=backend)
    owner.start()

    deadline = time.monotonic() + 0.5
    while not owner.is_operational and time.monotonic() < deadline:
        time.sleep(0.01)

    captured_frames = []

    def reader_worker():
        for _ in range(20):
            f = owner.capture_frame()
            if f is not None:
                captured_frames.append(f.shape)
            time.sleep(0.005)

    threads = [threading.Thread(target=reader_worker) for _ in range(3)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=1.0)

    assert len(captured_frames) > 0
    assert all(shape == (480, 640, 3) for shape in captured_frames)
    owner.release()


def test_opencv_backend_fallback_open_failure_cleanup():
    """Verify that when primary and fallback opens both fail, all handles are released."""
    import cv2
    from unittest.mock import MagicMock, patch
    from app.vision.camera import OpenCVCameraBackend

    backend = OpenCVCameraBackend(device_index=99)
    mock_instance = MagicMock()
    mock_instance.isOpened.return_value = False

    with patch.object(cv2, "VideoCapture", return_value=mock_instance):
        success = backend.open()
        assert success is False
        assert backend.is_opened() is False
        assert mock_instance.release.call_count >= 1


def test_camera_shutdown_during_active_capture():
    """Verify clean shutdown when capture worker is actively processing frames."""
    backend = MockCameraBackend()
    owner = CameraOwner(backend=backend)
    assert owner.start() is True

    deadline = time.monotonic() + 0.5
    while not owner.is_operational and time.monotonic() < deadline:
        time.sleep(0.01)

    # Stop while active
    assert owner.is_operational is True
    owner.stop()

    assert owner.is_operational is False
    assert owner.capture_frame() is None
    owner.release()


def test_camera_stale_frame_rejection():
    """Verify that frames exceeding max_staleness_sec are rejected as None."""
    backend = MockCameraBackend()
    owner = CameraOwner(backend=backend, max_staleness_sec=0.10)
    assert owner.start() is True

    deadline = time.monotonic() + 0.5
    while not owner.is_operational and time.monotonic() < deadline:
        time.sleep(0.01)

    # Force timestamp to be stale (> 0.10s ago)
    with owner._lock:
        owner._latest_timestamp = time.time() - 0.50

    # Should reject stale frame
    assert owner.capture_frame() is None
    owner.release()


def test_camera_to_detector_integration_and_stale_detection_prevention():
    """Integration test: valid camera frames yield detections; degraded camera yields no detections."""
    from app.hardware.mock_hardware import MockDetector
    from app.hardware.interfaces import DetectionResult

    backend = MockCameraBackend()
    owner = CameraOwner(backend=backend)
    detector = MockDetector()
    detector.inject_detections([
        DetectionResult(
            class_id=0,
            class_name="person",
            confidence=0.88,
            box_xywh=(320.0, 240.0, 100.0, 100.0),
            box_xyxy=(270.0, 190.0, 370.0, 290.0),
        )
    ])

    # 1. Unstarted camera -> no frame -> no detection passed
    frame = owner.capture_frame()
    assert frame is None

    # 2. Started camera -> valid frame -> detector called
    owner.start()
    deadline = time.monotonic() + 0.5
    while not owner.is_operational and time.monotonic() < deadline:
        time.sleep(0.01)

    valid_frame = owner.capture_frame()
    assert valid_frame is not None
    detections = detector.detect(valid_frame)
    assert len(detections) == 1
    assert detections[0].class_name == "person"

    # 3. Degraded / stopped camera -> frame returns None
    owner.release()
    stopped_frame = owner.capture_frame()
    assert stopped_frame is None


def test_camera_owner_single_owner_enforcement():
    """Verify that starting a new CameraOwner stops and releases any prior active instance."""
    backend1 = MockCameraBackend()
    owner1 = CameraOwner(backend=backend1)
    assert owner1.start() is True
    assert owner1.is_running is True
    assert backend1.open_call_count == 1

    # Start a second owner instance
    backend2 = MockCameraBackend()
    owner2 = CameraOwner(backend=backend2)
    assert owner2.start() is True
    assert owner2.is_running is True

    # First owner must have been cleanly stopped and backend released
    assert owner1.is_running is False
    assert backend1.release_call_count >= 1

    owner2.release()
    assert owner2.is_running is False


def test_camera_owner_worker_thread_terminates_on_terminal_fault():
    """Verify capture worker thread terminates and does not spin as a zombie on terminal fault."""
    class FailingBackend(MockCameraBackend):
        def read(self) -> Tuple[bool, Optional[np.ndarray]]:
            return False, None

    backend = FailingBackend()
    owner = CameraOwner(
        backend=backend,
        max_consecutive_drops=2,
        max_recovery_attempts=2,
        recovery_backoff_sec=0.01,
    )
    assert owner.start() is True
    worker_thread = owner._thread
    assert worker_thread is not None
    assert worker_thread.is_alive()

    # Wait for terminal fault
    deadline = time.monotonic() + 1.5
    while not owner.has_terminal_fault and time.monotonic() < deadline:
        time.sleep(0.02)

    assert owner.has_terminal_fault is True
    # The worker thread must exit cleanly within a brief window
    worker_thread.join(timeout=0.5)
    assert not worker_thread.is_alive()
    assert owner.is_running is False
    owner.release()


def test_camera_owner_stop_releases_backend_handle():
    """Verify stop() releases the backend device handle, freeing device node for subsequent use."""
    backend = MockCameraBackend()
    owner = CameraOwner(backend=backend)
    assert owner.start() is True
    assert backend.is_opened() is True

    owner.stop()
    assert owner.is_running is False
    assert backend.is_opened() is False
    assert backend.release_call_count == 1
    owner.release()
