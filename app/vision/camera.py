"""Camera Subsystem for SS_PROJECT_VCR_FRESH.

Implements the Single Camera Owner pattern for USB webcam (Logitech C270).
Features:
- Drains V4L2 kernel frame buffer via a background capture thread to eliminate latency.
- Guaranteed release of failed handles before any fallback open attempt (prevents FD leaks).
- Bounded recovery policy on repeated frame drops with backoff and terminal fault state.
- Verified frame dimensions and staleness validation.
- Deterministic shutdown verifying worker thread termination.
- Hardware-free simulation mode support via injectable backends.
"""
from __future__ import annotations

import glob
import os
import sys
import threading
import time
from typing import Any, Dict, List, Optional, Tuple
import numpy as np

from app.config import CameraConfig
from app.constants import CAMERA_DEVICE_INDEX, CAMERA_FPS, CAMERA_HEIGHT, CAMERA_WIDTH
from app.hardware.interfaces import CameraInterface
from app.logger import get_logger

logger = get_logger("camera")


class OpenCVCameraBackend:
    """Real OpenCV VideoCapture device backend with guaranteed handle cleanup."""

    def __init__(
        self,
        device_index: int = CAMERA_DEVICE_INDEX,
        width: int = CAMERA_WIDTH,
        height: int = CAMERA_HEIGHT,
        fps: int = CAMERA_FPS,
    ):
        self.device_index = int(device_index)
        self.width = int(width)
        self.height = int(height)
        self.fps = int(fps)
        self._cap: Any = None
        self._lock = threading.RLock()

    def _find_candidate_indices(self) -> List[int]:
        """Find candidate capture indices on Linux (excluding platform codec/ISP nodes 10-31)."""
        candidates: List[int] = [self.device_index]
        if sys.platform.startswith("linux"):
            for by_id in sorted(glob.glob("/dev/v4l/by-id/*index0")):
                try:
                    real = os.path.realpath(by_id)
                    if real.startswith("/dev/video"):
                        idx = int(real.replace("/dev/video", ""))
                        if idx not in candidates:
                            candidates.append(idx)
                except (ValueError, OSError):
                    pass
            for idx in range(5):
                if idx not in candidates and os.path.exists(f"/dev/video{idx}"):
                    candidates.append(idx)
        return candidates

    def open(self) -> bool:
        """Open physical camera device, cleanly releasing any existing handle first."""
        with self._lock:
            self.release()

            import cv2

            candidate_indices = self._find_candidate_indices()
            cap = None

            for dev_idx in candidate_indices:
                logger.info(
                    "Opening camera device index %d (%dx%d @ %d FPS)...",
                    dev_idx,
                    self.width,
                    self.height,
                    self.fps,
                )

                # Try primary backend (V4L2 on Linux)
                if sys.platform.startswith("linux"):
                    try:
                        cap = cv2.VideoCapture(dev_idx, cv2.CAP_V4L2)
                    except Exception as exc:
                        logger.debug("cv2.CAP_V4L2 open raised on index %d: %s", dev_idx, exc)
                        cap = None

                # Fallback if V4L2 failed or unsupported
                if cap is None or not cap.isOpened():
                    if cap is not None:
                        try:
                            cap.release()
                        except Exception:
                            pass
                    try:
                        cap = cv2.VideoCapture(dev_idx)
                    except Exception as exc:
                        logger.error("Default VideoCapture open failed on index %d: %s", dev_idx, exc)
                        cap = None

                if cap is not None and cap.isOpened():
                    self.device_index = dev_idx
                    break

                if cap is not None:
                    try:
                        cap.release()
                    except Exception:
                        pass
                cap = None

            if cap is None or not cap.isOpened():
                if cap is not None:
                    try:
                        cap.release()
                    except Exception:
                        pass
                logger.error("Failed to open camera device index %d.", self.device_index)
                self._cap = None
                return False

            # Configure properties
            try:
                cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
                cap.set(cv2.CAP_PROP_FPS, self.fps)
            except Exception as exc:
                logger.warning("Error configuring camera properties: %s", exc)

            self._cap = cap
            return True

    def is_opened(self) -> bool:
        with self._lock:
            return self._cap is not None and self._cap.isOpened()

    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        with self._lock:
            if self._cap is None or not self._cap.isOpened():
                return False, None
            try:
                return self._cap.read()
            except Exception as exc:
                logger.error("Exception during cv2.VideoCapture.read(): %s", exc)
                return False, None

    def release(self) -> None:
        with self._lock:
            if self._cap is not None:
                try:
                    self._cap.release()
                except Exception as exc:
                    logger.debug("Exception releasing VideoCapture: %s", exc)
                finally:
                    self._cap = None


class CameraOwner(CameraInterface):
    """Authoritative single camera owner providing shared, fresh optical frames."""

    _active_owner: Optional["CameraOwner"] = None
    _owner_lock = threading.RLock()

    def __init__(
        self,
        config: Optional[CameraConfig] = None,
        backend: Optional[Any] = None,
        max_staleness_sec: float = 2.00,
        max_consecutive_drops: int = 30,
        max_recovery_attempts: int = 5,
        recovery_backoff_sec: float = 0.15,
    ):
        self.config = config or CameraConfig()
        self.backend = backend or OpenCVCameraBackend(
            device_index=self.config.device_index,
            width=self.config.width,
            height=self.config.height,
            fps=self.config.fps,
        )
        self.max_staleness_sec = float(max_staleness_sec)
        self.max_consecutive_drops = int(max_consecutive_drops)
        self.max_recovery_attempts = int(max_recovery_attempts)
        self.recovery_backoff_sec = float(recovery_backoff_sec)

        self._is_running = False
        self._is_operational = False
        self._has_terminal_fault = False
        self._recovery_attempts = 0
        self._thread: Optional[threading.Thread] = None
        self._lock = threading.RLock()

        self._latest_frame: Optional[np.ndarray] = None
        self._latest_timestamp: float = 0.0
        self._consecutive_drops: int = 0
        self._total_frames_captured: int = 0

    @property
    def is_running(self) -> bool:
        """Returns True if the background capture thread is running."""
        with self._lock:
            return self._is_running

    @property
    def is_operational(self) -> bool:
        """Returns True only if the camera is opened, capturing frames, and within error limits."""
        with self._lock:
            return self._is_operational and self._is_running

    @property
    def has_terminal_fault(self) -> bool:
        """Returns True if recovery attempts were exhausted."""
        with self._lock:
            return self._has_terminal_fault

    def is_opened(self) -> bool:
        with self._lock:
            return self.backend.is_opened() if self.backend else False

    def start(self) -> bool:
        """Open camera backend and launch background frame-draining thread."""
        prev_owner = None
        with CameraOwner._owner_lock:
            if CameraOwner._active_owner is not None and CameraOwner._active_owner is not self:
                prev_owner = CameraOwner._active_owner
            CameraOwner._active_owner = self

        if prev_owner is not None and prev_owner.is_running:
            logger.warning(
                "Another CameraOwner instance is active; stopping previous instance."
            )
            try:
                prev_owner.stop()
            except Exception as exc:
                logger.error("Failed to stop previous CameraOwner: %s", exc)

        with self._lock:
            if self._is_running:
                return True

            logger.info("Starting CameraOwner background capture thread...")
            if not self.backend.is_opened():
                opened = self.backend.open()
                if not opened:
                    self._is_operational = False
                    logger.error("CameraOwner start failed: backend could not be opened.")
                    return False

            self._is_running = True
            self._consecutive_drops = 0
            self._is_operational = False  # Becomes True when first valid frame arrives
            self._has_terminal_fault = False
            self._recovery_attempts = 0

            self._thread = threading.Thread(
                target=self._capture_worker, daemon=True, name="CameraWorker"
            )
            self._thread.start()
            logger.info("CameraOwner capture worker started.")
            return True

    def _attempt_recovery(self) -> bool:
        """Controlled recovery policy when consecutive frame drops exceed threshold."""
        if self._recovery_attempts >= self.max_recovery_attempts:
            self._has_terminal_fault = True
            logger.error(
                "Camera recovery budget exhausted (%d/%d attempts). Entering terminal fault state.",
                self._recovery_attempts,
                self.max_recovery_attempts,
            )
            try:
                self.backend.release()
            except Exception:
                pass
            return False

        self._recovery_attempts += 1
        logger.warning(
            "Attempting camera recovery (attempt %d/%d) after %d drops...",
            self._recovery_attempts,
            self.max_recovery_attempts,
            self._consecutive_drops,
        )

        try:
            self.backend.release()
            time.sleep(self.recovery_backoff_sec)
            opened = self.backend.open()
            if opened:
                self._consecutive_drops = 0
                logger.info("Camera recovery successful; backend reopened.")
                return True
            else:
                logger.warning("Camera recovery reopen failed.")
                if self._recovery_attempts >= self.max_recovery_attempts:
                    self._has_terminal_fault = True
                    try:
                        self.backend.release()
                    except Exception:
                        pass
                return False
        except Exception as exc:
            logger.error("Exception during camera recovery: %s", exc)
            return False

    def _capture_worker(self) -> None:
        """Continuously grab frames to prevent V4L2 driver queue staleness."""
        target_delay = 1.0 / max(1, self.config.fps)

        while self._is_running:
            if self._has_terminal_fault:
                break

            loop_start = time.perf_counter()
            try:
                success, frame = self.backend.read()
            except Exception as exc:
                logger.error("Camera backend read raised an exception: %s", exc)
                success, frame = False, None
            now = time.time()

            with self._lock:
                if not self._is_running or self._has_terminal_fault:
                    break

                if success and frame is not None:
                    h, w = frame.shape[:2]
                    if (w, h) == (self.config.width, self.config.height):
                        self._latest_frame = frame
                        self._latest_timestamp = now
                        self._consecutive_drops = 0
                        self._total_frames_captured += 1
                        self._is_operational = True
                    else:
                        logger.warning(
                            "Frame dimension mismatch: got %dx%d, expected %dx%d",
                            w,
                            h,
                            self.config.width,
                            self.config.height,
                        )
                        self._handle_frame_drop()
                else:
                    self._handle_frame_drop()

            if self._has_terminal_fault:
                with self._lock:
                    self._is_running = False
                    self._is_operational = False
                break

            elapsed = time.perf_counter() - loop_start
            sleep_time = max(0.001, target_delay - elapsed)
            time.sleep(sleep_time)

    def _handle_frame_drop(self) -> None:
        """Record drop and trigger recovery if threshold exceeded."""
        self._consecutive_drops += 1
        if self._consecutive_drops >= self.max_consecutive_drops:
            self._is_operational = False
            self._latest_frame = None
            if not self._has_terminal_fault:
                self._attempt_recovery()
            if self._has_terminal_fault:
                self._is_running = False

    def capture_frame(self) -> Optional[np.ndarray]:
        """Return the latest fresh optical frame as a uint8 BGR array [H, W, 3].
        Returns None if frame is dropped, stale (> max_staleness_sec), or camera is degraded.
        """
        frame, _ = self.get_latest_frame_with_timestamp()
        return frame

    def get_latest_frame_with_timestamp(self) -> Tuple[Optional[np.ndarray], float]:
        """Return latest frame and its acquisition timestamp."""
        with self._lock:
            if not self._is_running or not self._is_operational or self._latest_frame is None:
                return None, 0.0

            age = time.time() - self._latest_timestamp
            if age > self.max_staleness_sec:
                logger.warning(
                    "Captured frame is stale (age %.3fs > threshold %.3fs). Dropping.",
                    age,
                    self.max_staleness_sec,
                )
                return None, self._latest_timestamp

            return self._latest_frame.copy(), self._latest_timestamp

    def stop(self) -> None:
        """Stop background capture worker thread and release backend camera hardware."""
        with self._lock:
            self._is_running = False
            self._is_operational = False

        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=1.5)
            if self._thread.is_alive():
                logger.error("CameraWorker thread did not terminate within timeout.")
            else:
                logger.info("CameraOwner worker stopped.")

        with self._lock:
            if self.backend and hasattr(self.backend, "is_opened") and self.backend.is_opened():
                try:
                    self.backend.release()
                except Exception as exc:
                    logger.error("Exception releasing backend in stop(): %s", exc)

        with CameraOwner._owner_lock:
            if CameraOwner._active_owner is self:
                CameraOwner._active_owner = None

    def release(self) -> None:
        """Permanently stop capture and release backend camera hardware."""
        self.stop()
        with self._lock:
            if self.backend and hasattr(self.backend, "is_opened") and self.backend.is_opened():
                try:
                    self.backend.release()
                except Exception as exc:
                    logger.debug("Exception releasing backend in release(): %s", exc)
            self._latest_frame = None
            self._latest_timestamp = 0.0
            self._is_operational = False
            self._is_running = False
            logger.info("CameraOwner backend released.")
