"""Microphone Audio Capture Subsystem for SS_PROJECT_VCR_FRESH.

Features:
- Dynamically discovers and selects the Logitech C270 USB microphone (never assumes index 0).
- Verifies format and converts audio to 16000 Hz 16-bit mono PCM required by Vosk.
- Employs a bounded ring buffer (queue.Queue) to prevent unbounded memory growth and lag.
- Single owner: enforces one authoritative microphone stream instance.
- Clean and responsive shutdown without thread leaks.
- Hardware-free simulation mode support via dependency injection.
"""
from __future__ import annotations

import queue
import sys
import threading
import time
from typing import Any, Dict, List, Optional, Tuple
import numpy as np

from app.config import AudioConfig
from app.constants import AUDIO_CHANNELS, AUDIO_CHUNK_SIZE, AUDIO_SAMPLE_RATE
from app.hardware.interfaces import AudioSourceInterface
from app.logger import get_logger

logger = get_logger("audio")


def find_logitech_c270_device_index(devices: Optional[List[Dict[str, Any]]] = None) -> Optional[int]:
    """Inspect audio input devices to discover the Logitech C270 USB microphone."""
    if devices is None:
        try:
            import sounddevice as sd
            devices = list(sd.query_devices())
        except Exception as exc:
            logger.warning("Unable to query sounddevice audio devices: %s", exc)
            return None

    input_devices = [
        (idx, dev, str(dev.get("name", "")).casefold())
        for idx, dev in enumerate(devices)
        if int(dev.get("max_input_channels", 0)) > 0
    ]
    # Prefer a positive Logitech/C270 identity match over generic USB audio.
    for idx, dev, name in input_devices:
        if "c270" in name or "logitech" in name or "0x046d" in name or "0x46d" in name:
            logger.info("Selected Logitech microphone index %d ('%s')", idx, dev.get("name"))
            return idx

    # A generic USB audio name is acceptable only when exactly one USB input
    # exists; multiple devices are ambiguous and must not select the wrong mic.
    generic_usb = [(idx, dev, name) for idx, dev, name in input_devices
                   if ("usb audio" in name or "usb pnp" in name or "webcam" in name)]
    if len(generic_usb) == 1:
        idx, dev, _ = generic_usb[0]
        logger.info("Selected sole generic USB microphone index %d ('%s')", idx, dev.get("name"))
        return idx

    logger.warning("Logitech C270 microphone not identified unambiguously; refusing a default-device fallback.")
    return None


def convert_and_resample_pcm(
    raw_samples: np.ndarray,
    input_sample_rate: int,
    target_sample_rate: int = AUDIO_SAMPLE_RATE,
) -> bytes:
    """Convert input numpy array to 16-bit signed PCM mono at 16000 Hz."""
    # Convert multi-channel to mono by averaging channels
    if raw_samples.ndim > 1 and raw_samples.shape[1] > 1:
        mono_samples = np.mean(raw_samples, axis=1)
    else:
        mono_samples = raw_samples.flatten()

    # Resample if needed
    if input_sample_rate != target_sample_rate and len(mono_samples) > 0:
        # Linear interpolation resampling
        old_indices = np.linspace(0, len(mono_samples) - 1, len(mono_samples))
        new_length = int(len(mono_samples) * (target_sample_rate / input_sample_rate))
        new_indices = np.linspace(0, len(mono_samples) - 1, new_length)
        mono_samples = np.interp(new_indices, old_indices, mono_samples)

    # Convert float [-1.0, 1.0] to int16 [-32768, 32767]
    if mono_samples.dtype != np.int16:
        # If float, clip and scale
        if np.issubdtype(mono_samples.dtype, np.floating):
            clipped = np.clip(mono_samples, -1.0, 1.0)
            int16_samples = (clipped * 32767.0).astype(np.int16)
        else:
            int16_samples = mono_samples.astype(np.int16)
    else:
        int16_samples = mono_samples

    return int16_samples.tobytes()


class MicrophoneAudioSource(AudioSourceInterface):
    """Authoritative audio source streaming 16 kHz mono PCM from USB microphone."""

    def __init__(
        self,
        config: Optional[AudioConfig] = None,
        device_index: Optional[int] = None,
        max_buffered_chunks: int = 50,  # ~3.2 seconds at 16 kHz / 1024 chunk
    ):
        self.config = config or AudioConfig()
        self.requested_device_index = device_index
        self.max_buffered_chunks = max(5, int(max_buffered_chunks))

        self._audio_queue: queue.Queue[bytes] = queue.Queue(maxsize=self.max_buffered_chunks)
        self._is_streaming = False
        self._is_closed = False
        self._stream: Any = None
        self._lock = threading.Lock()
        self._active_device_index: Optional[int] = None

    @property
    def active_device_index(self) -> Optional[int]:
        return self._active_device_index

    def _audio_callback(self, indata: np.ndarray, frames: int, time_info: Any, status: Any) -> None:
        """SoundDevice callback called from low-level audio driver thread."""
        if not self._is_streaming:
            return

        pcm_bytes = convert_and_resample_pcm(
            indata,
            input_sample_rate=self.config.sample_rate,
            target_sample_rate=AUDIO_SAMPLE_RATE,
        )

        try:
            self._audio_queue.put_nowait(pcm_bytes)
        except queue.Full:
            # Buffer overrun: drop oldest chunk to maintain real-time latency
            try:
                self._audio_queue.get_nowait()
                self._audio_queue.put_nowait(pcm_bytes)
            except Exception:
                pass

    def start_stream(self) -> None:
        """Start audio stream acquisition."""
        with self._lock:
            if self._is_streaming or self._is_closed:
                return

            import sounddevice as sd

            # Resolve device index
            dev_idx = self.requested_device_index
            if dev_idx is None:
                dev_idx = find_logitech_c270_device_index()
                if dev_idx is None:
                    raise RuntimeError(
                        "Logitech C270 microphone was not identified. Refusing to silently use the system-default microphone; "
                        "inspect sounddevice.query_devices() and configure the intended input device."
                    )

            self._active_device_index = dev_idx
            logger.info("Opening sounddevice InputStream on device index %s...", dev_idx)

            try:
                self._stream = sd.InputStream(
                    device=dev_idx,
                    samplerate=self.config.sample_rate,
                    channels=self.config.channels,
                    dtype="int16",
                    blocksize=self.config.chunk_size,
                    callback=self._audio_callback,
                )
                self._stream.start()
                self._is_streaming = True
                logger.info("Microphone audio stream started successfully.")
            except Exception as exc:
                logger.error("Failed to start sounddevice audio stream: %s", exc)
                self._is_streaming = False
                raise

    def read_chunk(self) -> Optional[bytes]:
        """Read a single raw 16-bit 16 kHz mono PCM chunk. Non-blocking."""
        if not self._is_streaming or self._is_closed:
            return None
        try:
            return self._audio_queue.get_nowait()
        except queue.Empty:
            return None

    def stop_stream(self) -> None:
        """Stop and pause microphone audio stream."""
        with self._lock:
            if not self._is_streaming:
                return
            self._is_streaming = False

            if self._stream:
                try:
                    self._stream.stop()
                except Exception as exc:
                    logger.warning("Error stopping sounddevice stream: %s", exc)

        logger.info("Microphone audio stream stopped.")

    def close(self) -> None:
        """Permanently close and release audio hardware device."""
        self.stop_stream()
        with self._lock:
            if self._stream:
                try:
                    self._stream.close()
                except Exception as exc:
                    logger.warning("Error closing sounddevice stream: %s", exc)
                self._stream = None
            self._is_closed = True

        # Clear queued audio
        while not self._audio_queue.empty():
            try:
                self._audio_queue.get_nowait()
            except Exception:
                break
        logger.info("Microphone audio source closed.")
