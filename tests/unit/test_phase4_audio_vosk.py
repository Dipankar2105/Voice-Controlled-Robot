"""Phase 4 unit tests: Microphone audio capture, PCM resampling, bounded queues,
and offline Vosk speech recognition with grammar constraints.
"""
from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch
import numpy as np
import pytest

from app.audio.audio_source import (
    MicrophoneAudioSource,
    convert_and_resample_pcm,
    find_logitech_c270_device_index,
)
from app.audio.vosk_recognizer import (
    REVIEWED_COMMAND_VOCABULARY,
    VoskSpeechRecognizer,
    build_grammar_json,
)
from app.config import AudioConfig, get_project_root


# ---------------------------------------------------------------------------
# 1. Device Discovery Tests
# ---------------------------------------------------------------------------

def test_find_logitech_c270_device_index():
    """Device search matches Logitech C270, ignoring output-only devices."""
    mock_devices = [
        {"name": "Speakers (Realtek Audio)", "max_input_channels": 0, "default_samplerate": 48000},
        {"name": "C270 HD Webcam Output", "max_input_channels": 0, "default_samplerate": 48000},  # Output only!
        {"name": "Built-in Microphone", "max_input_channels": 2, "default_samplerate": 44100},
        {"name": "HD Webcam C270 Audio", "max_input_channels": 1, "default_samplerate": 16000},  # Target!
    ]

    found_idx = find_logitech_c270_device_index(mock_devices)
    assert found_idx == 3


def test_find_logitech_c270_fallback_when_not_found():
    """When Logitech C270 is absent, device search safely returns None."""
    mock_devices = [
        {"name": "Realtek Audio", "max_input_channels": 2, "default_samplerate": 48000},
        {"name": "Stereo Mix", "max_input_channels": 2, "default_samplerate": 44100},
    ]
    found_idx = find_logitech_c270_device_index(mock_devices)
    assert found_idx is None

    # Two generic USB devices are ambiguous and must not silently select the first.
    ambiguous = [
        {"name": "USB Audio Device 1", "max_input_channels": 1, "default_samplerate": 16000},
        {"name": "USB Audio Device 2", "max_input_channels": 1, "default_samplerate": 16000},
    ]
    assert find_logitech_c270_device_index(ambiguous) is None


# ---------------------------------------------------------------------------
# 2. Format Conversion & Resampling Tests
# ---------------------------------------------------------------------------

def test_convert_and_resample_pcm_stereo_to_mono_and_downsample():
    """Stereo float32 audio at 48000 Hz is downsampled to 16000 Hz mono 16-bit PCM."""
    # 480 samples at 48 kHz = 10 ms of audio
    # 2 channels (stereo)
    stereo_48k = np.ones((480, 2), dtype=np.float32) * 0.50

    pcm_bytes = convert_and_resample_pcm(
        raw_samples=stereo_48k,
        input_sample_rate=48000,
        target_sample_rate=16000,
    )

    # 480 samples downsampled 3:1 -> 160 samples. Each int16 sample is 2 bytes -> 320 bytes
    assert len(pcm_bytes) == 160 * 2

    # Check sample values: 0.50 * 32767 ≈ 16383
    samples = np.frombuffer(pcm_bytes, dtype=np.int16)
    assert len(samples) == 160
    assert np.all(samples >= 16380) and np.all(samples <= 16385)


def test_convert_and_resample_pcm_clipping():
    """Extreme float samples outside [-1.0, 1.0] must be clipped safely without overflow."""
    out_of_bounds = np.array([-2.5, 0.0, 3.8], dtype=np.float32)
    pcm_bytes = convert_and_resample_pcm(out_of_bounds, input_sample_rate=16000, target_sample_rate=16000)
    samples = np.frombuffer(pcm_bytes, dtype=np.int16)

    assert samples[0] == -32767
    assert samples[1] == 0
    assert samples[2] == 32767


# ---------------------------------------------------------------------------
# 3. Audio Source Bounded Queue & Lifecycle Tests
# ---------------------------------------------------------------------------

def test_microphone_audio_source_bounded_queue():
    """Queue drops oldest chunks when overflowing to avoid latency and memory growth."""
    source = MicrophoneAudioSource(max_buffered_chunks=5)
    source._is_streaming = True

    # Push 10 distinct chunks into callback
    for i in range(10):
        dummy_chunk = np.full((160,), fill_value=(i / 10.0), dtype=np.float32)
        source._audio_callback(dummy_chunk, 160, None, None)

    # Buffer queue maxsize is 5; exactly 5 chunks should remain (the newest 5: indices 5, 6, 7, 8, 9)
    chunks_read = []
    while True:
        chunk = source.read_chunk()
        if chunk is None:
            break
        chunks_read.append(chunk)

    assert len(chunks_read) == 5

    # Check that first chunk read corresponds to index 5 (oldest was dropped)
    first_chunk_samples = np.frombuffer(chunks_read[0], dtype=np.int16)
    expected_val = int(0.50 * 32767.0)
    assert abs(first_chunk_samples[0] - expected_val) < 20


def test_microphone_audio_source_lifecycle_mocked():
    """Microphone stream can start, pause, and close without leaking resources."""
    pytest.importorskip("sounddevice", reason="sounddevice is not installed in this test environment")
    mock_stream = MagicMock()

    with patch("sounddevice.InputStream", return_value=mock_stream), \
         patch("sounddevice.query_devices", return_value=[{"name": "Logitech C270", "max_input_channels": 1}]):
        source = MicrophoneAudioSource()
        source.start_stream()
        assert source._is_streaming is True
        mock_stream.start.assert_called_once()

        source.stop_stream()
        assert source._is_streaming is False
        mock_stream.stop.assert_called_once()

        source.close()
        assert source._is_closed is True
        mock_stream.close.assert_called_once()


# ---------------------------------------------------------------------------
# 4. Vosk Model Loading & Grammar Tests
# ---------------------------------------------------------------------------

def test_vosk_missing_model_directory_fails_closed(tmp_path: Path):
    """Missing model directory raises FileNotFoundError without silent fallback."""
    fake_path = tmp_path / "nonexistent_vosk_dir"
    recognizer = VoskSpeechRecognizer(model_path=fake_path)

    with pytest.raises(FileNotFoundError, match="Offline Vosk model directory missing"):
        recognizer.load_model()


def test_vosk_corrupt_model_directory_fails_closed(tmp_path: Path):
    """Incomplete model directory missing key Kaldi files raises FileNotFoundError."""
    corrupt_dir = tmp_path / "corrupt_model"
    corrupt_dir.mkdir()
    (corrupt_dir / "conf").mkdir()
    (corrupt_dir / "conf" / "model.conf").write_text("test")

    recognizer = VoskSpeechRecognizer(model_path=corrupt_dir)
    with pytest.raises(FileNotFoundError, match="Vosk model component corrupt or missing"):
        recognizer.load_model()


def test_grammar_vocabulary_coverage():
    """Grammar JSON contains essential robot control, safety, and 26 target classes."""
    grammar_json_str = build_grammar_json()
    tokens = json.loads(grammar_json_str)

    # Core control commands
    assert "stop" in tokens
    assert "halt" in tokens
    assert "forward" in tokens
    assert "backward" in tokens
    assert "turn left" in tokens
    assert "turn right" in tokens

    # Targets & Classes
    assert "person" in tokens
    assert "bottle" in tokens
    assert "chair" in tokens
    assert "red target" in tokens
    assert "blue target" in tokens
    assert "scissors" in tokens
    assert "find bottle" in tokens
    assert "search for scissors" in tokens
    assert "look for red target" in tokens
    assert "wait" in tokens
    assert "find seat" in tokens
    assert "find notebook" in tokens
    assert "find mug" in tokens


def test_vosk_recognizer_with_preserved_model():
    """Load the cryptographically preserved Vosk model and feed synthetic waveform."""
    pytest.importorskip("vosk", reason="Vosk native package is not installed in this test environment")
    recognizer = VoskSpeechRecognizer()
    recognizer.load_model()

    # Model and KaldiRecognizer should be instantiated
    assert recognizer._model is not None
    assert recognizer._recognizer is not None

    # Feed 0.5s of synthetic 16 kHz 16-bit silence
    silence = b"\x00" * 16000
    is_complete = recognizer.accept_waveform(silence)

    # Partial result should be callable and return string
    partial = recognizer.get_partial_result()
    assert isinstance(partial, str)

    # Reset recognizer
    recognizer.reset()
    assert recognizer.get_partial_result() == ""
    assert recognizer.get_result() == ""


def test_audio_source_single_initialization_and_idempotency():
    """Microphone stream start_stream is idempotent and initializes backend stream only once."""
    pytest.importorskip("sounddevice", reason="sounddevice is not installed in this test environment")
    mock_stream = MagicMock()

    with patch("sounddevice.InputStream", return_value=mock_stream), \
         patch("sounddevice.query_devices", return_value=[{"name": "Logitech C270", "max_input_channels": 1}]):
        source = MicrophoneAudioSource()

        # Call start_stream multiple times
        source.start_stream()
        source.start_stream()
        source.start_stream()

        # Backend InputStream must only be created once
        mock_stream.start.assert_called_once()
        assert source._is_streaming is True

        source.close()
        assert source._is_closed is True


def test_microphone_stream_open_failure_cleanup():
    """Stream initialization error must clean up partial stream and mark source not streaming."""
    pytest.importorskip("sounddevice", reason="sounddevice is not installed in this test environment")

    with patch("sounddevice.InputStream", side_effect=IOError("Device busy or unavailable")), \
         patch("sounddevice.query_devices", return_value=[{"name": "Logitech C270", "max_input_channels": 1}]):
        source = MicrophoneAudioSource()
        with pytest.raises(IOError, match="Device busy"):
            source.start_stream()

        assert source._is_streaming is False
        assert source._stream is None


def test_microphone_stream_closed_cannot_restart():
    """Calling start_stream on a closed MicrophoneAudioSource must raise RuntimeError."""
    source = MicrophoneAudioSource()
    source.close()
    assert source._is_closed is True

    with pytest.raises(RuntimeError, match="Cannot start audio stream on a closed"):
        source.start_stream()


def test_convert_and_resample_pcm_empty_and_nan_handling():
    """Empty or non-finite inputs must not raise and must return safe PCM bytes."""
    # Empty array
    assert convert_and_resample_pcm(np.array([], dtype=np.float32), 16000) == b""

    # Array containing NaNs and Infs
    corrupt = np.array([float("nan"), float("inf"), float("-inf"), 0.5], dtype=np.float32)
    pcm_bytes = convert_and_resample_pcm(corrupt, 16000, 16000)
    samples = np.frombuffer(pcm_bytes, dtype=np.int16)
    assert len(samples) == 4
    assert samples[0] == 0  # nan -> 0
    assert samples[1] == 32767  # +inf -> 1.0 -> 32767
    assert samples[2] == -32767  # -inf -> -1.0 -> -32767
    assert abs(samples[3] - 16383) < 10


def test_vosk_accept_waveform_malformed_and_odd_bytes():
    """Recognizer safely handles empty and odd-byte PCM chunks without Kaldi crashes."""
    recognizer = VoskSpeechRecognizer()
    recognizer._recognizer = MagicMock()
    recognizer._recognizer.AcceptWaveform.return_value = False
    recognizer._recognizer.PartialResult.return_value = '{"partial": ""}'

    # Empty bytes -> False
    assert recognizer.accept_waveform(b"") is False

    # Odd length bytes (3 bytes) -> truncated to 2 bytes and fed
    assert recognizer.accept_waveform(b"\x00\x01\x02") is False
    # Verify AcceptWaveform was called with 2 bytes
    recognizer._recognizer.AcceptWaveform.assert_called_with(b"\x00\x01")


def test_vosk_accept_waveform_exception_handling():
    """Exceptions in Kaldi AcceptWaveform are caught and return False safely."""
    recognizer = VoskSpeechRecognizer()
    recognizer._recognizer = MagicMock()
    recognizer._recognizer.AcceptWaveform.side_effect = RuntimeError("Kaldi internal decode error")

    assert recognizer.accept_waveform(b"\x00\x00" * 100) is False


def test_vosk_recognizer_close_and_resource_cleanup():
    """close() resets Kaldi recognizer and releases model references."""
    recognizer = VoskSpeechRecognizer()
    recognizer._recognizer = MagicMock()
    recognizer._model = MagicMock()
    recognizer._vosk = MagicMock()

    recognizer.close()
    assert recognizer._recognizer is None
    assert recognizer._model is None
    assert recognizer._vosk is None
    assert recognizer.get_result() == ""
    assert recognizer.get_partial_result() == ""

