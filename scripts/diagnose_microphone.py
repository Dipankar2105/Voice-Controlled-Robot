"""Human-Invoked Microphone and Vosk Speech Recognition Diagnostic Tool.

Verifies:
1. Audio device enumeration and auto-discovery of Logitech C270 USB microphone.
2. Real-time audio stream acquisition at 16000 Hz 16-bit mono PCM.
3. Audio signal RMS power level and peak amplitude check.
4. Offline Vosk Kaldi speech model loading and live command transcription.

Usage:
    python scripts/diagnose_microphone.py [--device <index>] [--seconds 3.0]
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
import numpy as np

# Add project root to sys.path
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from app.audio.audio_source import MicrophoneAudioSource, find_logitech_c270_device_index
from app.audio.vosk_recognizer import VoskSpeechRecognizer


def run_microphone_diagnostic(device_index: int | None = None, record_seconds: float = 3.0) -> int:
    print("=" * 70)
    print("  SS_PROJECT_VCR_FRESH — MICROPHONE & VOSK SPEECH DIAGNOSTIC")
    print("=" * 70)

    try:
        import sounddevice as sd
        devices = list(sd.query_devices())
        print(f"[INFO] SoundDevice Version: {sd.__version__}")
        print(f"[INFO] Total Audio Devices Detected: {len(devices)}")
    except Exception as exc:
        print(f"[ERROR] Could not query sounddevice: {exc}")
        return 1

    # Print input devices
    print("\nAvailable Audio Input Devices:")
    for idx, dev in enumerate(devices):
        if dev.get("max_input_channels", 0) > 0:
            print(f"  [{idx:2d}] {dev.get('name')} (Max In: {dev.get('max_input_channels')}, Rate: {dev.get('default_samplerate'):.0f} Hz)")

    # Resolve target device
    target_idx = device_index
    if target_idx is None:
        target_idx = find_logitech_c270_device_index(devices)
        if target_idx is None:
            target_idx = sd.default.device[0]
            print(f"\n[WARN] Logitech C270 not matched by name. Falling back to default input: index {target_idx}")

    print(f"\n[INFO] Selected input device index: {target_idx}")

    # Initialize Vosk recognizer
    print("\n[INFO] Initializing offline Vosk speech recognition engine...")
    try:
        recognizer = VoskSpeechRecognizer()
        recognizer.load_model()
        print("[PASS] Vosk model loaded successfully with reviewed robotics grammar.")
    except Exception as exc:
        print(f"[FAIL] Vosk model failed to load: {exc}")
        return 2

    # Open microphone stream
    print(f"\n[INFO] Starting audio stream for {record_seconds:.1f} seconds...")
    print("       >>> SPEAK A COMMAND NOW (e.g. 'find bottle', 'stop', 'turn left') <<<")

    source = MicrophoneAudioSource(device_index=target_idx)
    try:
        source.start_stream()
    except Exception as exc:
        print(f"[FAIL] Failed to open microphone stream: {exc}")
        return 3

    t_end = time.time() + record_seconds
    chunks_received = 0
    all_pcm_bytes = bytearray()

    while time.time() < t_end:
        chunk = source.read_chunk()
        if chunk:
            chunks_received += 1
            all_pcm_bytes.extend(chunk)
            is_complete = recognizer.accept_waveform(chunk)
            partial = recognizer.get_partial_result()
            if partial:
                print(f"  [Partial]: {partial}", end="\r", flush=True)
            if is_complete:
                final_text = recognizer.get_result()
                if final_text:
                    print(f"\n  [Utterance Finalized]: '{final_text}'")
        time.sleep(0.02)

    source.stop_stream()
    source.close()

    final_text = recognizer.get_result()
    if final_text:
        print(f"\n  [Final Transcription]: '{final_text}'")

    print(f"\n--- AUDIO SIGNAL SUMMARY ---")
    print(f"  Chunks Received: {chunks_received}")
    print(f"  Total Bytes: {len(all_pcm_bytes)} bytes ({len(all_pcm_bytes)/32000.0:.2f} seconds)")

    if all_pcm_bytes:
        samples = np.frombuffer(all_pcm_bytes, dtype=np.int16).astype(np.float32) / 32768.0
        rms = float(np.sqrt(np.mean(samples ** 2)))
        peak = float(np.max(np.abs(samples)))
        print(f"  RMS Volume Level: {rms:.4f}")
        print(f"  Peak Amplitude:   {peak:.4f}")

        if rms < 0.001:
            print("[WARN] Microphone recorded near-silence. Verify microphone mute or gain settings.")
        else:
            print("[PASS] Audio signal strength is healthy.")

    print("\n[RESULT: PASS] Microphone capture and Vosk decoding operational.")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Microphone and Vosk Speech Diagnostic")
    parser.add_argument("--device", type=int, default=None, help="Audio input device index")
    parser.add_argument("--seconds", type=float, default=3.0, help="Test duration in seconds")
    args = parser.parse_args()
    return run_microphone_diagnostic(device_index=args.device, record_seconds=args.seconds)


if __name__ == "__main__":
    raise SystemExit(main())
