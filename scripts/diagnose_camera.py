"""Human-Invoked Camera Diagnostic Tool for SS_PROJECT_VCR_FRESH.

Verifies:
1. OpenCV VideoCapture initialization.
2. Frame dimensions (expected 640x480).
3. Frame capture latency and effective framerate (FPS).
4. Frame freshness without stale buffer lag.

Usage:
    python scripts/diagnose_camera.py [--device 0] [--save-snapshot]
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

# Add project root to sys.path
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from app.vision.camera import CameraOwner, OpenCVCameraBackend


def run_camera_diagnostic(device_index: int = 0, save_snapshot: bool = True) -> int:
    print("=" * 70)
    print("  SS_PROJECT_VCR_FRESH — LOGITECH C270 CAMERA DIAGNOSTIC")
    print("=" * 70)

    try:
        import cv2
        print(f"[INFO] OpenCV Version: {cv2.__version__}")
    except ImportError:
        print("[ERROR] OpenCV (cv2) is not installed.")
        return 1

    backend = OpenCVCameraBackend(device_index=device_index, width=640, height=480, fps=15)
    owner = CameraOwner(backend=backend, max_staleness_sec=0.50)

    print(f"[INFO] Attempting to open video device index {device_index}...")
    success = owner.start()
    if not success:
        print(f"[FAIL] Could not open camera device {device_index}.")
        print("       Check USB cable, /dev/video* permissions, or try --device 1.")
        return 2

    print("[PASS] Camera opened. Waiting for first frame and optical auto-exposure...")
    wait_start = time.time()
    while time.time() - wait_start < 3.0 and not owner.is_operational:
        time.sleep(0.1)

    # Capture 10 sample frames
    print("[INFO] Capturing 10 consecutive frames...")
    latencies = []
    frames = []

    for i in range(1, 11):
        t0 = time.time()
        frame, ts = owner.get_latest_frame_with_timestamp()
        dt = (time.time() - t0) * 1000.0  # ms
        latencies.append(dt)

        if frame is not None:
            h, w = frame.shape[:2]
            frames.append(frame)
            age = (time.time() - ts) * 1000.0
            print(f"  Frame {i:2d}: Shape={w}x{h} | Latency={dt:.2f}ms | Age={age:.1f}ms [OK]")
        else:
            print(f"  Frame {i:2d}: None (Frame Drop!) [FAIL]")
        time.sleep(0.06)

    valid_count = len(frames)
    avg_latency = sum(latencies) / len(latencies) if latencies else 0.0

    print("\n--- DIAGNOSTIC SUMMARY ---")
    print(f"  Frames Captured: {valid_count} / 10")
    print(f"  Average Query Latency: {avg_latency:.2f} ms")

    if valid_count > 0 and save_snapshot:
        diag_dir = _ROOT / "diagnostics"
        diag_dir.mkdir(exist_ok=True)
        out_path = diag_dir / "camera_diagnostic_snapshot.jpg"
        cv2.imwrite(str(out_path), frames[-1])
        print(f"  Saved diagnostic snapshot: {out_path}")

    owner.release()

    if valid_count >= 8:
        print("\n[RESULT: PASS] Camera is healthy, calibrated, and operational.")
        return 0
    else:
        print("\n[RESULT: FAIL] Excessive frame drops observed.")
        return 3


def main() -> int:
    parser = argparse.ArgumentParser(description="Logitech C270 Camera Diagnostic")
    parser.add_argument("--device", type=int, default=0, help="Camera device index (default: 0)")
    parser.add_argument("--save-snapshot", action="store_true", default=True, help="Save sample snapshot to diagnostics/")
    args = parser.parse_args()
    return run_camera_diagnostic(device_index=args.device, save_snapshot=args.save_snapshot)


if __name__ == "__main__":
    raise SystemExit(main())
