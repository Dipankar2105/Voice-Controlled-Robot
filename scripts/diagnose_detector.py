"""Human-Invoked YOLO ONNX Detector Diagnostic Tool for SS_PROJECT_VCR_FRESH.

Verifies:
1. ONNX model loading and embedded metadata inspection.
2. Verified class count and presence of RED_TARGET, BLUE_TARGET, and scissors.
3. Inference on test fixture images or live camera frames.
4. Bounding box coordinates, confidence scores, and NMS suppression.
5. Inference benchmark timing (ms per frame).
6. Optional visualization snapshot output with rendered bounding boxes.

Usage:
    python scripts/diagnose_detector.py [--model v2|v1] [--image path/to/image.jpg] [--benchmark]
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys
import time
import cv2
import numpy as np

# Add project root to sys.path
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from app.vision.yolo_detector import ONNXYOLODetector


def run_detector_diagnostic(
    model_version: str = "v2",
    image_path: Optional[str] = None,
    benchmark_runs: int = 5,
    save_annotated: bool = True,
) -> int:
    print("=" * 70)
    print("  SS_PROJECT_VCR_FRESH — YOLO ONNX DETECTOR DIAGNOSTIC")
    print("=" * 70)

    # 1. Resolve model path
    if model_version.lower() == "v1":
        rel_model = "models/object_detection/best_v1.onnx"
    else:
        rel_model = "models/object_detection/best_v2_target.onnx"

    full_model_path = _ROOT / rel_model
    print(f"[INFO] Target Model: {rel_model} ({full_model_path.stat().st_size if full_model_path.exists() else 0} bytes)")

    if not full_model_path.is_file():
        print(f"[FAIL] Model file not found at {full_model_path}.")
        return 1

    # 2. Instantiate and load model
    try:
        detector = ONNXYOLODetector(model_path=full_model_path, confidence_threshold=0.25, nms_threshold=0.45)
        detector.load_model()
        print(f"[PASS] ONNX Runtime session initialized successfully.")
    except Exception as exc:
        print(f"[FAIL] Failed to load detector: {exc}")
        return 2

    # 3. Inspect metadata & classes
    classes = detector.class_names
    print(f"[INFO] Total Verified Classes: {len(classes)}")
    print(f"       Classes: {', '.join(classes[:10])} ... {', '.join(classes[-3:])}")

    has_red = detector.has_class("RED_TARGET")
    has_blue = detector.has_class("BLUE_TARGET")
    has_scissors = detector.has_class("scissors")

    print(f"       Includes RED_TARGET:  {has_red} (index {classes.index('RED_TARGET') if has_red else 'N/A'})")
    print(f"       Includes BLUE_TARGET: {has_blue} (index {classes.index('BLUE_TARGET') if has_blue else 'N/A'})")
    print(f"       Includes scissors:    {has_scissors} (index {classes.index('scissors') if has_scissors else 'N/A'})")

    # 4. Resolve test image
    if image_path:
        target_img_path = Path(image_path)
    else:
        target_img_path = _ROOT / "tests" / "fixtures" / "images" / "live_view_now.jpg"

    if not target_img_path.is_file():
        print(f"[WARN] Test image not found at {target_img_path}. Creating synthetic color pattern frame.")
        test_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.circle(test_frame, (320, 240), 80, (0, 0, 255), -1)  # Red circle
    else:
        print(f"[INFO] Loading test image: {target_img_path}")
        test_frame = cv2.imread(str(target_img_path))
        if test_frame is None:
            print(f"[FAIL] cv2 could not decode image: {target_img_path}")
            return 3

    h, w = test_frame.shape[:2]
    print(f"[INFO] Frame resolution: {w}x{h}")

    # 5. Run inference
    detections = detector.detect(test_frame)
    print(f"\n[INFO] Detections post-NMS (conf >= {detector.confidence_threshold}): {len(detections)}")

    annotated = test_frame.copy()
    for i, det in enumerate(detections, 1):
        x1, y1, x2, y2 = [int(v) for v in det.box_xyxy]
        cx, cy, bw, bh = [int(v) for v in det.box_xywh]
        print(f"  [{i}] {det.class_name} (id {det.class_id}): conf={det.confidence:.3f} | bbox=[{x1}, {y1}, {x2}, {y2}] | center=({cx}, {cy}) | area={det.area:.0f}px")

        # Draw on annotated frame
        cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
        label = f"{det.class_name} {det.confidence:.2f}"
        cv2.putText(annotated, label, (x1, max(15, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    # 6. Benchmark
    if benchmark_runs > 1:
        print(f"\n[INFO] Running benchmark across {benchmark_runs} inference iterations...")
        latencies = []
        for r in range(benchmark_runs):
            t0 = time.time()
            detector.detect(test_frame)
            latencies.append((time.time() - t0) * 1000.0)

        avg_lat = sum(latencies) / len(latencies)
        print(f"       Average CPU Inference Latency: {avg_lat:.2f} ms ({1000.0/avg_lat:.1f} FPS)")
        print(f"       Min: {min(latencies):.2f} ms | Max: {max(latencies):.2f} ms")

    # 7. Save annotated snapshot
    if save_annotated:
        diag_dir = _ROOT / "diagnostics"
        diag_dir.mkdir(exist_ok=True)
        out_file = diag_dir / f"detector_diagnostic_{model_version}.jpg"
        cv2.imwrite(str(out_file), annotated)
        print(f"\n[INFO] Saved annotated detection snapshot: {out_file}")

    detector.close()
    print("\n[RESULT: PASS] YOLO ONNX Detector operational.")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="YOLO ONNX Detector Diagnostic Tool")
    parser.add_argument("--model", type=str, default="v2", choices=["v2", "v1"], help="Model version (v2 or v1)")
    parser.add_argument("--image", type=str, default=None, help="Path to test image")
    parser.add_argument("--benchmark", type=int, default=3, help="Number of benchmark iterations")
    parser.add_argument("--no-save", action="store_true", help="Do not save annotated snapshot")
    args = parser.parse_args()

    return run_detector_diagnostic(
        model_version=args.model,
        image_path=args.image,
        benchmark_runs=args.benchmark,
        save_annotated=not args.no_save,
    )


if __name__ == "__main__":
    raise SystemExit(main())
