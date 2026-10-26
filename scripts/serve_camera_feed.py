"""Live Camera and YOLO Vision Web Server for SS_PROJECT_VCR.

Streams live Logitech C270 video with real-time YOLO object detection bounding boxes
to a web browser at http://0.0.0.0:8000.
"""
from __future__ import annotations

import asyncio
import os
import sys
import time
from pathlib import Path
from typing import AsyncGenerator

import cv2
import numpy as np
import uvicorn
from fastapi import FastAPI
from fastapi.responses import HTMLResponse, StreamingResponse

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from app.vision.camera import CameraOwner, OpenCVCameraBackend
from app.vision.yolo_detector import ONNXYOLODetector

app = FastAPI(title="SS_PROJECT_VCR Live Camera Feed")

# Global instances
backend = OpenCVCameraBackend(device_index=0, width=640, height=480, fps=15)
camera = CameraOwner(backend=backend, max_staleness_sec=0.50)
model_path = _ROOT / "models" / "object_detection" / "best_v2_target.onnx"
detector = ONNXYOLODetector(model_path=model_path, confidence_threshold=0.25, nms_threshold=0.45)

HTML_PAGE = """<!DOCTYPE html>
<html>
<head>
  <title>SS_PROJECT_VCR — Live Camera & YOLO Feed</title>
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <style>
    body {
      background: #0b0f19;
      color: #e2e8f0;
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
      margin: 0;
      padding: 20px;
      display: flex;
      flex-direction: column;
      align-items: center;
    }
    .header {
      background: #151d30;
      border: 1px solid #23314e;
      border-radius: 8px;
      padding: 14px 24px;
      margin-bottom: 20px;
      width: 100%;
      max-width: 680px;
      display: flex;
      justify-content: space-between;
      align-items: center;
      box-sizing: border-box;
    }
    .title { font-weight: 700; font-size: 1.2rem; color: #38bdf8; }
    .badge {
      background: #064e3b;
      color: #34d399;
      padding: 4px 10px;
      border-radius: 4px;
      font-size: 0.8rem;
      font-weight: 700;
    }
    .video-card {
      background: #151d30;
      border: 1px solid #23314e;
      border-radius: 8px;
      padding: 12px;
      width: 100%;
      max-width: 680px;
      box-sizing: border-box;
      text-align: center;
    }
    img.stream {
      width: 100%;
      max-width: 640px;
      height: auto;
      border-radius: 4px;
      border: 1px solid #334155;
    }
    .info {
      margin-top: 12px;
      font-size: 0.9rem;
      color: #94a3b8;
    }
  </style>
</head>
<body>
  <div class="header">
    <div class="title">SS_PROJECT_VCR — Live Camera & YOLO Feed</div>
    <div class="badge">LIVE STREAM</div>
  </div>
  <div class="video-card">
    <img class="stream" src="/stream" alt="Live Camera Feed" />
    <div class="info">Logitech C270 (640x480) &bull; YOLO ONNX Real-Time Detection</div>
  </div>
</body>
</html>
"""

@app.get("/", response_class=HTMLResponse)
def index():
    return HTMLResponse(content=HTML_PAGE)

async def mjpeg_frame_generator() -> AsyncGenerator[bytes, None]:
    last_detect_time = 0.0
    cached_detections = []
    
    while True:
        frame, ts = camera.get_latest_frame_with_timestamp()
        if frame is not None:
            now = time.time()
            # Run YOLO inference every 150ms (~6.6 Hz) to balance Pi 4 CPU load
            if (now - last_detect_time) > 0.15:
                try:
                    cached_detections = detector.detect(frame)
                    last_detect_time = now
                except Exception as e:
                    pass
            
            # Draw detections
            for det in cached_detections:
                x1, y1, x2, y2 = [int(v) for v in det.box_xyxy]
                color = (0, 255, 0)
                if "target" in det.class_name.lower():
                    color = (0, 0, 255)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                label = f"{det.class_name} {det.confidence:.2f}"
                cv2.putText(frame, label, (x1, max(20, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            
            # Encode as JPEG
            success, jpeg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
            if success:
                yield (b"--frame\r\n"
                       b"Content-Type: image/jpeg\r\n\r\n" + jpeg.tobytes() + b"\r\n")
        await asyncio.sleep(0.04)

@app.get("/stream")
async def stream():
    return StreamingResponse(
        mjpeg_frame_generator(),
        media_type="multipart/x-mixed-replace; boundary=frame"
    )

def main():
    print("[INFO] Initializing camera...")
    camera.start()
    time.sleep(1.8)
    print("[INFO] Loading YOLO ONNX model...")
    detector.load_model()
    print("[INFO] Starting server on http://0.0.0.0:8000...")
    try:
        uvicorn.run(app, host="0.0.0.0", port=8000, log_level="warning")
    finally:
        camera.stop()
        camera.release()
        detector.close()

if __name__ == "__main__":
    main()
