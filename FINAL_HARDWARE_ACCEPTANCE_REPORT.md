# FINAL HARDWARE ACCEPTANCE & MASTER INTEGRATION REPORT
## SS_PROJECT_VCR_FRESH — Clean-Room Embedded Linux 4WD Robot

**Date:** 2026-10-22  
**Platform:** Raspberry Pi 4 Model B (Debian GNU/Linux 6.18.50 aarch64)  
**Host & Port:** `http://10.248.131.117:8000` / `http://127.0.0.1:8000`  
**Operational Mode:** `REAL_HARDWARE`  
**Overall Acceptance Gate:** **BENCH, GUI, VOICE PIPELINE & SOFTWARE ACCEPTED — PENDING SUPERVISED PHYSICAL FLOOR MOVEMENT (TEST 6A)**

---

## 1. Executive Summary

A systematic master integration and verification audit of `SS_Project_VCR_FRESH` was conducted. All six robot subsystems (motors, ultrasonic, camera, audio, speech, YOLO detector) have been audited, integrated, unit tested, and verified on live Raspberry Pi hardware.

The previous floor test returned HTTP 200 because the API and FSM executed cleanly, but the chassis only shook because a 250 ms pulse was insufficient to overcome static friction ($\mu_s$) and chassis inertia (~1.2 kg). The software has been updated to support a floor-calibrated, bounded, and configurable pulse duration (`manual_pulse_sec: 0.75`), with an asynchronous preemption loop ensuring emergency and active stops halt actuation within 20 ms.

The web-based control GUI has been launched and verified in the active desktop browser environment. Offline voice recognition through the Logitech C270 USB microphone and Vosk Kaldi speech engine has been comprehensively tested and verified end-to-end.

---

## 2. Master System Baseline & Configuration

### A. Active Configuration Invariants
- **Operational Mode:** `REAL_HARDWARE` (enforced via `.robot_token` authorization for mutating control routes).
- **Dual H-Bridge Motor Driver (L298N):**
  - Left Channel (A): ENA (`GPIO 18` / 100 Hz PWM), IN1 (`GPIO 17`), IN2 (`GPIO 27`).
  - Right Channel (B): ENB (`GPIO 13` / 100 Hz PWM), IN3 (`GPIO 5`), IN4 (`GPIO 22`) — inverted for chassis mirrored symmetry.
  - Commanded Speed: `0.75` (75% PWM duty cycle).
  - Configured Manual Pulse Duration: `0.75 s` (750 ms), bounded within $[0.05, 1.50]\text{ s}$.
  - Stop Behavior: High-Z free coasting (PWM duty cycles pulled to 0.0%, directional pins pulled to LOW).
- **Proximity Safety (HC-SR04):**
  - Pins: TRIG (`GPIO 23`), ECHO (`GPIO 24` with 5V -> 3.3V voltage divider and internal pull-down).
  - Safety Zones:
    - Critical Zone: $d < 15.0\text{ cm}$ (Emergency Stop; motion completely inhibited).
    - Blocked Zone: $15.0 \le d < 25.0\text{ cm}$ (Forward motion strictly refused).
    - Provisional Target Stop: $30.0\text{ cm}$ (Must remain strictly above blocked threshold).
    - Warning Zone: $25.0 \le d < 50.0\text{ cm}$ (Throttled speed alert).
    - Clear Zone: $d \ge 50.0\text{ cm}$ (Uninhibited forward motion).
    - Transducer Blind Zone: $d < 2.0\text{ cm}$ (Fail-closed CRITICAL underflow).
- **Vision (Logitech C270 & YOLO ONNX):**
  - Resolution: 640x480 @ 15 FPS via V4L2 background worker.
  - Model: `models/object_detection/best_v2_target.onnx` (26 verified classes, 320x320 input).
  - Confidence Threshold: `0.25`, NMS: `0.45`, Optical Centering Deadband: $\pm 40\text{ px}$.
- **Audio & Offline Voice (Logitech C270 & Vosk):**
  - Hardware: Card 3 Device 0 (`hw:3,0`, USB audio index 1).
  - Sampling: 16 kHz 16-bit mono PCM via dedicated background capture thread.
  - Model: Offline Kaldi small model (`models/voice/vosk-model/`).
  - Grammar: 1,162 robotics commands and object classes.

---

## 3. End-to-End Integration Audit & Remediations

### A. Root Causes Identified & Addressed
1. **Pulse Duration vs. Chassis Inertia:**
   - *Finding:* Manual directional movement was previously hardcoded to 250 ms (`pulse_sec = 0.25`). While sufficient for free-spinning bench wheels, it expired before static friction, gearbox backlash, and chassis weight (~1.2 kg) could be overcome on the floor.
   - *Fix:* Made manual movement duration configurable (`manual_pulse_sec` in `MotionConfig` and `default_config.yaml`), set to `0.75 s` (750 ms), bounded within $[0.05, 1.50]\text{ s}$.
2. **Emergency Preemption During Motion:**
   - *Finding:* An uninterrupted `time.sleep()` blocked immediate preemption until the pulse expired.
   - *Fix:* Replaced with a responsive 20 ms polling loop wrapped in `try...finally: self.motors.stop()`. If `/api/emergency-stop` or `/api/stop` is received, motor power is removed within 20 ms.
3. **Target Telemetry Bounding Box Geometries:**
   - *Finding:* In `app/runtime.py`, accessing `d.label` caused an `AttributeError` when detections were present.
   - *Fix:* Standardized detection geometry access to support `class_name` and `box_xywh`.
4. **Server Shutdown & Streaming Cleanup:**
   - *Finding:* When terminating the daemon, active MJPEG streaming connections could block uvicorn termination while the listen socket was closed.
   - *Fix:* Added client disconnect checking (`await request.is_disconnected()`) and uvicorn lifespan shutdown event handlers to trigger runtime resource release promptly.

---

## 4. Software Regression Test Results

Executed via `.venv/bin/pytest -v`:
- **Total Tests Collected:** 175
- **Passed:** **174**
- **Failed:** **0**
- **Skipped:** **1** (`test_external_source_archive_is_not_required_for_portable_tests`, environment-dependent archive check).
- **Pass Rate:** **100%** of active tests.

### Subsystem Unit Test Breakdown:
| Test Module | Coverage Area | Tests Run | Result |
| :--- | :--- | :--- | :--- |
| `test_asset_integrity.py` | Model checksums and cryptographic integrity | 3 | **PASS** |
| `test_config.py` | Pin validation, speed bounds, safety ordering | 10 | **PASS** |
| `test_mock_hardware.py` | Isolated driver mocks and interfaces | 5 | **PASS** |
| `test_phase1_scaffold.py` | Import safety, directory layout | 4 | **PASS** (1 skip) |
| `test_phase2_runtime.py` | Runtime startup, shutdown, health states | 9 | **PASS** |
| `test_phase3_hardware.py` | Motor polarity, PWM clamping, ultrasonic safety | 11 | **PASS** |
| `test_phase4_audio_vosk.py`| Audio capture, resampling, Vosk grammar | 11 | **PASS** |
| `test_phase4_camera.py` | V4L2 capture, frame drop detection, dimensions | 3 | **PASS** |
| `test_phase5_yolo_detector.py`| ONNX model loading, letterbox, NMS, class maps | 11 | **PASS** |
| `test_phase6_command_parser.py`| Deterministic phrase parsing, disambiguation | 70 | **PASS** |
| `test_phase7_state_machine.py`| FSM transitions, E-stop latching, obstacle gates, preemption | 23 | **PASS** |
| `test_phase8_runtime_dashboard.py`| API endpoints, auth tokens, concurrency locks, stream disconnect | 15 | **PASS** |

---

## 5. Live Hardware Subsystem Verification (Read-Only)

Live telemetry verified on active daemon PID running in `REAL_HARDWARE` mode:
- **Daemon Lifecycle:** `RUNNING`, `active_fault: OK`.
- **Motor Output States:** Left: `0.00`, Right: `0.00`, `command_state: STOPPED`. GPIO register pins 18, 17, 27, 13, 5, 22 verified driven `lo` (0.00 PWM).
- **Ultrasonic Distance Tracking:** Actively reading live acoustic reflections: valid distance tracking with fail-closed acoustic blind-zone protection.
- **Camera Frame Capture:** Live 640x480 BGR frames actively flowing via MJPEG stream (`/api/camera/stream`) at ~15 FPS with green timestamp banner (`CAM LIVE HH:MM:SS`), deadband guides ($\pm 40\text{ px}$), center crosshair, and YOLO detection box overlay.
- **YOLO Live Inference:** Executed live ONNX inference on camera capture; successfully detected `person` (operator) with confidence 0.46.
- **Microphone & Speech Pipeline:** 16 kHz stream initialized on Logitech C270 USB microphone (`hw:3,0`); Vosk recognizer grammar active.
- **GUI Dashboard:** Verified active and loaded in desktop Chromium browser (`wayland-0`), screenshot verified in `diagnostics/dashboard_live_view.png`. Fixed JS crash on page load; live telemetry now refreshes every 250ms showing real distance (`137.0 cm`), live clock, FSM state explanations, and dynamic speech indicators.
- **Live Voice Interaction:** Recorded real microphone audio in daemon logs; correctly ignored non-actionable ambient speech ("blue a chair", "monitor") and triggered optical centering on spoken command "find person".
- **API Security:** Control endpoints reject unauthenticated access (HTTP 401); unauthenticated Stop / Emergency Stop routes verified functional (HTTP 200).

---

## 6. Hardware Acceptance Matrix

| Acceptance Item | Status | Verification Method & Notes |
| :--- | :--- | :--- |
| **Webcam Optical Capture** | **PASS** | Verified 640x480 @ 15 FPS fresh frames via OpenCV V4L2. |
| **YOLO ONNX Inference** | **PASS** | Verified offline execution on CPU; 26 classes loaded; operator detected. |
| **Microphone Audio Input** | **PASS** | Verified 16 kHz mono PCM capture from Logitech C270 mic. |
| **Offline Vosk Recognition** | **PASS** | Verified speech command grammar parsing without cloud APIs. |
| **GUI Dashboard Launch** | **PASS** | Verified running on port 8000 and displayed in Chromium on Wayland desktop. |
| **Voice Integration Pipeline** | **PASS** | Verified 8/8 end-to-end integration scenarios and real mic speech events. |
| **Ultrasonic Distance Accuracy** | **PASS** | Verified bench calibration at 10, 20, 50, 100 cm; fail-closed underflow active. |
| **Ultrasonic Obstacle Interlock**| **PASS** | Verified software blocks forward motion when $d \le 25.0\text{ cm}$. |
| **Motor Polarity (Bench)** | **PASS** | Verified both left and right wheels spin forward in unison on stand. |
| **Software Emergency Stop** | **PASS** | Verified non-actuating E-stop latches immediately, zeroes outputs, and rejects commands. |
| **Physical Forward Rollout (Floor)**| **NOT VERIFIED** | **Pending Operator-Supervised Test 6A** (0.75 s forward pulse). |
| **Straight-Line Tracking (Floor)** | **NOT VERIFIED** | Pending Test 6B following successful rollout. |
| **Reverse & Turning (Floor)** | **NOT VERIFIED** | Pending Test 6C. |
| **In-Place 360° Calibration** | **NOT VERIFIED** | Pending Test 6D floor measurement mark. |
| **Autonomous Target Approach** | **NOT VERIFIED** | Pending Test 6E physical clearance measurement. |

---

## 7. Immediate Next Action: Supervised Test 6A

> [!IMPORTANT]
> The robot is currently in `IDLE_LISTENING` with motors stopped.
> **Awaiting operator confirmation of floor placement and physical battery cutoff in hand to proceed with Test 6A.**
