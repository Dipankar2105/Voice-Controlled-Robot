# SS_Project_VCR — Forensic Audit Remediation & Current Status

**Status:** SOFTWARE REMEDIATION COMPLETED; PENDING PHYSICAL HARDWARE VERIFICATION ON PI.
Software-side audit repairs (Vosk vocabulary typo removal, microphone overflow diagnostics, worker thread health monitoring, enhanced startup diagnostics, and camera pipeline candidate fallback/recovery) are implemented and unit tested. Physical hardware startup remains blocked by the HC-SR04 ultrasonic sensor returning FAULT readings, requiring physical inspection on the Raspberry Pi.

## Current Snapshot

- **Primary YOLO Detector:** `models/object_detection/best_v2_target.onnx` (26 classes, 169 ms CPU latency on Pi, verified, preserved).
- **Secondary Reference Detector:** `models/object_detection/best_v1.onnx` (preserved).
- **Offline Speech Model:** `models/voice/vosk-model/` (Vosk KaldiRecognizer with clean reviewed vocabulary, typo `"backwad"` removed).
- **Audio Capture:** Logitech C270 USB microphone (`hw:3,0`, index 1, 16 kHz mono 16-bit PCM, buffer overflow and ALSA diagnostics active).
- **Camera Frame Owner:** Logitech C270 USB camera (`/dev/video0` or validated candidate, 640x480 @ 15 FPS, candidate fallback and bounded retry recovery active).
- **Acoustic Ranging:** HC-SR04 ultrasonic sensor (TRIG: 23, ECHO: 24 with pull-down). **Hardware Fault Alert:** Currently returns FAULT readings on startup; fail-safe interlock prevents motor activation.
- **Motor Control & Mapping:** L298N 4WD controller (ENA: 18, IN1: 17, IN2: 27, ENB: 13, IN3: 5, IN4: 22, 100 Hz PWM, verified).
- **Unified Telemetry & Dashboard:** Port 8000 (`http://127.0.0.1:8000/`), live MJPEG camera stream at `/api/camera/stream` with telemetry overlay; polling telemetry at `/api/status`.
- **Security & Concurrency:** In real-hardware mode, mutating commands require `ROBOT_API_TOKEN` (>= 24 chars). STOP and EMERGENCY STOP remain unauthenticated because they strictly remove actuation.

## Latest Automated Software Verification

- **Full Pytest Suite:** All unit and integration test suites passing across configuration, state machine, camera, detector, Vosk speech, command parsing, ultrasonic sensing, and end-to-end workflows.
- `python -m app.main --check-config`: Passed with 0 errors (validates static assets and YAML structure only; does NOT verify physical sensors).
- `git diff --check`: Passed with 0 whitespace errors.

## Subsystem Status: Physical vs Software Tested

| Subsystem / Capability | Status | Evidence / Notes |
|---|---|---|
| **Motor Primitives** (`FORWARD`, `REVERSE`, `TURN_LEFT`, `TURN_RIGHT`) | **Bench/Floor Verified (Prior Session)** | Verified in earlier hardware session on floor at 50% PWM. Safety interlocks remain enforced. |
| **Logitech C270 Camera** | **Software Hardened (Pending Pi Retest)** | Candidate fallback, permission detection, and bounded retry recovery implemented and verified with mocks. Physical USB cable retest pending on Pi. |
| **YOLO ONNX Object Detector** | **Model & Inference Verified** | ONNX Runtime session inference verified (26 classes, 169 ms / 5.9 FPS on Pi CPU, asset checksums verified). |
| **Logitech C270 Microphone** | **Software Hardened (Pending Pi Retest)** | Driver input overflow counting and non-blocking callback diagnostics added. Audio capture was verified in prior session. |
| **Vosk Offline Speech Engine** | **Software Verified & Cleaned** | Misspelled `"backwad"` vocabulary typo removed; valid grammar generation and Kaldi model load verified. |
| **HC-SR04 Ultrasonic Distance** | **HARDWARE FAULT PENDING PHYSICAL REPAIR** | Startup check correctly halts motors because sensor returns FAULT readings on Pi. Wiring/divider inspection required before live start. |
| **Worker Thread Health Monitoring** | **Software Verified** | Real-time thread liveness tracking and fault reporting verified; unhandled worker death flags system as degraded. |
| **End-to-End Navigation & FSM** | **Software Verified** | State transitions, deadband centering, standoff arrival, obstacle aborts verified using mocks. |
| **Full Robot Physical Acceptance** | **PENDING PHYSICAL PI VERIFICATION** | Requires physical HC-SR04 wiring repair, live ultrasonic ping check, and elevated motor pre-check. |

## Checklist for Remaining Physical Verification on Raspberry Pi

1. **HC-SR04 Physical Sensor & Wiring Inspection (Priority 1):**
   - Inspect TRIG wire from Pi BCM 23 (Pin 16) to sensor TRIG pad.
   - Inspect ECHO wire from sensor ECHO pad through the $1\,\text{k}\Omega / 2\,\text{k}\Omega$ voltage divider to Pi BCM 24 (Pin 18).
   - Verify sensor 5V VCC and ground connections.
   - Verify raw acoustic pings directly using `python scripts/live_ultrasonic.py`.
2. **Logitech C270 USB Connection (Priority 2):**
   - Reseat USB cable in Pi USB port; test with `python scripts/diagnose_camera.py`.
3. **Supervised Commissioning (Priority 3):**
   - Elevate robot chassis so all four wheels rotate freely in the air.
   - Run `python scripts/commission_hardware.py --interactive` with supervisor present.
