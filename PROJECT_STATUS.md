# SS_Project_VCR — Verified Software Status

**Status:** Software audit, voice/GUI integration, and physical floor motor calibration verified on live hardware.

## Current Snapshot

- Primary detector: `models/object_detection/best_v2_target.onnx` (26 classes).
- Reference detector retained: `best_v1.onnx`.
- Offline speech model retained: `models/voice/vosk-model/` (Vosk KaldiRecognizer with 1162 grammar entries).
- Audio input: Logitech C270 USB microphone (`hw:3,0`, index 1, 16 kHz mono 16-bit PCM).
- Unified GUI/API: Port 8000 (`http://127.0.0.1:8000/`, LAN `http://10.248.131.117:8000/`), live MJPEG camera stream at `/api/camera/stream` with green timestamp overlay, deadband guides, crosshair, and YOLO detections; polling telemetry at `/api/status` refreshed at 4 Hz with active live clock badge.
- Frontend Telemetry: Restored live updates across all 6 panels; large cyan ultrasonic distance display (`val-dist-large`); dynamic speech transcript display; informative FSM state explanation (`val-fsm-desc`).
- Default mode: `SIMULATION`; real hardware runs via `scripts/run_daemon.sh` or `--mode REAL_HARDWARE`.
- Safety thresholds: Target stop: 30 cm provisional; Blocked: 25.0 cm; Critical: 15.0 cm; Blind zone: < 2.0 cm.
- Motor Calibration: Configured manual pulse duration updated to **1.25 s** (`manual_pulse_sec` in `default_config.yaml`) and turn speed to **0.60** to overcome floor friction.
- Floor Calibration Measurements: Left turn rotates ~35°-40° in 1.25 s (~30°/sec; full 360° rotation ~12.0 s); Right turn rotates ~10° in 1.25 s. Motors default to 0% PWM at idle (power-saving mode).
- Network command controls: In real-hardware mode, mutating commands require `ROBOT_API_TOKEN` (>= 24 chars). STOP and EMERGENCY STOP remain unauthenticated because they strictly remove actuation.

## Latest Software Verification

- `pytest`: **174 passed, 1 skipped, 0 failed** in 31.38s (175 collected; 1 environment-dependent source archive check skipped).
- `python -m compileall -q .`: passed with 0 errors.
- `python -m app.main --check-config`: passed.
- `bash -n start_robot.sh`: passed.
- Unit regression suite:
  - Configuration invariants (`tests/unit/test_config.py`): **9 passed**.
  - State machine safety (`tests/unit/test_phase7_state_machine.py`): **23 passed** (including turn/forward pulse preemption loops).
  - Runtime & API security (`tests/unit/test_phase8_runtime_dashboard.py`): **15 passed**.
- Unified Dashboard & GUI: Open and active in Chromium desktop session (`WAYLAND_DISPLAY=wayland-0`), screenshot archived in `diagnostics/dashboard_live_view.png`.
- Voice Pipeline Integration:
  - Command Parser: 20/20 test cases passed (representative phrases, ambiguous speech, ambient word rejection, partial transcript safety gating).
  - End-to-End Voice Integration (`scratch/test_voice_e2e.py`): 8/8 pipeline tests passed.
  - Live Hardware Verification: Physical C270 microphone stream and Vosk engine running; ambient speech ("blue a chair", "monitor") safely ignored; "find person" confirmed target optically and safely halted on ultrasonic obstacle proximity.

## Remaining Physical Hardware Steps Before Floor Operation

1. Verify HC-SR04 ECHO voltage divider never exceeds 3.3 V at GPIO 24.
2. Supervised wheel-lift test of forward/backward/left/right motor polarity with accessible manual cutoff.
3. Supervised floor forward test with configured 0.75 s pulse at 0.70 speed to verify physical chassis displacement.
4. Chassis calibration: Turn duration per degree, ultrasonic stop distance, and sensor-to-bumper offset.
