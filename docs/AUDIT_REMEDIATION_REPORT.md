# Source Audit and Remediation Report

## Scope and evidence

The audit used the uploaded `SS_Project_VCR_FRESH.zip` as the code snapshot. The supplied Phase 1–9 reports were treated as claims to verify, not proof that the archive itself had been tested. The uploaded archive was read-only and remains untouched: **62,927,377 bytes**, SHA-256 `f9275383b276a6fd6de7e0b0c32bd3fe352167e5fbd5ead03f2c2fc2111c63b4`. All code fixes are in a separate extracted workspace and exported to a new ZIP.

The report's earlier claim of 152 passing tests did not match the first test run from the uploaded snapshot: collection initially failed because optional native dependencies were imported at module import time. The audit added regression tests and made imports dependency-safe for simulation and host-side testing. The final count is recorded in `PROJECT_STATUS.md`.

## Findings and fixes

### 1. Autonomous target arrival could override obstacle safety

**Finding:** The approach state checked the target-stop threshold before checking `BLOCKED`/`CRITICAL`, and could declare arrival based on distance alone without confirming the requested object remained visible and centred. The default target-stop value was 5 cm, inside the blocked/critical region.

**Fix:** The default is now 30 cm provisionally. Configuration validation enforces `critical < blocked < target_stop < warning`. Obstacle checks take priority; arrival requires a valid fresh ultrasonic reading plus a fresh, centred detection of the requested target. Large bounding-box area alone cannot complete a mission. Missing/invalid sensors fail closed. Search/centering turns are refused when the forward sensor reports the stop zone, because side clearance is unknown.

**Calibration caveat:** 30 cm is not a measured chassis stopping distance. It must be validated physically, including the ultrasonic sensor's offset from the front bumper.

### 2. Real-hardware mode had no complete driver construction path

**Finding:** The runtime's real-hardware initialization path expected injected drivers, while the normal application entry point constructed the runtime without injecting them. The advertised real-hardware launch path therefore could not initialize the robot from configuration alone.

**Fix:** Real mode now constructs the RPi GPIO adapter, L298N motor controller, HC-SR04 sensor, camera owner, microphone source, Vosk recognizer, and YOLO detector when missing. It refuses known mock implementations in real mode and does not silently substitute simulation hardware. Model loading, sensor readiness, audio startup, and first valid camera frame are checked before motors are enabled. Startup failures stop/disable motors and release partially initialized resources.

**Limitation:** This factory path was not executed on physical Raspberry Pi hardware in this environment.

### 3. Ultrasonic sensor initialization could deadlock

**Finding:** The sensor used a non-reentrant lock while `start()`/`get_reading()` could call `initialize()` under the same lock.

**Fix:** The lock is re-entrant and regression tests exercise an uninitialized read. NaN, infinity, zero, negative, stale, and invalid readings are rejected instead of being treated as clear.

### 4. YOLO preprocessing and suppression were unsafe/inconsistent

**Finding:** Preprocessing directly stretched a 640×480 frame to 320×320, distorting aspect ratio; NMS was class-agnostic; non-finite model outputs could crash coordinate conversion.

**Fix:** The detector now letterboxes while preserving aspect ratio, inversely maps boxes back to source coordinates, applies class-aware NMS, discards non-finite predictions, and validates input/output tensor shapes and class metadata before declaring readiness.

**Limitation:** The actual ONNX Runtime inference tests were skipped here because `onnxruntime` is unavailable. Model asset checksums and mocked preprocessing/NMS tests passed; this is not proof of Pi inference speed or detection accuracy.

### 5. Vosk grammar did not include complete object-command phrases

**Finding:** A grammar containing independent words such as `find` and `bottle` did not explicitly guarantee the phrase `find bottle` was accepted under grammar-restricted decoding.

**Fix:** The generated grammar now includes complete `find`/`locate`/`search for`/`look for` phrases and common articles for supported classes and aliases. Documented aliases `seat`, `notebook`, `mug`, and stop phrase `wait` now agree with the command parser.

**Limitation:** Native Vosk loading and microphone capture were not executed here because `vosk` and `sounddevice` are unavailable.

### 6. Microphone selection could silently choose the wrong input

**Finding:** If a Logitech C270 could not be identified, the audio source silently fell back to the system-default input.

**Fix:** The selector prefers explicit Logitech/C270 matches; a generic USB input is accepted only when unambiguous. Otherwise startup fails with a diagnostic rather than using an unknown microphone.

### 7. Telemetry was not strictly read-only and could report stale sensor data as healthy

**Finding:** The health path could call the ultrasonic sensor while responding to a status request. A last-good distance could remain cached after sampling errors.

**Fix:** Health/status reads use cached data only. Sensor exceptions invalidate the cache, freshness is checked, and absent/stale readings are reported as `FAULT` rather than `CLEAR`. Forward movement is refused without a valid sensor reading.

### 8. Background exceptions and emergency-stop races

**Finding:** Mission-worker exceptions were logged without necessarily stopping/disable motors. A manual pulse could finish after an E-stop and overwrite the latched state with `IDLE_LISTENING`. E-stop clearing did not require a valid, unblocked sensor reading.

**Fix:** Mission-worker exceptions stop and disable motors and move the FSM to `FAULT`; manual pulse completion no longer overwrites a preempting stop state; E-stop clearing is refused until the sensor is valid and the blocked zone is clear. The API now reports a failed clear honestly.

### 9. Network motor-control endpoints were unauthenticated

**Finding:** The dashboard was served on a network interface, with wildcard CORS and no token protection for movement commands.

**Fix:** Wildcard CORS was removed. Real-hardware mode requires a `ROBOT_API_TOKEN` of at least 24 characters before runtime/hardware initialization. The dashboard prompts for the token; it is sent in a header and is not embedded in HTML. Command dispatch and E-stop clearing require the token. STOP and EMERGENCY STOP remain available without a token because they only remove actuation.

### 10. Launcher preflight and mode override defects

**Finding:** The shell launcher captured the inverted status of a failed preflight, and the frozen configuration object was mutated directly for `--mode` overrides.

**Fix:** The shell launcher now preserves the actual preflight exit code; the Windows launcher uses a dedicated failure label; mode overrides use `dataclasses.replace` so `--mode REAL_HARDWARE` is applied correctly.

### 11. Camera readiness was reported too early

**Finding:** The camera owner marked itself operational as soon as the capture thread started, before a valid frame existed, and backend read exceptions could kill the worker.

**Fix:** Operational status now begins only after a valid frame; backend read exceptions are caught and counted; runtime startup waits for a first frame before enabling motors.

## Test evidence

The latest suite run in this audit environment reported **161 passed, 10 skipped**. Skips were caused by unavailable native `onnxruntime`, `vosk`, and `sounddevice` dependencies and the original ZIP not being included beside the extracted workspace. The following checks also passed:

- Python bytecode compilation (`python -m compileall -q .`)
- Static configuration/model preflight (`python -m app.main --check-config`)
- Shell syntax validation (`bash -n start_robot.sh`)
- Simulation server smoke test with intentional timeout and shutdown
- Preserved model asset checksums

## Remaining limitations / release gate

- No physical motor actuation, camera capture, microphone capture, ultrasonic measurement, or Pi deployment was performed during this audit.
- The single forward HC-SR04 cannot establish side or rear clearance. Manual reverse/turn commands remain bounded but require supervision; use additional sensors if autonomous side/rear avoidance is required.
- Wheel direction, PWM deadband, stopping distance, 360-degree turn duration, and sensor-to-bumper offset remain uncalibrated.
- Run the complete suite with all dependencies installed on the actual Raspberry Pi, then complete `HARDWARE_ACCEPTANCE_GUIDE.md` with wheels elevated before any floor trial.
