# SS_PROJECT_VCR_FRESH — Autonomous Voice-Controlled Mobile Robot

A clean-room, zero-legacy implementation of an embedded voice-controlled autonomous mobile robot built for the **Raspberry Pi 4 Model B**. Integrates offline voice recognition (Vosk), deep learning object detection (YOLO ONNX), 4WD motor actuation (L298N), forward acoustic safety ranging (HC-SR04), and a real-time web telemetry dashboard into one authoritative runtime lifecycle.

---

## 1. System Overview & Architecture

```
               +--------------------------------------------------+
               |         Local Web Dashboard & REST API          |
               |     (Read-Only Telemetry / Live MJPEG Stream)    |
               +------------------------+-------------------------+
                                        |
               +------------------------v-------------------------+
               |      Authoritative RobotRuntime Lifecycle        |
               +------------------------+-------------------------+
                                        |
       +--------------------------------+--------------------------------+
       |                                |                                |
+------v------+                 +-------v-------+                +-------v-------+
|  Command    |                 |  Robot FSM    |                | Multi-Thread  |
|  Parser     |                 |  Navigation   |                |  Background   |
| (Whole-Token|                 |  (13 States)  |                |   Workers     |
+------+------+                 +-------+-------+                +-------+-------+
       |                                |                                |
       |                                |                                |
+------v------+         +---------------+---------------+        +-------v-------+
| Offline Vosk|         |               |               |        | Shared Camera |
| Speech Engine         |               |               |        | Frame Owner   |
| (16 kHz Mono)   +-----v-----+   +-----v-----+   +-----v-----+  | (Single Inst) |
+-------------+   |   L298N   |   |  HC-SR04  |   | YOLO ONNX |  +---------------+
                  | 4WD Motors|   | Ultrasonic|   | Detector  |
                  +-----------+   +-----------+   +-----------+
```

### Key Hardware Specifications
- **Host Compute:** Raspberry Pi 4 Model B (Broadcom BCM2711, Quad-core Cortex-A72 @ 1.5/1.8 GHz, ARMv8 64-bit).
- **Operating System:** Raspberry Pi OS 64-bit (Debian 12 Bookworm recommended).
- **Vision:** Logitech C270 USB Webcam (640x480 @ 15 FPS) + YOLO ONNX (`best_v2_target.onnx`).
- **Microphone:** Logitech C270 built-in USB microphone (16 kHz mono PCM capture).
- **Offline Speech Engine:** Vosk Small English Kaldi model (`vosk-model-small-en-us-0.15`).
- **Actuation:** L298N Dual H-Bridge driving 4 DC motors via RPi.GPIO software PWM (100 Hz).
- **Proximity Safety:** HC-SR04 Ultrasonic Distance Sensor with mandatory resistive voltage divider on ECHO (GPIO 24).

---

## 2. Clean Installation Guide (Raspberry Pi OS)

### A. System Package Installation (Debian / Pi OS)
On the target Raspberry Pi, update apt repositories and install required audio/video and build libraries:
```bash
sudo apt-get update
sudo apt-get install -y --no-install-recommends \
    python3-venv \
    python3-pip \
    libportaudio2 \
    libasound2-dev \
    libgl1 \
    libglib2.0-0 \
    v4l-utils
```

### B. User Permissions (GPIO, Video & Audio Access)
Ensure the robot user account belongs to the required device groups:
```bash
sudo usermod -aG gpio,video,audio,dialout $USER
# Log out and log back in for group changes to take effect
```

### C. Python Virtual Environment Setup
Create an isolated virtual environment to prevent modifying system packages:
```bash
cd SS_Project_VCR_FRESH
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
```

---

## 3. One-Command Startup & Launcher

### Single-Command Start (Linux / Raspberry Pi)
```bash
chmod +x start_robot.sh
./start_robot.sh
```

### Single-Command Start (Windows Simulation)
```cmd
start_robot.bat
```

### Direct CLI Commands
```bash
# 1. Non-invasive preflight check (validates assets and config without hardware actuation)
python -m app.main --check-config

# 2. Launch authoritative runtime server on default port 8000
python -m app.main --serve

# 3. Hardware mode requires an out-of-band operator token (24+ characters).
export ROBOT_API_TOKEN="$(python3 -c 'import secrets; print(secrets.token_urlsafe(32))')"
python -m app.main --serve --host 0.0.0.0 --port 8080 --mode REAL_HARDWARE
```

In real-hardware mode, the dashboard prompts for `ROBOT_API_TOKEN`. It is never embedded in the page. Command dispatch and E-stop clearing require the token; STOP and EMERGENCY STOP remain unauthenticated because they only remove actuation. The dashboard uses same-origin requests (wildcard CORS is disabled). Keep the robot on a trusted network and retain a physical power cutoff.

- **Local Telemetry Dashboard:** `http://localhost:8000/`
- **Read-Only Telemetry API:** `http://localhost:8000/api/status`
- **Shared Optical Video Stream:** `http://localhost:8000/api/camera/stream`

---

## 4. Supported Voice & Text Commands

### Movement & Teleoperation
| Command Voice Phrase | Action | Intent | Safety Invariant |
|---|---|---|---|
| `"move forward"`, `"go forward"`, `"forward"` | `MOVE_FORWARD` | `MOVEMENT` | Gated by ultrasonic sensor ($> 25$ cm required) |
| `"move backward"`, `"go back"`, `"backward"` | `MOVE_BACKWARD` | `MOVEMENT` | Bounded pulse (no side/rear ultrasonic coverage) |
| `"turn left"`, `"rotate left"`, `"left"` | `TURN_LEFT` | `MOVEMENT` | Bounded in-place differential pulse |
| `"turn right"`, `"rotate right"`, `"right"` | `TURN_RIGHT` | `MOVEMENT` | Bounded in-place differential pulse |
| `"stop"`, `"halt"`, `"wait"` | `STOP` | `CONTROL` | Immediate active motor braking |
| `"emergency stop"`, `"e-stop"` | `EMERGENCY_STOP` | `CONTROL` | Latched software disable; halts all actuation |

### Autonomous Object Search
| Command Phrase | Action | Target Class | Supported Model Aliases |
|---|---|---|---|
| `"find backpack"` | `SEARCH_TARGET` | `backpack` | `bag`, `pack`, `schoolbag` |
| `"find cup"` | `SEARCH_TARGET` | `cup` | `cup`, `mug` |
| `"find bottle"` | `SEARCH_TARGET` | `bottle` | `water bottle` |
| `"find scissors"` | `SEARCH_TARGET` | `scissors` | `scissor`, `shears`, `pair of scissors` |
| `"find red target"` | `SEARCH_TARGET` | `RED_TARGET` | `red object`, `red` |
| `"find blue target"` | `SEARCH_TARGET` | `BLUE_TARGET` | `blue object`, `blue` |
| `"find phone"` | `SEARCH_TARGET` | `cell_phone` | `mobile phone`, `cell phone` |
| `"find trash can"` | `SEARCH_TARGET` | `trash_can` | `garbage can`, `trash`, `bin` |
| `"find chair"` | `SEARCH_TARGET` | `chair` | `chair`, `seat` |
| `"find laptop"` | `SEARCH_TARGET` | `laptop` | `laptop`, `notebook` |

### Control & Utility
| Command Phrase | Action | Description |
|---|---|---|
| `"cancel search"`, `"stop scanning"` | `CANCEL_SEARCH` | Safely aborts active search or approach mission |
| `"resume listening"` | `RESUME_LISTENING` | Resets terminal mission states and resumes listening |
| `"status"`, `"system status"` | `GET_STATUS` | Reports system health snapshot |
| `"speed slow"` / `"speed normal"` / `"speed fast"` | `SET_SPEED` | Adjusts manual movement speed within $[0.20, 0.80]$ bounds |

---

## 5. Wiring Reference & Mandatory Safety Protocols

```
               Raspberry Pi 4 Physical Header (BCM Pinout)
           3.3V Power [01] [02] 5.0V Power
                      [03] [04] 5.0V Power
                      [05] [06] Ground (Common GND)
                      [07] [08]
               Ground [09] [10]
        IN1 (GPIO 17) [11] [12] ENA (GPIO 18) [PWM Ch A]
        IN2 (GPIO 27) [13] [14] Ground
        IN3 (GPIO 22) [15] [16] TRIG (GPIO 23) [Ultrasonic]
             3.3V Out [17] [18] ECHO (GPIO 24) [Via Divider]
                      [19] [20] Ground
                      [21] [22]
                      [23] [24]
               Ground [25] [26]
                      [27] [28]
        IN4 (GPIO 05) [29] [30] Ground
                      [31] [32]
        ENB (GPIO 13) [33] [34] Ground [PWM Ch B]
                      [35] [36]
                      [37] [38]
               Ground [39] [40]
```

> [!WARNING]
> **HC-SR04 ECHO VOLTAGE DIVIDER IS MANDATORY:**
> The HC-SR04 outputs 5.0V on ECHO. Raspberry Pi GPIOs are strictly **3.3V max**. You must place a voltage divider ($1\,\text{k}\Omega$ series, $2\,\text{k}\Omega$ shunt) between ECHO and GPIO 24.
>
> **COMMON GROUND IS MANDATORY:**
> The external motor battery ground ($-$), L298N `GND` terminal, and Raspberry Pi header `GND` pin must be bonded together.

---

## 6. Hardware Commissioning & Acceptance Testing

Before running the robot on the floor, execute the interactive acceptance suite:
```bash
# Elevate chassis so wheels rotate in the air!
python scripts/run_acceptance_tests.py --all
```
See detailed step-by-step procedures in [docs/HARDWARE_ACCEPTANCE_GUIDE.md](file:///c:/Users/Dipankar%20Pimple/Downloads/SS_VCR_FINAL_CLEAN/SS_Project_VCR_FRESH/docs/HARDWARE_ACCEPTANCE_GUIDE.md).

---

## 7. Troubleshooting & Fault Codes

| Fault Code | Cause | Recommended Remediation |
|---|---|---|
| `FAULT_CONFIG_INVALID` | Duplicate GPIO assignment, invalid speed bound, or non-monotonic threshold | Check `config/default_config.yaml` against pinout table |
| `FAULT_ASSET_MISSING` | Preserved ONNX model file or Vosk directory missing | Run `python -m pytest tests/unit/test_asset_integrity.py` |
| `FAULT_HARDWARE_INIT_FAILED` | GPIO pin setup failed or `REAL_HARDWARE` mode requested on non-Pi | Verify running on Raspberry Pi ARM Linux or set `mode: SIMULATION` |
| `FAULT_SENSOR_TIMEOUT` | HC-SR04 sensor unresponsive or wiring disconnected | Check TRIG/ECHO wiring, 5V power supply, and ground connection |
| `FAULT_CAMERA_FRAME_DROP` | Camera disconnected or USB bandwidth saturated | Reconnect Logitech C270 to USB 3.0 (blue port); run `diagnose_camera.py` |
| `FAULT_EMERGENCY_STOP_LATCHED` | Emergency stop was triggered via voice or web dashboard | Press "Clear E-Stop" button in dashboard or post to `/api/clear-estop` |

---

## 8. Graceful Shutdown

To shut down cleanly:
1. Press `Ctrl+C` in the terminal running `./start_robot.sh`.
2. The runtime signal handler catches `SIGINT`, cancels any active mission, stops motor actuation, releases the camera backend, closes the audio stream, cleans up GPIO pins, and exits with code 0.
