# HARDWARE ACCEPTANCE & CALIBRATION GUIDE
## SS_PROJECT_VCR_FRESH — Clean-Room Embedded Linux Robot

> [!CAUTION]
> **MANDATORY SAFETY PROTOCOL BEFORE ANY PHYSICAL ACTUATION:**
> 1. Elevate the robot chassis on a stable stand so **all four wheels spin freely in the air**.
> 2. Confirm a **physical battery cutoff switch** or battery disconnect plug is within immediate arm's reach.
> 3. Verify the **HC-SR04 ECHO voltage divider** (strictly 3.3V maximum input to Raspberry Pi GPIO 24).
> 4. Never claim that physical movement, obstacle stopping, rotation coverage, or target stopping distance has been validated without supervised testing on the physical robot.

---

## 1. GPIO Pin Mapping Reference

| Subsystem | Signal Function | BCM GPIO Pin | Header Physical Pin | Electrical Specification |
|---|---|---|---|---|
| **Left Motors (Channel A)** | ENA (PWM Speed) | **GPIO 18** | Pin 12 | 3.3V Logic PWM (100 Hz default) |
| | IN1 (Direction 1) | **GPIO 17** | Pin 11 | 3.3V Logic OUT |
| | IN2 (Direction 2) | **GPIO 27** | Pin 13 | 3.3V Logic OUT |
| **Right Motors (Channel B)** | ENB (PWM Speed) | **GPIO 13** | Pin 33 | 3.3V Logic PWM (100 Hz default) |
| | IN3 (Direction 3) | **GPIO 22** | Pin 15 | 3.3V Logic OUT |
| | IN4 (Direction 4) | **GPIO 5** | Pin 29 | 3.3V Logic OUT |
| **HC-SR04 Ultrasonic** | TRIG (Trigger Pulse) | **GPIO 23** | Pin 16 | 3.3V Logic OUT (10 µs pulse) |
| | ECHO (Echo Return) | **GPIO 24** | Pin 18 | **Must be level-shifted to 3.3V** via divider |
| **Power & Ground** | Common Ground | **GND** | Pin 6, 14, 20, 39 | Common Ground between Pi, L298N, and Battery (-) |

### HC-SR04 ECHO Voltage Divider Circuit
The HC-SR04 is powered by +5V and its ECHO pin outputs a 5.0V TTL pulse. Directly connecting this 5.0V pulse to Raspberry Pi GPIO 24 will damage the SoC input circuitry.

```
HC-SR04 ECHO (5V OUT) ----[ 1.0 kΩ Resistor ]----+----[ Raspberry Pi GPIO 24 (IN) ]
                                                |
                                        [ 2.0 kΩ Resistor ]
                                                |
                                             [ GND ]
```
- Calculated Voltage at GPIO 24: $5.0\,\text{V} \times \frac{2.0\,\text{k}\Omega}{1.0\,\text{k}\Omega + 2.0\,\text{k}\Omega} = 3.33\,\text{V}$ (Safe for Pi 3.3V logic).

---

## 2. Step-by-Step Hardware Acceptance Procedures

### Test 1: Camera Capture & Freshness
- **Objective:** Verify Logitech C270 USB webcam opens, drains V4L2 buffers continuously, and delivers 640x480 BGR frames with staleness $< 0.50$ seconds.
- **Execution:**
  ```bash
  python scripts/diagnose_camera.py
  ```
- **Pass Criteria:**
  - Device `/dev/video0` opens cleanly.
  - Resolved dimensions: exactly 640 width by 480 height.
  - Frame rate: ~15 FPS.
  - Frame freshness age: $< 0.10$ seconds under continuous capture.

---

### Test 2: Webcam Microphone & Offline Vosk Speech Recognition
- **Objective:** Enumerate USB audio input devices, verify Logitech C270 microphone capture at 16 kHz mono PCM, and decode voice commands offline without cloud APIs.
- **Execution:**
  ```bash
  python scripts/diagnose_microphone.py
  ```
- **Pass Criteria:**
  - Audio device index identified as "C270" or "USB Audio".
  - Audio stream captures 16-bit 16000 Hz mono PCM chunks without buffer underflows.
  - Speaking "find backpack" or "move forward" returns correct string transcripts.

---

### Test 3: Ultrasonic Readings at Known Physical Distances
- **Objective:** Validate acoustic flight time math and sensor accuracy across operational ranges.
- **Execution:**
  ```bash
  python scripts/run_acceptance_tests.py --test ultrasonic
  ```
- **Procedure:**
  1. Place a vertical board at 10.0 cm from transducer face. Verify reading is $10 \pm 1.5$ cm.
  2. Place board at 20.0 cm. Verify reading is $20 \pm 2.0$ cm.
  3. Place board at 50.0 cm. Verify reading is $50 \pm 3.0$ cm.
  4. Test Transducer Blind Zone: Place object at $< 2.0$ cm. Verify system evaluates reading as `CRITICAL` fail-closed.
- **Pass Criteria:** All samples within acceptable tolerances; blind zone evaluates fail-closed.

---

### Test 4: Autonomous Target Stop Band (30 cm provisional)
- **Objective:** Verify that autonomous arrival requires a fresh centred target detection plus physical ultrasonic distance; bounding-box area alone must never end a mission.
- **Procedure:**
  1. Confirm `target_stop_distance_cm: 30.0` and `obstacle_blocked_cm: 25.0`.
  2. Supervise an approach to a centred target at approximately 35 cm.
  3. Verify that a valid centred target at 25–30 cm causes `TARGET_REACHED` with motor speeds at zero.
  4. Verify that any return at or below 25 cm causes `BLOCKED`, not `TARGET_REACHED`.
  5. At 28 cm, hide or move the target off-centre; verify no success claim or blind in-place turn occurs.
- **Pass Criteria:** Arrival only in the 25–30 cm warning band with the requested target still visible and centred. The 30 cm value is provisional; calibrate physical stopping distance before floor operation and never set target-stop inside the blocked zone.

---

### Test 5: Motor Direction & PWM with Wheels Lifted
- **Objective:** Verify dual-channel differential drive polarity and independent channel control without ground contact.
- **Execution:**
  ```bash
  python scripts/run_acceptance_tests.py --test motors
  ```
- **Procedure:**
  1. Elevate robot chassis. Wheels MUST be off the ground.
  2. Command FORWARD at speed 0.40 for 1.0 second. Verify all four wheels rotate forward.
  3. Command REVERSE at speed 0.40 for 1.0 second. Verify all four wheels rotate in reverse.
  4. Command TURN LEFT. Verify left wheels reverse and right wheels move forward.
  5. Command TURN RIGHT. Verify right wheels reverse and left wheels move forward.
- **Pass Criteria:** No polarity inversion; PWM switching operates smoothly at 100 Hz.

---

### Test 6: Forward Obstacle Stopping Interlock
- **Objective:** Ensure forward motor motion is strictly gated by the ultrasonic safety supervisor.
- **Execution:**
  ```bash
  python scripts/run_acceptance_tests.py --test obstacle
  ```
- **Procedure:**
  1. Position an obstacle 15.0 cm in front of the robot.
  2. Attempt a manual forward command: `"move forward"`.
- **Pass Criteria:** The state machine transitions to `BLOCKED`, motors remain at 0.0 speed, and log records: `Forward motion refused: obstacle at 15.0 cm (BLOCKED)`.

---

### Test 7: In-Place Search-Sector Rotation Duration Calibration
- **Objective:** Calibrate the precise turn duration required for a full 360° rotation on the physical floor surface.
- **Execution:**
  ```bash
  python scripts/run_acceptance_tests.py --test rotation
  ```
- **Procedure:**
  1. Place robot on the operational surface (tile, wood, or carpet).
  2. Run the rotation calibration tool and time exactly one complete 360° turn at speed 0.40 (`TURN_SPEED`).
  3. Enter the measured duration into `config/default_config.yaml`:
     ```yaml
     motion:
       calibrated_360_duration_sec: 11.45  # Example measured value
     ```
- **Pass Criteria:** Recorded calibration value divides cleanly into 8 bounded $45^\circ$ search sectors.

---

### Test 8: Motor Deadband & Stall Speed Threshold (`MIN_SPEED`)
- **Objective:** Determine the minimum PWM duty cycle where motors overcome internal gearbox friction on the ground.
- **Procedure:**
  1. Increment PWM duty cycle from 0.10 to 0.40 in steps of 0.05.
  2. Note the threshold where all wheels reliably start spinning without stalling.
  3. Set `MIN_SPEED` in `config/default_config.yaml` to $\text{threshold} + 0.05$ safety margin.
- **Pass Criteria:** Motors never stall into high-current static states at configured speeds.

---

### Test 9: Optical Centering Deadband Test
- **Objective:** Confirm bounding box visual alignment and turn pulse responsiveness.
- **Procedure:**
  1. Place target object at camera horizontal center $x = 320$. Verify status reports `CENTERED`.
  2. Move object to $x = 220$ ($< 320 - 40$). Verify state machine commands bounded right-wheel clockwise pulse to re-center.
  3. Move object to $x = 420$ ($> 320 + 40$). Verify state machine commands counter-clockwise pulse to re-center.
- **Pass Criteria:** Target re-centers within $\pm 40$ pixels without oscillating.

---

### Test 10: Target Approach & Ultrasonic Stopping Distance
- **Objective:** Verify forward approach pulses are gated by distance and halt at configured stop distance.
- **Procedure:**
  1. Place target object at 80 cm distance in clear line of sight.
  2. Run search mission: `find cup`.
  3. Verify robot advances in bounded forward pulses (speed 0.40, pulse duration 0.15s, settling delay 0.08s).
  4. Verify robot transitions to `TARGET_REACHED` when ultrasonic reading drops below `target_stop_distance_cm` (30.0 cm provisional).
- **Pass Criteria:** The robot stops in the configured warning-band approach window, and the measured sensor-to-bumper offset plus final bumper clearance are recorded. Adjust the provisional 30 cm threshold only after measurement, and keep it strictly above the 25 cm blocked threshold. Never relies on bounding-box area for stopping.

---

### Test 11: End-to-End Voice Mission & Emergency Stop Latching
- **Objective:** Complete autonomous voice-driven object search mission with emergency stop interruption.
- **Procedure:**
  1. Start launcher: `./start_robot.sh`.
  2. Speak: `"find backpack"`. Verify robot searches in sectors, centers on backpack, and approaches.
  3. While robot is approaching, speak loudly or click: `"emergency stop"`.
  4. Verify all motor speeds drop to 0.00 instantly, motors software disabled, and FSM latches `EMERGENCY_STOP`.
  5. Attempt `"move forward"`. Verify command is rejected with `Emergency stop is LATCHED`.
  6. Click or post `"clear e-stop"`. Verify state returns to `IDLE_LISTENING`.
- **Pass Criteria:** Zero actuation during latched emergency stop; safe state recovery.

---

## 3. Unverified Physical Behaviors & Operating Limitations

Until physical bench tests and supervised floor runs are completed on the physical Raspberry Pi hardware, the following physical properties remain unverified:
1. **Floor Surface Friction Variance:** Carpet vs tile has dramatically different friction coefficients, requiring distinct `calibrated_360_duration_sec` tuning.
2. **Battery Voltage Sag:** Under high motor load, motor battery voltage drop may brownout the Raspberry Pi if external supplies do not share an isolated regulator.
3. **Ambient Acoustic Noise:** In high-noise environments, Vosk offline acoustic model confidence may drop. Maintain a distance of $< 1.5$ meters from the C270 microphone when issuing spoken commands.
4. **Lighting Conditions:** YOLO ONNX detector confidence depends on adequate indoor lighting ($> 150$ lux). Very dark rooms will degrade visual object detection.
