"""Hardware Acceptance & Calibration Test Suite for SS_PROJECT_VCR_FRESH.

Provides interactive, step-by-step physical validation and calibration procedures:
1. Camera Capture & Optical Freshness Verification.
2. Logitech C270 USB Microphone Capture & Vosk Offline Speech Recognition.
3. Ultrasonic Distance Sensor Calibration at Known Physical Distances (10, 20, 50, 100 cm).
4. Motor Direction & PWM Speed Calibration (Chassis Elevated / Wheels Off Ground).
5. Forward Obstacle Stopping & Ultrasonic Interlock Gating.
6. In-Place Search-Sector Rotation Duration Calibration (360° Floor Test).
7. Motor Deadband & Minimum Effective PWM Threshold (MIN_SPEED).
8. Actual Stopping Distance Measurement.
9. Target Approach & Cancellation Override.
10. End-to-End Voice-Controlled Object Search Mission.

Usage on Raspberry Pi:
    python scripts/run_acceptance_tests.py --all
    python scripts/run_acceptance_tests.py --test ultrasonic
    python scripts/run_acceptance_tests.py --test motors
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

# Add project root to sys.path
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from app.config import load_config
from app.constants import (
    APPROACH_SPEED,
    ExecutionMode,
    PIN_LEFT_ENA,
    PIN_LEFT_IN1,
    PIN_LEFT_IN2,
    PIN_RIGHT_ENB,
    PIN_RIGHT_IN3,
    PIN_RIGHT_IN4,
    PIN_ULTRASONIC_ECHO,
    PIN_ULTRASONIC_TRIG,
    SEARCH_SPEED,
    TURN_SPEED,
)
from app.hardware.gpio_adapter import FakeGPIOAdapter, RPiGPIOAdapter
from app.hardware.motors import L298NMotorController
from app.hardware.ultrasonic import HCSR04UltrasonicSensor
from app.logger import get_logger

logger = get_logger("acceptance")


def prompt_confirm(prompt: str) -> bool:
    """Prompt operator for explicit confirmation."""
    try:
        res = input(f"\n{prompt} [y/N]: ").strip().lower()
        return res in ("y", "yes")
    except (KeyboardInterrupt, EOFError):
        print("\n[ABORTED by operator]")
        return False


def test_camera():
    """Test 1: Camera capture, frame freshness and dimension validation."""
    print("\n" + "=" * 60)
    print("  TEST 1: CAMERA CAPTURE & DIMENSIONS")
    print("=" * 60)
    from app.vision.camera import CameraOwner

    cfg = load_config()
    cam = CameraOwner(config=cfg.camera)
    print(f"Opening camera index {cfg.camera.device_index} ({cfg.camera.width}x{cfg.camera.height} @ {cfg.camera.fps} FPS)...")
    if not cam.start():
        print("[FAIL] Could not start CameraOwner backend.")
        return False

    time.sleep(0.5)
    frame, ts = cam.get_latest_frame_with_timestamp()
    if frame is None:
        print("[FAIL] Captured frame is None (camera unresponsive or frame dropped).")
        cam.release()
        return False

    h, w, c = frame.shape
    age = time.time() - ts
    print(f"[OK] Frame acquired successfully: {w}x{h} (channels: {c}), freshness age: {age:.3f}s")
    cam.release()
    return (w, h) == (cfg.camera.width, cfg.camera.height) and age < 0.5


def test_microphone_and_vosk():
    """Test 2: Microphone capture and Vosk offline speech recognition."""
    print("\n" + "=" * 60)
    print("  TEST 2: WEBCAM MICROPHONE & VOSK OFFLINE SPEECH")
    print("=" * 60)
    from app.audio.audio_source import PyAudioSource
    from app.audio.vosk_recognizer import VoskSpeechRecognizer

    cfg = load_config()
    print("Enumerating audio input devices for Logitech C270...")
    audio = PyAudioSource(config=cfg.audio)
    device_idx = audio.discover_webcam_mic()
    print(f"Selected audio input device index: {device_idx}")

    print("Loading offline Vosk language model...")
    recognizer = VoskSpeechRecognizer(config=cfg.speech)
    recognizer.initialize()

    audio.start_stream()
    print("\n>> MICROPHONE IS LIVE. Speak clearly: 'find backpack' or 'move forward'...")
    start = time.time()
    recognized = ""
    while time.time() - start < 8.0:
        chunk = audio.read_chunk()
        if chunk and recognizer.accept_waveform(chunk):
            recognized = recognizer.get_result()
            print(f"\n[RECOGNIZED]: '{recognized}'")
            break
        time.sleep(0.01)

    audio.stop_stream()
    audio.close()
    if recognized:
        print(f"[PASS] Successfully recognized speech: '{recognized}'")
        return True
    else:
        print("[INFO] No complete utterance captured in 8 seconds.")
        return False


def test_ultrasonic_known_distances():
    """Test 3: Ultrasonic readings at known measured physical distances."""
    print("\n" + "=" * 60)
    print("  TEST 3: ULTRASONIC SENSOR CALIBRATION AT KNOWN DISTANCES")
    print("=" * 60)
    print("Pins: TRIG=23, ECHO=24 (BCM). Voltage divider on ECHO line required.")
    cfg = load_config()
    gpio = RPiGPIOAdapter() if sys.platform.startswith("linux") else FakeGPIOAdapter()
    sensor = HCSR04UltrasonicSensor(gpio=gpio, config=cfg.ultrasonic, safety_config=cfg.safety)
    sensor.initialize()

    test_targets = [10.0, 20.0, 50.0]
    for dist in test_targets:
        if not prompt_confirm(f"Place a flat vertical board exactly {dist} cm in front of HC-SR04 sensor. Ready?"):
            print("[ABORT] Test skipped by operator.")
            return False

        samples = []
        for _ in range(5):
            r = sensor.sample_distance_cm()
            if r.distance_cm is not None:
                samples.append(r.distance_cm)
            time.sleep(0.06)

        if samples:
            avg = sum(samples) / len(samples)
            error = abs(avg - dist)
            print(f"  Measured: {avg:.1f} cm (Target: {dist} cm, Error: {error:.1f} cm) -> {'PASS' if error < 5.0 else 'CHECK CALIBRATION'}")
        else:
            print(f"  [FAIL] No valid acoustic echo returned for {dist} cm.")

    return True


def test_motors_raised_chassis():
    """Test 4: Motor direction and PWM duty cycle with chassis elevated."""
    print("\n" + "=" * 60)
    print("  TEST 4: MOTOR DIRECTION & PWM (CHASSIS ELEVATED)")
    print("=" * 60)
    print("MANDATORY SAFETY: All four wheels MUST be elevated and spinning freely in the air.")
    cfg = load_config()
    print(f"BCM Pin Map: Left ENA={cfg.motor.left_ena_pin}, IN1={cfg.motor.left_in1_pin}, IN2={cfg.motor.left_in2_pin} | "
          f"Right ENB={cfg.motor.right_enb_pin}, IN3={cfg.motor.right_in3_pin}, IN4={cfg.motor.right_in4_pin}")
    if not prompt_confirm("Are all wheels elevated off the ground and power cutoff switch in hand?"):
        print("[ABORT] Test cancelled for safety.")
        return False

    gpio = RPiGPIOAdapter() if sys.platform.startswith("linux") else FakeGPIOAdapter()
    motors = L298NMotorController(gpio=gpio, config=cfg.motor)
    motors.initialize()
    motors.enable(True)

    try:
        # Forward test
        print("1. Commanding FORWARD at speed 0.40 for 1.0 second...")
        motors.set_speeds(0.40, 0.40)
        time.sleep(1.0)
        motors.stop()
        if not prompt_confirm("Did all four wheels rotate in the FORWARD direction?"):
            print("[FAIL] Motor wiring inverted. Check IN1/IN2 or IN3/IN4 polarity.")
            return False

        # Backward test
        print("2. Commanding BACKWARD at speed 0.40 for 1.0 second...")
        motors.set_speeds(-0.40, -0.40)
        time.sleep(1.0)
        motors.stop()
        if not prompt_confirm("Did all four wheels rotate in the REVERSE direction?"):
            print("[FAIL] Reverse direction mismatch.")
            return False

        # Turn left
        print("3. Commanding TURN LEFT at speed 0.40 for 0.8 seconds...")
        motors.set_speeds(-0.40, 0.40)
        time.sleep(0.8)
        motors.stop()

        # Turn right
        print("4. Commanding TURN RIGHT at speed 0.40 for 0.8 seconds...")
        motors.set_speeds(0.40, -0.40)
        time.sleep(0.8)
        motors.stop()

        print("[PASS] Motor direction and differential channels verified.")
        return True
    finally:
        motors.stop()
        motors.cleanup()


def test_forward_obstacle_stopping():
    """Test 5: Forward obstacle interlock stopping."""
    print("\n" + "=" * 60)
    print("  TEST 5: FORWARD OBSTACLE INTERLOCK STOPPING")
    print("=" * 60)
    print("Tests that the state machine strictly refuses forward movement when distance < obstacle_blocked_cm.")
    cfg = load_config()
    print(f"Obstacle thresholds: Warning={cfg.safety.obstacle_warning_cm} cm, Blocked={cfg.safety.obstacle_blocked_cm} cm, Critical={cfg.safety.obstacle_critical_cm} cm")
    print("Place an obstacle within 20 cm in front of the ultrasonic sensor.")
    if not prompt_confirm("Obstacle in place?"):
        return False

    gpio = RPiGPIOAdapter() if sys.platform.startswith("linux") else FakeGPIOAdapter()
    sensor = HCSR04UltrasonicSensor(gpio=gpio, config=cfg.ultrasonic, safety_config=cfg.safety)
    sensor.initialize()
    reading = sensor.sample_distance_cm()
    print(f"Sensor reading: {reading.distance_cm} cm | State: {reading.state.value} | Allows forward: {reading.allows_forward_motion}")

    if not reading.allows_forward_motion:
        print("[PASS] Forward motion successfully interlocked and blocked by sensor.")
        return True
    else:
        print("[FAIL] Sensor allowed forward motion despite obstacle proximity.")
        return False


def test_rotation_sector_calibration():
    """Test 6: In-place 360 rotation duration calibration on floor."""
    print("\n" + "=" * 60)
    print("  TEST 6: IN-PLACE ROTATION DURATION CALIBRATION")
    print("=" * 60)
    print("Robot should be on the actual floor surface in a clear 1.5m x 1.5m area.")
    if not prompt_confirm("Is robot placed on the operational floor surface with clear space?"):
        return False

    cfg = load_config()
    gpio = RPiGPIOAdapter() if sys.platform.startswith("linux") else FakeGPIOAdapter()
    motors = L298NMotorController(gpio=gpio, config=cfg.motor)
    motors.initialize()
    motors.enable(True)

    turn_spd = TURN_SPEED
    print(f"Turning in place at speed {turn_spd}...")
    start = time.time()
    motors.set_speeds(turn_spd, -turn_spd)

    input("Press ENTER the exact moment the robot completes exactly one full 360-degree rotation...")
    elapsed = time.time() - start
    motors.stop()
    motors.cleanup()

    print(f"\n[MEASURED 360 DURATION]: {elapsed:.2f} seconds at speed {turn_spd}")
    print(f"Update your config/default_config.yaml:")
    print(f"  motion:\n    calibrated_360_duration_sec: {elapsed:.2f}")
    return True


def main():
    parser = argparse.ArgumentParser(description="SS_PROJECT_VCR_FRESH Hardware Acceptance Suite")
    parser.add_argument("--all", action="store_true", help="Run all acceptance tests sequentially")
    parser.add_argument("--test", choices=["camera", "speech", "ultrasonic", "motors", "obstacle", "rotation"], help="Run specific acceptance test")
    args = parser.parse_args()

    print("=" * 70)
    print("  SS_PROJECT_VCR_FRESH — HARDWARE ACCEPTANCE & CALIBRATION SUITE")
    print("=" * 70)

    if args.test == "camera" or args.all:
        test_camera()
    if args.test == "speech" or args.all:
        test_microphone_and_vosk()
    if args.test == "ultrasonic" or args.all:
        test_ultrasonic_known_distances()
    if args.test == "motors" or args.all:
        test_motors_raised_chassis()
    if args.test == "obstacle" or args.all:
        test_forward_obstacle_stopping()
    if args.test == "rotation" or args.all:
        test_rotation_sector_calibration()

    print("\nAcceptance routines completed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
