"""Interactive Physical Hardware Commissioning Utility for SS_PROJECT_VCR_FRESH.

MANDATORY SAFETY PROTOCOL:
1. Elevate robot chassis so all four wheels rotate freely in the air.
2. Confirm a physical battery power cutoff switch is within immediate reach.
3. Verify voltage divider on HC-SR04 ECHO line (must NOT exceed 3.3V).
4. Run this script manually on the Raspberry Pi:
   python scripts/commission_hardware.py --interactive
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

from app.config import AppConfig, load_config
from app.constants import ExecutionMode
from app.hardware.gpio_adapter import FakeGPIOAdapter, RPiGPIOAdapter
from app.hardware.motors import L298NMotorController
from app.hardware.ultrasonic import HCSR04UltrasonicSensor


def run_commissioning(interactive: bool = True, force_mock: bool = False) -> int:
    print("=" * 70)
    print("  SS_PROJECT_VCR_FRESH — HARDWARE COMMISSIONING CHECKLIST")
    print("=" * 70)

    cfg = load_config()

    # Determine GPIO adapter
    if force_mock or not sys.platform.startswith("linux"):
        print("\n[INFO] Running in SIMULATION / BENCH TEST mode (FakeGPIOAdapter).")
        print("       Physical GPIO will NOT be energized.")
        gpio = FakeGPIOAdapter()
    else:
        print("\n[INFO] Initializing physical RPiGPIOAdapter on Linux BCM GPIO...")
        try:
            gpio = RPiGPIOAdapter()
        except Exception as exc:
            print(f"[ERROR] Failed to initialize physical GPIO: {exc}")
            return 1

    # Step 1: Preflight Mechanical & Electrical Checks
    print("\n--- STEP 1: PHYSICAL PREFLIGHT SAFETY INSPECTION ---")
    print("  1. Is the robot chassis elevated with wheels off the ground? (MANDATORY)")
    print("  2. Is an external motor battery connected to L298N VMS terminal?")
    print("  3. Is there a physical power cutoff switch immediately reachable?")
    print("  4. Is common ground connected between battery (-), L298N GND, and Pi GND?")
    if interactive:
        response = input("\nHave all 4 preflight conditions been physically verified? (yes/no): ").strip().lower()
        if response != "yes":
            print("[ABORT] Commissioning aborted by operator.")
            return 2

    # Step 2: Electrical Voltage Divider Check
    print("\n--- STEP 2: HC-SR04 ECHO VOLTAGE DIVIDER INSPECTION ---")
    print("  HC-SR04 ECHO output delivers 5.0V TTL pulses.")
    print("  Raspberry Pi GPIO 24 is strictly 3.3V tolerant.")
    print("  A resistive divider (e.g. 1k series, 2k shunt or 10k/10k) MUST reduce ECHO to <= 3.3V.")
    if interactive:
        response = input("\nHas the voltage divider on GPIO 24 been measured or verified? (yes/no): ").strip().lower()
        if response != "yes":
            print("[ABORT] Commissioning aborted: ECHO line overvoltage risks SoC destruction.")
            return 2

    # Step 3: Ultrasonic Sensor Verification
    print("\n--- STEP 3: ULTRASONIC SENSOR SAMPLING ---")
    sensor = HCSR04UltrasonicSensor(gpio=gpio, config=cfg.ultrasonic, safety_config=cfg.safety)
    sensor.initialize()
    print("Taking 5 distance samples...")
    for i in range(1, 6):
        reading = sensor.sample_distance_cm()
        print(f"  Sample {i}: Distance={reading.distance_cm} cm | State={reading.state.value} | Valid={reading.is_valid}")
        time.sleep(0.1)

    # Step 4: Supervised Low-Speed Motor Pulse (0.20s @ 30% PWM)
    print("\n--- STEP 4: SUPERVISED LOW-SPEED MOTOR ACTUATION ---")
    motors = L298NMotorController(gpio=gpio, config=cfg.motor, motion_config=cfg.motion)
    motors.initialize()
    motors.enable(True)

    print("Command details: 0.20-second pulse at 30% PWM forward.")
    if interactive:
        response = input("\nExecute 0.20-second forward pulse now? (yes/no): ").strip().lower()
        if response == "yes":
            print("Pulsing motors forward for 0.20 seconds...")
            motors.move_pulse(left=0.30, right=0.30, duration_sec=0.20)
            print("Motors stopped. PWM set to 0%, directional pins LOW.")
            direction_ok = input("Did left and right wheels rotate FORWARD? (yes/no): ").strip().lower()
            if direction_ok != "yes":
                print("[WARNING] Motor polarity mismatch detected! Swap IN1/IN2 or IN3/IN4 wires.")
        else:
            print("Motor actuation skipped by operator.")

    # Step 5: Cleanup and Final Status
    motors.stop()
    motors.cleanup()
    sensor.stop()

    print("\n" + "=" * 70)
    print("  COMMISSIONING PROCEDURE FINISHED — HARDWARE SAFE & DE-ENERGIZED")
    print("=" * 70)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="SS_PROJECT_VCR_FRESH Hardware Commissioning")
    parser.add_argument("--interactive", action="store_true", default=True, help="Run with interactive safety prompts")
    parser.add_argument("--non-interactive", action="store_false", dest="interactive", help="Run non-interactively (bench test only)")
    parser.add_argument("--force-mock", action="store_true", help="Force FakeGPIOAdapter even on Linux")
    args = parser.parse_args()
    return run_commissioning(interactive=args.interactive, force_mock=args.force_mock)


if __name__ == "__main__":
    raise SystemExit(main())
