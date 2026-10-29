#!/usr/bin/env python3
"""Phase 1 Isolated Motor Channel Diagnostic Tool for SS_PROJECT_VCR_FRESH.

Safety Protocols:
- Never energizes outputs during import or preflight.
- Fails closed on SIGINT (Ctrl+C) or any exception.
- Allows isolated single-channel testing to verify Left and Right channels independently.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from app.config import load_config
from app.constants import ExecutionMode
from app.hardware.gpio_adapter import FakeGPIOAdapter, RPiGPIOAdapter
from app.hardware.motors import L298NMotorController


def run_isolated_test(target: str, speed: float = 0.30, duration_sec: float = 0.35, dry_run: bool = False) -> int:
    cfg = load_config()
    print("=" * 65)
    print("  PHASE 1: MOTOR ISOLATED CHANNEL DIAGNOSTIC TEST")
    print("=" * 65)
    print(f"Target: {target.upper()} | Speed: {speed*100:.0f}% PWM | Duration: {duration_sec:.2f}s")
    print(f"Hardware Pins: Left(ENA={cfg.motor.left_ena_pin}, IN1={cfg.motor.left_in1_pin}, IN2={cfg.motor.left_in2_pin})")
    print(f"               Right(ENB={cfg.motor.right_enb_pin}, IN3={cfg.motor.right_in3_pin}, IN4={cfg.motor.right_in4_pin})")
    print(f"Execution Mode: {'DRY RUN (FakeGPIO)' if dry_run or not sys.platform.startswith('linux') else 'REAL HARDWARE (RPi.GPIO)'}")
    print("=" * 65)

    if dry_run or not sys.platform.startswith("linux"):
        gpio = FakeGPIOAdapter()
    else:
        gpio = RPiGPIOAdapter()

    motor = L298NMotorController(gpio=gpio, config=cfg.motor, motion_config=cfg.motion)

    try:
        print("\n[STEP 1] Initializing GPIO and zeroing all outputs (IN1..4 LOW, PWM 0%)...")
        motor.initialize()
        motor.stop()
        print("  Outputs commanded: ALL LOW, 0% PWM.")
        time.sleep(1.0)

        if target == "hold_stop":
            print("\n[TEST: HOLD_STOP] Holding outputs in STOP state for 3 seconds...")
            print("  Observe wheels: If either side spins, hardware has a short or external power leak.")
            for c in range(3, 0, -1):
                print(f"  Holding... {c}s remaining")
                time.sleep(1.0)
            print("  HOLD_STOP completed.")
            return 0

        print(f"\n[STEP 2] Preparing to pulse: {target.upper()} in 3 seconds...")
        print("  Keep battery cutoff switch in hand.")
        for c in range(3, 0, -1):
            print(f"  {c}...")
            time.sleep(1.0)

        motor.enable(True)
        if not motor.is_enabled:
            print("[ERROR] Failed to enable motor driver output.")
            return 1

        print(f"  >>> PULSING {target.upper()} NOW ({duration_sec:.2f}s @ {speed*100:.0f}%) <<<")

        if target in ("left_only", "left_forward"):
            print(f"  Commanding: LEFT FORWARD (IN1={cfg.motor.left_in1_pin} HIGH, IN2={cfg.motor.left_in2_pin} LOW, ENA={speed*100:.0f}%)")
            motor.set_speeds(left=speed, right=0.0)
        elif target == "left_reverse":
            print(f"  Commanding: LEFT REVERSE (IN1={cfg.motor.left_in1_pin} LOW, IN2={cfg.motor.left_in2_pin} HIGH, ENA={speed*100:.0f}%)")
            motor.set_speeds(left=-speed, right=0.0)
        elif target in ("right_only", "right_forward"):
            print(f"  Commanding: RIGHT FORWARD (IN3={cfg.motor.right_in3_pin} HIGH, IN4={cfg.motor.right_in4_pin} LOW, ENB={speed*100:.0f}%)")
            motor.set_speeds(left=0.0, right=speed)
        elif target == "right_reverse":
            print(f"  Commanding: RIGHT REVERSE (IN3={cfg.motor.right_in3_pin} LOW, IN4={cfg.motor.right_in4_pin} HIGH, ENB={speed*100:.0f}%)")
            motor.set_speeds(left=0.0, right=-speed)
        elif target == "both_forward":
            print(f"  Commanding: BOTH FORWARD (Left + Right forward)")
            motor.set_speeds(left=speed, right=speed)
        elif target == "both_reverse":
            print(f"  Commanding: BOTH REVERSE (Left + Right reverse)")
            motor.set_speeds(left=-speed, right=-speed)
        else:
            print(f"Unknown target: {target}")
            return 1

        time.sleep(duration_sec)
        motor.stop()
        motor.enable(False)
        print("  >>> PULSE COMPLETE. ALL OUTPUTS DE-ENERGIZED. <<<")
        return 0

    except KeyboardInterrupt:
        print("\n[INTERRUPT] Aborted by operator (Ctrl+C). Emergency stop engaged.")
        return 130
    except Exception as exc:
        print(f"\n[ERROR] Exception during diagnostic: {exc}")
        return 1
    finally:
        try:
            motor.stop()
            motor.enable(False)
            motor.cleanup()
        except Exception:
            pass
        print("[CLEANUP] Pins released and returned to safe state.\n")


def main() -> int:
    parser = argparse.ArgumentParser(description="Phase 1 Motor Channel Diagnostic")
    parser.add_argument(
        "--target",
        choices=[
            "hold_stop", "left_only", "left_forward", "left_reverse",
            "right_only", "right_forward", "right_reverse",
            "both_forward", "both_reverse",
        ],
        default="hold_stop",
        help="Motion target to test",
    )
    parser.add_argument("--speed", type=float, default=0.30, help="PWM speed (0.10 to 0.50)")
    parser.add_argument("--duration", type=float, default=0.35, help="Pulse duration in seconds")
    parser.add_argument("--dry-run", action="store_true", help="Simulate with FakeGPIOAdapter")
    args = parser.parse_args()

    clamped_speed = max(0.10, min(0.80, args.speed))
    clamped_duration = max(0.10, min(1.00, args.duration))
    return run_isolated_test(target=args.target, speed=clamped_speed, duration_sec=clamped_duration, dry_run=args.dry_run)


if __name__ == "__main__":
    raise SystemExit(main())
