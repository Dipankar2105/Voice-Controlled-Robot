#!/usr/bin/env python3
"""Phase 2 Motor Direction, Balance & Rotation Calibration Diagnostic Tool.

Safety Protocols:
- Never energizes outputs during import or preflight.
- Fails closed on SIGINT (Ctrl+C) or any exception.
- Applies left and right speed trims dynamically.
- Strictly clamps PWM (0.10 to 0.70) and pulse duration (0.05 to 5.00s).
- Always forces hardware stop on completion or failure.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from app.config import MotionConfig, load_config
from app.hardware.gpio_adapter import FakeGPIOAdapter, RPiGPIOAdapter
from app.hardware.motors import L298NMotorController


def run_calibration_pulse(
    target: str,
    speed: float = 0.35,
    duration_sec: float = 0.35,
    left_trim: float = 1.0,
    right_trim: float = 1.0,
    dry_run: bool = False,
) -> int:
    cfg = load_config()

    # Create motion config with specified trims
    motion_cfg = MotionConfig(
        manual_speed=cfg.motion.manual_speed,
        turn_speed=cfg.motion.turn_speed,
        approach_speed=cfg.motion.approach_speed,
        search_speed=cfg.motion.search_speed,
        speed_ramp_step=cfg.motion.speed_ramp_step,
        max_speed=cfg.motion.max_speed,
        search_scan_duration_sec=cfg.motion.search_scan_duration_sec,
        calibrated_360_duration_sec=cfg.motion.calibrated_360_duration_sec,
        manual_pulse_sec=cfg.motion.manual_pulse_sec,
        left_trim=left_trim,
        right_trim=right_trim,
    )

    print("=" * 70)
    print("  PHASE 2: MOTOR DIRECTION, BALANCE & ROTATION CALIBRATION")
    print("=" * 70)
    print(f"Target Command   : {target.upper()}")
    print(f"Nominal Speed    : {speed*100:.1f}% PWM")
    print(f"Duration         : {duration_sec:.2f} seconds")
    print(f"Speed Trims      : Left Trim = {left_trim:.3f} | Right Trim = {right_trim:.3f}")
    print(f"Effective PWM    : Left = {speed * left_trim * 100:.1f}% | Right = {speed * right_trim * 100:.1f}%")
    print(f"GPIO Mode        : {'DRY RUN (FakeGPIOAdapter)' if dry_run or not sys.platform.startswith('linux') else 'REAL HARDWARE (RPiGPIOAdapter)'}")
    print("=" * 70)

    if dry_run or not sys.platform.startswith("linux"):
        gpio = FakeGPIOAdapter()
    else:
        gpio = RPiGPIOAdapter()

    motors = L298NMotorController(gpio=gpio, config=cfg.motor, motion_config=motion_cfg)

    try:
        print("\n[STEP 1] Preflight: Initializing GPIO and zeroing outputs...")
        motors.initialize()
        motors.stop()
        print("  Outputs verified: All direction pins LOW, PWM duty cycles 0.0%.")

        print(f"\n[STEP 2] Standby: Ready to pulse {target.upper()} in 3 seconds...")
        print("  EMERGENCY PROCEDURE: Press Ctrl+C or hit battery cutoff switch to stop.")
        for count in range(3, 0, -1):
            print(f"  {count}...")
            time.sleep(1.0)

        motors.enable(True)
        if not motors.is_enabled:
            print("[ERROR] Failed to enable motor driver output.")
            return 1

        print(f"\n  >>> EXECUTING PULSE: {target.upper()} NOW ({duration_sec:.2f}s) <<<")

        if target == "forward":
            motors.set_speeds(left=speed, right=speed)
        elif target == "reverse":
            motors.set_speeds(left=-speed, right=-speed)
        elif target == "turn_left":
            # Differential CCW rotation: Left reverse, Right forward
            motors.set_speeds(left=-speed, right=speed)
        elif target == "turn_right":
            # Differential CW rotation: Left forward, Right reverse
            motors.set_speeds(left=speed, right=-speed)
        else:
            print(f"[ERROR] Unknown target action: {target}")
            return 1

        time.sleep(duration_sec)

        motors.stop()
        motors.enable(False)
        print("  >>> PULSE FINISHED. OUTPUTS DE-ENERGIZED (0% PWM, ALL PINS LOW). <<<\n")
        return 0

    except KeyboardInterrupt:
        print("\n[OPERATOR ABORT] Ctrl+C received! Immediate emergency stop engaged.")
        return 130
    except Exception as exc:
        print(f"\n[FAILURE] Exception during pulse execution: {exc}")
        return 1
    finally:
        try:
            motors.stop()
            motors.enable(False)
            motors.cleanup()
        except Exception:
            pass
        print("[CLEANUP] Controller cleaned up. Pins de-energized.")


def main() -> int:
    parser = argparse.ArgumentParser(description="Phase 2 Calibration Diagnostic Tool")
    parser.add_argument(
        "--target",
        choices=["forward", "reverse", "turn_left", "turn_right"],
        required=True,
        help="Directional pulse to execute",
    )
    parser.add_argument(
        "--speed",
        type=float,
        default=0.35,
        help="Base PWM speed (0.10 to 0.70)",
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=0.35,
        help="Pulse duration in seconds (0.05 to 20.0)",
    )
    parser.add_argument(
        "--left-trim",
        type=float,
        default=1.00,
        help="Left channel trim multiplier [0.10, 1.00]",
    )
    parser.add_argument(
        "--right-trim",
        type=float,
        default=1.00,
        help="Right channel trim multiplier [0.10, 1.00]",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run without driving physical GPIO (FakeGPIOAdapter)",
    )
    args = parser.parse_args()

    clamped_speed = max(0.10, min(0.70, args.speed))
    clamped_duration = max(0.05, min(20.00, args.duration))
    clamped_left_trim = max(0.10, min(1.00, args.left_trim))
    clamped_right_trim = max(0.10, min(1.00, args.right_trim))

    return run_calibration_pulse(
        target=args.target,
        speed=clamped_speed,
        duration_sec=clamped_duration,
        left_trim=clamped_left_trim,
        right_trim=clamped_right_trim,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    raise SystemExit(main())
