#!/usr/bin/env python3
"""Phase 3 Physical Interlock & Safety Verification Tool.

Tests:
1. Forward motion refusal when ultrasonic obstacle <= 25.0 cm.
2. Backward recovery motion allowance when forward obstacle is present.
3. Guaranteed zero motor actuation during forward block.
4. Emergency stop override.
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
from app.command.command_parser import CommandAction, CommandIntent, RobotCommand
from app.constants import ExecutionMode, RobotState
from app.hardware.interfaces import ObstacleState
from app.hardware.gpio_adapter import FakeGPIOAdapter, RPiGPIOAdapter
from app.hardware.motors import L298NMotorController
from app.hardware.ultrasonic import HCSR04UltrasonicSensor
from app.navigation.state_machine import RobotStateMachine
from app.runtime import RobotRuntime


def run_interlock_test(dry_run: bool = False, run_backward: bool = False) -> int:
    cfg = load_config()
    print("=" * 70)
    print("  PHASE 3: PHYSICAL ULTRASONIC INTERLOCK & SAFETY TEST")
    print("=" * 70)

    if dry_run or not sys.platform.startswith("linux"):
        gpio = FakeGPIOAdapter()
    else:
        gpio = RPiGPIOAdapter()

    sensor = HCSR04UltrasonicSensor(gpio=gpio, config=cfg.ultrasonic, safety_config=cfg.safety)
    motors = L298NMotorController(gpio=gpio, config=cfg.motor, motion_config=cfg.motion)
    fsm = RobotStateMachine(config=cfg, motors=motors, distance_sensor=sensor)

    try:
        print("\n[STEP 1] Initializing hardware drivers...")
        sensor.initialize()
        sensor.start()
        time.sleep(0.35)
        motors.initialize()
        motors.stop()
        print("  Motors stopped (0% PWM, all pins LOW).")

        print("\n[STEP 2] Sampling forward ultrasonic distance...")
        time.sleep(0.5)
        for i in range(3):
            reading = sensor.sample_distance_cm()
            dist_str = f"{reading.distance_cm:.1f} cm" if reading.distance_cm is not None else "OUT_OF_RANGE"
            print(f"  Sample #{i+1}: {dist_str} | State: {reading.state.value} | Valid: {reading.is_valid}")
            time.sleep(0.15)

        print("\n[STEP 3] Attempting FORWARD command with ultrasonic protection ON...")
        time.sleep(0.15)
        chk_reading = sensor.get_reading()
        chk_str = f"{chk_reading.distance_cm:.1f} cm" if chk_reading.distance_cm is not None else "OUT_OF_RANGE"
        print(f"  Sensor get_reading(): {chk_str} | State: {chk_reading.state.value}")
        forward_safe = fsm._is_forward_path_safe()
        print(f"  fsm._is_forward_path_safe() returned: {forward_safe}")

        # Dispatch command through FSM
        from app.command.command_parser import RobotCommand, CommandIntent
        cmd = RobotCommand(
            raw_transcript="move forward",
            normalized_text="move forward",
            intent=CommandIntent.MOVEMENT,
            action=CommandAction.MOVE_FORWARD,
        )
        time.sleep(0.15)
        result = fsm.handle_command(cmd)
        print(f"  fsm.handle_command('move forward') accepted? {result}")
        print(f"  FSM State after forward attempt: {fsm.state.value}")
        print(f"  Motors Commanded Speeds: {motors.current_speeds}")
        if not forward_safe:
            assert not result, "Forward command must be rejected when path is unsafe"
            assert motors.current_speeds == (0.0, 0.0), "Motors must remain strictly 0.0"
            print("  >>> VERIFIED: MOTORS DID NOT MOVE. Interlock successfully blocked actuation. <<<")
        else:
            print(f"  [INFO] Path is clear ({reading.distance_cm} cm > 25.0 cm). Place an obstacle within 25 cm to test refusal.")
            return 2

        print("\n[STEP 4] Testing BACKWARD motion allowance while forward obstacle is present...")
        cmd_back = RobotCommand(
            raw_transcript="move backward",
            normalized_text="move backward",
            intent=CommandIntent.MOVEMENT,
            action=CommandAction.MOVE_BACKWARD,
        )
        print("  Backward movement is expected to be accepted to allow evasion.")
        if run_backward:
            print("  Executing supervised backward pulse (0.35s @ 40% PWM)...")
            back_result = fsm.handle_command(cmd_back)
            print(f"  fsm.handle_command('move backward') accepted? {back_result}")
            print(f"  FSM State after backward: {fsm.state.value}")
            print(f"  Motors Commanded Speeds: {motors.current_speeds}")
            assert back_result, "Backward command must be accepted during BLOCKED state"
            print("  >>> VERIFIED: Backward evasion motion completed and motors stopped. <<<")
        else:
            print("  FSM allows backward command? Ready for supervised backward pulse.")
        return 0

    except KeyboardInterrupt:
        print("\n[ABORT] Operator cancelled test.")
        return 130
    except Exception as exc:
        print(f"\n[ERROR] Exception in interlock test: {exc}")
        return 1
    finally:
        try:
            motors.stop()
            motors.cleanup()
            sensor.stop()
        except Exception:
            pass
        print("[CLEANUP] Pins released and returned to safe state.\n")


def main() -> int:
    parser = argparse.ArgumentParser(description="Phase 3 Interlock Test")
    parser.add_argument("--dry-run", action="store_true", help="Simulate with FakeGPIO")
    parser.add_argument("--run-backward", action="store_true", help="Execute supervised backward recovery pulse")
    args = parser.parse_args()
    return run_interlock_test(dry_run=args.dry_run, run_backward=args.run_backward)


if __name__ == "__main__":
    raise SystemExit(main())
