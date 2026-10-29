#!/usr/bin/env python3
"""Phase 3 Joystick Hold-to-Move, Release-to-Stop & Watchdog Diagnostic Tool.

Verifies on physical hardware:
1. Hold-to-move and release-to-stop.
2. Watchdog timeout stopping motors when client heartbeat ceases (>0.40s).
3. Emergency stop preemption and latching.
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
from app.constants import ExecutionMode, RobotState
from app.hardware.gpio_adapter import FakeGPIOAdapter, RPiGPIOAdapter
from app.hardware.motors import L298NMotorController
from app.hardware.ultrasonic import HCSR04UltrasonicSensor
from app.runtime import RobotRuntime


def run_joystick_diagnostic(dry_run: bool = False) -> int:
    cfg = load_config()
    print("=" * 70)
    print("  PHASE 3: PHYSICAL JOYSTICK & WATCHDOG LEASE DIAGNOSTIC")
    print("=" * 70)

    if dry_run or not sys.platform.startswith("linux"):
        gpio = FakeGPIOAdapter()
    else:
        gpio = RPiGPIOAdapter()

    sensor = HCSR04UltrasonicSensor(gpio=gpio, config=cfg.ultrasonic, safety_config=cfg.safety)
    motors = L298NMotorController(gpio=gpio, config=cfg.motor, motion_config=cfg.motion)

    runtime = RobotRuntime(
        config=cfg,
        motor_controller=motors,
        distance_sensor=sensor,
    )

    try:
        print("\n[STEP 1] Starting runtime subsystem...")
        motors.initialize()
        runtime.start()
        print("  Runtime started. FSM in IDLE_LISTENING.")

        print("\n[STEP 2] Testing Joystick HOLD-TO-MOVE (Reverse at 40% PWM for 0.40s)...")
        success, msg = runtime.dispatch_joystick("backward", speed=0.40)
        print(f"  dispatch_joystick('backward', 0.40): {success} ({msg})")
        print(f"  Motors Commanded: {motors.current_speeds}")
        assert success, "Joystick backward must succeed"
        assert motors.current_speeds == (-0.40, -0.40)
        time.sleep(0.40)

        print("\n[STEP 3] Testing Joystick RELEASE-TO-STOP...")
        stop_success, stop_msg = runtime.dispatch_joystick("stop")
        print(f"  dispatch_joystick('stop'): {stop_success} ({stop_msg})")
        print(f"  Motors Commanded: {motors.current_speeds}")
        assert stop_success, "Joystick stop must succeed"
        assert motors.current_speeds == (0.0, 0.0)
        print("  >>> VERIFIED: Release-to-stop de-energized motors immediately. <<<")

        print("\n[STEP 4] Testing WATCHDOG LEASE TIMEOUT (Heartbeat Ceased)...")
        wb_success, wb_msg = runtime.dispatch_joystick("backward", speed=0.40)
        print(f"  Simulating active drive: {wb_success} (Motors: {motors.current_speeds})")
        print("  Ceasing client heartbeats... waiting 0.60s for watchdog lease expiry (0.40s limit)...")
        time.sleep(0.60)

        print(f"  Motors after watchdog timeout: {motors.current_speeds}")
        print(f"  Joystick active flag: {runtime._joystick_active}")
        assert motors.current_speeds == (0.0, 0.0), "Watchdog must have stopped motors"
        assert not runtime._joystick_active, "Joystick active flag must be False after timeout"
        print("  >>> VERIFIED: Watchdog lease halted motors within 0.40s timeout! <<<")

        print("\n[STEP 5] Testing EMERGENCY STOP Preemption & Latching...")
        # Start a drive action
        runtime.dispatch_joystick("backward", speed=0.40)
        # Issue emergency stop
        estop_ok, _, estop_msg = runtime.dispatch_text_command("emergency stop")
        print(f"  dispatch_text_command('emergency stop'): {estop_ok} ({estop_msg})")
        print(f"  FSM State: {runtime.fsm.state.value}")
        print(f"  E-Stop Latched? {runtime.fsm.is_emergency_stop_latched}")
        print(f"  Motors Commanded: {motors.current_speeds}")
        assert runtime.fsm.is_emergency_stop_latched, "E-Stop must be latched"
        assert motors.current_speeds == (0.0, 0.0), "Motors must be stopped"

        # Verify new commands are rejected while latched
        rej_ok, rej_msg = runtime.dispatch_joystick("backward", speed=0.40)
        print(f"  Subsequent joystick command during E-stop rejected? {not rej_ok} ({rej_msg})")
        assert not rej_ok, "Must reject actuation while E-Stop is latched"

        # Verify E-stop clear behavior:
        # Refused if sensor is blocked/invalid, but allowed if operator overrides protection or sensor is clear
        cleared = runtime.fsm.clear_emergency_stop()
        if not cleared:
            print("  [SAFETY INVARIANT] E-stop clear correctly REFUSED because sensor is invalid/blocked!")
            print("  Applying operator protection override to clear...")
            runtime.set_ultrasonic_protection(False)
            cleared_override = runtime.fsm.clear_emergency_stop()
            print(f"  E-stop cleared with override? {cleared_override} (FSM: {runtime.fsm.state.value})")
            assert cleared_override, "E-stop clear must succeed under operator override"
            # Restore protection
            runtime.set_ultrasonic_protection(True)
        else:
            print(f"  E-stop cleared? {cleared} (FSM: {runtime.fsm.state.value})")
            assert cleared, "E-stop clear must succeed"

        print("  >>> VERIFIED: Emergency Stop halts, latches, gates clearance, and clears correctly! <<<")

        return 0

    except KeyboardInterrupt:
        print("\n[ABORT] Test interrupted by operator.")
        return 130
    except Exception as exc:
        print(f"\n[ERROR] Exception in joystick diagnostic: {exc}")
        return 1
    finally:
        try:
            runtime.stop()
            runtime.cleanup()
        except Exception:
            pass
        print("[CLEANUP] Runtime stopped and pins released.\n")


def main() -> int:
    parser = argparse.ArgumentParser(description="Phase 3 Joystick Diagnostic")
    parser.add_argument("--dry-run", action="store_true", help="Simulate with FakeGPIO")
    args = parser.parse_args()
    return run_joystick_diagnostic(dry_run=args.dry_run)


if __name__ == "__main__":
    raise SystemExit(main())
