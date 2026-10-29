#!/usr/bin/env python3
"""Phase 5 Supervised Autonomous Target Search & Approach Diagnostic Tool.

Executes a supervised end-to-end autonomous mission:
1. Preflight safety check (perception, ultrasonic distance, motor zeroing).
2. Spoken or commanded search intent (e.g. 'bottle').
3. Multi-frame target confirmation.
4. Optical centering alignment.
5. Ultrasonic-gated forward approach pulses toward the 30 cm standoff line.
6. Target reached arrival and fail-closed stopping.

Safety Protocols:
- Never energizes outputs during preflight.
- Fails closed on SIGINT (Ctrl+C) or any exception.
- Forward motion strictly refused if ultrasonic distance <= 25.0 cm.
- Clean stop and de-energization immediately upon reaching 30.0 cm standoff.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from dataclasses import replace
from app.config import load_config
from app.constants import ExecutionMode, FaultCode, RobotState
from app.hardware.gpio_adapter import FakeGPIOAdapter, RPiGPIOAdapter
from app.hardware.mock_hardware import (
    MockCamera,
    MockDetector,
    MockDistanceSensor,
    MockMotorController,
)
from app.hardware.motors import L298NMotorController
from app.hardware.ultrasonic import HCSR04UltrasonicSensor
from app.navigation.state_machine import RobotStateMachine
from app.vision.camera import CameraOwner, OpenCVCameraBackend
from app.vision.yolo_detector import ONNXYOLODetector


def run_mission(
    target: str = "bottle",
    dry_run: bool = False,
    max_steps: int = 60,
    target_stop_cm: float | None = None,
    blocked_cm: float | None = None,
    critical_cm: float | None = None,
) -> int:
    cfg = load_config()

    if target_stop_cm is not None or blocked_cm is not None or critical_cm is not None:
        target_stop = target_stop_cm if target_stop_cm is not None else cfg.safety.target_stop_distance_cm
        blocked = blocked_cm if blocked_cm is not None else cfg.safety.obstacle_blocked_cm
        critical = critical_cm if critical_cm is not None else cfg.safety.obstacle_critical_cm
        safety_cfg = replace(
            cfg.safety,
            target_stop_distance_cm=target_stop,
            obstacle_blocked_cm=blocked,
            obstacle_critical_cm=critical,
        )
        cfg = replace(cfg, safety=safety_cfg)

    print("=" * 70)
    print("  PHASE 5: SUPERVISED AUTONOMOUS SEARCH & APPROACH MISSION")
    print("=" * 70)
    print(f"Target Object   : {target.upper()}")
    print(f"Target Standoff : {cfg.safety.target_stop_distance_cm:.1f} cm (Blocked: {cfg.safety.obstacle_blocked_cm:.1f} cm, Critical: {cfg.safety.obstacle_critical_cm:.1f} cm)")
    print(f"Sector Turn     : {cfg.motion.search_turn_pulse_sec:.2f}s @ {cfg.motion.turn_speed*100:.0f}% PWM")
    print(f"Centering Pulse : {cfg.motion.centering_turn_pulse_sec:.2f}s @ {cfg.motion.turn_speed*100:.0f}% PWM")
    print(f"Approach Pulse  : {cfg.motion.approach_pulse_sec:.2f}s @ {cfg.motion.approach_speed*100:.0f}% PWM")
    print(f"Execution Mode  : {'DRY RUN (Simulated)' if dry_run else 'REAL HARDWARE'}")
    print("=" * 70)

    if dry_run:
        motors = MockMotorController(initially_enabled=True)
        sensor = MockDistanceSensor(initial_distance_cm=55.0)
        camera = MockCamera(width=640, height=480)
        detector = MockDetector()
        from app.hardware.interfaces import DetectionResult
        detector.detect = lambda frame: [DetectionResult(class_name=target, confidence=0.90, box_xywh=(320, 240, 100, 200), box_xyxy=(270, 140, 370, 340), class_id=12)]
        gpio = None
    else:
        gpio = RPiGPIOAdapter()
        motors = L298NMotorController(gpio=gpio, config=cfg.motor, motion_config=cfg.motion)
        motors.initialize()
        motors.stop()
        sensor = HCSR04UltrasonicSensor(gpio=gpio, config=cfg.ultrasonic, safety_config=cfg.safety)
        cam_backend = OpenCVCameraBackend(
            device_index=cfg.camera.device_index,
            width=cfg.camera.width,
            height=cfg.camera.height,
            fps=cfg.camera.fps,
        )
        camera = CameraOwner(backend=cam_backend, max_staleness_sec=0.50)
        detector = ONNXYOLODetector(
            model_path=_ROOT / cfg.vision.model_path,
            confidence_threshold=cfg.vision.confidence_threshold,
            nms_threshold=cfg.vision.nms_threshold,
        )

    fsm = RobotStateMachine(
        config=cfg,
        motors=motors,
        distance_sensor=sensor,
        camera=camera,
        detector=detector,
    )

    try:
        print("\n[STEP 1] Preflight Initialization...")
        if not dry_run:
            motors.initialize()
            motors.stop()
            sensor.initialize()
            sensor.start()
            if not camera.start():
                print("[ERROR] Camera failed to initialize.")
                return 1
            detector.load_model()
            time.sleep(0.50)  # Settle sensors

        init_dist = sensor.get_reading()
        dist_str = f"{init_dist.distance_cm:.1f} cm" if init_dist and init_dist.distance_cm else "UNKNOWN"
        print(f"  Ultrasonic Distance: {dist_str} (State: {init_dist.state.value if init_dist else 'NONE'})")
        print("  Motors: De-energized (0% PWM, all pins LOW).")

        print("\n[STEP 2] Standby: Starting autonomous mission in 3 seconds...")
        print("  EMERGENCY PROCEDURE: Press Ctrl+C or disconnect battery cutoff to abort.")
        for count in range(3, 0, -1):
            print(f"  {count}...")
            time.sleep(1.0)

        if not dry_run:
            motors.enable(True)

        print(f"\n  >>> LAUNCHING MISSION: SEARCH FOR '{target.upper()}' <<<\n")
        fsm.start_search_mission(target_class=target)

        step_count = 0
        while step_count < max_steps:
            step_count += 1
            current_state = fsm.state
            reading = sensor.get_reading() if sensor else None
            dist_val = f"{reading.distance_cm:.1f} cm" if reading and reading.distance_cm else "N/A"

            det_info = ""
            if fsm.context and fsm.context.target_detection:
                td = fsm.context.target_detection
                det_info = f" | Target: {td.class_name} ({td.confidence:.2f}) cx={td.box_xywh[0]:.1f} offset={td.box_xywh[0]-320.0:+.1f}px"

            print(f"[Step {step_count:2d}] State: {current_state.value:<16} | Distance: {dist_val:<8}{det_info}")

            if current_state in (RobotState.TARGET_REACHED, RobotState.TARGET_NOT_FOUND, RobotState.BLOCKED, RobotState.FAULT, RobotState.CANCELLED):
                break

            fsm.step_mission()
            time.sleep(0.05)

        print("\n" + "=" * 70)
        final_state = fsm.state
        print(f"Mission Finished with State: {final_state.value}")
        if fsm.context and fsm.context.terminal_message:
            print(f"Details: {fsm.context.terminal_message}")
        if final_state == RobotState.TARGET_REACHED:
            final_dist = sensor.get_reading() if sensor else None
            final_cm = f"{final_dist.distance_cm:.1f} cm" if final_dist and final_dist.distance_cm else "N/A"
            print(f">>> SUCCESS: TARGET REACHED SAFELY AT STANDOFF DISTANCE (Measured Distance: {final_cm})! <<<")
            return 0
        elif final_state == RobotState.BLOCKED:
            print(">>> NOTICE: Motion stopped due to obstacle blocking the path. <<<")
            return 0
        else:
            print(f">>> Mission ended with status: {final_state.value} <<<")
            return 0

    except KeyboardInterrupt:
        print("\n\n[ABORT] Operator emergency interrupt (Ctrl+C) received! Halting immediately.")
        return 130
    except Exception as exc:
        print(f"\n[FAILURE] Exception during autonomous mission: {exc}")
        return 1
    finally:
        try:
            if motors:
                motors.stop()
                if hasattr(motors, "enable"):
                    motors.enable(False)
                motors.cleanup()
            if sensor and hasattr(sensor, "stop"):
                sensor.stop()
            if camera and hasattr(camera, "release"):
                camera.release()
            if detector and hasattr(detector, "close"):
                detector.close()
        except Exception:
            pass
        print("[CLEANUP] Motors stopped, sensors de-energized, and GPIO pins released.\n")


def main() -> int:
    parser = argparse.ArgumentParser(description="Phase 5 Autonomous Search & Approach Mission Tool")
    parser.add_argument("--target", type=str, default="bottle", help="Target object class to search and approach")
    parser.add_argument("--dry-run", action="store_true", help="Simulate with mock hardware")
    parser.add_argument("--max-steps", type=int, default=60, help="Maximum FSM step iterations")
    parser.add_argument("--target-stop", type=float, default=None, help="Target standoff distance in cm")
    parser.add_argument("--blocked", type=float, default=None, help="Obstacle blocked distance in cm")
    parser.add_argument("--critical", type=float, default=None, help="Obstacle critical distance in cm")
    args = parser.parse_args()

    return run_mission(
        target=args.target,
        dry_run=args.dry_run,
        max_steps=args.max_steps,
        target_stop_cm=args.target_stop,
        blocked_cm=args.blocked,
        critical_cm=args.critical,
    )


if __name__ == "__main__":
    raise SystemExit(main())
