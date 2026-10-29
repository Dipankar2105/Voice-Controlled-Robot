#!/usr/bin/env python3
"""Phase 3 Live Interactive Voice Command Execution Tool.

Features:
- Live streaming from Logitech C270 USB microphone.
- Real-time offline speech recognition using Vosk Kaldi grammar.
- Deterministic CommandParser routing.
- Motor actuation for verified movement commands:
  - 'move backward' / 'backward' -> Reverse pulse
  - 'turn left' / 'left' -> CCW rotation pulse
  - 'turn right' / 'right' -> CW rotation pulse
  - 'move forward' / 'forward' -> Forward pulse (strictly gated by ultrasonic <= 25cm interlock)
  - 'stop' / 'halt' / 'emergency stop' -> Immediate stop
- Safe termination on Ctrl+C or timeout with full GPIO pin cleanup.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from app.audio.audio_source import MicrophoneAudioSource, find_logitech_c270_device_index
from app.audio.vosk_recognizer import VoskSpeechRecognizer
from app.command.command_parser import CommandAction, CommandIntent, CommandParser
from app.config import load_config
from app.constants import ExecutionMode, RobotState
from app.hardware.gpio_adapter import FakeGPIOAdapter, RPiGPIOAdapter
from app.hardware.motors import L298NMotorController
from app.hardware.ultrasonic import HCSR04UltrasonicSensor
from app.navigation.state_machine import RobotStateMachine


def run_live_voice_session(duration_sec: float = 30.0, dry_run: bool = False) -> int:
    cfg = load_config()

    print("=" * 70)
    print("  PHASE 3: LIVE INTERACTIVE VOSK VOICE CONTROL")
    print("=" * 70)
    print("Available Spoken Commands:")
    print("  Directional : 'move forward', 'move backward', 'turn left', 'turn right'")
    print("  Safety      : 'stop', 'halt', 'emergency stop'")
    print(f"Safety Gate : Ultrasonic Obstacle Interlock (< {cfg.safety.obstacle_blocked_cm} cm blocks forward)")
    print(f"Session Time: {duration_sec:.0f} seconds (or press Ctrl+C to finish)")
    print(f"Execution   : {'DRY RUN (FakeGPIO)' if dry_run or not sys.platform.startswith('linux') else 'REAL HARDWARE (RPiGPIO)'}")
    print("=" * 70)

    if dry_run or not sys.platform.startswith("linux"):
        gpio = FakeGPIOAdapter()
    else:
        gpio = RPiGPIOAdapter()

    # Hardware components
    motors = L298NMotorController(gpio=gpio, config=cfg.motor, motion_config=cfg.motion)
    sensor = HCSR04UltrasonicSensor(gpio=gpio, config=cfg.ultrasonic, safety_config=cfg.safety)
    fsm = RobotStateMachine(config=cfg, motors=motors, distance_sensor=sensor)
    parser = CommandParser()

    # Audio & Vosk
    print("\n[STEP 1] Initializing audio and loading Vosk speech engine...")
    import sounddevice as sd
    devices = list(sd.query_devices())
    dev_idx = find_logitech_c270_device_index(devices)
    if dev_idx is None:
        dev_idx = sd.default.device[0]

    audio_source = MicrophoneAudioSource(config=cfg.audio, device_index=dev_idx)
    speech_recognizer = VoskSpeechRecognizer(config=cfg.audio)
    speech_recognizer.load_model()
    print("  Vosk model loaded and speech grammar compiled successfully.")

    try:
        print("\n[STEP 2] Initializing motor driver and ultrasonic sensor...")
        motors.initialize()
        motors.stop()
        sensor.initialize()
        sensor.start()  # Start continuous ~14 Hz background median-filtered acoustic sampling
        time.sleep(0.35)  # Allow initial median filter window to fill
        init_reading = sensor.get_reading()
        init_dist = f"{init_reading.distance_cm:.1f} cm" if init_reading.distance_cm is not None else "NONE"
        print(f"  Ultrasonic background worker active. Current reading: {init_dist} ({init_reading.state.value})")

        print("\n[STEP 3] Starting live audio capture...")
        audio_source.start_stream()
        print("\n" + "=" * 70)
        print("  >>> [READY] ROBOT IS LISTENING! SPEAK INTO MICROPHONE NOW! <<<")
        print("=" * 70 + "\n")

        t_end = time.time() + duration_sec
        last_partial = ""

        while time.time() < t_end:
            time_left = int(t_end - time.time())
            chunk = audio_source.read_chunk()
            if not chunk:
                time.sleep(0.01)
                continue

            accepted = speech_recognizer.accept_waveform(chunk)
            if accepted:
                transcript = speech_recognizer.get_result()
                if transcript and transcript.strip():
                    print(f"\n[VOICE HEARD] Transcript: '{transcript}'")
                    cmd = parser.parse(transcript)
                    print(f"  Parsed Action : {cmd.action.value} (Intent: {cmd.intent.value})")

                    if cmd.action == CommandAction.EMERGENCY_STOP:
                        print("  >>> EMERGENCY STOP ACTIVATED! <<<")
                        fsm.handle_command(cmd)
                        motors.stop()
                    elif cmd.action == CommandAction.STOP:
                        print("  >>> STOP COMMANDED. Halting motors. <<<")
                        fsm.handle_command(cmd)
                        motors.stop()
                    elif cmd.intent == CommandIntent.MOVEMENT:
                        # Forward obstacle check via synchronized median-filtered reading
                        if cmd.action == CommandAction.MOVE_FORWARD:
                            reading = sensor.get_reading()
                            dist_cm = reading.distance_cm if reading.distance_cm is not None else 999.0
                            print(f"  Forward ultrasonic reading: {dist_cm:.1f} cm (State: {reading.state.value})")
                            if reading.state.value in ("BLOCKED", "CRITICAL") or dist_cm <= cfg.safety.obstacle_blocked_cm:
                                print(f"  >>> SAFETY INTERLOCK BLOCKED: Obstacle at {dist_cm:.1f} cm <= {cfg.safety.obstacle_blocked_cm} cm! FORWARD REFUSED. <<<")
                                motors.stop()
                                continue

                        print(f"  >>> EXECUTING MOVEMENT: {cmd.action.value} <<<")
                        fsm.handle_command(cmd)
                        print(f"  >>> MOVEMENT PULSE COMPLETED. MOTORS STOPPED. <<<\n")
                    else:
                        print(f"  Command '{cmd.action.value}' acknowledged (no motor actuation).")
            else:
                partial = speech_recognizer.get_partial_result()
                if partial and partial != last_partial:
                    last_partial = partial
                    # Print partial cleanly in terminal
                    print(f"  [listening...] {partial}", end="\r", flush=True)

        print("\n[SESSION FINISHED] Live voice session duration completed.")
        return 0

    except KeyboardInterrupt:
        print("\n\n[ABORT] Voice session interrupted by operator (Ctrl+C).")
        return 130
    except Exception as exc:
        print(f"\n[ERROR] Exception during live voice session: {exc}")
        return 1
    finally:
        try:
            motors.stop()
            motors.cleanup()
            sensor.stop()
            audio_source.stop_stream()
            audio_source.close()
        except Exception:
            pass
        print("[CLEANUP] Motors zeroed, audio closed, and GPIO pins released.\n")


def main() -> int:
    parser = argparse.ArgumentParser(description="Live Interactive Voice Control")
    parser.add_argument("--seconds", type=float, default=30.0, help="Live session timeout in seconds")
    parser.add_argument("--dry-run", action="store_true", help="Simulate with FakeGPIO")
    args = parser.parse_args()
    return run_live_voice_session(duration_sec=args.seconds, dry_run=args.dry_run)


if __name__ == "__main__":
    raise SystemExit(main())
