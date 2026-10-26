"""Entrypoint for SS_PROJECT_VCR_FRESH.

Coordinates CLI arguments, preflight configuration checks, and the unified runtime lifecycle.
Guarantees:
- Single authoritative runtime instance.
- No real motors or physical GPIO energized in SIMULATION mode or preflight checks.
- Clean shutdown on SIGINT/SIGTERM releasing all resources.
- Browser/kiosk is NOT automatically opened unless explicitly configured.
"""
from __future__ import annotations

import argparse
import os
from dataclasses import replace
import signal
import sys
import webbrowser
from pathlib import Path

# Ensure project root is in sys.path for direct CLI execution
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from app.config import ConfigurationError, load_config
from app.constants import ExecutionMode
from app.logger import get_logger
from app.runtime import RobotRuntime

logger = get_logger("main")
__version__ = "2.0.0"


def check_config(config_path: str | None = None) -> int:
    """Perform purely static configuration validation without hardware initialization."""
    base_dir = Path(__file__).resolve().parent.parent
    try:
        cfg = load_config(config_path)
    except ConfigurationError as exc:
        print(f"CONFIGURATION ERROR: {exc}", file=sys.stderr)
        return 2
    except Exception as exc:
        print(f"ERROR: Failed to load configuration: {exc}", file=sys.stderr)
        return 1

    missing_assets = cfg.validate_asset_files(base_dir)
    if missing_assets:
        print("PRESERVED ASSET CHECK FAILED:", file=sys.stderr)
        for missing in missing_assets:
            print(f"  - {missing}", file=sys.stderr)
        return 3

    print(f"SS_PROJECT_VCR_FRESH v{__version__} — Configuration Preflight")
    print(f"  - Robot Name: {cfg.robot_name}")
    print(f"  - Mode: {cfg.mode.value}")
    print(f"  - GPIO Pins: ENA={cfg.motor.left_ena_pin}, IN1={cfg.motor.left_in1_pin}, IN2={cfg.motor.left_in2_pin}")
    print(f"               ENB={cfg.motor.right_enb_pin}, IN3={cfg.motor.right_in3_pin}, IN4={cfg.motor.right_in4_pin}")
    print(f"               TRIG={cfg.ultrasonic.trig_pin}, ECHO={cfg.ultrasonic.echo_pin}")
    print(f"  - PWM Frequency: {cfg.motor.pwm_frequency_hz} Hz")
    print(f"  - Speeds: Manual={cfg.motion.manual_speed}, Turn={cfg.motion.turn_speed}, Approach={cfg.motion.approach_speed}, Search={cfg.motion.search_speed}")
    print(f"  - Safety Thresholds: Warning={cfg.safety.obstacle_warning_cm} cm, Target stop={cfg.safety.target_stop_distance_cm} cm (provisional), Blocked={cfg.safety.obstacle_blocked_cm} cm, Critical={cfg.safety.obstacle_critical_cm} cm")
    print("  - Hardware Initialization: NONE (Static preflight only)")
    print("Preflight check PASSED.")
    return 0


def run_server(
    config_path: str | None = None,
    host: str = "0.0.0.0",
    port: int = 8000,
    open_browser: bool = False,
    mode_override: str | None = None,
) -> int:
    """Run unified robot runtime and web dashboard server with clean shutdown handling."""
    import uvicorn
    from app.server import create_app

    cfg = load_config(config_path)
    if mode_override:
        cfg = replace(cfg, mode=ExecutionMode(mode_override))

    control_token = os.environ.get("ROBOT_API_TOKEN")
    if cfg.mode == ExecutionMode.REAL_HARDWARE and (not control_token or len(control_token) < 24):
        logger.error("REAL_HARDWARE mode requires ROBOT_API_TOKEN with at least 24 characters; no hardware was initialized.")
        return 2

    runtime = RobotRuntime(config=cfg)

    def shutdown_handler(signum, frame):
        logger.info("Signal %s received. Performing graceful runtime shutdown...", signum)
        runtime.cleanup()
        sys.exit(0)

    signal.signal(signal.SIGINT, shutdown_handler)
    signal.signal(signal.SIGTERM, shutdown_handler)

    try:
        logger.info("Initializing runtime...")
        runtime.initialize()
        logger.info("Starting runtime subsystems...")
        runtime.start()

        app = create_app(runtime, control_token=control_token)

        server_url = f"http://{host if host != '0.0.0.0' else '127.0.0.1'}:{port}"
        print(f"\n=======================================================")
        print(f" SS_PROJECT_VCR_FRESH Unified Runtime Dashboard Active")
        print(f" Mode: {cfg.mode.value}")
        print(f" Dashboard URL: {server_url}")
        print(f" REST API: {server_url}/api/status")
        print(f" Video Stream: {server_url}/api/camera/stream")
        print(f" Press Ctrl+C to stop cleanly.")
        print(f"=======================================================\n")

        if open_browser:
            webbrowser.open(server_url)

        uvicorn.run(app, host=host, port=port, log_level="warning")
        return 0
    except Exception as exc:
        logger.error("Runtime server encountered an error: %s", exc)
        return 1
    finally:
        logger.info("Shutting down and cleaning up runtime resources...")
        runtime.cleanup()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="SS_PROJECT_VCR_FRESH: Clean-room Voice Controlled Robot runtime."
    )
    parser.add_argument(
        "--version", action="version", version=f"SS_PROJECT_VCR_FRESH {__version__}"
    )
    parser.add_argument(
        "--check-config",
        action="store_true",
        help="Perform static configuration check and verify preserved asset presence without hardware actuation.",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to custom configuration YAML file.",
    )
    parser.add_argument(
        "--serve",
        action="store_true",
        help="Launch the authoritative runtime and local web dashboard server.",
    )
    parser.add_argument(
        "--host",
        type=str,
        default="0.0.0.0",
        help="Host interface to bind server (default: 0.0.0.0).",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Port number to bind server (default: 8000).",
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["SIMULATION", "REAL_HARDWARE"],
        default=None,
        help="Override execution mode (default: SIMULATION).",
    )
    parser.add_argument(
        "--open-browser",
        action="store_true",
        default=False,
        help="Automatically launch a local web browser tab upon server start (default: False).",
    )

    args = parser.parse_args(argv)

    if args.check_config:
        return check_config(args.config)

    if args.serve:
        return run_server(
            config_path=args.config,
            host=args.host,
            port=args.port,
            open_browser=args.open_browser,
            mode_override=args.mode,
        )

    print(f"SS_PROJECT_VCR_FRESH v{__version__} scaffold initialized.")
    print("Use --check-config to validate static configuration and asset integrity.")
    print("Use --serve to launch the unified runtime and local status dashboard.")
    print("No hardware initialized; no motors energized.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
