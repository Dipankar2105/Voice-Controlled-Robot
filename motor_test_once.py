import time
from app.config import load_config
from app.constants import ExecutionMode
from app.hardware.gpio_adapter import get_gpio_adapter
from app.hardware.motors import L298NMotorController

motor = None

try:
    cfg = load_config()
    gpio = get_gpio_adapter(ExecutionMode.REAL_HARDWARE)

    motor = L298NMotorController(
        gpio=gpio,
        config=cfg.motor,
        motion_config=cfg.motion,
    )
    motor.initialize()
    motor.stop()

    print("Motor driver initialized. Wheels must be off the ground.")
    print("Look at BOTH LEFT and RIGHT wheels. Pulsing FORWARD in 5 seconds...")
    for c in range(5, 0, -1):
        print(f"  {c}...")
        time.sleep(1.0)
    print("  >>> PULSING BOTH WHEELS FORWARD NOW! <<<")

    motor.enable(True)
    motor.move_pulse(0.40, 0.40, 0.50)
    motor.stop()
    motor.enable(False)

    print("Pulse complete. Motor driver disabled.")

except KeyboardInterrupt:
    print("Test cancelled.")

finally:
    if motor is not None:
        try:
            motor.stop()
            motor.enable(False)
            motor.cleanup()
        except Exception as exc:
            print("Cleanup warning:", exc)
