import sys
import time
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from app.config import load_config
from app.constants import ExecutionMode
from app.hardware.gpio_adapter import get_gpio_adapter
from app.hardware.ultrasonic import HCSR04UltrasonicSensor

def main():
    cfg = load_config()
    gpio = get_gpio_adapter(ExecutionMode.REAL_HARDWARE)
    sensor = HCSR04UltrasonicSensor(gpio=gpio, config=cfg.ultrasonic, safety_config=cfg.safety)
    sensor.initialize()

    print("Live ultrasonic sampling started (10 seconds)...")
    t_end = time.time() + 10.0
    sample_num = 0

    while time.time() < t_end:
        sample_num += 1
        r = sensor.sample_distance_cm()
        if r.is_valid and r.distance_cm is not None:
            print(f"#{sample_num:02d}: {r.distance_cm:5.1f} cm | State: {r.state.value:8s} | VALID [OK]")
        else:
            d_str = f"{r.distance_cm:5.1f} cm" if r.distance_cm is not None else "None    "
            print(f"#{sample_num:02d}: {d_str} | State: {r.state.value:8s} | INVALID (Out of range / timeout)")
        time.sleep(0.15)

    sensor.stop()
    print("Sampling complete. Pins released.")

if __name__ == "__main__":
    main()
