import sys, time
from pathlib import Path
_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))
from app.config import load_config
from app.hardware.gpio_adapter import RPiGPIOAdapter
from app.hardware.motors import L298NMotorController

cfg = load_config()
print("Loaded config:")
print(f"  Left ENA={cfg.motor.left_ena_pin}, IN1={cfg.motor.left_in1_pin}, IN2={cfg.motor.left_in2_pin}")
print(f"  Right ENB={cfg.motor.right_enb_pin}, IN3={cfg.motor.right_in3_pin}, IN4={cfg.motor.right_in4_pin}")
print(f"  PWM freq={cfg.motor.pwm_frequency_hz}")
