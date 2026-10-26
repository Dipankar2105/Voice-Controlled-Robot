"""Hardware Abstraction Layer (HAL) for SS_PROJECT_VCR_FRESH."""
from app.hardware.gpio_adapter import (
    FakeGPIOAdapter,
    GPIOAdapter,
    RPiGPIOAdapter,
    get_gpio_adapter,
)
from app.hardware.interfaces import (
    AudioSourceInterface,
    CameraInterface,
    DetectionResult,
    DetectorInterface,
    DistanceSensorInterface,
    MotorControllerInterface,
    ObstacleState,
    SpeechRecognizerInterface,
    UltrasonicReading,
)
from app.hardware.mock_hardware import (
    MockAudioSource,
    MockCamera,
    MockDetector,
    MockDistanceSensor,
    MockMotorController,
    MockSpeechRecognizer,
)
from app.hardware.motors import L298NMotorController
from app.hardware.ultrasonic import HCSR04UltrasonicSensor

__all__ = [
    # Interfaces & Value Types
    "MotorControllerInterface",
    "DistanceSensorInterface",
    "CameraInterface",
    "AudioSourceInterface",
    "DetectorInterface",
    "SpeechRecognizerInterface",
    "ObstacleState",
    "UltrasonicReading",
    "DetectionResult",
    # Hardware Adapters
    "GPIOAdapter",
    "FakeGPIOAdapter",
    "RPiGPIOAdapter",
    "get_gpio_adapter",
    "L298NMotorController",
    "HCSR04UltrasonicSensor",
    # Simulation Mocks
    "MockMotorController",
    "MockDistanceSensor",
    "MockCamera",
    "MockAudioSource",
    "MockDetector",
    "MockSpeechRecognizer",
]
