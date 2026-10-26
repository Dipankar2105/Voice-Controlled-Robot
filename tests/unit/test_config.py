"""Unit tests for SS_PROJECT_VCR_FRESH configuration module."""
from pathlib import Path
import pytest

from app.config import (
    AppConfig,
    ConfigurationError,
    MotorDriverConfig,
    UltrasonicConfig,
    MotionConfig,
    SafetyConfig,
    CameraConfig,
    VisionConfig,
    AudioConfig,
    CommunicationConfig,
    load_config,
    load_config_from_dict,
)


def test_default_config_loading_and_validity():
    """Default configuration must load and pass all validations."""
    base_dir = Path(__file__).resolve().parent.parent.parent
    config_file = base_dir / "config" / "default_config.yaml"
    cfg = load_config(config_file)

    assert cfg.robot_name == "VoiceRobot4WD"
    assert cfg.motor.pwm_frequency_hz == 100
    assert cfg.motor.left_ena_pin == 18
    assert cfg.ultrasonic.trig_pin == 23
    assert cfg.ultrasonic.echo_pin == 24
    assert cfg.motion.manual_speed == 0.75
    assert cfg.motion.manual_pulse_sec == 0.75
    assert cfg.motion.turn_speed == 0.40
    assert cfg.safety.obstacle_critical_cm == 15.0
    assert cfg.safety.obstacle_blocked_cm == 25.0
    assert cfg.safety.obstacle_warning_cm == 50.0
    assert cfg.vision.input_width == 320
    assert cfg.audio.sample_rate == 16000


def test_duplicate_gpio_pins_rejected():
    """Attempting to assign the same GPIO pin to multiple functions must raise ConfigurationError."""
    # Set left ENA and ultrasonic TRIG to pin 18
    motor_cfg = MotorDriverConfig(left_ena_pin=18)
    ultrasonic_cfg = UltrasonicConfig(trig_pin=18)

    cfg = AppConfig(motor=motor_cfg, ultrasonic=ultrasonic_cfg)
    with pytest.raises(ConfigurationError, match="Duplicate GPIO pin assignment"):
        cfg.validate()


def test_invalid_pin_number_rejected():
    """Pins outside BCM 0-40 range must be rejected."""
    motor_cfg = MotorDriverConfig(left_ena_pin=99)
    cfg = AppConfig(motor=motor_cfg)
    with pytest.raises(ConfigurationError, match="Invalid BCM GPIO pin number"):
        cfg.validate()


def test_pwm_frequency_limits():
    """PWM frequency must be bounded within safe operational range."""
    cfg = AppConfig(motor=MotorDriverConfig(pwm_frequency_hz=5))
    with pytest.raises(ConfigurationError, match="outside safe operating range"):
        cfg.validate()


def test_speed_bounds_and_relationships():
    """Speeds must be strictly within (0.0, 1.0] and manual_speed <= max_speed."""
    cfg_zero = AppConfig(motion=MotionConfig(manual_speed=0.0))
    with pytest.raises(ConfigurationError, match="violates valid range"):
        cfg_zero.validate()

    cfg_exceed = AppConfig(motion=MotionConfig(manual_speed=0.9, max_speed=0.8))
    with pytest.raises(ConfigurationError, match="manual_speed .* cannot exceed max_speed"):
        cfg_exceed.validate()

    cfg_pulse_zero = AppConfig(motion=MotionConfig(manual_pulse_sec=0.0))
    with pytest.raises(ConfigurationError, match="manual_pulse_sec .* must be within"):
        cfg_pulse_zero.validate()


def test_obstacle_threshold_monotonicity():
    """Critical < blocked < target stop < warning so arrival cannot bypass the blocked zone."""
    # Warning smaller than blocked
    invalid_safety = SafetyConfig(
        obstacle_warning_cm=20.0,
        obstacle_blocked_cm=25.0,
        obstacle_critical_cm=15.0,
    )
    cfg = AppConfig(safety=invalid_safety)
    with pytest.raises(ConfigurationError, match="mandatory monotonic ordering"):
        cfg.validate()


def test_target_stop_must_be_outside_blocked_zone():
    """A target-stop threshold inside the blocked zone is unsafe and rejected."""
    cfg = AppConfig(safety=SafetyConfig(target_stop_distance_cm=20.0))
    with pytest.raises(ConfigurationError, match="mandatory monotonic ordering"):
        cfg.validate()


def test_vision_and_audio_invariants():
    """Vision model inputs and audio sample rate invariants must be enforced."""
    cfg_vis = AppConfig(vision=VisionConfig(input_width=640))
    with pytest.raises(ConfigurationError, match="must match YOLO ONNX contract 320x320"):
        cfg_vis.validate()

    cfg_audio = AppConfig(audio=AudioConfig(sample_rate=44100))
    with pytest.raises(ConfigurationError, match="strictly requires 16000 Hz"):
        cfg_audio.validate()


def test_asset_files_presence_check():
    """Validation method correctly checks presence of preserved model assets."""
    base_dir = Path(__file__).resolve().parent.parent.parent
    cfg = load_config(base_dir / "config" / "default_config.yaml")
    missing = cfg.validate_asset_files(base_dir)
    assert missing == [], f"Expected preserved assets to be present, but found missing: {missing}"
