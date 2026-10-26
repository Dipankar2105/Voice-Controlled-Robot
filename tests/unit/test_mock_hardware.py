"""Unit tests for SS_PROJECT_VCR_FRESH hardware abstraction and mocks."""
import numpy as np
import pytest

from app.hardware.interfaces import ObstacleState, UltrasonicReading
from app.hardware.mock_hardware import (
    MockAudioSource,
    MockCamera,
    MockDistanceSensor,
    MockMotorController,
)


def test_mock_motor_enable_and_clamping():
    """Mock motor controller must strictly respect enable gate and speed bounds [-1.0, 1.0]."""
    motors = MockMotorController(initially_enabled=False)
    assert not motors.is_enabled

    # Setting speeds while disabled has zero effect
    motors.set_speeds(0.8, 0.8)
    assert motors.current_speeds == (0.0, 0.0)

    # Enable and test valid speeds
    motors.enable(True)
    assert motors.is_enabled
    motors.set_speeds(0.5, -0.5)
    assert motors.current_speeds == (0.5, -0.5)

    # Test clamping beyond [-1.0, 1.0]
    motors.set_speeds(1.5, -2.0)
    assert motors.current_speeds == (1.0, -1.0)

    # Test stop
    motors.stop()
    assert motors.current_speeds == (0.0, 0.0)
    assert motors.stop_count == 1

    # Disabling stops motors immediately
    motors.enable(True)
    motors.set_speeds(0.6, 0.6)
    motors.enable(False)
    assert motors.current_speeds == (0.0, 0.0)


def test_mock_distance_sensor_classification():
    """Distance measurements must be deterministically classified into correct ObstacleState."""
    sensor = MockDistanceSensor(
        warning_cm=50.0,
        blocked_cm=25.0,
        critical_cm=15.0,
        min_cm=2.0,
        max_cm=400.0,
    )

    # Clear zone
    sensor.set_distance(60.0)
    r = sensor.get_reading()
    assert r.state == ObstacleState.CLEAR
    assert r.allows_forward_motion
    assert r.is_valid

    # Warning zone
    sensor.set_distance(35.0)
    r = sensor.get_reading()
    assert r.state == ObstacleState.WARNING
    assert r.allows_forward_motion
    assert r.is_valid

    # Blocked zone
    sensor.set_distance(20.0)
    r = sensor.get_reading()
    assert r.state == ObstacleState.BLOCKED
    assert not r.allows_forward_motion
    assert r.is_valid

    # Critical zone
    sensor.set_distance(10.0)
    r = sensor.get_reading()
    assert r.state == ObstacleState.CRITICAL
    assert not r.allows_forward_motion
    assert r.is_valid

    # Transducer Blind Zone (< 2.0 cm) -> Fail-closed CRITICAL
    sensor.set_distance(1.5)
    r = sensor.get_reading()
    assert r.state == ObstacleState.CRITICAL
    assert not r.allows_forward_motion

    # Out of Range (> 400.0 cm)
    sensor.set_distance(450.0)
    r = sensor.get_reading()
    assert r.state == ObstacleState.OUT_OF_RANGE
    assert not r.allows_forward_motion

    # Sensor Fault / Timeout (None or negative)
    sensor.set_distance(None)
    r = sensor.get_reading()
    assert r.state == ObstacleState.FAULT
    assert not r.allows_forward_motion
    assert not r.is_valid


def test_mock_distance_sensor_sequence():
    """Sensor correctly steps through a programmed distance sequence."""
    sensor = MockDistanceSensor()
    sensor.set_sequence([100.0, 40.0, 10.0])

    assert sensor.get_reading().state == ObstacleState.CLEAR
    assert sensor.get_reading().state == ObstacleState.WARNING
    assert sensor.get_reading().state == ObstacleState.CRITICAL


def test_mock_camera_frame_capture_and_injection():
    """Mock camera returns properly shaped numpy arrays and supports frame injection."""
    cam = MockCamera(width=640, height=480)
    assert cam.is_opened()

    frame = cam.capture_frame()
    assert frame is not None
    assert frame.shape == (480, 640, 3)
    assert frame.dtype == np.uint8

    # Frame injection
    test_img = np.ones((480, 640, 3), dtype=np.uint8) * 255
    cam.inject_frame(test_img)
    captured = cam.capture_frame()
    assert np.array_equal(captured, test_img)

    # Frame drop simulation
    cam.inject_frame(None)
    assert cam.capture_frame() is None

    # Release
    cam.release()
    assert not cam.is_opened()
    assert cam.capture_frame() is None


def test_mock_audio_source_pcm_streaming():
    """Mock audio source streams queued PCM chunks."""
    audio = MockAudioSource(chunk_size=1024, sample_rate=16000)

    # Reading while stream is stopped returns None
    assert audio.read_chunk() is None

    audio.start_stream()
    # Generates silence chunk by default (1024 samples * 2 bytes = 2048 bytes)
    chunk = audio.read_chunk()
    assert chunk is not None
    assert len(chunk) == 2048
    assert chunk == b"\x00" * 2048

    # Enqueue custom data
    custom_pcm = b"\x01\x02" * 1024
    audio.queue_pcm_data(custom_pcm)
    assert audio.read_chunk() == custom_pcm

    audio.stop_stream()
    assert audio.read_chunk() is None

    audio.close()
    assert audio._is_closed
