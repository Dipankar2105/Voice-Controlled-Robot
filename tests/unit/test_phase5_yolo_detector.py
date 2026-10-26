"""Phase 5 unit tests: YOLO ONNX object detector, metadata inspection,
class-count verification, coordinate conversion, NMS, and target filtering.
"""
from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch
import cv2
import numpy as np
import pytest

from app.config import VisionConfig, get_project_root
from app.hardware.interfaces import DetectionResult
from app.vision.yolo_detector import (
    ONNXYOLODetector,
    is_target_match,
    normalize_target_query,
)


@pytest.fixture
def project_root() -> Path:
    return get_project_root()


@pytest.fixture
def sample_fixture_image(project_root: Path) -> np.ndarray:
    img_path = project_root / "tests" / "fixtures" / "images" / "live_view_now.jpg"
    assert img_path.is_file(), f"Fixture image missing: {img_path}"
    img = cv2.imread(str(img_path))
    assert img is not None
    return img


# ---------------------------------------------------------------------------
# 1. Metadata and Class Count Tests
# ---------------------------------------------------------------------------

def test_v2_model_metadata_and_class_count(project_root: Path):
    pytest.importorskip("onnxruntime", reason="onnxruntime is not installed in this test environment")
    """best_v2_target.onnx must have 26 classes, including RED_TARGET, BLUE_TARGET, and scissors."""
    model_path = project_root / "models" / "object_detection" / "best_v2_target.onnx"
    detector = ONNXYOLODetector(model_path=model_path)
    detector.load_model()

    assert detector.is_ready is True
    classes = detector.class_names
    assert len(classes) == 26

    # Verify critical target classes and their exact indices
    assert detector.has_class("RED_TARGET") is True
    assert detector.has_class("BLUE_TARGET") is True
    assert detector.has_class("scissors") is True

    assert classes[4] == "RED_TARGET"
    assert classes[5] == "BLUE_TARGET"
    assert classes[25] == "scissors"
    assert classes[0] == "person"
    assert classes[1] == "chair"

    detector.close()


def test_v1_model_metadata_and_class_count(project_root: Path):
    pytest.importorskip("onnxruntime", reason="onnxruntime is not installed in this test environment")
    """best_v1.onnx must have 25 classes and must NOT include scissors."""
    model_path = project_root / "models" / "object_detection" / "best_v1.onnx"
    detector = ONNXYOLODetector(model_path=model_path)
    detector.load_model()

    assert detector.is_ready is True
    classes = detector.class_names
    assert len(classes) == 25

    assert detector.has_class("RED_TARGET") is True
    assert detector.has_class("BLUE_TARGET") is True
    assert detector.has_class("scissors") is False

    assert classes[4] == "RED_TARGET"
    assert classes[5] == "BLUE_TARGET"

    detector.close()


# ---------------------------------------------------------------------------
# 2. Known Fixture Inference and Expected Detections
# ---------------------------------------------------------------------------

def test_yolo_detection_on_known_fixture(sample_fixture_image: np.ndarray, project_root: Path):
    pytest.importorskip("onnxruntime", reason="onnxruntime is not installed in this test environment")
    """Inference on live_view_now.jpg detects person and chair above confidence threshold."""
    model_path = project_root / "models" / "object_detection" / "best_v2_target.onnx"
    detector = ONNXYOLODetector(model_path=model_path, confidence_threshold=0.25, nms_threshold=0.45)
    detector.load_model()

    detections = detector.detect(sample_fixture_image)
    assert len(detections) >= 1

    # Check top detection
    top_det = detections[0]
    assert top_det.class_name == "person"
    assert top_det.confidence >= 0.50

    # Bounding box bounds check
    x1, y1, x2, y2 = top_det.box_xyxy
    assert 0.0 <= x1 < x2 <= 640.0
    assert 0.0 <= y1 < y2 <= 480.0

    # Mathematical consistency
    cx, cy, w, h = top_det.box_xywh
    assert abs(cx - (x1 + x2) / 2.0) <= 1.0
    assert abs(cy - (y1 + y2) / 2.0) <= 1.0
    assert abs(w - (x2 - x1)) <= 1.0
    assert abs(h - (y2 - y1)) <= 1.0

    # Target specific query
    person_det = detector.detect_target(sample_fixture_image, "person")
    assert person_det is not None
    assert person_det.class_name == "person"

    chair_det = detector.detect_target(sample_fixture_image, "chair")
    assert chair_det is not None
    assert chair_det.class_name == "chair"

    detector.close()


# ---------------------------------------------------------------------------
# 3. Coordinate Conversion & Rescaling Math
# ---------------------------------------------------------------------------

def test_coordinate_conversion_and_rescaling(project_root: Path):
    """Bounding box coordinates from 320x320 model space rescale correctly to 640x480 frame."""
    # Synthetic frame: 640x480
    frame = np.zeros((480, 640, 3), dtype=np.uint8)

    detector = ONNXYOLODetector()
    # Mock session output
    # Synthetic box at (cx=160, cy=160, w=80, h=80) in 320x320 space
    # Class 0 ('person') score = 0.90
    num_anchors = 2100
    num_classes = 26
    synthetic_output = np.zeros((1, 4 + num_classes, num_anchors), dtype=np.float32)

    # Put detection at anchor 0
    synthetic_output[0, 0, 0] = 160.0  # cx
    synthetic_output[0, 1, 0] = 160.0  # cy
    synthetic_output[0, 2, 0] = 80.0   # w
    synthetic_output[0, 3, 0] = 80.0   # h
    synthetic_output[0, 4, 0] = 0.90   # class 0 ('person')

    mock_session = MagicMock()
    mock_session.run.return_value = [synthetic_output]
    mock_session.get_inputs.return_value = [MagicMock(name="images", shape=[1, 3, 320, 320])]
    mock_session.get_outputs.return_value = [MagicMock(name="output0", shape=[1, 30, 2100])]
    mock_session.get_modelmeta.return_value.custom_metadata_map = {
        "names": "{0: 'person', 1: 'chair'}"
    }

    detector._session = mock_session
    detector._input_name = "images"
    detector._output_name = "output0"
    detector._id_to_name = {0: "person", 1: "chair"}
    detector._name_to_id = {"person": 0, "chair": 1}
    detector._is_ready = True

    results = detector.detect(frame)
    assert len(results) == 1

    det = results[0]
    # 640x480 is letterboxed to 320x320 with scale 0.5 and 40px top/bottom padding.
    # Model center (160,160) maps to source center (320,240); 80x80 maps to 160x160.
    assert det.box_xywh == (320.0, 240.0, 160.0, 160.0)
    assert det.box_xyxy == (240.0, 160.0, 400.0, 320.0)
    assert det.confidence == 0.90
    assert det.class_name == "person"


def test_class_aware_nms_preserves_overlapping_different_classes():
    """Two overlapping anchors from different classes must not suppress one another."""
    detector = ONNXYOLODetector()
    num_anchors = 2100
    synthetic_output = np.zeros((1, 4 + 26, num_anchors), dtype=np.float32)
    for anchor, cid, score in ((0, 0, 0.90), (1, 1, 0.85)):
        synthetic_output[0, 0, anchor] = 160.0
        synthetic_output[0, 1, anchor] = 160.0
        synthetic_output[0, 2, anchor] = 80.0
        synthetic_output[0, 3, anchor] = 80.0
        synthetic_output[0, 4 + cid, anchor] = score
    mock_session = MagicMock()
    mock_session.run.return_value = [synthetic_output]
    detector._session = mock_session
    detector._input_name = "images"
    detector._output_name = "output0"
    detector._id_to_name = {0: "person", 1: "chair"}
    detector._name_to_id = {"person": 0, "chair": 1}
    detector._is_ready = True

    results = detector.detect(np.zeros((480, 640, 3), dtype=np.uint8))
    assert {result.class_name for result in results} == {"person", "chair"}


def test_nonfinite_model_predictions_are_discarded():
    """NaN/Inf outputs must fail closed rather than crash during box conversion."""
    detector = ONNXYOLODetector()
    output = np.zeros((1, 4 + 26, 2100), dtype=np.float32)
    output[0, 0, 0] = np.nan
    output[0, 1, 0] = 160.0
    output[0, 2, 0] = 80.0
    output[0, 3, 0] = 80.0
    output[0, 4, 0] = 0.99
    mock_session = MagicMock()
    mock_session.run.return_value = [output]
    detector._session = mock_session
    detector._input_name = "images"
    detector._output_name = "output0"
    detector._id_to_name = {0: "person"}
    detector._name_to_id = {"person": 0}
    detector._is_ready = True
    assert detector.detect(np.zeros((480, 640, 3), dtype=np.uint8)) == []


# ---------------------------------------------------------------------------
# 4. Target Query Normalization and Alias Matching
# ---------------------------------------------------------------------------

def test_target_query_normalization_and_aliases():
    """Target name queries match across plural, prefix, and alias variations."""
    assert normalize_target_query("find scissors") == "scissors"
    assert normalize_target_query("look for the scissor") == "scissors"
    assert normalize_target_query("search for shears") == "scissors"
    assert normalize_target_query("the bottle") == "bottle"
    assert normalize_target_query("water bottle") == "bottle"
    assert normalize_target_query("red target") == "red_target"
    assert normalize_target_query("RED_TARGET") == "red_target"
    assert normalize_target_query("blue") == "blue_target"
    assert normalize_target_query("cell phone") == "cell_phone"

    # Match tests
    assert is_target_match("scissors", "scissor") is True
    assert is_target_match("scissors", "find the shears") is True
    assert is_target_match("RED_TARGET", "red") is True
    assert is_target_match("RED_TARGET", "red target") is True
    assert is_target_match("bottle", "water bottle") is True
    assert is_target_match("person", "find a person") is True

    # Negative matches
    assert is_target_match("person", "bottle") is False
    assert is_target_match("RED_TARGET", "blue target") is False


def test_unsupported_class_requests(sample_fixture_image: np.ndarray, project_root: Path):
    pytest.importorskip("onnxruntime", reason="onnxruntime is not installed in this test environment")
    """Unsupported classes safely return None without throwing or hallucinating."""
    model_path = project_root / "models" / "object_detection" / "best_v2_target.onnx"
    detector = ONNXYOLODetector(model_path=model_path)
    detector.load_model()

    # Request an object definitely not in the room
    det = detector.detect_target(sample_fixture_image, "airplane")
    assert det is None

    # Check has_class for nonexistent class
    assert detector.has_class("rocket") is False
    detector.close()


# ---------------------------------------------------------------------------
# 5. Missing Model Fails Closed & Exception Handling
# ---------------------------------------------------------------------------

def test_missing_model_file_fails_closed(tmp_path: Path):
    """Missing model file raises FileNotFoundError and leaves detector unready."""
    fake_path = tmp_path / "nonexistent_model.onnx"
    detector = ONNXYOLODetector(model_path=fake_path)

    with pytest.raises(FileNotFoundError, match="YOLO ONNX model file not found"):
        detector.load_model()

    assert detector.is_ready is False
    assert detector.class_names == []


def test_invalid_input_frames_handled_safely(project_root: Path):
    """None or invalid dimension inputs return empty detections without crashing."""
    model_path = project_root / "models" / "object_detection" / "best_v2_target.onnx"
    detector = ONNXYOLODetector(model_path=model_path)

    # Invalid frames are rejected before model loading or native runtime access.
    # Pass None
    assert detector.detect(None) == []

    # Pass 2D grayscale array
    assert detector.detect(np.zeros((480, 640), dtype=np.uint8)) == []

    # Pass 4D array
    assert detector.detect(np.zeros((1, 480, 640, 3), dtype=np.uint8)) == []

    # Pass zero-sized array
    assert detector.detect(np.zeros((0, 0, 3), dtype=np.uint8)) == []

    detector.close()


# ---------------------------------------------------------------------------
# 6. Performance Timing
# ---------------------------------------------------------------------------

def test_detector_performance_timing(sample_fixture_image: np.ndarray, project_root: Path):
    pytest.importorskip("onnxruntime", reason="onnxruntime is not installed in this test environment")
    """Detector records inference latency timing without claiming real-time performance."""
    model_path = project_root / "models" / "object_detection" / "best_v2_target.onnx"
    detector = ONNXYOLODetector(model_path=model_path)
    detector.load_model()

    detector.detect(sample_fixture_image)
    assert detector.last_inference_latency_ms > 0.0
    assert detector.last_inference_latency_ms < 5000.0  # Reasonable sanity bound

    detector.close()
