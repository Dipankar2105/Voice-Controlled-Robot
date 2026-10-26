"""YOLO ONNX Object Detector Subsystem for SS_PROJECT_VCR_FRESH.

Features:
- Executes offline inference via the verified ONNX Runtime (CPUExecutionProvider).
- Reads embedded model metadata ('names') to ensure 100% verified class-name-to-index mapping.
- Correctly decodes YOLOv8 outputs: [1, 4 + num_classes, num_anchors].
- Rescales normalized bounding boxes back to original optical frame coordinates (640x480).
- Applies confidence thresholding and Non-Maximum Suppression (NMS).
- Validates model architecture and fails closed if weights are missing or corrupted.
- Provides target name normalization and alias resolution for voice mission commands.
- Never falls back to fake detections in real hardware mode.
"""
from __future__ import annotations

import ast
from pathlib import Path
import time
from typing import Any, Dict, List, Optional, Tuple
import cv2
import numpy as np

from app.config import VisionConfig, get_project_root
from app.constants import (
    REL_PATH_FALLBACK_YOLO_ONNX,
    REL_PATH_PRIMARY_YOLO_ONNX,
    YOLO_CONFIDENCE_THRESHOLD,
    YOLO_INPUT_HEIGHT,
    YOLO_INPUT_WIDTH,
    YOLO_NMS_THRESHOLD,
)
from app.hardware.interfaces import DetectionResult, DetectorInterface
from app.logger import get_logger

logger = get_logger("detector")


def letterbox_image(frame: np.ndarray, target_width: int, target_height: int, fill_value: int = 114):
    """Aspect-preserving YOLO letterbox. Returns (image, scale, pad_left, pad_top)."""
    if frame is None or frame.size == 0:
        raise ValueError("Cannot preprocess an empty frame")
    src_h, src_w = frame.shape[:2]
    scale = min(target_width / float(src_w), target_height / float(src_h))
    resized_w = max(1, int(round(src_w * scale)))
    resized_h = max(1, int(round(src_h * scale)))
    resized = cv2.resize(frame, (resized_w, resized_h), interpolation=cv2.INTER_LINEAR)
    pad_w = target_width - resized_w
    pad_h = target_height - resized_h
    left = pad_w // 2
    right = pad_w - left
    top = pad_h // 2
    bottom = pad_h - top
    padded = cv2.copyMakeBorder(resized, top, bottom, left, right, cv2.BORDER_CONSTANT,
                                value=(fill_value, fill_value, fill_value))
    return padded, scale, float(left), float(top)


def normalize_target_query(name: str) -> str:
    """Normalize user or voice target queries for robust alias matching."""
    if not name:
        return ""
    q = str(name).lower().strip().replace("_", " ")

    # Strip conversational prefixes
    for prefix in ("search for ", "find ", "locate ", "look for ", "the ", "a ", "an "):
        if q.startswith(prefix):
            q = q[len(prefix):].strip()

    # Aliases
    if any(s in q for s in ("scissor", "shears")):
        return "scissors"
    if "bottle" in q:
        return "bottle"
    if q in ("red", "red target", "red object", "red_target"):
        return "red_target"
    if q in ("blue", "blue target", "blue object", "blue_target"):
        return "blue_target"
    if any(s in q for s in ("cell phone", "phone", "mobile")):
        return "cell_phone"
    if "trash" in q:
        return "trash_can"
    if "plant" in q:
        return "potted_plant"
    if "remote" in q:
        return "remote_control"
    if "extinguisher" in q:
        return "fire_extinguisher"
    if "board" in q:
        return "whiteboard"

    return q.replace(" ", "_")


def is_target_match(detected_class: str, requested_target: str) -> bool:
    """Check if a detected YOLO class name matches a requested search target."""
    c_norm = normalize_target_query(detected_class)
    r_norm = normalize_target_query(requested_target)
    return bool(c_norm and r_norm and c_norm == r_norm)


class ONNXYOLODetector(DetectorInterface):
    """Authoritative YOLO object detector using ONNX Runtime."""

    def __init__(
        self,
        config: Optional[VisionConfig] = None,
        model_path: Optional[Path | str] = None,
        confidence_threshold: Optional[float] = None,
        nms_threshold: Optional[float] = None,
        execution_providers: Optional[List[str]] = None,
    ):
        self.config = config or VisionConfig()
        root = get_project_root()

        if model_path is None:
            self.model_path = root / self.config.model_path
        else:
            p = Path(model_path)
            self.model_path = p if p.is_absolute() else (root / p)

        self.confidence_threshold = float(
            confidence_threshold if confidence_threshold is not None else self.config.confidence_threshold
        )
        self.nms_threshold = float(
            nms_threshold if nms_threshold is not None else self.config.nms_threshold
        )
        self.input_width = self.config.input_width
        self.input_height = self.config.input_height
        self.providers = execution_providers or ["CPUExecutionProvider"]

        self._ort = None
        self._session = None
        self._input_name: str = ""
        self._output_name: str = ""
        self._id_to_name: Dict[int, str] = {}
        self._name_to_id: Dict[str, int] = {}
        self._is_ready: bool = False
        self._last_inference_latency_ms: float = 0.0

    @property
    def is_ready(self) -> bool:
        """Returns True if ONNX model session is initialized and ready for inference."""
        return self._is_ready and self._session is not None

    @property
    def class_names(self) -> List[str]:
        """Return ordered list of verified class names supported by the loaded model."""
        if not self._is_ready:
            return []
        return [self._id_to_name[i] for i in sorted(self._id_to_name.keys())]

    @property
    def last_inference_latency_ms(self) -> float:
        """Latency in milliseconds of the most recent inference run."""
        return self._last_inference_latency_ms

    def load_model(self) -> None:
        """Initialize ONNX Runtime inference session and parse embedded class metadata."""
        if self._is_ready and self._session is not None:
            return

        logger.info("Loading YOLO ONNX model from '%s'...", self.model_path)
        if not self.model_path.is_file() or self.model_path.stat().st_size == 0:
            self._is_ready = False
            raise FileNotFoundError(
                f"YOLO ONNX model file not found: {self.model_path}. "
                "Silent fallback to fake detections is strictly prohibited."
            )

        try:
            import onnxruntime as ort
        except ImportError as exc:
            raise RuntimeError("The 'onnxruntime' package is required for YOLO inference. Install requirements.txt.") from exc
        self._ort = ort

        # Configure session options optimized for Raspberry Pi / CPU execution
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = 2  # Bound to 2 cores on Raspberry Pi 4 to prevent thermal throttling
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        try:
            self._session = ort.InferenceSession(
                str(self.model_path),
                sess_options=opts,
                providers=self.providers,
            )
        except Exception as exc:
            self._is_ready = False
            logger.error("Failed to initialize ONNX Runtime session: %s", exc)
            raise RuntimeError(f"Failed to load ONNX model {self.model_path}: {exc}") from exc

        # Validate input tensor
        inputs = self._session.get_inputs()
        if not inputs:
            raise RuntimeError(f"ONNX model {self.model_path} has no input tensors.")
        self._input_name = inputs[0].name
        input_shape = list(inputs[0].shape)
        expected_input = [1, 3, self.input_height, self.input_width]
        if len(input_shape) != 4 or any(
            isinstance(actual, int) and actual != expected
            for actual, expected in zip(input_shape, expected_input)
        ):
            raise RuntimeError(
                f"Unsupported YOLO input tensor shape {input_shape}; expected NCHW {expected_input}."
            )

        # Validate output tensor
        outputs = self._session.get_outputs()
        if not outputs:
            raise RuntimeError(f"ONNX model {self.model_path} has no output tensors.")
        self._output_name = outputs[0].name
        output_shape = list(outputs[0].shape)
        if len(output_shape) != 3:
            raise RuntimeError(f"Unsupported YOLO output tensor shape {output_shape}; expected [1, channels, anchors].")
        if isinstance(output_shape[0], int) and output_shape[0] != 1:
            raise RuntimeError(f"Unsupported YOLO batch size in output shape {output_shape}; expected batch 1.")

        # Parse embedded metadata directly from ONNX model
        meta = self._session.get_modelmeta().custom_metadata_map
        names_raw = meta.get("names", "")
        if not names_raw:
            raise ValueError(
                f"ONNX model {self.model_path} lacks required 'names' metadata dictionary. "
                "Cannot establish verified class-to-index mapping."
            )

        try:
            parsed_names = ast.literal_eval(names_raw)
            if not isinstance(parsed_names, dict):
                raise ValueError("Metadata 'names' is not a dictionary.")
        except Exception as exc:
            raise ValueError(f"Failed to parse model class names metadata: {exc}") from exc

        self._id_to_name = {int(k): str(v) for k, v in parsed_names.items()}
        if sorted(self._id_to_name) != list(range(len(self._id_to_name))):
            raise ValueError("ONNX class metadata IDs must be contiguous and zero-based.")
        channel_count = output_shape[1]
        if isinstance(channel_count, int) and channel_count != len(self._id_to_name) + 4:
            raise ValueError(
                f"ONNX output channel count {channel_count} does not match 4 box channels + "
                f"{len(self._id_to_name)} metadata classes. Unsupported output layout."
            )
        self._name_to_id = {v: k for k, v in self._id_to_name.items()}
        self._is_ready = True

        logger.info(
            "YOLO ONNX model loaded successfully (%d classes, input: %s, output: %s).",
            len(self._id_to_name),
            inputs[0].shape,
            outputs[0].shape,
        )

    def has_class(self, class_name: str) -> bool:
        """Check if the currently loaded model supports the specified class or alias."""
        if not self._is_ready:
            self.load_model()
        target_norm = normalize_target_query(class_name)
        for existing in self._id_to_name.values():
            if normalize_target_query(existing) == target_norm:
                return True
        return False

    def detect(self, frame: np.ndarray) -> List[DetectionResult]:
        """Perform object detection inference on an optical BGR frame [H, W, 3].
        
        Returns:
            List of DetectionResult instances post-NMS, sorted by confidence descending.
        """
        if frame is None or not isinstance(frame, np.ndarray) or frame.ndim != 3 or frame.shape[2] != 3:
            return []

        orig_h, orig_w = frame.shape[:2]
        if orig_h <= 0 or orig_w <= 0:
            return []

        if not self._is_ready or self._session is None:
            self.load_model()

        t0 = time.time()

        # Standard YOLO letterbox preserves aspect ratio; inverse geometry is
        # applied to every predicted box below.
        try:
            resized, scale, pad_left, pad_top = letterbox_image(frame, self.input_width, self.input_height)
        except Exception as exc:
            logger.error("YOLO preprocessing failed: %s", exc)
            return []
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        blob = np.ascontiguousarray(
            np.transpose(rgb.astype(np.float32) / 255.0, (2, 0, 1))[None, ...],
            dtype=np.float32,
        )

        # Inference
        try:
            outputs = self._session.run([self._output_name], {self._input_name: blob})
        except Exception as exc:
            logger.error("ONNX inference failed: %s", exc)
            self._last_inference_latency_ms = (time.time() - t0) * 1000.0
            raise

        self._last_inference_latency_ms = (time.time() - t0) * 1000.0

        raw_output = outputs[0]  # shape: [1, 4 + num_classes, num_anchors]
        if raw_output.ndim != 3 or raw_output.shape[0] != 1:
            logger.warning("Unexpected output tensor shape: %s", raw_output.shape)
            return []

        # Transpose to [num_anchors, 4 + num_classes]
        preds = raw_output[0].T  # e.g. [2100, 30]

        boxes = preds[:, :4]  # [cx, cy, w, h] in letterboxed model space
        scores = preds[:, 4:]  # class probabilities
        finite_rows = np.isfinite(boxes).all(axis=1) & np.isfinite(scores).all(axis=1)
        if not np.any(finite_rows):
            logger.warning("Detector output contained no finite predictions.")
            return []
        boxes = boxes[finite_rows]
        scores = scores[finite_rows]

        max_scores = np.max(scores, axis=1)
        max_cids = np.argmax(scores, axis=1)

        # Confidence filtering
        mask = max_scores >= self.confidence_threshold
        if not np.any(mask):
            return []

        filtered_boxes = boxes[mask]
        filtered_scores = max_scores[mask]
        filtered_cids = max_cids[mask]

        # Invert letterbox padding and uniform scale to recover original pixels.
        cx_orig = (filtered_boxes[:, 0] - pad_left) / scale
        cy_orig = (filtered_boxes[:, 1] - pad_top) / scale
        w_orig = filtered_boxes[:, 2] / scale
        h_orig = filtered_boxes[:, 3] / scale

        x1 = np.clip(cx_orig - (w_orig / 2.0), 0.0, float(orig_w))
        y1 = np.clip(cy_orig - (h_orig / 2.0), 0.0, float(orig_h))
        x2 = np.clip(cx_orig + (w_orig / 2.0), 0.0, float(orig_w))
        y2 = np.clip(cy_orig + (h_orig / 2.0), 0.0, float(orig_h))

        # Build candidate boxes for OpenCV NMS: [x, y, w, h]
        boxes_nms: List[List[int]] = []
        scores_nms: List[float] = []
        valid_indices: List[int] = []

        for i in range(len(filtered_scores)):
            bx1, by1, bx2, by2 = x1[i], y1[i], x2[i], y2[i]
            bw = bx2 - bx1
            bh = by2 - by1
            if bw < 2 or bh < 2:
                continue  # Filter out degenerated boxes
            boxes_nms.append([int(bx1), int(by1), int(bw), int(bh)])
            scores_nms.append(float(filtered_scores[i]))
            valid_indices.append(i)

        if not boxes_nms:
            return []

        # Class-aware NMS prevents a high-confidence class from suppressing a
        # different overlapping class (e.g. person in front of chair).
        kept_global_indices: List[int] = []
        valid_class_ids = [int(filtered_cids[i]) for i in valid_indices]
        for cid in sorted(set(valid_class_ids)):
            local_positions = [j for j, candidate_idx in enumerate(valid_indices)
                               if int(filtered_cids[candidate_idx]) == cid]
            class_boxes = [boxes_nms[j] for j in local_positions]
            class_scores = [scores_nms[j] for j in local_positions]
            indices = cv2.dnn.NMSBoxes(
                class_boxes, class_scores,
                score_threshold=self.confidence_threshold,
                nms_threshold=self.nms_threshold,
            )
            if len(indices) > 0:
                flat = indices.flatten() if hasattr(indices, "flatten") else [i[0] for i in indices]
                kept_global_indices.extend(valid_indices[local_positions[int(j)]] for j in flat)

        results: List[DetectionResult] = []
        for orig_idx in kept_global_indices:
            cid = int(filtered_cids[orig_idx])
            cname = self._id_to_name.get(cid, f"class_{cid}")
            conf = float(filtered_scores[orig_idx])
            bx1 = float(x1[orig_idx])
            by1 = float(y1[orig_idx])
            bx2 = float(x2[orig_idx])
            by2 = float(y2[orig_idx])
            bw = bx2 - bx1
            bh = by2 - by1
            bcx = bx1 + (bw / 2.0)
            bcy = by1 + (bh / 2.0)
            res = DetectionResult(
                class_id=cid,
                class_name=cname,
                confidence=round(conf, 4),
                box_xywh=(
                    round(bcx, 2),
                    round(bcy, 2),
                    round(bw, 2),
                    round(bh, 2),
                ),
                box_xyxy=(
                    round(bx1, 2),
                    round(by1, 2),
                    round(bx2, 2),
                    round(by2, 2),
                ),
            )
            results.append(res)

        # Sort by confidence descending
        results.sort(key=lambda r: r.confidence, reverse=True)
        return results

    def detect_target(self, frame: np.ndarray, target_name: str) -> Optional[DetectionResult]:
        """Find the highest confidence detection that matches the requested target query."""
        all_detections = self.detect(frame)
        for det in all_detections:
            if is_target_match(det.class_name, target_name):
                return det
        return None

    def close(self) -> None:
        """Release ONNX Runtime session."""
        self._session = None
        self._is_ready = False
        logger.info("ONNXYOLODetector closed.")
