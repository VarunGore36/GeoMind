"""Concrete model implementations."""

from __future__ import annotations

import random
from typing import Any, Optional

import numpy as np

from .interfaces import BaseModelInterface, InferenceResult
from .models import BoundingBox, Detection


class DummyModel(BaseModelInterface):
    """
    Dummy model for testing - returns random detections.

    Useful for testing the pipeline without a real model.
    """

    def __init__(
        self,
        weights_path: str = "dummy",
        device: str = "cpu",
        confidence_threshold: float = 0.25,
        iou_threshold: float = 0.45,
        input_size: int = 640,
        detection_probability: float = 0.3,
        **kwargs,
    ):
        super().__init__(
            weights_path=weights_path,
            device=device,
            confidence_threshold=confidence_threshold,
            iou_threshold=iou_threshold,
            input_size=input_size,
            **kwargs,
        )
        self.detection_probability = detection_probability

    def _load_model(self) -> Any:
        # No actual model to load
        return "dummy_model"

    def _run_inference(self, image: np.ndarray) -> list[Detection]:
        """Generate random detections for testing."""
        h, w = image.shape[:2]

        # Randomly decide whether to generate a detection
        if random.random() > self.detection_probability:
            return []

        # Generate 1-3 random detections
        num_detections = random.randint(1, 3)
        detections = []

        for _ in range(num_detections):
            # Random class (0=smoke, 1=fire for testing)
            class_id = random.choice([0, 1])
            class_name = "smoke" if class_id == 0 else "fire"

            # Random confidence above threshold
            confidence = random.uniform(self.confidence_threshold, 1.0)

            # Random bounding box
            x_min = random.uniform(0.0, 0.7)
            y_min = random.uniform(0.0, 0.7)
            x_max = random.uniform(x_min + 0.1, min(x_min + 0.3, 1.0))
            y_max = random.uniform(y_min + 0.1, min(y_min + 0.3, 1.0))

            detections.append(
                Detection(
                    class_id=class_id,
                    class_name=class_name,
                    confidence=confidence,
                    bbox=BoundingBox(
                        x_min=x_min,
                        y_min=y_min,
                        x_max=x_max,
                        y_max=y_max,
                    ),
                )
            )

        return detections


class YOLOModel(BaseModelInterface):
    """
    YOLO-based model implementation using Ultralytics YOLO.

    Requires: pip install ultralytics
    """

    def __init__(
        self,
        weights_path: str = "yolov8n.pt",
        device: str = "cpu",
        confidence_threshold: float = 0.25,
        iou_threshold: float = 0.45,
        input_size: int = 640,
        half_precision: bool = False,
        classes: list[int] | None = None,
        **kwargs,
    ):
        super().__init__(
            weights_path=weights_path,
            device=device,
            confidence_threshold=confidence_threshold,
            iou_threshold=iou_threshold,
            input_size=input_size,
            half_precision=half_precision,
            classes=classes or [],
            **kwargs,
        )
        self._yolo_model: Any = None

    def _load_model(self) -> Any:
        try:
            from ultralytics import YOLO
        except ImportError:
            raise RuntimeError(
                "ultralytics not installed. Install with: pip install ultralytics"
            )

        model = YOLO(self.weights_path)
        model.to(self.device)
        if self.half_precision and self.device != "cpu":
            model.half()
        return model

    def _run_inference(self, image: np.ndarray) -> list[Detection]:
        """Run YOLO inference and convert to Detection list."""
        import cv2

        # Preprocess
        processed = self._preprocess(image)
        h, w = image.shape[:2]

        # Run inference
        results = self._model(
            processed,
            conf=self.confidence_threshold,
            iou=self.iou_threshold,
            classes=self.classes if self.classes else None,
            verbose=False,
        )

        detections = []
        for result in results:
            if result.boxes is None:
                continue

            boxes = result.boxes.xyxyn.cpu().numpy()  # Normalized xyxy
            confidences = result.boxes.conf.cpu().numpy()
            class_ids = result.boxes.cls.cpu().numpy().astype(int)

            for box, conf, cls_id in zip(boxes, confidences, class_ids):
                x_min, y_min, x_max, y_max = box

                # Scale back to original image coordinates (approximately)
                # Since we used letterbox padding, we need to adjust
                # For simplicity, we use the normalized coords directly
                detections.append(
                    Detection(
                        class_id=int(cls_id),
                        class_name=self._get_class_name(int(cls_id)),
                        confidence=float(conf),
                        bbox=BoundingBox(
                            x_min=float(x_min),
                            y_min=float(y_min),
                            x_max=float(x_max),
                            y_max=float(y_max),
                        ),
                    )
                )

        return detections

    def _get_class_name(self, class_id: int) -> str:
        """Map YOLO class ID to name. Override for custom models."""
        # COCO class names - in practice, use model.names
        coco_names = {
            0: "person", 1: "bicycle", 2: "car", 3: "motorcycle",
            4: "airplane", 5: "bus", 6: "train", 7: "truck",
            8: "boat", 9: "traffic light", 10: "fire hydrant",
        }
        # For wildfire detection, we'd use a custom model with classes like "smoke", "fire"
        return f"class_{class_id}"


class ONNXModel(BaseModelInterface):
    """
    ONNX Runtime model implementation for wildfire smoke detection.

    Supports ONNX models exported from YOLO, custom CNNs, or other architectures.
    Includes proper letterbox preprocessing and NMS postprocessing.
    """

    # Default class names for wildfire detection
    DEFAULT_CLASSES = ["smoke", "fire"]

    def __init__(
        self,
        weights_path: str,
        device: str = "cpu",
        confidence_threshold: float = 0.25,
        iou_threshold: float = 0.45,
        input_size: int = 640,
        half_precision: bool = False,
        classes: list[int] | None = None,
        class_names: list[str] | None = None,
        providers: list[str] | None = None,
        **kwargs,
    ):
        super().__init__(
            weights_path=weights_path,
            device=device,
            confidence_threshold=confidence_threshold,
            iou_threshold=iou_threshold,
            input_size=input_size,
            half_precision=half_precision,
            classes=classes or [],
            **kwargs,
        )
        self.class_names = class_names or self.DEFAULT_CLASSES
        self.providers = providers or self._get_default_providers(device)
        self._session: Optional[Any] = None
        self._input_name: Optional[str] = None
        self._output_names: Optional[list[str]] = None

    def _get_default_providers(self, device: str) -> list[str]:
        """Get default ONNX Runtime providers based on device."""
        if device == "cuda":
            return ["CUDAExecutionProvider", "CPUExecutionProvider"]
        return ["CPUExecutionProvider"]

    def _load_model(self) -> Any:
        import onnxruntime as ort

        session_options = ort.SessionOptions()
        session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        self._session = ort.InferenceSession(
            self.weights_path,
            sess_options=session_options,
            providers=self.providers,
        )

        # Get input/output names
        self._input_name = self._session.get_inputs()[0].name
        self._output_names = [out.name for out in self._session.get_outputs()]

        return self._session

    def _preprocess(self, image: np.ndarray) -> np.ndarray:
        """
        Preprocess image with letterbox padding to maintain aspect ratio.

        Returns:
            Preprocessed image tensor (1, 3, H, W) normalized to [0, 1]
        """
        import cv2

        h, w = image.shape[:2]
        scale = min(self.input_size / w, self.input_size / h)
        new_w, new_h = int(w * scale), int(h * scale)

        # Resize with aspect ratio preserved
        resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

        # Create letterboxed image
        padded = np.full((self.input_size, self.input_size, 3), 114, dtype=np.uint8)
        pad_top = (self.input_size - new_h) // 2
        pad_left = (self.input_size - new_w) // 2
        padded[pad_top:pad_top + new_h, pad_left:pad_left + new_w] = resized

        # Convert to tensor: HWC -> CHW, normalize to [0, 1]
        tensor = padded.transpose(2, 0, 1).astype(np.float32) / 255.0
        tensor = np.expand_dims(tensor, axis=0)  # Add batch dimension

        # Store preprocessing info for postprocessing
        self._preprocess_info = {
            "scale": scale,
            "pad_left": pad_left,
            "pad_top": pad_top,
            "original_w": w,
            "original_h": h,
        }

        return tensor

    def _run_inference(self, image: np.ndarray) -> list[Detection]:
        """Run ONNX inference and convert to Detection list."""
        if self._session is None:
            self.load()

        # Preprocess
        input_tensor = self._preprocess(image)

        # Run inference
        outputs = self._session.run(self._output_names, {self._input_name: input_tensor})

        # Postprocess - handle different output formats
        detections = self._postprocess(outputs[0])

        return detections

    def _postprocess(self, output: np.ndarray) -> list[Detection]:
        """
        Postprocess ONNX output to Detection list.

        Handles YOLO-style output: (batch, num_boxes, 5 + num_classes)
        where each box is [x_center, y_center, width, height, obj_conf, class_probs...]
        """
        # output shape: (1, num_boxes, 5 + num_classes) or (num_boxes, 5 + num_classes)
        if output.ndim == 3:
            output = output[0]  # Remove batch dimension

        num_classes = output.shape[1] - 5
        if num_classes <= 0:
            return []

        # Extract boxes, objectness, class probs
        boxes = output[:, :4]  # x_center, y_center, width, height (normalized)
        obj_conf = output[:, 4]
        class_probs = output[:, 5:]

        # Get class with highest probability
        class_ids = np.argmax(class_probs, axis=1)
        class_conf = class_probs[np.arange(len(class_ids)), class_ids]

        # Final confidence = obj_conf * class_conf
        confidences = obj_conf * class_conf

        # Filter by confidence threshold
        mask = confidences >= self.confidence_threshold
        boxes = boxes[mask]
        confidences = confidences[mask]
        class_ids = class_ids[mask]

        if len(boxes) == 0:
            return []

        # Convert from center format to corner format (normalized)
        x_center, y_center, width, height = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
        x_min = x_center - width / 2
        y_min = y_center - height / 2
        x_max = x_center + width / 2
        y_max = y_center + height / 2

        # Apply letterbox correction
        info = getattr(self, "_preprocess_info", {})
        scale = info.get("scale", 1.0)
        pad_left = info.get("pad_left", 0)
        pad_top = info.get("pad_top", 0)
        orig_w = info.get("original_w", self.input_size)
        orig_h = info.get("original_h", self.input_size)

        # Correct coordinates for letterbox padding
        x_min = (x_min * self.input_size - pad_left) / scale / orig_w
        y_min = (y_min * self.input_size - pad_top) / scale / orig_h
        x_max = (x_max * self.input_size - pad_left) / scale / orig_w
        y_max = (y_max * self.input_size - pad_top) / scale / orig_h

        # Clip to valid range
        x_min = np.clip(x_min, 0.0, 1.0)
        y_min = np.clip(y_min, 0.0, 1.0)
        x_max = np.clip(x_max, 0.0, 1.0)
        y_max = np.clip(y_max, 0.0, 1.0)

        # Apply NMS
        keep_indices = self._nms(
            list(zip(x_min, y_min, x_max, y_max)),
            confidences,
            self.iou_threshold,
        )

        detections = []
        for idx in keep_indices:
            class_id = int(class_ids[idx])
            class_name = self._get_class_name(class_id)
            detections.append(
                Detection(
                    class_id=class_id,
                    class_name=class_name,
                    confidence=float(confidences[idx]),
                    bbox=BoundingBox(
                        x_min=float(x_min[idx]),
                        y_min=float(y_min[idx]),
                        x_max=float(x_max[idx]),
                        y_max=float(y_max[idx]),
                    ),
                )
            )

        return detections

    def _nms(
        self,
        boxes: list[tuple[float, float, float, float]],
        scores: np.ndarray,
        iou_threshold: float,
    ) -> list[int]:
        """Non-Maximum Suppression."""
        if len(boxes) == 0:
            return []

        boxes_arr = np.array(boxes)
        x1 = boxes_arr[:, 0]
        y1 = boxes_arr[:, 1]
        x2 = boxes_arr[:, 2]
        y2 = boxes_arr[:, 3]

        areas = (x2 - x1) * (y2 - y1)
        order = scores.argsort()[::-1]

        keep = []
        while order.size > 0:
            i = order[0]
            keep.append(i)

            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])

            w = np.maximum(0.0, xx2 - xx1)
            h = np.maximum(0.0, yy2 - yy1)
            inter = w * h
            iou = inter / (areas[i] + areas[order[1:]] - inter)

            order = order[1:][iou <= iou_threshold]

        return keep

    def _get_class_name(self, class_id: int) -> str:
        """Map class ID to name."""
        if 0 <= class_id < len(self.class_names):
            return self.class_names[class_id]
        return f"class_{class_id}"

    def unload(self) -> None:
        self._session = None
        self._input_name = None
        self._output_names = None


def create_model(model_config) -> BaseModelInterface:
    """
    Factory function to create model instance from config.

    Args:
        model_config: ModelConfig instance with type and parameters.

    Returns:
        ModelInterface instance.
    """
    model_type = model_config.type.lower()

    if model_type == "dummy":
        return DummyModel(
            weights_path=model_config.weights_path,
            device=model_config.device,
            confidence_threshold=model_config.confidence_threshold,
            iou_threshold=model_config.iou_threshold,
            input_size=model_config.input_size,
            half_precision=model_config.half_precision,
            classes=model_config.classes,
        )
    elif model_type in ("yolo", "ultralytics"):
        return YOLOModel(
            weights_path=model_config.weights_path,
            device=model_config.device,
            confidence_threshold=model_config.confidence_threshold,
            iou_threshold=model_config.iou_threshold,
            input_size=model_config.input_size,
            half_precision=model_config.half_precision,
            classes=model_config.classes,
        )
    elif model_type in ("onnx", "onnxruntime"):
        return ONNXModel(
            weights_path=model_config.weights_path,
            device=model_config.device,
            confidence_threshold=model_config.confidence_threshold,
            iou_threshold=model_config.iou_threshold,
            input_size=model_config.input_size,
            half_precision=model_config.half_precision,
            classes=model_config.classes,
            class_names=getattr(model_config, 'class_names', None),
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")