"""Concrete model implementations."""

from __future__ import annotations

import random
from typing import Any

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
    else:
        raise ValueError(f"Unknown model type: {model_type}")