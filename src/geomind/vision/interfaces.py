"""Vision Engine model interfaces for interchangeable models."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np


@dataclass(frozen=True, slots=True)
class InferenceResult:
    """Result of a single inference call."""

    detections: list[Any]  # Detection objects
    inference_time_ms: float
    model_version: str


class Detection(Protocol):
    """Protocol for detection objects."""
    class_id: int
    class_name: str
    confidence: float
    bbox: Any


class BoundingBox(Protocol):
    """Protocol for bounding box."""
    x_min: float
    y_min: float
    x_max: float
    y_max: float


class ModelInterface(Protocol):
    """Protocol for interchangeable detection models."""

    @property
    def model_version(self) -> str: ...

    @property
    def input_size(self) -> int: ...

    @property
    def confidence_threshold(self) -> float: ...

    @property
    def iou_threshold(self) -> float: ...

    @abstractmethod
    def load(self) -> None:
        """Load the model into memory."""

    @abstractmethod
    def warmup(self) -> None:
        """Run a dummy inference to warm up the model."""

    @abstractmethod
    def infer(self, image: np.ndarray) -> InferenceResult:
        """
        Run inference on a single image.

        Args:
            image: Input image as numpy array (H, W, C) in BGR format.

        Returns:
            InferenceResult with detections and metadata.
        """

    @abstractmethod
    def infer_batch(self, images: list[np.ndarray]) -> list[InferenceResult]:
        """
        Run inference on a batch of images.

        Args:
            images: List of input images as numpy arrays.

        Returns:
            List of InferenceResult for each image.
        """

    @abstractmethod
    def unload(self) -> None:
        """Unload the model and free resources."""


class BaseModelInterface(ABC):
    """Base class for model implementations."""

    def __init__(
        self,
        weights_path: str,
        device: str = "cpu",
        confidence_threshold: float = 0.25,
        iou_threshold: float = 0.45,
        input_size: int = 640,
        half_precision: bool = False,
        classes: list[int] | None = None,
    ):
        self.weights_path = weights_path
        self.device = device
        self.confidence_threshold = confidence_threshold
        self.iou_threshold = iou_threshold
        self.input_size = input_size
        self.half_precision = half_precision
        self.classes = classes or []
        self._loaded = False
        self._model: Any = None

    @property
    def model_version(self) -> str:
        return self.__class__.__name__

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    @abstractmethod
    def _load_model(self) -> Any:
        """Load the actual model. Implemented by subclasses."""

    @abstractmethod
    def _run_inference(self, image: np.ndarray) -> list[Detection]:
        """Run inference on preprocessed image. Returns list of Detection."""

    def load(self) -> None:
        if self._loaded:
            return
        self._model = self._load_model()
        self._loaded = True

    def warmup(self) -> None:
        if not self._loaded:
            self.load()
        # Create dummy image and run inference
        import numpy as np
        dummy = np.zeros((self.input_size, self.input_size, 3), dtype=np.uint8)
        self._run_inference(dummy)

    def infer(self, image: np.ndarray) -> "InferenceResult":
        if not self._loaded:
            self.load()

        import time
        start = time.perf_counter()

        detections = self._run_inference(image)

        inference_time_ms = (time.perf_counter() - start) * 1000

        return InferenceResult(
            detections=detections,
            inference_time_ms=inference_time_ms,
            model_version=self.model_version,
        )

    def infer_batch(self, images: list[np.ndarray]) -> list["InferenceResult"]:
        return [self.infer(img) for img in images]

    def unload(self) -> None:
        self._model = None
        self._loaded = False

    def _preprocess(self, image: np.ndarray) -> np.ndarray:
        """Preprocess image for inference. Override if needed."""
        import cv2
        # Resize with letterboxing to maintain aspect ratio
        h, w = image.shape[:2]
        scale = min(self.input_size / w, self.input_size / h)
        new_w, new_h = int(w * scale), int(h * scale)
        resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

        # Create padded image
        padded = np.full((self.input_size, self.input_size, 3), 114, dtype=np.uint8)
        padded[:new_h, :new_w] = resized

        return padded

    def _postprocess(
        self,
        raw_output: Any,
        original_shape: tuple[int, int],
    ) -> list[Detection]:
        """Postprocess model output to Detection list. Override in subclasses."""
        raise NotImplementedError