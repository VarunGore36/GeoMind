"""Tests for vision model implementations."""

from __future__ import annotations

import numpy as np
import pytest

from geomind.vision.config import ModelConfig
from geomind.vision.models_impl import DummyModel, YOLOModel, create_model
from geomind.vision.interfaces import BaseModelInterface


class TestDummyModel:
    @pytest.fixture
    def model(self):
        return DummyModel(detection_probability=1.0)  # Always detect

    def test_load_and_warmup(self, model):
        assert not model.is_loaded
        model.load()
        assert model.is_loaded
        model.warmup()  # Should not raise

    def test_infer_returns_detections(self, model):
        model.load()
        image = np.zeros((480, 640, 3), dtype=np.uint8)
        result = model.infer(image)

        assert hasattr(result, "detections")
        assert hasattr(result, "inference_time_ms")
        assert hasattr(result, "model_version")
        assert isinstance(result.detections, list)
        assert result.inference_time_ms > 0

    def test_unload(self, model):
        model.load()
        model.unload()
        assert not model.is_loaded

    def test_detection_probability_zero(self):
        """With probability 0, no detections should be returned."""
        model = DummyModel(detection_probability=0.0)
        model.load()
        image = np.zeros((480, 640, 3), dtype=np.uint8)
        result = model.infer(image)
        assert len(result.detections) == 0

    def test_detection_probability_one(self):
        """With probability 1, detections should always be returned."""
        model = DummyModel(detection_probability=1.0)
        model.load()
        image = np.zeros((480, 640, 3), dtype=np.uint8)
        result = model.infer(image)
        assert len(result.detections) > 0


class TestModelFactory:
    def test_create_dummy_model(self):
        config = ModelConfig(type="dummy", weights_path="test")
        model = create_model(config)
        assert isinstance(model, DummyModel)

    def test_create_yolo_model(self):
        config = ModelConfig(type="yolo", weights_path="test")
        # This will fail if ultralytics not installed, which is expected
        try:
            model = create_model(config)
            assert isinstance(model, YOLOModel)
        except RuntimeError as e:
            assert "ultralytics not installed" in str(e)

    def test_unknown_model_type(self):
        # Pydantic will validate at config creation time
        with pytest.raises(Exception):
            ModelConfig(type="unknown")


class TestModelInterface:
    def test_base_model_interface_cannot_instantiate(self):
        with pytest.raises(TypeError):
            BaseModelInterface()

    def test_model_version_property(self):
        model = DummyModel()
        assert model.model_version == "DummyModel"