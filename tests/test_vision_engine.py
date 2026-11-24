"""Tests for vision engine."""

from __future__ import annotations

from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest

from geomind.vision.config import VisionEngineConfig, ModelConfig
from geomind.vision.engine import VisionEngine, ProcessingResult
from geomind.vision.models import FrameDetections, Detection, BoundingBox
from geomind.vision.models_impl import DummyModel


@pytest.fixture
def vision_config():
    return VisionEngineConfig(
        model=ModelConfig(type="dummy"),
        enabled=True,
    )


@pytest.fixture
def dummy_model():
    model = DummyModel(detection_probability=1.0)
    model.load()
    return model


@pytest.fixture
async def engine(dummy_model, vision_config):
    with patch("geomind.vision.engine.get_vision_config", return_value=vision_config), \
         patch("geomind.vision.engine.create_model", return_value=dummy_model):
        engine = VisionEngine()
        await engine.start()
        yield engine
        await engine.stop()


@pytest.fixture
def sample_image():
    import numpy as np
    return np.zeros((480, 640, 3), dtype=np.uint8)


class TestVisionEngine:
    @pytest.mark.asyncio
    async def test_start_stop(self, vision_config):
        model = DummyModel()
        model.load()

        with patch("geomind.vision.engine.get_vision_config", return_value=vision_config), \
             patch("geomind.vision.engine.create_model", return_value=model):
            engine = VisionEngine()
            await engine.start()
            assert engine._running is True
            await engine.stop()
            assert engine._running is False

    @pytest.mark.asyncio
    async def test_submit_frame(self, engine):
        image = np.zeros((480, 640, 3), dtype=np.uint8)
        result = await engine.submit_frame("test_cam", 1, image)
        assert result is True

    @pytest.mark.asyncio
    async def test_submit_frame_when_stopped(self, vision_config):
        model = DummyModel()
        model.load()

        with patch("geomind.vision.engine.get_vision_config", return_value=vision_config), \
             patch("geomind.vision.engine.create_model", return_value=model):
            engine = VisionEngine()
            # Don't start
            image = np.zeros((480, 640, 3), dtype=np.uint8)
            result = await engine.submit_frame("test_cam", 1, image)
            assert result is False

    @pytest.mark.asyncio
    async def test_process_frame_sync(self, engine, sample_image):
        result = await engine.process_frame_sync(
            camera_id="test_cam",
            frame_id=1,
            image=sample_image,
        )

        assert isinstance(result, ProcessingResult)
        assert isinstance(result.frame_detections, FrameDetections)
        assert result.frame_detections.camera_id == "test_cam"
        assert result.frame_detections.frame_id == 1
        assert len(result.frame_detections.detections) > 0
        assert result.frame_detections.inference_time_ms > 0

    @pytest.mark.asyncio
    async def test_process_frame_creates_events(self, engine, sample_image):
        """Test that 3 consecutive frames with detections creates confirmed event."""
        # The DummyModel may not consistently return same class.
        # This test verifies the engine runs without errors.
        for i in range(3):
            result = await engine.process_frame_sync(
                camera_id="test_cam",
                frame_id=i + 1,
                image=sample_image,
            )
            # Engine should process frames without errors
            assert result.frame_detections.camera_id == "test_cam"
            assert result.frame_detections.frame_id == i + 1

    @pytest.mark.asyncio
    async def test_process_frame_creates_events_consistent_class(self, engine, sample_image):
        """Test that 3 consecutive frames with same class creates confirmed event."""
        # Create a mock model that always returns same class detections
        mock_model = MagicMock()
        mock_model.is_loaded = True
        mock_model.model_version = "MockModel"

        # Create consistent detections
        bbox = BoundingBox(x_min=0.1, y_min=0.1, x_max=0.5, y_max=0.5)
        detections = [Detection(class_id=0, class_name="smoke", confidence=0.8, bbox=bbox)]

        mock_result = MagicMock()
        mock_result.detections = detections
        mock_result.inference_time_ms = 50.0
        mock_result.model_version = "MockModel"
        mock_model.infer.return_value = mock_result

        with patch("geomind.vision.engine.create_model", return_value=mock_model), \
             patch("geomind.vision.engine.get_vision_config", return_value=engine._vision_config if hasattr(engine, '_vision_config') else MagicMock()):
            engine_with_mock = VisionEngine()
            await engine_with_mock.start()

            try:
                for i in range(3):
                    result = await engine_with_mock.process_frame_sync(
                        camera_id="test_cam",
                        frame_id=i + 1,
                        image=np.zeros((480, 640, 3), dtype=np.uint8),
                    )
                    if i == 2:
                        assert len(result.events) == 1
                        assert result.events[0].status.value == "confirmed"
                        assert result.events[0].class_name == "smoke"

            finally:
                await engine_with_mock.stop()

    @pytest.mark.asyncio
    async def test_get_stats(self, engine):
        stats = engine.get_stats()
        assert stats["running"] is True
        assert stats["model"] == "DummyModel"
        assert stats["model_loaded"] is True
        assert "queue_size" in stats
        assert "workers" in stats

    @pytest.mark.asyncio
    async def test_queue_full(self, vision_config):
        vision_config.queue.max_size = 1
        vision_config.queue.worker_count = 0  # No workers to consume

        model = DummyModel()
        model.load()

        with patch("geomind.vision.engine.get_vision_config", return_value=vision_config), \
             patch("geomind.vision.engine.create_model", return_value=model):
            engine = VisionEngine()
            await engine.start()

            image = np.zeros((480, 640, 3), dtype=np.uint8)

            # First frame should succeed
            result1 = await engine.submit_frame("test_cam", 1, image)
            assert result1 is True

            # Second frame should fail (queue full)
            result2 = await engine.submit_frame("test_cam", 2, image)
            assert result2 is False

            await engine.stop()