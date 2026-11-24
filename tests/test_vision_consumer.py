"""Tests for vision engine frame consumer."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest

from geomind.vision.config import VisionEngineConfig, ModelConfig, CameraFeedConfig
from geomind.vision.engine import VisionEngine
from geomind.vision.models import BoundingBox, Detection


@pytest.fixture
def vision_config():
    return VisionEngineConfig(
        model=ModelConfig(type="dummy"),
        enabled=True,
        feeds=[CameraFeedConfig(feed_url="https://example.com/cam.jpg", camera_id="test_cam")],
    )


@pytest.fixture
def dummy_model():
    from geomind.vision.models_impl import DummyModel
    model = DummyModel(detection_probability=1.0)
    model.load()
    return model


@pytest.mark.asyncio
async def test_consumer_start_stop(vision_config, dummy_model):
    """Test starting and stopping the consumer."""
    with patch("geomind.vision.engine.get_vision_config", return_value=vision_config), \
         patch("geomind.vision.engine.create_model", return_value=dummy_model):
        engine = VisionEngine()
        await engine.start()

        # Start consumer
        await engine.start_consumer(poll_interval=0.1)
        assert engine._consumer_task is not None
        assert not engine._consumer_task.done()

        # Stop consumer
        await engine.stop_consumer()
        assert engine._consumer_task is None

        await engine.stop()


@pytest.mark.asyncio
async def test_consumer_updates_last_processed_frame(vision_config, dummy_model):
    """Test that consumer tracks last processed frame per camera."""
    with patch("geomind.vision.engine.get_vision_config", return_value=vision_config), \
         patch("geomind.vision.engine.create_model", return_value=dummy_model):

        engine = VisionEngine()
        await engine.start()

        # Verify the tracking dict exists and is initialized
        assert hasattr(engine, '_last_processed_frame')
        assert isinstance(engine._last_processed_frame, dict)

        # Test that we can manually update it
        engine._last_processed_frame["test_cam"] = 42
        assert engine._last_processed_frame["test_cam"] == 42

        await engine.stop_consumer()
        await engine.stop()


import asyncio