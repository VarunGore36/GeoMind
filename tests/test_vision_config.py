"""Tests for vision engine configuration."""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest
import yaml

from geomind.vision.config import (
    VisionEngineConfig,
    ModelConfig,
    TemporalValidationConfig,
    get_vision_config,
)


class TestVisionConfig:
    def test_model_config_defaults(self):
        config = ModelConfig()
        assert config.type == "dummy"
        assert config.device == "cpu"
        assert config.confidence_threshold == 0.25
        assert config.iou_threshold == 0.45
        assert config.input_size == 640
        assert config.half_precision is False

    def test_temporal_config_defaults(self):
        config = TemporalValidationConfig()
        assert config.min_consecutive_frames == 3
        assert config.frame_confidence_threshold == 0.3
        assert config.max_frame_gap_seconds == 60.0
        assert config.min_confidence_sum == 1.0

    def test_vision_engine_config_defaults(self):
        config = VisionEngineConfig()
        assert config.model.type == "dummy"
        assert config.temporal.min_consecutive_frames == 3
        assert config.enabled is True
        assert config.log_level == "INFO"

    def test_load_config_from_yaml(self):
        config_data = {
            "model": {
                "type": "dummy",
                "weights_path": "./models/test.pt",
                "confidence_threshold": 0.5,
            },
            "temporal": {
                "min_consecutive_frames": 5,
            },
            "enabled": False,
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump(config_data, f)
            temp_path = f.name

        try:
            # Override the settings to use our temp file
            from geomind.vision.config import Settings
            settings = Settings(config_path=temp_path)
            config = VisionEngineConfig(**yaml.safe_load(open(temp_path)))

            assert config.model.type == "dummy"
            assert config.model.confidence_threshold == 0.5
            assert config.temporal.min_consecutive_frames == 5
            assert config.enabled is False
        finally:
            Path(temp_path).unlink()