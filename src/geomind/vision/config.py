"""Vision Engine configuration."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class ModelConfig(BaseModel):
    """Configuration for the detection model."""

    # Model type - can be extended for different models
    type: Literal["yolo", "custom", "dummy"] = "dummy"
    # Path to model weights or model identifier
    weights_path: str = "./models/yolov8n.pt"
    # Device to run inference on (cpu, cuda, mps)
    device: str = "cpu"
    # Confidence threshold for initial detection
    confidence_threshold: float = Field(default=0.25, ge=0.0, le=1.0)
    # IoU threshold for NMS
    iou_threshold: float = Field(default=0.45, ge=0.0, le=1.0)
    # Input image size (square)
    input_size: int = Field(default=640, ge=320, le=1280)
    # Classes to detect (COCO class IDs: 0=person, etc. For wildfire smoke, custom model needed)
    classes: list[int] = Field(default_factory=list)  # Empty = all classes
    # Half precision (FP16) for faster inference
    half_precision: bool = False


class TemporalValidationConfig(BaseModel):
    """Configuration for temporal validation across frames."""

    # Minimum consecutive frames with detection to confirm
    min_consecutive_frames: int = Field(default=3, ge=1)
    # Confidence threshold for a frame to count towards confirmation
    frame_confidence_threshold: float = Field(default=0.3, ge=0.0, le=1.0)
    # Maximum time gap between frames (seconds) to consider consecutive
    max_frame_gap_seconds: float = Field(default=60.0, gt=0)
    # Minimum total confidence sum across consecutive frames
    min_confidence_sum: float = Field(default=1.0, ge=0.0)


class DatabaseConfig(BaseModel):
    """Database configuration."""

    url: str = "sqlite+aiosqlite:///./vision_detections.db"


class QueueConfig(BaseModel):
    """Frame queue configuration for async processing."""

    max_size: int = Field(default=1000, ge=1)
    worker_count: int = Field(default=2, ge=1)
    batch_size: int = Field(default=1, ge=1)
    batch_timeout_ms: int = Field(default=100, ge=1)


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_prefix="GEOMIND_VISION_",
        extra="ignore",
    )
    config_path: str = "vision.yaml"


class VisionEngineConfig(BaseModel):
    """Main Vision Engine configuration."""

    model: ModelConfig = ModelConfig()
    temporal: TemporalValidationConfig = TemporalValidationConfig()
    database: DatabaseConfig = DatabaseConfig()
    queue: QueueConfig = QueueConfig()
    enabled: bool = True
    log_level: str = "INFO"


@lru_cache
def get_vision_config() -> VisionEngineConfig:
    """Load and cache the vision engine configuration."""
    settings = Settings()
    config_path = Path(settings.config_path)
    if config_path.exists():
        with open(config_path) as f:
            data = yaml.safe_load(f)
        return VisionEngineConfig(**data)
    return VisionEngineConfig()