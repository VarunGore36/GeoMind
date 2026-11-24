"""Vision Engine - Wildfire smoke detection module."""

from __future__ import annotations

from .config import VisionEngineConfig, get_vision_config
from .database import vision_db
from .engine import VisionEngine, vision_engine
from .models import (
    BoundingBox,
    Detection,
    DetectionEvent,
    DetectionEventResponse,
    DetectionStatus,
    FrameDetections,
    FrameDetectionResponse,
)
from .temporal import TemporalValidator

__version__ = "0.1.0"

__all__ = [
    "VisionEngineConfig",
    "get_vision_config",
    "vision_db",
    "VisionEngine",
    "vision_engine",
    "BoundingBox",
    "Detection",
    "DetectionEvent",
    "DetectionEventResponse",
    "DetectionStatus",
    "FrameDetections",
    "FrameDetectionResponse",
    "TemporalValidator",
]