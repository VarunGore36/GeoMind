"""Vision Engine data models."""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field
from sqlalchemy import Column, DateTime, Float, ForeignKey, Index, Integer, String, Text
from sqlalchemy.orm import declarative_base, relationship

Base = declarative_base()


class DetectionClass(str, Enum):
    """Supported detection classes."""

    SMOKE = "smoke"
    FIRE = "fire"
    SMOKE_PLUME = "smoke_plume"
    UNKNOWN = "unknown"


class BoundingBox(BaseModel):
    """Bounding box in normalized coordinates (0-1)."""

    x_min: float = Field(ge=0.0, le=1.0)
    y_min: float = Field(ge=0.0, le=1.0)
    x_max: float = Field(ge=0.0, le=1.0)
    y_max: float = Field(ge=0.0, le=1.0)

    def to_absolute(self, width: int, height: int) -> tuple[int, int, int, int]:
        """Convert to absolute pixel coordinates."""
        return (
            int(self.x_min * width),
            int(self.y_min * height),
            int(self.x_max * width),
            int(self.y_max * height),
        )

    @classmethod
    def from_absolute(
        cls, x_min: int, y_min: int, x_max: int, y_max: int, width: int, height: int
    ) -> "BoundingBox":
        """Create from absolute pixel coordinates."""
        return cls(
            x_min=x_min / width,
            y_min=y_min / height,
            x_max=x_max / width,
            y_max=y_max / height,
        )


class Detection(BaseModel):
    """A single detection result."""

    class_id: int
    class_name: str
    confidence: float = Field(ge=0.0, le=1.0)
    bbox: BoundingBox


class FrameDetections(BaseModel):
    """All detections for a single frame."""

    camera_id: str
    frame_id: int  # References camera_frames.frame_number
    timestamp: datetime
    detections: list[Detection]
    inference_time_ms: float
    model_version: str


class DetectionStatus(str, Enum):
    """Status of a detection event after temporal validation."""

    PENDING = "pending"          # Single detection, not yet confirmed
    CONFIRMED = "confirmed"      # Temporal validation passed
    REJECTED = "rejected"        # Temporal validation failed


class DetectionEvent(BaseModel):
    """A detection event after temporal validation."""

    event_id: str
    camera_id: str
    first_frame_id: int
    last_frame_id: int
    first_timestamp: datetime
    last_timestamp: datetime
    class_name: str
    max_confidence: float
    avg_confidence: float
    frame_count: int
    status: DetectionStatus
    bbox: Optional[BoundingBox] = None  # Union of bboxes or latest


class DetectionRecord(Base):
    """Database record for a detection event."""

    __tablename__ = "detection_events"
    __table_args__ = (
        Index("ix_detection_camera_time", "camera_id", "last_timestamp"),
        Index("ix_detection_status", "status"),
    )

    id = Column(Integer, primary_key=True, autoincrement=True)
    event_id = Column(String(64), nullable=False, unique=True, index=True)
    camera_id = Column(String(64), nullable=False, index=True)
    first_frame_id = Column(Integer, nullable=False)
    last_frame_id = Column(Integer, nullable=False)
    first_timestamp = Column(DateTime, nullable=False)
    last_timestamp = Column(DateTime, nullable=False, index=True)
    class_name = Column(String(32), nullable=False)
    max_confidence = Column(Float, nullable=False)
    avg_confidence = Column(Float, nullable=False)
    frame_count = Column(Integer, nullable=False)
    status = Column(String(16), nullable=False, index=True)
    bbox_json = Column(Text, nullable=True)  # JSON serialized BoundingBox
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)


class FrameDetectionRecord(Base):
    """Database record for individual frame detections (for audit trail)."""

    __tablename__ = "frame_detections"
    __table_args__ = (
        Index("ix_frame_detection_camera_frame", "camera_id", "frame_id"),
        Index("ix_frame_detection_event", "event_id"),
    )

    id = Column(Integer, primary_key=True, autoincrement=True)
    camera_id = Column(String(64), nullable=False, index=True)
    frame_id = Column(Integer, nullable=False)
    timestamp = Column(DateTime, nullable=False, index=True)
    event_id = Column(String(64), ForeignKey("detection_events.event_id"), nullable=True, index=True)
    detections_json = Column(Text, nullable=False)  # JSON serialized list[Detection]
    inference_time_ms = Column(Float, nullable=False)
    model_version = Column(String(64), nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)


class DetectionEventResponse(BaseModel):
    """API response for a detection event."""

    event_id: str
    camera_id: str
    first_frame_id: int
    last_frame_id: int
    first_timestamp: datetime
    last_timestamp: datetime
    class_name: str
    max_confidence: float
    avg_confidence: float
    frame_count: int
    status: DetectionStatus
    bbox: Optional[BoundingBox] = None

    @classmethod
    def from_record(cls, record: DetectionRecord) -> "DetectionEventResponse":
        bbox = None
        if record.bbox_json:
            import json
            bbox = BoundingBox(**json.loads(record.bbox_json))
        return cls(
            event_id=record.event_id,
            camera_id=record.camera_id,
            first_frame_id=record.first_frame_id,
            last_frame_id=record.last_frame_id,
            first_timestamp=record.first_timestamp,
            last_timestamp=record.last_timestamp,
            class_name=record.class_name,
            max_confidence=record.max_confidence,
            avg_confidence=record.avg_confidence,
            frame_count=record.frame_count,
            status=DetectionStatus(record.status),
            bbox=bbox,
        )


class FrameDetectionResponse(BaseModel):
    """API response for frame detections."""

    camera_id: str
    frame_id: int
    timestamp: datetime
    detections: list[Detection]
    inference_time_ms: float
    model_version: str