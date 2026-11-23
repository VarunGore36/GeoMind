"""Camera pipeline data models."""

from __future__ import annotations

from datetime import datetime
from typing import Optional

from pydantic import BaseModel
from sqlalchemy import Column, DateTime, Float, Integer, String, Text, UniqueConstraint
from sqlalchemy.orm import declarative_base

Base = declarative_base()


class FrameMetadata(BaseModel):
    camera_id: str
    timestamp: datetime
    frame_number: int
    file_path: str
    file_size: int
    width: int
    height: int
    content_hash: str
    fetch_duration_ms: float
    status_code: int
    error: Optional[str] = None


class FrameRecord(Base):
    __tablename__ = "camera_frames"
    __table_args__ = (
        UniqueConstraint("camera_id", "frame_number", name="uq_camera_frame"),
    )

    id = Column(Integer, primary_key=True, autoincrement=True)
    camera_id = Column(String(64), nullable=False, index=True)
    timestamp = Column(DateTime, nullable=False, index=True)
    frame_number = Column(Integer, nullable=False)
    file_path = Column(Text, nullable=False)
    file_size = Column(Integer, nullable=False)
    width = Column(Integer, nullable=False)
    height = Column(Integer, nullable=False)
    content_hash = Column(String(64), nullable=False, index=True)
    fetch_duration_ms = Column(Float, nullable=False)
    status_code = Column(Integer, nullable=False)
    error = Column(Text, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)


class CameraStatus(BaseModel):
    camera_id: str
    is_active: bool
    last_fetch: Optional[datetime] = None
    last_success: Optional[datetime] = None
    consecutive_failures: int = 0
    total_frames: int = 0
    total_errors: int = 0
    latest_frame: Optional[FrameMetadata] = None