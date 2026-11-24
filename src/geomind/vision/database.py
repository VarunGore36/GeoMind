"""Vision Engine database layer."""

from __future__ import annotations

import json
from contextlib import asynccontextmanager
from datetime import datetime
from typing import AsyncGenerator, Optional

from sqlalchemy import desc, select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from .config import get_vision_config
from .models import (
    Base,
    Detection,
    DetectionEvent,
    DetectionEventResponse,
    DetectionRecord,
    DetectionStatus,
    FrameDetectionRecord,
    FrameDetections,
)


class VisionDatabase:
    """Async database layer for vision detections."""

    def __init__(self, url: str | None = None):
        config = get_vision_config()
        self.engine = create_async_engine(url or config.database.url, echo=False)
        self.session_factory = async_sessionmaker(
            self.engine, expire_on_commit=False
        )

    async def init(self) -> None:
        """Initialize database tables."""
        async with self.engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)

    async def close(self) -> None:
        """Close database connections."""
        await self.engine.dispose()

    @asynccontextmanager
    async def session(self) -> AsyncGenerator[AsyncSession, None]:
        """Get a database session."""
        async with self.session_factory() as session:
            try:
                yield session
                await session.commit()
            except Exception:
                await session.rollback()
                raise

    # --- Detection Events ---

    async def save_detection_event(
        self,
        event_id: str,
        camera_id: str,
        first_frame_id: int,
        last_frame_id: int,
        first_timestamp: datetime,
        last_timestamp: datetime,
        class_name: str,
        max_confidence: float,
        avg_confidence: float,
        frame_count: int,
        status: DetectionStatus,
        bbox: dict | None = None,
    ) -> DetectionRecord:
        """Save a new detection event."""
        async with self.session() as session:
            record = DetectionRecord(
                event_id=event_id,
                camera_id=camera_id,
                first_frame_id=first_frame_id,
                last_frame_id=last_frame_id,
                first_timestamp=first_timestamp,
                last_timestamp=last_timestamp,
                class_name=class_name,
                max_confidence=max_confidence,
                avg_confidence=avg_confidence,
                frame_count=frame_count,
                status=status.value,
                bbox_json=json.dumps(bbox) if bbox else None,
            )
            session.add(record)
            await session.flush()
            return record

    async def update_detection_event(
        self,
        event_id: str,
        last_frame_id: int | None = None,
        last_timestamp: datetime | None = None,
        max_confidence: float | None = None,
        avg_confidence: float | None = None,
        frame_count: int | None = None,
        status: DetectionStatus | None = None,
        bbox: dict | None = None,
    ) -> Optional[DetectionRecord]:
        """Update an existing detection event."""
        async with self.session() as session:
            result = await session.execute(
                select(DetectionRecord).where(DetectionRecord.event_id == event_id)
            )
            record = result.scalar_one_or_none()
            if not record:
                return None

            if last_frame_id is not None:
                record.last_frame_id = last_frame_id
            if last_timestamp is not None:
                record.last_timestamp = last_timestamp
            if max_confidence is not None:
                record.max_confidence = max_confidence
            if avg_confidence is not None:
                record.avg_confidence = avg_confidence
            if frame_count is not None:
                record.frame_count = frame_count
            if status is not None:
                record.status = status.value
            if bbox is not None:
                record.bbox_json = json.dumps(bbox)
            record.updated_at = datetime.utcnow()

            await session.flush()
            return record

    async def get_detection_event(self, event_id: str) -> Optional[DetectionEventResponse]:
        """Get a detection event by ID."""
        async with self.session() as session:
            result = await session.execute(
                select(DetectionRecord).where(DetectionRecord.event_id == event_id)
            )
            record = result.scalar_one_or_none()
            if not record:
                return None
            return DetectionEventResponse.from_record(record)

    async def get_latest_event_for_camera(
        self, camera_id: str
    ) -> Optional[DetectionEventResponse]:
        """Get the latest detection event for a camera."""
        async with self.session() as session:
            result = await session.execute(
                select(DetectionRecord)
                .where(DetectionRecord.camera_id == camera_id)
                .order_by(desc(DetectionRecord.last_timestamp))
                .limit(1)
            )
            record = result.scalar_one_or_none()
            if not record:
                return None
            return DetectionEventResponse.from_record(record)

    async def list_detection_events(
        self,
        camera_id: str | None = None,
        status: DetectionStatus | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> list[DetectionEventResponse]:
        """List detection events with optional filters."""
        async with self.session() as session:
            query = select(DetectionRecord).order_by(
                desc(DetectionRecord.last_timestamp)
            )
            if camera_id:
                query = query.where(DetectionRecord.camera_id == camera_id)
            if status:
                query = query.where(DetectionRecord.status == status.value)
            query = query.limit(limit).offset(offset)

            result = await session.execute(query)
            records = result.scalars().all()
            return [DetectionEventResponse.from_record(r) for r in records]

    # --- Frame Detections (audit trail) ---

    async def save_frame_detections(
        self,
        camera_id: str,
        frame_id: int,
        timestamp: datetime,
        detections: list[Detection],
        inference_time_ms: float,
        model_version: str,
        event_id: str | None = None,
    ) -> FrameDetectionRecord:
        """Save frame detections for audit trail."""
        async with self.session() as session:
            record = FrameDetectionRecord(
                camera_id=camera_id,
                frame_id=frame_id,
                timestamp=timestamp,
                event_id=event_id,
                detections_json=json.dumps([d.model_dump() for d in detections]),
                inference_time_ms=inference_time_ms,
                model_version=model_version,
            )
            session.add(record)
            await session.flush()
            return record

    async def get_frame_detections(
        self, camera_id: str, frame_id: int
    ) -> Optional[FrameDetectionRecord]:
        """Get frame detections by camera and frame ID."""
        async with self.session() as session:
            result = await session.execute(
                select(FrameDetectionRecord).where(
                    FrameDetectionRecord.camera_id == camera_id,
                    FrameDetectionRecord.frame_id == frame_id,
                )
            )
            return result.scalar_one_or_none()

    async def get_recent_frame_detections(
        self, camera_id: str, limit: int = 100
    ) -> list[FrameDetectionRecord]:
        """Get recent frame detections for a camera."""
        async with self.session() as session:
            result = await session.execute(
                select(FrameDetectionRecord)
                .where(FrameDetectionRecord.camera_id == camera_id)
                .order_by(desc(FrameDetectionRecord.timestamp))
                .limit(limit)
            )
            return list(result.scalars().all())


vision_db = VisionDatabase()