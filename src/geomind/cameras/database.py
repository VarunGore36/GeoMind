"""Camera pipeline database layer."""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import AsyncGenerator, Optional

from sqlalchemy import desc, select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from .config import get_camera_config
from .models import Base, FrameMetadata, FrameRecord


class CameraDatabase:
    def __init__(self, url: str | None = None):
        config = get_camera_config()
        self.engine = create_async_engine(url or config.database.url, echo=False)
        self.session_factory = async_sessionmaker(self.engine, class_=AsyncSession, expire_on_commit=False)

    async def init(self) -> None:
        async with self.engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)

    async def close(self) -> None:
        await self.engine.dispose()

    @asynccontextmanager
    async def session(self) -> AsyncGenerator[AsyncSession, None]:
        async with self.session_factory() as session:
            try:
                yield session
                await session.commit()
            except Exception:
                await session.rollback()
                raise

    async def save_frame(self, meta: FrameMetadata) -> FrameRecord:
        async with self.session() as session:
            record = FrameRecord(
                camera_id=meta.camera_id, timestamp=meta.timestamp,
                frame_number=meta.frame_number, file_path=meta.file_path,
                file_size=meta.file_size, width=meta.width, height=meta.height,
                content_hash=meta.content_hash, fetch_duration_ms=meta.fetch_duration_ms,
                status_code=meta.status_code, error=meta.error,
            )
            session.add(record)
            await session.flush()
            return record

    async def get_latest_frame(self, camera_id: str) -> Optional[FrameMetadata]:
        async with self.session() as session:
            result = await session.execute(
                select(FrameRecord).where(FrameRecord.camera_id == camera_id)
                .order_by(desc(FrameRecord.frame_number)).limit(1)
            )
            record = result.scalar_one_or_none()
            if record:
                return FrameMetadata(
                    camera_id=record.camera_id, timestamp=record.timestamp,
                    frame_number=record.frame_number, file_path=record.file_path,
                    file_size=record.file_size, width=record.width,
                    height=record.height, content_hash=record.content_hash,
                    fetch_duration_ms=record.fetch_duration_ms,
                    status_code=record.status_code, error=record.error,
                )
            return None

    async def get_frame_count(self, camera_id: str) -> int:
        async with self.session() as session:
            result = await session.execute(
                select(FrameRecord).where(FrameRecord.camera_id == camera_id)
            )
            return len(result.scalars().all())

    async def cleanup_old_frames(self, camera_id: str, max_frames: int) -> int:
        async with self.session() as session:
            result = await session.execute(
                select(FrameRecord).where(FrameRecord.camera_id == camera_id)
                .order_by(FrameRecord.frame_number)
            )
            records = result.scalars().all()
            if len(records) <= max_frames:
                return 0
            for record in records[:-max_frames]:
                await session.delete(record)
            return len(records) - max_frames


db = CameraDatabase()