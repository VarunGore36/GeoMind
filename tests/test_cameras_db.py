"""Tests for camera database."""

from __future__ import annotations

from datetime import datetime

import pytest
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.pool import StaticPool
from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker

from geomind.cameras.database import CameraDatabase
from geomind.cameras.models import Base, FrameMetadata


@pytest.fixture
async def test_db():
    engine = create_async_engine(
        "sqlite+aiosqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    session_factory = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
    db = CameraDatabase()
    db.engine = engine
    db.session_factory = session_factory
    yield db
    await engine.dispose()


@pytest.mark.asyncio
async def test_save_and_get_frame(test_db):
    await test_db.init()
    meta = FrameMetadata(
        camera_id="test_cam", timestamp=datetime.utcnow(), frame_number=1,
        file_path="/frames/test.jpg", file_size=1024, width=1920, height=1080,
        content_hash="abc123", fetch_duration_ms=150.0, status_code=200,
    )
    record = await test_db.save_frame(meta)
    assert record.id is not None

    latest = await test_db.get_latest_frame("test_cam")
    assert latest is not None
    assert latest.frame_number == 1


@pytest.mark.asyncio
async def test_cleanup_old_frames(test_db):
    await test_db.init()
    for i in range(5):
        meta = FrameMetadata(
            camera_id="test_cam", timestamp=datetime.utcnow(), frame_number=i + 1,
            file_path=f"/frames/test_{i}.jpg", file_size=1024, width=1920, height=1080,
            content_hash=f"hash{i}", fetch_duration_ms=150.0, status_code=200,
        )
        await test_db.save_frame(meta)

    deleted = await test_db.cleanup_old_frames("test_cam", max_frames=3)
    assert deleted == 2
    count = await test_db.get_frame_count("test_cam")
    assert count == 3