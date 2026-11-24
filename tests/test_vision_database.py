"""Tests for vision database layer."""

from __future__ import annotations

from datetime import datetime

import pytest
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.pool import StaticPool
from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker

from geomind.vision.database import VisionDatabase
from geomind.vision.models import (
    Base,
    Detection,
    DetectionEvent,
    DetectionStatus,
    BoundingBox,
)


@pytest.fixture
async def test_db():
    engine = create_async_engine(
        "sqlite+aiosqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    session_factory = async_sessionmaker(engine, expire_on_commit=False)
    db = VisionDatabase()
    db.engine = engine
    db.session_factory = session_factory
    yield db
    await engine.dispose()


@pytest.fixture
def sample_bbox():
    return BoundingBox(x_min=0.1, y_min=0.1, x_max=0.5, y_max=0.5)


@pytest.fixture
def sample_detection(sample_bbox):
    return Detection(
        class_id=0,
        class_name="smoke",
        confidence=0.85,
        bbox=sample_bbox,
    )


class TestVisionDatabase:
    @pytest.mark.asyncio
    async def test_init(self, test_db):
        await test_db.init()
        # Should not raise

    @pytest.mark.asyncio
    async def test_save_and_get_detection_event(self, test_db, sample_bbox):
        await test_db.init()

        event = await test_db.save_detection_event(
            event_id="test-event-1",
            camera_id="test_cam",
            first_frame_id=1,
            last_frame_id=3,
            first_timestamp=datetime.utcnow(),
            last_timestamp=datetime.utcnow(),
            class_name="smoke",
            max_confidence=0.9,
            avg_confidence=0.8,
            frame_count=3,
            status=DetectionStatus.CONFIRMED,
            bbox=sample_bbox.model_dump(),
        )

        assert event.id is not None
        assert event.event_id == "test-event-1"

        # Retrieve
        retrieved = await test_db.get_detection_event("test-event-1")
        assert retrieved is not None
        assert retrieved.event_id == "test-event-1"
        assert retrieved.camera_id == "test_cam"
        assert retrieved.class_name == "smoke"
        assert retrieved.status == DetectionStatus.CONFIRMED
        assert retrieved.bbox is not None
        assert retrieved.bbox.x_min == 0.1

    @pytest.mark.asyncio
    async def test_update_detection_event(self, test_db, sample_bbox):
        await test_db.init()

        await test_db.save_detection_event(
            event_id="test-event-1",
            camera_id="test_cam",
            first_frame_id=1,
            last_frame_id=1,
            first_timestamp=datetime.utcnow(),
            last_timestamp=datetime.utcnow(),
            class_name="smoke",
            max_confidence=0.8,
            avg_confidence=0.8,
            frame_count=1,
            status=DetectionStatus.PENDING,
        )

        # Update to confirmed
        updated = await test_db.update_detection_event(
            event_id="test-event-1",
            last_frame_id=3,
            frame_count=3,
            status=DetectionStatus.CONFIRMED,
            max_confidence=0.9,
            avg_confidence=0.85,
        )

        assert updated is not None
        assert updated.status == "confirmed"
        assert updated.frame_count == 3

        # Verify
        retrieved = await test_db.get_detection_event("test-event-1")
        assert retrieved.status.value == "confirmed"
        assert retrieved.frame_count == 3

    @pytest.mark.asyncio
    async def test_list_detection_events(self, test_db):
        await test_db.init()

        for i in range(5):
            await test_db.save_detection_event(
                event_id=f"test-event-{i}",
                camera_id="test_cam",
                first_frame_id=1,
                last_frame_id=1,
                first_timestamp=datetime.utcnow(),
                last_timestamp=datetime.utcnow(),
                class_name="smoke",
                max_confidence=0.8,
                avg_confidence=0.8,
                frame_count=1,
                status=DetectionStatus.CONFIRMED,
            )

        events = await test_db.list_detection_events(limit=10)
        assert len(events) == 5

        # Test filter by camera
        events = await test_db.list_detection_events(camera_id="test_cam", limit=10)
        assert len(events) == 5

        events = await test_db.list_detection_events(camera_id="other_cam", limit=10)
        assert len(events) == 0

    @pytest.mark.asyncio
    async def test_get_latest_event_for_camera(self, test_db):
        await test_db.init()

        await test_db.save_detection_event(
            event_id="test-event-1",
            camera_id="test_cam",
            first_frame_id=1,
            last_frame_id=1,
            first_timestamp=datetime.utcnow(),
            last_timestamp=datetime.utcnow(),
            class_name="smoke",
            max_confidence=0.8,
            avg_confidence=0.8,
            frame_count=1,
            status=DetectionStatus.CONFIRMED,
        )

        latest = await test_db.get_latest_event_for_camera("test_cam")
        assert latest is not None
        assert latest.event_id == "test-event-1"

        latest = await test_db.get_latest_event_for_camera("other_cam")
        assert latest is None