"""Tests for camera scheduler."""

from __future__ import annotations

from datetime import datetime
from unittest.mock import AsyncMock, patch, MagicMock

import pytest

from geomind.cameras.config import CameraFeedConfig, CameraPipelineConfig, StorageConfig
from geomind.cameras.models import CameraStatus, FrameMetadata
from geomind.cameras.scheduler import CameraScheduler


@pytest.fixture
def mock_config():
    config = CameraPipelineConfig(
        feeds=[CameraFeedConfig(feed_url="https://example.com/cam.jpg", camera_id="test_cam", fetch_interval_seconds=30)],
        storage=StorageConfig(),
    )
    return config


@pytest.fixture
def mock_deps(mock_config):
    with patch("geomind.cameras.scheduler.CameraFetcher") as mock_fetcher_class, \
         patch("geomind.cameras.scheduler.db") as mock_db, \
         patch("geomind.cameras.scheduler.get_camera_config", return_value=mock_config):

        mock_fetcher_instance = AsyncMock()
        mock_fetcher_instance.start = AsyncMock()
        mock_fetcher_instance.stop = AsyncMock()
        mock_fetcher_instance.fetch_and_save = AsyncMock()
        mock_fetcher_class.return_value = mock_fetcher_instance
        
        mock_db.init = AsyncMock()
        mock_db.close = AsyncMock()
        mock_db.save_frame = AsyncMock()
        yield mock_fetcher_class, mock_db


@pytest.mark.asyncio
async def test_scheduler_start_stop(mock_deps):
    mock_fetcher_class, mock_db = mock_deps
    sched = CameraScheduler()
    await sched.start()
    assert sched._running is True
    mock_db.init.assert_called_once()
    mock_fetcher_class.assert_called_once()
    await sched.stop()
    assert sched._running is False
    mock_db.close.assert_called_once()


@pytest.mark.asyncio
async def test_get_status(mock_deps):
    mock_fetcher_class, mock_db = mock_deps
    sched = CameraScheduler()
    await sched.start()
    st = sched.get_status("test_cam")
    assert st is not None
    assert st.camera_id == "test_cam"
    await sched.stop()


@pytest.mark.asyncio
async def test_trigger_fetch(mock_deps):
    mock_fetcher_class, mock_db = mock_deps
    mock_fetcher_instance = mock_fetcher_class.return_value
    mock_fetcher_instance.fetch_and_save = AsyncMock(return_value=FrameMetadata(
        camera_id="test_cam", timestamp=datetime.utcnow(), frame_number=1,
        file_path="/frames/test.jpg", file_size=1024, width=1920, height=1080,
        content_hash="abc", fetch_duration_ms=100.0, status_code=200,
    ))
    mock_db.save_frame = AsyncMock()

    sched = CameraScheduler()
    await sched.start()
    meta = await sched.trigger_fetch("test_cam")
    assert meta is not None
    mock_db.save_frame.assert_called_once()
    await sched.stop()