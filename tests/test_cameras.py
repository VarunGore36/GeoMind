"""Tests for camera pipeline."""

from __future__ import annotations

from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from geomind.cameras.config import CameraFeedConfig, CameraPipelineConfig, StorageConfig
from geomind.cameras.fetcher import CameraFetcher, FetchResult
from geomind.cameras.storage import FrameStorage


class TestCameraConfig:
    def test_camera_feed_config_defaults(self):
        feed = CameraFeedConfig(feed_url="https://example.com/cam.jpg", camera_id="test")
        assert feed.fetch_interval_seconds == 30
        assert feed.timeout_seconds == 10
        assert feed.max_retries == 3

    def test_storage_config_defaults(self):
        storage = StorageConfig()
        assert storage.deduplicate is True
        assert storage.similarity_threshold == 0.95

    def test_pipeline_config(self):
        config = CameraPipelineConfig(
            feeds=[CameraFeedConfig(feed_url="https://example.com/cam.jpg", camera_id="test")],
            storage=StorageConfig(),
        )
        assert len(config.feeds) == 1
        assert config.feeds[0].camera_id == "test"


class TestFrameStorage:
    @pytest.fixture
    def temp_storage(self, tmp_path):
        return FrameStorage(base_path=str(tmp_path))

    @pytest.fixture
    def sample_image(self):
        import numpy as np
        import cv2
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        img[:, :] = (255, 128, 64)
        _, buf = cv2.imencode(".jpg", img)
        return buf.tobytes()

    def test_save_frame(self, temp_storage, sample_image):
        path, size, w, h, chash = temp_storage.save_frame(
            "test_cam", sample_image, datetime.utcnow(), 1
        )
        assert size == len(sample_image)
        assert w == 100
        assert h == 100
        assert len(chash) == 64

    def test_duplicate_detection(self, temp_storage, sample_image):
        temp_storage.save_frame("test_cam", sample_image, datetime.utcnow(), 1)
        phash = temp_storage._compute_phash(sample_image)
        assert temp_storage.is_duplicate("test_cam", sample_image, phash) is True

    def test_cleanup_old_frames(self, temp_storage, sample_image):
        temp_storage.max_frames = 3
        for i in range(5):
            temp_storage.save_frame("test_cam", sample_image, datetime.utcnow(), i + 1)
        frames = list(temp_storage._camera_dir("test_cam").glob("*.jpg"))
        assert len(frames) == 3


class TestCameraFetcher:
    @pytest.fixture
    def fetcher(self):
        with patch("geomind.cameras.fetcher.httpx.AsyncClient") as mock:
            mock_client = AsyncMock()
            mock.return_value = mock_client
            feed = CameraFeedConfig(feed_url="https://example.com/cam.jpg", camera_id="test")
            f = CameraFetcher(feed)
            f.client = mock_client
            yield f, mock_client

    @pytest.mark.asyncio
    async def test_fetch_success(self, fetcher):
        f, mock = fetcher
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.content = b"fake"
        mock.get.return_value = mock_resp

        result = await f.fetch_frame()
        assert result.success is True
        assert result.image_data == b"fake"

    @pytest.mark.asyncio
    async def test_fetch_retry(self, fetcher):
        f, mock = fetcher
        import httpx
        mock.get.side_effect = [
            httpx.TimeoutException("timeout"),
            MagicMock(status_code=200, content=b"ok"),
        ]
        f.feed_config.max_retries = 2
        f.feed_config.retry_delay_seconds = 0

        result = await f.fetch_frame()
        assert result.success is True
        assert mock.get.call_count == 2

    @pytest.mark.asyncio
    async def test_fetch_and_save_duplicate(self, fetcher):
        f, mock = fetcher
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.content = b"fake"
        mock.get.return_value = mock_resp

        with patch("geomind.cameras.fetcher.storage") as mock_store:
            mock_store._compute_phash.return_value = "phash"
            mock_store.is_duplicate.return_value = True

            meta = await f.fetch_and_save()
            assert meta.error == "Duplicate frame"