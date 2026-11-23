"""Tests for camera API."""

from __future__ import annotations

from datetime import datetime
from unittest.mock import patch, MagicMock

import pytest
from fastapi.testclient import TestClient

from geomind.main import create_app
from geomind.cameras.models import CameraStatus, FrameMetadata


@pytest.fixture
def client():
    with patch("geomind.cameras.scheduler.scheduler.start"), \
         patch("geomind.cameras.scheduler.scheduler.stop"):
        app = create_app()
        return TestClient(app)


def test_camera_health(client):
    with patch("geomind.cameras.api.scheduler") as mock_sched:
        mock_sched.get_all_status.return_value = {
            "test_cam": CameraStatus(camera_id="test_cam", is_active=True, total_frames=5)
        }
        resp = client.get("/api/v1/cameras/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "healthy"
        assert data["feeds"] == 1


def test_list_cameras(client):
    with patch("geomind.cameras.api.scheduler") as mock_sched:
        mock_sched.get_all_status.return_value = {
            "test_cam": CameraStatus(camera_id="test_cam", is_active=True)
        }
        resp = client.get("/api/v1/cameras")
        assert resp.status_code == 200
        assert "test_cam" in resp.json()


def test_get_camera_not_found(client):
    with patch("geomind.cameras.api.scheduler") as mock_sched:
        mock_sched.get_status.return_value = None
        resp = client.get("/api/v1/cameras/unknown")
        assert resp.status_code == 404


def test_get_latest_meta(client):
    with patch("geomind.cameras.api.scheduler") as mock_sched:
        meta = FrameMetadata(
            camera_id="test_cam", timestamp=datetime.utcnow(), frame_number=1,
            file_path="/frames/test.jpg", file_size=1024, width=1920, height=1080,
            content_hash="abc", fetch_duration_ms=100.0, status_code=200,
        )
        mock_sched.get_status.return_value = CameraStatus(
            camera_id="test_cam", is_active=True, latest_frame=meta
        )
        resp = client.get("/api/v1/cameras/test_cam/latest")
        assert resp.status_code == 200
        assert resp.json()["frame_number"] == 1