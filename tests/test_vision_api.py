"""Tests for vision API endpoints."""

from __future__ import annotations

from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from geomind.vision.main import create_vision_app
from geomind.vision.models import DetectionEventResponse, DetectionStatus, BoundingBox


@pytest.fixture
def client():
    with patch("geomind.vision.engine.vision_engine.start", new_callable=AsyncMock), \
         patch("geomind.vision.engine.vision_engine.stop", new_callable=AsyncMock):
        app = create_vision_app()
        return TestClient(app)


def test_health_check(client):
    with patch("geomind.vision.api.vision_engine") as mock_engine:
        mock_engine._running = True
        mock_engine.get_stats.return_value = {"running": True, "model": "DummyModel"}

        response = client.get("/api/v1/detections/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert data["stats"]["running"] is True


def test_list_detection_events(client):
    with patch("geomind.vision.api.vision_db") as mock_db:
        mock_event = MagicMock(spec=DetectionEventResponse)
        mock_event.event_id = "test-1"
        mock_event.camera_id = "test_cam"
        mock_event.first_frame_id = 1
        mock_event.last_frame_id = 3
        mock_event.first_timestamp = datetime.utcnow()
        mock_event.last_timestamp = datetime.utcnow()
        mock_event.class_name = "smoke"
        mock_event.max_confidence = 0.9
        mock_event.avg_confidence = 0.8
        mock_event.frame_count = 3
        mock_event.status = DetectionStatus.CONFIRMED
        mock_event.bbox = None

        mock_db.list_detection_events = AsyncMock(return_value=[mock_event])

        response = client.get("/api/v1/detections/")
        assert response.status_code == 200
        data = response.json()
        assert len(data) == 1
        assert data[0]["event_id"] == "test-1"


def test_get_detection_event_not_found(client):
    with patch("geomind.vision.api.vision_db") as mock_db:
        mock_db.get_detection_event = AsyncMock(return_value=None)

        response = client.get("/api/v1/detections/nonexistent")
        assert response.status_code == 404


def test_get_latest_event_for_camera(client):
    with patch("geomind.vision.api.vision_db") as mock_db:
        mock_event = MagicMock(spec=DetectionEventResponse)
        mock_event.event_id = "test-1"
        mock_event.camera_id = "test_cam"
        mock_event.first_frame_id = 1
        mock_event.last_frame_id = 3
        mock_event.first_timestamp = datetime.utcnow()
        mock_event.last_timestamp = datetime.utcnow()
        mock_event.class_name = "smoke"
        mock_event.max_confidence = 0.9
        mock_event.avg_confidence = 0.8
        mock_event.frame_count = 3
        mock_event.status = DetectionStatus.CONFIRMED
        mock_event.bbox = None

        mock_db.get_latest_event_for_camera = AsyncMock(return_value=mock_event)

        response = client.get("/api/v1/detections/camera/test_cam/latest")
        assert response.status_code == 200
        data = response.json()
        assert data["event_id"] == "test-1"


def test_get_pending_events(client):
    with patch("geomind.vision.api.vision_engine") as mock_engine:
        mock_pending = MagicMock()
        mock_pending.event_id = "pending-1"
        mock_pending.camera_id = "test_cam"
        mock_pending.class_name = "smoke"
        mock_pending.first_frame_id = 1
        mock_pending.last_frame_id = 1
        mock_pending.frame_count = 1
        mock_pending.confidences = [0.8]

        mock_engine.validator.get_pending_events.return_value = [mock_pending]

        response = client.get("/api/v1/detections/camera/test_cam/pending")
        assert response.status_code == 200
        data = response.json()
        assert data["pending_count"] == 1
        assert len(data["events"]) == 1