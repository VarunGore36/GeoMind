"""Tests for WebSocket API endpoints."""

from __future__ import annotations

from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from geomind.main import create_app
from geomind.ws.models import DetectionEventMessage, WSChannelType


@pytest.fixture
def client():
    with patch("geomind.ws.manager.connection_manager.start", new_callable=AsyncMock), \
         patch("geomind.ws.manager.connection_manager.stop", new_callable=AsyncMock), \
         patch("geomind.cameras.scheduler.scheduler.start", new_callable=AsyncMock), \
         patch("geomind.cameras.scheduler.scheduler.stop", new_callable=AsyncMock):
        app = create_app()
        return TestClient(app)


def test_ws_stats_endpoint(client):
    with patch("geomind.ws.api.connection_manager") as mock_manager:
        mock_manager.get_stats.return_value = {
            "total_connections": 5,
            "total_subscriptions": 10,
            "channels": {"detections": 3, "cameras": 2},
            "running": True,
        }

        response = client.get("/api/v1/ws/stats")
        assert response.status_code == 200
        data = response.json()
        assert data["total_connections"] == 5
        assert data["channels"]["detections"] == 3


def test_ws_channels_endpoint(client):
    with patch("geomind.ws.api.connection_manager") as mock_manager:
        mock_manager.channel_subscribers = {
            "detections": {"client1", "client2"},
            "cameras": {"client1"},
        }

        response = client.get("/api/v1/ws/channels")
        assert response.status_code == 200
        data = response.json()
        assert data["detections"] == 2
        assert data["cameras"] == 1


def test_ws_connections_endpoint(client):
    with patch("geomind.ws.api.connection_manager") as mock_manager:
        mock_conn = MagicMock()
        mock_conn.client_id = "client1"
        mock_conn.user_id = "user123"
        mock_conn.roles = {"user"}
        mock_conn.subscriptions = {"detections"}
        mock_conn.connected_at = datetime.utcnow()
        mock_conn.last_heartbeat = datetime.utcnow()

        mock_manager.connections = {"client1": mock_conn}

        response = client.get("/api/v1/ws/connections")
        assert response.status_code == 200
        data = response.json()
        assert data["total"] == 1
        assert len(data["connections"]) == 1
        assert data["connections"][0]["client_id"] == "client1"


def test_ws_stats_endpoint_disabled(client):
    """Test WS stats endpoint works even when WS is disabled (REST still works)."""
    with patch("geomind.ws.manager.connection_manager") as mock_manager:
        mock_manager.get_stats.return_value = {
            "total_connections": 0,
            "total_subscriptions": 0,
            "channels": {},
            "running": False,
        }
        response = client.get("/api/v1/ws/stats")
        assert response.status_code == 200