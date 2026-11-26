"""Tests for WebSocket connection manager."""

from __future__ import annotations

from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import WebSocket

from geomind.ws.config import WSConfigMain, WSConfig, WSAuthConfig
from geomind.ws.manager import Connection, ConnectionManager
from geomind.ws.models import (
    SubscribeMessage,
    UnsubscribeMessage,
    HeartbeatMessage,
    AckMessage,
    WSChannelType,
    DetectionEventMessage,
    DetectionConfirmedMessage,
    DetectionPendingMessage,
    CameraStatusMessage,
    serialize_ws_message,
)


@pytest.fixture
def ws_config():
    return WSConfigMain(
        ws=WSConfig(max_connections=10, heartbeat_interval_seconds=5),
        auth=WSAuthConfig(allow_anonymous=True, required_roles={}),
        enabled=True,
    )


@pytest.fixture
def mock_websocket():
    ws = AsyncMock(spec=WebSocket)
    ws.send_text = AsyncMock()
    ws.close = AsyncMock()
    ws.accept = AsyncMock()
    return ws


@pytest.fixture
async def manager(ws_config):
    with patch("geomind.ws.manager.get_ws_config", return_value=ws_config):
        manager = ConnectionManager()
        await manager.start()
        yield manager
        await manager.stop()


@pytest.fixture
def mock_ws():
    ws = AsyncMock(spec=WebSocket)
    ws.send_text = AsyncMock()
    ws.close = AsyncMock()
    ws.accept = AsyncMock()
    return ws


class TestConnection:
    @pytest.mark.asyncio
    async def test_connection_creation(self, ws_config, mock_ws):
        with patch("geomind.ws.manager.get_ws_config", return_value=ws_config):
            conn = Connection(
                websocket=mock_ws,
                client_id="test_client",
                user_id="user_123",
                roles={"user"},
            )
            assert conn.client_id == "test_client"
            assert conn.user_id == "user_123"
            assert conn.roles == {"user"}
            assert conn.subscriptions == set()

    @pytest.mark.asyncio
    async def test_send_message(self, ws_config, mock_ws):
        with patch("geomind.ws.manager.get_ws_config", return_value=ws_config):
            conn = Connection(
                websocket=mock_ws,
                client_id="test_client",
            )
            result = await conn.send("test message")
            assert result is True
            mock_ws.send_text.assert_called_once_with("test message")

    @pytest.mark.asyncio
    async def test_can_access_channel(self, ws_config):
        with patch("geomind.ws.manager.get_ws_config", return_value=ws_config):
            conn = Connection(
                websocket=AsyncMock(spec=WebSocket),
                client_id="test_client",
                roles={"user", "admin"},
            )
            required_roles = {
                "detections:*": ["user"],
                "admin:*": ["admin"],
            }
            assert conn.can_access_channel("detections:smoke", required_roles) is True
            assert conn.can_access_channel("admin:panel", required_roles) is True
            assert conn.can_access_channel("unknown:channel", required_roles) is True


class TestConnectionManager:
    @pytest.mark.asyncio
    async def test_start_stop(self, ws_config):
        with patch("geomind.ws.manager.get_ws_config", return_value=ws_config):
            manager = ConnectionManager()
            await manager.start()
            assert manager._running is True
            await manager.stop()
            assert manager._running is False

    @pytest.mark.asyncio
    async def test_connect_disconnect(self, manager, mock_websocket):
        conn = await manager.connect(mock_websocket, client_id="test_client")
        assert conn.client_id == "test_client"
        assert "test_client" in manager.connections

        await manager.disconnect("test_client")
        assert "test_client" not in manager.connections

    @pytest.mark.asyncio
    async def test_max_connections(self, ws_config, mock_websocket):
        ws_config.ws.max_connections = 1
        with patch("geomind.ws.manager.get_ws_config", return_value=ws_config):
            manager = ConnectionManager()
            await manager.start()

            # First connection should succeed
            ws1 = AsyncMock(spec=WebSocket)
            ws1.send_text = AsyncMock()
            ws1.close = AsyncMock()
            ws1.accept = AsyncMock()
            await manager.connect(ws1, client_id="client1")

            # Second connection should fail
            ws2 = AsyncMock(spec=WebSocket)
            ws2.send_text = AsyncMock()
            ws2.close = AsyncMock()
            ws2.accept = AsyncMock()

            with pytest.raises(RuntimeError, match="Max connections reached"):
                await manager.connect(ws2, client_id="client2")

            await manager.stop()

    @pytest.mark.asyncio
    async def test_subscribe_unsubscribe(self, manager, mock_websocket):
        with patch("geomind.ws.manager.get_ws_config", return_value=ws_config):
            conn = await manager.connect(mock_websocket, client_id="test_client")

            # Subscribe
            success = await manager.subscribe("test_client", "detections:smoke")
            assert success is True
            assert "detections:smoke" in manager.channel_subscribers
            assert "test_client" in manager.channel_subscribers["detections:smoke"]

            # Unsubscribe
            success = await manager.unsubscribe("test_client", "detections:smoke")
            assert success is True
            assert "detections:smoke" not in manager.channel_subscribers

    @pytest.mark.asyncio
    async def test_broadcast_to_channel(self, manager, mock_websocket):
        with patch("geomind.ws.manager.get_ws_config", return_value=ws_config):
            conn1 = await manager.connect(AsyncMock(spec=WebSocket), client_id="client1")
            conn2 = await manager.connect(AsyncMock(spec=WebSocket), client_id="client2")

            await manager.subscribe("client1", "detections")
            await manager.subscribe("client2", "detections")

            message = {"type": "test", "data": "hello"}
            sent = await manager.broadcast_to_channel("detections", message)

            assert sent == 2

    @pytest.mark.asyncio
    async def test_get_stats(self, manager, mock_websocket):
        await manager.connect(mock_websocket, client_id="client1")
        await manager.connect(AsyncMock(spec=WebSocket), client_id="client2")

        await manager.subscribe("client1", "detections")
        await manager.subscribe("client2", "detections")
        await manager.subscribe("client1", "cameras")

        stats = manager.get_stats()
        assert stats["total_connections"] == 2
        assert stats["total_subscriptions"] == 3
        assert stats["channels"]["detections"] == 2
        assert stats["channels"]["cameras"] == 1


class TestPublishFunctions:
    @pytest.mark.asyncio
    async def test_publish_detection_event(self, manager, mock_websocket):
        with patch("geomind.ws.api.connection_manager", manager):
            from geomind.ws.api import publish_detection_event

            await manager.connect(mock_websocket, client_id="client1")
            await manager.subscribe("client1", "detections")

            await publish_detection_event(
                event_id="evt-1",
                camera_id="cam_001",
                frame_id=42,
                timestamp=datetime.utcnow(),
                class_name="smoke",
                confidence=0.85,
                bbox={"x_min": 0.1, "y_min": 0.1, "x_max": 0.5, "y_max": 0.5},
                status="confirmed",
            )

            # The message should be queued for broadcast
            # (actual delivery happens in broadcast loop)


class TestSerializeMessages:
    def test_serialize_detection_event(self):
        msg = DetectionEventMessage(
            channel="detections",
            event_id="evt-1",
            camera_id="cam_001",
            frame_id=1,
            timestamp=datetime.utcnow(),
            class_name="smoke",
            confidence=0.85,
            bbox={"x_min": 0.1, "y_min": 0.1, "x_max": 0.5, "y_max": 0.5},
            status="confirmed",
        )
        json_str = serialize_ws_message(msg)
        assert '"type":"detection"' in json_str
        assert '"event_id":"evt-1"' in json_str
        assert '"class_name":"smoke"' in json_str