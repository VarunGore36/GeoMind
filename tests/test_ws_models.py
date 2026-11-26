"""Tests for WebSocket models."""

from __future__ import annotations

from datetime import datetime

import pytest

from geomind.ws.models import (
    WSMessageType,
    WSChannelType,
    SubscribeMessage,
    UnsubscribeMessage,
    HeartbeatMessage,
    AckMessage,
    ErrorMessage,
    DetectionEventMessage,
    DetectionConfirmedMessage,
    DetectionPendingMessage,
    CameraStatusMessage,
    parse_ws_message,
    serialize_ws_message,
)


class TestWSMessageTypes:
    def test_subscribe_message(self):
        msg = SubscribeMessage(
            channels=["detections:confirmed", "cameras"],
            filters={"camera_id": "cam_001"},
        )
        assert msg.type == "subscribe"
        assert msg.channels == ["detections:confirmed", "cameras"]
        assert msg.filters == {"camera_id": "cam_001"}

    def test_unsubscribe_message(self):
        msg = UnsubscribeMessage(channels=["detections:pending"])
        assert msg.type == "unsubscribe"
        assert msg.channels == ["detections:pending"]

    def test_heartbeat_message(self):
        msg = HeartbeatMessage(payload={"client_time": "2024-01-01T00:00:00"})
        assert msg.type == "heartbeat"
        assert msg.payload == {"client_time": "2024-01-01T00:00:00"}

    def test_ack_message(self):
        msg = AckMessage(original_id="msg-123", success=True, message="OK")
        assert msg.type == "ack"
        assert msg.original_id == "msg-123"
        assert msg.success is True

    def test_error_message(self):
        msg = ErrorMessage(code="FORBIDDEN", message="Access denied")
        assert msg.type == "error"
        assert msg.code == "FORBIDDEN"

    def test_detection_event_message(self):
        msg = DetectionEventMessage(
            channel="detections",
            event_id="evt-123",
            camera_id="cam_001",
            frame_id=42,
            timestamp=datetime.utcnow(),
            class_name="smoke",
            confidence=0.85,
            bbox={"x_min": 0.1, "y_min": 0.1, "x_max": 0.5, "y_max": 0.5},
            status="confirmed",
        )
        assert msg.type == "detection"
        assert msg.channel == "detections"
        assert msg.event_id == "evt-123"

    def test_detection_confirmed_message(self):
        msg = DetectionConfirmedMessage(
            event_id="evt-123",
            camera_id="cam_001",
            first_frame_id=10,
            last_frame_id=12,
            first_timestamp=datetime.utcnow(),
            last_timestamp=datetime.utcnow(),
            class_name="smoke",
            max_confidence=0.9,
            avg_confidence=0.85,
            frame_count=3,
            bbox={"x_min": 0.1, "y_min": 0.1, "x_max": 0.5, "y_max": 0.5},
        )
        assert msg.type == "detection_confirmed"

    def test_detection_pending_message(self):
        msg = DetectionPendingMessage(
            event_id="evt-123",
            camera_id="cam_001",
            frame_id=10,
            timestamp=datetime.utcnow(),
            class_name="smoke",
            confidence=0.7,
            bbox={"x_min": 0.1, "y_min": 0.1, "x_max": 0.5, "y_max": 0.5},
        )
        assert msg.type == "detection_pending"

    def test_camera_status_message(self):
        msg = CameraStatusMessage(
            camera_id="cam_001",
            status="active",
            last_frame_id=100,
            last_frame_timestamp=datetime.utcnow(),
            total_frames=1000,
            total_errors=5,
            fps=29.5,
        )
        assert msg.type == "camera_status"
        assert msg.camera_id == "cam_001"
        assert msg.status == "active"


class TestParseWSMessage:
    def test_parse_subscribe(self):
        data = {
            "type": "subscribe",
            "id": "msg-1",
            "channels": ["detections"],
            "filters": {"camera_id": "cam_001"},
        }
        msg = parse_ws_message(data)
        assert msg is not None
        assert isinstance(msg, SubscribeMessage)
        assert msg.channels == ["detections"]
        assert msg.filters == {"camera_id": "cam_001"}

    def test_parse_unsubscribe(self):
        data = {
            "type": "unsubscribe",
            "id": "msg-2",
            "channels": ["detections:pending"],
        }
        msg = parse_ws_message(data)
        assert msg is not None
        assert isinstance(msg, UnsubscribeMessage)

    def test_parse_heartbeat(self):
        data = {
            "type": "heartbeat",
            "id": "msg-3",
            "payload": {"client_time": "2024-01-01T00:00:00"},
        }
        msg = parse_ws_message(data)
        assert msg is not None
        assert isinstance(msg, HeartbeatMessage)

    def test_parse_invalid(self):
        data = {"type": "invalid_type"}
        msg = parse_ws_message(data)
        assert msg is None

    def test_parse_missing_type(self):
        data = {"id": "msg-1"}
        msg = parse_ws_message(data)
        assert msg is None


class TestSerializeWSMessage:
    def test_serialize_subscribe(self):
        msg = SubscribeMessage(channels=["detections"], filters={"camera_id": "cam_001"})
        json_str = serialize_ws_message(msg)
        assert '"type":"subscribe"' in json_str
        assert '"channels":["detections"]' in json_str

    def test_serialize_detection_event(self):
        msg = DetectionEventMessage(
            channel="detections",
            event_id="evt-1",
            camera_id="cam_001",
            frame_id=1,
            timestamp=datetime(2024, 1, 1, 12, 0, 0),
            class_name="smoke",
            confidence=0.9,
            bbox={"x_min": 0.1, "y_min": 0.1, "x_max": 0.5, "y_max": 0.5},
            status="confirmed",
        )
        json_str = serialize_ws_message(msg)
        assert '"type":"detection"' in json_str
        assert '"event_id":"evt-1"' in json_str