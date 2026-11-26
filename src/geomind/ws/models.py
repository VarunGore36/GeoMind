"""WebSocket models and message types."""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any, Optional
from uuid import UUID, uuid4

from pydantic import BaseModel, Field, field_validator


class WSMessageType(str, Enum):
    """WebSocket message types."""

    # Client -> Server
    SUBSCRIBE = "subscribe"
    UNSUBSCRIBE = "unsubscribe"
    HEARTBEAT = "heartbeat"
    ACK = "ack"

    # Server -> Client
    DETECTION = "detection"
    DETECTION_CONFIRMED = "detection_confirmed"
    DETECTION_PENDING = "detection_pending"
    CAMERA_STATUS = "camera_status"
    ERROR = "error"
    SUBSCRIBED = "subscribed"
    UNSUBSCRIBED = "unsubscribed"


class WSChannelType(str, Enum):
    """WebSocket channel types."""

    DETECTIONS = "detections"
    DETECTIONS_CONFIRMED = "detections:confirmed"
    DETECTIONS_PENDING = "detections:pending"
    CAMERAS = "cameras"
    SYSTEM = "system"


class WSMessage(BaseModel):
    """Base WebSocket message."""

    type: WSMessageType
    id: str = Field(default_factory=lambda: str(uuid4()))
    timestamp: datetime = Field(default_factory=datetime.utcnow)

    @field_validator("timestamp", mode="before")
    @classmethod
    def parse_timestamp(cls, v: Any) -> datetime:
        if isinstance(v, str):
            return datetime.fromisoformat(v.replace("Z", "+00:00"))
        return v


class SubscribeMessage(WSMessage):
    """Subscribe to channels."""

    type: WSMessageType = WSMessageType.SUBSCRIBE
    channels: list[str] = Field(..., min_length=1)
    filters: Optional[dict[str, Any]] = None


class UnsubscribeMessage(WSMessage):
    """Unsubscribe from channels."""

    type: WSMessageType = WSMessageType.UNSUBSCRIBE
    channels: list[str] = Field(..., min_length=1)


class HeartbeatMessage(WSMessage):
    """Heartbeat/ping message."""

    type: WSMessageType = WSMessageType.HEARTBEAT
    payload: Optional[dict[str, Any]] = None


class AckMessage(WSMessage):
    """Acknowledgment message."""

    type: WSMessageType = WSMessageType.ACK
    original_id: str
    success: bool = True
    message: Optional[str] = None


class ErrorMessage(WSMessage):
    """Error message."""

    type: WSMessageType = WSMessageType.ERROR
    code: str
    message: str
    details: Optional[dict[str, Any]] = None


class SubscribedMessage(WSMessage):
    """Subscription confirmation."""

    type: WSMessageType = WSMessageType.SUBSCRIBED
    channels: list[str]
    filters: Optional[dict[str, Any]] = None


class UnsubscribedMessage(WSMessage):
    """Unsubscription confirmation."""

    type: WSMessageType = WSMessageType.UNSUBSCRIBED
    channels: list[str]


# Detection event messages (server -> client)
class DetectionEventMessage(WSMessage):
    """Detection event broadcast."""

    type: WSMessageType = WSMessageType.DETECTION
    channel: WSChannelType
    event_id: str
    camera_id: str
    frame_id: int
    timestamp: datetime
    class_name: str
    confidence: float
    bbox: dict[str, float]  # x_min, y_min, x_max, y_max (normalized 0-1)
    status: str  # pending, confirmed, rejected


class DetectionConfirmedMessage(WSMessage):
    """Confirmed detection event."""

    type: WSMessageType = WSMessageType.DETECTION_CONFIRMED
    event_id: str
    camera_id: str
    first_frame_id: int
    last_frame_id: int
    first_timestamp: datetime
    last_timestamp: datetime
    class_name: str
    max_confidence: float
    avg_confidence: float
    frame_count: int
    bbox: dict[str, float]


class DetectionPendingMessage(WSMessage):
    """Pending detection event."""

    type: WSMessageType = WSMessageType.DETECTION_PENDING
    event_id: str
    camera_id: str
    frame_id: int
    timestamp: datetime
    class_name: str
    confidence: float
    bbox: dict[str, float]


class CameraStatusMessage(WSMessage):
    """Camera status update."""

    type: WSMessageType = WSMessageType.CAMERA_STATUS
    camera_id: str
    status: str  # active, inactive, error
    last_frame_id: Optional[int] = None
    last_frame_timestamp: Optional[datetime] = None
    total_frames: int
    total_errors: int
    fps: Optional[float] = None


# Union of all message types for parsing
WSIncomingMessage = (
    SubscribeMessage | UnsubscribeMessage | HeartbeatMessage | AckMessage
)
WSOutgoingMessage = (
    DetectionEventMessage
    | DetectionConfirmedMessage
    | DetectionPendingMessage
    | CameraStatusMessage
    | HeartbeatMessage
    | AckMessage
    | ErrorMessage
    | SubscribedMessage
    | UnsubscribedMessage
)

AnyWSMessage = WSIncomingMessage | WSOutgoingMessage


def parse_ws_message(data: dict[str, Any]) -> Optional[WSIncomingMessage]:
    """Parse incoming WebSocket message."""
    try:
        msg_type = data.get("type")
        if not msg_type:
            return None

        if msg_type == WSMessageType.SUBSCRIBE:
            return SubscribeMessage(**data)
        elif msg_type == WSMessageType.UNSUBSCRIBE:
            return UnsubscribeMessage(**data)
        elif msg_type == WSMessageType.HEARTBEAT:
            return HeartbeatMessage(**data)
        elif msg_type == WSMessageType.ACK:
            return AckMessage(**data)
        return None
    except Exception:
        return None


def serialize_ws_message(message: WSOutgoingMessage) -> str:
    """Serialize outgoing WebSocket message to JSON."""
    return message.model_dump_json(exclude_none=True)