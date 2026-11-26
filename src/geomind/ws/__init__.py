"""WebSocket module for real-time detection streaming."""

from .config import WSConfigMain, get_ws_config
from .manager import ConnectionManager, connection_manager
from .models import (
    WSMessageType,
    WSChannelType,
    WSMessage,
    SubscribeMessage,
    UnsubscribeMessage,
    HeartbeatMessage,
    AckMessage,
    ErrorMessage,
    DetectionEventMessage,
    DetectionConfirmedMessage,
    DetectionPendingMessage,
    CameraStatusMessage,
)
from .api import (
    router as ws_router,
    ws_info_router,
    publish_detection_event,
    publish_detection_confirmed,
    publish_detection_pending,
    publish_camera_status,
)
from .manager import ConnectionManager, connection_manager

__version__ = "0.1.0"

__all__ = [
    # Config
    "WSConfigMain",
    "get_ws_config",
    # Manager
    "ConnectionManager",
    "connection_manager",
    # Models
    "WSMessageType",
    "WSChannelType",
    "WSMessage",
    "SubscribeMessage",
    "UnsubscribeMessage",
    "HeartbeatMessage",
    "AckMessage",
    "ErrorMessage",
    "DetectionEventMessage",
    "DetectionConfirmedMessage",
    "DetectionPendingMessage",
    "CameraStatusMessage",
    # API
    "ws_router",
    "ws_info_router",
    "publish_detection_event",
    "publish_detection_confirmed",
    "publish_detection_pending",
    "publish_camera_status",
    # Manager
    "ConnectionManager",
    "connection_manager",
]