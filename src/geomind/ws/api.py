"""WebSocket API routes."""

from __future__ import annotations

from typing import Optional

from fastapi import (
    APIRouter,
    Depends,
    HTTPException,
    Query,
    WebSocket,
    WebSocketDisconnect,
    status,
)
from fastapi.security import APIKeyQuery

from ..auth.dependencies import get_current_user
from ..auth.models import UserRecord, UserRole
from .config import get_ws_config
from .manager import connection_manager
from .models import (
    WSChannelType,
    WSMessageType,
    parse_ws_message,
    serialize_ws_message,
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

router = APIRouter(prefix="/ws", tags=["websocket"])


# API key query for WebSocket auth
ws_api_key_query = APIKeyQuery(name="token", auto_error=False)


async def get_ws_user(
    websocket: WebSocket,
    token: Optional[str] = Query(None),
    api_key: Optional[str] = Depends(ws_api_key_query),
) -> Optional[UserRecord]:
    """Authenticate WebSocket connection."""
    from ..auth.utils import decode_token, verify_api_key, hash_token
    from ..auth.database import auth_db

    config = get_ws_config()
    if not config.enabled:
        return None

    ws_auth = config.auth

    # Try token from query param
    token = token or api_key

    # Try Authorization header
    if not token:
        auth_header = websocket.headers.get("Authorization")
        if auth_header and auth_header.startswith(ws_auth.token_prefix):
            token = auth_header[len(ws_auth.token_prefix):]

    if not token:
        if ws_auth.allow_anonymous:
            return None
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication required",
        )

    # Try JWT token
    from ..auth.utils import decode_token, verify_api_key, hash_token
    payload = decode_token(token)
    if payload and payload.get("type") == "access":
        user_id = payload.get("sub")
        if user_id:
            from ..auth.database import auth_db
            user = await auth_db.get_user_by_id(user_id)
            if user and user.is_active:
                return user

    # Try API key
    if token:
        key_hash = hash_token(token)
        api_key = await auth_db.get_api_key_by_hash(key_hash)
        if api_key and api_key.is_active:
            user = await auth_db.get_user_by_id(api_key.user_id)
            if user and user.is_active:
                return user

    if ws_auth.allow_anonymous:
        return None

    raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Invalid or expired authentication",
    )


def check_channel_access(user: Optional[UserRecord], channel: str, required_roles: dict) -> bool:
    """Check if user has access to a channel."""
    if not required_roles:
        return True

    user_roles = {user.role.value} if user else set()

    for pattern, roles in required_roles.items():
        if _match_channel(pattern, channel):
            if not roles or any(role in user_roles for role in roles):
                return True
            return False
    return True


def _match_channel(pattern: str, channel: str) -> bool:
    """Match channel against pattern (supports wildcards)."""
    if pattern == channel:
        return True
    if pattern.endswith("*"):
        prefix = pattern[:-1]
        return channel.startswith(prefix)
    return False


@router.websocket("/connect")
async def websocket_endpoint(
    websocket: WebSocket,
    user: Optional[UserRecord] = Depends(get_ws_user),
):
    """WebSocket endpoint for real-time detection streaming."""
    config = get_ws_config()
    if not config.enabled:
        await websocket.close(code=1011, reason="WebSocket not enabled")
        return

    # Authenticate
    user = await get_ws_user(websocket)
    if not user and not get_ws_config().auth.allow_anonymous:
        await websocket.close(code=4001, reason="Authentication required")
        return

    client_id = f"user_{user.id}" if user else f"anon_{uuid.uuid4().hex[:8]}"
    roles = {user.role.value} if user else set()

    # Connect
    conn = await connection_manager.connect(
        websocket=websocket,
        client_id=client_id,
        user_id=str(user.id) if user else None,
        roles={user.role.value} if user else set(),
    )

    try:
        while True:
            # Receive message
            data = await websocket.receive_text()

            # Parse message
            try:
                data = json.loads(data)
            except json.JSONDecodeError:
                error = ErrorMessage(code="INVALID_JSON", message="Invalid JSON")
                await conn.send_json(error.model_dump(mode="json"))
                continue

            # Parse message
            message = parse_ws_message(data)
            if not message:
                error = ErrorMessage(code="INVALID_MESSAGE", message="Invalid message format")
                await conn.send_json(error.model_dump(mode="json"))
                continue

            # Handle message based on type
            if isinstance(message, SubscribeMessage):
                await handle_subscribe(message, conn, user)
            elif isinstance(message, UnsubscribeMessage):
                await handle_unsubscribe(message, conn)
            elif isinstance(message, HeartbeatMessage):
                await handle_heartbeat(message, conn)
            elif isinstance(message, AckMessage):
                # Client acknowledging our message - can be ignored or logged
                pass
            else:
                error = ErrorMessage(
                    code="UNSUPPORTED_MESSAGE",
                    message=f"Message type {message.type} not supported"
                )
                await conn.send_json(error.model_dump(mode="json"))

    except WebSocketDisconnect:
        pass
    except Exception as e:
        logger.exception("WebSocket error: %s", e)
    finally:
        await connection_manager.disconnect(conn.client_id)


async def handle_subscribe(
    message: SubscribeMessage,
    conn,
    user: Optional[UserRecord],
):
    """Handle subscription request."""
    config = get_ws_config()
    results = []

    for channel in message.channels:
        # Check access
        if not check_channel_access(None, channel, config.auth.required_roles):
            error = ErrorMessage(
                code="FORBIDDEN",
                message=f"Access denied to channel: {channel}",
            )
            await conn.send_json(error.model_dump(mode="json"))
            results.append({"channel": channel, "success": False, "error": "Forbidden"})
            continue

        # Apply filters if provided
        filters = message.filters

        success = await connection_manager.subscribe(message.client_id, channel, filters)
        results.append({"channel": channel, "success": success})

    # Send acknowledgment
    ack = AckMessage(
        original_id=message.id,
        success=all(r["success"] for r in results),
        message=f"Subscribed to {len([r for r in results if r['success']])}/{len(results)} channels",
    )
    await conn.send_json(ack.model_dump(mode="json"))


async def handle_unsubscribe(
    message: UnsubscribeMessage,
    conn,
):
    """Handle unsubscription request."""
    results = []

    for channel in message.channels:
        success = await connection_manager.unsubscribe(message.client_id, channel)
        results.append({"channel": channel, "success": success})

    ack = AckMessage(
        original_id=message.id,
        success=all(r["success"] for r in results),
        message=f"Unsubscribed from {len([r for r in results if r['success']])}/{len(results)} channels",
    )
    await conn.send_json(ack.model_dump(mode="json"))


async def handle_heartbeat(message: HeartbeatMessage, conn):
    """Handle heartbeat/ping."""
    # Update last heartbeat
    conn.last_heartbeat = datetime.utcnow()

    # Send pong/ack
    ack = AckMessage(
        original_id=message.id,
        success=True,
        message="pong",
    )
    await conn.send_json(ack.model_dump(mode="json"))


# Event publishing functions (called from vision engine)
async def publish_detection_event(
    event_id: str,
    camera_id: str,
    frame_id: int,
    timestamp: datetime,
    class_name: str,
    confidence: float,
    bbox: dict,
    status: str,
    channel: WSChannelType = WSChannelType.DETECTIONS,
):
    """Publish a detection event to subscribers."""
    message = DetectionEventMessage(
        channel=channel,
        event_id=event_id,
        camera_id=camera_id,
        frame_id=frame_id,
        timestamp=timestamp,
        class_name=class_name,
        confidence=confidence,
        bbox=bbox,
        status=status,
    )
    await connection_manager.queue_broadcast(channel.value, message.model_dump(mode="json"))


async def publish_detection_confirmed(
    event_id: str,
    camera_id: str,
    first_frame_id: int,
    last_frame_id: int,
    first_timestamp: datetime,
    last_timestamp: datetime,
    class_name: str,
    max_confidence: float,
    avg_confidence: float,
    frame_count: int,
    bbox: dict,
):
    """Publish a confirmed detection event."""
    message = DetectionConfirmedMessage(
        event_id=event_id,
        camera_id=camera_id,
        first_frame_id=first_frame_id,
        last_frame_id=last_frame_id,
        first_timestamp=first_timestamp,
        last_timestamp=last_timestamp,
        class_name=class_name,
        max_confidence=max_confidence,
        avg_confidence=avg_confidence,
        frame_count=frame_count,
        bbox=bbox,
    )
    await connection_manager.queue_broadcast(
        WSChannelType.DETECTIONS_CONFIRMED.value,
        message.model_dump(mode="json"),
    )


async def publish_detection_pending(
    event_id: str,
    camera_id: str,
    frame_id: int,
    timestamp: datetime,
    class_name: str,
    confidence: float,
    bbox: dict,
):
    """Publish a pending detection event."""
    message = DetectionPendingMessage(
        event_id=event_id,
        camera_id=camera_id,
        frame_id=frame_id,
        timestamp=timestamp,
        class_name=class_name,
        confidence=confidence,
        bbox=bbox,
    )
    await connection_manager.queue_broadcast(
        WSChannelType.DETECTIONS_PENDING.value,
        message.model_dump(mode="json"),
    )


async def publish_camera_status(
    camera_id: str,
    status: str,
    last_frame_id: Optional[int] = None,
    last_frame_timestamp: Optional[datetime] = None,
    total_frames: int = 0,
    total_errors: int = 0,
    fps: Optional[float] = None,
):
    """Publish camera status update."""
    message = CameraStatusMessage(
        camera_id=camera_id,
        status=status,
        last_frame_id=last_frame_id,
        last_frame_timestamp=last_frame_timestamp,
        total_frames=total_frames,
        total_errors=total_errors,
        fps=fps,
    )
    await connection_manager.queue_broadcast(
        WSChannelType.CAMERAS.value,
        message.model_dump(mode="json"),
    )


# REST endpoints for WebSocket info
ws_info_router = APIRouter(prefix="/ws", tags=["websocket"])


@ws_info_router.get("/stats")
async def get_ws_stats():
    """Get WebSocket connection statistics."""
    return connection_manager.get_stats()


@ws_info_router.get("/channels")
async def list_channels():
    """List active WebSocket channels and subscriber counts."""
    return {
        channel: len(subscribers)
        for channel, subscribers in connection_manager.channel_subscribers.items()
    }


@ws_info_router.get("/connections")
async def list_connections():
    """List active WebSocket connections (admin only)."""
    return {
        "total": len(connection_manager.connections),
        "connections": [
            {
                "client_id": c.client_id,
                "user_id": c.user_id,
                "roles": list(c.roles),
                "subscriptions": list(c.subscriptions),
                "connected_at": c.connected_at.isoformat(),
                "last_heartbeat": c.last_heartbeat.isoformat(),
            }
            for c in connection_manager.connections.values()
        ]
    }


# Import at bottom to avoid circular imports
import json
import uuid
from datetime import datetime
from ..auth.database import auth_db
from ..auth.utils import decode_token, verify_api_key, hash_token