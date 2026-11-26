"""WebSocket connection manager."""

from __future__ import annotations

import asyncio
import json
import logging
import uuid
from collections import defaultdict
from datetime import datetime
from typing import Any, Optional, Set, Dict, List

from fastapi import WebSocket, WebSocketDisconnect

from .config import get_ws_config
from .models import (
    WSMessageType,
    WSChannelType,
    WSMessage,
    SubscribeMessage,
    UnsubscribeMessage,
    HeartbeatMessage,
    AckMessage,
    ErrorMessage,
    SubscribedMessage,
    UnsubscribedMessage,
    DetectionEventMessage,
    DetectionConfirmedMessage,
    DetectionPendingMessage,
    CameraStatusMessage,
    HeartbeatMessage as OutgoingHeartbeatMessage,
    AckMessage as OutgoingAckMessage,
    ErrorMessage as OutgoingErrorMessage,
    SubscribedMessage,
    UnsubscribedMessage,
    parse_ws_message,
    serialize_ws_message,
)

logger = logging.getLogger(__name__)


class Connection:
    """Represents a single WebSocket connection."""

    def __init__(
        self,
        websocket: WebSocket,
        client_id: str,
        user_id: Optional[str] = None,
        roles: Optional[Set[str]] = None,
    ):
        self.websocket = websocket
        self.client_id = client_id
        self.user_id = user_id
        self.roles = roles or set()
        self.subscriptions: Set[str] = set()
        self.connected_at = datetime.utcnow()
        self.last_heartbeat = datetime.utcnow()
        self.message_queue: asyncio.Queue[str] = asyncio.Queue(maxsize=1000)
        self._send_task: Optional[asyncio.Task] = None

    async def send(self, message: str) -> bool:
        """Send a message to the client."""
        try:
            await self.websocket.send_text(message)
            return True
        except Exception as e:
            logger.warning("Failed to send message to %s: %s", self.client_id, e)
            return False

    async def send_json(self, message: dict) -> bool:
        """Send a JSON message to the client."""
        return await self.send(json.dumps(message))

    def can_access_channel(self, channel: str, required_roles: Dict[str, List[str]]) -> bool:
        """Check if connection has required roles for channel."""
        if not required_roles:
            return True

        for pattern, roles in required_roles.items():
            if self._match_channel(pattern, channel):
                if not roles or any(role in self.roles for role in roles):
                    return True
                return False
        return True  # No specific role requirement

    def _match_channel(self, pattern: str, channel: str) -> bool:
        """Match channel against pattern (supports wildcards)."""
        if pattern == channel:
            return True
        if pattern.endswith("*"):
            prefix = pattern[:-1]
            return channel.startswith(prefix)
        return False


class ConnectionManager:
    """Manages WebSocket connections and message routing."""

    def __init__(self):
        self.config = get_ws_config()
        self.connections: Dict[str, Connection] = {}
        self.channel_subscribers: Dict[str, Set[str]] = defaultdict(set)
        self._heartbeat_task: Optional[asyncio.Task] = None
        self._cleanup_task: Optional[asyncio.Task] = None
        self._running = False
        self._broadcast_queue: asyncio.Queue = asyncio.Queue(maxsize=10000)

    async def start(self) -> None:
        """Start the connection manager."""
        if self._running:
            return

        self._running = True
        self._heartbeat_task = asyncio.create_task(self._heartbeat_loop())
        self._cleanup_task = asyncio.create_task(self._cleanup_loop())
        self._broadcast_task = asyncio.create_task(self._broadcast_loop())

    async def stop(self) -> None:
        """Stop the connection manager."""
        self._running = False

        if self._heartbeat_task:
            self._heartbeat_task.cancel()
        if self._cleanup_task:
            self._cleanup_task.cancel()
        if hasattr(self, '_broadcast_task') and self._broadcast_task:
            self._broadcast_task.cancel()

        # Close all connections
        for conn in list(self.connections.values()):
            await conn.websocket.close(code=1001, reason="Server shutting down")

        await asyncio.gather(
            self._heartbeat_task,
            self._cleanup_task,
            return_exceptions=True,
        )

    async def connect(
        self,
        websocket: WebSocket,
        client_id: Optional[str] = None,
        user_id: Optional[str] = None,
        roles: Optional[Set[str]] = None,
    ) -> Connection:
        """Accept a new WebSocket connection."""
        if len(self.connections) >= self.config.ws.max_connections:
            await websocket.close(code=1008, reason="Max connections reached")
            raise RuntimeError("Max connections reached")

        client_id = client_id or str(uuid.uuid4())

        # Check per-client limit
        client_connections = sum(
            1 for c in self.connections.values() if c.user_id == client_id
        )
        if client_connections >= self.config.ws.max_connections_per_client:
            await websocket.close(code=1008, reason="Max connections per client reached")
            raise RuntimeError("Max connections per client reached")

        await websocket.accept()

        conn = Connection(
            websocket=websocket,
            client_id=client_id,
            user_id=user_id,
            roles=set(),
        )
        self.connections[client_id] = conn

        # Start send task for this connection
        conn._send_task = asyncio.create_task(self._send_loop(conn))

        logger.info("Client %s connected (total: %d)", client_id, len(self.connections))
        return conn

    async def disconnect(self, client_id: str) -> None:
        """Disconnect a client."""
        conn = self.connections.pop(client_id, None)
        if not conn:
            return

        # Unsubscribe from all channels
        for channel in list(conn.subscriptions):
            await self.unsubscribe(client_id, channel)

        if conn._send_task:
            conn._send_task.cancel()
            try:
                await conn._send_task
            except asyncio.CancelledError:
                pass

        try:
            await conn.websocket.close()
        except Exception:
            pass

        logger.info("Client %s disconnected (total: %d)", client_id, len(self.connections))

    async def subscribe(self, client_id: str, channel: str, filters: Optional[dict] = None) -> bool:
        """Subscribe a client to a channel."""
        conn = self.connections.get(client_id)
        if not conn:
            return False

        # Check authorization
        if not conn.can_access_channel(channel, self.config.auth.required_roles):
            return False

        # Check if already subscribed
        if channel in conn.subscriptions:
            return True

        # Add subscription
        conn.subscriptions.add(channel)
        self.channel_subscribers[channel].add(client_id)

        # Send confirmation
        await conn.send_json(
            SubscribedMessage(
                channels=[channel],
                filters=filters,
            ).model_dump(mode="json")
        )

        logger.debug("Client %s subscribed to %s", client_id, channel)
        return True

    async def unsubscribe(self, client_id: str, channel: str) -> bool:
        """Unsubscribe a client from a channel."""
        conn = self.connections.get(client_id)
        if not conn:
            return False

        if channel not in conn.subscriptions:
            return False

        conn.subscriptions.discard(channel)
        self.channel_subscribers[channel].discard(client_id)

        # Clean up empty channel
        if not self.channel_subscribers[channel]:
            del self.channel_subscribers[channel]

        await conn.send_json(
            UnsubscribedMessage(channels=[channel]).model_dump(mode="json")
        )

        logger.debug("Client %s unsubscribed from %s", client_id, channel)
        return True

    async def send_to_client(self, client_id: str, message: dict) -> bool:
        """Send a message to a specific client."""
        conn = self.connections.get(client_id)
        if not conn:
            return False
        return await conn.send_json(message)

    async def broadcast_to_channel(
        self,
        channel: str,
        message: dict,
        exclude: Optional[Set[str]] = None,
    ) -> int:
        """Broadcast a message to all subscribers of a channel."""
        subscribers = self.channel_subscribers.get(channel, set()).copy()
        if exclude:
            subscribers -= exclude

        sent = 0
        for client_id in subscribers:
            conn = self.connections.get(client_id)
            if conn and await conn.send_json(message):
                sent += 1

        return sent

    async def broadcast(self, message: dict, exclude: Optional[Set[str]] = None) -> int:
        """Broadcast a message to all connected clients."""
        sent = 0
        for client_id, conn in self.connections.items():
            if exclude and client_id in exclude:
                continue
            if await conn.send_json(message):
                sent += 1
        return sent

    async def _send_loop(self, conn: Connection) -> None:
        """Background task to send queued messages."""
        try:
            while True:
                message = await conn.message_queue.get()
                if not await conn.send(message):
                    break
        except asyncio.CancelledError:
            pass
        except Exception as e:
            logger.warning("Send loop error for %s: %s", conn.client_id, e)

    async def queue_message(self, client_id: str, message: str) -> bool:
        """Queue a message for a client."""
        conn = self.connections.get(client_id)
        if not conn:
            return False
        try:
            conn.message_queue.put_nowait(message)
            return True
        except asyncio.QueueFull:
            logger.warning("Message queue full for client %s", client_id)
            return False

    async def _heartbeat_loop(self) -> None:
        """Send periodic heartbeats and check for stale connections."""
        while self._running:
            try:
                await asyncio.sleep(self.config.ws.heartbeat_interval_seconds)

                now = datetime.utcnow()
                stale_connections = []

                for client_id, conn in self.connections.items():
                    # Check if heartbeat timeout exceeded
                    if (now - conn.last_heartbeat).total_seconds() > self.config.ws.connection_timeout_seconds:
                        stale_connections.append(client_id)
                        continue

                    # Send heartbeat
                    heartbeat = OutgoingHeartbeatMessage(
                        payload={"server_time": datetime.utcnow().isoformat()}
                    )
                    await conn.send_json(heartbeat.model_dump(mode="json"))

                # Disconnect stale connections
                for client_id in stale_connections:
                    logger.info("Disconnecting stale client %s", client_id)
                    await self.disconnect(client_id)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.exception("Heartbeat loop error: %s", e)

    async def _cleanup_loop(self) -> None:
        """Periodic cleanup of stale data."""
        while self._running:
            try:
                await asyncio.sleep(60)  # Run every minute

                # Clean up empty channel subscriber sets
                empty_channels = [
                    ch for ch, subs in self.channel_subscribers.items() if not subs
                ]
                for ch in empty_channels:
                    del self.channel_subscribers[ch]

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.exception("Cleanup loop error: %s", e)

    async def _broadcast_loop(self) -> None:
        """Process broadcast queue."""
        while self._running:
            try:
                channel, message, exclude = await self._broadcast_queue.get()
                await self.broadcast_to_channel(channel, message, exclude)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.exception("Broadcast loop error: %s", e)

    async def queue_broadcast(
        self,
        channel: str,
        message: dict,
        exclude: Optional[Set[str]] = None,
    ) -> bool:
        """Queue a message for broadcast to a channel."""
        try:
            self._broadcast_queue.put_nowait((channel, message, exclude or set()))
            return True
        except asyncio.QueueFull:
            logger.warning("Broadcast queue full, dropping message for channel %s", channel)
            return False

    def get_stats(self) -> dict:
        """Get connection manager statistics."""
        return {
            "total_connections": len(self.connections),
            "total_subscriptions": sum(len(c.subscriptions) for c in self.connections.values()),
            "channels": {
                channel: len(subscribers)
                for channel, subscribers in self.channel_subscribers.items()
            },
            "running": self._running,
        }


# Global instance
connection_manager = ConnectionManager()