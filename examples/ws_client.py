#!/usr/bin/env python3
"""
WebSocket client example for GeoMind real-time detection streaming.

This example demonstrates how to connect to the GeoMind WebSocket API,
subscribe to detection channels, and handle incoming messages.

Usage:
    # Basic usage (anonymous if allowed)
    python examples/ws_client.py --url ws://localhost:8000/ws/connect

    # With JWT token
    python examples/ws_client.py --url ws://localhost:8000/ws/connect --token <JWT_TOKEN>

    # With API key
    python examples/ws_client.py --url ws://localhost:8000/ws/connect --api-key <API_KEY>

    # Subscribe to specific channels
    python examples/ws_client.py --url ws://localhost:8000/ws/connect --channels detections detections:confirmed
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import signal
import sys
from datetime import datetime
from typing import Any
from urllib.parse import urlencode

import aiohttp

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


class GeomindWSClient:
    """WebSocket client for GeoMind detection streaming."""

    def __init__(
        self,
        url: str,
        token: str | None = None,
        api_key: str | None = None,
        channels: list[str] | None = None,
        filters: dict[str, Any] | None = None,
        heartbeat_interval: float = 30.0,
    ):
        self.url = url
        self.token = token
        self.api_key = api_key
        self.channels = channels or ["detections", "detections:confirmed", "cameras"]
        self.filters = filters or {}
        self.heartbeat_interval = heartbeat_interval

        self.ws: aiohttp.ClientWebSocketResponse | None = None
        self.session: aiohttp.ClientSession | None = None
        self.running = False
        self.message_id = 0
        self.subscribed_channels: set[str] = set()

    def _next_id(self) -> str:
        self.message_id += 1
        return f"msg_{self.message_id}"

    def _build_url(self) -> str:
        """Build WebSocket URL with authentication."""
        params = {}
        if self.token:
            params["token"] = self.token
        elif self.api_key:
            params["token"] = self.api_key

        if params:
            return f"{self.url}?{urlencode(params)}"
        return self.url

    async def connect(self) -> None:
        """Connect to WebSocket server."""
        self.session = aiohttp.ClientSession()
        ws_url = self._build_url()
        logger.info(f"Connecting to {ws_url}")

        try:
            self.ws = await self.session.ws_connect(ws_url, heartbeat=self.heartbeat_interval)
            logger.info("Connected successfully")
        except aiohttp.ClientError as e:
            logger.error(f"Connection failed: {e}")
            await self.close()
            raise

    async def close(self) -> None:
        """Close connection."""
        self.running = False
        if self.ws and not self.ws.closed:
            await self.ws.close()
        if self.session and not self.session.closed:
            await self.session.close()
        logger.info("Disconnected")

    async def send(self, message: dict[str, Any]) -> None:
        """Send a message."""
        if not self.ws or self.ws.closed:
            raise RuntimeError("Not connected")
        await self.ws.send_str(json.dumps(message))

    async def subscribe(self, channels: list[str] | None = None, filters: dict[str, Any] | None = None) -> None:
        """Subscribe to channels."""
        channels = channels or self.channels
        filters = filters or self.filters

        message = {
            "type": "subscribe",
            "id": self._next_id(),
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "channels": channels,
            "filters": filters,
        }
        await self.send(message)
        logger.info(f"Subscribed to: {channels}")

    async def unsubscribe(self, channels: list[str]) -> None:
        """Unsubscribe from channels."""
        message = {
            "type": "unsubscribe",
            "id": self._next_id(),
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "channels": channels,
        }
        await self.send(message)
        logger.info(f"Unsubscribed from: {channels}")

    async def heartbeat(self, payload: dict[str, Any] | None = None) -> None:
        """Send heartbeat/ping."""
        message = {
            "type": "heartbeat",
            "id": self._next_id(),
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "payload": payload or {},
        }
        await self.send(message)

    async def handle_message(self, message: dict[str, Any]) -> None:
        """Handle incoming message."""
        msg_type = message.get("type")

        if msg_type == "detection":
            self._handle_detection(message)
        elif msg_type == "detection_confirmed":
            self._handle_detection_confirmed(message)
        elif msg_type == "detection_pending":
            self._handle_detection_pending(message)
        elif msg_type == "camera_status":
            self._handle_camera_status(message)
        elif msg_type == "error":
            self._handle_error(message)
        elif msg_type == "ack":
            self._handle_ack(message)
        elif msg_type == "subscribed":
            self._handle_subscribed(message)
        elif msg_type == "unsubscribed":
            self._handle_unsubscribed(message)
        else:
            logger.warning(f"Unknown message type: {msg_type}")

    def _handle_detection(self, message: dict[str, Any]) -> None:
        """Handle detection event."""
        logger.info(
            f"🔥 DETECTION: camera={message['camera_id']} "
            f"class={message['class_name']} "
            f"conf={message['confidence']:.2f} "
            f"status={message['status']} "
            f"bbox={message['bbox']} "
            f"frame={message['frame_id']}"
        )

    def _handle_detection_confirmed(self, message: dict[str, Any]) -> None:
        """Handle confirmed detection event."""
        logger.info(
            f"✅ CONFIRMED: camera={message['camera_id']} "
            f"class={message['class_name']} "
            f"max_conf={message['max_confidence']:.2f} "
            f"avg_conf={message['avg_confidence']:.2f} "
            f"frames={message['frame_count']} "
            f"event_id={message['event_id']}"
        )

    def _handle_detection_pending(self, message: dict[str, Any]) -> None:
        """Handle pending detection event."""
        logger.info(
            f"⏳ PENDING: camera={message['camera_id']} "
            f"class={message['class_name']} "
            f"conf={message['confidence']:.2f} "
            f"frame={message['frame_id']} "
            f"event_id={message['event_id']}"
        )

    def _handle_camera_status(self, message: dict[str, Any]) -> None:
        """Handle camera status update."""
        logger.info(
            f"📷 CAMERA: {message['camera_id']} "
            f"status={message['status']} "
            f"frames={message['total_frames']} "
            f"errors={message['total_errors']} "
            f"fps={message.get('fps', 'N/A')}"
        )

    def _handle_error(self, message: dict[str, Any]) -> None:
        """Handle error message."""
        logger.error(f"❌ ERROR: {message['code']} - {message['message']}")

    def _handle_ack(self, message: dict[str, Any]) -> None:
        """Handle acknowledgment."""
        if not message.get("success", True):
            logger.warning(f"⚠️ ACK failed: {message.get('message', 'Unknown error')}")
        else:
            logger.debug(f"✓ ACK: {message.get('message', 'OK')}")

    def _handle_subscribed(self, message: dict[str, Any]) -> None:
        """Handle subscription confirmation."""
        self.subscribed_channels.update(message.get("channels", []))
        logger.info(f"✓ Subscribed: {message.get('channels', [])}")

    def _handle_unsubscribed(self, message: dict[str, Any]) -> None:
        """Handle unsubscription confirmation."""
        for ch in message.get("channels", []):
            self.subscribed_channels.discard(ch)
        logger.info(f"✓ Unsubscribed: {message.get('channels', [])}")

    async def run(self) -> None:
        """Run the client main loop."""
        await self.connect()
        await self.subscribe()

        self.running = True
        last_heartbeat = asyncio.get_event_loop().time()

        try:
            async for msg in self.ws:
                if not self.running:
                    break

                if msg.type == aiohttp.WSMsgType.TEXT:
                    try:
                        data = json.loads(msg.data)
                        await self.handle_message(data)
                    except json.JSONDecodeError:
                        logger.warning(f"Invalid JSON: {msg.data}")
                elif msg.type == aiohttp.WSMsgType.ERROR:
                    logger.error(f"WebSocket error: {self.ws.exception()}")
                    break
                elif msg.type == aiohttp.WSMsgType.CLOSED:
                    logger.info("WebSocket closed by server")
                    break

                # Send heartbeat periodically
                now = asyncio.get_event_loop().time()
                if now - last_heartbeat >= self.heartbeat_interval:
                    await self.heartbeat()
                    last_heartbeat = now

        except asyncio.CancelledError:
            pass
        except Exception as e:
            logger.exception(f"Client error: {e}")
        finally:
            await self.close()


async def main():
    parser = argparse.ArgumentParser(
        description="GeoMind WebSocket client for real-time detection streaming",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Connect anonymously (if allowed)
  python examples/ws_client.py --url ws://localhost:8000/ws/connect

  # With JWT token
  python examples/ws_client.py --url ws://localhost:8000/ws/connect --token <JWT>

  # With API key
  python examples/ws_client.py --url ws://localhost:8000/ws/connect --api-key <API_KEY>

  # Subscribe to specific channels
  python examples/ws_client.py --url ws://localhost:8000/ws/connect --channels detections detections:confirmed

  # With filters (camera_id, class_name, min_confidence)
  python examples/ws_client.py --url ws://localhost:8000/ws/connect --filters '{"camera_id": "cam_1", "min_confidence": 0.5}'
        """,
    )
    parser.add_argument(
        "--url",
        default="ws://localhost:8000/ws/connect",
        help="WebSocket URL (default: ws://localhost:8000/ws/connect)",
    )
    parser.add_argument(
        "--token",
        help="JWT access token for authentication",
    )
    parser.add_argument(
        "--api-key",
        help="API key for authentication",
    )
    parser.add_argument(
        "--channels",
        nargs="+",
        default=["detections", "detections:confirmed", "cameras"],
        help="Channels to subscribe to (default: detections detections:confirmed cameras)",
    )
    parser.add_argument(
        "--filters",
        type=json.loads,
        default={},
        help='JSON filters for subscription (e.g., \'{"camera_id": "cam_1"}\')',
    )
    parser.add_argument(
        "--heartbeat",
        type=float,
        default=30.0,
        help="Heartbeat interval in seconds (default: 30)",
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable verbose logging",
    )

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Validate auth
    if args.token and args.api_key:
        logger.error("Cannot use both --token and --api-key")
        return 1

    client = GeomindWSClient(
        url=args.url,
        token=args.token,
        api_key=args.api_key,
        channels=args.channels,
        filters=args.filters,
        heartbeat_interval=args.heartbeat,
    )

    # Handle signals
    loop = asyncio.get_event_loop()

    def signal_handler():
        logger.info("Shutdown signal received")
        client.running = False

    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, signal_handler)
        except NotImplementedError:
            # Windows doesn't support add_signal_handler
            pass

    try:
        await client.run()
    except KeyboardInterrupt:
        logger.info("Interrupted")
    except Exception as e:
        logger.exception(f"Fatal error: {e}")
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))