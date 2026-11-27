"""
GeoMind API Client

Provides synchronous and asynchronous clients for the GeoMind API.
"""

from __future__ import annotations

import asyncio
import json
import logging
from contextlib import asynccontextmanager
from datetime import datetime
from typing import Any, AsyncIterator, Optional
from urllib.parse import urljoin

import httpx
import websockets

from .models import (
    LocationIntelligenceResponse,
    Camera,
    Frame,
    DetectionEvent,
    CameraStatus,
)

logger = logging.getLogger(__name__)


class GeomindClient:
    """
    Synchronous client for GeoMind API.

    Example:
        import os
        client = GeomindClient(
            base_url="http://localhost:8000",
            api_key=os.getenv("GEOMIND_API_KEY")
        )
        result = client.location_intelligence(lat=52.52, lon=13.405, radius=500)
    """

    def __init__(
        self,
        base_url: str,
        api_key: Optional[str] = None,
        token: Optional[str] = None,
        timeout: float = 30.0,
        verify_ssl: bool = True,
    ):
        """
        Initialize client.

        Args:
            base_url: Base URL of GeoMind API (e.g., "http://localhost:8000")
            api_key: API key for authentication
            token: JWT access token for authentication
            timeout: Request timeout in seconds
            verify_ssl: Whether to verify SSL certificates
        """
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.token = token
        self.timeout = timeout
        self.verify_ssl = verify_ssl

        self._client = httpx.Client(
            base_url=self.base_url,
            timeout=timeout,
            verify=verify_ssl,
        )
        self._setup_auth()

    def _setup_auth(self) -> None:
        """Configure authentication headers."""
        if self.api_key:
            self._client.headers["Authorization"] = f"Bearer {self.api_key}"
        elif self.token:
            self._client.headers["Authorization"] = f"Bearer {self.token}"

    def _request(
        self,
        method: str,
        path: str,
        **kwargs,
    ) -> httpx.Response:
        """Make HTTP request with error handling."""
        url = urljoin(self.base_url + "/", path.lstrip("/"))
        response = self._client.request(method, url, **kwargs)
        response.raise_for_status()
        return response

    def close(self) -> None:
        """Close the HTTP client."""
        self._client.close()

    def __enter__(self) -> "GeomindClient":
        return self

    def __exit__(self, *args) -> None:
        self.close()

    # Location Intelligence

    def location_intelligence(
        self,
        lat: float,
        lon: float,
        radius: float = 500.0,
        water_radius: Optional[float] = None,
        use_cache: bool = True,
    ) -> LocationIntelligenceResponse:
        """
        Get location intelligence for a point.

        Args:
            lat: Latitude in decimal degrees
            lon: Longitude in decimal degrees
            radius: Search radius in meters (default: 500)
            water_radius: Water search radius in meters (default: radius + 2000)
            use_cache: Whether to use cached results (default: True)

        Returns:
            LocationIntelligenceResponse with features and provenance
        """
        params = {
            "lat": lat,
            "lon": lon,
            "radius_m": radius,
            "use_cache": use_cache,
        }
        if water_radius:
            params["water_radius_m"] = water_radius

        response = self._request("GET", "/v1/location-intelligence", params=params)
        return LocationIntelligenceResponse.model_validate(response.json())

    # Health

    def health(self) -> dict:
        """Check API health (liveness)."""
        response = self._request("GET", "/health")
        return response.json()

    def readiness(self) -> dict:
        """Check API readiness (database + PostGIS)."""
        response = self._request("GET", "/health/ready")
        return response.json()

    # Cameras

    def list_cameras(self) -> list[Camera]:
        """List all cameras."""
        response = self._request("GET", "/api/v1/cameras")
        return [Camera.model_validate(c) for c in response.json()]

    def get_camera(self, camera_id: str) -> Camera:
        """Get camera by ID."""
        response = self._request("GET", f"/api/v1/cameras/{camera_id}")
        return Camera.model_validate(response.json())

    def get_latest_frame(self, camera_id: str) -> Frame:
        """Get latest frame metadata for a camera."""
        response = self._request("GET", f"/api/v1/cameras/{camera_id}/latest")
        return Frame.model_validate(response.json())

    def get_latest_frame_image(self, camera_id: str) -> bytes:
        """Get latest frame image for a camera."""
        response = self._request("GET", f"/api/v1/cameras/{camera_id}/latest/image")
        return response.content

    def trigger_camera_fetch(self, camera_id: str) -> dict:
        """Manually trigger frame fetch for a camera."""
        response = self._request("POST", f"/api/v1/cameras/{camera_id}/trigger")
        return response.json()

    # WebSocket streaming (async)

    async def stream_detections(
        self,
        channels: Optional[list[str]] = None,
        filters: Optional[dict] = None,
        token: Optional[str] = None,
    ) -> AsyncIterator[DetectionEvent | CameraStatus]:
        """
        Stream real-time detection events via WebSocket.

        Args:
            channels: Channels to subscribe to (default: all detection channels)
            filters: Subscription filters (camera_id, class_name, min_confidence)
            token: Override authentication token

        Yields:
            DetectionEvent or CameraStatus objects

        Example:
            async for event in client.stream_detections():
                if isinstance(event, DetectionEvent):
                    print(f"Fire detected: {event.confidence}")
        """
        ws_url = self.base_url.replace("http", "ws") + "/ws/connect"
        auth_token = token or self.api_key or self.token

        params = {}
        if auth_token:
            params["token"] = auth_token

        if params:
            ws_url += "?" + "&".join(f"{k}={v}" for k, v in params.items())

        channels = channels or ["detections", "detections:confirmed", "cameras"]
        filters = filters or {}

        async with websockets.connect(ws_url) as ws:
            # Subscribe
            subscribe_msg = {
                "type": "subscribe",
                "id": f"sub_{datetime.utcnow().timestamp()}",
                "timestamp": datetime.utcnow().isoformat() + "Z",
                "channels": channels,
                "filters": filters,
            }
            await ws.send(json.dumps(subscribe_msg))

            # Wait for subscription confirmation
            ack = await ws.recv()
            ack_data = json.loads(ack)
            if not ack_data.get("success", False):
                raise RuntimeError(f"Subscription failed: {ack_data.get('message')}")

            logger.info(f"Subscribed to channels: {channels}")

            # Message loop
            async for message in ws:
                try:
                    data = json.loads(message)
                    msg_type = data.get("type")

                    if msg_type == "detection":
                        yield DetectionEvent(**data)
                    elif msg_type == "detection_confirmed":
                        yield DetectionEvent(**data)  # Same model, different type field
                    elif msg_type == "detection_pending":
                        yield DetectionEvent(**data)
                    elif msg_type == "camera_status":
                        yield CameraStatus(**data)
                    elif msg_type == "error":
                        logger.error(f"WebSocket error: {data.get('code')} - {data.get('message')}")
                    elif msg_type == "ack":
                        logger.debug(f"WebSocket ack: {data.get('message')}")

                except json.JSONDecodeError:
                    logger.warning(f"Invalid JSON: {message}")
                except Exception as e:
                    logger.exception(f"Error processing message: {e}")

    async def stream_detections_raw(
        self,
        channels: Optional[list[str]] = None,
        filters: Optional[dict] = None,
        token: Optional[str] = None,
    ) -> AsyncIterator[dict]:
        """Stream raw WebSocket messages as dictionaries."""
        ws_url = self.base_url.replace("http", "ws") + "/ws/connect"
        auth_token = token or self.api_key or self.token

        params = {}
        if auth_token:
            params["token"] = auth_token

        if params:
            ws_url += "?" + "&".join(f"{k}={v}" for k, v in params.items())

        channels = channels or ["detections", "detections:confirmed", "cameras"]
        filters = filters or {}

        async with websockets.connect(ws_url) as ws:
            subscribe_msg = {
                "type": "subscribe",
                "id": f"sub_{datetime.utcnow().timestamp()}",
                "timestamp": datetime.utcnow().isoformat() + "Z",
                "channels": channels,
                "filters": filters,
            }
            await ws.send(json.dumps(subscribe_msg))

            async for message in ws:
                try:
                    yield json.loads(message)
                except json.JSONDecodeError:
                    continue


class AsyncGeomindClient:
    """
    Asynchronous client for GeoMind API.

    Example:
        async with AsyncGeomindClient(base_url="http://localhost:8000", api_key="key") as client:
            result = await client.location_intelligence(lat=52.52, lon=13.405, radius=500)
    """

    def __init__(
        self,
        base_url: str,
        api_key: Optional[str] = None,
        token: Optional[str] = None,
        timeout: float = 30.0,
        verify_ssl: bool = True,
    ):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.token = token
        self.timeout = timeout
        self.verify_ssl = verify_ssl

        self._client: Optional[httpx.AsyncClient] = None

    async def _get_client(self) -> httpx.AsyncClient:
        """Get or create async HTTP client."""
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(
                base_url=self.base_url,
                timeout=self.timeout,
                verify=self.verify_ssl,
            )
            if self.api_key:
                self._client.headers["Authorization"] = f"Bearer {self.api_key}"
            elif self.token:
                self._client.headers["Authorization"] = f"Bearer {self.token}"
        return self._client

    async def close(self) -> None:
        """Close the HTTP client."""
        if self._client and not self._client.is_closed:
            await self._client.aclose()

    async def __aenter__(self) -> "AsyncGeomindClient":
        return self

    async def __aexit__(self, *args) -> None:
        await self.close()

    async def _request(
        self,
        method: str,
        path: str,
        **kwargs,
    ) -> httpx.Response:
        """Make HTTP request with error handling."""
        client = await self._get_client()
        url = urljoin(self.base_url + "/", path.lstrip("/"))
        response = await client.request(method, url, **kwargs)
        response.raise_for_status()
        return response

    # Location Intelligence

    async def location_intelligence(
        self,
        lat: float,
        lon: float,
        radius: float = 500.0,
        water_radius: Optional[float] = None,
        use_cache: bool = True,
    ) -> LocationIntelligenceResponse:
        """Get location intelligence for a point (async)."""
        params = {
            "lat": lat,
            "lon": lon,
            "radius_m": radius,
            "use_cache": use_cache,
        }
        if water_radius:
            params["water_radius_m"] = water_radius

        response = await self._request("GET", "/v1/location-intelligence", params=params)
        return LocationIntelligenceResponse.model_validate(response.json())

    # Health

    async def health(self) -> dict:
        """Check API health (liveness)."""
        response = await self._request("GET", "/health")
        return response.json()

    async def readiness(self) -> dict:
        """Check API readiness."""
        response = await self._request("GET", "/health/ready")
        return response.json()

    # Cameras

    async def list_cameras(self) -> list[Camera]:
        """List all cameras."""
        response = await self._request("GET", "/api/v1/cameras")
        return [Camera.model_validate(c) for c in response.json()]

    async def get_camera(self, camera_id: str) -> Camera:
        """Get camera by ID."""
        response = await self._request("GET", f"/api/v1/cameras/{camera_id}")
        return Camera.model_validate(response.json())

    async def get_latest_frame(self, camera_id: str) -> Frame:
        """Get latest frame metadata for a camera."""
        response = await self._request("GET", f"/api/v1/cameras/{camera_id}/latest")
        return Frame.model_validate(response.json())

    async def get_latest_frame_image(self, camera_id: str) -> bytes:
        """Get latest frame image for a camera."""
        response = await self._request("GET", f"/api/v1/cameras/{camera_id}/latest/image")
        return response.content

    async def trigger_camera_fetch(self, camera_id: str) -> dict:
        """Manually trigger frame fetch for a camera."""
        response = await self._request("POST", f"/api/v1/cameras/{camera_id}/trigger")
        return response.json()

    # WebSocket streaming (reuse sync client's implementation)
    stream_detections = GeomindClient.stream_detections
    stream_detections_raw = GeomindClient.stream_detections_raw