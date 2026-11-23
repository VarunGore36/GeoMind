"""Camera feed fetcher."""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass
from datetime import datetime
from typing import Optional

import httpx

from .config import get_camera_config
from .models import FrameMetadata
from .storage import storage


@dataclass
class FetchResult:
    success: bool
    image_data: bytes | None = None
    status_code: int = 0
    error: str | None = None
    fetch_duration_ms: float = 0.0


class CameraFetcher:
    def __init__(self, feed_config):
        self.feed_config = feed_config
        self.client: Optional[httpx.AsyncClient] = None
        self.frame_counter = 0

    async def start(self) -> None:
        self.client = httpx.AsyncClient(
            timeout=httpx.Timeout(self.feed_config.timeout_seconds),
            follow_redirects=True,
        )

    async def stop(self) -> None:
        if self.client:
            await self.client.aclose()
            self.client = None

    async def fetch_frame(self) -> FetchResult:
        if not self.client:
            await self.start()

        last_error = None
        for attempt in range(self.feed_config.max_retries):
            start = time.perf_counter()
            try:
                resp = await self.client.get(self.feed_config.feed_url)
                dur = (time.perf_counter() - start) * 1000
                if resp.status_code == 200:
                    return FetchResult(True, resp.content, resp.status_code, None, dur)
                last_error = f"HTTP {resp.status_code}"
                return FetchResult(False, None, resp.status_code, last_error, dur)
            except httpx.TimeoutException:
                dur = (time.perf_counter() - start) * 1000
                last_error = f"Timeout after {self.feed_config.timeout_seconds}s"
            except httpx.RequestError as e:
                dur = (time.perf_counter() - start) * 1000
                last_error = f"Request error: {e}"
            except Exception as e:
                dur = (time.perf_counter() - start) * 1000
                last_error = f"Unexpected: {e}"

            if attempt < self.feed_config.max_retries - 1:
                await asyncio.sleep(self.feed_config.retry_delay_seconds)

        return FetchResult(False, None, 0, last_error or "Max retries", dur)

    async def fetch_and_save(self) -> FrameMetadata:
        self.frame_counter += 1
        result = await self.fetch_frame()
        ts = datetime.utcnow()

        if not result.success or not result.image_data:
            return FrameMetadata(
                camera_id=self.feed_config.camera_id,
                timestamp=ts,
                frame_number=self.frame_counter,
                file_path="", file_size=0, width=0, height=0,
                content_hash="", fetch_duration_ms=result.fetch_duration_ms,
                status_code=result.status_code, error=result.error,
            )

        phash = storage._compute_phash(result.image_data)
        if storage.is_duplicate(self.feed_config.camera_id, result.image_data, phash):
            return FrameMetadata(
                camera_id=self.feed_config.camera_id, timestamp=ts,
                frame_number=self.frame_counter, file_path="", file_size=0,
                width=0, height=0, content_hash=phash,
                fetch_duration_ms=result.fetch_duration_ms,
                status_code=result.status_code, error="Duplicate frame",
            )

        path, size, w, h, chash = storage.save_frame(
            self.feed_config.camera_id, result.image_data, ts, self.frame_counter
        )
        return FrameMetadata(
            camera_id=self.feed_config.camera_id, timestamp=ts,
            frame_number=self.frame_counter, file_path=path,
            file_size=size, width=w, height=h, content_hash=chash,
            fetch_duration_ms=result.fetch_duration_ms,
            status_code=result.status_code,
        )