"""Camera pipeline scheduler."""

from __future__ import annotations

import logging
from typing import Optional

from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.interval import IntervalTrigger

from .config import get_camera_config
from .database import db
from .fetcher import CameraFetcher
from .models import CameraStatus, FrameMetadata

logger = logging.getLogger(__name__)


class CameraScheduler:
    def __init__(self):
        self.scheduler = AsyncIOScheduler()
        self.fetchers: dict[str, CameraFetcher] = {}
        self.status: dict[str, CameraStatus] = {}
        self._running = False

    def _init_status(self, camera_id: str) -> None:
        if camera_id not in self.status:
            self.status[camera_id] = CameraStatus(camera_id=camera_id, is_active=False)

    async def start(self) -> None:
        if self._running:
            return

        config = get_camera_config()
        await db.init()

        for feed in config.feeds:
            self._init_status(feed.camera_id)
            fetcher = CameraFetcher(feed)
            await fetcher.start()
            self.fetchers[feed.camera_id] = fetcher

            self.scheduler.add_job(
                self._fetch_and_store,
                trigger=IntervalTrigger(seconds=feed.fetch_interval_seconds),
                args=[feed.camera_id],
                id=f"fetch_{feed.camera_id}",
                replace_existing=True,
                max_instances=1,
                coalesce=True,
            )

        self.scheduler.start()
        self._running = True
        logger.info("Camera scheduler started for %d feeds", len(config.feeds))

    async def stop(self) -> None:
        if not self._running:
            return
        self.scheduler.shutdown(wait=True)
        for fetcher in self.fetchers.values():
            await fetcher.stop()
        await db.close()
        self._running = False
        logger.info("Camera scheduler stopped")

    async def _fetch_and_store(self, camera_id: str) -> Optional[FrameMetadata]:
        self._init_status(camera_id)
        fetcher = self.fetchers.get(camera_id)
        st = self.status[camera_id]
        if not fetcher:
            return None

        st.is_active = True
        try:
            meta = await fetcher.fetch_and_save()
            if meta:
                if meta.file_path and meta.error != "Duplicate frame":
                    await db.save_frame(meta)
                    st.total_frames += 1
                    st.last_success = meta.timestamp
                    st.latest_frame = meta
                st.last_fetch = meta.timestamp
                st.consecutive_failures = 0
            else:
                st.consecutive_failures += 1
                st.total_errors += 1
        except Exception as e:
            logger.exception("Fetch error for %s", camera_id)
            st.consecutive_failures += 1
            st.total_errors += 1
        return self.status[camera_id].latest_frame

    def get_status(self, camera_id: str) -> Optional[CameraStatus]:
        return self.status.get(camera_id)

    def get_all_status(self) -> dict[str, CameraStatus]:
        return self.status.copy()

    async def trigger_fetch(self, camera_id: str) -> Optional[FrameMetadata]:
        return await self._fetch_and_store(camera_id)


scheduler = CameraScheduler()