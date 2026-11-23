"""Camera pipeline API routes."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse

from .config import get_camera_config
from .database import db
from .models import CameraStatus, FrameMetadata
from .scheduler import scheduler
from .storage import storage

router = APIRouter(prefix="/cameras", tags=["cameras"])


@router.get("/health")
async def health_check():
    config = get_camera_config()
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "feeds": len(config.feeds),
        "cameras": scheduler.get_all_status(),
    }


@router.get("", response_model=dict[str, CameraStatus])
async def list_cameras() -> dict[str, CameraStatus]:
    return scheduler.get_all_status()


@router.get("/{camera_id}", response_model=CameraStatus)
async def get_camera(camera_id: str) -> CameraStatus:
    st = scheduler.get_status(camera_id)
    if not st:
        raise HTTPException(404, f"Camera {camera_id} not found")
    return st


@router.get("/{camera_id}/latest", response_model=FrameMetadata)
async def get_latest_meta(camera_id: str) -> FrameMetadata:
    st = scheduler.get_status(camera_id)
    if not st:
        raise HTTPException(404, f"Camera {camera_id} not found")
    if not st.latest_frame:
        meta = await db.get_latest_frame(camera_id)
        if not meta:
            raise HTTPException(404, "No frames available")
        return meta
    return st.latest_frame


@router.get("/{camera_id}/latest/image")
async def get_latest_image(camera_id: str):
    st = scheduler.get_status(camera_id)
    if not st:
        raise HTTPException(404, f"Camera {camera_id} not found")

    path = storage.get_latest_frame_path(camera_id)
    if path and path.exists():
        return FileResponse(path, media_type="image/jpeg")

    meta = await db.get_latest_frame(camera_id)
    if meta and meta.file_path:
        path = Path(meta.file_path)
        if path.exists():
            return FileResponse(path, media_type="image/jpeg")

    raise HTTPException(404, "No frame image available")


@router.post("/{camera_id}/fetch", response_model=FrameMetadata)
async def trigger_fetch(camera_id: str) -> FrameMetadata:
    st = scheduler.get_status(camera_id)
    if not st:
        raise HTTPException(404, f"Camera {camera_id} not found")
    meta = await scheduler.trigger_fetch(camera_id)
    if not meta:
        raise HTTPException(500, "Fetch failed")
    return meta