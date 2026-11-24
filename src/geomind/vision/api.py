"""Vision Engine API routes."""

from __future__ import annotations

from datetime import datetime
from typing import Optional

import cv2
from fastapi import APIRouter, HTTPException, Query

from .database import vision_db
from .engine import vision_engine
from .models import (
    DetectionEventResponse,
    DetectionStatus,
    FrameDetectionResponse,
)
from .storage import vision_storage as storage

router = APIRouter(prefix="/detections", tags=["detections"])


@router.get("/health")
async def health_check() -> dict:
    """Health check for vision engine."""
    return {
        "status": "healthy" if vision_engine._running else "stopped",
        "stats": vision_engine.get_stats(),
    }


@router.get("", response_model=list[DetectionEventResponse])
async def list_detection_events(
    camera_id: Optional[str] = Query(None, description="Filter by camera ID"),
    status: Optional[DetectionStatus] = Query(None, description="Filter by status"),
    limit: int = Query(100, ge=1, le=1000),
    offset: int = Query(0, ge=0),
) -> list[DetectionEventResponse]:
    """List detection events with optional filters."""
    events = await vision_db.list_detection_events(
        camera_id=camera_id, status=status, limit=limit, offset=offset
    )
    return events


@router.get("/{event_id}", response_model=DetectionEventResponse)
async def get_detection_event(event_id: str) -> DetectionEventResponse:
    """Get a specific detection event by ID."""
    event = await vision_db.get_detection_event(event_id)
    if not event:
        raise HTTPException(status_code=404, detail=f"Detection event {event_id} not found")
    return event


@router.get("/camera/{camera_id}/latest", response_model=Optional[DetectionEventResponse])
async def get_latest_event_for_camera(camera_id: str) -> Optional[DetectionEventResponse]:
    """Get the latest detection event for a camera."""
    event = await vision_db.get_latest_event_for_camera(camera_id)
    return event


@router.get("/camera/{camera_id}/frames", response_model=list[FrameDetectionResponse])
async def get_recent_frame_detections(
    camera_id: str,
    limit: int = Query(100, ge=1, le=1000),
) -> list[FrameDetectionResponse]:
    """Get recent frame detections for a camera."""
    records = await vision_db.get_recent_frame_detections(camera_id, limit)
    return [
        FrameDetectionResponse(
            camera_id=r.camera_id,
            frame_id=r.frame_id,
            timestamp=r.timestamp,
            detections=r.detections_json,
            inference_time_ms=r.inference_time_ms,
            model_version=r.model_version,
        )
        for r in records
    ]


@router.get("/camera/{camera_id}/pending")
async def get_pending_events(camera_id: Optional[str] = None) -> dict:
    """Get currently pending (unconfirmed) detection events."""
    pending = vision_engine.validator.get_pending_events()
    if camera_id:
        pending = [p for p in pending if p.camera_id == camera_id]

    return {
        "pending_count": len(pending),
        "events": [
            {
                "event_id": p.event_id,
                "camera_id": p.camera_id,
                "class_name": p.class_name,
                "first_frame_id": p.first_frame_id,
                "last_frame_id": p.last_frame_id,
                "frame_count": p.frame_count,
                "avg_confidence": sum(p.confidences) / len(p.confidences) if p.confidences else 0,
            }
            for p in pending
        ],
    }


@router.post("/process-frame")
async def process_frame_sync(
    camera_id: str,
    frame_id: int,
    timestamp: Optional[datetime] = None,
) -> dict:
    """
    Synchronously process a frame from the camera pipeline.

    This endpoint fetches the frame from storage, runs inference,
    and applies temporal validation.
    """
    # Get frame from storage
    frame_path = storage.get_latest_frame_path(camera_id)
    if not frame_path:
        raise HTTPException(status_code=404, detail=f"No frames found for camera {camera_id}")

    image = cv2.imread(str(frame_path))
    if image is None:
        raise HTTPException(status_code=500, detail="Failed to read frame image")

    result = await vision_engine.process_frame_sync(
        camera_id=camera_id,
        frame_id=frame_id,
        image=image,
        timestamp=timestamp,
    )

    return {
        "frame_detections": result.frame_detections.model_dump(),
        "events": [e.model_dump() for e in result.events],
        "stats": vision_engine.get_stats(),
    }