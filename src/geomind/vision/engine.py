"""Vision Engine - main inference pipeline."""

from __future__ import annotations

import asyncio
import logging
import uuid
from dataclasses import dataclass
from datetime import datetime
from typing import Optional

import cv2
import numpy as np

from .config import get_vision_config
from .database import vision_db
from .interfaces import BaseModelInterface
from .models import FrameDetections, FrameDetectionRecord
from .models_impl import create_model
from .temporal import TemporalValidator

logger = logging.getLogger(__name__)


@dataclass
class ProcessingResult:
    """Result of processing a single frame."""

    frame_detections: FrameDetections
    frame_record: FrameDetectionRecord
    events: list  # DetectionEvent list


class VisionEngine:
    """
    Main vision inference engine.

    Processes frames from camera feeds, runs detection, applies temporal
    validation, and persists results.
    """

    def __init__(self, model: BaseModelInterface | None = None):
        self.config = get_vision_config()
        self.model = model or create_model(self.config.model)
        self.validator = TemporalValidator()
        self._running = False
        self._frame_queue: asyncio.Queue | None = None
        self._worker_tasks: list[asyncio.Task] = []

    async def start(self) -> None:
        """Initialize the engine."""
        if self._running:
            return

        # Load and warm up model
        self.model.load()
        self.model.warmup()

        # Initialize database
        await vision_db.init()

        # Create frame queue
        queue_config = self.config.queue
        self._frame_queue = asyncio.Queue(maxsize=queue_config.max_size)

        # Start worker tasks
        for i in range(queue_config.worker_count):
            task = asyncio.create_task(self._worker(f"worker-{i}"))
            self._worker_tasks.append(task)

        self._running = True
        logger.info(
            "Vision engine started with %d workers", queue_config.worker_count
        )

    async def stop(self) -> None:
        """Stop the engine."""
        if not self._running:
            return

        self._running = False

        # Cancel workers
        for task in self._worker_tasks:
            task.cancel()
        await asyncio.gather(*self._worker_tasks, return_exceptions=True)
        self._worker_tasks.clear()

        # Close database
        await vision_db.close()

        # Unload model
        self.model.unload()

        logger.info("Vision engine stopped")

    async def submit_frame(
        self,
        camera_id: str,
        frame_id: int,
        image: np.ndarray,
        timestamp: datetime | None = None,
    ) -> bool:
        """
        Submit a frame for async processing.

        Returns:
            True if frame was queued, False if queue is full.
        """
        if not self._running or self._frame_queue is None:
            logger.warning("Vision engine not running, cannot submit frame")
            return False

        frame_data = {
            "camera_id": camera_id,
            "frame_id": frame_id,
            "image": image,
            "timestamp": timestamp or datetime.utcnow(),
        }

        try:
            self._frame_queue.put_nowait(frame_data)
            return True
        except asyncio.QueueFull:
            logger.warning("Frame queue full, dropping frame for camera %s", camera_id)
            return False

    async def process_frame_sync(
        self,
        camera_id: str,
        frame_id: int,
        image: np.ndarray,
        timestamp: datetime | None = None,
    ) -> ProcessingResult:
        """
        Process a frame synchronously (for testing or low-throughput use).

        Returns:
            ProcessingResult with frame detections and any events.
        """
        ts = timestamp or datetime.utcnow()

        # Run inference
        result = self.model.infer(image)

        # Create frame detections
        frame_detections = FrameDetections(
            camera_id=camera_id,
            frame_id=frame_id,
            timestamp=ts,
            detections=result.detections,
            inference_time_ms=result.inference_time_ms,
            model_version=result.model_version,
        )

        # Save frame detections (audit trail)
        frame_record = await vision_db.save_frame_detections(
            camera_id=camera_id,
            frame_id=frame_id,
            timestamp=ts,
            detections=result.detections,
            inference_time_ms=result.inference_time_ms,
            model_version=result.model_version,
        )

        # Apply temporal validation
        events = self.validator.process_frame_detections(frame_detections)

        # Save events to database
        for event in events:
            await self._save_event(event, frame_record)

        return ProcessingResult(
            frame_detections=frame_detections,
            frame_record=frame_record,
            events=events,
        )

    async def _worker(self, name: str) -> None:
        """Background worker to process frames from queue."""
        logger.debug("Vision worker %s started", name)

        while self._running:
            try:
                # Wait for frame with timeout
                frame_data = await asyncio.wait_for(
                    self._frame_queue.get(), timeout=1.0
                )

                try:
                    await self.process_frame_sync(
                        camera_id=frame_data["camera_id"],
                        frame_id=frame_data["frame_id"],
                        image=frame_data["image"],
                        timestamp=frame_data["timestamp"],
                    )
                except Exception as e:
                    logger.exception("Error processing frame in %s: %s", name, e)

            except asyncio.TimeoutError:
                continue
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.exception("Worker %s error: %s", name, e)

        logger.debug("Vision worker %s stopped", name)

    async def _save_event(self, event, frame_record: FrameDetectionRecord) -> None:
        """Save a detection event to database and link frame record."""
        if event.status.value == "confirmed":
            # Update the detection event with final details
            await vision_db.update_detection_event(
                event_id=event.event_id,
                last_frame_id=event.last_frame_id,
                last_timestamp=event.last_timestamp,
                max_confidence=event.max_confidence,
                avg_confidence=event.avg_confidence,
                frame_count=event.frame_count,
                status=event.status,
                bbox=event.bbox.model_dump() if event.bbox else None,
            )
        elif event.status.value == "rejected":
            # Save the rejected event for audit
            await vision_db.save_detection_event(
                event_id=event.event_id,
                camera_id=event.camera_id,
                first_frame_id=event.first_frame_id,
                last_frame_id=event.last_frame_id,
                first_timestamp=event.first_timestamp,
                last_timestamp=event.last_timestamp,
                class_name=event.class_name,
                max_confidence=event.max_confidence,
                avg_confidence=event.avg_confidence,
                frame_count=event.frame_count,
                status=event.status,
                bbox=event.bbox.model_dump() if event.bbox else None,
            )

        # Update the frame record with event_id (for both confirmed and rejected)
        await vision_db.update_frame_detection_event_id(
            frame_record_id=frame_record.id,
            event_id=event.event_id,
        )

    def get_stats(self) -> dict:
        """Get engine statistics."""
        return {
            "running": self._running,
            "model": self.model.model_version,
            "model_loaded": self.model.is_loaded,
            "queue_size": self._frame_queue.qsize() if self._frame_queue else 0,
            "pending_events": len(self.validator.get_pending_events()),
            "workers": len(self._worker_tasks),
        }


vision_engine = VisionEngine()