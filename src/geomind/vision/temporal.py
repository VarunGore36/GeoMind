"""Temporal validation for detection events."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Optional

from .config import get_vision_config
from .models import (
    BoundingBox,
    Detection,
    DetectionEvent,
    DetectionStatus,
    FrameDetections,
)


@dataclass
class PendingEvent:
    """A detection event being tracked for temporal validation."""

    event_id: str
    camera_id: str
    class_name: str
    first_frame_id: int
    last_frame_id: int
    first_timestamp: datetime
    last_timestamp: datetime
    confidences: list[float] = field(default_factory=list)
    bboxes: list[BoundingBox] = field(default_factory=list)
    frame_count: int = 0
    status: DetectionStatus = DetectionStatus.PENDING


class TemporalValidator:
    """
    Validates detection events across consecutive frames.

    A detection is only confirmed if it appears in N consecutive frames
    with sufficient confidence, preventing false positives from single
    low-confidence detections.
    """

    def __init__(self):
        config = get_vision_config()
        self.temporal_config = config.temporal
        self._pending: dict[str, PendingEvent] = {}  # key: "camera_id:class_name"

    def _make_key(self, camera_id: str, class_name: str) -> str:
        return f"{camera_id}:{class_name}"

    def process_frame_detections(
        self, frame_detections: FrameDetections
    ) -> list[DetectionEvent]:
        """
        Process detections from a single frame and return any confirmed events.

        Returns:
            List of DetectionEvent that were confirmed or rejected in this frame.
        """
        results = []
        current_time = frame_detections.timestamp

        # Group detections by class for this frame
        detections_by_class: dict[str, list[Detection]] = defaultdict(list)
        for det in frame_detections.detections:
            if det.confidence >= self.temporal_config.frame_confidence_threshold:
                detections_by_class[det.class_name].append(det)

        # Update or create pending events for classes detected in this frame
        detected_keys = set()
        for class_name, detections in detections_by_class.items():
            key = self._make_key(frame_detections.camera_id, class_name)
            detected_keys.add(key)

            if key in self._pending:
                pending = self._pending[key]
                # Check if frame is within acceptable gap (time and frame ID)
                time_gap = (current_time - pending.last_timestamp).total_seconds()
                frame_gap = frame_detections.frame_id - pending.last_frame_id

                # Allow frame gaps up to max_frame_gap_frames (default 5)
                # to handle dropped frames from camera pipeline
                max_frame_gap = getattr(self.temporal_config, 'max_frame_gap_frames', 5)

                if time_gap <= self.temporal_config.max_frame_gap_seconds and frame_gap <= max_frame_gap:
                    # Consecutive frame - update pending event
                    pending.last_frame_id = frame_detections.frame_id
                    pending.last_timestamp = current_time
                    pending.confidences.extend(d.confidence for d in detections)
                    pending.bboxes.extend(d.bbox for d in detections)
                    pending.frame_count += 1

                    # Check if confirmed
                    if (
                        pending.frame_count >= self.temporal_config.min_consecutive_frames
                        and sum(pending.confidences) >= self.temporal_config.min_confidence_sum
                    ):
                        pending.status = DetectionStatus.CONFIRMED
                        results.append(self._create_event(pending))
                        del self._pending[key]
                else:
                    # Gap too large - reject old event, start new
                    if pending.frame_count > 0:
                        pending.status = DetectionStatus.REJECTED
                        results.append(self._create_event(pending))
                    # Start fresh
                    self._pending[key] = self._create_pending(
                        frame_detections, detections, key
                    )
            else:
                # New detection class for this camera
                self._pending[key] = self._create_pending(
                    frame_detections, detections, key
                )

        # Check for stale pending events (not detected in this frame)
        stale_keys = [k for k in self._pending if k not in detected_keys]
        for key in stale_keys:
            pending = self._pending[key]
            # Check if enough time has passed to consider it stale
            time_gap = (current_time - pending.last_timestamp).total_seconds()
            if time_gap > self.temporal_config.max_frame_gap_seconds:
                if pending.frame_count > 0:
                    pending.status = DetectionStatus.REJECTED
                    results.append(self._create_event(pending))
                del self._pending[key]

        return results

    def _create_pending(
        self, frame_detections: FrameDetections, detections: list[Detection], key: str
    ) -> PendingEvent:
        import uuid
        return PendingEvent(
            event_id=str(uuid.uuid4()),
            camera_id=frame_detections.camera_id,
            class_name=detections[0].class_name,
            first_frame_id=frame_detections.frame_id,
            last_frame_id=frame_detections.frame_id,
            first_timestamp=frame_detections.timestamp,
            last_timestamp=frame_detections.timestamp,
            confidences=[d.confidence for d in detections],
            bboxes=[d.bbox for d in detections],
            frame_count=1,
        )

    def _create_event(self, pending: PendingEvent) -> DetectionEvent:
        """Create a DetectionEvent from a pending event."""
        # Compute union bbox (encompassing all detections)
        bbox = None
        if pending.bboxes:
            x_min = min(b.x_min for b in pending.bboxes)
            y_min = min(b.y_min for b in pending.bboxes)
            x_max = max(b.x_max for b in pending.bboxes)
            y_max = max(b.y_max for b in pending.bboxes)
            bbox = BoundingBox(
                x_min=x_min, y_min=y_min, x_max=x_max, y_max=y_max
            )

        return DetectionEvent(
            event_id=pending.event_id,
            camera_id=pending.camera_id,
            first_frame_id=pending.first_frame_id,
            last_frame_id=pending.last_frame_id,
            first_timestamp=pending.first_timestamp,
            last_timestamp=pending.last_timestamp,
            class_name=pending.class_name,
            max_confidence=max(pending.confidences) if pending.confidences else 0.0,
            avg_confidence=sum(pending.confidences) / len(pending.confidences)
            if pending.confidences
            else 0.0,
            frame_count=pending.frame_count,
            status=pending.status,
            bbox=bbox,
        )

    def get_pending_events(self) -> list[PendingEvent]:
        """Get all currently pending events."""
        return list(self._pending.values())

    def clear_pending(self, camera_id: str | None = None) -> None:
        """Clear pending events, optionally for a specific camera."""
        if camera_id is None:
            self._pending.clear()
        else:
            keys_to_delete = [k for k in self._pending if k.startswith(f"{camera_id}:")]
            for key in keys_to_delete:
                del self._pending[key]