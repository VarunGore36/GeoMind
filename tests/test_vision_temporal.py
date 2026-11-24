"""Tests for temporal validation."""

from __future__ import annotations

from datetime import datetime, timedelta

import pytest

from geomind.vision.config import VisionEngineConfig, TemporalValidationConfig
from geomind.vision.models import BoundingBox, Detection, FrameDetections
from geomind.vision.temporal import TemporalValidator


@pytest.fixture
def validator():
    config = VisionEngineConfig(
        temporal=TemporalValidationConfig(
            min_consecutive_frames=3,
            frame_confidence_threshold=0.3,
            max_frame_gap_seconds=60.0,
            min_confidence_sum=1.0,
        )
    )
    return TemporalValidator()


@pytest.fixture
def sample_frame():
    def _make_frame(frame_id: int, detections: list, timestamp: datetime | None = None):
        return FrameDetections(
            camera_id="test_cam",
            frame_id=frame_id,
            timestamp=timestamp or datetime.utcnow(),
            detections=detections,
            inference_time_ms=100.0,
            model_version="DummyModel",
        )
    return _make_frame


@pytest.fixture
def smoke_detection():
    bbox = BoundingBox(x_min=0.1, y_min=0.1, x_max=0.5, y_max=0.5)
    return Detection(class_id=0, class_name="smoke", confidence=0.8, bbox=bbox)


def test_single_detection_not_confirmed(validator, sample_frame, smoke_detection):
    """A single detection should not be confirmed."""
    frame = sample_frame(1, [smoke_detection])
    events = validator.process_frame_detections(frame)
    assert len(events) == 0
    assert len(validator.get_pending_events()) == 1


def test_three_consecutive_detections_confirmed(validator, sample_frame, smoke_detection):
    """Three consecutive detections should be confirmed."""
    base_time = datetime.utcnow()

    # Frame 1
    frame1 = sample_frame(1, [smoke_detection], base_time)
    events = validator.process_frame_detections(frame1)
    assert len(events) == 0

    # Frame 2
    frame2 = sample_frame(2, [smoke_detection], base_time + timedelta(seconds=30))
    events = validator.process_frame_detections(frame2)
    assert len(events) == 0

    # Frame 3 - should confirm
    frame3 = sample_frame(3, [smoke_detection], base_time + timedelta(seconds=60))
    events = validator.process_frame_detections(frame3)
    assert len(events) == 1
    assert events[0].status.value == "confirmed"
    assert events[0].frame_count == 3


def test_gap_between_frames_rejects(validator, sample_frame, smoke_detection):
    """Large time gap between frames should reject the event."""
    base_time = datetime.utcnow()

    # Frame 1
    frame1 = sample_frame(1, [smoke_detection], base_time)
    validator.process_frame_detections(frame1)

    # Frame 2 - after 2 minutes (exceeds max_frame_gap_seconds=60)
    frame2 = sample_frame(2, [smoke_detection], base_time + timedelta(minutes=2))
    events = validator.process_frame_detections(frame2)
    assert len(events) == 1
    assert events[0].status.value == "rejected"


def test_low_confidence_ignored(validator, sample_frame):
    """Detections below frame_confidence_threshold should be ignored."""
    bbox = BoundingBox(x_min=0.1, y_min=0.1, x_max=0.5, y_max=0.5)
    low_conf = Detection(class_id=0, class_name="smoke", confidence=0.1, bbox=bbox)

    frame = sample_frame(1, [low_conf])
    events = validator.process_frame_detections(frame)
    assert len(events) == 0
    assert len(validator.get_pending_events()) == 0


def test_different_class_separate_tracking(validator, sample_frame, smoke_detection):
    """Different classes should be tracked independently."""
    bbox = BoundingBox(x_min=0.1, y_min=0.1, x_max=0.5, y_max=0.5)
    fire_detection = Detection(class_id=1, class_name="fire", confidence=0.8, bbox=bbox)

    base_time = datetime.utcnow()

    # Smoke detections - 3 consecutive frames to confirm
    frame1 = sample_frame(1, [smoke_detection], base_time)
    events = validator.process_frame_detections(frame1)
    assert len(events) == 0

    frame2 = sample_frame(2, [smoke_detection], base_time + timedelta(seconds=30))
    events = validator.process_frame_detections(frame2)
    assert len(events) == 0

    frame3 = sample_frame(3, [smoke_detection], base_time + timedelta(seconds=60))
    events = validator.process_frame_detections(frame3)
    assert len(events) == 1
    assert events[0].class_name == "smoke"
    assert events[0].status.value == "confirmed"

    # Fire detections - separate tracking, starts fresh
    frame4 = sample_frame(4, [fire_detection], base_time + timedelta(seconds=90))
    events = validator.process_frame_detections(frame4)
    assert len(events) == 0

    frame5 = sample_frame(5, [fire_detection], base_time + timedelta(seconds=120))
    events = validator.process_frame_detections(frame5)
    assert len(events) == 0

    frame6 = sample_frame(6, [fire_detection], base_time + timedelta(seconds=150))
    events = validator.process_frame_detections(frame6)
    assert len(events) == 1
    assert events[0].class_name == "fire"
    assert events[0].status.value == "confirmed"


def test_stale_event_rejected(validator, sample_frame, smoke_detection):
    """Events not seen for max_frame_gap_seconds should be rejected."""
    base_time = datetime.utcnow()

    frame1 = sample_frame(1, [smoke_detection], base_time)
    validator.process_frame_detections(frame1)

    # No detection for > 60 seconds
    frame2 = sample_frame(2, [], base_time + timedelta(minutes=2))
    events = validator.process_frame_detections(frame2)

    assert len(events) == 1
    assert events[0].status.value == "rejected"