"""Tests for vision engine models."""

from __future__ import annotations

from datetime import datetime

import pytest

from geomind.vision.models import (
    BoundingBox,
    Detection,
    DetectionClass,
    DetectionEvent,
    DetectionStatus,
    FrameDetections,
)


class TestBoundingBox:
    def test_bbox_creation(self):
        bbox = BoundingBox(x_min=0.1, y_min=0.2, x_max=0.5, y_max=0.6)
        assert bbox.x_min == 0.1
        assert bbox.y_min == 0.2
        assert bbox.x_max == 0.5
        assert bbox.y_max == 0.6

    def test_bbox_validation(self):
        with pytest.raises(ValueError):
            BoundingBox(x_min=-0.1, y_min=0.0, x_max=0.5, y_max=0.5)
        with pytest.raises(ValueError):
            BoundingBox(x_min=0.0, y_min=0.0, x_max=1.1, y_max=0.5)

    def test_bbox_to_absolute(self):
        bbox = BoundingBox(x_min=0.25, y_min=0.25, x_max=0.75, y_max=0.75)
        abs_coords = bbox.to_absolute(100, 100)
        assert abs_coords == (25, 25, 75, 75)

    def test_bbox_from_absolute(self):
        bbox = BoundingBox.from_absolute(10, 20, 50, 60, 100, 100)
        assert bbox.x_min == 0.1
        assert bbox.y_min == 0.2
        assert bbox.x_max == 0.5
        assert bbox.y_max == 0.6


class TestDetection:
    def test_detection_creation(self):
        bbox = BoundingBox(x_min=0.1, y_min=0.1, x_max=0.5, y_max=0.5)
        det = Detection(class_id=0, class_name="smoke", confidence=0.85, bbox=bbox)
        assert det.class_id == 0
        assert det.class_name == "smoke"
        assert det.confidence == 0.85

    def test_detection_confidence_validation(self):
        bbox = BoundingBox(x_min=0.0, y_min=0.0, x_max=1.0, y_max=1.0)
        with pytest.raises(ValueError):
            Detection(class_id=0, class_name="smoke", confidence=1.5, bbox=bbox)
        with pytest.raises(ValueError):
            Detection(class_id=0, class_name="smoke", confidence=-0.1, bbox=bbox)


class TestFrameDetections:
    def test_frame_detections_creation(self):
        bbox = BoundingBox(x_min=0.1, y_min=0.1, x_max=0.5, y_max=0.5)
        det = Detection(class_id=0, class_name="smoke", confidence=0.85, bbox=bbox)
        frame_det = FrameDetections(
            camera_id="test_cam",
            frame_id=1,
            timestamp=datetime.utcnow(),
            detections=[det],
            inference_time_ms=150.0,
            model_version="DummyModel",
        )
        assert frame_det.camera_id == "test_cam"
        assert frame_det.frame_id == 1
        assert len(frame_det.detections) == 1


class TestDetectionEvent:
    def test_detection_event_creation(self):
        bbox = BoundingBox(x_min=0.1, y_min=0.1, x_max=0.5, y_max=0.5)
        event = DetectionEvent(
            event_id="test-event-1",
            camera_id="test_cam",
            first_frame_id=1,
            last_frame_id=3,
            first_timestamp=datetime.utcnow(),
            last_timestamp=datetime.utcnow(),
            class_name="smoke",
            max_confidence=0.9,
            avg_confidence=0.8,
            frame_count=3,
            status=DetectionStatus.CONFIRMED,
            bbox=bbox,
        )
        assert event.event_id == "test-event-1"
        assert event.status == DetectionStatus.CONFIRMED


class TestDetectionStatus:
    def test_status_values(self):
        assert DetectionStatus.PENDING.value == "pending"
        assert DetectionStatus.CONFIRMED.value == "confirmed"
        assert DetectionStatus.REJECTED.value == "rejected"