"""Tests for metrics collector."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from prometheus_client import CollectorRegistry

from geomind.metrics.config import MetricsConfigMain, MetricsConfig
from geomind.metrics.metrics import MetricsCollector, track_time


class TestMetricsCollector:
    @pytest.fixture
    def metrics_config(self):
        return MetricsConfigMain(
            enabled=True,
            metrics=MetricsConfig(
                enabled=True,
                port=9090,
                path="/metrics",
                namespace="test",
            ),
        )

    @pytest.fixture
    def collector(self, metrics_config):
        # Use a fresh registry for each test to avoid duplicate metric registration
        registry = CollectorRegistry()
        collector = MetricsCollector(registry=registry)
        collector.config = metrics_config
        collector.initialize()
        yield collector

    def test_initialization(self, collector):
        assert collector._initialized is True
        assert "http_requests_total" in collector._metrics
        assert "vision_inferences_total" in collector._metrics
        assert "ws_connections_active" in collector._metrics

    def test_increment_counter(self, collector):
        collector.increment_counter("vision_inferences_total", {"model_type": "yolo", "status": "success"})
        # Can't easily test counter value without exposing internal state

    def test_set_gauge(self, collector):
        collector.set_gauge("ws_connections_active", 5)
        collector.set_gauge("vision_temporal_pending", 3, {"camera_id": "cam_001", "class_name": "smoke"})

    def test_observe_histogram(self, collector):
        collector.observe_histogram("vision_inference_duration_seconds", 0.15, {"model_type": "yolo"})
        collector.observe_histogram("http_request_duration_seconds", 0.05, {"method": "GET", "endpoint": "/api/v1/test"})

    def test_track_helpers(self, collector):
        collector.track_http_request("GET", "/api/v1/test", 200, 0.05)
        collector.track_camera_fetch("cam_001", "success", 0.15)
        collector.track_vision_inference("yolo", "success", 0.2)
        collector.track_detection("yolo", "smoke", "confirmed")
        collector.track_detection_confirmed("cam_001", "smoke")
        collector.track_detection_rejected("cam_001", "fire")
        collector.set_pending_detections("cam_001", "smoke", 2)
        collector.track_ws_message("received", "subscribe")
        collector.track_ws_message("sent", "detection", "detections")
        collector.set_ws_connections(10)
        collector.set_ws_subscriptions("detections", 5)
        collector.track_db_query("select", "camera_frames", 0.01)
        collector.set_db_pool_connections("idle", 5)
        collector.set_db_pool_connections("active", 3)
        collector.track_overpass_request("success", 1.5)
        collector.track_elevation_request("opentopodata", "success", 0.3)
        collector.track_auth_login("password", "success")
        collector.track_token_refresh("success")
        collector.set_active_api_keys(25)

    def test_get_metrics_output(self, collector):
        output = collector.get_metrics_output()
        assert isinstance(output, bytes)
        assert len(output) > 0

    def test_disabled_collector(self, metrics_config):
        metrics_config.enabled = False
        collector = MetricsCollector()
        collector.config = metrics_config
        collector.initialize()
        # Should not raise errors
        collector.increment_counter("test", {})
        collector.set_gauge("test", 1)
        collector.observe_histogram("test", 1.0)
        output = collector.get_metrics_output()
        assert output == b""

    @pytest.mark.asyncio
    async def test_track_time_decorator_async(self, collector):
        @track_time("vision_inference_duration_seconds", {"model_type": "test"})
        async def async_func():
            return "result"

        result = await async_func()
        assert result == "result"

    def test_track_time_decorator_sync(self, collector):
        @track_time("vision_inference_duration_seconds", {"model_type": "test"})
        def sync_func():
            return "result"

        result = sync_func()
        assert result == "result"