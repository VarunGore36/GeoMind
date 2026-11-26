"""Prometheus metrics collector for GeoMind."""

from __future__ import annotations

import time
from contextlib import contextmanager
from functools import wraps
from typing import Any, Callable, Optional

from prometheus_client import Counter, Gauge, Histogram, CollectorRegistry, generate_latest
from prometheus_client.core import CollectorRegistry as PrometheusCollectorRegistry

from .config import get_metrics_config

# Try to import the global registry, fall back to custom if needed
try:
    from prometheus_client import REGISTRY as DEFAULT_REGISTRY
    _default_registry = DEFAULT_REGISTRY
except ImportError:
    _default_registry = None


class MetricsCollector:
    """Centralized Prometheus metrics collector."""

    def __init__(self, registry: Optional[PrometheusCollectorRegistry] = None):
        self.config = get_metrics_config()
        self.registry = registry or _default_registry
        self._initialized = False
        self._metrics: dict[str, Any] = {}

    def initialize(self) -> None:
        """Initialize all metrics."""
        if self._initialized:
            return

        if not self.config.enabled:
            self._initialized = True
            return

        c = self.config.metrics
        ns = c.namespace
        sub = c.subsystem

        # --- Request/HTTP Metrics ---
        self._metrics["http_requests_total"] = Counter(
            "http_requests_total",
            "Total HTTP requests",
            ["method", "endpoint", "status"],
            namespace=ns,
            subsystem=sub,
            registry=self.registry,
        )

        self._metrics["http_request_duration_seconds"] = Histogram(
            "http_request_duration_seconds",
            "HTTP request latency in seconds",
            ["method", "endpoint"],
            buckets=c.latency_buckets,
            namespace=ns,
            subsystem=sub,
            registry=self.registry,
        )

        self._metrics["http_requests_in_progress"] = Gauge(
            "http_requests_in_progress",
            "HTTP requests currently in progress",
            ["method", "endpoint"],
            namespace=ns,
            subsystem=sub,
            registry=self.registry,
        )

        # --- Camera Pipeline Metrics ---
        self._metrics["camera_frames_fetched_total"] = Counter(
            "camera_frames_fetched_total",
            "Total frames fetched from camera feeds",
            ["camera_id", "status"],
            namespace=ns,
            subsystem=sub,
            registry=self.registry,
        )

        self._metrics["camera_fetch_duration_seconds"] = Histogram(
            "camera_fetch_duration_seconds",
            "Camera frame fetch latency in seconds",
            ["camera_id"],
            buckets=c.latency_buckets,
            namespace=ns,
            subsystem=sub,
            registry=self.registry,
        )

        self._metrics["camera_frames_stored_total"] = Counter(
            "camera_frames_stored_total",
            "Total frames stored to disk",
            ["camera_id"],
            namespace=ns,
            subsystem=sub,
            registry=self.registry,
        )

        self._metrics["camera_duplicates_skipped_total"] = Counter(
            "camera_duplicates_skipped_total",
            "Total duplicate frames skipped",
            ["camera_id"],
            namespace=ns,
            subsystem=sub,
            registry=self.registry,
        )

        # --- Vision Engine Metrics ---
        self._metrics["vision_inferences_total"] = Counter(
            "vision_inferences_total",
            "Total vision inference runs",
            ["model_type", "status"],
            namespace=ns,
            subsystem=sub,
            registry=self.registry,
        )

        self._metrics["vision_inference_duration_seconds"] = Histogram(
            "vision_inference_duration_seconds",
            "Vision inference latency in seconds",
            ["model_type"],
            buckets=c.latency_buckets,
            namespace=ns,
            subsystem=sub,
            registry=self.registry,
        )

        self._metrics["vision_detections_total"] = Counter(
            "vision_detections_total",
            "Total detections produced",
            ["model_type", "class_name", "status"],
            namespace=ns,
            subsystem=sub,
            registry=self.registry,
        )

        self._metrics["vision_detections_confirmed_total"] = Counter(
            "vision_detections_confirmed_total",
            "Total confirmed detections (after temporal validation)",
            ["camera_id", "class_name"],
            namespace=ns,
            subsystem=sub,
            registry=self.registry,
        )

        self._metrics["vision_detections_rejected_total"] = Counter(
            "vision_detections_rejected_total",
            "Total rejected detections (failed temporal validation)",
            ["camera_id", "class_name"],
            namespace=ns,
            subsystem=sub,
            registry=self.registry,
        )

        self._metrics["vision_temporal_pending"] = Gauge(
            "vision_temporal_pending",
            "Number of pending detection events awaiting confirmation",
            ["camera_id", "class_name"],
            namespace=ns,
            subsystem=sub,
            registry=self.registry,
        )

        # --- WebSocket Metrics ---
        self._metrics["ws_connections_active"] = Gauge(
            "ws_connections_active",
            "Active WebSocket connections",
            namespace=ns,
            subsystem=sub,
            registry=self.registry,
        )

        self._metrics["ws_messages_received_total"] = Counter(
            "ws_messages_received_total",
            "Total WebSocket messages received",
            ["message_type"],
            namespace=ns,
            subsystem=sub,
            registry=self.registry,
        )

        self._metrics["ws_messages_sent_total"] = Counter(
            "ws_messages_sent_total",
            "Total WebSocket messages sent",
            ["message_type", "channel"],
            namespace=ns,
            subsystem=sub,
            registry=self.registry,
        )

        self._metrics["ws_subscriptions_active"] = Gauge(
            "ws_subscriptions_active",
            "Active WebSocket subscriptions",
            ["channel"],
            namespace=ns,
            subsystem=sub,
            registry=self.registry,
        )

        # --- Database Metrics ---
        self._metrics["db_query_duration_seconds"] = Histogram(
            "db_query_duration_seconds",
            "Database query latency in seconds",
            ["operation", "table"],
            buckets=c.latency_buckets,
            namespace=ns,
            subsystem=sub,
            registry=self.registry,
        )

        self._metrics["db_pool_connections"] = Gauge(
            "db_pool_connections",
            "Database pool connections",
            ["state"],
            namespace=ns,
            subsystem=sub,
            registry=self.registry,
        )

        # --- Overpass/OSM Metrics ---
        self._metrics["overpass_requests_total"] = Counter(
            "overpass_requests_total",
            "Total Overpass API requests",
            ["status"],
            namespace=ns,
            subsystem=sub,
            registry=self.registry,
        )

        self._metrics["overpass_request_duration_seconds"] = Histogram(
            "overpass_request_duration_seconds",
            "Overpass API request latency",
            buckets=c.latency_buckets,
            namespace=ns,
            subsystem=sub,
            registry=self.registry,
        )

        # --- Elevation Metrics ---
        self._metrics["elevation_requests_total"] = Counter(
            "elevation_requests_total",
            "Total elevation API requests",
            ["provider", "status"],
            namespace=ns,
            subsystem=sub,
            registry=self.registry,
        )

        self._metrics["elevation_request_duration_seconds"] = Histogram(
            "elevation_request_duration_seconds",
            "Elevation API request latency",
            buckets=c.latency_buckets,
            namespace=ns,
            subsystem=sub,
            registry=self.registry,
        )

        # --- Auth Metrics ---
        self._metrics["auth_login_total"] = Counter(
            "auth_login_total",
            "Total login attempts",
            ["method", "status"],
            namespace=ns,
            subsystem=sub,
            registry=self.registry,
        )

        self._metrics["auth_token_refresh_total"] = Counter(
            "auth_token_refresh_total",
            "Total token refresh attempts",
            ["status"],
            namespace=ns,
            subsystem=sub,
            registry=self.registry,
        )

        self._metrics["api_keys_active"] = Gauge(
            "api_keys_active",
            "Active API keys",
            namespace=ns,
            subsystem=sub,
            registry=self.registry,
        )

        # Add default labels
        for metric in self._metrics.values():
            for label_name, label_value in self.config.metrics.default_labels.items():
                if hasattr(metric, "_labelnames") and label_name not in metric._labelnames:
                    pass  # Can't add labels after creation

        self._initialized = True

    def get_metric(self, name: str) -> Optional[Any]:
        """Get a metric by name."""
        return self._metrics.get(name)

    def increment_counter(self, name: str, labels: dict[str, str] = None, value: float = 1.0) -> None:
        """Increment a counter metric."""
        if not self.config.enabled:
            return
        metric = self._metrics.get(name)
        if metric and hasattr(metric, "inc"):
            if labels:
                metric.labels(**labels).inc(value)
            else:
                metric.inc(value)

    def set_gauge(self, name: str, value: float, labels: dict[str, str] = None) -> None:
        """Set a gauge metric."""
        if not self.config.enabled:
            return
        metric = self._metrics.get(name)
        if metric and hasattr(metric, "set"):
            if labels:
                metric.labels(**labels).set(value)
            else:
                metric.set(value)

    def observe_histogram(self, name: str, value: float, labels: dict[str, str] = None) -> None:
        """Observe a histogram metric."""
        if not self.config.enabled:
            return
        metric = self._metrics.get(name)
        if metric and hasattr(metric, "observe"):
            if labels:
                metric.labels(**labels).observe(value)
            else:
                metric.observe(value)

    @contextmanager
    def time_operation(self, histogram_name: str, labels: dict[str, str] = None):
        """Context manager to time an operation."""
        if not self.config.enabled:
            yield
            return
        start = time.perf_counter()
        try:
            yield
        finally:
            duration = time.perf_counter() - start
            self.observe_histogram(histogram_name, duration, labels)

    def track_http_request(self, method: str, endpoint: str, status: int, duration: float) -> None:
        """Track an HTTP request."""
        if not self.config.enabled:
            return
        labels = {"method": method, "endpoint": endpoint, "status": str(status)}
        self._metrics["http_requests_total"].labels(**labels).inc()
        self._metrics["http_request_duration_seconds"].labels(method=method, endpoint=endpoint).observe(duration)

    def track_camera_fetch(self, camera_id: str, status: str, duration: float) -> None:
        """Track a camera frame fetch."""
        if not self.config.enabled:
            return
        self._metrics["camera_frames_fetched_total"].labels(camera_id=camera_id, status=status).inc()
        self._metrics["camera_fetch_duration_seconds"].labels(camera_id=camera_id).observe(duration)

    def track_vision_inference(self, model_type: str, status: str, duration: float) -> None:
        """Track a vision inference."""
        if not self.config.enabled:
            return
        self._metrics["vision_inferences_total"].labels(model_type=model_type, status=status).inc()
        self._metrics["vision_inference_duration_seconds"].labels(model_type=model_type).observe(duration)

    def track_detection(self, model_type: str, class_name: str, status: str) -> None:
        """Track a detection."""
        if not self.config.enabled:
            return
        self._metrics["vision_detections_total"].labels(model_type=model_type, class_name=class_name, status=status).inc()

    def track_detection_confirmed(self, camera_id: str, class_name: str) -> None:
        """Track a confirmed detection."""
        if not self.config.enabled:
            return
        self._metrics["vision_detections_confirmed_total"].labels(camera_id=camera_id, class_name=class_name).inc()

    def track_detection_rejected(self, camera_id: str, class_name: str) -> None:
        """Track a rejected detection."""
        if not self.config.enabled:
            return
        self._metrics["vision_detections_rejected_total"].labels(camera_id=camera_id, class_name=class_name).inc()

    def set_pending_detections(self, camera_id: str, class_name: str, count: int) -> None:
        """Set pending detection count."""
        if not self.config.enabled:
            return
        self._metrics["vision_temporal_pending"].labels(camera_id=camera_id, class_name=class_name).set(count)

    def track_ws_message(self, direction: str, message_type: str, channel: str = None) -> None:
        """Track WebSocket message."""
        if not self.config.enabled:
            return
        if direction == "received":
            self._metrics["ws_messages_received_total"].labels(message_type=message_type).inc()
        elif direction == "sent":
            self._metrics["ws_messages_sent_total"].labels(message_type=message_type, channel=channel or "").inc()

    def set_ws_connections(self, count: int) -> None:
        """Set active WebSocket connections count."""
        if not self.config.enabled:
            return
        self._metrics["ws_connections_active"].set(count)

    def set_ws_subscriptions(self, channel: str, count: int) -> None:
        """Set WebSocket subscriptions count for a channel."""
        if not self.config.enabled:
            return
        self._metrics["ws_subscriptions_active"].labels(channel=channel).set(count)

    def track_db_query(self, operation: str, table: str, duration: float) -> None:
        """Track database query."""
        if not self.config.enabled:
            return
        self._metrics["db_query_duration_seconds"].labels(operation=operation, table=table).observe(duration)

    def set_db_pool_connections(self, state: str, count: int) -> None:
        """Set database pool connection count."""
        if not self.config.enabled:
            return
        self._metrics["db_pool_connections"].labels(state=state).set(count)

    def track_overpass_request(self, status: str, duration: float) -> None:
        """Track Overpass API request."""
        if not self.config.enabled:
            return
        self._metrics["overpass_requests_total"].labels(status=status).inc()
        self._metrics["overpass_request_duration_seconds"].observe(duration)

    def track_elevation_request(self, provider: str, status: str, duration: float) -> None:
        """Track elevation request."""
        if not self.config.enabled:
            return
        self._metrics["elevation_requests_total"].labels(provider=provider, status=status).inc()
        self._metrics["elevation_request_duration_seconds"].observe(duration)

    def track_auth_login(self, method: str, status: str) -> None:
        """Track login attempt."""
        if not self.config.enabled:
            return
        self._metrics["auth_login_total"].labels(method=method, status=status).inc()

    def track_token_refresh(self, status: str) -> None:
        """Track token refresh."""
        if not self.config.enabled:
            return
        self._metrics["auth_token_refresh_total"].labels(status=status).inc()

    def set_active_api_keys(self, count: int) -> None:
        """Set active API keys count."""
        if not self.config.enabled:
            return
        self._metrics["api_keys_active"].set(count)

    def get_metrics_output(self) -> bytes:
        """Get Prometheus metrics output."""
        if not self.config.enabled:
            return b""
        return generate_latest(self.registry)


def track_time(histogram_name: str, labels: dict[str, str] = None):
    """Decorator to time a function."""
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            collector = metrics_collector
            if not collector.config.enabled:
                return await func(*args, **kwargs)
            start = time.perf_counter()
            try:
                return await func(*args, **kwargs)
            finally:
                duration = time.perf_counter() - start
                collector.observe_histogram(histogram_name, duration, labels)

        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            collector = metrics_collector
            if not collector.config.enabled:
                return func(*args, **kwargs)
            start = time.perf_counter()
            try:
                return func(*args, **kwargs)
            finally:
                duration = time.perf_counter() - start
                collector.observe_histogram(histogram_name, duration, labels)

        import asyncio
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        return sync_wrapper
    return decorator


# Global metrics collector instance
metrics_collector = MetricsCollector()