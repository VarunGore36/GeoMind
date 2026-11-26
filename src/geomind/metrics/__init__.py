"""Prometheus metrics module for GeoMind."""

from .config import MetricsConfigMain, get_metrics_config
from .metrics import MetricsCollector, metrics_collector

# Initialize metrics collector at import time
metrics_collector.initialize()

__version__ = "0.1.0"

__all__ = [
    "MetricsConfigMain",
    "get_metrics_config",
    "MetricsCollector",
    "metrics_collector",
]