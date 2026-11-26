"""Prometheus metrics configuration."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class MetricsConfig(BaseModel):
    """Prometheus metrics configuration."""

    enabled: bool = True
    port: int = Field(default=9090, ge=1, le=65535)
    path: str = "/metrics"
    namespace: str = "geomind"
    subsystem: str = ""

    # Histogram buckets for latency
    latency_buckets: list[float] = Field(
        default_factory=lambda: [
            0.005, 0.01, 0.025, 0.05, 0.075, 0.1, 0.25, 0.5, 0.75, 1.0, 2.5, 5.0, 7.5, 10.0
        ]
    )

    # Default labels
    default_labels: dict[str, str] = Field(default_factory=dict)


class MetricsConfigMain(BaseModel):
    """Main metrics configuration."""

    metrics: MetricsConfig = MetricsConfig()
    enabled: bool = True


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_prefix="GEOMIND_METRICS_",
        extra="ignore",
    )
    config_path: str = "metrics.yaml"


@lru_cache
def get_metrics_config() -> "MetricsConfigMain":
    """Load and cache the metrics configuration."""
    import yaml
    from pathlib import Path

    settings = Settings()
    config_path = Path(settings.config_path)
    if config_path.exists():
        with open(config_path) as f:
            data = yaml.safe_load(f)
        return MetricsConfigMain(**data)
    return MetricsConfigMain()