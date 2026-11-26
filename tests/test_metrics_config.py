"""Tests for metrics configuration."""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest
import yaml

from geomind.metrics.config import MetricsConfigMain, MetricsConfig, get_metrics_config


class TestMetricsConfig:
    def test_metrics_config_defaults(self):
        config = MetricsConfig()
        assert config.enabled is True
        assert config.port == 9090
        assert config.path == "/metrics"
        assert config.namespace == "geomind"

    def test_metrics_config_main_defaults(self):
        config = MetricsConfigMain()
        assert config.enabled is True
        assert config.metrics.namespace == "geomind"

    def test_load_config_from_yaml(self):
        config_data = {
            "metrics": {
                "enabled": True,
                "port": 9091,
                "namespace": "custom",
                "latency_buckets": [0.01, 0.1, 1.0],
            },
            "enabled": False,
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump(config_data, f)
            temp_path = f.name

        try:
            from geomind.metrics.config import Settings
            settings = Settings(config_path=temp_path)
            config = MetricsConfigMain(**yaml.safe_load(open(temp_path)))

            assert config.metrics.port == 9091
            assert config.metrics.namespace == "custom"
            assert config.metrics.latency_buckets == [0.01, 0.1, 1.0]
            assert config.enabled is False
        finally:
            Path(temp_path).unlink()