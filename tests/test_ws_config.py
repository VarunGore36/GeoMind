"""Tests for WebSocket configuration."""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest
import yaml

from geomind.ws.config import WSConfigMain, WSConfig, WSAuthConfig, get_ws_config


class TestWSConfig:
    def test_ws_config_defaults(self):
        config = WSConfig()
        assert config.max_connections == 1000
        assert config.heartbeat_interval_seconds == 30
        assert config.redis_enabled is False

    def test_auth_config_defaults(self):
        config = WSAuthConfig()
        assert config.allow_anonymous is False
        assert "detections:*" in config.required_roles

    def test_ws_config_main_defaults(self):
        config = WSConfigMain()
        assert config.enabled is True
        assert config.log_level == "INFO"

    def test_load_config_from_yaml(self):
        config_data = {
            "ws": {
                "max_connections": 500,
                "heartbeat_interval_seconds": 15,
            },
            "auth": {
                "allow_anonymous": True,
            },
            "enabled": False,
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump(config_data, f)
            temp_path = f.name

        try:
            from geomind.ws.config import Settings
            settings = Settings(config_path=temp_path)
            config = WSConfigMain(**yaml.safe_load(open(temp_path)))

            assert config.ws.max_connections == 500
            assert config.ws.heartbeat_interval_seconds == 15
            assert config.auth.allow_anonymous is True
            assert config.enabled is False
        finally:
            Path(temp_path).unlink()