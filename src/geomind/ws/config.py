"""WebSocket configuration."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class WSConfig(BaseModel):
    """WebSocket server configuration."""

    # Connection settings
    max_connections: int = Field(default=1000, ge=1)
    max_connections_per_client: int = Field(default=10, ge=1)
    heartbeat_interval_seconds: int = Field(default=30, ge=5)
    connection_timeout_seconds: int = Field(default=300, ge=30)

    # Message settings
    max_message_size_bytes: int = Field(default=1_048_576, ge=1024)  # 1MB
    max_queue_size: int = Field(default=1000, ge=10)
    ping_interval_seconds: int = Field(default=20, ge=5)
    ping_timeout_seconds: int = Field(default=10, ge=2)

    # Redis for scaling
    redis_url: str = "redis://localhost:6379/0"
    redis_enabled: bool = False
    channel_prefix: str = "geomind:ws"

    # Message types
    supported_message_types: list[str] = Field(
        default_factory=lambda: [
            "subscribe",
            "unsubscribe",
            "detection",
            "heartbeat",
            "ack",
            "error",
        ]
    )


class WSAuthConfig(BaseModel):
    """WebSocket authentication configuration."""

    # Token validation
    token_query_param: str = "token"
    token_header_name: str = "Authorization"
    token_prefix: str = "Bearer "

    # Allow anonymous connections (for public streams)
    allow_anonymous: bool = False

    # Required roles for different channel types
    required_roles: dict[str, list[str]] = Field(
        default_factory=lambda: {
            "detections:*": ["viewer", "user", "admin", "service"],
            "cameras:*": ["viewer", "user", "admin", "service"],
            "detections:confirmed:*": ["user", "admin", "service"],
            "detections:pending:*": ["admin", "service"],
            "system:*": ["admin", "service"],
        }
    )


class WSConfigMain(BaseModel):
    """Main WebSocket configuration."""

    ws: WSConfig = WSConfig()
    auth: WSAuthConfig = WSAuthConfig()
    enabled: bool = True
    log_level: str = "INFO"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_prefix="GEOMIND_WS_",
        extra="ignore",
    )
    config_path: str = "ws.yaml"


@lru_cache
def get_ws_config() -> "WSConfigMain":
    """Load and cache the WebSocket configuration."""
    import yaml
    from pathlib import Path

    settings = Settings()
    config_path = Path(settings.config_path)
    if config_path.exists():
        with open(config_path) as f:
            data = yaml.safe_load(f)
        return WSConfigMain(**data)
    return WSConfigMain()