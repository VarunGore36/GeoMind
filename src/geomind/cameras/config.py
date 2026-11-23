"""Camera pipeline configuration."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class CameraFeedConfig(BaseModel):
    feed_url: str
    camera_id: str
    fetch_interval_seconds: int = 30
    timeout_seconds: int = 10
    max_retries: int = 3
    retry_delay_seconds: int = 5


class StorageConfig(BaseModel):
    path: str = "./frames"
    max_frames_per_camera: int = 1000
    deduplicate: bool = True
    similarity_threshold: float = 0.95


class DatabaseConfig(BaseModel):
    url: str = "sqlite+aiosqlite:///./cam_pipeline.db"


class RedisConfig(BaseModel):
    url: str = "redis://localhost:6379/0"
    enabled: bool = False


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_prefix="GEOMIND_CAM_",
        extra="ignore",
    )
    config_path: str = "cameras.yaml"


class CameraPipelineConfig(BaseModel):
    feeds: list[CameraFeedConfig]
    storage: StorageConfig = StorageConfig()
    database: DatabaseConfig = DatabaseConfig()
    redis: RedisConfig = RedisConfig()


@lru_cache
def get_camera_config() -> CameraPipelineConfig:
    settings = Settings()
    with open(settings.config_path) as f:
        data = yaml.safe_load(f)
    return CameraPipelineConfig(**data)