"""Authentication configuration."""

from __future__ import annotations

from functools import lru_cache
from typing import Literal

from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class JWTConfig(BaseModel):
    """JWT token configuration."""

    secret_key: str = Field(default="dev-secret-change-in-production", min_length=32)
    algorithm: Literal["HS256", "HS384", "HS512", "RS256"] = "HS256"
    access_token_expire_minutes: int = Field(default=30, ge=1)
    refresh_token_expire_days: int = Field(default=7, ge=1)
    issuer: str = "geomind"
    audience: str = "geomind-api"


class APIKeyConfig(BaseModel):
    """API Key configuration."""

    header_name: str = "X-API-Key"
    query_param: str = "api_key"
    key_length: int = 32
    prefix: str = "gm_"


class AuthConfig(BaseModel):
    """Main authentication configuration."""

    jwt: JWTConfig = JWTConfig()
    api_key: APIKeyConfig = APIKeyConfig()
    enabled: bool = True
    # Paths that don't require authentication
    public_paths: list[str] = Field(
        default_factory=lambda: [
            "/",
            "/health",
            "/health/ready",
            "/docs",
            "/openapi.json",
            "/redoc",
        ]
    )


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_prefix="GEOMIND_AUTH_",
        extra="ignore",
    )
    config_path: str = "auth.yaml"


@lru_cache
def get_auth_config() -> AuthConfig:
    """Load and cache the authentication configuration."""
    import yaml
    from pathlib import Path

    settings = Settings()
    config_path = Path(settings.config_path)
    if config_path.exists():
        with open(config_path) as f:
            data = yaml.safe_load(f)
        return AuthConfig(**data)
    return AuthConfig()