"""Application configuration loaded from the environment."""

from __future__ import annotations

from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Runtime configuration for the Location Intelligence API.

    Every field can be overridden with a ``GEOMIND_``-prefixed environment
    variable (or a ``.env`` file), e.g. ``GEOMIND_DATABASE_URL``.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_prefix="GEOMIND_",
        extra="ignore",
        case_sensitive=False,
    )

    # --- Application -----------------------------------------------------
    app_name: str = "GeoMind Location Intelligence API"
    version: str = "0.1.0"
    debug: bool = False
    log_level: str = "INFO"

    # --- Database --------------------------------------------------------
    database_url: str = "postgresql+asyncpg://geomind:geomind@localhost:5432/geomind"

    # --- HTTP ------------------------------------------------------------
    http_timeout_s: float = Field(default=30.0, gt=0)
    user_agent: str = "GeoMind/0.1 (+https://github.com/VarunGore36/GeoMind)"

    # --- Overpass / OpenStreetMap ---------------------------------------
    overpass_endpoints: list[str] = Field(
        default_factory=lambda: [
            "https://overpass-api.de/api/interpreter",
            "https://maps.mail.ru/osm/tools/overpass/api/interpreter",
            "https://overpass.kumi.systems/api/interpreter",
        ]
    )
    overpass_timeout_s: int = Field(default=25, gt=0)
    overpass_max_retries: int = Field(default=2, ge=0)
    overpass_backoff_s: float = Field(default=1.5, ge=0)

    # --- Elevation -------------------------------------------------------
    elevation_endpoint: str = "https://api.opentopodata.org/v1/srtm30m"
    elevation_fallback_endpoint: str = "https://api.open-elevation.com/api/v1/lookup"
    elevation_samples: int = Field(default=5, ge=1, le=25)

    # --- Query limits ----------------------------------------------------
    min_radius_m: float = Field(default=10.0, gt=0)
    max_radius_m: float = Field(default=5000.0, gt=0)
    default_radius_m: float = Field(default=500.0, gt=0)
    water_search_buffer_m: float = Field(default=2000.0, ge=0)

    # --- Caching ---------------------------------------------------------
    cache_ttl_s: int = Field(default=86_400, ge=0)


@lru_cache
def get_settings() -> Settings:
    """Return a process-wide cached :class:`Settings` instance."""

    return Settings()
