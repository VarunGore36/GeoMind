"""Shared pytest fixtures."""

from __future__ import annotations

import os
from datetime import UTC, datetime

import pytest
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine
from sqlalchemy.pool import StaticPool

from geomind.db import models  # noqa: F401  (register tables on Base.metadata)
from geomind.db.base import Base
from geomind.vision.models import Base as VisionBase
from geomind.schemas.location import (
    BoundingBox,
    BuildingDensity,
    CacheInfo,
    DataSource,
    ElevationInfo,
    Features,
    LocationIntelligenceResponse,
    OsmQueryInfo,
    Provenance,
    QueryInfo,
    RoadDensity,
    VegetationCoverage,
    WaterProximity,
)


def build_response(
    latitude: float, longitude: float, radius_m: float
) -> LocationIntelligenceResponse:
    """Build a fully-populated response for API contract tests."""

    area = 3.141592653589793 * radius_m * radius_m
    now = datetime.now(UTC)
    return LocationIntelligenceResponse(
        query=QueryInfo(
            latitude=latitude,
            longitude=longitude,
            radius_m=radius_m,
            area_m2=area,
            bbox=BoundingBox(
                min_lat=latitude - 0.01,
                min_lon=longitude - 0.01,
                max_lat=latitude + 0.01,
                max_lon=longitude + 0.01,
            ),
        ),
        features=Features(
            building_density=BuildingDensity(
                value=0.2, building_count=10, footprint_area_m2=area * 0.2
            ),
            road_density=RoadDensity(
                value=1.5, road_length_m=area * 1.5 / 1000.0, road_count=4
            ),
            vegetation_coverage=VegetationCoverage(
                value=0.3, vegetation_area_m2=area * 0.3, feature_count=2
            ),
            water_proximity=WaterProximity(
                nearest_distance_m=120.0,
                found_within_m=radius_m + 2000.0,
                water_coverage_ratio=0.0,
                nearest_feature=None,
            ),
            elevation=ElevationInfo(
                value=34.0, min_m=30.0, max_m=40.0, mean_m=35.0, samples=5, dataset="srtm30m"
            ),
        ),
        provenance=Provenance(
            generated_at=now,
            sources=[
                DataSource(
                    name="OpenStreetMap",
                    dataset="buildings, roads, vegetation, water",
                    provider="Overpass API",
                    url="https://www.openstreetmap.org/copyright",
                    license="ODbL-1.0",
                    attribution="© OpenStreetMap contributors",
                )
            ],
            cache=CacheInfo(hit=False, fetched_at=now, ttl_s=86400),
            osm=OsmQueryInfo(
                endpoint="https://overpass-api.de/api/interpreter",
                feature_count=16,
                fetched_at=now,
            ),
        ),
    )


@pytest.fixture
def sample_response():
    return build_response


@pytest.fixture
async def db_session():
    """Yield a session against a disposable PostGIS schema.

    Skips unless ``GEOMIND_TEST_DATABASE_URL`` points at a PostGIS database.
    """

    url = os.environ.get("GEOMIND_TEST_DATABASE_URL")
    if not url:
        pytest.skip("GEOMIND_TEST_DATABASE_URL is not set")

    engine = create_async_engine(url)
    async with engine.begin() as connection:
        await connection.exec_driver_sql("CREATE EXTENSION IF NOT EXISTS postgis")
        await connection.run_sync(Base.metadata.drop_all)
        await connection.run_sync(Base.metadata.create_all)

    session_factory = async_sessionmaker(engine, expire_on_commit=False)
    async with session_factory() as session:
        yield session

    await engine.dispose()


@pytest.fixture
async def vision_db():
    """Yield a vision database instance with in-memory SQLite."""
    from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker
    from sqlalchemy.pool import StaticPool
    from geomind.vision.database import VisionDatabase
    from geomind.vision.models import Base as VisionBase

    engine = create_async_engine(
        "sqlite+aiosqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    async with engine.begin() as conn:
        await conn.run_sync(VisionBase.metadata.create_all)

    session_factory = async_sessionmaker(engine, expire_on_commit=False)
    db = VisionDatabase()
    db.engine = engine
    db.session_factory = session_factory

    yield db

    await engine.dispose()
