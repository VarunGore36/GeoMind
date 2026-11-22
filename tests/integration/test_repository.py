"""Integration tests for the PostGIS metric queries.

Run with a PostGIS database:

    GEOMIND_TEST_DATABASE_URL=postgresql+asyncpg://geomind:geomind@localhost:5432/geomind \
        pytest -m integration
"""

from __future__ import annotations

import pytest
from sqlalchemy import text

from geomind.geo import utm_srid_for
from geomind.repositories.features import FeatureRepository

pytestmark = pytest.mark.integration

LAT = 52.52
LON = 13.405
SRID = utm_srid_for(LON, LAT)
RADIUS_M = 300.0

_INSERT_AREA = text(
    """
WITH c AS (
    SELECT ST_Transform(
        ST_SetSRID(ST_MakePoint(:lon, :lat), 4326), CAST(:srid AS integer)
    ) AS p
)
INSERT INTO osm_features (id, osm_type, osm_id, feature_type, geom)
SELECT
    gen_random_uuid(),
    'way',
    :osm_id,
    :feature_type,
    ST_Transform(
        ST_MakeEnvelope(
            ST_X(p) + :x0, ST_Y(p) + :y0, ST_X(p) + :x1, ST_Y(p) + :y1, CAST(:srid AS integer)
        ),
        4326
    )
FROM c
"""
)

_INSERT_LINE = text(
    """
WITH c AS (
    SELECT ST_Transform(
        ST_SetSRID(ST_MakePoint(:lon, :lat), 4326), CAST(:srid AS integer)
    ) AS p
)
INSERT INTO osm_features (id, osm_type, osm_id, feature_type, geom)
SELECT
    gen_random_uuid(),
    'way',
    :osm_id,
    'road',
    ST_Transform(
        ST_SetSRID(
            ST_MakeLine(
                ST_MakePoint(ST_X(p) + :x0, ST_Y(p) + :y0),
                ST_MakePoint(ST_X(p) + :x1, ST_Y(p) + :y1)
            ),
            CAST(:srid AS integer)
        ),
        4326
    )
FROM c
"""
)


async def _insert_area(session, osm_id, feature_type, x0, y0, x1, y1) -> None:
    await session.execute(
        _INSERT_AREA,
        {
            "lat": LAT,
            "lon": LON,
            "srid": SRID,
            "osm_id": osm_id,
            "feature_type": feature_type,
            "x0": x0,
            "y0": y0,
            "x1": x1,
            "y1": y1,
        },
    )


async def test_area_metrics(db_session) -> None:
    # 100x100 m building centred on the point.
    await _insert_area(db_session, 1, "building", -50, -50, 50, 50)
    # 100x100 m vegetation patch 200 m north (fully inside the 300 m circle).
    await _insert_area(db_session, 2, "vegetation", -50, 150, 50, 250)
    # 100x100 m water patch 200 m east (fully inside the circle).
    await _insert_area(db_session, 3, "water", 150, -50, 250, 50)
    # 200 m road through the centre.
    await db_session.execute(
        _INSERT_LINE,
        {
            "lat": LAT,
            "lon": LON,
            "srid": SRID,
            "osm_id": 4,
            "x0": -100,
            "y0": 0,
            "x1": 100,
            "y1": 0,
        },
    )
    await db_session.commit()

    repo = FeatureRepository(db_session)
    metrics = await repo.area_metrics(LAT, LON, RADIUS_M, SRID)

    assert metrics.building_count == 1
    assert metrics.building_area_m2 == pytest.approx(10_000, rel=0.01)
    assert metrics.vegetation_count == 1
    assert metrics.vegetation_area_m2 == pytest.approx(10_000, rel=0.01)
    assert metrics.water_area_m2 == pytest.approx(10_000, rel=0.01)
    assert metrics.road_count == 1
    assert metrics.road_length_m == pytest.approx(200, rel=0.01)


async def test_nearest_water(db_session) -> None:
    await _insert_area(db_session, 10, "water", 150, -50, 250, 50)
    await db_session.commit()

    repo = FeatureRepository(db_session)
    nearest = await repo.nearest_water(LAT, LON, search_radius_m=2_000)

    assert nearest is not None
    assert nearest.osm_id == 10
    assert nearest.distance_m == pytest.approx(150, rel=0.02)


async def test_nearest_water_returns_none_when_out_of_range(db_session) -> None:
    await _insert_area(db_session, 11, "water", 5_000, 5_000, 5_100, 5_100)
    await db_session.commit()

    repo = FeatureRepository(db_session)
    nearest = await repo.nearest_water(LAT, LON, search_radius_m=1_000)
    assert nearest is None
