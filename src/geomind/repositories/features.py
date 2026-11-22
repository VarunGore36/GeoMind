"""PostGIS-backed metric queries.

All heavy spatial work happens in the database: circles are buffered in a
locally-appropriate UTM projection (metres), features are clipped to the
circle, and areas/lengths are aggregated. This keeps the Python layer thin and
lets PostGIS use its spatial indexes.

Clipping is performed in EPSG:4326 (small coordinate values) and the clipped
geometry is then transformed to the local UTM zone for measurement. Clipping
directly in UTM can hit GEOS robustness limits at large easting/northing
values, which silently produces empty intersections.
"""

from __future__ import annotations

from dataclasses import dataclass

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession


@dataclass(frozen=True, slots=True)
class AreaMetrics:
    building_count: int
    building_area_m2: float
    road_count: int
    road_length_m: float
    vegetation_count: int
    vegetation_area_m2: float
    water_area_m2: float


@dataclass(frozen=True, slots=True)
class NearestWater:
    osm_type: str
    osm_id: int
    name: str | None
    distance_m: float


_AREA_METRICS_SQL = text(
    """
WITH params AS (
    SELECT ST_Transform(
        ST_SetSRID(
            ST_MakePoint(CAST(:lon AS double precision), CAST(:lat AS double precision)),
            4326
        ),
        CAST(:srid AS integer)
    ) AS center
),
circle AS (
    SELECT
        ST_Buffer(center, CAST(:radius AS double precision)) AS geom,
        ST_Transform(
            ST_Buffer(center, CAST(:radius AS double precision)), 4326
        ) AS geom_4326
    FROM params
)
SELECT
    (
        SELECT count(*) FROM osm_features f, circle c
        WHERE f.feature_type = 'building'
          AND ST_Intersects(f.geom, c.geom_4326)
    ) AS building_count,
    (
        SELECT COALESCE(sum(
            ST_Area(ST_Transform(
                ST_Intersection(
                    ST_CollectionExtract(ST_MakeValid(f.geom), 3),
                    c.geom_4326
                ),
                CAST(:srid AS integer)
            ))
        ), 0)
        FROM osm_features f, circle c
        WHERE f.feature_type = 'building'
          AND ST_Intersects(f.geom, c.geom_4326)
    ) AS building_area,
    (
        SELECT count(*) FROM osm_features f, circle c
        WHERE f.feature_type = 'road'
          AND ST_Intersects(f.geom, c.geom_4326)
    ) AS road_count,
    (
        SELECT COALESCE(sum(
            ST_Length(
                ST_Intersection(
                    ST_SnapToGrid(f.geom, 0.000001)::geography,
                    c.geom_4326::geography
                )
            )
        ), 0)
        FROM osm_features f, circle c
        WHERE f.feature_type = 'road'
          AND ST_Intersects(f.geom, c.geom_4326)
    ) AS road_length,
    (
        SELECT count(*) FROM osm_features f, circle c
        WHERE f.feature_type = 'vegetation'
          AND ST_Intersects(f.geom, c.geom_4326)
    ) AS vegetation_count,
    (
        SELECT COALESCE(sum(
            ST_Area(ST_Transform(
                ST_Intersection(
                    ST_CollectionExtract(ST_MakeValid(f.geom), 3),
                    c.geom_4326
                ),
                CAST(:srid AS integer)
            ))
        ), 0)
        FROM osm_features f, circle c
        WHERE f.feature_type = 'vegetation'
          AND ST_Intersects(f.geom, c.geom_4326)
    ) AS vegetation_area,
    (
        SELECT COALESCE(sum(
            ST_Area(ST_Transform(
                ST_Intersection(
                    ST_CollectionExtract(ST_MakeValid(f.geom), 3),
                    c.geom_4326
                ),
                CAST(:srid AS integer)
            ))
        ), 0)
        FROM osm_features f, circle c
        WHERE f.feature_type = 'water'
          AND ST_Intersects(f.geom, c.geom_4326)
    ) AS water_area
"""
)

_NEAREST_WATER_SQL = text(
    """
SELECT
    f.osm_type,
    f.osm_id,
    f.name,
    ST_Distance(
        f.geom::geography,
        ST_SetSRID(
            ST_MakePoint(CAST(:lon AS double precision), CAST(:lat AS double precision)),
            4326
        )::geography
    ) AS distance_m
FROM osm_features f
WHERE f.feature_type = 'water'
  AND ST_DWithin(
        f.geom::geography,
        ST_SetSRID(
            ST_MakePoint(CAST(:lon AS double precision), CAST(:lat AS double precision)),
            4326
        )::geography,
        CAST(:search_radius AS double precision)
  )
ORDER BY distance_m ASC
LIMIT 1
"""
)


class FeatureRepository:
    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def area_metrics(
        self, latitude: float, longitude: float, radius_m: float, srid: int
    ) -> AreaMetrics:
        row = (
            await self._session.execute(
                _AREA_METRICS_SQL,
                {
                    "lat": latitude,
                    "lon": longitude,
                    "radius": radius_m,
                    "srid": srid,
                },
            )
        ).one()
        return AreaMetrics(
            building_count=int(row.building_count),
            building_area_m2=float(row.building_area),
            road_count=int(row.road_count),
            road_length_m=float(row.road_length),
            vegetation_count=int(row.vegetation_count),
            vegetation_area_m2=float(row.vegetation_area),
            water_area_m2=float(row.water_area),
        )

    async def nearest_water(
        self, latitude: float, longitude: float, search_radius_m: float
    ) -> NearestWater | None:
        row = (
            await self._session.execute(
                _NEAREST_WATER_SQL,
                {
                    "lat": latitude,
                    "lon": longitude,
                    "search_radius": search_radius_m,
                },
            )
        ).first()
        if row is None:
            return None
        return NearestWater(
            osm_type=row.osm_type,
            osm_id=int(row.osm_id),
            name=row.name,
            distance_m=float(row.distance_m),
        )
