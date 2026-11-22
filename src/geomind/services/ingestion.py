"""Ingestion of OSM features with a simple coverage cache.

A request is served from the cache when a previous ingestion run covered the
same point and at least the same radius within the configured TTL. Otherwise
features are fetched from Overpass, upserted, and the run is recorded for
provenance.
"""

from __future__ import annotations

import logging
from typing import Any

from geoalchemy2.elements import WKTElement
from shapely.geometry import shape
from sqlalchemy import func, text
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.ext.asyncio import AsyncSession

from ..core.config import Settings
from ..db.models import IngestionArea, OsmFeature
from ..providers.base import NormalizedFeature, OsmFetchResult, OsmProvider
from ..schemas.location import CacheInfo, OsmQueryInfo

logger = logging.getLogger(__name__)

_UPSERT_CHUNK = 500

_COVERING_AREA_SQL = text(
    """
SELECT radius_m, feature_count, osm_endpoint, fetched_at
FROM ingestion_areas
WHERE fetched_at > now() - make_interval(secs => :ttl)
  AND radius_m >= CAST(:radius AS double precision)
  AND ST_DWithin(
        center::geography,
        ST_SetSRID(
            ST_MakePoint(CAST(:lon AS double precision), CAST(:lat AS double precision)),
            4326
        )::geography,
        radius_m - CAST(:radius AS double precision)
  )
ORDER BY radius_m ASC
LIMIT 1
"""
)


class IngestionService:
    def __init__(
        self, session: AsyncSession, osm_provider: OsmProvider, settings: Settings
    ) -> None:
        self._session = session
        self._osm = osm_provider
        self._settings = settings

    async def ensure_coverage(
        self, latitude: float, longitude: float, radius_m: float, refresh: bool = False
    ) -> tuple[CacheInfo, OsmQueryInfo]:
        covering = None if refresh else await self._find_covering_area(
            latitude, longitude, radius_m
        )

        if covering is not None:
            logger.info("Serving from cache (radius_m=%s)", covering["radius_m"])
            return (
                CacheInfo(
                    hit=True,
                    fetched_at=covering["fetched_at"],
                    ttl_s=self._settings.cache_ttl_s,
                ),
                OsmQueryInfo(
                    endpoint=covering["osm_endpoint"],
                    feature_count=int(covering["feature_count"]),
                    fetched_at=covering["fetched_at"],
                ),
            )

        water_radius = radius_m + self._settings.water_search_buffer_m
        result = await self._osm.fetch_features(
            latitude, longitude, radius_m, water_radius
        )
        await self._upsert_features(result.features)
        await self._record_area(latitude, longitude, radius_m, result)
        await self._session.commit()

        return (
            CacheInfo(
                hit=False, fetched_at=result.fetched_at, ttl_s=self._settings.cache_ttl_s
            ),
            OsmQueryInfo(
                endpoint=result.endpoint,
                feature_count=len(result.features),
                fetched_at=result.fetched_at,
            ),
        )

    async def _find_covering_area(
        self, latitude: float, longitude: float, radius_m: float
    ) -> dict[str, Any] | None:
        row = (
            await self._session.execute(
                _COVERING_AREA_SQL,
                {
                    "ttl": self._settings.cache_ttl_s,
                    "radius": radius_m,
                    "lat": latitude,
                    "lon": longitude,
                },
            )
        ).first()
        if row is None:
            return None
        return {
            "radius_m": float(row.radius_m),
            "feature_count": int(row.feature_count),
            "osm_endpoint": row.osm_endpoint,
            "fetched_at": row.fetched_at,
        }

    async def _upsert_features(self, features: list[NormalizedFeature]) -> None:
        if not features:
            return

        # Guard against duplicate conflict targets within a single statement.
        unique: dict[tuple[str, int, str], NormalizedFeature] = {
            (f.osm_type, f.osm_id, f.feature_type): f for f in features
        }
        rows = list(unique.values())

        for start in range(0, len(rows), _UPSERT_CHUNK):
            chunk = rows[start : start + _UPSERT_CHUNK]
            values = [
                {
                    "osm_type": feature.osm_type,
                    "osm_id": feature.osm_id,
                    "feature_type": feature.feature_type,
                    "name": feature.name,
                    "tags": feature.tags,
                    "source": feature.source,
                    "geom": WKTElement(shape(feature.geometry).wkt, srid=4326),
                }
                for feature in chunk
            ]
            statement = insert(OsmFeature).values(values)
            statement = statement.on_conflict_do_update(
                constraint="uq_osm_feature",
                set_={
                    "name": statement.excluded.name,
                    "tags": statement.excluded.tags,
                    "geom": statement.excluded.geom,
                    "source": statement.excluded.source,
                    "fetched_at": func.now(),
                },
            )
            await self._session.execute(statement)

    async def _record_area(
        self,
        latitude: float,
        longitude: float,
        radius_m: float,
        result: OsmFetchResult,
    ) -> None:
        area = IngestionArea(
            center=WKTElement(f"POINT({longitude} {latitude})", srid=4326),
            radius_m=radius_m,
            feature_count=len(result.features),
            osm_endpoint=result.endpoint,
            fetched_at=result.fetched_at,
        )
        self._session.add(area)
