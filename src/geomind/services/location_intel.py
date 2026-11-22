"""Orchestration of the Location Intelligence analysis."""

from __future__ import annotations

import logging
import math
from datetime import UTC, datetime

from sqlalchemy.ext.asyncio import AsyncSession

from ..core.config import Settings
from ..core.errors import InvalidQueryError
from ..geo import bbox_for_radius, circle_area_m2, sample_ring_points, utm_srid_for
from ..providers.base import ElevationProvider, OsmProvider
from ..repositories.features import AreaMetrics, FeatureRepository, NearestWater
from ..schemas.location import (
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
    WaterFeatureSummary,
    WaterProximity,
)
from .ingestion import IngestionService
from .metrics import ratio, road_density_km_per_km2, summarize_elevation

logger = logging.getLogger(__name__)

OSM_SOURCE = DataSource(
    name="OpenStreetMap",
    dataset="buildings, roads, vegetation, water",
    provider="Overpass API",
    url="https://www.openstreetmap.org/copyright",
    license="ODbL-1.0",
    attribution="© OpenStreetMap contributors",
)


def _round(value: float | None, digits: int) -> float | None:
    return None if value is None else round(value, digits)


class LocationIntelligenceService:
    def __init__(
        self,
        session: AsyncSession,
        osm_provider: OsmProvider,
        elevation_provider: ElevationProvider,
        settings: Settings,
    ) -> None:
        self._settings = settings
        self._osm_provider = osm_provider
        self._elevation_provider = elevation_provider
        self._repo = FeatureRepository(session)
        self._ingestion = IngestionService(session, osm_provider, settings)

    async def analyze(
        self,
        latitude: float,
        longitude: float,
        radius_m: float,
        refresh: bool = False,
    ) -> LocationIntelligenceResponse:
        self._validate(latitude, longitude, radius_m)

        srid = utm_srid_for(longitude, latitude)
        cache, osm_info = await self._ingestion.ensure_coverage(
            latitude, longitude, radius_m, refresh
        )
        metrics = await self._repo.area_metrics(latitude, longitude, radius_m, srid)
        nearest = await self._repo.nearest_water(
            latitude, longitude, radius_m + self._settings.water_search_buffer_m
        )
        elevation, elevation_source = await self._elevation(latitude, longitude, radius_m)

        return self._build_response(
            latitude=latitude,
            longitude=longitude,
            radius_m=radius_m,
            metrics=metrics,
            nearest=nearest,
            elevation=elevation,
            elevation_source=elevation_source,
            cache=cache,
            osm_info=osm_info,
        )

    def _validate(self, latitude: float, longitude: float, radius_m: float) -> None:
        if not (math.isfinite(latitude) and -90.0 <= latitude <= 90.0):
            raise InvalidQueryError("latitude must be a finite value in [-90, 90]")
        if not (math.isfinite(longitude) and -180.0 <= longitude <= 180.0):
            raise InvalidQueryError("longitude must be a finite value in [-180, 180]")
        if not (
            math.isfinite(radius_m)
            and self._settings.min_radius_m <= radius_m <= self._settings.max_radius_m
        ):
            raise InvalidQueryError(
                "radius_m is outside the allowed range",
                details={
                    "min_radius_m": self._settings.min_radius_m,
                    "max_radius_m": self._settings.max_radius_m,
                },
            )

    async def _elevation(
        self, latitude: float, longitude: float, radius_m: float
    ) -> tuple[ElevationInfo, DataSource | None]:
        ring_points = max(0, self._settings.elevation_samples - 1)
        points = sample_ring_points(latitude, longitude, radius_m, ring_points)
        try:
            result = await self._elevation_provider.lookup(points)
        except Exception as exc:  # noqa: BLE001 - elevation is optional enrichment
            logger.warning("Elevation lookup failed; omitting elevation: %s", exc)
            return ElevationInfo(samples=0), None

        if not result.samples:
            return ElevationInfo(samples=0, dataset=result.dataset), None

        summary = summarize_elevation(result.samples)
        centre_sample = min(
            result.samples,
            key=lambda sample: (sample.latitude - latitude) ** 2
            + (sample.longitude - longitude) ** 2,
        )
        info = ElevationInfo(
            value=_round(centre_sample.elevation_m, 1),
            min_m=_round(summary.min_m, 1),
            max_m=_round(summary.max_m, 1),
            mean_m=_round(summary.mean_m, 1),
            samples=summary.samples,
            dataset=result.dataset,
        )
        source = DataSource(
            name="Elevation",
            dataset=result.dataset,
            provider=result.provider,
            url=result.url,
            license="Public domain (SRTM)",
            attribution="Elevation data via OpenTopoData / Open-Elevation",
        )
        return info, source

    def _build_response(
        self,
        *,
        latitude: float,
        longitude: float,
        radius_m: float,
        metrics: AreaMetrics,
        nearest: NearestWater | None,
        elevation: ElevationInfo,
        elevation_source: DataSource | None,
        cache: CacheInfo,
        osm_info: OsmQueryInfo,
    ) -> LocationIntelligenceResponse:
        area_m2 = circle_area_m2(radius_m)
        bbox = bbox_for_radius(latitude, longitude, radius_m)
        water_buffer = self._settings.water_search_buffer_m

        features = Features(
            building_density=BuildingDensity(
                value=_round(ratio(metrics.building_area_m2, area_m2), 4),
                building_count=metrics.building_count,
                footprint_area_m2=round(metrics.building_area_m2, 1),
            ),
            road_density=RoadDensity(
                value=_round(
                    road_density_km_per_km2(metrics.road_length_m, area_m2), 3
                ),
                road_length_m=round(metrics.road_length_m, 1),
                road_count=metrics.road_count,
            ),
            vegetation_coverage=VegetationCoverage(
                value=_round(ratio(metrics.vegetation_area_m2, area_m2), 4),
                vegetation_area_m2=round(metrics.vegetation_area_m2, 1),
                feature_count=metrics.vegetation_count,
            ),
            water_proximity=WaterProximity(
                nearest_distance_m=_round(nearest.distance_m, 1) if nearest else None,
                found_within_m=radius_m + water_buffer,
                water_coverage_ratio=_round(ratio(metrics.water_area_m2, area_m2), 4),
                nearest_feature=(
                    WaterFeatureSummary(
                        osm_type=nearest.osm_type,
                        osm_id=nearest.osm_id,
                        name=nearest.name,
                        distance_m=round(nearest.distance_m, 1),
                    )
                    if nearest
                    else None
                ),
            ),
            elevation=elevation,
        )

        sources = [OSM_SOURCE]
        if elevation_source is not None:
            sources.append(elevation_source)

        provenance = Provenance(
            generated_at=datetime.now(UTC),
            sources=sources,
            cache=cache,
            osm=osm_info,
        )
        query = QueryInfo(
            latitude=latitude,
            longitude=longitude,
            radius_m=radius_m,
            area_m2=round(area_m2, 1),
            bbox=BoundingBox(
                min_lat=bbox.min_lat,
                min_lon=bbox.min_lon,
                max_lat=bbox.max_lat,
                max_lon=bbox.max_lon,
            ),
        )
        return LocationIntelligenceResponse(
            query=query, features=features, provenance=provenance
        )
