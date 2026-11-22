"""End-to-end service test with fake providers against a real PostGIS schema."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from geomind.core.config import Settings
from geomind.providers.base import (
    ElevationResult,
    ElevationSample,
    NormalizedFeature,
    OsmFetchResult,
)
from geomind.services.location_intel import LocationIntelligenceService

pytestmark = pytest.mark.integration

LAT = 52.52
LON = 13.405
RADIUS = 500.0


def _square(lon: float, lat: float, half: float = 0.0005) -> dict:
    return {
        "type": "Polygon",
        "coordinates": [
            [
                [lon - half, lat - half],
                [lon + half, lat - half],
                [lon + half, lat + half],
                [lon - half, lat + half],
                [lon - half, lat - half],
            ]
        ],
    }


def _features() -> list[NormalizedFeature]:
    return [
        NormalizedFeature("way", 1, "building", _square(LON, LAT), name="Building"),
        NormalizedFeature(
            "way",
            2,
            "road",
            {
                "type": "LineString",
                "coordinates": [[LON - 0.001, LAT], [LON + 0.001, LAT]],
            },
        ),
        NormalizedFeature("way", 3, "vegetation", _square(LON, LAT + 0.002)),
        NormalizedFeature("way", 4, "water", _square(LON + 0.002, LAT)),
    ]


class FakeOsmProvider:
    def __init__(self, features: list[NormalizedFeature]) -> None:
        self._features = features
        self.calls = 0

    async def fetch_features(self, latitude, longitude, radius_m, water_radius_m):
        self.calls += 1
        return OsmFetchResult(
            features=self._features,
            endpoint="fake://overpass",
            fetched_at=datetime.now(UTC),
            raw_element_count=len(self._features),
        )


class FakeElevationProvider:
    async def lookup(self, points):
        return ElevationResult(
            samples=[
                ElevationSample(lat, lon, 40.0 + index)
                for index, (lat, lon) in enumerate(points)
            ],
            dataset="srtm30m",
            provider="Fake",
            url="fake://elevation",
        )


async def test_full_analysis_and_cache_hit(db_session) -> None:
    osm = FakeOsmProvider(_features())
    service = LocationIntelligenceService(
        session=db_session,
        osm_provider=osm,
        elevation_provider=FakeElevationProvider(),
        settings=Settings(),
    )

    result = await service.analyze(LAT, LON, RADIUS)

    assert result.features.building_density.value is not None
    assert result.features.building_density.value > 0
    assert result.features.road_density.value is not None
    assert result.features.road_density.value > 0
    assert result.features.vegetation_coverage.value is not None
    assert result.features.vegetation_coverage.value > 0
    assert result.features.water_proximity.nearest_distance_m is not None
    assert result.features.water_proximity.water_coverage_ratio is not None
    assert result.features.water_proximity.water_coverage_ratio > 0
    assert result.features.elevation.value == 40.0
    assert result.features.elevation.samples == 5
    assert len(result.provenance.sources) == 2
    assert result.provenance.cache.hit is False
    assert osm.calls == 1

    cached = await service.analyze(LAT, LON, RADIUS)
    assert cached.provenance.cache.hit is True
    assert osm.calls == 1
