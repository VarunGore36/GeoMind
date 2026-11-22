"""Provider abstractions and normalised data-transfer objects.

Providers isolate external data sources from the rest of the application. The
service layer only sees :class:`NormalizedFeature` and :class:`ElevationSample`
objects, which makes it trivial to swap sources or inject fakes in tests.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Protocol

# Coarse feature taxonomy used by the metrics layer.
FEATURE_BUILDING = "building"
FEATURE_ROAD = "road"
FEATURE_VEGETATION = "vegetation"
FEATURE_WATER = "water"


@dataclass(frozen=True, slots=True)
class NormalizedFeature:
    """A single geospatial feature ready to be persisted."""

    osm_type: str
    osm_id: int
    feature_type: str
    geometry: dict[str, Any]  # GeoJSON geometry (Polygon/LineString/MultiPolygon)
    name: str | None = None
    tags: dict[str, str] = field(default_factory=dict)
    source: str = "osm"


@dataclass(frozen=True, slots=True)
class OsmFetchResult:
    features: list[NormalizedFeature]
    endpoint: str
    fetched_at: datetime
    raw_element_count: int


@dataclass(frozen=True, slots=True)
class ElevationSample:
    latitude: float
    longitude: float
    elevation_m: float


@dataclass(frozen=True, slots=True)
class ElevationResult:
    samples: list[ElevationSample]
    dataset: str
    provider: str
    url: str


class OsmProvider(Protocol):
    async def fetch_features(
        self, latitude: float, longitude: float, radius_m: float, water_radius_m: float
    ) -> OsmFetchResult: ...


class ElevationProvider(Protocol):
    async def lookup(self, points: list[tuple[float, float]]) -> ElevationResult: ...
