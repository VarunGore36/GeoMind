"""OpenStreetMap data via the Overpass API.

The Overpass API is queried for the features needed by the Location
Intelligence metrics, then responses are normalised into GeoJSON geometries.
Multiple public endpoints are tried in order with exponential backoff, because
the free instances are frequently at capacity.
"""

from __future__ import annotations

import asyncio
import logging
from datetime import UTC, datetime
from typing import Any

import httpx

from ..core.config import Settings
from ..core.errors import DataSourceUnavailableError
from .base import (
    FEATURE_BUILDING,
    FEATURE_ROAD,
    FEATURE_VEGETATION,
    FEATURE_WATER,
    NormalizedFeature,
    OsmFetchResult,
)

logger = logging.getLogger(__name__)

VEGETATION_LANDUSE = {
    "forest",
    "meadow",
    "grass",
    "farmland",
    "orchard",
    "vineyard",
    "village_green",
    "recreation_ground",
}
VEGETATION_NATURAL = {"wood", "scrub", "grassland", "heath", "wetland"}
VEGETATION_LEISURE = {"park", "garden", "nature_reserve", "golf_course"}

WATERWAY_VALUES = "^(river|stream|canal|drain|ditch)$"
VEGETATION_LANDUSE_RE = "^(forest|meadow|grass|farmland|orchard|vineyard|village_green|recreation_ground)$"
VEGETATION_NATURAL_RE = "^(wood|scrub|grassland|heath|wetland)$"
VEGETATION_LEISURE_RE = "^(park|garden|nature_reserve|golf_course)$"


def build_overpass_query(
    latitude: float, longitude: float, radius_m: float, water_radius_m: float, timeout_s: int
) -> str:
    """Build the Overpass QL query for one analysis area."""

    r = int(round(radius_m))
    w = int(round(water_radius_m))
    point = f"{latitude},{longitude}"
    clauses = [
        f'way["building"](around:{r},{point});',
        f'way["highway"](around:{r},{point});',
        f'way["landuse"~"{VEGETATION_LANDUSE_RE}"](around:{r},{point});',
        f'way["natural"~"{VEGETATION_NATURAL_RE}"](around:{r},{point});',
        f'way["leisure"~"{VEGETATION_LEISURE_RE}"](around:{r},{point});',
        f'way["natural"="water"](around:{w},{point});',
        f'way["water"](around:{w},{point});',
        f'way["landuse"="reservoir"](around:{w},{point});',
        f'way["waterway"~"{WATERWAY_VALUES}"](around:{w},{point});',
        f'relation["natural"="water"](around:{w},{point});',
        f'relation["water"](around:{w},{point});',
        f'relation["waterway"~"{WATERWAY_VALUES}"](around:{w},{point});',
    ]
    body = "\n  ".join(clauses)
    return f"[out:json][timeout:{timeout_s}];\n(\n  {body}\n);\nout geom;"


def _classify(tags: dict[str, str]) -> str | None:
    if not tags:
        return None
    if "building" in tags:
        return FEATURE_BUILDING
    if "highway" in tags:
        return FEATURE_ROAD
    if (
        tags.get("natural") == "water"
        or "water" in tags
        or tags.get("landuse") == "reservoir"
        or "waterway" in tags
    ):
        return FEATURE_WATER
    if (
        tags.get("landuse") in VEGETATION_LANDUSE
        or tags.get("natural") in VEGETATION_NATURAL
        or tags.get("leisure") in VEGETATION_LEISURE
    ):
        return FEATURE_VEGETATION
    return None


def _is_area(tags: dict[str, str], feature_type: str) -> bool:
    """Whether the feature should be treated as a polygon (area) or a line."""

    if feature_type in (FEATURE_BUILDING, FEATURE_VEGETATION):
        return True
    if feature_type == FEATURE_WATER:
        is_water_area = (
            tags.get("natural") == "water"
            or "water" in tags
            or tags.get("landuse") == "reservoir"
        )
        return is_water_area or "waterway" not in tags
    return False


def _coords_from_geometry(geometry: list[dict[str, Any]] | None) -> list[tuple[float, float]]:
    coords: list[tuple[float, float]] = []
    for point in geometry or []:
        lat, lon = point.get("lat"), point.get("lon")
        if lat is None or lon is None:
            continue
        coords.append((float(lon), float(lat)))
    return coords


def _close_ring(coords: list[tuple[float, float]]) -> list[tuple[float, float]] | None:
    if len(coords) < 3:
        return None
    if coords[0] != coords[-1]:
        coords = [*coords, coords[0]]
    if len(coords) < 4:
        return None
    return coords


def _polygon(coords: list[tuple[float, float]]) -> dict[str, Any] | None:
    ring = _close_ring(coords)
    if ring is None:
        return None
    return {"type": "Polygon", "coordinates": [ring]}


def _linestring(coords: list[tuple[float, float]]) -> dict[str, Any] | None:
    if len(coords) < 2:
        return None
    return {"type": "LineString", "coordinates": coords}


def _rings_from_segments(
    segments: list[list[tuple[float, float]]], tolerance: float = 1e-9
) -> list[list[tuple[float, float]]]:
    """Chain unordered way segments into closed rings (best effort)."""

    def same(a: tuple[float, float], b: tuple[float, float]) -> bool:
        return abs(a[0] - b[0]) <= tolerance and abs(a[1] - b[1]) <= tolerance

    remaining = [list(seg) for seg in segments if len(seg) >= 2]
    rings: list[list[tuple[float, float]]] = []

    while remaining:
        ring = remaining.pop(0)
        extended = True
        while extended and not same(ring[0], ring[-1]):
            extended = False
            for index, segment in enumerate(remaining):
                if same(ring[-1], segment[0]):
                    ring.extend(segment[1:])
                elif same(ring[-1], segment[-1]):
                    ring.extend(reversed(segment[:-1]))
                elif same(ring[0], segment[-1]):
                    ring = segment[:-1] + ring
                elif same(ring[0], segment[0]):
                    ring = list(reversed(segment[1:])) + ring
                else:
                    continue
                remaining.pop(index)
                extended = True
                break

        closed = _close_ring(ring)
        if closed is not None:
            rings.append(closed)

    return rings


def _relation_geometry(
    members: list[dict[str, Any]], is_area: bool
) -> dict[str, Any] | None:
    segments: list[list[tuple[float, float]]] = []
    for member in members:
        if member.get("type") != "way":
            continue
        if is_area and member.get("role") not in ("outer", "", None):
            continue
        coords = _coords_from_geometry(member.get("geometry"))
        if len(coords) >= 2:
            segments.append(coords)

    if not segments:
        return None

    if not is_area:
        if len(segments) == 1:
            return _linestring(segments[0])
        return {"type": "MultiLineString", "coordinates": segments}

    rings = _rings_from_segments(segments)
    if not rings:
        return None
    if len(rings) == 1:
        return {"type": "Polygon", "coordinates": [rings[0]]}
    return {"type": "MultiPolygon", "coordinates": [[ring] for ring in rings]}


def _feature_from_element(element: dict[str, Any]) -> NormalizedFeature | None:
    tags = {str(k): str(v) for k, v in (element.get("tags") or {}).items()}
    feature_type = _classify(tags)
    if feature_type is None:
        return None

    element_type = element.get("type")
    osm_id = element.get("id")
    if element_type not in ("way", "relation") or osm_id is None:
        return None

    is_area = _is_area(tags, feature_type)
    if element_type == "way":
        coords = _coords_from_geometry(element.get("geometry"))
        geometry = _polygon(coords) if is_area else _linestring(coords)
    else:
        geometry = _relation_geometry(element.get("members") or [], is_area)

    if geometry is None:
        return None

    return NormalizedFeature(
        osm_type=element_type,
        osm_id=int(osm_id),
        feature_type=feature_type,
        geometry=geometry,
        name=tags.get("name"),
        tags=tags,
    )


def parse_overpass_payload(payload: dict[str, Any]) -> list[NormalizedFeature]:
    """Normalise an Overpass JSON payload into deduplicated features."""

    features: dict[tuple[str, int, str], NormalizedFeature] = {}
    for element in payload.get("elements") or []:
        feature = _feature_from_element(element)
        if feature is not None:
            features[(feature.osm_type, feature.osm_id, feature.feature_type)] = feature
    return list(features.values())


class OverpassProvider:
    """Fetches OSM features from Overpass with endpoint failover."""

    def __init__(self, client: httpx.AsyncClient, settings: Settings) -> None:
        self._client = client
        self._settings = settings

    async def fetch_features(
        self, latitude: float, longitude: float, radius_m: float, water_radius_m: float
    ) -> OsmFetchResult:
        query = build_overpass_query(
            latitude,
            longitude,
            radius_m,
            water_radius_m,
            self._settings.overpass_timeout_s,
        )

        attempts: list[str] = []
        max_retries = self._settings.overpass_max_retries
        for attempt in range(max_retries + 1):
            for endpoint in self._settings.overpass_endpoints:
                try:
                    response = await self._client.post(endpoint, data={"data": query})
                    if response.status_code != 200:
                        attempts.append(f"{endpoint}: HTTP {response.status_code}")
                        continue
                    payload = response.json()
                    features = parse_overpass_payload(payload)
                    logger.info(
                        "Overpass returned %d elements (%d usable) from %s",
                        len(payload.get("elements") or []),
                        len(features),
                        endpoint,
                    )
                    return OsmFetchResult(
                        features=features,
                        endpoint=endpoint,
                        fetched_at=datetime.now(UTC),
                        raw_element_count=len(payload.get("elements") or []),
                    )
                except (httpx.HTTPError, ValueError) as exc:  # noqa: PERF203
                    attempts.append(f"{endpoint}: {exc}")

            if attempt < max_retries:
                delay = self._settings.overpass_backoff_s * (2**attempt)
                logger.warning("Overpass attempt %d failed; retrying in %.1fs", attempt + 1, delay)
                await asyncio.sleep(delay)

        raise DataSourceUnavailableError(
            "All Overpass endpoints failed; try again shortly.",
            details={"attempts": attempts[-10:]},
        )
