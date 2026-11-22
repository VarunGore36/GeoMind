"""Pure geospatial helpers (no I/O, no database).

These functions are deliberately dependency-light so they can be unit tested
without a database or network access.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

EARTH_RADIUS_M = 6_371_008.8  # IUGG mean Earth radius


@dataclass(frozen=True, slots=True)
class BBox:
    min_lat: float
    min_lon: float
    max_lat: float
    max_lon: float


def utm_srid_for(lon: float, lat: float) -> int:
    """Return the EPSG code of the UTM zone containing ``(lon, lat)``.

    UTM keeps metric area/length computations accurate at city scale, which is
    what we need for density metrics. ``EPSG:326xx`` is used in the northern
    hemisphere and ``EPSG:327xx`` in the southern hemisphere.
    """

    zone = int((lon + 180.0) / 6.0) + 1
    zone = max(1, min(zone, 60))
    return (32600 if lat >= 0 else 32700) + zone


def destination_point(
    lat: float, lon: float, bearing_deg: float, distance_m: float
) -> tuple[float, float]:
    """Return the ``(lat, lon)`` reached by travelling ``distance_m`` metres.

    Uses the spherical "destination point given distance and bearing" formula.
    """

    angular = distance_m / EARTH_RADIUS_M
    bearing = math.radians(bearing_deg)
    lat1 = math.radians(lat)
    lon1 = math.radians(lon)

    sin_lat2 = math.sin(lat1) * math.cos(angular) + math.cos(lat1) * math.sin(
        angular
    ) * math.cos(bearing)
    lat2 = math.asin(max(-1.0, min(1.0, sin_lat2)))
    lon2 = lon1 + math.atan2(
        math.sin(bearing) * math.sin(angular) * math.cos(lat1),
        math.cos(angular) - math.sin(lat1) * sin_lat2,
    )
    # Normalise longitude to [-180, 180].
    lon2 = (math.degrees(lon2) + 540.0) % 360.0 - 180.0
    return math.degrees(lat2), lon2


def bbox_for_radius(lat: float, lon: float, radius_m: float) -> BBox:
    """Return a latitude/longitude bounding box enclosing a circle."""

    north = destination_point(lat, lon, 0.0, radius_m)
    south = destination_point(lat, lon, 180.0, radius_m)
    east = destination_point(lat, lon, 90.0, radius_m)
    west = destination_point(lat, lon, 270.0, radius_m)

    return BBox(
        min_lat=min(south[0], north[0]),
        min_lon=min(west[1], east[1]),
        max_lat=max(south[0], north[0]),
        max_lon=max(west[1], east[1]),
    )


def sample_ring_points(
    lat: float, lon: float, radius_m: float, count: int
) -> list[tuple[float, float]]:
    """Return the centre plus ``count`` points on a ring at half the radius.

    Used to characterise elevation across the analysis area rather than at a
    single point.
    """

    points: list[tuple[float, float]] = [(lat, lon)]
    if count <= 0 or radius_m <= 0:
        return points
    ring_radius = radius_m / 2.0
    for i in range(count):
        bearing = 360.0 * i / count
        points.append(destination_point(lat, lon, bearing, ring_radius))
    return points


def circle_area_m2(radius_m: float) -> float:
    return math.pi * radius_m * radius_m


def clamp_ratio(value: float) -> float:
    """Clamp a coverage ratio to ``[0, 1]``.

    Overlapping OSM footprints can push a raw coverage slightly above 1; a
    coverage ratio is only meaningful in ``[0, 1]``.
    """

    return max(0.0, min(1.0, value))
