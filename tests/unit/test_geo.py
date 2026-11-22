"""Tests for pure geospatial helpers."""

from __future__ import annotations

import math

from geomind.geo import (
    bbox_for_radius,
    circle_area_m2,
    clamp_ratio,
    destination_point,
    sample_ring_points,
    utm_srid_for,
)


def test_utm_srid_northern_hemisphere() -> None:
    assert utm_srid_for(13.405, 52.52) == 32633  # Berlin


def test_utm_srid_southern_hemisphere() -> None:
    assert utm_srid_for(-58.38, -34.60) == 32721  # Buenos Aires


def test_utm_srid_clamps_longitude() -> None:
    assert utm_srid_for(180.0, 10.0) == 32660
    assert utm_srid_for(-180.0, 10.0) == 32601


def test_bbox_contains_center_and_is_ordered() -> None:
    bbox = bbox_for_radius(52.52, 13.405, 500.0)
    assert bbox.min_lat < 52.52 < bbox.max_lat
    assert bbox.min_lon < 13.405 < bbox.max_lon
    assert bbox.min_lat < bbox.max_lat
    assert bbox.min_lon < bbox.max_lon


def test_destination_point_moves_expected_distance() -> None:
    lat, lon = destination_point(0.0, 0.0, 0.0, 1000.0)
    # 1000 m north is ~0.00899 degrees of latitude.
    assert math.isclose(lat, 0.008993, abs_tol=1e-4)
    assert math.isclose(lon, 0.0, abs_tol=1e-9)


def test_destination_point_normalises_longitude() -> None:
    _, lon = destination_point(0.0, 179.999, 90.0, 1000.0)
    assert -180.0 <= lon <= 180.0


def test_sample_ring_points_includes_center() -> None:
    points = sample_ring_points(52.52, 13.405, 500.0, 4)
    assert len(points) == 5
    assert points[0] == (52.52, 13.405)
    assert all(point != points[0] for point in points[1:])


def test_sample_ring_points_with_zero_ring() -> None:
    assert sample_ring_points(52.52, 13.405, 500.0, 0) == [(52.52, 13.405)]


def test_circle_area() -> None:
    assert math.isclose(circle_area_m2(100.0), math.pi * 10_000.0)


def test_clamp_ratio() -> None:
    assert clamp_ratio(-0.5) == 0.0
    assert clamp_ratio(0.5) == 0.5
    assert clamp_ratio(1.5) == 1.0
