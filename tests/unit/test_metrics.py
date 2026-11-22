"""Tests for pure metric computations."""

from __future__ import annotations

import math

from geomind.providers.base import ElevationSample
from geomind.services.metrics import (
    ratio,
    road_density_km_per_km2,
    summarize_elevation,
)


def test_ratio() -> None:
    assert ratio(25.0, 100.0) == 0.25
    assert ratio(0.0, 100.0) == 0.0


def test_ratio_clamps_and_handles_zero_denominator() -> None:
    assert ratio(150.0, 100.0) == 1.0
    assert ratio(10.0, 0.0) is None


def test_road_density() -> None:
    # 1 km of road in a 1 km² area => 1 km/km².
    assert math.isclose(road_density_km_per_km2(1000.0, 1_000_000.0), 1.0)


def test_road_density_zero_area() -> None:
    assert road_density_km_per_km2(1000.0, 0.0) is None


def test_summarize_elevation() -> None:
    samples = [
        ElevationSample(52.52, 13.405, 40.0),
        ElevationSample(52.521, 13.405, 30.0),
        ElevationSample(52.519, 13.405, 50.0),
    ]
    summary = summarize_elevation(samples)
    assert summary.value == 40.0
    assert summary.min_m == 30.0
    assert summary.max_m == 50.0
    assert summary.mean_m == 40.0
    assert summary.samples == 3


def test_summarize_elevation_empty() -> None:
    summary = summarize_elevation([])
    assert summary.value is None
    assert summary.samples == 0
