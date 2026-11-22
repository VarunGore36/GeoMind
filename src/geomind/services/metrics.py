"""Pure metric computations (no I/O).

Kept separate from the orchestrator so the maths can be unit tested directly.
"""

from __future__ import annotations

from dataclasses import dataclass

from ..providers.base import ElevationSample


@dataclass(frozen=True, slots=True)
class ElevationSummary:
    value: float | None
    min_m: float | None
    max_m: float | None
    mean_m: float | None
    samples: int


def ratio(part: float, whole: float) -> float | None:
    """Return ``part / whole`` clamped to ``[0, 1]``, or ``None``."""

    if whole <= 0:
        return None
    return max(0.0, min(1.0, part / whole))


def road_density_km_per_km2(length_m: float, area_m2: float) -> float | None:
    """Convert a road length in metres and an area in m² to km/km²."""

    if area_m2 <= 0:
        return None
    return (length_m / 1000.0) / (area_m2 / 1_000_000.0)


def summarize_elevation(samples: list[ElevationSample]) -> ElevationSummary:
    """Summarise elevation samples; ``value`` is the first (centre) sample."""

    if not samples:
        return ElevationSummary(value=None, min_m=None, max_m=None, mean_m=None, samples=0)

    elevations = [sample.elevation_m for sample in samples]
    return ElevationSummary(
        value=elevations[0],
        min_m=min(elevations),
        max_m=max(elevations),
        mean_m=sum(elevations) / len(elevations),
        samples=len(elevations),
    )
