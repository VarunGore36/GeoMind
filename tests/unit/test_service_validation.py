"""Service-level input validation (no DB or network)."""

from __future__ import annotations

import pytest

from geomind.core.config import Settings
from geomind.core.errors import InvalidQueryError
from geomind.services.location_intel import LocationIntelligenceService


class _Unused:
    """Placeholder provider; validation happens before any provider call."""


def _service() -> LocationIntelligenceService:
    return LocationIntelligenceService(None, _Unused(), _Unused(), Settings())  # type: ignore[arg-type]


async def test_radius_below_minimum_is_rejected() -> None:
    with pytest.raises(InvalidQueryError):
        await _service().analyze(52.52, 13.405, 1.0)


async def test_radius_above_maximum_is_rejected() -> None:
    with pytest.raises(InvalidQueryError):
        await _service().analyze(52.52, 13.405, 99_999.0)


async def test_latitude_out_of_range_is_rejected() -> None:
    with pytest.raises(InvalidQueryError):
        await _service().analyze(100.0, 13.405, 500.0)


async def test_longitude_out_of_range_is_rejected() -> None:
    with pytest.raises(InvalidQueryError):
        await _service().analyze(52.52, 200.0, 500.0)
