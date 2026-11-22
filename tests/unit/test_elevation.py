"""Tests for the elevation provider (HTTP mocked with respx)."""

from __future__ import annotations

import httpx
import pytest
import respx
from httpx import Response

from geomind.core.config import Settings
from geomind.core.errors import DataSourceUnavailableError
from geomind.providers.elevation import OpenTopoDataProvider


@pytest.fixture
def settings() -> Settings:
    return Settings()


@respx.mock
async def test_opentopodata_success(settings: Settings) -> None:
    respx.get(url__startswith=settings.elevation_endpoint).mock(
        return_value=Response(
            200,
            json={
                "status": "OK",
                "results": [
                    {
                        "dataset": "srtm30m",
                        "elevation": 40.0,
                        "location": {"lat": 52.52, "lng": 13.405},
                    },
                    {
                        "dataset": "srtm30m",
                        "elevation": 45.0,
                        "location": {"lat": 52.521, "lng": 13.405},
                    },
                ],
            },
        )
    )

    async with httpx.AsyncClient() as client:
        provider = OpenTopoDataProvider(client, settings)
        result = await provider.lookup([(52.52, 13.405), (52.521, 13.405)])

    assert result.provider == "OpenTopoData"
    assert result.dataset == "srtm30m"
    assert [sample.elevation_m for sample in result.samples] == [40.0, 45.0]


@respx.mock
async def test_falls_back_to_open_elevation(settings: Settings) -> None:
    respx.get(url__startswith=settings.elevation_endpoint).mock(
        return_value=Response(500)
    )
    respx.get(url__startswith=settings.elevation_fallback_endpoint).mock(
        return_value=Response(
            200,
            json={
                "results": [
                    {"latitude": 52.52, "longitude": 13.405, "elevation": 44.5}
                ]
            },
        )
    )

    async with httpx.AsyncClient() as client:
        provider = OpenTopoDataProvider(client, settings)
        result = await provider.lookup([(52.52, 13.405)])

    assert result.provider == "Open-Elevation"
    assert result.samples[0].elevation_m == 44.5


@respx.mock
async def test_raises_when_all_elevation_sources_fail(settings: Settings) -> None:
    respx.get(url__startswith=settings.elevation_endpoint).mock(
        return_value=Response(500)
    )
    respx.get(url__startswith=settings.elevation_fallback_endpoint).mock(
        return_value=Response(503)
    )

    async with httpx.AsyncClient() as client:
        provider = OpenTopoDataProvider(client, settings)
        with pytest.raises(DataSourceUnavailableError):
            await provider.lookup([(52.52, 13.405)])
