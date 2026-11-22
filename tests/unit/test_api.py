"""API contract tests using an injected fake service (no DB, no network)."""

from __future__ import annotations

import httpx

from geomind.api.deps import get_location_service
from geomind.core.errors import DataSourceUnavailableError
from geomind.main import create_app


class FakeService:
    def __init__(self, response) -> None:
        self._response = response

    async def analyze(self, latitude, longitude, radius_m, refresh: bool = False):
        return self._response


class FailingService:
    async def analyze(self, latitude, longitude, radius_m, refresh: bool = False):
        raise DataSourceUnavailableError("overpass is down")


async def _get(app, path: str, params: dict | None = None) -> httpx.Response:
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        return await client.get(path, params=params)


def _app_with(service) -> object:
    app = create_app()
    app.dependency_overrides[get_location_service] = lambda: service
    return app


async def test_health() -> None:
    app = create_app()
    response = await _get(app, "/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


async def test_root_metadata() -> None:
    app = create_app()
    response = await _get(app, "/")
    assert response.status_code == 200
    assert response.json()["docs"] == "/docs"


async def test_location_intelligence_ok(sample_response) -> None:
    expected = sample_response(52.52, 13.405, 500.0)
    app = _app_with(FakeService(expected))

    response = await _get(
        app,
        "/v1/location-intelligence",
        params={"lat": 52.52, "lon": 13.405, "radius_m": 500},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["query"]["latitude"] == 52.52
    assert body["features"]["building_density"]["unit"] == "ratio"
    assert body["features"]["elevation"]["dataset"] == "srtm30m"
    assert body["provenance"]["sources"][0]["name"] == "OpenStreetMap"
    assert body["provenance"]["sources"][0]["license"] == "ODbL-1.0"


async def test_location_intelligence_defaults_radius(sample_response) -> None:
    expected = sample_response(52.52, 13.405, 500.0)
    app = _app_with(FakeService(expected))

    response = await _get(
        app, "/v1/location-intelligence", params={"lat": 52.52, "lon": 13.405}
    )
    assert response.status_code == 200


async def test_invalid_latitude_returns_422(sample_response) -> None:
    app = _app_with(FakeService(sample_response(0, 0, 500)))
    response = await _get(
        app, "/v1/location-intelligence", params={"lat": 100, "lon": 13.405}
    )
    assert response.status_code == 422


async def test_radius_below_fastapi_floor_returns_422(sample_response) -> None:
    app = _app_with(FakeService(sample_response(52.52, 13.405, 500)))
    response = await _get(
        app,
        "/v1/location-intelligence",
        params={"lat": 52.52, "lon": 13.405, "radius_m": 0},
    )
    assert response.status_code == 422


async def test_data_source_failure_returns_503() -> None:
    app = _app_with(FailingService())
    response = await _get(
        app,
        "/v1/location-intelligence",
        params={"lat": 52.52, "lon": 13.405, "radius_m": 500},
    )
    assert response.status_code == 503
    body = response.json()
    assert body["error"]["type"] == "data_source_unavailable"
