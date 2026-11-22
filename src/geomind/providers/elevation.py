"""Elevation lookups via public open datasets.

OpenTopoData is used as the primary source (SRTM by default) with
Open-Elevation as a fallback. Both are free, key-less public services, which
keeps the first module easy to run without extra credentials.
"""

from __future__ import annotations

import logging

import httpx

from ..core.config import Settings
from ..core.errors import DataSourceUnavailableError
from .base import ElevationResult, ElevationSample

logger = logging.getLogger(__name__)


def _locations_param(points: list[tuple[float, float]]) -> str:
    return "|".join(f"{lat:.6f},{lon:.6f}" for lat, lon in points)


class OpenTopoDataProvider:
    """Batched elevation lookups with a fallback provider."""

    def __init__(self, client: httpx.AsyncClient, settings: Settings) -> None:
        self._client = client
        self._settings = settings

    async def lookup(self, points: list[tuple[float, float]]) -> ElevationResult:
        if not points:
            return ElevationResult(
                samples=[],
                dataset=self._dataset_name(),
                provider="OpenTopoData",
                url=self._settings.elevation_endpoint,
            )

        try:
            return await self._lookup_opentopodata(points)
        except (httpx.HTTPError, ValueError, KeyError, TypeError) as exc:
            logger.warning("OpenTopoData failed (%s); trying Open-Elevation", exc)

        return await self._lookup_open_elevation(points)

    def _dataset_name(self) -> str:
        return self._settings.elevation_endpoint.rstrip("/").split("/")[-1] or "unknown"

    async def _lookup_opentopodata(self, points: list[tuple[float, float]]) -> ElevationResult:
        response = await self._client.get(
            self._settings.elevation_endpoint,
            params={"locations": _locations_param(points)},
        )
        response.raise_for_status()
        payload = response.json()
        if payload.get("status") not in (None, "OK"):
            raise ValueError(f"OpenTopoData status: {payload.get('status')}")

        samples: list[ElevationSample] = []
        for result in payload.get("results") or []:
            elevation = result.get("elevation")
            if elevation is None:
                continue
            location = result.get("location") or {}
            samples.append(
                ElevationSample(
                    latitude=float(location.get("lat", 0.0)),
                    longitude=float(location.get("lng", 0.0)),
                    elevation_m=float(elevation),
                )
            )
        return ElevationResult(
            samples=samples,
            dataset=payload.get("results", [{}])[0].get("dataset", self._dataset_name())
            if payload.get("results")
            else self._dataset_name(),
            provider="OpenTopoData",
            url=self._settings.elevation_endpoint,
        )

    async def _lookup_open_elevation(self, points: list[tuple[float, float]]) -> ElevationResult:
        try:
            response = await self._client.get(
                self._settings.elevation_fallback_endpoint,
                params={"locations": _locations_param(points)},
            )
            response.raise_for_status()
            payload = response.json()
            samples = [
                ElevationSample(
                    latitude=float(item["latitude"]),
                    longitude=float(item["longitude"]),
                    elevation_m=float(item["elevation"]),
                )
                for item in payload.get("results") or []
                if item.get("elevation") is not None
            ]
        except (httpx.HTTPError, ValueError, KeyError, TypeError) as exc:
            raise DataSourceUnavailableError(
                "Elevation services are unavailable; elevation will be omitted.",
                details={"error": str(exc)},
            ) from exc

        return ElevationResult(
            samples=samples,
            dataset="srtm/aster (Open-Elevation)",
            provider="Open-Elevation",
            url=self._settings.elevation_fallback_endpoint,
        )
