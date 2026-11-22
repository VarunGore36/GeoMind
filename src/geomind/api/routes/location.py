"""Location Intelligence endpoint."""

from __future__ import annotations

from fastapi import APIRouter, Depends, Query

from ...core.config import get_settings
from ...schemas.location import LocationIntelligenceResponse
from ...services.location_intel import LocationIntelligenceService
from ..deps import get_location_service

router = APIRouter(prefix="/v1", tags=["location intelligence"])


@router.get(
    "/location-intelligence",
    response_model=LocationIntelligenceResponse,
    summary="Extract geospatial features around a point",
    description=(
        "Returns building density, road density, vegetation coverage, water "
        "proximity, and elevation for a circular area. Data comes from "
        "OpenStreetMap (via Overpass) and open elevation datasets; the response "
        "includes full provenance and source metadata."
    ),
)
async def location_intelligence(
    lat: float = Query(..., ge=-90, le=90, description="Latitude in WGS84 degrees."),
    lon: float = Query(..., ge=-180, le=180, description="Longitude in WGS84 degrees."),
    radius_m: float | None = Query(
        default=None,
        ge=1,
        le=10000,
        description="Analysis radius in metres (defaults to server config).",
    ),
    refresh: bool = Query(
        default=False,
        description="Bypass the local coverage cache and refetch from Overpass.",
    ),
    service: LocationIntelligenceService = Depends(get_location_service),
) -> LocationIntelligenceResponse:
    effective_radius = radius_m if radius_m is not None else get_settings().default_radius_m
    return await service.analyze(lat, lon, effective_radius, refresh=refresh)
