"""FastAPI dependencies."""

from __future__ import annotations

from fastapi import Depends, Request
from sqlalchemy.ext.asyncio import AsyncSession

from ..core.config import get_settings
from ..db.session import get_session
from ..providers.elevation import OpenTopoDataProvider
from ..providers.overpass import OverpassProvider
from ..services.location_intel import LocationIntelligenceService


def get_osm_provider(request: Request) -> OverpassProvider:
    return request.app.state.osm_provider


def get_elevation_provider(request: Request) -> OpenTopoDataProvider:
    return request.app.state.elevation_provider


async def get_location_service(
    session: AsyncSession = Depends(get_session),
    osm_provider: OverpassProvider = Depends(get_osm_provider),
    elevation_provider: OpenTopoDataProvider = Depends(get_elevation_provider),
) -> LocationIntelligenceService:
    return LocationIntelligenceService(
        session=session,
        osm_provider=osm_provider,
        elevation_provider=elevation_provider,
        settings=get_settings(),
    )
