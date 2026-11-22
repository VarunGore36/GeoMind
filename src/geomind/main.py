"""GeoMind FastAPI application factory."""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import AsyncIterator

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from . import __version__
from .api.routes import health, location
from .core.config import get_settings
from .core.errors import GeoMindError
from .core.logging import configure_logging
from .providers.elevation import OpenTopoDataProvider
from .providers.overpass import OverpassProvider


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    settings = get_settings()
    configure_logging(settings.log_level)

    # One shared HTTP client for all outbound provider calls.
    async with httpx.AsyncClient(
        timeout=settings.http_timeout_s,
        headers={"User-Agent": settings.user_agent},
    ) as client:
        app.state.http_client = client
        app.state.osm_provider = OverpassProvider(client, settings)
        app.state.elevation_provider = OpenTopoDataProvider(client, settings)
        yield


def create_app() -> FastAPI:
    settings = get_settings()
    app = FastAPI(
        title=settings.app_name,
        version=settings.version,
        description=(
            "GeoMind turns open geospatial data into structured location "
            "intelligence: building density, road density, vegetation coverage, "
            "water proximity, and elevation — with full data provenance."
        ),
        lifespan=lifespan,
    )

    app.include_router(health.router)
    app.include_router(location.router)

    @app.exception_handler(GeoMindError)
    async def _geomind_error_handler(_: Request, exc: GeoMindError) -> JSONResponse:
        return JSONResponse(status_code=exc.status_code, content=exc.to_dict())

    @app.get("/", tags=["meta"], summary="Service metadata")
    async def root() -> dict[str, str]:
        return {
            "name": settings.app_name,
            "version": __version__,
            "docs": "/docs",
            "openapi": "/openapi.json",
        }

    return app


app = create_app()
