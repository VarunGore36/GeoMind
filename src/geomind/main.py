"""GeoMind FastAPI application factory."""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import AsyncIterator

import httpx
from fastapi import FastAPI, Request, Response
from fastapi.responses import JSONResponse

from . import __version__
from .api.routes import health, location
from .auth import auth_router
from .cameras.api import router as camera_router
from .cameras.scheduler import scheduler as camera_scheduler
from .core.config import get_settings
from .core.errors import GeoMindError
from .core.logging import configure_logging
from .metrics import metrics_collector
from .providers.elevation import OpenTopoDataProvider
from .providers.overpass import OverpassProvider
from .ws import ws_router, ws_info_router
from .ws.manager import connection_manager


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    settings = get_settings()
    configure_logging(settings.log_level)

    # Initialize metrics
    metrics_collector.initialize()

    # Start camera scheduler
    await camera_scheduler.start()

    # Start WebSocket connection manager
    from .ws.config import get_ws_config
    ws_config = get_ws_config()
    if ws_config.enabled:
        await connection_manager.start()

    # One shared HTTP client for all outbound provider calls.
    async with httpx.AsyncClient(
        timeout=settings.http_timeout_s,
        headers={"User-Agent": settings.user_agent},
    ) as client:
        app.state.http_client = client
        app.state.osm_provider = OverpassProvider(client, settings)
        app.state.elevation_provider = OpenTopoDataProvider(client, settings)
        yield

    # Shutdown camera scheduler
    await camera_scheduler.stop()

    # Stop WebSocket connection manager
    ws_config = get_ws_config()
    if ws_config.enabled:
        await connection_manager.stop()

    # Cleanup metrics
    # metrics_collector is long-lived, no explicit shutdown needed


def create_app() -> FastAPI:
    settings = get_settings()
    app = FastAPI(
        title=settings.app_name,
        version=settings.version,
        description=(
            "GeoMind turns open geospatial data into structured location "
            "intelligence: building density, road density, vegetation coverage, "
            "water proximity, elevation, and wildfire detection — "
            "with full data provenance, authentication, and real-time streaming."
        ),
        lifespan=lifespan,
    )

    app.include_router(health.router)
    app.include_router(location.router)
    app.include_router(camera_router, prefix="/api/v1")
    app.include_router(auth_router, prefix="/api/v1")
    app.include_router(ws_router, prefix="/api/v1")
    app.include_router(ws_info_router, prefix="/api/v1")

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

    @app.get("/metrics", tags=["metrics"], summary="Prometheus metrics")
    async def metrics() -> Response:
        metrics_data = metrics_collector.get_metrics_output()
        return Response(content=metrics_data, media_type="text/plain; version=0.0.4; charset=utf-8")

    @app.middleware("http")
    async def metrics_middleware(request: Request, call_next):
        import time
        start = time.perf_counter()
        response = await call_next(request)
        duration = time.perf_counter() - start
        metrics_collector.track_http_request(
            method=request.method,
            endpoint=request.url.path,
            status=response.status_code,
            duration=duration,
        )
        return response

    return app


app = create_app()
