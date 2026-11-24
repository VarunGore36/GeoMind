"""Vision Engine main entry point."""

from __future__ import annotations

import asyncio
import logging
import signal
from contextlib import asynccontextmanager
from typing import AsyncIterator

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from . import __version__
from .api import router as vision_router
from .config import get_vision_config
from ..core.config import get_settings
from ..core.errors import GeoMindError
from ..core.logging import configure_logging
from .engine import vision_engine

logger = logging.getLogger(__name__)


@asynccontextmanager
async def vision_lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Vision engine lifespan handler."""
    settings = get_settings()
    vision_config = get_vision_config()

    if vision_config.enabled:
        logger.info("Starting Vision Engine...")
        await vision_engine.start()
        logger.info("Vision Engine started")

        # Integrate with camera pipeline if available
        try:
            from ..cameras.scheduler import scheduler as camera_scheduler
            # Hook into camera pipeline frame processing
            await _integrate_with_camera_pipeline()
        except ImportError:
            logger.info("Camera pipeline not available, running standalone")

    try:
        yield
    finally:
        if vision_config.enabled:
            logger.info("Stopping Vision Engine...")
            await vision_engine.stop()
            logger.info("Vision Engine stopped")


async def _integrate_with_camera_pipeline() -> None:
    """Hook into the camera pipeline to process frames as they arrive."""
    # This would be implemented to hook into the camera scheduler
    # For now, frames are submitted via API or external triggers
    pass


def create_vision_app() -> FastAPI:
    """Create the Vision Engine FastAPI application."""
    vision_config = get_vision_config()
    settings = get_settings()

    app = FastAPI(
        title="GeoMind Vision Engine",
        version=__version__,
        description="Wildfire smoke detection from camera feeds",
        lifespan=vision_lifespan,
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(vision_router, prefix="/api/v1")

    @app.exception_handler(GeoMindError)
    async def _vision_error_handler(_, exc: GeoMindError):
        from fastapi.responses import JSONResponse
        return JSONResponse(status_code=exc.status_code, content=exc.to_dict())

    @app.get("/", tags=["meta"])
    async def root():
        return {
            "name": "GeoMind Vision Engine",
            "version": __version__,
            "docs": "/docs",
            "health": "/api/v1/detections/health",
        }

    return app


app = create_vision_app()


async def _run_frame_consumer() -> None:
    """Run the frame consumer loop for testing."""
    from ..cameras.storage import storage
    from .engine import vision_engine
    import cv2

    vision_config = get_vision_config()
    if not vision_config.enabled:
        return

    logger.info("Starting frame consumer loop...")
    await vision_engine.start()

    try:
        while True:
            # This would be replaced with actual frame consumption from camera pipeline
            await asyncio.sleep(1)
    except asyncio.CancelledError:
        pass
    finally:
        await vision_engine.stop()


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "src.geomind.vision.main:app",
        host="0.0.0.0",
        port=8001,
        reload=True,
    )