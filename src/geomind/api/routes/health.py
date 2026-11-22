"""Health and readiness endpoints."""

from __future__ import annotations

from fastapi import APIRouter, Depends
from fastapi.responses import JSONResponse
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from ... import __version__
from ...db.session import get_session

router = APIRouter(tags=["health"])


@router.get("/health", summary="Liveness probe")
async def health() -> dict[str, str]:
    return {"status": "ok", "version": __version__}


@router.get("/health/ready", summary="Readiness probe (database + PostGIS)")
async def readiness(session: AsyncSession = Depends(get_session)) -> JSONResponse:
    try:
        await session.execute(text("SELECT 1"))
        postgis_version = (await session.execute(text("SELECT PostGIS_Version()"))).scalar_one()
    except Exception as exc:  # noqa: BLE001 - report any readiness failure
        return JSONResponse(
            status_code=503,
            content={
                "status": "not_ready",
                "database": "unavailable",
                "detail": str(exc),
            },
        )

    return JSONResponse(
        status_code=200,
        content={
            "status": "ready",
            "database": "ok",
            "postgis": postgis_version,
        },
    )
