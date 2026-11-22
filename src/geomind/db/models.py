"""PostGIS-backed models.

A single ``osm_features`` table stores every normalised feature. Features are
tagged with a coarse ``feature_type`` (building/road/vegetation/water) so the
same ingestion pipeline can serve all metrics.
"""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import Any

from geoalchemy2 import Geometry
from sqlalchemy import (
    BigInteger,
    DateTime,
    Float,
    Integer,
    String,
    UniqueConstraint,
    func,
    text,
)
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column

from .base import Base


class OsmFeature(Base):
    __tablename__ = "osm_features"
    __table_args__ = (
        UniqueConstraint("osm_type", "osm_id", "feature_type", name="uq_osm_feature"),
    )

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    osm_type: Mapped[str] = mapped_column(String(16), nullable=False)
    osm_id: Mapped[int] = mapped_column(BigInteger, nullable=False)
    feature_type: Mapped[str] = mapped_column(String(32), nullable=False, index=True)
    name: Mapped[str | None] = mapped_column(String(512), nullable=True)
    tags: Mapped[dict[str, Any]] = mapped_column(
        JSONB, nullable=False, default=dict, server_default=text("'{}'::jsonb")
    )
    source: Mapped[str] = mapped_column(
        String(32), nullable=False, default="osm", server_default=text("'osm'")
    )
    fetched_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )
    geom: Mapped[Any] = mapped_column(
        Geometry(geometry_type="GEOMETRY", srid=4326, spatial_index=True), nullable=False
    )


class IngestionArea(Base):
    """Records a circular area that has been fetched from Overpass.

    Used to decide whether a request can be served from the local cache and to
    expose provenance (when and from where data was fetched).
    """

    __tablename__ = "ingestion_areas"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    center: Mapped[Any] = mapped_column(
        Geometry(geometry_type="POINT", srid=4326, spatial_index=True), nullable=False
    )
    radius_m: Mapped[float] = mapped_column(Float, nullable=False)
    feature_count: Mapped[int] = mapped_column(
        Integer, nullable=False, default=0, server_default=text("0")
    )
    osm_endpoint: Mapped[str | None] = mapped_column(String(256), nullable=True)
    fetched_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )
