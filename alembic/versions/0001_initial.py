"""Initial PostGIS schema.

Revision ID: 0001_initial
Revises:
Create Date: 2026-11-22

"""
from __future__ import annotations

from collections.abc import Sequence

import geoalchemy2
import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "0001_initial"
down_revision: str | None = None
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.execute("CREATE EXTENSION IF NOT EXISTS postgis")

    op.create_table(
        "osm_features",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True, nullable=False),
        sa.Column("osm_type", sa.String(length=16), nullable=False),
        sa.Column("osm_id", sa.BigInteger(), nullable=False),
        sa.Column("feature_type", sa.String(length=32), nullable=False),
        sa.Column("name", sa.String(length=512), nullable=True),
        sa.Column(
            "tags",
            postgresql.JSONB(astext_type=sa.Text()),
            nullable=False,
            server_default=sa.text("'{}'::jsonb"),
        ),
        sa.Column(
            "source", sa.String(length=32), nullable=False, server_default=sa.text("'osm'")
        ),
        sa.Column(
            "fetched_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("now()"),
        ),
        sa.Column(
            "geom",
            geoalchemy2.types.Geometry(
                geometry_type="GEOMETRY", srid=4326, spatial_index=False
            ),
            nullable=False,
        ),
        sa.UniqueConstraint("osm_type", "osm_id", "feature_type", name="uq_osm_feature"),
    )
    op.create_index("ix_osm_features_feature_type", "osm_features", ["feature_type"])
    op.create_index(
        "ix_osm_features_geom", "osm_features", ["geom"], postgresql_using="gist"
    )

    op.create_table(
        "ingestion_areas",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True, nullable=False),
        sa.Column(
            "center",
            geoalchemy2.types.Geometry(
                geometry_type="POINT", srid=4326, spatial_index=False
            ),
            nullable=False,
        ),
        sa.Column("radius_m", sa.Float(), nullable=False),
        sa.Column("feature_count", sa.Integer(), nullable=False, server_default=sa.text("0")),
        sa.Column("osm_endpoint", sa.String(length=256), nullable=True),
        sa.Column(
            "fetched_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("now()"),
        ),
    )
    op.create_index(
        "ix_ingestion_areas_center", "ingestion_areas", ["center"], postgresql_using="gist"
    )


def downgrade() -> None:
    op.drop_index("ix_ingestion_areas_center", table_name="ingestion_areas")
    op.drop_table("ingestion_areas")
    op.drop_index("ix_osm_features_geom", table_name="osm_features")
    op.drop_index("ix_osm_features_feature_type", table_name="osm_features")
    op.drop_table("osm_features")
