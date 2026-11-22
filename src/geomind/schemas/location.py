"""Pydantic schemas for the Location Intelligence API."""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field


class BoundingBox(BaseModel):
    min_lat: float
    min_lon: float
    max_lat: float
    max_lon: float


class QueryInfo(BaseModel):
    latitude: float
    longitude: float
    radius_m: float
    area_m2: float
    bbox: BoundingBox


class BuildingDensity(BaseModel):
    value: float | None = Field(
        None, description="Building footprint area divided by analysis area (0-1)."
    )
    unit: Literal["ratio"] = "ratio"
    building_count: int
    footprint_area_m2: float
    definition: str = "Building footprint area / analysis area"


class RoadDensity(BaseModel):
    value: float | None = Field(
        None, description="Total road length per square kilometre of analysis area."
    )
    unit: Literal["km/km2"] = "km/km2"
    road_length_m: float
    road_count: int
    definition: str = "Total road length / analysis area"


class VegetationCoverage(BaseModel):
    value: float | None = Field(
        None, description="Vegetated land area divided by analysis area (0-1)."
    )
    unit: Literal["ratio"] = "ratio"
    vegetation_area_m2: float
    feature_count: int
    definition: str = (
        "Vegetated land area / analysis area "
        "(landuse=forest|meadow|grass|farmland|orchard|vineyard, "
        "natural=wood|scrub|grassland|heath|wetland, leisure=park|garden)"
    )


class WaterFeatureSummary(BaseModel):
    osm_type: str
    osm_id: int
    name: str | None = None
    distance_m: float


class WaterProximity(BaseModel):
    nearest_distance_m: float | None = Field(
        None, description="Distance to the nearest water feature, in metres."
    )
    found_within_m: float
    water_coverage_ratio: float | None = Field(
        None, description="Water area inside the analysis circle / analysis area (0-1)."
    )
    nearest_feature: WaterFeatureSummary | None = None
    definition: str = (
        "Distance from the query point to the nearest water feature within the "
        "search buffer, plus water coverage inside the analysis area"
    )


class ElevationInfo(BaseModel):
    value: float | None = Field(None, description="Elevation at the query point, in metres.")
    unit: Literal["m"] = "m"
    min_m: float | None = None
    max_m: float | None = None
    mean_m: float | None = None
    samples: int = 0
    dataset: str | None = None


class Features(BaseModel):
    building_density: BuildingDensity
    road_density: RoadDensity
    vegetation_coverage: VegetationCoverage
    water_proximity: WaterProximity
    elevation: ElevationInfo


class DataSource(BaseModel):
    name: str
    dataset: str
    provider: str
    url: str
    license: str
    attribution: str


class CacheInfo(BaseModel):
    hit: bool
    fetched_at: datetime | None = None
    ttl_s: int


class OsmQueryInfo(BaseModel):
    endpoint: str | None = None
    feature_count: int = 0
    fetched_at: datetime | None = None


class Provenance(BaseModel):
    generated_at: datetime
    sources: list[DataSource]
    cache: CacheInfo
    osm: OsmQueryInfo


class LocationIntelligenceResponse(BaseModel):
    query: QueryInfo
    features: Features
    provenance: Provenance
