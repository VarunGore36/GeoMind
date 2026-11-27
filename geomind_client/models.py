"""
Pydantic models for GeoMind API responses.

These models match the API response schemas for type-safe client usage.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Optional
from uuid import UUID

from pydantic import BaseModel, Field


# Location Intelligence Models

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
    value: Optional[float] = None
    building_count: int = 0
    footprint_area_m2: float = 0.0


class RoadDensity(BaseModel):
    value: Optional[float] = None
    road_length_m: float = 0.0
    road_count: int = 0


class VegetationCoverage(BaseModel):
    value: Optional[float] = None
    vegetation_area_m2: float = 0.0
    feature_count: int = 0


class WaterProximity(BaseModel):
    nearest_distance_m: Optional[float] = None
    found_within_m: float = 0.0
    water_coverage_ratio: Optional[float] = None
    nearest_feature: Optional[dict] = None


class ElevationInfo(BaseModel):
    value: Optional[float] = None
    min_m: Optional[float] = None
    max_m: Optional[float] = None
    mean_m: Optional[float] = None
    samples: int = 0
    dataset: str = "unknown"


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
    fetched_at: datetime
    ttl_s: int


class OsmQueryInfo(BaseModel):
    endpoint: str
    feature_count: int
    fetched_at: datetime


class Provenance(BaseModel):
    generated_at: datetime
    sources: list[DataSource]
    cache: CacheInfo
    osm: OsmQueryInfo


class LocationIntelligenceResponse(BaseModel):
    query: QueryInfo
    features: Features
    provenance: Provenance


# Camera Models

class Camera(BaseModel):
    camera_id: str
    feed_url: str
    fetch_interval_seconds: int
    enabled: bool = True
    last_fetch: Optional[datetime] = None
    last_error: Optional[str] = None
    total_frames: int = 0
    total_errors: int = 0


class Frame(BaseModel):
    id: UUID
    camera_id: str
    frame_number: int
    timestamp: datetime
    width: int
    height: int
    format: str
    size_bytes: int
    phash: Optional[str] = None
    is_duplicate: bool = False


# WebSocket Models

class DetectionEvent(BaseModel):
    type: str
    id: str
    timestamp: datetime
    channel: Optional[str] = None
    event_id: str
    camera_id: str
    frame_id: int
    class_name: str
    confidence: float
    bbox: dict[str, float]
    status: str = "pending"
    # Confirmed fields
    first_frame_id: Optional[int] = None
    last_frame_id: Optional[int] = None
    first_timestamp: Optional[datetime] = None
    last_timestamp: Optional[datetime] = None
    max_confidence: Optional[float] = None
    avg_confidence: Optional[float] = None
    frame_count: Optional[int] = None


class CameraStatus(BaseModel):
    type: str
    id: str
    timestamp: datetime
    camera_id: str
    status: str
    last_frame_id: Optional[int] = None
    last_frame_timestamp: Optional[datetime] = None
    total_frames: int = 0
    total_errors: int = 0
    fps: Optional[float] = None