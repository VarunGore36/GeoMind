"""
GeoMind Python Client Library

A simple, typed client for the GeoMind Location Intelligence API.

Usage:
    from geomind_client import GeomindClient
    import os

    client = GeomindClient(
        base_url="https://api.geomind.example.com",
        api_key=os.getenv("GEOMIND_API_KEY")
    )

    # Location intelligence
    result = client.location_intelligence(lat=52.52, lon=13.405, radius=500)
    print(f"Building density: {result.features.building_density.value}")

    # Camera management
    cameras = client.list_cameras()
    latest = client.get_latest_frame("camera_1")

    # WebSocket streaming
    async for detection in client.stream_detections():
        print(f"Detection: {detection.class_name} ({detection.confidence})")
"""

from .client import GeomindClient
from .models import (
    LocationIntelligenceResponse,
    Features,
    BuildingDensity,
    RoadDensity,
    VegetationCoverage,
    WaterProximity,
    ElevationInfo,
    Provenance,
    Camera,
    Frame,
    DetectionEvent,
    CameraStatus,
)

__version__ = "0.1.0"
__all__ = [
    "GeomindClient",
    "LocationIntelligenceResponse",
    "Features",
    "BuildingDensity",
    "RoadDensity",
    "VegetationCoverage",
    "WaterProximity",
    "ElevationInfo",
    "Provenance",
    "Camera",
    "Frame",
    "DetectionEvent",
    "CameraStatus",
]