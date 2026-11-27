# GeoMind Python Client

Official Python client library for the [GeoMind Location Intelligence API](https://github.com/VarunGore36/GeoMind).

## Installation

```bash
pip install geomind-client
```

Or install with async support:

```bash
pip install geomind-client[async]
```

## Quick Start

```python
from geomind_client import GeomindClient
import os

# Initialize client (use environment variables for secrets)
client = GeomindClient(
    base_url="http://localhost:8000",
    api_key=os.getenv("GEOMIND_API_KEY")  # or token=os.getenv("GEOMIND_TOKEN")
)

# Get location intelligence
result = client.location_intelligence(
    lat=52.52,      # Berlin latitude
    lon=13.405,     # Berlin longitude
    radius=500      # 500 meter radius
)

print(f"Building density: {result.features.building_density.value:.2%}")
print(f"Road density: {result.features.road_density.value:.2f} km/km²")
print(f"Vegetation coverage: {result.features.vegetation_coverage.value:.2%}")
print(f"Nearest water: {result.features.water_proximity.nearest_distance_m:.0f}m")
print(f"Elevation: {result.features.elevation.value:.1f}m")

# Access provenance
print(f"Data sources: {[s.name for s in result.provenance.sources]}")
print(f"Cache hit: {result.provenance.cache.hit}")
```

## Async Usage

```python
import asyncio
import os
from geomind_client import AsyncGeomindClient

async def main():
    async with AsyncGeomindClient(
        base_url="http://localhost:8000",
        api_key=os.getenv("GEOMIND_API_KEY")
    ) as client:
        result = await client.location_intelligence(lat=52.52, lon=13.405, radius=500)
        print(f"Building count: {result.features.building_density.building_count}")

asyncio.run(main())
```

## Real-time WebSocket Streaming

```python
import asyncio
import os
from geomind_client import GeomindClient

async def stream_detections():
    client = GeomindClient(
        base_url="http://localhost:8000",
        api_key=os.getenv("GEOMIND_API_KEY")
    )

    # Stream fire/smoke detections
    async for event in client.stream_detections(
        channels=["detections:confirmed"],  # Only confirmed detections
        filters={"class_name": "fire", "min_confidence": 0.7}
    ):
        print(f"🔥 FIRE: camera={event.camera_id} conf={event.confidence:.2f} "
              f"bbox={event.bbox} frame={event.frame_id}")

asyncio.run(stream_detections())
```

## Camera Management

```python
# List cameras
cameras = client.list_cameras()
for cam in cameras:
    print(f"{cam.camera_id}: {cam.feed_url} ({cam.total_frames} frames)")

# Get latest frame
frame = client.get_latest_frame("camera_1")
print(f"Frame {frame.frame_number} at {frame.timestamp}")

# Get frame image
image_bytes = client.get_latest_frame_image("camera_1")
with open("latest.jpg", "wb") as f:
    f.write(image_bytes)

# Trigger manual fetch
client.trigger_camera_fetch("camera_1")
```

## Authentication

The client supports two authentication methods (use environment variables for secrets):

1. **API Key** (recommended for services):
   ```python
   import os
   client = GeomindClient(base_url="...", api_key=os.getenv("GEOMIND_API_KEY"))
   ```

2. **JWT Token** (for user sessions):
   ```python
   import os
   client = GeomindClient(base_url="...", token=os.getenv("GEOMIND_TOKEN"))
   ```

## Error Handling

```python
from geomind_client import GeomindClient
import httpx

client = GeomindClient(base_url="http://localhost:8000", api_key="key")

try:
    result = client.location_intelligence(lat=52.52, lon=13.405, radius=500)
except httpx.HTTPStatusError as e:
    if e.response.status_code == 401:
        print("Authentication failed")
    elif e.response.status_code == 429:
        print("Rate limited")
    elif e.response.status_code == 503:
        print("Service unavailable")
    else:
        print(f"API error: {e.response.status_code}")
except httpx.RequestError as e:
    print(f"Network error: {e}")
```

## Configuration

```python
client = GeomindClient(
    base_url="https://api.geomind.example.com",
    api_key="key",
    timeout=60.0,        # Request timeout (seconds)
    verify_ssl=True,     # Verify SSL certificates
)
```

## Development

```bash
# Install in development mode
pip install -e ./geomind_client[dev]

# Run tests
pytest geomind_client/tests/

# Lint
ruff check geomind_client/

# Type check
mypy geomind_client/
```

## License

MIT License - see LICENSE file for details.