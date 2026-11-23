# Camera Pipeline

Multi-feed camera ingestion pipeline for GeoMind.

## Features

- **Multi-feed support**: Configure multiple camera feeds with different intervals
- **Perceptual deduplication**: pHash-based detection of visually identical frames
- **Periodic fetching**: APScheduler with configurable intervals and exponential backoff
- **Frame storage**: Organized by camera ID with timestamped filenames
- **Metadata persistence**: Async SQLAlchemy with PostgreSQL/SQLite support
- **REST API**: FastAPI endpoints for frame access and camera status
- **Health monitoring**: `/health` endpoint and standalone check script

## Configuration

Create `cameras.yaml` from `cameras.yaml.example`:

```yaml
feeds:
  - feed_url: "https://your-camera.com/image.jpg"
    camera_id: "my_camera"
    fetch_interval_seconds: 30
    timeout_seconds: 10
    max_retries: 3
    retry_delay_seconds: 5

storage:
  path: "./frames"
  max_frames_per_camera: 1000
  deduplicate: true
  similarity_threshold: 0.95

database:
  url: "postgresql+asyncpg://user:pass@host/db"

redis:
  url: "redis://localhost:6379/0"
  enabled: false
```

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/v1/cameras/health` | Health check with all camera statuses |
| GET | `/api/v1/cameras` | List all cameras |
| GET | `/api/v1/cameras/{id}` | Get camera status |
| GET | `/api/v1/cameras/{id}/latest` | Latest frame metadata |
| GET | `/api/v1/cameras/{id}/latest/image` | Latest frame as JPEG |
| POST | `/api/v1/cameras/{id}/fetch` | Trigger manual fetch |

## Running

### Docker Compose (recommended)
```bash
docker-compose up -d
```

### Local Development
```bash
# Install camera dependencies
pip install -e ".[camera]"

# Run with config
GEOMIND_CAM_CONFIG_PATH=cameras.yaml python -m src.geomind.main
```

## Health Check

```bash
# Basic check
python scripts/cam_health_check.py --url http://localhost:8000

# JSON output for monitoring systems
python scripts/cam_health_check.py --url http://localhost:8000 --json

# Exit with error code on failure
python scripts/cam_health_check.py --url http://localhost:8000 --exit-code
```

## Architecture

```
┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│   Camera    │────▶│  Scheduler  │────▶│   Fetcher   │
│   Feeds     │     │ (APScheduler)│    │ (HTTP +     │
└─────────────┘     └─────────────┘     │  Retry)     │
                                         └──────┬──────┘
                                                │
                              ┌─────────────────┼─────────────────┐
                              ▼                 ▼                 ▼
                        ┌─────────────┐  ┌─────────────┐  ┌─────────────┐
                        │  Storage    │  │  Database   │  │    API      │
                        │ (Files +    │  │ (SQLAlchemy)│  │ (FastAPI)   │
                        │  Dedup)     │  └─────────────┘  └─────────────┘
                        └─────────────┘
```

## Deduplication

Frames are deduplicated using perceptual hashing (pHash):
1. Compute 64-bit pHash of each new frame
2. Compare with previous frame's pHash for same camera
3. If similarity ≥ 95% (configurable), skip saving

## Testing

```bash
# Run camera tests
make test-cam

# Or directly
pytest tests/test_cameras.py tests/test_cameras_db.py tests/test_cameras_api.py tests/test_cameras_scheduler.py -v
```