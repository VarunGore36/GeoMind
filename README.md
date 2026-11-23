<div align="center">

# 🌍 GeoMind

**AI-powered geospatial intelligence platform.**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115+-009688.svg)](https://fastapi.tiangolo.com/)
[![PostGIS](https://img.shields.io/badge/PostGIS-3.4+-4B7BEC.svg)](https://postgis.net/)
[![Tests](https://img.shields.io/badge/tests-54%20passing-brightgreen.svg)](https://github.com/VarunGore36/GeoMind/actions)
[![Docker](https://img.shields.io/badge/Docker-ready-2496ED.svg)](https://www.docker.com/)

</div>

---

GeoMind transforms open geospatial data — satellite imagery, OpenStreetMap, elevation models, and live camera feeds — into **structured, queryable, actionable intelligence** through developer-friendly APIs.

## 🎯 Vision

Make geographic intelligence accessible through **simple APIs** and **scalable infrastructure** — no GIS expertise required.

---

## 🛰️ Core Modules

### 1. Location Intelligence API (`/api/v1/location-intelligence`)

Extract real-world metrics around any coordinate:

| Metric | Description | Unit |
|--------|-------------|------|
| **Building Density** | Footprint area ÷ analysis area | ratio (0–1) |
| **Road Density** | Total road length ÷ area | km/km² |
| **Vegetation Coverage** | Vegetated land ÷ area | ratio (0–1) |
| **Water Proximity** | Distance to nearest water + coverage | meters + ratio |
| **Elevation** | Center + ring samples (min/max/mean) | meters |

**Data sources:** OpenStreetMap (via Overpass API), SRTM elevation (OpenTopoData/Open-Elevation)  
**Provenance:** Every response includes source metadata, cache status, timestamps, and licenses (ODbL, public domain).

```bash
curl "http://localhost:8000/api/v1/location-intelligence?lat=52.52&lon=13.405&radius_m=500"
```

---

### 2. Camera Pipeline (`/api/v1/cameras`)

Ingest live MJPEG/JPEG feeds, deduplicate, and serve latest frames.

| Feature | Details |
|---------|---------|
| **Multi-feed** | Configure N cameras with individual intervals |
| **Deduplication** | Perceptual hash (pHash) — skips visually identical frames |
| **Scheduling** | APScheduler with per-feed intervals + exponential backoff retries |
| **Storage** | Files organized by `camera_id/YYYYMMDD_HHMMSS_NNNNNN.jpg` |
| **Persistence** | Async SQLAlchemy (PostgreSQL/SQLite) for frame metadata |
| **API** | Health, list, latest frame (JSON + JPEG), manual trigger |

**Config (`cameras.yaml`):**
```yaml
feeds:
  - feed_url: "https://images.webcams.travel/api/cam/1471069946/image.jpg"
    camera_id: "nyc_times_square"
    fetch_interval_seconds: 30
    timeout_seconds: 10
    max_retries: 3
```

---

## 🚀 Quick Start

### Docker (Recommended)

```bash
git clone https://github.com/VarunGore36/GeoMind
cd GeoMind

# Start PostGIS + Redis + API
docker-compose up -d

# API at http://localhost:8000
# Docs at http://localhost:8000/docs
```

### Local Development

```bash
# Install with camera extras
pip install -e ".[camera]"

# Copy config template
cp cameras.yaml.example cameras.yaml
# Edit cameras.yaml with your feeds

# Run
GEOMIND_CAM_CONFIG_PATH=cameras.yaml python -m src.geomind.main
```

---

## 📡 API Reference

### Location Intelligence

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/api/v1/location-intelligence` | Extract features around a point |
| Query params | `lat`, `lon`, `radius_m` (optional), `refresh` (optional) | WGS84 coords, radius in meters |

### Camera Pipeline

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/api/v1/cameras/health` | All cameras status + frame counts |
| `GET` | `/api/v1/cameras` | List configured cameras |
| `GET` | `/api/v1/cameras/{id}` | Single camera status |
| `GET` | `/api/v1/cameras/{id}/latest` | Latest frame metadata |
| `GET` | `/api/v1/cameras/{id}/latest/image` | Latest frame as JPEG |
| `POST` | `/api/v1/cameras/{id}/fetch` | Trigger immediate fetch |

### Health & Meta

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/health` | Liveness probe |
| `GET` | `/health/ready` | Readiness (DB + PostGIS) |
| `GET` | `/` | Service metadata |

---

## 📊 Example Response

**Location Intelligence:**
```json
{
  "query": { "latitude": 52.52, "longitude": 13.405, "radius_m": 500, "area_m2": 785398 },
  "features": {
    "building_density": { "value": 0.21, "building_count": 244, "footprint_area_m2": 164983 },
    "road_density": { "value": 57.3, "unit": "km/km2", "road_length_m": 44970 },
    "vegetation_coverage": { "value": 0.26, "vegetation_area_m2": 208049 },
    "water_proximity": { "nearest_distance_m": 125, "water_coverage_ratio": 0.056 },
    "elevation": { "value": 49, "min_m": 33, "max_m": 58, "mean_m": 44.2, "samples": 5 }
  },
  "provenance": {
    "sources": [
      { "name": "OpenStreetMap", "license": "ODbL-1.0" },
      { "name": "Elevation", "dataset": "srtm30m", "license": "Public domain" }
    ],
    "cache": { "hit": false, "ttl_s": 86400 }
  }
}
```

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                         GeoMind API                              │
├──────────────────────────────┬──────────────────────────────────┤
│  Location Intelligence       │  Camera Pipeline                 │
│  ─────────────────────────   │  ────────────────────            │
│  Overpass API (OSM)    ────▶ │  Camera Feeds (MJPEG/JPEG)       │
│  OpenTopoData (SRTM)  ────▶  │  ────────────────────            │
│  ─────────────────────────   │  APScheduler ◀──── Intervals      │
│  PostGIS (metrics)           │  ────────────────────            │
│  ─────────────────────────   │  Fetcher (HTTP + Retry)           │
│  FastAPI /v1/location-intel  │  ────────────────────            │
│                              │  pHash Dedup                      │
│                              │  ────────────────────            │
│                              │  Files + Postgres                 │
│                              │  ────────────────────            │
│                              │  FastAPI /v1/cameras              │
└──────────────────────────────┴──────────────────────────────────┘
```

- **Async throughout** — FastAPI + asyncpg + SQLAlchemy 2.0
- **PostGIS** — spatial indexes, UTM projection for accurate metrics
- **Caching** — 24h TTL for OSM coverage areas
- **Deduplication** — 64-bit perceptual hash, ≥95% similarity skips save

---

## ⚙️ Configuration

All settings via environment variables (`GEOMIND_` prefix) or YAML:

| File | Purpose |
|------|---------|
| `.env` | Runtime overrides (DB URL, API keys, log level) |
| `cameras.yaml` | Camera feeds, storage, database, Redis |
| `pyproject.toml` | Dependencies, tool config (ruff, mypy, pytest) |

Key env vars:
```bash
GEOMIND_DATABASE_URL=postgresql+asyncpg://user:pass@host/db
GEOMIND_CAM_REDIS__URL=redis://localhost:6379/0
GEOMIND_LOG_LEVEL=INFO
```

---

## 🧪 Testing

```bash
# All tests (54 passing)
make test

# Camera tests only
make test-cam

# With coverage
pytest --cov=src --cov-report=html
```

**Test breakdown:**
- 36 unit/integration tests for Location Intelligence
- 18 camera pipeline tests (config, storage, fetcher, DB, API, scheduler)

---

## 🐳 Docker

```yaml
# docker-compose.yml services:
#   db      - postgis/postgis:16-3.4 (PostGIS 3.4)
#   redis   - redis:7-alpine (for camera caching)
#   api     - geomind:latest (FastAPI + Uvicorn)
```

```bash
docker-compose up -d          # Start
docker-compose logs -f api    # View logs
docker-compose down           # Stop
```

---

## 📁 Project Structure

```
GeoMind/
├── src/geomind/
│   ├── cameras/              # Camera pipeline module
│   │   ├── config.py         # Pydantic config models
│   │   ├── models.py         # SQLAlchemy + Pydantic models
│   │   ├── storage.py        # File storage + pHash dedup
│   │   ├── fetcher.py        # HTTP fetcher with retries
│   │   ├── database.py       # Async SQLAlchemy CRUD
│   │   ├── scheduler.py      # APScheduler orchestration
│   │   ├── api.py            # FastAPI routes
│   │   └── README.md         # Module docs
│   ├── core/                 # Config, errors, logging
│   ├── db/                   # Base, session, models
│   ├── providers/            # Overpass, Elevation
│   ├── repositories/         # PostGIS metric queries
│   ├── services/             # Ingestion, metrics, orchestration
│   ├── api/                  # Health, Location Intelligence routes
│   └── main.py               # FastAPI app factory
├── tests/                    # 54 tests (unit + integration)
├── alembic/                  # DB migrations
├── scripts/                  # cam_health_check.py
├── static/                   # Web viewer (index.html)
├── cameras.yaml              # Camera config (gitignored)
├── cameras.yaml.example      # Template
├── docker-compose.yml
├── Dockerfile
├── Makefile
├── pyproject.toml
├── requirements.txt
├── CHANGELOG.md
└── LICENSE (MIT)
```

---

## 🤝 Contributing

```bash
# Format & lint
make format
make lint

# Run tests
make test

# Submit PR
```

Code style: `ruff` (line-length 100), type hints required (`mypy` strict).

---

## 📄 License

MIT License — see [LICENSE](LICENSE).

---

## 🙋‍♂️ Author

**Varun Gore** — [@VarunGore36](https://github.com/VarunGore36)

---

<div align="center">

**Built with FastAPI, PostGIS, and open geospatial data.**

</div>