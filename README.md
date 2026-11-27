<div align="center">

# 🌍 GeoMind

**AI-powered geospatial intelligence platform.**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115+-009688.svg)](https://fastapi.tiangolo.com/)
[![PostGIS](https://img.shields.io/badge/PostGIS-3.4+-4B7BEC.svg)](https://postgis.net/)
[![Tests](https://img.shields.io/badge/tests-154%20passing-brightgreen.svg)](https://github.com/VarunGore36/GeoMind/actions)
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

### 3. Vision Engine — Wildfire Detection (`/api/v1/vision`)

Real-time smoke and fire detection from camera frames using ONNX Runtime.

| Feature | Details |
|---------|---------|
| **Model** | ONNX Runtime (YOLO-style) with custom wildfire classes (smoke=0, fire=1) |
| **Temporal Validation** | Requires N consecutive frames (default 3) with confidence threshold |
| **Detection States** | pending → confirmed / rejected |
| **Queue** | Async frame processing with configurable workers & batch size |
| **Storage** | SQLite/PostgreSQL for detection events & confirmed incidents |
| **API** | List events, latest per camera, real-time WebSocket streaming |

**Config (`vision.yaml`):**
```yaml
model:
  type: "onnx"           # dummy, yolo, wildfire, onnx, onnxruntime
  weights_path: "./models/wildfire_yolov8n.onnx"
  device: "cpu"          # cpu, cuda, mps
  confidence_threshold: 0.25
  iou_threshold: 0.45
  input_size: 640
  classes: [0, 1]        # smoke, fire
temporal:
  min_consecutive_frames: 3
  frame_confidence_threshold: 0.3
  max_frame_gap_seconds: 60.0
```

---

### 4. Authentication & Authorization (`/api/v1/auth`)

JWT + API Key authentication with Role-Based Access Control.

| Feature | Details |
|---------|---------|
| **JWT** | Access (15m) + Refresh (7d) tokens, HS256 |
| **API Keys** | Hashed storage, per-key rate limits, expiration |
| **Roles** | admin, user, viewer, service |
| **Rate Limiting** | Token bucket (per-minute + per-hour) by IP / API key / Bearer token |
| **Endpoints** | Login, refresh, logout, register, user management, API key CRUD |

---

### 5. Real-time WebSocket Streaming (`/ws/connect`)

Pub/Sub for live detection events and camera status.

| Channel | Events |
|---------|--------|
| `detections` | All detection events (pending + confirmed) |
| `detections:confirmed` | Only temporally validated detections |
| `detections:pending` | Pending detections awaiting confirmation |
| `cameras` | Camera status updates (active, error, fps) |
| `system` | System notifications |

**Features:** JWT/API key auth, channel subscriptions with filters (camera_id, class_name, min_confidence), heartbeat/ping-pong, auto-reconnect.

---

### 6. Monitoring & Observability

| Component | Details |
|-----------|---------|
| **Prometheus** | `/metrics` endpoint with 50+ custom metrics |
| **Grafana** | Pre-built dashboard (16 panels) |
| **Alerting** | 25+ alert rules (API, DB, cache, vision, auth, business) |
| **Recording Rules** | Pre-computed percentiles, rates, ratios |
| **Health** | `/health` (liveness), `/health/ready` (readiness) |

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
# Dev server (hot reload) at http://localhost:8001
```

### Local Development

```bash
# Install with all extras
pip install -e ".[camera,vision,auth,dev]"

# Copy config templates
cp cameras.yaml.example cameras.yaml
cp vision.yaml.example vision.yaml
cp auth.yaml.example auth.yaml
cp metrics.yaml.example metrics.yaml

# Edit configs with your settings
# ...

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

### Vision Engine

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/api/v1/vision/events` | List detection events (paginated, filterable) |
| `GET` | `/api/v1/vision/events/latest/{camera_id}` | Latest event for camera |
| `GET` | `/api/v1/vision/health` | Vision engine health |

### Authentication

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/api/v1/auth/login` | Get access + refresh tokens |
| `POST` | `/api/v1/auth/refresh` | Refresh access token |
| `POST` | `/api/v1/auth/logout` | Invalidate tokens |
| `POST` | `/api/v1/auth/register` | Register new user |
| `GET` | `/api/v1/auth/users` | List users (admin) |
| `GET` | `/api/v1/auth/users/{id}` | Get user |
| `PATCH` | `/api/v1/auth/users/{id}` | Update user |
| `DELETE` | `/api/v1/auth/users/{id}` | Delete user |
| `POST` | `/api/v1/auth/api-keys` | Create API key |
| `GET` | `/api/v1/auth/api-keys` | List API keys |
| `DELETE` | `/api/v1/auth/api-keys/{id}` | Revoke API key |

### WebSocket

| Method | Endpoint | Description |
|--------|----------|-------------|
| `WS` | `/ws/connect` | WebSocket connection endpoint |
| `GET` | `/api/v1/ws/stats` | Connection statistics |
| `GET` | `/api/v1/ws/channels` | Active channels + subscriber counts |
| `GET` | `/api/v1/ws/connections` | List connections (admin) |

### Health & Meta

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/health` | Liveness probe |
| `GET` | `/health/ready` | Readiness (DB + PostGIS) |
| `GET` | `/` | Service metadata |
| `GET` | `/metrics` | Prometheus metrics |

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
├──────────────────────────────┼──────────────────────────────────┤
│  Vision Engine               │  WebSocket / Auth / Monitoring   │
│  ─────────────────────────   │  ─────────────────────────       │
│  ONNX Runtime (YOLO)         │  JWT + API Keys (RBAC)           │
│  Temporal Validation (N=3)   │  Token Bucket Rate Limiting      │
│  SQLite/Postgres Events      │  WebSocket Pub/Sub               │
│  ─────────────────────────   │  Prometheus /metrics             │
│  FastAPI /v1/vision          │  Grafana Dashboards              │
│                              │  Alerting Rules                  │
└──────────────────────────────┴──────────────────────────────────┘
```

- **Async throughout** — FastAPI + asyncpg + SQLAlchemy 2.0
- **PostGIS** — spatial indexes, UTM projection for accurate metrics
- **Caching** — 24h TTL for OSM coverage areas, Redis for camera frames
- **Deduplication** — 64-bit perceptual hash, ≥95% similarity skips save
- **Multi-stage Docker** — minimal runtime image (~150MB), dev image with hot reload

---

## ⚙️ Configuration

All settings via environment variables (`GEOMIND_` prefix) or YAML:

| File | Purpose |
|------|---------|
| `.env` | Runtime overrides (DB URL, API keys, log level) |
| `cameras.yaml` | Camera feeds, storage, database, Redis |
| `vision.yaml` | Vision engine, model, temporal validation |
| `auth.yaml` | JWT, API keys, rate limiting, RBAC |
| `metrics.yaml` | Prometheus metrics config |
| `pyproject.toml` | Dependencies, tool config (ruff, mypy, pytest) |

Key env vars:
```bash
GEOMIND_DATABASE_URL=postgresql+asyncpg://user:pass@host/db
GEOMIND_CAM_REDIS__URL=redis://localhost:6379/0
GEOMIND_VISION_MODEL__TYPE=onnx
GEOMIND_VISION_MODEL__WEIGHTS_PATH=./models/wildfire_yolov8n.onnx
GEOMIND_AUTH_JWT_SECRET=your-secret
GEOMIND_LOG_LEVEL=INFO
```

---

## 🧪 Testing

```bash
# All tests (154 passing)
make test

# Camera tests only
make test-cam

# Vision tests only
make test-vision

# With coverage
pytest --cov=src --cov-report=html

# Integration tests (requires PostGIS)
GEOMIND_TEST_DATABASE_URL=postgresql+asyncpg://... pytest tests/integration/
```

**Test breakdown:**
- 36 unit/integration tests for Location Intelligence
- 18 camera pipeline tests (config, storage, fetcher, DB, API, scheduler)
- Vision engine tests (model, temporal validation, database, API)
- Auth tests (JWT, API keys, RBAC, rate limiting)
- WebSocket tests (manager, models, API)
- Metrics tests (collector, middleware)

---

## 🐳 Docker

```yaml
# docker-compose.yml services:
#   db      - postgis/postgis:16-3.4 (PostGIS 3.4)
#   redis   - redis:7-alpine (for camera caching)
#   api     - geomind:latest (FastAPI + Uvicorn, 4 workers)
#   api-dev - geomind:dev (hot reload on port 8001)
```

```bash
docker-compose up -d          # Start production
docker-compose up -d api-dev  # Start dev server
docker-compose logs -f api    # View logs
docker-compose down           # Stop
```

**Multi-stage build targets:**
- `runtime` — minimal production image (~150MB)
- `dev` — development image with pytest, ruff, mypy
- `builder` — dependency compilation stage

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
│   ├── vision/               # Vision engine module
│   │   ├── config.py         # Model, temporal, queue config
│   │   ├── models.py         # Detection events, frames
│   │   ├── models_impl.py    # ONNX, YOLO, Wildfire, Dummy models
│   │   ├── interfaces.py     # Model protocols
│   │   ├── engine.py         # Async inference engine
│   │   ├── temporal.py       # Temporal validation logic
│   │   ├── database.py       # Event storage
│   │   ├── api.py            # FastAPI routes
│   │   └── README.md         # Module docs
│   ├── auth/                 # Authentication module
│   │   ├── models.py         # User, API key, role models
│   │   ├── database.py       # User/key storage
│   │   ├── utils.py          # JWT, hashing, tokens
│   │   ├── dependencies.py   # FastAPI auth deps
│   │   ├── rate_limit.py     # Token bucket rate limiting
│   │   ├── api.py            # Auth routes
│   │   └── README.md         # Module docs
│   ├── ws/                   # WebSocket module
│   │   ├── models.py         # WS message models
│   │   ├── manager.py        # Connection manager + pub/sub
│   │   ├── api.py            # WS endpoint + publishing
│   │   ├── config.py         # WS config
│   │   └── README.md         # Module docs
│   ├── metrics/              # Prometheus metrics
│   │   ├── collector.py      # Custom metrics
│   │   ├── middleware.py     # Request metrics middleware
│   │   └── README.md         # Module docs
│   ├── core/                 # Config, errors, logging
│   ├── db/                   # Base, session, engine proxy
│   ├── providers/            # Overpass, Elevation
│   ├── repositories/         # PostGIS metric queries
│   ├── services/             # Ingestion, metrics, orchestration
│   ├── api/                  # Health, Location Intelligence routes
│   └── main.py               # FastAPI app factory
├── geomind_client/           # Python client library
│   ├── client.py             # Sync + Async clients
│   ├── models.py             # Pydantic response models
│   ├── pyproject.toml        # Package config
│   └── README.md             # Client docs
├── scripts/                  # Utility scripts
│   ├── download_model.py     # Model download + placeholder
│   ├── train_wildfire.py     # Training pipeline scaffold
│   ├── validate_model.py     # ONNX model validation
│   ├── satellite_ingest.py   # Sentinel-2/Landsat ingestion
│   └── generate_openapi.py   # OpenAPI spec generation
├── examples/                 # Usage examples
│   ├── ws_client.py          # Python WS client
│   ├── ws_client.js          # Node.js/Browser WS client
│   └── ws_client.html        # Browser demo
├── monitoring/               # Observability configs
│   ├── prometheus_alerts.yml        # 25+ alert rules
│   ├── prometheus_recording_rules.yml # Recording rules
│   └── grafana_dashboard.json       # 16-panel dashboard
├── tests/                    # 154 tests (unit + integration)
├── alembic/                  # DB migrations
├── data/                     # Training data (gitignored)
├── models/                   # Model weights (gitignored)
├── frames/                   # Camera frames (gitignored)
├── cameras.yaml              # Camera config (gitignored)
├── vision.yaml               # Vision config (gitignored)
├── auth.yaml                 # Auth config (gitignored)
├── metrics.yaml              # Metrics config (gitignored)
├── docker-compose.yml
├── Dockerfile
├── Makefile
├── pyproject.toml
├── requirements.txt
├── CHANGELOG.md
└── LICENSE (MIT)
```

---

## 🛠️ Scripts & Examples

| Script | Purpose |
|--------|---------|
| `scripts/download_model.py` | Download/convert YOLO models, create ONNX placeholders |
| `scripts/train_wildfire.py` | Training pipeline scaffold (prepare, train, export, validate) |
| `scripts/validate_model.py` | Validate ONNX models + benchmark + Vision Engine integration test |
| `scripts/satellite_ingest.py` | Sentinel-2/Landsat search, download, indices (NDVI, NBR, NDWI...) |
| `scripts/generate_openapi.py` | Generate OpenAPI 3.1 spec (JSON/YAML) with filtering |
| `examples/ws_client.py` | Python async WebSocket client with auth, filters, reconnection |
| `examples/ws_client.js` | Universal JS/Node.js WebSocket client |
| `examples/ws_client.html` | Browser demo (vanilla JS, no deps) |

---

## 🤝 Contributing

```bash
# Format & lint
make format
make lint

# Run tests
make test

# Type check
make typecheck

# Submit PR
```

Code style: `ruff` (line-length 100), type hints required (`mypy`).
CI: GitHub Actions (lint, unit tests, integration tests, Docker build, security scan).

---

## 📄 License

MIT License — see [LICENSE](LICENSE).

---

## 🙋‍♂️ Author

**Varun Gore** — [@VarunGore36](https://github.com/VarunGore36)

---

<div align="center">

**Built with FastAPI, PostGIS, ONNX Runtime, and open geospatial data.**

</div>