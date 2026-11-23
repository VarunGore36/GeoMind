# Changelog

All notable changes to GeoMind will be documented in this file.

## [0.2.0] - 2026-11-24

### Added
- **Camera Pipeline Module** (`src/geomind/cameras/`)
  - Multi-feed camera ingestion with configurable intervals
  - Perceptual hash (pHash) deduplication to skip visually identical frames
  - Async SQLAlchemy database layer for frame metadata persistence
  - APScheduler-based periodic fetching with exponential backoff retries
  - FastAPI REST API for frame access and camera status monitoring
  - Frame storage organized by camera ID with timestamped filenames
  - Health check endpoint and monitoring script
  - Unit tests for config, storage, fetcher, database, scheduler, and API

### Changed
- Updated FastAPI app lifespan to start/stop camera scheduler
- Added Redis service to docker-compose for optional caching
- Added camera configuration via `cameras.yaml`

### Dependencies
- Added camera pipeline optional dependencies: apscheduler, opencv-python-headless, numpy, pillow, redis, pyyaml, aiosqlite

## [0.1.0] - 2026-11-22

### Added
- Location Intelligence API with FastAPI and PostGIS
- OpenStreetMap data via Overpass API with failover
- Elevation data via OpenTopoData with Open-Elevation fallback
- Building density, road density, vegetation coverage, water proximity metrics
- Full data provenance in API responses
- Docker and docker-compose support
- Alembic migrations for PostGIS schema
- 40 passing unit and integration tests