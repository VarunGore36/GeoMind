# syntax=docker/dockerfile:1.4

# ==============================================================================
# Base stage - shared dependencies
# ==============================================================================
FROM python:3.12-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONPATH=/app/src

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    libpq-dev \
    libgdal-dev \
    gdal-bin \
    libgeos-dev \
    libproj-dev \
    && rm -rf /var/lib/apt/lists/*

# ==============================================================================
# Builder stage - install dependencies
# ==============================================================================
FROM base AS builder

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy dependency files first for better caching
COPY pyproject.toml README.md LICENSE requirements.txt ./

# Install Python dependencies
RUN pip install --no-deps . && \
    pip list --format=freeze > /app/requirements-lock.txt

# ==============================================================================
# Runtime stage - minimal runtime image
# ==============================================================================
FROM base AS runtime

# Copy installed packages from builder
COPY --from=builder /usr/local/lib/python3.12/site-packages /usr/local/lib/python3.12/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Copy application code
COPY src ./src
COPY alembic.ini ./
COPY alembic ./alembic

# Copy configuration files
COPY cameras.yaml.example ./cameras.yaml
COPY vision.yaml.example ./vision.yaml
COPY auth.yaml.example ./auth.yaml
COPY metrics.yaml.example ./metrics.yaml

# Copy models directory (placeholder model)
COPY models ./models

# Create non-root user
RUN useradd --create-home --shell /bin/bash appuser && \
    chown -R appuser:appuser /app
USER appuser

EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=10s --retries=3 \
    CMD python -c "import httpx; httpx.get('http://localhost:8000/health', timeout=5).raise_for_status()" || exit 1

# Run migrations and start server
CMD ["sh", "-c", "alembic upgrade head && uvicorn geomind.main:app --host 0.0.0.0 --port 8000 --workers 4"]

# ==============================================================================
# Development stage - with dev tools
# ==============================================================================
FROM builder AS dev

# Install dev dependencies
RUN pip install pytest pytest-asyncio pytest-cov httpx respx ruff mypy

# Copy all source files
COPY . .

# Create non-root user
RUN useradd --create-home --shell /bin/bash appuser && \
    chown -R appuser:appuser /app
USER appuser

EXPOSE 8000

CMD ["uvicorn", "geomind.main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]