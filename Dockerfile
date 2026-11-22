# syntax=docker/dockerfile:1

FROM python:3.12-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# Install the package and its runtime dependencies first for better layer caching.
COPY pyproject.toml README.md LICENSE ./
COPY src ./src
RUN pip install .

# Runtime assets needed to migrate/start the app.
COPY alembic.ini ./
COPY alembic ./alembic

EXPOSE 8000

# Apply migrations, then serve.
CMD ["sh", "-c", "alembic upgrade head && uvicorn geomind.main:app --host 0.0.0.0 --port 8000"]
