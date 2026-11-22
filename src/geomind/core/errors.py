"""Typed application errors mapped to HTTP responses by the API layer."""

from __future__ import annotations

from typing import Any


class GeoMindError(Exception):
    """Base class for expected, user-facing errors."""

    status_code: int = 500
    error_type: str = "internal_error"

    def __init__(self, message: str, *, details: dict[str, Any] | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.details = details or {}

    def to_dict(self) -> dict[str, Any]:
        return {
            "error": {
                "type": self.error_type,
                "message": self.message,
                "details": self.details,
            }
        }


class InvalidQueryError(GeoMindError):
    status_code = 422
    error_type = "invalid_query"


class DataSourceUnavailableError(GeoMindError):
    status_code = 503
    error_type = "data_source_unavailable"


class UpstreamError(GeoMindError):
    status_code = 502
    error_type = "upstream_error"
