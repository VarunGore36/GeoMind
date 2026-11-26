"""Rate limiting middleware for GeoMind."""

from __future__ import annotations

import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Optional

from fastapi import HTTPException, Request, Response
from starlette.middleware.base import BaseHTTPMiddleware

from .config import get_settings


@dataclass
class RateLimitConfig:
    """Configuration for rate limiting."""

    enabled: bool = True
    requests_per_minute: int = 60
    requests_per_hour: int = 1000
    burst_allowance: int = 10
    # Paths that are exempt from rate limiting
    exempt_paths: list[str] = field(default_factory=lambda: [
        "/health",
        "/health/ready",
        "/metrics",
        "/docs",
        "/openapi.json",
        "/redoc",
    ])


class TokenBucket:
    """Token bucket algorithm for rate limiting."""

    def __init__(self, capacity: int, refill_rate: float):
        self.capacity = capacity
        self.tokens = float(capacity)
        self.refill_rate = refill_rate  # tokens per second
        self.last_refill = time.time()

    def consume(self, tokens: int = 1) -> bool:
        """Try to consume tokens. Returns True if successful."""
        self._refill()
        if self.tokens >= tokens:
            self.tokens -= tokens
            return True
        return False

    def _refill(self) -> None:
        now = time.time()
        elapsed = now - self.last_refill
        new_tokens = elapsed * self.refill_rate
        self.tokens = min(self.capacity, self.tokens + new_tokens)
        self.last_refill = now


class RateLimiter:
    """Rate limiter using token bucket algorithm with in-memory storage."""

    def __init__(self, config: RateLimitConfig):
        self.config = config
        self.buckets: dict[str, TokenBucket] = defaultdict(
            lambda: TokenBucket(
                capacity=config.requests_per_minute + config.burst_allowance,
                refill_rate=config.requests_per_minute / 60.0,
            )
        )
        self.hourly_buckets: dict[str, TokenBucket] = defaultdict(
            lambda: TokenBucket(
                capacity=config.requests_per_hour + config.burst_allowance,
                refill_rate=config.requests_per_hour / 3600.0,
            )
        )

    def is_allowed(self, key: str) -> tuple[bool, dict[str, int]]:
        """Check if request is allowed. Returns (allowed, headers)."""
        if not self.config.enabled:
            return True, {}

        minute_bucket = self.buckets[key]
        hour_bucket = self.hourly_buckets[key]

        minute_allowed = minute_bucket.consume()
        hour_allowed = hour_bucket.consume()

        allowed = minute_allowed and hour_allowed

        # Calculate remaining requests
        minute_remaining = int(minute_bucket.tokens)
        hour_remaining = int(hour_bucket.tokens)

        headers = {
            "X-RateLimit-Limit-Minute": str(self.config.requests_per_minute + self.config.burst_allowance),
            "X-RateLimit-Remaining-Minute": str(max(0, minute_remaining)),
            "X-RateLimit-Limit-Hour": str(self.config.requests_per_hour + self.config.burst_allowance),
            "X-RateLimit-Remaining-Hour": str(max(0, hour_remaining)),
        }

        if not allowed:
            # Calculate retry-after
            minute_reset = (minute_bucket.capacity - minute_bucket.tokens) / minute_bucket.refill_rate if minute_bucket.refill_rate > 0 else 60
            hour_reset = (hour_bucket.capacity - hour_bucket.tokens) / hour_bucket.refill_rate if hour_bucket.refill_rate > 0 else 3600
            retry_after = int(max(minute_reset, hour_reset))
            headers["Retry-After"] = str(retry_after)

        return allowed, headers

    def get_stats(self, key: str) -> dict:
        """Get rate limit stats for a key."""
        minute_bucket = self.buckets[key]
        hour_bucket = self.hourly_buckets[key]

        return {
            "minute_limit": self.config.requests_per_minute + self.config.burst_allowance,
            "minute_remaining": int(minute_bucket.tokens),
            "hour_limit": self.config.requests_per_hour + self.config.burst_allowance,
            "hour_remaining": int(hour_bucket.tokens),
        }


# Global rate limiter instance
_rate_limiter: Optional[RateLimiter] = None


def get_rate_limiter() -> RateLimiter:
    """Get or create the global rate limiter instance."""
    global _rate_limiter
    if _rate_limiter is None:
        settings = get_settings()
        config = RateLimitConfig(
            enabled=getattr(settings, "RATE_LIMIT_ENABLED", True),
            requests_per_minute=getattr(settings, "RATE_LIMIT_PER_MINUTE", 60),
            requests_per_hour=getattr(settings, "RATE_LIMIT_PER_HOUR", 1000),
            burst_allowance=getattr(settings, "RATE_LIMIT_BURST", 10),
        )
        _rate_limiter = RateLimiter(config)
    return _rate_limiter


class RateLimitMiddleware(BaseHTTPMiddleware):
    """FastAPI middleware for rate limiting."""

    def __init__(self, app, rate_limiter: Optional[RateLimiter] = None):
        super().__init__(app)
        self.rate_limiter = rate_limiter or get_rate_limiter()

    async def dispatch(self, request: Request, call_next):
        # Check if path is exempt
        if any(request.url.path.startswith(path) for path in self.rate_limiter.config.exempt_paths):
            return await call_next(request)

        # Get client identifier (IP address or API key)
        client_id = self._get_client_id(request)

        # Check rate limit
        allowed, headers = self.rate_limiter.is_allowed(client_id)

        if not allowed:
            return Response(
                content='{"detail": "Rate limit exceeded"}',
                status_code=429,
                headers=headers,
                media_type="application/json",
            )

        response = await call_next(request)

        # Add rate limit headers to response
        for key, value in headers.items():
            response.headers[key] = value

        return response

    def _get_client_id(self, request: Request) -> str:
        """Extract client identifier from request."""
        # Check for API key
        api_key = request.headers.get("X-API-Key") or request.query_params.get("api_key")
        if api_key:
            return f"apikey:{api_key}"

        # Check for JWT token
        auth_header = request.headers.get("Authorization")
        if auth_header and auth_header.startswith("Bearer "):
            return f"bearer:{auth_header[7:20]}"  # Truncate token for privacy

        # Fall back to IP address
        forwarded = request.headers.get("X-Forwarded-For")
        if forwarded:
            ip = forwarded.split(",")[0].strip()
        else:
            ip = request.client.host if request.client else "unknown"
        return f"ip:{ip}"