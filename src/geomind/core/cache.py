"""Redis-based caching layer for GeoMind."""

from __future__ import annotations

import json
import logging
from contextlib import asynccontextmanager
from datetime import datetime, timedelta
from typing import Any, Optional, Union

import redis.asyncio as redis
from pydantic import BaseModel, Field

from .config import get_settings

logger = logging.getLogger(__name__)


class CacheConfig(BaseModel):
    """Cache configuration."""

    enabled: bool = True
    url: str = "redis://localhost:6379/1"
    default_ttl: int = Field(default=300, ge=1)  # 5 minutes default
    max_connections: int = 50
    socket_timeout: float = 5.0
    socket_connect_timeout: float = 5.0
    decode_responses: bool = True
    key_prefix: str = "geomind:"
    # Serialization
    default_serializer: str = "json"  # json, pickle, msgpack


class CacheEntry(BaseModel):
    """Cache entry with metadata."""

    key: str
    value: Any
    created_at: datetime = Field(default_factory=datetime.utcnow)
    expires_at: Optional[datetime] = None
    hits: int = 0
    size_bytes: int = 0


class CacheManager:
    """Async Redis cache manager with JSON serialization."""

    def __init__(self, config: Optional[CacheConfig] = None):
        self.config = config or CacheConfig()
        self._client: Optional[redis.Redis] = None
        self._pool: Optional[redis.ConnectionPool] = None

    async def connect(self) -> None:
        """Initialize Redis connection pool."""
        if self._pool is not None:
            return

        settings = get_settings()
        url = self.config.url if self.config.url != "redis://localhost:6379/1" else getattr(
            settings, "REDIS_URL", "redis://localhost:6379/1"
        )

        self._pool = redis.ConnectionPool.from_url(
            url,
            max_connections=self.config.max_connections,
            socket_timeout=self.config.socket_timeout,
            socket_connect_timeout=self.config.socket_connect_timeout,
            decode_responses=self.config.decode_responses,
        )
        self._client = redis.Redis(connection_pool=self._pool)

        # Test connection
        await self._client.ping()
        logger.info("Cache connected to %s", url)

    async def close(self) -> None:
        """Close Redis connections."""
        if self._client:
            await self._client.aclose()
        if self._pool:
            await self._pool.disconnect()
        self._client = None
        self._pool = None

    def _make_key(self, key: str) -> str:
        """Add prefix to key."""
        return f"{self.config.key_prefix}{key}"

    def _serialize(self, value: Any) -> str:
        """Serialize value to string."""
        if self.config.default_serializer == "json":
            return json.dumps(value, default=str)
        elif self.config.default_serializer == "pickle":
            import pickle
            return pickle.dumps(value).hex()
        return json.dumps(value, default=str)

    def _deserialize(self, data: str) -> Any:
        """Deserialize value from string."""
        if self.config.default_serializer == "json":
            return json.loads(data)
        elif self.config.default_serializer == "pickle":
            import pickle
            return pickle.loads(bytes.fromhex(data))
        return json.loads(data)

    async def get(self, key: str, default: Any = None) -> Any:
        """Get value from cache."""
        if not self.config.enabled or not self._client:
            return default

        key = self._make_key(key)
        try:
            data = await self._client.get(key)
            if data is None:
                return default

            # Update hit count
            await self._client.hincrby(f"{key}:meta", "hits", 1)
            return self._deserialize(data)
        except Exception as e:
            logger.warning("Cache get error for key %s: %s", key, e)
            return default

    async def set(
        self,
        key: str,
        value: Any,
        ttl: Optional[int] = None,
        nx: bool = False,
        xx: bool = False,
    ) -> bool:
        """Set value in cache with optional TTL."""
        if not self.config.enabled or not self._client:
            return False

        key = self._make_key(key)
        ttl = ttl or self.config.default_ttl
        serialized = self._serialize(value)

        try:
            if nx:
                result = await self._client.set(key, serialized, ex=ttl, nx=True)
            elif xx:
                result = await self._client.set(key, serialized, ex=ttl, xx=True)
            else:
                result = await self._client.set(key, serialized, ex=ttl)

            if result:
                # Store metadata
                meta_key = f"{key}:meta"
                await self._client.hset(meta_key, mapping={
                    "created_at": datetime.utcnow().isoformat(),
                    "expires_at": (datetime.utcnow() + timedelta(seconds=ttl)).isoformat() if ttl else "",
                    "hits": "0",
                    "size_bytes": str(len(serialized.encode())),
                })
                await self._client.expire(meta_key, ttl)
            return bool(result)
        except Exception as e:
            logger.warning("Cache set error for key %s: %s", key, e)
            return False

    async def delete(self, *keys: str) -> int:
        """Delete keys from cache."""
        if not self.config.enabled or not self._client:
            return 0

        keys = [self._make_key(k) for k in keys]
        meta_keys = [f"{k}:meta" for k in keys]
        all_keys = keys + meta_keys

        try:
            return await self._client.delete(*all_keys)
        except Exception as e:
            logger.warning("Cache delete error: %s", e)
            return 0

    async def exists(self, key: str) -> bool:
        """Check if key exists."""
        if not self.config.enabled or not self._client:
            return False

        key = self._make_key(key)
        try:
            return await self._client.exists(key) > 0
        except Exception:
            return False

    async def expire(self, key: str, ttl: int) -> bool:
        """Set expiration on key."""
        if not self.config.enabled or not self._client:
            return False

        key = self._make_key(key)
        try:
            return await self._client.expire(key, ttl)
        except Exception:
            return False

    async def ttl(self, key: str) -> int:
        """Get time to live for key."""
        if not self.config.enabled or not self._client:
            return -2

        key = self._make_key(key)
        try:
            return await self._client.ttl(key)
        except Exception:
            return -2

    async def increment(self, key: str, amount: int = 1) -> Optional[int]:
        """Increment a numeric value."""
        if not self.config.enabled or not self._client:
            return None

        key = self._make_key(key)
        try:
            return await self._client.incrby(key, amount)
        except Exception:
            return None

    async def get_many(self, keys: list[str]) -> dict[str, Any]:
        """Get multiple keys at once."""
        if not self.config.enabled or not self._client:
            return {}

        prefixed_keys = [self._make_key(k) for k in keys]
        try:
            values = await self._client.mget(prefixed_keys)
            result = {}
            for key, value in zip(keys, values):
                if value is not None:
                    result[key] = self._deserialize(value)
            return result
        except Exception as e:
            logger.warning("Cache get_many error: %s", e)
            return {}

    async def set_many(
        self,
        mapping: dict[str, Any],
        ttl: Optional[int] = None,
    ) -> int:
        """Set multiple keys at once."""
        if not self.config.enabled or not self._client:
            return 0

        ttl = ttl or self.config.default_ttl
        prefixed_mapping = {self._make_key(k): self._serialize(v) for k, v in mapping.items()}

        try:
            pipe = self._client.pipeline()
            for key, value in prefixed_mapping.items():
                pipe.set(key, value, ex=ttl)
                meta_key = f"{key}:meta"
                pipe.hset(meta_key, mapping={
                    "created_at": datetime.utcnow().isoformat(),
                    "expires_at": (datetime.utcnow() + timedelta(seconds=ttl)).isoformat(),
                    "hits": "0",
                    "size_bytes": str(len(value.encode())),
                })
                pipe.expire(meta_key, ttl)
            await pipe.execute()
            return len(mapping)
        except Exception as e:
            logger.warning("Cache set_many error: %s", e)
            return 0

    async def get_info(self, key: str) -> Optional[CacheEntry]:
        """Get cache entry metadata."""
        if not self.config.enabled or not self._client:
            return None

        key = self._make_key(key)
        try:
            meta = await self._client.hgetall(f"{key}:meta")
            if not meta:
                return None

            value = await self._client.get(key)
            return CacheEntry(
                key=key,
                value=self._deserialize(value) if value else None,
                created_at=datetime.fromisoformat(meta.get("created_at", datetime.utcnow().isoformat())),
                expires_at=datetime.fromisoformat(meta["expires_at"]) if meta.get("expires_at") else None,
                hits=int(meta.get("hits", 0)),
                size_bytes=int(meta.get("size_bytes", 0)),
            )
        except Exception:
            return None

    async def clear_pattern(self, pattern: str) -> int:
        """Delete all keys matching pattern."""
        if not self.config.enabled or not self._client:
            return 0

        pattern = self._make_key(pattern)
        try:
            keys = []
            async for key in self._client.scan_iter(match=pattern):
                keys.append(key)
                keys.append(f"{key}:meta")
            if keys:
                return await self._client.delete(*keys)
            return 0
        except Exception as e:
            logger.warning("Cache clear_pattern error: %s", e)
            return 0

    async def get_stats(self) -> dict:
        """Get cache statistics."""
        if not self.config.enabled or not self._client:
            return {"enabled": False}

        try:
            info = await self._client.info("stats")
            memory = await self._client.info("memory")
            keyspace = await self._client.info("keyspace")

            return {
                "enabled": True,
                "connected_clients": info.get("connected_clients", 0),
                "total_commands": info.get("total_commands_processed", 0),
                "keyspace_hits": info.get("keyspace_hits", 0),
                "keyspace_misses": info.get("keyspace_misses", 0),
                "used_memory": memory.get("used_memory_human", "N/A"),
                "used_memory_peak": memory.get("used_memory_peak_human", "N/A"),
                "keyspace": keyspace,
            }
        except Exception as e:
            logger.warning("Cache stats error: %s", e)
            return {"enabled": True, "error": str(e)}


# Global cache instance
_cache: Optional[CacheManager] = None


def get_cache(config: Optional[CacheConfig] = None) -> CacheManager:
    """Get or create the global cache instance."""
    global _cache
    if _cache is None:
        _cache = CacheManager(config)
    return _cache


async def init_cache(config: Optional[CacheConfig] = None) -> CacheManager:
    """Initialize and connect the cache."""
    cache = get_cache()
    await cache.connect()
    return cache


async def close_cache() -> None:
    """Close the global cache connection."""
    global _cache
    if _cache:
        await _cache.close()
        _cache = None