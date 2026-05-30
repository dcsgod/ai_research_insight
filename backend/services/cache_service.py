"""
Redis-backed caching service with connection pooling and namespace management.

Features:
- Async Redis client via redis.asyncio
- Connection pooling (shared pool across requests)
- Namespace-based key isolation per domain (papers, repos, trends, …)
- Transparent JSON serialisation / deserialisation for complex objects
- ``cache()`` decorator for async functions
- TTL defaults with per-call overrides
- Atomic batch operations (mget, mset)
"""

from __future__ import annotations

import asyncio
import functools
import hashlib
import json
import logging
from datetime import datetime, date
from typing import Any, Callable, Dict, List, Optional, TypeVar, Union
from uuid import UUID

import redis.asyncio as aioredis
from redis.asyncio import ConnectionPool
from redis.asyncio.client import Redis
from redis.exceptions import ConnectionError, RedisError, TimeoutError

from backend.core.config import get_settings
from backend.core.logging import get_logger

logger = get_logger(__name__)

F = TypeVar("F", bound=Callable[..., Any])


# ---------------------------------------------------------------------------
# Custom JSON encoder for UUID, datetime, date
# ---------------------------------------------------------------------------


class _ExtendedEncoder(json.JSONEncoder):
    """JSON encoder that handles UUID, datetime, and date objects."""

    def default(self, obj: Any) -> Any:
        if isinstance(obj, UUID):
            return str(obj)
        if isinstance(obj, (datetime, date)):
            return obj.isoformat()
        return super().default(obj)


def _serialize(value: Any) -> str:
    """Serialise ``value`` to a JSON string."""
    return json.dumps(value, cls=_ExtendedEncoder, ensure_ascii=False)


def _deserialize(raw: Optional[str]) -> Any:
    """Deserialise a JSON string; return ``None`` if input is None."""
    if raw is None:
        return None
    try:
        return json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        logger.warning("Cache deserialisation error; returning raw value")
        return raw


# ---------------------------------------------------------------------------
# Connection pool (module-level singleton)
# ---------------------------------------------------------------------------

_pool: Optional[ConnectionPool] = None


def _get_pool() -> ConnectionPool:
    """Return (or create) the shared Redis connection pool."""
    global _pool
    if _pool is None:
        settings = get_settings()
        _pool = ConnectionPool.from_url(
            settings.REDIS_URL,
            max_connections=settings.REDIS_MAX_CONNECTIONS,
            decode_responses=True,
            socket_timeout=5.0,
            socket_connect_timeout=5.0,
            retry_on_timeout=True,
        )
        logger.debug(
            "Redis connection pool created",
            extra={"max_connections": settings.REDIS_MAX_CONNECTIONS},
        )
    return _pool


async def close_pool() -> None:
    """Gracefully close the shared Redis connection pool."""
    global _pool
    if _pool is not None:
        await _pool.aclose()
        _pool = None
        logger.info("Redis connection pool closed")


# ---------------------------------------------------------------------------
# Key builder
# ---------------------------------------------------------------------------

NAMESPACES = {
    "papers": "ppr",
    "repos": "rep",
    "trends": "trd",
    "topics": "top",
    "insights": "ins",
    "forecasts": "frc",
    "ingestion": "ing",
    "dashboard": "dsh",
    "global": "glb",
}


def build_key(namespace: str, *parts: Any, version: str = "v1") -> str:
    """
    Build a namespaced cache key.

    Args:
        namespace: Logical namespace string (must be in NAMESPACES).
        *parts:    Additional key components (stringified and joined with ':').
        version:   Cache version prefix for schema invalidation.

    Returns:
        A colon-delimited string key, e.g. ``"v1:ppr:list:arxiv:page1"``.

    Example::

        key = build_key("papers", "list", source="arxiv", page=1)
    """
    ns = NAMESPACES.get(namespace, namespace)
    safe_parts = [str(p).replace(":", "_").replace(" ", "_") for p in parts if p is not None]
    return ":".join([version, ns] + safe_parts)


def build_hash_key(namespace: str, **kwargs: Any) -> str:
    """
    Build a cache key by hashing arbitrary keyword arguments.

    Useful when the query parameters are complex or too long for a readable key.

    Returns:
        A key ending with an 8-char MD5 hex digest of the serialised kwargs.
    """
    payload = json.dumps(kwargs, sort_keys=True, default=str)
    digest = hashlib.md5(payload.encode()).hexdigest()[:8]
    return build_key(namespace, digest)


# ---------------------------------------------------------------------------
# CacheService class
# ---------------------------------------------------------------------------


class CacheService:
    """
    Async Redis caching service.

    Intended to be used as a per-request or application-level singleton.
    The underlying connection pool is shared at module level.

    Example::

        cache = CacheService()
        await cache.set("my_key", {"data": 42}, ttl=300)
        value = await cache.get("my_key")
    """

    def __init__(self) -> None:
        self._settings = get_settings()

    def _client(self) -> Redis:
        """Return a Redis client backed by the shared pool."""
        return aioredis.Redis(connection_pool=_get_pool())

    # -------------------------------------------------------------------------
    # Core operations
    # -------------------------------------------------------------------------

    async def get(self, key: str) -> Optional[Any]:
        """
        Retrieve and deserialise a cached value.

        Args:
            key: Cache key string.

        Returns:
            Deserialised Python object, or ``None`` if the key doesn't exist.
        """
        try:
            async with self._client() as client:
                raw = await client.get(key)
                if raw is None:
                    return None
                return _deserialize(raw)
        except (ConnectionError, TimeoutError) as exc:
            logger.warning("Redis GET failed", extra={"key": key, "error": str(exc)})
            return None
        except RedisError as exc:
            logger.error("Redis GET error", exc_info=exc, extra={"key": key})
            return None

    async def set(
        self,
        key: str,
        value: Any,
        ttl: Optional[int] = None,
    ) -> bool:
        """
        Serialise and store a value with an optional TTL.

        Args:
            key:   Cache key string.
            value: Any JSON-serialisable Python object.
            ttl:   Time-to-live in seconds; uses ``CACHE_DEFAULT_TTL`` if None.

        Returns:
            True on success, False on failure.
        """
        ttl = ttl if ttl is not None else self._settings.CACHE_DEFAULT_TTL
        try:
            async with self._client() as client:
                serialised = _serialize(value)
                await client.setex(key, ttl, serialised)
                return True
        except (ConnectionError, TimeoutError) as exc:
            logger.warning("Redis SET failed", extra={"key": key, "error": str(exc)})
            return False
        except RedisError as exc:
            logger.error("Redis SET error", exc_info=exc, extra={"key": key})
            return False

    async def delete(self, key: str) -> bool:
        """
        Delete a cache entry.

        Args:
            key: Cache key string.

        Returns:
            True if the key was deleted, False if it didn't exist or on error.
        """
        try:
            async with self._client() as client:
                result = await client.delete(key)
                return result > 0
        except RedisError as exc:
            logger.warning("Redis DELETE error", exc_info=exc, extra={"key": key})
            return False

    async def exists(self, key: str) -> bool:
        """Return True if the key exists in Redis."""
        try:
            async with self._client() as client:
                return bool(await client.exists(key))
        except RedisError as exc:
            logger.warning("Redis EXISTS error", exc_info=exc, extra={"key": key})
            return False

    async def ttl(self, key: str) -> int:
        """
        Return remaining TTL in seconds, or -1 if no TTL, -2 if key absent.
        """
        try:
            async with self._client() as client:
                return await client.ttl(key)
        except RedisError:
            return -2

    async def expire(self, key: str, seconds: int) -> bool:
        """Reset the TTL of an existing key."""
        try:
            async with self._client() as client:
                return bool(await client.expire(key, seconds))
        except RedisError:
            return False

    # -------------------------------------------------------------------------
    # Batch operations
    # -------------------------------------------------------------------------

    async def mget(self, keys: List[str]) -> List[Optional[Any]]:
        """Retrieve multiple keys in a single round-trip."""
        if not keys:
            return []
        try:
            async with self._client() as client:
                raws = await client.mget(*keys)
                return [_deserialize(r) for r in raws]
        except RedisError as exc:
            logger.warning("Redis MGET error", exc_info=exc)
            return [None] * len(keys)

    async def mset(
        self,
        mapping: Dict[str, Any],
        ttl: Optional[int] = None,
    ) -> bool:
        """
        Store multiple key-value pairs.

        Note: ``MSET`` does not support per-key TTL; uses a pipeline with
        individual ``SETEX`` calls to apply TTL uniformly.
        """
        if not mapping:
            return True
        ttl = ttl if ttl is not None else self._settings.CACHE_DEFAULT_TTL
        try:
            async with self._client() as client:
                async with client.pipeline(transaction=False) as pipe:
                    for key, value in mapping.items():
                        pipe.setex(key, ttl, _serialize(value))
                    await pipe.execute()
                return True
        except RedisError as exc:
            logger.error("Redis MSET error", exc_info=exc)
            return False

    # -------------------------------------------------------------------------
    # Pattern-based invalidation
    # -------------------------------------------------------------------------

    async def delete_pattern(self, pattern: str) -> int:
        """
        Delete all keys matching a glob ``pattern``.

        Uses SCAN to avoid blocking the server. Returns the count of
        deleted keys.

        Warning:
            Use with care in production — scanning large keyspaces is slow.
        """
        deleted = 0
        try:
            async with self._client() as client:
                async for key in client.scan_iter(match=pattern, count=100):
                    await client.delete(key)
                    deleted += 1
            logger.debug("Cache pattern purge", extra={"pattern": pattern, "deleted": deleted})
        except RedisError as exc:
            logger.error("Redis SCAN/DELETE error", exc_info=exc)
        return deleted

    async def invalidate_namespace(self, namespace: str) -> int:
        """Delete all keys belonging to a namespace."""
        ns = NAMESPACES.get(namespace, namespace)
        return await self.delete_pattern(f"*:{ns}:*")

    # -------------------------------------------------------------------------
    # Health check
    # -------------------------------------------------------------------------

    async def health_check(self) -> Dict[str, Any]:
        """Ping Redis and return a status dict."""
        try:
            async with self._client() as client:
                pong = await client.ping()
                info = await client.info("server")
                return {
                    "status": "ok",
                    "ping": str(pong),
                    "redis_version": info.get("redis_version"),
                }
        except Exception as exc:
            return {"status": "error", "detail": str(exc)}


# ---------------------------------------------------------------------------
# cache() decorator
# ---------------------------------------------------------------------------


def cache(
    namespace: str,
    ttl: Optional[int] = None,
    key_prefix: Optional[str] = None,
    include_args: bool = True,
) -> Callable[[F], F]:
    """
    Decorator that caches the return value of an async function in Redis.

    The cache key is built from the ``namespace``, optional ``key_prefix``,
    and an MD5 digest of the function's positional and keyword arguments.

    Args:
        namespace:    Logical cache namespace (used for key building).
        ttl:          Time-to-live in seconds; falls back to settings default.
        key_prefix:   Optional static prefix appended before argument hash.
        include_args: If False, the key is based only on prefix (use for
                      functions with no meaningful arguments, e.g. dashboards).

    Usage::

        @cache(namespace="papers", ttl=60)
        async def get_trending_papers(limit: int, offset: int) -> list:
            ...
    """

    def decorator(func: F) -> F:
        @functools.wraps(func)
        async def wrapper(*args: Any, **kwargs: Any) -> Any:
            svc = CacheService()

            if include_args:
                # Combine args and kwargs into a deterministic string
                arg_parts = [str(a) for a in args] + [
                    f"{k}={v}" for k, v in sorted(kwargs.items())
                ]
                arg_hash = hashlib.md5(":".join(arg_parts).encode()).hexdigest()[:12]
                key_parts = [key_prefix or func.__name__, arg_hash]
            else:
                key_parts = [key_prefix or func.__name__]

            cache_key = build_key(namespace, *key_parts)

            # Try cache hit
            cached = await svc.get(cache_key)
            if cached is not None:
                logger.debug("Cache hit", extra={"key": cache_key})
                return cached

            # Cache miss → compute
            result = await func(*args, **kwargs)

            # Store result (non-blocking; errors are logged, not raised)
            await svc.set(cache_key, result, ttl=ttl)
            return result

        return wrapper  # type: ignore[return-value]

    return decorator


# ---------------------------------------------------------------------------
# Singleton accessor
# ---------------------------------------------------------------------------

_cache_service: Optional[CacheService] = None


def get_cache_service() -> CacheService:
    """Return the application-level CacheService singleton."""
    global _cache_service
    if _cache_service is None:
        _cache_service = CacheService()
    return _cache_service
