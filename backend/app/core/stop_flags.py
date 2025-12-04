"""Shared helpers to request/inspect job stop flags across workers and processes."""
import logging
import os
from typing import Optional

try:
    from redis import Redis
except Exception:  # pragma: no cover - optional dependency in local mode
    Redis = None  # type: ignore

logger = logging.getLogger(__name__)

_redis_conn: Optional["Redis"] = None
_STOP_KEY_PREFIX = "searchone:job_stop:"


def _get_redis_conn() -> Optional["Redis"]:
    """Lazily create a Redis connection if REDIS_URL is configured."""
    global _redis_conn
    if Redis is None:
        return None
    url = os.getenv("REDIS_URL")
    if not url:
        return None
    if _redis_conn is None:
        try:
            _redis_conn = Redis.from_url(url)
        except Exception as e:  # pragma: no cover - connectivity issues
            logger.debug("Failed to init redis connection for stop flags: %s", e)
            _redis_conn = None
    return _redis_conn


def request_stop(job_id: int, ttl_seconds: int = 3600) -> bool:
    """Set a stop flag for a job (read by workers to exit early)."""
    conn = _get_redis_conn()
    if not conn:
        return False
    try:
        conn.setex(f"{_STOP_KEY_PREFIX}{job_id}", ttl_seconds, "1")
        return True
    except Exception as e:  # pragma: no cover - connectivity issues
        logger.debug("Failed to set stop flag for job %s: %s", job_id, e)
        return False


def clear_stop(job_id: int) -> bool:
    """Remove a stop flag once a job is finished."""
    conn = _get_redis_conn()
    if not conn:
        return False
    try:
        conn.delete(f"{_STOP_KEY_PREFIX}{job_id}")
        return True
    except Exception as e:  # pragma: no cover - connectivity issues
        logger.debug("Failed to clear stop flag for job %s: %s", job_id, e)
        return False


def is_stop_requested(job_id: int) -> bool:
    """Return True if a stop flag is present in Redis for this job."""
    conn = _get_redis_conn()
    if not conn:
        return False
    try:
        return bool(conn.get(f"{_STOP_KEY_PREFIX}{job_id}"))
    except Exception as e:  # pragma: no cover - connectivity issues
        logger.debug("Failed to read stop flag for job %s: %s", job_id, e)
        return False
