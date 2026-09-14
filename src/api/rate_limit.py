"""Shared API rate limiter.

Uses Redis storage when REDIS_URL is set so limits are shared across
Vercel serverless instances. Falls back to in-memory with a warning
when REDIS_URL is absent — limits will not be enforced across instances
in that case.

Set REDIS_URL to a Redis connection string (e.g. ``redis://host:6379/0``
or an Upstash ``rediss://...`` URL) to enable shared enforcement.
"""

from __future__ import annotations

import logging
import os

from slowapi import Limiter
from slowapi.util import get_remote_address

logger = logging.getLogger(__name__)

_redis_url = os.environ.get("REDIS_URL", "").strip()

if _redis_url:
    # Log the host portion only — never log credentials that may be in the URL.
    _safe_url = _redis_url.split("@")[-1]
    logger.info("Rate limiter: using Redis backend (%s)", _safe_url)
    limiter = Limiter(key_func=get_remote_address, storage_uri=_redis_url)
else:
    logger.warning(
        "REDIS_URL is not set — rate limiter is using in-memory storage. "
        "Limits will NOT be shared across Vercel serverless instances. "
        "Set REDIS_URL to a shared Redis instance for production enforcement."
    )
    limiter = Limiter(key_func=get_remote_address)
