"""Minimal middleware for MHRAS API."""

import logging
import time
import uuid
from typing import Callable

from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware

logger = logging.getLogger(__name__)


class RequestLoggingMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        request_id = str(uuid.uuid4())
        request.state.request_id = request_id
        start = time.time()

        try:
            response = await call_next(request)
            elapsed = time.time() - start
            logger.info(
                "%s %s %s %.3fs",
                request.method, request.url.path, response.status_code, elapsed,
            )
            response.headers["X-Request-ID"] = request_id
            return response
        except Exception as e:
            elapsed = time.time() - start
            logger.error("%s %s FAILED %.3fs: %s", request.method, request.url.path, elapsed, e)
            raise
