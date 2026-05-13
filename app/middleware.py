"""Custom middleware: rate limiting and correlation IDs."""

import logging
import time
from collections import defaultdict

from fastapi import Request, status
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware

logger = logging.getLogger(__name__)

_request_counts: dict[str, list[float]] = defaultdict(list)
RATE_LIMIT = 60
WINDOW_SECONDS = 60


class RateLimitMiddleware(BaseHTTPMiddleware):
    """Sliding-window rate limiter: 60 requests per 60 seconds per client IP."""

    async def dispatch(self, request: Request, call_next):
        client_ip = request.client.host if request.client else "unknown"
        now = time.time()

        window = _request_counts[client_ip]
        # Prune expired timestamps
        _request_counts[client_ip] = [t for t in window if now - t < WINDOW_SECONDS]

        if len(_request_counts[client_ip]) >= RATE_LIMIT:
            logger.warning("Rate limit exceeded for %s", client_ip)
            return JSONResponse(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                content={"detail": "Rate limit exceeded. Max 60 requests per minute."},
            )

        _request_counts[client_ip].append(now)
        return await call_next(request)
