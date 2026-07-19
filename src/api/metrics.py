"""Prometheus metrics for MHRAS API.

Provides:
- HTTP request metrics (count, latency, in-flight) via middleware
- Screening-specific counters (risk levels, alerts, reviews)
- Firestore operation latency histogram

All metrics are exposed at /metrics in Prometheus exposition format.
"""

from __future__ import annotations

import time
from typing import Callable

from prometheus_client import (
    CollectorRegistry,
    Counter,
    Gauge,
    Histogram,
    generate_latest,
    CONTENT_TYPE_LATEST,
)
from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint

# ── Registry ───────────────────────────────────────────────────────────────

registry = CollectorRegistry()

# ── HTTP metrics ───────────────────────────────────────────────────────────

REQUEST_COUNT = Counter(
    "mhras_http_requests_total",
    "Total HTTP requests",
    ["method", "endpoint", "status_code"],
    registry=registry,
)

REQUEST_LATENCY = Histogram(
    "mhras_http_request_duration_seconds",
    "HTTP request latency in seconds",
    ["method", "endpoint"],
    buckets=(0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0),
    registry=registry,
)

IN_FLIGHT = Gauge(
    "mhras_http_requests_in_flight",
    "Number of HTTP requests currently being processed",
    registry=registry,
)

# ── Screening metrics ──────────────────────────────────────────────────────

SCREENINGS_TOTAL = Counter(
    "mhras_screenings_total",
    "Total screenings processed",
    ["risk_level"],
    registry=registry,
)

SCREENING_SCORE = Histogram(
    "mhras_screening_score",
    "Distribution of risk scores",
    buckets=(10, 20, 30, 40, 50, 60, 70, 80, 90, 100),
    registry=registry,
)

ALERTS_TRIGGERED = Counter(
    "mhras_alerts_triggered_total",
    "Total high-risk alerts triggered",
    registry=registry,
)

REVIEWS_CREATED = Counter(
    "mhras_reviews_created_total",
    "Total reviews created (cases requiring human review)",
    registry=registry,
)

BATCH_SIZE = Histogram(
    "mhras_batch_screening_size",
    "Batch screening request sizes",
    buckets=(1, 5, 10, 25, 50, 100),
    registry=registry,
)

BATCH_ITEMS = Counter(
    "mhras_batch_screening_items_total",
    "Total items processed in batch screenings",
    ["status"],
    registry=registry,
)

# ── Firestore metrics ──────────────────────────────────────────────────────

FIRESTORE_OPS = Counter(
    "mhras_firestore_operations_total",
    "Total Firestore operations",
    ["operation", "collection"],
    registry=registry,
)

FIRESTORE_LATENCY = Histogram(
    "mhras_firestore_operation_duration_seconds",
    "Firestore operation latency",
    ["operation"],
    buckets=(0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5),
    registry=registry,
)

# ── Auth metrics ───────────────────────────────────────────────────────────

AUTH_ATTEMPTS = Counter(
    "mhras_auth_attempts_total",
    "Total authentication attempts",
    ["method", "result"],
    registry=registry,
)


# ── Middleware ──────────────────────────────────────────────────────────────

def _normalise_path(path: str) -> str:
    """Collapse variable path segments for high-cardinality control.

    /reviews/abc123/assign → /reviews/{id}/assign
    /risk-score/patient_1  → /risk-score/{id}
    """
    parts = path.strip("/").split("/")
    out = []
    for i, p in enumerate(parts):
        if i >= 2 and len(p) > 16:
            out.append("{id}")
        elif p and p[0] == "{" and p[-1] == "}":
            out.append(p)
        else:
            out.append(p)
    return "/" + "/".join(out)


class PrometheusMiddleware(BaseHTTPMiddleware):
    """Record request count, latency, and in-flight gauge."""

    async def dispatch(
        self, request: Request, call_next: RequestResponseEndpoint,
    ) -> Response:
        if request.url.path == "/metrics":
            return await call_next(request)

        path = _normalise_path(request.url.path)
        method = request.method
        IN_FLIGHT.inc()
        start = time.perf_counter()

        try:
            response = await call_next(request)
        except Exception:
            REQUEST_COUNT.labels(method=method, endpoint=path, status_code="500").inc()
            raise
        finally:
            IN_FLIGHT.dec()

        elapsed = time.perf_counter() - start
        REQUEST_LATENCY.labels(method=method, endpoint=path).observe(elapsed)
        REQUEST_COUNT.labels(
            method=method, endpoint=path, status_code=str(response.status_code),
        ).inc()

        return response


# ── /metrics endpoint ──────────────────────────────────────────────────────

def metrics_response() -> Response:
    """Return Prometheus text exposition format."""
    return Response(
        content=generate_latest(registry),
        media_type=CONTENT_TYPE_LATEST,
    )
