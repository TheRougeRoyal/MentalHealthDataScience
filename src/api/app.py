"""MHRAS API application — Firebase/Firestore backend."""

import logging
import os
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from slowapi import _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded

from src.api.models import ErrorResponse
from src.api.metrics import PrometheusMiddleware, metrics_response
from src.logging_config import setup_logging
from src.firebase_admin import get_firestore_client
from src.risk_model import get_risk_model
from src.api.rate_limit import limiter

setup_logging()
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(_app: FastAPI):
    logger.info("Starting MHRAS API (Firebase backend)...")
    logger.info("API docs at /docs")
    yield
    logger.info("Shutting down MHRAS API...")


app = FastAPI(
    lifespan=lifespan,
    title="Mental Health Risk Assessment System API",
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

ALLOWED_ORIGINS = [
    origin.strip()
    for origin in os.environ.get(
        "CORS_ORIGINS",
        "https://mental-health-data-science.vercel.app,http://localhost:3000,http://127.0.0.1:3000,http://localhost:3001,http://127.0.0.1:3001",
    ).split(",")
    if origin.strip()
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["GET", "POST", "PATCH", "DELETE", "OPTIONS"],
    allow_headers=["Authorization", "Content-Type"],
)

app.add_middleware(PrometheusMiddleware)


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    errors = [
        {"field": ".".join(str(loc) for loc in e["loc"]), "message": e["msg"]}
        for e in exc.errors()
    ]
    return JSONResponse(
        status_code=status.HTTP_400_BAD_REQUEST,
        content=ErrorResponse(
            error="ValidationError",
            message="Request validation failed",
            details={"validation_errors": errors},
        ).model_dump(mode="json"),
    )


@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    logger.error("Unhandled exception: %s", exc, exc_info=True)
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content=ErrorResponse(
            error="InternalServerError",
            message="An unexpected error occurred",
        ).model_dump(mode="json"),
    )


# ── Include routers ────────────────────────────────────────────────────────

from src.api.auth import router as auth_router
from src.api.reviews import router as reviews_router
from src.api.endpoints import router as endpoints_router
from src.api.admin import router as admin_router

app.include_router(auth_router)
app.include_router(endpoints_router)
app.include_router(admin_router)
app.include_router(reviews_router)


# ── Startup / shutdown ─────────────────────────────────────────────────────

# ── Prometheus metrics ─────────────────────────────────────────────────────

@app.get("/metrics", tags=["Monitoring"], include_in_schema=False)
async def prometheus_metrics():
    return metrics_response()


# ── Health / root ──────────────────────────────────────────────────────────

def _readiness_status() -> tuple[dict, int]:
    checks = {"firestore": "unavailable", "model": "unavailable"}

    try:
        db = get_firestore_client()
        if db is not None:
            db.collection("_health").document("readiness").get()
            checks["firestore"] = "ok"
    except Exception:
        logger.exception("Readiness Firestore probe failed")

    try:
        model = get_risk_model()
        if model is not None and callable(getattr(model, "assess", None)):
            checks["model"] = "ok"
    except Exception:
        logger.exception("Readiness model probe failed")

    ready = all(value == "ok" for value in checks.values())
    body = {
        "status": "ready" if ready else "not_ready",
        "service": "MHRAS API",
        "version": "2.0.0",
        "checks": checks,
    }
    return body, 200 if ready else 503


@app.get("/live", tags=["Health"])
async def liveness_check():
    """Confirm the process is running without checking external services."""
    return {"status": "alive", "service": "MHRAS API", "version": "2.0.0"}


@app.get("/ready", tags=["Health"])
async def readiness_check():
    body, status_code = _readiness_status()
    return JSONResponse(status_code=status_code, content=body)


@app.get("/health", tags=["Health"])
async def health_check():
    """Backward-compatible readiness endpoint for existing monitors."""
    body, status_code = _readiness_status()
    return JSONResponse(status_code=status_code, content=body)


@app.get("/", tags=["Root"])
async def root():
    return {
        "service": "Mental Health Risk Assessment System API",
        "version": "2.0.0",
        "status": "operational",
        "docs": "/docs",
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("src.api.app:app", host="0.0.0.0", port=8000, reload=True)
