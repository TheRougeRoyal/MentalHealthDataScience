"""MHRAS API application — Firebase/Firestore backend."""

import logging
import os
from fastapi import FastAPI, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

from src.api.models import ErrorResponse
from src.api.metrics import PrometheusMiddleware, metrics_response
from src.logging_config import setup_logging

setup_logging()
logger = logging.getLogger(__name__)

limiter = Limiter(key_func=get_remote_address, default_limits=["120/minute"])

app = FastAPI(
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

@app.on_event("startup")
async def startup_event():
    logger.info("Starting MHRAS API (Firebase backend)...")
    logger.info("API docs at /docs")


@app.on_event("shutdown")
async def shutdown_event():
    logger.info("Shutting down MHRAS API...")


# ── Prometheus metrics ─────────────────────────────────────────────────────

@app.get("/metrics", tags=["Monitoring"], include_in_schema=False)
async def prometheus_metrics():
    return metrics_response()


# ── Health / root ──────────────────────────────────────────────────────────

@app.get("/health", tags=["Health"])
async def health_check(request: Request):
    return {"status": "healthy", "service": "MHRAS API", "version": "2.0.0"}


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
