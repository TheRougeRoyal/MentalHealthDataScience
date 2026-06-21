"""MHRAS API application — clean minimal setup."""

import logging
from fastapi import FastAPI, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware

from src.api.models import ErrorResponse
from src.logging_config import setup_logging
from src.database import init_db, check_health, engine

setup_logging()
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Mental Health Risk Assessment System API",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


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

app.include_router(auth_router)
app.include_router(endpoints_router)
app.include_router(reviews_router)


# ── Startup / shutdown ─────────────────────────────────────────────────────

@app.on_event("startup")
async def startup_event():
    logger.info("Starting MHRAS API...")
    init_db()
    if not check_health():
        logger.error("Database health-check failed during startup")
    else:
        logger.info("Database health-check passed")
    logger.info("API docs at /docs")


@app.on_event("shutdown")
async def shutdown_event():
    logger.info("Shutting down MHRAS API...")
    engine.dispose()
    logger.info("MHRAS API shutdown complete")


# ── Health / root ──────────────────────────────────────────────────────────

@app.get("/health", tags=["Health"])
async def health_check():
    return {"status": "healthy", "service": "MHRAS API", "version": "1.0.0"}


@app.get("/", tags=["Root"])
async def root():
    return {
        "service": "Mental Health Risk Assessment System API",
        "version": "1.0.0",
        "status": "operational",
        "docs": "/docs",
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("src.api.app:app", host="0.0.0.0", port=8000, reload=True)
