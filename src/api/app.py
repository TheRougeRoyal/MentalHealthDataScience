"""Main FastAPI application with middleware and error handlers"""

import logging
from fastapi import FastAPI, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, PlainTextResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import ValidationError as PydanticValidationError
from prometheus_client import Counter, Histogram, Gauge, generate_latest, CONTENT_TYPE_LATEST

from src.api.endpoints import app as endpoints_app
from src.api.middleware import (
    RequestLoggingMiddleware,
    AuthenticationMiddleware,
    ErrorHandlingMiddleware,
    TimeoutMiddleware,
    CORSMiddleware as CustomCORSMiddleware
)
from src.api.models import ErrorResponse
from src.logging_config import setup_logging

# Prometheus metrics
REQUEST_COUNT = Counter(
    'mhras_http_requests_total',
    'Total HTTP requests',
    ['method', 'endpoint', 'status']
)

REQUEST_DURATION = Histogram(
    'mhras_http_request_duration_seconds',
    'HTTP request duration in seconds',
    ['method', 'endpoint']
)

SCREENING_COUNT = Counter(
    'mhras_screenings_total',
    'Total screening requests',
    ['risk_level']
)

PREDICTION_DURATION = Histogram(
    'mhras_prediction_duration_seconds',
    'Prediction generation duration in seconds'
)

HUMAN_REVIEW_QUEUE = Gauge(
    'mhras_human_review_queue_size',
    'Number of cases in human review queue'
)

DRIFT_SCORE = Gauge(
    'mhras_drift_score',
    'Current data drift score',
    ['feature']
)

MODEL_INFERENCE_DURATION = Histogram(
    'mhras_model_inference_duration_seconds',
    'Model inference duration in seconds',
    ['model_type']
)

ERROR_COUNT = Counter(
    'mhras_errors_total',
    'Total errors',
    ['error_type']
)

# Setup logging
setup_logging()
logger = logging.getLogger(__name__)

# Create main application
app = FastAPI(
    title="Mental Health Risk Assessment System API",
    description="API for mental health risk screening and prediction",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json"
)


# Add CORS middleware
cors_config = CustomCORSMiddleware.get_config()
app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_config["allow_origins"],
    allow_credentials=cors_config["allow_credentials"],
    allow_methods=cors_config["allow_methods"],
    allow_headers=cors_config["allow_headers"],
)

# Add custom middleware (order matters - first added is outermost)
app.add_middleware(TimeoutMiddleware, timeout_seconds=10.0)
app.add_middleware(ErrorHandlingMiddleware)
app.add_middleware(AuthenticationMiddleware)
app.add_middleware(RequestLoggingMiddleware)


# Exception handlers
@app.exception_handler(RequestValidationError)
async def validation_exception_handler(
    request: Request,
    exc: RequestValidationError
) -> JSONResponse:
    """
    Handle Pydantic validation errors.
    
    Args:
        request: Request that caused the error
        exc: Validation error
    
    Returns:
        JSON response with validation error details
    """
    request_id = getattr(request.state, "request_id", "unknown")
    
    # Extract validation error details
    errors = []
    for error in exc.errors():
        errors.append({
            "field": ".".join(str(loc) for loc in error["loc"]),
            "message": error["msg"],
            "type": error["type"]
        })
    
    logger.warning(
        f"Validation error",
        extra={
            "request_id": request_id,
            "errors": errors
        }
    )
    
    error_response = ErrorResponse(
        error="ValidationError",
        message="Request validation failed",
        details={"validation_errors": errors}
    )
    
    return JSONResponse(
        status_code=status.HTTP_400_BAD_REQUEST,
        content=error_response.model_dump(),
        headers={"X-Request-ID": request_id}
    )


@app.exception_handler(PydanticValidationError)
async def pydantic_validation_exception_handler(
    request: Request,
    exc: PydanticValidationError
) -> JSONResponse:
    """
    Handle Pydantic validation errors from models.
    
    Args:
        request: Request that caused the error
        exc: Validation error
    
    Returns:
        JSON response with validation error details
    """
    request_id = getattr(request.state, "request_id", "unknown")
    
    logger.warning(
        f"Pydantic validation error",
        extra={
            "request_id": request_id,
            "error": str(exc)
        }
    )
    
    error_response = ErrorResponse(
        error="ValidationError",
        message="Data validation failed",
        details={"error": str(exc)}
    )
    
    return JSONResponse(
        status_code=status.HTTP_400_BAD_REQUEST,
        content=error_response.model_dump(),
        headers={"X-Request-ID": request_id}
    )


@app.exception_handler(Exception)
async def general_exception_handler(
    request: Request,
    exc: Exception
) -> JSONResponse:
    """
    Handle all unhandled exceptions.
    
    Args:
        request: Request that caused the error
        exc: Exception
    
    Returns:
        JSON response with error details
    """
    request_id = getattr(request.state, "request_id", "unknown")
    
    logger.error(
        f"Unhandled exception",
        extra={
            "request_id": request_id,
            "error": str(exc)
        },
        exc_info=True
    )
    
    error_response = ErrorResponse(
        error="InternalServerError",
        message="An unexpected error occurred",
        details={}
    )
    
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content=error_response.model_dump(),
        headers={"X-Request-ID": request_id}
    )


# Include routers from endpoints
app.include_router(endpoints_app.router if hasattr(endpoints_app, 'router') else endpoints_app)

# Include data science endpoints
from src.api.ds_endpoints import router as ds_router, initialize_ds_components
app.include_router(ds_router)


# Global integration instance
_integration = None


# Startup and shutdown events
@app.on_event("startup")
async def startup_event():
    """Initialize application on startup"""
    global _integration
    
    logger.info("Starting MHRAS API...")
    
    # Initialize integration
    from src.integration import get_integration
    _integration = get_integration()
    
    # Initialize data science components
    try:
        initialize_ds_components()
        logger.info("Data science components initialized")
    except Exception as e:
        logger.warning(f"Failed to initialize data science components: {e}")
    
    # Perform health check
    health = _integration.health_check()
    logger.info(f"System health check: {health['overall_status']}")
    
    logger.info("API documentation available at /docs")


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    global _integration
    
    logger.info("Shutting down MHRAS API...")
    
    if _integration:
        _integration.shutdown()
    
    logger.info("MHRAS API shutdown complete")


# Health check endpoint
@app.get("/health", tags=["Health"])
async def health_check():
    """
    Health check endpoint.
    
    Returns:
        Health status
    """
    return {
        "status": "healthy",
        "service": "MHRAS API",
        "version": "1.0.0"
    }


# Prometheus metrics endpoint
@app.get("/metrics", tags=["Monitoring"])
async def metrics():
    """
    Prometheus metrics endpoint.
    
    Returns:
        Prometheus metrics in text format
    """
    return PlainTextResponse(
        content=generate_latest(),
        media_type=CONTENT_TYPE_LATEST
    )


@app.get("/", tags=["Root"])
async def root():
    """
    Root endpoint.
    
    Returns:
        Service information
    """
    return {
        "service": "Mental Health Risk Assessment System API",
        "version": "1.0.0",
        "status": "operational",
        "documentation": "/docs"
    }


if __name__ == "__main__":
    import uvicorn
    
    # Run the application
    uvicorn.run(
        "src.api.app:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
