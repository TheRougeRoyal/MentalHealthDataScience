"""Middleware for MHRAS API"""

import logging
import time
import uuid
from typing import Callable
from fastapi import Request, Response
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.types import ASGIApp

from src.api.models import ErrorResponse
from src.exceptions import (
    ValidationError,
    ConsentError,
    InferenceError,
    InterpretabilityError,
    ModelNotFoundError,
    EnsembleError
)

logger = logging.getLogger(__name__)


class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """
    Middleware for logging all API requests and responses.
    
    Logs:
    - Request ID
    - Method and path
    - Client IP
    - User agent
    - Response status code
    - Response time
    """
    
    async def dispatch(
        self,
        request: Request,
        call_next: Callable
    ) -> Response:
        """
        Process request and log details.
        
        Args:
            request: Incoming request
            call_next: Next middleware/endpoint
        
        Returns:
            Response from endpoint
        """
        # Generate request ID
        request_id = str(uuid.uuid4())
        request.state.request_id = request_id
        
        # Log request
        start_time = time.time()
        
        logger.info(
            f"Request started",
            extra={
                "request_id": request_id,
                "method": request.method,
                "path": request.url.path,
                "client_ip": request.client.host if request.client else "unknown",
                "user_agent": request.headers.get("user-agent", "unknown")
            }
        )
        
        # Process request
        try:
            response = await call_next(request)
            
            # Calculate response time
            elapsed_time = time.time() - start_time
            
            # Log response
            logger.info(
                f"Request completed",
                extra={
                    "request_id": request_id,
                    "method": request.method,
                    "path": request.url.path,
                    "status_code": response.status_code,
                    "response_time_ms": round(elapsed_time * 1000, 2)
                }
            )
            
            # Add request ID to response headers
            response.headers["X-Request-ID"] = request_id
            response.headers["X-Response-Time"] = f"{elapsed_time:.3f}s"
            
            return response
            
        except Exception as e:
            elapsed_time = time.time() - start_time
            
            logger.error(
                f"Request failed",
                extra={
                    "request_id": request_id,
                    "method": request.method,
                    "path": request.url.path,
                    "error": str(e),
                    "response_time_ms": round(elapsed_time * 1000, 2)
                },
                exc_info=True
            )
            
            raise


class AuthenticationMiddleware(BaseHTTPMiddleware):
    """
    Middleware for authentication on protected endpoints.
    
    Note: This is a placeholder. Actual authentication is handled
    by the verify_authentication dependency in endpoints.
    This middleware can be used for additional auth-related processing.
    """
    
    async def dispatch(
        self,
        request: Request,
        call_next: Callable
    ) -> Response:
        """
        Process authentication.
        
        Args:
            request: Incoming request
            call_next: Next middleware/endpoint
        
        Returns:
            Response from endpoint
        """
        # Skip authentication for public endpoints
        public_paths = ["/", "/health", "/docs", "/openapi.json", "/redoc"]
        
        if request.url.path in public_paths:
            return await call_next(request)
        
        # For protected endpoints, authentication is handled by dependencies
        # This middleware can add additional auth-related processing
        
        response = await call_next(request)
        return response


class ErrorHandlingMiddleware(BaseHTTPMiddleware):
    """
    Middleware for centralized error handling.
    
    Catches exceptions and returns appropriate HTTP responses with
    standardized error format.
    """
    
    async def dispatch(
        self,
        request: Request,
        call_next: Callable
    ) -> Response:
        """
        Handle errors and return appropriate responses.
        
        Args:
            request: Incoming request
            call_next: Next middleware/endpoint
        
        Returns:
            Response (may be error response)
        """
        try:
            response = await call_next(request)
            return response
            
        except ValidationError as e:
            # 400 Bad Request
            return self._create_error_response(
                status_code=400,
                error="ValidationError",
                message=str(e),
                request=request
            )
        
        except ConsentError as e:
            # 403 Forbidden
            return self._create_error_response(
                status_code=403,
                error="ConsentError",
                message=str(e),
                request=request
            )
        
        except ModelNotFoundError as e:
            # 503 Service Unavailable
            return self._create_error_response(
                status_code=503,
                error="ModelNotFoundError",
                message=str(e),
                request=request
            )
        
        except InferenceError as e:
            # 503 Service Unavailable
            return self._create_error_response(
                status_code=503,
                error="InferenceError",
                message=str(e),
                request=request
            )
        
        except EnsembleError as e:
            # 503 Service Unavailable
            return self._create_error_response(
                status_code=503,
                error="EnsembleError",
                message=str(e),
                request=request
            )
        
        except InterpretabilityError as e:
            # 500 Internal Server Error (non-critical, can continue)
            logger.warning(f"Interpretability error (non-critical): {e}")
            return self._create_error_response(
                status_code=500,
                error="InterpretabilityError",
                message=str(e),
                request=request
            )
        
        except TimeoutError as e:
            # 504 Gateway Timeout
            return self._create_error_response(
                status_code=504,
                error="TimeoutError",
                message=str(e),
                request=request
            )
        
        except Exception as e:
            # 500 Internal Server Error
            logger.error(f"Unhandled exception: {e}", exc_info=True)
            return self._create_error_response(
                status_code=500,
                error="InternalServerError",
                message="An unexpected error occurred",
                request=request
            )
    
    def _create_error_response(
        self,
        status_code: int,
        error: str,
        message: str,
        request: Request,
        details: dict = None
    ) -> JSONResponse:
        """
        Create standardized error response.
        
        Args:
            status_code: HTTP status code
            error: Error type
            message: Error message
            request: Original request
            details: Optional additional details
        
        Returns:
            JSONResponse with error information
        """
        request_id = getattr(request.state, "request_id", "unknown")
        
        error_response = ErrorResponse(
            error=error,
            message=message,
            details=details or {}
        )
        
        logger.error(
            f"Error response",
            extra={
                "request_id": request_id,
                "status_code": status_code,
                "error": error,
                "message": message
            }
        )
        
        return JSONResponse(
            status_code=status_code,
            content=error_response.model_dump(),
            headers={"X-Request-ID": request_id}
        )


class TimeoutMiddleware(BaseHTTPMiddleware):
    """
    Middleware for enforcing request timeouts.
    
    Ensures requests complete within specified time limits.
    """
    
    def __init__(
        self,
        app: ASGIApp,
        timeout_seconds: float = 10.0
    ):
        """
        Initialize timeout middleware.
        
        Args:
            app: ASGI application
            timeout_seconds: Maximum request duration
        """
        super().__init__(app)
        self.timeout_seconds = timeout_seconds
    
    async def dispatch(
        self,
        request: Request,
        call_next: Callable
    ) -> Response:
        """
        Enforce timeout on request processing.
        
        Args:
            request: Incoming request
            call_next: Next middleware/endpoint
        
        Returns:
            Response from endpoint
        
        Raises:
            TimeoutError: If request exceeds timeout
        """
        import asyncio
        
        start_time = time.time()
        
        try:
            # Process request with timeout
            response = await asyncio.wait_for(
                call_next(request),
                timeout=self.timeout_seconds
            )
            
            return response
            
        except asyncio.TimeoutError:
            elapsed_time = time.time() - start_time
            
            logger.error(
                f"Request timeout after {elapsed_time:.3f}s "
                f"(limit: {self.timeout_seconds}s)"
            )
            
            raise TimeoutError(
                f"Request exceeded timeout of {self.timeout_seconds}s"
            )


class CORSMiddleware:
    """
    CORS middleware configuration.
    
    Note: FastAPI provides built-in CORSMiddleware.
    This is a placeholder for custom CORS configuration.
    """
    
    @staticmethod
    def get_config():
        """
        Get CORS configuration.
        
        Returns:
            Dictionary with CORS settings
        """
        return {
            "allow_origins": ["*"],  # In production, specify allowed origins
            "allow_credentials": True,
            "allow_methods": ["*"],
            "allow_headers": ["*"],
        }
