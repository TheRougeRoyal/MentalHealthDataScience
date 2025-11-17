"""API layer components for MHRAS"""

from src.api.app import app
from src.api.models import (
    ScreeningRequest,
    ScreeningResponse,
    RiskScore,
    RiskScoreResponse,
    ExplanationRequest,
    ExplanationResponse,
    ExplanationSummary,
    ResourceRecommendation,
    RiskLevel,
    ErrorResponse
)
from src.api.auth import Authenticator, AuthResult, authenticator
from src.api.middleware import (
    RequestLoggingMiddleware,
    AuthenticationMiddleware,
    ErrorHandlingMiddleware,
    TimeoutMiddleware
)

__all__ = [
    'app',
    'ScreeningRequest',
    'ScreeningResponse',
    'RiskScore',
    'RiskScoreResponse',
    'ExplanationRequest',
    'ExplanationResponse',
    'ExplanationSummary',
    'ResourceRecommendation',
    'RiskLevel',
    'ErrorResponse',
    'Authenticator',
    'AuthResult',
    'authenticator',
    'RequestLoggingMiddleware',
    'AuthenticationMiddleware',
    'ErrorHandlingMiddleware',
    'TimeoutMiddleware'
]
