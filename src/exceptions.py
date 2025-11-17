"""Base exception classes for MHRAS error handling"""


class MHRASException(Exception):
    """Base exception for all MHRAS errors"""
    
    def __init__(self, message: str, details: dict = None):
        self.message = message
        self.details = details or {}
        super().__init__(self.message)


class ValidationError(MHRASException):
    """Raised when data validation fails"""
    pass


class ConsentError(MHRASException):
    """Raised when consent verification fails or consent is missing"""
    pass


class AuthenticationError(MHRASException):
    """Raised when authentication fails"""
    pass


class AuthorizationError(MHRASException):
    """Raised when authorization fails"""
    pass


class ProcessingError(MHRASException):
    """Raised when data processing or ETL fails"""
    pass


class ModelError(MHRASException):
    """Raised when model inference or loading fails"""
    pass


class TimeoutError(MHRASException):
    """Raised when operations exceed time limits"""
    pass


class ConfigurationError(MHRASException):
    """Raised when configuration is invalid or missing"""
    pass


class ModelNotFoundError(MHRASException):
    """Raised when a model is not found in the registry"""
    pass


class ModelRegistrationError(MHRASException):
    """Raised when model registration fails"""
    pass


class InferenceError(MHRASException):
    """Raised when model inference fails"""
    pass


class EnsembleError(MHRASException):
    """Raised when ensemble prediction fails"""
    pass


class DataProcessingError(MHRASException):
    """Raised when data processing fails"""
    pass


class ScreeningError(MHRASException):
    """Raised when screening workflow fails"""
    pass


class InterpretabilityError(MHRASException):
    """Raised when model interpretability/explanation generation fails"""
    pass
