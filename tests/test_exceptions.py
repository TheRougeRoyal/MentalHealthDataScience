"""Tests for exception classes"""

import pytest
from src.exceptions import (
    MHRASException,
    ValidationError,
    ConsentError,
    AuthenticationError,
    ProcessingError,
    ModelError,
)


def test_base_exception():
    """Test base MHRAS exception"""
    error = MHRASException("Test error", {"key": "value"})
    
    assert str(error) == "Test error"
    assert error.message == "Test error"
    assert error.details == {"key": "value"}


def test_validation_error():
    """Test validation error"""
    error = ValidationError("Invalid data", {"field": "email"})
    
    assert isinstance(error, MHRASException)
    assert str(error) == "Invalid data"
    assert error.details["field"] == "email"


def test_consent_error():
    """Test consent error"""
    error = ConsentError("Consent not found")
    
    assert isinstance(error, MHRASException)
    assert str(error) == "Consent not found"


def test_authentication_error():
    """Test authentication error"""
    error = AuthenticationError("Invalid token")
    
    assert isinstance(error, MHRASException)
    assert str(error) == "Invalid token"


def test_processing_error():
    """Test processing error"""
    error = ProcessingError("ETL failed")
    
    assert isinstance(error, MHRASException)
    assert str(error) == "ETL failed"


def test_model_error():
    """Test model error"""
    error = ModelError("Model not found")
    
    assert isinstance(error, MHRASException)
    assert str(error) == "Model not found"
