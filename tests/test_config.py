"""Tests for configuration management"""

import pytest
from src.config import Settings, DatabaseConfig, APIConfig, MLConfig


def test_settings_default_values():
    """Test that settings load with default values"""
    settings = Settings()
    
    assert settings.environment == "development"
    assert settings.debug is False
    assert settings.database.host == "localhost"
    assert settings.api.port == 8000


def test_database_config():
    """Test database configuration"""
    db_config = DatabaseConfig()
    
    assert db_config.host == "localhost"
    assert db_config.port == 5432
    assert db_config.name == "mhras"
    assert db_config.pool_size == 10


def test_api_config():
    """Test API configuration"""
    api_config = APIConfig()
    
    assert api_config.host == "0.0.0.0"
    assert api_config.port == 8000
    assert api_config.timeout == 5


def test_ml_config():
    """Test ML configuration"""
    ml_config = MLConfig()
    
    assert ml_config.inference_timeout == 2
    assert ml_config.risk_threshold_high == 51.0
    assert ml_config.risk_threshold_critical == 75.0
