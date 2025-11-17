#!/usr/bin/env python3
"""
Integration verification script.

Verifies that all MHRAS components are properly integrated and functional.
"""

import sys
import traceback
from typing import List, Tuple

def test_imports() -> Tuple[bool, str]:
    """Test that all modules can be imported"""
    try:
        # Core modules
        from src.integration import MHRASIntegration, get_integration
        from src.screening_service import ScreeningService, ScreeningRequest
        from src.main import initialize_app, run_api_server
        from src.cli import cli
        
        # API modules
        from src.api.app import app
        from src.api.endpoints import initialize_components
        from src.api.auth import Authenticator
        from src.api.middleware import RequestLoggingMiddleware
        from src.api.models import ScreeningRequest as APIScreeningRequest
        
        # ML modules
        from src.ml.model_registry import ModelRegistry
        from src.ml.inference_engine import InferenceEngine
        from src.ml.ensemble_predictor import EnsemblePredictor
        from src.ml.interpretability import InterpretabilityEngine
        from src.ml.feature_pipeline import FeatureEngineeringPipeline
        
        # Governance modules
        from src.governance.audit_logger import AuditLogger
        from src.governance.human_review_queue import HumanReviewQueue
        from src.governance.drift_monitor import DriftMonitor
        from src.governance.consent import ConsentVerifier
        from src.governance.anonymization import Anonymizer
        
        # Processing modules
        from src.processing.etl_pipeline import ETLPipeline
        from src.processing.cleaning import DataCleaner
        from src.processing.imputation import Imputer
        from src.processing.normalization import Normalizer
        
        # Recommendation modules
        from src.recommendations.recommendation_engine import RecommendationEngine
        from src.recommendations.resource_catalog import ResourceCatalog
        
        # Database modules
        from src.database.connection import DatabaseConnection
        from src.database.repositories import (
            PredictionRepository,
            AuditLogRepository,
            ConsentRepository,
            HumanReviewQueueRepository
        )
        
        return True, "All imports successful"
    except Exception as e:
        return False, f"Import failed: {str(e)}\n{traceback.format_exc()}"


def test_integration_initialization() -> Tuple[bool, str]:
    """Test that integration can be initialized"""
    try:
        from src.integration import MHRASIntegration
        
        integration = MHRASIntegration()
        
        # Check all components
        assert integration.model_registry is not None
        assert integration.inference_engine is not None
        assert integration.ensemble_predictor is not None
        assert integration.interpretability_engine is not None
        assert integration.feature_pipeline is not None
        assert integration.etl_pipeline is not None
        assert integration.data_validator is not None
        assert integration.anonymizer is not None
        assert integration.consent_verifier is not None
        assert integration.audit_logger is not None
        assert integration.human_review_queue is not None
        assert integration.drift_monitor is not None
        assert integration.recommendation_engine is not None
        assert integration.screening_service is not None
        
        integration.shutdown()
        
        return True, "Integration initialized successfully"
    except Exception as e:
        return False, f"Integration initialization failed: {str(e)}\n{traceback.format_exc()}"


def test_component_access() -> Tuple[bool, str]:
    """Test that components can be accessed"""
    try:
        from src.integration import get_integration
        
        integration = get_integration()
        
        # Test getters
        screening_service = integration.get_screening_service()
        model_registry = integration.get_model_registry()
        recommendation_engine = integration.get_recommendation_engine()
        audit_logger = integration.get_audit_logger()
        human_review_queue = integration.get_human_review_queue()
        drift_monitor = integration.get_drift_monitor()
        
        assert screening_service is not None
        assert model_registry is not None
        assert recommendation_engine is not None
        assert audit_logger is not None
        assert human_review_queue is not None
        assert drift_monitor is not None
        
        return True, "Component access successful"
    except Exception as e:
        return False, f"Component access failed: {str(e)}\n{traceback.format_exc()}"


def test_health_check() -> Tuple[bool, str]:
    """Test health check functionality"""
    try:
        from src.integration import get_integration
        
        integration = get_integration()
        health = integration.health_check()
        
        assert "overall_status" in health
        assert "components" in health
        assert "timestamp" in health
        
        return True, f"Health check successful: {health['overall_status']}"
    except Exception as e:
        return False, f"Health check failed: {str(e)}\n{traceback.format_exc()}"


def test_statistics() -> Tuple[bool, str]:
    """Test statistics collection"""
    try:
        from src.integration import get_integration
        
        integration = get_integration()
        stats = integration.get_statistics()
        
        assert "timestamp" in stats
        assert "screening_service" in stats
        assert "review_queue" in stats
        assert "models" in stats
        
        return True, "Statistics collection successful"
    except Exception as e:
        return False, f"Statistics collection failed: {str(e)}\n{traceback.format_exc()}"


def test_api_app() -> Tuple[bool, str]:
    """Test that API app can be created"""
    try:
        from src.api.app import app
        
        assert app is not None
        assert hasattr(app, 'routes')
        
        # Check key routes exist
        routes = [route.path for route in app.routes]
        assert "/" in routes
        assert "/health" in routes
        assert "/metrics" in routes
        
        return True, f"API app created with {len(routes)} routes"
    except Exception as e:
        return False, f"API app creation failed: {str(e)}\n{traceback.format_exc()}"


def test_cli() -> Tuple[bool, str]:
    """Test that CLI can be imported"""
    try:
        from src.cli import cli, db, models, review, audit, system
        
        assert cli is not None
        assert db is not None
        assert models is not None
        assert review is not None
        assert audit is not None
        assert system is not None
        
        return True, "CLI imported successfully"
    except Exception as e:
        return False, f"CLI import failed: {str(e)}\n{traceback.format_exc()}"


def run_verification():
    """Run all verification tests"""
    print("=" * 70)
    print("MHRAS Integration Verification")
    print("=" * 70)
    print()
    
    tests = [
        ("Module Imports", test_imports),
        ("Integration Initialization", test_integration_initialization),
        ("Component Access", test_component_access),
        ("Health Check", test_health_check),
        ("Statistics Collection", test_statistics),
        ("API Application", test_api_app),
        ("CLI Tool", test_cli),
    ]
    
    results: List[Tuple[str, bool, str]] = []
    
    for test_name, test_func in tests:
        print(f"Testing: {test_name}...", end=" ")
        success, message = test_func()
        results.append((test_name, success, message))
        
        if success:
            print("✅ PASS")
            print(f"  {message}")
        else:
            print("❌ FAIL")
            print(f"  {message}")
        print()
    
    # Summary
    print("=" * 70)
    print("Summary")
    print("=" * 70)
    
    passed = sum(1 for _, success, _ in results if success)
    total = len(results)
    
    print(f"Tests Passed: {passed}/{total}")
    print()
    
    if passed == total:
        print("✅ ALL TESTS PASSED - Integration is complete and functional!")
        return 0
    else:
        print("❌ SOME TESTS FAILED - Please review errors above")
        print()
        print("Failed tests:")
        for test_name, success, message in results:
            if not success:
                print(f"  - {test_name}")
        return 1


if __name__ == "__main__":
    sys.exit(run_verification())
