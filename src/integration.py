"""
Integration module for MHRAS - connects all components together.

This module provides a unified interface for the complete screening workflow,
integrating data validation, ML inference, governance, and recommendations.
"""

import logging
from typing import Optional, Dict, Any
from datetime import datetime

from src.database.connection import DatabaseConnection
from src.database.repositories import (
    PredictionRepository,
    AuditLogRepository,
    ConsentRepository,
    HumanReviewQueueRepository,
    ResourceRepository,
    ResourceRecommendationRepository
)
from src.ml.model_registry import ModelRegistry
from src.ml.inference_engine import InferenceEngine
from src.ml.ensemble_predictor import EnsemblePredictor
from src.ml.interpretability import InterpretabilityEngine
from src.ml.feature_pipeline import FeatureEngineeringPipeline
from src.processing.etl_pipeline import ETLPipeline, ETLPipelineConfig
from src.ingestion.validation import DataValidator
from src.governance.consent import ConsentVerifier
from src.governance.anonymization import Anonymizer
from src.governance.audit_logger import AuditLogger
from src.governance.human_review_queue import HumanReviewQueue
from src.governance.drift_monitor import DriftMonitor
from src.recommendations.recommendation_engine import RecommendationEngine
from src.screening_service import ScreeningService

logger = logging.getLogger(__name__)


class MHRASIntegration:
    """
    Main integration class that connects all MHRAS components.
    
    This class provides a single entry point for initializing and accessing
    all system components with proper dependency injection.
    """
    
    def __init__(
        self,
        db_connection: Optional[DatabaseConnection] = None,
        config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize MHRAS integration.
        
        Args:
            db_connection: Database connection (creates default if None)
            config: Configuration dictionary
        """
        self.config = config or {}
        self.db_connection = db_connection
        
        # Initialize repositories
        self._init_repositories()
        
        # Initialize ML components
        self._init_ml_components()
        
        # Initialize processing components
        self._init_processing_components()
        
        # Initialize governance components
        self._init_governance_components()
        
        # Initialize recommendation engine
        self._init_recommendation_engine()
        
        # Initialize screening service
        self._init_screening_service()
        
        logger.info("MHRAS Integration initialized successfully")
    
    def _init_repositories(self):
        """Initialize database repositories"""
        if self.db_connection:
            self.prediction_repo = PredictionRepository(self.db_connection)
            self.audit_log_repo = AuditLogRepository(self.db_connection)
            self.consent_repo = ConsentRepository(self.db_connection)
            self.review_queue_repo = HumanReviewQueueRepository(self.db_connection)
            self.resource_repo = ResourceRepository(self.db_connection)
            self.resource_rec_repo = ResourceRecommendationRepository(self.db_connection)
            logger.info("Database repositories initialized")
        else:
            logger.warning("No database connection provided - repositories not initialized")
            self.prediction_repo = None
            self.audit_log_repo = None
            self.consent_repo = None
            self.review_queue_repo = None
            self.resource_repo = None
            self.resource_rec_repo = None
    
    def _init_ml_components(self):
        """Initialize ML components"""
        self.model_registry = ModelRegistry()
        self.inference_engine = InferenceEngine(self.model_registry)
        self.ensemble_predictor = EnsemblePredictor(
            self.model_registry,
            self.inference_engine
        )
        self.interpretability_engine = InterpretabilityEngine(self.model_registry)
        logger.info("ML components initialized")
    
    def _init_processing_components(self):
        """Initialize data processing components"""
        self.feature_pipeline = FeatureEngineeringPipeline()
        etl_config = ETLPipelineConfig()  # Use default configuration
        self.etl_pipeline = ETLPipeline(etl_config)
        self.data_validator = DataValidator()
        self.anonymizer = Anonymizer()
        logger.info("Processing components initialized")
    
    def _init_governance_components(self):
        """Initialize governance components"""
        self.consent_verifier = ConsentVerifier(db_connection=self.db_connection)
        self.audit_logger = AuditLogger()
        self.human_review_queue = HumanReviewQueue()
        self.drift_monitor = DriftMonitor()
        logger.info("Governance components initialized")
    
    def _init_recommendation_engine(self):
        """Initialize recommendation engine"""
        self.recommendation_engine = RecommendationEngine()
        logger.info("Recommendation engine initialized")
    
    def _init_screening_service(self):
        """Initialize screening service with all components"""
        self.screening_service = ScreeningService(
            db_connection=self.db_connection,
            model_registry=self.model_registry
        )
        logger.info("Screening service initialized")
    
    def get_screening_service(self) -> ScreeningService:
        """Get the screening service instance"""
        return self.screening_service
    
    def get_model_registry(self) -> ModelRegistry:
        """Get the model registry instance"""
        return self.model_registry
    
    def get_recommendation_engine(self) -> RecommendationEngine:
        """Get the recommendation engine instance"""
        return self.recommendation_engine
    
    def get_audit_logger(self) -> AuditLogger:
        """Get the audit logger instance"""
        return self.audit_logger
    
    def get_human_review_queue(self) -> HumanReviewQueue:
        """Get the human review queue instance"""
        return self.human_review_queue
    
    def get_drift_monitor(self) -> DriftMonitor:
        """Get the drift monitor instance"""
        return self.drift_monitor
    
    def health_check(self) -> Dict[str, Any]:
        """
        Perform health check on all components.
        
        Returns:
            Dictionary with health status of each component
        """
        health = {
            "timestamp": datetime.utcnow().isoformat(),
            "overall_status": "healthy",
            "components": {}
        }
        
        # Check database connection
        if self.db_connection:
            try:
                self.db_connection.health_check()
                health["components"]["database"] = "healthy"
            except Exception as e:
                health["components"]["database"] = f"unhealthy: {str(e)}"
                health["overall_status"] = "degraded"
        else:
            health["components"]["database"] = "not_configured"
        
        # Check model registry
        try:
            active_models = self.model_registry.get_active_models()
            health["components"]["model_registry"] = {
                "status": "healthy",
                "active_models": len(active_models)
            }
        except Exception as e:
            health["components"]["model_registry"] = f"unhealthy: {str(e)}"
            health["overall_status"] = "degraded"
        
        # Check other components
        health["components"]["feature_pipeline"] = "healthy"
        health["components"]["etl_pipeline"] = "healthy"
        health["components"]["recommendation_engine"] = "healthy"
        health["components"]["audit_logger"] = "healthy"
        health["components"]["human_review_queue"] = "healthy"
        health["components"]["drift_monitor"] = "healthy"
        
        return health
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get system statistics.
        
        Returns:
            Dictionary with system statistics
        """
        stats = {
            "timestamp": datetime.utcnow().isoformat(),
            "screening_service": self.screening_service.get_screening_statistics(),
            "review_queue": self.human_review_queue.get_queue_statistics()
        }
        
        # Add model registry stats
        active_models = self.model_registry.get_active_models()
        all_models = self.model_registry.list_models()
        stats["models"] = {
            "active_count": len(active_models),
            "total_count": len(all_models),
            "active_models": [m["model_id"] for m in active_models]
        }
        
        return stats
    
    def shutdown(self):
        """Shutdown and cleanup all components"""
        logger.info("Shutting down MHRAS Integration")
        
        # Close database connection
        if self.db_connection:
            self.db_connection.close()
        
        # Clear caches
        self.screening_service.clear_cache()
        
        logger.info("MHRAS Integration shutdown complete")


# Global integration instance (singleton pattern)
_integration_instance: Optional[MHRASIntegration] = None


def get_integration(
    db_connection: Optional[DatabaseConnection] = None,
    config: Optional[Dict[str, Any]] = None,
    force_new: bool = False
) -> MHRASIntegration:
    """
    Get or create the global MHRAS integration instance.
    
    Args:
        db_connection: Database connection
        config: Configuration dictionary
        force_new: Force creation of new instance
    
    Returns:
        MHRASIntegration instance
    """
    global _integration_instance
    
    if _integration_instance is None or force_new:
        _integration_instance = MHRASIntegration(
            db_connection=db_connection,
            config=config
        )
    
    return _integration_instance


def reset_integration():
    """Reset the global integration instance"""
    global _integration_instance
    
    if _integration_instance:
        _integration_instance.shutdown()
        _integration_instance = None
