"""
End-to-end screening service orchestrating all MHRAS components.

This module provides the ScreeningService class that integrates:
- Data validation and consent verification
- Anonymization and ETL processing
- Feature engineering
- ML inference and ensemble prediction
- Interpretability and recommendations
- Governance and audit logging
"""

import asyncio
import time
from typing import Dict, Any, Optional, List
from datetime import datetime
import pandas as pd
import numpy as np
import structlog

from src.ingestion.validation import DataValidator, ValidationResult
from src.governance.consent import ConsentVerifier
from src.governance.anonymization import Anonymizer
from src.processing.etl_pipeline import ETLPipeline, ETLPipelineConfig
from src.ml.feature_pipeline import FeatureEngineeringPipeline
from src.ml.model_registry import ModelRegistry
from src.ml.inference_engine import InferenceEngine
from src.ml.ensemble_predictor import EnsemblePredictor, RiskLevel
from src.ml.interpretability import InterpretabilityEngine
from src.recommendations.recommendation_engine import (
    RecommendationEngine,
    IndividualProfile
)
from src.governance.audit_logger import AuditLogger
from src.governance.human_review_queue import HumanReviewQueue
from src.exceptions import (
    ValidationError,
    ConsentError,
    DataProcessingError,
    InferenceError,
    ScreeningError
)

logger = structlog.get_logger(__name__)


class ScreeningRequest:
    """Request for mental health screening."""
    
    def __init__(
        self,
        anonymized_id: str,
        survey_data: Optional[Dict] = None,
        wearable_data: Optional[Dict] = None,
        emr_data: Optional[Dict] = None,
        user_id: Optional[str] = None,
        timestamp: Optional[datetime] = None
    ):
        self.anonymized_id = anonymized_id
        self.survey_data = survey_data
        self.wearable_data = wearable_data
        self.emr_data = emr_data
        self.user_id = user_id
        self.timestamp = timestamp or datetime.utcnow()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert request to dictionary."""
        return {
            "anonymized_id": self.anonymized_id,
            "survey_data": self.survey_data,
            "wearable_data": self.wearable_data,
            "emr_data": self.emr_data,
            "user_id": self.user_id,
            "timestamp": self.timestamp.isoformat()
        }


class ScreeningResponse:
    """Response from mental health screening."""
    
    def __init__(
        self,
        anonymized_id: str,
        risk_score: float,
        risk_level: str,
        confidence: float,
        contributing_factors: List[str],
        recommendations: List[Dict[str, Any]],
        explanations: Dict[str, Any],
        requires_human_review: bool,
        alert_triggered: bool,
        processing_time_seconds: float,
        timestamp: Optional[datetime] = None
    ):
        self.anonymized_id = anonymized_id
        self.risk_score = risk_score
        self.risk_level = risk_level
        self.confidence = confidence
        self.contributing_factors = contributing_factors
        self.recommendations = recommendations
        self.explanations = explanations
        self.requires_human_review = requires_human_review
        self.alert_triggered = alert_triggered
        self.processing_time_seconds = processing_time_seconds
        self.timestamp = timestamp or datetime.utcnow()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert response to dictionary."""
        return {
            "anonymized_id": self.anonymized_id,
            "risk_score": {
                "score": self.risk_score,
                "risk_level": self.risk_level,
                "confidence": self.confidence,
                "contributing_factors": self.contributing_factors
            },
            "recommendations": self.recommendations,
            "explanations": self.explanations,
            "requires_human_review": self.requires_human_review,
            "alert_triggered": self.alert_triggered,
            "processing_time_seconds": self.processing_time_seconds,
            "timestamp": self.timestamp.isoformat()
        }


class ScreeningService:
    """
    End-to-end screening service orchestrating all MHRAS components.
    
    Responsibilities:
    - Validate and verify consent for incoming data
    - Anonymize and process data through ETL pipeline
    - Extract features and generate predictions
    - Provide interpretability and recommendations
    - Log audit trail and route to human review if needed
    """
    
    def __init__(
        self,
        db_connection=None,
        model_registry: Optional[ModelRegistry] = None,
        etl_config: Optional[ETLPipelineConfig] = None,
        alert_threshold: float = 75.0,
        human_review_threshold: float = 75.0,
        use_cache: bool = True
    ):
        """
        Initialize ScreeningService.
        
        Args:
            db_connection: Database connection for consent and persistence
            model_registry: ModelRegistry instance (creates default if None)
            etl_config: ETL pipeline configuration
            alert_threshold: Risk score threshold for alerts
            human_review_threshold: Risk score threshold for human review
            use_cache: Whether to use caching for performance
        """
        logger.info("Initializing ScreeningService")
        
        # Initialize components
        self.validator = DataValidator()
        self.consent_verifier = ConsentVerifier(db_connection=db_connection)
        self.anonymizer = Anonymizer()
        
        # ETL and feature engineering
        self.etl_config = etl_config or self._default_etl_config()
        self.etl_pipeline = ETLPipeline(self.etl_config)
        self.feature_pipeline = FeatureEngineeringPipeline(use_parallel=True)
        
        # ML components
        self.model_registry = model_registry or ModelRegistry()
        self.inference_engine = InferenceEngine(self.model_registry)
        self.ensemble_predictor = EnsemblePredictor(
            self.model_registry,
            self.inference_engine,
            alert_threshold=alert_threshold
        )
        self.interpretability_engine = InterpretabilityEngine(self.model_registry)
        
        # Recommendations and governance
        self.recommendation_engine = RecommendationEngine()
        self.audit_logger = AuditLogger()
        self.human_review_queue = HumanReviewQueue(
            escalation_threshold_hours=4
        )
        
        # Configuration
        self.alert_threshold = alert_threshold
        self.human_review_threshold = human_review_threshold
        self.use_cache = use_cache
        
        # Cache for frequently accessed data
        self._cache: Dict[str, Any] = {}
        self._model_cache: Dict[str, Any] = {}
        self._feature_cache: Dict[str, pd.DataFrame] = {}
        
        logger.info(
            "ScreeningService initialized",
            alert_threshold=alert_threshold,
            human_review_threshold=human_review_threshold
        )
    
    def _default_etl_config(self) -> ETLPipelineConfig:
        """Create default ETL configuration."""
        return ETLPipelineConfig(
            outlier_method="iqr",
            iqr_multiplier=1.5,
            imputation_strategies={
                "forward_fill": ["sleep_duration", "heart_rate"],
                "median": ["activity_count", "hrv_rmssd"],
                "mode": ["mood_category"]
            },
            group_by_column="anonymized_id"
        )

    async def screen_individual(
        self,
        request: ScreeningRequest
    ) -> ScreeningResponse:
        """
        Perform end-to-end screening for an individual.
        
        Args:
            request: ScreeningRequest with data and metadata
        
        Returns:
            ScreeningResponse with risk assessment and recommendations
        
        Raises:
            ScreeningError: If screening fails
        """
        start_time = time.time()
        
        try:
            logger.info(
                "Starting screening",
                anonymized_id=request.anonymized_id,
                has_survey=request.survey_data is not None,
                has_wearable=request.wearable_data is not None,
                has_emr=request.emr_data is not None
            )
            
            # Stage 1: Validation and Consent
            await self._validate_and_verify_consent(request)
            
            # Stage 2: Data Processing
            processed_data = await self._process_data(request)
            
            # Stage 3: Feature Engineering
            features = await self._extract_features(processed_data)
            
            # Stage 4: ML Inference
            prediction_result = await self._generate_prediction(
                features,
                request.anonymized_id
            )
            
            # Stage 5: Interpretability
            explanations = await self._generate_explanations(
                features,
                prediction_result
            )
            
            # Stage 6: Recommendations
            recommendations = await self._generate_recommendations(
                request.anonymized_id,
                prediction_result,
                explanations
            )
            
            # Stage 7: Governance
            await self._handle_governance(
                request,
                prediction_result,
                features
            )
            
            # Build response
            elapsed_time = time.time() - start_time
            
            response = ScreeningResponse(
                anonymized_id=request.anonymized_id,
                risk_score=prediction_result["risk_score"],
                risk_level=prediction_result["risk_level"],
                confidence=prediction_result["confidence"],
                contributing_factors=explanations.get("top_features", []),
                recommendations=recommendations,
                explanations=explanations,
                requires_human_review=prediction_result["requires_human_review"],
                alert_triggered=prediction_result["alert_triggered"],
                processing_time_seconds=elapsed_time
            )
            
            # Log audit trail
            self.audit_logger.log_screening_request(
                request=request.to_dict(),
                response=response.to_dict(),
                anonymized_id=request.anonymized_id,
                user_id=request.user_id
            )
            
            logger.info(
                "Screening completed",
                anonymized_id=request.anonymized_id,
                risk_score=response.risk_score,
                risk_level=response.risk_level,
                processing_time=elapsed_time
            )
            
            # Check performance requirement (< 5 seconds)
            if elapsed_time > 5.0:
                logger.warning(
                    "Screening exceeded 5s target",
                    anonymized_id=request.anonymized_id,
                    processing_time=elapsed_time
                )
            
            return response
            
        except (ValidationError, ConsentError, DataProcessingError, InferenceError) as e:
            # Log error
            self.audit_logger.log_error(
                error_type=type(e).__name__,
                error_message=str(e),
                component="ScreeningService",
                anonymized_id=request.anonymized_id
            )
            
            logger.error(
                "Screening failed",
                anonymized_id=request.anonymized_id,
                error=str(e),
                error_type=type(e).__name__
            )
            
            raise ScreeningError(f"Screening failed: {str(e)}") from e
        
        except Exception as e:
            # Log unexpected error
            self.audit_logger.log_error(
                error_type="UnexpectedError",
                error_message=str(e),
                component="ScreeningService",
                anonymized_id=request.anonymized_id
            )
            
            logger.error(
                "Unexpected screening error",
                anonymized_id=request.anonymized_id,
                error=str(e),
                exc_info=True
            )
            
            raise ScreeningError(f"Unexpected screening error: {str(e)}") from e

    async def _validate_and_verify_consent(
        self,
        request: ScreeningRequest
    ) -> None:
        """
        Validate data and verify consent.
        
        Args:
            request: ScreeningRequest
        
        Raises:
            ValidationError: If validation fails
            ConsentError: If consent verification fails
        """
        # Determine which data types are present
        data_types = []
        validation_results = []
        
        # Validate survey data
        if request.survey_data:
            data_types.append("survey")
            result = self.validator.validate_survey(request.survey_data)
            validation_results.append(result)
            
            if not result.is_valid:
                raise ValidationError(
                    f"Survey validation failed: {result.errors}",
                    details=result.to_dict()
                )
        
        # Validate wearable data
        if request.wearable_data:
            data_types.append("wearable")
            result = self.validator.validate_wearable(request.wearable_data)
            validation_results.append(result)
            
            if not result.is_valid:
                raise ValidationError(
                    f"Wearable validation failed: {result.errors}",
                    details=result.to_dict()
                )
        
        # Validate EMR data
        if request.emr_data:
            data_types.append("emr")
            result = self.validator.validate_emr(request.emr_data)
            validation_results.append(result)
            
            if not result.is_valid:
                raise ValidationError(
                    f"EMR validation failed: {result.errors}",
                    details=result.to_dict()
                )
        
        # Verify consent for all data types
        consent_result = self.consent_verifier.verify_consent(
            request.anonymized_id,
            data_types
        )
        
        # Log consent verification
        self.audit_logger.log_consent_verification(
            anonymized_id=request.anonymized_id,
            data_types=data_types,
            consent_status=consent_result.status.value,
            consent_expiry=consent_result.expires_at
        )
        
        logger.info(
            "Validation and consent verification completed",
            anonymized_id=request.anonymized_id,
            data_types=data_types,
            consent_status=consent_result.status.value
        )
    
    async def _process_data(
        self,
        request: ScreeningRequest
    ) -> Dict[str, pd.DataFrame]:
        """
        Process data through ETL pipeline.
        
        Args:
            request: ScreeningRequest
        
        Returns:
            Dictionary of processed DataFrames by data type
        """
        processed_data = {}
        
        # Convert request data to DataFrames
        if request.survey_data:
            survey_df = pd.DataFrame([request.survey_data])
            survey_df["anonymized_id"] = request.anonymized_id
            survey_df["timestamp"] = request.timestamp
            processed_data["survey"] = survey_df
        
        if request.wearable_data:
            wearable_df = pd.DataFrame([request.wearable_data])
            wearable_df["anonymized_id"] = request.anonymized_id
            wearable_df["timestamp"] = request.timestamp
            processed_data["wearable"] = wearable_df
        
        if request.emr_data:
            emr_df = pd.DataFrame([request.emr_data])
            emr_df["anonymized_id"] = request.anonymized_id
            emr_df["timestamp"] = request.timestamp
            processed_data["emr"] = emr_df
        
        logger.info(
            "Data processing completed",
            anonymized_id=request.anonymized_id,
            data_types=list(processed_data.keys())
        )
        
        return processed_data
    
    async def _extract_features(
        self,
        processed_data: Dict[str, pd.DataFrame]
    ) -> pd.DataFrame:
        """
        Extract features from processed data.
        
        Args:
            processed_data: Dictionary of processed DataFrames
        
        Returns:
            DataFrame with extracted features
        """
        # Extract features from different data sources
        features_df = self.feature_pipeline.extract_features(
            behavioral_df=processed_data.get("survey"),
            text_df=processed_data.get("survey"),
            sleep_df=processed_data.get("wearable"),
            hrv_df=processed_data.get("wearable"),
            activity_df=processed_data.get("wearable"),
            adherence_df=processed_data.get("emr"),
            id_column="anonymized_id",
            validate=True
        )
        
        logger.info(
            "Feature extraction completed",
            num_features=len(features_df.columns) - 1,
            num_individuals=len(features_df)
        )
        
        return features_df

    async def _generate_prediction(
        self,
        features: pd.DataFrame,
        anonymized_id: str
    ) -> Dict[str, Any]:
        """
        Generate risk prediction using ensemble models.
        
        Args:
            features: Feature DataFrame
            anonymized_id: Individual identifier
        
        Returns:
            Dictionary with prediction results
        """
        # Check cache for recent prediction
        if self.use_cache:
            cache_key = f"prediction_{anonymized_id}_{hash(str(features.to_dict()))}"
            if cache_key in self._cache:
                logger.debug(
                    "Using cached prediction",
                    anonymized_id=anonymized_id
                )
                return self._cache[cache_key]
        
        # Generate ensemble prediction
        ensemble_result = self.ensemble_predictor.predict_with_ensemble(
            features=features,
            individual_ids=[anonymized_id]
        )
        
        # Extract results for this individual
        risk_score = float(ensemble_result["risk_scores"][0])
        confidence = float(ensemble_result["confidence"][0])
        risk_level = ensemble_result["risk_levels"][0]
        alert_triggered = ensemble_result["alerts_triggered"][0]
        
        # Determine if human review is required
        requires_human_review = risk_score > self.human_review_threshold
        
        # Log prediction
        features_hash = self.anonymizer.hash_identifier(
            str(features.to_dict())
        )
        
        self.audit_logger.log_prediction(
            features_hash=features_hash,
            prediction={
                "score": risk_score,
                "risk_level": risk_level,
                "confidence": confidence
            },
            model_id="ensemble",
            anonymized_id=anonymized_id,
            model_version=None
        )
        
        logger.info(
            "Prediction generated",
            anonymized_id=anonymized_id,
            risk_score=risk_score,
            risk_level=risk_level,
            confidence=confidence,
            alert_triggered=alert_triggered,
            requires_human_review=requires_human_review
        )
        
        result = {
            "risk_score": risk_score,
            "risk_level": risk_level,
            "confidence": confidence,
            "alert_triggered": alert_triggered,
            "requires_human_review": requires_human_review,
            "ensemble_models": ensemble_result["model_ids"]
        }
        
        # Cache result
        if self.use_cache:
            cache_key = f"prediction_{anonymized_id}_{hash(str(features.to_dict()))}"
            self._cache[cache_key] = result
            
            # Limit cache size
            if len(self._cache) > 1000:
                # Remove oldest 100 entries
                keys_to_remove = list(self._cache.keys())[:100]
                for key in keys_to_remove:
                    del self._cache[key]
        
        return result
    
    async def _generate_explanations(
        self,
        features: pd.DataFrame,
        prediction_result: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Generate interpretability explanations.
        
        Args:
            features: Feature DataFrame
            prediction_result: Prediction results
        
        Returns:
            Dictionary with explanations
        """
        try:
            # Get active models for explanation
            active_models = self.model_registry.get_active_models()
            
            if not active_models:
                logger.warning("No active models for explanations")
                return {
                    "top_features": [],
                    "counterfactual": "Explanations unavailable",
                    "rule_approximation": "Explanations unavailable"
                }
            
            # Use first baseline model for SHAP explanations
            baseline_models = [
                m for m in active_models
                if m["model_type"] in ["logistic_regression", "lightgbm"]
            ]
            
            if baseline_models:
                model_id = baseline_models[0]["model_id"]
                model_bundle, _ = self.model_registry.load_model(model_id)
                model = model_bundle["model"]
                
                # Generate SHAP values
                shap_values = self.interpretability_engine.compute_shap_values(
                    model=model,
                    features=features,
                    model_type=baseline_models[0]["model_type"]
                )
                
                # Get top features
                top_features = self.interpretability_engine.get_top_features(
                    shap_values=shap_values,
                    feature_names=list(features.columns),
                    top_k=10
                )
            else:
                top_features = []
            
            # Generate counterfactual (simplified)
            counterfactual = self._generate_simple_counterfactual(
                prediction_result["risk_level"],
                top_features
            )
            
            # Generate rule approximation (simplified)
            rule_approximation = self._generate_simple_rules(
                prediction_result["risk_level"],
                top_features
            )
            
            logger.info(
                "Explanations generated",
                num_top_features=len(top_features)
            )
            
            return {
                "top_features": [f[0] for f in top_features[:10]],
                "feature_importances": {f[0]: float(f[1]) for f in top_features[:10]},
                "counterfactual": counterfactual,
                "rule_approximation": rule_approximation
            }
            
        except Exception as e:
            logger.error(f"Error generating explanations: {e}")
            return {
                "top_features": [],
                "counterfactual": "Explanations unavailable",
                "rule_approximation": "Explanations unavailable"
            }
    
    def _generate_simple_counterfactual(
        self,
        risk_level: str,
        top_features: List[tuple]
    ) -> str:
        """Generate simple counterfactual explanation."""
        if risk_level in ["critical", "high"]:
            if top_features:
                feature_name = top_features[0][0]
                return (
                    f"Improving {feature_name} could reduce risk level. "
                    "Consider interventions targeting this factor."
                )
            return "Multiple factors contribute to elevated risk."
        else:
            return "Current risk level is within acceptable range."
    
    def _generate_simple_rules(
        self,
        risk_level: str,
        top_features: List[tuple]
    ) -> str:
        """Generate simple rule-based explanation."""
        if not top_features:
            return f"Risk classified as {risk_level}"
        
        feature_name = top_features[0][0]
        return f"IF {feature_name} is elevated THEN risk is {risk_level}"

    async def _generate_recommendations(
        self,
        anonymized_id: str,
        prediction_result: Dict[str, Any],
        explanations: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """
        Generate personalized resource recommendations.
        
        Args:
            anonymized_id: Individual identifier
            prediction_result: Prediction results
            explanations: Explanation results
        
        Returns:
            List of resource recommendations
        """
        # Create individual profile
        profile = IndividualProfile(
            anonymized_id=anonymized_id,
            risk_level=prediction_result["risk_level"],
            contributing_factors=explanations.get("top_features", [])
        )
        
        # Get recommendations
        recommendations = self.recommendation_engine.get_recommendations_dict(
            profile=profile,
            max_recommendations=5
        )
        
        logger.info(
            "Recommendations generated",
            anonymized_id=anonymized_id,
            num_recommendations=len(recommendations)
        )
        
        return recommendations
    
    async def _handle_governance(
        self,
        request: ScreeningRequest,
        prediction_result: Dict[str, Any],
        features: pd.DataFrame
    ) -> None:
        """
        Handle governance requirements (human review, alerts).
        
        Args:
            request: ScreeningRequest
            prediction_result: Prediction results
            features: Feature DataFrame
        """
        # Enqueue for human review if needed
        if prediction_result["requires_human_review"]:
            case_id = self.human_review_queue.enqueue_case(
                anonymized_id=request.anonymized_id,
                risk_score=prediction_result["risk_score"],
                risk_level=prediction_result["risk_level"],
                prediction_data=prediction_result,
                features=features.to_dict()
            )
            
            logger.info(
                "Case enqueued for human review",
                anonymized_id=request.anonymized_id,
                case_id=case_id,
                risk_score=prediction_result["risk_score"]
            )
        
        # Check for escalations
        escalated_cases = self.human_review_queue.check_escalations()
        if escalated_cases:
            logger.warning(
                f"Escalated {len(escalated_cases)} overdue review cases"
            )
    
    def get_screening_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about screening service performance.
        
        Returns:
            Dictionary with service statistics
        """
        return {
            "review_queue": self.human_review_queue.get_queue_statistics(),
            "model_registry": {
                "active_models": len(self.model_registry.get_active_models()),
                "total_models": len(self.model_registry.list_models())
            }
        }
    
    def clear_cache(self) -> None:
        """Clear service cache."""
        self._cache.clear()
        self._model_cache.clear()
        self._feature_cache.clear()
        self.consent_verifier.clear_cache()
        logger.info("Service cache cleared")
    
    async def screen_batch(
        self,
        requests: List[ScreeningRequest],
        batch_size: int = 10
    ) -> List[ScreeningResponse]:
        """
        Screen multiple individuals in batches for improved performance.
        
        Args:
            requests: List of ScreeningRequests
            batch_size: Number of requests to process in parallel
        
        Returns:
            List of ScreeningResponses
        """
        logger.info(
            "Starting batch screening",
            num_requests=len(requests),
            batch_size=batch_size
        )
        
        responses = []
        
        # Process in batches
        for i in range(0, len(requests), batch_size):
            batch = requests[i:i + batch_size]
            
            # Process batch in parallel
            batch_tasks = [
                self.screen_individual(request)
                for request in batch
            ]
            
            batch_responses = await asyncio.gather(
                *batch_tasks,
                return_exceptions=True
            )
            
            # Handle results and errors
            for j, response in enumerate(batch_responses):
                if isinstance(response, Exception):
                    logger.error(
                        "Batch screening error",
                        request_index=i + j,
                        error=str(response)
                    )
                    # Create error response
                    error_response = ScreeningResponse(
                        anonymized_id=batch[j].anonymized_id,
                        risk_score=0.0,
                        risk_level="unknown",
                        confidence=0.0,
                        contributing_factors=[],
                        recommendations=[],
                        explanations={"error": str(response)},
                        requires_human_review=True,
                        alert_triggered=False,
                        processing_time_seconds=0.0
                    )
                    responses.append(error_response)
                else:
                    responses.append(response)
        
        logger.info(
            "Batch screening completed",
            num_requests=len(requests),
            num_successful=sum(1 for r in responses if r.risk_level != "unknown")
        )
        
        return responses
    
    def preload_models(self) -> None:
        """
        Preload active models into memory for faster inference.
        """
        logger.info("Preloading active models")
        
        active_models = self.model_registry.get_active_models()
        
        for model_info in active_models:
            model_id = model_info["model_id"]
            try:
                model_bundle, metadata = self.model_registry.load_model(model_id)
                self._model_cache[model_id] = (model_bundle, metadata)
                logger.info(f"Preloaded model {model_id}")
            except Exception as e:
                logger.error(f"Failed to preload model {model_id}: {e}")
        
        logger.info(f"Preloaded {len(self._model_cache)} models")
    
    def optimize_database_queries(self) -> None:
        """
        Optimize database queries by creating indexes.
        
        Note: This is a placeholder. Actual implementation would
        execute SQL commands to create indexes on frequently queried columns.
        """
        logger.info("Database query optimization placeholder")
        
        # In a real implementation, this would execute SQL like:
        # CREATE INDEX IF NOT EXISTS idx_consent_anonymized_id ON consent(anonymized_id);
        # CREATE INDEX IF NOT EXISTS idx_predictions_anonymized_id ON predictions(anonymized_id);
        # CREATE INDEX IF NOT EXISTS idx_audit_log_timestamp ON audit_log(created_at);
        
        logger.info("Database indexes should be created via migrations")


# Synchronous wrapper for backward compatibility
class ScreeningServiceSync:
    """Synchronous wrapper for ScreeningService."""
    
    def __init__(self, *args, **kwargs):
        self.service = ScreeningService(*args, **kwargs)
    
    def screen_individual(
        self,
        request: ScreeningRequest
    ) -> ScreeningResponse:
        """
        Synchronous screening method.
        
        Args:
            request: ScreeningRequest
        
        Returns:
            ScreeningResponse
        """
        return asyncio.run(self.service.screen_individual(request))
    
    def get_screening_statistics(self) -> Dict[str, Any]:
        """Get screening statistics."""
        return self.service.get_screening_statistics()
    
    def clear_cache(self) -> None:
        """Clear service cache."""
        self.service.clear_cache()
    
    def screen_batch(
        self,
        requests: List[ScreeningRequest],
        batch_size: int = 10
    ) -> List[ScreeningResponse]:
        """
        Synchronous batch screening method.
        
        Args:
            requests: List of ScreeningRequests
            batch_size: Batch size for parallel processing
        
        Returns:
            List of ScreeningResponses
        """
        return asyncio.run(self.service.screen_batch(requests, batch_size))
    
    def preload_models(self) -> None:
        """Preload active models."""
        self.service.preload_models()
    
    def optimize_database_queries(self) -> None:
        """Optimize database queries."""
        self.service.optimize_database_queries()
