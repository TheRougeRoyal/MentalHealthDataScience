"""Configuration management using Pydantic settings"""

from typing import List, Optional
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class DatabaseConfig(BaseSettings):
    """Database configuration"""
    
    host: str = Field(default="localhost", description="Database host")
    port: int = Field(default=5432, description="Database port")
    name: str = Field(default="mhras", description="Database name")
    user: str = Field(default="mhras_user", description="Database user")
    password: str = Field(default="", description="Database password")
    pool_size: int = Field(default=10, description="Connection pool size")
    
    model_config = SettingsConfigDict(
        env_prefix="DB_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )


class APIConfig(BaseSettings):
    """API configuration"""
    
    host: str = Field(default="0.0.0.0", description="API host")
    port: int = Field(default=8000, description="API port")
    workers: int = Field(default=4, description="Number of workers")
    timeout: int = Field(default=5, description="Request timeout in seconds")
    max_request_size: int = Field(default=10485760, description="Max request size in bytes (10MB)")
    
    model_config = SettingsConfigDict(
        env_prefix="API_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )


class MLConfig(BaseSettings):
    """Machine learning configuration"""
    
    model_storage_path: str = Field(default="models/", description="Path to model storage")
    inference_timeout: int = Field(default=2, description="Inference timeout in seconds")
    ensemble_weights: Optional[str] = Field(default=None, description="JSON string of ensemble weights")
    risk_threshold_high: float = Field(default=51.0, description="High risk threshold")
    risk_threshold_critical: float = Field(default=75.0, description="Critical risk threshold")
    
    model_config = SettingsConfigDict(
        env_prefix="ML_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )


class LoggingConfig(BaseSettings):
    """Logging configuration"""
    
    level: str = Field(default="INFO", description="Logging level")
    format: str = Field(default="json", description="Log format (json or text)")
    output: str = Field(default="stdout", description="Log output (stdout or file path)")
    
    model_config = SettingsConfigDict(
        env_prefix="LOG_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )


class SecurityConfig(BaseSettings):
    """Security configuration"""
    
    jwt_secret: str = Field(default="change-me-in-production", description="JWT secret key")
    jwt_algorithm: str = Field(default="HS256", description="JWT algorithm")
    jwt_expiry_hours: int = Field(default=24, description="JWT expiry in hours")
    anonymization_salt: str = Field(default="change-me-in-production", description="Salt for anonymization")
    
    model_config = SettingsConfigDict(
        env_prefix="SECURITY_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )


class GovernanceConfig(BaseSettings):
    """Governance configuration"""
    
    audit_log_retention_days: int = Field(default=90, description="Audit log retention in days")
    human_review_threshold: float = Field(default=75.0, description="Risk score threshold for human review")
    review_escalation_hours: int = Field(default=4, description="Hours before review escalation")
    drift_threshold: float = Field(default=0.3, description="Drift detection threshold")
    
    model_config = SettingsConfigDict(
        env_prefix="GOVERNANCE_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )


class ExperimentTrackingConfig(BaseSettings):
    """Experiment tracking configuration"""
    
    storage_backend: str = Field(default="filesystem", description="Storage backend (filesystem or s3)")
    artifacts_path: str = Field(default="experiments/artifacts", description="Path to artifacts storage")
    db_table_prefix: str = Field(default="", description="Database table prefix for experiment tables")
    auto_log_git_commit: bool = Field(default=True, description="Automatically log git commit hash")
    
    model_config = SettingsConfigDict(
        env_prefix="EXPERIMENT_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )


class DataVersioningConfig(BaseSettings):
    """Data versioning configuration"""
    
    storage_path: str = Field(default="data/versions", description="Path to versioned data storage")
    compression: str = Field(default="gzip", description="Compression algorithm (gzip, bz2, xz, none)")
    deduplication: bool = Field(default=True, description="Enable dataset deduplication")
    max_dataset_size_mb: int = Field(default=1000, description="Maximum dataset size in MB")
    
    model_config = SettingsConfigDict(
        env_prefix="DATA_VERSION_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )


class FeatureStoreConfig(BaseSettings):
    """Feature store configuration"""
    
    cache_backend: str = Field(default="memory", description="Cache backend (memory, redis)")
    redis_url: str = Field(default="redis://localhost:6379", description="Redis connection URL")
    cache_ttl: int = Field(default=3600, description="Cache TTL in seconds")
    online_serving_timeout_ms: int = Field(default=100, description="Online serving timeout in milliseconds")
    enable_caching: bool = Field(default=True, description="Enable feature caching")
    
    model_config = SettingsConfigDict(
        env_prefix="FEATURE_STORE_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )


class EDAConfig(BaseSettings):
    """EDA module configuration"""
    
    max_dataset_size: int = Field(default=1000000, description="Maximum dataset size in rows for EDA")
    visualization_dpi: int = Field(default=300, description="DPI for generated visualizations")
    report_template: str = Field(default="default", description="Report template name")
    
    model_config = SettingsConfigDict(
        env_prefix="EDA_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )


class ModelCardConfig(BaseSettings):
    """Model card configuration"""
    
    template: str = Field(default="default", description="Model card template name")
    include_shap: bool = Field(default=True, description="Include SHAP visualizations")
    include_fairness: bool = Field(default=True, description="Include fairness metrics")
    
    model_config = SettingsConfigDict(
        env_prefix="MODEL_CARD_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )


class Settings(BaseSettings):
    """Main application settings"""
    
    environment: str = Field(default="development", description="Environment (development, staging, production)")
    debug: bool = Field(default=False, description="Debug mode")
    
    # Sub-configurations
    database: DatabaseConfig = Field(default_factory=DatabaseConfig)
    api: APIConfig = Field(default_factory=APIConfig)
    ml: MLConfig = Field(default_factory=MLConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)
    security: SecurityConfig = Field(default_factory=SecurityConfig)
    governance: GovernanceConfig = Field(default_factory=GovernanceConfig)
    experiment_tracking: ExperimentTrackingConfig = Field(default_factory=ExperimentTrackingConfig)
    data_versioning: DataVersioningConfig = Field(default_factory=DataVersioningConfig)
    feature_store: FeatureStoreConfig = Field(default_factory=FeatureStoreConfig)
    eda: EDAConfig = Field(default_factory=EDAConfig)
    model_card: ModelCardConfig = Field(default_factory=ModelCardConfig)
    
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )
    
    @property
    def DATABASE_URL(self) -> str:
        """Construct PostgreSQL connection URL from database config."""
        return (
            f"postgresql://{self.database.user}:{self.database.password}"
            f"@{self.database.host}:{self.database.port}/{self.database.name}"
        )


# Global settings instance
settings = Settings()
