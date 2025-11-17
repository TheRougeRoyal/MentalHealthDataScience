"""Data validation module for MHRAS ingestion layer"""

import json
import time
from pathlib import Path
from typing import Dict, List, Optional
from dataclasses import dataclass
from enum import Enum

import jsonschema
from jsonschema import Draft7Validator

from src.exceptions import ValidationError


class DataSourceType(Enum):
    """Enumeration of supported data source types"""
    SURVEY = "survey"
    WEARABLE = "wearable"
    EMR = "emr"


@dataclass
class ValidationResult:
    """Result of data validation"""
    is_valid: bool
    errors: List[str]
    warnings: List[str]
    validation_time_ms: float
    source_type: str
    
    def to_dict(self) -> Dict:
        """Convert validation result to dictionary"""
        return {
            "is_valid": self.is_valid,
            "errors": self.errors,
            "warnings": self.warnings,
            "validation_time_ms": self.validation_time_ms,
            "source_type": self.source_type
        }


class DataValidator:
    """
    Validates incoming data against predefined JSON schemas.
    
    Responsibilities:
    - Load and cache JSON schemas for each data source type
    - Validate required fields, data types, and value ranges
    - Return detailed error messages for validation failures
    - Complete validation within 100ms per record
    """
    
    def __init__(self, schema_dir: Optional[Path] = None):
        """
        Initialize DataValidator with schema directory.
        
        Args:
            schema_dir: Directory containing JSON schema files.
                       Defaults to src/ingestion/schemas/
        """
        if schema_dir is None:
            # Default to schemas directory relative to this file
            schema_dir = Path(__file__).parent / "schemas"
        
        self.schema_dir = Path(schema_dir)
        self._schema_cache: Dict[str, Dict] = {}
        self._validator_cache: Dict[str, Draft7Validator] = {}
        
        # Pre-load schemas on initialization
        self._load_all_schemas()
    
    def _load_all_schemas(self) -> None:
        """Pre-load all schemas into cache"""
        for source_type in DataSourceType:
            try:
                self.get_schema(source_type.value)
            except Exception as e:
                # Log warning but don't fail initialization
                print(f"Warning: Could not load schema for {source_type.value}: {e}")
    
    def get_schema(self, source_type: str) -> Dict:
        """
        Load and cache JSON schema for a data source type.
        
        Args:
            source_type: Type of data source (survey, wearable, emr)
            
        Returns:
            JSON schema dictionary
            
        Raises:
            ValidationError: If schema file not found or invalid
        """
        # Check cache first
        if source_type in self._schema_cache:
            return self._schema_cache[source_type]
        
        # Load schema from file
        schema_file = self.schema_dir / f"{source_type}_schema.json"
        
        if not schema_file.exists():
            raise ValidationError(
                f"Schema file not found for source type: {source_type}",
                details={"schema_file": str(schema_file)}
            )
        
        try:
            with open(schema_file, 'r') as f:
                schema = json.load(f)
            
            # Validate that the schema itself is valid
            Draft7Validator.check_schema(schema)
            
            # Cache the schema and validator
            self._schema_cache[source_type] = schema
            self._validator_cache[source_type] = Draft7Validator(schema)
            
            return schema
            
        except json.JSONDecodeError as e:
            raise ValidationError(
                f"Invalid JSON in schema file: {schema_file}",
                details={"error": str(e)}
            )
        except jsonschema.SchemaError as e:
            raise ValidationError(
                f"Invalid JSON schema for {source_type}",
                details={"error": str(e)}
            )
    
    def _validate_against_schema(
        self, 
        data: Dict, 
        source_type: str
    ) -> ValidationResult:
        """
        Internal method to validate data against schema.
        
        Args:
            data: Data dictionary to validate
            source_type: Type of data source
            
        Returns:
            ValidationResult with validation outcome
        """
        start_time = time.time()
        errors = []
        warnings = []
        
        try:
            # Get cached validator
            if source_type not in self._validator_cache:
                self.get_schema(source_type)
            
            validator = self._validator_cache[source_type]
            
            # Validate data
            validation_errors = list(validator.iter_errors(data))
            
            if validation_errors:
                for error in validation_errors:
                    # Build detailed error message
                    path = ".".join(str(p) for p in error.path) if error.path else "root"
                    error_msg = f"Field '{path}': {error.message}"
                    errors.append(error_msg)
            
            # Check for additional warnings (e.g., missing optional fields)
            self._check_data_quality_warnings(data, source_type, warnings)
            
        except Exception as e:
            errors.append(f"Validation error: {str(e)}")
        
        end_time = time.time()
        validation_time_ms = (end_time - start_time) * 1000
        
        return ValidationResult(
            is_valid=len(errors) == 0,
            errors=errors,
            warnings=warnings,
            validation_time_ms=validation_time_ms,
            source_type=source_type
        )
    
    def _check_data_quality_warnings(
        self, 
        data: Dict, 
        source_type: str, 
        warnings: List[str]
    ) -> None:
        """
        Check for data quality issues that don't fail validation.
        
        Args:
            data: Data dictionary
            source_type: Type of data source
            warnings: List to append warnings to
        """
        # Source-specific quality checks
        if source_type == DataSourceType.SURVEY.value:
            responses = data.get("responses", {})
            if not responses:
                warnings.append("Survey has no responses")
            
            # Check for critical indicators
            if responses.get("suicidal_ideation") is True:
                warnings.append("Critical: Suicidal ideation reported")
            if responses.get("self_harm") is True:
                warnings.append("Critical: Self-harm reported")
        
        elif source_type == DataSourceType.WEARABLE.value:
            metrics = data.get("metrics", {})
            data_quality = data.get("data_quality", {})
            
            # Check data completeness
            completeness = data_quality.get("completeness_percent", 100)
            if completeness < 70:
                warnings.append(f"Low data completeness: {completeness}%")
            
            # Check wear time
            wear_time = data_quality.get("wear_time_minutes", 1440)
            if wear_time < 720:  # Less than 12 hours
                warnings.append(f"Low wear time: {wear_time} minutes")
        
        elif source_type == DataSourceType.EMR.value:
            # Check for high-risk indicators
            risk_factors = data.get("risk_factors", {})
            if risk_factors.get("suicide_attempts", 0) > 0:
                warnings.append("Critical: History of suicide attempts")
            
            hospitalizations = data.get("hospitalizations", [])
            involuntary_admissions = [h for h in hospitalizations if h.get("involuntary")]
            if involuntary_admissions:
                warnings.append("Critical: History of involuntary hospitalization")
    
    def validate_survey(self, data: Dict) -> ValidationResult:
        """
        Validate survey response data.
        
        Args:
            data: Survey data dictionary
            
        Returns:
            ValidationResult with validation outcome
            
        Raises:
            ValidationError: If validation takes longer than 100ms
        """
        result = self._validate_against_schema(data, DataSourceType.SURVEY.value)
        
        if result.validation_time_ms > 100:
            raise ValidationError(
                f"Validation exceeded time limit: {result.validation_time_ms:.2f}ms",
                details={"limit_ms": 100, "actual_ms": result.validation_time_ms}
            )
        
        return result
    
    def validate_wearable(self, data: Dict) -> ValidationResult:
        """
        Validate wearable device data.
        
        Args:
            data: Wearable data dictionary
            
        Returns:
            ValidationResult with validation outcome
            
        Raises:
            ValidationError: If validation takes longer than 100ms
        """
        result = self._validate_against_schema(data, DataSourceType.WEARABLE.value)
        
        if result.validation_time_ms > 100:
            raise ValidationError(
                f"Validation exceeded time limit: {result.validation_time_ms:.2f}ms",
                details={"limit_ms": 100, "actual_ms": result.validation_time_ms}
            )
        
        return result
    
    def validate_emr(self, data: Dict) -> ValidationResult:
        """
        Validate Electronic Medical Records data.
        
        Args:
            data: EMR data dictionary
            
        Returns:
            ValidationResult with validation outcome
            
        Raises:
            ValidationError: If validation takes longer than 100ms
        """
        result = self._validate_against_schema(data, DataSourceType.EMR.value)
        
        if result.validation_time_ms > 100:
            raise ValidationError(
                f"Validation exceeded time limit: {result.validation_time_ms:.2f}ms",
                details={"limit_ms": 100, "actual_ms": result.validation_time_ms}
            )
        
        return result
    
    def validate(self, data: Dict, source_type: str) -> ValidationResult:
        """
        Generic validation method that routes to specific validator.
        
        Args:
            data: Data dictionary to validate
            source_type: Type of data source (survey, wearable, emr)
            
        Returns:
            ValidationResult with validation outcome
            
        Raises:
            ValidationError: If source_type is invalid
        """
        try:
            source_enum = DataSourceType(source_type)
        except ValueError:
            raise ValidationError(
                f"Invalid source type: {source_type}",
                details={
                    "valid_types": [t.value for t in DataSourceType]
                }
            )
        
        if source_enum == DataSourceType.SURVEY:
            return self.validate_survey(data)
        elif source_enum == DataSourceType.WEARABLE:
            return self.validate_wearable(data)
        elif source_enum == DataSourceType.EMR:
            return self.validate_emr(data)
