"""Tests for data validation module"""

import pytest
from datetime import datetime
from pathlib import Path

from src.ingestion.validation import DataValidator, ValidationResult, DataSourceType
from src.exceptions import ValidationError


@pytest.fixture
def validator():
    """Create DataValidator instance for testing"""
    return DataValidator()


class TestDataValidator:
    """Test suite for DataValidator class"""
    
    def test_initialization(self, validator):
        """Test validator initializes correctly"""
        assert validator is not None
        assert validator.schema_dir.exists()
        assert len(validator._schema_cache) > 0
    
    def test_get_schema_survey(self, validator):
        """Test loading survey schema"""
        schema = validator.get_schema("survey")
        assert schema is not None
        assert schema["title"] == "Survey Data Schema"
        assert "properties" in schema
    
    def test_get_schema_wearable(self, validator):
        """Test loading wearable schema"""
        schema = validator.get_schema("wearable")
        assert schema is not None
        assert schema["title"] == "Wearable Data Schema"
        assert "properties" in schema
    
    def test_get_schema_emr(self, validator):
        """Test loading EMR schema"""
        schema = validator.get_schema("emr")
        assert schema is not None
        assert schema["title"] == "EMR Data Schema"
        assert "properties" in schema
    
    def test_get_schema_invalid_type(self, validator):
        """Test error handling for invalid schema type"""
        with pytest.raises(ValidationError) as exc_info:
            validator.get_schema("invalid_type")
        assert "not found" in str(exc_info.value)
    
    def test_schema_caching(self, validator):
        """Test that schemas are cached after first load"""
        schema1 = validator.get_schema("survey")
        schema2 = validator.get_schema("survey")
        assert schema1 is schema2  # Same object reference


class TestSurveyValidation:
    """Test suite for survey data validation"""
    
    def test_valid_survey_data(self, validator):
        """Test validation of valid survey data"""
        data = {
            "survey_id": "survey_123",
            "timestamp": "2025-11-17T10:30:00Z",
            "responses": {
                "mood_score": 7,
                "anxiety_level": 5,
                "depression_level": 8,
                "stress_level": 6,
                "sleep_quality": 3,
                "social_support": 4,
                "suicidal_ideation": False,
                "self_harm": False,
                "substance_use": "none",
                "medication_adherence": "good",
                "therapy_engagement": 85,
                "life_events": ["job_change"],
                "free_text_response": "Feeling better this week"
            },
            "completion_time_seconds": 180,
            "device_type": "mobile"
        }
        
        result = validator.validate_survey(data)
        assert result.is_valid is True
        assert len(result.errors) == 0
        assert result.source_type == "survey"
        assert result.validation_time_ms < 100
    
    def test_minimal_valid_survey(self, validator):
        """Test validation with only required fields"""
        data = {
            "survey_id": "survey_456",
            "timestamp": "2025-11-17T10:30:00Z",
            "responses": {}
        }
        
        result = validator.validate_survey(data)
        assert result.is_valid is True
        assert "no responses" in " ".join(result.warnings).lower()
    
    def test_survey_missing_required_field(self, validator):
        """Test validation fails when required field is missing"""
        data = {
            "survey_id": "survey_789",
            # Missing timestamp
            "responses": {
                "mood_score": 5
            }
        }
        
        result = validator.validate_survey(data)
        assert result.is_valid is False
        assert len(result.errors) > 0
        assert any("timestamp" in error.lower() for error in result.errors)
    
    def test_survey_invalid_mood_score(self, validator):
        """Test validation fails for out-of-range mood score"""
        data = {
            "survey_id": "survey_999",
            "timestamp": "2025-11-17T10:30:00Z",
            "responses": {
                "mood_score": 15  # Invalid: max is 10
            }
        }
        
        result = validator.validate_survey(data)
        assert result.is_valid is False
        assert any("mood_score" in error.lower() for error in result.errors)
    
    def test_survey_invalid_enum_value(self, validator):
        """Test validation fails for invalid enum value"""
        data = {
            "survey_id": "survey_enum",
            "timestamp": "2025-11-17T10:30:00Z",
            "responses": {
                "substance_use": "invalid_value"  # Not in enum
            }
        }
        
        result = validator.validate_survey(data)
        assert result.is_valid is False
        assert any("substance_use" in error.lower() for error in result.errors)
    
    def test_survey_suicidal_ideation_warning(self, validator):
        """Test warning is generated for suicidal ideation"""
        data = {
            "survey_id": "survey_critical",
            "timestamp": "2025-11-17T10:30:00Z",
            "responses": {
                "suicidal_ideation": True,
                "mood_score": 2
            }
        }
        
        result = validator.validate_survey(data)
        assert result.is_valid is True
        assert any("suicidal ideation" in warning.lower() for warning in result.warnings)


class TestWearableValidation:
    """Test suite for wearable data validation"""
    
    def test_valid_wearable_data(self, validator):
        """Test validation of valid wearable data"""
        data = {
            "device_id": "device_abc123",
            "device_type": "smartwatch",
            "start_timestamp": "2025-11-17T00:00:00Z",
            "end_timestamp": "2025-11-17T23:59:59Z",
            "metrics": {
                "heart_rate": {
                    "average_bpm": 72,
                    "resting_bpm": 58,
                    "max_bpm": 145,
                    "min_bpm": 52
                },
                "hrv": {
                    "rmssd": 45.5,
                    "sdnn": 62.3,
                    "lf_hf_ratio": 1.8
                },
                "sleep": {
                    "total_minutes": 420,
                    "deep_sleep_minutes": 90,
                    "rem_sleep_minutes": 105,
                    "light_sleep_minutes": 210,
                    "awake_minutes": 15,
                    "interruptions": 2,
                    "efficiency_percent": 96.4,
                    "bedtime": "2025-11-16T23:00:00Z",
                    "wake_time": "2025-11-17T07:00:00Z"
                },
                "activity": {
                    "steps": 8500,
                    "distance_meters": 6800,
                    "calories_burned": 2200,
                    "active_minutes": 45,
                    "sedentary_minutes": 480,
                    "intensity_zones": {
                        "light_minutes": 30,
                        "moderate_minutes": 15,
                        "vigorous_minutes": 5
                    }
                },
                "stress_score": 35,
                "spo2_percent": 98.5,
                "skin_temperature_celsius": 36.2
            },
            "data_quality": {
                "completeness_percent": 95.5,
                "wear_time_minutes": 1380
            }
        }
        
        result = validator.validate_wearable(data)
        assert result.is_valid is True
        assert len(result.errors) == 0
        assert result.source_type == "wearable"
        assert result.validation_time_ms < 100
    
    def test_wearable_missing_required_field(self, validator):
        """Test validation fails when required field is missing"""
        data = {
            "device_id": "device_xyz",
            "start_timestamp": "2025-11-17T00:00:00Z",
            # Missing end_timestamp
            "metrics": {}
        }
        
        result = validator.validate_wearable(data)
        assert result.is_valid is False
        assert any("end_timestamp" in error.lower() for error in result.errors)
    
    def test_wearable_invalid_heart_rate(self, validator):
        """Test validation fails for invalid heart rate"""
        data = {
            "device_id": "device_hr",
            "start_timestamp": "2025-11-17T00:00:00Z",
            "end_timestamp": "2025-11-17T23:59:59Z",
            "metrics": {
                "heart_rate": {
                    "average_bpm": 250  # Invalid: max is 220
                }
            }
        }
        
        result = validator.validate_wearable(data)
        assert result.is_valid is False
        assert any("average_bpm" in error.lower() for error in result.errors)
    
    def test_wearable_low_completeness_warning(self, validator):
        """Test warning for low data completeness"""
        data = {
            "device_id": "device_incomplete",
            "start_timestamp": "2025-11-17T00:00:00Z",
            "end_timestamp": "2025-11-17T23:59:59Z",
            "metrics": {},
            "data_quality": {
                "completeness_percent": 50.0,
                "wear_time_minutes": 600
            }
        }
        
        result = validator.validate_wearable(data)
        assert result.is_valid is True
        assert any("completeness" in warning.lower() for warning in result.warnings)
        assert any("wear time" in warning.lower() for warning in result.warnings)


class TestEMRValidation:
    """Test suite for EMR data validation"""
    
    def test_valid_emr_data(self, validator):
        """Test validation of valid EMR data"""
        data = {
            "record_id": "emr_12345",
            "timestamp": "2025-11-17T10:30:00Z",
            "record_type": "assessment",
            "diagnoses": [
                {
                    "code": "F32.1",
                    "description": "Major depressive disorder, single episode, moderate",
                    "date_diagnosed": "2025-01-15",
                    "status": "active",
                    "severity": "moderate"
                }
            ],
            "medications": [
                {
                    "name": "Sertraline",
                    "dosage": "50mg",
                    "frequency": "once daily",
                    "start_date": "2025-01-20",
                    "status": "active",
                    "prescriber": "Dr. Smith"
                }
            ],
            "encounters": [
                {
                    "date": "2025-11-10T14:00:00Z",
                    "type": "therapy_session",
                    "provider_type": "therapist",
                    "duration_minutes": 50,
                    "chief_complaint": "Follow-up therapy session",
                    "disposition": "follow_up_scheduled"
                }
            ],
            "assessments": [
                {
                    "date": "2025-11-10",
                    "type": "mental_status",
                    "score": 12,
                    "interpretation": "moderate",
                    "notes": "Patient showing improvement"
                }
            ],
            "risk_factors": {
                "suicide_attempts": 0,
                "self_harm_history": False,
                "family_history_mental_illness": True,
                "substance_abuse_history": False,
                "trauma_history": False
            }
        }
        
        result = validator.validate_emr(data)
        assert result.is_valid is True
        assert len(result.errors) == 0
        assert result.source_type == "emr"
        assert result.validation_time_ms < 100
    
    def test_emr_missing_required_field(self, validator):
        """Test validation fails when required field is missing"""
        data = {
            "record_id": "emr_67890",
            "timestamp": "2025-11-17T10:30:00Z"
            # Missing record_type
        }
        
        result = validator.validate_emr(data)
        assert result.is_valid is False
        assert any("record_type" in error.lower() for error in result.errors)
    
    def test_emr_invalid_diagnosis_code(self, validator):
        """Test validation fails for invalid diagnosis code format"""
        data = {
            "record_id": "emr_diag",
            "timestamp": "2025-11-17T10:30:00Z",
            "record_type": "diagnosis",
            "diagnoses": [
                {
                    "code": "invalid code!",  # Invalid format
                    "description": "Test diagnosis"
                }
            ]
        }
        
        result = validator.validate_emr(data)
        assert result.is_valid is False
        assert any("code" in error.lower() for error in result.errors)
    
    def test_emr_suicide_attempt_warning(self, validator):
        """Test warning for suicide attempt history"""
        data = {
            "record_id": "emr_risk",
            "timestamp": "2025-11-17T10:30:00Z",
            "record_type": "assessment",
            "risk_factors": {
                "suicide_attempts": 2,
                "self_harm_history": True
            }
        }
        
        result = validator.validate_emr(data)
        assert result.is_valid is True
        assert any("suicide attempts" in warning.lower() for warning in result.warnings)
    
    def test_emr_involuntary_hospitalization_warning(self, validator):
        """Test warning for involuntary hospitalization"""
        data = {
            "record_id": "emr_hosp",
            "timestamp": "2025-11-17T10:30:00Z",
            "record_type": "encounter",
            "hospitalizations": [
                {
                    "admission_date": "2025-01-01T00:00:00Z",
                    "discharge_date": "2025-01-05T00:00:00Z",
                    "reason": "Acute psychiatric crisis",
                    "facility_type": "psychiatric",
                    "involuntary": True
                }
            ]
        }
        
        result = validator.validate_emr(data)
        assert result.is_valid is True
        assert any("involuntary" in warning.lower() for warning in result.warnings)


class TestGenericValidation:
    """Test suite for generic validation method"""
    
    def test_validate_with_source_type_survey(self, validator):
        """Test generic validate method with survey type"""
        data = {
            "survey_id": "test",
            "timestamp": "2025-11-17T10:30:00Z",
            "responses": {}
        }
        
        result = validator.validate(data, "survey")
        assert result.is_valid is True
        assert result.source_type == "survey"
    
    def test_validate_with_source_type_wearable(self, validator):
        """Test generic validate method with wearable type"""
        data = {
            "device_id": "test",
            "start_timestamp": "2025-11-17T00:00:00Z",
            "end_timestamp": "2025-11-17T23:59:59Z",
            "metrics": {}
        }
        
        result = validator.validate(data, "wearable")
        assert result.is_valid is True
        assert result.source_type == "wearable"
    
    def test_validate_with_source_type_emr(self, validator):
        """Test generic validate method with EMR type"""
        data = {
            "record_id": "test",
            "timestamp": "2025-11-17T10:30:00Z",
            "record_type": "assessment"
        }
        
        result = validator.validate(data, "emr")
        assert result.is_valid is True
        assert result.source_type == "emr"
    
    def test_validate_with_invalid_source_type(self, validator):
        """Test generic validate method with invalid source type"""
        data = {"test": "data"}
        
        with pytest.raises(ValidationError) as exc_info:
            validator.validate(data, "invalid")
        
        assert "invalid source type" in str(exc_info.value).lower()


class TestValidationResult:
    """Test suite for ValidationResult dataclass"""
    
    def test_validation_result_to_dict(self):
        """Test converting ValidationResult to dictionary"""
        result = ValidationResult(
            is_valid=True,
            errors=[],
            warnings=["Test warning"],
            validation_time_ms=25.5,
            source_type="survey"
        )
        
        result_dict = result.to_dict()
        assert result_dict["is_valid"] is True
        assert result_dict["errors"] == []
        assert result_dict["warnings"] == ["Test warning"]
        assert result_dict["validation_time_ms"] == 25.5
        assert result_dict["source_type"] == "survey"


class TestPerformance:
    """Test suite for validation performance"""
    
    def test_validation_performance_survey(self, validator):
        """Test survey validation completes within 100ms"""
        data = {
            "survey_id": "perf_test",
            "timestamp": "2025-11-17T10:30:00Z",
            "responses": {
                "mood_score": 5,
                "anxiety_level": 10,
                "depression_level": 15,
                "stress_level": 7,
                "sleep_quality": 3,
                "social_support": 4,
                "suicidal_ideation": False,
                "self_harm": False,
                "substance_use": "occasional",
                "medication_adherence": "moderate",
                "therapy_engagement": 70,
                "life_events": ["job_change", "relationship_change"],
                "free_text_response": "This is a longer text response to test performance"
            }
        }
        
        result = validator.validate_survey(data)
        assert result.validation_time_ms < 100
    
    def test_validation_performance_wearable(self, validator):
        """Test wearable validation completes within 100ms"""
        data = {
            "device_id": "perf_device",
            "device_type": "smartwatch",
            "start_timestamp": "2025-11-17T00:00:00Z",
            "end_timestamp": "2025-11-17T23:59:59Z",
            "metrics": {
                "heart_rate": {"average_bpm": 72, "resting_bpm": 58},
                "hrv": {"rmssd": 45.5, "sdnn": 62.3},
                "sleep": {"total_minutes": 420, "efficiency_percent": 95.0},
                "activity": {"steps": 8500, "active_minutes": 45}
            }
        }
        
        result = validator.validate_wearable(data)
        assert result.validation_time_ms < 100
    
    def test_validation_performance_emr(self, validator):
        """Test EMR validation completes within 100ms"""
        data = {
            "record_id": "perf_emr",
            "timestamp": "2025-11-17T10:30:00Z",
            "record_type": "assessment",
            "diagnoses": [
                {"code": "F32.1", "description": "Depression", "status": "active"}
            ],
            "medications": [
                {"name": "Sertraline", "status": "active"}
            ]
        }
        
        result = validator.validate_emr(data)
        assert result.validation_time_ms < 100
