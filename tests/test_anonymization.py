"""Tests for data anonymization"""

import pytest
from src.governance.anonymization import Anonymizer


class TestAnonymizer:
    """Test Anonymizer class"""
    
    def test_hash_identifier_consistency(self):
        """Test that same identifier produces same hash"""
        anonymizer = Anonymizer(salt="test_salt")
        
        hash1 = anonymizer.hash_identifier("user123")
        hash2 = anonymizer.hash_identifier("user123")
        
        assert hash1 == hash2
        assert len(hash1) == 64  # SHA-256 produces 64 hex characters
    
    def test_hash_identifier_different_values(self):
        """Test that different identifiers produce different hashes"""
        anonymizer = Anonymizer(salt="test_salt")
        
        hash1 = anonymizer.hash_identifier("user123")
        hash2 = anonymizer.hash_identifier("user456")
        
        assert hash1 != hash2
    
    def test_hash_identifier_empty_string(self):
        """Test hashing empty string"""
        anonymizer = Anonymizer(salt="test_salt")
        
        result = anonymizer.hash_identifier("")
        
        assert result == ""
    
    def test_anonymize_record_basic(self):
        """Test basic record anonymization"""
        anonymizer = Anonymizer(salt="test_salt")
        
        data = {
            "user_id": "12345",
            "name": "John Doe",
            "age": 30,
            "score": 85.5
        }
        
        result = anonymizer.anonymize_record(data, ["user_id", "name"])
        
        assert result["user_id"] != "12345"
        assert result["name"] != "John Doe"
        assert result["age"] == 30  # Not in PII fields
        assert result["score"] == 85.5  # Not in PII fields
        assert len(result["user_id"]) == 64
        assert len(result["name"]) == 64
    
    def test_anonymize_record_with_none(self):
        """Test anonymizing record with None values"""
        anonymizer = Anonymizer(salt="test_salt")
        
        data = {
            "user_id": "12345",
            "name": None,
            "email": "test@example.com"
        }
        
        result = anonymizer.anonymize_record(data, ["user_id", "name", "email"])
        
        assert result["user_id"] != "12345"
        assert result["name"] is None  # None values preserved
        assert result["email"] != "test@example.com"
    
    def test_anonymize_text_email(self):
        """Test redacting email addresses from text"""
        anonymizer = Anonymizer()
        
        text = "Contact me at john.doe@example.com for more info"
        result = anonymizer.anonymize_text(text)
        
        assert "john.doe@example.com" not in result
        assert "[REDACTED]" in result
    
    def test_anonymize_text_phone(self):
        """Test redacting phone numbers from text"""
        anonymizer = Anonymizer()
        
        text = "Call me at 555-123-4567 or (555) 987-6543"
        result = anonymizer.anonymize_text(text)
        
        assert "555-123-4567" not in result
        assert "555-987-6543" not in result
        assert "[REDACTED]" in result
    
    def test_anonymize_text_multiple_patterns(self):
        """Test redacting multiple PII patterns"""
        anonymizer = Anonymizer()
        
        text = "Email: test@example.com, Phone: 555-1234, SSN: 123-45-6789"
        result = anonymizer.anonymize_text(text)
        
        assert "test@example.com" not in result
        assert "123-45-6789" not in result
        assert result.count("[REDACTED]") >= 2
    
    def test_anonymize_text_custom_replacement(self):
        """Test using custom replacement string"""
        anonymizer = Anonymizer()
        
        text = "Contact: john@example.com"
        result = anonymizer.anonymize_text(text, replacement="***")
        
        assert "john@example.com" not in result
        assert "***" in result
    
    def test_anonymize_text_specific_patterns(self):
        """Test redacting only specific patterns"""
        anonymizer = Anonymizer()
        
        text = "Email: test@example.com, Phone: 555-1234"
        result = anonymizer.anonymize_text(text, redact_patterns=["email"])
        
        assert "test@example.com" not in result
        assert "555-1234" in result  # Phone not redacted
    
    def test_anonymize_text_empty(self):
        """Test anonymizing empty text"""
        anonymizer = Anonymizer()
        
        result = anonymizer.anonymize_text("")
        
        assert result == ""
    
    def test_create_anonymized_id_single(self):
        """Test creating anonymized ID from single identifier"""
        anonymizer = Anonymizer(salt="test_salt")
        
        anon_id = anonymizer.create_anonymized_id("user123")
        
        assert len(anon_id) == 64
        assert anon_id == anonymizer.hash_identifier("user123")
    
    def test_create_anonymized_id_multiple(self):
        """Test creating composite anonymized ID"""
        anonymizer = Anonymizer(salt="test_salt")
        
        anon_id = anonymizer.create_anonymized_id("user123", "session456")
        
        assert len(anon_id) == 64
        # Should be different from individual hashes
        assert anon_id != anonymizer.hash_identifier("user123")
        assert anon_id != anonymizer.hash_identifier("session456")
    
    def test_verify_anonymization_valid(self):
        """Test verifying properly anonymized data"""
        anonymizer = Anonymizer(salt="test_salt")
        
        data = {
            "user_id": "12345",
            "email": "test@example.com"
        }
        
        anonymized = anonymizer.anonymize_record(data, ["user_id", "email"])
        is_valid = anonymizer.verify_anonymization(anonymized, ["user_id", "email"])
        
        assert is_valid
    
    def test_verify_anonymization_invalid(self):
        """Test detecting non-anonymized data"""
        anonymizer = Anonymizer(salt="test_salt")
        
        data = {
            "user_id": "12345",  # Not anonymized
            "email": "test@example.com"
        }
        
        is_valid = anonymizer.verify_anonymization(data, ["user_id", "email"])
        
        assert not is_valid
    
    def test_anonymize_nested_record(self):
        """Test anonymizing nested data structures"""
        anonymizer = Anonymizer(salt="test_salt")
        
        data = {
            "user": {
                "id": "12345",
                "profile": {
                    "email": "test@example.com"
                }
            },
            "session_id": "session789"
        }
        
        result = anonymizer.anonymize_nested_record(
            data,
            pii_fields=["user.id", "session_id"],
            text_fields=["user.profile.email"]
        )
        
        assert result["user"]["id"] != "12345"
        assert len(result["user"]["id"]) == 64
        assert result["session_id"] != "session789"
        assert "test@example.com" not in result["user"]["profile"]["email"]
    
    def test_different_salts_produce_different_hashes(self):
        """Test that different salts produce different hashes"""
        anonymizer1 = Anonymizer(salt="salt1")
        anonymizer2 = Anonymizer(salt="salt2")
        
        hash1 = anonymizer1.hash_identifier("user123")
        hash2 = anonymizer2.hash_identifier("user123")
        
        assert hash1 != hash2
