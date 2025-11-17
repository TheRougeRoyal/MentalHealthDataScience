"""Tests for consent verification"""

import pytest
from datetime import datetime, timedelta
from src.governance.consent import ConsentVerifier, ConsentResult, ConsentStatus
from src.exceptions import ConsentError


class TestConsentVerifier:
    """Test ConsentVerifier class"""
    
    def test_verify_consent_valid(self):
        """Test consent verification with valid consent"""
        verifier = ConsentVerifier(db_connection=None)
        
        result = verifier.verify_consent(
            anonymized_id="test_user_123",
            data_types=["survey", "wearable"]
        )
        
        assert result.is_valid
        assert result.status == ConsentStatus.VALID
        assert result.anonymized_id == "test_user_123"
    
    def test_verify_consent_caching(self):
        """Test that consent results are cached"""
        verifier = ConsentVerifier(db_connection=None, cache_ttl_seconds=60)
        
        # First call
        result1 = verifier.verify_consent(
            anonymized_id="test_user_456",
            data_types=["survey"]
        )
        
        # Second call should hit cache
        result2 = verifier.verify_consent(
            anonymized_id="test_user_456",
            data_types=["survey"]
        )
        
        assert result1.anonymized_id == result2.anonymized_id
        assert result1.status == result2.status
    
    def test_get_consent_expiry(self):
        """Test getting consent expiry date"""
        verifier = ConsentVerifier(db_connection=None)
        
        expiry = verifier.get_consent_expiry("test_user_789")
        
        assert expiry is not None
        assert isinstance(expiry, datetime)
        assert expiry > datetime.now()
    
    def test_clear_cache_specific(self):
        """Test clearing specific cache entry"""
        verifier = ConsentVerifier(db_connection=None)
        
        # Add to cache
        verifier.verify_consent("test_user_clear", ["survey"])
        
        # Clear specific entry
        verifier.clear_cache("test_user_clear")
        
        # Cache should be empty for this user
        assert "test_user_clear" not in verifier._consent_cache
    
    def test_clear_cache_all(self):
        """Test clearing entire cache"""
        verifier = ConsentVerifier(db_connection=None)
        
        # Add multiple entries
        verifier.verify_consent("user1", ["survey"])
        verifier.verify_consent("user2", ["wearable"])
        
        # Clear all
        verifier.clear_cache()
        
        assert len(verifier._consent_cache) == 0
    
    def test_consent_result_properties(self):
        """Test ConsentResult properties"""
        result = ConsentResult(
            anonymized_id="test_id",
            status=ConsentStatus.VALID,
            data_types=["survey", "emr"],
            granted_at=datetime.now(),
            expires_at=datetime.now() + timedelta(days=365)
        )
        
        assert result.is_valid
        assert result.anonymized_id == "test_id"
        assert "survey" in result.data_types
        assert "emr" in result.data_types
    
    def test_consent_result_invalid_status(self):
        """Test ConsentResult with invalid status"""
        result = ConsentResult(
            anonymized_id="test_id",
            status=ConsentStatus.EXPIRED,
            data_types=["survey"]
        )
        
        assert not result.is_valid
    
    def test_consent_result_repr(self):
        """Test ConsentResult string representation"""
        result = ConsentResult(
            anonymized_id="test_id",
            status=ConsentStatus.VALID,
            data_types=["survey"]
        )
        
        repr_str = repr(result)
        assert "test_id" in repr_str
        assert "valid" in repr_str
