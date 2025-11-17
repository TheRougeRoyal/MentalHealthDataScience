"""Tests for API authentication"""

import pytest
from datetime import datetime, timedelta
from jose import jwt

from src.api.auth import Authenticator, AuthResult, TokenData


class TestAuthenticator:
    """Test Authenticator class"""
    
    def test_initialization(self):
        """Test authenticator initializes correctly"""
        auth = Authenticator(
            secret_key="test-secret",
            algorithm="HS256",
            access_token_expire_minutes=30
        )
        
        assert auth.secret_key == "test-secret"
        assert auth.algorithm == "HS256"
        assert auth.access_token_expire_minutes == 30
        assert len(auth._revoked_tokens) == 0
        assert len(auth._token_cache) == 0
    
    def test_generate_token(self):
        """Test token generation"""
        auth = Authenticator(secret_key="test-secret")
        
        token = auth.generate_token("user123", role="clinician")
        
        assert token is not None
        assert isinstance(token, str)
        assert len(token) > 0
        
        # Decode token to verify payload
        payload = jwt.decode(token, "test-secret", algorithms=["HS256"])
        assert payload["user_id"] == "user123"
        assert payload["role"] == "clinician"
        assert "exp" in payload
        assert "iat" in payload
    
    def test_generate_token_custom_expiry(self):
        """Test token generation with custom expiry"""
        auth = Authenticator(secret_key="test-secret")
        
        token = auth.generate_token("user456", expiry_minutes=120)
        
        payload = jwt.decode(token, "test-secret", algorithms=["HS256"])
        exp_time = datetime.fromtimestamp(payload["exp"])
        iat_time = datetime.fromtimestamp(payload["iat"])
        
        # Should expire in approximately 120 minutes
        time_diff = (exp_time - iat_time).total_seconds() / 60
        assert 119 < time_diff < 121
    
    def test_verify_token_valid(self):
        """Test verifying valid token"""
        auth = Authenticator(secret_key="test-secret")
        
        token = auth.generate_token("user789", role="admin")
        result = auth.verify_token(token)
        
        assert result.authenticated is True
        assert result.user_id == "user789"
        assert result.role == "admin"
        assert result.error is None
    
    def test_verify_token_expired(self):
        """Test verifying expired token"""
        auth = Authenticator(secret_key="test-secret")
        
        # Create expired token
        expire = datetime.utcnow() - timedelta(minutes=10)
        payload = {
            "user_id": "user_expired",
            "role": "user",
            "exp": expire,
            "iat": datetime.utcnow() - timedelta(minutes=20)
        }
        token = jwt.encode(payload, "test-secret", algorithm="HS256")
        
        result = auth.verify_token(token)
        
        assert result.authenticated is False
        assert "expired" in result.error.lower()
    
    def test_verify_token_invalid_signature(self):
        """Test verifying token with invalid signature"""
        auth = Authenticator(secret_key="test-secret")
        
        # Create token with different secret
        token = jwt.encode(
            {"user_id": "user123", "role": "user", "exp": datetime.utcnow() + timedelta(hours=1)},
            "wrong-secret",
            algorithm="HS256"
        )
        
        result = auth.verify_token(token)
        
        assert result.authenticated is False
        assert result.error is not None
    
    def test_verify_token_missing_user_id(self):
        """Test verifying token without user_id"""
        auth = Authenticator(secret_key="test-secret")
        
        # Create token without user_id
        payload = {
            "role": "user",
            "exp": datetime.utcnow() + timedelta(hours=1)
        }
        token = jwt.encode(payload, "test-secret", algorithm="HS256")
        
        result = auth.verify_token(token)
        
        assert result.authenticated is False
        assert "invalid" in result.error.lower()
    
    def test_verify_token_caching(self):
        """Test that token verification results are cached"""
        auth = Authenticator(secret_key="test-secret")
        
        token = auth.generate_token("user_cache", role="user")
        
        # First verification
        result1 = auth.verify_token(token)
        assert auth.get_cache_size() == 1
        
        # Second verification should hit cache
        result2 = auth.verify_token(token)
        assert result1.user_id == result2.user_id
        assert auth.get_cache_size() == 1
    
    def test_revoke_token(self):
        """Test token revocation"""
        auth = Authenticator(secret_key="test-secret")
        
        token = auth.generate_token("user_revoke", role="user")
        
        # Verify token works initially
        result1 = auth.verify_token(token)
        assert result1.authenticated is True
        
        # Revoke token
        auth.revoke_token(token)
        assert auth.get_revoked_count() == 1
        
        # Verify token is now rejected
        result2 = auth.verify_token(token)
        assert result2.authenticated is False
        assert "revoked" in result2.error.lower()
    
    def test_revoke_token_clears_cache(self):
        """Test that revoking token removes it from cache"""
        auth = Authenticator(secret_key="test-secret")
        
        token = auth.generate_token("user_cache_revoke", role="user")
        
        # Add to cache
        auth.verify_token(token)
        assert auth.get_cache_size() == 1
        
        # Revoke token
        auth.revoke_token(token)
        
        # Cache should be cleared for this token
        assert auth.get_cache_size() == 0
    
    def test_clear_cache(self):
        """Test clearing token cache"""
        auth = Authenticator(secret_key="test-secret")
        
        # Generate and verify multiple tokens
        for i in range(3):
            token = auth.generate_token(f"user{i}", role="user")
            auth.verify_token(token)
        
        assert auth.get_cache_size() == 3
        
        # Clear cache
        auth.clear_cache()
        assert auth.get_cache_size() == 0
    
    def test_get_cache_size(self):
        """Test getting cache size"""
        auth = Authenticator(secret_key="test-secret")
        
        assert auth.get_cache_size() == 0
        
        token = auth.generate_token("user_size", role="user")
        auth.verify_token(token)
        
        assert auth.get_cache_size() == 1
    
    def test_get_revoked_count(self):
        """Test getting revoked token count"""
        auth = Authenticator(secret_key="test-secret")
        
        assert auth.get_revoked_count() == 0
        
        token1 = auth.generate_token("user1", role="user")
        token2 = auth.generate_token("user2", role="user")
        
        auth.revoke_token(token1)
        assert auth.get_revoked_count() == 1
        
        auth.revoke_token(token2)
        assert auth.get_revoked_count() == 2
    
    def test_multiple_roles(self):
        """Test tokens with different roles"""
        auth = Authenticator(secret_key="test-secret")
        
        roles = ["admin", "clinician", "user", "auditor"]
        
        for role in roles:
            token = auth.generate_token(f"user_{role}", role=role)
            result = auth.verify_token(token)
            
            assert result.authenticated is True
            assert result.role == role


class TestAuthResult:
    """Test AuthResult model"""
    
    def test_auth_result_authenticated(self):
        """Test authenticated result"""
        result = AuthResult(
            authenticated=True,
            user_id="user123",
            role="clinician"
        )
        
        assert result.authenticated is True
        assert result.user_id == "user123"
        assert result.role == "clinician"
        assert result.error is None
    
    def test_auth_result_not_authenticated(self):
        """Test not authenticated result"""
        result = AuthResult(
            authenticated=False,
            error="Invalid token"
        )
        
        assert result.authenticated is False
        assert result.user_id is None
        assert result.role is None
        assert result.error == "Invalid token"


class TestTokenData:
    """Test TokenData model"""
    
    def test_token_data_creation(self):
        """Test creating token data"""
        exp_time = datetime.utcnow() + timedelta(hours=1)
        
        token_data = TokenData(
            user_id="user123",
            role="admin",
            exp=exp_time
        )
        
        assert token_data.user_id == "user123"
        assert token_data.role == "admin"
        assert token_data.exp == exp_time
