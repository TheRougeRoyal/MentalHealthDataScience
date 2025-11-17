"""Authentication and authorization for MHRAS API"""

import logging
from datetime import datetime, timedelta
from typing import Dict, Optional
from jose import JWTError, jwt
from pydantic import BaseModel
from src.config import settings

logger = logging.getLogger(__name__)


class TokenData(BaseModel):
    """Token payload data"""
    user_id: str
    role: str
    exp: datetime


class AuthResult(BaseModel):
    """Authentication result"""
    authenticated: bool
    user_id: Optional[str] = None
    role: Optional[str] = None
    error: Optional[str] = None


class Authenticator:
    """
    Handles JWT token generation, validation, and revocation.
    
    Implements token-based authentication with:
    - JWT token generation with configurable expiration
    - Token signature verification
    - Token expiration checks
    - Token revocation support
    - Token caching for performance
    """
    
    def __init__(
        self,
        secret_key: Optional[str] = None,
        algorithm: str = "HS256",
        access_token_expire_minutes: int = 60
    ):
        """
        Initialize authenticator.
        
        Args:
            secret_key: Secret key for JWT signing (defaults to config)
            algorithm: JWT algorithm (default: HS256)
            access_token_expire_minutes: Token expiration time in minutes
        """
        self.secret_key = secret_key or getattr(settings, 'SECRET_KEY', 'default-secret-key-change-in-production')
        self.algorithm = algorithm
        self.access_token_expire_minutes = access_token_expire_minutes
        
        # Token revocation list (in production, use Redis or database)
        self._revoked_tokens: set = set()
        
        # Token cache for performance (in production, use Redis)
        self._token_cache: Dict[str, AuthResult] = {}
        
        logger.info("Authenticator initialized")
    
    def generate_token(
        self,
        user_id: str,
        role: str = "user",
        expiry_minutes: Optional[int] = None
    ) -> str:
        """
        Generate a JWT token for a user.
        
        Args:
            user_id: User identifier
            role: User role (e.g., 'admin', 'clinician', 'user')
            expiry_minutes: Custom expiration time (overrides default)
        
        Returns:
            JWT token string
        
        Example:
            >>> auth = Authenticator()
            >>> token = auth.generate_token("user123", role="clinician")
        """
        expiry = expiry_minutes or self.access_token_expire_minutes
        expire = datetime.utcnow() + timedelta(minutes=expiry)
        
        payload = {
            "user_id": user_id,
            "role": role,
            "exp": expire,
            "iat": datetime.utcnow()
        }
        
        token = jwt.encode(payload, self.secret_key, algorithm=self.algorithm)
        
        logger.info(f"Generated token for user {user_id} with role {role}")
        return token
    
    def verify_token(self, token: str) -> AuthResult:
        """
        Verify a JWT token.
        
        Performs:
        - Signature verification
        - Expiration check
        - Revocation check
        - Cache lookup for performance
        
        Args:
            token: JWT token string
        
        Returns:
            AuthResult with authentication status and user info
        
        Example:
            >>> auth = Authenticator()
            >>> result = auth.verify_token(token)
            >>> if result.authenticated:
            ...     print(f"User {result.user_id} authenticated")
        """
        # Check cache first
        if token in self._token_cache:
            cached_result = self._token_cache[token]
            # Verify cached result is still valid (not expired)
            try:
                payload = jwt.decode(
                    token,
                    self.secret_key,
                    algorithms=[self.algorithm]
                )
                return cached_result
            except JWTError:
                # Token expired or invalid, remove from cache
                del self._token_cache[token]
        
        # Check if token is revoked
        if token in self._revoked_tokens:
            logger.warning("Attempted use of revoked token")
            return AuthResult(
                authenticated=False,
                error="Token has been revoked"
            )
        
        try:
            # Decode and verify token
            payload = jwt.decode(
                token,
                self.secret_key,
                algorithms=[self.algorithm]
            )
            
            user_id = payload.get("user_id")
            role = payload.get("role")
            
            if not user_id:
                logger.warning("Token missing user_id")
                return AuthResult(
                    authenticated=False,
                    error="Invalid token payload"
                )
            
            result = AuthResult(
                authenticated=True,
                user_id=user_id,
                role=role
            )
            
            # Cache the result
            self._token_cache[token] = result
            
            logger.debug(f"Token verified for user {user_id}")
            return result
            
        except jwt.ExpiredSignatureError:
            logger.warning("Token has expired")
            return AuthResult(
                authenticated=False,
                error="Token has expired"
            )
        except JWTError as e:
            logger.warning(f"Token verification failed: {str(e)}")
            return AuthResult(
                authenticated=False,
                error=f"Invalid token: {str(e)}"
            )
        except Exception as e:
            logger.error(f"Unexpected error during token verification: {str(e)}")
            return AuthResult(
                authenticated=False,
                error="Authentication error"
            )
    
    def revoke_token(self, token: str) -> None:
        """
        Revoke a token.
        
        Adds token to revocation list and removes from cache.
        In production, this should persist to a database or Redis.
        
        Args:
            token: JWT token to revoke
        
        Example:
            >>> auth = Authenticator()
            >>> auth.revoke_token(token)
        """
        self._revoked_tokens.add(token)
        
        # Remove from cache if present
        if token in self._token_cache:
            del self._token_cache[token]
        
        logger.info("Token revoked")
    
    def clear_cache(self) -> None:
        """
        Clear the token cache.
        
        Useful for testing or when cache needs to be refreshed.
        """
        self._token_cache.clear()
        logger.info("Token cache cleared")
    
    def get_cache_size(self) -> int:
        """
        Get the current size of the token cache.
        
        Returns:
            Number of cached tokens
        """
        return len(self._token_cache)
    
    def get_revoked_count(self) -> int:
        """
        Get the number of revoked tokens.
        
        Returns:
            Number of revoked tokens
        """
        return len(self._revoked_tokens)


# Global authenticator instance
authenticator = Authenticator()
