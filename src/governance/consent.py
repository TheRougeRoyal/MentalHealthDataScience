"""Consent verification and management"""

from datetime import datetime, timedelta
from typing import List, Optional, Dict
from enum import Enum
import structlog

from src.exceptions import ConsentError
from src.config import settings


logger = structlog.get_logger(__name__)


class ConsentStatus(Enum):
    """Consent status enumeration"""
    VALID = "valid"
    EXPIRED = "expired"
    REVOKED = "revoked"
    NOT_FOUND = "not_found"


class ConsentResult:
    """Result of consent verification"""
    
    def __init__(
        self,
        anonymized_id: str,
        status: ConsentStatus,
        data_types: List[str],
        granted_at: Optional[datetime] = None,
        expires_at: Optional[datetime] = None,
        revoked_at: Optional[datetime] = None
    ):
        self.anonymized_id = anonymized_id
        self.status = status
        self.data_types = data_types
        self.granted_at = granted_at
        self.expires_at = expires_at
        self.revoked_at = revoked_at
    
    @property
    def is_valid(self) -> bool:
        """Check if consent is valid"""
        return self.status == ConsentStatus.VALID
    
    def __repr__(self) -> str:
        return (
            f"ConsentResult(anonymized_id={self.anonymized_id}, "
            f"status={self.status.value}, data_types={self.data_types})"
        )


class ConsentVerifier:
    """Verify consent status before processing data"""
    
    def __init__(self, db_connection=None, cache_ttl_seconds: int = 300):
        """
        Initialize ConsentVerifier
        
        Args:
            db_connection: Database connection (SQLAlchemy engine or connection)
            cache_ttl_seconds: Time-to-live for consent cache in seconds (default: 5 minutes)
        """
        self.db_connection = db_connection
        self.cache_ttl_seconds = cache_ttl_seconds
        self._consent_cache: Dict[str, tuple[ConsentResult, datetime]] = {}
        
        logger.info(
            "consent_verifier_initialized",
            cache_ttl_seconds=cache_ttl_seconds
        )
    
    def verify_consent(
        self,
        anonymized_id: str,
        data_types: List[str]
    ) -> ConsentResult:
        """
        Verify that valid consent exists for the specified data types
        
        Args:
            anonymized_id: Anonymized identifier for the individual
            data_types: List of data types to verify consent for (e.g., ['survey', 'wearable', 'emr'])
        
        Returns:
            ConsentResult object with verification details
        
        Raises:
            ConsentError: If consent verification fails or consent is invalid
        """
        logger.info(
            "verifying_consent",
            anonymized_id=anonymized_id,
            data_types=data_types
        )
        
        # Check cache first
        cached_result = self._get_from_cache(anonymized_id)
        if cached_result:
            # Verify cached consent covers requested data types
            if all(dt in cached_result.data_types for dt in data_types):
                logger.info(
                    "consent_cache_hit",
                    anonymized_id=anonymized_id,
                    status=cached_result.status.value
                )
                
                if not cached_result.is_valid:
                    raise ConsentError(
                        f"Consent is {cached_result.status.value}",
                        details={
                            "anonymized_id": anonymized_id,
                            "status": cached_result.status.value,
                            "data_types": data_types
                        }
                    )
                
                return cached_result
        
        # Query database for consent
        consent_result = self._query_consent_database(anonymized_id, data_types)
        
        # Cache the result
        self._add_to_cache(anonymized_id, consent_result)
        
        # Raise error if consent is not valid
        if not consent_result.is_valid:
            logger.warning(
                "consent_verification_failed",
                anonymized_id=anonymized_id,
                status=consent_result.status.value,
                data_types=data_types
            )
            raise ConsentError(
                f"Consent verification failed: {consent_result.status.value}",
                details={
                    "anonymized_id": anonymized_id,
                    "status": consent_result.status.value,
                    "data_types": data_types,
                    "granted_at": consent_result.granted_at.isoformat() if consent_result.granted_at else None,
                    "expires_at": consent_result.expires_at.isoformat() if consent_result.expires_at else None,
                    "revoked_at": consent_result.revoked_at.isoformat() if consent_result.revoked_at else None
                }
            )
        
        logger.info(
            "consent_verified",
            anonymized_id=anonymized_id,
            data_types=data_types
        )
        
        return consent_result
    
    def get_consent_expiry(self, anonymized_id: str) -> Optional[datetime]:
        """
        Get consent expiration date for an individual
        
        Args:
            anonymized_id: Anonymized identifier for the individual
        
        Returns:
            Expiration datetime or None if no expiration set
        """
        # Check cache first
        cached_result = self._get_from_cache(anonymized_id)
        if cached_result:
            return cached_result.expires_at
        
        # Query database
        consent_result = self._query_consent_database(anonymized_id, [])
        
        # Cache the result
        self._add_to_cache(anonymized_id, consent_result)
        
        return consent_result.expires_at
    
    def _query_consent_database(
        self,
        anonymized_id: str,
        data_types: List[str]
    ) -> ConsentResult:
        """
        Query the consent database for consent status
        
        Args:
            anonymized_id: Anonymized identifier
            data_types: List of data types to check
        
        Returns:
            ConsentResult object
        """
        if self.db_connection is None:
            # For testing/development without database
            logger.warning(
                "no_database_connection",
                message="Using mock consent verification"
            )
            return self._mock_consent_query(anonymized_id, data_types)
        
        try:
            # Execute SQL query
            query = """
                SELECT anonymized_id, data_types, granted_at, expires_at, revoked_at
                FROM consent
                WHERE anonymized_id = %s
            """
            
            # Use SQLAlchemy connection or raw psycopg2
            if hasattr(self.db_connection, 'execute'):
                result = self.db_connection.execute(query, (anonymized_id,))
                row = result.fetchone()
            else:
                # Assume it's a psycopg2 connection
                cursor = self.db_connection.cursor()
                cursor.execute(query, (anonymized_id,))
                row = cursor.fetchone()
                cursor.close()
            
            if not row:
                return ConsentResult(
                    anonymized_id=anonymized_id,
                    status=ConsentStatus.NOT_FOUND,
                    data_types=[]
                )
            
            # Parse row data
            _, db_data_types, granted_at, expires_at, revoked_at = row
            
            # Check if revoked
            if revoked_at is not None:
                return ConsentResult(
                    anonymized_id=anonymized_id,
                    status=ConsentStatus.REVOKED,
                    data_types=db_data_types,
                    granted_at=granted_at,
                    expires_at=expires_at,
                    revoked_at=revoked_at
                )
            
            # Check if expired
            if expires_at is not None and datetime.now() > expires_at:
                return ConsentResult(
                    anonymized_id=anonymized_id,
                    status=ConsentStatus.EXPIRED,
                    data_types=db_data_types,
                    granted_at=granted_at,
                    expires_at=expires_at
                )
            
            # Check if requested data types are covered
            if data_types and not all(dt in db_data_types for dt in data_types):
                missing_types = [dt for dt in data_types if dt not in db_data_types]
                logger.warning(
                    "consent_missing_data_types",
                    anonymized_id=anonymized_id,
                    missing_types=missing_types
                )
                return ConsentResult(
                    anonymized_id=anonymized_id,
                    status=ConsentStatus.NOT_FOUND,
                    data_types=db_data_types,
                    granted_at=granted_at,
                    expires_at=expires_at
                )
            
            # Consent is valid
            return ConsentResult(
                anonymized_id=anonymized_id,
                status=ConsentStatus.VALID,
                data_types=db_data_types,
                granted_at=granted_at,
                expires_at=expires_at
            )
            
        except Exception as e:
            logger.error(
                "consent_database_query_failed",
                anonymized_id=anonymized_id,
                error=str(e)
            )
            raise ConsentError(
                f"Failed to query consent database: {str(e)}",
                details={"anonymized_id": anonymized_id, "error": str(e)}
            )
    
    def _mock_consent_query(
        self,
        anonymized_id: str,
        data_types: List[str]
    ) -> ConsentResult:
        """
        Mock consent query for testing without database
        
        Args:
            anonymized_id: Anonymized identifier
            data_types: List of data types
        
        Returns:
            ConsentResult with VALID status
        """
        return ConsentResult(
            anonymized_id=anonymized_id,
            status=ConsentStatus.VALID,
            data_types=data_types or ['survey', 'wearable', 'emr'],
            granted_at=datetime.now() - timedelta(days=30),
            expires_at=datetime.now() + timedelta(days=335)
        )
    
    def _get_from_cache(self, anonymized_id: str) -> Optional[ConsentResult]:
        """
        Get consent result from cache if not expired
        
        Args:
            anonymized_id: Anonymized identifier
        
        Returns:
            Cached ConsentResult or None if not in cache or expired
        """
        if anonymized_id in self._consent_cache:
            result, cached_at = self._consent_cache[anonymized_id]
            
            # Check if cache entry is still valid
            if datetime.now() - cached_at < timedelta(seconds=self.cache_ttl_seconds):
                return result
            else:
                # Remove expired cache entry
                del self._consent_cache[anonymized_id]
        
        return None
    
    def _add_to_cache(self, anonymized_id: str, result: ConsentResult) -> None:
        """
        Add consent result to cache
        
        Args:
            anonymized_id: Anonymized identifier
            result: ConsentResult to cache
        """
        self._consent_cache[anonymized_id] = (result, datetime.now())
        
        # Simple cache size management - remove oldest entries if cache grows too large
        if len(self._consent_cache) > 10000:
            # Remove oldest 1000 entries
            sorted_cache = sorted(
                self._consent_cache.items(),
                key=lambda x: x[1][1]
            )
            for key, _ in sorted_cache[:1000]:
                del self._consent_cache[key]
    
    def clear_cache(self, anonymized_id: Optional[str] = None) -> None:
        """
        Clear consent cache
        
        Args:
            anonymized_id: If provided, clear only this entry. Otherwise clear all.
        """
        if anonymized_id:
            self._consent_cache.pop(anonymized_id, None)
            logger.info("consent_cache_cleared", anonymized_id=anonymized_id)
        else:
            self._consent_cache.clear()
            logger.info("consent_cache_cleared_all")
