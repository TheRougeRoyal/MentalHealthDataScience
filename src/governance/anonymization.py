"""Data anonymization and PII protection"""

import hashlib
import re
from typing import Dict, List, Any, Optional
import structlog

from src.config import settings


logger = structlog.get_logger(__name__)


class Anonymizer:
    """Anonymize personally identifiable information (PII)"""
    
    # Common PII patterns for text redaction
    PII_PATTERNS = {
        'email': r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
        'phone': r'\b(?:\+?1[-.]?)?\(?([0-9]{3})\)?[-.]?([0-9]{3})[-.]?([0-9]{4})\b',
        'ssn': r'\b\d{3}-\d{2}-\d{4}\b',
        'credit_card': r'\b\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}\b',
        'zip_code': r'\b\d{5}(?:-\d{4})?\b',
        'ip_address': r'\b(?:\d{1,3}\.){3}\d{1,3}\b',
    }
    
    def __init__(self, salt: Optional[str] = None):
        """
        Initialize Anonymizer
        
        Args:
            salt: Salt for hashing. If None, uses value from settings.
        """
        self.salt = salt or settings.security.anonymization_salt
        
        if self.salt == "change-me-in-production":
            logger.warning(
                "anonymization_using_default_salt",
                message="Using default salt - change in production!"
            )
        
        logger.info("anonymizer_initialized")
    
    def anonymize_record(
        self,
        data: Dict[str, Any],
        pii_fields: List[str]
    ) -> Dict[str, Any]:
        """
        Anonymize a data record by hashing specified PII fields
        
        Args:
            data: Dictionary containing the data record
            pii_fields: List of field names to anonymize
        
        Returns:
            Dictionary with PII fields replaced by hashes
        
        Example:
            >>> anonymizer = Anonymizer()
            >>> data = {"user_id": "12345", "name": "John Doe", "age": 30}
            >>> result = anonymizer.anonymize_record(data, ["user_id", "name"])
            >>> # result["user_id"] and result["name"] are now hashed
        """
        anonymized_data = data.copy()
        
        for field in pii_fields:
            if field in anonymized_data:
                original_value = anonymized_data[field]
                
                # Handle None values
                if original_value is None:
                    continue
                
                # Convert to string for hashing
                str_value = str(original_value)
                
                # Hash the identifier
                anonymized_data[field] = self.hash_identifier(str_value)
                
                logger.debug(
                    "field_anonymized",
                    field=field,
                    original_length=len(str_value)
                )
        
        return anonymized_data
    
    def hash_identifier(self, identifier: str) -> str:
        """
        Hash an identifier using SHA-256 with salt
        
        This ensures consistency - the same identifier always produces
        the same hash, allowing record linkage while protecting privacy.
        
        Args:
            identifier: The identifier to hash
        
        Returns:
            Hexadecimal hash string
        
        Example:
            >>> anonymizer = Anonymizer()
            >>> hash1 = anonymizer.hash_identifier("user123")
            >>> hash2 = anonymizer.hash_identifier("user123")
            >>> assert hash1 == hash2  # Consistent hashing
        """
        if not identifier:
            return ""
        
        # Combine identifier with salt
        salted_value = f"{identifier}{self.salt}"
        
        # Create SHA-256 hash
        hash_object = hashlib.sha256(salted_value.encode('utf-8'))
        hash_hex = hash_object.hexdigest()
        
        return hash_hex
    
    def anonymize_text(
        self,
        text: str,
        redact_patterns: Optional[List[str]] = None,
        replacement: str = "[REDACTED]"
    ) -> str:
        """
        Anonymize PII in free-text fields by redacting patterns
        
        Args:
            text: Text to anonymize
            redact_patterns: List of pattern names to redact (e.g., ['email', 'phone']).
                           If None, redacts all known patterns.
            replacement: String to replace PII with
        
        Returns:
            Text with PII redacted
        
        Example:
            >>> anonymizer = Anonymizer()
            >>> text = "Contact me at john@example.com or 555-123-4567"
            >>> result = anonymizer.anonymize_text(text)
            >>> # result: "Contact me at [REDACTED] or [REDACTED]"
        """
        if not text:
            return text
        
        anonymized_text = text
        patterns_to_use = redact_patterns or list(self.PII_PATTERNS.keys())
        
        redaction_count = 0
        
        for pattern_name in patterns_to_use:
            if pattern_name in self.PII_PATTERNS:
                pattern = self.PII_PATTERNS[pattern_name]
                
                # Count matches before redaction
                matches = re.findall(pattern, anonymized_text)
                if matches:
                    redaction_count += len(matches)
                    logger.debug(
                        "pii_pattern_found",
                        pattern_name=pattern_name,
                        count=len(matches)
                    )
                
                # Replace matches with redaction string
                anonymized_text = re.sub(pattern, replacement, anonymized_text)
        
        if redaction_count > 0:
            logger.info(
                "text_anonymized",
                redaction_count=redaction_count,
                original_length=len(text),
                anonymized_length=len(anonymized_text)
            )
        
        return anonymized_text
    
    def anonymize_nested_record(
        self,
        data: Dict[str, Any],
        pii_fields: List[str],
        text_fields: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Anonymize a nested data record with both identifier and text fields
        
        Args:
            data: Dictionary containing the data record (may be nested)
            pii_fields: List of field paths to hash (e.g., ['user.id', 'patient_id'])
            text_fields: List of field paths containing text to redact
        
        Returns:
            Dictionary with PII anonymized
        """
        anonymized_data = data.copy()
        
        # Anonymize identifier fields
        for field_path in pii_fields:
            self._anonymize_field_by_path(anonymized_data, field_path, hash_value=True)
        
        # Anonymize text fields
        if text_fields:
            for field_path in text_fields:
                self._anonymize_field_by_path(anonymized_data, field_path, hash_value=False)
        
        return anonymized_data
    
    def _anonymize_field_by_path(
        self,
        data: Dict[str, Any],
        field_path: str,
        hash_value: bool = True
    ) -> None:
        """
        Anonymize a field in a nested dictionary by path
        
        Args:
            data: Dictionary to modify in-place
            field_path: Dot-separated path to field (e.g., 'user.profile.email')
            hash_value: If True, hash the value. If False, redact PII patterns.
        """
        parts = field_path.split('.')
        current = data
        
        # Navigate to parent of target field
        for part in parts[:-1]:
            if part not in current or not isinstance(current[part], dict):
                return  # Path doesn't exist
            current = current[part]
        
        # Anonymize the target field
        field_name = parts[-1]
        if field_name in current:
            value = current[field_name]
            
            if value is None:
                return
            
            if hash_value:
                current[field_name] = self.hash_identifier(str(value))
            else:
                if isinstance(value, str):
                    current[field_name] = self.anonymize_text(value)
    
    def create_anonymized_id(self, *identifiers: str) -> str:
        """
        Create a composite anonymized ID from multiple identifiers
        
        Useful when you need to create a unique anonymous identifier
        from multiple fields (e.g., combining user_id and session_id)
        
        Args:
            *identifiers: Variable number of identifier strings
        
        Returns:
            Hashed composite identifier
        
        Example:
            >>> anonymizer = Anonymizer()
            >>> anon_id = anonymizer.create_anonymized_id("user123", "session456")
        """
        # Combine all identifiers with a separator
        combined = "|".join(str(i) for i in identifiers if i)
        
        return self.hash_identifier(combined)
    
    def verify_anonymization(self, data: Dict[str, Any], pii_fields: List[str]) -> bool:
        """
        Verify that specified PII fields have been anonymized (are hashes)
        
        Args:
            data: Dictionary to verify
            pii_fields: List of fields that should be anonymized
        
        Returns:
            True if all PII fields appear to be hashed, False otherwise
        """
        for field in pii_fields:
            if field in data:
                value = data[field]
                
                # Check if value looks like a SHA-256 hash (64 hex characters)
                if not isinstance(value, str) or len(value) != 64:
                    logger.warning(
                        "field_not_anonymized",
                        field=field,
                        value_type=type(value).__name__,
                        value_length=len(str(value)) if value else 0
                    )
                    return False
                
                # Check if it's hexadecimal
                try:
                    int(value, 16)
                except ValueError:
                    logger.warning(
                        "field_not_valid_hash",
                        field=field
                    )
                    return False
        
        return True
