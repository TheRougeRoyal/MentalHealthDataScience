"""Feature Store for centralized feature engineering and serving."""

import logging
import json
import hashlib
from datetime import datetime
from typing import List, Optional, Dict, Any, Callable
from uuid import UUID
import pandas as pd
from pydantic import BaseModel, Field, field_validator

from src.database.connection import DatabaseConnection
from src.config import settings
from src.exceptions import ValidationError

logger = logging.getLogger(__name__)


class FeatureDefinition(BaseModel):
    """Definition of a feature with transformation logic."""
    
    feature_id: Optional[UUID] = None
    feature_name: str = Field(..., min_length=1, max_length=255)
    feature_type: str = Field(..., pattern="^(numeric|categorical|embedding|text|boolean)$")
    description: Optional[str] = None
    transformation_code: str = Field(..., min_length=1)
    input_schema: Dict[str, str] = Field(default_factory=dict)
    output_schema: Dict[str, str] = Field(default_factory=dict)
    version: str = Field(default="1.0.0")
    dependencies: List[str] = Field(default_factory=list)
    owner: Optional[str] = None
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None
    
    @field_validator('input_schema', 'output_schema')
    @classmethod
    def validate_schema(cls, v: Dict[str, str]) -> Dict[str, str]:
        """Validate schema format."""
        if not isinstance(v, dict):
            raise ValueError("Schema must be a dictionary")
        for key, value in v.items():
            if not isinstance(key, str) or not isinstance(value, str):
                raise ValueError("Schema keys and values must be strings")
        return v
    
    @field_validator('dependencies')
    @classmethod
    def validate_dependencies(cls, v: List[str]) -> List[str]:
        """Validate dependencies format."""
        if not isinstance(v, list):
            raise ValueError("Dependencies must be a list")
        for dep in v:
            if not isinstance(dep, str):
                raise ValueError("Dependencies must be strings")
        return v


class FeatureValue(BaseModel):
    """Represents a computed feature value for an entity."""
    
    value_id: Optional[UUID] = None
    feature_id: UUID
    entity_id: str
    feature_value: Any
    computed_at: Optional[datetime] = None
    dataset_version_id: Optional[UUID] = None


class FeatureStoreError(Exception):
    """Base exception for feature store operations."""
    pass


class FeatureNotFoundError(FeatureStoreError):
    """Feature not found in store."""
    pass


class FeatureComputationError(FeatureStoreError):
    """Error during feature computation."""
    pass


class CacheBackend:
    """Abstract cache backend interface."""
    
    def get(self, key: str) -> Optional[Any]:
        """Get value from cache."""
        raise NotImplementedError
    
    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        """Set value in cache with optional TTL."""
        raise NotImplementedError
    
    def delete(self, key: str) -> None:
        """Delete value from cache."""
        raise NotImplementedError
    
    def clear(self) -> None:
        """Clear all cache entries."""
        raise NotImplementedError


class MemoryCache(CacheBackend):
    """In-memory cache implementation."""
    
    def __init__(self):
        self._cache: Dict[str, Any] = {}
        self._expiry: Dict[str, datetime] = {}
    
    def get(self, key: str) -> Optional[Any]:
        """Get value from cache."""
        if key in self._cache:
            # Check expiry
            if key in self._expiry:
                if datetime.now() > self._expiry[key]:
                    del self._cache[key]
                    del self._expiry[key]
                    return None
            return self._cache[key]
        return None
    
    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        """Set value in cache with optional TTL."""
        self._cache[key] = value
        if ttl:
            from datetime import timedelta
            self._expiry[key] = datetime.now() + timedelta(seconds=ttl)
    
    def delete(self, key: str) -> None:
        """Delete value from cache."""
        self._cache.pop(key, None)
        self._expiry.pop(key, None)
    
    def clear(self) -> None:
        """Clear all cache entries."""
        self._cache.clear()
        self._expiry.clear()


class RedisCache(CacheBackend):
    """Redis cache implementation."""
    
    def __init__(self, redis_url: str):
        try:
            import redis
            self._redis = redis.from_url(redis_url, decode_responses=False)
            # Test connection
            self._redis.ping()
            logger.info(f"Connected to Redis at {redis_url}")
        except ImportError:
            logger.warning("Redis library not installed, falling back to memory cache")
            raise
        except Exception as e:
            logger.warning(f"Failed to connect to Redis: {e}, falling back to memory cache")
            raise
    
    def get(self, key: str) -> Optional[Any]:
        """Get value from cache."""
        try:
            value = self._redis.get(key)
            if value:
                return json.loads(value)
            return None
        except Exception as e:
            logger.error(f"Redis get error: {e}")
            return None
    
    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        """Set value in cache with optional TTL."""
        try:
            serialized = json.dumps(value)
            if ttl:
                self._redis.setex(key, ttl, serialized)
            else:
                self._redis.set(key, serialized)
        except Exception as e:
            logger.error(f"Redis set error: {e}")
    
    def delete(self, key: str) -> None:
        """Delete value from cache."""
        try:
            self._redis.delete(key)
        except Exception as e:
            logger.error(f"Redis delete error: {e}")
    
    def clear(self) -> None:
        """Clear all cache entries."""
        try:
            self._redis.flushdb()
        except Exception as e:
            logger.error(f"Redis clear error: {e}")


class FeatureRepository:
    """Repository for feature store database operations."""
    
    def __init__(self, db_connection: DatabaseConnection):
        """
        Initialize repository.
        
        Args:
            db_connection: Database connection instance
        """
        self.db = db_connection
    
    def create_feature(self, feature: FeatureDefinition) -> UUID:
        """
        Create a new feature definition.
        
        Args:
            feature: FeatureDefinition instance
            
        Returns:
            UUID of created feature
        """
        with self.db.get_cursor() as cur:
            cur.execute(
                """
                INSERT INTO features (
                    feature_name, feature_type, description, transformation_code,
                    input_schema, output_schema, version, dependencies, owner
                )
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                RETURNING feature_id
                """,
                (
                    feature.feature_name,
                    feature.feature_type,
                    feature.description,
                    feature.transformation_code,
                    json.dumps(feature.input_schema),
                    json.dumps(feature.output_schema),
                    feature.version,
                    json.dumps(feature.dependencies),
                    feature.owner,
                )
            )
            result = cur.fetchone()
            feature_id = result["feature_id"]
            logger.info(f"Created feature {feature_id}: {feature.feature_name}")
            return feature_id
    
    def get_feature_by_name(self, feature_name: str) -> Optional[FeatureDefinition]:
        """
        Get feature by name.
        
        Args:
            feature_name: Feature name
            
        Returns:
            FeatureDefinition instance or None if not found
        """
        with self.db.get_cursor() as cur:
            cur.execute(
                "SELECT * FROM features WHERE feature_name = %s",
                (feature_name,)
            )
            row = cur.fetchone()
            if row:
                return FeatureDefinition(**row)
            return None
    
    def get_feature_by_id(self, feature_id: UUID) -> Optional[FeatureDefinition]:
        """
        Get feature by ID.
        
        Args:
            feature_id: Feature UUID
            
        Returns:
            FeatureDefinition instance or None if not found
        """
        with self.db.get_cursor() as cur:
            cur.execute(
                "SELECT * FROM features WHERE feature_id = %s",
                (str(feature_id),)
            )
            row = cur.fetchone()
            if row:
                return FeatureDefinition(**row)
            return None
    
    def list_features(self, limit: int = 100) -> List[FeatureDefinition]:
        """
        List all features.
        
        Args:
            limit: Maximum number of results
            
        Returns:
            List of FeatureDefinition instances
        """
        with self.db.get_cursor() as cur:
            cur.execute(
                """
                SELECT * FROM features
                ORDER BY created_at DESC
                LIMIT %s
                """,
                (limit,)
            )
            rows = cur.fetchall()
            return [FeatureDefinition(**row) for row in rows]
    
    def update_feature(
        self,
        feature_id: UUID,
        transformation_code: Optional[str] = None,
        version: Optional[str] = None,
        description: Optional[str] = None
    ) -> bool:
        """
        Update feature definition.
        
        Args:
            feature_id: Feature UUID
            transformation_code: Optional new transformation code
            version: Optional new version
            description: Optional new description
            
        Returns:
            True if updated, False if not found
        """
        updates = []
        values = []
        
        if transformation_code is not None:
            updates.append("transformation_code = %s")
            values.append(transformation_code)
        
        if version is not None:
            updates.append("version = %s")
            values.append(version)
        
        if description is not None:
            updates.append("description = %s")
            values.append(description)
        
        if not updates:
            return False
        
        values.append(str(feature_id))
        
        with self.db.get_cursor() as cur:
            cur.execute(
                f"""
                UPDATE features
                SET {', '.join(updates)}
                WHERE feature_id = %s
                RETURNING feature_id
                """,
                tuple(values)
            )
            result = cur.fetchone()
            return result is not None
    
    def store_feature_value(self, feature_value: FeatureValue) -> UUID:
        """
        Store a computed feature value.
        
        Args:
            feature_value: FeatureValue instance
            
        Returns:
            UUID of stored value
        """
        with self.db.get_cursor() as cur:
            cur.execute(
                """
                INSERT INTO feature_values (
                    feature_id, entity_id, feature_value, dataset_version_id
                )
                VALUES (%s, %s, %s, %s)
                RETURNING value_id
                """,
                (
                    str(feature_value.feature_id),
                    feature_value.entity_id,
                    json.dumps(feature_value.feature_value),
                    str(feature_value.dataset_version_id) if feature_value.dataset_version_id else None,
                )
            )
            result = cur.fetchone()
            value_id = result["value_id"]
            return value_id
    
    def get_feature_values(
        self,
        feature_id: UUID,
        entity_ids: List[str]
    ) -> List[FeatureValue]:
        """
        Get feature values for specific entities.
        
        Args:
            feature_id: Feature UUID
            entity_ids: List of entity IDs
            
        Returns:
            List of FeatureValue instances
        """
        with self.db.get_cursor() as cur:
            cur.execute(
                """
                SELECT * FROM feature_values
                WHERE feature_id = %s AND entity_id = ANY(%s)
                ORDER BY computed_at DESC
                """,
                (str(feature_id), entity_ids)
            )
            rows = cur.fetchall()
            return [FeatureValue(**row) for row in rows]
    
    def get_latest_feature_values(
        self,
        feature_id: UUID,
        entity_ids: List[str]
    ) -> Dict[str, Any]:
        """
        Get latest feature values for entities.
        
        Args:
            feature_id: Feature UUID
            entity_ids: List of entity IDs
            
        Returns:
            Dictionary mapping entity_id to feature value
        """
        with self.db.get_cursor() as cur:
            cur.execute(
                """
                SELECT DISTINCT ON (entity_id)
                    entity_id, feature_value
                FROM feature_values
                WHERE feature_id = %s AND entity_id = ANY(%s)
                ORDER BY entity_id, computed_at DESC
                """,
                (str(feature_id), entity_ids)
            )
            rows = cur.fetchall()
            return {row["entity_id"]: row["feature_value"] for row in rows}


class FeatureStore:
    """Centralized feature repository for feature engineering and serving."""
    
    def __init__(
        self,
        db_connection: DatabaseConnection,
        cache_backend: Optional[CacheBackend] = None
    ):
        """
        Initialize feature store.
        
        Args:
            db_connection: Database connection instance
            cache_backend: Optional cache backend for online serving
        """
        self.db = db_connection
        self.repository = FeatureRepository(db_connection)
        
        # Initialize cache backend
        if cache_backend:
            self.cache = cache_backend
        elif settings.feature_store.enable_caching:
            if settings.feature_store.cache_backend == "redis":
                try:
                    self.cache = RedisCache(settings.feature_store.redis_url)
                except Exception as e:
                    logger.warning(f"Failed to initialize Redis cache: {e}, using memory cache")
                    self.cache = MemoryCache()
            else:
                self.cache = MemoryCache()
        else:
            self.cache = None
        
        logger.info("Feature store initialized")
    
    def _generate_cache_key(self, feature_name: str, entity_id: str) -> str:
        """Generate cache key for feature value."""
        return f"feature:{feature_name}:{entity_id}"
    
    def _validate_schema(
        self,
        data: pd.DataFrame,
        schema: Dict[str, str],
        schema_name: str
    ) -> None:
        """
        Validate data against schema.
        
        Args:
            data: DataFrame to validate
            schema: Schema definition
            schema_name: Name for error messages
            
        Raises:
            ValidationError: If validation fails
        """
        if not schema:
            return
        
        # Check required columns
        missing_cols = set(schema.keys()) - set(data.columns)
        if missing_cols:
            raise ValidationError(
                f"{schema_name} validation failed: missing columns {missing_cols}"
            )
        
        # Check data types (basic validation)
        for col, expected_type in schema.items():
            if col not in data.columns:
                continue
            
            actual_dtype = str(data[col].dtype)
            
            # Map pandas dtypes to schema types
            if expected_type in ["int", "integer", "int64"]:
                if not pd.api.types.is_integer_dtype(data[col]):
                    raise ValidationError(
                        f"{schema_name} validation failed: column '{col}' "
                        f"expected {expected_type}, got {actual_dtype}"
                    )
            elif expected_type in ["float", "float64", "numeric"]:
                if not pd.api.types.is_numeric_dtype(data[col]):
                    raise ValidationError(
                        f"{schema_name} validation failed: column '{col}' "
                        f"expected {expected_type}, got {actual_dtype}"
                    )
            elif expected_type in ["str", "string", "object"]:
                if not pd.api.types.is_string_dtype(data[col]) and data[col].dtype != object:
                    raise ValidationError(
                        f"{schema_name} validation failed: column '{col}' "
                        f"expected {expected_type}, got {actual_dtype}"
                    )

    def register_feature(
        self,
        feature_name: str,
        feature_type: str,
        transformation_code: str,
        input_schema: Dict[str, str],
        output_schema: Dict[str, str],
        description: Optional[str] = None,
        version: str = "1.0.0",
        dependencies: Optional[List[str]] = None,
        owner: Optional[str] = None
    ) -> UUID:
        """
        Register a new feature definition.
        
        Args:
            feature_name: Unique feature name
            feature_type: Type of feature (numeric, categorical, embedding, text, boolean)
            transformation_code: Python code for feature transformation
            input_schema: Schema of input data (column_name -> type)
            output_schema: Schema of output feature (column_name -> type)
            description: Optional feature description
            version: Feature version (default: "1.0.0")
            dependencies: Optional list of dependent feature names
            owner: Optional owner identifier
            
        Returns:
            UUID of registered feature
            
        Raises:
            ValidationError: If feature definition is invalid
            FeatureStoreError: If feature already exists
        """
        # Check if feature already exists
        existing = self.repository.get_feature_by_name(feature_name)
        if existing:
            raise FeatureStoreError(f"Feature '{feature_name}' already exists")
        
        # Validate feature type
        valid_types = ["numeric", "categorical", "embedding", "text", "boolean"]
        if feature_type not in valid_types:
            raise ValidationError(
                f"Invalid feature_type '{feature_type}'. Must be one of {valid_types}"
            )
        
        # Validate dependencies exist
        if dependencies:
            for dep in dependencies:
                dep_feature = self.repository.get_feature_by_name(dep)
                if not dep_feature:
                    raise ValidationError(f"Dependency feature '{dep}' not found")
        
        # Create feature definition
        feature = FeatureDefinition(
            feature_name=feature_name,
            feature_type=feature_type,
            transformation_code=transformation_code,
            input_schema=input_schema,
            output_schema=output_schema,
            description=description,
            version=version,
            dependencies=dependencies or [],
            owner=owner
        )
        
        feature_id = self.repository.create_feature(feature)
        logger.info(f"Registered feature '{feature_name}' with ID {feature_id}")
        
        return feature_id
    
    def _resolve_dependencies(
        self,
        feature_name: str,
        visited: Optional[set] = None
    ) -> List[str]:
        """
        Resolve feature dependencies in topological order.
        
        Args:
            feature_name: Feature name to resolve
            visited: Set of already visited features (for cycle detection)
            
        Returns:
            List of feature names in dependency order
            
        Raises:
            FeatureStoreError: If circular dependency detected
        """
        if visited is None:
            visited = set()
        
        if feature_name in visited:
            raise FeatureStoreError(
                f"Circular dependency detected involving feature '{feature_name}'"
            )
        
        visited.add(feature_name)
        
        feature = self.repository.get_feature_by_name(feature_name)
        if not feature:
            raise FeatureNotFoundError(f"Feature '{feature_name}' not found")
        
        result = []
        for dep in feature.dependencies:
            result.extend(self._resolve_dependencies(dep, visited.copy()))
        
        result.append(feature_name)
        return result
    
    def compute_features(
        self,
        feature_names: List[str],
        input_data: pd.DataFrame,
        entity_id_column: str = "entity_id"
    ) -> pd.DataFrame:
        """
        Compute features from raw data.
        
        Args:
            feature_names: List of feature names to compute
            input_data: Input DataFrame with raw data
            entity_id_column: Column name containing entity IDs
            
        Returns:
            DataFrame with computed features
            
        Raises:
            FeatureNotFoundError: If feature not found
            FeatureComputationError: If computation fails
            ValidationError: If schema validation fails
        """
        if entity_id_column not in input_data.columns:
            raise ValidationError(
                f"Entity ID column '{entity_id_column}' not found in input data"
            )
        
        # Resolve dependencies for all features
        all_features = []
        for feature_name in feature_names:
            deps = self._resolve_dependencies(feature_name)
            all_features.extend(deps)
        
        # Remove duplicates while preserving order
        seen = set()
        ordered_features = []
        for f in all_features:
            if f not in seen:
                seen.add(f)
                ordered_features.append(f)
        
        # Compute features in dependency order
        result_data = input_data.copy()
        
        for feature_name in ordered_features:
            feature = self.repository.get_feature_by_name(feature_name)
            if not feature:
                raise FeatureNotFoundError(f"Feature '{feature_name}' not found")
            
            # Validate input schema
            self._validate_schema(
                result_data,
                feature.input_schema,
                f"Feature '{feature_name}' input"
            )
            
            # Execute transformation code safely
            try:
                # Create a restricted namespace for code execution
                namespace = {
                    'pd': pd,
                    'data': result_data,
                    '__builtins__': {
                        'len': len,
                        'str': str,
                        'int': int,
                        'float': float,
                        'bool': bool,
                        'list': list,
                        'dict': dict,
                        'set': set,
                        'tuple': tuple,
                        'min': min,
                        'max': max,
                        'sum': sum,
                        'abs': abs,
                        'round': round,
                    }
                }
                
                # Execute transformation
                exec(feature.transformation_code, namespace)
                
                # Get result (transformation should assign to 'result' variable)
                if 'result' not in namespace:
                    raise FeatureComputationError(
                        f"Transformation code for '{feature_name}' must assign to 'result' variable"
                    )
                
                computed_feature = namespace['result']
                
                # Add computed feature to result data
                if isinstance(computed_feature, pd.Series):
                    result_data[feature_name] = computed_feature
                elif isinstance(computed_feature, pd.DataFrame):
                    # Merge computed features
                    for col in computed_feature.columns:
                        result_data[col] = computed_feature[col]
                else:
                    raise FeatureComputationError(
                        f"Transformation for '{feature_name}' must return pd.Series or pd.DataFrame"
                    )
                
                logger.debug(f"Computed feature '{feature_name}'")
                
            except Exception as e:
                raise FeatureComputationError(
                    f"Failed to compute feature '{feature_name}': {str(e)}"
                )
        
        # Return only requested features plus entity ID
        output_columns = [entity_id_column] + feature_names
        available_columns = [col for col in output_columns if col in result_data.columns]
        
        return result_data[available_columns]
    
    def get_features(
        self,
        feature_names: List[str],
        entity_ids: List[str],
        mode: str = "online"
    ) -> pd.DataFrame:
        """
        Retrieve feature values for entities.
        
        Args:
            feature_names: List of feature names
            entity_ids: List of entity IDs
            mode: Serving mode ("online" or "batch")
            
        Returns:
            DataFrame with feature values
            
        Raises:
            FeatureNotFoundError: If feature not found
        """
        result_data = []
        
        for entity_id in entity_ids:
            entity_features = {"entity_id": entity_id}
            
            for feature_name in feature_names:
                # Check cache first (for online mode)
                if mode == "online" and self.cache:
                    cache_key = self._generate_cache_key(feature_name, entity_id)
                    cached_value = self.cache.get(cache_key)
                    if cached_value is not None:
                        entity_features[feature_name] = cached_value
                        continue
                
                # Get feature definition
                feature = self.repository.get_feature_by_name(feature_name)
                if not feature:
                    raise FeatureNotFoundError(f"Feature '{feature_name}' not found")
                
                # Get latest value from database
                values = self.repository.get_latest_feature_values(
                    feature.feature_id,
                    [entity_id]
                )
                
                if entity_id in values:
                    feature_value = values[entity_id]
                    entity_features[feature_name] = feature_value
                    
                    # Cache for online serving
                    if mode == "online" and self.cache:
                        cache_key = self._generate_cache_key(feature_name, entity_id)
                        self.cache.set(
                            cache_key,
                            feature_value,
                            ttl=settings.feature_store.cache_ttl
                        )
                else:
                    entity_features[feature_name] = None
            
            result_data.append(entity_features)
        
        return pd.DataFrame(result_data)
    
    def materialize_features(
        self,
        feature_names: List[str],
        input_data: pd.DataFrame,
        entity_id_column: str = "entity_id",
        dataset_version_id: Optional[UUID] = None
    ) -> int:
        """
        Pre-compute and store features for later retrieval.
        
        Args:
            feature_names: List of feature names to materialize
            input_data: Input DataFrame with raw data
            entity_id_column: Column name containing entity IDs
            dataset_version_id: Optional dataset version ID for lineage
            
        Returns:
            Number of feature values stored
            
        Raises:
            FeatureNotFoundError: If feature not found
            FeatureComputationError: If computation fails
        """
        # Compute features
        computed = self.compute_features(
            feature_names,
            input_data,
            entity_id_column
        )
        
        # Store feature values
        stored_count = 0
        
        for feature_name in feature_names:
            if feature_name not in computed.columns:
                logger.warning(f"Feature '{feature_name}' not in computed results")
                continue
            
            feature = self.repository.get_feature_by_name(feature_name)
            if not feature:
                raise FeatureNotFoundError(f"Feature '{feature_name}' not found")
            
            # Store each entity's feature value
            for _, row in computed.iterrows():
                entity_id = str(row[entity_id_column])
                feature_value = row[feature_name]
                
                # Skip null values
                if pd.isna(feature_value):
                    continue
                
                # Convert numpy types to Python types for JSON serialization
                if hasattr(feature_value, 'item'):
                    feature_value = feature_value.item()
                
                value = FeatureValue(
                    feature_id=feature.feature_id,
                    entity_id=entity_id,
                    feature_value=feature_value,
                    dataset_version_id=dataset_version_id
                )
                
                self.repository.store_feature_value(value)
                stored_count += 1
                
                # Invalidate cache
                if self.cache:
                    cache_key = self._generate_cache_key(feature_name, entity_id)
                    self.cache.delete(cache_key)
        
        logger.info(f"Materialized {stored_count} feature values for {len(feature_names)} features")
        return stored_count
    
    def get_feature_definition(self, feature_name: str) -> FeatureDefinition:
        """
        Get feature definition by name.
        
        Args:
            feature_name: Feature name
            
        Returns:
            FeatureDefinition instance
            
        Raises:
            FeatureNotFoundError: If feature not found
        """
        feature = self.repository.get_feature_by_name(feature_name)
        if not feature:
            raise FeatureNotFoundError(f"Feature '{feature_name}' not found")
        return feature
    
    def get_feature(self, feature_name: str) -> Optional[FeatureDefinition]:
        """
        Get feature by name (alias for get_feature_definition).

        Args:
            feature_name: Name of the feature

        Returns:
            FeatureDefinition or None if not found
        """
        return self.get_feature_definition(feature_name)

    def list_features(self, limit: int = 100) -> List[FeatureDefinition]:
        """
        List all registered features.
        
        Args:
            limit: Maximum number of results
            
        Returns:
            List of FeatureDefinition instances
        """
        return self.repository.list_features(limit)
