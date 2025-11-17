"""Tests for feature store system."""

import pytest
import pandas as pd
from datetime import datetime
from uuid import UUID, uuid4

from src.ds.feature_store import (
    FeatureStore,
    FeatureDefinition,
    FeatureValue,
    FeatureRepository,
    FeatureStoreError,
    FeatureNotFoundError,
    FeatureComputationError,
    MemoryCache,
    RedisCache,
)
from src.exceptions import ValidationError
from src.database.connection import DatabaseConnection


class TestFeatureDefinition:
    """Test feature definition model."""

    def test_feature_definition_valid(self):
        """Test valid feature definition creation."""
        feature = FeatureDefinition(
            feature_name="age_group",
            feature_type="categorical",
            transformation_code="result = pd.cut(data['age'], bins=[0, 18, 65, 100])",
            input_schema={"age": "int"},
            output_schema={"age_group": "categorical"},
            version="1.0.0",
            dependencies=[],
            owner="data_team"
        )
        assert feature.feature_name == "age_group"
        assert feature.feature_type == "categorical"
        assert feature.version == "1.0.0"

    def test_feature_definition_invalid_type(self):
        """Test feature definition with invalid type."""
        with pytest.raises(ValueError):
            FeatureDefinition(
                feature_name="test",
                feature_type="invalid_type",
                transformation_code="result = data['x']",
                input_schema={},
                output_schema={}
            )

    def test_feature_definition_schema_validation(self):
        """Test schema validation."""
        with pytest.raises(ValueError):
            FeatureDefinition(
                feature_name="test",
                feature_type="numeric",
                transformation_code="result = data['x']",
                input_schema={"col": 123},  # Invalid: value must be string
                output_schema={}
            )


class TestMemoryCache:
    """Test memory cache backend."""

    def test_cache_set_get(self):
        """Test setting and getting cache values."""
        cache = MemoryCache()
        cache.set("key1", "value1")
        assert cache.get("key1") == "value1"

    def test_cache_get_nonexistent(self):
        """Test getting nonexistent key."""
        cache = MemoryCache()
        assert cache.get("nonexistent") is None

    def test_cache_delete(self):
        """Test deleting cache entry."""
        cache = MemoryCache()
        cache.set("key1", "value1")
        cache.delete("key1")
        assert cache.get("key1") is None

    def test_cache_clear(self):
        """Test clearing all cache entries."""
        cache = MemoryCache()
        cache.set("key1", "value1")
        cache.set("key2", "value2")
        cache.clear()
        assert cache.get("key1") is None
        assert cache.get("key2") is None

    def test_cache_ttl_expiry(self):
        """Test cache TTL expiry."""
        import time
        cache = MemoryCache()
        cache.set("key1", "value1", ttl=1)
        assert cache.get("key1") == "value1"
        time.sleep(1.1)
        assert cache.get("key1") is None


@pytest.fixture
def mock_db_connection(mocker):
    """Create mock database connection."""
    mock_conn = mocker.Mock(spec=DatabaseConnection)
    mock_cursor = mocker.MagicMock()
    mock_conn.get_cursor.return_value.__enter__.return_value = mock_cursor
    return mock_conn, mock_cursor


@pytest.fixture
def feature_repository(mock_db_connection):
    """Create feature repository for testing."""
    mock_conn, _ = mock_db_connection
    return FeatureRepository(mock_conn)


@pytest.fixture
def feature_store(mock_db_connection):
    """Create feature store for testing."""
    mock_conn, _ = mock_db_connection
    cache = MemoryCache()
    return FeatureStore(mock_conn, cache_backend=cache)


class TestFeatureRepository:
    """Test feature repository database operations."""

    def test_create_feature(self, feature_repository, mock_db_connection):
        """Test creating a feature."""
        _, mock_cursor = mock_db_connection
        
        feature_id = uuid4()
        mock_cursor.fetchone.return_value = {"feature_id": feature_id}
        
        feature = FeatureDefinition(
            feature_name="test_feature",
            feature_type="numeric",
            transformation_code="result = data['x'] * 2",
            input_schema={"x": "float"},
            output_schema={"test_feature": "float"},
            version="1.0.0"
        )
        
        result_id = feature_repository.create_feature(feature)
        assert result_id == feature_id
        assert mock_cursor.execute.called

    def test_get_feature_by_name(self, feature_repository, mock_db_connection):
        """Test getting feature by name."""
        _, mock_cursor = mock_db_connection
        
        mock_cursor.fetchone.return_value = {
            "feature_id": uuid4(),
            "feature_name": "test_feature",
            "feature_type": "numeric",
            "description": None,
            "transformation_code": "result = data['x']",
            "input_schema": {"x": "float"},
            "output_schema": {"test_feature": "float"},
            "version": "1.0.0",
            "dependencies": [],
            "owner": None,
            "created_at": datetime.utcnow(),
            "updated_at": datetime.utcnow()
        }
        
        feature = feature_repository.get_feature_by_name("test_feature")
        assert feature is not None
        assert feature.feature_name == "test_feature"

    def test_get_feature_not_found(self, feature_repository, mock_db_connection):
        """Test getting nonexistent feature."""
        _, mock_cursor = mock_db_connection
        mock_cursor.fetchone.return_value = None
        
        feature = feature_repository.get_feature_by_name("nonexistent")
        assert feature is None

    def test_list_features(self, feature_repository, mock_db_connection):
        """Test listing features."""
        _, mock_cursor = mock_db_connection
        
        mock_cursor.fetchall.return_value = [
            {
                "feature_id": uuid4(),
                "feature_name": "feature1",
                "feature_type": "numeric",
                "description": None,
                "transformation_code": "result = data['x']",
                "input_schema": {},
                "output_schema": {},
                "version": "1.0.0",
                "dependencies": [],
                "owner": None,
                "created_at": datetime.utcnow(),
                "updated_at": datetime.utcnow()
            }
        ]
        
        features = feature_repository.list_features()
        assert len(features) == 1
        assert features[0].feature_name == "feature1"

    def test_store_feature_value(self, feature_repository, mock_db_connection):
        """Test storing feature value."""
        _, mock_cursor = mock_db_connection
        
        value_id = uuid4()
        mock_cursor.fetchone.return_value = {"value_id": value_id}
        
        feature_value = FeatureValue(
            feature_id=uuid4(),
            entity_id="entity_123",
            feature_value=42.5
        )
        
        result_id = feature_repository.store_feature_value(feature_value)
        assert result_id == value_id
        assert mock_cursor.execute.called


class TestFeatureStore:
    """Test feature store operations."""

    def test_register_feature(self, feature_store, mock_db_connection):
        """Test registering a new feature."""
        _, mock_cursor = mock_db_connection
        
        # Mock feature doesn't exist
        mock_cursor.fetchone.side_effect = [
            None,  # Check if exists
            {"feature_id": uuid4()}  # Create feature
        ]
        
        feature_id = feature_store.register_feature(
            feature_name="test_feature",
            feature_type="numeric",
            transformation_code="result = data['x'] * 2",
            input_schema={"x": "float"},
            output_schema={"test_feature": "float"}
        )
        
        assert isinstance(feature_id, UUID)

    def test_register_feature_already_exists(self, feature_store, mock_db_connection):
        """Test registering feature that already exists."""
        _, mock_cursor = mock_db_connection
        
        # Mock feature exists
        mock_cursor.fetchone.return_value = {
            "feature_id": uuid4(),
            "feature_name": "test_feature",
            "feature_type": "numeric",
            "description": None,
            "transformation_code": "result = data['x']",
            "input_schema": {},
            "output_schema": {},
            "version": "1.0.0",
            "dependencies": [],
            "owner": None,
            "created_at": datetime.utcnow(),
            "updated_at": datetime.utcnow()
        }
        
        with pytest.raises(FeatureStoreError):
            feature_store.register_feature(
                feature_name="test_feature",
                feature_type="numeric",
                transformation_code="result = data['x']",
                input_schema={},
                output_schema={}
            )

    def test_register_feature_invalid_type(self, feature_store):
        """Test registering feature with invalid type."""
        with pytest.raises(ValidationError):
            feature_store.register_feature(
                feature_name="test_feature",
                feature_type="invalid_type",
                transformation_code="result = data['x']",
                input_schema={},
                output_schema={}
            )

    def test_compute_features_simple(self, feature_store, mock_db_connection):
        """Test computing simple feature."""
        _, mock_cursor = mock_db_connection
        
        # Mock feature definition
        mock_cursor.fetchone.return_value = {
            "feature_id": uuid4(),
            "feature_name": "double_x",
            "feature_type": "numeric",
            "description": None,
            "transformation_code": "result = data['x'] * 2",
            "input_schema": {"x": "float"},
            "output_schema": {"double_x": "float"},
            "version": "1.0.0",
            "dependencies": [],
            "owner": None,
            "created_at": datetime.utcnow(),
            "updated_at": datetime.utcnow()
        }
        
        input_data = pd.DataFrame({
            "entity_id": ["e1", "e2", "e3"],
            "x": [1.0, 2.0, 3.0]
        })
        
        result = feature_store.compute_features(
            feature_names=["double_x"],
            input_data=input_data
        )
        
        assert "entity_id" in result.columns
        assert "double_x" in result.columns
        assert list(result["double_x"]) == [2.0, 4.0, 6.0]

    def test_compute_features_missing_entity_id(self, feature_store):
        """Test computing features without entity_id column."""
        input_data = pd.DataFrame({
            "x": [1.0, 2.0, 3.0]
        })
        
        with pytest.raises(ValidationError):
            feature_store.compute_features(
                feature_names=["test_feature"],
                input_data=input_data
            )

    def test_compute_features_with_dependencies(self, feature_store, mock_db_connection):
        """Test computing features with dependencies."""
        _, mock_cursor = mock_db_connection
        
        # Mock two features: base and dependent
        def mock_fetchone_side_effect(*args, **kwargs):
            call_count = mock_cursor.fetchone.call_count
            if call_count == 1:  # First call for dependent feature
                return {
                    "feature_id": uuid4(),
                    "feature_name": "dependent_feature",
                    "feature_type": "numeric",
                    "description": None,
                    "transformation_code": "result = data['base_feature'] + 10",
                    "input_schema": {"base_feature": "float"},
                    "output_schema": {"dependent_feature": "float"},
                    "version": "1.0.0",
                    "dependencies": ["base_feature"],
                    "owner": None,
                    "created_at": datetime.utcnow(),
                    "updated_at": datetime.utcnow()
                }
            else:  # Second call for base feature
                return {
                    "feature_id": uuid4(),
                    "feature_name": "base_feature",
                    "feature_type": "numeric",
                    "description": None,
                    "transformation_code": "result = data['x'] * 2",
                    "input_schema": {"x": "float"},
                    "output_schema": {"base_feature": "float"},
                    "version": "1.0.0",
                    "dependencies": [],
                    "owner": None,
                    "created_at": datetime.utcnow(),
                    "updated_at": datetime.utcnow()
                }
        
        mock_cursor.fetchone.side_effect = mock_fetchone_side_effect
        
        input_data = pd.DataFrame({
            "entity_id": ["e1", "e2"],
            "x": [1.0, 2.0]
        })
        
        result = feature_store.compute_features(
            feature_names=["dependent_feature"],
            input_data=input_data
        )
        
        assert "dependent_feature" in result.columns
        assert list(result["dependent_feature"]) == [12.0, 14.0]

    def test_get_features_online(self, feature_store, mock_db_connection):
        """Test getting features in online mode."""
        _, mock_cursor = mock_db_connection
        
        # Mock feature definition
        feature_id = uuid4()
        mock_cursor.fetchone.return_value = {
            "feature_id": feature_id,
            "feature_name": "test_feature",
            "feature_type": "numeric",
            "description": None,
            "transformation_code": "result = data['x']",
            "input_schema": {},
            "output_schema": {},
            "version": "1.0.0",
            "dependencies": [],
            "owner": None,
            "created_at": datetime.utcnow(),
            "updated_at": datetime.utcnow()
        }
        
        # Mock feature values
        mock_cursor.fetchall.return_value = [
            {"entity_id": "e1", "feature_value": 10.5},
            {"entity_id": "e2", "feature_value": 20.5}
        ]
        
        result = feature_store.get_features(
            feature_names=["test_feature"],
            entity_ids=["e1", "e2"],
            mode="online"
        )
        
        assert len(result) == 2
        assert "test_feature" in result.columns

    def test_get_features_with_cache(self, feature_store, mock_db_connection):
        """Test getting features with caching."""
        _, mock_cursor = mock_db_connection
        
        feature_id = uuid4()
        mock_cursor.fetchone.return_value = {
            "feature_id": feature_id,
            "feature_name": "test_feature",
            "feature_type": "numeric",
            "description": None,
            "transformation_code": "result = data['x']",
            "input_schema": {},
            "output_schema": {},
            "version": "1.0.0",
            "dependencies": [],
            "owner": None,
            "created_at": datetime.utcnow(),
            "updated_at": datetime.utcnow()
        }
        
        # First call - cache miss
        mock_cursor.fetchall.return_value = [
            {"entity_id": "e1", "feature_value": 10.5}
        ]
        
        result1 = feature_store.get_features(
            feature_names=["test_feature"],
            entity_ids=["e1"],
            mode="online"
        )
        
        # Second call - should use cache
        result2 = feature_store.get_features(
            feature_names=["test_feature"],
            entity_ids=["e1"],
            mode="online"
        )
        
        assert result1.equals(result2)

    def test_materialize_features(self, feature_store, mock_db_connection):
        """Test materializing features."""
        _, mock_cursor = mock_db_connection
        
        feature_id = uuid4()
        
        # Mock feature definition
        mock_cursor.fetchone.return_value = {
            "feature_id": feature_id,
            "feature_name": "test_feature",
            "feature_type": "numeric",
            "description": None,
            "transformation_code": "result = data['x'] * 2",
            "input_schema": {"x": "float"},
            "output_schema": {"test_feature": "float"},
            "version": "1.0.0",
            "dependencies": [],
            "owner": None,
            "created_at": datetime.utcnow(),
            "updated_at": datetime.utcnow()
        }
        
        # Mock store_feature_value
        value_id = uuid4()
        mock_cursor.fetchone.side_effect = [
            {  # First call for feature definition
                "feature_id": feature_id,
                "feature_name": "test_feature",
                "feature_type": "numeric",
                "description": None,
                "transformation_code": "result = data['x'] * 2",
                "input_schema": {"x": "float"},
                "output_schema": {"test_feature": "float"},
                "version": "1.0.0",
                "dependencies": [],
                "owner": None,
                "created_at": datetime.utcnow(),
                "updated_at": datetime.utcnow()
            },
            {"value_id": value_id},  # Store value for e1
            {  # Get feature definition again
                "feature_id": feature_id,
                "feature_name": "test_feature",
                "feature_type": "numeric",
                "description": None,
                "transformation_code": "result = data['x'] * 2",
                "input_schema": {"x": "float"},
                "output_schema": {"test_feature": "float"},
                "version": "1.0.0",
                "dependencies": [],
                "owner": None,
                "created_at": datetime.utcnow(),
                "updated_at": datetime.utcnow()
            },
            {"value_id": value_id},  # Store value for e2
        ]
        
        input_data = pd.DataFrame({
            "entity_id": ["e1", "e2"],
            "x": [1.0, 2.0]
        })
        
        count = feature_store.materialize_features(
            feature_names=["test_feature"],
            input_data=input_data
        )
        
        assert count == 2

    def test_get_feature_definition(self, feature_store, mock_db_connection):
        """Test getting feature definition."""
        _, mock_cursor = mock_db_connection
        
        mock_cursor.fetchone.return_value = {
            "feature_id": uuid4(),
            "feature_name": "test_feature",
            "feature_type": "numeric",
            "description": "Test description",
            "transformation_code": "result = data['x']",
            "input_schema": {},
            "output_schema": {},
            "version": "1.0.0",
            "dependencies": [],
            "owner": None,
            "created_at": datetime.utcnow(),
            "updated_at": datetime.utcnow()
        }
        
        feature = feature_store.get_feature_definition("test_feature")
        assert feature.feature_name == "test_feature"
        assert feature.description == "Test description"

    def test_get_feature_definition_not_found(self, feature_store, mock_db_connection):
        """Test getting nonexistent feature definition."""
        _, mock_cursor = mock_db_connection
        mock_cursor.fetchone.return_value = None
        
        with pytest.raises(FeatureNotFoundError):
            feature_store.get_feature_definition("nonexistent")

    def test_list_features(self, feature_store, mock_db_connection):
        """Test listing features."""
        _, mock_cursor = mock_db_connection
        
        mock_cursor.fetchall.return_value = [
            {
                "feature_id": uuid4(),
                "feature_name": "feature1",
                "feature_type": "numeric",
                "description": None,
                "transformation_code": "result = data['x']",
                "input_schema": {},
                "output_schema": {},
                "version": "1.0.0",
                "dependencies": [],
                "owner": None,
                "created_at": datetime.utcnow(),
                "updated_at": datetime.utcnow()
            },
            {
                "feature_id": uuid4(),
                "feature_name": "feature2",
                "feature_type": "categorical",
                "description": None,
                "transformation_code": "result = data['y']",
                "input_schema": {},
                "output_schema": {},
                "version": "1.0.0",
                "dependencies": [],
                "owner": None,
                "created_at": datetime.utcnow(),
                "updated_at": datetime.utcnow()
            }
        ]
        
        features = feature_store.list_features()
        assert len(features) == 2
        assert features[0].feature_name == "feature1"
        assert features[1].feature_name == "feature2"

    def test_circular_dependency_detection(self, feature_store, mock_db_connection):
        """Test detection of circular dependencies."""
        _, mock_cursor = mock_db_connection
        
        # Mock circular dependency: A depends on B, B depends on A
        def mock_fetchone_side_effect(*args, **kwargs):
            call_count = mock_cursor.fetchone.call_count
            if call_count % 2 == 1:
                return {
                    "feature_id": uuid4(),
                    "feature_name": "feature_a",
                    "feature_type": "numeric",
                    "description": None,
                    "transformation_code": "result = data['feature_b']",
                    "input_schema": {},
                    "output_schema": {},
                    "version": "1.0.0",
                    "dependencies": ["feature_b"],
                    "owner": None,
                    "created_at": datetime.utcnow(),
                    "updated_at": datetime.utcnow()
                }
            else:
                return {
                    "feature_id": uuid4(),
                    "feature_name": "feature_b",
                    "feature_type": "numeric",
                    "description": None,
                    "transformation_code": "result = data['feature_a']",
                    "input_schema": {},
                    "output_schema": {},
                    "version": "1.0.0",
                    "dependencies": ["feature_a"],
                    "owner": None,
                    "created_at": datetime.utcnow(),
                    "updated_at": datetime.utcnow()
                }
        
        mock_cursor.fetchone.side_effect = mock_fetchone_side_effect
        
        input_data = pd.DataFrame({
            "entity_id": ["e1"],
            "x": [1.0]
        })
        
        with pytest.raises(FeatureStoreError, match="Circular dependency"):
            feature_store.compute_features(
                feature_names=["feature_a"],
                input_data=input_data
            )
