"""Tests for data version control system."""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime
from uuid import UUID, uuid4
import tempfile
import shutil

from src.ds.data_versioning import (
    DataVersionControl,
    DatasetVersion,
    DriftReport,
    DataLineage,
    DataVersionRepository
)
from src.ds.storage import FileSystemStorage
from src.database.connection import DatabaseConnection


@pytest.fixture
def temp_storage_dir():
    """Create temporary storage directory."""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    shutil.rmtree(temp_dir)


@pytest.fixture
def storage_backend(temp_storage_dir):
    """Create storage backend for testing."""
    return FileSystemStorage(base_path=temp_storage_dir)


@pytest.fixture
def mock_db_connection(mocker):
    """Create mock database connection."""
    mock_conn = mocker.Mock(spec=DatabaseConnection)
    mock_cursor = mocker.MagicMock()
    mock_conn.get_cursor.return_value.__enter__.return_value = mock_cursor
    return mock_conn, mock_cursor


@pytest.fixture
def data_version_control(storage_backend, mock_db_connection):
    """Create data version control for testing."""
    db_conn, _ = mock_db_connection
    return DataVersionControl(storage_backend, db_conn)


@pytest.fixture
def sample_dataframe():
    """Create sample dataframe for testing."""
    return pd.DataFrame({
        'numeric_col': [1, 2, 3, 4, 5],
        'categorical_col': ['a', 'b', 'a', 'c', 'b'],
        'float_col': [1.1, 2.2, 3.3, 4.4, 5.5]
    })



class TestDatasetVersionModel:
    """Test DatasetVersion model."""

    def test_dataset_version_valid(self):
        """Test valid dataset version creation."""
        version = DatasetVersion(
            dataset_name="test_dataset",
            version="1",
            source="test_source",
            num_rows=100,
            num_columns=5,
            schema={"col1": "int64", "col2": "object"},
            statistics={"mean": 50.0},
            storage_uri="file:///path/to/data",
            content_hash="abc123"
        )
        assert version.dataset_name == "test_dataset"
        assert version.version == "1"
        assert version.num_rows == 100


class TestDriftReportModel:
    """Test DriftReport model."""

    def test_drift_report_valid(self):
        """Test valid drift report creation."""
        report = DriftReport(
            version_id1=uuid4(),
            version_id2=uuid4(),
            drift_detected=True,
            drift_score=0.25,
            feature_drifts={"col1": 0.3, "col2": 0.2},
            statistical_tests={"col1": {"test": "ks", "p_value": 0.01}},
            recommendations=["Monitor model performance"]
        )
        assert report.drift_detected is True
        assert report.drift_score == 0.25


class TestDataVersionControl:
    """Test DataVersionControl functionality."""

    def test_register_dataset(self, data_version_control, sample_dataframe, mock_db_connection):
        """Test registering a new dataset."""
        _, mock_cursor = mock_db_connection
        
        version_id = uuid4()
        mock_cursor.fetchone.side_effect = [
            None,  # find_duplicate_by_hash
            None,  # get_latest_dataset_version
            {"version_id": version_id}  # create_dataset_version
        ]
        
        dataset_version = data_version_control.register_dataset(
            dataset=sample_dataframe,
            dataset_name="test_dataset",
            source="test_source"
        )
        
        assert dataset_version.version_id == version_id
        assert dataset_version.dataset_name == "test_dataset"
        assert dataset_version.num_rows == 5
        assert dataset_version.num_columns == 3
        assert dataset_version.version == "1"

    def test_register_dataset_with_version(self, data_version_control, sample_dataframe, mock_db_connection):
        """Test registering dataset with explicit version."""
        _, mock_cursor = mock_db_connection
        
        version_id = uuid4()
        mock_cursor.fetchone.side_effect = [
            None,  # find_duplicate_by_hash
            {"version_id": version_id}  # create_dataset_version
        ]
        
        dataset_version = data_version_control.register_dataset(
            dataset=sample_dataframe,
            dataset_name="test_dataset",
            source="test_source",
            version="custom_v1"
        )
        
        assert dataset_version.version == "custom_v1"


    def test_register_dataset_deduplication(self, data_version_control, sample_dataframe, mock_db_connection):
        """Test dataset deduplication by content hash."""
        _, mock_cursor = mock_db_connection
        
        existing_version_id = uuid4()
        existing_version = DatasetVersion(
            version_id=existing_version_id,
            dataset_name="test_dataset",
            version="1",
            source="test_source",
            num_rows=5,
            num_columns=3,
            schema={},
            statistics={},
            storage_uri="file:///existing",
            content_hash="same_hash"
        )
        
        mock_cursor.fetchone.return_value = {
            "version_id": existing_version_id,
            "dataset_name": "test_dataset",
            "version": "1",
            "source": "test_source",
            "num_rows": 5,
            "num_columns": 3,
            "schema": {},
            "statistics": {},
            "storage_uri": "file:///existing",
            "metadata": None,
            "parent_version_id": None,
            "created_at": datetime.utcnow(),
            "content_hash": "same_hash"
        }
        
        dataset_version = data_version_control.register_dataset(
            dataset=sample_dataframe,
            dataset_name="test_dataset",
            source="test_source"
        )
        
        # Should return existing version
        assert dataset_version.version_id == existing_version_id

    def test_register_dataset_auto_increment_version(self, data_version_control, sample_dataframe, mock_db_connection):
        """Test automatic version incrementing."""
        _, mock_cursor = mock_db_connection
        
        latest_version = DatasetVersion(
            version_id=uuid4(),
            dataset_name="test_dataset",
            version="5",
            source="test_source",
            num_rows=5,
            num_columns=3,
            schema={},
            statistics={},
            storage_uri="file:///latest",
            content_hash="old_hash"
        )
        
        new_version_id = uuid4()
        mock_cursor.fetchone.side_effect = [
            None,  # find_duplicate_by_hash
            {  # get_latest_dataset_version
                "version_id": latest_version.version_id,
                "dataset_name": "test_dataset",
                "version": "5",
                "source": "test_source",
                "num_rows": 5,
                "num_columns": 3,
                "schema": {},
                "statistics": {},
                "storage_uri": "file:///latest",
                "metadata": None,
                "parent_version_id": None,
                "created_at": datetime.utcnow(),
                "content_hash": "old_hash"
            },
            {"version_id": new_version_id}  # create_dataset_version
        ]
        
        dataset_version = data_version_control.register_dataset(
            dataset=sample_dataframe,
            dataset_name="test_dataset",
            source="test_source"
        )
        
        assert dataset_version.version == "6"

    def test_get_dataset_latest(self, data_version_control, sample_dataframe, mock_db_connection):
        """Test retrieving latest dataset version."""
        _, mock_cursor = mock_db_connection
        
        version_id = uuid4()
        
        # First register a dataset
        mock_cursor.fetchone.side_effect = [
            None,  # find_duplicate_by_hash
            None,  # get_latest_dataset_version (for register)
            {"version_id": version_id},  # create_dataset_version
            {  # get_latest_dataset_version (for get)
                "version_id": version_id,
                "dataset_name": "test_dataset",
                "version": "1",
                "source": "test_source",
                "num_rows": 5,
                "num_columns": 3,
                "schema": {"numeric_col": "int64", "categorical_col": "object", "float_col": "float64"},
                "statistics": {},
                "storage_uri": f"file://{data_version_control.storage.base_path}/datasets/test_dataset/1/data.pkl.gz",
                "metadata": None,
                "parent_version_id": None,
                "created_at": datetime.utcnow(),
                "content_hash": "hash123"
            }
        ]
        
        # Register dataset
        data_version_control.register_dataset(
            dataset=sample_dataframe,
            dataset_name="test_dataset",
            source="test_source"
        )
        
        # Retrieve latest
        data, dataset_version = data_version_control.get_dataset("test_dataset")
        
        assert dataset_version.version == "1"
        assert len(data) == 5
        assert list(data.columns) == ['numeric_col', 'categorical_col', 'float_col']


    def test_get_dataset_specific_version(self, data_version_control, sample_dataframe, mock_db_connection):
        """Test retrieving specific dataset version."""
        _, mock_cursor = mock_db_connection
        
        version_id = uuid4()
        
        mock_cursor.fetchone.side_effect = [
            None,  # find_duplicate_by_hash
            None,  # get_latest_dataset_version
            {"version_id": version_id},  # create_dataset_version
            {  # get_dataset_version_by_name_and_version
                "version_id": version_id,
                "dataset_name": "test_dataset",
                "version": "1",
                "source": "test_source",
                "num_rows": 5,
                "num_columns": 3,
                "schema": {},
                "statistics": {},
                "storage_uri": f"file://{data_version_control.storage.base_path}/datasets/test_dataset/1/data.pkl.gz",
                "metadata": None,
                "parent_version_id": None,
                "created_at": datetime.utcnow(),
                "content_hash": "hash123"
            }
        ]
        
        # Register dataset
        data_version_control.register_dataset(
            dataset=sample_dataframe,
            dataset_name="test_dataset",
            source="test_source"
        )
        
        # Retrieve specific version
        data, dataset_version = data_version_control.get_dataset("test_dataset", version="1")
        
        assert dataset_version.version == "1"
        assert len(data) == 5

    def test_get_dataset_not_found(self, data_version_control, mock_db_connection):
        """Test retrieving non-existent dataset."""
        _, mock_cursor = mock_db_connection
        
        mock_cursor.fetchone.return_value = None
        
        with pytest.raises(ValueError, match="not found"):
            data_version_control.get_dataset("nonexistent_dataset")

    def test_list_versions(self, data_version_control, mock_db_connection):
        """Test listing dataset versions."""
        _, mock_cursor = mock_db_connection
        
        mock_cursor.fetchall.return_value = [
            {
                "version_id": uuid4(),
                "dataset_name": "test_dataset",
                "version": "2",
                "source": "test_source",
                "num_rows": 10,
                "num_columns": 3,
                "schema": {},
                "statistics": {},
                "storage_uri": "file:///v2",
                "metadata": None,
                "parent_version_id": None,
                "created_at": datetime.utcnow(),
                "content_hash": "hash2"
            },
            {
                "version_id": uuid4(),
                "dataset_name": "test_dataset",
                "version": "1",
                "source": "test_source",
                "num_rows": 5,
                "num_columns": 3,
                "schema": {},
                "statistics": {},
                "storage_uri": "file:///v1",
                "metadata": None,
                "parent_version_id": None,
                "created_at": datetime.utcnow(),
                "content_hash": "hash1"
            }
        ]
        
        versions = data_version_control.list_versions("test_dataset")
        
        assert len(versions) == 2
        assert versions[0].version == "2"
        assert versions[1].version == "1"

    def test_track_transformation(self, data_version_control, mock_db_connection):
        """Test tracking data transformation lineage."""
        _, mock_cursor = mock_db_connection
        
        input_version_id = uuid4()
        output_version_id = uuid4()
        lineage_id = uuid4()
        
        mock_cursor.fetchone.return_value = {"lineage_id": lineage_id}
        
        result_id = data_version_control.track_transformation(
            input_version_id=input_version_id,
            output_version_id=output_version_id,
            transformation_type="filter",
            transformation_code="df[df['col'] > 5]"
        )
        
        assert result_id == lineage_id


    def test_get_lineage_upstream(self, data_version_control, mock_db_connection):
        """Test getting upstream lineage."""
        _, mock_cursor = mock_db_connection
        
        version_id = uuid4()
        parent_version_id = uuid4()
        lineage_id = uuid4()
        
        mock_cursor.fetchall.return_value = [
            {
                # DatasetVersion fields
                "version_id": parent_version_id,
                "dataset_name": "parent_dataset",
                "version": "1",
                "source": "source",
                "num_rows": 100,
                "num_columns": 5,
                "schema": {},
                "statistics": {},
                "storage_uri": "file:///parent",
                "metadata": None,
                "parent_version_id": None,
                "created_at": datetime.utcnow(),
                "content_hash": "hash",
                # DataLineage fields
                "lineage_id": lineage_id,
                "input_version_id": parent_version_id,
                "output_version_id": version_id,
                "transformation_type": "filter",
                "transformation_code": "code"
            }
        ]
        
        lineage = data_version_control.get_lineage(version_id, direction="upstream")
        
        assert len(lineage) == 1
        dataset_version, data_lineage = lineage[0]
        assert dataset_version.version_id == parent_version_id
        assert data_lineage.transformation_type == "filter"

    def test_get_lineage_downstream(self, data_version_control, mock_db_connection):
        """Test getting downstream lineage."""
        _, mock_cursor = mock_db_connection
        
        version_id = uuid4()
        child_version_id = uuid4()
        
        mock_cursor.fetchall.return_value = [
            {
                # DatasetVersion fields
                "version_id": child_version_id,
                "dataset_name": "child_dataset",
                "version": "1",
                "source": "source",
                "num_rows": 50,
                "num_columns": 3,
                "schema": {},
                "statistics": {},
                "storage_uri": "file:///child",
                "metadata": None,
                "parent_version_id": version_id,
                "created_at": datetime.utcnow(),
                "content_hash": "hash",
                # DataLineage fields
                "lineage_id": uuid4(),
                "input_version_id": version_id,
                "output_version_id": child_version_id,
                "transformation_type": "aggregate",
                "transformation_code": "groupby"
            }
        ]
        
        lineage = data_version_control.get_lineage(version_id, direction="downstream")
        
        assert len(lineage) == 1
        dataset_version, data_lineage = lineage[0]
        assert dataset_version.version_id == child_version_id
        assert data_lineage.transformation_type == "aggregate"

    def test_get_lineage_invalid_direction(self, data_version_control):
        """Test getting lineage with invalid direction."""
        with pytest.raises(ValueError, match="Invalid direction"):
            data_version_control.get_lineage(uuid4(), direction="invalid")

    def test_visualize_lineage(self, data_version_control, mock_db_connection):
        """Test lineage visualization data generation."""
        _, mock_cursor = mock_db_connection
        
        version_id = uuid4()
        parent_id = uuid4()
        
        # Mock get_dataset_version_by_id
        mock_cursor.fetchone.side_effect = [
            {
                "version_id": version_id,
                "dataset_name": "current",
                "version": "2",
                "source": "source",
                "num_rows": 100,
                "num_columns": 5,
                "schema": {},
                "statistics": {},
                "storage_uri": "file:///current",
                "metadata": None,
                "parent_version_id": parent_id,
                "created_at": datetime.utcnow(),
                "content_hash": "hash"
            },
            {
                "version_id": parent_id,
                "dataset_name": "parent",
                "version": "1",
                "source": "source",
                "num_rows": 200,
                "num_columns": 10,
                "schema": {},
                "statistics": {},
                "storage_uri": "file:///parent",
                "metadata": None,
                "parent_version_id": None,
                "created_at": datetime.utcnow(),
                "content_hash": "hash2"
            }
        ]
        
        # Mock lineage queries
        mock_cursor.fetchall.side_effect = [
            [  # upstream for version_id
                {
                    "version_id": parent_id,
                    "dataset_name": "parent",
                    "version": "1",
                    "source": "source",
                    "num_rows": 200,
                    "num_columns": 10,
                    "schema": {},
                    "statistics": {},
                    "storage_uri": "file:///parent",
                    "metadata": None,
                    "parent_version_id": None,
                    "created_at": datetime.utcnow(),
                    "content_hash": "hash2",
                    "lineage_id": uuid4(),
                    "input_version_id": parent_id,
                    "output_version_id": version_id,
                    "transformation_type": "filter",
                    "transformation_code": None
                }
            ],
            [],  # downstream for version_id
            [],  # upstream for parent_id
            []   # downstream for parent_id
        ]
        
        viz_data = data_version_control.visualize_lineage(version_id, max_depth=2)
        
        assert "nodes" in viz_data
        assert "edges" in viz_data
        assert len(viz_data["nodes"]) == 2
        assert len(viz_data["edges"]) == 1


    def test_detect_drift_numerical(self, data_version_control, mock_db_connection):
        """Test drift detection for numerical features."""
        _, mock_cursor = mock_db_connection
        
        version_id1 = uuid4()
        version_id2 = uuid4()
        
        # Create two datasets with different distributions
        df1 = pd.DataFrame({
            'numeric_col': np.random.normal(0, 1, 100),
            'categorical_col': ['a'] * 50 + ['b'] * 50
        })
        
        df2 = pd.DataFrame({
            'numeric_col': np.random.normal(5, 1, 100),  # Different mean
            'categorical_col': ['a'] * 50 + ['b'] * 50
        })
        
        # Register both datasets
        mock_cursor.fetchone.side_effect = [
            None,  # get_drift_report (not cached)
            {  # get_dataset_version_by_id for version1
                "version_id": version_id1,
                "dataset_name": "test_dataset",
                "version": "1",
                "source": "source",
                "num_rows": 100,
                "num_columns": 2,
                "schema": {},
                "statistics": {},
                "storage_uri": f"file://{data_version_control.storage.base_path}/datasets/test_dataset/1/data.pkl.gz",
                "metadata": None,
                "parent_version_id": None,
                "created_at": datetime.utcnow(),
                "content_hash": "hash1"
            },
            {  # get_dataset_version_by_id for version2
                "version_id": version_id2,
                "dataset_name": "test_dataset",
                "version": "2",
                "source": "source",
                "num_rows": 100,
                "num_columns": 2,
                "schema": {},
                "statistics": {},
                "storage_uri": f"file://{data_version_control.storage.base_path}/datasets/test_dataset/2/data.pkl.gz",
                "metadata": None,
                "parent_version_id": None,
                "created_at": datetime.utcnow(),
                "content_hash": "hash2"
            },
            {"report_id": uuid4()}  # create_drift_report
        ]
        
        # Save datasets to storage
        uri1 = data_version_control._save_dataset(df1, "test_dataset", "1")
        uri2 = data_version_control._save_dataset(df2, "test_dataset", "2")
        
        drift_report = data_version_control.detect_drift(version_id1, version_id2)
        
        assert drift_report.drift_detected is True
        assert 'numeric_col' in drift_report.feature_drifts
        assert drift_report.feature_drifts['numeric_col'] > 0.5  # High drift expected

    def test_detect_drift_categorical(self, data_version_control, mock_db_connection):
        """Test drift detection for categorical features."""
        _, mock_cursor = mock_db_connection
        
        version_id1 = uuid4()
        version_id2 = uuid4()
        
        # Create two datasets with different categorical distributions
        df1 = pd.DataFrame({
            'categorical_col': ['a'] * 70 + ['b'] * 30
        })
        
        df2 = pd.DataFrame({
            'categorical_col': ['a'] * 30 + ['b'] * 70  # Reversed distribution
        })
        
        mock_cursor.fetchone.side_effect = [
            None,  # get_drift_report
            {  # version1
                "version_id": version_id1,
                "dataset_name": "test_dataset",
                "version": "1",
                "source": "source",
                "num_rows": 100,
                "num_columns": 1,
                "schema": {},
                "statistics": {},
                "storage_uri": f"file://{data_version_control.storage.base_path}/datasets/test_dataset/1/data.pkl.gz",
                "metadata": None,
                "parent_version_id": None,
                "created_at": datetime.utcnow(),
                "content_hash": "hash1"
            },
            {  # version2
                "version_id": version_id2,
                "dataset_name": "test_dataset",
                "version": "2",
                "source": "source",
                "num_rows": 100,
                "num_columns": 1,
                "schema": {},
                "statistics": {},
                "storage_uri": f"file://{data_version_control.storage.base_path}/datasets/test_dataset/2/data.pkl.gz",
                "metadata": None,
                "parent_version_id": None,
                "created_at": datetime.utcnow(),
                "content_hash": "hash2"
            },
            {"report_id": uuid4()}
        ]
        
        uri1 = data_version_control._save_dataset(df1, "test_dataset", "1")
        uri2 = data_version_control._save_dataset(df2, "test_dataset", "2")
        
        drift_report = data_version_control.detect_drift(version_id1, version_id2)
        
        assert 'categorical_col' in drift_report.statistical_tests
        assert drift_report.statistical_tests['categorical_col']['test'] == 'chi_square'

    def test_detect_drift_cached(self, data_version_control, mock_db_connection):
        """Test drift detection returns cached report."""
        _, mock_cursor = mock_db_connection
        
        version_id1 = uuid4()
        version_id2 = uuid4()
        report_id = uuid4()
        
        # Mock existing drift report
        mock_cursor.fetchone.return_value = {
            "report_id": report_id,
            "version_id1": version_id1,
            "version_id2": version_id2,
            "drift_detected": True,
            "drift_score": 0.3,
            "feature_drifts": {"col1": 0.3},
            "statistical_tests": {},
            "created_at": datetime.utcnow()
        }
        
        drift_report = data_version_control.detect_drift(version_id1, version_id2)
        
        assert drift_report.report_id == report_id
        assert drift_report.drift_score == 0.3

    def test_compute_statistics(self, data_version_control, sample_dataframe):
        """Test computing dataset statistics."""
        stats = data_version_control._compute_statistics(sample_dataframe)
        
        assert "numeric_features" in stats
        assert "categorical_features" in stats
        assert "missing_values" in stats
        
        assert "numeric_col" in stats["numeric_features"]
        assert "mean" in stats["numeric_features"]["numeric_col"]
        assert stats["numeric_features"]["numeric_col"]["mean"] == 3.0
        
        assert "categorical_col" in stats["categorical_features"]
        assert stats["categorical_features"]["categorical_col"]["unique_count"] == 3

    def test_compute_dataset_hash(self, data_version_control, sample_dataframe):
        """Test computing dataset content hash."""
        hash1 = data_version_control._compute_dataset_hash(sample_dataframe)
        hash2 = data_version_control._compute_dataset_hash(sample_dataframe)
        
        # Same data should produce same hash
        assert hash1 == hash2
        
        # Different data should produce different hash
        different_df = sample_dataframe.copy()
        different_df['numeric_col'] = different_df['numeric_col'] * 2
        hash3 = data_version_control._compute_dataset_hash(different_df)
        
        assert hash1 != hash3

    def test_save_and_load_dataset(self, data_version_control, sample_dataframe):
        """Test saving and loading dataset."""
        uri = data_version_control._save_dataset(sample_dataframe, "test_dataset", "1")
        
        assert uri is not None
        assert "test_dataset" in uri
        
        loaded_df = data_version_control._load_dataset(uri)
        
        pd.testing.assert_frame_equal(sample_dataframe, loaded_df)
