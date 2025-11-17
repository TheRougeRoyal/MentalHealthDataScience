"""Data version control system for dataset versioning and lineage tracking."""

import hashlib
import logging
import gzip
import pickle
from datetime import datetime
from typing import Optional, Dict, Any, List, Tuple
from uuid import UUID
import pandas as pd
import numpy as np
from scipy import stats
from pydantic import BaseModel, Field

from src.database.connection import DatabaseConnection
from src.ds.storage import StorageBackend

logger = logging.getLogger(__name__)


class DatasetVersion(BaseModel):
    """Represents a versioned dataset."""
    
    version_id: Optional[UUID] = None
    dataset_name: str
    version: str
    source: str
    num_rows: int
    num_columns: int
    schema: Dict[str, str]  # column -> dtype
    statistics: Dict[str, Any]  # summary stats
    storage_uri: str
    metadata: Optional[Dict[str, Any]] = None
    parent_version_id: Optional[UUID] = None
    created_at: Optional[datetime] = None
    content_hash: Optional[str] = None


class DriftReport(BaseModel):
    """Data drift analysis report."""
    
    report_id: Optional[UUID] = None
    version_id1: UUID
    version_id2: UUID
    drift_detected: bool
    drift_score: float
    feature_drifts: Dict[str, float]  # feature -> drift score
    statistical_tests: Dict[str, Dict[str, Any]]
    recommendations: List[str]
    created_at: Optional[datetime] = None


class DataLineage(BaseModel):
    """Data transformation lineage record."""
    
    lineage_id: Optional[UUID] = None
    input_version_id: UUID
    output_version_id: UUID
    transformation_type: str
    transformation_code: Optional[str] = None
    created_at: Optional[datetime] = None


class DataVersionRepository:
    """Repository for data versioning database operations."""
    
    def __init__(self, db_connection: DatabaseConnection):
        """
        Initialize repository.
        
        Args:
            db_connection: Database connection instance
        """
        self.db = db_connection
    
    def create_dataset_version(self, dataset_version: DatasetVersion) -> UUID:
        """
        Create a new dataset version.
        
        Args:
            dataset_version: DatasetVersion model instance
            
        Returns:
            UUID of created dataset version
        """
        with self.db.get_cursor() as cur:
            cur.execute(
                """
                INSERT INTO dataset_versions (
                    dataset_name, version, source, num_rows, num_columns,
                    schema, statistics, storage_uri, metadata, parent_version_id,
                    content_hash
                )
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                RETURNING version_id
                """,
                (
                    dataset_version.dataset_name,
                    dataset_version.version,
                    dataset_version.source,
                    dataset_version.num_rows,
                    dataset_version.num_columns,
                    dataset_version.schema,
                    dataset_version.statistics,
                    dataset_version.storage_uri,
                    dataset_version.metadata,
                    str(dataset_version.parent_version_id) if dataset_version.parent_version_id else None,
                    dataset_version.content_hash,
                )
            )
            result = cur.fetchone()
            version_id = result["version_id"]
            logger.info(f"Created dataset version {version_id}: {dataset_version.dataset_name} v{dataset_version.version}")
            return version_id
    
    def get_dataset_version_by_id(self, version_id: UUID) -> Optional[DatasetVersion]:
        """
        Get dataset version by ID.
        
        Args:
            version_id: Version UUID
            
        Returns:
            DatasetVersion instance or None if not found
        """
        with self.db.get_cursor() as cur:
            cur.execute(
                "SELECT * FROM dataset_versions WHERE version_id = %s",
                (str(version_id),)
            )
            row = cur.fetchone()
            return DatasetVersion(**row) if row else None
    
    def get_dataset_version_by_name_and_version(
        self,
        dataset_name: str,
        version: str
    ) -> Optional[DatasetVersion]:
        """
        Get dataset version by name and version string.
        
        Args:
            dataset_name: Dataset name
            version: Version string
            
        Returns:
            DatasetVersion instance or None if not found
        """
        with self.db.get_cursor() as cur:
            cur.execute(
                """
                SELECT * FROM dataset_versions
                WHERE dataset_name = %s AND version = %s
                """,
                (dataset_name, version)
            )
            row = cur.fetchone()
            return DatasetVersion(**row) if row else None
    
    def get_latest_dataset_version(self, dataset_name: str) -> Optional[DatasetVersion]:
        """
        Get latest version of a dataset.
        
        Args:
            dataset_name: Dataset name
            
        Returns:
            DatasetVersion instance or None if not found
        """
        with self.db.get_cursor() as cur:
            cur.execute(
                """
                SELECT * FROM dataset_versions
                WHERE dataset_name = %s
                ORDER BY created_at DESC
                LIMIT 1
                """,
                (dataset_name,)
            )
            row = cur.fetchone()
            return DatasetVersion(**row) if row else None
    
    def list_dataset_versions(self, dataset_name: str) -> List[DatasetVersion]:
        """
        List all versions of a dataset.
        
        Args:
            dataset_name: Dataset name
            
        Returns:
            List of DatasetVersion instances
        """
        with self.db.get_cursor() as cur:
            cur.execute(
                """
                SELECT * FROM dataset_versions
                WHERE dataset_name = %s
                ORDER BY created_at DESC
                """,
                (dataset_name,)
            )
            rows = cur.fetchall()
            return [DatasetVersion(**row) for row in rows]
    
    def find_duplicate_by_hash(
        self,
        dataset_name: str,
        content_hash: str
    ) -> Optional[DatasetVersion]:
        """
        Find existing dataset version with same content hash.
        
        Args:
            dataset_name: Dataset name
            content_hash: Content hash to search for
            
        Returns:
            DatasetVersion instance or None if not found
        """
        with self.db.get_cursor() as cur:
            cur.execute(
                """
                SELECT * FROM dataset_versions
                WHERE dataset_name = %s AND content_hash = %s
                ORDER BY created_at DESC
                LIMIT 1
                """,
                (dataset_name, content_hash)
            )
            row = cur.fetchone()
            return DatasetVersion(**row) if row else None
    
    def create_lineage(self, lineage: DataLineage) -> UUID:
        """
        Create a lineage record.
        
        Args:
            lineage: DataLineage model instance
            
        Returns:
            UUID of created lineage record
        """
        with self.db.get_cursor() as cur:
            cur.execute(
                """
                INSERT INTO data_lineage (
                    input_version_id, output_version_id,
                    transformation_type, transformation_code
                )
                VALUES (%s, %s, %s, %s)
                RETURNING lineage_id
                """,
                (
                    str(lineage.input_version_id),
                    str(lineage.output_version_id),
                    lineage.transformation_type,
                    lineage.transformation_code,
                )
            )
            result = cur.fetchone()
            lineage_id = result["lineage_id"]
            logger.info(f"Created lineage record {lineage_id}")
            return lineage_id
    
    def get_upstream_lineage(self, version_id: UUID) -> List[Tuple[DatasetVersion, DataLineage]]:
        """
        Get upstream lineage (parent datasets).
        
        Args:
            version_id: Version UUID
            
        Returns:
            List of (DatasetVersion, DataLineage) tuples
        """
        with self.db.get_cursor() as cur:
            cur.execute(
                """
                SELECT dv.*, dl.*
                FROM data_lineage dl
                JOIN dataset_versions dv ON dl.input_version_id = dv.version_id
                WHERE dl.output_version_id = %s
                ORDER BY dl.created_at DESC
                """,
                (str(version_id),)
            )
            rows = cur.fetchall()
            
            results = []
            for row in rows:
                # Split row into dataset version and lineage parts
                dv_data = {k: v for k, v in row.items() if k in DatasetVersion.__fields__}
                dl_data = {k: v for k, v in row.items() if k in DataLineage.__fields__}
                results.append((DatasetVersion(**dv_data), DataLineage(**dl_data)))
            
            return results
    
    def get_downstream_lineage(self, version_id: UUID) -> List[Tuple[DatasetVersion, DataLineage]]:
        """
        Get downstream lineage (derived datasets).
        
        Args:
            version_id: Version UUID
            
        Returns:
            List of (DatasetVersion, DataLineage) tuples
        """
        with self.db.get_cursor() as cur:
            cur.execute(
                """
                SELECT dv.*, dl.*
                FROM data_lineage dl
                JOIN dataset_versions dv ON dl.output_version_id = dv.version_id
                WHERE dl.input_version_id = %s
                ORDER BY dl.created_at DESC
                """,
                (str(version_id),)
            )
            rows = cur.fetchall()
            
            results = []
            for row in rows:
                # Split row into dataset version and lineage parts
                dv_data = {k: v for k, v in row.items() if k in DatasetVersion.__fields__}
                dl_data = {k: v for k, v in row.items() if k in DataLineage.__fields__}
                results.append((DatasetVersion(**dv_data), DataLineage(**dl_data)))
            
            return results
    
    def create_drift_report(self, drift_report: DriftReport) -> UUID:
        """
        Create a drift report.
        
        Args:
            drift_report: DriftReport model instance
            
        Returns:
            UUID of created drift report
        """
        with self.db.get_cursor() as cur:
            cur.execute(
                """
                INSERT INTO drift_reports (
                    version_id1, version_id2, drift_detected, drift_score,
                    feature_drifts, statistical_tests
                )
                VALUES (%s, %s, %s, %s, %s, %s)
                RETURNING report_id
                """,
                (
                    str(drift_report.version_id1),
                    str(drift_report.version_id2),
                    drift_report.drift_detected,
                    drift_report.drift_score,
                    drift_report.feature_drifts,
                    drift_report.statistical_tests,
                )
            )
            result = cur.fetchone()
            report_id = result["report_id"]
            logger.info(f"Created drift report {report_id}")
            return report_id
    
    def get_drift_report(self, version_id1: UUID, version_id2: UUID) -> Optional[DriftReport]:
        """
        Get existing drift report between two versions.
        
        Args:
            version_id1: First version UUID
            version_id2: Second version UUID
            
        Returns:
            DriftReport instance or None if not found
        """
        with self.db.get_cursor() as cur:
            cur.execute(
                """
                SELECT * FROM drift_reports
                WHERE (version_id1 = %s AND version_id2 = %s)
                   OR (version_id1 = %s AND version_id2 = %s)
                ORDER BY created_at DESC
                LIMIT 1
                """,
                (str(version_id1), str(version_id2), str(version_id2), str(version_id1))
            )
            row = cur.fetchone()
            return DriftReport(**row) if row else None


class DataVersionControl:
    """Data versioning and lineage tracking system."""
    
    def __init__(
        self,
        storage_backend: StorageBackend,
        db_connection: DatabaseConnection,
        compression: bool = True
    ):
        """
        Initialize data version control.
        
        Args:
            storage_backend: Storage backend for datasets
            db_connection: Database connection
            compression: Whether to compress datasets (default: True)
        """
        self.storage = storage_backend
        self.db = db_connection
        self.repository = DataVersionRepository(db_connection)
        self.compression = compression
    
    def _compute_dataset_hash(self, data: pd.DataFrame) -> str:
        """
        Compute content hash of dataset for deduplication.
        
        Args:
            data: DataFrame to hash
            
        Returns:
            SHA256 hash of dataset content
        """
        # Create a deterministic representation of the dataframe
        hash_obj = hashlib.sha256()
        
        # Hash column names and types
        schema_str = str(sorted([(col, str(dtype)) for col, dtype in data.dtypes.items()]))
        hash_obj.update(schema_str.encode())
        
        # Hash data content
        for col in sorted(data.columns):
            col_data = data[col].values
            # Convert to bytes for hashing
            if col_data.dtype == object:
                # For object types, convert to string
                col_bytes = str(col_data.tolist()).encode()
            else:
                col_bytes = col_data.tobytes()
            hash_obj.update(col_bytes)
        
        return hash_obj.hexdigest()
    
    def _compute_statistics(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Compute summary statistics for dataset.
        
        Args:
            data: DataFrame to analyze
            
        Returns:
            Dictionary of statistics
        """
        stats = {
            "numeric_features": {},
            "categorical_features": {},
            "missing_values": {},
        }
        
        for col in data.columns:
            # Missing values
            missing_count = data[col].isna().sum()
            if missing_count > 0:
                stats["missing_values"][col] = int(missing_count)
            
            # Numeric features
            if pd.api.types.is_numeric_dtype(data[col]):
                col_stats = {
                    "mean": float(data[col].mean()) if not data[col].isna().all() else None,
                    "std": float(data[col].std()) if not data[col].isna().all() else None,
                    "min": float(data[col].min()) if not data[col].isna().all() else None,
                    "max": float(data[col].max()) if not data[col].isna().all() else None,
                    "median": float(data[col].median()) if not data[col].isna().all() else None,
                }
                stats["numeric_features"][col] = col_stats
            
            # Categorical features
            elif pd.api.types.is_object_dtype(data[col]) or pd.api.types.is_categorical_dtype(data[col]):
                unique_count = data[col].nunique()
                stats["categorical_features"][col] = {
                    "unique_count": int(unique_count),
                    "top_values": data[col].value_counts().head(5).to_dict() if unique_count > 0 else {},
                }
        
        return stats
    
    def _save_dataset(self, data: pd.DataFrame, dataset_name: str, version: str) -> str:
        """
        Save dataset to storage.
        
        Args:
            data: DataFrame to save
            dataset_name: Dataset name
            version: Version string
            
        Returns:
            Storage URI
        """
        # Create path for dataset
        path = f"datasets/{dataset_name}/{version}/data.pkl"
        
        # Serialize dataframe
        data_bytes = pickle.dumps(data)
        
        # Compress if enabled
        if self.compression:
            data_bytes = gzip.compress(data_bytes)
            path += ".gz"
        
        # Save to storage
        uri = self.storage.save_artifact(data_bytes, path)
        logger.info(f"Saved dataset to {uri}")
        
        return uri
    
    def _load_dataset(self, uri: str) -> pd.DataFrame:
        """
        Load dataset from storage.
        
        Args:
            uri: Storage URI
            
        Returns:
            DataFrame
        """
        # Load from storage
        data_bytes = self.storage.load_artifact(uri)
        
        # Decompress if needed
        if uri.endswith(".gz"):
            data_bytes = gzip.decompress(data_bytes)
        
        # Deserialize dataframe
        data = pickle.loads(data_bytes)
        
        return data

    def register_dataset(
        self,
        dataset: pd.DataFrame,
        dataset_name: str,
        source: str,
        version: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        parent_version_id: Optional[UUID] = None
    ) -> DatasetVersion:
        """
        Register a new dataset version.
        
        Args:
            dataset: DataFrame to register
            dataset_name: Name of the dataset
            source: Source description
            version: Optional version string (auto-generated if not provided)
            metadata: Optional metadata dictionary
            parent_version_id: Optional parent version for lineage
            
        Returns:
            DatasetVersion instance
        """
        # Compute content hash for deduplication
        content_hash = self._compute_dataset_hash(dataset)
        
        # Check for duplicate
        existing = self.repository.find_duplicate_by_hash(dataset_name, content_hash)
        if existing:
            logger.info(f"Dataset {dataset_name} with same content already exists as version {existing.version}")
            return existing
        
        # Generate version if not provided
        if version is None:
            # Get latest version and increment
            latest = self.repository.get_latest_dataset_version(dataset_name)
            if latest:
                # Try to parse version as integer and increment
                try:
                    latest_num = int(latest.version)
                    version = str(latest_num + 1)
                except ValueError:
                    # If not numeric, use timestamp
                    version = datetime.now().strftime("%Y%m%d%H%M%S")
            else:
                version = "1"
        
        # Compute statistics
        statistics = self._compute_statistics(dataset)
        
        # Get schema
        schema = {col: str(dtype) for col, dtype in dataset.dtypes.items()}
        
        # Save dataset to storage
        storage_uri = self._save_dataset(dataset, dataset_name, version)
        
        # Create dataset version record
        dataset_version = DatasetVersion(
            dataset_name=dataset_name,
            version=version,
            source=source,
            num_rows=len(dataset),
            num_columns=len(dataset.columns),
            schema=schema,
            statistics=statistics,
            storage_uri=storage_uri,
            metadata=metadata,
            parent_version_id=parent_version_id,
            content_hash=content_hash,
        )
        
        # Save to database
        version_id = self.repository.create_dataset_version(dataset_version)
        dataset_version.version_id = version_id
        
        logger.info(f"Registered dataset {dataset_name} version {version} with {len(dataset)} rows")
        
        return dataset_version
    
    def get_dataset(
        self,
        dataset_name: str,
        version: Optional[str] = None
    ) -> Tuple[pd.DataFrame, DatasetVersion]:
        """
        Retrieve a specific dataset version.
        
        Args:
            dataset_name: Name of the dataset
            version: Optional version string (retrieves latest if not provided)
            
        Returns:
            Tuple of (DataFrame, DatasetVersion)
            
        Raises:
            ValueError: If dataset not found
        """
        # Get dataset version metadata
        if version is None:
            dataset_version = self.repository.get_latest_dataset_version(dataset_name)
            if not dataset_version:
                raise ValueError(f"Dataset {dataset_name} not found")
        else:
            dataset_version = self.repository.get_dataset_version_by_name_and_version(
                dataset_name, version
            )
            if not dataset_version:
                raise ValueError(f"Dataset {dataset_name} version {version} not found")
        
        # Load dataset from storage
        data = self._load_dataset(dataset_version.storage_uri)
        
        logger.info(f"Retrieved dataset {dataset_name} version {dataset_version.version}")
        
        return data, dataset_version
    
    def list_versions(self, dataset_name: str) -> List[DatasetVersion]:
        """
        List all versions of a dataset.
        
        Args:
            dataset_name: Name of the dataset
            
        Returns:
            List of DatasetVersion instances
        """
        return self.repository.list_dataset_versions(dataset_name)
    
    def list_datasets(self) -> Dict[str, List[Dict[str, Any]]]:
        """
        List all datasets with their versions.
        
        Returns:
            Dictionary mapping dataset names to list of version info
        """
        with self.db.get_cursor() as cur:
            cur.execute(
                """
                SELECT dataset_name, version_id, version, num_rows, num_columns, 
                       source, created_at
                FROM dataset_versions
                ORDER BY dataset_name, created_at DESC
                """
            )
            rows = cur.fetchall()
        
        # Group by dataset name
        datasets = {}
        for row in rows:
            dataset_name = row[0]
            if dataset_name not in datasets:
                datasets[dataset_name] = []
            
            datasets[dataset_name].append({
                'version_id': str(row[1]),
                'version': row[2],
                'num_rows': row[3],
                'num_columns': row[4],
                'source': row[5],
                'created_at': row[6]
            })
        
        return datasets
    
    def track_transformation(
        self,
        input_version_id: UUID,
        output_version_id: UUID,
        transformation_type: str,
        transformation_code: Optional[str] = None
    ) -> UUID:
        """
        Track data transformation lineage.
        
        Args:
            input_version_id: Input dataset version UUID
            output_version_id: Output dataset version UUID
            transformation_type: Type of transformation (e.g., "filter", "aggregate", "join")
            transformation_code: Optional code/description of transformation
            
        Returns:
            UUID of created lineage record
        """
        lineage = DataLineage(
            input_version_id=input_version_id,
            output_version_id=output_version_id,
            transformation_type=transformation_type,
            transformation_code=transformation_code,
        )
        
        lineage_id = self.repository.create_lineage(lineage)
        logger.info(f"Tracked transformation {transformation_type} from {input_version_id} to {output_version_id}")
        
        return lineage_id
    
    def get_lineage(
        self,
        dataset_version_id: UUID,
        direction: str = "upstream"
    ) -> List[Tuple[DatasetVersion, DataLineage]]:
        """
        Get dataset lineage (upstream or downstream).
        
        Args:
            dataset_version_id: Dataset version UUID
            direction: "upstream" for parent datasets, "downstream" for derived datasets
            
        Returns:
            List of (DatasetVersion, DataLineage) tuples
            
        Raises:
            ValueError: If invalid direction
        """
        if direction == "upstream":
            return self.repository.get_upstream_lineage(dataset_version_id)
        elif direction == "downstream":
            return self.repository.get_downstream_lineage(dataset_version_id)
        else:
            raise ValueError(f"Invalid direction: {direction}. Must be 'upstream' or 'downstream'")
    
    def visualize_lineage(
        self,
        dataset_version_id: UUID,
        max_depth: int = 3
    ) -> Dict[str, Any]:
        """
        Create lineage visualization data structure.
        
        Args:
            dataset_version_id: Dataset version UUID
            max_depth: Maximum depth to traverse
            
        Returns:
            Dictionary with nodes and edges for visualization
        """
        nodes = {}
        edges = []
        
        def traverse(version_id: UUID, depth: int, visited: set):
            if depth > max_depth or version_id in visited:
                return
            
            visited.add(version_id)
            
            # Get version info
            version = self.repository.get_dataset_version_by_id(version_id)
            if version:
                nodes[str(version_id)] = {
                    "name": f"{version.dataset_name} v{version.version}",
                    "dataset_name": version.dataset_name,
                    "version": version.version,
                    "num_rows": version.num_rows,
                    "created_at": version.created_at.isoformat() if version.created_at else None,
                }
            
            # Get upstream lineage
            upstream = self.repository.get_upstream_lineage(version_id)
            for parent_version, lineage in upstream:
                edges.append({
                    "from": str(parent_version.version_id),
                    "to": str(version_id),
                    "transformation": lineage.transformation_type,
                })
                traverse(parent_version.version_id, depth + 1, visited)
            
            # Get downstream lineage
            downstream = self.repository.get_downstream_lineage(version_id)
            for child_version, lineage in downstream:
                edges.append({
                    "from": str(version_id),
                    "to": str(child_version.version_id),
                    "transformation": lineage.transformation_type,
                })
                traverse(child_version.version_id, depth + 1, visited)
        
        traverse(dataset_version_id, 0, set())
        
        return {
            "nodes": list(nodes.values()),
            "edges": edges,
        }
    
    def detect_drift(
        self,
        dataset_version_id1: UUID,
        dataset_version_id2: UUID,
        drift_threshold: float = 0.05
    ) -> DriftReport:
        """
        Detect statistical drift between two dataset versions.
        
        Args:
            dataset_version_id1: First dataset version UUID
            dataset_version_id2: Second dataset version UUID
            drift_threshold: P-value threshold for drift detection (default: 0.05)
            
        Returns:
            DriftReport instance
        """
        # Check for existing report
        existing_report = self.repository.get_drift_report(
            dataset_version_id1, dataset_version_id2
        )
        if existing_report:
            logger.info("Returning cached drift report")
            return existing_report
        
        # Get dataset versions
        version1 = self.repository.get_dataset_version_by_id(dataset_version_id1)
        version2 = self.repository.get_dataset_version_by_id(dataset_version_id2)
        
        if not version1 or not version2:
            raise ValueError("One or both dataset versions not found")
        
        # Load datasets
        data1 = self._load_dataset(version1.storage_uri)
        data2 = self._load_dataset(version2.storage_uri)
        
        # Find common columns
        common_cols = set(data1.columns) & set(data2.columns)
        
        feature_drifts = {}
        statistical_tests = {}
        
        for col in common_cols:
            col1 = data1[col].dropna()
            col2 = data2[col].dropna()
            
            if len(col1) == 0 or len(col2) == 0:
                continue
            
            # Numerical features: Kolmogorov-Smirnov test
            if pd.api.types.is_numeric_dtype(data1[col]) and pd.api.types.is_numeric_dtype(data2[col]):
                try:
                    statistic, p_value = stats.ks_2samp(col1, col2)
                    feature_drifts[col] = float(statistic)
                    statistical_tests[col] = {
                        "test": "kolmogorov_smirnov",
                        "statistic": float(statistic),
                        "p_value": float(p_value),
                        "drift_detected": p_value < drift_threshold,
                    }
                except Exception as e:
                    logger.warning(f"Failed to compute KS test for {col}: {e}")
            
            # Categorical features: Chi-square test
            elif pd.api.types.is_object_dtype(data1[col]) or pd.api.types.is_categorical_dtype(data1[col]):
                try:
                    # Get value counts
                    counts1 = col1.value_counts()
                    counts2 = col2.value_counts()
                    
                    # Align categories
                    all_categories = set(counts1.index) | set(counts2.index)
                    aligned1 = [counts1.get(cat, 0) for cat in all_categories]
                    aligned2 = [counts2.get(cat, 0) for cat in all_categories]
                    
                    # Chi-square test
                    if sum(aligned1) > 0 and sum(aligned2) > 0:
                        statistic, p_value = stats.chisquare(
                            f_obs=aligned2,
                            f_exp=aligned1,
                        )
                        
                        # Normalize statistic to [0, 1] range
                        drift_score = min(1.0, statistic / (len(all_categories) * 10))
                        
                        feature_drifts[col] = float(drift_score)
                        statistical_tests[col] = {
                            "test": "chi_square",
                            "statistic": float(statistic),
                            "p_value": float(p_value),
                            "drift_detected": p_value < drift_threshold,
                        }
                except Exception as e:
                    logger.warning(f"Failed to compute chi-square test for {col}: {e}")
        
        # Compute aggregate drift score
        if feature_drifts:
            aggregate_drift_score = float(np.mean(list(feature_drifts.values())))
        else:
            aggregate_drift_score = 0.0
        
        # Determine if drift detected
        drift_detected = any(
            test.get("drift_detected", False)
            for test in statistical_tests.values()
        )
        
        # Generate recommendations
        recommendations = self._generate_drift_recommendations(
            feature_drifts, statistical_tests, aggregate_drift_score
        )
        
        # Create drift report
        drift_report = DriftReport(
            version_id1=dataset_version_id1,
            version_id2=dataset_version_id2,
            drift_detected=drift_detected,
            drift_score=aggregate_drift_score,
            feature_drifts=feature_drifts,
            statistical_tests=statistical_tests,
            recommendations=recommendations,
        )
        
        # Save to database
        report_id = self.repository.create_drift_report(drift_report)
        drift_report.report_id = report_id
        
        logger.info(f"Detected drift between versions: drift_score={aggregate_drift_score:.3f}, drift_detected={drift_detected}")
        
        return drift_report
    
    def _generate_drift_recommendations(
        self,
        feature_drifts: Dict[str, float],
        statistical_tests: Dict[str, Dict[str, Any]],
        aggregate_drift_score: float
    ) -> List[str]:
        """
        Generate recommendations based on drift analysis.
        
        Args:
            feature_drifts: Feature-level drift scores
            statistical_tests: Statistical test results
            aggregate_drift_score: Aggregate drift score
            
        Returns:
            List of recommendation strings
        """
        recommendations = []
        
        # Overall drift assessment
        if aggregate_drift_score < 0.1:
            recommendations.append("Low drift detected. Data distributions are similar.")
        elif aggregate_drift_score < 0.3:
            recommendations.append("Moderate drift detected. Monitor model performance closely.")
        else:
            recommendations.append("High drift detected. Consider retraining models with new data.")
        
        # Feature-specific recommendations
        high_drift_features = [
            feature for feature, score in feature_drifts.items()
            if score > 0.3
        ]
        
        if high_drift_features:
            recommendations.append(
                f"Features with high drift: {', '.join(high_drift_features[:5])}. "
                "Investigate these features for data quality issues or distribution changes."
            )
        
        # Statistical significance
        significant_features = [
            feature for feature, test in statistical_tests.items()
            if test.get("drift_detected", False)
        ]
        
        if significant_features:
            recommendations.append(
                f"{len(significant_features)} features show statistically significant drift. "
                "Review feature engineering and preprocessing steps."
            )
        
        return recommendations
