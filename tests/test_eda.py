"""
Unit tests for EDA Module
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import tempfile
import shutil

from src.ds.eda import EDAModule, EDAReport, DataQualityIssue


@pytest.fixture
def sample_data():
    """Create sample dataset for testing"""
    np.random.seed(42)
    
    data = pd.DataFrame({
        'age': np.random.randint(18, 80, 100),
        'score': np.random.normal(50, 10, 100),
        'income': np.random.exponential(50000, 100),
        'category': np.random.choice(['A', 'B', 'C'], 100),
        'status': np.random.choice(['active', 'inactive'], 100)
    })
    
    # Add some missing values
    data.loc[0:5, 'score'] = np.nan
    data.loc[10:12, 'category'] = np.nan
    
    # Add some outliers
    data.loc[95:99, 'income'] = [500000, 600000, 700000, 800000, 900000]
    
    # Add duplicate rows
    data = pd.concat([data, data.iloc[0:3]], ignore_index=True)
    
    return data


@pytest.fixture
def eda_module():
    """Create EDA module instance"""
    return EDAModule()


@pytest.fixture
def temp_output_dir():
    """Create temporary directory for test outputs"""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    shutil.rmtree(temp_dir)


def test_generate_summary_statistics_numerical(eda_module, sample_data):
    """Test summary statistics generation for numerical features"""
    stats = eda_module.generate_summary_statistics(sample_data)
    
    # Check numerical columns
    assert 'age' in stats
    assert 'score' in stats
    assert 'income' in stats
    
    # Check statistical measures
    age_stats = stats['age']
    assert 'mean' in age_stats
    assert 'std' in age_stats
    assert 'min' in age_stats
    assert 'max' in age_stats
    assert 'median' in age_stats
    assert 'q25' in age_stats
    assert 'q75' in age_stats
    
    # Verify reasonable values
    assert 18 <= age_stats['min'] <= 80
    assert 18 <= age_stats['max'] <= 80


def test_generate_summary_statistics_categorical(eda_module, sample_data):
    """Test summary statistics generation for categorical features"""
    stats = eda_module.generate_summary_statistics(sample_data)
    
    # Check categorical columns
    assert 'category' in stats
    assert 'status' in stats
    
    # Check categorical measures
    cat_stats = stats['category']
    assert 'count' in cat_stats
    assert 'unique' in cat_stats
    assert 'top' in cat_stats
    assert 'freq' in cat_stats
    
    # Verify reasonable values
    assert cat_stats['unique'] <= 3  # A, B, C


def test_detect_missing_values(eda_module, sample_data):
    """Test detection of missing values"""
    issues = eda_module.detect_data_quality_issues(sample_data)
    
    # Find missing value issues
    missing_issues = [i for i in issues if i.issue_type == 'missing']
    
    assert len(missing_issues) > 0
    
    # Check that score column missing values are detected
    score_issues = [i for i in missing_issues if i.column == 'score']
    assert len(score_issues) > 0
    assert score_issues[0].affected_rows == 6


def test_detect_outliers(eda_module, sample_data):
    """Test detection of outliers"""
    issues = eda_module.detect_data_quality_issues(sample_data)
    
    # Find outlier issues
    outlier_issues = [i for i in issues if i.issue_type == 'outlier']
    
    assert len(outlier_issues) > 0
    
    # Check that income outliers are detected
    income_issues = [i for i in outlier_issues if i.column == 'income']
    assert len(income_issues) > 0


def test_detect_duplicates(eda_module, sample_data):
    """Test detection of duplicate rows"""
    issues = eda_module.detect_data_quality_issues(sample_data)
    
    # Find duplicate issues
    duplicate_issues = [i for i in issues if i.issue_type == 'duplicate']
    
    assert len(duplicate_issues) > 0
    assert duplicate_issues[0].affected_rows == 3


def test_analyze_correlations(eda_module, sample_data):
    """Test correlation analysis"""
    corr_matrix = eda_module.analyze_correlations(sample_data)
    
    # Check that correlation matrix is generated
    assert not corr_matrix.empty
    assert corr_matrix.shape[0] == corr_matrix.shape[1]
    
    # Check that numerical columns are included
    assert 'age' in corr_matrix.columns
    assert 'score' in corr_matrix.columns
    assert 'income' in corr_matrix.columns
    
    # Check diagonal is 1 (self-correlation)
    for col in corr_matrix.columns:
        assert abs(corr_matrix.loc[col, col] - 1.0) < 0.01


def test_analyze_dataset(eda_module, sample_data):
    """Test full dataset analysis"""
    report = eda_module.analyze_dataset(sample_data, dataset_name="test_dataset")
    
    # Check report structure
    assert isinstance(report, EDAReport)
    assert report.dataset_name == "test_dataset"
    assert report.num_rows == len(sample_data)
    assert report.num_columns == len(sample_data.columns)
    
    # Check summary statistics
    assert len(report.summary_statistics) > 0
    
    # Check missing values
    assert 'score' in report.missing_values
    assert report.missing_values['score'] == 6
    
    # Check data types
    assert len(report.data_types) == len(sample_data.columns)
    
    # Check quality issues
    assert len(report.quality_issues) > 0
    
    # Check recommendations
    assert len(report.recommendations) > 0


def test_generate_visualizations(eda_module, sample_data, temp_output_dir):
    """Test visualization generation"""
    viz_paths = eda_module.generate_visualizations(sample_data, output_dir=temp_output_dir)
    
    # Check that visualizations were created
    assert len(viz_paths) > 0
    
    # Check that files exist
    for path in viz_paths:
        assert Path(path).exists()
        assert Path(path).suffix == '.png'
    
    # Check for expected visualizations
    viz_names = [Path(p).stem for p in viz_paths]
    assert 'distributions' in viz_names
    assert 'boxplots' in viz_names
    assert 'correlation_heatmap' in viz_names
    assert 'missing_values' in viz_names


def test_export_report_html(eda_module, sample_data, temp_output_dir):
    """Test HTML report export"""
    # Generate report
    report = eda_module.analyze_dataset(sample_data, dataset_name="test_dataset")
    
    # Export as HTML
    output_path = Path(temp_output_dir) / "report.html"
    exported_path = eda_module.export_report(report, format="html", output_path=str(output_path))
    
    # Check that file was created
    assert Path(exported_path).exists()
    
    # Check content
    with open(exported_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    assert '<!DOCTYPE html>' in content
    assert 'test_dataset' in content
    assert 'Data Quality Issues' in content
    assert 'Summary Statistics' in content
    assert 'Recommendations' in content


def test_export_report_markdown(eda_module, sample_data, temp_output_dir):
    """Test Markdown report export"""
    # Generate report
    report = eda_module.analyze_dataset(sample_data, dataset_name="test_dataset")
    
    # Export as Markdown
    output_path = Path(temp_output_dir) / "report.md"
    exported_path = eda_module.export_report(report, format="markdown", output_path=str(output_path))
    
    # Check that file was created
    assert Path(exported_path).exists()
    
    # Check content
    with open(exported_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    assert '# Exploratory Data Analysis Report' in content
    assert 'test_dataset' in content
    assert '## Data Quality Issues' in content
    assert '## Summary Statistics' in content
    assert '## Recommendations' in content


def test_quality_issue_severity_levels(eda_module):
    """Test that quality issues have appropriate severity levels"""
    # Create data with varying levels of missing values
    data = pd.DataFrame({
        'critical_missing': [np.nan] * 60 + [1] * 40,  # 60% missing
        'high_missing': [np.nan] * 25 + [1] * 75,      # 25% missing
        'medium_missing': [np.nan] * 10 + [1] * 90,    # 10% missing
        'low_missing': [np.nan] * 2 + [1] * 98         # 2% missing
    })
    
    issues = eda_module.detect_data_quality_issues(data)
    
    # Find issues by column
    issues_by_col = {i.column: i for i in issues if i.issue_type == 'missing'}
    
    assert issues_by_col['critical_missing'].severity == 'critical'
    assert issues_by_col['high_missing'].severity == 'high'
    assert issues_by_col['medium_missing'].severity == 'medium'
    assert issues_by_col['low_missing'].severity == 'low'


def test_empty_dataframe(eda_module):
    """Test handling of empty DataFrame"""
    empty_data = pd.DataFrame()
    
    stats = eda_module.generate_summary_statistics(empty_data)
    assert len(stats) == 0
    
    issues = eda_module.detect_data_quality_issues(empty_data)
    assert len(issues) == 0


def test_single_column_dataframe(eda_module):
    """Test handling of single column DataFrame"""
    single_col_data = pd.DataFrame({'value': [1, 2, 3, 4, 5]})
    
    stats = eda_module.generate_summary_statistics(single_col_data)
    assert 'value' in stats
    
    corr_matrix = eda_module.analyze_correlations(single_col_data)
    assert corr_matrix.shape == (1, 1)


def test_constant_column_detection(eda_module):
    """Test detection of constant (zero variance) columns"""
    data = pd.DataFrame({
        'constant': [5] * 100,
        'variable': np.random.randn(100)
    })
    
    issues = eda_module.detect_data_quality_issues(data)
    
    # Find constant column issues
    constant_issues = [i for i in issues if i.column == 'constant' and 'constant value' in i.description]
    assert len(constant_issues) > 0


def test_high_cardinality_detection(eda_module):
    """Test detection of high cardinality categorical columns"""
    data = pd.DataFrame({
        'high_cardinality': [f'value_{i}' for i in range(95)],
        'normal': ['A'] * 50 + ['B'] * 45
    })
    
    issues = eda_module.detect_data_quality_issues(data)
    
    # Find high cardinality issues
    cardinality_issues = [i for i in issues if i.column == 'high_cardinality' and 'cardinality' in i.description]
    assert len(cardinality_issues) > 0
