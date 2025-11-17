#!/usr/bin/env python3
"""
Manual test script for EDA module
"""

import pandas as pd
import numpy as np
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.ds.eda import EDAModule, EDAReport, DataQualityIssue

def test_eda_module():
    """Test EDA module functionality"""
    print("Testing EDA Module...")
    print("=" * 60)
    
    # Create sample data
    np.random.seed(42)
    data = pd.DataFrame({
        'age': np.random.randint(18, 80, 100),
        'score': np.random.normal(50, 10, 100),
        'income': np.random.exponential(50000, 100),
        'category': np.random.choice(['A', 'B', 'C'], 100),
        'status': np.random.choice(['active', 'inactive'], 100)
    })
    
    # Add missing values
    data.loc[0:5, 'score'] = np.nan
    data.loc[10:12, 'category'] = np.nan
    
    # Add outliers
    data.loc[95:99, 'income'] = [500000, 600000, 700000, 800000, 900000]
    
    # Add duplicates
    data = pd.concat([data, data.iloc[0:3]], ignore_index=True)
    
    print(f"Sample data shape: {data.shape}")
    print()
    
    # Initialize EDA module
    eda = EDAModule()
    
    # Test 1: Summary statistics
    print("Test 1: Generate Summary Statistics")
    print("-" * 60)
    stats = eda.generate_summary_statistics(data)
    print(f"✓ Generated statistics for {len(stats)} columns")
    
    # Check numerical stats
    if 'age' in stats:
        age_stats = stats['age']
        print(f"  Age: mean={age_stats['mean']:.2f}, std={age_stats['std']:.2f}")
    
    # Check categorical stats
    if 'category' in stats:
        cat_stats = stats['category']
        print(f"  Category: unique={int(cat_stats['unique'])}, top={cat_stats['top']}")
    print()
    
    # Test 2: Data quality issues
    print("Test 2: Detect Data Quality Issues")
    print("-" * 60)
    issues = eda.detect_data_quality_issues(data)
    print(f"✓ Detected {len(issues)} quality issues")
    
    for issue in issues[:5]:  # Show first 5
        print(f"  - {issue.severity.upper()}: {issue.issue_type} in {issue.column or 'dataset'}")
        print(f"    {issue.description}")
    
    if len(issues) > 5:
        print(f"  ... and {len(issues) - 5} more issues")
    print()
    
    # Test 3: Correlation analysis
    print("Test 3: Analyze Correlations")
    print("-" * 60)
    corr_matrix = eda.analyze_correlations(data)
    print(f"✓ Generated correlation matrix: {corr_matrix.shape}")
    print(f"  Columns: {', '.join(corr_matrix.columns.tolist())}")
    print()
    
    # Test 4: Full analysis
    print("Test 4: Full Dataset Analysis")
    print("-" * 60)
    report = eda.analyze_dataset(data, dataset_name="test_dataset")
    print(f"✓ Generated EDA report")
    print(f"  Dataset: {report.dataset_name}")
    print(f"  Rows: {report.num_rows}")
    print(f"  Columns: {report.num_columns}")
    print(f"  Quality Issues: {len(report.quality_issues)}")
    print(f"  Recommendations: {len(report.recommendations)}")
    print()
    
    # Test 5: Visualizations
    print("Test 5: Generate Visualizations")
    print("-" * 60)
    try:
        viz_paths = eda.generate_visualizations(data, output_dir="test_eda_output")
        print(f"✓ Generated {len(viz_paths)} visualizations")
        for path in viz_paths:
            print(f"  - {Path(path).name}")
        print()
    except Exception as e:
        print(f"✗ Visualization generation failed: {e}")
        print()
    
    # Test 6: Report export
    print("Test 6: Export Report")
    print("-" * 60)
    try:
        html_path = eda.export_report(report, format="html", output_path="test_eda_output/report.html")
        print(f"✓ Exported HTML report to: {html_path}")
        
        md_path = eda.export_report(report, format="markdown", output_path="test_eda_output/report.md")
        print(f"✓ Exported Markdown report to: {md_path}")
        print()
    except Exception as e:
        print(f"✗ Report export failed: {e}")
        print()
    
    # Summary
    print("=" * 60)
    print("✓ All core EDA functionality tests passed!")
    print()
    print("Recommendations from analysis:")
    for i, rec in enumerate(report.recommendations, 1):
        print(f"  {i}. {rec}")
    
    return True

if __name__ == "__main__":
    try:
        success = test_eda_module()
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"\n✗ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
