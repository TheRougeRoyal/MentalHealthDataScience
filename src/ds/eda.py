"""
Exploratory Data Analysis (EDA) Module

This module provides automated exploratory data analysis capabilities including
statistical summaries, data quality issue detection, visualizations, and report generation.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime
from pathlib import Path
from pydantic import BaseModel, Field
import warnings
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings('ignore')
sns.set_style("whitegrid")


class DataQualityIssue(BaseModel):
    """Represents a data quality issue detected during EDA"""
    
    issue_type: str = Field(..., description="Type of issue: missing, outlier, duplicate, inconsistent")
    severity: str = Field(..., description="Severity level: low, medium, high, critical")
    column: Optional[str] = Field(None, description="Column name if issue is column-specific")
    description: str = Field(..., description="Human-readable description of the issue")
    affected_rows: int = Field(..., description="Number of rows affected by this issue")
    recommendation: str = Field(..., description="Recommended action to address the issue")


class EDAReport(BaseModel):
    """Comprehensive EDA report"""
    
    dataset_name: str
    num_rows: int
    num_columns: int
    summary_statistics: Dict[str, Dict[str, float]]
    missing_values: Dict[str, int]
    data_types: Dict[str, str]
    outliers: Dict[str, List[int]]
    correlations: Optional[List[List[float]]] = None
    quality_issues: List[DataQualityIssue]
    visualizations: List[str] = Field(default_factory=list)
    recommendations: List[str]
    generated_at: datetime = Field(default_factory=datetime.now)
    
    class Config:
        arbitrary_types_allowed = True


class EDAModule:
    """Automated exploratory data analysis module"""
    
    def __init__(self):
        """Initialize EDA module"""
        self.report: Optional[EDAReport] = None
    
    def generate_summary_statistics(self, data: pd.DataFrame) -> Dict[str, Dict[str, float]]:
        """
        Generate statistical summaries for numerical and categorical features
        
        Args:
            data: Input DataFrame
            
        Returns:
            Dictionary mapping column names to their statistics
        """
        summary_stats = {}
        
        # Numerical features
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            col_data = data[col].dropna()
            if len(col_data) > 0:
                summary_stats[col] = {
                    'count': float(len(col_data)),
                    'mean': float(col_data.mean()),
                    'std': float(col_data.std()),
                    'min': float(col_data.min()),
                    'q25': float(col_data.quantile(0.25)),
                    'median': float(col_data.median()),
                    'q75': float(col_data.quantile(0.75)),
                    'max': float(col_data.max()),
                    'skewness': float(col_data.skew()),
                    'kurtosis': float(col_data.kurtosis())
                }
        
        # Categorical features
        categorical_cols = data.select_dtypes(include=['object', 'category']).columns
        for col in categorical_cols:
            col_data = data[col].dropna()
            if len(col_data) > 0:
                value_counts = col_data.value_counts()
                summary_stats[col] = {
                    'count': float(len(col_data)),
                    'unique': float(col_data.nunique()),
                    'top': str(value_counts.index[0]) if len(value_counts) > 0 else 'N/A',
                    'freq': float(value_counts.iloc[0]) if len(value_counts) > 0 else 0.0,
                    'missing': float(data[col].isna().sum())
                }
        
        return summary_stats
    
    def detect_data_quality_issues(self, data: pd.DataFrame) -> List[DataQualityIssue]:
        """
        Detect data quality issues including missing values, outliers, and duplicates
        
        Args:
            data: Input DataFrame
            
        Returns:
            List of detected data quality issues
        """
        issues = []
        total_rows = len(data)
        
        # Check for missing values
        missing_counts = data.isna().sum()
        for col, missing_count in missing_counts.items():
            if missing_count > 0:
                missing_pct = (missing_count / total_rows) * 100
                
                if missing_pct >= 50:
                    severity = "critical"
                    recommendation = f"Consider dropping column '{col}' or using advanced imputation"
                elif missing_pct >= 20:
                    severity = "high"
                    recommendation = f"Apply imputation strategy for column '{col}'"
                elif missing_pct >= 5:
                    severity = "medium"
                    recommendation = f"Review missing value pattern in column '{col}'"
                else:
                    severity = "low"
                    recommendation = f"Minor missing values in column '{col}', simple imputation sufficient"
                
                issues.append(DataQualityIssue(
                    issue_type="missing",
                    severity=severity,
                    column=col,
                    description=f"Column '{col}' has {missing_count} missing values ({missing_pct:.1f}%)",
                    affected_rows=int(missing_count),
                    recommendation=recommendation
                ))
        
        # Check for outliers in numerical columns using IQR method
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            col_data = data[col].dropna()
            if len(col_data) > 0:
                Q1 = col_data.quantile(0.25)
                Q3 = col_data.quantile(0.75)
                IQR = Q3 - Q1
                
                lower_bound = Q1 - 3 * IQR
                upper_bound = Q3 + 3 * IQR
                
                outliers = ((col_data < lower_bound) | (col_data > upper_bound)).sum()
                if outliers > 0:
                    outlier_pct = (outliers / len(col_data)) * 100
                    
                    if outlier_pct >= 10:
                        severity = "high"
                        recommendation = f"Investigate outliers in '{col}', consider robust scaling or transformation"
                    elif outlier_pct >= 5:
                        severity = "medium"
                        recommendation = f"Review outliers in '{col}', may need capping or removal"
                    else:
                        severity = "low"
                        recommendation = f"Few outliers in '{col}', monitor but may be valid extreme values"
                    
                    issues.append(DataQualityIssue(
                        issue_type="outlier",
                        severity=severity,
                        column=col,
                        description=f"Column '{col}' has {outliers} outliers ({outlier_pct:.1f}%)",
                        affected_rows=int(outliers),
                        recommendation=recommendation
                    ))
        
        # Check for duplicate rows
        duplicate_count = data.duplicated().sum()
        if duplicate_count > 0:
            duplicate_pct = (duplicate_count / total_rows) * 100
            
            if duplicate_pct >= 10:
                severity = "high"
                recommendation = "Remove duplicate rows before analysis"
            elif duplicate_pct >= 5:
                severity = "medium"
                recommendation = "Review and remove duplicate rows"
            else:
                severity = "low"
                recommendation = "Few duplicates detected, verify if intentional"
            
            issues.append(DataQualityIssue(
                issue_type="duplicate",
                severity=severity,
                column=None,
                description=f"Dataset has {duplicate_count} duplicate rows ({duplicate_pct:.1f}%)",
                affected_rows=int(duplicate_count),
                recommendation=recommendation
            ))
        
        # Check for constant columns (zero variance)
        for col in numeric_cols:
            if data[col].nunique() == 1:
                issues.append(DataQualityIssue(
                    issue_type="inconsistent",
                    severity="medium",
                    column=col,
                    description=f"Column '{col}' has constant value (zero variance)",
                    affected_rows=total_rows,
                    recommendation=f"Consider removing column '{col}' as it provides no information"
                ))
        
        # Check for high cardinality in categorical columns
        categorical_cols = data.select_dtypes(include=['object', 'category']).columns
        for col in categorical_cols:
            unique_count = data[col].nunique()
            unique_pct = (unique_count / total_rows) * 100
            
            if unique_pct >= 90:
                issues.append(DataQualityIssue(
                    issue_type="inconsistent",
                    severity="medium",
                    column=col,
                    description=f"Column '{col}' has very high cardinality ({unique_count} unique values)",
                    affected_rows=total_rows,
                    recommendation=f"Consider if '{col}' should be treated as identifier rather than categorical feature"
                ))
        
        return issues
    
    def analyze_correlations(
        self,
        data: pd.DataFrame,
        method: str = "pearson"
    ) -> pd.DataFrame:
        """
        Compute correlation matrix for numerical features
        
        Args:
            data: Input DataFrame
            method: Correlation method ('pearson', 'spearman', 'kendall')
            
        Returns:
            Correlation matrix as DataFrame
        """
        numeric_data = data.select_dtypes(include=[np.number])
        
        if numeric_data.empty:
            return pd.DataFrame()
        
        correlation_matrix = numeric_data.corr(method=method)
        return correlation_matrix
    
    def _generate_recommendations(
        self,
        data: pd.DataFrame,
        quality_issues: List[DataQualityIssue]
    ) -> List[str]:
        """
        Generate actionable recommendations based on data analysis
        
        Args:
            data: Input DataFrame
            quality_issues: List of detected quality issues
            
        Returns:
            List of recommendations
        """
        recommendations = []
        
        # Count issues by severity
        critical_issues = sum(1 for issue in quality_issues if issue.severity == "critical")
        high_issues = sum(1 for issue in quality_issues if issue.severity == "high")
        
        if critical_issues > 0:
            recommendations.append(
                f"Address {critical_issues} critical data quality issue(s) before proceeding with modeling"
            )
        
        if high_issues > 0:
            recommendations.append(
                f"Resolve {high_issues} high-severity issue(s) to improve data quality"
            )
        
        # Check for imbalanced dataset
        if len(data) < 100:
            recommendations.append(
                "Dataset is small (< 100 rows). Consider collecting more data for robust analysis"
            )
        
        # Check for high dimensionality
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 50:
            recommendations.append(
                f"High dimensionality detected ({len(numeric_cols)} numerical features). "
                "Consider feature selection or dimensionality reduction"
            )
        
        # Check correlation matrix for multicollinearity
        if len(numeric_cols) > 1:
            corr_matrix = self.analyze_correlations(data)
            high_corr_pairs = []
            
            for i in range(len(corr_matrix.columns)):
                for j in range(i + 1, len(corr_matrix.columns)):
                    if abs(corr_matrix.iloc[i, j]) > 0.9:
                        high_corr_pairs.append(
                            (corr_matrix.columns[i], corr_matrix.columns[j], corr_matrix.iloc[i, j])
                        )
            
            if high_corr_pairs:
                recommendations.append(
                    f"Found {len(high_corr_pairs)} highly correlated feature pair(s). "
                    "Consider removing redundant features to reduce multicollinearity"
                )
        
        if not recommendations:
            recommendations.append("Data quality looks good. Proceed with feature engineering and modeling")
        
        return recommendations
    
    def analyze_dataset(
        self,
        data: pd.DataFrame,
        dataset_name: str = "dataset",
        target_column: Optional[str] = None
    ) -> EDAReport:
        """
        Generate comprehensive EDA report orchestrating full workflow
        
        Args:
            data: Input DataFrame
            dataset_name: Name of the dataset
            target_column: Optional target column for supervised learning
            
        Returns:
            Complete EDA report
        """
        # Generate summary statistics
        summary_stats = self.generate_summary_statistics(data)
        
        # Detect data quality issues
        quality_issues = self.detect_data_quality_issues(data)
        
        # Compute correlations
        corr_matrix = self.analyze_correlations(data)
        
        # Extract missing values
        missing_values = {col: int(data[col].isna().sum()) for col in data.columns}
        
        # Extract data types
        data_types = {col: str(dtype) for col, dtype in data.dtypes.items()}
        
        # Detect outliers (store indices)
        outliers = {}
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            col_data = data[col].dropna()
            if len(col_data) > 0:
                Q1 = col_data.quantile(0.25)
                Q3 = col_data.quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 3 * IQR
                upper_bound = Q3 + 3 * IQR
                
                outlier_indices = data[
                    (data[col] < lower_bound) | (data[col] > upper_bound)
                ].index.tolist()
                
                if outlier_indices:
                    outliers[col] = outlier_indices[:100]  # Limit to first 100
        
        # Generate recommendations
        recommendations = self._generate_recommendations(data, quality_issues)
        
        # Create report
        report = EDAReport(
            dataset_name=dataset_name,
            num_rows=len(data),
            num_columns=len(data.columns),
            summary_statistics=summary_stats,
            missing_values=missing_values,
            data_types=data_types,
            outliers=outliers,
            correlations=corr_matrix.values.tolist() if not corr_matrix.empty else None,
            quality_issues=quality_issues,
            recommendations=recommendations
        )
        
        self.report = report
        return report
    
    def generate_visualizations(
        self,
        data: pd.DataFrame,
        output_dir: str = "eda_visualizations"
    ) -> List[str]:
        """
        Generate and save visualizations for the dataset
        
        Args:
            data: Input DataFrame
            output_dir: Directory to save visualizations
            
        Returns:
            List of paths to generated visualization files
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        visualization_paths = []
        
        # Get numeric and categorical columns
        numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = data.select_dtypes(include=['object', 'category']).columns.tolist()
        
        # 1. Distribution plots for numerical features
        if numeric_cols:
            n_numeric = len(numeric_cols)
            n_cols = min(3, n_numeric)
            n_rows = (n_numeric + n_cols - 1) // n_cols
            
            fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows))
            if n_numeric == 1:
                axes = [axes]
            else:
                axes = axes.flatten() if n_numeric > 1 else [axes]
            
            for idx, col in enumerate(numeric_cols):
                ax = axes[idx]
                col_data = data[col].dropna()
                
                if len(col_data) > 0:
                    # Histogram with KDE
                    ax.hist(col_data, bins=30, alpha=0.7, color='skyblue', edgecolor='black', density=True)
                    
                    # Add KDE if enough data points
                    if len(col_data) > 10:
                        col_data.plot(kind='kde', ax=ax, color='red', linewidth=2)
                    
                    ax.set_title(f'Distribution of {col}', fontsize=10, fontweight='bold')
                    ax.set_xlabel(col)
                    ax.set_ylabel('Density')
                    ax.grid(True, alpha=0.3)
            
            # Hide unused subplots
            for idx in range(n_numeric, len(axes)):
                axes[idx].set_visible(False)
            
            plt.tight_layout()
            dist_path = output_path / "distributions.png"
            plt.savefig(dist_path, dpi=300, bbox_inches='tight')
            plt.close()
            visualization_paths.append(str(dist_path))
        
        # 2. Box plots for outlier visualization
        if numeric_cols:
            n_numeric = len(numeric_cols)
            n_cols = min(3, n_numeric)
            n_rows = (n_numeric + n_cols - 1) // n_cols
            
            fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows))
            if n_numeric == 1:
                axes = [axes]
            else:
                axes = axes.flatten() if n_numeric > 1 else [axes]
            
            for idx, col in enumerate(numeric_cols):
                ax = axes[idx]
                col_data = data[col].dropna()
                
                if len(col_data) > 0:
                    ax.boxplot(col_data, vert=True, patch_artist=True,
                              boxprops=dict(facecolor='lightblue', alpha=0.7),
                              medianprops=dict(color='red', linewidth=2))
                    ax.set_title(f'Box Plot of {col}', fontsize=10, fontweight='bold')
                    ax.set_ylabel(col)
                    ax.grid(True, alpha=0.3, axis='y')
            
            # Hide unused subplots
            for idx in range(n_numeric, len(axes)):
                axes[idx].set_visible(False)
            
            plt.tight_layout()
            box_path = output_path / "boxplots.png"
            plt.savefig(box_path, dpi=300, bbox_inches='tight')
            plt.close()
            visualization_paths.append(str(box_path))
        
        # 3. Correlation heatmap
        if len(numeric_cols) > 1:
            corr_matrix = self.analyze_correlations(data)
            
            if not corr_matrix.empty:
                fig, ax = plt.subplots(figsize=(max(10, len(numeric_cols) * 0.8), 
                                                max(8, len(numeric_cols) * 0.7)))
                
                sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm',
                           center=0, square=True, linewidths=1, cbar_kws={"shrink": 0.8},
                           ax=ax, vmin=-1, vmax=1)
                
                ax.set_title('Feature Correlation Heatmap', fontsize=14, fontweight='bold', pad=20)
                plt.tight_layout()
                
                corr_path = output_path / "correlation_heatmap.png"
                plt.savefig(corr_path, dpi=300, bbox_inches='tight')
                plt.close()
                visualization_paths.append(str(corr_path))
        
        # 4. Missing value pattern visualization
        missing_counts = data.isna().sum()
        if missing_counts.sum() > 0:
            missing_data = missing_counts[missing_counts > 0].sort_values(ascending=False)
            
            fig, ax = plt.subplots(figsize=(10, max(6, len(missing_data) * 0.4)))
            
            bars = ax.barh(range(len(missing_data)), missing_data.values, color='coral')
            ax.set_yticks(range(len(missing_data)))
            ax.set_yticklabels(missing_data.index)
            ax.set_xlabel('Number of Missing Values', fontsize=12)
            ax.set_title('Missing Values by Feature', fontsize=14, fontweight='bold')
            ax.grid(True, alpha=0.3, axis='x')
            
            # Add value labels on bars
            for i, (bar, value) in enumerate(zip(bars, missing_data.values)):
                pct = (value / len(data)) * 100
                ax.text(value, i, f' {value} ({pct:.1f}%)', 
                       va='center', fontsize=9)
            
            plt.tight_layout()
            missing_path = output_path / "missing_values.png"
            plt.savefig(missing_path, dpi=300, bbox_inches='tight')
            plt.close()
            visualization_paths.append(str(missing_path))
        
        # 5. Categorical feature distribution (top categories)
        if categorical_cols:
            for col in categorical_cols[:5]:  # Limit to first 5 categorical columns
                value_counts = data[col].value_counts().head(10)
                
                if len(value_counts) > 0:
                    fig, ax = plt.subplots(figsize=(10, 6))
                    
                    bars = ax.barh(range(len(value_counts)), value_counts.values, color='steelblue')
                    ax.set_yticks(range(len(value_counts)))
                    ax.set_yticklabels(value_counts.index)
                    ax.set_xlabel('Count', fontsize=12)
                    ax.set_title(f'Top Categories in {col}', fontsize=14, fontweight='bold')
                    ax.grid(True, alpha=0.3, axis='x')
                    
                    # Add value labels
                    for i, (bar, value) in enumerate(zip(bars, value_counts.values)):
                        ax.text(value, i, f' {value}', va='center', fontsize=9)
                    
                    plt.tight_layout()
                    cat_path = output_path / f"categorical_{col}.png"
                    plt.savefig(cat_path, dpi=300, bbox_inches='tight')
                    plt.close()
                    visualization_paths.append(str(cat_path))
        
        return visualization_paths
    
    def export_report(
        self,
        report: EDAReport,
        format: str = "html",
        output_path: str = "eda_report.html"
    ) -> str:
        """
        Export EDA report in specified format
        
        Args:
            report: EDA report to export
            format: Export format ('html' or 'markdown')
            output_path: Path to save the report
            
        Returns:
            Path to the exported report
        """
        if format.lower() == "html":
            return self._export_html(report, output_path)
        elif format.lower() == "markdown":
            return self._export_markdown(report, output_path)
        else:
            raise ValueError(f"Unsupported format: {format}. Use 'html' or 'markdown'")
    
    def _export_html(self, report: EDAReport, output_path: str) -> str:
        """Export report as HTML"""
        html_content = self._generate_html_report(report)
        
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        return str(output_file)
    
    def _export_markdown(self, report: EDAReport, output_path: str) -> str:
        """Export report as Markdown"""
        md_content = self._generate_markdown_report(report)
        
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(md_content)
        
        return str(output_file)
    
    def _generate_html_report(self, report: EDAReport) -> str:
        """Generate HTML report content"""
        
        # Build quality issues HTML
        issues_html = ""
        if report.quality_issues:
            issues_by_severity = {
                'critical': [],
                'high': [],
                'medium': [],
                'low': []
            }
            
            for issue in report.quality_issues:
                issues_by_severity[issue.severity].append(issue)
            
            for severity in ['critical', 'high', 'medium', 'low']:
                issues = issues_by_severity[severity]
                if issues:
                    severity_color = {
                        'critical': '#dc3545',
                        'high': '#fd7e14',
                        'medium': '#ffc107',
                        'low': '#17a2b8'
                    }[severity]
                    
                    issues_html += f'<h4 style="color: {severity_color}; text-transform: capitalize;">{severity} Severity Issues</h4>'
                    issues_html += '<ul>'
                    
                    for issue in issues:
                        col_info = f" (Column: {issue.column})" if issue.column else ""
                        issues_html += f'''
                        <li>
                            <strong>{issue.issue_type.capitalize()}{col_info}</strong>: {issue.description}
                            <br><em>Recommendation: {issue.recommendation}</em>
                        </li>
                        '''
                    
                    issues_html += '</ul>'
        else:
            issues_html = '<p style="color: green;">No data quality issues detected!</p>'
        
        # Build summary statistics HTML
        stats_html = '<table style="width: 100%; border-collapse: collapse; margin: 20px 0;">'
        stats_html += '''
        <thead>
            <tr style="background-color: #f8f9fa; border-bottom: 2px solid #dee2e6;">
                <th style="padding: 12px; text-align: left; border: 1px solid #dee2e6;">Feature</th>
                <th style="padding: 12px; text-align: left; border: 1px solid #dee2e6;">Type</th>
                <th style="padding: 12px; text-align: left; border: 1px solid #dee2e6;">Statistics</th>
            </tr>
        </thead>
        <tbody>
        '''
        
        for col, stats in report.summary_statistics.items():
            dtype = report.data_types.get(col, 'unknown')
            
            if 'mean' in stats:  # Numerical
                stats_str = f"Mean: {stats['mean']:.2f}, Std: {stats['std']:.2f}, Min: {stats['min']:.2f}, Max: {stats['max']:.2f}"
            else:  # Categorical
                stats_str = f"Unique: {int(stats.get('unique', 0))}, Top: {stats.get('top', 'N/A')}"
            
            stats_html += f'''
            <tr style="border-bottom: 1px solid #dee2e6;">
                <td style="padding: 10px; border: 1px solid #dee2e6;"><strong>{col}</strong></td>
                <td style="padding: 10px; border: 1px solid #dee2e6;">{dtype}</td>
                <td style="padding: 10px; border: 1px solid #dee2e6;">{stats_str}</td>
            </tr>
            '''
        
        stats_html += '</tbody></table>'
        
        # Build visualizations HTML
        viz_html = ""
        if report.visualizations:
            viz_html = '<div style="margin: 20px 0;">'
            for viz_path in report.visualizations:
                viz_name = Path(viz_path).stem.replace('_', ' ').title()
                viz_html += f'''
                <div style="margin: 20px 0;">
                    <h4>{viz_name}</h4>
                    <img src="{viz_path}" alt="{viz_name}" style="max-width: 100%; height: auto; border: 1px solid #dee2e6; border-radius: 4px;">
                </div>
                '''
            viz_html += '</div>'
        
        # Build recommendations HTML
        rec_html = '<ul>'
        for rec in report.recommendations:
            rec_html += f'<li>{rec}</li>'
        rec_html += '</ul>'
        
        # Complete HTML template
        html = f'''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>EDA Report - {report.dataset_name}</title>
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif;
            line-height: 1.6;
            color: #333;
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
            background-color: #f8f9fa;
        }}
        .container {{
            background-color: white;
            padding: 30px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        h1 {{
            color: #2c3e50;
            border-bottom: 3px solid #3498db;
            padding-bottom: 10px;
        }}
        h2 {{
            color: #34495e;
            margin-top: 30px;
            border-bottom: 2px solid #ecf0f1;
            padding-bottom: 8px;
        }}
        h3 {{
            color: #7f8c8d;
        }}
        .metadata {{
            background-color: #ecf0f1;
            padding: 15px;
            border-radius: 4px;
            margin: 20px 0;
        }}
        .metadata p {{
            margin: 5px 0;
        }}
        ul {{
            line-height: 1.8;
        }}
        .footer {{
            margin-top: 40px;
            padding-top: 20px;
            border-top: 1px solid #dee2e6;
            text-align: center;
            color: #6c757d;
            font-size: 0.9em;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>Exploratory Data Analysis Report</h1>
        
        <div class="metadata">
            <p><strong>Dataset:</strong> {report.dataset_name}</p>
            <p><strong>Rows:</strong> {report.num_rows:,}</p>
            <p><strong>Columns:</strong> {report.num_columns}</p>
            <p><strong>Generated:</strong> {report.generated_at.strftime('%Y-%m-%d %H:%M:%S')}</p>
        </div>
        
        <h2>Data Quality Issues</h2>
        {issues_html}
        
        <h2>Summary Statistics</h2>
        {stats_html}
        
        <h2>Visualizations</h2>
        {viz_html if viz_html else '<p>No visualizations generated.</p>'}
        
        <h2>Recommendations</h2>
        {rec_html}
        
        <div class="footer">
            <p>Generated by Mental Health Risk Assessment System - EDA Module</p>
        </div>
    </div>
</body>
</html>
        '''
        
        return html
    
    def _generate_markdown_report(self, report: EDAReport) -> str:
        """Generate Markdown report content"""
        
        md = f'''# Exploratory Data Analysis Report

## Dataset Information

- **Dataset:** {report.dataset_name}
- **Rows:** {report.num_rows:,}
- **Columns:** {report.num_columns}
- **Generated:** {report.generated_at.strftime('%Y-%m-%d %H:%M:%S')}

## Data Quality Issues

'''
        
        if report.quality_issues:
            issues_by_severity = {
                'critical': [],
                'high': [],
                'medium': [],
                'low': []
            }
            
            for issue in report.quality_issues:
                issues_by_severity[issue.severity].append(issue)
            
            for severity in ['critical', 'high', 'medium', 'low']:
                issues = issues_by_severity[severity]
                if issues:
                    md += f'\n### {severity.capitalize()} Severity Issues\n\n'
                    
                    for issue in issues:
                        col_info = f" (Column: {issue.column})" if issue.column else ""
                        md += f'- **{issue.issue_type.capitalize()}{col_info}**: {issue.description}\n'
                        md += f'  - *Recommendation: {issue.recommendation}*\n'
        else:
            md += 'No data quality issues detected!\n'
        
        md += '\n## Summary Statistics\n\n'
        md += '| Feature | Type | Statistics |\n'
        md += '|---------|------|------------|\n'
        
        for col, stats in report.summary_statistics.items():
            dtype = report.data_types.get(col, 'unknown')
            
            if 'mean' in stats:  # Numerical
                stats_str = f"Mean: {stats['mean']:.2f}, Std: {stats['std']:.2f}"
            else:  # Categorical
                stats_str = f"Unique: {int(stats.get('unique', 0))}"
            
            md += f'| {col} | {dtype} | {stats_str} |\n'
        
        md += '\n## Visualizations\n\n'
        if report.visualizations:
            for viz_path in report.visualizations:
                viz_name = Path(viz_path).stem.replace('_', ' ').title()
                md += f'### {viz_name}\n\n'
                md += f'![{viz_name}]({viz_path})\n\n'
        else:
            md += 'No visualizations generated.\n'
        
        md += '\n## Recommendations\n\n'
        for rec in report.recommendations:
            md += f'- {rec}\n'
        
        md += '\n---\n\n'
        md += '*Generated by Mental Health Risk Assessment System - EDA Module*\n'
        
        return md
