"""
Data validation module for handling null values and data quality checks.
Provides utilities for analyzing and managing missing data in datasets.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataValidator:
    """Handles data validation with focus on null value analysis and data quality."""
    
    def __init__(self, df: pd.DataFrame):
        """Initialize validator with a DataFrame."""
        self.df = df
        self.null_report = {}
        logger.info(f"DataValidator initialized with {len(df)} rows and {len(df.columns)} columns")
    
    def analyze_nulls(self) -> Dict[str, Any]:
        """
        Comprehensive analysis of null values in the dataset.
        
        Returns:
            Dictionary containing null statistics for each column
        """
        logger.info("Starting null value analysis")
        
        total_rows = len(self.df)
        null_analysis = {}
        
        for column in self.df.columns:
            null_count = self.df[column].isnull().sum()
            null_percentage = (null_count / total_rows) * 100
            
            null_analysis[column] = {
                'null_count': int(null_count),
                'null_percentage': round(null_percentage, 2),
                'non_null_count': int(total_rows - null_count),
                'data_type': str(self.df[column].dtype),
                'unique_values': int(self.df[column].nunique()),
                'has_nulls': bool(null_count > 0)
            }
            
            if null_count > 0:
                logger.info(f"Column '{column}': {null_count} nulls ({null_percentage:.2f}%)")
        
        self.null_report = null_analysis
        return null_analysis
    
    def get_null_patterns(self) -> Dict[str, Any]:
        """
        Identify patterns in null values across rows.
        
        Returns:
            Dictionary with null pattern analysis
        """
        logger.info("Analyzing null patterns")
        
        null_mask = self.df.isnull()
        rows_with_nulls = null_mask.any(axis=1).sum()
        complete_rows = len(self.df) - rows_with_nulls
        
        null_counts_per_row = null_mask.sum(axis=1)
        
        patterns = {
            'total_rows': len(self.df),
            'rows_with_any_null': int(rows_with_nulls),
            'complete_rows': int(complete_rows),
            'completeness_rate': round((complete_rows / len(self.df)) * 100, 2),
            'max_nulls_in_row': int(null_counts_per_row.max()),
            'avg_nulls_per_row': round(null_counts_per_row.mean(), 2),
            'columns_always_together_null': self._find_correlated_nulls()
        }
        
        logger.info(f"Found {rows_with_nulls} rows with nulls out of {len(self.df)}")
        return patterns
    
    def _find_correlated_nulls(self, threshold: float = 0.8) -> List[Tuple[str, str, float]]:
        """
        Find columns that tend to have nulls together.
        
        Args:
            threshold: Correlation threshold (0-1)
            
        Returns:
            List of column pairs with high null correlation
        """
        null_mask = self.df.isnull().astype(int)
        corr_matrix = null_mask.corr()
        
        correlated_pairs = []
        for i in range(len(corr_matrix.columns)):
            for j in range(i + 1, len(corr_matrix.columns)):
                corr_value = corr_matrix.iloc[i, j]
                if abs(corr_value) >= threshold:
                    col1 = corr_matrix.columns[i]
                    col2 = corr_matrix.columns[j]
                    correlated_pairs.append((col1, col2, round(corr_value, 3)))
                    logger.info(f"High null correlation between '{col1}' and '{col2}': {corr_value:.3f}")
        
        return correlated_pairs
    
    def suggest_handling_strategy(self) -> Dict[str, str]:
        """
        Suggest strategies for handling null values based on data characteristics.
        
        Returns:
            Dictionary with column names and suggested strategies
        """
        logger.info("Generating null handling suggestions")
        
        if not self.null_report:
            self.analyze_nulls()
        
        strategies = {}
        
        for column, stats in self.null_report.items():
            if not stats['has_nulls']:
                strategies[column] = "No action needed - no nulls"
                continue
            
            null_pct = stats['null_percentage']
            dtype = stats['data_type']
            
            if null_pct > 70:
                strategies[column] = "Consider dropping - too many nulls (>70%)"
            elif null_pct > 40:
                strategies[column] = "High null rate - investigate data collection issue"
            elif 'float' in dtype or 'int' in dtype:
                if null_pct < 5:
                    strategies[column] = "Impute with median or mean"
                else:
                    strategies[column] = "Consider advanced imputation or indicator variable"
            elif 'object' in dtype:
                if null_pct < 10:
                    strategies[column] = "Impute with mode or 'Unknown'"
                else:
                    strategies[column] = "Create 'Missing' category or drop if not critical"
            elif 'datetime' in dtype:
                strategies[column] = "Forward/backward fill or interpolate if time series"
            else:
                strategies[column] = "Analyze context for appropriate strategy"
        
        return strategies
    
    def validate_completeness(self, required_columns: List[str]) -> Tuple[bool, List[str]]:
        """
        Check if required columns have no null values.
        
        Args:
            required_columns: List of column names that must be complete
            
        Returns:
            Tuple of (is_valid, list_of_incomplete_columns)
        """
        logger.info(f"Validating completeness for {len(required_columns)} required columns")
        
        incomplete_columns = []
        
        for col in required_columns:
            if col not in self.df.columns:
                logger.warning(f"Required column '{col}' not found in dataset")
                incomplete_columns.append(f"{col} (missing)")
            elif self.df[col].isnull().any():
                null_count = self.df[col].isnull().sum()
                incomplete_columns.append(f"{col} ({null_count} nulls)")
                logger.warning(f"Required column '{col}' has {null_count} null values")
        
        is_valid = len(incomplete_columns) == 0
        
        if is_valid:
            logger.info("All required columns are complete")
        else:
            logger.warning(f"Validation failed: {len(incomplete_columns)} incomplete columns")
        
        return is_valid, incomplete_columns
    
    def get_summary_report(self) -> str:
        """
        Generate a human-readable summary report of data quality.
        
        Returns:
            Formatted string report
        """
        if not self.null_report:
            self.analyze_nulls()
        
        patterns = self.get_null_patterns()
        strategies = self.suggest_handling_strategy()
        
        report = []
        report.append("=" * 60)
        report.append("DATA QUALITY REPORT - NULL VALUE ANALYSIS")
        report.append("=" * 60)
        report.append(f"\nDataset Shape: {len(self.df)} rows × {len(self.df.columns)} columns")
        report.append(f"Complete Rows: {patterns['complete_rows']} ({patterns['completeness_rate']}%)")
        report.append(f"Rows with Nulls: {patterns['rows_with_any_null']}")
        report.append(f"Average Nulls per Row: {patterns['avg_nulls_per_row']}")
        
        report.append("\n" + "-" * 40)
        report.append("COLUMNS WITH NULL VALUES:")
        report.append("-" * 40)
        
        null_columns = [(col, stats) for col, stats in self.null_report.items() if stats['has_nulls']]
        null_columns.sort(key=lambda x: x[1]['null_percentage'], reverse=True)
        
        for col, stats in null_columns:
            report.append(f"\n{col}:")
            report.append(f"  - Null Count: {stats['null_count']} ({stats['null_percentage']}%)")
            report.append(f"  - Data Type: {stats['data_type']}")
            report.append(f"  - Suggested Strategy: {strategies[col]}")
        
        if patterns['columns_always_together_null']:
            report.append("\n" + "-" * 40)
            report.append("CORRELATED NULL PATTERNS:")
            report.append("-" * 40)
            for col1, col2, corr in patterns['columns_always_together_null']:
                report.append(f"  {col1} ↔ {col2}: {corr:.3f} correlation")
        
        report.append("\n" + "=" * 60)
        
        return "\n".join(report)


def load_and_validate(filepath: str, required_columns: Optional[List[str]] = None) -> Tuple[pd.DataFrame, DataValidator]:
    """
    Convenience function to load CSV and validate data.
    
    Args:
        filepath: Path to CSV file
        required_columns: Optional list of required columns
        
    Returns:
        Tuple of (DataFrame, DataValidator instance)
    """
    logger.info(f"Loading data from {filepath}")
    
    try:
        df = pd.read_csv(filepath)
        validator = DataValidator(df)
        
        validator.analyze_nulls()
        
        if required_columns:
            is_valid, incomplete = validator.validate_completeness(required_columns)
            if not is_valid:
                logger.warning(f"Data validation issues found: {incomplete}")
        
        return df, validator
        
    except Exception as e:
        logger.error(f"Failed to load or validate data: {str(e)}")
        raise