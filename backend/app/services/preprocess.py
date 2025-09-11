"""
Data preprocessing service for Interactive Spectral Clustering Platform.

Provides data cleaning, scaling, feature selection, and quality assessment
for machine learning pipelines with comprehensive statistics and validation.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional, Tuple, Union
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.feature_selection import VarianceThreshold
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.decomposition import PCA
from scipy import stats
import warnings
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)

@dataclass
class PreprocessingConfig:
    """Configuration for preprocessing pipeline."""
    
    # Scaling options
    scaler_type: str = "standard"  # "standard", "minmax", "robust", "none"
    
    # Missing value handling
    missing_strategy: str = "mean"  # "mean", "median", "mode", "drop", "knn"
    missing_threshold: float = 0.5  # Drop columns with >50% missing values
    
    # Feature selection
    variance_threshold: float = 0.01  # Remove low-variance features
    
    # Outlier detection
    outlier_method: str = "iqr"  # "iqr", "zscore", "isolation", "none"
    outlier_threshold: float = 3.0  # Standard deviations or IQR multiplier
    
    # Data validation
    min_samples: int = 10  # Minimum number of samples required
    max_features: int = 1000  # Maximum number of features allowed


@dataclass
class DatasetStats:
    """Comprehensive dataset statistics."""
    
    # Basic info
    shape: Tuple[int, int]
    memory_usage: float  # MB
    dtypes: Dict[str, str]
    
    # Missing values
    missing_counts: Dict[str, int]
    missing_percentages: Dict[str, float]
    total_missing: int
    
    # Numerical statistics
    numerical_stats: Dict[str, Dict[str, float]]  # column -> {mean, std, min, max, etc.}
    correlations: Optional[Dict[str, float]]  # High correlations (>0.9)
    
    # Categorical statistics
    categorical_stats: Dict[str, Dict[str, Any]]  # column -> {unique_count, top_values}
    
    # Data quality
    duplicate_rows: int
    constant_columns: List[str]
    high_cardinality_columns: List[str]  # >50% unique values
    skewed_columns: List[str]  # |skewness| > 2
    
    # Outliers
    outlier_counts: Dict[str, int]
    
    # Recommendations
    preprocessing_recommendations: List[str]


class DataPreprocessor:
    """Comprehensive data preprocessing and analysis service."""
    
    def __init__(self):
        """Initialize the preprocessor."""
        self.scalers = {
            "standard": StandardScaler,
            "minmax": MinMaxScaler,
            "robust": RobustScaler
        }
        
        self.imputers = {
            "mean": lambda: SimpleImputer(strategy="mean"),
            "median": lambda: SimpleImputer(strategy="median"),
            "mode": lambda: SimpleImputer(strategy="most_frequent"),
            "knn": lambda: KNNImputer(n_neighbors=5)
        }
    
    def analyze_dataset(self, data: np.ndarray, column_names: Optional[List[str]] = None) -> DatasetStats:
        """
        Comprehensive dataset analysis and statistics generation.
        
        Args:
            data: Input dataset as numpy array
            column_names: Optional column names
            
        Returns:
            DatasetStats object with comprehensive analysis
        """
        try:
            # Convert to DataFrame for easier analysis
            if column_names is None:
                column_names = [f"feature_{i}" for i in range(data.shape[1])]
            
            df = pd.DataFrame(data, columns=column_names)
            
            # Basic info
            shape = df.shape
            memory_usage = df.memory_usage(deep=True).sum() / 1024 / 1024  # MB
            dtypes = {col: str(dtype) for col, dtype in df.dtypes.items()}
            
            # Missing values analysis
            missing_counts = df.isnull().sum().to_dict()
            missing_percentages = (df.isnull().sum() / len(df) * 100).to_dict()
            total_missing = df.isnull().sum().sum()
            
            # Numerical statistics
            numerical_cols = df.select_dtypes(include=[np.number]).columns
            numerical_stats = {}
            
            for col in numerical_cols:
                if col in df.columns:
                    series = df[col].dropna()
                    if len(series) > 0:
                        numerical_stats[col] = {
                            "count": len(series),
                            "mean": float(series.mean()),
                            "std": float(series.std()),
                            "min": float(series.min()),
                            "25%": float(series.quantile(0.25)),
                            "50%": float(series.median()),
                            "75%": float(series.quantile(0.75)),
                            "max": float(series.max()),
                            "skewness": float(stats.skew(series)),
                            "kurtosis": float(stats.kurtosis(series))
                        }
            
            # Correlation analysis
            correlations = None
            if len(numerical_cols) > 1:
                corr_matrix = df[numerical_cols].corr()
                # Find high correlations (>0.9, excluding diagonal)
                high_corr = {}
                for i in range(len(corr_matrix.columns)):
                    for j in range(i+1, len(corr_matrix.columns)):
                        corr_val = corr_matrix.iloc[i, j]
                        if abs(corr_val) > 0.9:
                            pair = f"{corr_matrix.columns[i]}-{corr_matrix.columns[j]}"
                            high_corr[pair] = float(corr_val)
                correlations = high_corr if high_corr else None
            
            # Categorical statistics
            categorical_cols = df.select_dtypes(include=["object", "category"]).columns
            categorical_stats = {}
            
            for col in categorical_cols:
                if col in df.columns:
                    series = df[col].dropna()
                    if len(series) > 0:
                        value_counts = series.value_counts()
                        categorical_stats[col] = {
                            "unique_count": len(value_counts),
                            "top_value": str(value_counts.index[0]) if len(value_counts) > 0 else None,
                            "top_frequency": int(value_counts.iloc[0]) if len(value_counts) > 0 else 0,
                            "top_percentage": float(value_counts.iloc[0] / len(series) * 100) if len(value_counts) > 0 else 0
                        }
            
            # Data quality checks
            duplicate_rows = df.duplicated().sum()
            
            # Constant columns
            constant_columns = []
            for col in df.columns:
                if df[col].nunique() <= 1:
                    constant_columns.append(col)
            
            # High cardinality columns
            high_cardinality_columns = []
            for col in df.columns:
                unique_ratio = df[col].nunique() / len(df)
                if unique_ratio > 0.5 and df[col].nunique() > 10:
                    high_cardinality_columns.append(col)
            
            # Skewed columns
            skewed_columns = []
            for col in numerical_cols:
                if col in numerical_stats:
                    skewness = abs(numerical_stats[col]["skewness"])
                    if skewness > 2:
                        skewed_columns.append(col)
            
            # Outlier detection
            outlier_counts = {}
            for col in numerical_cols:
                outliers = self._detect_outliers_iqr(df[col].dropna())
                outlier_counts[col] = len(outliers)
            
            # Generate recommendations
            recommendations = self._generate_recommendations(
                df, missing_percentages, constant_columns, 
                high_cardinality_columns, skewed_columns, correlations
            )
            
            return DatasetStats(
                shape=shape,
                memory_usage=memory_usage,
                dtypes=dtypes,
                missing_counts=missing_counts,
                missing_percentages=missing_percentages,
                total_missing=total_missing,
                numerical_stats=numerical_stats,
                correlations=correlations,
                categorical_stats=categorical_stats,
                duplicate_rows=duplicate_rows,
                constant_columns=constant_columns,
                high_cardinality_columns=high_cardinality_columns,
                skewed_columns=skewed_columns,
                outlier_counts=outlier_counts,
                preprocessing_recommendations=recommendations
            )
            
        except Exception as e:
            logger.error(f"Error analyzing dataset: {str(e)}")
            raise ValueError(f"Failed to analyze dataset: {str(e)}")
    
    def preprocess_data(
        self, 
        data: np.ndarray, 
        config: PreprocessingConfig,
        column_names: Optional[List[str]] = None
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Apply preprocessing pipeline to the data.
        
        Args:
            data: Input dataset
            config: Preprocessing configuration
            column_names: Optional column names
            
        Returns:
            Tuple of (preprocessed_data, preprocessing_info)
        """
        try:
            if column_names is None:
                column_names = [f"feature_{i}" for i in range(data.shape[1])]
            
            df = pd.DataFrame(data, columns=column_names)
            preprocessing_info = {
                "original_shape": df.shape,
                "steps_applied": [],
                "removed_columns": [],
                "outliers_removed": 0
            }
            
            # 1. Remove columns with too many missing values
            missing_threshold = config.missing_threshold
            high_missing_cols = []
            for col in df.columns:
                missing_pct = df[col].isnull().sum() / len(df)
                if missing_pct > missing_threshold:
                    high_missing_cols.append(col)
            
            if high_missing_cols:
                df = df.drop(columns=high_missing_cols)
                preprocessing_info["removed_columns"].extend(high_missing_cols)
                preprocessing_info["steps_applied"].append(
                    f"Removed {len(high_missing_cols)} columns with >{missing_threshold*100}% missing values"
                )
            
            # 2. Handle missing values
            if config.missing_strategy != "drop":
                numerical_cols = df.select_dtypes(include=[np.number]).columns
                if len(numerical_cols) > 0 and df[numerical_cols].isnull().any().any():
                    imputer = self.imputers[config.missing_strategy]()
                    df[numerical_cols] = imputer.fit_transform(df[numerical_cols])
                    preprocessing_info["steps_applied"].append(
                        f"Imputed missing values using {config.missing_strategy} strategy"
                    )
            elif config.missing_strategy == "drop":
                original_len = len(df)
                df = df.dropna()
                dropped_rows = original_len - len(df)
                if dropped_rows > 0:
                    preprocessing_info["steps_applied"].append(
                        f"Dropped {dropped_rows} rows with missing values"
                    )
            
            # 3. Remove constant columns
            constant_cols = []
            for col in df.columns:
                if df[col].nunique() <= 1:
                    constant_cols.append(col)
            
            if constant_cols:
                df = df.drop(columns=constant_cols)
                preprocessing_info["removed_columns"].extend(constant_cols)
                preprocessing_info["steps_applied"].append(
                    f"Removed {len(constant_cols)} constant columns"
                )
            
            # 4. Variance threshold feature selection
            if config.variance_threshold > 0:
                numerical_cols = df.select_dtypes(include=[np.number]).columns
                if len(numerical_cols) > 1:
                    selector = VarianceThreshold(threshold=config.variance_threshold)
                    selected_data = selector.fit_transform(df[numerical_cols])
                    selected_cols = numerical_cols[selector.get_support()]
                    removed_cols = numerical_cols[~selector.get_support()]
                    
                    # Update dataframe
                    df = df.drop(columns=removed_cols)
                    if len(removed_cols) > 0:
                        preprocessing_info["removed_columns"].extend(removed_cols.tolist())
                        preprocessing_info["steps_applied"].append(
                            f"Removed {len(removed_cols)} low-variance features"
                        )
            
            # 5. Outlier removal
            if config.outlier_method != "none":
                numerical_cols = df.select_dtypes(include=[np.number]).columns
                outlier_indices = set()
                
                for col in numerical_cols:
                    if config.outlier_method == "iqr":
                        outliers = self._detect_outliers_iqr(df[col])
                    elif config.outlier_method == "zscore":
                        outliers = self._detect_outliers_zscore(df[col], config.outlier_threshold)
                    else:
                        outliers = []
                    
                    outlier_indices.update(outliers)
                
                if outlier_indices:
                    df = df.drop(index=list(outlier_indices))
                    preprocessing_info["outliers_removed"] = len(outlier_indices)
                    preprocessing_info["steps_applied"].append(
                        f"Removed {len(outlier_indices)} outlier rows using {config.outlier_method} method"
                    )
            
            # 6. Scaling
            if config.scaler_type != "none":
                numerical_cols = df.select_dtypes(include=[np.number]).columns
                if len(numerical_cols) > 0:
                    scaler = self.scalers[config.scaler_type]()
                    df[numerical_cols] = scaler.fit_transform(df[numerical_cols])
                    preprocessing_info["steps_applied"].append(
                        f"Applied {config.scaler_type} scaling to numerical features"
                    )
            
            # Validate final dataset
            if len(df) < config.min_samples:
                raise ValueError(f"Dataset has too few samples ({len(df)}) after preprocessing")
            
            if df.shape[1] > config.max_features:
                logger.warning(f"Dataset has many features ({df.shape[1]}), consider feature selection")
            
            preprocessing_info["final_shape"] = df.shape
            
            return df.values, preprocessing_info
            
        except Exception as e:
            logger.error(f"Error preprocessing data: {str(e)}")
            raise ValueError(f"Failed to preprocess data: {str(e)}")
    
    def _detect_outliers_iqr(self, series: pd.Series) -> List[int]:
        """Detect outliers using IQR method."""
        Q1 = series.quantile(0.25)
        Q3 = series.quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR
        
        outliers = series[(series < lower) | (series > upper)].index.tolist()
        return outliers
    
    def _detect_outliers_zscore(self, series: pd.Series, threshold: float = 3.0) -> List[int]:
        """Detect outliers using Z-score method."""
        z_scores = np.abs(stats.zscore(series.dropna()))
        outliers = series[z_scores > threshold].index.tolist()
        return outliers
    
    def _generate_recommendations(
        self,
        df: pd.DataFrame,
        missing_percentages: Dict[str, float],
        constant_columns: List[str],
        high_cardinality_columns: List[str],
        skewed_columns: List[str],
        correlations: Optional[Dict[str, float]]
    ) -> List[str]:
        """Generate preprocessing recommendations based on data analysis."""
        recommendations = []
        
        # Missing values
        high_missing = [col for col, pct in missing_percentages.items() if pct > 20]
        if high_missing:
            recommendations.append(
                f"Consider removing or carefully imputing {len(high_missing)} columns with >20% missing values"
            )
        
        # Constant columns
        if constant_columns:
            recommendations.append(
                f"Remove {len(constant_columns)} constant columns (no variance)"
            )
        
        # High cardinality
        if high_cardinality_columns:
            recommendations.append(
                f"Consider encoding or binning {len(high_cardinality_columns)} high-cardinality columns"
            )
        
        # Skewed features
        if skewed_columns:
            recommendations.append(
                f"Consider log transformation for {len(skewed_columns)} highly skewed features"
            )
        
        # High correlations
        if correlations:
            recommendations.append(
                f"Consider removing one feature from {len(correlations)} highly correlated pairs"
            )
        
        # General recommendations
        if df.shape[1] > 100:
            recommendations.append("Consider dimensionality reduction (PCA) for high-dimensional data")
        
        if len(df) < 100:
            recommendations.append("Small dataset - be cautious with complex models and cross-validation")
        
        return recommendations
