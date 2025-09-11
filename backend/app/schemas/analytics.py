"""
Schemas for data processing and analytics endpoints.

Defines request/response models for preprocessing, statistics,
exports, and batch processing functionality.
"""

from pydantic import BaseModel, Field, validator
from typing import Dict, Any, List, Optional, Union
from datetime import datetime
from enum import Enum


# ===== Preprocessing Schemas =====

class ScalerType(str, Enum):
    """Supported scaling methods."""
    STANDARD = "standard"
    MINMAX = "minmax"
    ROBUST = "robust"
    NONE = "none"


class MissingStrategy(str, Enum):
    """Missing value handling strategies."""
    MEAN = "mean"
    MEDIAN = "median"
    MODE = "mode"
    DROP = "drop"
    KNN = "knn"


class OutlierMethod(str, Enum):
    """Outlier detection methods."""
    IQR = "iqr"
    ZSCORE = "zscore"
    ISOLATION = "isolation"
    NONE = "none"


class PreprocessingRequest(BaseModel):
    """Request for data preprocessing."""
    
    # Scaling configuration
    scaler_type: ScalerType = ScalerType.STANDARD
    
    # Missing value handling
    missing_strategy: MissingStrategy = MissingStrategy.MEAN
    missing_threshold: float = Field(0.5, ge=0.0, le=1.0, description="Drop columns with this fraction of missing values")
    
    # Feature selection
    variance_threshold: float = Field(0.01, ge=0.0, description="Remove features with variance below this threshold")
    
    # Outlier detection
    outlier_method: OutlierMethod = OutlierMethod.IQR
    outlier_threshold: float = Field(3.0, gt=0.0, description="Threshold for outlier detection")
    
    # Data validation
    min_samples: int = Field(10, gt=0, description="Minimum number of samples required")
    max_features: int = Field(1000, gt=0, description="Maximum number of features allowed")


class PreprocessingResponse(BaseModel):
    """Response from data preprocessing."""
    
    original_shape: List[int]
    final_shape: List[int]
    steps_applied: List[str]
    removed_columns: List[str]
    outliers_removed: int
    processing_time_seconds: float
    
    class Config:
        schema_extra = {
            "example": {
                "original_shape": [1000, 20],
                "final_shape": [950, 18],
                "steps_applied": [
                    "Removed 2 columns with >50% missing values",
                    "Imputed missing values using mean strategy",
                    "Removed 50 outlier rows using IQR method",
                    "Applied standard scaling to numerical features"
                ],
                "removed_columns": ["feature_5", "feature_12"],
                "outliers_removed": 50,
                "processing_time_seconds": 2.35
            }
        }


# ===== Statistics Schemas =====

class DatasetStatsResponse(BaseModel):
    """Comprehensive dataset statistics response."""
    
    # Basic information
    shape: List[int]
    memory_usage_mb: float
    dtypes: Dict[str, str]
    
    # Missing values
    missing_counts: Dict[str, int]
    missing_percentages: Dict[str, float]
    total_missing: int
    
    # Numerical statistics
    numerical_stats: Dict[str, Dict[str, float]]
    correlations: Optional[Dict[str, float]]
    
    # Categorical statistics
    categorical_stats: Dict[str, Dict[str, Any]]
    
    # Data quality metrics
    duplicate_rows: int
    constant_columns: List[str]
    high_cardinality_columns: List[str]
    skewed_columns: List[str]
    outlier_counts: Dict[str, int]
    
    # Recommendations
    preprocessing_recommendations: List[str]
    
    class Config:
        schema_extra = {
            "example": {
                "shape": [1000, 15],
                "memory_usage_mb": 0.12,
                "dtypes": {"feature_0": "float64", "feature_1": "int64"},
                "missing_counts": {"feature_0": 5, "feature_1": 0},
                "missing_percentages": {"feature_0": 0.5, "feature_1": 0.0},
                "total_missing": 5,
                "numerical_stats": {
                    "feature_0": {
                        "count": 995,
                        "mean": 1.5,
                        "std": 0.8,
                        "min": -2.1,
                        "max": 4.2,
                        "skewness": 0.1,
                        "kurtosis": -0.5
                    }
                },
                "correlations": {"feature_0-feature_1": 0.95},
                "categorical_stats": {},
                "duplicate_rows": 3,
                "constant_columns": [],
                "high_cardinality_columns": ["feature_10"],
                "skewed_columns": ["feature_5"],
                "outlier_counts": {"feature_0": 12, "feature_1": 8},
                "preprocessing_recommendations": [
                    "Consider removing one feature from 1 highly correlated pairs",
                    "Consider log transformation for 1 highly skewed features"
                ]
            }
        }


# ===== Export Schemas =====

class ExportFormat(str, Enum):
    """Supported export formats."""
    CSV = "csv"
    JSON = "json"
    HTML = "html"
    ZIP = "zip"


class ExportRequest(BaseModel):
    """Request for exporting clustering results."""
    
    format: ExportFormat
    include_original_data: bool = True
    include_metadata: bool = True
    
    class Config:
        schema_extra = {
            "example": {
                "format": "zip",
                "include_original_data": True,
                "include_metadata": True
            }
        }


# ===== Batch Processing Schemas =====

class BatchJobRequest(BaseModel):
    """Individual job configuration within a batch."""
    
    dataset_id: str = Field(..., description="Dataset identifier")
    algorithm: str = Field(..., description="Clustering algorithm name")
    parameters: Dict[str, Any] = Field(default_factory=dict, description="Algorithm parameters")
    
    class Config:
        schema_extra = {
            "example": {
                "dataset_id": "dataset_123",
                "algorithm": "spectral",
                "parameters": {
                    "n_clusters": 3,
                    "gamma": 1.0,
                    "affinity": "rbf"
                }
            }
        }


class BatchRequest(BaseModel):
    """Request for batch processing."""
    
    name: str = Field(..., description="Batch name")
    description: Optional[str] = Field(None, description="Batch description")
    jobs: List[BatchJobRequest] = Field(..., min_items=1, description="List of clustering jobs")
    
    # Execution settings
    max_parallel_jobs: int = Field(3, ge=1, le=10, description="Maximum parallel jobs")
    stop_on_error: bool = Field(False, description="Stop batch if any job fails")
    timeout_minutes: int = Field(60, ge=1, le=1440, description="Batch timeout in minutes")
    
    # Notification settings
    notify_on_completion: bool = Field(False, description="Send notification when complete")
    email: Optional[str] = Field(None, description="Email for notifications")
    
    @validator('email')
    def validate_email(cls, v, values):
        if values.get('notify_on_completion') and not v:
            raise ValueError('Email required when notify_on_completion is True')
        return v
    
    class Config:
        schema_extra = {
            "example": {
                "name": "Multi-dataset clustering analysis",
                "description": "Compare clustering algorithms across different datasets",
                "jobs": [
                    {
                        "dataset_id": "dataset_1",
                        "algorithm": "spectral",
                        "parameters": {"n_clusters": 3}
                    },
                    {
                        "dataset_id": "dataset_1",
                        "algorithm": "dbscan",
                        "parameters": {"eps": 0.5, "min_samples": 5}
                    }
                ],
                "max_parallel_jobs": 2,
                "stop_on_error": False,
                "timeout_minutes": 30
            }
        }


class JobStatus(str, Enum):
    """Job execution status."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


class BatchStatus(str, Enum):
    """Batch execution status."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class BatchJobResponse(BaseModel):
    """Individual job status response."""
    
    id: str
    dataset_id: str
    algorithm: str
    parameters: Dict[str, Any]
    status: JobStatus
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    duration_seconds: Optional[float] = None


class BatchProgressResponse(BaseModel):
    """Batch progress information."""
    
    total_jobs: int
    completed_jobs: int
    failed_jobs: int
    progress_percentage: float


class BatchResponse(BaseModel):
    """Batch execution response."""
    
    id: str
    name: str
    description: Optional[str]
    status: BatchStatus
    progress: BatchProgressResponse
    
    # Timing information
    created_at: datetime
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    
    # Job details
    jobs: List[BatchJobResponse]
    
    class Config:
        schema_extra = {
            "example": {
                "id": "batch_12345",
                "name": "Multi-dataset analysis",
                "description": "Comparing algorithms",
                "status": "running",
                "progress": {
                    "total_jobs": 4,
                    "completed_jobs": 2,
                    "failed_jobs": 0,
                    "progress_percentage": 50.0
                },
                "created_at": "2023-12-01T10:00:00",
                "started_at": "2023-12-01T10:01:00",
                "completed_at": None,
                "jobs": [
                    {
                        "id": "job_1",
                        "dataset_id": "dataset_1",
                        "algorithm": "spectral",
                        "parameters": {"n_clusters": 3},
                        "status": "completed",
                        "duration_seconds": 5.2
                    }
                ]
            }
        }


class BatchSummaryResponse(BaseModel):
    """Batch summary statistics."""
    
    batch_id: str
    name: str
    status: BatchStatus
    
    execution_time: Dict[str, Any]
    job_statistics: Dict[str, Any]
    results_by_algorithm: Dict[str, Any]
    results_by_dataset: Dict[str, Any]
    performance_metrics: Dict[str, Any]
    
    class Config:
        schema_extra = {
            "example": {
                "batch_id": "batch_12345",
                "name": "Multi-dataset analysis",
                "status": "completed",
                "execution_time": {
                    "started_at": "2023-12-01T10:00:00",
                    "completed_at": "2023-12-01T10:15:00",
                    "duration_minutes": 15.0
                },
                "job_statistics": {
                    "total": 4,
                    "completed": 4,
                    "failed": 0,
                    "success_rate": 100.0
                },
                "results_by_algorithm": {
                    "spectral": {
                        "job_count": 2,
                        "avg_clusters": 3.5,
                        "avg_duration": 4.2
                    }
                }
            }
        }


class BatchListResponse(BaseModel):
    """Response for listing batches."""
    
    batches: List[Dict[str, Any]]
    total_count: int
    
    class Config:
        schema_extra = {
            "example": {
                "batches": [
                    {
                        "id": "batch_1",
                        "name": "Analysis 1",
                        "status": "completed",
                        "total_jobs": 3,
                        "completed_jobs": 3,
                        "failed_jobs": 0,
                        "created_at": "2023-12-01T09:00:00",
                        "completed_at": "2023-12-01T09:10:00"
                    }
                ],
                "total_count": 1
            }
        }
