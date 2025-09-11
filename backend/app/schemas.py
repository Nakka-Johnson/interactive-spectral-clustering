"""
Pydantic schemas for Interactive Spectral Clustering Platform.

This module defines all the Pydantic models for request/response validation
and API documentation. Supports Phase 2 and Phase 3 frontend requirements.
"""

from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Union
from uuid import UUID

from pydantic import BaseModel, Field, validator


class AlgorithmId(str, Enum):
    """Supported clustering algorithms."""
    KMEANS = "kmeans"
    SPECTRAL = "spectral"
    DBSCAN = "dbscan"
    HIERARCHICAL = "hierarchical"
    GMM = "gmm"


class RunStatus(str, Enum):
    """Clustering run status."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


class PreprocessingOptions(BaseModel):
    """Data preprocessing configuration."""
    normalize: bool = Field(False, description="Apply Min-Max normalization")
    standardize: bool = Field(True, description="Apply Z-score standardization")
    handle_missing: str = Field("drop", description="Missing value strategy: drop, mean, median")
    pca_components: Optional[int] = Field(None, description="Number of PCA components")
    pca_variance_threshold: Optional[float] = Field(0.95, description="PCA variance retention")


class ParameterMap(BaseModel):
    """Dynamic parameter mapping for different algorithms."""
    
    # K-Means parameters
    n_clusters: Optional[int] = Field(None, description="Number of clusters")
    max_iter: Optional[int] = Field(300, description="Maximum iterations")
    tol: Optional[float] = Field(1e-4, description="Convergence tolerance")
    
    # Spectral clustering parameters
    gamma: Optional[float] = Field(1.0, description="RBF kernel gamma parameter")
    eigen_solver: Optional[str] = Field("arpack", description="Eigenvalue solver method")
    n_neighbors: Optional[int] = Field(10, description="Number of neighbors for k-NN graph")
    
    # DBSCAN parameters
    eps: Optional[float] = Field(0.5, description="Maximum distance between samples")
    min_samples: Optional[int] = Field(5, description="Minimum samples in neighborhood")
    
    # Hierarchical clustering parameters
    linkage: Optional[str] = Field("ward", description="Linkage criterion")
    distance_threshold: Optional[float] = Field(None, description="Distance threshold")
    
    # GMM parameters
    covariance_type: Optional[str] = Field("full", description="Covariance type")
    
    @validator('linkage')
    def validate_linkage(cls, v):
        """Validate hierarchical clustering linkage."""
        valid_linkages = ['ward', 'complete', 'average', 'single']
        if v and v not in valid_linkages:
            raise ValueError(f"Linkage must be one of {valid_linkages}")
        return v
    
    @validator('covariance_type')
    def validate_covariance_type(cls, v):
        """Validate GMM covariance type."""
        valid_types = ['full', 'tied', 'diag', 'spherical']
        if v and v not in valid_types:
            raise ValueError(f"Covariance type must be one of {valid_types}")
        return v


class DatasetMeta(BaseModel):
    """Dataset metadata response."""
    dataset_id: str = Field(..., description="Unique dataset identifier")
    filename: str = Field(..., description="Original filename")
    upload_time: datetime = Field(..., description="Upload timestamp")
    total_rows: int = Field(..., description="Number of rows")
    total_columns: int = Field(..., description="Number of columns")
    file_size_bytes: int = Field(..., description="File size in bytes")
    column_names: List[str] = Field(..., description="List of column names")
    sample_data: Optional[List[List[Any]]] = Field(None, description="Sample rows for preview")


class DatasetPreview(BaseModel):
    """Dataset preview response for Phase 2."""
    columns: List[str] = Field(..., description="Column names")
    rows: List[List[Any]] = Field(..., description="Preview rows (up to 50)")
    total_rows: int = Field(..., description="Total number of rows")
    total_columns: int = Field(..., description="Total number of columns")
    preview_rows: int = Field(..., description="Number of preview rows returned")


class RunRequest(BaseModel):
    """Clustering run request."""
    dataset_id: str = Field(..., description="Dataset identifier")
    algorithms: List[AlgorithmId] = Field(..., description="Selected algorithms to run")
    parameters: Dict[AlgorithmId, ParameterMap] = Field(..., description="Algorithm parameters")
    preprocessing: PreprocessingOptions = Field(default_factory=PreprocessingOptions)
    run_name: Optional[str] = Field(None, description="Optional run name")
    
    @validator('algorithms')
    def validate_algorithms(cls, v):
        """Ensure at least one algorithm is selected."""
        if not v:
            raise ValueError("At least one algorithm must be selected")
        return v


class ClusteringMetrics(BaseModel):
    """Clustering evaluation metrics."""
    silhouette_score: Optional[float] = Field(None, description="Silhouette coefficient")
    calinski_harabasz_score: Optional[float] = Field(None, description="Calinski-Harabasz index")
    davies_bouldin_score: Optional[float] = Field(None, description="Davies-Bouldin index")
    inertia: Optional[float] = Field(None, description="Within-cluster sum of squares")
    n_clusters_found: int = Field(..., description="Number of clusters found")
    n_noise_points: Optional[int] = Field(None, description="Number of noise points (DBSCAN)")


class AlgorithmResult(BaseModel):
    """Results for a single algorithm."""
    algorithm: AlgorithmId = Field(..., description="Algorithm identifier")
    status: RunStatus = Field(..., description="Algorithm execution status")
    execution_time: float = Field(..., description="Execution time in seconds")
    labels: Optional[List[int]] = Field(None, description="Cluster labels")
    cluster_centers: Optional[List[List[float]]] = Field(None, description="Cluster centers")
    metrics: Optional[ClusteringMetrics] = Field(None, description="Evaluation metrics")
    error_message: Optional[str] = Field(None, description="Error message if failed")
    parameters_used: Optional[ParameterMap] = Field(None, description="Parameters used")


class EmbeddingRequest(BaseModel):
    """Embedding generation request."""
    dataset_id: str = Field(..., description="Dataset identifier")
    method: str = Field(..., description="Embedding method: pca, tsne, umap")
    n_components: int = Field(2, description="Number of dimensions")
    
    # PCA parameters
    
    # t-SNE parameters
    perplexity: Optional[float] = Field(30.0, description="t-SNE perplexity")
    learning_rate: Optional[float] = Field(200.0, description="t-SNE learning rate")
    n_iter: Optional[int] = Field(1000, description="t-SNE iterations")
    
    # UMAP parameters
    n_neighbors: Optional[int] = Field(15, description="UMAP n_neighbors")
    min_dist: Optional[float] = Field(0.1, description="UMAP min_dist")
    
    @validator('method')
    def validate_method(cls, v):
        """Validate embedding method."""
        valid_methods = ['pca', 'tsne', 'umap']
        if v.lower() not in valid_methods:
            raise ValueError(f"Method must be one of {valid_methods}")
        return v.lower()


class EmbeddingResponse(BaseModel):
    """Embedding generation response."""
    embedding_id: str = Field(..., description="Unique embedding identifier")
    method: str = Field(..., description="Embedding method used")
    n_components: int = Field(..., description="Number of dimensions")
    coordinates: List[List[float]] = Field(..., description="Embedding coordinates")
    explained_variance_ratio: Optional[List[float]] = Field(None, description="PCA explained variance")
    execution_time: float = Field(..., description="Execution time in seconds")


class ClusteringRun(BaseModel):
    """Complete clustering run response."""
    run_id: str = Field(..., description="Unique run identifier")
    dataset_id: str = Field(..., description="Dataset identifier")
    run_name: Optional[str] = Field(None, description="Run name")
    status: RunStatus = Field(..., description="Overall run status")
    created_at: datetime = Field(..., description="Creation timestamp")
    completed_at: Optional[datetime] = Field(None, description="Completion timestamp")
    total_execution_time: Optional[float] = Field(None, description="Total execution time")
    
    # Results
    algorithms_results: List[AlgorithmResult] = Field(..., description="Per-algorithm results")
    preprocessing_applied: PreprocessingOptions = Field(..., description="Preprocessing options used")
    
    # Dataset info
    dataset_info: Optional[DatasetMeta] = Field(None, description="Dataset metadata")
    
    # Error handling
    error_message: Optional[str] = Field(None, description="Run-level error message")


class RunResponse(BaseModel):
    """Simple run creation response."""
    run_id: str = Field(..., description="Created run identifier")
    status: RunStatus = Field(..., description="Initial status")
    message: str = Field(..., description="Success message")


class GridSearchRequest(BaseModel):
    """Grid search request for hyperparameter optimization."""
    dataset_id: str = Field(..., description="Dataset identifier")
    algorithm: AlgorithmId = Field(..., description="Algorithm to optimize")
    parameter_grid: Dict[str, List[Any]] = Field(..., description="Parameter grid")
    cv_folds: int = Field(5, description="Cross-validation folds")
    scoring_metric: str = Field("silhouette", description="Optimization metric")
    preprocessing: PreprocessingOptions = Field(default_factory=PreprocessingOptions)
    
    @validator('scoring_metric')
    def validate_scoring_metric(cls, v):
        """Validate scoring metric."""
        valid_metrics = ['silhouette', 'calinski_harabasz', 'davies_bouldin']
        if v not in valid_metrics:
            raise ValueError(f"Scoring metric must be one of {valid_metrics}")
        return v


class GridSearchResult(BaseModel):
    """Grid search optimization result."""
    search_id: str = Field(..., description="Search identifier")
    algorithm: AlgorithmId = Field(..., description="Algorithm optimized")
    best_params: ParameterMap = Field(..., description="Best parameters found")
    best_score: float = Field(..., description="Best cross-validation score")
    results: List[Dict[str, Any]] = Field(..., description="All parameter combinations tested")
    execution_time: float = Field(..., description="Total optimization time")


class HealthResponse(BaseModel):
    """Health check response."""
    status: str = Field(..., description="Service status")
    timestamp: datetime = Field(..., description="Check timestamp")
    version: str = Field(..., description="API version")
    database_connected: bool = Field(..., description="Database connection status")
    dependencies: Dict[str, str] = Field(..., description="Dependency versions")


class ErrorResponse(BaseModel):
    """Standardized error response."""
    code: str = Field(..., description="Error code")
    message: str = Field(..., description="Human-readable error message")
    details: Optional[Dict[str, Any]] = Field(None, description="Additional error details")
    correlation_id: Optional[str] = Field(None, description="Request correlation ID")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Error timestamp")


# Export schemas for easy importing
__all__ = [
    "AlgorithmId",
    "RunStatus", 
    "PreprocessingOptions",
    "ParameterMap",
    "DatasetMeta",
    "DatasetPreview",
    "RunRequest",
    "ClusteringMetrics",
    "AlgorithmResult",
    "EmbeddingRequest",
    "EmbeddingResponse",
    "ClusteringRun",
    "RunResponse",
    "GridSearchRequest",
    "GridSearchResult",
    "HealthResponse",
    "ErrorResponse",
]
