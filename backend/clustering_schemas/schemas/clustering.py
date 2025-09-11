"""
Clustering schemas for Interactive Spectral Clustering Platform.

This module contains all Pydantic models for clustering operations,
including request/response models, algorithm configurations, and data structures.
"""

from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Union
from uuid import UUID, uuid4

from pydantic import BaseModel, Field, validator


class AlgorithmId(str, Enum):
    """Supported clustering algorithms."""
    SPECTRAL = "spectral"
    KMEANS = "kmeans"
    DBSCAN = "dbscan"
    GMM = "gmm"
    AGGLOMERATIVE = "agglomerative"


class RunStatus(str, Enum):
    """Clustering run execution status."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class EmbeddingMethod(str, Enum):
    """Available embedding methods for dimensionality reduction."""
    PCA = "pca"
    TSNE = "tsne"
    UMAP = "umap"
    SPECTRAL = "spectral"


# Type alias for flexible parameter mapping
ParameterMap = Dict[str, Union[int, float, str, bool, List[Union[int, float]]]]


class DatasetRef(BaseModel):
    """Reference to a dataset for clustering operations."""
    dataset_id: str = Field(..., description="Unique identifier for the dataset")
    name: str = Field(..., description="Human-readable dataset name")
    shape: tuple[int, int] = Field(..., description="Dataset dimensions (rows, cols)")
    
    class Config:
        schema_extra = {
            "example": {
                "dataset_id": "iris_001",
                "name": "Iris Dataset",
                "shape": [150, 4]
            }
        }


class ClusteringRequest(BaseModel):
    """Request model for clustering operations."""
    algorithm: AlgorithmId = Field(..., description="Clustering algorithm to use")
    parameters: ParameterMap = Field(default_factory=dict, description="Algorithm-specific parameters")
    dataset_ref: DatasetRef = Field(..., description="Reference to the dataset")
    run_id: Optional[str] = Field(default_factory=lambda: str(uuid4()), description="Optional run identifier")
    
    @validator('parameters')
    def validate_parameters(cls, v, values):
        """Validate parameters based on the selected algorithm."""
        algorithm = values.get('algorithm')
        if algorithm == AlgorithmId.SPECTRAL:
            required_params = {'n_clusters'}
            if not required_params.issubset(v.keys()):
                raise ValueError(f"Spectral clustering requires parameters: {required_params}")
        elif algorithm == AlgorithmId.KMEANS:
            if 'n_clusters' not in v:
                raise ValueError("K-means requires 'n_clusters' parameter")
        elif algorithm == AlgorithmId.DBSCAN:
            required_params = {'eps', 'min_samples'}
            if not required_params.issubset(v.keys()):
                raise ValueError(f"DBSCAN requires parameters: {required_params}")
        return v
    
    class Config:
        schema_extra = {
            "example": {
                "algorithm": "spectral",
                "parameters": {
                    "n_clusters": 3,
                    "gamma": 1.0,
                    "affinity": "rbf"
                },
                "dataset_ref": {
                    "dataset_id": "iris_001",
                    "name": "Iris Dataset", 
                    "shape": [150, 4]
                },
                "run_id": "run_001"
            }
        }


class ClusteringMetrics(BaseModel):
    """Clustering evaluation metrics."""
    silhouette_score: Optional[float] = Field(None, description="Silhouette coefficient")
    calinski_harabasz_score: Optional[float] = Field(None, description="Calinski-Harabasz index")
    davies_bouldin_score: Optional[float] = Field(None, description="Davies-Bouldin index")
    inertia: Optional[float] = Field(None, description="Within-cluster sum of squares")
    n_clusters: int = Field(..., description="Number of clusters found")
    
    class Config:
        schema_extra = {
            "example": {
                "silhouette_score": 0.55,
                "calinski_harabasz_score": 561.63,
                "davies_bouldin_score": 0.92,
                "inertia": 78.85,
                "n_clusters": 3
            }
        }


class ClusteringRun(BaseModel):
    """Clustering run status and metadata."""
    run_id: str = Field(..., description="Unique run identifier")
    status: RunStatus = Field(default=RunStatus.PENDING, description="Current run status")
    algorithm: AlgorithmId = Field(..., description="Algorithm used")
    parameters: ParameterMap = Field(..., description="Parameters used")
    dataset_ref: DatasetRef = Field(..., description="Dataset reference")
    created_at: datetime = Field(default_factory=datetime.utcnow, description="Run creation time")
    started_at: Optional[datetime] = Field(None, description="Run start time")
    completed_at: Optional[datetime] = Field(None, description="Run completion time")
    error_message: Optional[str] = Field(None, description="Error message if failed")
    
    class Config:
        schema_extra = {
            "example": {
                "run_id": "run_001",
                "status": "completed",
                "algorithm": "spectral",
                "parameters": {"n_clusters": 3, "gamma": 1.0},
                "dataset_ref": {
                    "dataset_id": "iris_001",
                    "name": "Iris Dataset",
                    "shape": [150, 4]
                },
                "created_at": "2024-01-15T10:30:00Z",
                "started_at": "2024-01-15T10:30:01Z", 
                "completed_at": "2024-01-15T10:30:05Z"
            }
        }


class ClusteringResponse(BaseModel):
    """Response model for clustering operations."""
    run: ClusteringRun = Field(..., description="Run metadata")
    labels: Optional[List[int]] = Field(None, description="Cluster labels for each data point")
    cluster_centers: Optional[List[List[float]]] = Field(None, description="Cluster centroids")
    metrics: Optional[ClusteringMetrics] = Field(None, description="Clustering evaluation metrics")
    
    class Config:
        schema_extra = {
            "example": {
                "run": {
                    "run_id": "run_001",
                    "status": "completed",
                    "algorithm": "spectral"
                },
                "labels": [0, 0, 1, 1, 2, 2],
                "cluster_centers": [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]],
                "metrics": {
                    "silhouette_score": 0.55,
                    "n_clusters": 3
                }
            }
        }


class GridSearchRequest(BaseModel):
    """Request model for hyperparameter grid search."""
    algorithm: AlgorithmId = Field(..., description="Algorithm for grid search")
    parameter_grid: Dict[str, List[Union[int, float, str]]] = Field(..., description="Parameter grid to search")
    dataset_ref: DatasetRef = Field(..., description="Dataset reference")
    cv_folds: int = Field(default=5, description="Cross-validation folds")
    scoring_metric: str = Field(default="silhouette", description="Optimization metric")
    run_id: Optional[str] = Field(default_factory=lambda: str(uuid4()), description="Optional run identifier")
    
    @validator('parameter_grid')
    def validate_parameter_grid(cls, v, values):
        """Ensure parameter grid is not empty and contains valid parameters."""
        if not v:
            raise ValueError("Parameter grid cannot be empty")
        
        algorithm = values.get('algorithm')
        if algorithm == AlgorithmId.SPECTRAL:
            valid_params = {'n_clusters', 'gamma', 'affinity', 'eigen_solver'}
        elif algorithm == AlgorithmId.KMEANS:
            valid_params = {'n_clusters', 'init', 'n_init', 'max_iter'}
        elif algorithm == AlgorithmId.DBSCAN:
            valid_params = {'eps', 'min_samples', 'metric'}
        else:
            valid_params = set(v.keys())  # Accept all for other algorithms
        
        invalid_params = set(v.keys()) - valid_params
        if invalid_params:
            raise ValueError(f"Invalid parameters for {algorithm}: {invalid_params}")
        
        return v
    
    class Config:
        schema_extra = {
            "example": {
                "algorithm": "spectral",
                "parameter_grid": {
                    "n_clusters": [2, 3, 4, 5],
                    "gamma": [0.1, 1.0, 10.0],
                    "affinity": ["rbf", "nearest_neighbors"]
                },
                "dataset_ref": {
                    "dataset_id": "iris_001",
                    "name": "Iris Dataset",
                    "shape": [150, 4]
                },
                "cv_folds": 5,
                "scoring_metric": "silhouette"
            }
        }


class GridSearchResult(BaseModel):
    """Individual result from grid search."""
    parameters: ParameterMap = Field(..., description="Parameter combination tested")
    score: float = Field(..., description="Cross-validation score")
    std_score: float = Field(..., description="Standard deviation of CV scores")
    rank: int = Field(..., description="Rank of this parameter combination")
    
    class Config:
        schema_extra = {
            "example": {
                "parameters": {"n_clusters": 3, "gamma": 1.0},
                "score": 0.55,
                "std_score": 0.02,
                "rank": 1
            }
        }


class GridSearchResponse(BaseModel):
    """Response model for grid search operations."""
    run_id: str = Field(..., description="Grid search run identifier")
    status: RunStatus = Field(..., description="Grid search status")
    algorithm: AlgorithmId = Field(..., description="Algorithm used")
    best_parameters: Optional[ParameterMap] = Field(None, description="Best parameter combination found")
    best_score: Optional[float] = Field(None, description="Best cross-validation score")
    results: List[GridSearchResult] = Field(default_factory=list, description="All grid search results")
    created_at: datetime = Field(default_factory=datetime.utcnow, description="Grid search creation time")
    completed_at: Optional[datetime] = Field(None, description="Grid search completion time")
    
    class Config:
        schema_extra = {
            "example": {
                "run_id": "gridsearch_001",
                "status": "completed",
                "algorithm": "spectral",
                "best_parameters": {"n_clusters": 3, "gamma": 1.0},
                "best_score": 0.55,
                "results": [
                    {
                        "parameters": {"n_clusters": 3, "gamma": 1.0},
                        "score": 0.55,
                        "std_score": 0.02,
                        "rank": 1
                    }
                ]
            }
        }


class EmbeddingRequest(BaseModel):
    """Request model for dimensionality reduction/embedding."""
    method: EmbeddingMethod = Field(..., description="Embedding method to use")
    parameters: ParameterMap = Field(default_factory=dict, description="Method-specific parameters")
    dataset_ref: DatasetRef = Field(..., description="Dataset reference")
    target_dimensions: int = Field(default=2, description="Target dimensionality")
    run_id: Optional[str] = Field(default_factory=lambda: str(uuid4()), description="Optional run identifier")
    
    @validator('target_dimensions')
    def validate_target_dimensions(cls, v):
        """Ensure target dimensions is positive."""
        if v <= 0:
            raise ValueError("Target dimensions must be positive")
        return v
    
    @validator('parameters')
    def validate_embedding_parameters(cls, v, values):
        """Validate parameters based on the selected embedding method."""
        method = values.get('method')
        if method == EmbeddingMethod.TSNE:
            if 'perplexity' in v and (v['perplexity'] <= 0 or v['perplexity'] >= 50):
                raise ValueError("t-SNE perplexity should be between 0 and 50")
        elif method == EmbeddingMethod.UMAP:
            if 'n_neighbors' in v and v['n_neighbors'] <= 1:
                raise ValueError("UMAP n_neighbors must be greater than 1")
        return v
    
    class Config:
        schema_extra = {
            "example": {
                "method": "tsne",
                "parameters": {
                    "perplexity": 30.0,
                    "learning_rate": 200.0,
                    "n_iter": 1000
                },
                "dataset_ref": {
                    "dataset_id": "iris_001",
                    "name": "Iris Dataset",
                    "shape": [150, 4]
                },
                "target_dimensions": 2
            }
        }


class EmbeddingResult(BaseModel):
    """Result of dimensionality reduction."""
    coordinates: List[List[float]] = Field(..., description="Embedded coordinates")
    explained_variance_ratio: Optional[List[float]] = Field(None, description="Explained variance ratio (for PCA)")
    stress: Optional[float] = Field(None, description="Stress value (for MDS-like methods)")
    
    class Config:
        schema_extra = {
            "example": {
                "coordinates": [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]],
                "explained_variance_ratio": [0.7, 0.2],
                "stress": 0.15
            }
        }


class EmbeddingResponse(BaseModel):
    """Response model for embedding operations."""
    run_id: str = Field(..., description="Embedding run identifier")
    status: RunStatus = Field(..., description="Embedding status")
    method: EmbeddingMethod = Field(..., description="Embedding method used")
    parameters: ParameterMap = Field(..., description="Parameters used")
    dataset_ref: DatasetRef = Field(..., description="Dataset reference")
    result: Optional[EmbeddingResult] = Field(None, description="Embedding result")
    created_at: datetime = Field(default_factory=datetime.utcnow, description="Embedding creation time")
    completed_at: Optional[datetime] = Field(None, description="Embedding completion time")
    
    class Config:
        schema_extra = {
            "example": {
                "run_id": "embedding_001",
                "status": "completed",
                "method": "tsne",
                "parameters": {"perplexity": 30.0},
                "dataset_ref": {
                    "dataset_id": "iris_001",
                    "name": "Iris Dataset",
                    "shape": [150, 4]
                },
                "result": {
                    "coordinates": [[1.0, 2.0], [3.0, 4.0]],
                    "stress": 0.15
                }
            }
        }
