"""
Grid Search schemas for parameter optimization.

This module defines the request/response models for grid search functionality,
allowing users to run multiple clustering experiments across parameter grids.
"""

from typing import Dict, List, Any, Optional, Union
from pydantic import BaseModel, Field
from datetime import datetime
from enum import Enum

class GridSearchStatus(str, Enum):
    """Status of a grid search experiment."""
    PENDING = "pending"
    RUNNING = "running" 
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"

class ParameterGrid(BaseModel):
    """
    Parameter grid specification for a single algorithm.
    
    Each parameter can be:
    - A single value (no grid search for this param)
    - A list of values to try
    - A range specification with min, max, step
    """
    algorithm: str = Field(..., description="Algorithm name (e.g., 'spectral', 'dbscan')")
    parameters: Dict[str, Union[Any, List[Any], Dict[str, Any]]] = Field(
        ..., 
        description="Parameter specifications - can be values, lists, or range objects"
    )
    
    class Config:
        schema_extra = {
            "example": {
                "algorithm": "spectral",
                "parameters": {
                    "n_clusters": [2, 3, 4, 5],
                    "gamma": {"min": 0.1, "max": 2.0, "step": 0.3},
                    "affinity": ["rbf", "linear"],
                    "random_state": 42
                }
            }
        }

class GridSearchRequest(BaseModel):
    """Request for grid search parameter optimization."""
    dataset_id: str = Field(..., description="ID of the dataset to use")
    experiment_name: str = Field(..., description="Name for this experiment")
    description: Optional[str] = Field(None, description="Optional description")
    
    parameter_grids: List[ParameterGrid] = Field(
        ..., 
        description="Parameter grids for algorithms to test"
    )
    
    optimization_metric: str = Field(
        default="silhouette_score",
        description="Metric to optimize (silhouette_score, davies_bouldin_score, etc.)"
    )
    maximize_metric: bool = Field(
        default=True,
        description="Whether to maximize (True) or minimize (False) the optimization metric"
    )
    
    use_gpu: bool = Field(default=True, description="Whether to use GPU acceleration")
    max_concurrent_runs: int = Field(
        default=3, 
        ge=1, 
        le=10,
        description="Maximum number of concurrent runs"
    )
    
    class Config:
        schema_extra = {
            "example": {
                "dataset_id": "abc123",
                "experiment_name": "Spectral Clustering Optimization",
                "description": "Finding optimal parameters for spectral clustering",
                "parameter_grids": [
                    {
                        "algorithm": "spectral",
                        "parameters": {
                            "n_clusters": [2, 3, 4, 5],
                            "gamma": [0.1, 0.5, 1.0, 2.0],
                            "affinity": ["rbf", "linear"]
                        }
                    }
                ],
                "optimization_metric": "silhouette_score",
                "maximize_metric": True,
                "use_gpu": True,
                "max_concurrent_runs": 2
            }
        }

class GridSearchRun(BaseModel):
    """Individual run within a grid search experiment."""
    run_id: str = Field(..., description="Unique identifier for this run")
    algorithm: str = Field(..., description="Algorithm used")
    parameters: Dict[str, Any] = Field(..., description="Parameter values for this run")
    status: str = Field(..., description="Run status")
    
    # Results (populated after completion)
    metrics: Optional[Dict[str, float]] = Field(None, description="Clustering metrics")
    execution_time: Optional[float] = Field(None, description="Execution time in seconds")
    gpu_used: Optional[bool] = Field(None, description="Whether GPU was used")
    error_message: Optional[str] = Field(None, description="Error message if failed")
    
    created_at: datetime = Field(default_factory=datetime.utcnow)
    completed_at: Optional[datetime] = Field(None)

class GridSearchExperiment(BaseModel):
    """Grid search experiment with multiple runs."""
    group_id: str = Field(..., description="Unique identifier for the experiment")
    experiment_name: str = Field(..., description="Name of the experiment")
    description: Optional[str] = Field(None, description="Description")
    dataset_id: str = Field(..., description="Dataset used")
    
    # Configuration
    optimization_metric: str = Field(..., description="Metric being optimized")
    maximize_metric: bool = Field(..., description="Whether to maximize the metric")
    total_runs: int = Field(..., description="Total number of runs in the grid")
    
    # Status tracking
    status: GridSearchStatus = Field(default=GridSearchStatus.PENDING)
    completed_runs: int = Field(default=0, description="Number of completed runs")
    failed_runs: int = Field(default=0, description="Number of failed runs")
    
    # Results
    best_run_id: Optional[str] = Field(None, description="ID of the best performing run")
    best_score: Optional[float] = Field(None, description="Best metric score achieved")
    best_parameters: Optional[Dict[str, Any]] = Field(None, description="Best parameters found")
    
    # Timing
    created_at: datetime = Field(default_factory=datetime.utcnow)
    started_at: Optional[datetime] = Field(None)
    completed_at: Optional[datetime] = Field(None)
    
    # Runs
    runs: List[GridSearchRun] = Field(default_factory=list, description="Individual runs")

class GridSearchSummary(BaseModel):
    """Summary of a grid search experiment."""
    group_id: str
    experiment_name: str
    status: GridSearchStatus
    total_runs: int
    completed_runs: int
    failed_runs: int
    
    best_run: Optional[GridSearchRun] = Field(None, description="Best performing run")
    progress_percentage: float = Field(..., description="Completion percentage")
    
    created_at: datetime
    started_at: Optional[datetime]
    completed_at: Optional[datetime]
    estimated_completion: Optional[datetime] = Field(
        None, 
        description="Estimated completion time"
    )

class LeaderboardEntry(BaseModel):
    """Entry in the grid search leaderboard."""
    rank: int = Field(..., description="Rank based on optimization metric")
    run_id: str = Field(..., description="Run identifier")
    experiment_name: str = Field(..., description="Experiment name")
    algorithm: str = Field(..., description="Algorithm used")
    parameters: Dict[str, Any] = Field(..., description="Parameter values")
    
    # Key metrics
    optimization_score: float = Field(..., description="Score for the optimization metric")
    silhouette_score: Optional[float] = None
    davies_bouldin_score: Optional[float] = None
    calinski_harabasz_score: Optional[float] = None
    
    # Execution details
    execution_time: float = Field(..., description="Execution time in seconds")
    gpu_used: bool = Field(..., description="Whether GPU was used")
    completed_at: datetime = Field(..., description="Completion timestamp")

class GridSearchResponse(BaseModel):
    """Response after submitting a grid search request."""
    group_id: str = Field(..., description="Experiment group identifier")
    message: str = Field(..., description="Status message")
    total_runs: int = Field(..., description="Total number of runs scheduled")
    estimated_duration: str = Field(..., description="Estimated total duration")
    polling_url: str = Field(..., description="URL to check progress")
    websocket_url: str = Field(..., description="WebSocket URL for real-time updates")
