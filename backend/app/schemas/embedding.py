"""
Schemas for dimensionality reduction and embedding endpoints.

Provides request/response models for generating 2D embeddings from high-dimensional data
using various dimensionality reduction techniques (PCA, t-SNE, UMAP).
"""

from typing import Optional, Dict, Any, List, Literal
from pydantic import BaseModel, Field, validator


class EmbeddingRequest(BaseModel):
    """Request model for generating 2D embeddings."""
    
    dataset_id: str = Field(..., description="ID of the dataset to embed")
    method: Literal["pca", "tsne", "umap"] = Field(
        ..., 
        description="Dimensionality reduction method to use"
    )
    parameters: Dict[str, Any] = Field(
        default_factory=dict,
        description="Method-specific parameters"
    )
    
    @validator('parameters')
    def validate_parameters(cls, v, values):
        """Validate parameters based on the chosen method."""
        method = values.get('method')
        
        if method == 'pca':
            # PCA parameters validation
            valid_params = {'n_components', 'random_state', 'svd_solver'}
            invalid_params = set(v.keys()) - valid_params
            if invalid_params:
                raise ValueError(f"Invalid PCA parameters: {invalid_params}")
                
            if 'n_components' in v and v['n_components'] != 2:
                raise ValueError("n_components must be 2 for visualization")
                
        elif method == 'tsne':
            # t-SNE parameters validation
            valid_params = {
                'n_components', 'perplexity', 'learning_rate', 'n_iter', 
                'random_state', 'init', 'metric', 'early_exaggeration'
            }
            invalid_params = set(v.keys()) - valid_params
            if invalid_params:
                raise ValueError(f"Invalid t-SNE parameters: {invalid_params}")
                
            if 'n_components' in v and v['n_components'] != 2:
                raise ValueError("n_components must be 2 for visualization")
                
            if 'perplexity' in v and (v['perplexity'] < 5 or v['perplexity'] > 50):
                raise ValueError("perplexity must be between 5 and 50")
                
        elif method == 'umap':
            # UMAP parameters validation
            valid_params = {
                'n_components', 'n_neighbors', 'min_dist', 'metric', 
                'random_state', 'spread', 'learning_rate'
            }
            invalid_params = set(v.keys()) - valid_params
            if invalid_params:
                raise ValueError(f"Invalid UMAP parameters: {invalid_params}")
                
            if 'n_components' in v and v['n_components'] != 2:
                raise ValueError("n_components must be 2 for visualization")
                
            if 'n_neighbors' in v and (v['n_neighbors'] < 2 or v['n_neighbors'] > 200):
                raise ValueError("n_neighbors must be between 2 and 200")
                
        return v


class EmbeddingPoint(BaseModel):
    """A single 2D point in the embedding space."""
    
    x: float = Field(..., description="X coordinate in 2D space")
    y: float = Field(..., description="Y coordinate in 2D space")
    original_index: int = Field(..., description="Index in original dataset")


class EmbeddingResponse(BaseModel):
    """Response model for embedding generation."""
    
    dataset_id: str = Field(..., description="ID of the embedded dataset")
    method: str = Field(..., description="Dimensionality reduction method used")
    parameters: Dict[str, Any] = Field(..., description="Parameters used for embedding")
    points: List[EmbeddingPoint] = Field(..., description="2D embedding points")
    execution_time: float = Field(..., description="Time taken to generate embedding (seconds)")
    cache_hit: bool = Field(..., description="Whether result was retrieved from cache")
    explained_variance_ratio: Optional[List[float]] = Field(
        None, 
        description="Explained variance ratio for PCA (if applicable)"
    )
    kl_divergence: Optional[float] = Field(
        None,
        description="Final KL divergence for t-SNE (if applicable)"
    )
    
    class Config:
        schema_extra = {
            "example": {
                "dataset_id": "dataset_123",
                "method": "pca",
                "parameters": {"n_components": 2, "random_state": 42},
                "points": [
                    {"x": 1.5, "y": -0.8, "original_index": 0},
                    {"x": -2.1, "y": 1.3, "original_index": 1}
                ],
                "execution_time": 0.234,
                "cache_hit": False,
                "explained_variance_ratio": [0.65, 0.23]
            }
        }


class EmbeddingCacheKey(BaseModel):
    """Model for generating cache keys for embeddings."""
    
    dataset_id: str
    method: str
    parameters: Dict[str, Any]
    
    def to_cache_key(self) -> str:
        """Generate a string cache key from the embedding parameters."""
        import hashlib
        import json
        
        # Sort parameters for consistent hashing
        sorted_params = json.dumps(self.parameters, sort_keys=True)
        content = f"{self.dataset_id}:{self.method}:{sorted_params}"
        
        return hashlib.md5(content.encode()).hexdigest()


class AvailableMethodsResponse(BaseModel):
    """Response model for available embedding methods."""
    
    methods: List[str] = Field(..., description="List of available embedding methods")
    default_parameters: Dict[str, Dict[str, Any]] = Field(
        ..., 
        description="Default parameters for each method"
    )
    method_info: Dict[str, Dict[str, Any]] = Field(
        ...,
        description="Information about each method including description and parameter ranges"
    )
    
    class Config:
        schema_extra = {
            "example": {
                "methods": ["pca", "tsne", "umap"],
                "default_parameters": {
                    "pca": {"n_components": 2, "random_state": 42},
                    "tsne": {"n_components": 2, "perplexity": 30, "random_state": 42},
                    "umap": {"n_components": 2, "n_neighbors": 15, "min_dist": 0.1}
                },
                "method_info": {
                    "pca": {
                        "description": "Principal Component Analysis - linear dimensionality reduction",
                        "speed": "fast",
                        "preserves": "global_structure"
                    }
                }
            }
        }
