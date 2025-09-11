"""
Embedding service for dimensionality reduction and visualization.

Provides PCA, t-SNE, and UMAP embeddings with caching and graceful fallbacks.
"""

import time
import hashlib
import json
import logging
from typing import Dict, Any, List, Optional, Tuple
import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from sklearn.utils import check_array

from ..schemas.embedding import (
    EmbeddingRequest, 
    EmbeddingResponse, 
    EmbeddingPoint,
    EmbeddingCacheKey,
    AvailableMethodsResponse
)

logger = logging.getLogger(__name__)


class EmbeddingService:
    """Service for generating 2D embeddings from high-dimensional data."""
    
    def __init__(self):
        """Initialize the embedding service with cache."""
        self.cache: Dict[str, EmbeddingResponse] = {}
        self.scaler = StandardScaler()
        
        # Check for UMAP availability
        self.umap_available = self._check_umap_availability()
        
        # Default parameters for each method
        self.default_parameters = {
            'pca': {
                'n_components': 2,
                'random_state': 42,
                'svd_solver': 'auto'
            },
            'tsne': {
                'n_components': 2,
                'perplexity': 30.0,
                'learning_rate': 200.0,
                'n_iter': 1000,
                'random_state': 42,
                'init': 'random',
                'metric': 'euclidean'
            },
            'umap': {
                'n_components': 2,
                'n_neighbors': 15,
                'min_dist': 0.1,
                'metric': 'euclidean',
                'random_state': 42
            }
        }
        
        # Method information
        self.method_info = {
            'pca': {
                'description': 'Principal Component Analysis - linear dimensionality reduction',
                'speed': 'fast',
                'preserves': 'global_structure',
                'best_for': 'Linear relationships, global structure preservation'
            },
            'tsne': {
                'description': 't-Distributed Stochastic Neighbor Embedding - nonlinear reduction',
                'speed': 'slow',
                'preserves': 'local_structure',
                'best_for': 'Local neighborhoods, cluster visualization'
            },
            'umap': {
                'description': 'Uniform Manifold Approximation and Projection - fast nonlinear reduction',
                'speed': 'medium',
                'preserves': 'both_local_and_global',
                'best_for': 'Balance of local and global structure'
            }
        }
    
    def _check_umap_availability(self) -> bool:
        """Check if UMAP is available."""
        try:
            import umap
            return True
        except ImportError:
            logger.warning("UMAP not available. Install with: pip install umap-learn")
            return False
    
    def get_available_methods(self) -> AvailableMethodsResponse:
        """Get list of available embedding methods and their parameters."""
        methods = ['pca', 'tsne']
        if self.umap_available:
            methods.append('umap')
        
        # Filter method info and parameters based on availability
        available_params = {method: self.default_parameters[method] for method in methods}
        available_info = {method: self.method_info[method] for method in methods}
        
        return AvailableMethodsResponse(
            methods=methods,
            default_parameters=available_params,
            method_info=available_info
        )
    
    def _generate_cache_key(self, request: EmbeddingRequest, data_hash: str) -> str:
        """Generate cache key for the embedding request."""
        cache_key_model = EmbeddingCacheKey(
            dataset_id=f"{request.dataset_id}:{data_hash}",
            method=request.method,
            parameters=request.parameters
        )
        return cache_key_model.to_cache_key()
    
    def _hash_data(self, data: np.ndarray) -> str:
        """Generate hash of the data for cache key."""
        return hashlib.md5(data.tobytes()).hexdigest()[:16]
    
    def _prepare_data(self, data: np.ndarray) -> np.ndarray:
        """Prepare data for embedding (validation and scaling)."""
        # Validate input
        data = check_array(data, dtype=np.float32, ensure_2d=True)
        
        # Check minimum requirements
        if data.shape[0] < 4:
            raise ValueError("Need at least 4 data points for embedding")
        
        if data.shape[1] < 2:
            raise ValueError("Data must have at least 2 dimensions")
        
        # Scale data for better embedding results
        data_scaled = self.scaler.fit_transform(data)
        
        return data_scaled
    
    def _embed_pca(self, data: np.ndarray, parameters: Dict[str, Any]) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Generate PCA embedding."""
        # Merge with defaults
        params = {**self.default_parameters['pca'], **parameters}
        params['n_components'] = 2  # Force 2D for visualization
        
        logger.info(f"Generating PCA embedding with parameters: {params}")
        
        pca = PCA(**params)
        embedding = pca.fit_transform(data)
        
        # Additional info
        extra_info = {
            'explained_variance_ratio': pca.explained_variance_ratio_.tolist()
        }
        
        return embedding, extra_info
    
    def _embed_tsne(self, data: np.ndarray, parameters: Dict[str, Any]) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Generate t-SNE embedding."""
        # Merge with defaults
        params = {**self.default_parameters['tsne'], **parameters}
        params['n_components'] = 2  # Force 2D for visualization
        
        # Adjust perplexity based on data size
        max_perplexity = min(50, (data.shape[0] - 1) // 3)
        if params['perplexity'] > max_perplexity:
            params['perplexity'] = max_perplexity
            logger.warning(f"Reduced perplexity to {max_perplexity} due to small dataset")
        
        logger.info(f"Generating t-SNE embedding with parameters: {params}")
        
        tsne = TSNE(**params)
        embedding = tsne.fit_transform(data)
        
        # Additional info
        extra_info = {
            'kl_divergence': float(tsne.kl_divergence_)
        }
        
        return embedding, extra_info
    
    def _embed_umap(self, data: np.ndarray, parameters: Dict[str, Any]) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Generate UMAP embedding."""
        if not self.umap_available:
            raise ValueError("UMAP is not available. Install with: pip install umap-learn")
        
        import umap
        
        # Merge with defaults
        params = {**self.default_parameters['umap'], **parameters}
        params['n_components'] = 2  # Force 2D for visualization
        
        # Adjust n_neighbors based on data size
        max_neighbors = min(200, data.shape[0] - 1)
        if params['n_neighbors'] > max_neighbors:
            params['n_neighbors'] = max_neighbors
            logger.warning(f"Reduced n_neighbors to {max_neighbors} due to small dataset")
        
        logger.info(f"Generating UMAP embedding with parameters: {params}")
        
        reducer = umap.UMAP(**params)
        embedding = reducer.fit_transform(data)
        
        # Additional info (UMAP doesn't provide standard metrics)
        extra_info = {}
        
        return embedding, extra_info
    
    async def generate_embedding(
        self, 
        request: EmbeddingRequest, 
        data: np.ndarray
    ) -> EmbeddingResponse:
        """Generate 2D embedding from high-dimensional data."""
        start_time = time.time()
        
        # Prepare data
        try:
            data_prepared = self._prepare_data(data)
        except Exception as e:
            logger.error(f"Data preparation failed: {e}")
            raise ValueError(f"Invalid data: {e}")
        
        # Generate cache key
        data_hash = self._hash_data(data_prepared)
        cache_key = self._generate_cache_key(request, data_hash)
        
        # Check cache
        if cache_key in self.cache:
            logger.info(f"Cache hit for embedding: {request.method}")
            cached_result = self.cache[cache_key]
            cached_result.cache_hit = True
            return cached_result
        
        # Generate embedding based on method
        try:
            if request.method == 'pca':
                embedding, extra_info = self._embed_pca(data_prepared, request.parameters)
            elif request.method == 'tsne':
                embedding, extra_info = self._embed_tsne(data_prepared, request.parameters)
            elif request.method == 'umap':
                embedding, extra_info = self._embed_umap(data_prepared, request.parameters)
            else:
                raise ValueError(f"Unknown embedding method: {request.method}")
        
        except Exception as e:
            logger.error(f"Embedding generation failed: {e}")
            raise ValueError(f"Failed to generate {request.method} embedding: {e}")
        
        # Convert to response format
        points = [
            EmbeddingPoint(
                x=float(embedding[i, 0]),
                y=float(embedding[i, 1]),
                original_index=i
            )
            for i in range(embedding.shape[0])
        ]
        
        execution_time = time.time() - start_time
        
        # Create response
        response = EmbeddingResponse(
            dataset_id=request.dataset_id,
            method=request.method,
            parameters=request.parameters,
            points=points,
            execution_time=execution_time,
            cache_hit=False,
            **extra_info
        )
        
        # Cache the result
        self.cache[cache_key] = response
        
        logger.info(
            f"Generated {request.method} embedding: "
            f"{len(points)} points in {execution_time:.3f}s"
        )
        
        return response
    
    def clear_cache(self) -> int:
        """Clear the embedding cache and return number of entries cleared."""
        count = len(self.cache)
        self.cache.clear()
        logger.info(f"Cleared {count} cached embeddings")
        return count
    
    def get_cache_info(self) -> Dict[str, Any]:
        """Get information about the current cache state."""
        return {
            'cached_embeddings': len(self.cache),
            'cache_size_mb': sum(
                len(str(response.dict()).encode()) 
                for response in self.cache.values()
            ) / (1024 * 1024),
            'available_methods': self.get_available_methods().methods
        }
