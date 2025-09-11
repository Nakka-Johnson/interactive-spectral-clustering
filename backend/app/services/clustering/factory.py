"""
Clustering algorithm factory and registry for Interactive Spectral Clustering Platform.

This module provides a clean interface for registering and creating clustering algorithms
with proper parameter validation and GPU/CPU fallback support.
"""

import time
import numpy as np
from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import logging

logger = logging.getLogger(__name__)

class AlgorithmType(Enum):
    """Supported clustering algorithm types."""
    SPECTRAL = "spectral"
    KMEANS = "kmeans"
    DBSCAN = "dbscan"
    GMM = "gmm"
    AGGLOMERATIVE = "agglomerative"

@dataclass
class ClusteringResult:
    """Result of a clustering operation."""
    labels: np.ndarray
    metrics: Dict[str, float]
    runtime_ms: float
    algorithm: str
    parameters: Dict[str, Any]
    n_clusters_found: int
    gpu_used: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            'labels': self.labels.tolist(),
            'metrics': self.metrics,
            'runtime_ms': self.runtime_ms,
            'algorithm': self.algorithm,
            'parameters': self.parameters,
            'n_clusters_found': self.n_clusters_found,
            'gpu_used': self.gpu_used
        }

class BaseClusteringAlgorithm(ABC):
    """Base class for all clustering algorithms."""
    
    def __init__(self, use_gpu: bool = False):
        self.use_gpu = use_gpu and self._gpu_available()
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        
    @abstractmethod
    def get_algorithm_type(self) -> AlgorithmType:
        """Return the algorithm type."""
        pass
    
    @abstractmethod
    def get_default_parameters(self) -> Dict[str, Any]:
        """Return default parameters for this algorithm."""
        pass
    
    @abstractmethod
    def validate_parameters(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Validate and normalize parameters. Return validated params."""
        pass
    
    @abstractmethod
    def _fit_predict(self, X: np.ndarray, params: Dict[str, Any]) -> np.ndarray:
        """Core clustering implementation. Return cluster labels."""
        pass
    
    def fit_predict(self, X: np.ndarray, params: Dict[str, Any] = None) -> ClusteringResult:
        """
        Fit the clustering algorithm and predict cluster labels.
        
        Args:
            X: Input data of shape (n_samples, n_features)
            params: Algorithm-specific parameters
            
        Returns:
            ClusteringResult with labels, metrics, and metadata
        """
        start_time = time.time()
        
        # Use default parameters if none provided
        if params is None:
            params = self.get_default_parameters()
        else:
            # Merge with defaults and validate
            default_params = self.get_default_parameters()
            default_params.update(params)
            params = self.validate_parameters(default_params)
        
        self.logger.info(f"Starting {self.get_algorithm_type().value} clustering with params: {params}")
        
        try:
            # Perform clustering
            labels = self._fit_predict(X, params)
            
            # Calculate runtime
            runtime_ms = (time.time() - start_time) * 1000
            
            # Compute metrics
            metrics = self._compute_metrics(X, labels)
            
            # Count clusters
            n_clusters_found = len(np.unique(labels[labels >= 0]))  # Exclude noise (-1)
            
            result = ClusteringResult(
                labels=labels,
                metrics=metrics,
                runtime_ms=runtime_ms,
                algorithm=self.get_algorithm_type().value,
                parameters=params,
                n_clusters_found=n_clusters_found,
                gpu_used=self.use_gpu
            )
            
            self.logger.info(f"Clustering completed in {runtime_ms:.2f}ms, found {n_clusters_found} clusters")
            return result
            
        except Exception as e:
            runtime_ms = (time.time() - start_time) * 1000
            self.logger.error(f"Clustering failed after {runtime_ms:.2f}ms: {str(e)}")
            raise
    
    def _compute_metrics(self, X: np.ndarray, labels: np.ndarray) -> Dict[str, float]:
        """Compute clustering evaluation metrics."""
        from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
        
        metrics = {}
        
        # Only compute metrics if we have valid clusters
        unique_labels = np.unique(labels)
        valid_labels = labels[labels >= 0]  # Exclude noise points
        
        if len(unique_labels) > 1 and len(valid_labels) > 1:
            try:
                # Silhouette score (higher is better, -1 to 1)
                if len(unique_labels) > 1:
                    metrics['silhouette_score'] = float(silhouette_score(X, labels))
                
                # Davies-Bouldin score (lower is better, >= 0)
                if len(unique_labels) > 1:
                    metrics['davies_bouldin_score'] = float(davies_bouldin_score(X, labels))
                
                # Calinski-Harabasz score (higher is better, >= 0)
                if len(unique_labels) > 1:
                    metrics['calinski_harabasz_score'] = float(calinski_harabasz_score(X, labels))
                
            except Exception as e:
                self.logger.warning(f"Failed to compute some metrics: {str(e)}")
                metrics['metric_error'] = str(e)
        else:
            self.logger.warning("Cannot compute metrics: insufficient clusters or all points are noise")
            metrics['warning'] = "Insufficient clusters for metric computation"
        
        return metrics
    
    def _gpu_available(self) -> bool:
        """Check if GPU is available for this algorithm."""
        try:
            import torch
            return torch.cuda.is_available()
        except ImportError:
            return False

class ClusteringFactory:
    """Factory for creating clustering algorithms."""
    
    def __init__(self):
        self._algorithms: Dict[AlgorithmType, type] = {}
        self.logger = logging.getLogger(f"{__name__}.ClusteringFactory")
    
    def register(self, algorithm_type: AlgorithmType, algorithm_class: type):
        """Register a clustering algorithm."""
        if not issubclass(algorithm_class, BaseClusteringAlgorithm):
            raise ValueError(f"Algorithm must inherit from BaseClusteringAlgorithm")
        
        self._algorithms[algorithm_type] = algorithm_class
        self.logger.info(f"Registered algorithm: {algorithm_type.value}")
    
    def create(self, algorithm_type: str, use_gpu: bool = False) -> BaseClusteringAlgorithm:
        """Create a clustering algorithm instance."""
        try:
            algo_enum = AlgorithmType(algorithm_type)
        except ValueError:
            available = [t.value for t in self._algorithms.keys()]
            raise ValueError(f"Unknown algorithm '{algorithm_type}'. Available: {available}")
        
        if algo_enum not in self._algorithms:
            available = [t.value for t in self._algorithms.keys()]
            raise ValueError(f"Algorithm '{algorithm_type}' not registered. Available: {available}")
        
        algorithm_class = self._algorithms[algo_enum]
        return algorithm_class(use_gpu=use_gpu)
    
    def get_available_algorithms(self) -> List[str]:
        """Get list of available algorithm names."""
        return [t.value for t in self._algorithms.keys()]
    
    def get_algorithm_info(self, algorithm_type: str) -> Dict[str, Any]:
        """Get information about an algorithm including default parameters."""
        algorithm = self.create(algorithm_type, use_gpu=False)
        return {
            'type': algorithm_type,
            'default_parameters': algorithm.get_default_parameters(),
            'gpu_supported': algorithm._gpu_available()
        }

# Global factory instance
clustering_factory = ClusteringFactory()

# Register all available algorithms
def register_all_algorithms():
    """Register all clustering algorithms with the factory."""
    # Import here to avoid circular imports
    from .spectral import SpectralClustering
    from .dbscan import DBSCANClustering
    from .gmm import GMMClustering
    from .agglomerative import AgglomerativeClustering
    
    # Register algorithms with their types
    clustering_factory.register(AlgorithmType.SPECTRAL, SpectralClustering)
    clustering_factory.register(AlgorithmType.DBSCAN, DBSCANClustering)
    clustering_factory.register(AlgorithmType.GMM, GMMClustering)
    clustering_factory.register(AlgorithmType.AGGLOMERATIVE, AgglomerativeClustering)

# Auto-register on import
register_all_algorithms()
