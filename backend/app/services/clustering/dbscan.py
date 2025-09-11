"""
DBSCAN (Density-Based Spatial Clustering of Applications with Noise) algorithm implementation.

This module provides DBSCAN clustering with parameter validation and optimization
for different data types and sizes.
"""

import numpy as np
from typing import Dict, Any
from .factory import BaseClusteringAlgorithm, AlgorithmType
import logging

logger = logging.getLogger(__name__)

class DBSCANClustering(BaseClusteringAlgorithm):
    """
    DBSCAN clustering for density-based cluster discovery.
    
    Automatically detects clusters of varying shapes and identifies outliers.
    Does not require specifying the number of clusters in advance.
    """
    
    def get_algorithm_type(self) -> AlgorithmType:
        return AlgorithmType.DBSCAN
    
    def get_default_parameters(self) -> Dict[str, Any]:
        return {
            'eps': 0.5,
            'min_samples': 5,
            'metric': 'euclidean',
            'algorithm': 'auto',  # 'auto', 'ball_tree', 'kd_tree', 'brute'
            'leaf_size': 30,
            'p': 2,  # Power parameter for Minkowski metric
            'n_jobs': -1,  # Use all available cores
        }
    
    def validate_parameters(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Validate DBSCAN parameters."""
        validated = params.copy()
        
        # Validate eps
        if validated.get('eps', 0.5) <= 0:
            validated['eps'] = 0.5
            logger.warning("eps must be positive, setting to 0.5")
        
        # Validate min_samples
        if validated.get('min_samples', 5) < 1:
            validated['min_samples'] = 5
            logger.warning("min_samples must be >= 1, setting to 5")
        
        # Validate metric
        valid_metrics = [
            'euclidean', 'manhattan', 'chebyshev', 'minkowski',
            'cosine', 'hamming', 'jaccard', 'braycurtis'
        ]
        if validated.get('metric', 'euclidean') not in valid_metrics:
            validated['metric'] = 'euclidean'
            logger.warning(f"Invalid metric, setting to euclidean. Valid options: {valid_metrics}")
        
        # Validate algorithm
        valid_algorithms = ['auto', 'ball_tree', 'kd_tree', 'brute']
        if validated.get('algorithm', 'auto') not in valid_algorithms:
            validated['algorithm'] = 'auto'
            logger.warning(f"Invalid algorithm, setting to auto. Valid options: {valid_algorithms}")
        
        # Validate leaf_size
        if validated.get('leaf_size', 30) < 1:
            validated['leaf_size'] = 30
        
        # Validate p (for Minkowski metric)
        if validated.get('p', 2) < 1:
            validated['p'] = 2
        
        return validated
    
    def _fit_predict(self, X: np.ndarray, params: Dict[str, Any]) -> np.ndarray:
        """Core DBSCAN implementation."""
        if self.use_gpu:
            return self._fit_predict_gpu(X, params)
        else:
            return self._fit_predict_cpu(X, params)
    
    def _fit_predict_gpu(self, X: np.ndarray, params: Dict[str, Any]) -> np.ndarray:
        """GPU-accelerated DBSCAN using cuML if available."""
        try:
            # Try to use cuML (RAPIDS) for GPU acceleration
            import cuml
            from cuml.cluster import DBSCAN as cuDBSCAN
            
            logger.info("Using GPU acceleration (cuML) for DBSCAN")
            
            # cuML DBSCAN parameters
            dbscan = cuDBSCAN(
                eps=params['eps'],
                min_samples=params['min_samples'],
                metric=params['metric']
            )
            
            labels = dbscan.fit_predict(X)
            
            # Convert from cuML to numpy if needed
            if hasattr(labels, 'to_array'):
                labels = labels.to_array()
            elif hasattr(labels, 'get'):
                labels = labels.get()
            
            return labels.astype(np.int32)
            
        except ImportError:
            logger.info("cuML not available, trying PyTorch implementation")
            return self._fit_predict_pytorch(X, params)
        except Exception as e:
            logger.warning(f"GPU DBSCAN failed: {str(e)}, falling back to CPU")
            return self._fit_predict_cpu(X, params)
    
    def _fit_predict_pytorch(self, X: np.ndarray, params: Dict[str, Any]) -> np.ndarray:
        """PyTorch-based DBSCAN implementation for GPU."""
        try:
            import torch
            
            if not torch.cuda.is_available():
                raise RuntimeError("CUDA not available")
            
            logger.info("Using PyTorch implementation for GPU DBSCAN")
            
            # Convert to GPU tensors
            X_gpu = torch.tensor(X, dtype=torch.float32).cuda()
            eps = params['eps']
            min_samples = params['min_samples']
            
            # Compute pairwise distances on GPU
            if params['metric'] == 'euclidean':
                distances = torch.cdist(X_gpu, X_gpu, p=2)
            elif params['metric'] == 'manhattan':
                distances = torch.cdist(X_gpu, X_gpu, p=1)
            else:
                # Fallback for unsupported metrics
                raise ValueError(f"Metric {params['metric']} not supported in PyTorch implementation")
            
            # Find neighbors within eps
            neighbors = distances <= eps
            
            # Count neighbors for each point
            neighbor_counts = torch.sum(neighbors, dim=1) - 1  # Exclude self
            
            # Identify core points
            core_points = neighbor_counts >= min_samples
            
            # Move back to CPU for cluster assignment (complex graph operations)
            neighbors_cpu = neighbors.cpu().numpy()
            core_points_cpu = core_points.cpu().numpy()
            
            # Perform cluster assignment on CPU
            labels = self._assign_clusters_cpu(neighbors_cpu, core_points_cpu)
            
            return labels
            
        except Exception as e:
            logger.warning(f"PyTorch DBSCAN failed: {str(e)}, falling back to CPU")
            return self._fit_predict_cpu(X, params)
    
    def _fit_predict_cpu(self, X: np.ndarray, params: Dict[str, Any]) -> np.ndarray:
        """CPU DBSCAN using scikit-learn."""
        from sklearn.cluster import DBSCAN
        
        logger.info("Using CPU DBSCAN (scikit-learn)")
        
        # Create DBSCAN instance
        dbscan = DBSCAN(
            eps=params['eps'],
            min_samples=params['min_samples'],
            metric=params['metric'],
            algorithm=params['algorithm'],
            leaf_size=params['leaf_size'],
            p=params['p'],
            n_jobs=params['n_jobs']
        )
        
        # Fit and predict
        labels = dbscan.fit_predict(X)
        
        return labels
    
    def _assign_clusters_cpu(self, neighbors: np.ndarray, core_points: np.ndarray) -> np.ndarray:
        """
        Assign cluster labels using CPU-based graph traversal.
        
        This implements the core DBSCAN algorithm for cluster assignment.
        """
        n_points = neighbors.shape[0]
        labels = np.full(n_points, -1, dtype=np.int32)  # Initialize as noise
        cluster_id = 0
        
        # Process each core point
        for i in range(n_points):
            if not core_points[i] or labels[i] != -1:
                continue  # Skip non-core points or already processed points
            
            # Start new cluster
            cluster_points = []
            stack = [i]
            
            while stack:
                current = stack.pop()
                if labels[current] != -1:
                    continue
                
                labels[current] = cluster_id
                cluster_points.append(current)
                
                if core_points[current]:
                    # Add unvisited neighbors to stack
                    neighbor_indices = np.where(neighbors[current])[0]
                    for neighbor in neighbor_indices:
                        if labels[neighbor] == -1:
                            stack.append(neighbor)
            
            cluster_id += 1
        
        return labels
    
    def get_cluster_statistics(self, labels: np.ndarray) -> Dict[str, Any]:
        """Get statistics about the clustering result."""
        unique_labels = np.unique(labels)
        n_clusters = len(unique_labels[unique_labels >= 0])  # Exclude noise (-1)
        n_noise = np.sum(labels == -1)
        n_points = len(labels)
        
        cluster_sizes = []
        for label in unique_labels:
            if label >= 0:
                cluster_sizes.append(np.sum(labels == label))
        
        return {
            'n_clusters': n_clusters,
            'n_noise_points': n_noise,
            'noise_ratio': n_noise / n_points if n_points > 0 else 0,
            'cluster_sizes': cluster_sizes,
            'min_cluster_size': min(cluster_sizes) if cluster_sizes else 0,
            'max_cluster_size': max(cluster_sizes) if cluster_sizes else 0,
            'avg_cluster_size': np.mean(cluster_sizes) if cluster_sizes else 0
        }
    
    def _compute_metrics(self, X: np.ndarray, labels: np.ndarray) -> Dict[str, float]:
        """Compute DBSCAN-specific metrics in addition to standard ones."""
        # Get standard metrics
        metrics = super()._compute_metrics(X, labels)
        
        # Add DBSCAN-specific statistics
        stats = self.get_cluster_statistics(labels)
        metrics.update({
            'n_clusters_found': stats['n_clusters'],
            'noise_ratio': stats['noise_ratio'],
            'avg_cluster_size': stats['avg_cluster_size']
        })
        
        return metrics
