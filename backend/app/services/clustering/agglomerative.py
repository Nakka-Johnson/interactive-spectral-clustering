"""
Agglomerative clustering algorithm implementation.

This module provides hierarchical agglomerative clustering with various
linkage criteria and distance metrics for bottom-up cluster formation.
"""

import numpy as np
from typing import Dict, Any, Optional, TYPE_CHECKING
from .factory import BaseClusteringAlgorithm, AlgorithmType
import logging

if TYPE_CHECKING:
    import torch

logger = logging.getLogger(__name__)

class AgglomerativeClustering(BaseClusteringAlgorithm):
    """
    Agglomerative hierarchical clustering for bottom-up cluster formation.
    
    Builds a hierarchy of clusters by iteratively merging the closest pairs
    using various linkage criteria and distance metrics.
    """
    
    def get_algorithm_type(self) -> AlgorithmType:
        return AlgorithmType.AGGLOMERATIVE
    
    def get_default_parameters(self) -> Dict[str, Any]:
        return {
            'n_clusters': 3,
            'linkage': 'ward',  # 'ward', 'complete', 'average', 'single'
            'metric': 'euclidean',  # 'euclidean', 'manhattan', 'cosine', 'precomputed'
            'memory': None,
            'connectivity': None,
            'compute_full_tree': 'auto',
            'distance_threshold': None,
            'compute_distances': False,
        }
    
    def validate_parameters(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Validate Agglomerative clustering parameters."""
        validated = params.copy()
        
        # Handle n_clusters vs distance_threshold
        if validated.get('distance_threshold') is not None:
            validated['n_clusters'] = None
            if validated.get('distance_threshold') <= 0:
                validated['distance_threshold'] = None
                validated['n_clusters'] = 3
                logger.warning("Invalid distance_threshold, setting n_clusters to 3")
        else:
            if validated.get('n_clusters', 3) is None or validated.get('n_clusters', 3) < 2:
                validated['n_clusters'] = 3
                logger.warning("n_clusters must be >= 2, setting to 3")
            if validated.get('n_clusters', 3) > 100:
                validated['n_clusters'] = 100
                logger.warning("n_clusters too large, setting to 100")
        
        # Validate linkage
        valid_linkages = ['ward', 'complete', 'average', 'single']
        if validated.get('linkage', 'ward') not in valid_linkages:
            validated['linkage'] = 'ward'
            logger.warning(f"Invalid linkage, setting to ward. Valid options: {valid_linkages}")
        
        # Validate metric based on linkage
        if validated.get('linkage') == 'ward':
            if validated.get('metric', 'euclidean') != 'euclidean':
                validated['metric'] = 'euclidean'
                logger.warning("Ward linkage requires euclidean metric")
        else:
            valid_metrics = ['euclidean', 'manhattan', 'cosine', 'precomputed']
            if validated.get('metric', 'euclidean') not in valid_metrics:
                validated['metric'] = 'euclidean'
                logger.warning(f"Invalid metric, setting to euclidean. Valid options: {valid_metrics}")
        
        return validated
    
    def _fit_predict(self, X: np.ndarray, params: Dict[str, Any]) -> np.ndarray:
        """Core Agglomerative clustering implementation."""
        if self.use_gpu:
            return self._fit_predict_gpu(X, params)
        else:
            return self._fit_predict_cpu(X, params)
    
    def _fit_predict_gpu(self, X: np.ndarray, params: Dict[str, Any]) -> np.ndarray:
        """GPU-accelerated Agglomerative clustering using PyTorch."""
        try:
            import torch
            
            if not torch.cuda.is_available():
                logger.warning("GPU requested but not available, falling back to CPU")
                return self._fit_predict_cpu(X, params)
            
            logger.info("Using GPU acceleration (PyTorch) for Agglomerative clustering")
            
            # Convert to GPU tensors
            X_gpu = torch.tensor(X, dtype=torch.float32).cuda()
            n_samples = X_gpu.shape[0]
            
            # For very large datasets, use scikit-learn (more memory efficient)
            if n_samples > 10000:
                logger.info("Large dataset detected, using CPU implementation")
                return self._fit_predict_cpu(X, params)
            
            # Compute distance matrix on GPU
            distance_matrix = self._compute_distance_matrix_gpu(X_gpu, params['metric'])
            
            # Run agglomerative clustering on GPU
            labels = self._agglomerative_clustering_gpu(
                distance_matrix, params['n_clusters'], params['linkage']
            )
            
            return labels.cpu().numpy()
            
        except Exception as e:
            logger.warning(f"GPU Agglomerative clustering failed: {str(e)}, falling back to CPU")
            return self._fit_predict_cpu(X, params)
    
    def _fit_predict_cpu(self, X: np.ndarray, params: Dict[str, Any]) -> np.ndarray:
        """CPU Agglomerative clustering using scikit-learn."""
        from sklearn.cluster import AgglomerativeClustering as SklearnAgglomerative
        
        logger.info("Using CPU Agglomerative clustering (scikit-learn)")
        
        # Create clustering instance
        clustering = SklearnAgglomerative(
            n_clusters=params['n_clusters'],
            linkage=params['linkage'],
            metric=params['metric'],
            memory=params['memory'],
            connectivity=params['connectivity'],
            compute_full_tree=params['compute_full_tree'],
            distance_threshold=params['distance_threshold'],
            compute_distances=params['compute_distances']
        )
        
        # Fit and predict
        labels = clustering.fit_predict(X)
        
        # Store model for metrics
        self._clustering_model = clustering
        
        return labels
    
    def _compute_distance_matrix_gpu(self, X: "torch.Tensor", metric: str) -> "torch.Tensor":
        """Compute pairwise distance matrix on GPU."""
        import torch
        
        n_samples = X.shape[0]
        
        if metric == 'euclidean':
            # Efficient euclidean distance computation
            dist_matrix = torch.cdist(X, X, p=2)
        elif metric == 'manhattan':
            dist_matrix = torch.cdist(X, X, p=1)
        elif metric == 'cosine':
            # Cosine distance = 1 - cosine similarity
            X_norm = torch.nn.functional.normalize(X, p=2, dim=1)
            similarity = torch.mm(X_norm, X_norm.T)
            dist_matrix = 1 - similarity
        else:
            # Default to euclidean
            dist_matrix = torch.cdist(X, X, p=2)
        
        # Set diagonal to zero (distance from point to itself)
        dist_matrix.fill_diagonal_(0)
        
        return dist_matrix
    
    def _agglomerative_clustering_gpu(self, distance_matrix: "torch.Tensor", 
                                     n_clusters: int, linkage: str) -> "torch.Tensor":
        """Perform agglomerative clustering on GPU."""
        import torch
        
        n_samples = distance_matrix.shape[0]
        
        # Initialize each point as its own cluster
        labels = torch.arange(n_samples, device=distance_matrix.device)
        cluster_sizes = torch.ones(n_samples, device=distance_matrix.device)
        active_clusters = torch.ones(n_samples, dtype=torch.bool, device=distance_matrix.device)
        
        # Current number of clusters
        current_clusters = n_samples
        
        # Agglomerative merging
        while current_clusters > n_clusters:
            # Find the minimum distance between active clusters
            active_mask = active_clusters.unsqueeze(0) & active_clusters.unsqueeze(1)
            
            # Set diagonal and inactive clusters to infinity
            masked_distances = distance_matrix.clone()
            masked_distances[~active_mask] = float('inf')
            masked_distances.fill_diagonal_(float('inf'))
            
            # Find minimum distance
            min_idx = torch.argmin(masked_distances)
            i, j = min_idx // n_samples, min_idx % n_samples
            
            # Merge clusters i and j (keep i, merge j into i)
            labels[labels == j.item()] = i.item()
            
            # Update cluster sizes
            cluster_sizes[i] += cluster_sizes[j]
            cluster_sizes[j] = 0
            
            # Update distance matrix based on linkage
            self._update_distances_gpu(
                distance_matrix, i, j, cluster_sizes, active_clusters, linkage
            )
            
            # Deactivate cluster j
            active_clusters[j] = False
            current_clusters -= 1
        
        # Relabel clusters to be contiguous
        unique_labels = torch.unique(labels)
        label_mapping = torch.zeros(n_samples, dtype=torch.long, device=distance_matrix.device)
        for new_label, old_label in enumerate(unique_labels):
            label_mapping[old_label] = new_label
        
        return label_mapping[labels]
    
    def _update_distances_gpu(self, distance_matrix: "torch.Tensor", i: "torch.Tensor", 
                             j: "torch.Tensor", cluster_sizes: "torch.Tensor",
                             active_clusters: "torch.Tensor", linkage: str):
        """Update distance matrix after merging clusters i and j."""
        import torch
        
        n_samples = distance_matrix.shape[0]
        
        for k in range(n_samples):
            if k == i or k == j or not active_clusters[k]:
                continue
            
            d_ik = distance_matrix[i, k]
            d_jk = distance_matrix[j, k]
            
            if linkage == 'single':
                # Single linkage: minimum distance
                new_distance = torch.min(d_ik, d_jk)
            elif linkage == 'complete':
                # Complete linkage: maximum distance
                new_distance = torch.max(d_ik, d_jk)
            elif linkage == 'average':
                # Average linkage: weighted by cluster sizes
                size_i = cluster_sizes[i] - cluster_sizes[j]  # Original size of i
                size_j = cluster_sizes[j]
                new_distance = (size_i * d_ik + size_j * d_jk) / (size_i + size_j)
            elif linkage == 'ward':
                # Ward linkage: minimize within-cluster variance
                size_i = cluster_sizes[i] - cluster_sizes[j]
                size_j = cluster_sizes[j]
                size_k = cluster_sizes[k]
                
                new_distance = torch.sqrt(
                    ((size_i + size_k) * d_ik**2 + (size_j + size_k) * d_jk**2 - 
                     size_k * distance_matrix[i, j]**2) / (size_i + size_j + size_k)
                )
            else:
                # Default to average
                new_distance = (d_ik + d_jk) / 2
            
            # Update distance matrix
            distance_matrix[i, k] = new_distance
            distance_matrix[k, i] = new_distance
    
    def _compute_metrics(self, X: np.ndarray, labels: np.ndarray) -> Dict[str, float]:
        """Compute Agglomerative-specific metrics in addition to standard ones."""
        # Get standard metrics
        metrics = super()._compute_metrics(X, labels)
        
        # Add Agglomerative-specific metrics if available
        if hasattr(self, '_clustering_model'):
            model = self._clustering_model
            
            # Add hierarchy information
            if hasattr(model, 'n_clusters_'):
                metrics['n_clusters_found'] = model.n_clusters_
            
            if hasattr(model, 'children_'):
                metrics['n_merges'] = len(model.children_)
            
            if hasattr(model, 'distances_') and model.distances_ is not None:
                metrics.update({
                    'max_merge_distance': float(np.max(model.distances_)),
                    'min_merge_distance': float(np.min(model.distances_)),
                    'avg_merge_distance': float(np.mean(model.distances_))
                })
        
        return metrics
