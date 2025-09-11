"""
Spectral Clustering algorithm implementation with GPU acceleration support.

This module provides a sophisticated spectral clustering implementation with
k-NN graph construction, normalized Laplacian computation, and eigensolver
optimizations. Supports both GPU (CuPy/PyTorch) and CPU (scikit-learn) execution.
"""

import numpy as np
from typing import Dict, Any
from .factory import BaseClusteringAlgorithm, AlgorithmType
import logging

logger = logging.getLogger(__name__)

class SpectralClustering(BaseClusteringAlgorithm):
    """
    Spectral clustering using normalized Laplacian with k-NN graph.
    
    Supports both GPU and CPU execution with automatic fallback.
    Uses RBF (Gaussian) similarity with adaptive sigma parameter.
    """
    
    def get_algorithm_type(self) -> AlgorithmType:
        return AlgorithmType.SPECTRAL
    
    def get_default_parameters(self) -> Dict[str, Any]:
        return {
            'n_clusters': 3,
            'sigma': 1.0,
            'n_neighbors': 10,
            'affinity': 'rbf',  # 'rbf', 'nearest_neighbors', 'polynomial'
            'eigen_solver': 'auto',  # 'auto', 'arpack', 'lobpcg'
            'assign_labels': 'kmeans',  # 'kmeans', 'discretize'
            'random_state': 42,
            'gamma': None,  # RBF kernel parameter (auto-computed if None)
            'n_components': None,  # Number of eigenvectors (defaults to n_clusters)
        }
    
    def validate_parameters(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Validate spectral clustering parameters."""
        validated = params.copy()
        
        # Validate n_clusters
        if validated.get('n_clusters', 1) < 2:
            validated['n_clusters'] = 2
        if validated.get('n_clusters', 1) > 50:
            validated['n_clusters'] = 50
        
        # Validate sigma
        if validated.get('sigma', 1.0) <= 0:
            validated['sigma'] = 1.0
        
        # Validate n_neighbors
        if validated.get('n_neighbors', 10) < 1:
            validated['n_neighbors'] = 10
        
        # Auto-compute gamma from sigma
        if validated.get('gamma') is None:
            validated['gamma'] = 1.0 / (2.0 * validated['sigma'] ** 2)
        
        # Set n_components to n_clusters if not specified
        if validated.get('n_components') is None:
            validated['n_components'] = validated['n_clusters']
        
        return validated
    
    def _fit_predict(self, X: np.ndarray, params: Dict[str, Any]) -> np.ndarray:
        """Core spectral clustering implementation."""
        if self.use_gpu:
            return self._fit_predict_gpu(X, params)
        else:
            return self._fit_predict_cpu(X, params)
    
    def _fit_predict_gpu(self, X: np.ndarray, params: Dict[str, Any]) -> np.ndarray:
        """GPU-accelerated spectral clustering using PyTorch/CuPy."""
        try:
            import torch
            if not torch.cuda.is_available():
                self.logger.warning("GPU requested but not available, falling back to CPU")
                return self._fit_predict_cpu(X, params)
            
            self.logger.info("Using GPU acceleration for spectral clustering")
            
            # Convert to PyTorch tensors on GPU
            X_gpu = torch.tensor(X, dtype=torch.float32).cuda()
            
            # Compute affinity matrix on GPU
            affinity = self._compute_affinity_gpu(X_gpu, params)
            
            # Compute normalized Laplacian on GPU
            laplacian = self._compute_laplacian_gpu(affinity)
            
            # Eigendecomposition (move to CPU for scipy/sklearn)
            laplacian_cpu = laplacian.cpu().numpy()
            
            # Compute eigenvectors
            embeddings = self._compute_embeddings(laplacian_cpu, params)
            
            # K-means on embeddings
            labels = self._assign_labels(embeddings, params)
            
            return labels
            
        except Exception as e:
            self.logger.warning(f"GPU spectral clustering failed: {str(e)}, falling back to CPU")
            return self._fit_predict_cpu(X, params)
    
    def _fit_predict_cpu(self, X: np.ndarray, params: Dict[str, Any]) -> np.ndarray:
        """CPU spectral clustering using scikit-learn."""
        from sklearn.cluster import SpectralClustering as SklearnSpectral
        
        self.logger.info("Using CPU spectral clustering")
        
        try:
            # Use scikit-learn's spectral clustering
            spectral = SklearnSpectral(
                n_clusters=params['n_clusters'],
                affinity=params['affinity'],
                gamma=params['gamma'],
                n_neighbors=params['n_neighbors'],
                eigen_solver=params['eigen_solver'],
                assign_labels=params['assign_labels'],
                random_state=params['random_state'],
                n_components=params['n_components']
            )
            
            labels = spectral.fit_predict(X)
            return labels
            
        except Exception as e:
            self.logger.error(f"CPU spectral clustering failed: {str(e)}")
            # Fallback to simple k-means if spectral clustering fails
            return self._fallback_kmeans(X, params)
    
    def _compute_affinity_gpu(self, X_gpu, params: Dict[str, Any]):
        """Compute RBF affinity matrix on GPU."""
        import torch
        
        gamma = params['gamma']
        n_neighbors = params['n_neighbors']
        
        # Compute pairwise squared distances
        X_norm = torch.sum(X_gpu**2, dim=1, keepdim=True)
        distances_sq = X_norm + X_norm.T - 2 * torch.mm(X_gpu, X_gpu.T)
        
        # RBF kernel
        affinity = torch.exp(-gamma * distances_sq)
        
        # Apply k-NN sparsification
        if n_neighbors < X_gpu.shape[0]:
            # Get k-nearest neighbors for each point
            _, indices = torch.topk(affinity, k=n_neighbors, dim=1, largest=True)
            
            # Create sparse affinity matrix
            sparse_affinity = torch.zeros_like(affinity)
            for i in range(X_gpu.shape[0]):
                sparse_affinity[i, indices[i]] = affinity[i, indices[i]]
            
            # Make symmetric
            affinity = (sparse_affinity + sparse_affinity.T) / 2
        
        return affinity
    
    def _compute_laplacian_gpu(self, affinity):
        """Compute normalized Laplacian on GPU."""
        import torch
        
        # Degree matrix
        degree = torch.sum(affinity, dim=1)
        
        # Avoid division by zero
        degree_inv_sqrt = torch.pow(degree + 1e-12, -0.5)
        degree_inv_sqrt = torch.diag(degree_inv_sqrt)
        
        # Normalized Laplacian: L = I - D^(-1/2) A D^(-1/2)
        identity = torch.eye(affinity.shape[0], device=affinity.device)
        laplacian = identity - torch.mm(torch.mm(degree_inv_sqrt, affinity), degree_inv_sqrt)
        
        return laplacian
    
    def _compute_embeddings(self, laplacian: np.ndarray, params: Dict[str, Any]) -> np.ndarray:
        """Compute embedding using eigendecomposition."""
        from scipy.sparse.linalg import eigsh
        from scipy.linalg import eigh
        
        n_components = params['n_components']
        eigen_solver = params['eigen_solver']
        
        try:
            if eigen_solver == 'auto':
                # Choose solver based on matrix size
                if laplacian.shape[0] < 500:
                    eigen_solver = 'dense'
                else:
                    eigen_solver = 'arpack'
            
            if eigen_solver == 'dense':
                # Use dense solver for small matrices
                eigenvalues, eigenvectors = eigh(laplacian)
                # Sort by eigenvalue (ascending)
                idx = np.argsort(eigenvalues)
                embeddings = eigenvectors[:, idx[:n_components]]
            else:
                # Use sparse solver for large matrices
                eigenvalues, eigenvectors = eigsh(
                    laplacian, 
                    k=n_components, 
                    which='SM',  # Smallest eigenvalues
                    sigma=0.0
                )
                embeddings = eigenvectors
            
            # Normalize embeddings
            from sklearn.preprocessing import normalize
            embeddings = normalize(embeddings, norm='l2', axis=1)
            
            return embeddings
            
        except Exception as e:
            self.logger.error(f"Eigendecomposition failed: {str(e)}")
            raise
    
    def _assign_labels(self, embeddings: np.ndarray, params: Dict[str, Any]) -> np.ndarray:
        """Assign cluster labels from embeddings."""
        assign_labels = params['assign_labels']
        n_clusters = params['n_clusters']
        random_state = params['random_state']
        
        if assign_labels == 'kmeans':
            from sklearn.cluster import KMeans
            kmeans = KMeans(
                n_clusters=n_clusters, 
                random_state=random_state,
                n_init=10
            )
            labels = kmeans.fit_predict(embeddings)
        elif assign_labels == 'discretize':
            # Discretization method (simple but fast)
            labels = self._discretize_embeddings(embeddings, n_clusters)
        else:
            raise ValueError(f"Unknown assign_labels method: {assign_labels}")
        
        return labels
    
    def _discretize_embeddings(self, embeddings: np.ndarray, n_clusters: int) -> np.ndarray:
        """Simple discretization of embeddings to cluster labels."""
        # Find the eigenvector with largest variation
        variations = np.var(embeddings, axis=0)
        best_eigenvector = embeddings[:, np.argmax(variations)]
        
        # Simple thresholding for binary clustering, extend for multi-class
        if n_clusters == 2:
            labels = (best_eigenvector > np.median(best_eigenvector)).astype(int)
        else:
            # Use quantiles for multi-class
            quantiles = np.linspace(0, 1, n_clusters + 1)[1:-1]
            thresholds = np.quantile(best_eigenvector, quantiles)
            labels = np.digitize(best_eigenvector, thresholds)
        
        return labels
    
    def _fallback_kmeans(self, X: np.ndarray, params: Dict[str, Any]) -> np.ndarray:
        """Fallback to k-means if spectral clustering fails."""
        from sklearn.cluster import KMeans
        
        self.logger.warning("Using k-means fallback")
        
        kmeans = KMeans(
            n_clusters=params['n_clusters'],
            random_state=params['random_state'],
            n_init=10
        )
        return kmeans.fit_predict(X)
