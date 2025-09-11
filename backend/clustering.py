import numpy as np
import torch
from typing import Dict, List, Any, Tuple, Union, Optional
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering, SpectralClustering
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import asyncio
import warnings

# Import graph utilities
from graph_utils import compute_eigendecomposition, spectral_embedding
from app import manager  # Import WebSocket manager for progress updates

# Import new factory pattern
from app.services.clustering.factory import clustering_factory, AlgorithmType

# Try to import UMAP
try:
    import umap
    UMAP_AVAILABLE = True
except ImportError:
    UMAP_AVAILABLE = False

class ClusteringMethod:
    """Base class for clustering methods"""
    
    def __init__(self, name: str):
        self.name = name
    
    async def fit_predict(self, X: np.ndarray, **kwargs) -> np.ndarray:
        """Fit clustering algorithm and return labels"""
        raise NotImplementedError
    
    def get_default_params(self) -> Dict[str, Any]:
        """Get default parameters for this method"""
        return {}

class ManualSpectralClustering(ClusteringMethod):
    """Manual implementation of spectral clustering using eigendecomposition"""
    
    def __init__(self):
        super().__init__("manual_spectral")
    
    async def fit_predict(self, X: np.ndarray, L: Optional[np.ndarray] = None, 
                         n_clusters: int = 3, **kwargs) -> np.ndarray:
        """
        Manual spectral clustering implementation
        
        Args:
            X: Data matrix (for fallback if L not provided)
            L: Precomputed Laplacian matrix
            n_clusters: Number of clusters
        
        Returns:
            Cluster labels
        """
        if L is None:
            # Fallback: compute spectral embedding from data
            from graph_utils import spectral_embedding
            embedding = spectral_embedding(X, n_components=n_clusters)
        else:
            # Use precomputed Laplacian
            eigenvals, eigenvecs = compute_eigendecomposition(L, k=n_clusters + 1, which='SM')
            
            # Skip the first eigenvector (constant for connected graphs)
            if isinstance(eigenvecs, torch.Tensor):
                embedding = eigenvecs[:, 1:n_clusters + 1].cpu().numpy()
            else:
                embedding = eigenvecs[:, 1:n_clusters + 1]
        
        # Apply K-means to the embedding
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        labels = kmeans.fit_predict(embedding)
        
        return labels
    
    def get_default_params(self) -> Dict[str, Any]:
        return {
            'n_clusters': 3,
            'affinity_method': 'rbf',
            'sigma': 1.0,
            'n_neighbors': 10
        }

class SklearnSpectralClustering(ClusteringMethod):
    """Scikit-learn SpectralClustering wrapper"""
    
    def __init__(self):
        super().__init__("sklearn_spectral")
    
    async def fit_predict(self, X: np.ndarray, n_clusters: int = 3, 
                         affinity: str = 'rbf', gamma: float = 1.0, 
                         n_neighbors: int = 10, **kwargs) -> np.ndarray:
        """
        Scikit-learn spectral clustering
        
        Args:
            X: Data matrix
            n_clusters: Number of clusters
            affinity: Affinity method ('rbf', 'nearest_neighbors')
            gamma: Kernel coefficient for rbf
            n_neighbors: Number of neighbors for knn affinity
        
        Returns:
            Cluster labels
        """
        # Convert affinity parameters
        if affinity == 'knn':
            affinity = 'nearest_neighbors'
        
        spectral = SpectralClustering(
            n_clusters=n_clusters,
            affinity=affinity,
            gamma=gamma,
            n_neighbors=n_neighbors,
            random_state=42
        )
        
        labels = spectral.fit_predict(X)
        return labels
    
    def get_default_params(self) -> Dict[str, Any]:
        return {
            'n_clusters': 3,
            'affinity': 'rbf',
            'gamma': 1.0,
            'n_neighbors': 10
        }

class KMeansClustering(ClusteringMethod):
    """K-Means clustering wrapper"""
    
    def __init__(self):
        super().__init__("kmeans")
    
    async def fit_predict(self, X: np.ndarray, n_clusters: int = 3, 
                         init: str = 'k-means++', max_iter: int = 300,
                         **kwargs) -> np.ndarray:
        """
        K-Means clustering
        
        Args:
            X: Data matrix
            n_clusters: Number of clusters
            init: Initialization method
            max_iter: Maximum iterations
        
        Returns:
            Cluster labels
        """
        kmeans = KMeans(
            n_clusters=n_clusters,
            init=init,
            max_iter=max_iter,
            random_state=42,
            n_init=10
        )
        
        labels = kmeans.fit_predict(X)
        return labels
    
    def get_default_params(self) -> Dict[str, Any]:
        return {
            'n_clusters': 3,
            'init': 'k-means++',
            'max_iter': 300
        }

class DBSCANClustering(ClusteringMethod):
    """DBSCAN clustering wrapper"""
    
    def __init__(self):
        super().__init__("dbscan")
    
    async def fit_predict(self, X: np.ndarray, eps: float = 0.5, 
                         min_samples: int = 5, **kwargs) -> np.ndarray:
        """
        DBSCAN clustering
        
        Args:
            X: Data matrix
            eps: Maximum distance between samples in a neighborhood
            min_samples: Minimum samples in a neighborhood for core point
        
        Returns:
            Cluster labels (noise points labeled as -1)
        """
        dbscan = DBSCAN(eps=eps, min_samples=min_samples)
        labels = dbscan.fit_predict(X)
        return labels
    
    def get_default_params(self) -> Dict[str, Any]:
        return {
            'eps': 0.5,
            'min_samples': 5
        }

class AgglomerativeClustering(ClusteringMethod):
    """Agglomerative clustering wrapper"""
    
    def __init__(self):
        super().__init__("agglomerative")
    
    async def fit_predict(self, X: np.ndarray, n_clusters: int = 3,
                         linkage: str = 'ward', **kwargs) -> np.ndarray:
        """
        Agglomerative clustering
        
        Args:
            X: Data matrix
            n_clusters: Number of clusters
            linkage: Linkage criterion
        
        Returns:
            Cluster labels
        """
        agglomerative = AgglomerativeClustering(
            n_clusters=n_clusters,
            linkage=linkage
        )
        
        labels = agglomerative.fit_predict(X)
        return labels
    
    def get_default_params(self) -> Dict[str, Any]:
        return {
            'n_clusters': 3,
            'linkage': 'ward'
        }

class GaussianMixtureClustering(ClusteringMethod):
    """Gaussian Mixture Model clustering wrapper"""
    
    def __init__(self):
        super().__init__("gmm")
    
    async def fit_predict(self, X: np.ndarray, n_components: int = 3,
                         covariance_type: str = 'full', max_iter: int = 100,
                         **kwargs) -> np.ndarray:
        """
        Gaussian Mixture Model clustering
        
        Args:
            X: Data matrix
            n_components: Number of mixture components (clusters)
            covariance_type: Type of covariance parameters ('full', 'tied', 'diag', 'spherical')
            max_iter: Maximum number of EM iterations
        
        Returns:
            Cluster labels
        """
        gmm = GaussianMixture(
            n_components=n_components,
            covariance_type=covariance_type,
            max_iter=max_iter,
            random_state=42
        )
        
        labels = gmm.fit_predict(X)
        return labels
    
    def get_default_params(self) -> Dict[str, Any]:
        return {
            'n_components': 3,
            'covariance_type': 'full',
            'max_iter': 100
        }

# Registry of available clustering methods
CLUSTERING_METHODS = {
    'manual_spectral': ManualSpectralClustering(),
    'sklearn_spectral': SklearnSpectralClustering(),
    'kmeans': KMeansClustering(),
    'dbscan': DBSCANClustering(),
    'agglomerative': AgglomerativeClustering(),
    'gmm': GaussianMixtureClustering()
}

def get_available_methods() -> List[str]:
    """Get list of available clustering method names (legacy)"""
    return list(CLUSTERING_METHODS.keys())

def get_available_advanced_algorithms() -> List[str]:
    """Get list of available advanced clustering algorithms"""
    return clustering_factory.get_available_algorithms()

def get_all_available_methods() -> Dict[str, List[str]]:
    """Get all available clustering methods grouped by type"""
    return {
        'legacy': get_available_methods(),
        'advanced': get_available_advanced_algorithms()
    }

def get_method_params(method_name: str) -> Dict[str, Any]:
    """Get default parameters for a clustering method"""
    if method_name not in CLUSTERING_METHODS:
        raise ValueError(f"Unknown clustering method: {method_name}")
    
    return CLUSTERING_METHODS[method_name].get_default_params()

def generate_visualization_coordinates(X: np.ndarray, 
                                     method: str = 'pca',
                                     n_components_2d: int = 2,
                                     n_components_3d: int = 3) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate 2D and 3D coordinates for visualization
    
    Args:
        X: Data matrix
        method: Dimensionality reduction method ('pca', 'tsne', 'umap')
        n_components_2d: Number of components for 2D visualization
        n_components_3d: Number of components for 3D visualization
    
    Returns:
        Tuple of (coords_2d, coords_3d)
    """
    # Reduce dimensionality if data is high-dimensional
    X_viz = X
    if X.shape[1] > 50:
        pca_pre = PCA(n_components=min(50, X.shape[0] - 1))
        X_viz = pca_pre.fit_transform(X)
    
    if method.lower() == 'pca':
        # PCA for both 2D and 3D
        pca_2d = PCA(n_components=n_components_2d)
        coords_2d = pca_2d.fit_transform(X_viz)
        
        pca_3d = PCA(n_components=n_components_3d)
        coords_3d = pca_3d.fit_transform(X_viz)
        
    elif method.lower() == 'tsne':
        # t-SNE for visualization
        tsne_2d = TSNE(n_components=n_components_2d, random_state=42, 
                       perplexity=min(30, X_viz.shape[0] - 1))
        coords_2d = tsne_2d.fit_transform(X_viz)
        
        tsne_3d = TSNE(n_components=n_components_3d, random_state=42,
                       perplexity=min(30, X_viz.shape[0] - 1))
        coords_3d = tsne_3d.fit_transform(X_viz)
        
    elif method.lower() == 'umap':
        if not UMAP_AVAILABLE:
            warnings.warn("UMAP not available, falling back to PCA")
            return generate_visualization_coordinates(X, 'pca', n_components_2d, n_components_3d)
        
        # UMAP for visualization
        umap_2d = umap.UMAP(n_components=n_components_2d, random_state=42)
        coords_2d = umap_2d.fit_transform(X_viz)
        
        umap_3d = umap.UMAP(n_components=n_components_3d, random_state=42)
        coords_3d = umap_3d.fit_transform(X_viz)
    
    else:
        raise ValueError(f"Unknown visualization method: {method}")
    
    return coords_2d, coords_3d

async def run_methods(X: np.ndarray, 
                     L: Optional[np.ndarray],
                     methods: List[str],
                     params: Dict[str, Any],
                     job_id: str,
                     visualization_method: str = 'pca') -> Dict[str, Any]:
    """
    Run multiple clustering methods and return results
    
    Args:
        X: Data matrix
        L: Precomputed Laplacian matrix (optional)
        methods: List of clustering method names
        params: Dictionary of parameters for each method
        job_id: Job ID for progress updates
        visualization_method: Method for generating visualization coordinates
    
    Returns:
        Dictionary containing:
        - labels: Dict mapping method names to cluster labels
        - coords2D: 2D coordinates for visualization
        - coords3D: 3D coordinates for visualization
        - method_info: Information about each method run
    """
    
    results = {
        'labels': {},
        'coords2D': [],
        'coords3D': [],
        'method_info': {}
    }
    
    # Generate visualization coordinates
    await emit_progress(job_id, 10, "Generating visualization coordinates...")
    try:
        coords_2d, coords_3d = generate_visualization_coordinates(
            X, method=visualization_method
        )
        results['coords2D'] = coords_2d.tolist()
        results['coords3D'] = coords_3d.tolist()
    except Exception as e:
        warnings.warn(f"Visualization generation failed: {e}, using PCA fallback")
        coords_2d, coords_3d = generate_visualization_coordinates(X, method='pca')
        results['coords2D'] = coords_2d.tolist()
        results['coords3D'] = coords_3d.tolist()
    
    # Run each clustering method
    total_methods = len(methods)
    for i, method_name in enumerate(methods):
        try:
            progress = 20 + int((i / total_methods) * 60)  # Progress from 20% to 80%
            await emit_progress(job_id, progress, f"Running {method_name} clustering...")
            
            if method_name not in CLUSTERING_METHODS:
                results['method_info'][method_name] = {'error': f'Unknown method: {method_name}'}
                continue
            
            method = CLUSTERING_METHODS[method_name]
            
            # Get method-specific parameters
            method_params = params.get(method_name, {})
            
            # Merge with default parameters
            default_params = method.get_default_params()
            default_params.update(method_params)
            
            # Run clustering
            if method_name == 'manual_spectral':
                labels = await method.fit_predict(X, L=L, **default_params)
            else:
                labels = await method.fit_predict(X, **default_params)
            
            results['labels'][method_name] = labels.tolist()
            results['method_info'][method_name] = {
                'success': True,
                'n_clusters': len(np.unique(labels)),
                'parameters': default_params
            }
            
            await emit_progress(job_id, progress + 5, f"{method_name} completed successfully")
            
        except Exception as e:
            error_msg = f"Error in {method_name}: {str(e)}"
            results['method_info'][method_name] = {'error': error_msg}
            await emit_progress(job_id, progress, error_msg)
            continue
    
    await emit_progress(job_id, 85, "All clustering methods completed")
    return results

async def emit_progress(job_id: str, progress: int, message: str):
    """Helper function to emit progress updates"""
    try:
        await manager.send_progress(job_id, progress, message)
    except Exception as e:
        print(f"Failed to send progress for job {job_id}: {e}")

async def run_advanced_clustering(
    X: np.ndarray, 
    algorithms: List[str], 
    params: Dict[str, Dict[str, Any]], 
    job_id: str,
    use_gpu: bool = True
) -> Dict[str, Any]:
    """
    Run advanced clustering algorithms using the factory pattern.
    
    Args:
        X: Data matrix
        algorithms: List of algorithm names to run
        params: Parameters for each algorithm
        job_id: Job ID for progress tracking
        use_gpu: Whether to use GPU acceleration
    
    Returns:
        Dictionary with clustering results and metrics
    """
    results = {
        'labels': {},
        'metrics': {},
        'algorithm_info': {}
    }
    
    available_algorithms = clustering_factory.get_available_algorithms()
    valid_algorithms = [alg for alg in algorithms if alg in available_algorithms]
    
    await emit_progress(job_id, 10, f"Starting advanced clustering with {len(valid_algorithms)} algorithms")
    
    for i, algorithm_name in enumerate(valid_algorithms):
        try:
            progress = 20 + (i * 60 // len(valid_algorithms))
            await emit_progress(job_id, progress, f"Running {algorithm_name} clustering")
            
            # Get algorithm instance
            algorithm = clustering_factory.create(algorithm_name, use_gpu=use_gpu)
            
            # Get algorithm-specific parameters
            algorithm_params = params.get(algorithm_name, {})
            
            # Run clustering
            clustering_result = await asyncio.to_thread(
                algorithm.fit_predict, X, algorithm_params
            )
            
            # Store results
            results['labels'][algorithm_name] = clustering_result.labels.tolist()
            results['metrics'][algorithm_name] = clustering_result.metrics
            results['algorithm_info'][algorithm_name] = {
                'success': True,
                'n_clusters': len(np.unique(clustering_result.labels)),
                'parameters': clustering_result.parameters,
                'gpu_used': clustering_result.gpu_used,
                'execution_time': clustering_result.execution_time
            }
            
            await emit_progress(job_id, progress + 10, f"{algorithm_name} completed successfully")
            
        except Exception as e:
            error_msg = f"Error in {algorithm_name}: {str(e)}"
            results['algorithm_info'][algorithm_name] = {'error': error_msg}
            await emit_progress(job_id, progress, error_msg)
            continue
    
    await emit_progress(job_id, 85, "Advanced clustering algorithms completed")
    return results

def validate_clustering_params(method_name: str, params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Validate and sanitize clustering parameters
    
    Args:
        method_name: Name of clustering method
        params: Parameters to validate
    
    Returns:
        Validated parameters dictionary
    """
    if method_name not in CLUSTERING_METHODS:
        raise ValueError(f"Unknown clustering method: {method_name}")
    
    validated_params = {}
    default_params = CLUSTERING_METHODS[method_name].get_default_params()
    
    for key, default_value in default_params.items():
        if key in params:
            param_value = params[key]
            
            # Type validation
            if isinstance(default_value, int):
                try:
                    validated_params[key] = int(param_value)
                except (ValueError, TypeError):
                    validated_params[key] = default_value
            elif isinstance(default_value, float):
                try:
                    validated_params[key] = float(param_value)
                except (ValueError, TypeError):
                    validated_params[key] = default_value
            elif isinstance(default_value, str):
                validated_params[key] = str(param_value)
            else:
                validated_params[key] = param_value
        else:
            validated_params[key] = default_value
    
    # Method-specific validation
    if method_name in ['kmeans', 'manual_spectral', 'sklearn_spectral', 'agglomerative']:
        validated_params['n_clusters'] = max(1, validated_params.get('n_clusters', 2))
    
    if method_name == 'gmm':
        validated_params['n_components'] = max(1, validated_params.get('n_components', 2))
        validated_params['max_iter'] = max(1, validated_params.get('max_iter', 100))
        # Validate covariance_type
        valid_cov_types = ['full', 'tied', 'diag', 'spherical']
        if validated_params.get('covariance_type') not in valid_cov_types:
            validated_params['covariance_type'] = 'full'
    
    if method_name == 'dbscan':
        validated_params['eps'] = max(0.001, validated_params.get('eps', 0.5))
        validated_params['min_samples'] = max(1, validated_params.get('min_samples', 5))
    
    return validated_params
