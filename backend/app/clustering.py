"""
Clustering algorithms implementation for Interactive Spectral Clustering Platform.

This module provides comprehensive clustering algorithms with preprocessing,
evaluation metrics, and result formatting for the web application.
"""

import time
import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Tuple, Optional, Union
from dataclasses import dataclass
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.cluster import (
    KMeans, DBSCAN, AgglomerativeClustering, 
    SpectralClustering, MeanShift, OPTICS
)
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import (
    silhouette_score, calinski_harabasz_score, davies_bouldin_score,
    adjusted_rand_score, normalized_mutual_info_score
)
from sklearn.neighbors import NearestNeighbors
import umap.umap_ as umap

from .api_schemas import AlgorithmId, RunStatus, ParameterMap

logger = logging.getLogger(__name__)


@dataclass
class ClusteringResult:
    """Container for clustering algorithm results."""
    algorithm_name: str
    labels: np.ndarray
    n_clusters: int
    n_noise_points: int
    execution_time: float
    parameters_used: Dict[str, Any]
    cluster_centers: Optional[np.ndarray] = None
    silhouette_score: Optional[float] = None
    calinski_harabasz_score: Optional[float] = None
    davies_bouldin_score: Optional[float] = None
    inertia: Optional[float] = None
    additional_metrics: Optional[Dict[str, float]] = None
    error_message: Optional[str] = None
    status: str = "completed"


@dataclass
class EmbeddingResult:
    """Container for dimensionality reduction results."""
    method: str
    coordinates: np.ndarray
    n_components: int
    execution_time: float
    parameters_used: Dict[str, Any]
    explained_variance_ratio: Optional[np.ndarray] = None


class DataPreprocessor:
    """Data preprocessing utilities for clustering."""
    
    @staticmethod
    def load_data(file_path: str) -> pd.DataFrame:
        """Load data from various file formats."""
        try:
            if file_path.endswith('.csv'):
                return pd.read_csv(file_path)
            elif file_path.endswith(('.xls', '.xlsx')):
                return pd.read_excel(file_path)
            elif file_path.endswith('.json'):
                return pd.read_json(file_path)
            elif file_path.endswith('.parquet'):
                return pd.read_parquet(file_path)
            else:
                raise ValueError(f"Unsupported file format: {file_path}")
        except Exception as e:
            logger.error(f"Failed to load data from {file_path}: {str(e)}")
            raise
    
    @staticmethod
    def preprocess_data(
        data: pd.DataFrame,
        config: Dict[str, Any]
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Preprocess data according to configuration.
        
        Args:
            data: Input DataFrame
            config: Preprocessing configuration
            
        Returns:
            Preprocessed data array and metadata
        """
        try:
            metadata = {
                "original_shape": data.shape,
                "original_columns": list(data.columns),
                "preprocessing_steps": []
            }
            
            df = data.copy()
            
            # Handle missing values
            missing_strategy = config.get("missing_values", "drop")
            if missing_strategy == "drop":
                initial_rows = len(df)
                df = df.dropna()
                rows_dropped = initial_rows - len(df)
                metadata["preprocessing_steps"].append(f"Dropped {rows_dropped} rows with missing values")
            elif missing_strategy == "mean":
                numeric_cols = df.select_dtypes(include=[np.number]).columns
                df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())
                metadata["preprocessing_steps"].append("Filled missing values with mean")
            elif missing_strategy == "median":
                numeric_cols = df.select_dtypes(include=[np.number]).columns
                df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())
                metadata["preprocessing_steps"].append("Filled missing values with median")
            
            # Select numeric columns only
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) == 0:
                raise ValueError("No numeric columns found for clustering")
            
            df_numeric = df[numeric_cols]
            metadata["selected_columns"] = list(numeric_cols)
            metadata["preprocessing_steps"].append(f"Selected {len(numeric_cols)} numeric columns")
            
            # Handle outliers
            outlier_strategy = config.get("outliers", "keep")
            if outlier_strategy == "remove_iqr":
                initial_rows = len(df_numeric)
                Q1 = df_numeric.quantile(0.25)
                Q3 = df_numeric.quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                df_numeric = df_numeric[~((df_numeric < lower_bound) | (df_numeric > upper_bound)).any(axis=1)]
                rows_removed = initial_rows - len(df_numeric)
                metadata["preprocessing_steps"].append(f"Removed {rows_removed} outlier rows using IQR method")
            elif outlier_strategy == "clip":
                for col in df_numeric.columns:
                    q01, q99 = df_numeric[col].quantile([0.01, 0.99])
                    df_numeric[col] = df_numeric[col].clip(lower=q01, upper=q99)
                metadata["preprocessing_steps"].append("Clipped outliers to 1st-99th percentile range")
            
            # Feature scaling
            scaling_method = config.get("scaling", "standard")
            if scaling_method == "standard":
                scaler = StandardScaler()
                scaled_data = scaler.fit_transform(df_numeric)
                metadata["preprocessing_steps"].append("Applied standard scaling (mean=0, std=1)")
            elif scaling_method == "minmax":
                scaler = MinMaxScaler()
                scaled_data = scaler.fit_transform(df_numeric)
                metadata["preprocessing_steps"].append("Applied min-max scaling (range 0-1)")
            elif scaling_method == "robust":
                scaler = RobustScaler()
                scaled_data = scaler.fit_transform(df_numeric)
                metadata["preprocessing_steps"].append("Applied robust scaling (median=0, IQR=1)")
            else:
                scaled_data = df_numeric.values
                metadata["preprocessing_steps"].append("No scaling applied")
            
            metadata["final_shape"] = scaled_data.shape
            
            logger.info(f"Preprocessed data: {metadata['original_shape']} -> {metadata['final_shape']}")
            return scaled_data, metadata
            
        except Exception as e:
            logger.error(f"Data preprocessing failed: {str(e)}")
            raise


class ClusteringEngine:
    """Main clustering engine with multiple algorithms."""
    
    def __init__(self):
        self.preprocessor = DataPreprocessor()
    
    def run_clustering(
        self,
        data: np.ndarray,
        algorithms: List[str],
        algorithm_params: Dict[str, ParameterMap],
        max_clusters: Optional[int] = None
    ) -> List[ClusteringResult]:
        """
        Run multiple clustering algorithms on the data.
        
        Args:
            data: Preprocessed data array
            algorithms: List of algorithm names to run
            algorithm_params: Parameters for each algorithm
            max_clusters: Maximum number of clusters to consider
            
        Returns:
            List of clustering results
        """
        results = []
        
        # Determine reasonable cluster range
        n_samples = data.shape[0]
        if max_clusters is None:
            max_clusters = min(10, max(2, int(np.sqrt(n_samples))))
        
        for algorithm in algorithms:
            try:
                if algorithm == AlgorithmId.KMEANS.value:
                    result = self._run_kmeans(data, algorithm_params.get(algorithm, {}), max_clusters)
                elif algorithm == AlgorithmId.SPECTRAL.value:
                    result = self._run_spectral(data, algorithm_params.get(algorithm, {}), max_clusters)
                elif algorithm == AlgorithmId.DBSCAN.value:
                    result = self._run_dbscan(data, algorithm_params.get(algorithm, {}))
                elif algorithm == AlgorithmId.HIERARCHICAL.value:
                    result = self._run_hierarchical(data, algorithm_params.get(algorithm, {}), max_clusters)
                elif algorithm == AlgorithmId.GAUSSIAN_MIXTURE.value:
                    result = self._run_gaussian_mixture(data, algorithm_params.get(algorithm, {}), max_clusters)
                elif algorithm == AlgorithmId.MEAN_SHIFT.value:
                    result = self._run_mean_shift(data, algorithm_params.get(algorithm, {}))
                elif algorithm == AlgorithmId.OPTICS.value:
                    result = self._run_optics(data, algorithm_params.get(algorithm, {}))
                else:
                    logger.warning(f"Unknown algorithm: {algorithm}")
                    continue
                
                # Calculate evaluation metrics
                if result.status == "completed" and len(np.unique(result.labels)) > 1:
                    self._calculate_metrics(data, result)
                
                results.append(result)
                logger.info(f"Completed {algorithm} clustering: {result.n_clusters} clusters")
                
            except Exception as e:
                logger.error(f"Failed to run {algorithm}: {str(e)}")
                results.append(ClusteringResult(
                    algorithm_name=algorithm,
                    labels=np.array([]),
                    n_clusters=0,
                    n_noise_points=0,
                    execution_time=0.0,
                    parameters_used=algorithm_params.get(algorithm, {}),
                    error_message=str(e),
                    status="failed"
                ))
        
        return results
    
    def _run_kmeans(
        self, 
        data: np.ndarray, 
        params: ParameterMap, 
        max_clusters: int
    ) -> ClusteringResult:
        """Run K-means clustering with optimal k selection."""
        start_time = time.time()
        
        # Extract parameters
        k_range = params.get("k_range", [2, max_clusters])
        init_method = params.get("init", "k-means++")
        max_iter = params.get("max_iter", 300)
        n_init = params.get("n_init", 10)
        random_state = params.get("random_state", 42)
        
        best_k = 2
        best_score = -np.inf
        best_labels = None
        best_centers = None
        best_inertia = None
        
        # Try different k values
        for k in range(max(2, k_range[0]), min(k_range[1] + 1, len(data))):
            try:
                kmeans = KMeans(
                    n_clusters=k,
                    init=init_method,
                    max_iter=max_iter,
                    n_init=n_init,
                    random_state=random_state
                )
                labels = kmeans.fit_predict(data)
                
                # Calculate silhouette score for k selection
                if len(np.unique(labels)) > 1:
                    score = silhouette_score(data, labels)
                    if score > best_score:
                        best_score = score
                        best_k = k
                        best_labels = labels
                        best_centers = kmeans.cluster_centers_
                        best_inertia = kmeans.inertia_
                        
            except Exception as e:
                logger.warning(f"K-means failed for k={k}: {str(e)}")
                continue
        
        execution_time = time.time() - start_time
        
        if best_labels is None:
            raise ValueError("K-means failed for all k values")
        
        return ClusteringResult(
            algorithm_name="kmeans",
            labels=best_labels,
            n_clusters=best_k,
            n_noise_points=0,
            execution_time=execution_time,
            parameters_used={**params, "selected_k": best_k},
            cluster_centers=best_centers,
            inertia=best_inertia
        )
    
    def _run_spectral(
        self, 
        data: np.ndarray, 
        params: ParameterMap, 
        max_clusters: int
    ) -> ClusteringResult:
        """Run Spectral clustering with optimal k selection."""
        start_time = time.time()
        
        # Extract parameters
        k_range = params.get("k_range", [2, max_clusters])
        affinity = params.get("affinity", "rbf")
        gamma = params.get("gamma", 1.0)
        random_state = params.get("random_state", 42)
        
        best_k = 2
        best_score = -np.inf
        best_labels = None
        
        # Try different k values
        for k in range(max(2, k_range[0]), min(k_range[1] + 1, len(data))):
            try:
                spectral = SpectralClustering(
                    n_clusters=k,
                    affinity=affinity,
                    gamma=gamma,
                    random_state=random_state,
                    n_jobs=-1
                )
                labels = spectral.fit_predict(data)
                
                # Calculate silhouette score for k selection
                if len(np.unique(labels)) > 1:
                    score = silhouette_score(data, labels)
                    if score > best_score:
                        best_score = score
                        best_k = k
                        best_labels = labels
                        
            except Exception as e:
                logger.warning(f"Spectral clustering failed for k={k}: {str(e)}")
                continue
        
        execution_time = time.time() - start_time
        
        if best_labels is None:
            raise ValueError("Spectral clustering failed for all k values")
        
        return ClusteringResult(
            algorithm_name="spectral",
            labels=best_labels,
            n_clusters=best_k,
            n_noise_points=0,
            execution_time=execution_time,
            parameters_used={**params, "selected_k": best_k}
        )
    
    def _run_dbscan(self, data: np.ndarray, params: ParameterMap) -> ClusteringResult:
        """Run DBSCAN clustering."""
        start_time = time.time()
        
        # Extract parameters
        eps = params.get("eps", None)
        min_samples = params.get("min_samples", 5)
        metric = params.get("metric", "euclidean")
        
        # Auto-estimate eps if not provided
        if eps is None:
            # Use nearest neighbors to estimate eps
            k = min(min_samples, len(data) - 1)
            nbrs = NearestNeighbors(n_neighbors=k).fit(data)
            distances, _ = nbrs.kneighbors(data)
            eps = np.mean(distances[:, -1])  # Mean distance to k-th neighbor
        
        dbscan = DBSCAN(eps=eps, min_samples=min_samples, metric=metric, n_jobs=-1)
        labels = dbscan.fit_predict(data)
        
        execution_time = time.time() - start_time
        
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        n_noise_points = list(labels).count(-1)
        
        return ClusteringResult(
            algorithm_name="dbscan",
            labels=labels,
            n_clusters=n_clusters,
            n_noise_points=n_noise_points,
            execution_time=execution_time,
            parameters_used={"eps": eps, "min_samples": min_samples, "metric": metric}
        )
    
    def _run_hierarchical(
        self, 
        data: np.ndarray, 
        params: ParameterMap, 
        max_clusters: int
    ) -> ClusteringResult:
        """Run Agglomerative hierarchical clustering."""
        start_time = time.time()
        
        # Extract parameters
        k_range = params.get("k_range", [2, max_clusters])
        linkage = params.get("linkage", "ward")
        metric = params.get("metric", "euclidean")
        
        best_k = 2
        best_score = -np.inf
        best_labels = None
        
        # Try different k values
        for k in range(max(2, k_range[0]), min(k_range[1] + 1, len(data))):
            try:
                agg = AgglomerativeClustering(
                    n_clusters=k,
                    linkage=linkage,
                    metric=metric if linkage != "ward" else "euclidean"
                )
                labels = agg.fit_predict(data)
                
                # Calculate silhouette score for k selection
                if len(np.unique(labels)) > 1:
                    score = silhouette_score(data, labels)
                    if score > best_score:
                        best_score = score
                        best_k = k
                        best_labels = labels
                        
            except Exception as e:
                logger.warning(f"Hierarchical clustering failed for k={k}: {str(e)}")
                continue
        
        execution_time = time.time() - start_time
        
        if best_labels is None:
            raise ValueError("Hierarchical clustering failed for all k values")
        
        return ClusteringResult(
            algorithm_name="hierarchical",
            labels=best_labels,
            n_clusters=best_k,
            n_noise_points=0,
            execution_time=execution_time,
            parameters_used={**params, "selected_k": best_k}
        )
    
    def _run_gaussian_mixture(
        self, 
        data: np.ndarray, 
        params: ParameterMap, 
        max_clusters: int
    ) -> ClusteringResult:
        """Run Gaussian Mixture Model clustering."""
        start_time = time.time()
        
        # Extract parameters
        k_range = params.get("k_range", [2, max_clusters])
        covariance_type = params.get("covariance_type", "full")
        max_iter = params.get("max_iter", 100)
        random_state = params.get("random_state", 42)
        
        best_k = 2
        best_score = -np.inf
        best_labels = None
        
        # Try different k values
        for k in range(max(2, k_range[0]), min(k_range[1] + 1, len(data))):
            try:
                gmm = GaussianMixture(
                    n_components=k,
                    covariance_type=covariance_type,
                    max_iter=max_iter,
                    random_state=random_state
                )
                labels = gmm.fit_predict(data)
                
                # Use BIC for model selection (lower is better)
                bic = gmm.bic(data)
                score = -bic  # Negate to use higher is better logic
                
                if score > best_score:
                    best_score = score
                    best_k = k
                    best_labels = labels
                    
            except Exception as e:
                logger.warning(f"Gaussian Mixture failed for k={k}: {str(e)}")
                continue
        
        execution_time = time.time() - start_time
        
        if best_labels is None:
            raise ValueError("Gaussian Mixture failed for all k values")
        
        return ClusteringResult(
            algorithm_name="gaussian_mixture",
            labels=best_labels,
            n_clusters=best_k,
            n_noise_points=0,
            execution_time=execution_time,
            parameters_used={**params, "selected_k": best_k}
        )
    
    def _run_mean_shift(self, data: np.ndarray, params: ParameterMap) -> ClusteringResult:
        """Run Mean Shift clustering."""
        start_time = time.time()
        
        # Extract parameters
        bandwidth = params.get("bandwidth", None)
        
        # Auto-estimate bandwidth if not provided
        if bandwidth is None:
            from sklearn.cluster import estimate_bandwidth
            bandwidth = estimate_bandwidth(data, quantile=0.2, n_samples=min(500, len(data)))
        
        mean_shift = MeanShift(bandwidth=bandwidth, n_jobs=-1)
        labels = mean_shift.fit_predict(data)
        
        execution_time = time.time() - start_time
        
        n_clusters = len(np.unique(labels))
        
        return ClusteringResult(
            algorithm_name="mean_shift",
            labels=labels,
            n_clusters=n_clusters,
            n_noise_points=0,
            execution_time=execution_time,
            parameters_used={"bandwidth": bandwidth},
            cluster_centers=mean_shift.cluster_centers_
        )
    
    def _run_optics(self, data: np.ndarray, params: ParameterMap) -> ClusteringResult:
        """Run OPTICS clustering."""
        start_time = time.time()
        
        # Extract parameters
        min_samples = params.get("min_samples", 5)
        max_eps = params.get("max_eps", np.inf)
        metric = params.get("metric", "euclidean")
        cluster_method = params.get("cluster_method", "xi")
        
        optics = OPTICS(
            min_samples=min_samples,
            max_eps=max_eps,
            metric=metric,
            cluster_method=cluster_method,
            n_jobs=-1
        )
        labels = optics.fit_predict(data)
        
        execution_time = time.time() - start_time
        
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        n_noise_points = list(labels).count(-1)
        
        return ClusteringResult(
            algorithm_name="optics",
            labels=labels,
            n_clusters=n_clusters,
            n_noise_points=n_noise_points,
            execution_time=execution_time,
            parameters_used={
                "min_samples": min_samples,
                "max_eps": max_eps,
                "metric": metric,
                "cluster_method": cluster_method
            }
        )
    
    def _calculate_metrics(self, data: np.ndarray, result: ClusteringResult) -> None:
        """Calculate clustering evaluation metrics."""
        try:
            labels = result.labels
            unique_labels = np.unique(labels)
            
            # Skip single cluster results
            if len(unique_labels) <= 1:
                return
            
            # Handle noise points for metrics calculation
            if -1 in unique_labels:
                # For metrics, exclude noise points
                mask = labels != -1
                if np.sum(mask) < 2:
                    return
                clean_data = data[mask]
                clean_labels = labels[mask]
            else:
                clean_data = data
                clean_labels = labels
            
            if len(np.unique(clean_labels)) <= 1:
                return
            
            # Calculate metrics
            result.silhouette_score = silhouette_score(clean_data, clean_labels)
            result.calinski_harabasz_score = calinski_harabasz_score(clean_data, clean_labels)
            result.davies_bouldin_score = davies_bouldin_score(clean_data, clean_labels)
            
        except Exception as e:
            logger.warning(f"Failed to calculate metrics for {result.algorithm_name}: {str(e)}")


class EmbeddingEngine:
    """Dimensionality reduction engine for visualization."""
    
    def generate_embedding(
        self,
        data: np.ndarray,
        method: str,
        n_components: int = 2,
        params: Optional[Dict[str, Any]] = None
    ) -> EmbeddingResult:
        """
        Generate dimensionality reduction embedding.
        
        Args:
            data: Input data array
            method: Embedding method (pca, tsne, umap)
            n_components: Number of dimensions for output
            params: Method-specific parameters
            
        Returns:
            Embedding result
        """
        if params is None:
            params = {}
        
        start_time = time.time()
        
        try:
            if method == "pca":
                result = self._generate_pca(data, n_components, params)
            elif method == "tsne":
                result = self._generate_tsne(data, n_components, params)
            elif method == "umap":
                result = self._generate_umap(data, n_components, params)
            else:
                raise ValueError(f"Unknown embedding method: {method}")
            
            result.execution_time = time.time() - start_time
            logger.info(f"Generated {method} embedding: {data.shape} -> {result.coordinates.shape}")
            return result
            
        except Exception as e:
            logger.error(f"Failed to generate {method} embedding: {str(e)}")
            raise
    
    def _generate_pca(
        self, 
        data: np.ndarray, 
        n_components: int, 
        params: Dict[str, Any]
    ) -> EmbeddingResult:
        """Generate PCA embedding."""
        random_state = params.get("random_state", 42)
        
        pca = PCA(n_components=n_components, random_state=random_state)
        coordinates = pca.fit_transform(data)
        
        return EmbeddingResult(
            method="pca",
            coordinates=coordinates,
            n_components=n_components,
            execution_time=0.0,  # Will be set by caller
            parameters_used=params,
            explained_variance_ratio=pca.explained_variance_ratio_
        )
    
    def _generate_tsne(
        self, 
        data: np.ndarray, 
        n_components: int, 
        params: Dict[str, Any]
    ) -> EmbeddingResult:
        """Generate t-SNE embedding."""
        perplexity = params.get("perplexity", min(30, max(5, len(data) // 4)))
        learning_rate = params.get("learning_rate", "auto")
        n_iter = params.get("n_iter", 1000)
        random_state = params.get("random_state", 42)
        
        # Limit data size for t-SNE performance
        if len(data) > 5000:
            logger.warning(f"Large dataset ({len(data)} samples) - consider using PCA first")
        
        tsne = TSNE(
            n_components=n_components,
            perplexity=perplexity,
            learning_rate=learning_rate,
            n_iter=n_iter,
            random_state=random_state,
            n_jobs=-1
        )
        coordinates = tsne.fit_transform(data)
        
        return EmbeddingResult(
            method="tsne",
            coordinates=coordinates,
            n_components=n_components,
            execution_time=0.0,  # Will be set by caller
            parameters_used={
                "perplexity": perplexity,
                "learning_rate": learning_rate,
                "n_iter": n_iter,
                "random_state": random_state
            }
        )
    
    def _generate_umap(
        self, 
        data: np.ndarray, 
        n_components: int, 
        params: Dict[str, Any]
    ) -> EmbeddingResult:
        """Generate UMAP embedding."""
        n_neighbors = params.get("n_neighbors", 15)
        min_dist = params.get("min_dist", 0.1)
        metric = params.get("metric", "euclidean")
        random_state = params.get("random_state", 42)
        
        mapper = umap.UMAP(
            n_components=n_components,
            n_neighbors=n_neighbors,
            min_dist=min_dist,
            metric=metric,
            random_state=random_state
        )
        coordinates = mapper.fit_transform(data)
        
        return EmbeddingResult(
            method="umap",
            coordinates=coordinates,
            n_components=n_components,
            execution_time=0.0,  # Will be set by caller
            parameters_used={
                "n_neighbors": n_neighbors,
                "min_dist": min_dist,
                "metric": metric,
                "random_state": random_state
            }
        )


# Factory functions
def create_clustering_engine() -> ClusteringEngine:
    """Create clustering engine instance."""
    return ClusteringEngine()


def create_embedding_engine() -> EmbeddingEngine:
    """Create embedding engine instance."""
    return EmbeddingEngine()
