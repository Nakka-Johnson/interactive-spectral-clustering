import numpy as np
from typing import Dict, List, Any, Tuple
from sklearn.metrics import (
    silhouette_score, 
    davies_bouldin_score, 
    calinski_harabasz_score,
    adjusted_rand_score,
    normalized_mutual_info_score,
    homogeneity_score,
    completeness_score,
    v_measure_score
)
from sklearn.datasets import make_blobs
import warnings

def compute_metrics(X: np.ndarray, labels_dict: Dict[str, List[int]]) -> Dict[str, Dict[str, float]]:
    """
    Compute clustering evaluation metrics for multiple methods
    
    Args:
        X: Data matrix (n_samples, n_features)
        labels_dict: Dictionary mapping method names to cluster labels
    
    Returns:
        Dictionary of metrics for each clustering method
        {
            'method_name': {
                'silhouette_score': float,
                'davies_bouldin_score': float,
                'calinski_harabasz_score': float,
                'n_clusters': int,
                'n_noise_points': int (for methods that can produce noise)
            }
        }
    """
    
    metrics_results = {}
    
    for method_name, labels_list in labels_dict.items():
        try:
            labels = np.array(labels_list)
            method_metrics = {}
            
            # Basic cluster information
            unique_labels = np.unique(labels)
            n_clusters = len(unique_labels)
            n_noise_points = np.sum(labels == -1) if -1 in unique_labels else 0
            
            method_metrics['n_clusters'] = n_clusters
            method_metrics['n_noise_points'] = n_noise_points
            
            # Check if we have valid clusters for metric computation
            valid_clusters = unique_labels[unique_labels != -1] if -1 in unique_labels else unique_labels
            n_valid_clusters = len(valid_clusters)
            
            if n_valid_clusters < 2:
                method_metrics.update({
                    'silhouette_score': None,
                    'davies_bouldin_score': None,
                    'calinski_harabasz_score': None,
                    'error': 'Insufficient clusters for metric computation'
                })
            else:
                # Filter out noise points for metrics that don't handle them
                if -1 in labels:
                    mask = labels != -1
                    X_filtered = X[mask]
                    labels_filtered = labels[mask]
                else:
                    X_filtered = X
                    labels_filtered = labels
                
                if len(X_filtered) == 0:
                    method_metrics.update({
                        'silhouette_score': None,
                        'davies_bouldin_score': None,
                        'calinski_harabasz_score': None,
                        'error': 'All points classified as noise'
                    })
                else:
                    # Compute metrics
                    try:
                        # Silhouette Score (higher is better, range: [-1, 1])
                        if len(np.unique(labels_filtered)) > 1:
                            silhouette = silhouette_score(X_filtered, labels_filtered)
                            method_metrics['silhouette_score'] = float(silhouette)
                        else:
                            method_metrics['silhouette_score'] = None
                    except Exception as e:
                        method_metrics['silhouette_score'] = None
                        method_metrics['silhouette_error'] = str(e)
                    
                    try:
                        # Davies-Bouldin Score (lower is better, range: [0, inf))
                        if len(np.unique(labels_filtered)) > 1:
                            db_score = davies_bouldin_score(X_filtered, labels_filtered)
                            method_metrics['davies_bouldin_score'] = float(db_score)
                        else:
                            method_metrics['davies_bouldin_score'] = None
                    except Exception as e:
                        method_metrics['davies_bouldin_score'] = None
                        method_metrics['db_error'] = str(e)
                    
                    try:
                        # Calinski-Harabasz Score (higher is better, range: [0, inf))
                        if len(np.unique(labels_filtered)) > 1:
                            ch_score = calinski_harabasz_score(X_filtered, labels_filtered)
                            method_metrics['calinski_harabasz_score'] = float(ch_score)
                        else:
                            method_metrics['calinski_harabasz_score'] = None
                    except Exception as e:
                        method_metrics['calinski_harabasz_score'] = None
                        method_metrics['ch_error'] = str(e)
            
            metrics_results[method_name] = method_metrics
            
        except Exception as e:
            metrics_results[method_name] = {
                'error': f'Failed to compute metrics: {str(e)}'
            }
    
    return metrics_results

def compute_external_metrics(true_labels: np.ndarray, pred_labels: np.ndarray) -> Dict[str, float]:
    """
    Compute external clustering evaluation metrics (when ground truth is available)
    
    Args:
        true_labels: Ground truth cluster labels
        pred_labels: Predicted cluster labels
    
    Returns:
        Dictionary of external metrics
    """
    
    metrics = {}
    
    try:
        # Adjusted Rand Index (higher is better, range: [-1, 1])
        ari = adjusted_rand_score(true_labels, pred_labels)
        metrics['adjusted_rand_score'] = float(ari)
    except Exception as e:
        metrics['adjusted_rand_score'] = None
        metrics['ari_error'] = str(e)
    
    try:
        # Normalized Mutual Information (higher is better, range: [0, 1])
        nmi = normalized_mutual_info_score(true_labels, pred_labels)
        metrics['normalized_mutual_info'] = float(nmi)
    except Exception as e:
        metrics['normalized_mutual_info'] = None
        metrics['nmi_error'] = str(e)
    
    try:
        # Homogeneity Score (higher is better, range: [0, 1])
        homogeneity = homogeneity_score(true_labels, pred_labels)
        metrics['homogeneity_score'] = float(homogeneity)
    except Exception as e:
        metrics['homogeneity_score'] = None
        metrics['homogeneity_error'] = str(e)
    
    try:
        # Completeness Score (higher is better, range: [0, 1])
        completeness = completeness_score(true_labels, pred_labels)
        metrics['completeness_score'] = float(completeness)
    except Exception as e:
        metrics['completeness_score'] = None
        metrics['completeness_error'] = str(e)
    
    try:
        # V-Measure Score (higher is better, range: [0, 1])
        v_measure = v_measure_score(true_labels, pred_labels)
        metrics['v_measure_score'] = float(v_measure)
    except Exception as e:
        metrics['v_measure_score'] = None
        metrics['v_measure_error'] = str(e)
    
    return metrics

def evaluate_clustering_stability(X: np.ndarray, 
                                clustering_func: callable,
                                n_iterations: int = 10,
                                subsample_ratio: float = 0.8) -> Dict[str, float]:
    """
    Evaluate clustering stability through subsampling
    
    Args:
        X: Data matrix
        clustering_func: Function that takes X and returns cluster labels
        n_iterations: Number of subsampling iterations
        subsample_ratio: Fraction of data to subsample
    
    Returns:
        Dictionary with stability metrics
    """
    
    from sklearn.utils import resample
    
    stability_scores = []
    n_samples = X.shape[0]
    subsample_size = int(n_samples * subsample_ratio)
    
    # Get reference clustering on full data
    try:
        reference_labels = clustering_func(X)
    except Exception as e:
        return {'error': f'Failed to compute reference clustering: {str(e)}'}
    
    for i in range(n_iterations):
        try:
            # Subsample data
            subsample_indices = resample(
                range(n_samples), 
                n_samples=subsample_size, 
                random_state=i
            )
            X_subsample = X[subsample_indices]
            
            # Cluster subsample
            subsample_labels = clustering_func(X_subsample)
            
            # Map back to original indices
            full_labels = np.full(n_samples, -1)
            full_labels[subsample_indices] = subsample_labels
            
            # Compute ARI with reference
            # Only consider points that were in the subsample
            mask = full_labels != -1
            if np.sum(mask) > 0:
                ari = adjusted_rand_score(
                    reference_labels[mask], 
                    full_labels[mask]
                )
                stability_scores.append(ari)
        
        except Exception as e:
            warnings.warn(f"Stability iteration {i} failed: {e}")
            continue
    
    if len(stability_scores) == 0:
        return {'error': 'All stability iterations failed'}
    
    return {
        'mean_stability': float(np.mean(stability_scores)),
        'std_stability': float(np.std(stability_scores)),
        'min_stability': float(np.min(stability_scores)),
        'max_stability': float(np.max(stability_scores)),
        'n_successful_iterations': len(stability_scores)
    }

def rank_clustering_methods(metrics_dict: Dict[str, Dict[str, float]]) -> List[Tuple[str, float]]:
    """
    Rank clustering methods based on multiple metrics
    
    Args:
        metrics_dict: Dictionary of metrics for each method
    
    Returns:
        List of (method_name, composite_score) tuples, sorted by score (higher is better)
    """
    
    rankings = []
    
    for method_name, metrics in metrics_dict.items():
        if 'error' in metrics:
            continue
        
        score = 0.0
        n_metrics = 0
        
        # Silhouette Score (higher is better)
        if metrics.get('silhouette_score') is not None:
            score += metrics['silhouette_score']
            n_metrics += 1
        
        # Davies-Bouldin Score (lower is better, so we negate it)
        if metrics.get('davies_bouldin_score') is not None:
            score -= metrics['davies_bouldin_score'] / 10  # Scale down
            n_metrics += 1
        
        # Calinski-Harabasz Score (higher is better, but scale it down)
        if metrics.get('calinski_harabasz_score') is not None:
            score += np.log(1 + metrics['calinski_harabasz_score']) / 10  # Log scale
            n_metrics += 1
        
        # Average the score
        if n_metrics > 0:
            composite_score = score / n_metrics
            rankings.append((method_name, composite_score))
    
    # Sort by score (higher is better)
    rankings.sort(key=lambda x: x[1], reverse=True)
    
    return rankings

# Unit Tests using sklearn.datasets.make_blobs
def test_metrics_on_synthetic_data():
    """
    Unit tests for metrics computation using synthetic data
    """
    
    # Generate synthetic data
    X, true_labels = make_blobs(
        n_samples=300, 
        centers=4, 
        cluster_std=0.60, 
        random_state=42
    )
    
    # Test internal metrics
    labels_dict = {
        'perfect': true_labels.tolist(),
        'random': np.random.randint(0, 4, 300).tolist(),
        'single_cluster': np.zeros(300).tolist()
    }
    
    print("Testing internal metrics...")
    internal_metrics = compute_metrics(X, labels_dict)
    
    for method, metrics in internal_metrics.items():
        print(f"\n{method}:")
        for metric_name, value in metrics.items():
            print(f"  {metric_name}: {value}")
    
    # Test external metrics
    print("\nTesting external metrics...")
    external_metrics = compute_external_metrics(true_labels, true_labels)
    print("Perfect clustering (should be 1.0 for most metrics):")
    for metric_name, value in external_metrics.items():
        print(f"  {metric_name}: {value}")
    
    # Test with random labels
    random_labels = np.random.randint(0, 4, 300)
    external_metrics_random = compute_external_metrics(true_labels, random_labels)
    print("\nRandom clustering (should be low for most metrics):")
    for metric_name, value in external_metrics_random.items():
        print(f"  {metric_name}: {value}")
    
    # Test ranking
    print("\nRanking methods...")
    rankings = rank_clustering_methods(internal_metrics)
    for method, score in rankings:
        print(f"  {method}: {score:.4f}")
    
    return True

def validate_metrics_input(X: np.ndarray, labels: np.ndarray) -> Tuple[bool, str]:
    """
    Validate input for metrics computation
    
    Args:
        X: Data matrix
        labels: Cluster labels
    
    Returns:
        Tuple of (is_valid, error_message)
    """
    
    if X.shape[0] == 0:
        return False, "Empty data matrix"
    
    if len(labels) == 0:
        return False, "Empty labels array"
    
    if X.shape[0] != len(labels):
        return False, f"Mismatch between data ({X.shape[0]}) and labels ({len(labels)}) length"
    
    if X.shape[1] == 0:
        return False, "Data matrix has no features"
    
    # Check for NaN values
    if np.isnan(X).any():
        return False, "Data matrix contains NaN values"
    
    if np.isnan(labels).any():
        return False, "Labels contain NaN values"
    
    return True, ""

if __name__ == "__main__":
    # Run unit tests
    test_metrics_on_synthetic_data()
