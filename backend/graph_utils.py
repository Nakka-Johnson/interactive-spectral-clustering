import numpy as np
import torch
import scipy.sparse as sp
from scipy.sparse.linalg import eigsh
from sklearn.neighbors import kneighbors_graph, NearestNeighbors
from sklearn.metrics.pairwise import rbf_kernel, pairwise_distances
from typing import Union, Tuple, Optional
import warnings

def compute_affinity(X: Union[np.ndarray, torch.Tensor], 
                    method: str = 'rbf',
                    sigma: float = 1.0,
                    n_neighbors: int = 10,
                    mutual: bool = False,
                    device: Optional[str] = None) -> Union[np.ndarray, torch.Tensor]:
    """
    Compute affinity matrix using various methods
    
    Args:
        X: Data matrix (n_samples, n_features)
        method: Affinity computation method ('rbf', 'knn', 'mutual_knn')
        sigma: Bandwidth parameter for RBF kernel
        n_neighbors: Number of neighbors for k-NN methods
        mutual: Whether to use mutual k-NN (only for knn methods)
        device: Device for torch computations ('cpu', 'cuda')
    
    Returns:
        Affinity matrix (n_samples, n_samples)
    """
    
    if isinstance(X, torch.Tensor):
        return _compute_affinity_torch(X, method, sigma, n_neighbors, mutual, device)
    else:
        return _compute_affinity_numpy(X, method, sigma, n_neighbors, mutual)

def _compute_affinity_torch(X: torch.Tensor,
                           method: str,
                           sigma: float,
                           n_neighbors: int,
                           mutual: bool,
                           device: Optional[str]) -> torch.Tensor:
    """Compute affinity matrix using PyTorch (GPU-accelerated)"""
    
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    X = X.to(device)
    n_samples = X.shape[0]
    
    if method == 'rbf':
        # Compute pairwise squared distances
        distances_sq = torch.cdist(X, X, p=2) ** 2
        
        # Compute RBF kernel: exp(-distances^2 / (2 * sigma^2))
        gamma = 1.0 / (2 * sigma ** 2)
        affinity = torch.exp(-gamma * distances_sq)
        
        # Set diagonal to 1 (self-similarity)
        affinity.fill_diagonal_(1.0)
        
    elif method in ['knn', 'mutual_knn']:
        # Compute k-nearest neighbors
        distances = torch.cdist(X, X, p=2)
        
        # Find k+1 nearest neighbors (including self)
        _, indices = torch.topk(distances, k=n_neighbors + 1, dim=1, largest=False)
        
        # Remove self from neighbors
        knn_indices = indices[:, 1:]
        
        # Create sparse k-NN graph
        affinity = torch.zeros(n_samples, n_samples, device=device)
        
        for i in range(n_samples):
            neighbors = knn_indices[i]
            if method == 'rbf':
                # Use RBF weights for k-NN connections
                neighbor_distances = distances[i, neighbors]
                gamma = 1.0 / (2 * sigma ** 2)
                weights = torch.exp(-gamma * neighbor_distances ** 2)
                affinity[i, neighbors] = weights
            else:
                # Binary connections
                affinity[i, neighbors] = 1.0
        
        # Make symmetric for mutual k-NN
        if method == 'mutual_knn' or mutual:
            affinity = (affinity + affinity.t()) / 2
            affinity = (affinity > 0).float()  # Binary mutual connections
        else:
            # Standard k-NN: make symmetric by taking max
            affinity = torch.maximum(affinity, affinity.t())
    
    else:
        raise ValueError(f"Unknown affinity method: {method}")
    
    return affinity

def _compute_affinity_numpy(X: np.ndarray,
                           method: str,
                           sigma: float,
                           n_neighbors: int,
                           mutual: bool) -> np.ndarray:
    """Compute affinity matrix using NumPy/scikit-learn (CPU)"""
    
    n_samples = X.shape[0]
    
    if method == 'rbf':
        # Compute RBF kernel
        gamma = 1.0 / (2 * sigma ** 2)
        affinity = rbf_kernel(X, gamma=gamma)
        
    elif method in ['knn', 'mutual_knn']:
        # Compute k-nearest neighbors graph
        if mutual or method == 'mutual_knn':
            # Mutual k-NN: symmetric connections only
            knn_graph = kneighbors_graph(
                X, n_neighbors=n_neighbors, 
                mode='connectivity', 
                include_self=False
            )
            # Make mutual: intersection of k-NN graphs
            knn_graph = knn_graph.multiply(knn_graph.T)
            affinity = knn_graph.toarray()
        else:
            # Standard k-NN
            knn_graph = kneighbors_graph(
                X, n_neighbors=n_neighbors,
                mode='connectivity',
                include_self=False
            )
            # Make symmetric by taking union
            knn_graph = knn_graph + knn_graph.T
            knn_graph.data = np.clip(knn_graph.data, 0, 1)  # Clip to binary
            affinity = knn_graph.toarray()
    
    else:
        raise ValueError(f"Unknown affinity method: {method}")
    
    return affinity

def compute_laplacian(A: Union[np.ndarray, torch.Tensor], 
                     normalized: bool = True,
                     return_sparse: bool = True) -> Union[np.ndarray, torch.Tensor, sp.spmatrix]:
    """
    Compute Laplacian matrix from affinity matrix
    
    Args:
        A: Affinity matrix (n_samples, n_samples)
        normalized: Whether to compute normalized Laplacian
        return_sparse: Whether to return sparse matrix (only for numpy)
    
    Returns:
        Laplacian matrix L
        - If normalized=True: L = I - D^(-1/2) * A * D^(-1/2)
        - If normalized=False: L = D - A
    """
    
    if isinstance(A, torch.Tensor):
        return _compute_laplacian_torch(A, normalized)
    else:
        return _compute_laplacian_numpy(A, normalized, return_sparse)

def _compute_laplacian_torch(A: torch.Tensor, normalized: bool) -> torch.Tensor:
    """Compute Laplacian using PyTorch"""
    
    device = A.device
    n = A.shape[0]
    
    # Compute degree matrix
    degrees = torch.sum(A, dim=1)
    
    # Handle isolated nodes (zero degree)
    degrees = torch.where(degrees == 0, torch.ones_like(degrees), degrees)
    
    if normalized:
        # Normalized Laplacian: L = I - D^(-1/2) * A * D^(-1/2)
        deg_inv_sqrt = torch.pow(degrees, -0.5)
        deg_inv_sqrt = torch.where(torch.isinf(deg_inv_sqrt), 
                                  torch.zeros_like(deg_inv_sqrt), 
                                  deg_inv_sqrt)
        
        # Create diagonal matrix D^(-1/2)
        D_inv_sqrt = torch.diag(deg_inv_sqrt)
        
        # Compute normalized Laplacian
        I = torch.eye(n, device=device)
        L = I - torch.matmul(torch.matmul(D_inv_sqrt, A), D_inv_sqrt)
    else:
        # Unnormalized Laplacian: L = D - A
        D = torch.diag(degrees)
        L = D - A
    
    return L

def _compute_laplacian_numpy(A: np.ndarray, 
                           normalized: bool, 
                           return_sparse: bool) -> Union[np.ndarray, sp.spmatrix]:
    """Compute Laplacian using NumPy/SciPy"""
    
    n = A.shape[0]
    
    # Convert to sparse if not already
    if not sp.issparse(A):
        A_sparse = sp.csr_matrix(A)
    else:
        A_sparse = A
    
    # Compute degrees
    degrees = np.array(A_sparse.sum(axis=1)).flatten()
    
    # Handle isolated nodes
    degrees = np.where(degrees == 0, 1, degrees)
    
    if normalized:
        # Normalized Laplacian
        deg_inv_sqrt = np.power(degrees, -0.5)
        deg_inv_sqrt[np.isinf(deg_inv_sqrt)] = 0
        
        # Create diagonal matrix
        D_inv_sqrt = sp.diags(deg_inv_sqrt, format='csr')
        
        # Compute normalized Laplacian
        I = sp.eye(n, format='csr')
        L = I - D_inv_sqrt @ A_sparse @ D_inv_sqrt
    else:
        # Unnormalized Laplacian
        D = sp.diags(degrees, format='csr')
        L = D - A_sparse
    
    if return_sparse:
        return L
    else:
        return L.toarray()

def compute_eigendecomposition(L: Union[np.ndarray, torch.Tensor, sp.spmatrix],
                              k: int,
                              which: str = 'SM',
                              return_torch: bool = False) -> Tuple[Union[np.ndarray, torch.Tensor], 
                                                                  Union[np.ndarray, torch.Tensor]]:
    """
    Compute eigendecomposition of Laplacian matrix
    
    Args:
        L: Laplacian matrix
        k: Number of eigenvalues/eigenvectors to compute
        which: Which eigenvalues to compute ('SM' for smallest, 'LM' for largest)
        return_torch: Whether to return torch tensors
    
    Returns:
        Tuple of (eigenvalues, eigenvectors)
    """
    
    if isinstance(L, torch.Tensor):
        return _compute_eigen_torch(L, k, which, return_torch)
    else:
        return _compute_eigen_numpy(L, k, which, return_torch)

def _compute_eigen_torch(L: torch.Tensor, 
                        k: int, 
                        which: str,
                        return_torch: bool) -> Tuple[Union[np.ndarray, torch.Tensor], 
                                                    Union[np.ndarray, torch.Tensor]]:
    """Compute eigendecomposition using PyTorch"""
    
    try:
        # Use torch.linalg.eigh for symmetric matrices
        eigenvals, eigenvecs = torch.linalg.eigh(L)
        
        # Sort eigenvalues and eigenvectors
        if which == 'SM':
            # Smallest eigenvalues first
            idx = torch.argsort(eigenvals)
        else:
            # Largest eigenvalues first
            idx = torch.argsort(eigenvals, descending=True)
        
        # Select k eigenvalues/eigenvectors
        selected_idx = idx[:k]
        selected_eigenvals = eigenvals[selected_idx]
        selected_eigenvecs = eigenvecs[:, selected_idx]
        
        if return_torch:
            return selected_eigenvals, selected_eigenvecs
        else:
            return selected_eigenvals.cpu().numpy(), selected_eigenvecs.cpu().numpy()
            
    except Exception as e:
        warnings.warn(f"PyTorch eigendecomposition failed: {e}. Falling back to NumPy.")
        # Fallback to NumPy
        L_np = L.cpu().numpy()
        return _compute_eigen_numpy(L_np, k, which, return_torch)

def _compute_eigen_numpy(L: Union[np.ndarray, sp.spmatrix], 
                        k: int, 
                        which: str,
                        return_torch: bool) -> Tuple[Union[np.ndarray, torch.Tensor], 
                                                    Union[np.ndarray, torch.Tensor]]:
    """Compute eigendecomposition using NumPy/SciPy"""
    
    try:
        if sp.issparse(L):
            # Use sparse eigenvalue solver
            if k >= L.shape[0] - 1:
                # Convert to dense if asking for too many eigenvalues
                L_dense = L.toarray()
                eigenvals, eigenvecs = np.linalg.eigh(L_dense)
            else:
                # Sparse solver
                eigenvals, eigenvecs = eigsh(L, k=k, which=which, sigma=0.0)
        else:
            # Dense eigenvalue decomposition
            eigenvals, eigenvecs = np.linalg.eigh(L)
            
            # Sort and select
            if which == 'SM':
                idx = np.argsort(eigenvals)
            else:
                idx = np.argsort(eigenvals)[::-1]
            
            eigenvals = eigenvals[idx[:k]]
            eigenvecs = eigenvecs[:, idx[:k]]
        
        if return_torch:
            return torch.tensor(eigenvals), torch.tensor(eigenvecs)
        else:
            return eigenvals, eigenvecs
            
    except Exception as e:
        raise RuntimeError(f"Eigendecomposition failed: {e}")

def spectral_embedding(X: Union[np.ndarray, torch.Tensor],
                      n_components: int,
                      affinity_method: str = 'rbf',
                      sigma: float = 1.0,
                      n_neighbors: int = 10,
                      normalized_laplacian: bool = True) -> Union[np.ndarray, torch.Tensor]:
    """
    Compute spectral embedding of data
    
    Args:
        X: Input data matrix
        n_components: Number of embedding dimensions
        affinity_method: Method for computing affinity matrix
        sigma: Bandwidth for RBF kernel
        n_neighbors: Number of neighbors for k-NN methods
        normalized_laplacian: Whether to use normalized Laplacian
    
    Returns:
        Spectral embedding (n_samples, n_components)
    """
    
    # Compute affinity matrix
    A = compute_affinity(X, method=affinity_method, sigma=sigma, n_neighbors=n_neighbors)
    
    # Compute Laplacian
    L = compute_laplacian(A, normalized=normalized_laplacian)
    
    # Compute eigendecomposition
    eigenvals, eigenvecs = compute_eigendecomposition(L, k=n_components + 1, which='SM')
    
    # Skip the first eigenvector (constant vector for connected graphs)
    if isinstance(eigenvecs, torch.Tensor):
        embedding = eigenvecs[:, 1:n_components + 1]
    else:
        embedding = eigenvecs[:, 1:n_components + 1]
    
    return embedding
