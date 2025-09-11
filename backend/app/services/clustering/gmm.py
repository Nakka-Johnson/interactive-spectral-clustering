"""
Gaussian Mixture Model (GMM) clustering algorithm implementation.

This module provides GMM clustering with various covariance types and 
initialization strategies for probabilistic cluster assignment.
"""

import numpy as np
from typing import Dict, Any, TYPE_CHECKING
from .factory import BaseClusteringAlgorithm, AlgorithmType
import logging

if TYPE_CHECKING:
    import torch

logger = logging.getLogger(__name__)

class GMMClustering(BaseClusteringAlgorithm):
    """
    Gaussian Mixture Model clustering for probabilistic cluster assignment.
    
    Uses Expectation-Maximization algorithm to fit Gaussian components
    and provides soft cluster assignments with probabilities.
    """
    
    def get_algorithm_type(self) -> AlgorithmType:
        return AlgorithmType.GMM
    
    def get_default_parameters(self) -> Dict[str, Any]:
        return {
            'n_components': 3,
            'covariance_type': 'full',  # 'full', 'tied', 'diag', 'spherical'
            'tol': 1e-3,
            'reg_covar': 1e-6,
            'max_iter': 100,
            'n_init': 1,
            'init_params': 'kmeans',  # 'kmeans', 'random'
            'weights_init': None,
            'means_init': None,
            'precisions_init': None,
            'random_state': 42,
            'warm_start': False,
            'verbose': 0,
        }
    
    def validate_parameters(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Validate GMM parameters."""
        validated = params.copy()
        
        # Validate n_components
        if validated.get('n_components', 3) < 1:
            validated['n_components'] = 3
            logger.warning("n_components must be >= 1, setting to 3")
        if validated.get('n_components', 3) > 50:
            validated['n_components'] = 50
            logger.warning("n_components too large, setting to 50")
        
        # Validate covariance_type
        valid_cov_types = ['full', 'tied', 'diag', 'spherical']
        if validated.get('covariance_type', 'full') not in valid_cov_types:
            validated['covariance_type'] = 'full'
            logger.warning(f"Invalid covariance_type, setting to full. Valid options: {valid_cov_types}")
        
        # Validate tolerance
        if validated.get('tol', 1e-3) <= 0:
            validated['tol'] = 1e-3
        
        # Validate regularization
        if validated.get('reg_covar', 1e-6) <= 0:
            validated['reg_covar'] = 1e-6
        
        # Validate max_iter
        if validated.get('max_iter', 100) < 1:
            validated['max_iter'] = 100
        if validated.get('max_iter', 100) > 1000:
            validated['max_iter'] = 1000
            logger.warning("max_iter too large, setting to 1000")
        
        # Validate n_init
        if validated.get('n_init', 1) < 1:
            validated['n_init'] = 1
        
        # Validate init_params
        valid_init = ['kmeans', 'random']
        if validated.get('init_params', 'kmeans') not in valid_init:
            validated['init_params'] = 'kmeans'
            logger.warning(f"Invalid init_params, setting to kmeans. Valid options: {valid_init}")
        
        return validated
    
    def _fit_predict(self, X: np.ndarray, params: Dict[str, Any]) -> np.ndarray:
        """Core GMM implementation."""
        if self.use_gpu:
            return self._fit_predict_gpu(X, params)
        else:
            return self._fit_predict_cpu(X, params)
    
    def _fit_predict_gpu(self, X: np.ndarray, params: Dict[str, Any]) -> np.ndarray:
        """GPU-accelerated GMM using PyTorch."""
        try:
            import torch
            from torch.distributions import MultivariateNormal
            
            if not torch.cuda.is_available():
                logger.warning("GPU requested but not available, falling back to CPU")
                return self._fit_predict_cpu(X, params)
            
            logger.info("Using GPU acceleration (PyTorch) for GMM")
            
            # Convert to GPU tensors
            X_gpu = torch.tensor(X, dtype=torch.float32).cuda()
            n_samples, n_features = X_gpu.shape
            n_components = params['n_components']
            max_iter = params['max_iter']
            tol = params['tol']
            reg_covar = params['reg_covar']
            
            # Initialize parameters
            means, covariances, weights = self._initialize_parameters_gpu(
                X_gpu, n_components, params
            )
            
            # EM algorithm
            log_likelihood_old = float('-inf')
            
            for iteration in range(max_iter):
                # E-step: compute responsibilities
                responsibilities = self._e_step_gpu(X_gpu, means, covariances, weights, reg_covar)
                
                # M-step: update parameters
                means, covariances, weights = self._m_step_gpu(
                    X_gpu, responsibilities, params['covariance_type'], reg_covar
                )
                
                # Check convergence
                log_likelihood = self._compute_log_likelihood_gpu(
                    X_gpu, means, covariances, weights, reg_covar
                )
                
                if abs(log_likelihood - log_likelihood_old) < tol:
                    logger.info(f"GMM converged after {iteration + 1} iterations")
                    break
                
                log_likelihood_old = log_likelihood
            
            # Predict cluster labels
            responsibilities = self._e_step_gpu(X_gpu, means, covariances, weights, reg_covar)
            labels = torch.argmax(responsibilities, dim=1).cpu().numpy()
            
            return labels
            
        except Exception as e:
            logger.warning(f"GPU GMM failed: {str(e)}, falling back to CPU")
            return self._fit_predict_cpu(X, params)
    
    def _fit_predict_cpu(self, X: np.ndarray, params: Dict[str, Any]) -> np.ndarray:
        """CPU GMM using scikit-learn."""
        from sklearn.mixture import GaussianMixture
        
        logger.info("Using CPU GMM (scikit-learn)")
        
        # Create GMM instance
        gmm = GaussianMixture(
            n_components=params['n_components'],
            covariance_type=params['covariance_type'],
            tol=params['tol'],
            reg_covar=params['reg_covar'],
            max_iter=params['max_iter'],
            n_init=params['n_init'],
            init_params=params['init_params'],
            weights_init=params['weights_init'],
            means_init=params['means_init'],
            precisions_init=params['precisions_init'],
            random_state=params['random_state'],
            warm_start=params['warm_start'],
            verbose=params['verbose']
        )
        
        # Fit and predict
        labels = gmm.fit_predict(X)
        
        # Store additional information for metrics
        self._gmm_model = gmm
        
        return labels
    
    def _initialize_parameters_gpu(self, X: "torch.Tensor", n_components: int, params: Dict[str, Any]):
        """Initialize GMM parameters on GPU."""
        import torch
        
        n_samples, n_features = X.shape
        
        if params['init_params'] == 'kmeans':
            # Use k-means for initialization
            from sklearn.cluster import KMeans
            kmeans = KMeans(n_clusters=n_components, random_state=params['random_state'])
            labels_cpu = kmeans.fit_predict(X.cpu().numpy())
            
            # Initialize means
            means = torch.zeros(n_components, n_features, device=X.device)
            for k in range(n_components):
                mask = (labels_cpu == k)
                if mask.sum() > 0:
                    means[k] = X[torch.tensor(mask, device=X.device)].mean(dim=0)
                else:
                    means[k] = X[torch.randint(0, n_samples, (1,))].squeeze()
        else:
            # Random initialization
            torch.manual_seed(params['random_state'])
            means = X[torch.randperm(n_samples)[:n_components]]
        
        # Initialize covariances
        covariances = torch.stack([torch.eye(n_features, device=X.device) for _ in range(n_components)])
        
        # Initialize weights
        weights = torch.ones(n_components, device=X.device) / n_components
        
        return means, covariances, weights
    
    def _e_step_gpu(self, X: "torch.Tensor", means: "torch.Tensor", covariances: "torch.Tensor", 
                    weights: "torch.Tensor", reg_covar: float) -> "torch.Tensor":
        """E-step: compute responsibilities on GPU."""
        import torch
        from torch.distributions import MultivariateNormal
        
        n_samples, n_features = X.shape
        n_components = means.shape[0]
        
        # Regularize covariances
        reg_eye = torch.eye(n_features, device=X.device) * reg_covar
        covariances_reg = covariances + reg_eye.unsqueeze(0)
        
        # Compute log probabilities
        log_probs = torch.zeros(n_samples, n_components, device=X.device)
        
        for k in range(n_components):
            try:
                dist = MultivariateNormal(means[k], covariances_reg[k])
                log_probs[:, k] = dist.log_prob(X) + torch.log(weights[k])
            except:
                # Fallback for numerical issues
                log_probs[:, k] = -torch.inf
        
        # Compute responsibilities using log-sum-exp trick
        log_probs_max = torch.max(log_probs, dim=1, keepdim=True)[0]
        probs = torch.exp(log_probs - log_probs_max)
        responsibilities = probs / torch.sum(probs, dim=1, keepdim=True)
        
        return responsibilities
    
    def _m_step_gpu(self, X: "torch.Tensor", responsibilities: "torch.Tensor", 
                    covariance_type: str, reg_covar: float):
        """M-step: update parameters on GPU."""
        import torch
        
        n_samples, n_features = X.shape
        n_components = responsibilities.shape[1]
        
        # Update weights
        weights = torch.mean(responsibilities, dim=0)
        
        # Update means
        means = torch.zeros(n_components, n_features, device=X.device)
        for k in range(n_components):
            means[k] = torch.sum(responsibilities[:, k:k+1] * X, dim=0) / torch.sum(responsibilities[:, k])
        
        # Update covariances
        covariances = torch.zeros(n_components, n_features, n_features, device=X.device)
        
        for k in range(n_components):
            diff = X - means[k:k+1]
            weighted_diff = responsibilities[:, k:k+1] * diff
            covariances[k] = torch.mm(weighted_diff.T, diff) / torch.sum(responsibilities[:, k])
            
            # Apply covariance type constraints
            if covariance_type == 'spherical':
                # Spherical: σ²I
                variance = torch.trace(covariances[k]) / n_features
                covariances[k] = variance * torch.eye(n_features, device=X.device)
            elif covariance_type == 'diag':
                # Diagonal: only diagonal elements
                covariances[k] = torch.diag(torch.diag(covariances[k]))
        
        if covariance_type == 'tied':
            # Tied: same covariance for all components
            tied_cov = torch.mean(covariances, dim=0)
            covariances = tied_cov.unsqueeze(0).repeat(n_components, 1, 1)
        
        return means, covariances, weights
    
    def _compute_log_likelihood_gpu(self, X: "torch.Tensor", means: "torch.Tensor", 
                                   covariances: "torch.Tensor", weights: "torch.Tensor", reg_covar: float) -> float:
        """Compute log-likelihood on GPU."""
        responsibilities = self._e_step_gpu(X, means, covariances, weights, reg_covar)
        return torch.sum(torch.log(torch.sum(responsibilities, dim=1))).item()
    
    def _compute_metrics(self, X: np.ndarray, labels: np.ndarray) -> Dict[str, float]:
        """Compute GMM-specific metrics in addition to standard ones."""
        # Get standard metrics
        metrics = super()._compute_metrics(X, labels)
        
        # Add GMM-specific metrics if available
        if hasattr(self, '_gmm_model'):
            gmm = self._gmm_model
            
            # Add likelihood-based metrics
            metrics.update({
                'log_likelihood': gmm.score(X),
                'aic': gmm.aic(X),
                'bic': gmm.bic(X),
                'n_components_used': gmm.n_components
            })
            
            # Add convergence information
            if hasattr(gmm, 'converged_'):
                metrics['converged'] = gmm.converged_
            if hasattr(gmm, 'n_iter_'):
                metrics['n_iterations'] = gmm.n_iter_
        
        return metrics
