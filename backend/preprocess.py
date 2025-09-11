import os
import json
import numpy as np
import pandas as pd
import torch
from typing import Tuple, Optional, List, Dict, Any
from sqlalchemy.orm import Session
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from io import StringIO

# Import database models and connection manager from app.py
from app import Dataset, SessionLocal, manager

# Check if UMAP is available
try:
    import umap
    UMAP_AVAILABLE = True
except ImportError:
    UMAP_AVAILABLE = False

def get_db():
    """Get database session"""
    db = SessionLocal()
    try:
        return db
    finally:
        pass  # Don't close here, caller will handle

async def emit_progress(job_id: str, progress: int, message: str):
    """Helper function to emit progress via WebSocket"""
    try:
        await manager.send_progress(job_id, progress, message)
    except Exception as e:
        print(f"Failed to send progress for job {job_id}: {e}")

def load_dataset_from_db(job_id: str, db: Session) -> Tuple[pd.DataFrame, List[str]]:
    """Load dataset from database by job_id"""
    dataset = db.query(Dataset).filter(Dataset.job_id == job_id).first()
    if not dataset:
        raise ValueError(f"Dataset with job_id {job_id} not found")
    
    # Parse CSV content
    df = pd.read_csv(StringIO(dataset.raw_csv))
    numeric_columns = json.loads(dataset.numeric_columns)
    
    return df, numeric_columns

def apply_filters(df: pd.DataFrame, 
                 column_filters: Optional[List[str]] = None,
                 row_conditions: Optional[Dict[str, Any]] = None) -> pd.DataFrame:
    """Apply user-specified column and row filters"""
    filtered_df = df.copy()
    
    # Apply column filters
    if column_filters:
        available_columns = [col for col in column_filters if col in filtered_df.columns]
        if available_columns:
            filtered_df = filtered_df[available_columns]
    
    # Apply row conditions (basic filtering)
    if row_conditions:
        for column, condition in row_conditions.items():
            if column in filtered_df.columns:
                if isinstance(condition, dict):
                    if 'min' in condition:
                        filtered_df = filtered_df[filtered_df[column] >= condition['min']]
                    if 'max' in condition:
                        filtered_df = filtered_df[filtered_df[column] <= condition['max']]
                    if 'values' in condition:
                        filtered_df = filtered_df[filtered_df[column].isin(condition['values'])]
    
    return filtered_df

def standardize_data(X: np.ndarray) -> Tuple[np.ndarray, StandardScaler]:
    """Apply StandardScaler to the data"""
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return X_scaled, scaler

def apply_pca_gpu(X: np.ndarray, n_components: int) -> Tuple[np.ndarray, object]:
    """Apply PCA using PyTorch on GPU if available"""
    try:
        # Check if CUDA is available
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Convert to torch tensor
        X_tensor = torch.tensor(X, dtype=torch.float32, device=device)
        
        # Center the data
        X_centered = X_tensor - torch.mean(X_tensor, dim=0, keepdim=True)
        
        # Compute PCA using SVD
        U, S, Vt = torch.pca_lowrank(X_centered, q=n_components)
        
        # Transform data
        X_pca = torch.matmul(X_centered, Vt.t())
        
        # Convert back to numpy
        X_pca_np = X_pca.cpu().numpy()
        
        # Create a simple PCA-like object for consistency
        class TorchPCA:
            def __init__(self, components, explained_variance_ratio):
                self.components_ = components
                self.explained_variance_ratio_ = explained_variance_ratio
                self.n_components_ = len(explained_variance_ratio)
        
        # Calculate explained variance ratio
        explained_variance = S.cpu().numpy() ** 2 / (X.shape[0] - 1)
        total_variance = np.sum(explained_variance)
        explained_variance_ratio = explained_variance / total_variance
        
        pca_obj = TorchPCA(Vt.cpu().numpy(), explained_variance_ratio)
        
        return X_pca_np, pca_obj
        
    except Exception as e:
        print(f"GPU PCA failed, falling back to CPU: {e}")
        # Fallback to sklearn PCA
        return apply_pca_cpu(X, n_components)

def apply_pca_cpu(X: np.ndarray, n_components: int) -> Tuple[np.ndarray, PCA]:
    """Apply PCA using scikit-learn on CPU"""
    pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(X)
    return X_pca, pca

def apply_dimensionality_reduction(X: np.ndarray, 
                                 dim_reducer: str, 
                                 n_components: int = 2,
                                 **kwargs) -> Tuple[np.ndarray, object]:
    """Apply dimensionality reduction technique"""
    
    if dim_reducer.upper() == 'PCA':
        return apply_pca_cpu(X, n_components)
    
    elif dim_reducer.upper() == 'TSNE':
        # For t-SNE, we typically use 2 or 3 components
        n_comp = min(n_components, 3)
        tsne = TSNE(
            n_components=n_comp,
            random_state=42,
            perplexity=min(30, X.shape[0] - 1),
            **kwargs
        )
        X_tsne = tsne.fit_transform(X)
        return X_tsne, tsne
    
    elif dim_reducer.upper() == 'UMAP':
        if not UMAP_AVAILABLE:
            raise ImportError("UMAP not available. Please install umap-learn: pip install umap-learn")
        
        umap_reducer = umap.UMAP(
            n_components=n_components,
            random_state=42,
            **kwargs
        )
        X_umap = umap_reducer.fit_transform(X)
        return X_umap, umap_reducer
    
    else:
        raise ValueError(f"Unsupported dimensionality reduction method: {dim_reducer}")

async def preprocess(job_id: str, 
                    use_pca: bool = False, 
                    n_components: int = 2,
                    dim_reducer: str = 'PCA',
                    column_filters: Optional[List[str]] = None,
                    row_conditions: Optional[Dict[str, Any]] = None,
                    use_gpu_pca: bool = True) -> Tuple[np.ndarray, np.ndarray]:
    """
    Main preprocessing function
    
    Args:
        job_id: Database job identifier
        use_pca: Whether to apply PCA for dimensionality reduction
        n_components: Number of components for PCA
        dim_reducer: Type of dimensionality reduction ('PCA', 'TSNE', 'UMAP')
        column_filters: List of columns to include
        row_conditions: Dictionary of row filtering conditions
        use_gpu_pca: Whether to use GPU acceleration for PCA
    
    Returns:
        Tuple of (X_scaled, embedding) where:
        - X_scaled: Standardized feature matrix
        - embedding: Low-dimensional embedding for plotting
    """
    
    db = get_db()
    
    try:
        # Step 1: Load data from database
        await emit_progress(job_id, 10, "Loading dataset from database...")
        df, numeric_columns = load_dataset_from_db(job_id, db)
        
        # Step 2: Apply filters
        await emit_progress(job_id, 20, "Applying data filters...")
        if column_filters:
            # Use only specified columns that are also numeric
            filtered_columns = [col for col in column_filters if col in numeric_columns]
        else:
            filtered_columns = numeric_columns
        
        if not filtered_columns:
            raise ValueError("No valid numeric columns found after filtering")
        
        # Apply row and column filters
        df_filtered = apply_filters(df, filtered_columns, row_conditions)
        
        if df_filtered.empty:
            raise ValueError("No data remaining after applying filters")
        
        # Step 3: Prepare feature matrix
        await emit_progress(job_id, 30, "Preparing feature matrix...")
        X = df_filtered[filtered_columns].values
        
        # Handle missing values
        if np.isnan(X).any():
            # Fill NaN with column means
            col_means = np.nanmean(X, axis=0)
            for i in range(X.shape[1]):
                X[np.isnan(X[:, i]), i] = col_means[i]
        
        # Step 4: Standardize data
        await emit_progress(job_id, 40, "Standardizing features...")
        X_scaled, scaler = standardize_data(X)
        
        # Step 5: Apply PCA if requested
        if use_pca:
            await emit_progress(job_id, 50, f"Applying PCA ({n_components} components)...")
            
            # Ensure n_components doesn't exceed feature count
            n_comp_pca = min(n_components, X_scaled.shape[1])
            
            if use_gpu_pca and torch.cuda.is_available():
                await emit_progress(job_id, 55, "Using GPU-accelerated PCA...")
                X_scaled, pca_obj = apply_pca_gpu(X_scaled, n_comp_pca)
            else:
                await emit_progress(job_id, 55, "Using CPU PCA...")
                X_scaled, pca_obj = apply_pca_cpu(X_scaled, n_comp_pca)
        
        # Step 6: Generate embedding for visualization
        await emit_progress(job_id, 70, f"Computing {dim_reducer} embedding for visualization...")
        
        # For embedding, use up to 50 components to avoid curse of dimensionality
        X_for_embedding = X_scaled
        if X_scaled.shape[1] > 50:
            await emit_progress(job_id, 75, "Reducing dimensions for embedding computation...")
            X_for_embedding, _ = apply_pca_cpu(X_scaled, 50)
        
        # Generate 2D embedding for plotting
        try:
            embedding_2d, reducer_obj = apply_dimensionality_reduction(
                X_for_embedding, 
                dim_reducer, 
                n_components=2
            )
        except Exception as e:
            # Fallback to PCA if other methods fail
            await emit_progress(job_id, 75, f"{dim_reducer} failed, using PCA fallback...")
            embedding_2d, reducer_obj = apply_pca_cpu(X_for_embedding, 2)
        
        await emit_progress(job_id, 90, "Preprocessing complete!")
        
        return X_scaled, embedding_2d
        
    except Exception as e:
        await emit_progress(job_id, 0, f"Preprocessing failed: {str(e)}")
        raise e
    
    finally:
        db.close()

def get_preprocessing_info(job_id: str) -> Dict[str, Any]:
    """Get information about available preprocessing options for a dataset"""
    db = get_db()
    
    try:
        df, numeric_columns = load_dataset_from_db(job_id, db)
        
        info = {
            "total_rows": len(df),
            "total_columns": len(df.columns),
            "numeric_columns": numeric_columns,
            "missing_values": df.isnull().sum().to_dict(),
            "data_types": df.dtypes.astype(str).to_dict(),
            "available_reducers": ["PCA", "TSNE"] + (["UMAP"] if UMAP_AVAILABLE else []),
            "gpu_available": torch.cuda.is_available()
        }
        
        return info
        
    finally:
        db.close()
