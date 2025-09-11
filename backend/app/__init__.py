"""
App package initialization for Interactive Spectral Clustering Platform.

This module exports the main components of the application for easy importing.
"""

from .api_schemas import *
from .db_models import *
from .db_service import *
from .clustering import *
from .endpoints import router

__version__ = "1.0.0"
__all__ = [
    # Schemas
    "AlgorithmId",
    "RunStatus", 
    "ParameterMap",
    "DatasetMeta",
    "RunRequest",
    "ClusteringRunResponse",
    "EmbeddingRequest",
    "EmbeddingResponse",
    "ErrorResponse",
    "SuccessResponse",
    
    # Database
    "Base",
    "Dataset",
    "ClusteringRun", 
    "AlgorithmResult",
    "Embedding",
    "GridSearch",
    "SystemMetrics",
    "create_database_engine",
    "create_session_factory",
    "init_database",
    "get_db_session",
    
    # Services
    "DatabaseService",
    "get_database_service",
    
    # Clustering
    "ClusteringEngine",
    "EmbeddingEngine",
    "DataPreprocessor",
    "ClusteringResult",
    "EmbeddingResult",
    "create_clustering_engine",
    "create_embedding_engine",
    
    # Endpoints
    "router",
]
