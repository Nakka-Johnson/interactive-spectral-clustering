"""
SQLAlchemy database models for Interactive Spectral Clustering Platform.

This module defines the database schema using SQLAlchemy ORM models.
Supports multi-tenancy and comprehensive metadata tracking.
"""

import uuid
from datetime import datetime
from typing import Optional

from sqlalchemy import (
    Boolean, Column, DateTime, Float, Integer, String, Text, JSON,
    ForeignKey, Index, create_engine
)
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship, sessionmaker, Session
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.sql import func


Base = declarative_base()


class TimestampMixin:
    """Mixin for created_at and updated_at timestamps."""
    created_at = Column(DateTime, default=func.now(), nullable=False)
    updated_at = Column(DateTime, default=func.now(), onupdate=func.now(), nullable=False)


class TenantMixin:
    """Mixin for multi-tenant support."""
    tenant_id = Column(String(255), nullable=True, index=True, comment="Tenant identifier for multi-tenancy")


class Dataset(Base, TimestampMixin, TenantMixin):
    """Dataset model for uploaded files."""
    
    __tablename__ = "datasets"
    
    # Primary key
    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    
    # Basic metadata
    filename = Column(String(255), nullable=False, comment="Original filename")
    file_path = Column(String(500), nullable=False, comment="Storage path")
    file_size_bytes = Column(Integer, nullable=False, comment="File size in bytes")
    
    # Data characteristics
    total_rows = Column(Integer, nullable=False, comment="Number of data rows")
    total_columns = Column(Integer, nullable=False, comment="Number of columns")
    column_names = Column(JSON, nullable=False, comment="List of column names")
    column_types = Column(JSON, nullable=True, comment="Detected column data types")
    
    # Processing status
    is_processed = Column(Boolean, default=False, comment="Whether dataset has been preprocessed")
    processing_error = Column(Text, nullable=True, comment="Processing error message if any")
    
    # Sample data for preview
    sample_data = Column(JSON, nullable=True, comment="Sample rows for quick preview")
    
    # Statistics
    missing_values_count = Column(JSON, nullable=True, comment="Missing values per column")
    data_summary = Column(JSON, nullable=True, comment="Basic statistical summary")
    
    # Relationships
    clustering_runs = relationship("ClusteringRun", back_populates="dataset", cascade="all, delete-orphan")
    embeddings = relationship("Embedding", back_populates="dataset", cascade="all, delete-orphan")
    
    # Indexes
    __table_args__ = (
        Index('idx_dataset_tenant_created', 'tenant_id', 'created_at'),
        Index('idx_dataset_filename', 'filename'),
    )
    
    def __repr__(self):
        return f"<Dataset(id={self.id}, filename={self.filename}, rows={self.total_rows})>"


class ClusteringRun(Base, TimestampMixin, TenantMixin):
    """Clustering run model for experiment tracking."""
    
    __tablename__ = "clustering_runs"
    
    # Primary key
    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    
    # Foreign keys
    dataset_id = Column(String(36), ForeignKey("datasets.id"), nullable=False)
    
    # Run metadata
    run_name = Column(String(255), nullable=True, comment="Optional run name")
    status = Column(String(50), nullable=False, default="pending", comment="Run status")
    
    # Configuration
    algorithms_requested = Column(JSON, nullable=False, comment="List of requested algorithms")
    preprocessing_config = Column(JSON, nullable=False, comment="Preprocessing configuration")
    
    # Execution tracking
    started_at = Column(DateTime, nullable=True, comment="Execution start time")
    completed_at = Column(DateTime, nullable=True, comment="Execution completion time")
    total_execution_time = Column(Float, nullable=True, comment="Total execution time in seconds")
    
    # Results storage
    results = Column(JSON, nullable=True, comment="Complete results data")
    best_algorithm = Column(String(50), nullable=True, comment="Best performing algorithm")
    best_score = Column(Float, nullable=True, comment="Best algorithm score")
    
    # Error handling
    error_message = Column(Text, nullable=True, comment="Error message if run failed")
    error_details = Column(JSON, nullable=True, comment="Detailed error information")
    
    # Metadata
    parameters_hash = Column(String(64), nullable=True, comment="Hash of run parameters for deduplication")
    tags = Column(JSON, nullable=True, comment="User-defined tags")
    notes = Column(Text, nullable=True, comment="User notes")
    
    # Relationships
    dataset = relationship("Dataset", back_populates="clustering_runs")
    algorithm_results = relationship("AlgorithmResult", back_populates="clustering_run", cascade="all, delete-orphan")
    
    # Indexes
    __table_args__ = (
        Index('idx_clustering_run_dataset', 'dataset_id'),
        Index('idx_clustering_run_status', 'status'),
        Index('idx_clustering_run_tenant_created', 'tenant_id', 'created_at'),
    )
    
    def __repr__(self):
        return f"<ClusteringRun(id={self.id}, dataset_id={self.dataset_id}, status={self.status})>"


class AlgorithmResult(Base, TimestampMixin):
    """Individual algorithm result within a clustering run."""
    
    __tablename__ = "algorithm_results"
    
    # Primary key
    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    
    # Foreign keys
    clustering_run_id = Column(String(36), ForeignKey("clustering_runs.id"), nullable=False)
    
    # Algorithm information
    algorithm_name = Column(String(50), nullable=False, comment="Algorithm identifier")
    parameters_used = Column(JSON, nullable=False, comment="Parameters used for this algorithm")
    
    # Execution tracking
    status = Column(String(50), nullable=False, default="pending", comment="Algorithm execution status")
    execution_time = Column(Float, nullable=True, comment="Execution time in seconds")
    
    # Results
    labels = Column(JSON, nullable=True, comment="Cluster labels array")
    cluster_centers = Column(JSON, nullable=True, comment="Cluster centers if applicable")
    n_clusters_found = Column(Integer, nullable=True, comment="Number of clusters found")
    n_noise_points = Column(Integer, nullable=True, comment="Number of noise points (DBSCAN)")
    
    # Evaluation metrics
    silhouette_score = Column(Float, nullable=True, comment="Silhouette coefficient")
    calinski_harabasz_score = Column(Float, nullable=True, comment="Calinski-Harabasz index")
    davies_bouldin_score = Column(Float, nullable=True, comment="Davies-Bouldin index")
    inertia = Column(Float, nullable=True, comment="Within-cluster sum of squares")
    
    # Additional algorithm-specific metrics
    additional_metrics = Column(JSON, nullable=True, comment="Algorithm-specific metrics")
    
    # Error handling
    error_message = Column(Text, nullable=True, comment="Error message if algorithm failed")
    
    # Relationships
    clustering_run = relationship("ClusteringRun", back_populates="algorithm_results")
    
    # Indexes
    __table_args__ = (
        Index('idx_algorithm_result_run_algorithm', 'clustering_run_id', 'algorithm_name'),
        Index('idx_algorithm_result_status', 'status'),
    )
    
    def __repr__(self):
        return f"<AlgorithmResult(id={self.id}, algorithm={self.algorithm_name}, status={self.status})>"


class Embedding(Base, TimestampMixin, TenantMixin):
    """Dimensionality reduction embeddings."""
    
    __tablename__ = "embeddings"
    
    # Primary key
    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    
    # Foreign keys
    dataset_id = Column(String(36), ForeignKey("datasets.id"), nullable=False)
    clustering_run_id = Column(String(36), ForeignKey("clustering_runs.id"), nullable=True)
    
    # Embedding metadata
    method = Column(String(50), nullable=False, comment="Embedding method: pca, tsne, umap")
    n_components = Column(Integer, nullable=False, comment="Number of dimensions")
    parameters_used = Column(JSON, nullable=False, comment="Method-specific parameters")
    
    # Results
    coordinates = Column(JSON, nullable=False, comment="Embedding coordinates")
    explained_variance_ratio = Column(JSON, nullable=True, comment="PCA explained variance ratios")
    
    # Execution tracking
    execution_time = Column(Float, nullable=True, comment="Execution time in seconds")
    
    # Relationships
    dataset = relationship("Dataset", back_populates="embeddings")
    
    # Indexes
    __table_args__ = (
        Index('idx_embedding_dataset_method', 'dataset_id', 'method'),
        Index('idx_embedding_tenant_created', 'tenant_id', 'created_at'),
    )
    
    def __repr__(self):
        return f"<Embedding(id={self.id}, method={self.method}, n_components={self.n_components})>"


class GridSearch(Base, TimestampMixin, TenantMixin):
    """Grid search hyperparameter optimization runs."""
    
    __tablename__ = "grid_searches"
    
    # Primary key
    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    
    # Foreign keys
    dataset_id = Column(String(36), ForeignKey("datasets.id"), nullable=False)
    
    # Search configuration
    algorithm = Column(String(50), nullable=False, comment="Algorithm being optimized")
    parameter_grid = Column(JSON, nullable=False, comment="Parameter grid specification")
    cv_folds = Column(Integer, nullable=False, default=5, comment="Cross-validation folds")
    scoring_metric = Column(String(50), nullable=False, comment="Optimization metric")
    preprocessing_config = Column(JSON, nullable=False, comment="Preprocessing configuration")
    
    # Execution tracking
    status = Column(String(50), nullable=False, default="pending", comment="Search status")
    started_at = Column(DateTime, nullable=True, comment="Search start time")
    completed_at = Column(DateTime, nullable=True, comment="Search completion time")
    execution_time = Column(Float, nullable=True, comment="Total search time in seconds")
    
    # Results
    best_params = Column(JSON, nullable=True, comment="Best parameters found")
    best_score = Column(Float, nullable=True, comment="Best cross-validation score")
    cv_results = Column(JSON, nullable=True, comment="Complete cross-validation results")
    
    # Error handling
    error_message = Column(Text, nullable=True, comment="Error message if search failed")
    
    # Indexes
    __table_args__ = (
        Index('idx_grid_search_dataset_algorithm', 'dataset_id', 'algorithm'),
        Index('idx_grid_search_status', 'status'),
        Index('idx_grid_search_tenant_created', 'tenant_id', 'created_at'),
    )
    
    def __repr__(self):
        return f"<GridSearch(id={self.id}, algorithm={self.algorithm}, status={self.status})>"


class SystemMetrics(Base, TimestampMixin):
    """System performance and usage metrics."""
    
    __tablename__ = "system_metrics"
    
    # Primary key
    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    
    # Metric data
    metric_name = Column(String(100), nullable=False, comment="Metric identifier")
    metric_value = Column(Float, nullable=False, comment="Metric value")
    metric_unit = Column(String(50), nullable=True, comment="Unit of measurement")
    
    # Context
    tenant_id = Column(String(255), nullable=True, index=True, comment="Tenant identifier")
    resource_id = Column(String(36), nullable=True, comment="Related resource ID")
    resource_type = Column(String(50), nullable=True, comment="Type of related resource")
    
    # Additional metadata
    tags = Column(JSON, nullable=True, comment="Additional metric tags")
    
    # Indexes
    __table_args__ = (
        Index('idx_system_metrics_name_created', 'metric_name', 'created_at'),
        Index('idx_system_metrics_tenant_name', 'tenant_id', 'metric_name'),
    )
    
    def __repr__(self):
        return f"<SystemMetrics(metric_name={self.metric_name}, value={self.metric_value})>"


# Database utility functions
def create_database_engine(database_url: str, echo: bool = False):
    """Create SQLAlchemy engine with optimal settings."""
    return create_engine(
        database_url,
        echo=echo,
        pool_pre_ping=True,  # Validate connections before use
        pool_recycle=3600,   # Recycle connections after 1 hour
    )


def create_session_factory(engine):
    """Create sessionmaker factory."""
    return sessionmaker(autocommit=False, autoflush=False, bind=engine)


def init_database(engine):
    """Initialize database tables."""
    Base.metadata.create_all(bind=engine)


def get_db_session(SessionLocal: sessionmaker):
    """Database session dependency generator."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


# Export models for easy importing
__all__ = [
    "Base",
    "TimestampMixin",
    "TenantMixin", 
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
]
