"""
Database service layer for Interactive Spectral Clustering Platform.

This module provides high-level database operations and business logic
for managing datasets, clustering runs, and results.
"""

import hashlib
import json
import logging
from datetime import datetime
from typing import List, Optional, Dict, Any, Tuple
from uuid import uuid4

from sqlalchemy.orm import Session
from sqlalchemy import desc, and_, or_
from sqlalchemy.exc import SQLAlchemyError

from .db_models import (
    Dataset, ClusteringRun, AlgorithmResult, Embedding, 
    GridSearch, SystemMetrics
)
from .api_schemas import (
    DatasetMeta, RunRequest, RunStatus, AlgorithmId,
    ClusteringRun, EmbeddingRequest
)

logger = logging.getLogger(__name__)


class DatabaseService:
    """High-level database operations service."""
    
    def __init__(self, db_session: Session):
        self.db = db_session
    
    # Dataset operations
    
    def create_dataset(
        self,
        filename: str,
        file_path: str,
        file_size_bytes: int,
        total_rows: int,
        total_columns: int,
        column_names: List[str],
        tenant_id: Optional[str] = None,
        column_types: Optional[Dict[str, str]] = None,
        sample_data: Optional[List[Dict]] = None,
        missing_values_count: Optional[Dict[str, int]] = None,
        data_summary: Optional[Dict[str, Any]] = None
    ) -> Dataset:
        """Create a new dataset record."""
        try:
            dataset = Dataset(
                id=str(uuid4()),
                filename=filename,
                file_path=file_path,
                file_size_bytes=file_size_bytes,
                total_rows=total_rows,
                total_columns=total_columns,
                column_names=column_names,
                column_types=column_types,
                sample_data=sample_data,
                missing_values_count=missing_values_count,
                data_summary=data_summary,
                tenant_id=tenant_id,
                is_processed=False
            )
            
            self.db.add(dataset)
            self.db.commit()
            self.db.refresh(dataset)
            
            logger.info(f"Created dataset {dataset.id} with {total_rows} rows")
            return dataset
            
        except SQLAlchemyError as e:
            self.db.rollback()
            logger.error(f"Failed to create dataset: {str(e)}")
            raise
    
    def get_dataset(self, dataset_id: str, tenant_id: Optional[str] = None) -> Optional[Dataset]:
        """Get dataset by ID with optional tenant filtering."""
        query = self.db.query(Dataset).filter(Dataset.id == dataset_id)
        
        if tenant_id is not None:
            query = query.filter(Dataset.tenant_id == tenant_id)
        
        return query.first()
    
    def list_datasets(
        self, 
        tenant_id: Optional[str] = None,
        limit: int = 50,
        offset: int = 0
    ) -> List[Dataset]:
        """List datasets with optional tenant filtering."""
        query = self.db.query(Dataset)
        
        if tenant_id is not None:
            query = query.filter(Dataset.tenant_id == tenant_id)
        
        return query.order_by(desc(Dataset.created_at)).offset(offset).limit(limit).all()
    
    def update_dataset_processing_status(
        self,
        dataset_id: str,
        is_processed: bool,
        processing_error: Optional[str] = None
    ) -> bool:
        """Update dataset processing status."""
        try:
            dataset = self.db.query(Dataset).filter(Dataset.id == dataset_id).first()
            if not dataset:
                return False
            
            dataset.is_processed = is_processed
            dataset.processing_error = processing_error
            dataset.updated_at = datetime.utcnow()
            
            self.db.commit()
            return True
            
        except SQLAlchemyError as e:
            self.db.rollback()
            logger.error(f"Failed to update dataset {dataset_id}: {str(e)}")
            return False
    
    # Clustering run operations
    
    def create_clustering_run(
        self,
        dataset_id: str,
        algorithms_requested: List[str],
        preprocessing_config: Dict[str, Any],
        tenant_id: Optional[str] = None,
        run_name: Optional[str] = None,
        tags: Optional[List[str]] = None,
        notes: Optional[str] = None
    ) -> ClusteringRun:
        """Create a new clustering run."""
        try:
            # Calculate parameters hash for deduplication
            params_str = json.dumps({
                "algorithms": sorted(algorithms_requested),
                "preprocessing": preprocessing_config
            }, sort_keys=True)
            parameters_hash = hashlib.sha256(params_str.encode()).hexdigest()
            
            clustering_run = ClusteringRun(
                id=str(uuid4()),
                dataset_id=dataset_id,
                run_name=run_name,
                status=RunStatus.PENDING.value,
                algorithms_requested=algorithms_requested,
                preprocessing_config=preprocessing_config,
                parameters_hash=parameters_hash,
                tags=tags,
                notes=notes,
                tenant_id=tenant_id
            )
            
            self.db.add(clustering_run)
            self.db.commit()
            self.db.refresh(clustering_run)
            
            logger.info(f"Created clustering run {clustering_run.id} for dataset {dataset_id}")
            return clustering_run
            
        except SQLAlchemyError as e:
            self.db.rollback()
            logger.error(f"Failed to create clustering run: {str(e)}")
            raise
    
    def get_clustering_run(
        self, 
        run_id: str, 
        tenant_id: Optional[str] = None
    ) -> Optional[ClusteringRun]:
        """Get clustering run by ID with optional tenant filtering."""
        query = self.db.query(ClusteringRun).filter(ClusteringRun.id == run_id)
        
        if tenant_id is not None:
            query = query.filter(ClusteringRun.tenant_id == tenant_id)
        
        return query.first()
    
    def list_clustering_runs(
        self,
        dataset_id: Optional[str] = None,
        tenant_id: Optional[str] = None,
        status: Optional[str] = None,
        limit: int = 50,
        offset: int = 0
    ) -> List[ClusteringRun]:
        """List clustering runs with filtering options."""
        query = self.db.query(ClusteringRun)
        
        if dataset_id:
            query = query.filter(ClusteringRun.dataset_id == dataset_id)
        
        if tenant_id is not None:
            query = query.filter(ClusteringRun.tenant_id == tenant_id)
        
        if status:
            query = query.filter(ClusteringRun.status == status)
        
        return query.order_by(desc(ClusteringRun.created_at)).offset(offset).limit(limit).all()
    
    def update_clustering_run_status(
        self,
        run_id: str,
        status: str,
        error_message: Optional[str] = None,
        error_details: Optional[Dict[str, Any]] = None,
        started_at: Optional[datetime] = None,
        completed_at: Optional[datetime] = None
    ) -> bool:
        """Update clustering run status and timing."""
        try:
            run = self.db.query(ClusteringRun).filter(ClusteringRun.id == run_id).first()
            if not run:
                return False
            
            run.status = status
            if error_message:
                run.error_message = error_message
            if error_details:
                run.error_details = error_details
            if started_at:
                run.started_at = started_at
            if completed_at:
                run.completed_at = completed_at
                if run.started_at:
                    run.total_execution_time = (completed_at - run.started_at).total_seconds()
            
            run.updated_at = datetime.utcnow()
            
            self.db.commit()
            return True
            
        except SQLAlchemyError as e:
            self.db.rollback()
            logger.error(f"Failed to update clustering run {run_id}: {str(e)}")
            return False
    
    def update_clustering_run_results(
        self,
        run_id: str,
        results: Dict[str, Any],
        best_algorithm: Optional[str] = None,
        best_score: Optional[float] = None
    ) -> bool:
        """Update clustering run with final results."""
        try:
            run = self.db.query(ClusteringRun).filter(ClusteringRun.id == run_id).first()
            if not run:
                return False
            
            run.results = results
            run.best_algorithm = best_algorithm
            run.best_score = best_score
            run.updated_at = datetime.utcnow()
            
            self.db.commit()
            return True
            
        except SQLAlchemyError as e:
            self.db.rollback()
            logger.error(f"Failed to update clustering run results {run_id}: {str(e)}")
            return False
    
    # Algorithm result operations
    
    def create_algorithm_result(
        self,
        clustering_run_id: str,
        algorithm_name: str,
        parameters_used: Dict[str, Any],
        status: str = "pending"
    ) -> AlgorithmResult:
        """Create a new algorithm result."""
        try:
            result = AlgorithmResult(
                id=str(uuid4()),
                clustering_run_id=clustering_run_id,
                algorithm_name=algorithm_name,
                parameters_used=parameters_used,
                status=status
            )
            
            self.db.add(result)
            self.db.commit()
            self.db.refresh(result)
            
            return result
            
        except SQLAlchemyError as e:
            self.db.rollback()
            logger.error(f"Failed to create algorithm result: {str(e)}")
            raise
    
    def update_algorithm_result(
        self,
        result_id: str,
        status: str,
        execution_time: Optional[float] = None,
        labels: Optional[List[int]] = None,
        n_clusters_found: Optional[int] = None,
        n_noise_points: Optional[int] = None,
        metrics: Optional[Dict[str, float]] = None,
        cluster_centers: Optional[List[List[float]]] = None,
        error_message: Optional[str] = None
    ) -> bool:
        """Update algorithm result with execution details."""
        try:
            result = self.db.query(AlgorithmResult).filter(AlgorithmResult.id == result_id).first()
            if not result:
                return False
            
            result.status = status
            if execution_time is not None:
                result.execution_time = execution_time
            if labels is not None:
                result.labels = labels
            if n_clusters_found is not None:
                result.n_clusters_found = n_clusters_found
            if n_noise_points is not None:
                result.n_noise_points = n_noise_points
            if cluster_centers is not None:
                result.cluster_centers = cluster_centers
            if error_message:
                result.error_message = error_message
            
            # Update individual metrics
            if metrics:
                for metric_name, value in metrics.items():
                    if hasattr(result, metric_name):
                        setattr(result, metric_name, value)
                    else:
                        # Store in additional_metrics
                        if not result.additional_metrics:
                            result.additional_metrics = {}
                        result.additional_metrics[metric_name] = value
            
            result.updated_at = datetime.utcnow()
            
            self.db.commit()
            return True
            
        except SQLAlchemyError as e:
            self.db.rollback()
            logger.error(f"Failed to update algorithm result {result_id}: {str(e)}")
            return False
    
    def get_algorithm_results(self, clustering_run_id: str) -> List[AlgorithmResult]:
        """Get all algorithm results for a clustering run."""
        return self.db.query(AlgorithmResult).filter(
            AlgorithmResult.clustering_run_id == clustering_run_id
        ).order_by(AlgorithmResult.created_at).all()
    
    # Embedding operations
    
    def create_embedding(
        self,
        dataset_id: str,
        method: str,
        n_components: int,
        parameters_used: Dict[str, Any],
        coordinates: List[List[float]],
        tenant_id: Optional[str] = None,
        clustering_run_id: Optional[str] = None,
        execution_time: Optional[float] = None,
        explained_variance_ratio: Optional[List[float]] = None
    ) -> Embedding:
        """Create a new embedding."""
        try:
            embedding = Embedding(
                id=str(uuid4()),
                dataset_id=dataset_id,
                clustering_run_id=clustering_run_id,
                method=method,
                n_components=n_components,
                parameters_used=parameters_used,
                coordinates=coordinates,
                execution_time=execution_time,
                explained_variance_ratio=explained_variance_ratio,
                tenant_id=tenant_id
            )
            
            self.db.add(embedding)
            self.db.commit()
            self.db.refresh(embedding)
            
            logger.info(f"Created {method} embedding {embedding.id} for dataset {dataset_id}")
            return embedding
            
        except SQLAlchemyError as e:
            self.db.rollback()
            logger.error(f"Failed to create embedding: {str(e)}")
            raise
    
    def get_embeddings(
        self,
        dataset_id: Optional[str] = None,
        clustering_run_id: Optional[str] = None,
        method: Optional[str] = None,
        tenant_id: Optional[str] = None
    ) -> List[Embedding]:
        """Get embeddings with filtering options."""
        query = self.db.query(Embedding)
        
        if dataset_id:
            query = query.filter(Embedding.dataset_id == dataset_id)
        
        if clustering_run_id:
            query = query.filter(Embedding.clustering_run_id == clustering_run_id)
        
        if method:
            query = query.filter(Embedding.method == method)
        
        if tenant_id is not None:
            query = query.filter(Embedding.tenant_id == tenant_id)
        
        return query.order_by(desc(Embedding.created_at)).all()
    
    # System metrics operations
    
    def record_metric(
        self,
        metric_name: str,
        metric_value: float,
        metric_unit: Optional[str] = None,
        tenant_id: Optional[str] = None,
        resource_id: Optional[str] = None,
        resource_type: Optional[str] = None,
        tags: Optional[Dict[str, Any]] = None
    ) -> SystemMetrics:
        """Record a system metric."""
        try:
            metric = SystemMetrics(
                id=str(uuid4()),
                metric_name=metric_name,
                metric_value=metric_value,
                metric_unit=metric_unit,
                tenant_id=tenant_id,
                resource_id=resource_id,
                resource_type=resource_type,
                tags=tags
            )
            
            self.db.add(metric)
            self.db.commit()
            self.db.refresh(metric)
            
            return metric
            
        except SQLAlchemyError as e:
            self.db.rollback()
            logger.error(f"Failed to record metric {metric_name}: {str(e)}")
            raise
    
    def get_metrics(
        self,
        metric_name: Optional[str] = None,
        tenant_id: Optional[str] = None,
        resource_id: Optional[str] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        limit: int = 1000
    ) -> List[SystemMetrics]:
        """Get system metrics with filtering options."""
        query = self.db.query(SystemMetrics)
        
        if metric_name:
            query = query.filter(SystemMetrics.metric_name == metric_name)
        
        if tenant_id is not None:
            query = query.filter(SystemMetrics.tenant_id == tenant_id)
        
        if resource_id:
            query = query.filter(SystemMetrics.resource_id == resource_id)
        
        if start_time:
            query = query.filter(SystemMetrics.created_at >= start_time)
        
        if end_time:
            query = query.filter(SystemMetrics.created_at <= end_time)
        
        return query.order_by(desc(SystemMetrics.created_at)).limit(limit).all()
    
    # Utility methods
    
    def check_duplicate_run(
        self,
        dataset_id: str,
        algorithms_requested: List[str],
        preprocessing_config: Dict[str, Any],
        tenant_id: Optional[str] = None
    ) -> Optional[ClusteringRun]:
        """Check for duplicate clustering run based on parameters hash."""
        params_str = json.dumps({
            "algorithms": sorted(algorithms_requested),
            "preprocessing": preprocessing_config
        }, sort_keys=True)
        parameters_hash = hashlib.sha256(params_str.encode()).hexdigest()
        
        query = self.db.query(ClusteringRun).filter(
            and_(
                ClusteringRun.dataset_id == dataset_id,
                ClusteringRun.parameters_hash == parameters_hash,
                ClusteringRun.status.in_([RunStatus.COMPLETED.value, RunStatus.RUNNING.value])
            )
        )
        
        if tenant_id is not None:
            query = query.filter(ClusteringRun.tenant_id == tenant_id)
        
        return query.first()
    
    def get_run_summary_stats(self, tenant_id: Optional[str] = None) -> Dict[str, Any]:
        """Get summary statistics for runs."""
        query = self.db.query(ClusteringRun)
        
        if tenant_id is not None:
            query = query.filter(ClusteringRun.tenant_id == tenant_id)
        
        total_runs = query.count()
        completed_runs = query.filter(ClusteringRun.status == RunStatus.COMPLETED.value).count()
        failed_runs = query.filter(ClusteringRun.status == RunStatus.FAILED.value).count()
        running_runs = query.filter(ClusteringRun.status == RunStatus.RUNNING.value).count()
        
        return {
            "total_runs": total_runs,
            "completed_runs": completed_runs,
            "failed_runs": failed_runs,
            "running_runs": running_runs,
            "success_rate": completed_runs / total_runs if total_runs > 0 else 0
        }


# Dependency injection helpers
def get_database_service(db: Session) -> DatabaseService:
    """Get database service instance."""
    return DatabaseService(db)
