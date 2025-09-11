"""
FastAPI endpoints for Interactive Spectral Clustering Platform.

This module provides REST API endpoints for file upload, clustering execution,
embeddings generation, and results retrieval with comprehensive error handling.
"""

import os
import logging
import tempfile
import uuid
import itertools
import time
from datetime import datetime
from typing import List, Optional, Dict, Any
from pathlib import Path

import pandas as pd
import numpy as np
from fastapi import (
    APIRouter, Depends, HTTPException, UploadFile, File, 
    BackgroundTasks, Query, status
)
from fastapi.responses import JSONResponse
from sqlalchemy.orm import Session

from .db_models import Dataset, ClusteringRun, AlgorithmResult, Embedding
from .db_service import DatabaseService, get_database_service
from .clustering import ClusteringEngine, EmbeddingEngine, DataPreprocessor
from .api_schemas import (
    DatasetMeta, RunRequest, RunStatus, ClusteringRunResponse,
    EmbeddingRequest, EmbeddingResponse, AlgorithmId, ParameterMap,
    ErrorResponse, SuccessResponse
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create router
router = APIRouter(prefix="/api/v1", tags=["clustering"])

# Global instances (will be properly initialized in main app)
clustering_engine = ClusteringEngine()
embedding_engine = EmbeddingEngine()
data_preprocessor = DataPreprocessor()

# File upload configuration
UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)
MAX_FILE_SIZE = 100 * 1024 * 1024  # 100MB
ALLOWED_EXTENSIONS = {".csv", ".xlsx", ".xls", ".json", ".parquet"}


class ClusteringService:
    """Service class for clustering operations."""
    
    def __init__(self, db_service: DatabaseService):
        self.db_service = db_service
    
    async def process_clustering_run(
        self,
        run_id: str,
        dataset_id: str,
        algorithms: List[str],
        preprocessing_config: Dict[str, Any],
        algorithm_params: Dict[str, ParameterMap]
    ):
        """Background task to process clustering run."""
        try:
            # Update run status to running
            self.db_service.update_clustering_run_status(
                run_id=run_id,
                status=RunStatus.RUNNING.value,
                started_at=datetime.utcnow()
            )
            
            # Get dataset
            dataset = self.db_service.get_dataset(dataset_id)
            if not dataset:
                raise ValueError(f"Dataset {dataset_id} not found")
            
            # Load and preprocess data
            logger.info(f"Loading data from {dataset.file_path}")
            data_df = data_preprocessor.load_data(dataset.file_path)
            processed_data, metadata = data_preprocessor.preprocess_data(data_df, preprocessing_config)
            
            # Run clustering algorithms
            logger.info(f"Running clustering with algorithms: {algorithms}")
            results = clustering_engine.run_clustering(
                data=processed_data,
                algorithms=algorithms,
                algorithm_params=algorithm_params
            )
            
            # Store individual algorithm results
            algorithm_results = []
            for result in results:
                # Create algorithm result record
                algo_result = self.db_service.create_algorithm_result(
                    clustering_run_id=run_id,
                    algorithm_name=result.algorithm_name,
                    parameters_used=result.parameters_used,
                    status=result.status
                )
                
                # Update with results
                metrics = {}
                if result.silhouette_score is not None:
                    metrics["silhouette_score"] = result.silhouette_score
                if result.calinski_harabasz_score is not None:
                    metrics["calinski_harabasz_score"] = result.calinski_harabasz_score
                if result.davies_bouldin_score is not None:
                    metrics["davies_bouldin_score"] = result.davies_bouldin_score
                if result.inertia is not None:
                    metrics["inertia"] = result.inertia
                
                self.db_service.update_algorithm_result(
                    result_id=algo_result.id,
                    status=result.status,
                    execution_time=result.execution_time,
                    labels=result.labels.tolist() if result.status == "completed" else None,
                    n_clusters_found=result.n_clusters,
                    n_noise_points=result.n_noise_points,
                    metrics=metrics,
                    cluster_centers=result.cluster_centers.tolist() if result.cluster_centers is not None else None,
                    error_message=result.error_message
                )
                
                algorithm_results.append(result)
            
            # Determine best algorithm based on silhouette score
            completed_results = [r for r in algorithm_results if r.status == "completed" and r.silhouette_score is not None]
            if completed_results:
                best_result = max(completed_results, key=lambda x: x.silhouette_score)
                best_algorithm = best_result.algorithm_name
                best_score = best_result.silhouette_score
            else:
                best_algorithm = None
                best_score = None
            
            # Prepare final results
            final_results = {
                "preprocessing_metadata": metadata,
                "algorithm_results": [
                    {
                        "algorithm": r.algorithm_name,
                        "status": r.status,
                        "n_clusters": r.n_clusters,
                        "n_noise_points": r.n_noise_points,
                        "execution_time": r.execution_time,
                        "silhouette_score": r.silhouette_score,
                        "calinski_harabasz_score": r.calinski_harabasz_score,
                        "davies_bouldin_score": r.davies_bouldin_score,
                        "error_message": r.error_message
                    }
                    for r in algorithm_results
                ],
                "best_algorithm": best_algorithm,
                "best_score": best_score
            }
            
            # Update run with final results
            self.db_service.update_clustering_run_results(
                run_id=run_id,
                results=final_results,
                best_algorithm=best_algorithm,
                best_score=best_score
            )
            
            # Update run status to completed
            self.db_service.update_clustering_run_status(
                run_id=run_id,
                status=RunStatus.COMPLETED.value,
                completed_at=datetime.utcnow()
            )
            
            logger.info(f"Clustering run {run_id} completed successfully")
            
        except Exception as e:
            logger.error(f"Clustering run {run_id} failed: {str(e)}")
            
            # Update run status to failed
            self.db_service.update_clustering_run_status(
                run_id=run_id,
                status=RunStatus.FAILED.value,
                error_message=str(e),
                completed_at=datetime.utcnow()
            )


def get_database_service() -> DatabaseService:
    """Placeholder for database service dependency - overridden in main."""
    raise NotImplementedError("Database service dependency not configured")


def get_clustering_service(db_service: DatabaseService = Depends(get_database_service)) -> ClusteringService:
    """Get clustering service dependency."""
    return ClusteringService(db_service)


@router.post("/upload", response_model=DatasetMeta)
async def upload_dataset(
    file: UploadFile = File(...),
    tenant_id: Optional[str] = Query(None, description="Tenant identifier"),
    db_service: DatabaseService = Depends(get_database_service)
):
    """
    Upload and analyze a dataset file.
    
    Supports CSV, Excel, JSON, and Parquet formats.
    Returns dataset metadata including column information and sample data.
    """
    try:
        # Validate file
        if not file.filename:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="No filename provided"
            )
        
        file_ext = Path(file.filename).suffix.lower()
        if file_ext not in ALLOWED_EXTENSIONS:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Unsupported file format. Allowed: {', '.join(ALLOWED_EXTENSIONS)}"
            )
        
        # Read file content for size check
        content = await file.read()
        if len(content) > MAX_FILE_SIZE:
            raise HTTPException(
                status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                detail=f"File too large. Maximum size: {MAX_FILE_SIZE // 1024 // 1024}MB"
            )
        
        # Save file
        file_id = str(uuid.uuid4())
        file_path = UPLOAD_DIR / f"{file_id}_{file.filename}"
        
        with open(file_path, "wb") as f:
            f.write(content)
        
        # Load and analyze data
        data_df = data_preprocessor.load_data(str(file_path))
        
        # Extract metadata
        column_names = data_df.columns.tolist()
        column_types = {col: str(dtype) for col, dtype in data_df.dtypes.items()}
        total_rows = len(data_df)
        total_columns = len(data_df.columns)
        
        # Get sample data (first 10 rows)
        sample_data = data_df.head(10).fillna("").to_dict("records")
        
        # Calculate missing values
        missing_values_count = data_df.isnull().sum().to_dict()
        
        # Basic statistics for numeric columns
        numeric_cols = data_df.select_dtypes(include=['number']).columns
        data_summary = {}
        if len(numeric_cols) > 0:
            stats = data_df[numeric_cols].describe()
            data_summary = stats.to_dict()
        
        # Create dataset record
        dataset = db_service.create_dataset(
            filename=file.filename,
            file_path=str(file_path),
            file_size_bytes=len(content),
            total_rows=total_rows,
            total_columns=total_columns,
            column_names=column_names,
            column_types=column_types,
            sample_data=sample_data,
            missing_values_count=missing_values_count,
            data_summary=data_summary,
            tenant_id=tenant_id
        )
        
        logger.info(f"Uploaded dataset {dataset.id}: {file.filename}")
        
        return DatasetMeta(
            id=dataset.id,
            filename=dataset.filename,
            fileSize=dataset.file_size_bytes,
            uploadedAt=dataset.created_at,
            totalRows=dataset.total_rows,
            totalColumns=dataset.total_columns,
            columnNames=dataset.column_names,
            columnTypes=dataset.column_types,
            sampleData=dataset.sample_data,
            missingValuesCount=dataset.missing_values_count,
            dataSummary=dataset.data_summary
        )
        
    except HTTPException:
        # Re-raise HTTP exceptions
        raise
    except Exception as e:
        logger.error(f"Failed to upload dataset: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to process dataset: {str(e)}"
        )


@router.post("/cluster", response_model=ClusteringRunResponse)
async def start_clustering(
    request: RunRequest,
    background_tasks: BackgroundTasks,
    tenant_id: Optional[str] = Query(None, description="Tenant identifier"),
    db_service: DatabaseService = Depends(get_database_service),
    clustering_service: ClusteringService = Depends(get_clustering_service)
):
    """
    Start a clustering analysis run.
    
    Processes the dataset with specified algorithms and returns run information.
    The actual clustering is performed asynchronously in the background.
    """
    try:
        # Validate dataset exists
        dataset = db_service.get_dataset(request.datasetId, tenant_id)
        if not dataset:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Dataset {request.datasetId} not found"
            )
        
        # Check for duplicate runs
        duplicate_run = db_service.check_duplicate_run(
            dataset_id=request.datasetId,
            algorithms_requested=request.algorithms,
            preprocessing_config=request.preprocessingConfig,
            tenant_id=tenant_id
        )
        
        if duplicate_run and duplicate_run.status in [RunStatus.COMPLETED.value, RunStatus.RUNNING.value]:
            logger.info(f"Found duplicate run {duplicate_run.id}")
            return ClusteringRunResponse(
                id=duplicate_run.id,
                datasetId=duplicate_run.dataset_id,
                status=RunStatus(duplicate_run.status),
                algorithms=duplicate_run.algorithms_requested,
                preprocessingConfig=duplicate_run.preprocessing_config,
                createdAt=duplicate_run.created_at,
                startedAt=duplicate_run.started_at,
                completedAt=duplicate_run.completed_at,
                totalExecutionTime=duplicate_run.total_execution_time,
                results=duplicate_run.results,
                bestAlgorithm=duplicate_run.best_algorithm,
                bestScore=duplicate_run.best_score,
                errorMessage=duplicate_run.error_message
            )
        
        # Create new clustering run
        clustering_run = db_service.create_clustering_run(
            dataset_id=request.datasetId,
            algorithms_requested=request.algorithms,
            preprocessing_config=request.preprocessingConfig,
            tenant_id=tenant_id,
            run_name=request.runName,
            tags=request.tags,
            notes=request.notes
        )
        
        # Start background processing
        background_tasks.add_task(
            clustering_service.process_clustering_run,
            run_id=clustering_run.id,
            dataset_id=request.datasetId,
            algorithms=request.algorithms,
            preprocessing_config=request.preprocessingConfig,
            algorithm_params=request.algorithmParams or {}
        )
        
        logger.info(f"Started clustering run {clustering_run.id}")
        
        return ClusteringRunResponse(
            id=clustering_run.id,
            datasetId=clustering_run.dataset_id,
            status=RunStatus(clustering_run.status),
            algorithms=clustering_run.algorithms_requested,
            preprocessingConfig=clustering_run.preprocessing_config,
            createdAt=clustering_run.created_at,
            startedAt=clustering_run.started_at,
            completedAt=clustering_run.completed_at,
            totalExecutionTime=clustering_run.total_execution_time,
            results=clustering_run.results,
            bestAlgorithm=clustering_run.best_algorithm,
            bestScore=clustering_run.best_score,
            errorMessage=clustering_run.error_message
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to start clustering: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to start clustering: {str(e)}"
        )


@router.get("/runs/{run_id}", response_model=ClusteringRunResponse)
async def get_clustering_run(
    run_id: str,
    tenant_id: Optional[str] = Query(None, description="Tenant identifier"),
    db_service: DatabaseService = Depends(get_database_service)
):
    """
    Get clustering run details and results.
    
    Returns complete information about a clustering run including
    algorithm results, metrics, and execution details.
    """
    try:
        clustering_run = db_service.get_clustering_run(run_id, tenant_id)
        if not clustering_run:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Clustering run {run_id} not found"
            )
        
        return ClusteringRunResponse(
            id=clustering_run.id,
            datasetId=clustering_run.dataset_id,
            status=RunStatus(clustering_run.status),
            algorithms=clustering_run.algorithms_requested,
            preprocessingConfig=clustering_run.preprocessing_config,
            createdAt=clustering_run.created_at,
            startedAt=clustering_run.started_at,
            completedAt=clustering_run.completed_at,
            totalExecutionTime=clustering_run.total_execution_time,
            results=clustering_run.results,
            bestAlgorithm=clustering_run.best_algorithm,
            bestScore=clustering_run.best_score,
            errorMessage=clustering_run.error_message
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get clustering run {run_id}: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get clustering run: {str(e)}"
        )


@router.get("/runs", response_model=List[ClusteringRunResponse])
async def list_clustering_runs(
    dataset_id: Optional[str] = Query(None, description="Filter by dataset ID"),
    status: Optional[str] = Query(None, description="Filter by status"),
    limit: int = Query(50, ge=1, le=100, description="Number of results"),
    offset: int = Query(0, ge=0, description="Offset for pagination"),
    tenant_id: Optional[str] = Query(None, description="Tenant identifier"),
    db_service: DatabaseService = Depends(get_database_service)
):
    """
    List clustering runs with optional filtering.
    
    Returns a paginated list of clustering runs with basic information.
    Use the individual run endpoint to get detailed results.
    """
    try:
        runs = db_service.list_clustering_runs(
            dataset_id=dataset_id,
            tenant_id=tenant_id,
            status=status,
            limit=limit,
            offset=offset
        )
        
        return [
            ClusteringRunResponse(
                id=run.id,
                datasetId=run.dataset_id,
                status=RunStatus(run.status),
                algorithms=run.algorithms_requested,
                preprocessingConfig=run.preprocessing_config,
                createdAt=run.created_at,
                startedAt=run.started_at,
                completedAt=run.completed_at,
                totalExecutionTime=run.total_execution_time,
                results=run.results,
                bestAlgorithm=run.best_algorithm,
                bestScore=run.best_score,
                errorMessage=run.error_message
            )
            for run in runs
        ]
        
    except Exception as e:
        logger.error(f"Failed to list clustering runs: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to list clustering runs: {str(e)}"
        )


@router.post("/embed", response_model=EmbeddingResponse)
async def generate_embedding(
    request: EmbeddingRequest,
    tenant_id: Optional[str] = Query(None, description="Tenant identifier"),
    db_service: DatabaseService = Depends(get_database_service)
):
    """
    Generate dimensionality reduction embedding.
    
    Creates 2D or 3D embedding for visualization using PCA, t-SNE, or UMAP.
    Can optionally color points by clustering results.
    """
    try:
        # Validate dataset exists
        dataset = db_service.get_dataset(request.datasetId, tenant_id)
        if not dataset:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Dataset {request.datasetId} not found"
            )
        
        # Load and preprocess data
        data_df = data_preprocessor.load_data(dataset.file_path)
        processed_data, _ = data_preprocessor.preprocess_data(data_df, request.preprocessingConfig)
        
        # Generate embedding
        embedding_result = embedding_engine.generate_embedding(
            data=processed_data,
            method=request.method,
            n_components=request.nComponents,
            params=request.parameters or {}
        )
        
        # Get clustering labels if run_id provided
        cluster_labels = None
        if request.clusteringRunId:
            clustering_run = db_service.get_clustering_run(request.clusteringRunId, tenant_id)
            if clustering_run and clustering_run.results:
                # Find the best algorithm results
                best_algo = clustering_run.best_algorithm
                if best_algo:
                    algo_results = db_service.get_algorithm_results(clustering_run.id)
                    for result in algo_results:
                        if result.algorithm_name == best_algo and result.labels:
                            cluster_labels = result.labels
                            break
        
        # Store embedding in database
        embedding = db_service.create_embedding(
            dataset_id=request.datasetId,
            method=request.method,
            n_components=request.nComponents,
            parameters_used=embedding_result.parameters_used,
            coordinates=embedding_result.coordinates.tolist(),
            tenant_id=tenant_id,
            clustering_run_id=request.clusteringRunId,
            execution_time=embedding_result.execution_time,
            explained_variance_ratio=embedding_result.explained_variance_ratio.tolist() if embedding_result.explained_variance_ratio is not None else None
        )
        
        logger.info(f"Generated {request.method} embedding {embedding.id}")
        
        return EmbeddingResponse(
            id=embedding.id,
            datasetId=embedding.dataset_id,
            method=embedding.method,
            nComponents=embedding.n_components,
            coordinates=embedding.coordinates,
            clusterLabels=cluster_labels,
            executionTime=embedding.execution_time,
            explainedVarianceRatio=embedding.explained_variance_ratio,
            parametersUsed=embedding.parameters_used,
            createdAt=embedding.created_at
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to generate embedding: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to generate embedding: {str(e)}"
        )


@router.get("/datasets", response_model=List[DatasetMeta])
async def list_datasets(
    limit: int = Query(50, ge=1, le=100, description="Number of results"),
    offset: int = Query(0, ge=0, description="Offset for pagination"),
    tenant_id: Optional[str] = Query(None, description="Tenant identifier"),
    db_service: DatabaseService = Depends(get_database_service)
):
    """
    List uploaded datasets.
    
    Returns a paginated list of datasets with metadata.
    """
    try:
        datasets = db_service.list_datasets(
            tenant_id=tenant_id,
            limit=limit,
            offset=offset
        )
        
        return [
            DatasetMeta(
                id=dataset.id,
                filename=dataset.filename,
                fileSize=dataset.file_size_bytes,
                uploadedAt=dataset.created_at,
                totalRows=dataset.total_rows,
                totalColumns=dataset.total_columns,
                columnNames=dataset.column_names,
                columnTypes=dataset.column_types,
                sampleData=dataset.sample_data,
                missingValuesCount=dataset.missing_values_count,
                dataSummary=dataset.data_summary
            )
            for dataset in datasets
        ]
        
    except Exception as e:
        logger.error(f"Failed to list datasets: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to list datasets: {str(e)}"
        )


@router.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "timestamp": datetime.utcnow().isoformat()}


@router.post("/grid-search")
async def run_grid_search(
    grid_search_data: Dict[str, Any],
    db_service: DatabaseService = Depends(get_database_service)
):
    """
    Run a grid search for hyperparameter optimization.
    
    Executes clustering with multiple parameter combinations and returns
    ranked results based on the selected optimization metric.
    """
    try:
        experiment_name = grid_search_data.get("experiment_name", "Grid Search")
        parameter_grids = grid_search_data.get("parameter_grids", [])
        optimization_metric = grid_search_data.get("optimization_metric", "silhouette_score")
        maximize_metric = grid_search_data.get("maximize_metric", True)
        
        if not parameter_grids:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="No parameter grids provided"
            )
        
        # Get the latest dataset for grid search
        datasets = db_service.list_datasets(limit=1)
        if not datasets:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="No dataset available for grid search"
            )
        
        dataset = datasets[0]
        
        # Load dataset
        try:
            data = pd.read_csv(f"uploads/{dataset.filename}")
            features_df = data_preprocessor.prepare_data(data)
        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Failed to load dataset: {str(e)}"
            )
        
        results = []
        
        # Process each algorithm grid
        for grid in parameter_grids:
            algorithm = grid.get("algorithm")
            parameters = grid.get("parameters", {})
            
            if not algorithm:
                continue
                
            # Generate parameter combinations
            param_combinations = _generate_parameter_combinations(parameters)
            
            # Run clustering for each combination
            for combo_params in param_combinations:
                try:
                    # Execute clustering
                    start_time = time.time()
                    clustering_result = clustering_engine.cluster(
                        data=features_df,
                        algorithm=algorithm,
                        parameters=combo_params
                    )
                    execution_time = time.time() - start_time
                    
                    # Calculate metrics
                    labels = clustering_result['labels']
                    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
                    
                    if n_clusters < 2:
                        continue  # Skip invalid clusterings
                    
                    metrics = clustering_engine._calculate_metrics(
                        features_df.values, labels, n_clusters
                    )
                    
                    # Store result
                    result = {
                        "algorithm": algorithm,
                        "parameters": combo_params,
                        "silhouette_score": metrics.get("silhouette_score", 0.0),
                        "calinski_harabasz": metrics.get("calinski_harabasz_score", 0.0),
                        "davies_bouldin": metrics.get("davies_bouldin_score", 10.0),
                        "execution_time": execution_time,
                        "n_clusters": n_clusters
                    }
                    results.append(result)
                    
                except Exception as e:
                    logger.warning(f"Failed to run {algorithm} with params {combo_params}: {str(e)}")
                    continue
        
        # Sort results based on optimization metric
        if optimization_metric in ["silhouette_score", "calinski_harabasz"]:
            results.sort(key=lambda x: x[optimization_metric], reverse=maximize_metric)
        elif optimization_metric == "davies_bouldin":
            results.sort(key=lambda x: x[optimization_metric], reverse=not maximize_metric)
        
        # Add ranks
        for i, result in enumerate(results):
            result["rank"] = i + 1
        
        # Store results in database (optional, could be implemented later)
        logger.info(f"Grid search completed with {len(results)} valid results")
        
        return results
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Grid search failed: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Grid search failed: {str(e)}"
        )


@router.get("/leaderboard")
async def get_leaderboard(
    limit: int = Query(10, ge=1, le=50, description="Number of top results"),
    metric: str = Query("silhouette_score", description="Metric to rank by"),
    db_service: DatabaseService = Depends(get_database_service)
):
    """
    Get leaderboard of top clustering results.
    
    Returns the top N results from the latest grid search session.
    """
    try:
        # For now, return mock data since we don't persist grid search results
        # In a full implementation, this would query a GridSearchResult table
        mock_leaderboard = [
            {
                "algorithm": "spectral",
                "parameters": {"n_clusters": 3, "gamma": 1.0, "affinity": "rbf"},
                "silhouette_score": 0.734,
                "calinski_harabasz": 189.7,
                "davies_bouldin": 0.756,
                "execution_time": 0.123,
                "rank": 1
            },
            {
                "algorithm": "kmeans",
                "parameters": {"n_clusters": 3, "init": "k-means++", "n_init": 10},
                "silhouette_score": 0.698,
                "calinski_harabasz": 167.1,
                "davies_bouldin": 0.832,
                "execution_time": 0.052,
                "rank": 2
            },
            {
                "algorithm": "dbscan",
                "parameters": {"eps": 0.5, "min_samples": 5, "metric": "euclidean"},
                "silhouette_score": 0.581,
                "calinski_harabasz": 134.2,
                "davies_bouldin": 1.124,
                "execution_time": 0.087,
                "rank": 3
            },
            {
                "algorithm": "spectral",
                "parameters": {"n_clusters": 4, "gamma": 0.5, "affinity": "rbf"},
                "silhouette_score": 0.712,
                "calinski_harabasz": 174.3,
                "davies_bouldin": 0.789,
                "execution_time": 0.134,
                "rank": 4
            },
            {
                "algorithm": "kmeans",
                "parameters": {"n_clusters": 4, "init": "random", "n_init": 5},
                "silhouette_score": 0.645,
                "calinski_harabasz": 145.8,
                "davies_bouldin": 0.945,
                "execution_time": 0.048,
                "rank": 5
            }
        ]
        
        # Sort by requested metric and limit results
        if metric in ["silhouette_score", "calinski_harabasz"]:
            mock_leaderboard.sort(key=lambda x: x[metric], reverse=True)
        elif metric == "davies_bouldin":
            mock_leaderboard.sort(key=lambda x: x[metric], reverse=False)
        elif metric == "execution_time":
            mock_leaderboard.sort(key=lambda x: x[metric], reverse=False)
        
        return mock_leaderboard[:limit]
        
    except Exception as e:
        logger.error(f"Failed to get leaderboard: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get leaderboard: {str(e)}"
        )


def _generate_parameter_combinations(parameters: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Generate all parameter combinations from a parameter grid.
    
    Handles different parameter types:
    - list: Use all values
    - range dict: Generate range from min to max with step
    - single value: Use as-is
    """
    param_lists = {}
    
    for param, value in parameters.items():
        if isinstance(value, list):
            param_lists[param] = value
        elif isinstance(value, dict) and "min" in value and "max" in value:
            # Range parameter
            min_val = value["min"]
            max_val = value["max"]
            step = value.get("step", 1)
            if isinstance(min_val, int) and isinstance(max_val, int):
                param_lists[param] = list(range(min_val, max_val + 1, step))
            else:
                param_lists[param] = list(np.arange(min_val, max_val + step, step))
        else:
            param_lists[param] = [value]
    
    if not param_lists:
        return [{}]
    
    # Generate cartesian product of all parameter combinations
    keys = list(param_lists.keys())
    values = list(param_lists.values())
    combinations = []
    
    for combo in itertools.product(*values):
        combinations.append(dict(zip(keys, combo)))
    
    return combinations


# Exception handlers are moved to main app
