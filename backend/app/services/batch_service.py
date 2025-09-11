"""
Batch processing service for Interactive Spectral Clustering Platform.

Provides queuing and batch execution of multiple clustering runs
across different datasets with progress tracking and result management.
"""

import asyncio
import uuid
from datetime import datetime
from typing import Dict, Any, List, Optional, Callable
from enum import Enum
from dataclasses import dataclass, asdict
import logging
from concurrent.futures import ThreadPoolExecutor
import json

logger = logging.getLogger(__name__)

class BatchStatus(Enum):
    """Batch job status enumeration."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"

class JobStatus(Enum):
    """Individual job status enumeration."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"

@dataclass
class BatchJob:
    """Individual job within a batch."""
    id: str
    dataset_id: str
    algorithm: str
    parameters: Dict[str, Any]
    status: JobStatus = JobStatus.PENDING
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    duration_seconds: Optional[float] = None

@dataclass
class BatchRequest:
    """Batch processing request configuration."""
    name: str
    description: Optional[str] = None
    jobs: List[Dict[str, Any]] = None  # List of job configurations
    
    # Execution settings
    max_parallel_jobs: int = 3
    stop_on_error: bool = False
    timeout_minutes: int = 60
    
    # Notification settings
    notify_on_completion: bool = False
    email: Optional[str] = None

@dataclass
class BatchExecution:
    """Batch execution tracking."""
    id: str
    name: str
    description: Optional[str]
    status: BatchStatus
    jobs: List[BatchJob]
    
    # Execution metadata
    created_at: datetime
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    
    # Progress tracking
    total_jobs: int = 0
    completed_jobs: int = 0
    failed_jobs: int = 0
    
    # Settings
    max_parallel_jobs: int = 3
    stop_on_error: bool = False
    timeout_minutes: int = 60
    
    # Results
    results: Dict[str, Any] = None

class BatchProcessor:
    """Service for managing and executing batch clustering jobs."""
    
    def __init__(self, clustering_service, dataset_service):
        """
        Initialize batch processor.
        
        Args:
            clustering_service: Clustering service instance
            dataset_service: Dataset service instance
        """
        self.clustering_service = clustering_service
        self.dataset_service = dataset_service
        self.batches: Dict[str, BatchExecution] = {}
        self.executor = ThreadPoolExecutor(max_workers=10)
        self._running_batches: Dict[str, asyncio.Task] = {}
    
    def create_batch(self, request: BatchRequest) -> str:
        """
        Create a new batch job.
        
        Args:
            request: Batch request configuration
            
        Returns:
            Batch ID string
        """
        try:
            batch_id = str(uuid.uuid4())
            
            # Create jobs from request
            jobs = []
            if request.jobs:
                for i, job_config in enumerate(request.jobs):
                    job = BatchJob(
                        id=f"{batch_id}_job_{i}",
                        dataset_id=job_config["dataset_id"],
                        algorithm=job_config["algorithm"],
                        parameters=job_config.get("parameters", {})
                    )
                    jobs.append(job)
            
            # Create batch execution
            batch = BatchExecution(
                id=batch_id,
                name=request.name,
                description=request.description,
                status=BatchStatus.PENDING,
                jobs=jobs,
                created_at=datetime.now(),
                total_jobs=len(jobs),
                max_parallel_jobs=request.max_parallel_jobs,
                stop_on_error=request.stop_on_error,
                timeout_minutes=request.timeout_minutes,
                results={}
            )
            
            self.batches[batch_id] = batch
            
            logger.info(f"Created batch {batch_id} with {len(jobs)} jobs")
            return batch_id
            
        except Exception as e:
            logger.error(f"Error creating batch: {str(e)}")
            raise ValueError(f"Failed to create batch: {str(e)}")
    
    async def execute_batch(self, batch_id: str) -> BatchExecution:
        """
        Execute a batch of clustering jobs.
        
        Args:
            batch_id: Batch identifier
            
        Returns:
            Updated batch execution
        """
        if batch_id not in self.batches:
            raise ValueError(f"Batch {batch_id} not found")
        
        batch = self.batches[batch_id]
        
        if batch.status != BatchStatus.PENDING:
            raise ValueError(f"Batch {batch_id} is not in pending status")
        
        try:
            # Start batch execution
            batch.status = BatchStatus.RUNNING
            batch.started_at = datetime.now()
            
            logger.info(f"Starting batch execution {batch_id} with {len(batch.jobs)} jobs")
            
            # Create execution task
            task = asyncio.create_task(self._execute_batch_jobs(batch))
            self._running_batches[batch_id] = task
            
            # Wait for completion
            await task
            
            # Clean up
            if batch_id in self._running_batches:
                del self._running_batches[batch_id]
            
            return batch
            
        except Exception as e:
            batch.status = BatchStatus.FAILED
            batch.completed_at = datetime.now()
            logger.error(f"Batch {batch_id} failed: {str(e)}")
            raise
    
    async def _execute_batch_jobs(self, batch: BatchExecution):
        """Execute all jobs in a batch with parallel processing."""
        
        try:
            # Semaphore to limit parallel jobs
            semaphore = asyncio.Semaphore(batch.max_parallel_jobs)
            
            async def execute_job(job: BatchJob):
                """Execute a single job."""
                async with semaphore:
                    await self._execute_single_job(batch, job)
            
            # Execute all jobs
            tasks = [execute_job(job) for job in batch.jobs]
            await asyncio.gather(*tasks, return_exceptions=True)
            
            # Update batch status
            batch.completed_at = datetime.now()
            
            if batch.failed_jobs > 0 and batch.stop_on_error:
                batch.status = BatchStatus.FAILED
            else:
                batch.status = BatchStatus.COMPLETED
            
            # Generate batch summary
            batch.results = self._generate_batch_summary(batch)
            
            logger.info(
                f"Batch {batch.id} completed: {batch.completed_jobs} successful, "
                f"{batch.failed_jobs} failed"
            )
            
        except Exception as e:
            batch.status = BatchStatus.FAILED
            batch.completed_at = datetime.now()
            logger.error(f"Error executing batch {batch.id}: {str(e)}")
            raise
    
    async def _execute_single_job(self, batch: BatchExecution, job: BatchJob):
        """Execute a single clustering job."""
        
        try:
            job.status = JobStatus.RUNNING
            job.start_time = datetime.now()
            
            logger.debug(f"Starting job {job.id} for dataset {job.dataset_id}")
            
            # Load dataset
            dataset = await self._load_dataset(job.dataset_id)
            if dataset is None:
                raise ValueError(f"Dataset {job.dataset_id} not found")
            
            # Execute clustering in thread pool
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                self.executor,
                self._run_clustering_job,
                dataset,
                job.algorithm,
                job.parameters
            )
            
            # Store results
            job.result = result
            job.status = JobStatus.COMPLETED
            job.end_time = datetime.now()
            job.duration_seconds = (job.end_time - job.start_time).total_seconds()
            
            batch.completed_jobs += 1
            
            logger.debug(f"Job {job.id} completed successfully")
            
        except Exception as e:
            job.status = JobStatus.FAILED
            job.error = str(e)
            job.end_time = datetime.now()
            if job.start_time:
                job.duration_seconds = (job.end_time - job.start_time).total_seconds()
            
            batch.failed_jobs += 1
            
            logger.error(f"Job {job.id} failed: {str(e)}")
            
            if batch.stop_on_error:
                # Mark remaining jobs as skipped
                for remaining_job in batch.jobs:
                    if remaining_job.status == JobStatus.PENDING:
                        remaining_job.status = JobStatus.SKIPPED
                
                raise ValueError(f"Batch stopped due to job failure: {str(e)}")
    
    def _run_clustering_job(
        self, 
        dataset: Dict[str, Any], 
        algorithm: str, 
        parameters: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Run a single clustering job synchronously.
        
        Args:
            dataset: Dataset information
            algorithm: Clustering algorithm name
            parameters: Algorithm parameters
            
        Returns:
            Clustering results
        """
        try:
            # Extract data
            data = dataset.get("data")
            if data is None:
                raise ValueError("Dataset has no data")
            
            # Execute clustering
            result = self.clustering_service.cluster_data(
                data=data,
                algorithm=algorithm,
                parameters=parameters
            )
            
            return {
                "algorithm": algorithm,
                "parameters": parameters,
                "labels": result.get("labels", []),
                "metrics": result.get("metrics", {}),
                "n_clusters": result.get("n_clusters", 0),
                "n_samples": len(data) if data is not None else 0
            }
            
        except Exception as e:
            logger.error(f"Error running clustering job: {str(e)}")
            raise
    
    async def _load_dataset(self, dataset_id: str) -> Optional[Dict[str, Any]]:
        """Load dataset by ID."""
        try:
            # This would integrate with your dataset service
            # For now, returning a placeholder
            return {"id": dataset_id, "data": None}
        except Exception as e:
            logger.error(f"Error loading dataset {dataset_id}: {str(e)}")
            return None
    
    def _generate_batch_summary(self, batch: BatchExecution) -> Dict[str, Any]:
        """Generate summary statistics for completed batch."""
        
        successful_jobs = [job for job in batch.jobs if job.status == JobStatus.COMPLETED]
        failed_jobs = [job for job in batch.jobs if job.status == JobStatus.FAILED]
        
        summary = {
            "batch_id": batch.id,
            "name": batch.name,
            "status": batch.status.value,
            "execution_time": {
                "started_at": batch.started_at.isoformat() if batch.started_at else None,
                "completed_at": batch.completed_at.isoformat() if batch.completed_at else None,
                "duration_minutes": None
            },
            "job_statistics": {
                "total": batch.total_jobs,
                "completed": len(successful_jobs),
                "failed": len(failed_jobs),
                "success_rate": (len(successful_jobs) / batch.total_jobs * 100) if batch.total_jobs > 0 else 0
            },
            "results_by_algorithm": {},
            "results_by_dataset": {},
            "performance_metrics": {}
        }
        
        # Calculate duration
        if batch.started_at and batch.completed_at:
            duration = batch.completed_at - batch.started_at
            summary["execution_time"]["duration_minutes"] = duration.total_seconds() / 60
        
        # Group results by algorithm
        for job in successful_jobs:
            algorithm = job.algorithm
            if algorithm not in summary["results_by_algorithm"]:
                summary["results_by_algorithm"][algorithm] = {
                    "job_count": 0,
                    "avg_clusters": 0,
                    "avg_duration": 0
                }
            
            alg_stats = summary["results_by_algorithm"][algorithm]
            alg_stats["job_count"] += 1
            
            if job.result:
                alg_stats["avg_clusters"] += job.result.get("n_clusters", 0)
            
            if job.duration_seconds:
                alg_stats["avg_duration"] += job.duration_seconds
        
        # Calculate averages
        for algorithm, stats in summary["results_by_algorithm"].items():
            if stats["job_count"] > 0:
                stats["avg_clusters"] = stats["avg_clusters"] / stats["job_count"]
                stats["avg_duration"] = stats["avg_duration"] / stats["job_count"]
        
        return summary
    
    def get_batch_status(self, batch_id: str) -> Optional[Dict[str, Any]]:
        """
        Get current status of a batch.
        
        Args:
            batch_id: Batch identifier
            
        Returns:
            Batch status information
        """
        if batch_id not in self.batches:
            return None
        
        batch = self.batches[batch_id]
        
        return {
            "id": batch.id,
            "name": batch.name,
            "status": batch.status.value,
            "progress": {
                "total_jobs": batch.total_jobs,
                "completed_jobs": batch.completed_jobs,
                "failed_jobs": batch.failed_jobs,
                "progress_percentage": (batch.completed_jobs / batch.total_jobs * 100) if batch.total_jobs > 0 else 0
            },
            "timing": {
                "created_at": batch.created_at.isoformat(),
                "started_at": batch.started_at.isoformat() if batch.started_at else None,
                "completed_at": batch.completed_at.isoformat() if batch.completed_at else None
            },
            "jobs": [
                {
                    "id": job.id,
                    "dataset_id": job.dataset_id,
                    "algorithm": job.algorithm,
                    "status": job.status.value,
                    "duration_seconds": job.duration_seconds,
                    "error": job.error
                }
                for job in batch.jobs
            ]
        }
    
    def cancel_batch(self, batch_id: str) -> bool:
        """
        Cancel a running batch.
        
        Args:
            batch_id: Batch identifier
            
        Returns:
            True if cancelled, False if not running
        """
        if batch_id not in self.batches:
            return False
        
        batch = self.batches[batch_id]
        
        if batch.status != BatchStatus.RUNNING:
            return False
        
        # Cancel the running task
        if batch_id in self._running_batches:
            task = self._running_batches[batch_id]
            task.cancel()
            del self._running_batches[batch_id]
        
        # Update batch status
        batch.status = BatchStatus.CANCELLED
        batch.completed_at = datetime.now()
        
        # Mark pending jobs as skipped
        for job in batch.jobs:
            if job.status == JobStatus.PENDING:
                job.status = JobStatus.SKIPPED
        
        logger.info(f"Batch {batch_id} cancelled")
        return True
    
    def list_batches(self) -> List[Dict[str, Any]]:
        """List all batches with summary information."""
        
        return [
            {
                "id": batch.id,
                "name": batch.name,
                "status": batch.status.value,
                "total_jobs": batch.total_jobs,
                "completed_jobs": batch.completed_jobs,
                "failed_jobs": batch.failed_jobs,
                "created_at": batch.created_at.isoformat(),
                "completed_at": batch.completed_at.isoformat() if batch.completed_at else None
            }
            for batch in self.batches.values()
        ]
    
    def get_batch_results(self, batch_id: str) -> Optional[Dict[str, Any]]:
        """Get detailed results for a completed batch."""
        
        if batch_id not in self.batches:
            return None
        
        batch = self.batches[batch_id]
        
        if batch.status not in [BatchStatus.COMPLETED, BatchStatus.FAILED]:
            return None
        
        return {
            "batch_info": asdict(batch),
            "summary": batch.results,
            "job_results": [
                {
                    "job_id": job.id,
                    "dataset_id": job.dataset_id,
                    "algorithm": job.algorithm,
                    "parameters": job.parameters,
                    "status": job.status.value,
                    "result": job.result,
                    "error": job.error,
                    "duration_seconds": job.duration_seconds
                }
                for job in batch.jobs
            ]
        }
