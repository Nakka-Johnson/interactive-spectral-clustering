"""
Grid Search Service for parameter optimization.

This module provides grid search functionality to optimize clustering parameters
across multiple algorithms and parameter combinations.
"""

import asyncio
import itertools
import uuid
import time
import logging
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor
import numpy as np

from app.schemas.grid_search import (
    GridSearchRequest, GridSearchExperiment, GridSearchRun, 
    GridSearchStatus, ParameterGrid, GridSearchSummary, LeaderboardEntry
)
from app.services.clustering.factory import clustering_factory

logger = logging.getLogger(__name__)

class GridSearchService:
    """Service for managing grid search experiments."""
    
    def __init__(self, max_workers: int = 3):
        self.experiments: Dict[str, GridSearchExperiment] = {}
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        self.running_experiments: Dict[str, asyncio.Task] = {}
        
    def generate_parameter_combinations(self, parameter_grid: ParameterGrid) -> List[Dict[str, Any]]:
        """
        Generate all parameter combinations from a parameter grid.
        
        Args:
            parameter_grid: Parameter grid specification
            
        Returns:
            List of parameter dictionaries
        """
        combinations = []
        param_keys = []
        param_values = []
        
        for param_name, param_spec in parameter_grid.parameters.items():
            param_keys.append(param_name)
            
            if isinstance(param_spec, list):
                # List of values to try
                param_values.append(param_spec)
            elif isinstance(param_spec, dict) and 'min' in param_spec:
                # Range specification
                min_val = param_spec['min']
                max_val = param_spec['max']
                step = param_spec.get('step', 1)
                
                if isinstance(min_val, float) or isinstance(step, float):
                    # Float range
                    values = []
                    current = min_val
                    while current <= max_val:
                        values.append(round(current, 6))  # Avoid floating point precision issues
                        current += step
                else:
                    # Integer range
                    values = list(range(min_val, max_val + 1, step))
                
                param_values.append(values)
            else:
                # Single value (no grid search for this parameter)
                param_values.append([param_spec])
        
        # Generate all combinations
        for combination in itertools.product(*param_values):
            param_dict = dict(zip(param_keys, combination))
            combinations.append(param_dict)
        
        return combinations
    
    def create_grid_search_experiment(
        self, 
        request: GridSearchRequest, 
        dataset_data: np.ndarray
    ) -> GridSearchExperiment:
        """
        Create a new grid search experiment.
        
        Args:
            request: Grid search request
            dataset_data: Dataset to use for clustering
            
        Returns:
            Created experiment
        """
        group_id = str(uuid.uuid4())
        
        # Generate all runs
        runs = []
        for parameter_grid in request.parameter_grids:
            algorithm = parameter_grid.algorithm
            param_combinations = self.generate_parameter_combinations(parameter_grid)
            
            for params in param_combinations:
                run_id = str(uuid.uuid4())
                run = GridSearchRun(
                    run_id=run_id,
                    algorithm=algorithm,
                    parameters=params,
                    status="pending"
                )
                runs.append(run)
        
        # Create experiment
        experiment = GridSearchExperiment(
            group_id=group_id,
            experiment_name=request.experiment_name,
            description=request.description,
            dataset_id=request.dataset_id,
            optimization_metric=request.optimization_metric,
            maximize_metric=request.maximize_metric,
            total_runs=len(runs),
            runs=runs
        )
        
        self.experiments[group_id] = experiment
        logger.info(f"Created grid search experiment {group_id} with {len(runs)} runs")
        
        return experiment
    
    async def execute_grid_search(
        self, 
        group_id: str, 
        dataset_data: np.ndarray,
        use_gpu: bool = True,
        max_concurrent: int = 3
    ):
        """
        Execute a grid search experiment.
        
        Args:
            group_id: Experiment identifier
            dataset_data: Dataset for clustering
            use_gpu: Whether to use GPU acceleration
            max_concurrent: Maximum concurrent runs
        """
        experiment = self.experiments.get(group_id)
        if not experiment:
            raise ValueError(f"Experiment {group_id} not found")
        
        experiment.status = GridSearchStatus.RUNNING
        experiment.started_at = datetime.utcnow()
        
        logger.info(f"Starting grid search {group_id} with {experiment.total_runs} runs")
        
        # Semaphore to limit concurrent runs
        semaphore = asyncio.Semaphore(max_concurrent)
        
        async def run_single_experiment(run: GridSearchRun):
            async with semaphore:
                await self._execute_single_run(run, dataset_data, use_gpu)
                
                # Update experiment progress
                if run.status == "completed":
                    experiment.completed_runs += 1
                elif run.status == "failed":
                    experiment.failed_runs += 1
                
                # Check if this is the best run so far
                await self._update_best_run(experiment, run)
                
                logger.info(
                    f"Run {run.run_id} completed. "
                    f"Progress: {experiment.completed_runs + experiment.failed_runs}/{experiment.total_runs}"
                )
        
        # Execute all runs concurrently
        tasks = [run_single_experiment(run) for run in experiment.runs]
        await asyncio.gather(*tasks, return_exceptions=True)
        
        # Mark experiment as completed
        experiment.status = GridSearchStatus.COMPLETED
        experiment.completed_at = datetime.utcnow()
        
        logger.info(f"Grid search {group_id} completed. Best score: {experiment.best_score}")
    
    async def _execute_single_run(
        self, 
        run: GridSearchRun, 
        dataset_data: np.ndarray,
        use_gpu: bool
    ):
        """Execute a single clustering run."""
        try:
            run.status = "running"
            start_time = time.time()
            
            # Create algorithm instance
            algorithm = clustering_factory.create(run.algorithm, use_gpu=use_gpu)
            
            # Run clustering in thread pool to avoid blocking
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                self.executor,
                algorithm.fit_predict,
                dataset_data,
                run.parameters
            )
            
            # Record results
            run.metrics = result.metrics
            run.execution_time = time.time() - start_time
            run.gpu_used = result.gpu_used
            run.status = "completed"
            run.completed_at = datetime.utcnow()
            
            logger.debug(f"Run {run.run_id} completed successfully")
            
        except Exception as e:
            run.status = "failed"
            run.error_message = str(e)
            run.execution_time = time.time() - start_time
            run.completed_at = datetime.utcnow()
            
            logger.error(f"Run {run.run_id} failed: {str(e)}")
    
    async def _update_best_run(self, experiment: GridSearchExperiment, run: GridSearchRun):
        """Update the best run for an experiment."""
        if run.status != "completed" or not run.metrics:
            return
        
        metric_value = run.metrics.get(experiment.optimization_metric)
        if metric_value is None:
            logger.warning(f"Optimization metric '{experiment.optimization_metric}' not found in run {run.run_id}")
            return
        
        # Check if this is the best run so far
        is_better = False
        if experiment.best_score is None:
            is_better = True
        elif experiment.maximize_metric:
            is_better = metric_value > experiment.best_score
        else:
            is_better = metric_value < experiment.best_score
        
        if is_better:
            experiment.best_run_id = run.run_id
            experiment.best_score = metric_value
            experiment.best_parameters = run.parameters.copy()
            
            logger.info(f"New best run: {run.run_id} with score {metric_value}")
    
    def get_experiment(self, group_id: str) -> Optional[GridSearchExperiment]:
        """Get an experiment by ID."""
        return self.experiments.get(group_id)
    
    def get_experiment_summary(self, group_id: str) -> Optional[GridSearchSummary]:
        """Get experiment summary."""
        experiment = self.experiments.get(group_id)
        if not experiment:
            return None
        
        # Calculate progress
        total_processed = experiment.completed_runs + experiment.failed_runs
        progress_percentage = (total_processed / experiment.total_runs) * 100 if experiment.total_runs > 0 else 0
        
        # Find best run
        best_run = None
        if experiment.best_run_id:
            best_run = next(
                (run for run in experiment.runs if run.run_id == experiment.best_run_id),
                None
            )
        
        # Estimate completion time
        estimated_completion = None
        if experiment.status == GridSearchStatus.RUNNING and total_processed > 0:
            elapsed = (datetime.utcnow() - experiment.started_at).total_seconds()
            avg_time_per_run = elapsed / total_processed
            remaining_runs = experiment.total_runs - total_processed
            estimated_completion = datetime.utcnow() + timedelta(seconds=remaining_runs * avg_time_per_run)
        
        return GridSearchSummary(
            group_id=group_id,
            experiment_name=experiment.experiment_name,
            status=experiment.status,
            total_runs=experiment.total_runs,
            completed_runs=experiment.completed_runs,
            failed_runs=experiment.failed_runs,
            best_run=best_run,
            progress_percentage=progress_percentage,
            created_at=experiment.created_at,
            started_at=experiment.started_at,
            completed_at=experiment.completed_at,
            estimated_completion=estimated_completion
        )
    
    def get_leaderboard(
        self, 
        limit: int = 50,
        optimization_metric: str = "silhouette_score"
    ) -> List[LeaderboardEntry]:
        """
        Get leaderboard of best runs across all experiments.
        
        Args:
            limit: Maximum number of entries to return
            optimization_metric: Metric to sort by
            
        Returns:
            Sorted list of leaderboard entries
        """
        entries = []
        
        for experiment in self.experiments.values():
            for run in experiment.runs:
                if run.status != "completed" or not run.metrics:
                    continue
                
                metric_value = run.metrics.get(optimization_metric)
                if metric_value is None:
                    continue
                
                entry = LeaderboardEntry(
                    rank=0,  # Will be set after sorting
                    run_id=run.run_id,
                    experiment_name=experiment.experiment_name,
                    algorithm=run.algorithm,
                    parameters=run.parameters,
                    optimization_score=metric_value,
                    silhouette_score=run.metrics.get("silhouette_score"),
                    davies_bouldin_score=run.metrics.get("davies_bouldin_score"),
                    calinski_harabasz_score=run.metrics.get("calinski_harabasz_score"),
                    execution_time=run.execution_time or 0.0,
                    gpu_used=run.gpu_used or False,
                    completed_at=run.completed_at or datetime.utcnow()
                )
                entries.append(entry)
        
        # Sort by optimization metric (descending for higher-is-better metrics)
        entries.sort(key=lambda x: x.optimization_score, reverse=True)
        
        # Assign ranks and limit results
        for i, entry in enumerate(entries[:limit]):
            entry.rank = i + 1
        
        return entries[:limit]
    
    async def start_experiment(
        self, 
        group_id: str, 
        dataset_data: np.ndarray,
        use_gpu: bool = True,
        max_concurrent: int = 3
    ):
        """Start a grid search experiment in the background."""
        if group_id in self.running_experiments:
            raise ValueError(f"Experiment {group_id} is already running")
        
        task = asyncio.create_task(
            self.execute_grid_search(group_id, dataset_data, use_gpu, max_concurrent)
        )
        self.running_experiments[group_id] = task
        
        # Clean up task when done
        def cleanup(task):
            if group_id in self.running_experiments:
                del self.running_experiments[group_id]
        
        task.add_done_callback(cleanup)
        
        return task
    
    def cancel_experiment(self, group_id: str):
        """Cancel a running experiment."""
        if group_id in self.running_experiments:
            task = self.running_experiments[group_id]
            task.cancel()
            
            # Update experiment status
            experiment = self.experiments.get(group_id)
            if experiment:
                experiment.status = GridSearchStatus.CANCELLED
                experiment.completed_at = datetime.utcnow()

# Global service instance
grid_search_service = GridSearchService()
