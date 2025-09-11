"""
Metrics service for Interactive Spectral Clustering Platform.

Provides Prometheus-compatible metrics collection for monitoring
application performance, usage, and system health.
"""

import time
from typing import Dict, Any, Optional
from prometheus_client import Counter, Histogram, Gauge, Info, generate_latest, CONTENT_TYPE_LATEST
from prometheus_client.core import CollectorRegistry
from fastapi import Request, Response
import psutil
import os

class MetricsCollector:
    """Prometheus metrics collector for the clustering platform."""
    
    def __init__(self):
        """Initialize metrics collector."""
        
        # Create custom registry to avoid conflicts
        self.registry = CollectorRegistry()
        
        # HTTP Request metrics
        self.http_requests_total = Counter(
            'http_requests_total',
            'Total number of HTTP requests',
            ['method', 'endpoint', 'status_code'],
            registry=self.registry
        )
        
        self.http_request_duration = Histogram(
            'http_request_duration_seconds',
            'HTTP request duration in seconds',
            ['method', 'endpoint'],
            registry=self.registry
        )
        
        # Clustering operation metrics
        self.clustering_operations_total = Counter(
            'clustering_operations_total',
            'Total number of clustering operations',
            ['algorithm', 'status'],
            registry=self.registry
        )
        
        self.clustering_duration = Histogram(
            'clustering_duration_seconds',
            'Clustering operation duration in seconds',
            ['algorithm'],
            buckets=[0.1, 0.5, 1.0, 2.5, 5.0, 10.0, 25.0, 50.0, 100.0, float('inf')],
            registry=self.registry
        )
        
        self.dataset_size = Histogram(
            'dataset_size_samples',
            'Size of datasets being processed',
            buckets=[10, 50, 100, 500, 1000, 5000, 10000, 50000, 100000, float('inf')],
            registry=self.registry
        )
        
        # System metrics
        self.cpu_usage = Gauge(
            'cpu_usage_percent',
            'Current CPU usage percentage',
            registry=self.registry
        )
        
        self.memory_usage = Gauge(
            'memory_usage_bytes',
            'Current memory usage in bytes',
            registry=self.registry
        )
        
        self.memory_available = Gauge(
            'memory_available_bytes',
            'Available memory in bytes',
            registry=self.registry
        )
        
        self.disk_usage = Gauge(
            'disk_usage_bytes',
            'Current disk usage in bytes',
            registry=self.registry
        )
        
        self.disk_available = Gauge(
            'disk_available_bytes',
            'Available disk space in bytes',
            registry=self.registry
        )
        
        # Database metrics
        self.database_connections = Gauge(
            'database_connections_active',
            'Number of active database connections',
            registry=self.registry
        )
        
        self.database_query_duration = Histogram(
            'database_query_duration_seconds',
            'Database query duration in seconds',
            ['query_type'],
            registry=self.registry
        )
        
        # Application metrics
        self.active_users = Gauge(
            'active_users_total',
            'Number of currently active users',
            registry=self.registry
        )
        
        self.uploaded_datasets = Counter(
            'uploaded_datasets_total',
            'Total number of uploaded datasets',
            ['file_type'],
            registry=self.registry
        )
        
        self.export_operations = Counter(
            'export_operations_total',
            'Total number of export operations',
            ['export_type', 'status'],
            registry=self.registry
        )
        
        # Batch processing metrics
        self.batch_jobs_total = Counter(
            'batch_jobs_total',
            'Total number of batch jobs',
            ['status'],
            registry=self.registry
        )
        
        self.batch_job_duration = Histogram(
            'batch_job_duration_seconds',
            'Batch job duration in seconds',
            buckets=[10, 30, 60, 300, 600, 1800, 3600, 7200, float('inf')],
            registry=self.registry
        )
        
        # Application info
        self.app_info = Info(
            'app_info',
            'Application information',
            registry=self.registry
        )
        
        # Set application info
        self.app_info.info({
            'version': '1.0.0',
            'environment': os.getenv('ENVIRONMENT', 'development'),
            'python_version': f"{psutil.sys.version_info.major}.{psutil.sys.version_info.minor}",
        })
        
        # Start background system metrics collection
        self._update_system_metrics()
    
    def record_http_request(self, method: str, endpoint: str, status_code: int, duration: float):
        """Record HTTP request metrics."""
        self.http_requests_total.labels(
            method=method,
            endpoint=endpoint,
            status_code=str(status_code)
        ).inc()
        
        self.http_request_duration.labels(
            method=method,
            endpoint=endpoint
        ).observe(duration)
    
    def record_clustering_operation(self, algorithm: str, duration: float, dataset_size: int, success: bool):
        """Record clustering operation metrics."""
        status = 'success' if success else 'failure'
        
        self.clustering_operations_total.labels(
            algorithm=algorithm,
            status=status
        ).inc()
        
        if success:
            self.clustering_duration.labels(algorithm=algorithm).observe(duration)
            self.dataset_size.observe(dataset_size)
    
    def record_dataset_upload(self, file_type: str):
        """Record dataset upload metrics."""
        self.uploaded_datasets.labels(file_type=file_type).inc()
    
    def record_export_operation(self, export_type: str, success: bool):
        """Record export operation metrics."""
        status = 'success' if success else 'failure'
        self.export_operations.labels(
            export_type=export_type,
            status=status
        ).inc()
    
    def record_batch_job(self, duration: float, success: bool):
        """Record batch job metrics."""
        status = 'success' if success else 'failure'
        
        self.batch_jobs_total.labels(status=status).inc()
        
        if success:
            self.batch_job_duration.observe(duration)
    
    def record_database_query(self, query_type: str, duration: float):
        """Record database query metrics."""
        self.database_query_duration.labels(query_type=query_type).observe(duration)
    
    def set_active_users(self, count: int):
        """Set current active users count."""
        self.active_users.set(count)
    
    def set_database_connections(self, count: int):
        """Set current database connections count."""
        self.database_connections.set(count)
    
    def _update_system_metrics(self):
        """Update system metrics."""
        try:
            # CPU usage
            cpu_percent = psutil.cpu_percent(interval=None)
            self.cpu_usage.set(cpu_percent)
            
            # Memory usage
            memory = psutil.virtual_memory()
            self.memory_usage.set(memory.used)
            self.memory_available.set(memory.available)
            
            # Disk usage
            disk = psutil.disk_usage('/')
            self.disk_usage.set(disk.used)
            self.disk_available.set(disk.free)
            
        except Exception:
            # Silently fail to avoid disrupting the application
            pass
    
    def get_metrics(self) -> str:
        """Get metrics in Prometheus format."""
        # Update system metrics before generating output
        self._update_system_metrics()
        
        return generate_latest(self.registry).decode('utf-8')
    
    def get_content_type(self) -> str:
        """Get Prometheus metrics content type."""
        return CONTENT_TYPE_LATEST

class MetricsMiddleware:
    """Middleware for automatic metrics collection."""
    
    def __init__(self, metrics_collector: MetricsCollector):
        """Initialize metrics middleware."""
        self.metrics = metrics_collector
    
    async def __call__(self, request: Request, call_next):
        """Process request and collect metrics."""
        start_time = time.time()
        
        # Process request
        response = await call_next(request)
        
        # Calculate duration
        duration = time.time() - start_time
        
        # Extract endpoint from path
        endpoint = self._extract_endpoint(request.url.path)
        
        # Record metrics
        self.metrics.record_http_request(
            method=request.method,
            endpoint=endpoint,
            status_code=response.status_code,
            duration=duration
        )
        
        return response
    
    def _extract_endpoint(self, path: str) -> str:
        """Extract endpoint pattern from request path."""
        # Normalize path to remove IDs and create consistent endpoint names
        
        # Remove query parameters
        path = path.split('?')[0]
        
        # Replace common ID patterns
        import re
        
        # Replace UUIDs
        path = re.sub(r'/[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}', '/{id}', path)
        
        # Replace numeric IDs
        path = re.sub(r'/\d+', '/{id}', path)
        
        # Replace file extensions
        path = re.sub(r'\.(csv|json|html|zip)$', '.{ext}', path)
        
        return path or '/'

# Global metrics collector instance
metrics_collector = MetricsCollector()
metrics_middleware = MetricsMiddleware(metrics_collector)
