"""
Health check service for Interactive Spectral Clustering Platform.

Provides comprehensive health monitoring including liveness, readiness,
and detailed system status checks for production deployments.
"""

import os
import time
import psutil
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
from sqlalchemy import text
from sqlalchemy.orm import Session
from pydantic import BaseModel
import asyncio

class HealthStatus(BaseModel):
    """Health check status model."""
    
    status: str  # "healthy", "unhealthy", "degraded"
    timestamp: str
    uptime_seconds: float
    checks: Dict[str, Any]

class ComponentCheck(BaseModel):
    """Individual component health check."""
    
    name: str
    status: str  # "pass", "fail", "warn"
    message: str
    duration_ms: float
    details: Optional[Dict[str, Any]] = None

class HealthChecker:
    """Comprehensive health checking service."""
    
    def __init__(self, db_session_factory):
        """Initialize health checker with database session factory."""
        self.db_session_factory = db_session_factory
        self.start_time = time.time()
        self.last_database_check = None
        self.database_check_cache_duration = 30  # seconds
    
    async def check_liveness(self) -> HealthStatus:
        """
        Basic liveness check - is the application running?
        This should be fast and only check core application functionality.
        """
        
        start_time = time.time()
        checks = {}
        
        # Check basic application state
        try:
            checks["application"] = ComponentCheck(
                name="application",
                status="pass",
                message="Application is running",
                duration_ms=0.1
            )
            
            overall_status = "healthy"
            
        except Exception as e:
            checks["application"] = ComponentCheck(
                name="application", 
                status="fail",
                message=f"Application check failed: {str(e)}",
                duration_ms=1.0
            )
            overall_status = "unhealthy"
        
        return HealthStatus(
            status=overall_status,
            timestamp=datetime.utcnow().isoformat(),
            uptime_seconds=time.time() - self.start_time,
            checks={k: v.dict() for k, v in checks.items()}
        )
    
    async def check_readiness(self) -> HealthStatus:
        """
        Readiness check - is the application ready to serve requests?
        This includes dependencies like database, external services, etc.
        """
        
        start_time = time.time()
        checks = {}
        overall_status = "healthy"
        
        # Check database connectivity
        database_check = await self._check_database()
        checks["database"] = database_check
        
        if database_check.status == "fail":
            overall_status = "unhealthy"
        elif database_check.status == "warn":
            overall_status = "degraded"
        
        # Check system resources
        resources_check = await self._check_system_resources()
        checks["system_resources"] = resources_check
        
        if resources_check.status == "fail":
            overall_status = "unhealthy"
        elif resources_check.status == "warn" and overall_status == "healthy":
            overall_status = "degraded"
        
        # Check disk space
        disk_check = await self._check_disk_space()
        checks["disk_space"] = disk_check
        
        if disk_check.status == "fail":
            overall_status = "unhealthy"
        elif disk_check.status == "warn" and overall_status == "healthy":
            overall_status = "degraded"
        
        # Check memory usage
        memory_check = await self._check_memory_usage()
        checks["memory"] = memory_check
        
        if memory_check.status == "fail":
            overall_status = "unhealthy"
        elif memory_check.status == "warn" and overall_status == "healthy":
            overall_status = "degraded"
        
        return HealthStatus(
            status=overall_status,
            timestamp=datetime.utcnow().isoformat(),
            uptime_seconds=time.time() - self.start_time,
            checks={k: v.dict() for k, v in checks.items()}
        )
    
    async def get_detailed_status(self) -> Dict[str, Any]:
        """Get comprehensive system status information."""
        
        # Get readiness status
        readiness = await self.check_readiness()
        
        # Add additional system information
        system_info = {
            "version": "1.0.0",  # This could be loaded from version file
            "environment": os.getenv("ENVIRONMENT", "development"),
            "hostname": os.getenv("HOSTNAME", "unknown"),
            "process_id": os.getpid(),
            "python_version": f"{psutil.sys.version_info.major}.{psutil.sys.version_info.minor}.{psutil.sys.version_info.micro}",
        }
        
        # CPU and memory stats
        cpu_percent = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage('/')
        
        system_metrics = {
            "cpu_usage_percent": cpu_percent,
            "memory_usage_percent": memory.percent,
            "memory_available_gb": round(memory.available / (1024**3), 2),
            "disk_usage_percent": disk.percent,
            "disk_free_gb": round(disk.free / (1024**3), 2),
            "load_average": os.getloadavg() if hasattr(os, 'getloadavg') else None
        }
        
        return {
            "status": readiness.status,
            "timestamp": readiness.timestamp,
            "uptime_seconds": readiness.uptime_seconds,
            "system_info": system_info,
            "system_metrics": system_metrics,
            "health_checks": readiness.checks
        }
    
    async def _check_database(self) -> ComponentCheck:
        """Check database connectivity and performance."""
        
        # Use cached result if recent
        if (self.last_database_check and 
            time.time() - self.last_database_check["timestamp"] < self.database_check_cache_duration):
            return ComponentCheck(**self.last_database_check["result"])
        
        start_time = time.time()
        
        try:
            db = self.db_session_factory()
            
            # Simple connectivity test
            result = db.execute(text("SELECT 1"))
            result.fetchone()
            
            # Performance test
            performance_start = time.time()
            db.execute(text("SELECT COUNT(*) FROM clustering_runs"))
            performance_duration = (time.time() - performance_start) * 1000
            
            db.close()
            
            duration_ms = (time.time() - start_time) * 1000
            
            # Determine status based on performance
            if duration_ms > 5000:  # 5 seconds
                status = "fail"
                message = f"Database response too slow: {duration_ms:.1f}ms"
            elif duration_ms > 1000:  # 1 second
                status = "warn"
                message = f"Database response slow: {duration_ms:.1f}ms"
            else:
                status = "pass"
                message = f"Database healthy: {duration_ms:.1f}ms"
            
            check_result = ComponentCheck(
                name="database",
                status=status,
                message=message,
                duration_ms=duration_ms,
                details={
                    "connection_test_ms": duration_ms,
                    "query_performance_ms": performance_duration
                }
            )
            
            # Cache the result
            self.last_database_check = {
                "timestamp": time.time(),
                "result": check_result.dict()
            }
            
            return check_result
            
        except Exception as e:
            duration_ms = (time.time() - start_time) * 1000
            
            check_result = ComponentCheck(
                name="database",
                status="fail", 
                message=f"Database connection failed: {str(e)}",
                duration_ms=duration_ms,
                details={"error": str(e)}
            )
            
            # Cache the failure
            self.last_database_check = {
                "timestamp": time.time(),
                "result": check_result.dict()
            }
            
            return check_result
    
    async def _check_system_resources(self) -> ComponentCheck:
        """Check system resource usage."""
        
        start_time = time.time()
        
        try:
            cpu_percent = psutil.cpu_percent(interval=0.1)
            load_avg = os.getloadavg() if hasattr(os, 'getloadavg') else [0, 0, 0]
            
            # Determine status
            if cpu_percent > 90 or load_avg[0] > psutil.cpu_count() * 2:
                status = "fail"
                message = f"High CPU usage: {cpu_percent:.1f}%, load: {load_avg[0]:.2f}"
            elif cpu_percent > 70 or load_avg[0] > psutil.cpu_count():
                status = "warn"
                message = f"Elevated CPU usage: {cpu_percent:.1f}%, load: {load_avg[0]:.2f}"
            else:
                status = "pass"
                message = f"CPU usage normal: {cpu_percent:.1f}%, load: {load_avg[0]:.2f}"
            
            duration_ms = (time.time() - start_time) * 1000
            
            return ComponentCheck(
                name="system_resources",
                status=status,
                message=message,
                duration_ms=duration_ms,
                details={
                    "cpu_percent": cpu_percent,
                    "load_average_1m": load_avg[0],
                    "load_average_5m": load_avg[1],
                    "load_average_15m": load_avg[2],
                    "cpu_count": psutil.cpu_count()
                }
            )
            
        except Exception as e:
            duration_ms = (time.time() - start_time) * 1000
            
            return ComponentCheck(
                name="system_resources",
                status="fail",
                message=f"Resource check failed: {str(e)}",
                duration_ms=duration_ms,
                details={"error": str(e)}
            )
    
    async def _check_disk_space(self) -> ComponentCheck:
        """Check available disk space."""
        
        start_time = time.time()
        
        try:
            disk_usage = psutil.disk_usage('/')
            used_percent = (disk_usage.used / disk_usage.total) * 100
            free_gb = disk_usage.free / (1024**3)
            
            # Determine status
            if used_percent > 95 or free_gb < 1:
                status = "fail"
                message = f"Critically low disk space: {used_percent:.1f}% used, {free_gb:.1f}GB free"
            elif used_percent > 85 or free_gb < 5:
                status = "warn"
                message = f"Low disk space: {used_percent:.1f}% used, {free_gb:.1f}GB free"
            else:
                status = "pass"
                message = f"Disk space healthy: {used_percent:.1f}% used, {free_gb:.1f}GB free"
            
            duration_ms = (time.time() - start_time) * 1000
            
            return ComponentCheck(
                name="disk_space",
                status=status,
                message=message,
                duration_ms=duration_ms,
                details={
                    "total_gb": round(disk_usage.total / (1024**3), 2),
                    "used_gb": round(disk_usage.used / (1024**3), 2),
                    "free_gb": round(free_gb, 2),
                    "used_percent": round(used_percent, 1)
                }
            )
            
        except Exception as e:
            duration_ms = (time.time() - start_time) * 1000
            
            return ComponentCheck(
                name="disk_space",
                status="fail",
                message=f"Disk check failed: {str(e)}",
                duration_ms=duration_ms,
                details={"error": str(e)}
            )
    
    async def _check_memory_usage(self) -> ComponentCheck:
        """Check memory usage."""
        
        start_time = time.time()
        
        try:
            memory = psutil.virtual_memory()
            used_percent = memory.percent
            available_gb = memory.available / (1024**3)
            
            # Determine status
            if used_percent > 95 or available_gb < 0.5:
                status = "fail"
                message = f"Critically low memory: {used_percent:.1f}% used, {available_gb:.1f}GB available"
            elif used_percent > 85 or available_gb < 2:
                status = "warn"
                message = f"High memory usage: {used_percent:.1f}% used, {available_gb:.1f}GB available"
            else:
                status = "pass"
                message = f"Memory usage healthy: {used_percent:.1f}% used, {available_gb:.1f}GB available"
            
            duration_ms = (time.time() - start_time) * 1000
            
            return ComponentCheck(
                name="memory",
                status=status,
                message=message,
                duration_ms=duration_ms,
                details={
                    "total_gb": round(memory.total / (1024**3), 2),
                    "used_gb": round(memory.used / (1024**3), 2),
                    "available_gb": round(available_gb, 2),
                    "used_percent": round(used_percent, 1),
                    "cached_gb": round(memory.cached / (1024**3), 2) if hasattr(memory, 'cached') else None
                }
            )
            
        except Exception as e:
            duration_ms = (time.time() - start_time) * 1000
            
            return ComponentCheck(
                name="memory",
                status="fail",
                message=f"Memory check failed: {str(e)}",
                duration_ms=duration_ms,
                details={"error": str(e)}
            )
