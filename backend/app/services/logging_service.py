"""
Structured logging service for Interactive Spectral Clustering Platform.

Provides comprehensive logging functionality with request tracing,
structured JSON output, and performance monitoring capabilities.
"""

import sys
import json
import time
import uuid
from typing import Dict, Any, Optional, Callable
from contextvars import ContextVar
from loguru import logger
from fastapi import Request, Response
from fastapi.routing import APIRoute
import traceback
from datetime import datetime

# Context variables for request tracking
request_id_ctx: ContextVar[Optional[str]] = ContextVar('request_id', default=None)
user_id_ctx: ContextVar[Optional[str]] = ContextVar('user_id', default=None)

class StructuredLogger:
    """Structured logging service with request tracing and JSON output."""
    
    def __init__(self):
        """Initialize the structured logger."""
        self.setup_logging()
    
    def setup_logging(self):
        """Configure loguru for structured JSON logging."""
        
        # Ensure logs directory exists
        import os
        os.makedirs("logs", exist_ok=True)
        
        # Remove default handler
        logger.remove()
        
        # Add structured JSON handler for production
        logger.add(
            sys.stdout,
            format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}",
            level="INFO",
            serialize=False,
            backtrace=True,
            diagnose=True,
            enqueue=True
        )
        
        # Add file handler for persistent logs
        logger.add(
            "logs/app.log",
            format=self._json_formatter,
            level="DEBUG",
            rotation="100 MB",
            retention="30 days",
            compression="gz",  # Fixed compression format
            serialize=False,
            backtrace=True,
            diagnose=True,
            enqueue=True
        )
        
        # Add error file handler
        logger.add(
            "logs/errors.log",
            format=self._json_formatter,
            level="ERROR",
            rotation="50 MB",
            retention="90 days",
            compression="gz",  # Fixed compression format
            serialize=False,
            backtrace=True,
            diagnose=True,
            enqueue=True
        )
    
    def _json_formatter(self, record: Dict[str, Any]) -> str:
        """Format log record as structured JSON."""
        
        # Base log structure
        log_entry = {
            "timestamp": record["time"].isoformat(),
            "level": record["level"].name,
            "logger": record["name"],
            "message": record["message"],
            "module": record.get("module"),
            "function": record.get("function"),
            "line": record.get("line")
        }
        
        # Add request context if available
        request_id = request_id_ctx.get()
        if request_id:
            log_entry["request_id"] = request_id
        
        user_id = user_id_ctx.get()
        if user_id:
            log_entry["user_id"] = user_id
        
        # Add exception info if present
        if record.get("exception"):
            log_entry["exception"] = {
                "type": record["exception"].type.__name__,
                "value": str(record["exception"].value),
                "traceback": record["exception"].traceback
            }
        
        # Add extra fields from record
        extra = record.get("extra", {})
        if extra:
            log_entry["extra"] = extra
        
        return json.dumps(log_entry, default=str)
    
    def get_logger(self, name: str = __name__):
        """Get a logger instance with the given name."""
        return logger.bind(logger_name=name)

class RequestTrackingRoute(APIRoute):
    """Custom APIRoute that adds request tracking and logging."""
    
    def get_route_handler(self) -> Callable:
        original_route_handler = super().get_route_handler()
        
        async def custom_route_handler(request: Request) -> Response:
            # Generate request ID
            request_id = str(uuid.uuid4())
            request_id_ctx.set(request_id)
            
            # Extract user ID if available
            user_id = None
            if hasattr(request.state, 'user'):
                user_id = getattr(request.state.user, 'username', None)
            user_id_ctx.set(user_id)
            
            # Log request start
            start_time = time.time()
            
            logger.info(
                "Request started",
                extra={
                    "request_id": request_id,
                    "method": request.method,
                    "path": request.url.path,
                    "query_params": dict(request.query_params),
                    "client_ip": request.client.host if request.client else None,
                    "user_agent": request.headers.get("user-agent"),
                    "user_id": user_id
                }
            )
            
            try:
                # Process request
                response = await original_route_handler(request)
                
                # Calculate duration
                duration = time.time() - start_time
                
                # Log successful response
                logger.info(
                    "Request completed",
                    extra={
                        "request_id": request_id,
                        "status_code": response.status_code,
                        "duration_ms": round(duration * 1000, 2),
                        "user_id": user_id
                    }
                )
                
                # Add request ID to response headers
                response.headers["X-Request-ID"] = request_id
                
                return response
                
            except Exception as e:
                # Calculate duration
                duration = time.time() - start_time
                
                # Log error
                logger.error(
                    f"Request failed: {str(e)}",
                    extra={
                        "request_id": request_id,
                        "error_type": type(e).__name__,
                        "error_message": str(e),
                        "duration_ms": round(duration * 1000, 2),
                        "user_id": user_id,
                        "traceback": traceback.format_exc()
                    }
                )
                
                raise
        
        return custom_route_handler

class ErrorEnvelope:
    """Standardized error response envelope."""
    
    @staticmethod
    def create_error_response(
        error_code: str,
        message: str,
        details: Optional[Dict[str, Any]] = None,
        request_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """Create a standardized error response."""
        
        if request_id is None:
            request_id = request_id_ctx.get()
        
        return {
            "error": {
                "code": error_code,
                "message": message,
                "details": details or {},
                "request_id": request_id,
                "timestamp": datetime.utcnow().isoformat()
            }
        }
    
    @staticmethod
    def validation_error(details: Dict[str, Any]) -> Dict[str, Any]:
        """Create a validation error response."""
        return ErrorEnvelope.create_error_response(
            error_code="VALIDATION_ERROR",
            message="Request validation failed",
            details=details
        )
    
    @staticmethod
    def not_found_error(resource: str, resource_id: str) -> Dict[str, Any]:
        """Create a not found error response."""
        return ErrorEnvelope.create_error_response(
            error_code="RESOURCE_NOT_FOUND",
            message=f"{resource} not found",
            details={"resource": resource, "id": resource_id}
        )
    
    @staticmethod
    def internal_error(message: str = "Internal server error") -> Dict[str, Any]:
        """Create an internal server error response."""
        return ErrorEnvelope.create_error_response(
            error_code="INTERNAL_ERROR",
            message=message
        )
    
    @staticmethod
    def authentication_error() -> Dict[str, Any]:
        """Create an authentication error response."""
        return ErrorEnvelope.create_error_response(
            error_code="AUTHENTICATION_ERROR",
            message="Authentication required or invalid"
        )
    
    @staticmethod
    def authorization_error() -> Dict[str, Any]:
        """Create an authorization error response."""
        return ErrorEnvelope.create_error_response(
            error_code="AUTHORIZATION_ERROR",
            message="Insufficient permissions"
        )

class PerformanceMonitor:
    """Performance monitoring and metrics collection."""
    
    def __init__(self):
        """Initialize performance monitor."""
        self.metrics = {
            "requests_total": 0,
            "requests_by_method": {},
            "requests_by_endpoint": {},
            "response_times": [],
            "errors_total": 0,
            "errors_by_type": {}
        }
    
    def record_request(
        self, 
        method: str, 
        endpoint: str, 
        duration_ms: float, 
        status_code: int
    ):
        """Record request metrics."""
        
        self.metrics["requests_total"] += 1
        
        # Track by method
        if method not in self.metrics["requests_by_method"]:
            self.metrics["requests_by_method"][method] = 0
        self.metrics["requests_by_method"][method] += 1
        
        # Track by endpoint
        if endpoint not in self.metrics["requests_by_endpoint"]:
            self.metrics["requests_by_endpoint"][endpoint] = 0
        self.metrics["requests_by_endpoint"][endpoint] += 1
        
        # Track response times (keep last 1000)
        self.metrics["response_times"].append(duration_ms)
        if len(self.metrics["response_times"]) > 1000:
            self.metrics["response_times"].pop(0)
        
        # Track errors
        if status_code >= 400:
            self.metrics["errors_total"] += 1
            error_type = f"{status_code // 100}xx"
            if error_type not in self.metrics["errors_by_type"]:
                self.metrics["errors_by_type"][error_type] = 0
            self.metrics["errors_by_type"][error_type] += 1
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get current metrics."""
        
        response_times = self.metrics["response_times"]
        
        metrics = {
            "requests": {
                "total": self.metrics["requests_total"],
                "by_method": self.metrics["requests_by_method"],
                "by_endpoint": self.metrics["requests_by_endpoint"]
            },
            "errors": {
                "total": self.metrics["errors_total"],
                "by_type": self.metrics["errors_by_type"],
                "rate": (self.metrics["errors_total"] / max(1, self.metrics["requests_total"])) * 100
            },
            "performance": {
                "avg_response_time_ms": sum(response_times) / len(response_times) if response_times else 0,
                "min_response_time_ms": min(response_times) if response_times else 0,
                "max_response_time_ms": max(response_times) if response_times else 0,
                "p95_response_time_ms": self._percentile(response_times, 95) if response_times else 0
            }
        }
        
        return metrics
    
    def _percentile(self, data: list, percentile: int) -> float:
        """Calculate percentile of data."""
        if not data:
            return 0
        
        sorted_data = sorted(data)
        index = int((percentile / 100) * len(sorted_data))
        return sorted_data[min(index, len(sorted_data) - 1)]

# Global instances (commented out to prevent auto-initialization)
# structured_logger = StructuredLogger()
# performance_monitor = PerformanceMonitor()

# Export logger instance (will be created when needed)
# app_logger = structured_logger.get_logger("app")

def get_request_id() -> Optional[str]:
    """Get current request ID from context."""
    return request_id_ctx.get()

def set_user_context(user_id: str):
    """Set user context for logging."""
    user_id_ctx.set(user_id)
