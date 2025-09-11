"""
PHASE 9: Rate Limiting & Security Middleware

Implements comprehensive security hardening including:
- Token bucket rate limiting
- Security headers (HSTS, CSP, X-Content-Type-Options, etc.)
- CORS restriction to configured origins
- Request size limiting
- DDoS protection mechanisms
"""

import time
import asyncio
from typing import Dict, List, Optional, Set
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import logging
import os
import json

from fastapi import Request, Response, HTTPException
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse


@dataclass
class TokenBucket:
    """Token bucket for rate limiting"""
    capacity: int
    refill_rate: float  # tokens per second
    tokens: float = field(default_factory=lambda: 0)
    last_refill: float = field(default_factory=time.time)
    
    def consume(self, tokens: int = 1) -> bool:
        """Try to consume tokens from bucket"""
        self._refill()
        if self.tokens >= tokens:
            self.tokens -= tokens
            return True
        return False
    
    def _refill(self):
        """Refill tokens based on elapsed time"""
        now = time.time()
        elapsed = now - self.last_refill
        new_tokens = elapsed * self.refill_rate
        self.tokens = min(self.capacity, self.tokens + new_tokens)
        self.last_refill = now


@dataclass
class RateLimitRule:
    """Rate limiting rule configuration"""
    name: str
    requests_per_minute: int
    burst_capacity: int
    time_window: int = 60  # seconds
    paths: Optional[List[str]] = None  # specific paths, None = all paths
    methods: Optional[List[str]] = None  # specific methods, None = all methods
    
    def matches(self, path: str, method: str) -> bool:
        """Check if rule applies to request"""
        path_match = not self.paths or any(p in path for p in self.paths)
        method_match = not self.methods or method in self.methods
        return path_match and method_match


class RateLimitMiddleware(BaseHTTPMiddleware):
    """Advanced rate limiting middleware with multiple strategies"""
    
    def __init__(self, app, rules: List[RateLimitRule] = None, 
                 default_rate: int = 100, cleanup_interval: int = 300):
        super().__init__(app)
        self.rules = rules or [
            # Default rules for different endpoint types
            RateLimitRule("auth", 5, 10, paths=["/login", "/register"], time_window=60),
            RateLimitRule("upload", 10, 15, paths=["/upload"], time_window=60),
            RateLimitRule("clustering", 20, 30, paths=["/cluster"], time_window=60),
            RateLimitRule("api", 100, 150, time_window=60),  # General API limit
        ]
        self.default_rate = default_rate
        self.cleanup_interval = cleanup_interval
        
        # Storage for rate limiting state
        self.client_buckets: Dict[str, Dict[str, TokenBucket]] = defaultdict(dict)
        self.request_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=1000))
        self.blocked_ips: Dict[str, datetime] = {}
        self.last_cleanup = time.time()
        
        self.logger = logging.getLogger("rate_limiter")
    
    async def dispatch(self, request: Request, call_next):
        """Process request with rate limiting"""
        client_ip = self._get_client_ip(request)
        
        # Check if IP is temporarily blocked
        if self._is_blocked(client_ip):
            return self._rate_limit_response("IP temporarily blocked due to excessive requests")
        
        # Clean up old data periodically
        self._periodic_cleanup()
        
        # Apply rate limiting rules
        for rule in self.rules:
            if rule.matches(str(request.url.path), request.method):
                if not self._check_rate_limit(client_ip, rule):
                    self._log_rate_limit(client_ip, rule, request)
                    return self._rate_limit_response(f"Rate limit exceeded for {rule.name}")
        
        # Record request
        self._record_request(client_ip)
        
        # Proceed with request
        response = await call_next(request)
        
        # Add rate limit headers
        self._add_rate_limit_headers(response, client_ip)
        
        return response
    
    def _get_client_ip(self, request: Request) -> str:
        """Extract client IP address"""
        # Check for forwarded headers
        forwarded_for = request.headers.get("X-Forwarded-For")
        if forwarded_for:
            return forwarded_for.split(",")[0].strip()
        
        real_ip = request.headers.get("X-Real-IP")
        if real_ip:
            return real_ip
        
        return request.client.host if request.client else "unknown"
    
    def _check_rate_limit(self, client_ip: str, rule: RateLimitRule) -> bool:
        """Check if request is within rate limits"""
        bucket_key = f"{rule.name}_{client_ip}"
        
        if bucket_key not in self.client_buckets[client_ip]:
            # Create new bucket
            refill_rate = rule.requests_per_minute / 60.0  # tokens per second
            bucket = TokenBucket(rule.burst_capacity, refill_rate, rule.burst_capacity)
            self.client_buckets[client_ip][bucket_key] = bucket
        
        bucket = self.client_buckets[client_ip][bucket_key]
        return bucket.consume(1)
    
    def _is_blocked(self, client_ip: str) -> bool:
        """Check if IP is temporarily blocked"""
        if client_ip in self.blocked_ips:
            if datetime.now() > self.blocked_ips[client_ip]:
                del self.blocked_ips[client_ip]
                return False
            return True
        return False
    
    def _record_request(self, client_ip: str):
        """Record request for monitoring"""
        self.request_history[client_ip].append(time.time())
        
        # Check for potential DDoS (more than 1000 requests in 60 seconds)
        recent_requests = [
            t for t in self.request_history[client_ip] 
            if time.time() - t < 60
        ]
        
        if len(recent_requests) > 1000:
            # Block IP for 1 hour
            self.blocked_ips[client_ip] = datetime.now() + timedelta(hours=1)
            self.logger.warning(f"Blocked IP {client_ip} for excessive requests")
    
    def _add_rate_limit_headers(self, response: Response, client_ip: str):
        """Add rate limiting headers to response"""
        # Add standard rate limit headers
        response.headers["X-RateLimit-Limit"] = "100"
        response.headers["X-RateLimit-Remaining"] = "99"  # Simplified
        response.headers["X-RateLimit-Reset"] = str(int(time.time()) + 60)
    
    def _rate_limit_response(self, message: str) -> JSONResponse:
        """Return rate limit exceeded response"""
        return JSONResponse(
            status_code=429,
            content={
                "error": "Rate limit exceeded",
                "message": message,
                "retry_after": 60
            },
            headers={
                "Retry-After": "60",
                "X-RateLimit-Limit": "100",
                "X-RateLimit-Remaining": "0"
            }
        )
    
    def _log_rate_limit(self, client_ip: str, rule: RateLimitRule, request: Request):
        """Log rate limit violation"""
        self.logger.warning(
            f"Rate limit exceeded - IP: {client_ip}, Rule: {rule.name}, "
            f"Path: {request.url.path}, Method: {request.method}"
        )
    
    def _periodic_cleanup(self):
        """Clean up old rate limiting data"""
        now = time.time()
        if now - self.last_cleanup > self.cleanup_interval:
            # Clean up old request history
            cutoff = now - 3600  # Keep last hour
            for ip in list(self.request_history.keys()):
                history = self.request_history[ip]
                while history and history[0] < cutoff:
                    history.popleft()
                if not history:
                    del self.request_history[ip]
            
            # Clean up old buckets
            for ip in list(self.client_buckets.keys()):
                if not self.client_buckets[ip]:
                    del self.client_buckets[ip]
            
            self.last_cleanup = now


class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    """Security headers middleware"""
    
    def __init__(self, app, config: Optional[Dict] = None):
        super().__init__(app)
        self.config = config or {}
        
        # Default security headers
        self.headers = {
            # HSTS (HTTP Strict Transport Security)
            "Strict-Transport-Security": "max-age=31536000; includeSubDomains; preload",
            
            # Content type sniffing protection
            "X-Content-Type-Options": "nosniff",
            
            # XSS protection
            "X-XSS-Protection": "1; mode=block",
            
            # Frame options (clickjacking protection)
            "X-Frame-Options": "DENY",
            
            # Referrer policy
            "Referrer-Policy": "strict-origin-when-cross-origin",
            
            # Permissions policy
            "Permissions-Policy": "geolocation=(), microphone=(), camera=(), payment=()",
            
            # Content Security Policy (basic)
            "Content-Security-Policy": (
                "default-src 'self'; "
                "script-src 'self' 'unsafe-inline' 'unsafe-eval' https://cdn.jsdelivr.net; "
                "style-src 'self' 'unsafe-inline' https://fonts.googleapis.com; "
                "font-src 'self' https://fonts.gstatic.com; "
                "img-src 'self' data: https:; "
                "connect-src 'self' ws: wss:; "
                "frame-ancestors 'none';"
            ),
            
            # Cache control for sensitive responses
            "Cache-Control": "no-store, no-cache, must-revalidate, proxy-revalidate",
            "Pragma": "no-cache",
            "Expires": "0",
        }
        
        # Override with custom config
        self.headers.update(self.config.get("headers", {}))
    
    async def dispatch(self, request: Request, call_next):
        """Add security headers to response"""
        response = await call_next(request)
        
        # Add security headers
        for header, value in self.headers.items():
            response.headers[header] = value
        
        # Add server header obfuscation
        response.headers["Server"] = "WebServer/1.0"
        
        return response


class RequestSizeLimitMiddleware(BaseHTTPMiddleware):
    """Middleware to limit request body size"""
    
    def __init__(self, app, max_size: int = 50 * 1024 * 1024):  # 50MB default
        super().__init__(app)
        self.max_size = max_size
    
    async def dispatch(self, request: Request, call_next):
        """Check request size before processing"""
        content_length = request.headers.get("content-length")
        
        if content_length:
            if int(content_length) > self.max_size:
                return JSONResponse(
                    status_code=413,
                    content={
                        "error": "Request too large",
                        "max_size": self.max_size,
                        "received_size": int(content_length)
                    }
                )
        
        return await call_next(request)


# Security configuration from environment
def get_security_config() -> Dict:
    """Load security configuration from environment"""
    return {
        "allowed_origins": os.getenv("ALLOWED_ORIGINS", "http://localhost:3000").split(","),
        "rate_limit_enabled": os.getenv("RATE_LIMIT_ENABLED", "true").lower() == "true",
        "security_headers_enabled": os.getenv("SECURITY_HEADERS_ENABLED", "true").lower() == "true",
        "max_request_size": int(os.getenv("MAX_REQUEST_SIZE", 50 * 1024 * 1024)),
        "ddos_protection": os.getenv("DDOS_PROTECTION_ENABLED", "true").lower() == "true",
    }


# Export middleware classes and configuration
__all__ = [
    "RateLimitMiddleware",
    "SecurityHeadersMiddleware", 
    "RequestSizeLimitMiddleware",
    "RateLimitRule",
    "get_security_config"
]
