"""
PHASE 9: Security Middleware Initialization

Configures and initializes all security middleware components
"""

from app.middleware.security import (
    RateLimitMiddleware,
    SecurityHeadersMiddleware,
    RequestSizeLimitMiddleware,
    RateLimitRule,
    get_security_config
)


def create_rate_limit_rules(config: dict) -> list:
    """Create rate limiting rules based on configuration"""
    return [
        # Authentication endpoints - very strict
        RateLimitRule(
            name="auth",
            requests_per_minute=config.get("auth_rate_limit", 5),
            burst_capacity=config.get("auth_rate_limit", 5) * 2,
            paths=["/login", "/register", "/token", "/refresh", "/logout"],
            time_window=60
        ),
        
        # Upload endpoints - moderate limits
        RateLimitRule(
            name="upload",
            requests_per_minute=config.get("upload_rate_limit", 10),
            burst_capacity=config.get("upload_rate_limit", 10) * 2,
            paths=["/upload", "/datasets"],
            methods=["POST", "PUT"],
            time_window=60
        ),
        
        # Clustering endpoints - computational limits
        RateLimitRule(
            name="clustering",
            requests_per_minute=config.get("clustering_rate_limit", 20),
            burst_capacity=config.get("clustering_rate_limit", 20) * 2,
            paths=["/cluster", "/run-clustering", "/clustering"],
            methods=["POST"],
            time_window=60
        ),
        
        # Analytics and admin endpoints
        RateLimitRule(
            name="analytics",
            requests_per_minute=50,
            burst_capacity=75,
            paths=["/analytics", "/admin", "/metrics"],
            time_window=60
        ),
        
        # General API endpoints
        RateLimitRule(
            name="api",
            requests_per_minute=config.get("default_rate_limit", 100),
            burst_capacity=config.get("default_rate_limit", 100) * 2,
            time_window=60
        )
    ]


def configure_security_middleware(app, config: dict = None):
    """Configure all security middleware for the FastAPI app"""
    
    if config is None:
        config = get_security_config()
    
    # 1. Request Size Limiting (first layer of defense)
    if config.get("max_request_size"):
        app.add_middleware(
            RequestSizeLimitMiddleware,
            max_size=config["max_request_size"]
        )
        print(f"✅ Request size limiting enabled: {config['max_request_size']} bytes")
    
    # 2. Rate Limiting (second layer)
    if config.get("rate_limit_enabled", True):
        rules = create_rate_limit_rules(config)
        app.add_middleware(
            RateLimitMiddleware,
            rules=rules,
            default_rate=config.get("default_rate_limit", 100)
        )
        print(f"✅ Rate limiting enabled with {len(rules)} rules")
    
    # 3. Security Headers (final layer)
    if config.get("security_headers_enabled", True):
        security_headers_config = {
            "headers": {
                "Strict-Transport-Security": f"max-age={config.get('hsts_max_age', 31536000)}; includeSubDomains; preload",
                "Content-Security-Policy": config.get('csp_policy', "default-src 'self'"),
            }
        }
        app.add_middleware(
            SecurityHeadersMiddleware,
            config=security_headers_config
        )
        print("✅ Security headers middleware enabled")
    
    return app


# Export configuration function
__all__ = ["configure_security_middleware", "create_rate_limit_rules"]
