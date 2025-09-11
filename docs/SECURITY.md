# Security Documentation

## Interactive Spectral Clustering Platform Security

### Security Overview

The Interactive Spectral Clustering Platform implements comprehensive security measures across all layers of the application stack to protect user data, prevent unauthorized access, and ensure system integrity.

## Authentication & Authorization

### JWT (JSON Web Tokens)

The platform uses JWT for stateless authentication with the following implementation:

#### Token Structure
- **Access Token**: Short-lived (30 minutes) for API access
- **Refresh Token**: Long-lived (7 days) for token renewal
- **Algorithm**: HS256 (HMAC with SHA-256)
- **Secret**: Environment-configurable strong secret key

#### Security Features
- **Token Rotation**: Automatic refresh token rotation on use
- **Blacklisting**: Invalidated tokens stored in Redis
- **Secure Storage**: httpOnly cookies for web clients
- **CSRF Protection**: Custom headers for state-changing operations

#### Configuration
```env
JWT_SECRET=your-super-secret-jwt-key-minimum-32-characters
ALGORITHM=HS256
ACCESS_TOKEN_EXPIRE_MINUTES=30
REFRESH_TOKEN_EXPIRE_DAYS=7
```

### Role-Based Access Control (RBAC)

#### User Roles
- **User**: Standard user with dataset and clustering permissions
- **Admin**: Full system access including user management
- **Service**: Internal service account for system operations

#### Permission Matrix
| Resource | User | Admin | Service |
|----------|------|-------|---------|
| Own Datasets | CRUD | CRUD | R |
| Others' Datasets | - | CRUD | R |
| Clustering Jobs | CRUD | CRUD | CRUD |
| User Management | R (self) | CRUD | R |
| System Metrics | - | R | CRUD |

## Input Validation & Sanitization

### File Upload Security

#### Validation Layers
1. **File Type Validation**: Whitelist of allowed MIME types
2. **File Extension Check**: Verification against allowed extensions
3. **Magic Number Validation**: Content-based file type detection
4. **Size Limits**: Configurable maximum file size (default: 50MB)
5. **Virus Scanning**: Integration with ClamAV (optional)

#### Implementation
```python
ALLOWED_EXTENSIONS = {'.csv', '.xlsx', '.xls', '.tsv'}
ALLOWED_MIME_TYPES = {
    'text/csv',
    'application/vnd.ms-excel',
    'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
}
MAX_FILE_SIZE = 50 * 1024 * 1024  # 50MB
```

### Data Validation

#### Input Sanitization
- **SQL Injection Prevention**: Parameterized queries with SQLAlchemy ORM
- **XSS Prevention**: Input encoding and CSP headers
- **Path Traversal Protection**: Filename sanitization
- **Command Injection**: No shell command execution with user input

#### Schema Validation
```python
# Example Pydantic model for request validation
class ClusteringJobCreate(BaseModel):
    dataset_id: int = Field(..., gt=0)
    algorithm: str = Field(..., regex=r'^(spectral|kmeans|dbscan)$')
    n_clusters: int = Field(..., ge=2, le=100)
    parameters: Dict[str, Any] = Field(default_factory=dict)
    
    @validator('parameters')
    def validate_parameters(cls, v):
        # Custom parameter validation logic
        return v
```

## API Security

### Rate Limiting

#### Multi-Tier Rate Limiting System
```python
RATE_LIMIT_RULES = {
    'auth': {'requests': 5, 'window': 60},      # 5 req/min
    'upload': {'requests': 10, 'window': 3600}, # 10 req/hour
    'clustering': {'requests': 20, 'window': 3600}, # 20 req/hour
    'api': {'requests': 100, 'window': 60},     # 100 req/min
    'websocket': {'requests': 50, 'window': 60} # 50 req/min
}
```

#### Implementation Features
- **Token Bucket Algorithm**: Burst capacity with sustained rate
- **IP-Based Tracking**: Per-IP address rate limiting
- **User-Based Tracking**: Per-authenticated user limits
- **DDoS Protection**: Automatic IP blocking on abuse
- **Redis Backend**: Distributed rate limiting state

#### Rate Limit Headers
```http
X-RateLimit-Limit: 100
X-RateLimit-Remaining: 95
X-RateLimit-Reset: 1640995200
X-RateLimit-Retry-After: 60
```

### CORS (Cross-Origin Resource Sharing)

#### Configuration
```python
CORS_SETTINGS = {
    'allow_origins': ['http://localhost:3000', 'https://yourdomain.com'],
    'allow_methods': ['GET', 'POST', 'PUT', 'DELETE', 'OPTIONS'],
    'allow_headers': ['Authorization', 'Content-Type', 'X-Requested-With'],
    'allow_credentials': True,
    'max_age': 3600
}
```

#### Security Considerations
- **No Wildcard Origins**: Explicit origin allowlist
- **Credentials Support**: Secure cookie handling
- **Preflight Caching**: Reduced preflight request overhead

### Security Headers

#### HTTP Security Headers
```python
SECURITY_HEADERS = {
    'Strict-Transport-Security': 'max-age=31536000; includeSubDomains',
    'X-Content-Type-Options': 'nosniff',
    'X-Frame-Options': 'DENY',
    'X-XSS-Protection': '1; mode=block',
    'Referrer-Policy': 'strict-origin-when-cross-origin',
    'Content-Security-Policy': "default-src 'self'; script-src 'self' 'unsafe-inline'; style-src 'self' 'unsafe-inline'",
    'Permissions-Policy': 'geolocation=(), microphone=(), camera=()'
}
```

#### Header Descriptions
- **HSTS**: Enforce HTTPS connections
- **X-Content-Type-Options**: Prevent MIME type sniffing
- **X-Frame-Options**: Prevent clickjacking attacks
- **CSP**: Restrict resource loading sources
- **Referrer-Policy**: Control referrer information leakage

## Data Protection

### Encryption

#### Data in Transit
- **TLS 1.3**: Modern encryption for all HTTP traffic
- **Certificate Management**: Automated certificate renewal
- **HSTS**: Force HTTPS connections
- **Perfect Forward Secrecy**: Ephemeral key exchange

#### Data at Rest
- **Database Encryption**: PostgreSQL transparent data encryption
- **File Encryption**: Uploaded files encrypted using AES-256
- **Secrets Management**: Environment variables and secure vaults
- **Key Rotation**: Regular encryption key updates

#### Implementation Example
```python
from cryptography.fernet import Fernet

class FileEncryption:
    def __init__(self, key: bytes):
        self.cipher_suite = Fernet(key)
    
    def encrypt_file(self, file_path: str) -> str:
        with open(file_path, 'rb') as file:
            encrypted_data = self.cipher_suite.encrypt(file.read())
        return base64.b64encode(encrypted_data).decode()
    
    def decrypt_file(self, encrypted_data: str) -> bytes:
        encrypted_bytes = base64.b64decode(encrypted_data)
        return self.cipher_suite.decrypt(encrypted_bytes)
```

### Privacy

#### Data Minimization
- **Purpose Limitation**: Data collected only for specified purposes
- **Retention Policies**: Automatic data deletion after retention period
- **User Consent**: Explicit consent for data processing
- **Data Portability**: Export functionality for user data

#### Anonymization
- **PII Removal**: Personal identifiers stripped from datasets
- **Pseudonymization**: Reversible anonymization for internal use
- **Aggregation**: Statistical aggregation to prevent re-identification
- **Differential Privacy**: Noise injection for privacy preservation

## Infrastructure Security

### Container Security

#### Docker Security Best Practices
```dockerfile
# Non-root user
RUN useradd -m -u 1000 appuser
USER appuser

# Minimal base image
FROM nvidia/cuda:11.8-runtime-ubuntu20.04

# Read-only filesystem (where possible)
COPY --chown=appuser:appuser . .

# Health checks
HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1
```

#### Security Scanning
- **Trivy**: Vulnerability scanning for container images
- **Snyk**: Dependency vulnerability detection
- **CIS Benchmarks**: Container security compliance
- **Runtime Security**: Falco for runtime threat detection

### Network Security

#### Network Isolation
```yaml
# Docker Compose network configuration
networks:
  clustering_network:
    driver: bridge
    ipam:
      config:
        - subnet: 172.20.0.0/16
```

#### Firewall Rules
- **Port Restrictions**: Only necessary ports exposed
- **Internal Communication**: Services communicate via private network
- **External Access**: Reverse proxy for public endpoints
- **VPN Access**: Administrative access through VPN

### Database Security

#### PostgreSQL Security Configuration
```sql
-- Connection security
ssl = on
ssl_cert_file = 'server.crt'
ssl_key_file = 'server.key'

-- Authentication
password_encryption = scram-sha-256
log_connections = on
log_disconnections = on

-- Access control
shared_preload_libraries = 'pg_stat_statements'
```

#### Security Features
- **Connection Encryption**: SSL/TLS for all database connections
- **User Isolation**: Separate database users for different services
- **Audit Logging**: Comprehensive database activity logging
- **Backup Encryption**: Encrypted database backups

## Monitoring & Incident Response

### Security Monitoring

#### Log Aggregation
```python
LOGGING_CONFIG = {
    'security_events': {
        'failed_logins': 'WARNING',
        'rate_limit_exceeded': 'WARNING',
        'suspicious_activity': 'ERROR',
        'privilege_escalation': 'CRITICAL'
    }
}
```

#### Metrics Collection
- **Authentication Failures**: Failed login attempts
- **Rate Limiting**: Rate limit violations
- **Anomaly Detection**: Unusual access patterns
- **System Health**: Security service availability

#### Alerting
```yaml
# Example Prometheus alerting rules
groups:
- name: security
  rules:
  - alert: HighFailedLoginRate
    expr: rate(failed_login_attempts[5m]) > 10
    for: 2m
    annotations:
      summary: High rate of failed login attempts
```

### Incident Response

#### Response Procedures
1. **Detection**: Automated monitoring and alerting
2. **Assessment**: Severity classification and impact analysis
3. **Containment**: Immediate threat mitigation
4. **Investigation**: Root cause analysis and evidence collection
5. **Recovery**: System restoration and security improvements
6. **Lessons Learned**: Post-incident review and process updates

#### Security Contacts
- **Security Team**: security@company.com
- **Incident Response**: incident@company.com
- **Emergency Hotline**: +1-XXX-XXX-XXXX

## Compliance & Auditing

### Audit Logging

#### Audit Events
```python
AUDIT_EVENTS = {
    'user_login': {'level': 'INFO', 'retention': '1_year'},
    'data_access': {'level': 'INFO', 'retention': '2_years'},
    'admin_action': {'level': 'WARNING', 'retention': '5_years'},
    'security_violation': {'level': 'ERROR', 'retention': '7_years'}
}
```

#### Log Format
```json
{
  "timestamp": "2024-01-01T00:00:00Z",
  "event_type": "user_login",
  "user_id": 1234,
  "ip_address": "192.168.1.100",
  "user_agent": "Mozilla/5.0...",
  "result": "success",
  "details": {...}
}
```

### Compliance Standards

#### GDPR Compliance
- **Data Protection Impact Assessment**: Regular DPIA reviews
- **Right to Erasure**: Data deletion functionality
- **Data Portability**: User data export capabilities
- **Consent Management**: Granular consent tracking

#### SOC 2 Type II
- **Security Controls**: Implementation of security frameworks
- **Availability Controls**: System uptime and redundancy
- **Processing Integrity**: Data processing accuracy
- **Confidentiality**: Information protection measures

## Security Configuration

### Environment Variables

#### Required Security Settings
```env
# JWT Configuration
JWT_SECRET=your-256-bit-secret-key-here
ALGORITHM=HS256
ACCESS_TOKEN_EXPIRE_MINUTES=30

# CORS Settings
CORS_ORIGINS=https://yourdomain.com,https://app.yourdomain.com

# Rate Limiting
RATE_LIMIT_BACKEND=redis://redis:6379/1
RATE_LIMIT_ENABLED=true

# Security Features
SECURITY_HEADERS_ENABLED=true
REQUEST_SIZE_LIMIT=52428800
FILE_UPLOAD_SIZE_LIMIT=52428800

# Database Security
DATABASE_SSL_MODE=require
DATABASE_SSL_CERT_PATH=/path/to/cert.pem

# Monitoring
SECURITY_MONITORING_ENABLED=true
AUDIT_LOG_LEVEL=INFO
```

### Production Deployment Checklist

#### Pre-Deployment Security Checks
- [ ] All default passwords changed
- [ ] Security headers configured
- [ ] TLS/SSL certificates installed
- [ ] Database encryption enabled
- [ ] Firewall rules configured
- [ ] Security monitoring enabled
- [ ] Backup encryption configured
- [ ] Access logs enabled
- [ ] Rate limiting configured
- [ ] CORS origins restricted

#### Post-Deployment Verification
- [ ] Vulnerability scan completed
- [ ] Penetration testing performed
- [ ] Security monitoring verified
- [ ] Incident response procedures tested
- [ ] Backup and recovery tested
- [ ] Access controls verified
- [ ] Audit logging functional
- [ ] Performance impact assessed

## Security Updates & Maintenance

### Regular Security Tasks

#### Daily
- Monitor security alerts and logs
- Review failed authentication attempts
- Check system health and availability

#### Weekly
- Update security signatures and rules
- Review access logs for anomalies
- Verify backup integrity

#### Monthly
- Security patch assessment and deployment
- Access review and cleanup
- Security metrics review

#### Quarterly
- Penetration testing
- Security architecture review
- Incident response plan testing
- Security training updates

### Emergency Procedures

#### Security Incident Response
1. **Immediate Actions**
   - Isolate affected systems
   - Preserve evidence
   - Notify security team

2. **Assessment**
   - Determine incident scope
   - Assess data exposure
   - Document timeline

3. **Communication**
   - Notify stakeholders
   - Prepare public statements
   - Coordinate with authorities

4. **Recovery**
   - Restore affected systems
   - Implement additional controls
   - Monitor for additional threats

#### Contact Information
- **Security Incident Email**: security-incident@company.com
- **Emergency Phone**: +1-XXX-XXX-XXXX
- **Security Team Slack**: #security-incidents
