# Operations Documentation

## Interactive Spectral Clustering Platform Operations

### Getting Started

This guide covers deployment, configuration, monitoring, and maintenance of the Interactive Spectral Clustering Platform.

## Quick Start

### Prerequisites

#### System Requirements
- **Operating System**: Linux (Ubuntu 20.04+ recommended), macOS, or Windows with WSL2
- **Docker**: Version 20.10+
- **Docker Compose**: Version 2.0+
- **Memory**: 8GB RAM minimum, 16GB recommended
- **Storage**: 50GB available space minimum
- **GPU**: NVIDIA GPU with CUDA 11.8+ (optional, for GPU acceleration)

#### Software Dependencies
- **Git**: For source code management
- **Make**: For build automation (optional)
- **curl**: For health checks and testing

### Environment Setup

#### 1. Clone Repository
```bash
git clone https://github.com/your-org/clustering-platform.git
cd clustering-platform
```

#### 2. Environment Configuration
```bash
# Copy environment template
cp .env.example .env

# Edit configuration
nano .env
```

#### 3. Start Services
```bash
# Development environment
docker-compose up -d

# Production environment
docker-compose -f docker-compose.yml --profile production up -d

# With monitoring
docker-compose -f docker-compose.yml --profile monitoring up -d
```

#### 4. Verify Installation
```bash
# Check service health
curl http://localhost:8000/health

# Check frontend
curl http://localhost:3000/health

# View logs
docker-compose logs -f api
```

## Configuration

### Environment Variables

#### Core Application Settings
```env
# Application
ENVIRONMENT=production
DEBUG=false
LOG_LEVEL=INFO

# Database
POSTGRES_DB=clustering_db
POSTGRES_USER=cluster_user
POSTGRES_PASSWORD=your-secure-password
POSTGRES_HOST=db
POSTGRES_PORT=5432
DATABASE_URL=postgresql://cluster_user:password@db:5432/clustering_db

# Redis
REDIS_HOST=redis
REDIS_PORT=6379
REDIS_PASSWORD=your-redis-password
```

#### Security Configuration
```env
# JWT Authentication
JWT_SECRET=your-super-secret-jwt-key-minimum-32-characters
ALGORITHM=HS256
ACCESS_TOKEN_EXPIRE_MINUTES=30
REFRESH_TOKEN_EXPIRE_DAYS=7

# CORS
CORS_ORIGINS=https://yourdomain.com,https://app.yourdomain.com

# Rate Limiting
RATE_LIMIT_ENABLED=true
RATE_LIMIT_BACKEND=redis://redis:6379/1
RATE_LIMIT_RPS=10
RATE_LIMIT_BURST=20
```

#### Performance Settings
```env
# GPU Configuration
GPU_ENABLED=true
CUDA_VISIBLE_DEVICES=0

# File Uploads
MAX_FILE_SIZE=52428800  # 50MB
UPLOAD_PATH=/app/uploads

# Database Connection Pool
DB_POOL_SIZE=20
DB_MAX_OVERFLOW=30
DB_POOL_TIMEOUT=30
```

#### Frontend Configuration
```env
# React/Vite Environment
VITE_API_BASE_URL=https://api.yourdomain.com
VITE_WS_URL=wss://api.yourdomain.com/ws
VITE_ENVIRONMENT=production
```

### Service Ports

| Service | Development Port | Production Port | Description |
|---------|-----------------|----------------|-------------|
| Frontend | 3000 | 80/443 | React application |
| Backend API | 8000 | 8000 | FastAPI backend |
| Database | 5432 | 5432 | PostgreSQL |
| Redis | 6379 | 6379 | Cache & rate limiting |
| Prometheus | 9090 | 9090 | Metrics collection |
| Grafana | 3001 | 3001 | Monitoring dashboard |

### Docker Compose Profiles

#### Available Profiles
```bash
# Development (basic services)
docker-compose up -d

# With frontend
docker-compose --profile frontend up -d

# With caching
docker-compose --profile cache up -d

# Production (with nginx)
docker-compose --profile production up -d

# Full monitoring stack
docker-compose --profile monitoring up -d

# All services
docker-compose --profile frontend --profile cache --profile production --profile monitoring up -d
```

## Deployment

### Development Deployment

#### Local Development
```bash
# Clone and setup
git clone https://github.com/your-org/clustering-platform.git
cd clustering-platform

# Configure environment
cp .env.example .env
# Edit .env with your settings

# Start development services
docker-compose up -d db redis
docker-compose up -d api

# Start frontend (in separate terminal)
cd frontend
npm install
npm run dev
```

#### Development with Hot Reload
```bash
# Backend hot reload
cd backend
pip install -r requirements.txt
uvicorn main:app --reload --host 0.0.0.0 --port 8000

# Frontend hot reload
cd frontend
npm run dev
```

### Production Deployment

#### Single Server Deployment
```bash
# Production environment setup
export ENVIRONMENT=production
export DOMAIN=yourdomain.com

# Deploy services
docker-compose -f docker-compose.yml --profile production up -d

# Setup nginx SSL (if using Let's Encrypt)
docker-compose exec nginx certbot --nginx -d $DOMAIN
```

#### Multi-Server Deployment

##### Database Server
```bash
# On database server
docker run -d \
  --name clustering_db \
  -e POSTGRES_DB=clustering_db \
  -e POSTGRES_USER=cluster_user \
  -e POSTGRES_PASSWORD=secure_password \
  -v postgres_data:/var/lib/postgresql/data \
  -p 5432:5432 \
  postgres:15-alpine
```

##### Application Server
```bash
# On application server
docker run -d \
  --name clustering_api \
  -e DATABASE_URL=postgresql://user:pass@db-server:5432/clustering_db \
  -e CORS_ORIGINS=https://yourdomain.com \
  -p 8000:8000 \
  your-registry/clustering-backend:latest
```

##### Load Balancer Configuration
```nginx
# /etc/nginx/sites-available/clustering
upstream backend {
    server app1:8000;
    server app2:8000;
    server app3:8000;
}

server {
    listen 80;
    server_name yourdomain.com;
    
    location / {
        proxy_pass http://backend;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }
}
```

### Container Registry

#### Building and Pushing Images
```bash
# Build images
docker build -t your-registry/clustering-backend:latest ./backend
docker build -t your-registry/clustering-frontend:latest ./frontend

# Push to registry
docker push your-registry/clustering-backend:latest
docker push your-registry/clustering-frontend:latest

# Pull and deploy
docker pull your-registry/clustering-backend:latest
docker-compose up -d
```

#### GitHub Container Registry
```bash
# Login to GitHub registry
echo $GITHUB_TOKEN | docker login ghcr.io -u username --password-stdin

# Tag and push
docker tag clustering-backend ghcr.io/your-org/clustering-backend:latest
docker push ghcr.io/your-org/clustering-backend:latest
```

## Database Management

### Initial Setup

#### Database Initialization
```bash
# Run database migrations
docker-compose exec api python -c "from app.database import init_db; init_db()"

# Create admin user
docker-compose exec api python -c "
from app.auth import create_user
create_user('admin', 'admin@example.com', 'secure_password', is_admin=True)
"
```

#### Schema Migrations
```bash
# Generate migration
docker-compose exec api alembic revision --autogenerate -m "Description"

# Run migrations
docker-compose exec api alembic upgrade head

# Rollback migration
docker-compose exec api alembic downgrade -1
```

### Backup and Restore

#### Database Backup
```bash
# Create backup
docker-compose exec db pg_dump -U cluster_user clustering_db > backup_$(date +%Y%m%d_%H%M%S).sql

# Automated backup script
#!/bin/bash
BACKUP_DIR="/backups"
DATE=$(date +%Y%m%d_%H%M%S)
docker-compose exec db pg_dump -U cluster_user clustering_db | gzip > $BACKUP_DIR/backup_$DATE.sql.gz

# Keep only last 7 days
find $BACKUP_DIR -name "backup_*.sql.gz" -mtime +7 -delete
```

#### Database Restore
```bash
# Restore from backup
gunzip -c backup_20240101_120000.sql.gz | docker-compose exec -T db psql -U cluster_user clustering_db

# Alternative restore method
docker-compose exec db psql -U cluster_user clustering_db < backup.sql
```

#### Backup to Cloud Storage
```bash
# AWS S3 backup
aws s3 cp backup_$(date +%Y%m%d_%H%M%S).sql.gz s3://your-backup-bucket/database/

# Google Cloud Storage
gsutil cp backup_$(date +%Y%m%d_%H%M%S).sql.gz gs://your-backup-bucket/database/
```

## Monitoring

### Health Checks

#### Service Health Endpoints
```bash
# API health check
curl http://localhost:8000/health

# Database health check
docker-compose exec db pg_isready -U cluster_user

# Redis health check
docker-compose exec redis redis-cli ping

# Frontend health check
curl http://localhost:3000/health
```

#### Automated Health Monitoring
```bash
#!/bin/bash
# health_check.sh
SERVICES=("api:8000" "frontend:3000" "db:5432" "redis:6379")

for service in "${SERVICES[@]}"; do
    IFS=':' read -r name port <<< "$service"
    if curl -f -s "http://localhost:$port/health" > /dev/null; then
        echo "✅ $name is healthy"
    else
        echo "❌ $name is unhealthy"
        # Send alert notification
    fi
done
```

### Logging

#### Log Configuration
```yaml
# docker-compose.yml logging
services:
  api:
    logging:
      driver: "json-file"
      options:
        max-size: "10m"
        max-file: "3"
```

#### Centralized Logging with ELK Stack
```yaml
# Add to docker-compose.yml
  elasticsearch:
    image: docker.elastic.co/elasticsearch/elasticsearch:7.15.0
    environment:
      - discovery.type=single-node
    volumes:
      - es_data:/usr/share/elasticsearch/data

  logstash:
    image: docker.elastic.co/logstash/logstash:7.15.0
    volumes:
      - ./logstash.conf:/usr/share/logstash/pipeline/logstash.conf

  kibana:
    image: docker.elastic.co/kibana/kibana:7.15.0
    ports:
      - "5601:5601"
```

#### Log Analysis Commands
```bash
# View recent logs
docker-compose logs -f --tail=100 api

# Search logs for errors
docker-compose logs api | grep -i error

# Export logs for analysis
docker-compose logs --no-color api > api_logs_$(date +%Y%m%d).log
```

### Metrics and Monitoring

#### Prometheus Configuration
```yaml
# monitoring/prometheus.yml
global:
  scrape_interval: 15s

scrape_configs:
  - job_name: 'clustering-api'
    static_configs:
      - targets: ['api:8000']
    metrics_path: '/metrics'
    
  - job_name: 'postgres'
    static_configs:
      - targets: ['postgres-exporter:9187']
```

#### Grafana Dashboard Setup
```bash
# Import Grafana dashboards
curl -X POST \
  http://admin:admin@localhost:3001/api/dashboards/db \
  -H 'Content-Type: application/json' \
  -d @monitoring/grafana/dashboards/api-dashboard.json
```

#### Key Metrics to Monitor
- **API Response Time**: Average request latency
- **Error Rate**: 4xx/5xx response percentage
- **Database Connections**: Active connection count
- **Memory Usage**: Application memory consumption
- **CPU Usage**: System CPU utilization
- **Disk Usage**: Storage space utilization
- **Active Users**: Concurrent user sessions
- **Clustering Jobs**: Job queue length and processing time

### Alerting

#### Prometheus Alerting Rules
```yaml
# monitoring/alerts.yml
groups:
- name: clustering-platform
  rules:
  - alert: HighErrorRate
    expr: rate(http_requests_total{status=~"5.."}[5m]) > 0.1
    for: 5m
    annotations:
      summary: High error rate detected
      
  - alert: DatabaseConnectionsHigh
    expr: postgres_connections > 80
    for: 2m
    annotations:
      summary: Database connection pool nearly exhausted
```

#### Slack Notifications
```bash
# Send alert to Slack
curl -X POST -H 'Content-type: application/json' \
  --data '{"text":"🚨 Alert: High error rate detected on clustering platform"}' \
  YOUR_SLACK_WEBHOOK_URL
```

## Performance Optimization

### Database Optimization

#### Query Performance
```sql
-- Create indexes for common queries
CREATE INDEX idx_clustering_jobs_user_id ON clustering_jobs(user_id);
CREATE INDEX idx_clustering_jobs_status ON clustering_jobs(status);
CREATE INDEX idx_datasets_user_id ON datasets(user_id);

-- Analyze query performance
EXPLAIN ANALYZE SELECT * FROM clustering_jobs WHERE user_id = 1;
```

#### Connection Pooling
```python
# SQLAlchemy configuration
DATABASE_CONFIG = {
    'pool_size': 20,
    'max_overflow': 30,
    'pool_timeout': 30,
    'pool_recycle': 3600,
    'pool_pre_ping': True
}
```

### Application Performance

#### API Response Optimization
```python
# Redis caching configuration
REDIS_CONFIG = {
    'host': 'redis',
    'port': 6379,
    'decode_responses': True,
    'max_connections': 50
}

# Cache frequently accessed data
@cache.cached(timeout=300)
def get_user_datasets(user_id: int):
    return database.query(Dataset).filter(Dataset.user_id == user_id).all()
```

#### GPU Optimization
```python
# GPU memory management
import torch

def optimize_gpu_memory():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.set_per_process_memory_fraction(0.8)
```

### Frontend Performance

#### Build Optimization
```javascript
// vite.config.js
export default {
  build: {
    rollupOptions: {
      output: {
        manualChunks: {
          vendor: ['react', 'react-dom'],
          charts: ['chart.js', 'd3'],
        }
      }
    }
  }
}
```

#### Code Splitting
```javascript
// Lazy loading components
const ClusteringPage = lazy(() => import('./pages/ClusteringPage'));
const DatasetPage = lazy(() => import('./pages/DatasetPage'));
```

## Security Operations

### SSL/TLS Configuration

#### Let's Encrypt Setup
```bash
# Install certbot
docker-compose exec nginx apk add certbot certbot-nginx

# Generate certificate
docker-compose exec nginx certbot --nginx -d yourdomain.com

# Auto-renewal cron job
0 12 * * * docker-compose exec nginx certbot renew --quiet
```

#### Custom SSL Certificates
```bash
# Generate self-signed certificate for development
openssl req -x509 -nodes -days 365 -newkey rsa:2048 \
  -keyout nginx/ssl/server.key \
  -out nginx/ssl/server.crt
```

### Security Hardening

#### Container Security
```dockerfile
# Use non-root user
RUN useradd -m -u 1000 appuser
USER appuser

# Read-only filesystem
docker run --read-only -v /tmp:/tmp:rw your-image

# Drop capabilities
docker run --cap-drop=ALL --cap-add=CHOWN your-image
```

#### Network Security
```bash
# Restrict docker network access
docker network create --internal clustering_internal

# Use firewall rules
ufw allow 22    # SSH
ufw allow 80    # HTTP
ufw allow 443   # HTTPS
ufw deny 5432   # PostgreSQL (internal only)
```

## Troubleshooting

### Common Issues

#### Database Connection Issues
```bash
# Check database status
docker-compose exec db pg_isready -U cluster_user

# View database logs
docker-compose logs db

# Reset database connection
docker-compose restart db api
```

#### Memory Issues
```bash
# Check memory usage
docker stats

# Increase memory limits
# In docker-compose.yml:
services:
  api:
    deploy:
      resources:
        limits:
          memory: 4G
```

#### GPU Issues
```bash
# Check GPU availability
docker run --rm --gpus all nvidia/cuda:11.8-base nvidia-smi

# Test GPU in container
docker-compose exec api python -c "import torch; print(torch.cuda.is_available())"
```

### Debugging Commands

#### Container Debugging
```bash
# Access container shell
docker-compose exec api bash

# Check running processes
docker-compose exec api ps aux

# View environment variables
docker-compose exec api env

# Check file permissions
docker-compose exec api ls -la /app
```

#### Network Debugging
```bash
# Test network connectivity
docker-compose exec api ping db
docker-compose exec api curl http://redis:6379

# View container networks
docker network ls
docker network inspect clustering_clustering_network
```

#### Performance Debugging
```bash
# Monitor resource usage
docker stats --no-stream

# Check application metrics
curl http://localhost:8000/metrics

# Profile application
docker-compose exec api python -m cProfile -o profile.stats app.py
```

### Log Analysis

#### Error Investigation
```bash
# Search for specific errors
docker-compose logs api | grep -i "error\|exception\|failed"

# Get error context
docker-compose logs api | grep -A 5 -B 5 "specific error message"

# Export logs for analysis
docker-compose logs --no-color > full_logs_$(date +%Y%m%d).log
```

#### Performance Analysis
```bash
# Analyze response times
docker-compose logs api | grep "INFO.*ms" | awk '{print $NF}' | sort -n

# Count request types
docker-compose logs api | grep -o "GET\|POST\|PUT\|DELETE" | sort | uniq -c
```

## Maintenance

### Regular Maintenance Tasks

#### Daily Tasks
```bash
#!/bin/bash
# daily_maintenance.sh

# Check service health
docker-compose ps

# Clean up old logs
docker system prune -f

# Backup database
./scripts/backup_database.sh

# Check disk usage
df -h

# Monitor key metrics
curl -s http://localhost:8000/metrics | grep -E "(cpu|memory|disk)"
```

#### Weekly Tasks
```bash
#!/bin/bash
# weekly_maintenance.sh

# Update Docker images
docker-compose pull
docker-compose up -d

# Clean unused Docker resources
docker system prune -af

# Rotate logs
logrotate /etc/logrotate.d/docker

# Security scan
docker run --rm -v /var/run/docker.sock:/var/run/docker.sock \
  aquasec/trivy image clustering-api:latest
```

#### Monthly Tasks
```bash
#!/bin/bash
# monthly_maintenance.sh

# Update dependencies
cd backend && pip-review --auto
cd frontend && npm audit fix

# Database maintenance
docker-compose exec db psql -U cluster_user -d clustering_db -c "VACUUM ANALYZE;"

# Certificate renewal check
docker-compose exec nginx certbot certificates

# Performance review
docker stats --no-stream > performance_report_$(date +%Y%m).log
```

### Scaling Operations

#### Horizontal Scaling
```bash
# Scale API instances
docker-compose up -d --scale api=3

# Add load balancer configuration
# Update nginx.conf with upstream configuration
```

#### Vertical Scaling
```yaml
# Increase resource limits
services:
  api:
    deploy:
      resources:
        limits:
          memory: 8G
          cpus: '4'
```

### Disaster Recovery

#### Backup Strategy
```bash
#!/bin/bash
# comprehensive_backup.sh

# Database backup
pg_dump clustering_db | gzip > db_backup_$(date +%Y%m%d).sql.gz

# File uploads backup
tar -czf uploads_backup_$(date +%Y%m%d).tar.gz /app/uploads

# Configuration backup
tar -czf config_backup_$(date +%Y%m%d).tar.gz .env docker-compose.yml nginx/

# Upload to cloud storage
aws s3 sync ./backups s3://your-backup-bucket/
```

#### Recovery Procedures
```bash
#!/bin/bash
# disaster_recovery.sh

# Restore database
gunzip -c db_backup_latest.sql.gz | psql clustering_db

# Restore uploads
tar -xzf uploads_backup_latest.tar.gz -C /

# Restart services
docker-compose up -d

# Verify system health
./scripts/health_check.sh
```

## Support and Documentation

### Getting Help

#### Community Support
- **GitHub Issues**: Report bugs and request features
- **Documentation**: Comprehensive guides and API documentation
- **Discord/Slack**: Real-time community support

#### Enterprise Support
- **Professional Services**: Implementation and customization
- **24/7 Support**: Critical issue response
- **Training**: Team training and workshops

### Contributing

#### Development Setup
```bash
# Fork and clone repository
git clone https://github.com/your-username/clustering-platform.git

# Install development dependencies
cd backend && pip install -r requirements-dev.txt
cd frontend && npm install

# Run tests
cd backend && pytest
cd frontend && npm test

# Submit pull request
git push origin feature-branch
```

#### Documentation Updates
- Update relevant documentation files
- Include code examples and screenshots
- Test all documented procedures
- Submit documentation pull requests
