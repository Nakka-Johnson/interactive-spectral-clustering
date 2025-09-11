# Interactive Spectral Clustering Platform - Backend

## Overview

FastAPI-based backend for the Interactive Spectral Clustering Platform, providing REST API endpoints for clustering analysis, data management, and real-time progress tracking via WebSockets.

## Features

- **Multiple Clustering Algorithms**: K-means, Spectral, DBSCAN, Agglomerative, Gaussian Mixture Model
- **Real-time Progress Tracking**: WebSocket-based progress updates
- **Database Integration**: SQLAlchemy ORM with PostgreSQL/SQLite support
- **Error Monitoring**: Sentry integration for production error tracking
- **API Documentation**: Auto-generated OpenAPI/Swagger documentation

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Environment Configuration

Copy the example environment file and configure:

```bash
cp .env.example .env
```

Edit `.env` with your configuration:

```env
# Database
DATABASE_URL=sqlite:///./clustering.db

# Sentry (optional - for error monitoring)
SENTRY_DSN=https://your-sentry-dsn@sentry.io/project-id

# Environment
ENVIRONMENT=development
```

### 3. Run the Server

```bash
python app.py
```

The API will be available at: http://localhost:8000

## Error Monitoring with Sentry

The backend includes Sentry integration for comprehensive error monitoring and performance tracking.

### Setup

1. **Create Sentry Account**: Visit [sentry.io](https://sentry.io) and create an account
2. **Create Project**: Create a new Python/FastAPI project
3. **Get DSN**: Copy your project's DSN from Settings → Projects → Your Project → Client Keys (DSN)
4. **Configure Environment**: Add the DSN to your `.env` file:

```env
SENTRY_DSN=https://your-public-key@o-organization-id.ingest.sentry.io/project-id
```

### Features Enabled

- **Error Tracking**: Automatic capture of unhandled exceptions
- **Performance Monitoring**: Request/response timing and performance insights
- **Integration**: FastAPI-specific error handling and request tracing
- **Environment Tagging**: Errors tagged with development/production environment
- **Privacy**: PII (Personally Identifiable Information) sending disabled by default

### Testing

Access the `/sentry-debug` endpoint in development mode to test error reporting:

```bash
curl http://localhost:8000/sentry-debug
```

### Configuration Options

The Sentry integration includes:

- **Traces Sample Rate**: 100% transaction capture for performance monitoring
- **Environment Tagging**: Automatically tags errors with current environment
- **FastAPI Integration**: Captures FastAPI-specific context and errors
- **ASGI Middleware**: Wraps the entire application for comprehensive monitoring

## API Documentation

Once running, visit:
- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

## Development

### Database Migrations

The application auto-creates database tables on startup. For production, consider using Alembic for proper migrations.

### Testing Sentry

In development mode, you can test Sentry integration:

1. Ensure `ENVIRONMENT=development` in your `.env`
2. Configure a valid `SENTRY_DSN`
3. Visit `/sentry-debug` endpoint to trigger a test error
4. Check your Sentry dashboard for the captured error

### Environment Variables

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `DATABASE_URL` | No | `sqlite:///./clustering.db` | Database connection string |
| `SENTRY_DSN` | No | `None` | Sentry DSN for error monitoring |
| `ENVIRONMENT` | No | `development` | Environment tag for monitoring |
| `API_HOST` | No | `0.0.0.0` | Host to bind the server |
| `API_PORT` | No | `8000` | Port to bind the server |

## Architecture

- **FastAPI**: Modern, fast web framework for building APIs
- **SQLAlchemy**: Database ORM with support for multiple backends
- **WebSockets**: Real-time communication for progress updates
- **Pydantic**: Data validation and serialization
- **Sentry**: Error monitoring and performance tracking

## Security

- **CORS**: Configured for frontend origins
- **Exception Handling**: Global exception handler for consistent error responses
- **Input Validation**: Pydantic models for request validation
- **Environment Separation**: Different configurations for development/production
