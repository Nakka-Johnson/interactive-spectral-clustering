# Interactive Spectral Clustering Platform - Complete Architecture

## Overview
A production-ready, multi-tenant clustering platform that provides comprehensive machine learning clustering capabilities with modern web architecture, scientific visualization, and enterprise-grade features.

## 🏗️ System Architecture

### High-Level Architecture
```
┌─────────────────────────────────────────────────────────────────┐
│                    CLIENT LAYER                                 │
├─────────────────────────────────────────────────────────────────┤
│ Web Browser │ Mobile Browser │ API Clients │ CLI Tools          │
└─────────────────────────────────────────────────────────────────┘
                                ↓
┌─────────────────────────────────────────────────────────────────┐
│                   FRONTEND LAYER                                │
├─────────────────────────────────────────────────────────────────┤
│ React 19.1.1 + TypeScript 5.4.5 │ Material-UI v7.3.2           │
│ Zustand State Management         │ React Router v7.8.2          │
│ Chart.js + Three.js             │ Real-time WebSocket           │
└─────────────────────────────────────────────────────────────────┘
                                ↓
┌─────────────────────────────────────────────────────────────────┐
│                   API GATEWAY LAYER                             │
├─────────────────────────────────────────────────────────────────┤
│ FastAPI Server │ WebSocket Handler │ Authentication Service     │
│ Rate Limiting  │ Request Validation │ CORS Middleware           │
│ Health Checks  │ Metrics Collection │ Error Handling            │
└─────────────────────────────────────────────────────────────────┘
                                ↓
┌─────────────────────────────────────────────────────────────────┐
│                  BUSINESS LOGIC LAYER                           │
├─────────────────────────────────────────────────────────────────┤
│ Clustering Engine      │ Grid Search Service │ Export Service   │
│ Data Preprocessing     │ Metrics Evaluation  │ Batch Processing │
│ Embedding Service      │ Experiment Manager  │ Report Generator │
│ Health Monitoring      │ Logging Service     │ Authentication   │
└─────────────────────────────────────────────────────────────────┘
                                ↓
┌─────────────────────────────────────────────────────────────────┐
│                   COMPUTATION LAYER                             │
├─────────────────────────────────────────────────────────────────┤
│ Scikit-learn Models    │ FAISS GPU Indexing │ CUDA Acceleration │
│ NumPy/Pandas Computing │ Matplotlib/Seaborn │ ReportLab PDFs    │
│ Memory Management      │ Parallel Processing │ Result Caching    │
└─────────────────────────────────────────────────────────────────┘
                                ↓
┌─────────────────────────────────────────────────────────────────┐
│                    DATA LAYER                                   │
├─────────────────────────────────────────────────────────────────┤
│ PostgreSQL Database    │ Redis Cache        │ File System       │
│ Multi-tenant Schema    │ Session Storage    │ CSV/Upload Storage │
│ SQLAlchemy ORM         │ Result Caching     │ Export Files       │
└─────────────────────────────────────────────────────────────────┘
```

## 📁 Project Structure

```
interactive-spectral-clustering/
│
├── frontend/                           # React TypeScript Application
│   ├── src/
│   │   ├── components/                 # Reusable UI Components
│   │   │   ├── layout/
│   │   │   │   ├── AppShell.tsx        # Main layout with navigation
│   │   │   │   └── index.ts
│   │   │   ├── ErrorBoundary.tsx       # Error handling wrapper
│   │   │   ├── FileUploader.tsx        # Drag-drop file upload
│   │   │   ├── DatasetPreview.tsx      # Data preview tables
│   │   │   ├── FeatureFilter.tsx       # Column selection interface
│   │   │   ├── AlgorithmConfigPanel.tsx # Algorithm parameter config
│   │   │   ├── ProgressBar.tsx         # Real-time progress tracking
│   │   │   ├── ResultsVisualization.tsx # 2D/3D cluster plots
│   │   │   ├── MetricsComparison.tsx   # Performance metrics display
│   │   │   ├── ExperimentHistory.tsx   # Experiment tracking
│   │   │   └── ReportPanel.tsx         # Report generation UI
│   │   ├── features/                   # Feature-based modules
│   │   │   ├── upload/
│   │   │   │   ├── FileUpload.tsx      # Upload workflow
│   │   │   │   ├── UploadPage.tsx      # Upload page container
│   │   │   │   └── index.ts
│   │   │   ├── config/
│   │   │   │   ├── ConfigPage.tsx      # Algorithm configuration
│   │   │   │   └── index.ts
│   │   │   ├── visualize/
│   │   │   │   ├── VisualizationPage.tsx # Main visualization page
│   │   │   │   └── index.ts
│   │   │   ├── experiments/
│   │   │   │   ├── ExperimentDashboard.tsx # Experiment management
│   │   │   │   ├── Leaderboard.tsx     # Grid search results
│   │   │   │   └── index.ts
│   │   │   └── params/
│   │   │       ├── GridSearchForm.tsx  # Grid search configuration
│   │   │       └── index.ts
│   │   ├── pages/                      # Page-level components
│   │   │   ├── ReportPage.tsx          # PDF report generation
│   │   │   ├── EmbeddingVisualizationPage.tsx # Dimensionality reduction
│   │   │   ├── ErrorHandlingDemo.tsx   # Error handling examples
│   │   │   └── Phase2TestPage.tsx      # Development testing
│   │   ├── services/
│   │   │   ├── api.ts                  # HTTP client with auth
│   │   │   └── rateLimitService.ts     # Rate limiting client
│   │   ├── store/
│   │   │   ├── appStore.ts             # Global application state
│   │   │   └── useClusteringStore.ts   # Clustering-specific state
│   │   ├── theme/
│   │   │   └── neoDarkTheme.ts         # Material-UI theme
│   │   ├── utils/
│   │   │   ├── auth.ts                 # Authentication utilities
│   │   │   ├── logger.ts               # Frontend logging
│   │   │   └── dataUtils.ts            # Data processing helpers
│   │   ├── App.tsx                     # Main application with routing
│   │   └── index.tsx                   # Application entry point
│   ├── public/
│   ├── package.json                    # Dependencies and scripts
│   └── tsconfig.json                   # TypeScript configuration
│
├── backend/                            # FastAPI Production Server
│   ├── app/
│   │   ├── models/                     # Database Models
│   │   │   ├── auth.py                 # User, Tenant, JWT schemas
│   │   │   ├── clustering.py           # Clustering run models
│   │   │   └── __init__.py
│   │   ├── services/                   # Business Logic Services
│   │   │   ├── auth_service.py         # JWT authentication & RBAC
│   │   │   ├── grid_search_service.py  # Hyperparameter optimization
│   │   │   ├── export_service.py       # PDF/CSV export generation
│   │   │   ├── embedding_service.py    # Dimensionality reduction
│   │   │   ├── batch_service.py        # Batch processing
│   │   │   ├── logging_service.py      # Structured logging
│   │   │   ├── health_service.py       # System health monitoring
│   │   │   ├── metrics_service.py      # Performance metrics
│   │   │   ├── preprocess.py           # Data preprocessing
│   │   │   └── __init__.py
│   │   ├── middleware/                 # FastAPI Middleware
│   │   │   ├── security.py             # Rate limiting & security
│   │   │   └── __init__.py
│   │   ├── routes/                     # API Route Handlers
│   │   │   ├── auth.py                 # Authentication endpoints
│   │   │   └── __init__.py
│   │   ├── database/                   # Database Configuration
│   │   │   ├── connection.py           # Session management
│   │   │   └── __init__.py
│   │   └── __init__.py
│   ├── clustering.py                   # Core clustering algorithms
│   ├── evaluation.py                   # Metrics computation
│   ├── graph_utils.py                  # Graph algorithms
│   ├── app.py                          # Main FastAPI application
│   ├── requirements.txt                # Python dependencies
│   └── Dockerfile                      # Container configuration
│
├── database/
│   ├── clustering.db                   # SQLite development database
│   └── README.md
│
├── docs/                              # Documentation
│   ├── api.md                         # API documentation
│   ├── architecture.md                # Legacy architecture docs
│   ├── usage.md                       # User guides
│   └── README.md
│
├── docker-compose.yml                 # Container orchestration
├── README.md                          # Project overview
└── LICENSE                            # MIT License
```

## 💾 Database Architecture

### Multi-Tenant Schema
```sql
-- Organization/Tenant Management
CREATE TABLE tenants (
    id INTEGER PRIMARY KEY,
    name VARCHAR(255) UNIQUE NOT NULL,
    domain VARCHAR(255) UNIQUE,
    description TEXT,
    status VARCHAR(50) DEFAULT 'active',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- User Management with RBAC
CREATE TABLE users (
    id INTEGER PRIMARY KEY,
    username VARCHAR(255) UNIQUE NOT NULL,
    email VARCHAR(255) UNIQUE NOT NULL,
    full_name VARCHAR(255),
    hashed_password VARCHAR(255) NOT NULL,
    tenant_id INTEGER REFERENCES tenants(id),
    role VARCHAR(50) DEFAULT 'user', -- 'admin' or 'user'
    is_active BOOLEAN DEFAULT true,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Dataset Storage with Tenant Isolation
CREATE TABLE datasets (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id INTEGER REFERENCES tenants(id),
    user_id INTEGER REFERENCES users(id),
    filename VARCHAR(255) NOT NULL,
    original_filename VARCHAR(255),
    file_size INTEGER,
    data JSON, -- Processed dataset
    metadata JSON, -- Shape, columns, statistics
    preprocessing_info JSON,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Clustering Execution Results
CREATE TABLE clustering_runs (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id INTEGER REFERENCES users(id),
    dataset_id UUID REFERENCES datasets(id),
    method VARCHAR(50) NOT NULL, -- 'spectral', 'kmeans', 'dbscan', etc.
    parameters JSON NOT NULL,
    results JSON, -- Labels, centroids, etc.
    metrics JSON, -- Silhouette, Davies-Bouldin, etc.
    execution_time FLOAT,
    status VARCHAR(50) DEFAULT 'pending',
    error_message TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Experiment Session Management
CREATE TABLE experiment_sessions (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id INTEGER REFERENCES tenants(id),
    user_id INTEGER REFERENCES users(id),
    session_id VARCHAR(255) UNIQUE NOT NULL,
    session_name VARCHAR(255),
    description TEXT,
    job_ids JSON, -- Array of clustering run IDs
    run_ids JSON, -- Array of completed run IDs
    grid_search_config JSON,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- System Performance Monitoring
CREATE TABLE system_metrics (
    id INTEGER PRIMARY KEY,
    endpoint VARCHAR(255),
    method VARCHAR(10),
    response_time FLOAT,
    status_code INTEGER,
    user_id INTEGER REFERENCES users(id),
    correlation_id VARCHAR(255),
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Grid Search Experiment Tracking
CREATE TABLE grid_search_experiments (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    group_id VARCHAR(255) UNIQUE NOT NULL,
    user_id INTEGER REFERENCES users(id),
    experiment_name VARCHAR(255),
    parameter_grids JSON NOT NULL,
    optimization_metric VARCHAR(50),
    maximize_metric BOOLEAN DEFAULT true,
    status VARCHAR(50) DEFAULT 'pending',
    total_runs INTEGER,
    completed_runs INTEGER,
    best_score FLOAT,
    best_parameters JSON,
    results JSON, -- Array of all run results
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```

## 🔌 API Architecture

### Authentication & Authorization
```
POST /auth/register         # User registration
POST /auth/login           # JWT token generation
POST /auth/refresh         # Token refresh
GET  /auth/me              # Current user info
POST /auth/logout          # Token invalidation
```

### Core Clustering Operations
```
POST /upload               # Dataset upload with validation
POST /cluster             # Execute clustering algorithm
GET  /results/{run_id}    # Get clustering results
POST /preprocess          # Data preprocessing operations
GET  /datasets/{id}/stats # Dataset statistics and analysis
```

### Grid Search & Optimization
```
POST /grid-search         # Submit grid search experiment
GET  /grid-search/{id}    # Get experiment status
GET  /leaderboard         # Top performing configurations
DELETE /grid-search/{id}  # Cancel running experiment
```

### Report Generation & Export
```
POST /export/report       # Generate PDF reports (Executive, Technical, Detailed, Comparison)
GET  /export/report/preview # Preview report sections and data summary
GET  /export/{run_id}     # Export clustering results (CSV, JSON)
```

### Embedding & Dimensionality Reduction
```
POST /embed               # Run dimensionality reduction (PCA, t-SNE, UMAP)
GET  /embed/methods       # Available embedding methods
GET  /embed/{job_id}      # Get embedding results
```

### System Monitoring
```
GET  /health              # System health status
GET  /metrics            # Performance metrics
GET  /version            # API version info
```

## 🛠️ Technology Stack

### Frontend Technologies
- **Framework**: React 19.1.1 with TypeScript 5.4.5
- **UI Library**: Material-UI (MUI) v7.3.2 with Grid2 components
- **State Management**: Zustand for global state, React hooks for local state
- **Routing**: React Router v7.8.2 with protected routes
- **Charts & Visualization**: 
  - Chart.js with react-chartjs-2 for 2D plots
  - Three.js via @react-three/fiber for 3D cluster visualization
  - D3.js for custom data visualizations
- **HTTP Client**: Axios with JWT interceptors and error handling
- **Development**: Vite build system, ESLint, Prettier, hot reload
- **Type Safety**: Strict TypeScript with zero compilation errors

### Backend Technologies
- **Framework**: FastAPI with automatic OpenAPI documentation
- **Language**: Python 3.8+ with type hints and async/await
- **Machine Learning**:
  - Scikit-learn for clustering algorithms
  - NumPy/Pandas for data processing
  - FAISS for GPU-accelerated similarity search
  - CUDA for GPU computation acceleration
- **Database**: 
  - PostgreSQL for production with multi-tenant schema
  - SQLAlchemy ORM with relationship management
  - Redis for caching and session storage
- **Authentication**: JWT tokens with bcrypt password hashing
- **PDF Generation**: ReportLab for professional report creation
- **Task Processing**: Background job processing for long-running operations
- **Monitoring**: Structured logging with correlation IDs

### Infrastructure & DevOps
- **Containerization**: Docker multi-stage builds for both services
- **Database**: Multi-tenant PostgreSQL with automated data isolation
- **Caching**: Redis for session storage and computed result caching
- **File Storage**: Local filesystem with planned cloud storage integration
- **Monitoring**: 
  - Health endpoints with system status checks
  - Performance metrics collection and analysis
  - Request tracing with correlation IDs
  - Structured JSON logging for observability
- **Security**: 
  - JWT authentication with role-based access control
  - Rate limiting to prevent abuse
  - Input validation and sanitization
  - CORS middleware for cross-origin requests
- **Development**: 
  - Automated port cleanup and process management
  - Type checking integration in build pipeline
  - Hot reload for rapid development

## 🔐 Security Architecture

### Authentication & Authorization
- **Multi-tenant JWT Authentication**: Secure token-based auth with tenant isolation
- **Role-Based Access Control (RBAC)**: Admin and user roles with permission scoping
- **Password Security**: bcrypt hashing with salt for password storage
- **Token Management**: Automatic token refresh with secure storage

### Data Security
- **Tenant Isolation**: Database-level data separation between organizations
- **Input Validation**: Comprehensive validation for all API inputs
- **SQL Injection Prevention**: Parameterized queries via SQLAlchemy ORM
- **File Upload Security**: Type validation and size limits for uploaded files

### API Security
- **Rate Limiting**: Configurable rate limits per endpoint and user
- **CORS Configuration**: Properly configured cross-origin resource sharing
- **Request Validation**: Automatic request/response validation via Pydantic
- **Error Handling**: Secure error responses without sensitive information leakage

## 📊 Core Features

### Clustering Algorithms
- **Spectral Clustering**: Gaussian RBF kernel with normalized Laplacian
- **K-Means**: Traditional centroid-based clustering with k-means++
- **DBSCAN**: Density-based clustering with noise detection
- **Agglomerative**: Hierarchical clustering with multiple linkage methods
- **Gaussian Mixture Models**: Probabilistic clustering with EM algorithm

### Performance Metrics
- **Silhouette Score**: Cluster cohesion and separation measurement
- **Davies-Bouldin Index**: Cluster compactness and separation ratio
- **Calinski-Harabasz Index**: Ratio of between-cluster to within-cluster dispersion
- **Adjusted Rand Index**: Similarity measure for cluster assignments
- **Execution Time**: Performance benchmarking for algorithm comparison

### Data Processing
- **CSV Upload**: Drag-and-drop file upload with validation
- **Data Preprocessing**: Missing value handling, outlier detection, normalization
- **Feature Selection**: Interactive column selection and filtering
- **Dimensionality Reduction**: PCA, t-SNE, UMAP for visualization
- **Statistical Analysis**: Comprehensive dataset statistics and recommendations

### Visualization & Reporting
- **Interactive 2D/3D Plots**: Real-time cluster visualization with WebGL
- **Performance Dashboards**: Metrics comparison across algorithms
- **Experiment Tracking**: Historical run management and comparison
- **PDF Report Generation**: Professional reports with multiple formats:
  - Executive Summary for stakeholders
  - Technical Report for researchers
  - Detailed Analysis for data scientists
  - Algorithm Comparison for decision making

### Grid Search & Optimization
- **Automated Hyperparameter Tuning**: Exhaustive parameter space exploration
- **Multi-Algorithm Comparison**: Parallel execution across different algorithms
- **Leaderboard System**: Ranking based on configurable optimization metrics
- **Real-time Progress Tracking**: Live updates during grid search execution

## 🚀 Deployment Architecture

### Development Environment
- **Frontend**: React development server on port 3000/3001
- **Backend**: FastAPI server on port 8000 with auto-reload
- **Database**: Local PostgreSQL or SQLite for development
- **Hot Reload**: Automatic code reloading for rapid development

### Production Deployment
- **Container Orchestration**: Docker Compose for multi-service deployment
- **Load Balancing**: Nginx reverse proxy for high availability
- **Database**: PostgreSQL with connection pooling and backup strategies
- **Caching**: Redis cluster for session storage and result caching
- **Monitoring**: Prometheus metrics with Grafana dashboards
- **Logging**: Centralized logging with ELK stack integration

### Scalability Features
- **Horizontal Scaling**: Stateless architecture supports multiple instances
- **Database Optimization**: Indexed queries and connection pooling
- **Caching Strategy**: Multi-level caching for frequently accessed data
- **Async Processing**: Background job processing for long-running operations
- **GPU Acceleration**: CUDA support for large-scale dataset processing

## 📈 Performance Characteristics

### Frontend Performance
- **Bundle Size**: Optimized with code splitting and lazy loading
- **Rendering**: Virtual DOM with React optimization techniques
- **Memory Management**: Efficient state management with automatic cleanup
- **Network**: HTTP/2 support with request batching and caching

### Backend Performance
- **Async Operations**: FastAPI async/await for concurrent request handling
- **Database**: Connection pooling and query optimization
- **Computation**: NumPy vectorization and optional GPU acceleration
- **Memory**: Efficient data structures with garbage collection optimization

### Machine Learning Performance
- **Algorithm Efficiency**: Optimized scikit-learn implementations
- **Large Dataset Support**: FAISS integration for datasets >10k points
- **GPU Acceleration**: CUDA-enabled operations where available
- **Memory Optimization**: Streaming data processing for large files

## 🔧 Development Workflow

### Code Organization
- **Feature-Based Structure**: Frontend organized by user-facing features
- **Service Layer Architecture**: Backend business logic in dedicated services
- **Database Abstraction**: ORM models with clear relationship management
- **API Design**: RESTful endpoints with consistent naming conventions

### Quality Assurance
- **Type Safety**: 100% TypeScript coverage with strict type checking
- **Code Standards**: ESLint and Prettier for consistent code formatting
- **Error Handling**: Comprehensive error boundaries and graceful failures
- **Testing**: Unit tests for critical business logic and API endpoints

### Monitoring & Observability
- **Health Checks**: Automated system health monitoring
- **Performance Metrics**: Request timing and resource utilization tracking
- **Error Tracking**: Structured error logging with stack traces
- **User Analytics**: Usage patterns and feature adoption tracking

This architecture provides a solid foundation for an enterprise-grade clustering platform with modern development practices, comprehensive security, and production-ready deployment capabilities.
