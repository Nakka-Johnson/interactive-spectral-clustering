# Interactive Spectral Clustering Platform

A comprehensive, research-grade clustering analysis platform designed for scientific data exploration, algorithm comparison, and experiment management. Built with modern web technologies for optimal performance and user experience.

> **GDPR Compliance**: This tool only processes user-supplied, non-sensitive CSV data. No personal data is stored or transmitted.

![Platform Overview](docs/screenshots/platform-overview.png)

## 🎯 Project Overview

The Interactive Spectral Clustering Platform is a full-stack web application that enables researchers and data scientists to:

- **Upload and explore** CSV datasets with comprehensive statistical analysis
- **Configure and compare** multiple clustering algorithms (K-means, Spectral, DBSCAN, Agglomerative, Gaussian Mixture Model)
- **Visualize results** in 2D/3D with interactive charts and dimensionality reduction
- **Evaluate performance** using multiple clustering metrics (Silhouette, Davies-Bouldin, Calinski-Harabasz)
- **Manage experiments** with history tracking, comparison tools, and report generation
- **Accelerate computation** with GPU support for large datasets

### Key Features

- 🚀 **GPU-Accelerated Computing**: CUDA support for large-scale clustering
- 📊 **Multi-Algorithm Support**: K-means, Spectral (manual + sklearn), DBSCAN, Agglomerative, Gaussian Mixture Model
- 🎨 **Advanced Visualization**: 2D/3D plots with PCA, t-SNE, UMAP dimensionality reduction
- 📈 **Comprehensive Metrics**: Silhouette, Davies-Bouldin, Calinski-Harabasz, Adjusted Rand Index
- 🔄 **Real-time Progress**: WebSocket-based progress tracking with live logs
- 📋 **Experiment Management**: History tracking, comparison tools, and detailed reports
- 🛡️ **Error Monitoring**: Sentry integration for production error tracking and performance monitoring
- 🎛️ **Scientific UI**: Research-grade interface with professional styling
- 📱 **Responsive Design**: Works seamlessly across desktop and tablet devices

## 🏗️ Architecture

The platform follows a modern microservices architecture with clear separation of concerns:

```mermaid
graph TB
    subgraph "Frontend Layer"
        UI[React TypeScript UI]
        Store[Zustand State Management]
        API[API Client Layer]
    end
    
    subgraph "Backend Layer"
        FastAPI[FastAPI Server]
        WS[WebSocket Handler]
        Auth[Authentication]
    end
    
    subgraph "Processing Layer"
        Cluster[Clustering Engine]
        GPU[GPU Acceleration]
        Eval[Metrics Evaluation]
    end
    
    subgraph "Data Layer"
        DB[(PostgreSQL Database)]
        Files[File Storage]
        Cache[Redis Cache]
    end
    
    UI --> Store
    Store --> API
    API --> FastAPI
    FastAPI --> WS
    FastAPI --> Cluster
    Cluster --> GPU
    Cluster --> Eval
    FastAPI --> DB
    FastAPI --> Files
    FastAPI --> Cache
    
    style UI fill:#e1f5fe
    style FastAPI fill:#f3e5f5
    style Cluster fill:#e8f5e8
    style DB fill:#fff3e0
```

For detailed architecture documentation, see [architecture.md](docs/architecture.md).

## 🚀 Quick Start

### Prerequisites

- **Python 3.9+** with CUDA support (optional, for GPU acceleration)
- **Node.js 16+** and npm
- **PostgreSQL 13+**
- **Docker & Docker Compose** (recommended)

### Option 1: Docker Setup (Recommended)

```bash
# Clone the repository
git clone <repository-url>
cd interactive-spectral-clustering

# Start all services with Docker Compose
docker-compose up -d

# The application will be available at:
# Frontend: http://localhost:3000
# Backend API: http://localhost:8000
# API Documentation: http://localhost:8000/docs
```

### Option 2: Manual Setup

#### Backend Setup

```bash
# Navigate to backend directory
cd backend

# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Set up environment variables
cp .env.example .env
# Edit .env with your database credentials

# Initialize database
alembic upgrade head

# Start the server
uvicorn app:app --reload --host 0.0.0.0 --port 8000
```

#### Frontend Setup

```bash
# Navigate to frontend directory
cd frontend

# Install dependencies
npm install

# Start development server
npm start

# Build for production
npm run build
```

#### Database Setup

```sql
-- Create PostgreSQL database
CREATE DATABASE clustering_platform;
CREATE USER clustering_user WITH PASSWORD 'your_password';
GRANT ALL PRIVILEGES ON DATABASE clustering_platform TO clustering_user;
```

## 📖 Usage Guide

### 1. Upload Data Tab
- **Drag & drop** or select CSV files for analysis
- **Preview data** with automatic column detection and type inference
- **Validate format** with built-in checks for data quality
- **View statistics** including missing values, data types, and basic metrics

### 2. Data Exploration Tab
- **Statistical overview** with descriptive statistics for all columns
- **Data quality assessment** including missing values and outliers
- **Column analysis** with histograms and distribution plots
- **Correlation analysis** with interactive correlation matrix

### 3. Configuration Tab
- **Algorithm selection**: Choose from K-means, Spectral, DBSCAN, Agglomerative, Gaussian Mixture Model
- **Parameter tuning**: Configure clusters, sigma, neighbors, and advanced options
- **Dimensionality reduction**: Select PCA, t-SNE, or UMAP for visualization
- **Data filtering**: Choose specific columns and apply row filters

### 4. Visualization Tab
- **2D/3D scatter plots** with cluster coloring and interactive controls
- **Dimensionality reduction** visualization with multiple algorithms
- **Cluster boundaries** and centroid visualization
- **Export options** for high-quality plots and data

### 5. Metrics Tab
- **Performance evaluation** using multiple clustering metrics
- **Silhouette analysis** with per-cluster and overall scores
- **Comparative analysis** across different algorithms
- **Statistical significance** testing for algorithm comparison

### 6. History Tab
- **Experiment tracking** with automatic saving of all runs
- **Search and filter** experiments by date, algorithm, or parameters
- **Comparison tools** for side-by-side analysis
- **Export capabilities** for experiment data and results

### 7. Report Generation Tab
- **PDF reports** with comprehensive analysis and visualizations
- **HTML reports** for interactive sharing
- **Custom templates** for different analysis types
- **Citation support** for academic publications

## 🔌 API Reference

The platform provides a comprehensive REST API for programmatic access:

### Authentication
```bash
# Get access token
POST /auth/login
{
  "username": "your_username",
  "password": "your_password"
}
```

### Data Upload
```bash
# Upload CSV file
POST /api/upload
Content-Type: multipart/form-data
```

### Clustering Operations
```bash
# Start clustering analysis
POST /api/clustering/run
{
  "job_id": "dataset_123",
  "methods": ["kmeans", "spectral"],
  "n_clusters": 3,
  "use_pca": false
}

# Get results
GET /api/clustering/results/{job_id}
```

### Real-time Updates
```javascript
// WebSocket connection for progress updates
const socket = io('ws://localhost:8000');
socket.on('progress_update', (data) => {
  console.log(`Progress: ${data.progress}% - ${data.message}`);
});
```

**Complete API Documentation**: [http://localhost:8000/docs](http://localhost:8000/docs) (Swagger UI)

**OpenAPI Specification**: [http://localhost:8000/openapi.json](http://localhost:8000/openapi.json)

## 🧪 Testing

### Backend Tests
```bash
cd backend
pytest tests/ -v --cov=app
```

### Frontend Tests
```bash
cd frontend
npm test
npm run test:coverage
```

### Integration Tests
```bash
# Run full test suite
docker-compose -f docker-compose.test.yml up --abort-on-container-exit
```

## 🚀 Deployment

### Production Docker Setup
```bash
# Build production images
docker-compose -f docker-compose.prod.yml build

# Deploy to production
docker-compose -f docker-compose.prod.yml up -d
```

### Environment Configuration
```bash
# Backend environment variables
DATABASE_URL=postgresql://user:pass@localhost/clustering_platform
REDIS_URL=redis://localhost:6379/0
SECRET_KEY=your-secret-key
CUDA_VISIBLE_DEVICES=0,1  # For GPU acceleration

# Frontend environment variables
REACT_APP_API_URL=http://localhost:8000
REACT_APP_WS_URL=ws://localhost:8000
```

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guide](CONTRIBUTING.md) for details.

### Development Setup
1. Fork the repository
2. Create a feature branch: `git checkout -b feature/amazing-feature`
3. Make your changes and add tests
4. Ensure all tests pass: `npm test && pytest`
5. Commit your changes: `git commit -m 'Add amazing feature'`
6. Push to the branch: `git push origin feature/amazing-feature`
7. Open a Pull Request

### Code Style
- **Backend**: Black + isort + flake8
- **Frontend**: Prettier + ESLint
- **Commits**: Conventional Commits format

## 📊 Performance & Benchmarks

### Supported Dataset Sizes
- **Small**: < 1,000 rows (< 1 second)
- **Medium**: 1,000 - 100,000 rows (1-30 seconds)
- **Large**: 100,000 - 1M rows (30 seconds - 5 minutes, GPU recommended)
- **Extra Large**: 1M+ rows (GPU required, 5+ minutes)

### GPU Acceleration
The platform supports NVIDIA CUDA for significant performance improvements:
- **K-means**: Up to 50x speedup on large datasets
- **Spectral**: Up to 20x speedup with GPU eigenvalue computation
- **Memory**: Efficiently handles datasets up to available GPU memory

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## GDPR & Compliance

This platform only processes user-provided, non-sensitive CSV/Excel data. Uploaded files and results are auto-deleted after 30 days. No personal data is stored or transmitted.

### Data Processing Principles
- **Purpose Limitation**: Data is processed solely for clustering analysis and visualization
- **Data Minimization**: Only necessary data columns are processed for analysis
- **Storage Limitation**: All uploaded files and clustering results are automatically purged after 30 days
- **Transparency**: Users have full control over data upload, processing, and deletion

### Privacy Protection
- **No Personal Data Collection**: The platform does not collect, store, or process personal identifiable information (PII)
- **User-Controlled Data**: Users maintain full ownership and control of their uploaded datasets
- **Local Processing**: All computation occurs on the server without data transmission to third parties
- **Secure Deletion**: Files are securely deleted from storage after expiration period

### Technical Safeguards
- **Encrypted Transmission**: All data uploads use HTTPS encryption
- **Access Control**: JWT-based authentication protects user sessions
- **Audit Trail**: System logs track data processing activities (without storing actual data)
- **Regular Cleanup**: Automated background tasks ensure data retention compliance

### User Rights
- **Right to Access**: Users can view their uploaded datasets and results through the platform
- **Right to Deletion**: Data is automatically deleted after 30 days, or users can request immediate deletion
- **Right to Rectification**: Users can re-upload corrected datasets at any time
- **Data Portability**: Users can export their results and analysis in standard formats

## 🙏 Acknowledgments

- **scikit-learn** for machine learning algorithms
- **FastAPI** for high-performance web framework
- **React** and **Material-UI** for modern frontend
- **CUDA** ecosystem for GPU acceleration
- **PostgreSQL** for robust data storage

## 📞 Support

- **Documentation**: [docs/](docs/)
- **Issues**: [GitHub Issues](https://github.com/your-repo/issues)
- **Discussions**: [GitHub Discussions](https://github.com/your-repo/discussions)
- **Email**: support@clustering-platform.com

---

**Built with ❤️ for the research community**
