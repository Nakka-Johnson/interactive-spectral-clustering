# API Documentation

## Interactive Spectral Clustering Platform API

### Base URL
- **Development**: `http://localhost:8000`
- **Production**: `https://your-domain.com/api`

### Authentication

The API uses JWT (JSON Web Tokens) for authentication. Include the token in the Authorization header:

```
Authorization: Bearer <your-jwt-token>
```

### Response Format

All API responses follow a consistent JSON format:

```json
{
  "success": true,
  "data": {...},
  "message": "Success message",
  "timestamp": "2024-01-01T00:00:00Z"
}
```

Error responses:
```json
{
  "success": false,
  "error": {
    "code": "ERROR_CODE",
    "message": "Error description",
    "details": {...}
  },
  "timestamp": "2024-01-01T00:00:00Z"
}
```

## Authentication Endpoints

### POST /auth/register

Register a new user account.

**Request Body:**
```json
{
  "username": "string",
  "email": "string",
  "password": "string",
  "first_name": "string",
  "last_name": "string"
}
```

**Response:**
```json
{
  "success": true,
  "data": {
    "user": {
      "id": 1,
      "username": "johndoe",
      "email": "john@example.com",
      "first_name": "John",
      "last_name": "Doe",
      "is_active": true,
      "created_at": "2024-01-01T00:00:00Z"
    },
    "access_token": "eyJ0eXAiOiJKV1QiLCJhbGc...",
    "refresh_token": "eyJ0eXAiOiJKV1QiLCJhbGc...",
    "token_type": "bearer"
  }
}
```

**cURL Example:**
```bash
curl -X POST "http://localhost:8000/auth/register" \
  -H "Content-Type: application/json" \
  -d '{
    "username": "johndoe",
    "email": "john@example.com",
    "password": "securepassword123",
    "first_name": "John",
    "last_name": "Doe"
  }'
```

### POST /auth/login

Authenticate user and receive JWT tokens.

**Request Body:**
```json
{
  "username": "string",
  "password": "string"
}
```

**Response:**
```json
{
  "success": true,
  "data": {
    "access_token": "eyJ0eXAiOiJKV1QiLCJhbGc...",
    "refresh_token": "eyJ0eXAiOiJKV1QiLCJhbGc...",
    "token_type": "bearer",
    "expires_in": 1800
  }
}
```

**cURL Example:**
```bash
curl -X POST "http://localhost:8000/auth/login" \
  -H "Content-Type: application/json" \
  -d '{
    "username": "johndoe",
    "password": "securepassword123"
  }'
```

### POST /auth/refresh

Refresh an expired access token using the refresh token.

**Request Body:**
```json
{
  "refresh_token": "string"
}
```

**Response:**
```json
{
  "success": true,
  "data": {
    "access_token": "eyJ0eXAiOiJKV1QiLCJhbGc...",
    "token_type": "bearer",
    "expires_in": 1800
  }
}
```

**cURL Example:**
```bash
curl -X POST "http://localhost:8000/auth/refresh" \
  -H "Content-Type: application/json" \
  -d '{
    "refresh_token": "eyJ0eXAiOiJKV1QiLCJhbGc..."
  }'
```

### POST /auth/logout

Logout and invalidate tokens.

**Headers:**
```
Authorization: Bearer <access-token>
```

**Response:**
```json
{
  "success": true,
  "message": "Successfully logged out"
}
```

**cURL Example:**
```bash
curl -X POST "http://localhost:8000/auth/logout" \
  -H "Authorization: Bearer eyJ0eXAiOiJKV1QiLCJhbGc..."
```

## User Management

### GET /users/me

Get current user profile.

**Headers:**
```
Authorization: Bearer <access-token>
```

**Response:**
```json
{
  "success": true,
  "data": {
    "id": 1,
    "username": "johndoe",
    "email": "john@example.com",
    "first_name": "John",
    "last_name": "Doe",
    "is_active": true,
    "created_at": "2024-01-01T00:00:00Z",
    "last_login": "2024-01-01T12:00:00Z"
  }
}
```

**cURL Example:**
```bash
curl -X GET "http://localhost:8000/users/me" \
  -H "Authorization: Bearer eyJ0eXAiOiJKV1QiLCJhbGc..."
```

### PUT /users/me

Update current user profile.

**Headers:**
```
Authorization: Bearer <access-token>
```

**Request Body:**
```json
{
  "first_name": "string",
  "last_name": "string",
  "email": "string"
}
```

**cURL Example:**
```bash
curl -X PUT "http://localhost:8000/users/me" \
  -H "Authorization: Bearer eyJ0eXAiOiJKV1QiLCJhbGc..." \
  -H "Content-Type: application/json" \
  -d '{
    "first_name": "John",
    "last_name": "Smith",
    "email": "john.smith@example.com"
  }'
```

## Dataset Management

### POST /datasets/upload

Upload a dataset for clustering analysis.

**Headers:**
```
Authorization: Bearer <access-token>
Content-Type: multipart/form-data
```

**Request Body (Form Data):**
- `file`: CSV file (required)
- `name`: Dataset name (optional)
- `description`: Dataset description (optional)

**Response:**
```json
{
  "success": true,
  "data": {
    "id": 1,
    "name": "customer_data.csv",
    "description": "Customer segmentation data",
    "filename": "customer_data.csv",
    "file_size": 1024000,
    "rows": 10000,
    "columns": 15,
    "upload_time": "2024-01-01T00:00:00Z",
    "status": "uploaded"
  }
}
```

**cURL Example:**
```bash
curl -X POST "http://localhost:8000/datasets/upload" \
  -H "Authorization: Bearer eyJ0eXAiOiJKV1QiLCJhbGc..." \
  -F "file=@customer_data.csv" \
  -F "name=Customer Segmentation Data" \
  -F "description=E-commerce customer behavior data"
```

### GET /datasets/

Get list of user's datasets.

**Headers:**
```
Authorization: Bearer <access-token>
```

**Query Parameters:**
- `page`: Page number (default: 1)
- `size`: Items per page (default: 10)
- `sort`: Sort field (default: created_at)
- `order`: Sort order (asc/desc, default: desc)

**Response:**
```json
{
  "success": true,
  "data": {
    "items": [
      {
        "id": 1,
        "name": "customer_data.csv",
        "description": "Customer segmentation data",
        "file_size": 1024000,
        "rows": 10000,
        "columns": 15,
        "upload_time": "2024-01-01T00:00:00Z",
        "status": "uploaded"
      }
    ],
    "total": 1,
    "page": 1,
    "size": 10,
    "pages": 1
  }
}
```

**cURL Example:**
```bash
curl -X GET "http://localhost:8000/datasets/?page=1&size=10" \
  -H "Authorization: Bearer eyJ0eXAiOiJKV1QiLCJhbGc..."
```

### GET /datasets/{dataset_id}

Get specific dataset information.

**Headers:**
```
Authorization: Bearer <access-token>
```

**Response:**
```json
{
  "success": true,
  "data": {
    "id": 1,
    "name": "customer_data.csv",
    "description": "Customer segmentation data",
    "filename": "customer_data.csv",
    "file_size": 1024000,
    "rows": 10000,
    "columns": 15,
    "column_names": ["age", "income", "spending_score", ...],
    "column_types": {"age": "int64", "income": "float64", ...},
    "upload_time": "2024-01-01T00:00:00Z",
    "status": "uploaded",
    "preview": [
      {"age": 25, "income": 50000, "spending_score": 75},
      {"age": 30, "income": 60000, "spending_score": 80}
    ]
  }
}
```

**cURL Example:**
```bash
curl -X GET "http://localhost:8000/datasets/1" \
  -H "Authorization: Bearer eyJ0eXAiOiJKV1QiLCJhbGc..."
```

### DELETE /datasets/{dataset_id}

Delete a dataset.

**Headers:**
```
Authorization: Bearer <access-token>
```

**Response:**
```json
{
  "success": true,
  "message": "Dataset deleted successfully"
}
```

**cURL Example:**
```bash
curl -X DELETE "http://localhost:8000/datasets/1" \
  -H "Authorization: Bearer eyJ0eXAiOiJKV1QiLCJhbGc..."
```

## Clustering Operations

### POST /clustering/jobs

Create a new clustering job.

**Headers:**
```
Authorization: Bearer <access-token>
```

**Request Body:**
```json
{
  "dataset_id": 1,
  "algorithm": "spectral",
  "parameters": {
    "n_clusters": 5,
    "gamma": 1.0,
    "affinity": "rbf",
    "assign_labels": "kmeans",
    "random_state": 42
  },
  "preprocessing": {
    "normalize": true,
    "scale": true,
    "remove_outliers": false
  },
  "name": "Customer Segmentation Analysis",
  "description": "Spectral clustering of customer behavior data"
}
```

**Response:**
```json
{
  "success": true,
  "data": {
    "id": 1,
    "name": "Customer Segmentation Analysis",
    "description": "Spectral clustering of customer behavior data",
    "dataset_id": 1,
    "algorithm": "spectral",
    "parameters": {...},
    "preprocessing": {...},
    "status": "queued",
    "created_at": "2024-01-01T00:00:00Z",
    "progress": 0
  }
}
```

**cURL Example:**
```bash
curl -X POST "http://localhost:8000/clustering/jobs" \
  -H "Authorization: Bearer eyJ0eXAiOiJKV1QiLCJhbGc..." \
  -H "Content-Type: application/json" \
  -d '{
    "dataset_id": 1,
    "algorithm": "spectral",
    "parameters": {
      "n_clusters": 5,
      "gamma": 1.0,
      "affinity": "rbf"
    },
    "name": "Customer Segmentation",
    "description": "Clustering analysis"
  }'
```

### GET /clustering/jobs

Get list of clustering jobs.

**Headers:**
```
Authorization: Bearer <access-token>
```

**Query Parameters:**
- `page`: Page number (default: 1)
- `size`: Items per page (default: 10)
- `status`: Filter by status (queued, running, completed, failed)

**Response:**
```json
{
  "success": true,
  "data": {
    "items": [
      {
        "id": 1,
        "name": "Customer Segmentation Analysis",
        "dataset_id": 1,
        "algorithm": "spectral",
        "status": "completed",
        "progress": 100,
        "created_at": "2024-01-01T00:00:00Z",
        "completed_at": "2024-01-01T00:05:00Z",
        "execution_time": 300
      }
    ],
    "total": 1,
    "page": 1,
    "size": 10,
    "pages": 1
  }
}
```

**cURL Example:**
```bash
curl -X GET "http://localhost:8000/clustering/jobs?status=completed" \
  -H "Authorization: Bearer eyJ0eXAiOiJKV1QiLCJhbGc..."
```

### GET /clustering/jobs/{job_id}

Get specific clustering job details.

**Headers:**
```
Authorization: Bearer <access-token>
```

**Response:**
```json
{
  "success": true,
  "data": {
    "id": 1,
    "name": "Customer Segmentation Analysis",
    "description": "Spectral clustering of customer behavior data",
    "dataset_id": 1,
    "algorithm": "spectral",
    "parameters": {...},
    "preprocessing": {...},
    "status": "completed",
    "progress": 100,
    "created_at": "2024-01-01T00:00:00Z",
    "started_at": "2024-01-01T00:00:30Z",
    "completed_at": "2024-01-01T00:05:00Z",
    "execution_time": 270,
    "metrics": {
      "silhouette_score": 0.85,
      "calinski_harabasz_score": 1234.56,
      "davies_bouldin_score": 0.42
    }
  }
}
```

**cURL Example:**
```bash
curl -X GET "http://localhost:8000/clustering/jobs/1" \
  -H "Authorization: Bearer eyJ0eXAiOiJKV1QiLCJhbGc..."
```

### GET /clustering/jobs/{job_id}/results

Get clustering results and visualizations.

**Headers:**
```
Authorization: Bearer <access-token>
```

**Response:**
```json
{
  "success": true,
  "data": {
    "job_id": 1,
    "clusters": [0, 1, 2, 0, 1, ...],
    "cluster_centers": [
      [1.2, 3.4, 5.6],
      [2.1, 4.3, 6.5],
      [3.2, 5.4, 7.6]
    ],
    "metrics": {
      "silhouette_score": 0.85,
      "calinski_harabasz_score": 1234.56,
      "davies_bouldin_score": 0.42,
      "inertia": 1000.5
    },
    "visualizations": {
      "scatter_plot": "base64_encoded_image",
      "cluster_distribution": {...},
      "silhouette_plot": "base64_encoded_image"
    },
    "statistics": {
      "total_points": 10000,
      "n_clusters": 5,
      "cluster_sizes": [2000, 2500, 1500, 3000, 1000]
    }
  }
}
```

**cURL Example:**
```bash
curl -X GET "http://localhost:8000/clustering/jobs/1/results" \
  -H "Authorization: Bearer eyJ0eXAiOiJKV1QiLCJhbGc..."
```

### DELETE /clustering/jobs/{job_id}

Cancel or delete a clustering job.

**Headers:**
```
Authorization: Bearer <access-token>
```

**Response:**
```json
{
  "success": true,
  "message": "Clustering job deleted successfully"
}
```

**cURL Example:**
```bash
curl -X DELETE "http://localhost:8000/clustering/jobs/1" \
  -H "Authorization: Bearer eyJ0eXAiOiJKV1QiLCJhbGc..."
```

## System Endpoints

### GET /health

Health check endpoint (no authentication required).

**Response:**
```json
{
  "success": true,
  "data": {
    "status": "healthy",
    "timestamp": "2024-01-01T00:00:00Z",
    "version": "1.0.0",
    "services": {
      "database": "healthy",
      "redis": "healthy",
      "gpu": "available"
    }
  }
}
```

**cURL Example:**
```bash
curl -X GET "http://localhost:8000/health"
```

### GET /metrics

System metrics (requires authentication).

**Headers:**
```
Authorization: Bearer <access-token>
```

**Response:**
```json
{
  "success": true,
  "data": {
    "system": {
      "cpu_usage": 45.2,
      "memory_usage": 67.8,
      "disk_usage": 23.4,
      "gpu_usage": 12.5
    },
    "application": {
      "active_users": 25,
      "active_jobs": 3,
      "total_jobs_today": 15,
      "total_datasets": 1250
    }
  }
}
```

**cURL Example:**
```bash
curl -X GET "http://localhost:8000/metrics" \
  -H "Authorization: Bearer eyJ0eXAiOiJKV1QiLCJhbGc..."
```

## WebSocket Connections

### /ws/clustering/{job_id}

Real-time clustering job progress updates.

**Connection:**
```javascript
const ws = new WebSocket('ws://localhost:8000/ws/clustering/1?token=eyJ0eXAiOiJKV1Q...');

ws.onmessage = function(event) {
  const data = JSON.parse(event.data);
  console.log('Progress:', data.progress);
};
```

**Message Format:**
```json
{
  "type": "progress",
  "job_id": 1,
  "progress": 75,
  "status": "running",
  "message": "Computing spectral embedding...",
  "timestamp": "2024-01-01T00:03:00Z"
}
```

## Error Codes

| Code | Description |
|------|-------------|
| `AUTH_001` | Invalid credentials |
| `AUTH_002` | Token expired |
| `AUTH_003` | Token invalid |
| `AUTH_004` | Insufficient permissions |
| `DATA_001` | Dataset not found |
| `DATA_002` | Invalid file format |
| `DATA_003` | File too large |
| `DATA_004` | Dataset processing failed |
| `CLUSTER_001` | Clustering job not found |
| `CLUSTER_002` | Invalid clustering parameters |
| `CLUSTER_003` | Clustering algorithm failed |
| `CLUSTER_004` | Job already completed |
| `SYSTEM_001` | Database connection error |
| `SYSTEM_002` | GPU not available |
| `SYSTEM_003` | Rate limit exceeded |

## Rate Limiting

The API implements rate limiting to prevent abuse:

- **Authentication endpoints**: 5 requests per minute
- **Data upload**: 10 requests per hour
- **Clustering jobs**: 20 requests per hour
- **General API**: 100 requests per minute

Rate limit headers are included in responses:
```
X-RateLimit-Limit: 100
X-RateLimit-Remaining: 99
X-RateLimit-Reset: 1640995200
```

When rate limit is exceeded, you'll receive a 429 status code:
```json
{
  "success": false,
  "error": {
    "code": "SYSTEM_003",
    "message": "Rate limit exceeded",
    "details": {
      "retry_after": 60
    }
  }
}
```
