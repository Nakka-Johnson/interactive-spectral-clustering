import os
import uuid
import pandas as pd
import numpy as np
import concurrent.futures
# import torch  # Moved to lazy initialization to avoid blocking import
import logging
from typing import List, Dict, Any, Optional
from fastapi import FastAPI, File, UploadFile, WebSocket, HTTPException, Depends, status
from fastapi.routing import APIRouter
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from pydantic import BaseModel
from sqlalchemy import create_engine, Column, String, Text, DateTime, Float, Integer
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.dialects.postgresql import UUID
from datetime import datetime, timedelta
from passlib.context import CryptContext
from jose import JWTError, jwt
import json
import asyncio
from io import StringIO
import sentry_sdk
from sentry_sdk.integrations.fastapi import FastApiIntegration

# Setup basic logging as fallback
logger = logging.getLogger(__name__)

# Initialize Sentry
sentry_dsn = os.getenv("SENTRY_DSN")
if sentry_dsn:
    sentry_sdk.init(
        dsn=sentry_dsn,
        integrations=[FastApiIntegration(auto_enable=True)],
        traces_sample_rate=1.0,  # Capture 100% of transactions for performance monitoring
        environment=os.getenv("ENVIRONMENT", "development"),
        send_default_pii=False,  # Don't send personally identifiable information
    )

# Move database imports and initialization to lazy loading to avoid blocking
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./clustering.db")

def initialize_database():
    """Initialize database - called during app startup"""
    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker
    from models import Dataset, ClusteringRun, ExperimentSession, SystemMetrics, DatabaseManager, init_database, Base
    
    engine = create_engine(DATABASE_URL)
    SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    init_database(engine)
    print("✅ Database initialized")
    return engine, SessionLocal

# JWT Configuration
SECRET_KEY = os.getenv("SECRET_KEY", "your-secret-key-change-in-production")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30

# Password hashing
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# OAuth2 scheme
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

# JWT Models
class Token(BaseModel):
    access_token: str
    token_type: str

class TokenData(BaseModel):
    username: Optional[str] = None

class User(BaseModel):
    username: str
    email: Optional[str] = None
    disabled: Optional[bool] = None

class UserInDB(User):
    hashed_password: str

# Fake users database (replace with real database in production)
fake_users_db = {
    "testuser": {
        "username": "testuser", 
        "email": "test@example.com",
        "hashed_password": "$2b$12$EixZaYVK1fsbw1ZfbX3OXePaWxn96p36WQoeG6Lruj3vjPGga31lW",  # secret
        "disabled": False,
    }
}

# Authentication helper functions
def verify_password(plain_password, hashed_password):
    return pwd_context.verify(plain_password, hashed_password)

def get_password_hash(password):
    return pwd_context.hash(password)

def get_user(db, username: str):
    if username in db:
        user_dict = db[username]
        return UserInDB(**user_dict)

def authenticate_user(fake_db, username: str, password: str):
    user = get_user(fake_db, username)
    if not user:
        return False
    if not verify_password(password, user.hashed_password):
        return False
    return user

def create_access_token(data: dict, expires_delta: Optional[timedelta] = None):
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=15)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

# PHASE 8: Authentication system will be loaded during startup to avoid blocking import
use_new_auth = False  # Will be set during startup

def initialize_auth_system():
    """Initialize authentication system during startup"""
    global use_new_auth
    try:
        from app.services.auth_service import (
            AuthService, get_current_user, get_current_active_user,
            require_role, require_admin, get_tenant_filter
        )
        from app.models.auth import User as AuthUser, UserRole
        from app.database.connection import get_db
        
        # Use new auth system
        print("✅ PHASE 8 authentication system loaded")
        use_new_auth = True
        return True
        
    except ImportError as e:
        print(f"⚠️ PHASE 8 auth system not available, using legacy auth: {e}")
        use_new_auth = False
        return False
    
    # Legacy authentication functions (fallback)
    async def get_current_user(token: str = Depends(oauth2_scheme)):
        credentials_exception = HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Could not validate credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )
        try:
            payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
            username: str = payload.get("sub")
            if username is None:
                raise credentials_exception
            token_data = TokenData(username=username)
        except JWTError:
            raise credentials_exception
        user = get_user(fake_users_db, username=token_data.username)
        if user is None:
            raise credentials_exception
        return user

    async def get_current_active_user(current_user: User = Depends(get_current_user)):
        if current_user.disabled:
            raise HTTPException(status_code=400, detail="Inactive user")
        return current_user

# Pydantic Models
class ClusterRequest(BaseModel):
    job_id: str
    methods: List[str]
    n_clusters: int = 3
    sigma: float = 1.0  # sigma default justified by von Luxburg 2007
    n_neighbors: int = 10
    use_approximate_knn: bool = False  # Use approximate k-NN for large datasets
    use_pca: bool = False
    dim_reducer: str = "pca"  # Options: "pca", "tsne", "umap"
    random_state: int = 42

class UploadResponse(BaseModel):
    job_id: str
    columns: List[str]
    numeric_columns: List[str]
    shape: List[int]

class ClusterResponse(BaseModel):
    labels: Dict[str, List[int]]
    coords2D: List[List[float]]
    coords3D: List[List[float]]
    metrics: Dict[str, Dict[str, float]]

# FastAPI App with enhanced monitoring
from fastapi.routing import APIRouter

# Initialize the FastAPI app first to ensure it's always available
app = FastAPI(title="Interactive Spectral Clustering Platform", version="1.0.0")

# PHASE 9: Security Configuration
import os
from typing import List

# Load allowed origins from environment
ALLOWED_ORIGINS = os.getenv("ALLOWED_ORIGINS", "http://localhost:3000,http://127.0.0.1:3000").split(",")

# Configure CORS with restricted origins (PHASE 9 Security)
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,  # Restricted origins instead of "*"
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],  # Explicit methods
    allow_headers=["*"],
    expose_headers=["X-RateLimit-Limit", "X-RateLimit-Remaining", "X-RateLimit-Reset"]
)

# PHASE 9: Initialize Security Middleware
try:
    from app.middleware import configure_security_middleware
    
    # Apply security middleware stack
    app = configure_security_middleware(app)
    print("✅ PHASE 9 security middleware initialized successfully")
    
except ImportError as e:
    print(f"⚠️ PHASE 9 security middleware not available: {e}")
    print("   Falling back to basic security settings")

# Initialize PHASE 7 monitoring services
monitoring_enabled = True
if monitoring_enabled:
    try:
        from app.services.logging_service import (
            StructuredLogger, RequestTrackingRoute, ErrorEnvelope, PerformanceMonitor
        )
        from app.services.health_service import HealthChecker
        from app.services.metrics_service import MetricsCollector, MetricsMiddleware
        
        # Initialize services
        structured_logger = StructuredLogger()
        app_logger = structured_logger.get_logger()
        performance_monitor = PerformanceMonitor()
        metrics_collector = MetricsCollector()
        
        # Create metrics middleware
        metrics_middleware = MetricsMiddleware(metrics_collector)
        
        # Use structured logger
        logger = app_logger
        
        # Create custom router with request tracking
        class MonitoredAPIRouter(APIRouter):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                self.route_class = RequestTrackingRoute
        
        # Update app description for monitoring
        app.description = "Production-ready clustering platform with comprehensive monitoring"
        
        # Initialize health checker (will be initialized during startup)
        health_checker = None
        
        # Add metrics middleware
        app.middleware("http")(metrics_middleware)
        
        logger.info("✅ PHASE 7 monitoring services initialized successfully")
        
    except ImportError as e:
        # Fallback to basic logging
        logging.basicConfig(level=logging.INFO)
        logger = logging.getLogger(__name__)
        logger.warning(f"⚠️ PHASE 7 monitoring services not available: {e}")
        
        health_checker = None
        ErrorEnvelope = None
        performance_monitor = None
        metrics_collector = None
else:
    # Use basic logging for testing
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    health_checker = None
    ErrorEnvelope = None
    performance_monitor = None
    metrics_collector = None
    logger.info("⚠️ PHASE 7 monitoring services disabled for testing")

# Background cleanup task management (moved to startup to avoid blocking import)
cleanup_task = None
cleanup_executor = None

def initialize_cleanup_system():
    """Initialize cleanup system - called during app startup"""
    global cleanup_executor, connection_manager
    if cleanup_executor is None:
        from concurrent.futures import ThreadPoolExecutor
        cleanup_executor = ThreadPoolExecutor(max_workers=1)
        print("✅ Cleanup system initialized")
    if connection_manager is None:
        connection_manager = ConnectionManager()
        print("🔗 Initialized ConnectionManager")
    return cleanup_executor

@app.on_event("startup")
async def startup_event():
    """FastAPI startup event - schedule background cleanup task"""
    global cleanup_task, engine, SessionLocal, health_checker
    print("🚀 Starting up Interactive Spectral Clustering Platform...")
    
    # Initialize database
    engine, SessionLocal = initialize_database()
    
    # Initialize authentication system
    initialize_auth_system()
    
    # Initialize health checker
    try:
        from app.services.health_service import HealthChecker
        health_checker = HealthChecker(SessionLocal)
        print("✅ Health checker initialized")
    except ImportError:
        print("⚠️ Health checker not available")
    
    # Initialize GPU detection
    detect_gpu()
    
    # Initialize cleanup system
    initialize_cleanup_system()
    
    # Schedule background cleanup task to run every 24 hours
    cleanup_task = asyncio.create_task(schedule_cleanup_task())
    print("📅 Scheduled background cleanup task (runs every 24 hours)")
    
    print("✅ Application startup completed")

@app.on_event("shutdown")
async def shutdown_event():
    """FastAPI shutdown event - cleanup resources"""
    global cleanup_task, cleanup_executor
    print("🛑 Shutting down Interactive Spectral Clustering Platform...")
    
    if cleanup_task:
        cleanup_task.cancel()
        try:
            await cleanup_task
        except asyncio.CancelledError:
            pass
    
    cleanup_executor.shutdown(wait=True)
    print("✅ Cleanup completed")

async def schedule_cleanup_task():
    """Background task that runs cleanup every 24 hours"""
    while True:
        try:
            # Wait 24 hours between cleanup runs
            await asyncio.sleep(24 * 60 * 60)  # 24 hours in seconds
            
            # Run cleanup in thread pool to avoid blocking
            loop = asyncio.get_event_loop()
            deleted_count = await loop.run_in_executor(cleanup_executor, run_database_cleanup)
            
            print(f"🧹 Background cleanup completed: {deleted_count} expired runs deleted")
            
        except asyncio.CancelledError:
            print("🛑 Cleanup task cancelled")
            break
        except Exception as e:
            print(f"❌ Error in cleanup task: {e}")
            # Continue the loop even if cleanup fails

def run_database_cleanup() -> int:
    """Run database cleanup in a separate thread"""
    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker
    from models import DatabaseManager
    
    try:
        # Create a new database session for cleanup
        engine = create_engine(DATABASE_URL)
        SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
        
        with SessionLocal() as session:
            db_manager = DatabaseManager(session)
            deleted_count = db_manager.cleanup_expired_runs()
            return deleted_count
    except Exception as e:
        print(f"❌ Database cleanup error: {e}")
        return 0

# GPU Detection (moved to function to avoid blocking import)
use_gpu = None
# Database globals (initialized during startup)
engine, SessionLocal = None, None
health_checker = None

def detect_gpu():
    """Detect GPU availability - called lazily to avoid blocking import"""
    global use_gpu
    if use_gpu is None:
        try:
            import torch  # Import torch only when needed
            use_gpu = torch.cuda.is_available()
            if use_gpu:
                print(f"✅ GPU detected: {torch.cuda.get_device_name(0)}")
                print(f"   CUDA Version: {torch.version.cuda}")
                print(f"   GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
            else:
                print("⚠️  GPU not available, using CPU for computations")
        except ImportError:
            use_gpu = False
            print("⚠️  PyTorch not available, using CPU for computations")
    return use_gpu

# CORS Middleware
allowed_origins = os.getenv("ALLOWED_ORIGINS", "http://localhost:3000,http://localhost:5173").split(",")
app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# PHASE 8: Include authentication routes
try:
    from app.routes.auth import router as auth_router
    app.include_router(auth_router)
    logger.info("✅ PHASE 8 authentication routes included")
except ImportError as e:
    logger.warning(f"⚠️ Authentication routes not available: {e}")

# Sentry ASGI Middleware (only if Sentry is configured)
if sentry_dsn:
    from sentry_sdk.integrations.asgi import SentryAsgiMiddleware
    app = SentryAsgiMiddleware(app)

# Exception handlers
@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request, exc: RequestValidationError):
    """
    Handle FastAPI validation errors.
    
    Returns HTTP 400 with structured error response containing:
    - error: "Validation failed" 
    - details: List of validation errors from Pydantic
    
    This handler catches all request validation errors (invalid JSON, 
    missing required fields, type mismatches, etc.)
    """
    return JSONResponse(
        status_code=400,
        content={
            "error": "Validation failed",
            "details": exc.errors()
        }
    )

@app.exception_handler(Exception)
async def global_exception_handler(request, exc: Exception):
    """
    Handle all other unhandled exceptions.
    
    Returns HTTP 500 with generic error message without exposing 
    internal implementation details or stack traces to clients.
    
    Note: Sentry will still capture the full exception details 
    for debugging purposes if configured.
    """
    return JSONResponse(
        status_code=500,
        content={"error": "Internal server error"}
    )

# Dependency
def get_db():
    """Get database session - requires SessionLocal to be initialized"""
    if SessionLocal is None:
        raise HTTPException(status_code=500, detail="Database not initialized")
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

def get_db_manager(db: Session = Depends(get_db)):
    """Get database manager instance"""
    from models import DatabaseManager
    return DatabaseManager(db)

# WebSocket connections storage
class ConnectionManager:
    def __init__(self):
        self.active_connections: Dict[str, WebSocket] = {}
        self.connection_metadata: Dict[str, Dict] = {}

    async def connect(self, websocket: WebSocket, run_id: str):
        await websocket.accept()
        self.active_connections[run_id] = websocket
        self.connection_metadata[run_id] = {
            "connected_at": datetime.utcnow(),
            "last_ping": datetime.utcnow()
        }

    def disconnect(self, run_id: str):
        if run_id in self.active_connections:
            del self.active_connections[run_id]
        if run_id in self.connection_metadata:
            del self.connection_metadata[run_id]

    async def send_progress(self, run_id: str, progress: int, message: str = "", status: str = "running"):
        """Send progress update with enhanced payload"""
        if run_id in self.active_connections:
            try:
                payload = {
                    "type": "progress",
                    "data": {
                        "progress": progress,
                        "message": message,
                        "status": status,
                        "run_id": run_id,
                        "timestamp": datetime.utcnow().isoformat()
                    }
                }
                await self.active_connections[run_id].send_json(payload)
                
                # Update last activity
                if run_id in self.connection_metadata:
                    self.connection_metadata[run_id]["last_ping"] = datetime.utcnow()
                    
            except Exception as e:
                print(f"Failed to send progress to {run_id}: {e}")
                self.disconnect(run_id)

    async def send_completion(self, run_id: str, result: dict):
        """Send completion notification with results"""
        if run_id in self.active_connections:
            try:
                payload = {
                    "type": "completion",
                    "data": {
                        "progress": 100,
                        "message": "Clustering completed successfully!",
                        "status": "done",
                        "run_id": run_id,
                        "result": result,
                        "timestamp": datetime.utcnow().isoformat()
                    }
                }
                await self.active_connections[run_id].send_json(payload)
            except Exception as e:
                print(f"Failed to send completion to {run_id}: {e}")
                self.disconnect(run_id)

    async def send_error(self, run_id: str, error_message: str):
        """Send error notification"""
        if run_id in self.active_connections:
            try:
                payload = {
                    "type": "error",
                    "data": {
                        "progress": 0,
                        "message": f"Error: {error_message}",
                        "status": "error",
                        "run_id": run_id,
                        "error": error_message,
                        "timestamp": datetime.utcnow().isoformat()
                    }
                }
                await self.active_connections[run_id].send_json(payload)
            except Exception as e:
                print(f"Failed to send error to {run_id}: {e}")
                self.disconnect(run_id)

# Initialize connection manager lazily 
manager = None

def get_connection_manager():
    """Get connection manager instance - initialize if needed"""
    global manager
    if manager is None:
        manager = ConnectionManager()
    return manager

async def run_clustering_background(run_id: str, request: ClusterRequest, db_manager: DatabaseManager):
    """Background task to run clustering with progress updates"""
    try:
        # Update status to running
        db_manager.update_run_status(run_id, status="running", started_at=datetime.utcnow())
        
        # Send initial progress
        await get_connection_manager().send_progress(run_id, 5, "Starting clustering run...", "running")
        
        # Get dataset from database
        dataset = db_manager.get_dataset_by_job_id(request.job_id)
        if not dataset:
            raise Exception("Dataset not found")
        
        # Send progress update
        await manager.send_progress(run_id, 10, "Loading data...", "running")
        
        # Load data from file path
        file_path = dataset.file_path
        if file_path.endswith('.xlsx'):
            df = pd.read_excel(file_path)
        else:
            df = pd.read_csv(file_path)
        
        # Get numeric columns from metadata
        numeric_columns = dataset.columns.get("numeric_columns", [])
        
        # Preprocess data
        await manager.send_progress(run_id, 20, "Preprocessing data...", "running")
        X = preprocess_data(df, numeric_columns, request.use_pca, request.dim_reducer, random_state=request.random_state, use_gpu=use_gpu)
        
        # Compute affinity matrix
        await manager.send_progress(run_id, 40, "Computing affinity matrix...", "running")
        affinity = compute_affinity_matrix(X, request.sigma, request.n_neighbors, use_gpu=use_gpu, use_approximate_knn=request.use_approximate_knn)
        
        # Compute Laplacian
        await manager.send_progress(run_id, 50, "Computing Laplacian...", "running")
        laplacian = compute_laplacian(affinity)
        
        # Run clustering methods in parallel
        await manager.send_progress(run_id, 70, "Running clustering algorithms...", "running")
        labels_dict = await run_clustering_methods(X, request.methods, request.n_clusters, random_state=request.random_state)
        
        # Compute metrics
        await manager.send_progress(run_id, 85, "Computing metrics...", "running")
        metrics = compute_metrics(X, labels_dict)
        
        # Get visualization coordinates
        await manager.send_progress(run_id, 95, "Generating visualizations...", "running")
        coords_2d, coords_3d = get_visualization_coordinates(X, random_state=request.random_state)
        
        # Prepare results
        results = {
            "labels": labels_dict,
            "coords2D": coords_2d,
            "coords3D": coords_3d,
            "metrics": metrics,
            "processing_info": {
                "data_shape": list(X.shape),
                "gpu_used": use_gpu,
                "preprocessing": {
                    "use_pca": request.use_pca,
                    "dim_reducer": request.dim_reducer
                }
            }
        }
        
        # Update run with results
        db_manager.update_run_results(
            run_id=run_id,
            results=results,
            status="done",
            completed_at=datetime.utcnow()
        )
        
        # Send completion notification
        await manager.send_completion(run_id, results)
        
    except Exception as e:
        # Update run status to failed
        db_manager.update_run_status(
            run_id,
            status="error",
            error_message=str(e),
            completed_at=datetime.utcnow()
        )
        
        # Send error notification
        await manager.send_error(run_id, str(e))

# Helper Functions
def detect_numeric_columns(df: pd.DataFrame) -> List[str]:
    """Auto-detect numeric columns in DataFrame"""
    numeric_cols = []
    for col in df.columns:
        try:
            pd.to_numeric(df[col], errors='raise')
            numeric_cols.append(col)
        except (ValueError, TypeError):
            # Try to convert string numbers
            try:
                df_temp = df[col].astype(str).str.replace(',', '')
                pd.to_numeric(df_temp, errors='raise')
                numeric_cols.append(col)
            except:
                continue
    return numeric_cols

def preprocess_data(df: pd.DataFrame, numeric_cols: List[str], use_pca: bool = False, dim_reducer: str = "pca", n_components: int = 2, random_state: int = 42, use_gpu: bool = False):
    """Preprocess data with scaling and optional dimensionality reduction (PCA, t-SNE, UMAP)"""
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA
    
    # Select numeric columns and handle missing values
    X = df[numeric_cols].copy()
    X = X.fillna(X.mean())
    
    if use_gpu and torch.cuda.is_available():
        # GPU implementation using PyTorch
        try:
            # Convert to PyTorch tensor and move to GPU
            X_tensor = torch.tensor(X.values, dtype=torch.float32).cuda()
            
            # Standardize using PyTorch
            X_mean = torch.mean(X_tensor, dim=0)
            X_std = torch.std(X_tensor, dim=0)
            X_scaled_tensor = (X_tensor - X_mean) / (X_std + 1e-8)  # Add small epsilon for numerical stability
            
            # Optional dimensionality reduction using PyTorch
            if use_pca and n_components < len(numeric_cols):
                if dim_reducer.upper() == 'PCA':
                    # Center the data
                    X_centered = X_scaled_tensor - torch.mean(X_scaled_tensor, dim=0)
                    
                    # Compute SVD for PCA
                    U, S, V = torch.svd(X_centered)
                    X_scaled_tensor = torch.mm(X_centered, V[:, :n_components])
                else:
                    # For t-SNE and UMAP, fall back to CPU implementation
                    X_scaled = X_scaled_tensor.cpu().numpy()
                    use_gpu = False  # Continue with CPU for dimensionality reduction
            
            # Convert back to numpy
            if use_gpu:  # Only if we didn't fall back to CPU
                X_scaled = X_scaled_tensor.cpu().numpy()
            
        except Exception as e:
            print(f"GPU computation failed: {e}, falling back to CPU")
            use_gpu = False
    
    if not use_gpu:
        # CPU implementation using scikit-learn
        if 'X_scaled' not in locals():  # Only scale if not already done
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
        
        # Optional dimensionality reduction
        if use_pca and n_components < len(numeric_cols):
            if dim_reducer.upper() == 'PCA':
                pca = PCA(n_components=n_components, random_state=random_state)
                X_scaled = pca.fit_transform(X_scaled)
            elif dim_reducer.upper() == 'TSNE':
                from sklearn.manifold import TSNE
                # sigma default justified by von Luxburg 2007
                tsne = TSNE(n_components=n_components, random_state=random_state)
                X_scaled = tsne.fit_transform(X_scaled)
            elif dim_reducer.upper() == 'UMAP':
                try:
                    import umap
                    # sigma default justified by von Luxburg 2007
                    umap_reducer = umap.UMAP(n_components=n_components, random_state=random_state)
                    X_scaled = umap_reducer.fit_transform(X_scaled)
                except ImportError:
                    print("UMAP not available, falling back to PCA")
                    pca = PCA(n_components=n_components, random_state=random_state)
                    X_scaled = pca.fit_transform(X_scaled)
            else:
                print(f"Unknown dimensionality reducer '{dim_reducer}', using PCA as fallback")
                pca = PCA(n_components=n_components, random_state=random_state)
                X_scaled = pca.fit_transform(X_scaled)
    
    return X_scaled

def compute_affinity_matrix(X: np.ndarray, sigma: float = 1.0, n_neighbors: int = 10, use_gpu: bool = False, use_approximate_knn: bool = False):
    """Compute affinity matrix for spectral clustering"""
    from sklearn.neighbors import kneighbors_graph, NearestNeighbors
    from sklearn.metrics.pairwise import rbf_kernel
    
    if use_gpu and torch.cuda.is_available():
        # GPU implementation using PyTorch
        try:
            # Convert to PyTorch tensor and move to GPU
            X_tensor = torch.tensor(X, dtype=torch.float32).cuda()
            
            # Compute RBF kernel using PyTorch
            gamma = 1 / (2 * sigma**2)
            # Compute pairwise squared distances
            X_norm = torch.sum(X_tensor**2, dim=1, keepdim=True)
            distances_sq = X_norm + X_norm.T - 2 * torch.mm(X_tensor, X_tensor.T)
            
            # RBF kernel
            affinity_tensor = torch.exp(-gamma * distances_sq)
            
            # Convert back to numpy for k-neighbors computation (sklearn doesn't support GPU)
            affinity = affinity_tensor.cpu().numpy()
            
        except Exception as e:
            print(f"GPU affinity computation failed: {e}, falling back to CPU")
            use_gpu = False
    
    if not use_gpu:
        # CPU implementation using scikit-learn
        affinity = rbf_kernel(X, gamma=1/(2*sigma**2))
    
    # Make sparse using k-neighbors (always on CPU for now)
    if use_approximate_knn:
        # Use approximate k-NN with NearestNeighbors for better performance on large datasets
        nbrs = NearestNeighbors(n_neighbors=n_neighbors, n_jobs=-1).fit(X)
        knn_graph = nbrs.kneighbors_graph(mode='connectivity')
    else:
        # Use standard k-neighbors graph
        knn_graph = kneighbors_graph(X, n_neighbors=n_neighbors, mode='connectivity')
    
    affinity = affinity * knn_graph.toarray()
    
    return affinity

def compute_laplacian(affinity: np.ndarray):
    """Compute normalized Laplacian matrix"""
    from scipy import sparse
    
    # Degree matrix
    degree = np.sum(affinity, axis=1)
    degree_inv_sqrt = np.power(degree, -0.5)
    degree_inv_sqrt[np.isinf(degree_inv_sqrt)] = 0
    
    # Normalized Laplacian
    D_inv_sqrt = sparse.diags(degree_inv_sqrt)
    A = sparse.csr_matrix(affinity)
    L_norm = sparse.eye(affinity.shape[0]) - D_inv_sqrt @ A @ D_inv_sqrt
    
    return L_norm

def run_single_clustering_method(method: str, X: np.ndarray, n_clusters: int, random_state: int = 42, **kwargs):
    """Run a single clustering method - designed for parallel execution"""
    from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering, SpectralClustering
    
    if method == "kmeans":
        clusterer = KMeans(n_clusters=n_clusters, random_state=random_state)
        labels = clusterer.fit_predict(X)
    elif method == "spectral":
        clusterer = SpectralClustering(n_clusters=n_clusters, random_state=random_state)
        labels = clusterer.fit_predict(X)
    elif method == "dbscan":
        eps = kwargs.get('eps', 0.5)
        clusterer = DBSCAN(eps=eps)
        labels = clusterer.fit_predict(X)
    elif method == "agglomerative":
        clusterer = AgglomerativeClustering(n_clusters=n_clusters)
        labels = clusterer.fit_predict(X)
    else:
        raise ValueError(f"Unknown clustering method: {method}")
    
    return method, labels.tolist()

async def run_clustering_methods(X: np.ndarray, methods: List[str], n_clusters: int, random_state: int = 42, **kwargs):
    """Run multiple clustering methods in parallel"""
    from concurrent.futures import ProcessPoolExecutor
    import asyncio
    
    results = {}
    
    # For small datasets or single method, run sequentially to avoid overhead
    if len(methods) == 1 or X.shape[0] < 1000:
        for method in methods:
            method_name, labels = run_single_clustering_method(method, X, n_clusters, random_state, **kwargs)
            results[method_name] = labels
        return results
    
    # Run clustering methods in parallel for larger datasets
    try:
        with ProcessPoolExecutor() as executor:
            # Submit all tasks
            futures = [
                executor.submit(run_single_clustering_method, method, X, n_clusters, random_state, **kwargs)
                for method in methods
            ]
            
            # Wait for all futures to complete and collect results
            for future in concurrent.futures.as_completed(futures):
                try:
                    method_name, labels = future.result()
                    results[method_name] = labels
                except Exception as e:
                    print(f"Error in clustering method: {e}")
                    continue
    
    except Exception as e:
        print(f"Parallel processing failed: {e}, falling back to sequential")
        # Fallback to sequential processing
        for method in methods:
            try:
                method_name, labels = run_single_clustering_method(method, X, n_clusters, random_state, **kwargs)
                results[method_name] = labels
            except Exception as e:
                print(f"Error in method {method}: {e}")
                continue
    
    return results

def compute_metrics(X: np.ndarray, labels_dict: Dict[str, List[int]]):
    """Compute clustering metrics"""
    from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
    
    metrics = {}
    
    for method, labels in labels_dict.items():
        labels_array = np.array(labels)
        if len(np.unique(labels_array)) > 1:  # Need at least 2 clusters for metrics
            try:
                silhouette = silhouette_score(X, labels_array)
                davies_bouldin = davies_bouldin_score(X, labels_array)
                calinski_harabasz = calinski_harabasz_score(X, labels_array)
                
                metrics[method] = {
                    "silhouette_score": float(silhouette),
                    "davies_bouldin_score": float(davies_bouldin),
                    "calinski_harabasz_score": float(calinski_harabasz)
                }
            except Exception as e:
                metrics[method] = {"error": str(e)}
        else:
            metrics[method] = {"error": "Insufficient clusters for metrics"}
    
    return metrics

def get_visualization_coordinates(X: np.ndarray, random_state: int = 42):
    """Get 2D and 3D coordinates for visualization"""
    from sklearn.decomposition import PCA
    from sklearn.manifold import TSNE
    
    n_features = X.shape[1]
    
    # 2D coordinates
    if n_features >= 2:
        if n_features == 2:
            # If already 2D, use as-is
            coords_2d = X.copy()
        else:
            # Use PCA to reduce to 2D
            pca_2d = PCA(n_components=2, random_state=random_state)
            coords_2d = pca_2d.fit_transform(X)
    else:
        # If only 1D, duplicate the dimension
        coords_2d = np.column_stack([X.flatten(), np.zeros(X.shape[0])])
    
    # 3D coordinates
    if n_features >= 3:
        # Use PCA to reduce to 3D
        pca_3d = PCA(n_components=3, random_state=random_state)
        coords_3d = pca_3d.fit_transform(X)
    elif n_features == 2:
        # If 2D, add a zero dimension
        coords_3d = np.column_stack([X, np.zeros(X.shape[0])])
    else:
        # If 1D, add two zero dimensions
        coords_3d = np.column_stack([X.flatten(), np.zeros(X.shape[0]), np.zeros(X.shape[0])])
    
    return coords_2d.tolist(), coords_3d.tolist()

# API Endpoints

@app.post("/token", response_model=Token)
async def login_for_access_token(form_data: OAuth2PasswordRequestForm = Depends()):
    """JWT Token endpoint for authentication"""
    user = authenticate_user(fake_users_db, form_data.username, form_data.password)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"sub": user.username}, expires_delta=access_token_expires
    )
    return {"access_token": access_token, "token_type": "bearer"}

@app.post("/upload", response_model=UploadResponse)
async def upload_csv(
    file: UploadFile = File(...), 
    db_manager: DatabaseManager = Depends(get_db_manager),
    current_user: User = Depends(get_current_active_user)
):
    """Upload and process CSV or Excel file with enhanced validation"""
    try:
        # 1. Check file size limit (10MB)
        if file.size and file.size > 10 * 1024 * 1024:
            raise HTTPException(status_code=413, detail="File too large. Maximum size is 10MB")
        
        # 2. Check file content type
        allowed_content_types = [
            'text/csv',
            'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
            'application/csv',  # Some browsers send this for CSV
            'text/plain'        # Some browsers send this for CSV
        ]
        
        if file.content_type not in allowed_content_types:
            raise HTTPException(
                status_code=400, 
                detail=f"Unsupported file type: {file.content_type}. Supported types: CSV (.csv) and Excel (.xlsx)"
            )
        
        # Read file content
        content = await file.read()
        
        # Additional size check after reading (in case size wasn't provided in headers)
        if len(content) > 10 * 1024 * 1024:
            raise HTTPException(status_code=413, detail="File too large. Maximum size is 10MB")
        
        # 3. Parse file based on type and extension
        filename_lower = file.filename.lower() if file.filename else ""
        
        if filename_lower.endswith('.xlsx') or file.content_type == 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet':
            # Handle Excel files
            try:
                from io import BytesIO
                df = pd.read_excel(BytesIO(content))
                file_content_str = ""  # Excel files don't store as raw text
            except Exception as e:
                raise HTTPException(status_code=400, detail=f"Error reading Excel file: {str(e)}")
        
        elif filename_lower.endswith('.csv') or file.content_type in ['text/csv', 'application/csv', 'text/plain']:
            # Handle CSV files
            try:
                csv_content = content.decode('utf-8')
                df = pd.read_csv(StringIO(csv_content), on_bad_lines='error')
                file_content_str = csv_content
            except UnicodeDecodeError:
                # Try with different encodings
                try:
                    csv_content = content.decode('latin-1')
                    df = pd.read_csv(StringIO(csv_content), on_bad_lines='error')
                    file_content_str = csv_content
                except Exception as e:
                    raise HTTPException(status_code=400, detail=f"Error decoding CSV file: {str(e)}")
            except Exception as e:
                raise HTTPException(status_code=400, detail=f"Error parsing CSV file: {str(e)}")
        
        else:
            raise HTTPException(
                status_code=400, 
                detail="File extension not recognized. Please use .csv or .xlsx files"
            )
        
        # 4. Check column limit
        if df.shape[1] > 1000:
            raise HTTPException(status_code=400, detail="Too many columns. Maximum allowed is 1000 columns")
        
        # Basic data validation
        if df.empty:
            raise HTTPException(status_code=400, detail="File is empty or contains no data")
        
        if df.shape[0] == 0:
            raise HTTPException(status_code=400, detail="File contains no data rows")
        
        # Generate job ID
        job_id = str(uuid.uuid4())
        
        # Auto-detect numeric columns
        numeric_columns = detect_numeric_columns(df)
        
        if not numeric_columns:
            raise HTTPException(status_code=400, detail="No numeric columns found. At least one numeric column is required for clustering")
        
        # Save file to disk for efficiency (instead of storing in database)
        import os
        upload_dir = "uploads"
        os.makedirs(upload_dir, exist_ok=True)
        file_path = os.path.join(upload_dir, f"{job_id}_{file.filename}")
        
        with open(file_path, "wb") as f:
            f.write(content)
        
        # Prepare comprehensive column metadata as JSON
        columns_metadata = {
            "all_columns": df.columns.tolist(),
            "numeric_columns": numeric_columns,
            "categorical_columns": [col for col in df.columns if col not in numeric_columns],
            "data_types": df.dtypes.astype(str).to_dict(),
            "shape": list(df.shape),
            "missing_values": df.isnull().sum().to_dict(),
            "file_size_bytes": len(content),
            "filename": file.filename
        }
        
        # Store in database using refactored model
        dataset = db_manager.create_dataset(
            job_id=job_id,
            file_path=file_path,
            columns=columns_metadata
        )
        
        return UploadResponse(
            job_id=job_id,
            columns=df.columns.tolist(),
            numeric_columns=numeric_columns,
            shape=list(df.shape)
        )
        
    except HTTPException:
        # Re-raise HTTP exceptions as-is
        raise
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error processing file: {str(e)}")

@app.websocket("/ws/progress/{run_id}")
async def websocket_progress(websocket: WebSocket, run_id: str):
    """WebSocket endpoint for streaming progress updates with JWT authentication"""
    # Note: WebSocket JWT authentication is handled via query parameters
    # Usage: ws://localhost:8002/ws/progress/{run_id}?token=your_jwt_token
    try:
        # Accept the WebSocket connection
        await get_connection_manager().connect(websocket, run_id)
        
        # Send initial connection confirmation
        await manager.send_progress(run_id, 0, "Connected to progress stream")
        
        # Keep connection alive and handle any incoming messages
        while True:
            try:
                message = await websocket.receive_text()
                # Could handle client commands here (pause, cancel, etc.)
                if message == "ping":
                    await websocket.send_json({"type": "pong", "timestamp": datetime.utcnow().isoformat()})
            except Exception as e:
                print(f"WebSocket message error: {e}")
                break
    except Exception as e:
        print(f"WebSocket connection error: {e}")
    finally:
        manager.disconnect(run_id)

@app.post("/cluster", response_model=ClusterResponse)
async def cluster_data(
    request: ClusterRequest, 
    db_manager: DatabaseManager = Depends(get_db_manager),
    current_user: User = Depends(get_current_active_user)
):
    """Legacy endpoint - now redirects to new run-based system"""
    try:
        # Start a new clustering run using the new system
        run_result = await start_clustering_run(request, db_manager, current_user)
        run_id = run_result["run_id"]
        
        # Wait for completion or timeout (for backward compatibility)
        timeout_seconds = 300  # 5 minutes timeout
        start_time = datetime.utcnow()
        
        while True:
            await asyncio.sleep(1)  # Check every second
            
            # Check if timeout reached
            if (datetime.utcnow() - start_time).total_seconds() > timeout_seconds:
                raise HTTPException(status_code=408, detail="Clustering request timed out")
            
            # Check run status
            run = db_manager.get_clustering_run(run_id)
            if not run:
                raise HTTPException(status_code=404, detail="Run not found")
            
            if run.status == "done":
                # Extract results from metrics field
                results = run.metrics or {}
                return ClusterResponse(
                    labels=results.get("labels", {}),
                    coords2D=results.get("coords2D", []),
                    coords3D=results.get("coords3D", []),
                    metrics=results.get("metrics", {})
                )
            elif run.status == "error":
                raise HTTPException(status_code=500, detail=f"Clustering failed: {run.error_message}")
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Clustering failed: {str(e)}")

@app.get("/runs/{run_id}")
async def get_run_status(
    run_id: str,
    db_manager: DatabaseManager = Depends(get_db_manager),
    current_user: User = Depends(get_current_active_user)
):
    """Get current status and progress of a clustering run (fallback for WebSocket)"""
    try:
        # Get run from database
        run = db_manager.get_clustering_run(run_id)
        if not run:
            raise HTTPException(status_code=404, detail="Run not found")
        
        # Calculate progress based on status
        progress = 0
        if run.status == "queued":
            progress = 0
        elif run.status == "running":
            # If running, estimate progress based on time elapsed
            if run.started_at:
                elapsed = (datetime.utcnow() - run.started_at).total_seconds()
                # Rough estimate: most runs complete within 60 seconds
                progress = min(90, int(elapsed / 60 * 100))
            else:
                progress = 10
        elif run.status == "done":
            progress = 100
        elif run.status == "error":
            progress = 0
        
        # Return status information
        return {
            "run_id": run_id,
            "status": run.status,
            "progress": progress,
            "message": run.error_message if run.status == "error" else f"Run is {run.status}",
            "started_at": run.started_at.isoformat() if run.started_at else None,
            "completed_at": run.completed_at.isoformat() if run.completed_at else None,
            "algorithm": run.algorithm,
            "parameters": run.params,
            "metrics": run.metrics if run.status == "done" else None,
            "error_message": run.error_message,
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving run status: {str(e)}")

@app.post("/runs", response_model=dict)
async def start_clustering_run(
    request: ClusterRequest, 
    db_manager: DatabaseManager = Depends(get_db_manager),
    current_user: User = Depends(get_current_active_user)
):
    """Start a new clustering run and return run_id for progress tracking"""
    try:
        # Generate unique run ID
        run_id = str(uuid.uuid4())
        
        # Get dataset from database
        dataset = db_manager.get_dataset_by_job_id(request.job_id)
        if not dataset:
            raise HTTPException(status_code=404, detail="Dataset not found")
        
        # Create run record
        run = db_manager.create_clustering_run(
            job_id=request.job_id,
            methods=request.methods,
            params=request.dict(),
            run_id=run_id,
            status="queued"
        )
        
        # Start clustering as background task
        asyncio.create_task(
            run_clustering_background(run_id, request, db_manager)
        )
        
        return {
            "run_id": run_id,
            "status": "queued",
            "message": "Clustering run started. Use WebSocket or polling to track progress.",
            "websocket_url": f"/ws/progress/{run_id}",
            "polling_url": f"/runs/{run_id}"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error starting clustering run: {str(e)}")

@app.post("/cluster-advanced")
async def cluster_data_advanced(
    dataset_id: str,
    algorithms: List[str], 
    algorithm_params: Dict[str, Dict[str, Any]],
    use_gpu: bool = True,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """
    Run advanced clustering algorithms with detailed parameter control.
    
    This endpoint uses the new factory pattern to run sophisticated
    clustering algorithms with GPU acceleration and detailed parameter control.
    """
    try:
        # Get database manager
        db_manager = DatabaseManager(db)
        
        # Find dataset
        dataset = db_manager.get_dataset(dataset_id)
        if not dataset:
            raise HTTPException(status_code=404, detail="Dataset not found")
        
        # Load dataset
        data_dict = json.loads(dataset.data)
        if 'data' not in data_dict:
            raise HTTPException(status_code=400, detail="Invalid dataset format")
        
        X = np.array(data_dict['data'])
        
        # Generate run ID
        run_id = str(uuid.uuid4())
        
        # Import advanced clustering function
        from clustering import run_advanced_clustering
        
        # Run advanced clustering algorithms
        results = await run_advanced_clustering(
            X=X,
            algorithms=algorithms,
            params=algorithm_params,
            job_id=run_id,
            use_gpu=use_gpu
        )
        
        # Store results in database
        run_data = {
            'run_id': run_id,
            'dataset_id': dataset_id,
            'algorithms': algorithms,
            'algorithm_params': algorithm_params,
            'results': results,
            'use_gpu': use_gpu,
            'status': 'completed'
        }
        
        db_manager.create_clustering_run(
            job_id=run_id,
            methods=algorithms,
            params=run_data,
            run_id=run_id,
            status="completed"
        )
        
        return {
            "run_id": run_id,
            "status": "completed",
            "results": results,
            "message": "Advanced clustering completed successfully"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error running advanced clustering: {str(e)}")

@app.get("/algorithms/advanced")
async def get_advanced_algorithms(current_user: User = Depends(get_current_active_user)):
    """Get available advanced clustering algorithms and their default parameters."""
    try:
        from app.services.clustering.factory import clustering_factory
        
        algorithms = clustering_factory.get_available_algorithms()
        algorithm_info = {}
        
        for alg in algorithms:
            algorithm_info[alg] = clustering_factory.get_algorithm_info(alg)
        
        return {
            "algorithms": algorithms,
            "algorithm_info": algorithm_info
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting algorithm info: {str(e)}")

@app.post("/grid-search")
async def create_grid_search(
    request: Dict[str, Any],  # Using Dict instead of GridSearchRequest to avoid import issues for now
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """
    Create and start a grid search experiment for parameter optimization.
    
    This endpoint accepts a grid search request, generates all parameter combinations,
    and schedules the experiments to run in the background.
    """
    try:
        # Import here to avoid circular imports
        from app.services.grid_search_service import grid_search_service
        from app.schemas.grid_search import GridSearchRequest
        
        # Parse request
        grid_request = GridSearchRequest(**request)
        
        # Get database manager
        db_manager = DatabaseManager(db)
        
        # Find dataset
        dataset = db_manager.get_dataset(grid_request.dataset_id)
        if not dataset:
            raise HTTPException(status_code=404, detail="Dataset not found")
        
        # Load dataset
        data_dict = json.loads(dataset.data)
        if 'data' not in data_dict:
            raise HTTPException(status_code=400, detail="Invalid dataset format")
        
        X = np.array(data_dict['data'])
        
        # Create experiment
        experiment = grid_search_service.create_grid_search_experiment(grid_request, X)
        
        # Start experiment in background
        await grid_search_service.start_experiment(
            experiment.group_id,
            X,
            use_gpu=grid_request.use_gpu,
            max_concurrent=grid_request.max_concurrent_runs
        )
        
        # Estimate duration (rough estimate: 30 seconds per run)
        estimated_seconds = experiment.total_runs * 30
        estimated_duration = f"{estimated_seconds // 60}m {estimated_seconds % 60}s"
        
        return {
            "group_id": experiment.group_id,
            "message": f"Grid search experiment started with {experiment.total_runs} runs",
            "total_runs": experiment.total_runs,
            "estimated_duration": estimated_duration,
            "polling_url": f"/grid-search/{experiment.group_id}",
            "websocket_url": f"/ws/grid-search/{experiment.group_id}"
        }
        
    except Exception as e:
        logger.error(f"Error creating grid search: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error creating grid search: {str(e)}")

@app.get("/grid-search/{group_id}")
async def get_grid_search_status(
    group_id: str,
    current_user: User = Depends(get_current_active_user)
):
    """Get the status and results of a grid search experiment."""
    try:
        from app.services.grid_search_service import grid_search_service
        
        summary = grid_search_service.get_experiment_summary(group_id)
        if not summary:
            raise HTTPException(status_code=404, detail="Grid search experiment not found")
        
        return summary
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting grid search status: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error getting grid search status: {str(e)}")

@app.get("/grid-search/{group_id}/details")
async def get_grid_search_details(
    group_id: str,
    current_user: User = Depends(get_current_active_user)
):
    """Get detailed results of a grid search experiment including all runs."""
    try:
        from app.services.grid_search_service import grid_search_service
        
        experiment = grid_search_service.get_experiment(group_id)
        if not experiment:
            raise HTTPException(status_code=404, detail="Grid search experiment not found")
        
        return experiment
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting grid search details: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error getting grid search details: {str(e)}")

@app.get("/leaderboard")
async def get_leaderboard(
    limit: int = 50,
    metric: str = "silhouette_score",
    current_user: User = Depends(get_current_active_user)
):
    """
    Get leaderboard of best clustering runs across all experiments.
    
    Args:
        limit: Maximum number of entries to return (default: 50)
        metric: Optimization metric to sort by (default: silhouette_score)
    """
    try:
        from app.services.grid_search_service import grid_search_service
        
        leaderboard = grid_search_service.get_leaderboard(
            limit=min(limit, 100),  # Cap at 100 entries
            optimization_metric=metric
        )
        
        return {
            "metric": metric,
            "total_entries": len(leaderboard),
            "leaderboard": leaderboard
        }
        
    except Exception as e:
        logger.error(f"Error getting leaderboard: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error getting leaderboard: {str(e)}")

@app.delete("/grid-search/{group_id}")
async def cancel_grid_search(
    group_id: str,
    current_user: User = Depends(get_current_active_user)
):
    """Cancel a running grid search experiment."""
    try:
        from app.services.grid_search_service import grid_search_service
        
        experiment = grid_search_service.get_experiment(group_id)
        if not experiment:
            raise HTTPException(status_code=404, detail="Grid search experiment not found")
        
        grid_search_service.cancel_experiment(group_id)
        
        return {"message": f"Grid search experiment {group_id} cancelled"}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error cancelling grid search: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error cancelling grid search: {str(e)}")

# =============================================================================
# REPORT EXPORT ENDPOINTS
# =============================================================================

@app.post("/export/report")
async def export_report(
    report_type: str,
    experiment_name: str = "Clustering Analysis",
    include_sections: List[str] = ["executive_summary", "dataset_overview", "methods", "results", "metrics", "conclusions"],
    current_user: User = Depends(get_current_active_user)
):
    """Export a PDF report with selectable report types and sections."""
    try:
        # Validate report type
        valid_report_types = ['executive', 'detailed', 'technical', 'comparison']
        if report_type not in valid_report_types:
            raise HTTPException(
                status_code=400, 
                detail=f"Invalid report type. Must be one of: {', '.join(valid_report_types)}"
            )
        
        # Get the latest clustering results for the user
        # In a real implementation, you'd get this from the database
        # For now, we'll create mock data
        
        # Mock dataset stats
        dataset_stats = {
            "shape": [1000, 10],
            "memory_usage": 2.5,
            "column_names": [f"feature_{i}" for i in range(10)]
        }
        
        # Mock run data
        run_data = {
            "id": "report_001",
            "algorithm": "spectral",
            "dataset_name": experiment_name,
            "parameters": {
                "n_clusters": 3,
                "gamma": 1.0,
                "affinity": "rbf"
            },
            "execution_time": 2.45
        }
        
        # Mock clustering labels
        labels = np.random.randint(0, 3, 1000)  # 3 clusters, 1000 samples
        
        # Mock preprocessing info
        preprocessing_info = {
            "steps_applied": [
                "StandardScaler normalization",
                "Missing value imputation",
                "Feature selection"
            ],
            "removed_columns": ["col_1", "col_5"]
        }
        
        # Mock experiment results for comparison reports
        experiment_results = [
            {
                "algorithm": "spectral",
                "n_clusters": 3,
                "silhouette_score": 0.75,
                "execution_time": 2.45
            },
            {
                "algorithm": "dbscan",
                "n_clusters": 4,
                "silhouette_score": 0.68,
                "execution_time": 1.23
            },
            {
                "algorithm": "gmm",
                "n_clusters": 5,
                "silhouette_score": 0.72,
                "execution_time": 3.10
            }
        ]
        
        # Generate PDF report
        if export_service is None:
            raise HTTPException(status_code=500, detail="Export service not available")
        
        pdf_content = export_service.generate_pdf_report(
            report_type=report_type,
            run_data=run_data,
            labels=labels,
            dataset_stats=dataset_stats,
            preprocessing_info=preprocessing_info,
            experiment_results=experiment_results if report_type == 'comparison' else None
        )
        
        # Create response with PDF
        filename = f"{report_type}_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
        
        return Response(
            content=pdf_content,
            media_type="application/pdf",
            headers={
                "Content-Disposition": f"attachment; filename={filename}",
                "Content-Type": "application/pdf"
            }
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error generating PDF report: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error generating PDF report: {str(e)}")

@app.get("/export/report/preview")
async def get_report_preview(
    report_type: str,
    current_user: User = Depends(get_current_active_user)
):
    """Get a preview of the report sections and data summary."""
    try:
        # Validate report type
        valid_report_types = ['executive', 'detailed', 'technical', 'comparison']
        if report_type not in valid_report_types:
            raise HTTPException(
                status_code=400, 
                detail=f"Invalid report type. Must be one of: {', '.join(valid_report_types)}"
            )
        
        # Define sections for each report type
        sections_map = {
            'executive': [
                "Executive Summary",
                "Project Overview", 
                "Key Findings",
                "Dataset Summary",
                "Recommendations"
            ],
            'detailed': [
                "Executive Summary",
                "Methodology",
                "Algorithm Parameters",
                "Clustering Results",
                "Performance Metrics",
                "Dataset Information"
            ],
            'technical': [
                "Technical Overview",
                "Data Preprocessing",
                "Algorithm Implementation",
                "Performance Metrics",
                "Computational Details",
                "Code Documentation"
            ],
            'comparison': [
                "Comparison Overview",
                "Dataset Information",
                "Algorithm Performance Table",
                "Metric Comparisons",
                "Recommendations"
            ]
        }
        
        # Mock data summary
        data_summary = {
            "dataset_records": 1000,
            "analyses_count": 3,
            "estimated_pages": {
                'executive': 2,
                'detailed': 5,
                'technical': 8,
                'comparison': 4
            }.get(report_type, 3)
        }
        
        return {
            "report_type": report_type,
            "sections": sections_map.get(report_type, []),
            "data_summary": data_summary
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting report preview: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error getting report preview: {str(e)}")

# =============================================================================
# CORE ENDPOINTS
# =============================================================================@app.get("/")
async def root():
    """Health check endpoint"""
    return {"message": "Interactive Spectral Clustering Platform API", "version": "1.0.0"}

@app.get("/users/me", response_model=User)
async def read_users_me(current_user: User = Depends(get_current_active_user)):
    """Get current user information"""
    return current_user

@app.get("/sentry-debug")
async def trigger_error():
    """Test endpoint to trigger Sentry error reporting (development only)"""
    if os.getenv("ENVIRONMENT") == "development":
        division_by_zero = 1 / 0
    return {"message": "This endpoint is only available in development mode"}

# =============================================================================
# EMBEDDING ENDPOINTS (PHASE 5)
# =============================================================================

# Global embedding service instance
embedding_service = None

def get_embedding_service():
    """Get or create the global embedding service instance."""
    global embedding_service
    if embedding_service is None:
        from app.services.embedding_service import EmbeddingService
        embedding_service = EmbeddingService()
    return embedding_service

@app.get("/embed/methods")
async def get_embedding_methods():
    """Get available embedding methods and their parameters."""
    try:
        service = get_embedding_service()
        return service.get_available_methods()
    except Exception as e:
        logger.error(f"Error getting embedding methods: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error getting embedding methods: {str(e)}")

@app.post("/embed")
async def generate_embedding(request: dict, db: Session = Depends(get_db)):
    """Generate 2D embedding for visualization."""
    try:
        # Import here to avoid circular imports
        from app.services.embedding_service import EmbeddingService
        from app.schemas.embedding import EmbeddingRequest
        
        # Parse request
        embedding_request = EmbeddingRequest(**request)
        
        # Get database manager
        db_manager = DatabaseManager(db)
        
        # Find dataset
        dataset = db_manager.get_dataset(embedding_request.dataset_id)
        if not dataset:
            raise HTTPException(status_code=404, detail="Dataset not found")
        
        # Load dataset
        data_dict = json.loads(dataset.data)
        if 'data' not in data_dict:
            raise HTTPException(status_code=400, detail="Invalid dataset format")
        
        X = np.array(data_dict['data'])
        
        # Generate embedding
        service = get_embedding_service()
        result = await service.generate_embedding(embedding_request, X)
        
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error generating embedding: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error generating embedding: {str(e)}")

@app.delete("/embed/cache")
async def clear_embedding_cache():
    """Clear the embedding cache."""
    try:
        service = get_embedding_service()
        cleared_count = service.clear_cache()
        return {"message": f"Cleared {cleared_count} cached embeddings"}
    except Exception as e:
        logger.error(f"Error clearing embedding cache: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error clearing embedding cache: {str(e)}")

@app.get("/embed/cache/info")
async def get_embedding_cache_info():
    """Get information about the embedding cache."""
    try:
        service = get_embedding_service()
        return service.get_cache_info()
    except Exception as e:
        logger.error(f"Error getting cache info: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error getting cache info: {str(e)}")


# ===== PHASE 6: Data Processing & Analytics Endpoints =====

# Import new services and schemas
try:
    from app.services.preprocess import DataPreprocessor, PreprocessingConfig
    from app.services.export_service import ExportService
    from app.services.batch_service import BatchProcessor, BatchRequest as BatchRequestClass
    from app.schemas.analytics import (
        PreprocessingRequest, PreprocessingResponse, DatasetStatsResponse,
        ExportRequest, ExportFormat, BatchRequest, BatchResponse, BatchSummaryResponse,
        BatchListResponse
    )
    
    # Initialize services
    data_preprocessor = DataPreprocessor()
    export_service = ExportService()
    # Note: batch_processor will be initialized with clustering_service when needed
    
    print("✅ PHASE 6 services and schemas loaded successfully")
except ImportError as e:
    print(f"⚠️  PHASE 6 modules not loaded: {e}")
    data_preprocessor = None
    export_service = None


@app.get("/datasets/{dataset_id}/stats", response_model=DatasetStatsResponse)
async def get_dataset_stats(
    dataset_id: str, 
    current_user=Depends(get_current_active_user),
    db: Session = Depends(get_db) if use_new_auth else None
):
    """Get comprehensive statistics for a dataset with PHASE 8 tenant filtering."""
    if use_new_auth and db:
        # PHASE 8: Use new database session and tenant filtering
        dataset_query = db.query(Dataset).filter(Dataset.id == dataset_id)
        
        # Apply tenant filtering for non-admin users
        if current_user.role != UserRole.ADMIN:
            dataset_query = dataset_query.filter(Dataset.tenant_id == current_user.tenant_id)
        
        dataset = dataset_query.first()
    else:
        # Legacy: Use old session local
        db_legacy = SessionLocal()
        try:
            dataset = db_legacy.query(Dataset).filter(Dataset.id == dataset_id).first()
        finally:
            db_legacy.close()
    
    if not dataset:
        raise HTTPException(status_code=404, detail="Dataset not found")
    
    try:
        # Load dataset data
        data_dict = json.loads(dataset.data)
        data = np.array(data_dict['data'])
        column_names = data_dict.get('columns', [f"feature_{i}" for i in range(data.shape[1])])
        
        # Generate comprehensive statistics
        if data_preprocessor:
            stats = data_preprocessor.analyze_dataset(data, column_names)
            
            # Convert to response format
            return DatasetStatsResponse(
                shape=list(stats.shape),
                memory_usage_mb=stats.memory_usage,
                dtypes=stats.dtypes,
                missing_counts=stats.missing_counts,
                missing_percentages=stats.missing_percentages,
                total_missing=stats.total_missing,
                numerical_stats=stats.numerical_stats,
                correlations=stats.correlations,
                categorical_stats=stats.categorical_stats,
                duplicate_rows=stats.duplicate_rows,
                constant_columns=stats.constant_columns,
                high_cardinality_columns=stats.high_cardinality_columns,
                skewed_columns=stats.skewed_columns,
                outlier_counts=stats.outlier_counts,
                preprocessing_recommendations=stats.preprocessing_recommendations
            )
        else:
            raise HTTPException(status_code=500, detail="Preprocessing service not available")
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting dataset stats: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error analyzing dataset: {str(e)}")
    finally:
        if not use_new_auth and 'db_legacy' in locals():
            db_legacy.close()


@app.post("/datasets/{dataset_id}/preprocess", response_model=PreprocessingResponse)
async def preprocess_dataset(
    dataset_id: str,
    request: PreprocessingRequest,
    current_user: User = Depends(get_current_user)
):
    """Apply preprocessing pipeline to a dataset."""
    db = SessionLocal()
    try:
        start_time = datetime.now()
        
        # Get dataset
        dataset = db.query(Dataset).filter(Dataset.id == dataset_id).first()
        if not dataset:
            raise HTTPException(status_code=404, detail="Dataset not found")
        
        # Load dataset data
        data_dict = json.loads(dataset.data)
        data = np.array(data_dict['data'])
        column_names = data_dict.get('columns', [f"feature_{i}" for i in range(data.shape[1])])
        
        # Create preprocessing configuration
        config = PreprocessingConfig(
            scaler_type=request.scaler_type.value,
            missing_strategy=request.missing_strategy.value,
            missing_threshold=request.missing_threshold,
            variance_threshold=request.variance_threshold,
            outlier_method=request.outlier_method.value,
            outlier_threshold=request.outlier_threshold,
            min_samples=request.min_samples,
            max_features=request.max_features
        )
        
        # Apply preprocessing
        if data_preprocessor:
            processed_data, preprocessing_info = data_preprocessor.preprocess_data(
                data, config, column_names
            )
            
            # Calculate processing time
            end_time = datetime.now()
            processing_time = (end_time - start_time).total_seconds()
            
            # Create response
            return PreprocessingResponse(
                original_shape=preprocessing_info["original_shape"],
                final_shape=preprocessing_info["final_shape"],
                steps_applied=preprocessing_info["steps_applied"],
                removed_columns=preprocessing_info["removed_columns"],
                outliers_removed=preprocessing_info["outliers_removed"],
                processing_time_seconds=processing_time
            )
        else:
            raise HTTPException(status_code=500, detail="Preprocessing service not available")
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error preprocessing dataset: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error preprocessing dataset: {str(e)}")
    finally:
        db.close()


@app.get("/runs/{run_id}/labels.csv")
async def export_labels_csv(run_id: str, current_user: User = Depends(get_current_user)):
    """Export clustering labels as CSV."""
    db = SessionLocal()
    try:
        # Get clustering run
        run = db.query(ClusteringRun).filter(ClusteringRun.id == run_id).first()
        if not run:
            raise HTTPException(status_code=404, detail="Clustering run not found")
        
        # Get labels and original data
        result = json.loads(run.result)
        labels = np.array(result.get('labels', []))
        
        # Get original dataset
        dataset = db.query(Dataset).filter(Dataset.id == run.dataset_id).first()
        original_data = None
        column_names = None
        
        if dataset:
            data_dict = json.loads(dataset.data)
            original_data = np.array(data_dict['data'])
            column_names = data_dict.get('columns')
        
        # Generate CSV
        if export_service:
            csv_content = export_service.export_labels_csv(labels, original_data, column_names)
            
            from fastapi.responses import Response
            return Response(
                content=csv_content,
                media_type="text/csv",
                headers={"Content-Disposition": f"attachment; filename=labels_{run_id}.csv"}
            )
        else:
            raise HTTPException(status_code=500, detail="Export service not available")
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error exporting labels: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error exporting labels: {str(e)}")
    finally:
        db.close()


@app.get("/runs/{run_id}/report")
async def export_clustering_report(run_id: str, current_user: User = Depends(get_current_user)):
    """Export comprehensive clustering report as HTML."""
    db = SessionLocal()
    try:
        # Get clustering run
        run = db.query(ClusteringRun).filter(ClusteringRun.id == run_id).first()
        if not run:
            raise HTTPException(status_code=404, detail="Clustering run not found")
        
        # Get run data and results
        result = json.loads(run.result)
        labels = np.array(result.get('labels', []))
        
        run_data = {
            "id": run.id,
            "algorithm": run.algorithm,
            "parameters": json.loads(run.parameters),
            "dataset_name": f"Dataset {run.dataset_id}"
        }
        
        # Generate HTML report
        if export_service:
            html_content = export_service.generate_clustering_report(run_data, labels)
            
            from fastapi.responses import Response
            return Response(
                content=html_content,
                media_type="text/html",
                headers={"Content-Disposition": f"attachment; filename=report_{run_id}.html"}
            )
        else:
            raise HTTPException(status_code=500, detail="Export service not available")
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error generating report: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error generating report: {str(e)}")
    finally:
        db.close()


@app.get("/runs/{run_id}/bundle.zip")
async def export_bundle_zip(run_id: str, current_user: User = Depends(get_current_user)):
    """Export complete analysis bundle as ZIP."""
    db = SessionLocal()
    try:
        # Get clustering run
        run = db.query(ClusteringRun).filter(ClusteringRun.id == run_id).first()
        if not run:
            raise HTTPException(status_code=404, detail="Clustering run not found")
        
        # Get run data and results
        result = json.loads(run.result)
        labels = np.array(result.get('labels', []))
        
        run_data = {
            "id": run.id,
            "algorithm": run.algorithm,
            "parameters": json.loads(run.parameters),
            "dataset_name": f"Dataset {run.dataset_id}",
            "created_at": run.created_at.isoformat() if run.created_at else None
        }
        
        # Get original dataset
        dataset = db.query(Dataset).filter(Dataset.id == run.dataset_id).first()
        original_data = None
        column_names = None
        
        if dataset:
            data_dict = json.loads(dataset.data)
            original_data = np.array(data_dict['data'])
            column_names = data_dict.get('columns')
        
        # Generate ZIP bundle
        if export_service:
            zip_content = export_service.create_export_bundle(
                run_data, labels, original_data, None, None, column_names
            )
            
            from fastapi.responses import Response
            return Response(
                content=zip_content,
                media_type="application/zip",
                headers={"Content-Disposition": f"attachment; filename=bundle_{run_id}.zip"}
            )
        else:
            raise HTTPException(status_code=500, detail="Export service not available")
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error creating export bundle: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error creating export bundle: {str(e)}")
    finally:
        db.close()


@app.post("/batch", response_model=dict)
async def create_batch_job(request: BatchRequest, current_user: User = Depends(get_current_user)):
    """Create a new batch clustering job."""
    try:
        # Import clustering service from the existing app
        from clustering import SpectralClusteringService
        clustering_service = SpectralClusteringService()
        
        # Create batch processor
        batch_processor = BatchProcessor(clustering_service, None)  # dataset_service placeholder
        
        # Convert request to internal format
        batch_request = BatchRequestClass(
            name=request.name,
            description=request.description,
            jobs=[
                {
                    "dataset_id": job.dataset_id,
                    "algorithm": job.algorithm,
                    "parameters": job.parameters
                }
                for job in request.jobs
            ],
            max_parallel_jobs=request.max_parallel_jobs,
            stop_on_error=request.stop_on_error,
            timeout_minutes=request.timeout_minutes,
            notify_on_completion=request.notify_on_completion,
            email=request.email
        )
        
        # Create batch
        batch_id = batch_processor.create_batch(batch_request)
        
        return {
            "batch_id": batch_id,
            "message": f"Batch job '{request.name}' created with {len(request.jobs)} jobs",
            "status": "created"
        }
        
    except Exception as e:
        logger.error(f"Error creating batch job: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error creating batch job: {str(e)}")


@app.get("/batch/{batch_id}", response_model=dict)
async def get_batch_status(batch_id: str, current_user: User = Depends(get_current_user)):
    """Get status of a batch job."""
    try:
        # This would need to be stored in a global batch processor instance
        # For now, return a placeholder response
        return {
            "batch_id": batch_id,
            "status": "not_implemented",
            "message": "Batch status tracking not fully implemented yet"
        }
        
    except Exception as e:
        logger.error(f"Error getting batch status: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error getting batch status: {str(e)}")


@app.get("/batch", response_model=dict)
async def list_batch_jobs(current_user: User = Depends(get_current_user)):
    """List all batch jobs."""
    try:
        # This would need persistent batch storage
        # For now, return a placeholder response
        return {
            "batches": [],
            "total_count": 0,
            "message": "Batch listing not fully implemented yet"
        }
        
    except Exception as e:
        logger.error(f"Error listing batches: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error listing batches: {str(e)}")


# ===== PHASE 7: Health Check and Metrics Endpoints =====

@app.get("/health/live")
async def health_liveness():
    """Liveness probe - basic application health check."""
    try:
        if health_checker:
            health_status = await health_checker.check_liveness()
            return health_status.dict()
        else:
            return {
                "status": "healthy",
                "timestamp": datetime.utcnow().isoformat(),
                "message": "Basic liveness check"
            }
    except Exception as e:
        logger.error(f"Liveness check failed: {str(e)}")
        return JSONResponse(
            status_code=503,
            content={
                "status": "unhealthy",
                "timestamp": datetime.utcnow().isoformat(),
                "error": str(e)
            }
        )


@app.get("/health/ready")
async def health_readiness():
    """Readiness probe - comprehensive dependency health check."""
    try:
        if health_checker:
            health_status = await health_checker.check_readiness()
            
            # Return appropriate status code based on health
            status_code = 200
            if health_status.status == "unhealthy":
                status_code = 503
            elif health_status.status == "degraded":
                status_code = 200  # Still ready, but degraded
            
            return JSONResponse(
                status_code=status_code,
                content=health_status.dict()
            )
        else:
            return {
                "status": "healthy", 
                "timestamp": datetime.utcnow().isoformat(),
                "message": "Basic readiness check"
            }
    except Exception as e:
        logger.error(f"Readiness check failed: {str(e)}")
        return JSONResponse(
            status_code=503,
            content={
                "status": "unhealthy",
                "timestamp": datetime.utcnow().isoformat(),
                "error": str(e)
            }
        )


@app.get("/health/status")
async def health_detailed_status():
    """Detailed system status with comprehensive metrics."""
    try:
        if health_checker:
            detailed_status = await health_checker.get_detailed_status()
            return detailed_status
        else:
            return {
                "status": "healthy",
                "timestamp": datetime.utcnow().isoformat(),
                "message": "Detailed health checks not available"
            }
    except Exception as e:
        logger.error(f"Detailed status check failed: {str(e)}")
        return JSONResponse(
            status_code=500,
            content={
                "status": "error",
                "timestamp": datetime.utcnow().isoformat(),
                "error": str(e)
            }
        )


@app.get("/metrics")
async def get_prometheus_metrics():
    """Prometheus-compatible metrics endpoint."""
    try:
        if metrics_collector:
            metrics_data = metrics_collector.get_metrics()
            return Response(
                content=metrics_data,
                media_type=metrics_collector.get_content_type()
            )
        else:
            return Response(
                content="# Metrics not available\n",
                media_type="text/plain"
            )
    except Exception as e:
        logger.error(f"Metrics collection failed: {str(e)}")
        return Response(
            content="# Metrics collection error\n",
            media_type="text/plain",
            status_code=500
        )


@app.get("/metrics/simple")
async def get_simple_metrics():
    """Simple JSON metrics for basic monitoring."""
    try:
        if performance_monitor:
            metrics_data = performance_monitor.get_metrics()
            return metrics_data
        else:
            return {
                "status": "metrics_not_available",
                "timestamp": datetime.utcnow().isoformat()
            }
    except Exception as e:
        logger.error(f"Simple metrics collection failed: {str(e)}")
        return JSONResponse(
            status_code=500,
            content={
                "error": "Metrics collection failed",
                "timestamp": datetime.utcnow().isoformat()
            }
        )


# Enhanced error handlers with structured responses
@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request, exc):
    """Handle request validation errors with structured response."""
    error_details = []
    for error in exc.errors():
        error_details.append({
            "field": " -> ".join(str(x) for x in error["loc"]),
            "message": error["msg"],
            "type": error["type"]
        })
    
    if ErrorEnvelope:
        response = ErrorEnvelope.validation_error({"validation_errors": error_details})
    else:
        response = {
            "error": "Validation failed",
            "details": error_details,
            "timestamp": datetime.utcnow().isoformat()
        }
    
    logger.warning(f"Validation error: {error_details}")
    
    return JSONResponse(
        status_code=422,
        content=response
    )


@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    """Handle HTTP exceptions with structured response."""
    if ErrorEnvelope:
        if exc.status_code == 404:
            response = ErrorEnvelope.not_found_error("Resource", "unknown")
        elif exc.status_code == 401:
            response = ErrorEnvelope.authentication_error()
        elif exc.status_code == 403:
            response = ErrorEnvelope.authorization_error()
        else:
            response = ErrorEnvelope.create_error_response(
                error_code=f"HTTP_{exc.status_code}",
                message=exc.detail
            )
    else:
        response = {
            "error": exc.detail,
            "status_code": exc.status_code,
            "timestamp": datetime.utcnow().isoformat()
        }
    
    logger.warning(f"HTTP {exc.status_code}: {exc.detail}")
    
    return JSONResponse(
        status_code=exc.status_code,
        content=response
    )


@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    """Handle general exceptions with structured response."""
    if ErrorEnvelope:
        response = ErrorEnvelope.internal_error("An unexpected error occurred")
    else:
        response = {
            "error": "Internal server error",
            "timestamp": datetime.utcnow().isoformat()
        }
    
    logger.error(f"Unhandled exception: {str(exc)}", exc_info=True)
    
    return JSONResponse(
        status_code=500,
        content=response
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
