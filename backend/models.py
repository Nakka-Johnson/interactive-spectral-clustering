from sqlalchemy import Column, String, Text, DateTime, Float, Integer, Boolean, JSON, ForeignKey
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship, Session
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.sql import func
from datetime import datetime
import uuid
import json
from typing import Dict, Any, List, Optional

# Import auth models for relationships
from app.models.auth import Base as AuthBase, Tenant, User

# Use the same Base from auth models to ensure consistency
Base = AuthBase

class Dataset(Base):
    """
    Refactored Dataset model - simplified to only essential fields
    PHASE 8: Added tenant_id and user_id for multi-tenancy
    """
    __tablename__ = "datasets"
    
    # Primary key - using job_id as primary key for simplicity
    job_id = Column(String(255), primary_key=True, index=True, nullable=False)
    
    # PHASE 8: Multi-tenancy fields
    tenant_id = Column(Integer, ForeignKey('tenants.id'), nullable=False, index=True)
    user_id = Column(Integer, ForeignKey('users.id'), nullable=False, index=True)
    
    # Upload metadata
    upload_time = Column(DateTime, default=datetime.utcnow, nullable=False)
    
    # File information - storing path instead of content for efficiency
    file_path = Column(String(500), nullable=False)  # Path to stored file
    
    # Dataset schema - stored as JSON for flexibility
    columns = Column(JSON, nullable=False)  # All column information as JSON
    
    # PHASE 8: Relationships
    tenant = relationship("Tenant", back_populates="datasets")
    user = relationship("User", back_populates="datasets")
    runs = relationship("ClusteringRun", back_populates="dataset", cascade="all, delete-orphan")
    
    def __repr__(self):
        return f"<Dataset(job_id='{self.job_id}', tenant_id={self.tenant_id}, user_id={self.user_id})>"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert dataset to dictionary"""
        return {
            'job_id': self.job_id,
            'tenant_id': self.tenant_id,
            'user_id': self.user_id,
            'upload_time': self.upload_time.isoformat() if self.upload_time else None,
            'file_path': self.file_path,
            'columns': self.columns
        }

class ClusteringRun(Base):
    """
    Refactored ClusteringRun model - simplified with expiration support
    PHASE 8: Added user_id for tracking who initiated the run
    """
    __tablename__ = "clustering_runs"
    
    # Primary key
    run_id = Column(String(255), primary_key=True, index=True, nullable=False)
    
    # Foreign key to dataset
    job_id = Column(String(255), ForeignKey('datasets.job_id'), nullable=False, index=True)
    
    # PHASE 8: Track which user initiated this run
    user_id = Column(Integer, ForeignKey('users.id'), nullable=False, index=True)
    
    # Run configuration and results - stored as JSON for flexibility
    methods = Column(JSON, nullable=False)  # List of clustering methods used
    params = Column(JSON, nullable=False)   # All parameters as JSON
    metrics = Column(JSON, nullable=False)  # All evaluation metrics as JSON
    
    # Run status tracking
    status = Column(String(50), default="queued", nullable=False)  # queued, running, done, error
    started_at = Column(DateTime, nullable=True)
    completed_at = Column(DateTime, nullable=True)
    error_message = Column(Text, nullable=True)
    algorithm = Column(String(50), nullable=True)  # Primary algorithm used
    
    # Timestamps with expiration support
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    expires_at = Column(DateTime, nullable=False)  # Automatic expiration for cleanup
    
    # Relationships
    dataset = relationship("Dataset", back_populates="runs")
    user = relationship("User")  # Reference to user who initiated the run
    
    # Property to get tenant_id through dataset relationship
    @property
    def tenant_id(self) -> Optional[int]:
        """Get tenant_id through dataset relationship"""
        return self.dataset.tenant_id if self.dataset else None
    
    def __repr__(self):
        return f"<ClusteringRun(run_id='{self.run_id}', job_id='{self.job_id}', user_id={self.user_id})>"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert clustering run to dictionary"""
        return {
            'run_id': self.run_id,
            'job_id': self.job_id,
            'user_id': self.user_id,
            'tenant_id': self.tenant_id,
            'methods': self.methods,
            'params': self.params,
            'metrics': self.metrics,
            'status': self.status,
            'algorithm': self.algorithm,
            'started_at': self.started_at.isoformat() if self.started_at else None,
            'completed_at': self.completed_at.isoformat() if self.completed_at else None,
            'error_message': self.error_message,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'expires_at': self.expires_at.isoformat() if self.expires_at else None
        }

class ExperimentSession(Base):
    """
    ExperimentSession model for grouping related clustering runs
    PHASE 8: Added tenant_id and user_id for multi-tenancy
    """
    __tablename__ = "experiment_sessions"
    
    # Primary key
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    
    # PHASE 8: Multi-tenancy fields
    tenant_id = Column(Integer, ForeignKey('tenants.id'), nullable=False, index=True)
    user_id = Column(Integer, ForeignKey('users.id'), nullable=False, index=True)
    
    # Session metadata
    session_id = Column(String(255), unique=True, index=True, nullable=False)
    session_name = Column(String(255), nullable=False)
    description = Column(Text)
    
    # Associated datasets and runs
    job_ids = Column(JSON)  # List of job_ids used in this session
    run_ids = Column(JSON)  # List of run_ids in this session
    
    # Session configuration
    default_methods = Column(JSON)  # Default clustering methods for this session
    default_params = Column(JSON)  # Default parameters
    
    # Timestamps
    created_at = Column(DateTime, server_default=func.now())
    updated_at = Column(DateTime, server_default=func.now(), onupdate=func.now())
    last_accessed = Column(DateTime, default=datetime.utcnow)
    
    # PHASE 8: Relationships
    tenant = relationship("Tenant")
    user = relationship("User")
    
    def __repr__(self):
        return f"<ExperimentSession(session_id='{self.session_id}', tenant_id={self.tenant_id}, user_id={self.user_id})>"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert experiment session to dictionary"""
        return {
            'id': str(self.id),
            'tenant_id': self.tenant_id,
            'user_id': self.user_id,
            'session_id': self.session_id,
            'session_name': self.session_name,
            'description': self.description,
            'job_ids': self.job_ids,
            'run_ids': self.run_ids,
            'default_methods': self.default_methods,
            'default_params': self.default_params,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'updated_at': self.updated_at.isoformat() if self.updated_at else None,
            'last_accessed': self.last_accessed.isoformat() if self.last_accessed else None
        }

class SystemMetrics(Base):
    """
    SystemMetrics model for tracking system performance and usage
    """
    __tablename__ = "system_metrics"
    
    # Primary key
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    
    # Metric identification
    metric_type = Column(String(100), nullable=False)  # cpu, memory, gpu, disk, etc.
    component = Column(String(100))  # specific component being measured
    
    # Metric values
    value = Column(Float, nullable=False)
    unit = Column(String(50))  # MB, %, seconds, etc.
    
    # Context
    job_id = Column(String(255), index=True)
    run_id = Column(String(255), index=True)
    additional_context = Column(JSON)  # Extra metadata
    
    # Timestamp
    recorded_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    
    def __repr__(self):
        return f"<SystemMetrics(type='{self.metric_type}', value={self.value}, unit='{self.unit}')>"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert system metrics to dictionary"""
        return {
            'id': str(self.id),
            'metric_type': self.metric_type,
            'component': self.component,
            'value': self.value,
            'unit': self.unit,
            'job_id': self.job_id,
            'run_id': self.run_id,
            'additional_context': self.additional_context,
            'recorded_at': self.recorded_at.isoformat() if self.recorded_at else None
        }

# Database helper functions
class DatabaseManager:
    """Helper class for database operations"""
    
    def __init__(self, session: Session):
        self.session = session
    
    def create_dataset(self, job_id: str, file_path: str, columns: Dict[str, Any], **kwargs) -> Dataset:
        """Create a new dataset record"""
        dataset = Dataset(
            job_id=job_id,
            file_path=file_path,
            columns=columns,
            upload_time=kwargs.get('upload_time', datetime.utcnow())
        )
        self.session.add(dataset)
        self.session.commit()
        self.session.refresh(dataset)
        return dataset
    
    def get_dataset_by_job_id(self, job_id: str) -> Optional[Dataset]:
        """Get dataset by job_id"""
        return self.session.query(Dataset).filter(Dataset.job_id == job_id).first()
    
    def create_clustering_run(self, job_id: str, methods: List[str], params: Dict[str, Any], 
                            metrics: Dict[str, Any] = None, expires_in_hours: int = 24, **kwargs) -> ClusteringRun:
        """Create a new clustering run record with automatic expiration"""
        from datetime import timedelta
        
        run_id = kwargs.get('run_id', str(uuid.uuid4()))
        expires_at = datetime.utcnow() + timedelta(hours=expires_in_hours)
        
        run = ClusteringRun(
            run_id=run_id,
            job_id=job_id,
            methods=methods,
            params=params,
            metrics=metrics or {},
            status=kwargs.get('status', 'queued'),
            algorithm=kwargs.get('algorithm', methods[0] if methods else None),
            started_at=kwargs.get('started_at'),
            completed_at=kwargs.get('completed_at'),
            error_message=kwargs.get('error_message'),
            expires_at=expires_at,
            created_at=kwargs.get('created_at', datetime.utcnow())
        )
        self.session.add(run)
        self.session.commit()
        self.session.refresh(run)
        return run
    
    def get_clustering_run(self, run_id: str) -> Optional[ClusteringRun]:
        """Get clustering run by run_id"""
        return self.session.query(ClusteringRun).filter(ClusteringRun.run_id == run_id).first()
    
    def update_run_results(self, run_id: str, results: dict, status: str = "done", completed_at: datetime = None):
        """Update clustering run with results"""
        run = self.get_clustering_run(run_id)
        if run:
            run.metrics = results
            run.status = status
            if completed_at:
                run.completed_at = completed_at
            self.session.commit()
            self.session.refresh(run)
        return run
    
    def get_runs_by_job_id(self, job_id: str) -> List[ClusteringRun]:
        """Get all non-expired clustering runs for a dataset"""
        current_time = datetime.utcnow()
        return (self.session.query(ClusteringRun)
                .filter(ClusteringRun.job_id == job_id)
                .filter(ClusteringRun.expires_at > current_time)
                .all())
    
    def update_run_status(self, run_id: str, **kwargs) -> Optional[ClusteringRun]:
        """Update clustering run data"""
        run = self.session.query(ClusteringRun).filter(ClusteringRun.run_id == run_id).first()
        if run:
            # Update any provided fields
            for key, value in kwargs.items():
                if hasattr(run, key):
                    setattr(run, key, value)
            self.session.commit()
            self.session.refresh(run)
        return run
    
    def get_recent_runs(self, limit: int = 10) -> List[ClusteringRun]:
        """Get most recent non-expired clustering runs"""
        current_time = datetime.utcnow()
        return (self.session.query(ClusteringRun)
                .filter(ClusteringRun.expires_at > current_time)
                .order_by(ClusteringRun.created_at.desc())
                .limit(limit)
                .all())
    
    def cleanup_expired_runs(self) -> int:
        """Clean up expired clustering runs and return count of deleted records"""
        current_time = datetime.utcnow()
        expired_runs = self.session.query(ClusteringRun).filter(
            ClusteringRun.expires_at <= current_time
        )
        count = expired_runs.count()
        expired_runs.delete()
        self.session.commit()
        return count
    
    def cleanup_old_records(self, days_old: int = 30):
        """Clean up old records to manage database size"""
        from datetime import timedelta
        cutoff_date = datetime.utcnow() - timedelta(days=days_old)
        
        # Delete old system metrics
        old_metrics = self.session.query(SystemMetrics).filter(
            SystemMetrics.recorded_at < cutoff_date
        )
        old_metrics.delete()
        
        # The expired runs are already handled by cleanup_expired_runs()
        
        self.session.commit()

# Utility functions for data validation
def validate_json_field(data: Any) -> bool:
    """Validate that data can be serialized to JSON"""
    try:
        json.dumps(data)
        return True
    except (TypeError, ValueError):
        return False

def generate_data_hash(data: str) -> str:
    """Generate SHA256 hash for data integrity checking"""
    import hashlib
    return hashlib.sha256(data.encode('utf-8')).hexdigest()

# Database initialization
def init_database(engine):
    """Initialize database tables"""
    Base.metadata.create_all(bind=engine)

def drop_database(engine):
    """Drop all database tables (use with caution)"""
    Base.metadata.drop_all(bind=engine)
