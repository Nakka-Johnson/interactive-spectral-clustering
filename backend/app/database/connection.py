"""
Database connection and session management for PHASE 8
Provides database session dependency injection for FastAPI routes
"""

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session
from typing import Generator
import os

# Database configuration
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./clustering.db")

# Create database engine
engine = create_engine(
    DATABASE_URL,
    # SQLite specific configuration
    connect_args={"check_same_thread": False} if DATABASE_URL.startswith("sqlite") else {}
)

# Create sessionmaker
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

def get_db() -> Generator[Session, None, None]:
    """
    Database session dependency
    Creates a new database session for each request and closes it when done
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

def init_db_connection():
    """
    Initialize database connection and create tables if needed
    Called during application startup
    """
    from app.models.auth import Base
    from models import Base as ModelsBase
    
    # Create all tables
    Base.metadata.create_all(bind=engine)
    ModelsBase.metadata.create_all(bind=engine)
    
    return engine
