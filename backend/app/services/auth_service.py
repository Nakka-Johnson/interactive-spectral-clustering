"""
JWT Authentication Service for PHASE 8
Handles JWT token creation, validation, and user authentication.
"""

import os
from datetime import datetime, timedelta, timezone
from typing import Optional, Dict, Any
import jwt
from fastapi import HTTPException, status, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from sqlalchemy.orm import Session
from loguru import logger

from app.models.auth import User, UserRole, TokenData, verify_password, create_user_dict
from app.database.connection import get_db

# JWT Configuration
SECRET_KEY = os.getenv("JWT_SECRET_KEY", "your-secret-key-change-in-production")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = int(os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES", "30"))

# Security scheme
security = HTTPBearer()

class AuthenticationError(Exception):
    """Custom authentication error"""
    pass

class AuthorizationError(Exception):
    """Custom authorization error"""
    pass

class AuthService:
    """Authentication service for JWT token management"""
    
    @staticmethod
    def create_access_token(data: dict, expires_delta: Optional[timedelta] = None) -> str:
        """Create a new JWT access token"""
        try:
            to_encode = data.copy()
            
            if expires_delta:
                expire = datetime.now(timezone.utc) + expires_delta
            else:
                expire = datetime.now(timezone.utc) + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
            
            to_encode.update({"exp": expire, "iat": datetime.now(timezone.utc)})
            
            encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
            
            logger.info(f"JWT token created for user: {data.get('email', 'unknown')}")
            return encoded_jwt
            
        except Exception as e:
            logger.error(f"Error creating JWT token: {e}")
            raise AuthenticationError("Failed to create authentication token")
    
    @staticmethod
    def verify_token(token: str) -> TokenData:
        """Verify and decode a JWT token"""
        try:
            payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
            
            user_id: int = payload.get("user_id")
            email: str = payload.get("email")
            username: str = payload.get("username")
            role: str = payload.get("role")
            tenant_id: int = payload.get("tenant_id")
            
            if user_id is None or email is None:
                raise AuthenticationError("Invalid token payload")
            
            token_data = TokenData(
                user_id=user_id,
                email=email,
                username=username,
                role=UserRole(role) if role else None,
                tenant_id=tenant_id
            )
            
            return token_data
            
        except jwt.ExpiredSignatureError:
            logger.warning("JWT token has expired")
            raise AuthenticationError("Token has expired")
        except jwt.JWTError as e:
            logger.warning(f"JWT validation error: {e}")
            raise AuthenticationError("Invalid token")
    
    @staticmethod
    def authenticate_user(db: Session, email: str, password: str) -> Optional[User]:
        """Authenticate a user with email and password"""
        try:
            user = db.query(User).filter(
                User.email == email,
                User.is_active == True
            ).first()
            
            if not user:
                logger.warning(f"Authentication failed: user not found - {email}")
                return None
            
            if not verify_password(password, user.hashed_password):
                logger.warning(f"Authentication failed: invalid password - {email}")
                return None
            
            # Update last login
            user.last_login = datetime.now(timezone.utc)
            db.commit()
            
            logger.info(f"User authenticated successfully: {email}")
            return user
            
        except Exception as e:
            logger.error(f"Error during user authentication: {e}")
            db.rollback()
            return None

# Dependency functions for FastAPI
async def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(security),
    db: Session = Depends(get_db)
) -> User:
    """
    FastAPI dependency to get the current authenticated user
    """
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    
    try:
        token = credentials.credentials
        token_data = AuthService.verify_token(token)
        
        if token_data.user_id is None:
            raise credentials_exception
            
    except AuthenticationError:
        raise credentials_exception
    
    user = db.query(User).filter(User.id == token_data.user_id).first()
    if user is None:
        raise credentials_exception
        
    return user

async def get_current_active_user(current_user: User = Depends(get_current_user)) -> User:
    """
    FastAPI dependency to get the current active user
    """
    if not current_user.is_active:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Inactive user"
        )
    return current_user

def require_role(required_role: UserRole):
    """
    Decorator factory for role-based access control
    Returns a dependency that checks if user has required role
    """
    def role_checker(current_user: User = Depends(get_current_active_user)) -> User:
        if current_user.role != required_role and current_user.role != UserRole.ADMIN:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Insufficient permissions"
            )
        return current_user
    
    return role_checker

def require_admin(current_user: User = Depends(get_current_active_user)) -> User:
    """
    FastAPI dependency that requires admin role
    """
    if current_user.role != UserRole.ADMIN:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Admin access required"
        )
    return current_user

def require_same_tenant_or_admin(target_tenant_id: int):
    """
    Dependency factory for tenant-based access control
    Allows access if user is admin or belongs to the same tenant
    """
    def tenant_checker(current_user: User = Depends(get_current_active_user)) -> User:
        if (current_user.role != UserRole.ADMIN and 
            current_user.tenant_id != target_tenant_id):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Access denied: different tenant"
            )
        return current_user
    
    return tenant_checker

class TenantFilter:
    """
    Utility class for applying tenant-based filtering to database queries
    """
    
    @staticmethod
    def apply_tenant_filter(query, model_class, current_user: User):
        """
        Apply tenant filtering to a SQLAlchemy query
        Admins see all data, regular users see only their tenant's data
        """
        if current_user.role == UserRole.ADMIN:
            return query  # Admins see everything
        
        # Regular users see only their tenant's data
        if hasattr(model_class, 'tenant_id'):
            return query.filter(model_class.tenant_id == current_user.tenant_id)
        elif hasattr(model_class, 'user_id'):
            # If no direct tenant_id, filter by user_id for user-owned resources
            return query.filter(model_class.user_id == current_user.id)
        
        return query

# Utility functions for testing and development
def create_test_token(user_data: Dict[str, Any]) -> str:
    """Create a test token for development/testing purposes"""
    return AuthService.create_access_token(user_data)

def decode_token_for_debug(token: str) -> Dict[str, Any]:
    """Decode a token for debugging purposes (without verification)"""
    try:
        return jwt.decode(token, options={"verify_signature": False})
    except Exception as e:
        logger.error(f"Error decoding token for debug: {e}")
        return {}
