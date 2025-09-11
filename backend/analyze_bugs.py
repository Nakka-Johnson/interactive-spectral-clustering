"""
Comprehensive Bug Analysis Script
Identifies all errors, issues, and problems in the project
"""

import os
import sys
import traceback
import json
from pathlib import Path

def analyze_backend():
    """Analyze backend for issues"""
    print("\n" + "="*50)
    print("🔍 BACKEND ANALYSIS")
    print("="*50)
    
    issues = []
    
    # Test basic imports
    try:
        print("1. Testing basic imports...")
        import pandas as pd
        import numpy as np
        from fastapi import FastAPI
        print("✅ Basic imports successful")
    except Exception as e:
        issues.append(f"❌ Basic imports failed: {e}")
        print(f"❌ Basic imports failed: {e}")
    
    # Test model imports
    try:
        print("2. Testing model imports...")
        from models import Dataset, ClusteringRun, ExperimentSession
        print("✅ Models import successful")
    except Exception as e:
        issues.append(f"❌ Models import failed: {e}")
        print(f"❌ Models import failed: {e}")
        traceback.print_exc()
    
    # Test auth imports
    try:
        print("3. Testing auth imports...")
        from app.models.auth import User, Tenant
        print("✅ Auth models import successful")
    except Exception as e:
        issues.append(f"❌ Auth models import failed: {e}")
        print(f"❌ Auth models import failed: {e}")
        traceback.print_exc()
    
    # Test service imports
    try:
        print("4. Testing service imports...")
        from app.services.auth_service import AuthService
        print("✅ Auth service import successful")
    except Exception as e:
        issues.append(f"❌ Auth service import failed: {e}")
        print(f"❌ Auth service import failed: {e}")
        traceback.print_exc()
    
    # Test route imports
    try:
        print("5. Testing route imports...")
        from app.routes.auth import router
        print("✅ Auth routes import successful")
    except Exception as e:
        issues.append(f"❌ Auth routes import failed: {e}")
        print(f"❌ Auth routes import failed: {e}")
        traceback.print_exc()
    
    # Test database connection
    try:
        print("6. Testing database connection...")
        from app.database.connection import get_db, engine
        print("✅ Database connection import successful")
    except Exception as e:
        issues.append(f"❌ Database connection failed: {e}")
        print(f"❌ Database connection failed: {e}")
        traceback.print_exc()
    
    # Test main app import step by step
    try:
        print("7. Testing main app import...")
        
        # Check if app.py exists and is readable
        if not os.path.exists('app.py'):
            issues.append("❌ app.py file not found")
            print("❌ app.py file not found")
        else:
            print("✅ app.py file exists")
            
            # Try to exec the file step by step
            with open('app.py', 'r') as f:
                content = f.read()
                
            # Check for syntax errors
            try:
                compile(content, 'app.py', 'exec')
                print("✅ app.py syntax is valid")
            except SyntaxError as e:
                issues.append(f"❌ Syntax error in app.py: {e}")
                print(f"❌ Syntax error in app.py: {e}")
                
            # Try importing
            try:
                import app
                if hasattr(app, 'app'):
                    print("✅ app.app variable exists")
                else:
                    issues.append("❌ app.app variable not found after import")
                    print("❌ app.app variable not found after import")
                    print(f"Available attributes: {[x for x in dir(app) if not x.startswith('_')]}")
            except Exception as e:
                issues.append(f"❌ app.py import failed: {e}")
                print(f"❌ app.py import failed: {e}")
                traceback.print_exc()
                
    except Exception as e:
        issues.append(f"❌ Main app analysis failed: {e}")
        print(f"❌ Main app analysis failed: {e}")
        traceback.print_exc()
    
    return issues

def analyze_database():
    """Analyze database for issues"""
    print("\n" + "="*50)
    print("🗄️ DATABASE ANALYSIS")
    print("="*50)
    
    issues = []
    
    try:
        # Check if database file exists
        db_path = "clustering.db"
        if os.path.exists(db_path):
            print(f"✅ Database file exists: {db_path}")
            print(f"   Size: {os.path.getsize(db_path)} bytes")
        else:
            issues.append(f"⚠️ Database file not found: {db_path}")
            print(f"⚠️ Database file not found: {db_path}")
        
        # Try to connect to database
        from sqlalchemy import create_engine, text
        engine = create_engine("sqlite:///./clustering.db")
        
        with engine.connect() as conn:
            # Check if auth tables exist
            tables = ['tenants', 'users', 'datasets', 'clustering_runs', 'experiment_sessions']
            for table in tables:
                try:
                    result = conn.execute(text(f"SELECT count(*) FROM {table}"))
                    count = result.fetchone()[0]
                    print(f"✅ Table '{table}' exists with {count} records")
                except Exception as e:
                    issues.append(f"❌ Table '{table}' issue: {e}")
                    print(f"❌ Table '{table}' issue: {e}")
        
    except Exception as e:
        issues.append(f"❌ Database analysis failed: {e}")
        print(f"❌ Database analysis failed: {e}")
        traceback.print_exc()
    
    return issues

def analyze_dependencies():
    """Check for missing dependencies"""
    print("\n" + "="*50)
    print("📦 DEPENDENCY ANALYSIS")
    print("="*50)
    
    issues = []
    
    required_packages = [
        'fastapi', 'uvicorn', 'pandas', 'numpy', 'scikit-learn',
        'sqlalchemy', 'passlib', 'python-jose', 'bcrypt',
        'pydantic', 'loguru', 'sentry-sdk'
    ]
    
    for package in required_packages:
        try:
            __import__(package.replace('-', '_'))
            print(f"✅ {package}")
        except ImportError:
            issues.append(f"❌ Missing package: {package}")
            print(f"❌ Missing package: {package}")
    
    return issues

def analyze_file_structure():
    """Check file structure for issues"""
    print("\n" + "="*50)
    print("📁 FILE STRUCTURE ANALYSIS")
    print("="*50)
    
    issues = []
    
    required_files = [
        'app.py',
        'models.py',
        'clustering.py',
        'evaluation.py',
        'app/models/auth.py',
        'app/services/auth_service.py',
        'app/routes/auth.py',
        'app/database/connection.py'
    ]
    
    for file_path in required_files:
        if os.path.exists(file_path):
            print(f"✅ {file_path}")
        else:
            issues.append(f"❌ Missing file: {file_path}")
            print(f"❌ Missing file: {file_path}")
    
    # Check for __init__.py files
    init_files = [
        'app/__init__.py',
        'app/models/__init__.py',
        'app/services/__init__.py',
        'app/routes/__init__.py',
        'app/database/__init__.py'
    ]
    
    for init_file in init_files:
        if os.path.exists(init_file):
            print(f"✅ {init_file}")
        else:
            issues.append(f"⚠️ Missing __init__.py: {init_file}")
            print(f"⚠️ Missing __init__.py: {init_file}")
    
    return issues

def main():
    """Run comprehensive analysis"""
    print("🔍 COMPREHENSIVE PROJECT ANALYSIS")
    print("="*60)
    
    all_issues = []
    
    # Change to backend directory
    if os.path.exists('backend'):
        os.chdir('backend')
        print("📂 Changed to backend directory")
    
    # Run all analyses
    all_issues.extend(analyze_file_structure())
    all_issues.extend(analyze_dependencies())
    all_issues.extend(analyze_database())
    all_issues.extend(analyze_backend())
    
    # Summary
    print("\n" + "="*60)
    print("📋 ISSUE SUMMARY")
    print("="*60)
    
    if all_issues:
        print(f"Found {len(all_issues)} issues:")
        for i, issue in enumerate(all_issues, 1):
            print(f"{i}. {issue}")
    else:
        print("✅ No critical issues found!")
    
    return all_issues

if __name__ == "__main__":
    issues = main()
    sys.exit(1 if issues else 0)
