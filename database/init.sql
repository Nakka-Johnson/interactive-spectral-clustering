-- Initialize clustering database
-- This script runs when the PostgreSQL container starts

-- Create extensions
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "pg_trgm";

-- Create indexes for performance
-- These will be created automatically by SQLAlchemy, but we can optimize here

-- Set default configuration
ALTER DATABASE clustering_db SET timezone TO 'UTC';
