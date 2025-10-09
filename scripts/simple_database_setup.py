#!/usr/bin/env python3
"""
Simple Database Setup
Lightweight database setup without heavy migration.
"""

import os
import sys
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def check_postgresql():
    """Check if PostgreSQL is available."""
    try:
        import psycopg2
        logger.info("✅ psycopg2 available")
        return True
    except ImportError:
        logger.error("❌ psycopg2 not installed. Run: pip install psycopg2-binary")
        return False

def check_database_connection():
    """Check database connection with multiple auth methods."""
    import psycopg2

    # Try different connection configurations
    connection_configs = [
        # Try with current user (peer authentication)
        {"host": "localhost", "port": "5432", "user": os.getenv("USER", "vivi"), "database": "postgres"},
        # Try with postgres user and no password (peer auth)
        {"host": "localhost", "port": "5432", "user": "postgres", "database": "postgres"},
        # Try with postgres user and password
        {"host": "localhost", "port": "5432", "user": "postgres", "password": "postgres", "database": "postgres"},
        # Try with different port in case Docker is running
        {"host": "localhost", "port": "5433", "user": "postgres", "password": "postgres", "database": "postgres"},
    ]

    for i, config in enumerate(connection_configs):
        try:
            logger.info(f"Trying connection method {i+1}: user={config.get('user')}, port={config.get('port')}")
            conn = psycopg2.connect(**config)
            conn.close()
            logger.info(f"✅ PostgreSQL connection successful with config {i+1}")

            # Store successful config for later use
            global db_config
            db_config = config
            return True

        except Exception as e:
            logger.warning(f"Connection method {i+1} failed: {e}")
            continue

    logger.error("❌ All PostgreSQL connection methods failed")
    logger.info("💡 Try these solutions:")
    logger.info("   1. Connect as your user: psql postgres")
    logger.info("   2. Set up Docker PostgreSQL: sudo docker run -d -p 5433:5432 -e POSTGRES_PASSWORD=postgres postgres:15")
    logger.info("   3. Check PostgreSQL authentication: sudo nano /etc/postgresql/15/main/pg_hba.conf")
    return False

def create_database():
    """Create the pixelated_empathy database."""
    try:
        import psycopg2
        from psycopg2.extensions import ISOLATION_LEVEL_AUTOCOMMIT
        
        # Connect to postgres database
        conn = psycopg2.connect(
            host="localhost",
            port="5432",
            user="postgres", 
            password="postgres",
            database="postgres"
        )
        conn.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)
        cursor = conn.cursor()
        
        # Check if database exists
        cursor.execute("SELECT 1 FROM pg_database WHERE datname = 'pixelated_empathy'")
        exists = cursor.fetchone()
        
        if not exists:
            cursor.execute("CREATE DATABASE pixelated_empathy")
            logger.info("✅ Database 'pixelated_empathy' created")
        else:
            logger.info("✅ Database 'pixelated_empathy' already exists")
            
        cursor.close()
        conn.close()
        return True
        
    except Exception as e:
        logger.error(f"❌ Database creation failed: {e}")
        return False

def test_database_schema():
    """Test if we can create a simple table."""
    try:
        import psycopg2
        
        conn = psycopg2.connect(
            host="localhost",
            port="5432",
            user="postgres",
            password="postgres", 
            database="pixelated_empathy"
        )
        cursor = conn.cursor()
        
        # Create a simple test table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS test_connection (
                id SERIAL PRIMARY KEY,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Insert a test record
        cursor.execute("INSERT INTO test_connection DEFAULT VALUES")
        
        # Query it back
        cursor.execute("SELECT COUNT(*) FROM test_connection")
        count = cursor.fetchone()[0]
        
        conn.commit()
        cursor.close()
        conn.close()
        
        logger.info(f"✅ Database schema test successful ({count} test records)")
        return True
        
    except Exception as e:
        logger.error(f"❌ Database schema test failed: {e}")
        return False

def main():
    """Main database setup."""
    logger.info("🗄️ SIMPLE DATABASE SETUP")
    
    # Step 1: Check dependencies
    if not check_postgresql():
        return False
    
    # Step 2: Check connection
    if not check_database_connection():
        return False
    
    # Step 3: Create database
    if not create_database():
        return False
    
    # Step 4: Test schema
    if not test_database_schema():
        return False
    
    logger.info("✅ DATABASE SETUP COMPLETED")
    logger.info("Next steps:")
    logger.info("  1. Wait for backup to complete")
    logger.info("  2. Run migration: uv run dataset_pipeline/migrate_conversations_to_db.py")
    logger.info("  3. Verify data: psql -h localhost -U postgres -d pixelated_empathy")
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
