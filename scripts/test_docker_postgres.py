import pytest
#!/usr/bin/env python3
"""
Test Docker PostgreSQL Connection
Quick test to verify the Docker PostgreSQL container is working.
"""

import psycopg2
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class TestModule(unittest.TestCase):
    def test_connection():
        """Test connection to Docker PostgreSQL."""
        try:
            # Connect to Docker PostgreSQL on port 5433
            conn = psycopg2.connect(
                host="localhost",
                port="5433",
                user="postgres",
                password="postgres",
                database="pixelated_empathy"
            )
            
            cursor = conn.cursor()
            cursor.execute("SELECT version()")
            version = cursor.fetchone()[0]
            
            logger.info(f"✅ Connected to PostgreSQL: {version}")
            
            # Test creating a table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS test_table (
                    id SERIAL PRIMARY KEY,
                    message TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Insert test data
            cursor.execute("INSERT INTO test_table (message) VALUES (%s)", ("Hello from Docker PostgreSQL!",))
            
            # Query it back
            cursor.execute("SELECT * FROM test_table ORDER BY id DESC LIMIT 1")
            result = cursor.fetchone()
            
            logger.info(f"✅ Test data: ID={result[0]}, Message='{result[1]}', Created={result[2]}")
            
            conn.commit()
            cursor.close()
            conn.close()
            
            logger.info("✅ Docker PostgreSQL is working perfectly!")
            return True
            
        except Exception as e:
            logger.error(f"❌ Connection failed: {e}")
            return False
    
if __name__ == "__main__":
    test_connection()
