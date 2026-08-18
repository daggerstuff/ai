#!/usr/bin/env python3
"""
Production Database Setup and Migration Orchestrator
Sets up PostgreSQL database and migrates conversation data safely.
"""

import logging
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import psycopg2
from psycopg2 import sql
from psycopg2.extensions import ISOLATION_LEVEL_AUTOCOMMIT

# Add dataset_pipeline to path
sys.path.append(str(Path(__file__).parent.parent / "dataset_pipeline"))


from migrate_conversations_to_db import ConversationDataMigrator

_MIN_CONVERSATIONS = 1000

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("database_setup.log"), logging.StreamHandler()],
)
logger = logging.getLogger(__name__)


class ProductionDatabaseSetup:
    """Production database setup and migration orchestrator."""

    def __init__(self):
        self.db_config = {
            "host": os.getenv("DB_HOST", "localhost"),
            "port": os.getenv("DB_PORT", "5432"),
            "user": os.getenv("DB_USER", "postgres"),
            "password": os.getenv("DB_PASSWORD", "postgres"),
            "database": os.getenv("DB_NAME", "pixelated_empathy"),
            "admin_database": os.getenv("DB_ADMIN_DB", "postgres"),
        }

        self.database_url = f"postgresql://{self.db_config['user']}:{self.db_config['password']}@{self.db_config['host']}:{self.db_config['port']}/{self.db_config['database']}"
        self.admin_url = f"postgresql://{self.db_config['user']}:{self.db_config['password']}@{self.db_config['host']}:{self.db_config['port']}/{self.db_config['admin_database']}"

    def check_postgresql_running(self) -> bool:
        """Check if PostgreSQL is running and accessible."""
        try:
            conn = psycopg2.connect(self.admin_url)
            conn.close()
            logger.info("✅ PostgreSQL is running and accessible")
            return True
        except Exception as e:
            logger.error(f"❌ PostgreSQL connection failed: {e}")
            return False

    def create_database(self) -> bool:
        """Create the main database if it doesn't exist."""
        try:
            # Connect to admin database
            conn = psycopg2.connect(self.admin_url)
            conn.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)
            cursor = conn.cursor()

            # Check if database exists
            cursor.execute(
                "SELECT 1 FROM pg_database WHERE datname = %s",
                (self.db_config["database"],),
            )
            exists = cursor.fetchone()

            if not exists:
                logger.info(f"Creating database: {self.db_config['database']}")
                cursor.execute(sql.SQL("CREATE DATABASE {}").format(sql.Identifier(self.db_config["database"])))
                logger.info("✅ Database created successfully")
            else:
                logger.info("✅ Database already exists")

            cursor.close()
            conn.close()
            return True

        except Exception as e:
            logger.error(f"❌ Database creation failed: {e}")
            return False

    def setup_database_extensions(self) -> bool:
        """Set up required PostgreSQL extensions."""
        try:
            conn = psycopg2.connect(self.database_url)
            conn.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)
            cursor = conn.cursor()

            # Enable required extensions
            queries = [
                ("uuid-ossp", 'CREATE EXTENSION IF NOT EXISTS "uuid-ossp"'),
                ("pg_trgm", 'CREATE EXTENSION IF NOT EXISTS "pg_trgm"'),
                ("btree_gin", 'CREATE EXTENSION IF NOT EXISTS "btree_gin"'),
            ]

            for ext, query in queries:
                try:
                    cursor.execute(query)
                    logger.info(f"✅ Extension enabled: {ext}")
                except Exception as e:
                    logger.warning(f"⚠️ Could not enable extension {ext}: {e}")

            cursor.close()
            conn.close()
            return True

        except Exception as e:
            logger.error(f"❌ Extension setup failed: {e}")
            return False

    def create_database_schema(self) -> bool:
        """Create database tables using the migration system."""
        try:
            logger.info("Creating database schema...")
            migrator = ConversationDataMigrator(self.database_url)

            # The migrator will create tables automatically
            stats = migrator.get_migration_stats()
            logger.info(f"✅ Database schema ready. Current stats: {stats}")

            migrator.close()
            return True

        except Exception as e:
            logger.error(f"❌ Schema creation failed: {e}")
            return False

    def run_data_migration(self) -> bool:
        """Run the conversation data migration."""
        try:
            logger.info("Starting conversation data migration...")

            # Run the migration script
            migration_script = Path(__file__).parent.parent / "dataset_pipeline" / "migrate_conversations_to_db.py"

            result = subprocess.run(
                [sys.executable, str(migration_script)], capture_output=True, text=True, check=False
            )

            if result.returncode == 0:
                logger.info("✅ Data migration completed successfully")
                logger.info(f"Migration output: {result.stdout}")
                return True
            logger.error(f"❌ Data migration failed: {result.stderr}")
            return False

        except Exception as e:
            logger.error(f"❌ Data migration error: {e}")
            return False

    def verify_migration(self) -> dict[str, Any]:
        """Verify the migration was successful."""
        try:
            migrator = ConversationDataMigrator(self.database_url)
            stats = migrator.get_migration_stats()
            migrator.close()

            logger.info(f"Migration verification stats: {stats}")

            # Check if we have reasonable data
            if stats.get("conversations", 0) > _MIN_CONVERSATIONS:
                logger.info("✅ Migration verification successful")
                return {"success": True, "stats": stats}
            logger.warning("⚠️ Migration verification: Low conversation count")
            return {
                "success": False,
                "stats": stats,
                "reason": "Low conversation count",
            }

        except Exception as e:
            logger.error(f"❌ Migration verification failed: {e}")
            return {"success": False, "error": str(e)}

    def setup_database_monitoring(self) -> bool:
        """Set up basic database monitoring."""
        try:
            conn = psycopg2.connect(self.database_url)
            cursor = conn.cursor()

            # Create monitoring views
            monitoring_sql = """
            -- Create monitoring view for conversation statistics
            CREATE OR REPLACE VIEW conversation_stats AS
            SELECT
                COUNT(*) as total_conversations,
                COUNT(DISTINCT source) as unique_sources,
                MIN(started_at) as earliest_conversation,
                MAX(started_at) as latest_conversation,
                AVG((SELECT COUNT(*) FROM messages
                     WHERE conversation_id = conversations.id)) as avg_messages_per_conversation
            FROM conversations;

            -- Create monitoring view for data quality
            CREATE OR REPLACE VIEW data_quality_stats AS
            SELECT
                source,
                COUNT(*) as conversation_count,
                AVG(CASE WHEN tier = 'TIER_1' THEN 1.0 ELSE 0.0 END) as tier_1_percentage,
                COUNT(DISTINCT category) as unique_categories
            FROM conversations
            GROUP BY source;
            """

            cursor.execute(monitoring_sql)
            conn.commit()

            cursor.close()
            conn.close()

            logger.info("✅ Database monitoring views created")
            return True

        except Exception as e:
            logger.error(f"❌ Database monitoring setup failed: {e}")
            return False


def main():
    """Main setup orchestrator."""
    logger.info("🚀 STARTING PRODUCTION DATABASE SETUP")

    setup = ProductionDatabaseSetup()
    success = True

    # Step 1: Check PostgreSQL
    if success and not setup.check_postgresql_running():
        logger.error("❌ PostgreSQL is not running. Please start PostgreSQL first.")
        success = False

    # Step 2: Create database
    if success and not setup.create_database():
        logger.error("❌ Database creation failed")
        success = False

    # Step 3: Setup extensions
    if success and not setup.setup_database_extensions():
        logger.error("❌ Extension setup failed")
        success = False

    # Step 4: Create schema
    if success and not setup.create_database_schema():
        logger.error("❌ Schema creation failed")
        success = False

    # Step 5: Run migration
    if success and not setup.run_data_migration():
        logger.error("❌ Data migration failed")
        success = False

    # Step 6: Verify migration
    verification = None
    if success:
        verification = setup.verify_migration()
        if not verification["success"]:
            logger.error(f"❌ Migration verification failed: {verification}")
            success = False

    # Step 7: Setup monitoring
    if success and not setup.setup_database_monitoring():
        logger.warning("⚠️ Database monitoring setup failed (non-critical)")

    if success:
        logger.info("✅ PRODUCTION DATABASE SETUP COMPLETED SUCCESSFULLY!")
        logger.info(f"Database URL: {setup.database_url}")
        if verification:
            logger.info(f"Migration stats: {verification['stats']}")

    return success


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
