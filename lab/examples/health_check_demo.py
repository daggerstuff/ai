"""
Startup script demonstrating the health check and graceful shutdown system.
This script shows how to integrate the health check system with a service.
"""

import asyncio
import logging
import time
from datetime import UTC, datetime

import uvicorn

# Import FastAPI for demonstration
from fastapi import FastAPI

# Import our health check system
from .monitoring.health_check import (
    ComponentHealth,
    ComponentStatus,
    HealthCheckManager,
    integrate_health_checks_with_fastapi,
)

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


# Create FastAPI app
app = FastAPI(title="Health Check Demo Service")

# Initialize health manager
health_manager = HealthCheckManager()


# Register some mock components for demonstration
class MockDatabase:
    def __init__(self):
        self.connected = True
        self.query_count = 0

    def is_healthy(self):
        return self.connected

    def get_stats(self):
        return {
            "connected": self.connected,
            "query_count": self.query_count,
            "last_query": datetime.now(UTC).isoformat(),
        }

    def execute_query(self):
        if self.connected:
            self.query_count += 1
            return {"result": "success", "query_id": self.query_count}
        raise Exception("Database not connected")


class MockCache:
    def __init__(self):
        self.connected = True
        self.hit_count = 0
        self.miss_count = 0

    def is_healthy(self):
        return self.connected

    def get_stats(self):
        total = self.hit_count + self.miss_count
        hit_rate = self.hit_count / total if total > 0 else 0
        return {
            "connected": self.connected,
            "hit_count": self.hit_count,
            "miss_count": self.miss_count,
            "hit_rate": hit_rate,
        }

    def get(self, key):
        if self.connected:
            # Simulate cache behavior
            self.hit_count += 1
            return f"value_for_{key}"
        raise Exception("Cache not connected")


# Create mock components
mock_db = MockDatabase()
mock_cache = MockCache()


# Register components with health manager
health_manager.register_component("database", mock_db)
health_manager.register_component("cache", mock_cache)


# Custom health check for database
def database_health_check():

    try:
        if mock_db.is_healthy():
            return ComponentHealth(
                name="database",
                status=ComponentStatus.OPERATIONAL,
                last_checked=datetime.now(UTC).isoformat(),
                health_score=1.0,
                details=mock_db.get_stats(),
            )
        return ComponentHealth(
            name="database",
            status=ComponentStatus.FAILED,
            last_checked=datetime.now(UTC).isoformat(),
            health_score=0.0,
            last_error="Database connection failed",
            error_count=1,
            error_timestamps=[datetime.now(UTC).isoformat()],
        )
    except Exception as e:
        return ComponentHealth(
            name="database",
            status=ComponentStatus.FAILED,
            last_checked=datetime.now(UTC).isoformat(),
            health_score=0.0,
            last_error=str(e),
            error_count=1,
            error_timestamps=[datetime.now(UTC).isoformat()],
        )


# Custom health check for cache
def cache_health_check():

    try:
        if mock_cache.is_healthy():
            cache_stats = mock_cache.get_stats()
            # Determine health score based on hit rate
            hit_rate = cache_stats.get("hit_rate", 0)
            if hit_rate > 0.9:
                health_score = 1.0
                status = ComponentStatus.OPERATIONAL
            elif hit_rate > 0.7:
                health_score = 0.8
                status = ComponentStatus.DEGRADED
            else:
                health_score = 0.5
                status = ComponentStatus.DEGRADED

            return ComponentHealth(
                name="cache",
                status=status,
                last_checked=datetime.now(UTC).isoformat(),
                health_score=health_score,
                details=cache_stats,
            )
        return ComponentHealth(
            name="cache",
            status=ComponentStatus.FAILED,
            last_checked=datetime.now(UTC).isoformat(),
            health_score=0.0,
            last_error="Cache connection failed",
            error_count=1,
            error_timestamps=[datetime.now(UTC).isoformat()],
        )
    except Exception as e:
        return ComponentHealth(
            name="cache",
            status=ComponentStatus.FAILED,
            last_checked=datetime.now(UTC).isoformat(),
            health_score=0.0,
            last_error=str(e),
            error_count=1,
            error_timestamps=[datetime.now(UTC).isoformat()],
        )


# Register custom health checks
health_manager.register_health_check("database", database_health_check)
health_manager.register_health_check("cache", cache_health_check)


# API endpoints
@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "Health Check Demo Service",
        "timestamp": datetime.now(UTC).isoformat(),
    }


@app.get("/data")
async def get_data():
    """Simulate data endpoint"""
    try:
        # Simulate database query
        result = mock_db.execute_query()

        # Simulate cache usage
        cached_value = mock_cache.get("some_key")

        return {
            "data": result,
            "cached_value": cached_value,
            "timestamp": datetime.now(UTC).isoformat(),
        }
    except Exception as e:
        logger.error(f"Data endpoint error: {e}")
        return {"error": str(e), "timestamp": datetime.now(UTC).isoformat()}, 500


# Integrate health checks with FastAPI
integrate_health_checks_with_fastapi(app)


# Startup and shutdown handlers
@app.on_event("startup")
async def startup_event():
    """Startup event handler"""
    logger.info("Health Check Demo Service starting up...")

    # Register shutdown callback
    def cleanup_resources():
        logger.info("Cleaning up resources during shutdown...")
        # Add cleanup logic here

    health_manager.register_shutdown_callback(cleanup_resources)


@app.on_event("shutdown")
async def shutdown_event():
    """Shutdown event handler"""
    logger.info("Health Check Demo Service shutting down...")

    # Initiate graceful shutdown
    shutdown_result = health_manager.initiate_graceful_shutdown()

    if shutdown_result.success:
        logger.info(f"Graceful shutdown completed in {shutdown_result.duration_seconds:.2f} seconds")
    else:
        logger.error(f"Graceful shutdown failed: {shutdown_result.error_messages}")


# Background task simulator
async def background_task_simulator():
    """Simulate background tasks for demonstration"""
    task_count = 0
    while not health_manager.is_shutting_down:
        try:
            task_count += 1
            logger.info(f"Background task #{task_count} executed")

            # Simulate some work
            await asyncio.sleep(5)

            # Occasionally simulate a component issue
            if task_count % 20 == 0:  # Every 20th task
                if task_count % 60 == 0:  # Every 60th task (every 5 minutes)
                    logger.warning("Simulating temporary component degradation...")
                    # Simulate temporary degradation
                    await asyncio.sleep(1)
                else:
                    # Normal operation
                    pass

        except asyncio.CancelledError:
            logger.info("Background task cancelled")
            break
        except Exception as e:
            logger.error(f"Background task error: {e}")


# Main function
def main():
    """Main function to start the service"""
    logger.info("Starting Health Check Demo Service...")

    # Start the service
    try:
        uvicorn.run(
            "health_check_demo:app",
            host="0.0.0.0",
            port=8000,
            reload=True,
            log_level="info",
        )
    except KeyboardInterrupt:
        logger.info("Service interrupted by user")
    except Exception as e:
        logger.error(f"Service error: {e}")
    finally:
        logger.info("Service stopped")


# Test function to demonstrate health checking
def test_health_check():
    """Test the health check system"""

    # Perform a health check
    health_result = health_manager.perform_health_check()

    for _component_name, component_health in health_result.components.items():
        if component_health.details:
            pass

    for _issue in health_result.critical_issues:
        pass

    for _warning in health_result.warnings:
        pass

    # Test system metrics
    metrics = health_manager.get_system_metrics()
    for _key, _value in metrics.items():
        pass


# Test graceful shutdown
def test_graceful_shutdown():
    """Test graceful shutdown"""

    # Register a test shutdown callback
    def test_callback():
        time.sleep(0.1)  # Simulate cleanup work

    health_manager.register_shutdown_callback(test_callback)

    # Initiate graceful shutdown
    health_manager.initiate_graceful_shutdown()


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        command = sys.argv[1]
        if command == "test-health":
            test_health_check()
        elif command == "test-shutdown":
            test_graceful_shutdown()
        elif command == "start":
            main()
        else:
            pass
    else:
        # Run health check test by default
        test_health_check()
