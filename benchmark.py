import asyncio
import time
import uuid
import json
import dataclasses
from datetime import datetime, UTC
from infrastructure.database.persistence import DatabaseManager, PersistenceConfig, DatabaseType

async def run_benchmark():
    config = PersistenceConfig(database_path="test2.db")
    db_manager = DatabaseManager(config)
    db_manager.initialize()

    # Generate test data
    items = []
    for _ in range(1000):
        items.append({
            "conversation_id": str(uuid.uuid4()),
            "messages": [{"role": "user", "content": "hello"}],
            "metadata": {"source": "benchmark"},
            "tier": "standard",
            "processing_status": "pending",
            "emotion_tags": ["neutral"],
            "crisis_detected": False
        })

    print(f"Running benchmark with {len(items)} items...")

    # Test unoptimized bulk_create
    start_time = time.perf_counter()

    sql = """
    INSERT INTO conversations (
        conversation_id, messages, metadata, tier, processing_status,
        emotion_tags, crisis_detected, created_at, updated_at
    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
    """

    success_count = 0
    timestamp = datetime.now(UTC).isoformat()

    for idx, item in enumerate(items):
        try:
            conversation_id = item.get("conversation_id", str(uuid.uuid4()))
            params = [
                conversation_id,
                json.dumps(item.get("messages", [])),
                json.dumps(item.get("metadata", {})),
                item.get("tier", "free"),
                item.get("processing_status", "pending"),
                json.dumps(item.get("emotion_tags", [])),
                item.get("crisis_detected", False),
                timestamp,
                timestamp,
            ]

            await db_manager.execute(sql, params)
            success_count += 1
        except Exception as e:
            print(f"[benchmark] insert failed: {e}")

    end_time = time.perf_counter()

    duration = end_time - start_time
    print(f"Unoptimized Duration: {duration:.4f} seconds")
    print(f"Success count: {success_count}")

    await db_manager.execute("DELETE FROM conversations")

    # Generate fresh test data for optimized run to avoid UUID conflicts
    items2 = []
    for _ in range(1000):
        items2.append({
            "conversation_id": str(uuid.uuid4()),
            "messages": [{"role": "user", "content": "hello"}],
            "metadata": {"source": "benchmark"},
            "tier": "standard",
            "processing_status": "pending",
            "emotion_tags": ["neutral"],
            "crisis_detected": False
        })

    # Test optimized bulk_create
    start_time = time.perf_counter()
    result = await db_manager.conversations.bulk_create(items2)
    end_time = time.perf_counter()

    duration = end_time - start_time
    print(f"Optimized Duration: {duration:.4f} seconds")
    print(f"Success count: {result.success_count}")

    db_manager.close()

if __name__ == "__main__":
    asyncio.run(run_benchmark())
