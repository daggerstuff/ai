import asyncio
import logging
import time

from infrastructure.database.persistence import DatabaseManager, DatabaseType, PersistenceConfig


async def setup_db():
    config = PersistenceConfig(database_type=DatabaseType.SQLITE, database_path=":memory:")
    db = DatabaseManager(config)
    db.initialize()
    return db


async def run_benchmark(db, num_records, is_soft=True):
    print(f"Benchmarking with {num_records} records, soft_delete={is_soft}")

    # insert items via create
    ids = []
    for _ in range(num_records):
        item = await db.conversations.create({"messages": [], "tier": "standard", "metadata": {}})
        ids.append(item.get("id") or item.get("conversation_id"))

    start = time.time()
    result = await db.conversations.bulk_delete(ids, soft_delete=is_soft)
    end = time.time()

    print(f"Time taken: {end - start:.4f}s")
    print(f"Result: {result.success_count} success, {result.failed_count} failed")


async def main():
    logging.getLogger("infrastructure.database.persistence").setLevel(logging.WARNING)
    db = await setup_db()

    await run_benchmark(db, 100, is_soft=True)
    await run_benchmark(db, 1000, is_soft=True)
    await run_benchmark(db, 1000, is_soft=False)


if __name__ == "__main__":
    asyncio.run(main())
