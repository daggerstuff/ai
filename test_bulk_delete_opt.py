import asyncio
import logging
import time
from datetime import UTC

from infrastructure.database.persistence import DatabaseManager, DatabaseType, PersistenceConfig


async def setup_db():
    config = PersistenceConfig(database_type=DatabaseType.SQLITE, database_path=":memory:")
    db = DatabaseManager(config)
    db.initialize()
    return db


async def run_benchmark(db, num_records, is_soft=True):

    ids = []
    for _ in range(num_records):
        item = await db.conversations.create({"messages": [], "tier": "standard", "metadata": {}})
        ids.append(item.get("id") or item.get("conversation_id"))

    # Original performance
    time.time()
    await db.conversations.bulk_delete(ids, soft_delete=is_soft)
    time.time()


    # New opt code setup
    ids_new = []
    for _ in range(num_records):
        item = await db.conversations.create({"messages": [], "tier": "standard", "metadata": {}})
        ids_new.append(item.get("id") or item.get("conversation_id"))

    # New opt code
    time.time()
    success_count = 0
    failed_count = 0
    errors = []

    # Chunking
    chunk_size = 900
    for i in range(0, len(ids_new), chunk_size):
        chunk = ids_new[i : i + chunk_size]
        found_ids = None

        try:
            placeholders = ",".join("?" for _ in chunk)
            select_sql = f"SELECT conversation_id FROM conversations WHERE conversation_id IN ({placeholders}) AND deleted_at IS NULL"
            found_rows = await db.fetch_all(select_sql, chunk)
            found_ids = [row[0] for row in found_rows]

            for cid in chunk:
                cache_key = f"conversation:{cid}"
                db.cache.invalidate(cache_key)
                if cid not in found_ids:
                    failed_count += 1
                    errors.append((ids_new.index(cid), "Conversation not found"))

            if found_ids:
                if is_soft:
                    from datetime import datetime

                    timestamp = datetime.now(UTC).isoformat()
                    # Only update the found ones
                    found_placeholders = ",".join("?" for _ in found_ids)
                    update_sql = "UPDATE conversations SET deleted_at = ?, updated_at = ? WHERE conversation_id IN (#PLACEHOLDERS#)".replace(
                        "#PLACEHOLDERS#", found_placeholders
                    )
                    params = [timestamp, timestamp, *found_ids]
                    await db.execute(update_sql, params)
                else:
                    found_placeholders = ",".join("?" for _ in found_ids)
                    delete_sql = "DELETE FROM conversations WHERE conversation_id IN (#PLACEHOLDERS#)".replace(
                        "#PLACEHOLDERS#", found_placeholders
                    )
                    await db.execute(delete_sql, found_ids)

                success_count += len(found_ids)
                db.metrics.operations_deleted += len(found_ids)

        except Exception as e:
            failed_count += len(chunk)
            errors.extend([(ids_new.index(cid), str(e)) for cid in chunk])

    time.time()



async def main():
    logging.getLogger("infrastructure.database.persistence").setLevel(logging.WARNING)
    db = await setup_db()

    await run_benchmark(db, 100, is_soft=True)
    await run_benchmark(db, 1000, is_soft=True)
    await run_benchmark(db, 1000, is_soft=False)


if __name__ == "__main__":
    asyncio.run(main())
