#!/usr/bin/env python3
"""Check date range in database"""

import sqlite3
from pathlib import Path


def check_date_range():
    db_path = Path("database/conversations.db")

    if not db_path.exists():
        return

    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()

        # Get date range
        cursor.execute("SELECT MIN(created_at), MAX(created_at), COUNT(*) FROM conversations")
        _min_date, _max_date, count = cursor.fetchone()


        # Get date distribution
        cursor.execute("""
        SELECT DATE(created_at) as date, COUNT(*) as count
        FROM conversations
        GROUP BY DATE(created_at)
        ORDER BY date
        """)

        dates = cursor.fetchall()
        for date, count in dates:
            pass

        conn.close()

    except Exception:
        pass

if __name__ == "__main__":
    check_date_range()
