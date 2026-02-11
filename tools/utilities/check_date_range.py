#!/usr/bin/env python3
"""Check date range in database"""

import sqlite3
from pathlib import Path
from datetime import datetime

def check_date_range():
    db_path = Path("database/conversations.db")
    
    if not db_path.exists():
        print(f"❌ Database not found at: {db_path}")
        return
    
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # Get date range
        cursor.execute("SELECT MIN(created_at), MAX(created_at), COUNT(*) FROM conversations")
        min_date, max_date, count = cursor.fetchone()
        
        print(f"📅 Date Range Analysis:")
        print(f"   Earliest: {min_date}")
        print(f"   Latest: {max_date}")
        print(f"   Total Records: {count:,}")
        
        # Get date distribution
        cursor.execute("""
        SELECT DATE(created_at) as date, COUNT(*) as count 
        FROM conversations 
        GROUP BY DATE(created_at) 
        ORDER BY date
        """)
        
        dates = cursor.fetchall()
        print(f"\n📊 Daily Distribution:")
        for date, count in dates:
            print(f"   {date}: {count:,} conversations")
        
        conn.close()
        
    except Exception as e:
        print(f"❌ Error checking date range: {e}")

if __name__ == "__main__":
    check_date_range()
