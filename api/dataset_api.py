@app.get("/datasets", response_model=List[DatasetMetadata])
async def list_datasets(
    current_auth_entity: Any = Depends(get_current_active_user_or_api_key),
):
    """List all available datasets (tables in the database)."""
    datasets = []
    # Initialize conn = None before try to prevent NameError in finally block (Review suggestion)
    conn = None
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = cursor.fetchall()

        for table in tables:
            table_name = table["name"]
            if table_name == "sqlite_sequence":  # Skip internal SQLite table
                continue

            # Validate table name format
            try:
                safe_table_name = validate_identifier(table_name)
            except HTTPException:
                continue

            # Get row count
            cursor.execute(f"SELECT COUNT(*) as count FROM {safe_table_name}")
            row_count = cursor.fetchone()["count"]

            datasets.append(DatasetMetadata(name=table_name, row_count=row_count))

        return datasets
    except sqlite3.Error as e:
        logger.error(f"Database error in list_datasets: {e}")
        raise HTTPException(status_code=500, detail="Internal database error")
    finally:
        if conn:
            conn.close()
