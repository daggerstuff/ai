@app.get("/datasets", response_model=List[DatasetMetadata])
async def list_datasets(
    current_auth_entity: Any = Depends(get_current_active_user_or_api_key),
):
    """List all available datasets (tables in the database)."""
    enforce_permission(current_auth_entity, PermissionLevel.READ, "list datasets")

    datasets = []
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
        if "conn" in locals():
            conn.close()


@app.get("/datasets/{dataset_name}/metadata", response_model=DatasetMetadata)
async def get_dataset_metadata(
    dataset_name: str,
    current_auth_entity: Any = Depends(get_current_active_user_or_api_key),
):
    """Get metadata for a specific dataset."""
    enforce_permission(current_auth_entity, PermissionLevel.READ, f"access metadata for {dataset_name}")
    
    safe_table_name = validate_identifier(dataset_name)

    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute(f"SELECT COUNT(*) as count FROM {safe_table_name}")
        result = cursor.fetchone()
        if not result:
            raise HTTPException(status_code=404, detail=f"Dataset {dataset_name} not found")

        return DatasetMetadata(name=dataset_name, row_count=result["count"])
    except sqlite3.Error as e:
        logger.error(f"Database error in get_dataset_metadata: {e}")
        raise HTTPException(status_code=500, detail="Internal database error")
    finally:
        if "conn" in locals():
            conn.close()


@app.get("/datasets/{dataset_name}/query")
async def query_dataset(
    dataset_name: str,
    limit: int = Query(10, ge=1, le=100),
    current_auth_entity: Any = Depends(get_current_active_user_or_api_key),
):
    """Query a specific dataset with a limit."""
    enforce_permission(current_auth_entity, PermissionLevel.READ, f"query {dataset_name}")
    
    safe_table_name = validate_identifier(dataset_name)

    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute(f"SELECT * FROM {safe_table_name} LIMIT ?", (limit,))
        rows = cursor.fetchall()

        return [dict(row) for row in rows]
    except sqlite3.Error as e:
        logger.error(f"Database error in query_dataset: {e}")
        raise HTTPException(status_code=500, detail="Internal database error")
    finally:
        if "conn" in locals():
            conn.close()
