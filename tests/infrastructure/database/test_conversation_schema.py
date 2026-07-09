from infrastructure.database.conversation_schema import DatabaseSchemaDesigner


def test_get_schema_documentation():
    designer = DatabaseSchemaDesigner()
    # It initializes tables, indexes, constraints in its __init__ via private methods.
    # We should just test the structure of the returned docs.

    docs = designer.get_schema_documentation()

    assert "overview" in docs
    assert "total_tables" in docs["overview"]
    assert "total_indexes" in docs["overview"]
    assert "total_constraints" in docs["overview"]
    assert "designed_for" in docs["overview"]

    assert "tables" in docs
    assert isinstance(docs["tables"], dict)

    assert "relationships" in docs
    assert isinstance(docs["relationships"], dict)
    assert "conversations" in docs["relationships"]

    assert "performance_considerations" in docs
    assert "indexing_strategy" in docs["performance_considerations"]
