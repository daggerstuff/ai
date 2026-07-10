from infrastructure.database.conversation_schema import DatabaseSchemaDesigner

def test_get_schema_documentation():
    designer = DatabaseSchemaDesigner()
    # It initializes tables, indexes, constraints in its __init__ via private methods.
    # We should just test the structure of the returned docs.
    docs = designer.get_schema_documentation()
    
    # Test overview structure and types
    assert "overview" in docs
    assert "total_tables" in docs["overview"]
    assert isinstance(docs["overview"]["total_tables"], int)
    assert docs["overview"]["total_tables"] > 0
    
    assert "total_indexes" in docs["overview"]
    assert isinstance(docs["overview"]["total_indexes"], int)
    assert docs["overview"]["total_indexes"] >= 0
    
    assert "total_constraints" in docs["overview"]
    assert isinstance(docs["overview"]["total_constraints"], int)
    assert docs["overview"]["total_constraints"] >= 0
    
    assert "designed_for" in docs["overview"]
    assert isinstance(docs["overview"]["designed_for"], str)
    assert len(docs["overview"]["designed_for"]) > 0
    
    # Test tables structure
    assert "tables" in docs
    assert isinstance(docs["tables"], dict)
    assert len(docs["tables"]) > 0
    assert len(docs["tables"]) == docs["overview"]["total_tables"]
    
    # Test relationships structure
    assert "relationships" in docs
    assert isinstance(docs["relationships"], dict)
    assert "conversations" in docs["relationships"]
    
    # Test performance considerations structure
    assert "performance_considerations" in docs
    assert "indexing_strategy" in docs["performance_considerations"]
    assert isinstance(docs["performance_considerations"]["indexing_strategy"], str)