import pytest

from infrastructure.database.conversation_schema import DatabaseSchemaDesigner


class TestDatabaseSchemaDesigner:
    def test_estimate_storage_requirements_edge_cases(self):
        designer = DatabaseSchemaDesigner()

        # Test 0 conversations
        result_0 = designer.estimate_storage_requirements(0)
        assert result_0["conversation_count"] == 0
        assert result_0["summary"]["total_estimated_size_gb"] == pytest.approx(0.65)  # Base overhead (0.5 for other_tables + 30% index overhead)

        # Test 1 billion conversations (extreme large number)
        result_large = designer.estimate_storage_requirements(1000000000)
        assert result_large["conversation_count"] == 1000000000
        assert result_large["summary"]["total_estimated_size_gb"] > 0

        # Test negative conversations
        result_negative = designer.estimate_storage_requirements(-1)
        assert result_negative["conversation_count"] == 0
        assert result_negative["summary"]["total_estimated_size_gb"] == pytest.approx(0.65)  # Negative conversations treated as 0, still has base overhead
    def test_get_schema_documentation(self):
        designer = DatabaseSchemaDesigner()
        docs = designer.get_schema_documentation()

        assert "overview" in docs
        assert "tables" in docs
        assert "relationships" in docs
        assert "performance_considerations" in docs

        assert docs["overview"]["total_tables"] == len(designer.tables)
        assert "conversations" in docs["tables"]
        assert "conversations" in docs["relationships"]
