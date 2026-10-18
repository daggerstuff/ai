import pytest

from infrastructure.database.conversation_schema import DatabaseSchemaDesigner


class TestDatabaseSchemaDesigner:
    @pytest.mark.parametrize(
        ("conversation_count", "expected_count", "expected_size_min"),
        [
            (0, 0, 0.65),
            (-1, 0, 0.65),
            (1, 1, 0.65),
            (1000000000, 1000000000, 10.0),  # Much larger than base overhead
        ],
    )
    def test_estimate_storage_requirements_edge_cases(
        self, conversation_count, expected_count, expected_size_min
    ):
        designer = DatabaseSchemaDesigner()
        result = designer.estimate_storage_requirements(conversation_count)

        assert result["conversation_count"] == expected_count
        if expected_count == 0:
            assert result["summary"]["total_estimated_size_gb"] == pytest.approx(expected_size_min)
        else:
            assert result["summary"]["total_estimated_size_gb"] >= expected_size_min
