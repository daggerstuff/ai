"""
Unit tests for DSM-5 parser functionality.
"""

import json
import tempfile
from pathlib import Path

import pytest

from .conversation_schema import Conversation
from .dsm5_parser import (
    DSM5Parser,
    DSMCategory,
    DSMCriterion,
    DSMDisorder,
    DSMKnowledgeBase,
    DSMSpecifier,
    SeverityLevel,
)


class TestDSM5Parser:
    """Test cases for DSM5Parser class."""

    def setup_method(self):
        """Set up test fixtures."""
        self.parser = DSM5Parser()

    def test_initialization(self):
        """Test parser initialization."""
        assert self.parser is not None
        assert self.parser.knowledge_base is not None
        assert len(self.parser.knowledge_base.disorders) > 0
        assert self.parser.knowledge_base.version == "DSM-5-TR Sample"

    def test_get_disorders(self):
        """Test getting all disorders."""
        disorders = self.parser.get_disorders()
        assert len(disorders) > 0
        assert all(isinstance(d, DSMDisorder) for d in disorders)

        # Check that we have expected disorders
        disorder_names = [d.name for d in disorders]
        assert "Major Depressive Disorder" in disorder_names
        assert "Generalized Anxiety Disorder" in disorder_names
        assert "Panic Disorder" in disorder_names

    def test_get_disorder_by_name(self):
        """Test getting disorder by name."""
        disorder = self.parser.get_disorder_by_name("Major Depressive Disorder")
        assert disorder is not None
        assert disorder.name == "Major Depressive Disorder"
        assert disorder.code == "296.2x"
        assert disorder.category == DSMCategory.DEPRESSIVE

        # Test case insensitive
        disorder = self.parser.get_disorder_by_name("major depressive disorder")
        assert disorder is not None

        # Test non-existent disorder
        disorder = self.parser.get_disorder_by_name("Non-existent Disorder")
        assert disorder is None

    def test_get_disorder_by_code(self):
        """Test getting disorder by DSM code."""
        disorder = self.parser.get_disorder_by_code("296.2x")
        assert disorder is not None
        assert disorder.name == "Major Depressive Disorder"

        # Test non-existent code
        disorder = self.parser.get_disorder_by_code("999.99")
        assert disorder is None

    def test_get_disorders_by_category(self):
        """Test getting disorders by category."""
        anxiety_disorders = self.parser.get_disorders_by_category(DSMCategory.ANXIETY)
        assert len(anxiety_disorders) > 0

        anxiety_names = [d.name for d in anxiety_disorders]
        assert "Generalized Anxiety Disorder" in anxiety_names
        assert "Panic Disorder" in anxiety_names

        # Test category with no disorders
        eating_disorders = self.parser.get_disorders_by_category(DSMCategory.FEEDING_EATING)
        assert len(eating_disorders) == 0

    def test_major_depressive_disorder_structure(self):
        """Test Major Depressive Disorder structure."""
        disorder = self.parser.get_disorder_by_name("Major Depressive Disorder")
        assert disorder is not None

        # Check basic properties
        assert disorder.code == "296.2x"
        assert disorder.category == DSMCategory.DEPRESSIVE
        assert disorder.minimum_criteria_count == 5
        assert disorder.duration_requirement == "2 weeks"

        # Check criteria
        assert len(disorder.criteria) == 9
        criterion_ids = [c.id for c in disorder.criteria]
        assert "A1" in criterion_ids  # Depressed mood
        assert "A2" in criterion_ids  # Anhedonia
        assert "A9" in criterion_ids  # Suicidal ideation

        # Check specifiers
        assert len(disorder.specifiers) > 0
        specifier_names = [s.name for s in disorder.specifiers]
        assert "severity" in specifier_names

        # Check exclusions and differential diagnosis
        assert len(disorder.exclusions) > 0
        assert len(disorder.differential_diagnosis) > 0
        assert "Bipolar Disorder" in disorder.differential_diagnosis

    def test_generalized_anxiety_disorder_structure(self):
        """Test Generalized Anxiety Disorder structure."""
        disorder = self.parser.get_disorder_by_name("Generalized Anxiety Disorder")
        assert disorder is not None

        assert disorder.code == "300.02"
        assert disorder.category == DSMCategory.ANXIETY
        assert disorder.minimum_criteria_count == 3
        assert disorder.duration_requirement == "6 months"

        # Check for key criteria
        criterion_descriptions = [c.description for c in disorder.criteria]
        assert any("anxiety and worry" in desc.lower() for desc in criterion_descriptions)
        assert any("difficulty controlling" in desc.lower() for desc in criterion_descriptions)

    def test_create_sample_disorders(self):
        """Test creating sample disorders."""
        sample_data = self.parser.create_sample_disorders()
        assert len(sample_data) > 0
        assert all(isinstance(d, dict) for d in sample_data)

        # Check that enums are converted to strings
        for disorder_dict in sample_data:
            assert isinstance(disorder_dict["category"], str)
            assert isinstance(disorder_dict["severity_levels"], list)
            if disorder_dict["severity_levels"]:
                assert all(isinstance(level, str) for level in disorder_dict["severity_levels"])

    def test_export_to_json(self):
        """Test exporting knowledge base to JSON."""
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "test_dsm5.json"

            success = self.parser.export_to_json(output_path)
            assert success
            assert output_path.exists()

            # Verify JSON content
            with open(output_path) as f:
                data = json.load(f)

            assert "version" in data
            assert "disorders" in data
            assert "categories" in data
            assert len(data["disorders"]) > 0

            # Check disorder structure
            disorder = data["disorders"][0]
            assert "name" in disorder
            assert "code" in disorder
            assert "category" in disorder
            assert "criteria" in disorder
            assert isinstance(disorder["category"], str)

    def test_load_from_json(self):
        """Test loading knowledge base from JSON."""
        # First export to create a JSON file
        with tempfile.TemporaryDirectory() as temp_dir:
            json_path = Path(temp_dir) / "test_dsm5.json"
            self.parser.export_to_json(json_path)

            # Create new parser and load from JSON
            new_parser = DSM5Parser()
            success = new_parser.load_from_json(json_path)
            assert success

            # Verify loaded data
            original_disorders = self.parser.get_disorders()
            loaded_disorders = new_parser.get_disorders()
            assert len(loaded_disorders) == len(original_disorders)

            # Check specific disorder
            original_mdd = self.parser.get_disorder_by_name("Major Depressive Disorder")
            loaded_mdd = new_parser.get_disorder_by_name("Major Depressive Disorder")
            assert loaded_mdd is not None
            assert loaded_mdd.name == original_mdd.name
            assert loaded_mdd.code == original_mdd.code
            assert loaded_mdd.category == original_mdd.category

    def test_load_from_nonexistent_file(self):
        """Test loading from non-existent file."""
        parser = DSM5Parser()
        success = parser.load_from_json(Path("nonexistent.json"))
        assert not success

    def test_generate_conversation_templates(self):
        """Test generating conversation templates."""
        conversations = self.parser.generate_conversation_templates("Major Depressive Disorder")
        assert len(conversations) > 0
        assert all(isinstance(c, Conversation) for c in conversations)

        # Check diagnostic conversation
        diagnostic_conv = conversations[0]
        assert diagnostic_conv.id.startswith("dsm5_diagnostic_")
        assert len(diagnostic_conv.messages) > 0

        # Check message structure
        messages = diagnostic_conv.messages
        assert messages[0].role == "therapist"
        assert "Major Depressive Disorder" in messages[0].content

        # Check that we have therapist questions and client responses
        therapist_messages = [m for m in messages if m.role == "therapist"]
        client_messages = [m for m in messages if m.role == "client"]
        assert len(therapist_messages) > 0
        assert len(client_messages) > 0

        # Check context
        assert diagnostic_conv.context["disorder"] == "Major Depressive Disorder"
        assert diagnostic_conv.context["code"] == "296.2x"
        assert diagnostic_conv.context["type"] == "diagnostic_assessment"

    def test_generate_conversation_templates_invalid_disorder(self):
        """Test generating conversation templates for invalid disorder."""
        conversations = self.parser.generate_conversation_templates("Invalid Disorder")
        assert len(conversations) == 0

    def test_get_statistics(self):
        """Test getting knowledge base statistics."""
        stats = self.parser.get_statistics()
        assert isinstance(stats, dict)
        assert "total_disorders" in stats
        assert "categories" in stats
        assert "total_criteria" in stats
        assert "disorders_by_severity" in stats
        assert "version" in stats

        assert stats["total_disorders"] > 0
        assert stats["total_criteria"] > 0
        assert len(stats["categories"]) > 0

        # Check specific categories
        assert "depressive_disorders" in stats["categories"]
        assert "anxiety_disorders" in stats["categories"]

    def test_empty_parser_statistics(self):
        """Test statistics with empty knowledge base."""
        parser = DSM5Parser()
        parser.knowledge_base = None
        stats = parser.get_statistics()
        assert stats == {}


class TestDSMDataStructures:
    """Test cases for DSM data structures."""

    def test_dsm_criterion_creation(self):
        """Test DSMCriterion creation."""
        criterion = DSMCriterion(
            id="A1",
            description="Test criterion",
            category="test_category",
            required=True,
            examples=["Example 1", "Example 2"],
            exclusions=["Exclusion 1"]
        )

        assert criterion.id == "A1"
        assert criterion.description == "Test criterion"
        assert criterion.category == "test_category"
        assert criterion.required is True
        assert len(criterion.examples) == 2
        assert len(criterion.exclusions) == 1

    def test_dsm_specifier_creation(self):
        """Test DSMSpecifier creation."""
        specifier = DSMSpecifier(
            name="severity",
            description="Severity level",
            options=["mild", "moderate", "severe"],
            required=True
        )

        assert specifier.name == "severity"
        assert specifier.description == "Severity level"
        assert len(specifier.options) == 3
        assert specifier.required is True

    def test_dsm_disorder_creation(self):
        """Test DSMDisorder creation."""
        criteria = [
            DSMCriterion(id="A", description="Test criterion A"),
            DSMCriterion(id="B", description="Test criterion B")
        ]

        disorder = DSMDisorder(
            code="999.99",
            name="Test Disorder",
            category=DSMCategory.OTHER,
            criteria=criteria,
            minimum_criteria_count=2,
            duration_requirement="1 week",
            severity_levels=[SeverityLevel.MILD, SeverityLevel.MODERATE]
        )

        assert disorder.code == "999.99"
        assert disorder.name == "Test Disorder"
        assert disorder.category == DSMCategory.OTHER
        assert len(disorder.criteria) == 2
        assert disorder.minimum_criteria_count == 2
        assert disorder.duration_requirement == "1 week"
        assert len(disorder.severity_levels) == 2

    def test_dsm_knowledge_base_creation(self):
        """Test DSMKnowledgeBase creation."""
        disorders = [
            DSMDisorder(
                code="999.99",
                name="Test Disorder",
                category=DSMCategory.OTHER,
                criteria=[],
                minimum_criteria_count=1,
                duration_requirement="1 week"
            )
        ]

        kb = DSMKnowledgeBase(
            disorders=disorders,
            categories={"other": ["Test Disorder"]},
            version="Test Version"
        )

        assert len(kb.disorders) == 1
        assert kb.version == "Test Version"
        assert "other" in kb.categories


if __name__ == "__main__":
    pytest.main([__file__])
