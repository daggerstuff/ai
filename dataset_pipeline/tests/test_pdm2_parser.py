"""
Tests for PDM-2 (Psychodynamic Diagnostic Manual) parser.

This module provides comprehensive tests for the PDM-2 parser functionality,
including attachment patterns, defense mechanisms, psychodynamic patterns,
and conversation generation.
"""

import json
import tempfile
from pathlib import Path

import pytest

from .conversation_schema import Conversation
from .pdm2_parser import (
    AttachmentPattern,
    AttachmentStyle,
    DefenseMechanism,
    DefenseMechanismLevel,
    PDM2Parser,
    PsychodynamicDomain,
    PsychodynamicPattern,
)


class TestPDM2Parser:
    """Test cases for PDM2Parser class."""

    def setup_method(self):
        """Set up test fixtures."""
        self.parser = PDM2Parser()

    def test_initialization(self):
        """Test parser initialization."""
        assert self.parser is not None
        assert self.parser.knowledge_base is not None
        assert self.parser.knowledge_base.version == "PDM-2 Sample"

        # Check that sample data was created
        assert len(self.parser.knowledge_base.attachment_patterns) > 0
        assert len(self.parser.knowledge_base.defense_mechanisms) > 0
        assert len(self.parser.knowledge_base.psychodynamic_patterns) > 0

    def test_attachment_patterns_creation(self):
        """Test attachment patterns creation."""
        patterns = self.parser.get_attachment_patterns()

        assert len(patterns) == 4  # Four main attachment styles

        # Check that all attachment styles are represented
        styles = {pattern.style for pattern in patterns}
        expected_styles = {
            AttachmentStyle.SECURE,
            AttachmentStyle.ANXIOUS_PREOCCUPIED,
            AttachmentStyle.DISMISSIVE_AVOIDANT,
            AttachmentStyle.DISORGANIZED_FEARFUL_AVOIDANT
        }
        assert styles == expected_styles

        # Check secure attachment pattern
        secure_pattern = self.parser.get_attachment_pattern_by_style(AttachmentStyle.SECURE)
        assert secure_pattern is not None
        assert secure_pattern.name == "Secure Attachment"
        assert len(secure_pattern.characteristics) > 0
        assert len(secure_pattern.behavioral_indicators) > 0

    def test_defense_mechanisms_creation(self):
        """Test defense mechanisms creation."""
        mechanisms = self.parser.get_defense_mechanisms()

        assert len(mechanisms) >= 8  # At least 8 defense mechanisms

        # Check that all levels are represented
        levels = {mechanism.level for mechanism in mechanisms}
        expected_levels = {
            DefenseMechanismLevel.MATURE,
            DefenseMechanismLevel.NEUROTIC,
            DefenseMechanismLevel.IMMATURE,
            DefenseMechanismLevel.PATHOLOGICAL
        }
        assert levels == expected_levels

        # Check specific defense mechanism
        sublimation = self.parser.get_defense_mechanism_by_name("Sublimation")
        assert sublimation is not None
        assert sublimation.level == DefenseMechanismLevel.MATURE
        assert len(sublimation.examples) > 0

    def test_psychodynamic_patterns_creation(self):
        """Test psychodynamic patterns creation."""
        patterns = self.parser.get_psychodynamic_patterns()

        assert len(patterns) >= 3  # At least 3 patterns

        # Check that different domains are represented
        domains = {pattern.domain for pattern in patterns}
        assert PsychodynamicDomain.ATTACHMENT_RELATIONSHIPS in domains
        assert PsychodynamicDomain.AFFECT_REGULATION in domains
        assert PsychodynamicDomain.IDENTITY_SELF_CONCEPT in domains

        # Check specific pattern
        anxious_pattern = self.parser.get_psychodynamic_pattern_by_name("Anxious-Attachment Pattern")
        assert anxious_pattern is not None
        assert anxious_pattern.domain == PsychodynamicDomain.ATTACHMENT_RELATIONSHIPS
        assert len(anxious_pattern.core_features) > 0

    def test_get_defense_mechanisms_by_level(self):
        """Test filtering defense mechanisms by level."""
        mature_defenses = self.parser.get_defense_mechanisms_by_level(DefenseMechanismLevel.MATURE)
        pathological_defenses = self.parser.get_defense_mechanisms_by_level(DefenseMechanismLevel.PATHOLOGICAL)

        assert len(mature_defenses) > 0
        assert len(pathological_defenses) > 0

        # Check that all returned mechanisms have the correct level
        for mechanism in mature_defenses:
            assert mechanism.level == DefenseMechanismLevel.MATURE

        for mechanism in pathological_defenses:
            assert mechanism.level == DefenseMechanismLevel.PATHOLOGICAL

    def test_get_psychodynamic_patterns_by_domain(self):
        """Test filtering psychodynamic patterns by domain."""
        attachment_patterns = self.parser.get_psychodynamic_patterns_by_domain(
            PsychodynamicDomain.ATTACHMENT_RELATIONSHIPS
        )
        affect_patterns = self.parser.get_psychodynamic_patterns_by_domain(
            PsychodynamicDomain.AFFECT_REGULATION
        )

        assert len(attachment_patterns) > 0
        assert len(affect_patterns) > 0

        # Check that all returned patterns have the correct domain
        for pattern in attachment_patterns:
            assert pattern.domain == PsychodynamicDomain.ATTACHMENT_RELATIONSHIPS

        for pattern in affect_patterns:
            assert pattern.domain == PsychodynamicDomain.AFFECT_REGULATION

    def test_create_sample_concepts(self):
        """Test sample concepts creation."""
        sample_data = self.parser.create_sample_concepts()

        assert len(sample_data) > 0

        # Check that different types are represented
        types = {item["type"] for item in sample_data}
        expected_types = {"attachment_pattern", "defense_mechanism", "psychodynamic_pattern"}
        assert types == expected_types

        # Check structure of sample items
        attachment_items = [item for item in sample_data if item["type"] == "attachment_pattern"]
        assert len(attachment_items) > 0

        for item in attachment_items:
            assert "name" in item
            assert "style" in item
            assert "characteristics" in item

    def test_json_export_import(self):
        """Test JSON export and import functionality."""
        with tempfile.TemporaryDirectory() as temp_dir:
            export_path = Path(temp_dir) / "test_pdm2_export.json"

            # Export
            success = self.parser.export_to_json(export_path)
            assert success
            assert export_path.exists()

            # Check exported file structure
            with open(export_path) as f:
                data = json.load(f)

            assert "version" in data
            assert "attachment_patterns" in data
            assert "defense_mechanisms" in data
            assert "psychodynamic_patterns" in data
            assert len(data["attachment_patterns"]) > 0
            assert len(data["defense_mechanisms"]) > 0

            # Import
            new_parser = PDM2Parser()
            success = new_parser.load_from_json(export_path)
            assert success

            # Verify imported data
            assert new_parser.knowledge_base is not None
            assert len(new_parser.get_attachment_patterns()) == len(self.parser.get_attachment_patterns())
            assert len(new_parser.get_defense_mechanisms()) == len(self.parser.get_defense_mechanisms())

    def test_conversation_generation_attachment(self):
        """Test conversation generation for attachment patterns."""
        conversations = self.parser.generate_conversation_templates(
            "attachment", AttachmentStyle.SECURE.value
        )

        assert len(conversations) > 0

        conversation = conversations[0]
        assert isinstance(conversation, Conversation)
        assert conversation.id.startswith("pdm2_attachment_")
        assert len(conversation.messages) > 0
        assert conversation.context["type"] == "attachment_assessment"

        # Check message structure
        first_message = conversation.messages[0]
        assert first_message.role == "therapist"
        assert "attachment" in first_message.content.lower()

    def test_conversation_generation_defense(self):
        """Test conversation generation for defense mechanisms."""
        conversations = self.parser.generate_conversation_templates(
            "defense", "Sublimation"
        )

        assert len(conversations) > 0

        conversation = conversations[0]
        assert isinstance(conversation, Conversation)
        assert conversation.id.startswith("pdm2_defense_")
        assert len(conversation.messages) > 0
        assert conversation.context["type"] == "defense_assessment"

    def test_conversation_generation_pattern(self):
        """Test conversation generation for psychodynamic patterns."""
        conversations = self.parser.generate_conversation_templates(
            "pattern", "Anxious-Attachment Pattern"
        )

        assert len(conversations) > 0

        conversation = conversations[0]
        assert isinstance(conversation, Conversation)
        assert conversation.id.startswith("pdm2_pattern_")
        assert len(conversation.messages) > 0
        assert conversation.context["type"] == "psychodynamic_assessment"

    def test_statistics(self):
        """Test statistics generation."""
        stats = self.parser.get_statistics()

        assert "total_attachment_patterns" in stats
        assert "total_defense_mechanisms" in stats
        assert "total_psychodynamic_patterns" in stats
        assert "defense_mechanisms_by_level" in stats
        assert "psychodynamic_patterns_by_domain" in stats
        assert "attachment_styles" in stats
        assert "version" in stats

        assert stats["total_attachment_patterns"] > 0
        assert stats["total_defense_mechanisms"] > 0
        assert stats["total_psychodynamic_patterns"] > 0

        # Check level breakdown
        level_counts = stats["defense_mechanisms_by_level"]
        assert "mature" in level_counts
        assert "pathological" in level_counts

        # Check domain breakdown
        domain_counts = stats["psychodynamic_patterns_by_domain"]
        assert len(domain_counts) > 0

    def test_empty_knowledge_base_handling(self):
        """Test handling of empty knowledge base."""
        parser = PDM2Parser()
        parser.knowledge_base = None

        assert parser.get_attachment_patterns() == []
        assert parser.get_defense_mechanisms() == []
        assert parser.get_psychodynamic_patterns() == []
        assert parser.get_attachment_pattern_by_style(AttachmentStyle.SECURE) is None
        assert parser.get_defense_mechanism_by_name("Sublimation") is None
        assert parser.get_statistics() == {}


class TestPDM2DataStructures:
    """Test cases for PDM-2 data structures."""

    def test_attachment_pattern_creation(self):
        """Test AttachmentPattern creation."""
        pattern = AttachmentPattern(
            style=AttachmentStyle.SECURE,
            name="Test Secure",
            description="Test description",
            characteristics=["characteristic1", "characteristic2"],
            behavioral_indicators=["indicator1"]
        )

        assert pattern.style == AttachmentStyle.SECURE
        assert pattern.name == "Test Secure"
        assert len(pattern.characteristics) == 2
        assert len(pattern.behavioral_indicators) == 1

    def test_defense_mechanism_creation(self):
        """Test DefenseMechanism creation."""
        mechanism = DefenseMechanism(
            name="Test Defense",
            level=DefenseMechanismLevel.MATURE,
            description="Test description",
            function="Test function",
            examples=["example1", "example2"]
        )

        assert mechanism.name == "Test Defense"
        assert mechanism.level == DefenseMechanismLevel.MATURE
        assert len(mechanism.examples) == 2

    def test_psychodynamic_pattern_creation(self):
        """Test PsychodynamicPattern creation."""
        pattern = PsychodynamicPattern(
            name="Test Pattern",
            domain=PsychodynamicDomain.ATTACHMENT_RELATIONSHIPS,
            description="Test description",
            core_features=["feature1", "feature2"],
            unconscious_conflicts=["conflict1"]
        )

        assert pattern.name == "Test Pattern"
        assert pattern.domain == PsychodynamicDomain.ATTACHMENT_RELATIONSHIPS
        assert len(pattern.core_features) == 2
        assert len(pattern.unconscious_conflicts) == 1


if __name__ == "__main__":
    pytest.main([__file__])
