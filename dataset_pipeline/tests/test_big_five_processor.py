"""
Unit tests for Big Five personality assessment processor.

Tests the comprehensive Big Five personality assessment processing functionality
including personality profiles, assessments, clinical guidelines, and
conversation generation.
"""

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from ai.dataset_pipeline.processing.big_five_processor import (
    AssessmentItem,
    AssessmentType,
    BigFiveProcessor,
    PersonalityFacet,
    PersonalityFactor,
)
from ai.dataset_pipeline.schemas.conversation_schema import Conversation, Message


class TestBigFiveProcessor(unittest.TestCase):
    """Test Big Five processor functionality."""

    def setUp(self):
        """Set up test fixtures."""
        self.processor = BigFiveProcessor()

    def test_initialization(self):
        """Test processor initialization."""
        assert self.processor.knowledge_base is not None
        assert len(self.processor.knowledge_base.personality_profiles) == 5
        assert len(self.processor.knowledge_base.assessments) == 2
        assert (
            "assessment_best_practices"
            in self.processor.knowledge_base.clinical_guidelines
        )
        assert (
            "mental_health_correlations"
            in self.processor.knowledge_base.research_findings
        )

    def test_personality_factors_enum(self):
        """Test PersonalityFactor enum values."""
        expected_factors = {
            "openness", "conscientiousness", "extraversion",
            "agreeableness", "neuroticism"
        }
        actual_factors = {factor.value for factor in PersonalityFactor}
        assert expected_factors == actual_factors

    def test_assessment_types_enum(self):
        """Test AssessmentType enum values."""
        expected_types = {
            "neo_pi_r", "big_five_inventory", "ten_item_personality_inventory",
            "big_five_inventory_2", "international_personality_item_pool"
        }
        actual_types = {assessment_type.value for assessment_type in AssessmentType}
        assert expected_types == actual_types

    def test_get_personality_profiles(self):
        """Test getting all personality profiles."""
        profiles = self.processor.get_personality_profiles()
        assert len(profiles) == 5

        # Check that all Big Five factors are represented
        factors = {profile.factor for profile in profiles}
        expected_factors = set(PersonalityFactor)
        assert factors == expected_factors

    def test_get_profile_by_factor(self):
        """Test getting specific personality profile by factor."""
        # Test valid factor
        openness_profile = self.processor.get_profile_by_factor(
            PersonalityFactor.OPENNESS
        )
        assert openness_profile is not None
        assert openness_profile.factor == PersonalityFactor.OPENNESS
        assert openness_profile.name == "Openness to Experience"

        # Test with empty knowledge base
        empty_processor = BigFiveProcessor()
        empty_processor.knowledge_base = None
        result = empty_processor.get_profile_by_factor(PersonalityFactor.OPENNESS)
        assert result is None

    def test_get_assessments(self):
        """Test getting all assessments."""
        assessments = self.processor.get_assessments()
        assert len(assessments) == 2

        # Check assessment types
        assessment_types = {assessment.type for assessment in assessments}
        expected_types = {AssessmentType.BFI, AssessmentType.TIPI}
        assert assessment_types == expected_types

    def test_get_assessment_by_type(self):
        """Test getting specific assessment by type."""
        # Test valid type
        bfi_assessment = self.processor.get_assessment_by_type(AssessmentType.BFI)
        assert bfi_assessment is not None
        assert bfi_assessment.type == AssessmentType.BFI
        assert bfi_assessment.name == "Big Five Inventory (BFI)"

        # Test with empty knowledge base
        empty_processor = BigFiveProcessor()
        empty_processor.knowledge_base = None
        result = empty_processor.get_assessment_by_type(AssessmentType.BFI)
        assert result is None

    def test_personality_profile_structure(self):
        """Test personality profile data structure."""
        openness_profile = self.processor.get_profile_by_factor(
            PersonalityFactor.OPENNESS
        )

        # Check required fields
        assert isinstance(openness_profile.factor, PersonalityFactor)
        assert isinstance(openness_profile.name, str)
        assert isinstance(openness_profile.description, str)
        assert isinstance(openness_profile.facets, list)
        assert isinstance(openness_profile.score_interpretations, dict)
        assert isinstance(openness_profile.clinical_considerations, list)
        assert isinstance(openness_profile.therapeutic_implications, list)

        # Check facets structure
        assert len(openness_profile.facets) > 0
        for facet in openness_profile.facets:
            assert isinstance(facet, PersonalityFacet)
            assert facet.factor == PersonalityFactor.OPENNESS
            assert isinstance(facet.name, str)
            assert isinstance(facet.description, str)

        # Check score interpretations
        assert "high" in openness_profile.score_interpretations
        assert "low" in openness_profile.score_interpretations

    def test_assessment_structure(self):
        """Test assessment data structure."""
        bfi_assessment = self.processor.get_assessment_by_type(AssessmentType.BFI)

        # Check required fields
        assert isinstance(bfi_assessment.name, str)
        assert isinstance(bfi_assessment.type, AssessmentType)
        assert isinstance(bfi_assessment.description, str)
        assert isinstance(bfi_assessment.items, list)
        assert isinstance(bfi_assessment.administration_time, str)
        assert isinstance(bfi_assessment.reliability_data, dict)
        assert isinstance(bfi_assessment.scoring_guidelines, list)
        assert isinstance(bfi_assessment.clinical_applications, list)

        # Check items structure
        assert len(bfi_assessment.items) > 0
        for item in bfi_assessment.items:
            assert isinstance(item, AssessmentItem)
            assert isinstance(item.id, str)
            assert isinstance(item.text, str)
            assert isinstance(item.factor, PersonalityFactor)
            assert isinstance(item.reverse_scored, bool)

        # Check reliability data
        assert len(bfi_assessment.reliability_data) > 0
        for _factor, reliability in bfi_assessment.reliability_data.items():
            assert isinstance(reliability, float)
            assert reliability >= 0.0
            assert reliability <= 1.0

    def test_clinical_guidelines_structure(self):
        """Test clinical guidelines data structure."""
        guidelines = self.processor.knowledge_base.clinical_guidelines

        expected_categories = {
            "assessment_best_practices",
            "interpretation_guidelines",
            "therapeutic_applications",
            "ethical_considerations"
        }

        assert set(guidelines.keys()) == expected_categories

        for _category, items in guidelines.items():
            assert isinstance(items, list)
            assert len(items) > 0
            for item in items:
                assert isinstance(item, str)
                assert len(item) > 0

    def test_research_findings_structure(self):
        """Test research findings data structure."""
        findings = self.processor.knowledge_base.research_findings

        expected_categories = {
            "mental_health_correlations",
            "treatment_outcomes",
            "developmental_patterns",
            "cultural_considerations"
        }

        assert set(findings.keys()) == expected_categories

        for _category, items in findings.items():
            assert isinstance(items, list)
            assert len(items) > 0
            for item in items:
                assert isinstance(item, str)
                assert len(item) > 0

    def test_generate_conversation_templates(self):
        """Test conversation template generation."""
        conversations = self.processor.generate_conversation_templates(
            PersonalityFactor.OPENNESS
        )

        assert len(conversations) == 2  # Assessment and therapeutic conversations

        for conversation in conversations:
            assert isinstance(conversation, Conversation)
            assert isinstance(conversation.conversation_id, str)
            assert isinstance(conversation.messages, list)
            assert len(conversation.messages) > 0
            assert isinstance(conversation.metadata, dict)
            assert conversation.source == "big_five_processor"

            # Check metadata
            assert "factor" in conversation.metadata
            assert conversation.metadata["factor"] == "openness"
            assert "type" in conversation.metadata

            # Check messages
            for message in conversation.messages:
                assert isinstance(message, Message)
                assert message.role in ["therapist", "client"]
                assert isinstance(message.content, str)
                assert len(message.content) > 0

    def test_generate_conversation_templates_invalid_factor(self):
        """Test conversation generation with invalid factor."""
        # Mock get_profile_by_factor to return None
        with patch.object(self.processor, "get_profile_by_factor", return_value=None):
            conversations = self.processor.generate_conversation_templates(
                PersonalityFactor.OPENNESS
            )
            assert len(conversations) == 0

    def test_export_to_json(self):
        """Test exporting knowledge base to JSON."""
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "big_five_export.json"

            # Test successful export
            result = self.processor.export_to_json(output_path)
            assert result
            assert output_path.exists()

            # Verify exported content
            with open(output_path, encoding="utf-8") as f:
                exported_data = json.load(f)

            assert "version" in exported_data
            assert "personality_profiles" in exported_data
            assert "assessments" in exported_data
            assert "clinical_guidelines" in exported_data
            assert "research_findings" in exported_data

            # Check personality profiles
            assert len(exported_data["personality_profiles"]) == 5
            for profile in exported_data["personality_profiles"]:
                assert "factor" in profile
                assert "name" in profile
                assert "facets" in profile

            # Check assessments
            assert len(exported_data["assessments"]) == 2
            for assessment in exported_data["assessments"]:
                assert "type" in assessment
                assert "name" in assessment
                assert "items" in assessment

    def test_export_to_json_no_knowledge_base(self):
        """Test export with no knowledge base."""
        empty_processor = BigFiveProcessor()
        empty_processor.knowledge_base = None

        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "empty_export.json"
            result = empty_processor.export_to_json(output_path)
            assert not result
            assert not output_path.exists()

    def test_get_statistics(self):
        """Test getting knowledge base statistics."""
        stats = self.processor.get_statistics()

        expected_keys = {
            "total_profiles", "total_assessments", "factors_covered",
            "assessment_types", "total_facets", "total_assessment_items", "version"
        }
        assert set(stats.keys()) == expected_keys

        assert stats["total_profiles"] == 5
        assert stats["total_assessments"] == 2
        assert len(stats["factors_covered"]) == 5
        assert len(stats["assessment_types"]) == 2
        assert stats["total_facets"] > 0
        assert stats["total_assessment_items"] > 0
        assert isinstance(stats["version"], str)

    def test_get_statistics_no_knowledge_base(self):
        """Test statistics with no knowledge base."""
        empty_processor = BigFiveProcessor()
        empty_processor.knowledge_base = None

        stats = empty_processor.get_statistics()
        assert stats == {}


if __name__ == "__main__":
    unittest.main()
