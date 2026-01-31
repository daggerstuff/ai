"""
Unit tests for psychology knowledge converter.

Tests the comprehensive psychology knowledge conversion functionality
including DSM-5, PDM-2, and Big Five knowledge conversion to therapeutic conversations.
"""

import json
import tempfile
import unittest
from pathlib import Path

from .conversation_schema import Conversation, Message
from .psychology_knowledge_converter import (
    ConversationStyle,
    ConversationTemplate,
    ConversationType,
    PsychologyKnowledgeConverter,
)


class TestPsychologyKnowledgeConverter(unittest.TestCase):
    """Test psychology knowledge converter functionality."""

    def setUp(self):
        """Set up test fixtures."""
        self.converter = PsychologyKnowledgeConverter()

    def test_initialization(self):
        """Test converter initialization."""
        assert self.converter.dsm5_parser is not None
        assert self.converter.pdm2_parser is not None
        assert self.converter.big_five_processor is not None
        assert self.converter.conversation_templates is not None

        # Check that templates were created
        assert "dsm5_diagnostic" in self.converter.conversation_templates
        assert "big_five_personality" in self.converter.conversation_templates

    def test_conversation_types_enum(self):
        """Test ConversationType enum values."""
        expected_types = {
            "diagnostic_assessment", "personality_exploration", "psychodynamic_exploration",
            "clinical_interview", "therapeutic_education", "symptom_assessment"
        }
        actual_types = {conv_type.value for conv_type in ConversationType}
        assert expected_types == actual_types

    def test_conversation_styles_enum(self):
        """Test ConversationStyle enum values."""
        expected_styles = {
            "structured_interview", "exploratory_dialogue", "educational_discussion",
            "assessment_focused", "supportive_inquiry"
        }
        actual_styles = {style.value for style in ConversationStyle}
        assert expected_styles == actual_styles

    def test_dsm5_template_creation(self):
        """Test DSM-5 conversation template creation."""
        dsm5_templates = self.converter._create_dsm5_templates()

        assert len(dsm5_templates) > 0

        for template in dsm5_templates:
            assert isinstance(template, ConversationTemplate)
            assert template.knowledge_source == "DSM-5"
            assert isinstance(template.learning_objectives, list)
            assert isinstance(template.clinical_focus, list)

    def test_big_five_template_creation(self):
        """Test Big Five conversation template creation."""
        big_five_templates = self.converter._create_big_five_templates()

        assert len(big_five_templates) > 0

        for template in big_five_templates:
            assert isinstance(template, ConversationTemplate)
            assert template.knowledge_source == "Big Five"
            assert isinstance(template.learning_objectives, list)
            assert isinstance(template.clinical_focus, list)

    def test_convert_dsm5_to_conversations(self):
        """Test DSM-5 knowledge conversion to conversations."""
        conversations = self.converter.convert_dsm5_to_conversations(count=2)

        assert len(conversations) > 0

        for conversation in conversations:
            assert isinstance(conversation, Conversation)
            assert isinstance(conversation.messages, list)
            assert len(conversation.messages) > 0
            assert conversation.source == "psychology_knowledge_converter"
            assert "knowledge_source" in conversation.context
            assert conversation.context["knowledge_source"] == "DSM-5"

            # Check messages structure
            for message in conversation.messages:
                assert isinstance(message, Message)
                assert message.role in ["therapist", "client"]
                assert isinstance(message.content, str)
                assert len(message.content) > 0
                assert isinstance(message.meta, dict)

    def test_convert_big_five_to_conversations(self):
        """Test Big Five knowledge conversion to conversations."""
        conversations = self.converter.convert_big_five_to_conversations(count=2)

        assert len(conversations) > 0

        for conversation in conversations:
            assert isinstance(conversation, Conversation)
            assert isinstance(conversation.messages, list)
            assert len(conversation.messages) > 0
            assert conversation.source == "psychology_knowledge_converter"
            assert "knowledge_source" in conversation.context
            assert conversation.context["knowledge_source"] == "Big Five"

    def test_dsm5_diagnostic_conversation_structure(self):
        """Test DSM-5 diagnostic conversation structure."""
        # Get a disorder for testing
        disorders = self.converter.dsm5_parser.get_disorders()
        assert len(disorders) > 0

        disorder = disorders[0]
        conversation = self.converter._generate_dsm5_diagnostic_conversation(disorder)

        # Check conversation structure
        assert isinstance(conversation, Conversation)
        assert "dsm5_diagnostic" in conversation.id
        assert conversation.source == "psychology_knowledge_converter"

        # Check context
        assert conversation.context["disorder"] == disorder.name
        assert conversation.context["knowledge_source"] == "DSM-5"
        assert conversation.context["conversation_type"] == "diagnostic_assessment"

        # Check messages
        assert len(conversation.messages) > 0

        # First message should be therapist introduction
        first_message = conversation.messages[0]
        assert first_message.role == "therapist"
        assert "diagnostic" in first_message.meta.get("type", "")

        # Should have systematic criterion assessment
        criterion_messages = [msg for msg in conversation.messages if "criterion_id" in msg.meta]
        assert len(criterion_messages) > 0

    def test_big_five_exploration_conversation_structure(self):
        """Test Big Five exploration conversation structure."""
        # Get a personality profile for testing
        profiles = self.converter.big_five_processor.get_personality_profiles()
        assert len(profiles) > 0

        profile = profiles[0]
        conversation = self.converter._generate_big_five_exploration_conversation(profile)

        # Check conversation structure
        assert isinstance(conversation, Conversation)
        assert "big_five_exploration" in conversation.id
        assert conversation.source == "psychology_knowledge_converter"

        # Check context
        assert conversation.context["personality_factor"] == profile.factor.value
        assert conversation.context["knowledge_source"] == "Big Five"
        assert conversation.context["conversation_type"] == "personality_exploration"

        # Check messages
        assert len(conversation.messages) > 0

        # First message should be therapist introduction
        first_message = conversation.messages[0]
        assert first_message.role == "therapist"
        assert "personality" in first_message.meta.get("type", "")

    def test_generate_comprehensive_dataset(self):
        """Test comprehensive dataset generation."""
        conversations = self.converter.generate_comprehensive_dataset(
            dsm5_count=2,
            big_five_count=2
        )

        assert len(conversations) > 0

        # Check knowledge source diversity
        knowledge_sources = set()
        for conversation in conversations:
            source = conversation.context.get("knowledge_source")
            if source:
                knowledge_sources.add(source)

        assert "DSM-5" in knowledge_sources
        assert "Big Five" in knowledge_sources

    def test_export_conversations_to_json(self):
        """Test exporting conversations to JSON."""
        # Generate test conversations
        conversations = self.converter.convert_dsm5_to_conversations(count=1)

        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "test_conversations.json"

            # Test successful export
            result = self.converter.export_conversations_to_json(conversations, output_path)
            assert result
            assert output_path.exists()

            # Verify exported content
            with open(output_path, encoding="utf-8") as f:
                exported_data = json.load(f)

            assert "conversations" in exported_data
            assert "metadata" in exported_data
            assert exported_data["metadata"]["total_conversations"] == len(conversations)
            assert "knowledge_sources" in exported_data["metadata"]

            # Check conversation structure
            assert len(exported_data["conversations"]) == len(conversations)
            for conv_data in exported_data["conversations"]:
                assert "id" in conv_data
                assert "messages" in conv_data
                assert "context" in conv_data
                assert "source" in conv_data

    def test_get_statistics(self):
        """Test conversation statistics generation."""
        conversations = self.converter.generate_comprehensive_dataset(
            dsm5_count=2,
            big_five_count=2
        )

        stats = self.converter.get_statistics(conversations)

        expected_keys = {
            "total_conversations", "knowledge_sources", "conversation_types",
            "average_messages_per_conversation", "total_messages"
        }
        assert set(stats.keys()) == expected_keys

        assert stats["total_conversations"] == len(conversations)
        assert stats["total_messages"] > 0
        assert stats["average_messages_per_conversation"] > 0
        assert isinstance(stats["knowledge_sources"], dict)
        assert isinstance(stats["conversation_types"], dict)

    def test_get_statistics_empty_list(self):
        """Test statistics with empty conversation list."""
        stats = self.converter.get_statistics([])
        assert stats == {}

    def test_conversation_message_quality(self):
        """Test quality of generated conversation messages."""
        conversations = self.converter.convert_dsm5_to_conversations(count=1)

        for conversation in conversations:
            for message in conversation.messages:
                # Check message content quality
                assert isinstance(message.content, str)
                assert len(message.content.strip()) > 10  # Reasonable length
                assert not message.content.startswith(" ")  # No leading spaces
                assert not message.content.endswith(" ")  # No trailing spaces

                # Check metadata presence
                assert isinstance(message.meta, dict)
                assert "knowledge_source" in message.meta

                # Role-specific checks
                if message.role == "therapist":
                    # Therapist messages should be professional
                    assert "I don't know" not in message.content.lower()
                elif message.role == "client":
                    # Client messages should be personal
                    assert any(word in message.content.lower() for word in ["i", "me", "my", "i'm", "i've"])

    def test_knowledge_integration_metadata(self):
        """Test that conversations properly integrate knowledge metadata."""
        conversations = self.converter.convert_dsm5_to_conversations(count=1)

        for conversation in conversations:
            # Check conversation-level metadata
            assert "learning_objectives" in conversation.meta
            assert isinstance(conversation.meta["learning_objectives"], list)
            assert len(conversation.meta["learning_objectives"]) > 0

            # Check message-level knowledge integration
            knowledge_messages = [msg for msg in conversation.messages if "knowledge_source" in msg.meta]
            assert len(knowledge_messages) > 0

            for msg in knowledge_messages:
                assert msg.meta["knowledge_source"] == "DSM-5"


if __name__ == "__main__":
    unittest.main()
