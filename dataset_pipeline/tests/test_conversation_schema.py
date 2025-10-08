"""
Unit tests for the conversation schema.
"""

import unittest

from ai.dataset_pipeline.conversation_schema import Conversation, Message


class TestConversationSchema(unittest.TestCase):

    def test_message_creation(self):
        """Test basic Message object creation."""
        msg = Message(role="user", content="Hello")
        assert msg.role == "user"
        assert msg.content == "Hello"
        assert msg.timestamp is not None

    def test_conversation_creation(self):
        """Test basic Conversation object creation."""
        conv = Conversation(source="test_source")
        assert conv.conversation_id is not None
        assert conv.source == "test_source"
        assert len(conv.messages) == 0

    def test_add_message(self):
        """Test adding a message to a conversation."""
        conv = Conversation()
        conv.add_message(role="assistant", content="How can I help?")
        assert len(conv.messages) == 1
        assert conv.messages[0].role == "assistant"

    def test_to_dict_serialization(self):
        """Test serialization of Conversation to dict."""
        conv = Conversation(source="serialization_test")
        conv.add_message(role="user", content="Test message")
        conv_dict = conv.to_dict()
        assert conv_dict["source"] == "serialization_test"
        assert len(conv_dict["messages"]) == 1
        assert conv_dict["messages"][0]["content"] == "Test message"

    def test_from_dict_deserialization(self):
        """Test deserialization from dict to Conversation."""
        data = {
            "conversation_id": "test-123",
            "source": "deserialization_test",
            "messages": [{"role": "user", "content": "Hello from dict"}]
        }
        conv = Conversation.from_dict(data)
        assert conv.conversation_id == "test-123"
        assert len(conv.messages) == 1
        assert conv.messages[0].content == "Hello from dict"

if __name__ == "__main__":
    unittest.main()
