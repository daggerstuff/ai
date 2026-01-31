"""
Tests for Dataset Format Adapters

Tests all adapter implementations and the adapter registry.
"""

import pytest
from dataset_format_adapters import (
    Psych8kAdapter,
    MentalHealthCounselingAdapter,
    SoulChatAdapter,
    CounselChatAdapter,
    LLAMA3MentalAdapter,
    TherapistSFTAdapter,
    NeuroQAAdapter,
    GenericChatMLAdapter,
    AdapterRegistry,
)


class TestPsych8kAdapter:
    """Test Psych8k adapter."""
    
    def test_adapt_valid_data(self):
        adapter = Psych8kAdapter()
        raw_data = {
            "Context": "I'm feeling anxious about work",
            "Response": "It's understandable to feel that way. Let's explore what specifically is causing your anxiety.",
        }
        
        conversation = adapter.adapt(raw_data)
        
        assert conversation is not None
        assert len(conversation.messages) == 2
        assert conversation.messages[0].role == "user"
        assert conversation.messages[0].content == "I'm feeling anxious about work"
        assert conversation.messages[1].role == "assistant"
        assert conversation.metadata["source"] == "psych8k"
    
    def test_adapt_missing_context(self):
        adapter = Psych8kAdapter()
        raw_data = {"Response": "Response text"}
        
        conversation = adapter.adapt(raw_data)
        assert conversation is None
    
    def test_adapt_missing_response(self):
        adapter = Psych8kAdapter()
        raw_data = {"Context": "Context text"}
        
        conversation = adapter.adapt(raw_data)
        assert conversation is None
    
    def test_format_name(self):
        adapter = Psych8kAdapter()
        assert adapter.get_format_name() == "psych8k"


class TestMentalHealthCounselingAdapter:
    """Test mental health counseling adapter."""
    
    def test_adapt_with_title_and_text(self):
        adapter = MentalHealthCounselingAdapter()
        raw_data = {
            "questionTitle": "Anxiety Help",
            "questionText": "I have severe anxiety. What should I do?",
            "answerText": "Anxiety is treatable. I recommend starting with deep breathing exercises.",
        }
        
        conversation = adapter.adapt(raw_data)
        
        assert conversation is not None
        assert len(conversation.messages) == 2
        assert "Anxiety Help" in conversation.messages[0].content
        assert "I have severe anxiety" in conversation.messages[0].content
        assert conversation.metadata["source"] == "mental_health_counseling"
        assert conversation.metadata["question_title"] == "Anxiety Help"
    
    def test_adapt_without_title(self):
        adapter = MentalHealthCounselingAdapter()
        raw_data = {
            "questionText": "How do I cope with stress?",
            "answerText": "Here are some coping strategies...",
        }
        
        conversation = adapter.adapt(raw_data)
        
        assert conversation is not None
        assert "How do I cope with stress?" in conversation.messages[0].content
    
    def test_format_name(self):
        adapter = MentalHealthCounselingAdapter()
        assert adapter.get_format_name() == "mental_health_counseling"


class TestSoulChatAdapter:
    """Test SoulChat adapter."""
    
    def test_adapt_with_input_context(self):
        adapter = SoulChatAdapter()
        raw_data = {
            "instruction": "I need help with depression",
            "input": "I've been feeling down for weeks",
            "output": "I understand. Depression is serious and I'm here to help.",
        }
        
        conversation = adapter.adapt(raw_data)
        
        assert conversation is not None
        assert "I need help with depression" in conversation.messages[0].content
        assert "I've been feeling down" in conversation.messages[0].content
        assert conversation.metadata["format"] == "psychological_counselor"
    
    def test_adapt_without_input(self):
        adapter = SoulChatAdapter()
        raw_data = {
            "instruction": "What is cognitive behavioral therapy?",
            "output": "CBT is a type of psychotherapy...",
        }
        
        conversation = adapter.adapt(raw_data)
        
        assert conversation is not None
        assert conversation.messages[0].content == "What is cognitive behavioral therapy?"


class TestCounselChatAdapter:
    """Test counsel-chat adapter."""
    
    def test_adapt_with_topic(self):
        adapter = CounselChatAdapter()
        raw_data = {
            "questionText": "How do I deal with relationship issues?",
            "answerText": "Relationships require open communication...",
            "topic": "relationships",
        }
        
        conversation = adapter.adapt(raw_data)
        
        assert conversation is not None
        assert conversation.metadata["topic"] == "relationships"
        assert conversation.metadata["format"] == "professional_counseling"


class TestLLAMA3MentalAdapter:
    """Test LLAMA3 mental adapter."""
    
    def test_adapt_conversations_format(self):
        adapter = LLAMA3MentalAdapter()
        raw_data = {
            "conversations": [
                {"from": "human", "value": "I'm struggling with anxiety"},
                {"from": "gpt", "value": "I understand. Let's work through this together."},
                {"from": "human", "value": "Thank you"},
                {"from": "gpt", "value": "You're welcome. Remember to practice self-care."},
            ]
        }
        
        conversation = adapter.adapt(raw_data)
        
        assert conversation is not None
        assert len(conversation.messages) == 4
        assert conversation.messages[0].role == "user"
        assert conversation.messages[1].role == "assistant"
        assert conversation.metadata["source"] == "llama3_mental"
    
    def test_adapt_empty_conversations(self):
        adapter = LLAMA3MentalAdapter()
        raw_data = {"conversations": []}
        
        conversation = adapter.adapt(raw_data)
        assert conversation is None
    
    def test_adapt_single_message(self):
        adapter = LLAMA3MentalAdapter()
        raw_data = {
            "conversations": [
                {"from": "human", "value": "Hello"},
            ]
        }
        
        conversation = adapter.adapt(raw_data)
        assert conversation is None  # Need at least 2 messages


class TestTherapistSFTAdapter:
    """Test therapist SFT adapter."""
    
    def test_adapt_valid_data(self):
        adapter = TherapistSFTAdapter()
        raw_data = {
            "input": "Client: I can't sleep at night",
            "output": "Therapist: Let's explore what might be keeping you awake.",
        }
        
        conversation = adapter.adapt(raw_data)
        
        assert conversation is not None
        assert conversation.metadata["format"] == "supervised_fine_tuning"


class TestNeuroQAAdapter:
    """Test neuro QA adapter."""
    
    def test_adapt_valid_qa(self):
        adapter = NeuroQAAdapter()
        raw_data = {
            "question": "What causes depression?",
            "answer": "Depression can be caused by various factors...",
        }
        
        conversation = adapter.adapt(raw_data)
        
        assert conversation is not None
        assert conversation.metadata["format"] == "neurology_psychology_qa"
        assert conversation.messages[0].content == "What causes depression?"


class TestGenericChatMLAdapter:
    """Test generic ChatML adapter."""
    
    def test_adapt_standard_format(self):
        adapter = GenericChatMLAdapter()
        raw_data = {
            "messages": [
                {"role": "user", "content": "Hello"},
                {"role": "assistant", "content": "Hi! How can I help?"},
            ]
        }
        
        conversation = adapter.adapt(raw_data)
        
        assert conversation is not None
        assert len(conversation.messages) == 2
        assert conversation.metadata["format"] == "generic_chatml"
    
    def test_adapt_with_system_message(self):
        adapter = GenericChatMLAdapter()
        raw_data = {
            "messages": [
                {"role": "system", "content": "You are a helpful therapist"},
                {"role": "user", "content": "I need help"},
                {"role": "assistant", "content": "I'm here to help"},
            ]
        }
        
        conversation = adapter.adapt(raw_data)
        
        assert conversation is not None
        assert len(conversation.messages) == 3
    
    def test_adapt_invalid_messages(self):
        adapter = GenericChatMLAdapter()
        raw_data = {
            "messages": [
                {"role": "invalid", "content": "test"},
            ]
        }
        
        conversation = adapter.adapt(raw_data)
        assert conversation is None  # Need at least 2 valid messages


class TestAdapterRegistry:
    """Test adapter registry."""
    
    def test_default_adapters_registered(self):
        registry = AdapterRegistry()
        
        adapters = registry.list_adapters()
        
        assert "psych8k" in adapters
        assert "mental_health_counseling" in adapters
        assert "soulchat" in adapters
        assert "counsel_chat" in adapters
        assert "llama3_mental" in adapters
        assert "therapist_sft" in adapters
        assert "neuro_qa" in adapters
        assert "generic_chatml" in adapters
    
    def test_get_adapter(self):
        registry = AdapterRegistry()
        
        adapter = registry.get_adapter("psych8k")
        
        assert adapter is not None
        assert isinstance(adapter, Psych8kAdapter)
    
    def test_get_nonexistent_adapter(self):
        registry = AdapterRegistry()
        
        adapter = registry.get_adapter("nonexistent")
        
        assert adapter is None
    
    def test_adapt_with_fallback_success(self):
        registry = AdapterRegistry()
        raw_data = {
            "Context": "I'm anxious",
            "Response": "Let's talk about it",
        }
        
        conversation = registry.adapt_with_fallback(
            raw_data,
            "psych8k",
            "test_source",
        )
        
        assert conversation is not None
        assert conversation.metadata["source"] == "psych8k"
    
    def test_adapt_with_fallback_uses_generic(self):
        registry = AdapterRegistry()
        raw_data = {
            "messages": [
                {"role": "user", "content": "Hello"},
                {"role": "assistant", "content": "Hi"},
            ],
            "source": "test",
        }
        
        # Try with psych8k format (should fail and fallback to generic)
        conversation = registry.adapt_with_fallback(
            raw_data,
            "psych8k",
            "fallback_test",
        )
        
        assert conversation is not None
        assert conversation.metadata["format"] == "generic_chatml"
        assert conversation.metadata["source"] == "fallback_test"
    
    def test_adapt_with_fallback_both_fail(self):
        registry = AdapterRegistry()
        raw_data = {
            "invalid": "data",
        }
        
        conversation = registry.adapt_with_fallback(
            raw_data,
            "psych8k",
            "test",
        )
        
        assert conversation is None
    
    def test_register_custom_adapter(self):
        registry = AdapterRegistry()
        
        class CustomAdapter(Psych8kAdapter):
            def get_format_name(self):
                return "custom_format"
        
        custom_adapter = CustomAdapter()
        registry.register_adapter(custom_adapter)
        
        assert "custom_format" in registry.list_adapters()
        retrieved = registry.get_adapter("custom_format")
        assert retrieved is custom_adapter


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
