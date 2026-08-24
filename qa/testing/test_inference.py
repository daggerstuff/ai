"""Unit tests for the inference pipeline (LLMProvider, MockLLMProvider,
OpenAIProvider, and InferenceEngine)."""

from unittest.mock import patch

import pytest

from ai.pipelines.model_training.core.inference import (
    InferenceEngine,
    LLMProvider,
    MockLLMProvider,
    OpenAIProvider,
)


class TestLLMProvider:
    """LLMProvider base class contract."""

    def test_generate_not_implemented(self):
        provider = LLMProvider()
        with pytest.raises(NotImplementedError):
            provider.generate([])


class TestMockLLMProvider:
    """MockLLMProvider provides canned responses."""

    def test_generate_returns_mock_response(self):
        provider = MockLLMProvider()
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Tell me about chest pain."},
        ]
        result = provider.generate(messages)
        assert result.startswith("Mock response to: ")
        assert "chest" in result

    def test_generate_with_no_user_message(self):
        provider = MockLLMProvider()
        result = provider.generate([{"role": "system", "content": "You are a bot."}])
        assert result == "Mock response to: ..."


class TestOpenAIProvider:
    """OpenAIProvider wraps the openai>=1.0.0 client."""

    def test_instantiate_without_api_key(self):
        """Can be constructed without any credentials — no crash."""
        provider = OpenAIProvider()
        assert provider.api_key is None
        assert provider.model == "gpt-4"
        assert provider.base_url is None
        assert provider._client is None  # lazy init

    def test_instantiate_with_custom_params(self):
        provider = OpenAIProvider(
            api_key="sk-test",
            model="gpt-4o",
            base_url="https://api.openai.com/v1",
        )
        assert provider.api_key == "sk-test"
        assert provider.model == "gpt-4o"
        assert provider.base_url == "https://api.openai.com/v1"

    def test_generate_without_api_key_returns_message(self):
        provider = OpenAIProvider()
        messages = [
            {"role": "system", "content": "You are a bot."},
            {"role": "user", "content": "Hello."},
        ]
        result = provider.generate(messages)
        assert result == "OpenAI API Key not set. Please provide it."

    @patch("openai.OpenAI")
    def test_generate_calls_client_with_messages(self, mock_openai):
        mock_instance = mock_openai.return_value
        mock_choice = mock_instance.chat.completions.create.return_value.choices[0]
        mock_choice.message.content = "Hello, how can I help?"

        provider = OpenAIProvider(api_key="sk-test")
        messages = [
            {"role": "system", "content": "You are a bot."},
            {"role": "user", "content": "Hi."},
        ]
        result = provider.generate(messages)

        assert result == "Hello, how can I help?"
        mock_openai.assert_called_once_with(api_key="sk-test", base_url=None)
        mock_instance.chat.completions.create.assert_called_once_with(model="gpt-4", messages=messages)

    @patch("openai.OpenAI")
    def test_lazy_client_reuses_instance(self, mock_openai):
        provider = OpenAIProvider(api_key="sk-test")
        c1 = provider._get_client()
        c2 = provider._get_client()
        assert c1 is c2  # same object
        mock_openai.assert_called_once()

    @patch("openai.OpenAI")
    def test_generate_with_base_url(self, mock_openai):
        mock_instance = mock_openai.return_value
        mock_choice = mock_instance.chat.completions.create.return_value.choices[0]
        mock_choice.message.content = "Response"

        provider = OpenAIProvider(api_key="sk-test", base_url="https://nim.example.com/v1")
        provider.generate([{"role": "user", "content": "Hi"}])

        mock_openai.assert_called_once_with(api_key="sk-test", base_url="https://nim.example.com/v1")

    @patch("openai.OpenAI")
    def test_generate_empty_content(self, mock_openai):
        mock_instance = mock_openai.return_value
        mock_choice = mock_instance.chat.completions.create.return_value.choices[0]
        mock_choice.message.content = None

        provider = OpenAIProvider(api_key="sk-test")
        result = provider.generate([{"role": "user", "content": "Hi"}])
        assert result == ""


class TestInferenceEngine:
    """InferenceEngine orchestrates the prompt building and provider call."""

    def test_build_system_prompt_contains_do_not_break_character(self):
        """S4.3 requirement: explicit DO NOT BREAK CHARACTER scaffolding."""
        provider = MockLLMProvider()
        engine = InferenceEngine(provider)

        persona = {
            "metadata": {"name": "John Doe", "role": "patient"},
            "clinical_profile": {
                "condition": "Chest Pain",
                "symptoms": [{"name": "chest_pain", "severity": 7}],
                "medical_history": "Hypertension",
                "medications": ["Lisinopril"],
            },
            "communication_style": {
                "tone": "anxious",
                "vocabulary_level": "layman",
                "verbosity": "medium",
            },
        }
        state = {
            "current_state": "presentation",
            "variables": {"pain_level": 7, "anxiety_level": 4},
        }

        prompt = engine.build_system_prompt(persona, state)
        assert "DO NOT BREAK CHARACTER" in prompt

    def test_generate_response_returns_string(self):
        provider = MockLLMProvider()
        engine = InferenceEngine(provider)

        persona = {
            "metadata": {"name": "Test", "role": "patient"},
            "clinical_profile": {
                "condition": "Anxiety",
                "symptoms": [],
                "medical_history": "None",
                "medications": [],
            },
            "communication_style": {
                "tone": "calm",
                "vocabulary_level": "layman",
                "verbosity": "low",
            },
        }
        state = {
            "current_state": "calm",
            "variables": {"pain_level": 0, "anxiety_level": 2},
            "history": [
                {"turn": 1, "action": "Patient greeted"},
                {"turn": 2, "action": "Patient described symptoms"},
            ],
        }

        result = engine.generate_response("How are you?", persona, state)
        assert isinstance(result, str)
        assert "Mock response to:" in result

    def test_build_messages_structure(self):
        """Messages list contains system message followed by user message."""
        provider = MockLLMProvider()
        engine = InferenceEngine(provider)

        persona = {
            "metadata": {"name": "Alice", "role": "patient"},
            "clinical_profile": {
                "condition": "Migraine",
                "symptoms": [],
                "medical_history": "None",
                "medications": [],
            },
            "communication_style": {
                "tone": "neutral",
                "vocabulary_level": "layman",
                "verbosity": "medium",
            },
        }
        state = {
            "current_state": "neutral",
            "variables": {"pain_level": 5, "anxiety_level": 3},
            "history": [],
        }

        messages = engine._build_messages("Hello", persona, state)

        assert len(messages) == 2
        assert messages[0]["role"] == "system"
        assert messages[1]["role"] == "user"
        assert "DO NOT BREAK CHARACTER" in messages[0]["content"]
        assert "Hello" in messages[1]["content"]

    @patch.object(OpenAIProvider, "generate", return_value="Hello, patient.")
    def test_generate_response_propagates_provider_output(self, mock_gen):
        provider = OpenAIProvider(api_key="sk-test")
        engine = InferenceEngine(provider)

        persona = {
            "metadata": {"name": "Doc", "role": "doctor"},
            "clinical_profile": {
                "condition": "Checkup",
                "symptoms": [],
                "medical_history": "None",
                "medications": [],
            },
            "communication_style": {
                "tone": "professional",
                "vocabulary_level": "medical",
                "verbosity": "high",
            },
        }
        state = {
            "current_state": "calm",
            "variables": {"pain_level": 0, "anxiety_level": 0},
            "history": [],
        }

        result = engine.generate_response("I have a headache.", persona, state)
        assert result == "Hello, patient."
