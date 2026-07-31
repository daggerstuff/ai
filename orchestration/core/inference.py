import os
from typing import Any

import openai


class LLMProvider:
    def generate(self, messages: list[dict[str, str]]) -> str:
        raise NotImplementedError


class MockLLMProvider(LLMProvider):
    def generate(self, messages: list[dict[str, str]]) -> str:
        user_msg = next((m["content"] for m in messages if m["role"] == "user"), "")
        return f"Mock response to: {user_msg[:20]}..."


class OpenAIProvider(LLMProvider):
    def __init__(
        self,
        api_key: str | None = None,
        model: str = "gpt-4",
        base_url: str | None = None,
    ):
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.model = model
        self.base_url = base_url
        self._client: openai.OpenAI | None = None

    def _get_client(self) -> openai.OpenAI:
        if self._client is None:
            self._client = openai.OpenAI(
                api_key=self.api_key,
                base_url=self.base_url,
            )
        return self._client

    def generate(self, messages: list[dict[str, str]]) -> str:
        if not self.api_key:
            return "OpenAI API Key not set. Please provide it."

        client = self._get_client()
        response = client.chat.completions.create(
            model=self.model,
            messages=messages,  # type: ignore[arg-type]
        )
        return response.choices[0].message.content or ""


class InferenceEngine:
    def __init__(self, provider: LLMProvider):
        self.provider = provider

    def build_system_prompt(
        self,
        persona_definition: dict[str, Any],
        current_state: dict[str, Any],
    ) -> str:
        metadata = persona_definition.get("metadata", {})
        profile = persona_definition.get("clinical_profile", {})
        style = persona_definition.get("communication_style", {})
        persona_definition.get("emotional_state", {})

        return f"""You are playing the role of {metadata.get("name")}, a {metadata.get("role")}.
Your condition is {profile.get("condition")}.
Symptoms: {", ".join([s["name"] for s in profile.get("symptoms", [])])}.
Medical History: {profile.get("medical_history")}.
Medications: {", ".join(profile.get("medications", []))}.

Communication Style:
- Tone: {style.get("tone")}
- Vocabulary: {style.get("vocabulary_level")}
- Verbosity: {style.get("verbosity")}

Current Emotional State: {current_state.get("current_state")}
Pain Level: {current_state.get("variables", {}).get("pain_level")}/10
Anxiety Level: {current_state.get("variables", {}).get("anxiety_level")}/10

DO NOT BREAK CHARACTER. You must stay in character at all times.
Only reveal hidden information if specifically asked or if appropriate for the situation.
"""

    def _build_messages(
        self,
        user_input: str,
        persona_definition: dict[str, Any],
        current_state: dict[str, Any],
    ) -> list[dict[str, str]]:
        system_prompt = self.build_system_prompt(persona_definition, current_state)
        history = current_state.get("history", [])
        context = "\n".join([f"Turn {h['turn']}: {h['action']}" for h in history[-5:]])
        full_prompt = f"Previous context:\n{context}\n\nStudent: {user_input}\nYou:"

        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": full_prompt},
        ]

    def generate_response(
        self,
        user_input: str,
        persona_definition: dict[str, Any],
        current_state: dict[str, Any],
    ) -> str:
        messages = self._build_messages(user_input, persona_definition, current_state)
        return self.provider.generate(messages)
