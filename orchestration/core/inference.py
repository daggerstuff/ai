import os
from typing import Any

import openai


class LLMProvider:
    def generate(self, prompt: str, system_prompt: str) -> str:
        raise NotImplementedError

class MockLLMProvider(LLMProvider):
    def generate(self, prompt: str, system_prompt: str) -> str:
        return f"Mock response to: {prompt[:20]}..."

class OpenAIProvider(LLMProvider):
    def __init__(self, api_key: str | None = None, model: str = "gpt-4"):
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.model = model
        if self.api_key:
            openai.api_key = self.api_key

    def generate(self, prompt: str, system_prompt: str) -> str:
        if not self.api_key:
            return "OpenAI API Key not set. Please provide it."

        response = openai.ChatCompletion.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}
            ]
        )
        return response.choices[0].message.content

class InferenceEngine:
    def __init__(self, provider: LLMProvider):
        self.provider = provider

    def build_system_prompt(self, persona_definition: dict[str, Any], current_state: dict[str, Any]) -> str:
        metadata = persona_definition.get("metadata", {})
        profile = persona_definition.get("clinical_profile", {})
        style = persona_definition.get("communication_style", {})
        persona_definition.get("emotional_state", {})

        return f"""You are playing the role of {metadata.get('name')}, a {metadata.get('role')}.
Your condition is {profile.get('condition')}.
Symptoms: {', '.join([s['name'] for s in profile.get('symptoms', [])])}.
Medical History: {profile.get('medical_history')}.
Medications: {', '.join(profile.get('medications', []))}.

Communication Style:
- Tone: {style.get('tone')}
- Vocabulary: {style.get('vocabulary_level')}
- Verbosity: {style.get('verbosity')}

Current Emotional State: {current_state.get('current_state')}
Pain Level: {current_state.get('variables', {}).get('pain_level')}/10
Anxiety Level: {current_state.get('variables', {}).get('anxiety_level')}/10

Strictly adhere to your persona. Do not break character.
Only reveal hidden information if specifically asked or if appropriate for the situation.
"""

    def generate_response(self, user_input: str, persona_definition: dict[str, Any], current_state: dict[str, Any]) -> str:
        system_prompt = self.build_system_prompt(persona_definition, current_state)
        # We could also include turn history here
        history = current_state.get("history", [])
        context = "\n".join([f"Turn {h['turn']}: {h['action']}" for h in history[-5:]]) # last 5 turns
        full_prompt = f"Previous context:\n{context}\n\nStudent: {user_input}\nYou:"

        return self.provider.generate(full_prompt, system_prompt)
