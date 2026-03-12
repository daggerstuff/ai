"""
GestaltSimulator - Offline Batch Mode Simulator.

Runs the GestaltEngine over existing dialogue pairs and uses the PersonaManager
to rewrite the AI patient's responses to be more human and defense-aware via
an OpenAI-compatible LLM backend (NVIDIA NIM by default).
"""

import json
import logging
import os
import re
import requests
import time
from typing import Any, Dict, List, Optional

from ai.core.utils.llm_capabilities import ensure_valid_key, get_best_available_gemini_model

try:
    from google import genai
    from google.genai import types
except ImportError:
    genai = None

from ai.core.gestalt_engine import OCEAN_TRAITS, PLUTCHIK_EMOTIONS, GestaltEngine
from ai.core.persona_manager import PersonaManager

logger = logging.getLogger(__name__)


_THINK_TAG_RE = re.compile(r"<think>.*?</think>\s*", re.IGNORECASE | re.DOTALL)


def _sanitize_llm_text(text: str) -> str:
    """Remove generation artifacts that should never be persisted."""
    if not text:
        return text
    text = text.strip()
    text = _THINK_TAG_RE.sub("", text)
    return text.strip()


class GestaltSimulator:
    """Offline batch simulator for regenerating dialogues with Gestalt behaviors."""

    def __init__(
        self,
        defense_model_path: Optional[str] = None,
        device: str = "cpu",
        api_key: str = None,
        nim_only: bool = False,
    ):
        self.gestalt_engine = GestaltEngine()
        if defense_model_path:
            logger.info("Loading defense model from %s", defense_model_path)
            try:
                self.gestalt_engine.load_defense_model(
                    defense_model_path, device=device
                )
            except Exception as exc:
                logger.warning(
                    "Could not load defense model, running in dry-run/mock mode: %s",
                    exc,
                )
        else:
            logger.info(
                "No defense model path provided, initializing GestaltEngine with "
                "NIM defaults."
            )
            self.gestalt_engine.load_defense_model()

        self.persona_manager = PersonaManager()

        self.llm_mode = "nim" if (nim_only or self.nim_api_key) else "gemini"
        self.gemini_client = None
        self.nim_model = (
            os.environ.get("NIM_MODEL")
            or os.environ.get("NVIDIA_OPENAI_MODEL")
            or os.environ.get("OPENAI_MODEL")
            or "meta/llama-3.1-405b-instruct"
        )
        self.nim_base_url = (
            os.environ.get("OPENAI_BASE_URL")
            or os.environ.get("NVIDIA_OPENAI_BASE_URL")
            or "https://integrate.api.nvidia.com/v1"
        ).rstrip("/")
        self.nim_api_key = api_key or os.getenv("NIM_API_KEY") or os.getenv("NVIDIA_API_KEY") or os.getenv("OPENAI_API_KEY")
        self.nim_headers = {
            "Content-Type": "application/json",
        }
        if self.nim_api_key:
            self.nim_headers["Authorization"] = f"Bearer {self.nim_api_key}"

        if not nim_only:
            try:
                gemini_key = os.environ.get("GOOGLE_CLOUD_API_KEY") or os.environ.get(
                    "GEMINI_API_KEY"
                )
                if gemini_key:
                    self.gemini_client = genai.Client(api_key=gemini_key) if genai else None
                else:
                    try:
                        self.gemini_client = genai.Client(api_key=ensure_valid_key())
                    except Exception:
                        self.gemini_client = None
            except Exception:
                self.gemini_client = None
        if not self.nim_api_key and not self.gemini_client:
            self.llm_mode = "mock"
            logger.warning(
                "No NVIDIA NIM or Gemini credentials found. "
                "Generation will be mocked."
            )

    def _call_nim(
        self,
        system_prompt: str,
        conversation_history: List[Dict[str, str]],
    ) -> str:
        """Call NIM-compatible OpenAI endpoint."""
        if not self.nim_api_key:
            raise RuntimeError("No NIM API key available.")

        messages = [
            {"role": "system", "content": system_prompt},
            *[
                {
                    "role": "user" if msg["role"] == "user" else "assistant",
                    "content": msg["content"],
                }
                for msg in conversation_history
            ],
        ]
        payload = {
            "model": self.nim_model,
            "messages": messages,
            "temperature": 0.7,
        }

        response = requests.post(
            f"{self.nim_base_url}/chat/completions",
            headers=self.nim_headers,
            json=payload,
            timeout=30,
        )
        response.raise_for_status()
        payload = response.json()
        choices = payload.get("choices", [])
        if not choices:
            return ""
        message = choices[0].get("message", {})
        return message.get("content", "") or ""

    def _call_gemini(
        self,
        system_prompt: str,
        conversation_history: List[Dict[str, str]],
    ) -> str:
        """Call Gemini API as a fallback mode."""
        if not self.gemini_client:
            raise RuntimeError("No Gemini client available.")

        config = types.GenerateContentConfig(
            system_instruction=system_prompt,
            temperature=0.7,
        )
        contents = [
            types.Content(
                role="user" if msg["role"] == "user" else "model",
                parts=[types.Part.from_text(text=msg["content"])],
            )
            for msg in conversation_history
        ]
        response = self.gemini_client.models.generate_content(
            model=get_best_available_gemini_model(self.gemini_client),
            contents=contents,
            config=config,
        )
        return response.text

    def _call_llm(
        self,
        system_prompt: str,
        conversation_history: List[Dict[str, str]],
        max_retries: int = 3,
    ) -> str:
        """Call the LLM to generate the next response."""
        if self.llm_mode == "mock":
            return (
                "I hear you, and I can stay with that. "
                "Tell me more about what that feels like."
            )

        for attempt in range(max_retries):
            try:
                if self.nim_api_key:
                    text = _sanitize_llm_text(self._call_nim(system_prompt, conversation_history))
                elif self.gemini_client:
                    text = _sanitize_llm_text(self._call_gemini(system_prompt, conversation_history))
                else:
                    raise RuntimeError("No LLM backend available.")

                if not text:
                    logger.debug("Empty LLM response on attempt %d", attempt + 1)
                    continue
                if self.persona_manager.validate_human_likeness(text):
                    return text
                logger.debug(
                    "Generation failed human likeness check on attempt %d", attempt + 1
                )
            except Exception as exc:
                logger.error("LLM API error on attempt %d: %s", attempt + 1, exc)
                time.sleep(2**attempt)

        return (
            "I need a moment to process this. "
            "Let's keep going and I can share what feels most true for me."
        )

    def simulate_turn(
        self,
        dialogue: List[Dict[str, str]],
        target_utterance: str,
        persona_id: str = None,
        persona_id_hint: str = None,
    ) -> Dict[str, Any]:
        """
        Simulate a single turn.

        1. Run GestaltEngine on the current dialogue.
        2. Get the persona directive.
        3. Inject directive into the system prompt, then generate a response.
        """
        selected_persona_id = persona_id_hint or persona_id
        persona = (
            self.persona_manager.get_persona(selected_persona_id)
            if selected_persona_id
            else self.persona_manager.get_random_persona()
        )
        if not persona:
            logger.warning(
                "Requested persona_id '%s' not found. Falling back to random.",
                selected_persona_id,
            )
            persona = self.persona_manager.get_random_persona()

        # Mock middle-of-the-road emotion/trait scores for batch regen.
        mock_plutchik = {e: 0.2 for e in PLUTCHIK_EMOTIONS}
        mock_plutchik["sadness"] = 0.6
        mock_ocean = {t: persona.traits.get(t, 0.5) for t in OCEAN_TRAITS}

        if self.gestalt_engine.defense_model_loaded:
            gestalt_state = self.gestalt_engine.analyze_gestalt(
                dialogue=dialogue,
                target_utterance=target_utterance,
                plutchik_scores=mock_plutchik,
                ocean_scores=mock_ocean,
            )
            directive = gestalt_state.persona_directive
        else:
            logger.debug(
                "Defense model not loaded, using default persona defense directive."
            )
            directive = (
                f"[System: Maintain your '{persona.default_defense}' "
                "defense mechanism.]"
            )
            gestalt_state = None

        if not (directive and directive.strip()):
            directive = (
                "[System: Maintain strong therapeutic boundaries and "
                "stay grounded in the patient perspective.]"
            )

        system_prompt = persona.generate_system_prompt()
        if directive:
            system_prompt += f"\n\nCRITICAL DIRECTIVE:\n{directive}"

        # Build LLM history: dialogue history first, then the newest user utterance.
        llm_history = []
        for turn in dialogue:
            role = (
                "user"
                if turn.get("speaker", "user")
                in ("human", "user", "client", "therapist")
                else "assistant"
            )
            llm_history.append({"role": role, "content": turn.get("text", "")})
        llm_history.append({"role": "user", "content": target_utterance})

        new_response = self._call_llm(system_prompt, llm_history)

        return {
            "original_utterance": target_utterance,
            "new_response": new_response,
            "persona_id": persona.archetype_id,
            "directive_used": directive,
            "gestalt_state": gestalt_state.__dict__ if gestalt_state else None,
        }

    def process_batch(
        self, input_file: str, output_file: str, max_records: int = 5000
    ) -> int:
        """
        Process a JSONL file of dialogue pairs and rewrite the assistant responses.

        Returns the number of records written.
        """
        logger.info("Starting batch simulation from %s → %s", input_file, output_file)

        processed_count = 0
        with (
            open(input_file, "r", encoding="utf-8") as infile,
            open(output_file, "w", encoding="utf-8") as outfile,
        ):
            for line in infile:
                if processed_count >= max_records:
                    break

                try:
                    record = json.loads(line)
                    messages = record.get("messages", [])

                    if len(messages) < 3:
                        continue
                    if (
                        messages[-1]["role"] != "assistant"
                        or messages[-2]["role"] != "user"
                    ):
                        continue

                    target_user_utterance = messages[-2]["content"]

                    history_for_engine = [
                        {
                            "speaker": (
                                "therapist" if msg["role"] == "user" else "client"
                            ),
                            "text": msg["content"],
                        }
                        for msg in messages[:-2]
                        if msg["role"] != "system"
                    ]
                    result = self.simulate_turn(
                        history_for_engine, target_user_utterance
                    )

                    record["messages"][-1]["content"] = result["new_response"]
                    record.setdefault("metadata", {})["gestalt_simulation"] = {
                        "persona_id": result["persona_id"],
                        "directive": result["directive_used"],
                    }

                    outfile.write(json.dumps(record, ensure_ascii=False) + "\n")
                    processed_count += 1

                    if processed_count % 100 == 0:
                        logger.info("Processed %d records...", processed_count)

                except Exception as exc:
                    logger.error("Error processing record %d: %s", processed_count, exc)

        logger.info(
            "Batch complete. Wrote %d records to %s", processed_count, output_file
        )
        return processed_count
