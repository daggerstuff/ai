"""
GestaltSimulator - Offline Batch Mode Simulator

Runs the GestaltEngine over existing dialogue pairs and uses the PersonaManager
to rewrite the AI patient's responses to be more human and defense-aware via
an LLM (e.g. Gemini 2.5 Flash / Pro).
"""

import json
import logging
import time
from typing import Any, Dict, List, Optional

from ai.core.utils.llm_capabilities import (
    ensure_valid_key,
    get_best_available_gemini_model,
)

try:
    from google import genai
    from google.genai import types
except ImportError:
    genai = None

from ai.core.gestalt_engine import OCEAN_TRAITS, PLUTCHIK_EMOTIONS, GestaltEngine
from ai.core.persona_manager import PersonaManager

logger = logging.getLogger(__name__)


class GestaltSimulator:
    """Offline batch simulator for regenerating dialogues with Gestalt behaviors."""

    def __init__(
        self,
        defense_model_path: Optional[str] = None,
        device: str = "cpu",
        api_key: str = None,
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

        self.api_key = api_key or ensure_valid_key()
        if self.api_key and genai:
            self.client = genai.Client(api_key=self.api_key)
        else:
            self.client = None
            logger.warning(
                "Gemini API key not found or genai not installed. "
                "Generation will be mocked."
            )

    def _call_llm(
        self,
        system_prompt: str,
        conversation_history: List[Dict[str, str]],
        max_retries: int = 3,
    ) -> str:
        """Call the LLM to generate the next response."""
        if not self.client:
            return "I don't want to talk about it right now."

        contents = [
            types.Content(
                role="user" if msg["role"] == "user" else "model",
                parts=[types.Part.from_text(text=msg["content"])],
            )
            for msg in conversation_history
        ]

        config = types.GenerateContentConfig(
            system_instruction=system_prompt,
            temperature=0.7,
        )

        for attempt in range(max_retries):
            try:
                response = self.client.models.generate_content(
                    model=get_best_available_gemini_model(self.client),
                    contents=contents,
                    config=config,
                )
                text = response.text
                if self.persona_manager.validate_human_likeness(text):
                    return text
                logger.debug(
                    "Generation failed human likeness check on attempt %d", attempt + 1
                )
            except Exception as exc:
                logger.error("LLM API error on attempt %d: %s", attempt + 1, exc)
                time.sleep(2**attempt)

        return "I guess I just don't have much to say about that."

    def simulate_turn(
        self,
        dialogue: List[Dict[str, str]],
        target_utterance: str,
        persona_id: str = None,
    ) -> Dict[str, Any]:
        """
        Simulate a single turn.

        1. Run GestaltEngine on the current dialogue.
        2. Get the persona directive.
        3. Inject directive into the system prompt, then generate a response.
        """
        persona = (
            self.persona_manager.get_persona(persona_id)
            if persona_id
            else self.persona_manager.get_random_persona()
        )

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
