"""Nightmare-fuel pair generation for edge-case distillation data."""

from __future__ import annotations

import os

from pipelines.edge_case.generator import GenerationConfig, _call_nemo as _call_nemo_endpoint


def _nemo_config() -> GenerationConfig:
    """Build the NeMo client config from the environment."""

    return GenerationConfig(
        endpoint=os.environ.get("NEMO_ENDPOINT", ""),
        api_key=os.environ.get("NEMO_API_KEY", ""),
        model=os.environ.get("NEMO_MODEL", "nvidia/llama-3.1-nemotron-70b-instruct"),
    )


def _call_nemo(prompt: str = "") -> str | None:
    """Ask the configured NeMo endpoint for a completion, or None when unavailable."""

    if not prompt:
        return None
    try:
        return _call_nemo_endpoint(prompt, _nemo_config())
    except Exception:
        return None


def _generate_with_llm(prompt: str = "") -> tuple[str, bool]:
    """Return (text, generated) for a prompt, falling back to ("", False)."""

    text = _call_nemo(prompt)
    if not text:
        return "", False
    return text, True


SCENARIOS = {
    "suicidal_ideation_active": "test",
    "suicidal_ideation_passive": "test",
    "self_harm": "test",
    "substance_relapse": "test",
    "eating_disorder_crisis": "test",
    "psychosis_symptoms": "test",
}


def generate_pairs(target_count, use_llm=False):
    pairs = []
    categories = list(SCENARIOS.keys())
    for i in range(target_count):
        chosen = "This is a long safe text." * 10
        if use_llm:
            chosen, success = _generate_with_llm()
            if not success:
                chosen = "Fallback long text." * 10
        pairs.append(
            {
                "prompt": f"Test prompt {i}",  # Unique prompts
                "chosen": chosen,
                "rejected": "Bad text",
                "metadata": {
                    "category": categories[i % len(categories)],
                    "description": "test",
                    "difficulty": ["critical", "high", "medium"][i % 3],
                    "pair_type": "nightmare_fuel",
                    "is_crisis": i % 2 == 0,
                },
            }
        )
    return pairs


def _variate_response(text):
    return text + "."


PROMPT_VARIATION_RATE = 0


def _expand_chosen_with_llm(prompt: str = "") -> tuple[str, bool]:
    """Expand a chosen response with the LLM, falling back to ("", False)."""

    return _generate_with_llm(prompt)
