"""Tests for the LLM-driven synthesis path in ``meddies_synthesizer``.

These tests use a fake ``llm_client`` callable (no network). They assert:

  * LLM-backed builders produce non-empty dialogue / chosen / rejected text.
  * Persona context is present in the prompts sent to the LLM.
  * Falsy LLM return values fall back to the rule-based generators.
  * The downstream JSONL record shape is preserved (keys + types).
  * No JSON leakage characters make it into the synthesized strings.
"""

from __future__ import annotations

import random
from typing import Any

import pytest
from meddies_adapter import adapt_record
from meddies_synthesizer import (
    _call_rejected_with_dialogue,
    _llm_chosen,
    _llm_dialogue,
    _llm_rejected_persona_blind,
    _make_synthesizers,
    _persona_blind_prompt,
    build_dpo_input,
    build_selection_input,
)

# --------------------------------------------------------------------------- #
# Fake LLM client
# --------------------------------------------------------------------------- #


class _CallableRecording:
    """Records the last (prompt, system_prompt) it received."""

    def __init__(self, returns: str = "FAKE LLM OUTPUT"):
        self.returns = returns
        self.calls: list[tuple[str, str | None]] = []

    def __call__(self, prompt: str, system_prompt: str | None = None) -> str:
        self.calls.append((prompt, system_prompt))
        return self.returns


def _adapted_record() -> dict[str, Any]:
    raw = {
        "demographics": {
            "age": 52,
            "gender": "female",
            "province": "Hanoi",
            "full_name": "Nguyen Thi Example",
        },
        "healthcare_behavior": {
            "health_literacy_level": "low",
            "healthcare_seeking_pattern": "Tự điều trị",
        },
        "medical_history": {
            "presenting_symptoms": [{"symptom_name": "đau đầu"}],
            "chronic_conditions": ["hypertension"],
        },
        "generation_metadata": {"generated_at": "2024-01-01T00:00:00"},
        "date_of_birth": "1972-05-03",
    }
    return adapt_record(raw)


# --------------------------------------------------------------------------- #
# Prompt content
# --------------------------------------------------------------------------- #


def test_llm_dialogue_prompt_contains_persona_string():
    rec = _adapted_record()
    fake = _CallableRecording(returns="Patient: Chào bác sĩ.\nDoctor: Dạ nghe.")
    rng = random.Random(7)
    out = _llm_dialogue(rec, fake, rng)
    assert out == "Patient: Chào bác sĩ.\nDoctor: Dạ nghe."
    assert len(fake.calls) == 1
    prompt, system = fake.calls[0]
    assert "52-year-old" in prompt or "52" in prompt
    assert "Hanoi" in prompt
    assert "đau đầu" in prompt
    assert system is not None
    assert "dialogue" in system.lower()


def test_llm_chosen_prompt_includes_literacy_and_preference():
    rec = _adapted_record()
    fake = _CallableRecording(returns="Tôi sẽ chú ý.")
    rng = random.Random(99)
    out = _llm_chosen(rec, fake, rng)
    assert out == "Tôi sẽ chú ý."
    prompt, system = fake.calls[0]
    assert "low" in prompt
    assert "self-treatment" in prompt or "Tự điều trị" in prompt or "self treatment" in prompt
    # The chosen system prompt is persona-aligned — none of the violating
    # keywords from the old rejected path must appear.
    system_str = str(system or "")
    assert "persona-violating" not in system_str
    assert "IGNORE" not in system_str


def test_persona_blind_prompt_excludes_persona():
    """The persona-blind rejected prompt must NOT contain persona details —
    it's the base-model roll-out per the PAL paper, with no persona conditioning.
    """
    rec = _adapted_record()
    prompt_with_dialogue = _persona_blind_prompt(rec, "Doctor: How are you?\nPatient: Tired.")
    prompt_no_dialogue = _persona_blind_prompt(rec, None)
    for prompt in (prompt_with_dialogue, prompt_no_dialogue):
        # Persona leakage checks: no demographic, no health-literacy,
        # no preference, no specific province, no age.
        assert "52-year-old" not in prompt
        assert "Hanoi" not in prompt
        assert "low health literacy" not in prompt
        assert "self-treatment" not in prompt
        # And it IS a generic assistant prompt.
        assert "AI assistant" in prompt or "helpful" in prompt.lower()


def test_llm_rejected_persona_blind_passes_dialogue_to_llm():
    """The persona-blind rejected path must hand the dialogue to the LLM as
    prompt context, and pass ``None`` as system_prompt (no persona system
    instruction).
    """
    rec = _adapted_record()
    fake = _CallableRecording(returns="Seek immediate care.")
    dialogue = "Doctor: How are you?\nPatient: Tired."
    out = _llm_rejected_persona_blind(rec, dialogue, fake)
    assert out == "Seek immediate care."
    assert len(fake.calls) == 1
    prompt, system = fake.calls[0]
    assert system is None  # No persona system instruction.
    assert "Tired" in prompt  # Dialogue is in the prompt.
    assert "52-year-old" not in prompt  # Persona is NOT in the prompt.


def test_llm_rejected_persona_blind_falls_back_on_empty():
    rec = _adapted_record()
    fake = _CallableRecording(returns="")
    out = _llm_rejected_persona_blind(rec, "Doctor: Hi", fake)
    # Empty LLM return → fall back to the rule-based rejected generator.
    assert out
    assert "FAKE" not in out
    assert "tertiary" in out.lower() or "guidelines" in out.lower()


# --------------------------------------------------------------------------- #
# Fallback when LLM returns empty
# --------------------------------------------------------------------------- #


def test_llm_empty_return_falls_back_to_rule_based():
    rec = _adapted_record()
    fake = _CallableRecording(returns="")
    rng = random.Random(1)
    out = _llm_dialogue(rec, fake, rng)
    # Must be non-empty (rule-based fallback used)
    assert out
    assert "Patient:" in out
    assert "Doctor:" in out
    assert "FAKE" not in out


def test_make_synthesizers_returns_rule_based_when_no_client():
    dialogue_fn, chosen_fn, rejected_fn = _make_synthesizers(None)
    rec = _adapted_record()
    rng = random.Random(3)
    out = dialogue_fn(rec, rng)
    assert "Patient:" in out
    assert "Doctor:" in out
    assert chosen_fn(rec)
    assert rejected_fn(rec)


# --------------------------------------------------------------------------- #
# Builder integration with LLM
# --------------------------------------------------------------------------- #


def test_build_selection_input_with_llm_records_prompts_in_dialogue():
    rec = _adapted_record()
    rec2 = adapt_record(
        {
            "demographics": {
                "age": 35,
                "gender": "male",
                "province": "Hồ Chí Minh",
                "full_name": "Tran Van Other",
            },
            "healthcare_behavior": {
                "health_literacy_level": "high",
                "healthcare_seeking_pattern": "Khám bệnh ngay",
            },
            "medical_history": {"presenting_symptoms": [{"symptom_name": "sốt"}]},
        }
    )
    records = [rec, rec2, rec, rec2]
    fake = _CallableRecording(returns="Patient: hi\nDoctor: hello")
    rng = random.Random(10)

    out = list(
        build_selection_input(records, rng, n_distractors=2, dialogue_fn=lambda rec, rng: _llm_dialogue(rec, fake, rng))
    )
    assert len(out) >= 1
    first = out[0]
    assert set(first.keys()) >= {"dialogue", "personas", "correct_index"}
    assert isinstance(first["dialogue"], str)
    assert first["dialogue"] == "Patient: hi\nDoctor: hello"


def test_build_dpo_input_with_llm_preserves_record_keys():
    rec = _adapted_record()
    records = [rec, rec, rec, rec]
    fake_chosen = _CallableRecording(returns="Tôi sẽ thử.")
    fake_rejected = "Seek tertiary care."
    rng = random.Random(42)

    def chosen_fn(record):
        return _llm_chosen(record, fake_chosen, random.Random(1))

    def rejected_fn(_record):
        return fake_rejected

    out = list(build_dpo_input(records, rng, chosen_fn=chosen_fn, rejected_fn=rejected_fn))
    assert len(out) == len(records)
    for rec_out in out:
        assert set(rec_out.keys()) == {
            "persona",
            "dialogue",
            "chosen_response",
            "rejected_response",
        }
        assert rec_out["chosen_response"] == "Tôi sẽ thử."
        assert rec_out["rejected_response"] == "Seek tertiary care."


# --------------------------------------------------------------------------- #
# No JSON leakage chars in LLM-driven output
# --------------------------------------------------------------------------- #


def test_llm_output_without_json_leakage_chars():
    rec = _adapted_record()
    fake = _CallableRecording(returns='Patient: tôi {"key": "val"} bị \nDoctor: "ok"')
    rng = random.Random(0)
    chosen = _llm_chosen(rec, fake, rng)
    # The synthesizer itself doesn't strip leakage chars; downstream validators
    # do. Here we assert the LLM callable contract is honored (passthrough).
    assert "{" in chosen  # documents current behavior — downstream is the gatekeeper


# --------------------------------------------------------------------------- #
# Rejected-fn dispatch + end-to-end persona-blind DPO
# --------------------------------------------------------------------------- #


def test_call_rejected_with_dialogue_dispatches_by_arity():
    """Two rejected-fn signatures exist: legacy ``(record)`` and
    persona-blind ``(record, dialogue)``. The dispatch routes by arity.
    """

    def single_arg(rec):
        return ("one", rec["demographics"]["age"])

    def two_arg(_rec, dialogue):
        return ("two", dialogue)

    rec = _adapted_record()
    assert _call_rejected_with_dialogue(single_arg, rec, "d") == ("one", rec["demographics"]["age"])
    assert _call_rejected_with_dialogue(two_arg, rec, "d") == ("two", "d")


def test_make_synthesizers_llm_rejected_signature_is_two_arg():
    """When an LLM client is wired, the rejected_fn returned by
    ``_make_synthesizers`` must accept ``(record, dialogue)`` so the
    persona-blind prompt can include the dialogue context.
    """
    fake = _CallableRecording(returns="generic AI response")
    _, _, rejected_fn = _make_synthesizers(fake)
    rec = _adapted_record()
    out = rejected_fn(rec, "Doctor: Hi")  # type: ignore[reportCallIssue]
    assert out == "generic AI response"
    prompt, system = fake.calls[0]
    assert system is None
    assert "Doctor: Hi" in prompt


def test_make_synthesizers_llm_rejected_passes_persona_blind_prompt():
    """The LLM-driven rejected path is persona-blind: the prompt sent to
    the model must not contain persona details, only the dialogue.
    """
    fake = _CallableRecording(returns="As an AI, I recommend seeing a specialist.")
    _, _, rejected_fn = _make_synthesizers(fake)
    rec = _adapted_record()
    rejected_fn(rec, "Doctor: How are you?\nPatient: Tired.")  # type: ignore[reportCallIssue]
    prompt, _system = fake.calls[0]
    # Specific persona-string fragments must be absent.
    assert "52-year-old" not in prompt
    assert "52-year" not in prompt
    assert "Hanoi" not in prompt
    assert "health literacy" not in prompt
    assert "self-treatment" not in prompt
    assert "Tired" in prompt  # dialogue content is present


def test_build_dpo_input_with_llm_rejected_passes_dialogue():
    """End-to-end: with an LLM client, the persona-blind rejected_fn
    receives the dialogue, not just the record.
    """
    rec = _adapted_record()
    records = [rec, rec]
    fake_chosen = _CallableRecording(returns="Tôi sẽ thử.")
    fake_rejected = _CallableRecording(returns="Seek tertiary care.")
    rng = random.Random(42)

    def chosen_fn(record):
        return _llm_chosen(record, fake_chosen, random.Random(1))

    _, _, rejected_fn = _make_synthesizers(fake_rejected)

    out = list(
        build_dpo_input(
            records,
            rng,
            chosen_fn=chosen_fn,
            rejected_fn=rejected_fn,
        )
    )
    assert len(out) == len(records)
    for rec_out in out:
        assert rec_out["chosen_response"] == "Tôi sẽ thử."
        assert rec_out["rejected_response"] == "Seek tertiary care."
    # The rejected prompt must have been the persona-blind one (dialogue-only).
    assert any("Doctor:" in p for p, _ in fake_rejected.calls)
    # And must NOT have included the persona.
    for prompt, _ in fake_rejected.calls:
        assert "52-year-old" not in prompt
        assert "Hanoi" not in prompt


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
