"""Tests for ``generate_dpo_pairs`` — PIX-4074 Phase 3.1 DPO preference pairs."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from generate_dpo_pairs import (
    CHARS_PER_TOKEN,
    DEFAULT_MAX_TOKENS,
    DEFAULT_MIN_RECORDS,
    SYSTEM_PROMPT,
    DpoPairExample,
    _has_json_leakage,
    _response_has_json_leakage,
    build_dpo_pair,
    build_prompt,
    estimate_tokens_text,
    format_dialogue_history,
    format_persona_safe,
    generate_dataset,
    is_chatml_compliant,
    main,
    validate_token_bounds,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _persona(
    age: int = 45,
    gender: str = "female",
    location: str = "Hanoi",
    health_literacy: str = "low",
    preference: str = "traditional medicine",
) -> dict[str, Any]:
    return {
        "demographics": {"age": age, "gender": gender, "location": location},
        "healthcare_behavior": {"health_literacy": health_literacy, "preference": preference},
    }


def _make_input_jsonl(path: Path, n: int) -> None:
    """Write n DPO input records to ``path``."""
    with path.open("w", encoding="utf-8") as f:
        for i in range(n):
            f.write(
                json.dumps(
                    {
                        "persona": _persona(age=30 + i),
                        "dialogue": f"Doctor: How are you?\nPatient: I feel sick #{i}",
                        "chosen_response": f"I feel very tired, doctor. I do not understand the big words. #{i}",
                        "rejected_response": f"Based on your clinical presentation, I recommend immediate pharmacological intervention. #{i}",
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )


def _write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


# ---------------------------------------------------------------------------
# format_persona_safe
# ---------------------------------------------------------------------------


def test_format_persona_safe_basic() -> None:
    result = format_persona_safe(_persona())
    assert isinstance(result, str)
    assert "45-year-old" in result
    assert "female" in result
    assert "Hanoi" in result


def test_format_persona_safe_no_json_leakage() -> None:
    result = format_persona_safe(_persona())
    for ch in "{}\"'":
        assert ch not in result


# ---------------------------------------------------------------------------
# format_dialogue_history
# ---------------------------------------------------------------------------


def test_format_dialogue_history_string() -> None:
    assert format_dialogue_history("Doctor: Hi") == "Doctor: Hi"


def test_format_dialogue_history_messages() -> None:
    msgs = [
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": "Hi there"},
    ]
    result = format_dialogue_history(msgs)
    assert "User: Hello" in result
    assert "Assistant: Hi there" in result


def test_format_dialogue_history_none() -> None:
    assert format_dialogue_history(None) == ""


def test_format_dialogue_history_invalid_type() -> None:
    invalid: Any = 123
    with pytest.raises(ValueError, match="str | list | None"):
        format_dialogue_history(invalid)


def test_format_dialogue_history_invalid_turn() -> None:
    invalid_turn: Any = "not a dict"
    with pytest.raises(ValueError, match="must be a dict"):
        format_dialogue_history([invalid_turn])  # type: ignore[list-item]


def test_format_dialogue_history_non_string_content() -> None:
    with pytest.raises(ValueError, match="must be a string"):
        format_dialogue_history([{"role": "user", "content": 42}])  # type: ignore[dict-item]


# ---------------------------------------------------------------------------
# build_prompt
# ---------------------------------------------------------------------------


def test_build_prompt_contains_system_instruction() -> None:
    prompt = build_prompt("a persona", "")
    assert SYSTEM_PROMPT in prompt


def test_build_prompt_contains_persona_and_dialogue() -> None:
    prompt = build_prompt("my persona string", "Doctor: Hi\nPatient: Hello")
    assert "my persona string" in prompt
    assert "Doctor: Hi" in prompt
    assert "Patient: Hello" in prompt
    assert "Generate the next response." in prompt


def test_build_prompt_no_curly_brace_leakage() -> None:
    """Curly braces would indicate JSON dict leakage from persona data.

    Note: apostrophes/quotes appear naturally in the system prompt
    ("patient's") — those are not JSON leakage.
    """
    prompt = build_prompt("a simple persona", "simple dialogue")
    for ch in "{}":
        assert ch not in prompt


# ---------------------------------------------------------------------------
# _has_json_leakage / _response_has_json_leakage
# ---------------------------------------------------------------------------


def test_has_json_leakage_detects_braces() -> None:
    assert _has_json_leakage('{"key": "value"}')
    assert _has_json_leakage("no braces here") is False


def test_has_json_leakage_detects_quotes() -> None:
    # Double quotes are JSON structural chars and must be flagged.
    assert _has_json_leakage('has "double" quotes')
    # Single quotes are legitimate natural-language punctuation (apostrophes,
    # possessives like "patient's") and are NOT flagged as JSON leakage.
    assert not _has_json_leakage("has 'single' quotes")
    assert _has_json_leakage("clean text") is False


def test_response_has_json_leakage_positive() -> None:
    assert _response_has_json_leakage('{"response": "data"}')


def test_response_has_json_leakage_negative() -> None:
    assert _response_has_json_leakage("a clean response with no special chars") is False


# ---------------------------------------------------------------------------
# is_chatml_compliant
# ---------------------------------------------------------------------------


def test_is_chatml_compliant_valid() -> None:
    assert is_chatml_compliant([{"role": "assistant", "content": "hi"}])


def test_is_chatml_compliant_empty() -> None:
    assert is_chatml_compliant([]) is False


def test_is_chatml_compliant_non_dict() -> None:
    assert is_chatml_compliant(["not a dict"]) is False  # type: ignore[list-item]


def test_is_chatml_compliant_bad_role() -> None:
    assert is_chatml_compliant([{"role": "narrator", "content": "hi"}]) is False


def test_is_chatml_compliant_non_string_content() -> None:
    assert is_chatml_compliant([{"role": "assistant", "content": 42}]) is False


# ---------------------------------------------------------------------------
# estimate_tokens_text
# ---------------------------------------------------------------------------


def test_estimate_tokens_text_positive() -> None:
    assert estimate_tokens_text("hello world") > 0


def test_estimate_tokens_text_ceil() -> None:
    # 8 chars / 4 chars-per-token = 2 tokens exactly
    assert estimate_tokens_text("12345678") == 2
    # 9 chars / 4 = 2.25 → ceil = 3
    assert estimate_tokens_text("123456789") == 3


# ---------------------------------------------------------------------------
# validate_token_bounds
# ---------------------------------------------------------------------------


def test_validate_token_bounds_ok() -> None:
    prompt = "short prompt"
    chosen = "short chosen"
    rejected = "short rejected"
    assert validate_token_bounds(prompt, chosen, rejected, max_tokens=DEFAULT_MAX_TOKENS)


def test_validate_token_bounds_exceeds() -> None:
    prompt = "x" * (DEFAULT_MAX_TOKENS * CHARS_PER_TOKEN)
    assert validate_token_bounds(prompt, "y", "z", max_tokens=DEFAULT_MAX_TOKENS) is False


def test_validate_token_bounds_uses_max_of_chosen_rejected() -> None:
    # chosen is small, rejected is huge
    prompt = "p"
    chosen = "c"
    rejected = "x" * (DEFAULT_MAX_TOKENS * CHARS_PER_TOKEN)
    assert validate_token_bounds(prompt, chosen, rejected, max_tokens=DEFAULT_MAX_TOKENS) is False


# ---------------------------------------------------------------------------
# build_dpo_pair
# ---------------------------------------------------------------------------


def test_build_dpo_pair_basic() -> None:
    example = build_dpo_pair(
        persona=_persona(),
        dialogue="Doctor: How are you?",
        chosen_response="I feel very tired, doctor.",
        rejected_response="Based on your presentation, I recommend intervention.",
    )
    assert isinstance(example, DpoPairExample)
    assert isinstance(example.prompt, str)
    assert len(example.chosen) == 1
    assert len(example.rejected) == 1
    assert example.chosen[0]["role"] == "assistant"
    assert example.rejected[0]["role"] == "assistant"
    assert example.chosen[0]["content"] == "I feel very tired, doctor."
    assert example.rejected[0]["content"] == "Based on your presentation, I recommend intervention."


def test_build_dpo_pair_prompt_contains_persona() -> None:
    example = build_dpo_pair(
        persona=_persona(age=42, location="Da Nang"),
        dialogue=None,
        chosen_response="I am tired.",
        rejected_response="I recommend pharmacological treatment.",
    )
    assert "42-year-old" in example.prompt
    assert "Da Nang" in example.prompt


def test_build_dpo_pair_rejects_empty_chosen() -> None:
    with pytest.raises(ValueError, match="chosen_response must be non-empty"):
        build_dpo_pair(_persona(), "dialogue", "", "rejected")


def test_build_dpo_pair_rejects_empty_rejected() -> None:
    with pytest.raises(ValueError, match="rejected_response must be non-empty"):
        build_dpo_pair(_persona(), "dialogue", "chosen", "")


def test_build_dpo_pair_rejects_identical_responses() -> None:
    with pytest.raises(ValueError, match="must differ"):
        build_dpo_pair(_persona(), "dialogue", "same response", "same response")


def test_build_dpo_pair_rejects_json_leakage_chosen() -> None:
    with pytest.raises(ValueError, match="chosen_response contains JSON"):
        build_dpo_pair(_persona(), "dialogue", '{"key": "value"}', "different")


def test_build_dpo_pair_rejects_json_leakage_rejected() -> None:
    # Double quotes are JSON structural chars; single quotes are not.
    with pytest.raises(ValueError, match="rejected_response contains JSON"):
        build_dpo_pair(_persona(), "dialogue", "clean chosen", 'has "double" quotes')


def test_build_dpo_pair_rejects_token_overflow() -> None:
    huge = "x" * (DEFAULT_MAX_TOKENS * CHARS_PER_TOKEN)
    with pytest.raises(ValueError, match="token bound"):
        build_dpo_pair(_persona(), "dialogue", huge, "short rejected")


def test_build_dpo_pair_metadata_fields() -> None:
    example = build_dpo_pair(
        persona=_persona(),
        dialogue="Doctor: Hi",
        chosen_response="chosen response text",
        rejected_response="rejected response text",
    )
    assert "persona_string" in example.metadata
    assert "n_dialogue_turns" in example.metadata
    assert "chosen_estimated_tokens" in example.metadata
    assert "rejected_estimated_tokens" in example.metadata
    assert "prompt_estimated_tokens" in example.metadata


def test_build_dpo_pair_n_dialogue_turns_list() -> None:
    dialogue = [
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": "Hi"},
    ]
    example = build_dpo_pair(_persona(), dialogue, "chosen", "rejected")
    assert example.metadata["n_dialogue_turns"] == 2


def test_build_dpo_pair_n_dialogue_turns_none() -> None:
    example = build_dpo_pair(_persona(), None, "chosen", "rejected")
    assert example.metadata["n_dialogue_turns"] == 0


def test_build_dpo_pair_n_dialogue_turns_string() -> None:
    example = build_dpo_pair(_persona(), "Doctor: Hi", "chosen", "rejected")
    assert example.metadata["n_dialogue_turns"] == 1


def test_build_dpo_pair_no_curly_brace_leakage_in_prompt() -> None:
    """Curly braces indicate JSON dict leakage; apostrophes are natural in English."""
    example = build_dpo_pair(_persona(), "Doctor: Hi", "chosen", "rejected")
    for ch in "{}":
        assert ch not in example.prompt


def test_build_dpo_pair_custom_max_tokens() -> None:
    example = build_dpo_pair(
        _persona(),
        "short",
        "chosen",
        "rejected",
        max_tokens=200,
    )
    assert validate_token_bounds(example.prompt, "chosen", "rejected", max_tokens=200)


# ---------------------------------------------------------------------------
# generate_dataset
# ---------------------------------------------------------------------------


def test_generate_dataset_roundtrip(tmp_path: Path) -> None:
    inp = tmp_path / "input.jsonl"
    out = tmp_path / "output.jsonl"
    _make_input_jsonl(inp, 5)

    # Use min_records=0 so we don't need 10K records for testing
    n = generate_dataset(inp, out, min_records=0)
    assert n == 5

    lines = out.read_text(encoding="utf-8").strip().split("\n")
    assert len(lines) == 5
    for line in lines:
        record = json.loads(line)
        assert "prompt" in record
        assert "chosen" in record
        assert "rejected" in record
        assert "metadata" in record
        assert isinstance(record["prompt"], str)
        assert isinstance(record["chosen"], list)
        assert isinstance(record["rejected"], list)
        assert len(record["chosen"]) == 1
        assert len(record["rejected"]) == 1
        assert record["chosen"][0]["role"] == "assistant"
        assert record["rejected"][0]["role"] == "assistant"
        assert record["chosen"][0]["content"] != record["rejected"][0]["content"]


def test_generate_dataset_reject_oversize(tmp_path: Path) -> None:
    inp = tmp_path / "input.jsonl"
    out = tmp_path / "output.jsonl"
    records = [
        {
            "persona": _persona(),
            "dialogue": "short",
            "chosen_response": "short chosen",
            "rejected_response": "short rejected",
        },
        {
            "persona": _persona(),
            "dialogue": "short",
            "chosen_response": "x" * (DEFAULT_MAX_TOKENS * CHARS_PER_TOKEN),
            "rejected_response": "short rejected",
        },
    ]
    _write_jsonl(inp, records)

    n = generate_dataset(inp, out, reject_oversize=True, min_records=0)
    assert n == 1  # only the first record passes


def test_generate_dataset_oversize_raises_without_flag(tmp_path: Path) -> None:
    inp = tmp_path / "input.jsonl"
    out = tmp_path / "output.jsonl"
    records = [
        {
            "persona": _persona(),
            "dialogue": "short",
            "chosen_response": "x" * (DEFAULT_MAX_TOKENS * CHARS_PER_TOKEN),
            "rejected_response": "short rejected",
        },
    ]
    _write_jsonl(inp, records)

    with pytest.raises(ValueError, match="token bound"):
        generate_dataset(inp, out, reject_oversize=False, min_records=0)


def test_generate_dataset_below_minimum_raises(tmp_path: Path) -> None:
    inp = tmp_path / "input.jsonl"
    out = tmp_path / "output.jsonl"
    _make_input_jsonl(inp, 3)

    with pytest.raises(ValueError, match="below minimum 10000"):
        generate_dataset(inp, out, min_records=DEFAULT_MIN_RECORDS)


def test_generate_dataset_identical_responses_raises(tmp_path: Path) -> None:
    inp = tmp_path / "input.jsonl"
    out = tmp_path / "output.jsonl"
    records = [
        {
            "persona": _persona(),
            "dialogue": "dialogue",
            "chosen_response": "same response",
            "rejected_response": "same response",
        },
    ]
    _write_jsonl(inp, records)

    with pytest.raises(ValueError, match="must differ"):
        generate_dataset(inp, out, min_records=0)


def test_generate_dataset_skips_blank_lines(tmp_path: Path) -> None:
    inp = tmp_path / "input.jsonl"
    out = tmp_path / "output.jsonl"
    with inp.open("w", encoding="utf-8") as f:
        f.write(
            json.dumps({"persona": _persona(), "dialogue": "d", "chosen_response": "c", "rejected_response": "r"})
            + "\n"
        )
        f.write("\n")
        f.write(
            json.dumps({"persona": _persona(), "dialogue": "d", "chosen_response": "c2", "rejected_response": "r2"})
            + "\n"
        )

    n = generate_dataset(inp, out, min_records=0)
    assert n == 2


def test_generate_dataset_preserves_unicode(tmp_path: Path) -> None:
    inp = tmp_path / "input.jsonl"
    out = tmp_path / "output.jsonl"
    records = [
        {
            "persona": _persona(location="Hà Nội"),
            "dialogue": "Bác sĩ: Bạn khỏe không? 🇻🇳",
            "chosen_response": "Tôi thấy mệt lắm, không hiểu từ khó. 🇻🇳",
            "rejected_response": "Dựa trên lâm sàng, tôi khuyên dùng thuốc. 🇻🇳",
        },
    ]
    _write_jsonl(inp, records)

    n = generate_dataset(inp, out, min_records=0)
    assert n == 1
    content = out.read_text(encoding="utf-8")
    assert "Hà Nội" in content
    assert "🇻🇳" in content
    assert "Tôi thấy mệt lắm" in content


# ---------------------------------------------------------------------------
# main (CLI)
# ---------------------------------------------------------------------------


def test_main_missing_input_returns_1(tmp_path: Path) -> None:
    inp = tmp_path / "nonexistent.jsonl"
    out = tmp_path / "output.jsonl"
    rc = main([str(inp), str(out)])
    assert rc == 1


def test_main_success_returns_0(tmp_path: Path) -> None:
    inp = tmp_path / "input.jsonl"
    out = tmp_path / "output.jsonl"
    _make_input_jsonl(inp, 5)
    rc = main([str(inp), str(out), "--min-records", "0"])
    assert rc == 0
    lines = out.read_text(encoding="utf-8").strip().split("\n")
    assert len(lines) == 5


def test_main_below_minimum_raises(tmp_path: Path) -> None:
    inp = tmp_path / "input.jsonl"
    out = tmp_path / "output.jsonl"
    _make_input_jsonl(inp, 2)
    with pytest.raises(ValueError, match="below minimum"):
        main([str(inp), str(out)])


def test_main_reject_oversize(tmp_path: Path) -> None:
    inp = tmp_path / "input.jsonl"
    out = tmp_path / "output.jsonl"
    records = [
        {
            "persona": _persona(),
            "dialogue": "short",
            "chosen_response": "short chosen",
            "rejected_response": "short rejected",
        },
        {
            "persona": _persona(),
            "dialogue": "short",
            "chosen_response": "x" * (DEFAULT_MAX_TOKENS * CHARS_PER_TOKEN),
            "rejected_response": "short",
        },
    ]
    _write_jsonl(inp, records)
    rc = main([str(inp), str(out), "--reject-oversize", "--min-records", "0"])
    assert rc == 0
    lines = out.read_text(encoding="utf-8").strip().split("\n")
    assert len(lines) == 1
