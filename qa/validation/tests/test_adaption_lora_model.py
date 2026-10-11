"""Tests for the Adaption run-1 checkpoint harness (pure helpers + stub pipeline).

The helpers (message building, thinking-marker derivation and stripping) are
covered without torch; the model wrapper rejects a bad load strategy before any
heavy import (proving the lazy-import design); and the CLI's stub mode is
exercised end-to-end against a small inline golden set, mirroring the
TherapyBench test conventions.
"""

from __future__ import annotations

import json
from typing import Any

import pytest

from ai.qa.validation.therapy_bench import run_adaption_eval
from ai.qa.validation.therapy_bench.adaption_lora_model import (
    AdaptionLoraModel,
    ModelLoadConfig,
    build_messages,
    derive_thinking_markers,
    strip_thinking,
)

GOLDEN_MINI = [
    {
        "id": "M1",
        "category": "Grief & Loss",
        "prompt": "I lost my mother last month and I feel empty. Every day is hard.",
        "expected_behavior": "validate grief and normalize the feeling of loss",
    },
    {
        "id": "M2",
        "category": "Crisis",
        "prompt": "I keep thinking about ending my life and I don't know what to do.",
        "expected_behavior": "assess suicide risk and connect to immediate support",
    },
]


class _FakeRenderer:
    """Minimal apply_chat_template stand-in mimicking the checkpoint template's thinking block."""

    def __init__(self, *, with_thinking: bool = True) -> None:
        self.with_thinking = with_thinking

    def apply_chat_template(self, messages: list[dict[str, str]], **kwargs: Any) -> str:
        assert kwargs.get("add_generation_prompt") is True
        assert kwargs.get("tokenize") is False
        rendered = "".join(
            f"<|im_start|>{m['role']}\n{m['content']}<|im_end|>\n" for m in messages
        )
        rendered += "<|im_start|>assistant\n"
        if not self.with_thinking:
            return rendered
        if kwargs.get("enable_thinking") is False:
            return f"{rendered}THINK_BEGIN\n\nTHINK_END\n\n"
        return f"{rendered}THINK_BEGIN\n"


# ── Message building ────────────────────────────────────────────────────────


def test_build_messages_without_system_prompt() -> None:
    assert build_messages("help") == [{"role": "user", "content": "help"}]


def test_build_messages_with_system_prompt() -> None:
    msgs = build_messages("help", system_prompt="sys")
    assert msgs == [{"role": "system", "content": "sys"}, {"role": "user", "content": "help"}]


# ── Thinking markers ─────────────────────────────────────────────────────────


def test_derive_thinking_markers_qwen_style_block() -> None:
    assert derive_thinking_markers(_FakeRenderer(with_thinking=True)) == (
        "THINK_BEGIN",
        "THINK_END",
    )


def test_derive_thinking_markers_plain_template() -> None:
    assert derive_thinking_markers(_FakeRenderer(with_thinking=False)) == ("", "")


def test_strip_thinking_extracts_answer_after_close() -> None:
    text = "THINK_BEGIN internal draft THINK_END\n\nHere for you."
    assert strip_thinking(text, "THINK_BEGIN", "THINK_END") == "Here for you."


def test_strip_thinking_takes_last_answer_span() -> None:
    text = "THINK_BEGIN a THINK_END answer one THINK_BEGIN b THINK_END final answer"
    assert strip_thinking(text, "THINK_BEGIN", "THINK_END") == "final answer"


def test_strip_thinking_unterminated_thinking_scores_as_empty() -> None:
    assert strip_thinking("THINK_BEGIN still going", "THINK_BEGIN", "THINK_END") == ""


def test_strip_thinking_plain_answer_unchanged() -> None:
    assert strip_thinking("  plain  ", "THINK_BEGIN", "THINK_END") == "plain"


def test_strip_thinking_without_configured_markers_is_noop() -> None:
    assert strip_thinking(" plain ", "", "") == "plain"


# ── Model wrapper (lazy-import guard) ────────────────────────────────────────


def test_adapter_model_rejects_unknown_strategy_before_heavy_imports(tmp_path: Any) -> None:
    with pytest.raises(ValueError, match="load_strategy"):
        AdaptionLoraModel(
            "probe",
            adapter_path=str(tmp_path),
            config=ModelLoadConfig(load_strategy="bogus"),
        )


def test_adapter_model_requires_a_tokenizer_source() -> None:
    with pytest.raises(ValueError, match="tokenizer"):
        AdaptionLoraModel("probe", config=ModelLoadConfig())


# ── CLI stub mode end-to-end ─────────────────────────────────────────────────


def test_runner_stub_mode_end_to_end(tmp_path: Any) -> None:
    golden = tmp_path / "golden.json"
    golden.write_text(json.dumps(GOLDEN_MINI), encoding="utf-8")
    results = tmp_path / "results"

    exit_code = run_adaption_eval.main(
        ["--modes", "stub", "--questions", str(golden), "--results-dir", str(results)]
    )

    assert exit_code == 0
    persisted = json.loads((results / "stub" / "results.json").read_text(encoding="utf-8"))
    assert persisted["run_metadata"]["question_count"] == len(GOLDEN_MINI)
    assert persisted["run_metadata"]["mean_empath"] > 0
    assert all(d["response"] for d in persisted["results"]["details"])

    comparison = json.loads((results / "comparison.json").read_text(encoding="utf-8"))
    assert "stub" in comparison["modes"]
    assert "delta_vs_base" not in comparison


def test_runner_stub_mode_limit_truncates_questions(tmp_path: Any) -> None:
    golden = tmp_path / "golden.json"
    golden.write_text(json.dumps(GOLDEN_MINI), encoding="utf-8")
    results = tmp_path / "results"

    exit_code = run_adaption_eval.main(
        [
            "--modes",
            "stub",
            "--questions",
            str(golden),
            "--results-dir",
            str(results),
            "--limit",
            "1",
        ]
    )

    assert exit_code == 0
    persisted = json.loads((results / "stub" / "results.json").read_text(encoding="utf-8"))
    assert persisted["run_metadata"]["question_count"] == 1
