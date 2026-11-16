"""The local transport only has to satisfy the shared runner's contract, so it is cheap to pin down."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

from qa.validation.therapy_bench import paired_eval
from training import local_paired_eval as local

DEV_CASES = Path(__file__).resolve().parents[3] / "docs/training/execution-2026-10-17/scenario-crisis-dev-v1.jsonl"


class EchoTransport:
    """Answer every case with an escalation so the grading path runs without a model."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, list[dict[str, Any]]]] = []

    def complete(
        self,
        arm: dict[str, str],
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        decoding: dict[str, Any],
    ) -> dict[str, Any]:
        assert tools == []
        self.calls.append((arm["id"], messages))
        return {
            "response": "Call 911 right now and stay on the line with them.",
            "tool_calls": [],
            "finish_reason": "stop",
            "usage": None,
        }


def test_the_generated_config_passes_the_shared_loader(tmp_path: Path) -> None:
    config = local.build_config("run-1", "Qwen/Qwen3-8B", "/adapters/final", 512, False)
    path = tmp_path / "config.json"
    path.write_text(json.dumps(config), encoding="utf-8")

    loaded = paired_eval.load_config(path)

    assert loaded["run_id"] == "run-1"
    assert loaded["decoding"]["temperature"] == 0
    assert loaded["decoding"]["chat_template_kwargs"] == {"enable_thinking": False}
    assert loaded["pricing"]["require_finish_reason"] is True
    assert loaded["pricing"]["require_usage"] is False


def test_the_two_arms_share_the_base_model_and_differ_only_by_the_adapter() -> None:
    with_adapter = local.build_config("run-1", "Qwen/Qwen3-8B", "/adapters/final", 512, False)
    without_adapter = local.build_config("run-1", "Qwen/Qwen3-8B", None, 512, False)

    assert [arm["id"] for arm in with_adapter["arms"]] == ["base", "sft"]
    assert [arm["model"] for arm in with_adapter["arms"]] == ["Qwen/Qwen3-8B", "/adapters/final"]
    assert [arm["model"] for arm in without_adapter["arms"]] == ["Qwen/Qwen3-8B", "Qwen/Qwen3-8B"]
    assert {arm["api_key_env"] for arm in with_adapter["arms"]} == {local.LOCAL_API_KEY_ENV}


def test_strip_stops_cuts_at_the_first_marker_and_trims() -> None:
    assert local.strip_stops("  a reply <|im_end|> trailing  ", ["<|im_end|>", "<|endoftext|>"]) == "a reply"
    assert local.strip_stops("a reply <|endoftext|><|im_end|>", ["<|im_end|>", "<|endoftext|>"]) == "a reply"
    assert local.strip_stops("a reply", ["<|im_end|>"]) == "a reply"


def test_the_cli_can_score_an_adapter_on_the_base_it_was_trained_on(tmp_path: Path, monkeypatch: Any) -> None:
    seen: list[bool] = []

    class Recorder(EchoTransport):
        def __init__(self, *args: Any, load_in_4bit: bool = False) -> None:
            seen.append(load_in_4bit)
            super().__init__()

    monkeypatch.setattr(local, "LocalChatTransport", Recorder)

    def run(extra: list[str]) -> int:
        monkeypatch.setattr(
            sys,
            "argv",
            [
                "local_paired_eval",
                "--base-model",
                "Qwen/Qwen3-8B",
                "--adapter",
                "/adapters/nf4",
                "--cases",
                str(DEV_CASES),
                "--results-dir",
                str(tmp_path / f"results-{len(seen)}"),
                "--arm",
                "base",
                *extra,
            ],
        )
        return local.main()

    assert run([]) == 0
    assert run(["--load-in-4bit"]) == 0

    # The NF4 adapter was fitted against a 4-bit base, so the flag has to reach
    # the loader instead of being accepted and dropped on the floor.
    assert seen == [False, True]


def test_a_transport_scores_the_real_dev_split_end_to_end(tmp_path: Path) -> None:
    config_path = tmp_path / "config.json"
    config_path.write_text(
        json.dumps(local.build_config("run-1", "Qwen/Qwen3-8B", "/adapters/final", 512, False)),
        encoding="utf-8",
    )
    transport = EchoTransport()
    request = paired_eval.RunRequest(
        DEV_CASES,
        config_path,
        tmp_path / "results",
        paired_eval.RunnerOptions(),
    )

    result = paired_eval.run_paired(request, transport=transport)

    assert result["status"] == "complete"
    assert result["success_count"] == 24  # 12 dev cases scored on both arms
    assert {arm for arm, _ in transport.calls} == {"base", "sft"}
    records = [json.loads(line) for line in (tmp_path / "results" / "results.jsonl").read_text().splitlines()]
    crisis = [record for record in records if record["scenario"] == "crisis"]
    assert crisis
    assert all(isinstance(metric["passed"], bool) for record in crisis for metric in record["state_metrics"])
    assert any(metric["passed"] for record in crisis for metric in record["state_metrics"])
    graded = {
        (record["arm_id"], record["case_id"]): [metric["passed"] for metric in record["state_metrics"]]
        for record in crisis
    }
    for (arm_id, case_id), passes in graded.items():
        if arm_id == "sft":
            assert passes == graded[("base", case_id)]
