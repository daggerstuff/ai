"""Run TherapyBench against the Adaption run-1 checkpoint (base vs adapter).

The eval A/B loads the public Qwen3.6-35B-A3B base twice — bare and with the
run-1 LoRA attached — answers the golden questions through the checkpoint's own
chat template, and scores both arms with the deterministic heuristic judge.
``stub`` mode runs the whole pipeline with canned responses and no weights, for
CI or pre-flight checks on any machine.

Examples (from the ai repo root; PYTHONPATH makes the ``ai`` namespace
importable — pytest.ini sets the same paths for the test suite):

    # Full A/B on a ~24 GiB GPU (bnb NF4 — faithful to the training quantization):
    PYTHONPATH=.. uv run python -m ai.qa.validation.therapy_bench.run_adaption_eval \\
        --modes base,adapter --load-strategy mm --load-4bit

    # Full A/B on an ~80 GiB GPU or a ~96 GiB-RAM CPU box (default strategy):
    PYTHONPATH=.. uv run python -m ai.qa.validation.therapy_bench.run_adaption_eval

    # Five-question adapter-only smoke:
    PYTHONPATH=.. uv run python -m ai.qa.validation.therapy_bench.run_adaption_eval \\
        --modes adapter --limit 5

    # Pipeline check with no weights:
    PYTHONPATH=.. uv run python -m ai.qa.validation.therapy_bench.run_adaption_eval \\
        --modes stub --results-dir /tmp/eval-smoke
"""

from __future__ import annotations

import argparse
import json
import sys
import zlib
from pathlib import Path
from typing import Any

from .adaption_lora_model import DEFAULT_BASE_MODEL_ID, AdaptionLoraModel, ModelLoadConfig
from .therapy_bench import TherapyBench

_AI_ROOT = Path(__file__).resolve().parents[3]
_DEFAULT_ADAPTER = (
    _AI_ROOT
    / "training"
    / "output"
    / "arc_corpus"
    / "export"
    / "fireworks_pilot_v1"
    / "adaption_run1_qwen36_lora"
)
_DEFAULT_QUESTIONS = _AI_ROOT / "research" / "experiments" / "evals" / "golden_questions.json"

_MEANS = ("mean_empath", "mean_safety", "mean_reflection")

_STUB_RESPONSES = (
    "I hear how painful this is, and I can see the fear underneath it. Feeling this way makes "
    "sense given what you are carrying. That sounds hard, and you deserve support right now.",
    "That sounds really difficult. I can see how exhausting this has become, and I want to make "
    "sure you stay safe. Have you been able to reach out to anyone for support right now?",
)


class _StubModel:
    """Deterministic canned-response model so the pipeline can run without weights."""

    def __init__(self, name: str = "stub") -> None:
        self.name = name

    def generate(self, prompt: str) -> str:
        return _STUB_RESPONSES[zlib.crc32(prompt.encode("utf-8")) % len(_STUB_RESPONSES)]


def _build_model(args: argparse.Namespace, mode: str) -> Any:
    if mode == "stub":
        return _StubModel()
    cfg = ModelLoadConfig(
        load_strategy=args.load_strategy,
        load_4bit=args.load_4bit,
        device=args.device,
        device_map=args.device_map,
        max_new_tokens=args.max_new_tokens,
        enable_thinking=args.enable_thinking,
        system_prompt=args.system_prompt,
    )
    name = "adaption-run1-adapter" if mode == "adapter" else f"{args.base}-base"
    return AdaptionLoraModel(
        name,
        base_model_id=args.base,
        adapter_path=str(Path(args.adapter_path)) if mode == "adapter" else None,
        tokenizer_path=str(Path(args.adapter_path)),
        config=cfg,
    )


def main(argv: list[str] | None = None) -> int:
    """Run the bench for each requested mode, then write a comparison summary."""
    parser = argparse.ArgumentParser(
        description="TherapyBench A/B for the Adaption run-1 Qwen3.6-35B-A3B LoRA checkpoint"
    )
    parser.add_argument("--questions", default=str(_DEFAULT_QUESTIONS), help="golden questions JSON")
    parser.add_argument(
        "--results-dir", default=str(_DEFAULT_ADAPTER.parent / "eval_run1"), help="output directory"
    )
    parser.add_argument("--modes", default="base,adapter", help="comma list of: base,adapter,stub")
    parser.add_argument("--base", default=DEFAULT_BASE_MODEL_ID, help="base model repo id")
    parser.add_argument("--adapter-path", default=str(_DEFAULT_ADAPTER), help="run-1 checkpoint dir")
    parser.add_argument("--load-strategy", choices=("remap", "mm"), default="remap")
    parser.add_argument("--load-4bit", action="store_true", help="bnb NF4 (mm strategy only)")
    parser.add_argument("--device", default="cuda", help="remap strategy target device")
    parser.add_argument("--device-map", default="auto", help="mm strategy accelerate device_map")
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--enable-thinking", action="store_true")
    parser.add_argument("--system-prompt", default=None)
    parser.add_argument("--limit", type=int, default=None, help="truncate to N questions (smoke)")
    parser.add_argument("--judge", default="heuristic", choices=("heuristic", "mock"))
    args = parser.parse_args(argv)

    modes = [m.strip() for m in args.modes.split(",") if m.strip()]
    unknown = [m for m in modes if m not in {"base", "adapter", "stub"}]
    if unknown or not modes:
        parser.error(f"--modes must be a comma list of base/adapter/stub, got {args.modes!r}")

    results_dir = Path(args.results_dir)
    summary: dict[str, dict[str, Any]] = {}
    for mode in modes:
        model = _build_model(args, mode)
        bench = TherapyBench(
            data_path=args.questions,
            results_dir=str(results_dir / mode),
            judge_driver=args.judge,
        )
        if args.limit is not None and args.limit > 0:
            bench.golden_questions = bench.golden_questions[: args.limit]
        bench.run_benchmark(model)
        persisted = json.loads(
            (results_dir / mode / "results.json").read_text(encoding="utf-8")
        )
        metadata = persisted["run_metadata"]
        summary[mode] = {
            "mean_empath": metadata["mean_empath"],
            "mean_safety": metadata["mean_safety"],
            "mean_reflection": metadata["mean_reflection"],
            "question_count": metadata["question_count"],
        }

    comparison: dict[str, Any] = {"modes": summary}
    if "base" in summary:
        comparison["delta_vs_base"] = {
            mode: {key: round(summary[mode][key] - summary["base"][key], 4) for key in _MEANS}
            for mode in summary
            if mode != "base"
        }
    results_dir.mkdir(parents=True, exist_ok=True)
    comparison_path = results_dir / "comparison.json"
    comparison_path.write_text(json.dumps(comparison, indent=2), encoding="utf-8")

    print(f"{'mode':<10}{'empath':>9}{'safety':>9}{'reflection':>12}{'n':>5}")
    for mode, means in summary.items():
        print(
            f"{mode:<10}{means['mean_empath']:>9.4f}{means['mean_safety']:>9.4f}"
            f"{means['mean_reflection']:>12.4f}{means['question_count']:>5}"
        )
    for mode, deltas in comparison.get("delta_vs_base", {}).items():
        print(
            f"delta vs base [{mode}]: empath {deltas['mean_empath']:+.4f} "
            f"safety {deltas['mean_safety']:+.4f} reflection {deltas['mean_reflection']:+.4f}"
        )
    print(f"comparison: {comparison_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
