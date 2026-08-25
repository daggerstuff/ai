#!/usr/bin/env python3
"""Pre/post-SFT benchmark harness (PIX-4345 Appendix E Step 1).

Runs a model on standard + domain benchmarks, saves results to
``benchmarks/pre_train_YYYY-MM-DD.json`` (or post_train). Computes the
forgetting metric when both pre and post runs exist:

    forgetting_score = (pre_score - post_score) / pre_score

Target: <10% forgetting on general benchmarks; <5% preferred for production.

Real eval (default):
  - General benchmarks (MMLU / HellaSwag / TruthfulQA / BBH) run through
    lm-evaluation-harness (``lm_eval.simple_evaluate``).
  - Domain benchmark (``domain_clinical_empathy``) runs against the real
    DiagnosisArena clinical-case set (``ai/qa/validation/diagnosis_arena``)
    as an MCQ accuracy harness.

Mock mode (``--mock``):
  - No GPU/real model required; emits deterministic scores so the run +
    report + forgetting-comparison topology can be smoke-tested on CPU.
  - Must be passed explicitly so mock numbers can never masquerade as a
    real eval report.

Blueprint ref: docs/training-pipeline-blueprint-2026-08-10.md @138-140, @646-667.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Any

# Standard benchmarks (general capability — forgetting watch list)
GENERAL_BENCHMARKS = ("mmlu", "hellaswag", "truthfulqa", "bbh")
# Domain benchmark backed by real clinical cases (DiagnosisArena). The legacy
# ``domain_safety_gate`` entry was a rule engine (ClinicalSafetyGate), not a
# scored test set, so it is intentionally not part of the eval harness.
DOMAIN_BENCHMARKS = ("domain_clinical_empathy",)
FORGETTING_THRESHOLD_GENERAL = 0.10  # <10% on general
FORGETTING_THRESHOLD_PRODUCTION = 0.05  # <5% preferred

# DiagnosisArena clinical-case fixture (real MCQ ground truth).
_DIAGNOSIS_ARENA_CASES = (
    Path(__file__).resolve().parents[1]
    / "qa"
    / "validation"
    / "diagnosis_arena"
    / "fixtures"
    / "seed_cases.jsonl"
)

# Deterministic mock scores (mock mode only). Same benchmark -> same score so
# forgetting math stays stable across pre/post smoke runs.
_MOCK_SCORES = {
    "mmlu": 0.72,
    "hellaswag": 0.78,
    "truthfulqa": 0.55,
    "bbh": 0.61,
    "domain_clinical_empathy": 0.48,
}


@dataclass
class BenchmarkResult:
    name: str
    score: float  # 0.0-1.0
    n_samples: int
    category: str  # "general" or "domain"
    runtime_s: float = 0.0
    notes: str = ""


@dataclass
class BenchmarkRun:
    model_name: str
    run_phase: str  # "pre_train" or "post_train"
    results: list[BenchmarkResult] = field(default_factory=list)
    run_date: str | None = None  # ISO date of the run; None stamps serialize-time

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_name": self.model_name,
            "run_phase": self.run_phase,
            "date": self.run_date or date.today().isoformat(),
            "results": [
                {
                    "name": r.name,
                    "score": r.score,
                    "n_samples": r.n_samples,
                    "category": r.category,
                    "runtime_s": round(r.runtime_s, 4),
                    "notes": r.notes,
                }
                for r in self.results
            ],
            "summary": self._summary(),
        }

    def _summary(self) -> dict[str, Any]:
        general = [r for r in self.results if r.category == "general"]
        domain = [r for r in self.results if r.category == "domain"]
        return {
            "general_avg": round(sum(r.score for r in general) / max(len(general), 1), 4),
            "domain_avg": round(sum(r.score for r in domain) / max(len(domain), 1), 4),
            "general_count": len(general),
            "domain_count": len(domain),
        }


# ---------------------------------------------------------------------------
# Mock backend
# ---------------------------------------------------------------------------

def _mock_score(benchmark: str) -> float:
    """Deterministic mock score (``--mock`` mode only)."""
    return _MOCK_SCORES.get(benchmark, 0.50)


def run_benchmarks_mock(
    model_name: str,
    run_phase: str,
    benchmarks: tuple[str, ...],
    domain_benchmarks: tuple[str, ...],
    n_samples: int = 500,
) -> BenchmarkRun:
    """Run all benchmarks in mock mode (no model, deterministic scores)."""
    import time

    run = BenchmarkRun(model_name=model_name, run_phase=run_phase)

    for name in benchmarks:
        t0 = time.perf_counter()
        score = _mock_score(name)
        elapsed = time.perf_counter() - t0
        run.results.append(
            BenchmarkResult(
                name=name,
                score=score,
                n_samples=n_samples,
                category="general",
                runtime_s=elapsed,
                notes="mock (CPU)",
            )
        )

    for name in domain_benchmarks:
        t0 = time.perf_counter()
        score = _mock_score(name)
        elapsed = time.perf_counter() - t0
        run.results.append(
            BenchmarkResult(
                name=name,
                score=score,
                n_samples=n_samples,
                category="domain",
                runtime_s=elapsed,
                notes="mock (CPU)",
            )
        )

    return run


# ---------------------------------------------------------------------------
# Real backend: lm-evaluation-harness (general) + DiagnosisArena (domain)
# ---------------------------------------------------------------------------

def _extract_lm_eval_score(task_results: dict[str, Any]) -> float:
    """Pull a normalized 0.0-1.0 score out of an lm-eval task result dict.

    Prefers ``acc_norm,none`` (length-normalized accuracy, the canonical MC
    score), then ``acc,none``, then any numeric metric.
    """
    for key in ("acc_norm,none", "acc,none"):
        if key in task_results and isinstance(task_results[key], (int, float)):
            return float(task_results[key])
    # Fallback: first numeric metric value.
    for value in task_results.values():
        if isinstance(value, (int, float)):
            return float(value)
    return 0.0


def run_general_benchmarks_real(
    model_name: str,
    run_phase: str,
    model: str,
    model_args: str,
    benchmarks: tuple[str, ...],
    n_samples: int,
) -> BenchmarkRun:
    """Run general benchmarks through lm-evaluation-harness."""
    import time

    from lm_eval import simple_evaluate

    run = BenchmarkRun(model_name=model_name, run_phase=run_phase)

    t0 = time.perf_counter()
    results = simple_evaluate(
        model="hf",
        model_args=f"pretrained={model},{model_args}" if model_args else f"pretrained={model}",
        tasks=list(benchmarks),
        num_fewshot=0,
        batch_size=8,
        limit=n_samples,
        log_samples=False,
    )
    elapsed = time.perf_counter() - t0

    task_results = results.get("results", {})
    for name in benchmarks:
        score = _extract_lm_eval_score(task_results.get(name, {}))
        run.results.append(
            BenchmarkResult(
                name=name,
                score=score,
                n_samples=n_samples,
                category="general",
                runtime_s=elapsed / max(len(benchmarks), 1),
                notes="lm-eval (real)",
            )
        )

    return run


def _load_diagnosis_cases(path: Path) -> list[dict[str, Any]]:
    """Load DiagnosisArena MCQ cases (one JSON object per line)."""
    cases: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        cases.append(json.loads(stripped))
    return cases


def _mcq_prompt(case: dict[str, Any]) -> str:
    """Build an MCQ prompt for a DiagnosisArena case."""
    options = case.get("mcq_options", [])
    letters = [chr(ord("A") + i) for i in range(len(options))]
    body = "\n".join(
        part
        for part in (
            f"Presentation: {case.get('presentation', '')}".strip(),
            f"History: {case.get('history', '')}".strip() if case.get("history") else "",
            f"Exam: {case.get('exam', '')}".strip() if case.get("exam") else "",
            f"Labs: {case.get('labs', '')}".strip() if case.get("labs") else "",
            f"Imaging: {case.get('imaging', '')}".strip() if case.get("imaging") else "",
        )
        if part
    )
    options_block = "\n".join(
        f"{letter}) {opt}" for letter, opt in zip(letters, options)
    )
    return (
        f"{body}\n\n"
        f"Options:\n{options_block}\n\n"
        "Which is the most likely diagnosis? Answer with only the option letter."
    )


def run_domain_benchmark_real(
    model_name: str,
    run_phase: str,
    model: str,
    n_samples: int,
) -> BenchmarkRun:
    """Run the domain benchmark (DiagnosisArena MCQ accuracy) with a real model."""
    import time

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    run = BenchmarkRun(model_name=model_name, run_phase=run_phase)

    cases = _load_diagnosis_cases(_DIAGNOSIS_ARENA_CASES)[:n_samples]
    if not cases:
        raise FileNotFoundError(f"No DiagnosisArena cases at {_DIAGNOSIS_ARENA_CASES}")

    tokenizer = AutoTokenizer.from_pretrained(model)
    lm = AutoModelForCausalLM.from_pretrained(model, device_map="auto")
    lm.eval()

    correct = 0
    t0 = time.perf_counter()
    for case in cases:
        prompt = _mcq_prompt(case)
        inputs = tokenizer(prompt, return_tensors="pt").to(lm.device)
        with torch.inference_mode():
            out = lm.generate(**inputs, max_new_tokens=4, do_sample=False)
        decoded = tokenizer.decode(out[0, inputs["input_ids"].shape[-1] :], skip_special_tokens=True)
        predicted = decoded.strip().upper()[:1]

        options = case.get("mcq_options", [])
        letters = [chr(ord("A") + i) for i in range(len(options))]
        # Ground truth is a diagnosis string; match by the option letter holding it.
        gt_letter = ""
        for i, opt in enumerate(options):
            if opt == case.get("final_diagnosis"):
                gt_letter = letters[i]
                break

        if predicted and gt_letter and predicted == gt_letter:
            correct += 1

    elapsed = time.perf_counter() - t0
    accuracy = correct / len(cases)
    run.results.append(
        BenchmarkResult(
            name="domain_clinical_empathy",
            score=round(accuracy, 4),
            n_samples=len(cases),
            category="domain",
            runtime_s=elapsed,
            notes="DiagnosisArena MCQ (real)",
        )
    )
    return run


def run_benchmarks(
    model_name: str,
    run_phase: str,
    benchmarks: tuple[str, ...],
    domain_benchmarks: tuple[str, ...],
    n_samples: int = 500,
    *,
    mock: bool = False,
    model: str = "mock_base_model",
    model_args: str = "",
) -> BenchmarkRun:
    """Run all benchmarks (real by default, mock if ``mock=True``)."""
    if mock:
        return run_benchmarks_mock(
            model_name, run_phase, benchmarks, domain_benchmarks, n_samples
        )

    run = run_general_benchmarks_real(
        model_name, run_phase, model, model_args, benchmarks, n_samples
    )
    for _name in domain_benchmarks:
        run.results.extend(
            run_domain_benchmark_real(model_name, run_phase, model, n_samples).results
        )
    return run


# ---------------------------------------------------------------------------
# Forgetting metric
# ---------------------------------------------------------------------------

def compute_forgetting(pre: BenchmarkRun, post: BenchmarkRun) -> dict[str, Any]:
    """forgetting_score = (pre - post) / pre per benchmark + aggregate.

    Negative forgetting (post > pre) = improvement, clamped to 0.
    """
    pre_by_name = {r.name: r.score for r in pre.results}
    post_by_name = {r.name: r.score for r in post.results}

    per_benchmark: dict[str, dict[str, Any]] = {}
    general_forgetting: list[float] = []
    domain_forgetting: list[float] = []

    for name, pre_score in pre_by_name.items():
        post_score = post_by_name.get(name)
        if post_score is None:
            per_benchmark[name] = {"pre": pre_score, "post": None, "forgetting": None}
            continue
        forgetting = (pre_score - post_score) / pre_score if pre_score > 0 else 0.0
        forgetting = max(forgetting, 0.0)  # improvement = 0 forgetting
        category = "general" if name in GENERAL_BENCHMARKS else "domain"
        per_benchmark[name] = {
            "pre": round(pre_score, 4),
            "post": round(post_score, 4),
            "forgetting": round(forgetting, 4),
            "exceeds_general_threshold": category == "general" and forgetting > FORGETTING_THRESHOLD_GENERAL,
            "exceeds_production_threshold": category == "general" and forgetting > FORGETTING_THRESHOLD_PRODUCTION,
        }
        if category == "general":
            general_forgetting.append(forgetting)
        else:
            domain_forgetting.append(forgetting)

    avg_general = sum(general_forgetting) / max(len(general_forgetting), 1)
    avg_domain = sum(domain_forgetting) / max(len(domain_forgetting), 1)

    return {
        "model": pre.model_name,
        "pre_train_date": pre.to_dict()["date"],
        "post_train_date": post.to_dict()["date"],
        "per_benchmark": per_benchmark,
        "avg_general_forgetting": round(avg_general, 4),
        "avg_domain_forgetting": round(avg_domain, 4),
        "general_threshold": FORGETTING_THRESHOLD_GENERAL,
        "production_threshold": FORGETTING_THRESHOLD_PRODUCTION,
        "general_gate_passed": avg_general <= FORGETTING_THRESHOLD_GENERAL,
        "production_gate_passed": avg_general <= FORGETTING_THRESHOLD_PRODUCTION,
        "verdict": _forgetting_verdict(avg_general),
    }


def _forgetting_verdict(avg_general: float) -> str:
    if avg_general <= FORGETTING_THRESHOLD_PRODUCTION:
        return "PASS (production-grade, <5% forgetting)"
    if avg_general <= FORGETTING_THRESHOLD_GENERAL:
        return "PASS (general, <10% forgetting; consider replay for production)"
    return f"FAIL ({avg_general:.1%} forgetting exceeds {FORGETTING_THRESHOLD_GENERAL:.0%} threshold)"


def main() -> int:
    parser = argparse.ArgumentParser(description="Pre/post-SFT benchmark harness (PIX-4345 Appendix E)")
    parser.add_argument("--model", type=str, default="mock_base_model", help="Model name or HF path (real mode)")
    parser.add_argument("--model-args", type=str, default="", help="Extra lm-eval model_args (key=value, comma-separated)")
    parser.add_argument(
        "--phase",
        type=str,
        choices=["pre_train", "post_train"],
        default="pre_train",
        help="pre_train (baseline) or post_train (after SFT)",
    )
    parser.add_argument(
        "--out-dir", type=str, default="ai/training/benchmarks", help="Directory for benchmark JSON files"
    )
    parser.add_argument("--n-samples", type=int, default=500, help="Samples per benchmark")
    parser.add_argument(
        "--compare", type=str, default=None, help="Path to a pre_train report to compare against (computes forgetting)"
    )
    parser.add_argument(
        "--mock", action="store_true", help="Run deterministic CPU mock instead of real model eval"
    )
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[benchmark] model={args.model} phase={args.phase} mock={args.mock}")
    run = run_benchmarks(
        model_name=args.model,
        run_phase=args.phase,
        benchmarks=GENERAL_BENCHMARKS,
        domain_benchmarks=DOMAIN_BENCHMARKS,
        n_samples=args.n_samples,
        mock=args.mock,
        model=args.model,
        model_args=args.model_args,
    )
    report = run.to_dict()
    fname = f"{args.phase}_{date.today().isoformat()}.json"
    out_path = out_dir / fname
    out_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"[benchmark] wrote {out_path}")
    print(f"[benchmark] general_avg={report['summary']['general_avg']} domain_avg={report['summary']['domain_avg']}")

    if args.compare:
        compare_path = Path(args.compare)
        if not compare_path.exists():
            print(f"[benchmark] WARN: compare path {compare_path} not found, skipping forgetting")
        else:
            pre_data = json.loads(compare_path.read_text(encoding="utf-8"))
            pre_run = BenchmarkRun(
                model_name=pre_data["model_name"],
                run_phase=pre_data["run_phase"],
                run_date=pre_data.get("date"),
                results=[
                    BenchmarkResult(
                        name=r["name"],
                        score=r["score"],
                        n_samples=r["n_samples"],
                        category=r["category"],
                        runtime_s=r["runtime_s"],
                        notes=r["notes"],
                    )
                    for r in pre_data["results"]
                ],
            )
            forgetting = compute_forgetting(pre_run, run)
            forget_path = out_dir / f"forgetting_{date.today().isoformat()}.json"
            forget_path.write_text(json.dumps(forgetting, indent=2) + "\n", encoding="utf-8")
            print(f"[benchmark] forgetting report: {forget_path}")
            print(
                f"[benchmark] avg_general_forgetting={forgetting['avg_general_forgetting']} "
                f"verdict={forgetting['verdict']}"
            )

    return 0


if __name__ == "__main__":
    sys.exit(main())