"""
DiagnosisArena reporter.

Computes aggregate metrics over a collection of per-case scores and emits
them as either structured JSON or human-readable Markdown.
"""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from collections.abc import Iterable
from datetime import UTC, datetime
from pathlib import Path

from .types import (
    BenchmarkSummary,
    CaseScore,
    DIAGNOSTIC_DIMENSIONS,
    Difficulty,
    ResponseFormat,
)


def case_score_from_judgment(case_score: CaseScore) -> CaseScore:
    """Pass-through identity for type clarity at the integration boundary."""
    return case_score


def summarize(
    model_label: str,
    format: ResponseFormat,
    scores: Iterable[CaseScore],
) -> BenchmarkSummary:
    """Aggregate per-case scores into a single BenchmarkSummary."""
    scores_list = list(scores)
    if not scores_list:
        return BenchmarkSummary(
            model_label=model_label,
            format=format,
            cases_evaluated=0,
            overall_accuracy=0.0,
            timestamp=datetime.now(UTC).isoformat(),
        )

    overall = sum(s.tier_numeric for s in scores_list) / len(scores_list)

    diff_buckets: dict[Difficulty, list[float]] = defaultdict(list)
    for s in scores_list:
        diff_buckets[s.difficulty].append(s.tier_numeric)
    by_difficulty = {d.value: (sum(v) / len(v) if v else 0.0) for d, v in diff_buckets.items()}

    dim_sums: dict[str, list[float]] = {name: [] for name in DIAGNOSTIC_DIMENSIONS}
    for s in scores_list:
        for dim_name, dim_value in s.dimensions.items():
            if dim_name in dim_sums:
                dim_sums[dim_name].append(dim_value)
    by_dimension = {name: (sum(v) / len(v) if v else 0.0) for name, v in dim_sums.items()}

    error_counter: Counter[str] = Counter()
    for s in scores_list:
        for err in s.error_taxonomy:
            error_counter[err] += 1

    return BenchmarkSummary(
        model_label=model_label,
        format=format,
        cases_evaluated=len(scores_list),
        overall_accuracy=overall,
        by_difficulty=by_difficulty,
        by_dimension=by_dimension,
        error_distribution=dict(error_counter),
        timestamp=datetime.now(UTC).isoformat(),
    )


def to_json(summary: BenchmarkSummary, *, indent: int | None = 2) -> str:
    """Serialize a BenchmarkSummary as JSON."""
    payload = {
        "model_label": summary.model_label,
        "format": summary.format.value,
        "cases_evaluated": summary.cases_evaluated,
        "overall_accuracy": summary.overall_accuracy,
        "by_difficulty": summary.by_difficulty,
        "by_dimension": summary.by_dimension,
        "error_distribution": summary.error_distribution,
        "timestamp": summary.timestamp,
    }
    return json.dumps(payload, indent=indent, ensure_ascii=False)


def to_markdown(summary: BenchmarkSummary) -> str:
    """Render a BenchmarkSummary as a Markdown report."""
    lines: list[str] = []
    lines.append(f"# DiagnosisArena Report: {summary.model_label}")
    lines.append("")
    lines.append(f"- **Format**: {summary.format.value}")
    lines.append(f"- **Cases evaluated**: {summary.cases_evaluated}")
    lines.append(f"- **Overall accuracy**: {summary.overall_accuracy:.2%}")
    lines.append(f"- **Timestamp**: {summary.timestamp}")
    lines.append("")

    if summary.by_difficulty:
        lines.append("## Accuracy by difficulty")
        lines.append("")
        lines.append("| Difficulty | Accuracy |")
        lines.append("| --- | --- |")
        for d in (Difficulty.SIMPLE, Difficulty.MODERATE, Difficulty.COMPLEX):
            v = summary.by_difficulty.get(d.value)
            if v is None:
                continue
            lines.append(f"| {d.value} | {v:.2%} |")
        lines.append("")

    if summary.by_dimension:
        lines.append("## Scores by diagnostic dimension")
        lines.append("")
        lines.append("| Dimension | Score |")
        lines.append("| --- | --- |")
        for dim in DIAGNOSTIC_DIMENSIONS:
            v = summary.by_dimension.get(dim)
            if v is None:
                continue
            lines.append(f"| {dim} | {v:.2%} |")
        lines.append("")

    if summary.error_distribution:
        lines.append("## Diagnostic error distribution")
        lines.append("")
        lines.append("| Error type | Count |")
        lines.append("| --- | --- |")
        for err, count in sorted(summary.error_distribution.items(), key=lambda x: -x[1]):
            lines.append(f"| {err} | {count} |")
        lines.append("")

    return "\n".join(lines)


def write_report(
    summary: BenchmarkSummary,
    out_dir: str | Path,
    *,
    also_json: bool = True,
) -> tuple[Path, Path | None]:
    """Persist a BenchmarkSummary to disk as Markdown and optional JSON."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    safe_label = "".join(c for c in summary.model_label if c.isalnum() or c in ("-", "_"))
    base = f"{safe_label}_{summary.format.value}_{summary.timestamp.replace(':', '')}"
    md_path = out / f"{base}.md"
    md_path.write_text(to_markdown(summary), encoding="utf-8")

    json_path: Path | None = None
    if also_json:
        json_path = out / f"{base}.json"
        json_path.write_text(to_json(summary), encoding="utf-8")

    return md_path, json_path


__all__ = ["summarize", "to_json", "to_markdown", "write_report", "case_score_from_judgment"]
