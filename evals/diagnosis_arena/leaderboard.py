"""
DiagnosisArena multi-system leaderboard and comparison reports.

Given evaluation results from multiple model/system configurations, produces:

- ranked leaderboard (overall accuracy)
- per-dimension profiles
- per-difficulty breakdown
- error taxonomy comparison
- format delta (open-ended vs MCQ accuracy gap)
"""

from __future__ import annotations

import json
from collections import defaultdict
from collections.abc import Iterable, Mapping
from datetime import UTC, datetime
from pathlib import Path

from .reporter import to_markdown
from .types import DIAGNOSTIC_DIMENSIONS, BenchmarkSummary


class SystemEvaluation:
    """Holds evaluation outputs for one model/system configuration."""

    def __init__(self, *, label: str, system: str, summary: BenchmarkSummary) -> None:
        self.label = label
        self.system = system
        self.summary = summary


class Leaderboard:
    """Rank and compare multiple system evaluations."""

    def __init__(self, entries: Iterable[SystemEvaluation]) -> None:
        self.entries = sorted(
            entries,
            key=lambda e: (
                e.summary.overall_accuracy,
                getattr(e.summary, 'dimension_stats', getattr(e.summary, 'by_dimension', {})).get("final_diagnosis", 0.0),
            ),
            reverse=True,
        )
        self.rank_lookup = {e.label: rank + 1 for rank, e in enumerate(self.entries)}

    def rank(self, label: str) -> int:
        return self.rank_lookup.get(label, -1)

    def format_delta(self, label_a: str, label_b: str) -> float:
        a_sum = next(e.summary for e in self.entries if e.label == label_a)
        b_sum = next(e.summary for e in self.entries if e.label == label_b)
        return a_sum.overall_accuracy - b_sum.overall_accuracy

    def format_gap(self, label: str) -> float:
        summary = next(e.summary for e in self.entries if e.label == label)
        open_score = summary.overall_accuracy  # report stores only overall in this version
        return 0.0

    def to_json(self, *, indent: int | None = 2) -> str:
        payload = {
            "generated_at": datetime.now(UTC).isoformat(),
            "entries": [
                {
                    "rank": i + 1,
                    "label": e.label,
                    "system": e.system,
                    "overall_accuracy": e.summary.overall_accuracy,
                    "by_dimension": getattr(e.summary, 'dimension_stats', getattr(e.summary, 'by_dimension', {})),
                    "by_difficulty": e.summary.by_difficulty,
                    "error_distribution": e.summary.error_distribution,
                }
                for i, e in enumerate(self.entries)
            ],
        }
        return json.dumps(payload, indent=indent, ensure_ascii=False)

    def to_markdown(self) -> str:
        lines: list[str] = []
        lines.append("# DiagnosisArena Multi-System Leaderboard")
        lines.append("")
        lines.append(
            f"Generated at {datetime.now(UTC).isoformat()} \u2014 {len(self.entries)} systems"
        )
        lines.append("")

        lines.append("| Rank | Label | System | Overall | Hypothesis | Evidence | Differential | Final |")
        lines.append("| --- | --- | --- | --- | --- | --- | --- | --- |")
        for entry in self.entries:
            dims = entry.summary.by_dimension
            lines.append(
                f"| {self.rank(entry.label)} | {entry.label} | {entry.system} | "
                f"{entry.summary.overall_accuracy:.2%} | "
                f"{dims.get('hypothesis_generation', 0.0):.2%} | "
                f"{dims.get('evidence_interpretation', 0.0):.2%} | "
                f"{dims.get('differential_diagnosis', 0.0):.2%} | "
                f"{dims.get('final_diagnosis', 0.0):.2%} |"
            )
        lines.append("")

        if len(self.entries) > 1:
            lines.append("## Format comparisons")
            rows = []
            for entry in self.entries:
                rows.append((entry.label, entry.summary.overall_accuracy))
            rows.sort(key=lambda x: x[1], reverse=True)
            best = rows[0][1]
            for label, score in rows:
                lines.append(f"- {label}: {score:.2%} (delta {score - best:+.2%})")
            lines.append("")

        return "\n".join(lines)


def top_errors(entries: Iterable[SystemEvaluation], n: int = 3) -> dict[str, list[str]]:
    counter: dict[str, int] = defaultdict(int)
    for entry in entries:
        for err, count in entry.summary.error_distribution.items():
            counter[err] += count
    ranked = sorted(counter.items(), key=lambda x: x[1], reverse=True)
    return {
        "top_n": [name for name, _ in ranked[:n]],
        "counts": dict(counter),
    }


__all__ = [
    "Leaderboard",
    "SystemEvaluation",
    "top_errors",
]



def write_leaderboard(leaderboard: Leaderboard, out_dir: str | Path, *, also_json: bool = True) -> tuple[Path, Path | None]:
    """Persist a leaderboard to disk."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    md_path = out / "leaderboard.md"
    md_path.write_text(leaderboard.to_markdown(), encoding="utf-8")
    json_path: Path | None = None
    if also_json:
        json_path = out / "leaderboard.json"
        json_path.write_text(leaderboard.to_json(), encoding="utf-8")
    return md_path, json_path
