"""
DiagnosisArena continuous evaluation pipeline.

Orchestrates multi-system, multi-format benchmarking for PIX-3909.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Mapping
from pathlib import Path

from .leaderboard import Leaderboard, SystemEvaluation, write_leaderboard
from .reporter import write_report
from .runner import ResponseProducer, run
from .types import BenchmarkSummary, ResponseFormat

logger = logging.getLogger(__name__)


def solve_case_for_system(system: str, *, history: str = "", notes: str = "") -> str:
    """Stub solver for seeding pipelines in experimental mode."""
    note = notes or "no additional context"
    return f"{system}: placeholder response ({note})"


class OpenAIBenchmarkPipeline:
    """Run a multi-system benchmark and emit leaderboard + per-system reports."""

    def __init__(
        self,
        benchmark,
        systems: Mapping[str, ResponseProducer],
        formats: tuple[ResponseFormat, ...] | None = None,
        *,
        judge=None,
    ) -> None:
        from .judge import HeuristicJudge

        if formats is None:
            formats = (ResponseFormat.OPEN_ENDED, ResponseFormat.MCQ)
        self.benchmark = benchmark
        self.systems = dict(systems)
        self.formats = list(formats)
        self.judge = judge or HeuristicJudge()
        self._results: dict[tuple[str, str], tuple[list, BenchmarkSummary]] = {}

    def run(self) -> dict[tuple[str, str], tuple[list, BenchmarkSummary]]:
        """Execute all (system, format) pairs."""
        self._results = {}
        total = len(self.systems) * len(self.formats)
        idx = 0
        for label, producer in self.systems.items():
            for fmt in self.formats:
                idx += 1
                scores, summary = run(
                    self.benchmark,
                    producer=producer,
                    judge=self.judge,
                    response_format=fmt,
                    model_label=label,
                )
                self._results[(label, fmt.value)] = (scores, summary)
                logger.info(
                    "evaluated %s / %s (%d/%d): accuracy=%.2f%%",
                    label,
                    fmt.value,
                    idx,
                    total,
                    summary.overall_accuracy * 100,
                )
        return self._results

    def summarize(self) -> Leaderboard:
        """Return a ranked leaderboard across systems (best format per system). """
        entries: list[SystemEvaluation] = []
        for label in self.systems:
            summaries = [s for (l, _), (_, s) in self._results.items() if l == label]
            if not summaries:
                continue
            best = max(summaries, key=lambda s: s.overall_accuracy)
            entries.append(SystemEvaluation(label=label, system=label, summary=best))
        return Leaderboard(entries)

    def write_reports(self, out_dir: str | Path, *, also_json: bool = True) -> dict:
        """Persist per-system reports plus a leaderboard markdown + json."""
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        leaderboard = self.summarize()
        write_leaderboard(leaderboard, out, also_json=also_json)
        detail_paths: dict[str, Path] = {}
        for (label, fmt), (_, summary) in self._results.items():
            md_path, _ = write_report(summary, out, also_json=also_json)
            detail_paths[f"{label}/{fmt}"] = md_path
        return {
            "leaderboard_md": out / "leaderboard.md",
            "leaderboard_json": out / "leaderboard.json",
            "system_reports": detail_paths,
        }


def run_multi_system_benchmark(
    benchmark,
    systems: Mapping[str, Callable],
    *,
    judge=None,
    formats: tuple[ResponseFormat, ...] | None = None,
    out_dir: str | Path | None = None,
) -> tuple[Leaderboard, dict]:
    """One-shot convenience wrapper: run benchmark and return leaderboard and file outputs."""
    pipeline = OpenAIBenchmarkPipeline(
        benchmark=benchmark,
        systems=systems,
        formats=formats,
        judge=judge,
    )
    pipeline.run()
    leaderboard = pipeline.summarize()
    outputs = {}
    if out_dir is not None:
        outputs = pipeline.write_reports(out_dir)
    return leaderboard, outputs


__all__ = [
    "OpenAIBenchmarkPipeline",
    "run_multi_system_benchmark",
    "solve_case_for_system",
]
