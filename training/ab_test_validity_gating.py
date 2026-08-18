"""A/B test framework for clinical validity gating.

Compares model output quality before vs. after clinical validity gating:

* Loads two prompt/response JSONL datasets (control and treatment).
* Computes per-sample clinical validity scores using
  :class:`training.clinical_validity_scorer.ClinicalValidityScorer`.
* Aggregates the full eval metric set from
  :func:`training.mental_health_eval._compute_metrics`.
* Runs paired comparisons of headline metrics with two-sample t-tests
  (``scipy.stats.ttest_ind``), Cohen's ``d`` effect size, and
  Bonferroni-adjusted significance.
* Writes a JSON report and a markdown summary to the output directory.

Designed for PIX-3742 ("EPIC: Clinical Validity Enhancement Pipeline")
checklist item: *A/B test results comparing model quality before/after
validity gating*.

Usage (CLI):

    uv run python -m training.ab_test_validity_gating \
        --control  runs/baseline/eval_samples.jsonl \
        --treatment runs/gated/eval_samples.jsonl \
        --output-dir reports/ab_validity_gating
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import statistics
import sys
from collections.abc import Iterable, Sequence
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path

from scipy import stats

from training.clinical_validity_scorer import ClinicalValidityScorer
from training.mental_health_eval import (
    _compute_metrics,
    _has_crisis_resource,
    _is_crisis_prompt,
    _load_dataset,
)

logger = logging.getLogger("ab_test_validity_gating")


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Two-sample size below which Cohen's d is undefined -> fallback to 0.0.
_MIN_EFFECT_SIZE_SAMPLE = 2

# Standard significance threshold used for both raw and adjusted p-values.
_ALPHA = 0.05


# Empathy keywords (kept consistent with ``mental_health_eval._compute_metrics``).
_EMPATHY_KEYWORDS: frozenset[str] = frozenset(
    {
        "understand",
        "hear you",
        "feeling",
        "empathy",
        "compassion",
        "validate",
        "valid",
        "support",
        "care",
        "sorry you",
        "that sounds",
        "difficult",
        "struggling",
        "here for you",
        "not alone",
        "listen",
        "safe space",
        "your feelings",
    }
)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class MetricComparison:
    """Two-sample comparison for a single scalar metric."""

    metric: str
    control_mean: float
    treatment_mean: float
    delta: float
    relative_lift: float | None
    cohens_d: float
    t_statistic: float
    p_value: float
    p_value_adjusted: float | None
    significant_at_005: bool
    significant_at_005_adjusted: bool | None


@dataclass
class ABTestReport:
    """End-to-end A/B test report."""

    control_path: str
    treatment_path: str
    control_samples: int
    treatment_samples: int
    generated_at: str
    scoring_version: str
    headline_significant: bool
    comparisons: list[MetricComparison] = field(default_factory=list)
    control_metrics: dict = field(default_factory=dict)
    treatment_metrics: dict = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Statistical helpers
# ---------------------------------------------------------------------------


def _cohens_d(control: Sequence[float], treatment: Sequence[float]) -> float:
    """Compute Cohen's ``d`` (treatment - control) with pooled std-dev.

    Returns ``0.0`` if either sample is empty or has zero variance.
    """
    if len(control) < _MIN_EFFECT_SIZE_SAMPLE or len(treatment) < _MIN_EFFECT_SIZE_SAMPLE:
        return 0.0
    mean_control = statistics.mean(control)
    mean_treatment = statistics.mean(treatment)
    var_control = statistics.variance(control)
    var_treatment = statistics.variance(treatment)
    pooled_var = ((len(control) - 1) * var_control + (len(treatment) - 1) * var_treatment) / (
        len(control) + len(treatment) - 2
    )
    if pooled_var <= 0.0:
        return 0.0
    return (mean_treatment - mean_control) / math.sqrt(pooled_var)


def _bonferroni(p_values: Sequence[float]) -> list[float]:
    """Min(p * n, 1.0) across the supplied ``p_values``."""
    n = max(len(p_values), 1)
    return [min(p * n, 1.0) for p in p_values]


def _relative_lift(control_mean: float, treatment_mean: float) -> float | None:
    """Percentage change from control to treatment, ``None`` when undefined."""
    if control_mean == 0 or math.isnan(control_mean):
        return None
    return (treatment_mean - control_mean) / abs(control_mean)


# ---------------------------------------------------------------------------
# Per-sample scoring & metric extraction
# ---------------------------------------------------------------------------


def _per_sample_clinical_scores(samples: Iterable[dict]) -> list[float]:
    """Compute ``ClinicalValidityScorer.score()`` for every sample's response."""
    return [ClinicalValidityScorer.score(sample.get("response", "")) for sample in samples]


def _collect_metric_values(
    per_sample_lookups: dict[str, Sequence[float]],
    aggregate_metrics: dict[str, float],
) -> dict[str, tuple[float, Sequence[float]]]:
    """Pair aggregate values with per-sample vectors for t-test comparisons.

    ``per_sample_lookups`` provides per-sample arrays for the metrics we can
    actually run significance tests on.  ``aggregate_metrics`` provides the
    aggregate values from :func:`_compute_metrics` for everything else.
    """
    pairs: dict[str, tuple[float, Sequence[float]]] = {}
    for metric_name, values in per_sample_lookups.items():
        if not values:
            continue
        pairs[metric_name] = (statistics.mean(values), values)
    for metric_name, aggregate in aggregate_metrics.items():
        if metric_name in pairs:
            continue
        try:
            float(aggregate)
        except (TypeError, ValueError):
            continue
        pairs[metric_name] = (float(aggregate), [])
    return pairs


def _per_sample_response_lengths(samples: Sequence[dict]) -> list[float]:
    return [float(len(s.get("response", "").split())) for s in samples]


def _per_sample_crisis_flags(samples: Sequence[dict]) -> list[float]:
    flags: list[float] = []
    for s in samples:
        prompt = s.get("prompt", "")
        response = (s.get("response") or "").lower()
        if not _is_crisis_prompt(prompt):
            continue
        flags.append(1.0 if _has_crisis_resource(response) else 0.0)
    return flags


def _per_sample_empathy_flags(samples: Sequence[dict]) -> list[float]:
    flags: list[float] = []
    for s in samples:
        response = (s.get("response") or "").lower()
        flags.append(1.0 if any(kw in response for kw in _EMPATHY_KEYWORDS) else 0.0)
    return flags


# ---------------------------------------------------------------------------
# Comparison core
# ---------------------------------------------------------------------------


# Headline metric: the primary key in the PIX-3742 A/B checklist.
HEADLINE_METRIC = "clinical_validity_mean"


def _compare_metric(
    name: str,
    control_pairs: dict[str, tuple[float, Sequence[float]]],
    treatment_pairs: dict[str, tuple[float, Sequence[float]]],
    n_metrics: int,
) -> MetricComparison | None:
    if name not in control_pairs or name not in treatment_pairs:
        return None

    control_mean, control_values = control_pairs[name]
    treatment_mean, treatment_values = treatment_pairs[name]

    if not control_values or not treatment_values:
        # No per-sample pairs available — fall back to a t-test against the
        # aggregated point estimates (degenerate single-value sample).
        t_stat, p_value = 0.0, 1.0
    else:
        t_stat, p_value = stats.ttest_ind(treatment_values, control_values, equal_var=False)
        if math.isnan(float(p_value)):
            # Identical distributions → no signal; keep neutral values.
            t_stat, p_value = 0.0, 1.0

    p_adjusted = min(float(p_value) * n_metrics, 1.0)
    d = _cohens_d(control_values or [control_mean], treatment_values or [treatment_mean])

    return MetricComparison(
        metric=name,
        control_mean=control_mean,
        treatment_mean=treatment_mean,
        delta=treatment_mean - control_mean,
        relative_lift=_relative_lift(control_mean, treatment_mean),
        cohens_d=d,
        t_statistic=float(t_stat),
        p_value=float(p_value),
        p_value_adjusted=float(p_adjusted),
        significant_at_005=bool(float(p_value) < _ALPHA),
        significant_at_005_adjusted=bool(float(p_adjusted) < _ALPHA),
    )


def compare(
    control_samples: Sequence[dict],
    treatment_samples: Sequence[dict],
) -> ABTestReport:
    """Run the A/B comparison and return an :class:`ABTestReport`."""
    if not control_samples:
        raise ValueError("control dataset is empty")
    if not treatment_samples:
        raise ValueError("treatment dataset is empty")

    control_metrics = _compute_metrics(list(control_samples))
    treatment_metrics = _compute_metrics(list(treatment_samples))

    per_sample_control = {
        "clinical_validity_mean": _per_sample_clinical_scores(control_samples),
        "response_length_mean": _per_sample_response_lengths(control_samples),
        "empathy_presence_rate": _per_sample_empathy_flags(control_samples),
        "crisis_citation_rate": _per_sample_crisis_flags(control_samples),
    }
    per_sample_treatment = {
        "clinical_validity_mean": _per_sample_clinical_scores(treatment_samples),
        "response_length_mean": _per_sample_response_lengths(treatment_samples),
        "empathy_presence_rate": _per_sample_empathy_flags(treatment_samples),
        "crisis_citation_rate": _per_sample_crisis_flags(treatment_samples),
    }

    control_pairs = _collect_metric_values(per_sample_control, control_metrics)
    treatment_pairs = _collect_metric_values(per_sample_treatment, treatment_metrics)

    union = sorted(set(control_pairs) | set(treatment_pairs))
    n_metrics = len(union)

    comparisons: list[MetricComparison] = []
    for name in union:
        comp = _compare_metric(name, control_pairs, treatment_pairs, n_metrics)
        if comp is not None:
            comparisons.append(comp)

    headline = next((c for c in comparisons if c.metric == HEADLINE_METRIC), None)
    headline_significant = bool(headline and headline.significant_at_005_adjusted)

    notes: list[str] = []
    if len(control_samples) != len(treatment_samples):
        notes.append(
            f"Sample-count mismatch: control={len(control_samples)}, "
            f"treatment={len(treatment_samples)}. Welch's t-test used."
        )
    notes.append(f"Bonferroni-adjusted alpha across {n_metrics} comparisons for significance flag.")

    return ABTestReport(
        control_path="<inline>",
        treatment_path="<inline>",
        control_samples=len(control_samples),
        treatment_samples=len(treatment_samples),
        generated_at=datetime.now(UTC).isoformat(),
        scoring_version=ClinicalValidityScorer.VERSION,
        headline_significant=headline_significant,
        comparisons=comparisons,
        control_metrics=control_metrics,
        treatment_metrics=treatment_metrics,
        notes=notes,
    )


def compare_paths(control_path: Path, treatment_path: Path) -> ABTestReport:
    """Convenience wrapper that loads JSONL datasets from disk then compares."""
    control = _load_dataset(control_path)
    treatment = _load_dataset(treatment_path)
    report = compare(control, treatment)
    report.control_path = str(control_path)
    report.treatment_path = str(treatment_path)
    return report


# ---------------------------------------------------------------------------
# Report writers
# ---------------------------------------------------------------------------


def to_dict(report: ABTestReport) -> dict:
    """Return a JSON-serialisable ``dict`` view of ``report``."""
    return asdict(report)


def write_json_report(report: ABTestReport, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(to_dict(report), f, indent=2, sort_keys=False)
        f.write("\n")


def write_markdown_report(report: ABTestReport, output_path: Path) -> None:
    """Render a human-readable summary alongside the JSON report."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    def _fmt(value: float | None, *, digits: int = 4) -> str:
        if value is None or (isinstance(value, float) and math.isnan(value)):
            return "n/a"
        return f"{value:.{digits}f}"

    lines: list[str] = []
    lines.append("# Clinical Validity Gating - A/B Test Report")
    lines.append("")
    lines.append(f"- Generated: {report.generated_at}")
    lines.append(f"- Scorer version: {report.scoring_version}")
    lines.append(f"- Control samples: {report.control_samples} (`{report.control_path}`)")
    lines.append(f"- Treatment samples: {report.treatment_samples} (`{report.treatment_path}`)")
    lines.append("")
    lines.append(
        f"**Headline metric (`{HEADLINE_METRIC}`) significantly improved "
        f"(Bonferroni-adjusted p < 0.05):** "
        f"{'YES' if report.headline_significant else 'NO'}"
    )
    lines.append("")
    lines.append("## Per-metric comparison")
    lines.append("")
    lines.append(
        "| Metric | Control mean | Treatment mean | Δ | Relative lift | Cohen's d | "
        "t | p (raw) | p (Bonf.) | Significant (raw / adj.) |"
    )
    lines.append("| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |")
    for c in report.comparisons:
        sig_raw = "yes" if c.significant_at_005 else "no"
        sig_adj = "yes" if c.significant_at_005_adjusted else "no"
        lift = "n/a" if c.relative_lift is None else f"{_fmt(c.relative_lift, digits=3)}"
        lines.append(
            "| "
            + " | ".join(
                [
                    c.metric,
                    _fmt(c.control_mean, digits=4),
                    _fmt(c.treatment_mean, digits=4),
                    _fmt(c.delta, digits=4),
                    lift,
                    _fmt(c.cohens_d, digits=3),
                    _fmt(c.t_statistic, digits=3),
                    f"{c.p_value:.3e}",
                    f"{c.p_value_adjusted:.3e}",
                    f"{sig_raw} / {sig_adj}",
                ]
            )
            + " |"
        )
    lines.append("")
    lines.append("## Aggregate metrics")
    lines.append("")
    lines.append("| Metric | Control | Treatment |")
    lines.append("| --- | --- | --- |")
    keys = sorted(set(report.control_metrics) | set(report.treatment_metrics))
    nan = float("nan")
    for key in keys:
        c_val = report.control_metrics.get(key)
        t_val = report.treatment_metrics.get(key)
        c_str = _fmt(c_val if c_val is not None else nan)
        t_str = _fmt(t_val if t_val is not None else nan)
        lines.append(f"| {key} | {c_str} | {t_str} |")
    lines.append("")
    if report.notes:
        lines.append("## Notes")
        lines.append("")
        for note in report.notes:
            lines.append(f"- {note}")
        lines.append("")

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="A/B test clinical validity gating on two prompt/response JSONL datasets.",
    )
    parser.add_argument("--control", type=str, required=True, help="Path to control JSONL (pre-gating)")
    parser.add_argument("--treatment", type=str, required=True, help="Path to treatment JSONL (post-gating)")
    parser.add_argument("--output-dir", type=str, required=True, help="Where to write JSON + markdown reports")
    parser.add_argument(
        "--summary-name",
        type=str,
        default="ab_validity_compare",
        help="Base filename (without extension) for the reports",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    parser = build_parser()
    args = parser.parse_args(argv)

    control_path = Path(args.control)
    treatment_path = Path(args.treatment)
    if not control_path.exists():
        logger.error("Control dataset not found: %s", control_path)
        return 2
    if not treatment_path.exists():
        logger.error("Treatment dataset not found: %s", treatment_path)
        return 2

    try:
        report = compare_paths(control_path, treatment_path)
    except ValueError as exc:
        logger.error("Comparison failed: %s", exc)
        return 2

    output_dir = Path(args.output_dir)
    json_out = output_dir / f"{args.summary_name}.json"
    md_out = output_dir / f"{args.summary_name}.md"
    write_json_report(report, json_out)
    write_markdown_report(report, md_out)
    logger.info("Wrote JSON report: %s", json_out)
    logger.info("Wrote markdown report: %s", md_out)
    logger.info(
        "Headline metric significant (Bonferroni-adjusted): %s",
        report.headline_significant,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
