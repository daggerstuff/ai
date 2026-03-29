#!/usr/bin/env python3
"""Generate a Stage 2 training report from saved artifacts and evaluation output."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate Stage 2 training report")
    parser.add_argument(
        "--model-dir",
        default="./therapeutic_ai_final_stage2",
        help="Directory containing artifact_manifest.json and tokenizer/artifacts",
    )
    parser.add_argument(
        "--evaluation-results",
        default="evaluation_results.json",
        help="Path to the evaluation results JSON file",
    )
    parser.add_argument(
        "--output",
        default="stage2_evaluation_report.md",
        help="Output path for the generated markdown report",
    )
    return parser.parse_args()


def load_json_if_exists(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def summarize_evaluation_results(results: dict[str, Any] | None) -> tuple[list[str], list[str]]:
    if not results:
        return ["- Evaluation results file not found."], []

    summary_lines: list[str] = []
    metric_lines: list[str] = []
    overall_scores: list[float] = []

    for eval_name, metrics in results.items():
        summary_lines.append(f"- `{eval_name}`: {', '.join(sorted(metrics.keys()))}")
        overall_score = metrics.get("overall_score")
        if isinstance(overall_score, (int, float)):
            overall_scores.append(float(overall_score))

        metric_parts = []
        for key, value in sorted(metrics.items()):
            if isinstance(value, (int, float)):
                metric_parts.append(f"{key}={value:.4f}")
            else:
                metric_parts.append(f"{key}={value}")
        metric_lines.append(f"- `{eval_name}`: {', '.join(metric_parts)}")

    if overall_scores:
        average_score = sum(overall_scores) / len(overall_scores)
        summary_lines.append(f"- Average overall score: {average_score:.4f}")

    return summary_lines, metric_lines


def build_report(
    *,
    model_dir: Path,
    artifact_manifest: dict[str, Any] | None,
    evaluation_results: dict[str, Any] | None,
) -> str:
    generated_at = datetime.now(timezone.utc).isoformat()
    artifact_lines = []

    if artifact_manifest:
        for key, value in artifact_manifest.items():
            artifact_lines.append(f"- {key}: {value}")
    else:
        artifact_lines.append("- artifact_manifest.json not found.")

    model_dir_lines = [
        f"- Model directory exists: {'yes' if model_dir.exists() else 'no'}",
        f"- Adapter directory exists: {'yes' if (model_dir / 'adapters').exists() else 'no'}",
        f"- Tokenizer config exists: {'yes' if (model_dir / 'tokenizer_config.json').exists() else 'no'}",
    ]

    summary_lines, metric_lines = summarize_evaluation_results(evaluation_results)

    return "\n".join(
        [
            "# Stage 2 Evaluation Report",
            "",
            f"- Generated at: {generated_at}",
            f"- Model directory: `{model_dir}`",
            "",
            "## Artifact Summary",
            *artifact_lines,
            "",
            "## Filesystem Checks",
            *model_dir_lines,
            "",
            "## Evaluation Summary",
            *summary_lines,
            "",
            "## Evaluation Metrics",
            *(metric_lines or ["- No evaluation metrics available yet."]),
            "",
            "## Readiness Notes",
            "- This report is generated from saved artifacts and evaluation JSON only.",
            "- If evaluation results are missing, run the model evaluation step once GPU resources are available.",
            "- Treat this report as the handoff artifact for Asana task closure once weights and eval outputs exist.",
            "",
        ]
    )


def main() -> None:
    args = parse_args()
    model_dir = Path(args.model_dir)
    artifact_manifest = load_json_if_exists(model_dir / "artifact_manifest.json")
    evaluation_results = load_json_if_exists(Path(args.evaluation_results))

    report = build_report(
        model_dir=model_dir,
        artifact_manifest=artifact_manifest,
        evaluation_results=evaluation_results,
    )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(report, encoding="utf-8")
    print(f"Wrote Stage 2 report to {output_path}")


if __name__ == "__main__":
    main()
