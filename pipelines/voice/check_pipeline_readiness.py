#!/usr/bin/env python3
"""Check Pixel Voice pipeline readiness without executing the pipeline."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[2]
REPORTS_DIR = SCRIPT_DIR / "reports"
REPORTS_DIR.mkdir(exist_ok=True)

PIPELINE_STAGES = [
    ("Audio Quality Control", "audio_quality_control.py"),
    ("Batch Transcription", "batch_transcribe.py"),
    ("Transcription Quality Filtering", "transcription_quality_filter.py"),
    ("Feature Extraction", "feature_extraction.py"),
    ("Personality & Emotion Clustering", "personality_emotion_clustering.py"),
    ("Dialogue Pair Construction", "dialogue_pair_constructor.py"),
    ("Dialogue Pair Validation", "dialogue_pair_validation.py"),
    ("Therapeutic Pair Generation", "generate_therapeutic_pairs.py"),
    ("Voice Quality Consistency", "voice_quality_consistency.py"),
    ("Voice Data Filtering/Optimization", "voice_data_filtering.py"),
    ("Pipeline Reporting", "pipeline_reporting.py"),
]

EXPECTED_PATHS = {
    "docs": [
        SCRIPT_DIR / "README.md",
        SCRIPT_DIR / "DEPLOYMENT.md",
        SCRIPT_DIR.parent.parent / "docs" / "pixel_voice_pipeline.md",
        SCRIPT_DIR.parent.parent / "docs" / "pixel_voice_pipeline_production_checklist.md",
    ],
    "runtime": [
        SCRIPT_DIR / "run_full_pipeline.py",
        SCRIPT_DIR / "setup_pixel_voice_env_uv.sh",
        SCRIPT_DIR / "pyproject.toml",
        SCRIPT_DIR / "docker-compose.yml",
    ],
    "artifacts": [
        SCRIPT_DIR / "logs",
        SCRIPT_DIR / "reports",
        SCRIPT_DIR / "data",
    ],
}


def to_display_path(path: Path) -> str:
    try:
        return str(path.relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def inspect_paths() -> dict:
    result: dict[str, list[dict[str, object]]] = {}
    for category, paths in EXPECTED_PATHS.items():
        result[category] = []
        for path in paths:
            result[category].append(
                {
                    "path": to_display_path(path),
                    "exists": path.exists(),
                    "type": "directory" if path.is_dir() else "file",
                }
            )
    return result


def inspect_stages() -> list[dict[str, object]]:
    stages = []
    for name, filename in PIPELINE_STAGES:
        path = SCRIPT_DIR / filename
        stages.append(
            {
                "name": name,
                "script": filename,
                "exists": path.exists(),
            }
        )
    return stages


def build_summary(stages: list[dict[str, object]], paths: dict) -> dict:
    missing_stage_scripts = [
        stage["script"] for stage in stages if not stage["exists"]
    ]
    missing_paths = [
        entry["path"]
        for entries in paths.values()
        for entry in entries
        if not entry["exists"]
    ]
    warnings = []
    if missing_stage_scripts:
        warnings.append(f"Missing stage scripts: {', '.join(missing_stage_scripts)}")
    if missing_paths:
        warnings.append(f"Missing expected paths: {', '.join(missing_paths)}")

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "pipeline_root": str(SCRIPT_DIR),
        "stage_count": len(stages),
        "missing_stage_script_count": len(missing_stage_scripts),
        "missing_expected_path_count": len(missing_paths),
        "ready_to_run": not missing_stage_scripts,
        "warnings": warnings,
    }


def main() -> int:
    stages = inspect_stages()
    paths = inspect_paths()
    summary = build_summary(stages, paths)

    report = {
        "summary": summary,
        "stages": stages,
        "paths": paths,
    }

    output_path = REPORTS_DIR / "pipeline_readiness_report.json"
    output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(json.dumps(summary, indent=2))
    print(f"[INFO] Report written to: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
