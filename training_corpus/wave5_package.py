"""Build a canonical corpus package from the closed wave-five authoring ledger."""

from __future__ import annotations

import argparse
import json
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .builder import CorpusBuildConfig, CorpusBuilder, CorpusBuildResult
from .expansion_authoring import DEFAULT_WAVE5_AUTHORING_LEDGER_PATH, load_authoring_ledger
from .synthesis import build_seed_registry

DEFAULT_WAVE5_SOURCE_DIR = Path(__file__).resolve().parent / "assets"
_WAVE5_PREFIX = "wave5_authored_seed"
DEFAULT_WAVE5_SOURCE_PATHS = {
    "simulation": DEFAULT_WAVE5_SOURCE_DIR / f"{_WAVE5_PREFIX}_simulation.jsonl",
    "evaluator": DEFAULT_WAVE5_SOURCE_DIR / f"{_WAVE5_PREFIX}_evaluator.jsonl",
    "benchmark": DEFAULT_WAVE5_SOURCE_DIR / f"{_WAVE5_PREFIX}_benchmark.jsonl",
}
DEFAULT_WAVE5_MANIFEST_PATH = DEFAULT_WAVE5_SOURCE_DIR / f"{_WAVE5_PREFIX}_manifest.json"
DEFAULT_WAVE5_REGISTRY_PATH = DEFAULT_WAVE5_SOURCE_DIR / f"{_WAVE5_PREFIX}_registry.json"


@dataclass(frozen=True)
class Wave5PackageBuildConfig:
    ledger_path: Path = DEFAULT_WAVE5_AUTHORING_LEDGER_PATH
    assets_dir: Path = DEFAULT_WAVE5_SOURCE_DIR
    registry_path: Path | None = None
    name: str = "pixelated-wave5-authored-corpus"
    version: str = "2026.04.10-wave5-authored"
    verify_reproducibility: bool = True


def _clone_json(value: Any) -> Any:
    return json.loads(json.dumps(value))


def _string_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(item).strip() for item in value if str(item).strip()]


def _base_metadata(row: dict[str, Any], *, artifact_type: str) -> dict[str, Any]:
    metadata = _clone_json(row.get("metadata") or {})
    metadata.setdefault("artifact_type", artifact_type)
    metadata.setdefault("source_origin", "generated_internal")
    metadata.setdefault("quality_score", 0.95)
    metadata.setdefault("safety_score", 0.95)
    return metadata


def _clinician_review_hook() -> dict[str, Any]:
    return {
        "status": "planned",
        "reviewer_role": "clinician",
        "reviewer_count": 0,
        "calibration_subset": True,
    }


def _benchmark_spec_output(row: dict[str, Any]) -> str:
    must_detect = _string_list(row.get("must_detect"))
    mistakes = _string_list(row.get("likely_therapist_mistakes"))
    rubric_items = row.get("rubric_items") if isinstance(row.get("rubric_items"), list) else []
    criteria: list[str] = []
    for item in rubric_items:
        if isinstance(item, dict):
            criterion = str(item.get("criterion") or "").strip()
            weight = item.get("weight")
            if criterion:
                criteria.append(f"{criterion} (weight {weight})" if isinstance(weight, int) else criterion)
        elif isinstance(item, str) and item.strip():
            criteria.append(item.strip())

    parts = [
        "Expected handling should detect "
        + (", ".join(must_detect) if must_detect else "the core hidden driver and rupture risk"),
        "avoid "
        + (", ".join(mistakes) if mistakes else "generic reassurance or premature interpretation"),
    ]
    if criteria:
        parts.append("and satisfy rubric items such as " + "; ".join(criteria[:4]))
    return ". ".join(parts) + "."


def _evaluator_spec_output(row: dict[str, Any]) -> str:
    required_signals = _string_list(row.get("required_signals"))
    fail_conditions = _string_list(row.get("fail_conditions"))
    output_contract = row.get("output_contract") if isinstance(row.get("output_contract"), dict) else {}
    outputs = _string_list(output_contract.get("required_outputs"))

    parts = [
        "A strong evaluator response should inspect "
        + (", ".join(required_signals) if required_signals else "the required therapeutic signals"),
        "flag failures such as "
        + (", ".join(fail_conditions) if fail_conditions else "missing the core task requirements"),
    ]
    if outputs:
        parts.append("and return " + ", ".join(outputs))
    notes = output_contract.get("notes")
    if isinstance(notes, str) and notes.strip():
        parts.append(notes.strip())
    return ". ".join(parts) + "."


def _build_simulation_record(row: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    record = _clone_json(row)
    record["lane"] = "simulation"
    record["metadata"] = _base_metadata(row, artifact_type="dialogue_seed_rows")
    return "simulation", record


def _build_benchmark_row_record(row: dict[str, Any]) -> tuple[str, dict[str, Any]] | None:
    record = {
        "input": str(row.get("prompt") or "").strip(),
        "output": str(row.get("expected_behavior") or "").strip(),
        "lane": "benchmark",
        "metadata": {
            **_base_metadata(row, artifact_type="benchmark_rows"),
            "clinician_review": _clinician_review_hook(),
        },
    }
    return ("benchmark", record) if record["input"] and record["output"] else None


def _build_benchmark_spec_record(row: dict[str, Any]) -> tuple[str, dict[str, Any]] | None:
    record = {
        "input": str(row.get("prompt") or row.get("title") or "").strip(),
        "output": _benchmark_spec_output(row),
        "lane": "benchmark",
        "metadata": {
            **_base_metadata(row, artifact_type="benchmark_specs"),
            "benchmark_slice": row.get("benchmark_slice"),
            "must_detect": _string_list(row.get("must_detect")),
            "likely_therapist_mistakes": _string_list(row.get("likely_therapist_mistakes")),
            "rubric_items": _clone_json(row.get("rubric_items") or []),
            "clinician_review": _clinician_review_hook(),
        },
    }
    return ("benchmark", record) if record["input"] and record["output"] else None


def _build_evaluator_spec_record(row: dict[str, Any]) -> tuple[str, dict[str, Any]] | None:
    record = {
        "input": str(row.get("task") or row.get("evaluator_id") or "").strip(),
        "output": _evaluator_spec_output(row),
        "lane": "evaluator",
        "metadata": {
            **_base_metadata(row, artifact_type="evaluator_specs"),
            "required_signals": _string_list(row.get("required_signals")),
            "fail_conditions": _string_list(row.get("fail_conditions")),
            "output_contract": _clone_json(row.get("output_contract") or {}),
            "rubric_items": [
                {"criterion": "Detects the required therapeutic signals", "weight": 4},
                {"criterion": "Flags the declared fail conditions", "weight": 3},
                {"criterion": "Produces the required structured outputs", "weight": 3},
            ],
            "clinician_review": _clinician_review_hook(),
        },
    }
    return ("evaluator", record) if record["input"] and record["output"] else None


_RECORD_BUILDERS: dict[str, Callable[[dict[str, Any]], tuple[str, dict[str, Any]] | None]] = {
    "benchmark_rows": _build_benchmark_row_record,
    "benchmark_specs": _build_benchmark_spec_record,
    "dialogue_seed_rows": _build_simulation_record,
    "evaluator_specs": _build_evaluator_spec_record,
}


def build_wave5_authored_records(ledger: dict[str, Any]) -> dict[str, tuple[dict[str, Any], ...]]:
    entries = ledger.get("entries")
    if not isinstance(entries, list):
        raise ValueError("Authoring ledger must contain entries.")

    lane_records: dict[str, list[dict[str, Any]]] = {
        "simulation": [],
        "evaluator": [],
        "benchmark": [],
    }

    for entry in entries:
        if not isinstance(entry, dict):
            continue

        artifact_type = str(entry.get("artifact_type") or "").strip()
        handler = _RECORD_BUILDERS.get(artifact_type)
        draft_rows = entry.get("draft_rows")
        if handler is None or not isinstance(draft_rows, list):
            continue

        for row in draft_rows:
            if not isinstance(row, dict):
                continue
            built = handler(row)
            if built is None:
                continue
            lane, record = built
            lane_records[lane].append(record)

    return {
        "simulation": tuple(lane_records["simulation"]),
        "evaluator": tuple(lane_records["evaluator"]),
        "benchmark": tuple(lane_records["benchmark"]),
    }


def materialize_wave5_authored_sources(
    *,
    ledger_path: Path = DEFAULT_WAVE5_AUTHORING_LEDGER_PATH,
    output_paths: dict[str, Path] | None = None,
    manifest_path: Path | None = DEFAULT_WAVE5_MANIFEST_PATH,
) -> dict[str, Path]:
    ledger = load_authoring_ledger(ledger_path)
    records = build_wave5_authored_records(ledger)
    resolved_paths = output_paths or DEFAULT_WAVE5_SOURCE_PATHS
    written: dict[str, Path] = {}

    for lane, rows in records.items():
        path = resolved_paths[lane]
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        written[lane] = path

    if manifest_path is not None:
        manifest = {
            "ledger_path": str(ledger_path),
            "version": ledger.get("version"),
            "outputs": {lane: str(path) for lane, path in written.items()},
            "record_counts": {lane: len(records[lane]) for lane in records},
        }
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        manifest_path.write_text(f"{json.dumps(manifest, indent=2)}\n", encoding="utf-8")
    return written


def ensure_wave5_authored_registry_materialized(
    *,
    ledger_path: Path = DEFAULT_WAVE5_AUTHORING_LEDGER_PATH,
    output_dir: Path = DEFAULT_WAVE5_SOURCE_DIR,
    registry_path: Path = DEFAULT_WAVE5_REGISTRY_PATH,
    manifest_path: Path | None = DEFAULT_WAVE5_MANIFEST_PATH,
) -> Path:
    resolved_manifest_path = None if manifest_path is None else output_dir / Path(manifest_path).name
    source_paths = materialize_wave5_authored_sources(
        ledger_path=ledger_path,
        output_paths={
            "simulation": output_dir / DEFAULT_WAVE5_SOURCE_PATHS["simulation"].name,
            "evaluator": output_dir / DEFAULT_WAVE5_SOURCE_PATHS["evaluator"].name,
            "benchmark": output_dir / DEFAULT_WAVE5_SOURCE_PATHS["benchmark"].name,
        },
        manifest_path=resolved_manifest_path,
    )
    registry_payload = build_seed_registry(source_paths, prefix=_WAVE5_PREFIX)
    registry_path.parent.mkdir(parents=True, exist_ok=True)
    registry_path.write_text(f"{json.dumps(registry_payload, indent=2)}\n", encoding="utf-8")
    return registry_path


def build_wave5_authored_corpus(
    output_dir: Path,
    config: Wave5PackageBuildConfig | None = None,
) -> CorpusBuildResult:
    resolved_config = config or Wave5PackageBuildConfig()
    resolved_registry = ensure_wave5_authored_registry_materialized(
        ledger_path=resolved_config.ledger_path,
        output_dir=resolved_config.assets_dir,
        registry_path=resolved_config.registry_path or (resolved_config.assets_dir / DEFAULT_WAVE5_REGISTRY_PATH.name),
        manifest_path=resolved_config.assets_dir / DEFAULT_WAVE5_MANIFEST_PATH.name,
    )
    builder = CorpusBuilder(
        CorpusBuildConfig(
            name=resolved_config.name,
            version=resolved_config.version,
            registry_path=resolved_registry,
            destination=output_dir,
            verify_reproducibility=resolved_config.verify_reproducibility,
        )
    )
    return builder.build()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output_dir", type=Path, help="Destination directory for the built wave-five corpus")
    parser.add_argument("--ledger-path", type=Path, default=DEFAULT_WAVE5_AUTHORING_LEDGER_PATH)
    parser.add_argument("--assets-dir", type=Path, default=DEFAULT_WAVE5_SOURCE_DIR)
    parser.add_argument("--registry-path", type=Path, default=None)
    parser.add_argument("--name", default="pixelated-wave5-authored-corpus")
    parser.add_argument("--version", default="2026.04.10-wave5-authored")
    parser.add_argument(
        "--no-repro",
        action="store_true",
        help="Disable reproducibility verification for faster local iteration",
    )
    args = parser.parse_args()
    build_wave5_authored_corpus(
        args.output_dir,
        Wave5PackageBuildConfig(
            ledger_path=args.ledger_path,
            assets_dir=args.assets_dir,
            registry_path=args.registry_path,
            name=args.name,
            version=args.version,
            verify_reproducibility=not args.no_repro,
        ),
    )


if __name__ == "__main__":
    main()
