"""Build row-level authoring ledgers from wave-five draft-pack cards."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

from .expansion_drafts import (
    DEFAULT_WAVE5_EXPANSION_DRAFT_PACK_PATH,
    ensure_default_expansion_draft_pack_materialized,
)

DEFAULT_WAVE5_AUTHORING_LEDGER_PATH = (
    Path(__file__).resolve().parent / "assets" / "wave5_authoring_ledger.json"
)
_ARTIFACT_PRIORITY = {
    "scenario_archetypes": 0,
    "client_state_profiles": 1,
    "dialogue_seed_rows": 2,
    "benchmark_specs": 3,
    "benchmark_rows": 4,
    "preference_pair_candidates": 5,
    "session_scaffolds": 6,
    "therapist_moves": 7,
    "evaluator_specs": 8,
}

_LIST_FIELDS = {
    "activation_cues",
    "candidate_outputs",
    "common_distortions",
    "discard_zones",
    "escalation_markers",
    "fail_conditions",
    "fields_to_mine",
    "lane_targets",
    "likely_therapist_mistakes",
    "must_detect",
    "repair_openings",
    "repair_opportunities",
    "required_signals",
    "sample_paths",
    "session_phases",
    "signals_to_mine",
    "turning_points",
    "writing_notes",
}
_OBJECT_FIELDS = {
    "candidate_a",
    "candidate_b",
    "metadata",
    "output_contract",
}
_PRIMARY_ID_FIELD_MAP = {
    "benchmark_rows": "row_id",
    "benchmark_specs": "benchmark_id",
    "client_state_profiles": "state_id",
    "dialogue_seed_rows": "row_id",
    "evaluator_specs": "evaluator_id",
    "preference_pair_candidates": "pair_id",
    "scenario_archetypes": "scenario_id",
    "session_scaffolds": "scaffold_id",
    "therapist_moves": "move_id",
}


def load_authoring_draft_pack(
    path: Path = DEFAULT_WAVE5_EXPANSION_DRAFT_PACK_PATH,
) -> dict[str, Any]:
    with open(path, encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"Expansion draft pack must be a JSON object: {path}")
    cards = payload.get("authoring_cards")
    if not isinstance(cards, list):
        raise ValueError(f"Expansion draft pack must contain authoring_cards: {path}")
    return payload


def build_authoring_ledger(
    draft_pack: dict[str, Any],
    *,
    version: str = "2026-04-09-wave5-authoring-ledger",
) -> dict[str, Any]:
    cards = draft_pack.get("authoring_cards")
    if not isinstance(cards, list):
        raise ValueError("Draft pack must contain authoring_cards.")

    ledger_entries: list[dict[str, Any]] = []
    by_source_key: Counter[str] = Counter()
    by_artifact_type: Counter[str] = Counter()
    total_target_rows = 0

    for card in cards:
        if not isinstance(card, dict):
            continue
        card_id = card.get("card_id")
        source_key = card.get("source_key")
        artifact_type = card.get("artifact_type")
        target_count = card.get("target_count")
        if not isinstance(card_id, str) or not isinstance(source_key, str) or not isinstance(artifact_type, str):
            continue
        if not isinstance(target_count, int) or target_count <= 0:
            continue

        ledger_entries.append(
            {
                "ledger_entry_id": card_id,
                "card_id": card_id,
                "ticket_id": card.get("ticket_id"),
                "source_key": source_key,
                "source_title": card.get("source_title"),
                "priority": card.get("priority"),
                "artifact_type": artifact_type,
                "target_count": target_count,
                "lane_targets": _string_list(card.get("lane_targets")),
                "required_fields": _string_list(card.get("required_fields")),
                "fields_to_mine": _string_list(card.get("fields_to_mine")),
                "extraction_focus": _string_list(card.get("extraction_focus")),
                "discard_zones": _string_list(card.get("discard_zones")),
                "writing_notes": _string_list(card.get("writing_notes")),
                "sample_paths": _string_list(card.get("sample_paths")),
                "sample_evidence": _sample_evidence(card.get("sample_evidence")),
                "row_template": _build_row_template(card),
                "progress": {
                    "drafted_count": 0,
                    "reviewed_count": 0,
                    "promoted_count": 0,
                    "remaining_count": target_count,
                },
                "draft_rows": [],
            }
        )
        by_source_key.update([source_key])
        by_artifact_type.update([artifact_type])
        total_target_rows += target_count

    return {
        "version": version,
        "draft_pack_version": draft_pack.get("version"),
        "ledger_entry_count": len(ledger_entries),
        "total_target_rows": total_target_rows,
        "entries": ledger_entries,
        "summary": {
            "by_source_key": dict(sorted(by_source_key.items())),
            "by_artifact_type": dict(sorted(by_artifact_type.items())),
        },
    }


def load_authoring_ledger(
    path: Path = DEFAULT_WAVE5_AUTHORING_LEDGER_PATH,
) -> dict[str, Any]:
    with open(path, encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"Authoring ledger must be a JSON object: {path}")
    entries = payload.get("entries")
    if not isinstance(entries, list):
        raise ValueError(f"Authoring ledger must contain entries: {path}")
    return payload


def load_authored_batch(path: Path) -> dict[str, Any]:
    with open(path, encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"Authored batch must be a JSON object: {path}")
    source_key = payload.get("source_key")
    artifacts = payload.get("artifacts")
    if not isinstance(source_key, str) or not source_key.strip():
        raise ValueError(f"Authored batch must contain source_key: {path}")
    if not isinstance(artifacts, dict):
        raise ValueError(f"Authored batch must contain artifacts: {path}")
    return payload


def build_authoring_target(
    ledger: dict[str, Any],
    *,
    source_key: str,
    max_entries: int | None = None,
    version: str | None = None,
) -> dict[str, Any]:
    entries = ledger.get("entries")
    if not isinstance(entries, list):
        raise ValueError("Authoring ledger must contain entries.")
    normalized_source_key = source_key.strip()
    if not normalized_source_key:
        raise ValueError("source_key is required.")

    selected_entries = [
        _clone_entry(entry)
        for entry in entries
        if isinstance(entry, dict) and entry.get("source_key") == normalized_source_key
    ]
    if not selected_entries:
        raise ValueError(f"No authoring-ledger entries found for source_key={normalized_source_key}.")

    selected_entries.sort(
        key=lambda entry: (
            str(entry.get("priority")),
            _ARTIFACT_PRIORITY.get(str(entry.get("artifact_type")), 99),
            -int(entry.get("target_count", 0)),
        )
    )
    if isinstance(max_entries, int) and max_entries > 0:
        selected_entries = selected_entries[:max_entries]

    target_rows = sum(
        int(entry.get("target_count", 0))
        for entry in selected_entries
        if isinstance(entry.get("target_count"), int)
    )
    source_title = next(
        (
            str(entry.get("source_title"))
            for entry in selected_entries
            if isinstance(entry.get("source_title"), str) and str(entry.get("source_title")).strip()
        ),
        normalized_source_key,
    )

    return {
        "version": version or f"{ledger.get('version')}-{normalized_source_key}-target",
        "ledger_version": ledger.get("version"),
        "source_key": normalized_source_key,
        "source_title": source_title,
        "entry_count": len(selected_entries),
        "target_rows": target_rows,
        "entries": selected_entries,
        "starter_batch": [_clone_entry(entry) for entry in selected_entries[: min(3, len(selected_entries))]],
        "summary": {
            "artifact_types": dict(
                sorted(
                    Counter(
                        str(entry.get("artifact_type"))
                        for entry in selected_entries
                        if isinstance(entry.get("artifact_type"), str)
                    ).items()
                )
            ),
            "lane_targets": dict(
                sorted(
                    Counter(
                        lane
                        for entry in selected_entries
                        if isinstance(entry, dict)
                        for lane in _string_list(entry.get("lane_targets"))
                    ).items()
                )
            ),
        },
    }


def apply_authored_batch(
    ledger: dict[str, Any],
    batch: dict[str, Any],
    *,
    applied_at: str | None = None,
) -> dict[str, Any]:
    source_key, batch_id, artifacts = _validate_batch_application_inputs(ledger, batch)
    cloned_ledger = _clone_entry(ledger)
    entry_index = {
        (str(entry.get("source_key")), str(entry.get("artifact_type"))): entry
        for entry in cloned_ledger["entries"]
        if isinstance(entry, dict)
    }

    batch_counts: dict[str, int] = {}
    for artifact_type, rows in artifacts.items():
        if not isinstance(artifact_type, str):
            continue
        if not isinstance(rows, list):
            raise ValueError(f"Authored batch artifact rows must be a list: {artifact_type}")
        entry = entry_index.get((source_key, artifact_type))
        if entry is None:
            raise ValueError(
                f"Authored batch references unknown ledger entry: source_key={source_key} artifact_type={artifact_type}"
            )
        normalized_rows = _normalize_batch_rows(entry, rows)
        existing_rows = entry.get("draft_rows")
        if not isinstance(existing_rows, list):
            existing_rows = []
        merged_rows = _merge_rows_for_entry(entry, existing_rows, normalized_rows)
        entry["draft_rows"] = merged_rows
        entry["progress"] = _progress_for_entry(entry, merged_rows)
        applied_batches = entry.get("applied_batches")
        if not isinstance(applied_batches, list):
            applied_batches = []
        if batch_id not in applied_batches:
            applied_batches.append(batch_id)
        entry["applied_batches"] = applied_batches
        if isinstance(applied_at, str) and applied_at.strip():
            entry["last_applied_at"] = applied_at.strip()
        batch_counts[artifact_type] = len(normalized_rows)

    cloned_ledger["progress_summary"] = _progress_summary(cloned_ledger["entries"])
    source_batches = cloned_ledger.get("applied_batches")
    if not isinstance(source_batches, list):
        source_batches = []
    if batch_id not in source_batches:
        source_batches.append(batch_id)
    cloned_ledger["applied_batches"] = source_batches
    if isinstance(applied_at, str) and applied_at.strip():
        cloned_ledger["last_applied_at"] = applied_at.strip()
    cloned_ledger["last_batch"] = {
        "batch_id": batch_id,
        "source_key": source_key,
        "artifact_counts": dict(sorted(batch_counts.items())),
    }
    return cloned_ledger


def materialize_authoring_ledger(ledger: dict[str, Any], output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(f"{json.dumps(ledger, indent=2)}\n", encoding="utf-8")
    return output_path


def ensure_default_authoring_ledger_materialized(
    *,
    draft_pack_path: Path = DEFAULT_WAVE5_EXPANSION_DRAFT_PACK_PATH,
    output_path: Path = DEFAULT_WAVE5_AUTHORING_LEDGER_PATH,
) -> Path:
    if output_path.exists():
        return output_path
    draft_pack_file = ensure_default_expansion_draft_pack_materialized(output_path=draft_pack_path)
    ledger = build_authoring_ledger(load_authoring_draft_pack(draft_pack_file))
    return materialize_authoring_ledger(ledger, output_path)


def write_authoring_ledger_report(
    output_dir: Path,
    *,
    draft_pack_path: Path = DEFAULT_WAVE5_EXPANSION_DRAFT_PACK_PATH,
    ledger_path: Path = DEFAULT_WAVE5_AUTHORING_LEDGER_PATH,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    if ledger_path.exists():
        ledger = load_authoring_ledger(ledger_path)
        ledger["progress_summary"] = _progress_summary(ledger.get("entries"))
    else:
        draft_pack_file = ensure_default_expansion_draft_pack_materialized(output_path=draft_pack_path)
        ledger = build_authoring_ledger(load_authoring_draft_pack(draft_pack_file))
        ledger["progress_summary"] = _progress_summary(ledger.get("entries"))
        materialize_authoring_ledger(ledger, ledger_path)
    (output_dir / "authoring_ledger.json").write_text(
        f"{json.dumps(ledger, indent=2)}\n",
        encoding="utf-8",
    )
    (output_dir / "authoring_ledger.md").write_text(
        _authoring_ledger_markdown(ledger),
        encoding="utf-8",
    )
    return ledger


def write_applied_authoring_batch_report(
    output_dir: Path,
    *,
    batch_path: Path,
    ledger_path: Path = DEFAULT_WAVE5_AUTHORING_LEDGER_PATH,
    applied_at: str | None = None,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    applied_ledger = apply_authored_batch(
        load_authoring_ledger(ledger_path),
        load_authored_batch(batch_path),
        applied_at=applied_at,
    )
    materialize_authoring_ledger(applied_ledger, ledger_path)
    (output_dir / "authoring_ledger.json").write_text(
        f"{json.dumps(applied_ledger, indent=2)}\n",
        encoding="utf-8",
    )
    (output_dir / "authoring_ledger.md").write_text(
        _authoring_ledger_markdown(applied_ledger),
        encoding="utf-8",
    )
    return applied_ledger


def materialize_authoring_target(target: dict[str, Any], output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(f"{json.dumps(target, indent=2)}\n", encoding="utf-8")
    return output_path


def write_authoring_target_report(
    output_dir: Path,
    *,
    source_key: str,
    ledger_path: Path = DEFAULT_WAVE5_AUTHORING_LEDGER_PATH,
    max_entries: int | None = None,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    ledger = load_authoring_ledger(ledger_path)
    target = build_authoring_target(
        ledger,
        source_key=source_key,
        max_entries=max_entries,
    )
    materialize_authoring_target(target, output_dir / "authoring_target.json")
    (output_dir / "authoring_target.md").write_text(
        _authoring_target_markdown(target),
        encoding="utf-8",
    )
    return target


def _build_row_template(card: dict[str, Any]) -> dict[str, Any]:
    template: dict[str, Any] = {}
    for field_name in _string_list(card.get("required_fields")):
        template[field_name] = _template_value(field_name)
    return template


def _template_value(field_name: str) -> Any:
    if field_name in _LIST_FIELDS:
        return []
    if field_name in _OBJECT_FIELDS:
        return {}
    return ""


def _string_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [item.strip() for item in value if isinstance(item, str) and item.strip()]


def _sample_evidence(value: Any) -> list[dict[str, str]]:
    if not isinstance(value, list):
        return []
    evidence: list[dict[str, str]] = []
    for item in value:
        if not isinstance(item, dict):
            continue
        label = item.get("label")
        excerpt = item.get("excerpt")
        if isinstance(label, str) and isinstance(excerpt, str):
            evidence.append({"label": label, "excerpt": excerpt})
    return evidence


def _clone_entry(entry: dict[str, Any]) -> dict[str, Any]:
    return json.loads(json.dumps(entry))


def _normalize_batch_rows(entry: dict[str, Any], rows: list[Any]) -> list[dict[str, Any]]:
    artifact_type = str(entry.get("artifact_type"))
    required_fields = _string_list(entry.get("required_fields"))
    primary_id_field = _PRIMARY_ID_FIELD_MAP.get(artifact_type)
    normalized_rows: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError(f"Authored row must be an object for artifact_type={artifact_type}")
        normalized_row = _clone_entry(row)
        missing_fields = [field for field in required_fields if field not in normalized_row]
        if missing_fields:
            raise ValueError(
                f"Authored row missing required fields for artifact_type={artifact_type}: {', '.join(missing_fields)}"
            )
        if primary_id_field is None:
            raise ValueError(f"Unsupported artifact_type for authored batch: {artifact_type}")
        row_id = normalized_row.get(primary_id_field)
        if not isinstance(row_id, str) or not row_id.strip():
            raise ValueError(
                f"Authored row must contain non-empty {primary_id_field} for artifact_type={artifact_type}"
            )
        normalized_id = row_id.strip()
        if normalized_id in seen_ids:
            raise ValueError(
                f"Duplicate authored row id in batch for artifact_type={artifact_type}: {normalized_id}"
            )
        seen_ids.add(normalized_id)
        normalized_row[primary_id_field] = normalized_id
        normalized_rows.append(normalized_row)
    return normalized_rows


def _validate_batch_application_inputs(
    ledger: dict[str, Any],
    batch: dict[str, Any],
) -> tuple[str, str, dict[str, Any]]:
    entries = ledger.get("entries")
    if not isinstance(entries, list):
        raise ValueError("Authoring ledger must contain entries.")
    source_key = batch.get("source_key")
    artifacts = batch.get("artifacts")
    batch_id = batch.get("batch_id")
    if not isinstance(source_key, str) or not source_key.strip():
        raise ValueError("Authored batch must contain source_key.")
    if not isinstance(artifacts, dict):
        raise ValueError("Authored batch must contain artifacts.")
    if not isinstance(batch_id, str) or not batch_id.strip():
        raise ValueError("Authored batch must contain batch_id.")
    return source_key, batch_id, artifacts


def _merge_rows_for_entry(
    entry: dict[str, Any],
    existing_rows: list[Any],
    new_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    artifact_type = str(entry.get("artifact_type"))
    primary_id_field = _PRIMARY_ID_FIELD_MAP[artifact_type]
    merged: dict[str, dict[str, Any]] = {}
    for row in existing_rows:
        if not isinstance(row, dict):
            continue
        row_id = row.get(primary_id_field)
        if isinstance(row_id, str) and row_id.strip():
            merged[row_id.strip()] = _clone_entry(row)
    for row in new_rows:
        merged[str(row[primary_id_field]).strip()] = _clone_entry(row)
    return [merged[row_id] for row_id in sorted(merged)]


def _progress_for_entry(entry: dict[str, Any], merged_rows: list[dict[str, Any]]) -> dict[str, int]:
    target_count = int(entry.get("target_count", 0))
    drafted_count = len(merged_rows)
    previous_progress = entry.get("progress")
    reviewed_count = 0
    promoted_count = 0
    if isinstance(previous_progress, dict):
        previous_reviewed = previous_progress.get("reviewed_count")
        previous_promoted = previous_progress.get("promoted_count")
        if isinstance(previous_reviewed, int):
            reviewed_count = min(previous_reviewed, drafted_count)
        if isinstance(previous_promoted, int):
            promoted_count = min(previous_promoted, reviewed_count)
    return {
        "drafted_count": drafted_count,
        "reviewed_count": reviewed_count,
        "promoted_count": promoted_count,
        "remaining_count": max(target_count - drafted_count, 0),
    }


def _progress_summary(entries: Any) -> dict[str, Any]:
    if not isinstance(entries, list):
        return {"drafted_rows": 0, "remaining_rows": 0, "by_source_key": {}, "by_artifact_type": {}}

    drafted_rows = 0
    remaining_rows = 0
    by_source_key: dict[str, dict[str, int]] = {}
    by_artifact_type: dict[str, dict[str, int]] = {}
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        progress = entry.get("progress") if isinstance(entry.get("progress"), dict) else {}
        drafted_count = progress.get("drafted_count") if isinstance(progress.get("drafted_count"), int) else 0
        remaining_count = progress.get("remaining_count") if isinstance(progress.get("remaining_count"), int) else 0
        target_count = entry.get("target_count") if isinstance(entry.get("target_count"), int) else 0
        source_key = str(entry.get("source_key"))
        artifact_type = str(entry.get("artifact_type"))
        drafted_rows += drafted_count
        remaining_rows += remaining_count
        by_source = by_source_key.setdefault(source_key, {"entry_count": 0, "target_rows": 0, "drafted_rows": 0, "remaining_rows": 0})
        by_source["entry_count"] += 1
        by_source["target_rows"] += target_count
        by_source["drafted_rows"] += drafted_count
        by_source["remaining_rows"] += remaining_count
        by_artifact = by_artifact_type.setdefault(artifact_type, {"entry_count": 0, "target_rows": 0, "drafted_rows": 0, "remaining_rows": 0})
        by_artifact["entry_count"] += 1
        by_artifact["target_rows"] += target_count
        by_artifact["drafted_rows"] += drafted_count
        by_artifact["remaining_rows"] += remaining_count
    return {
        "drafted_rows": drafted_rows,
        "remaining_rows": remaining_rows,
        "by_source_key": dict(sorted(by_source_key.items())),
        "by_artifact_type": dict(sorted(by_artifact_type.items())),
    }


def _authoring_ledger_markdown(ledger: dict[str, Any]) -> str:
    lines = [
        "# Wave-Five Authoring Ledger",
        "",
        f"- Version: `{ledger.get('version')}`",
        f"- Draft-pack version: `{ledger.get('draft_pack_version')}`",
        f"- Ledger entries: `{ledger.get('ledger_entry_count')}`",
        f"- Total target rows: `{ledger.get('total_target_rows')}`",
        "",
        "## Summary",
        "",
    ]
    summary = ledger.get("summary") if isinstance(ledger.get("summary"), dict) else {}
    progress_summary = ledger.get("progress_summary") if isinstance(ledger.get("progress_summary"), dict) else {}
    lines.append("### Draft Progress")
    lines.append("")
    lines.append(f"- Drafted rows: `{progress_summary.get('drafted_rows', 0)}`")
    lines.append(f"- Remaining rows: `{progress_summary.get('remaining_rows', 0)}`")
    lines.append("")
    for label, key in (("By Source", "by_source_key"), ("By Artifact Type", "by_artifact_type")):
        lines.append(f"### {label}")
        lines.append("")
        values = summary.get(key) if isinstance(summary, dict) else {}
        if isinstance(values, dict):
            for name, count in sorted(values.items()):
                lines.append(f"- `{name}`: `{count}`")
        lines.append("")
    for label, key in (("Progress By Source", "by_source_key"), ("Progress By Artifact Type", "by_artifact_type")):
        lines.append(f"### {label}")
        lines.append("")
        values = progress_summary.get(key) if isinstance(progress_summary, dict) else {}
        if isinstance(values, dict):
            for name, metrics in sorted(values.items()):
                if not isinstance(metrics, dict):
                    continue
                lines.append(
                    f"- `{name}`: drafted=`{metrics.get('drafted_rows', 0)}` / "
                    f"target=`{metrics.get('target_rows', 0)}` / remaining=`{metrics.get('remaining_rows', 0)}`"
                )
        lines.append("")

    lines.extend(["## Priority Worklist", ""])
    entries = ledger.get("entries")
    if isinstance(entries, list):
        for entry in sorted(
            [
                item
                for item in entries
                if isinstance(item, dict)
                and isinstance(item.get("progress"), dict)
                and int(item["progress"].get("remaining_count", 0)) > 0
            ],
            key=lambda item: (
                str(item.get("priority")),
                -int(item["progress"].get("remaining_count", 0)),
                _ARTIFACT_PRIORITY.get(str(item.get("artifact_type")), 99),
            ),
        )[:12]:
            lines.append(
                f"- `{entry['ledger_entry_id']}` | `{entry['artifact_type']}` | "
                f"remaining=`{entry['progress'].get('remaining_count', 0)}` | source=`{entry['source_key']}`"
            )
    lines.append("")
    return "\n".join(lines)


def _authoring_target_markdown(target: dict[str, Any]) -> str:
    lines = [
        "# Source Authoring Target",
        "",
        f"- Source key: `{target.get('source_key')}`",
        f"- Source title: `{target.get('source_title')}`",
        f"- Entry count: `{target.get('entry_count')}`",
        f"- Target rows: `{target.get('target_rows')}`",
        "",
        "## Summary",
        "",
    ]
    summary = target.get("summary") if isinstance(target.get("summary"), dict) else {}
    for label, key in (("Artifact Types", "artifact_types"), ("Lane Targets", "lane_targets")):
        lines.append(f"### {label}")
        lines.append("")
        values = summary.get(key) if isinstance(summary, dict) else {}
        if isinstance(values, dict):
            for name, count in sorted(values.items()):
                lines.append(f"- `{name}`: `{count}`")
        lines.append("")

    lines.extend(["## Starter Batch", ""])
    starter_batch = target.get("starter_batch")
    if isinstance(starter_batch, list):
        for entry in starter_batch:
            if not isinstance(entry, dict):
                continue
            lines.append(
                f"- `{entry['ledger_entry_id']}` | `{entry['artifact_type']}` | "
                f"target=`{entry['target_count']}`"
            )
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output_dir", type=Path, help="Directory to receive the authoring-ledger report")
    parser.add_argument(
        "--draft-pack-path",
        type=Path,
        default=DEFAULT_WAVE5_EXPANSION_DRAFT_PACK_PATH,
        help="Expansion draft-pack JSON path",
    )
    parser.add_argument(
        "--ledger-path",
        type=Path,
        default=DEFAULT_WAVE5_AUTHORING_LEDGER_PATH,
        help="Materialized authoring-ledger JSON asset path",
    )
    parser.add_argument(
        "--source-key",
        default=None,
        help="Optional source key for a focused authoring target bundle.",
    )
    parser.add_argument(
        "--max-entries",
        type=int,
        default=None,
        help="Optional cap for focused authoring target entries.",
    )
    parser.add_argument(
        "--apply-batch-path",
        type=Path,
        default=None,
        help="Optional authored-batch JSON to merge into the ledger before reporting.",
    )
    parser.add_argument(
        "--applied-at",
        default=None,
        help="Optional timestamp/label to record when applying an authored batch.",
    )
    args = parser.parse_args()
    if args.apply_batch_path is None:
        write_authoring_ledger_report(
            args.output_dir,
            draft_pack_path=args.draft_pack_path,
            ledger_path=args.ledger_path,
        )
    else:
        write_applied_authoring_batch_report(
            args.output_dir,
            batch_path=args.apply_batch_path,
            ledger_path=args.ledger_path,
            applied_at=args.applied_at,
        )
    if isinstance(args.source_key, str) and args.source_key.strip():
        write_authoring_target_report(
            args.output_dir / f"target_{args.source_key.strip()}",
            source_key=args.source_key,
            ledger_path=args.ledger_path,
            max_entries=args.max_entries,
        )


if __name__ == "__main__":
    main()
