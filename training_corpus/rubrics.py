"""Rubric normalization for evaluator and benchmark corpus lanes."""

from __future__ import annotations

from typing import Any

from .model import CorpusLane


def _metadata(raw: dict[str, Any]) -> dict[str, Any]:
    metadata = raw.get("metadata")
    if not isinstance(metadata, dict):
        return {}
    return metadata


def normalize_rubric_items(raw: dict[str, Any], lane: CorpusLane) -> list[dict[str, Any]]:
    metadata = _metadata(raw)

    if lane not in {"evaluator", "benchmark"}:
        return []

    candidates = metadata.get("rubric_items")
    if not isinstance(candidates, list):
        candidates = metadata.get("criteria")
    if not isinstance(candidates, list):
        rubric = metadata.get("rubric")
        if isinstance(rubric, dict):
            nested_items = rubric.get("items")
            if isinstance(nested_items, list):
                candidates = nested_items

    normalized: list[dict[str, Any]] = []
    if not isinstance(candidates, list):
        return normalized

    for index, item in enumerate(candidates):
        if not isinstance(item, dict):
            continue
        name = item.get("name") or item.get("criterion") or item.get("label")
        if not isinstance(name, str) or not name.strip():
            continue
        weight = item.get("weight", 1.0)
        normalized.append(
            {
                "criterion_id": str(item.get("criterion_id") or f"{lane}-{index + 1}"),
                "name": name.strip(),
                "weight": weight if isinstance(weight, (int, float)) else 1.0,
                "required": bool(item.get("required", True)),
                "notes": str(item.get("notes") or "").strip(),
            }
        )
    return normalized


def normalize_clinician_review(raw: dict[str, Any], lane: CorpusLane) -> dict[str, Any] | None:
    if lane not in {"evaluator", "benchmark"}:
        return None

    metadata = _metadata(raw)
    rubric_items = normalize_rubric_items(raw, lane)
    candidate = metadata.get("clinician_review")

    if isinstance(candidate, dict):
        status = str(candidate.get("status") or metadata.get("clinician_review_status") or "unreviewed")
        reviewer_role = str(candidate.get("reviewer_role") or "clinician")
        reviewer_count = candidate.get("reviewer_count", metadata.get("clinician_reviewer_count", 0))
        calibration_subset = bool(
            candidate.get("calibration_subset", metadata.get("calibration_subset", lane == "benchmark"))
        )
        return {
            "required": bool(candidate.get("required", True)),
            "status": status,
            "reviewer_role": reviewer_role,
            "reviewer_count": reviewer_count if isinstance(reviewer_count, int) else 0,
            "calibration_subset": calibration_subset,
            "notes": str(candidate.get("notes") or metadata.get("clinician_review_notes") or "").strip(),
        }

    if not rubric_items and not bool(metadata.get("clinician_review_required")):
        return None

    return {
        "required": True,
        "status": str(metadata.get("clinician_review_status") or "unreviewed"),
        "reviewer_role": str(metadata.get("clinician_reviewer_role") or "clinician"),
        "reviewer_count": int(metadata.get("clinician_reviewer_count") or 0),
        "calibration_subset": bool(metadata.get("calibration_subset", lane == "benchmark")),
        "notes": str(metadata.get("clinician_review_notes") or "").strip(),
    }
