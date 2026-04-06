"""
Helpers for applying canonical intake routing to orchestrator records.
"""

from __future__ import annotations

from typing import Any

from ai.pipelines.orchestrator.configs.stages import STAGE3_ID, STAGE4_ID
from ai.pipelines.orchestrator.data_splitter import DataSplitter
from ai.pipelines.orchestrator.ingestion.intake_gates import OrchestratorIntakeGates

_PROMOTED_FEEDER_ALLOWED_LANES: dict[str, set[str]] = {
    "edge_case": {STAGE3_ID},
    "nightmare_scenarios": {STAGE3_ID},
    "cot_reasoning": {STAGE3_ID},
    "voice_persona": {STAGE3_ID, STAGE4_ID},
    "youtube_transcript": {STAGE3_ID, STAGE4_ID},
    "dual_persona": {STAGE3_ID, STAGE4_ID},
}


def apply_intake_routing(
    records: list[dict[str, Any]],
    *,
    source_family: str,
    intake_gates: OrchestratorIntakeGates,
) -> list[dict[str, Any]]:
    """Apply canonical intake metadata to a batch of records."""
    routed_records: list[dict[str, Any]] = []

    for record in records:
        if not isinstance(record, dict):
            continue

        metadata = record.setdefault("metadata", {})
        decision = intake_gates.evaluate(
            source_family=source_family,
            record=record,
        )

        metadata["source_family"] = source_family
        metadata["stage"] = decision.target_lane
        metadata["intake_target_lane"] = decision.target_lane
        metadata["intake_route_reason"] = (
            decision.reasons[0] if decision.reasons else "Canonical intake route applied."
        )
        metadata["intake_route_reasons"] = list(decision.reasons)
        trusted_source = bool(metadata.get("trusted_source"))
        requires_human_review = decision.requires_human_review
        if trusted_source and decision.target_lane == "stage1_foundation":
            requires_human_review = False
            metadata["intake_route_reasons"].append(
                "Trusted Stage 1 source bypassed manual-review fallback."
            )
        metadata["requires_human_review"] = requires_human_review

        if decision.split:
            metadata["split"] = decision.split

        allowed_lanes = _PROMOTED_FEEDER_ALLOWED_LANES.get(decision.source_family)
        if allowed_lanes is not None and decision.target_lane not in allowed_lanes:
            continue

        routed_records.append(record)

    return routed_records


def split_records_with_preferences(records: list[dict[str, Any]]) -> Any:
    """Split records while honoring explicit metadata.split assignments."""
    splitter = DataSplitter(train_ratio=0.70, val_ratio=0.15, test_ratio=0.15)
    forced_splits: dict[str, list[dict[str, Any]]] = {"train": [], "val": [], "test": []}
    flexible_records: list[dict[str, Any]] = []

    for record in records:
        metadata = record.get("metadata", {})
        split_preference = metadata.get("split")
        if split_preference in forced_splits:
            forced_splits[split_preference].append(record)
        else:
            flexible_records.append(record)

    auto_split = splitter.split(list(flexible_records), shuffle=True, seed=42)
    auto_split.train = forced_splits["train"] + auto_split.train
    auto_split.val = forced_splits["val"] + auto_split.val
    auto_split.test = forced_splits["test"] + auto_split.test
    auto_split.metadata["forced_counts"] = {
        split_name: len(split_records)
        for split_name, split_records in forced_splits.items()
    }
    return auto_split


__all__ = ["apply_intake_routing", "split_records_with_preferences"]
