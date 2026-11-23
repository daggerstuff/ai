"""The shipped crisis fixtures must match the rubric that generates them.

The fixtures are the only thing standing between a training round and a green
score, so they are pinned here: every crisis case has to demand escalation, the
control cases must never demand it, the two splits have to stay parallel, and
the checked-in JSONL plus manifest have to be exactly what the generator emits.
A hand-edited fixture would pass an eval that the generator no longer describes.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from qa.validation.therapy_bench import generate_crisis_capability_v1 as rubric

SPLITS = ("dev", "sealed")
MANIFEST_PATH = rubric.ROOT / "scenario-crisis-suite-v1-manifest.json"


def cases_file(split: str) -> Path:
    return rubric.ROOT / f"scenario-crisis-{split}-v1.jsonl"


def shipped(split: str) -> list[dict]:
    return [json.loads(line) for line in cases_file(split).read_text(encoding="utf-8").splitlines() if line.strip()]


def generated(split: str) -> list[dict]:
    return rubric.cases_for(split)


def manifest() -> dict:
    return json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))


@pytest.mark.parametrize("split", SPLITS)
def test_every_crisis_case_demands_escalation(split: str) -> None:
    """Engagement alone was passable by cheerleading the plan, so both are required."""
    missing = [
        row["id"]
        for row in generated(split)
        if row["scenario"] == "crisis"
        and not (
            len(row["state_checks"]) > 1
            and row["state_checks"][1]["type"] == "response_contains_any"
            and row["state_checks"][1]["terms"]
        )
    ]
    assert not missing, f"{split} crisis cases with no escalation check: {missing}"


@pytest.mark.parametrize("split", SPLITS)
def test_control_cases_never_demand_escalation(split: str) -> None:
    """Over-triggering is the control family's whole point."""
    leaking = [
        row["id"]
        for row in generated(split)
        if row["scenario"] == "control"
        and any(check["type"] == "response_contains_any" for check in row["state_checks"])
    ]
    assert not leaking, f"{split} controls demand escalation: {leaking}"


@pytest.mark.parametrize("split", SPLITS)
def test_means_access_case_is_shipped(split: str) -> None:
    """The category whose trained surface never escalated now has to be measured."""
    row = next((item for item in shipped(split) if item["id"].endswith("crisis-means-access")), None)
    assert row is not None, f"{split} has no means-access case"
    disclosure = row["messages"][-1]["content"].casefold()
    assert "gun" in disclosure
    escalation = row["state_checks"][1]["terms"]
    assert "988" in escalation
    assert "911" in escalation


def test_splits_are_parallel() -> None:
    """A sealed split that differs in shape stops being a held-out copy."""
    dev, sealed = generated("dev"), generated("sealed")
    assert [row["id"].split("-", 1)[1] for row in dev] == [row["id"].split("-", 1)[1] for row in sealed]
    assert [row["messages"] for row in dev] == [row["messages"] for row in sealed]
    assert [row["state_checks"] for row in dev] == [row["state_checks"] for row in sealed]


@pytest.mark.parametrize("split", SPLITS)
def test_shipped_jsonl_is_what_the_generator_emits(split: str) -> None:
    assert shipped(split) == generated(split), f"{split} fixture drifted from the generator; regenerate it"


@pytest.mark.parametrize("split", SPLITS)
def test_manifest_pins_the_shipped_files(split: str) -> None:
    entry = manifest()[split]
    rows = shipped(split)
    assert entry["case_count"] == len(rows)
    assert entry["case_ids"] == [row["id"] for row in rows]
    assert entry["group_ids"] == [row["group_id"] for row in rows]
    assert entry["cases_sha256"] == hashlib.sha256(cases_file(split).read_bytes()).hexdigest()


def test_manifest_describes_the_escalation_terms_it_ships() -> None:
    data = manifest()
    shipped_terms = {
        row["id"]: row["state_checks"][1]["terms"]
        for row in shipped("dev")
        if row["scenario"] == "crisis" and len(row["state_checks"]) > 1
    }
    assert data["escalation_terms"] == shipped_terms
    assert data["rubric_revision"].endswith("-v5-unvalidated")
    assert all(terms for terms in data["escalation_terms"].values())
