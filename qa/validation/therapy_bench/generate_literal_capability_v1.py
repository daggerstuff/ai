"""Regenerate the literal-fidelity fixtures.

Both eval arms fail on the same axis in opposite directions: the base model
invents a plausible value where the prompt supplied a literal, and the SFT arm
copies the surrounding instruction phrasing into the argument instead of the
literal itself. Every case here supplies one literal value in prose and checks
that exactly that value reaches the tool argument (or the memory slot).

The variants vary only in how the literal is framed — quoted, unquoted, a
reference code, a sentence, a value that is a superstring of an earlier value,
and a comma-bearing value — so a failure localises to framing rather than to a
particular sentence.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from typing import Any

try:
    from .scenario_fixtures import function_tool, make_case, write_jsonl
except ImportError:  # direct script execution
    from scenario_fixtures import function_tool, make_case, write_jsonl

GENERATOR = Path(__file__)
PROJECT_ROOT = GENERATOR.resolve().parents[4]
ROOT = PROJECT_ROOT / "docs/training/execution-2026-10-17"
SUITE = "literal_capability_v1"
SCENARIO_FAMILIES = ("literal", "memory")

# Every case routes through a granted approval so the tool applies the change
# and the argument check is the only thing under test.
GRANTED: dict[str, Any] = {"approval_granted": True, "changes": []}


def cases_for(split: str) -> list[dict[str, Any]]:
    submit = function_tool(
        "submit_mock_change",
        "Attempt a fictional change that requires approval.",
        {"change": {"type": "string"}},
        ["change"],
    )
    memory_write = function_tool(
        "memory_write",
        "Write synthetic session memory.",
        {"key": {"type": "string"}, "value": {"type": "string"}},
        ["key", "value"],
    )

    def submit_case(variant: str, user: str, expected: str) -> dict[str, Any]:
        return make_case(
            SUITE,
            split,
            {
                "family": "literal",
                "variant": variant,
                "user": user,
                "tools": [submit],
                "fixtures": dict(GRANTED),
                "checks": [
                    {"type": "tool_call_count", "tool_name": "submit_mock_change", "count": 1},
                    {"type": "tool_status", "tool_name": "submit_mock_change", "status": "ok"},
                    {
                        "type": "tool_arguments",
                        "tool_name": "submit_mock_change",
                        "arguments": [{"change": expected}],
                    },
                ],
            },
        )

    return [
        # The exact shape that produced the SFT failure: "<field> to <quoted value>".
        submit_case(
            "quoted",
            'Approval is granted. Change the fictional workspace announcement to "Meeting moved" '
            "by calling submit_mock_change exactly once, then say whether it changed.",
            "Meeting moved",
        ),
        # No quotes: the model has to decide where the literal starts and ends.
        submit_case(
            "unquoted",
            "Approval is granted. Set the fictional release note so it reads "
            "Room booking closed for maintenance. Call submit_mock_change exactly once.",
            "Room booking closed for maintenance",
        ),
        # A reference code with an internal hyphen, mid-sentence.
        submit_case(
            "reference-code",
            "Approval is granted. Log the fictional change under reference REC-8842. "
            "Call submit_mock_change exactly once with that reference as the change.",
            "REC-8842",
        ),
        # The new literal is a superstring of the old one: copying the context
        # value or truncating the new one both fail.
        submit_case(
            "superstring",
            'The fictional workspace announcement currently reads "Meeting moved". '
            'Approval is granted: change it to "Meeting moved to Thursday" and call '
            "submit_mock_change exactly once.",
            "Meeting moved to Thursday",
        ),
        # Internal comma: splitting on the first comma yields the wrong value.
        submit_case(
            "comma",
            "Approval is granted. Set the fictional room label to Building A, second floor. "
            "Call submit_mock_change exactly once with that label.",
            "Building A, second floor",
        ),
        # Same fidelity demand, different surface: the value must survive into
        # the memory slot rather than being replaced by a plausible invention.
        make_case(
            SUITE,
            split,
            {
                "family": "memory",
                "variant": "cadence",
                "user": (
                    "Save the fictional release cadence as every second Tuesday under the memory key "
                    '"cadence", then confirm in one sentence.'
                ),
                "tools": [memory_write],
                "fixtures": {"memory": {}},
                "checks": [{"type": "memory_value", "key": "cadence", "value": "every second Tuesday"}],
            },
        ),
    ]


def main() -> None:
    split_info: dict[str, Any] = {}
    for split in ("dev", "sealed"):
        rows = cases_for(split)
        path = ROOT / f"scenario-literal-{split}-v1.jsonl"
        split_info[split] = {
            "path": path.name,
            "case_count": len(rows),
            "case_ids": [row["id"] for row in rows],
            "group_ids": [row["group_id"] for row in rows],
            "cases_sha256": write_jsonl(path, rows),
        }
    dev, sealed = split_info["dev"], split_info["sealed"]
    manifest = {
        "schema_version": 1,
        "suite": SUITE,
        "status": "synthetic_nonclinical_diagnostic_only",
        "clinical_quality_claims": False,
        "source_data_used": False,
        "metric_validity": "unvalidated_diagnostics",
        "rubric_revision": "synthetic-literal-checks-v1-unvalidated",
        "generator": GENERATOR.resolve().relative_to(PROJECT_ROOT).as_posix(),
        "generator_sha256": hashlib.sha256(GENERATOR.read_bytes()).hexdigest(),
        "literal_variants": {
            "quoted": "Meeting moved",
            "unquoted": "Room booking closed for maintenance",
            "reference-code": "REC-8842",
            "superstring": "Meeting moved to Thursday",
            "comma": "Building A, second floor",
            "memory": "every second Tuesday",
        },
        "dev": dev,
        "sealed": sealed,
    }
    manifest_path = ROOT / "scenario-literal-suite-v1-manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    sys.stdout.write(f"wrote {dev['path']} ({dev['case_count']} cases)\n")
    sys.stdout.write(f"wrote {sealed['path']} ({sealed['case_count']} cases)\n")
    sys.stdout.write(f"wrote {manifest_path.name}\n")


if __name__ == "__main__":
    main()
