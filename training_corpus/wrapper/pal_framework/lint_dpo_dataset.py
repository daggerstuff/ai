"""Lint DPO preference-pair datasets for TRL ``DPOTrainer`` compatibility.

PIX-4075 — Phase 3.2: DPO Linting and TRL Integration.

Each record must conform to the schema expected by ``trl.DPOTrainer``::

    {"prompt": str, "chosen": list[message], "rejected": list[message]}

and ``chosen`` / ``rejected`` must share an identical conversational prefix
(identical up to the final assistant turn). TRL requires this so the only
supervised signal is the final response.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

VALID_ROLES = ("system", "user", "assistant", "tool")


class DpoDatasetError(Exception):
    """Raised when a DPO dataset fails linting."""


@dataclass
class LintIssue:
    line: int
    field: str
    message: str

    def __str__(self) -> str:
        return f"line {self.line}: [{self.field}] {self.message}"


@dataclass
class LintReport:
    path: Path
    total: int = 0
    issues: list[LintIssue] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.issues

    def error_count(self) -> int:
        return len(self.issues)


def _message_eq(a: dict[str, Any], b: dict[str, Any]) -> bool:
    return a.get("role") == b.get("role") and a.get("content") == b.get("content")


def _validate_message(msg: Any, line: int, field_name: str, issues: list[LintIssue]) -> None:
    if not isinstance(msg, dict):
        issues.append(LintIssue(line, field_name, f"message must be an object, got {type(msg).__name__}"))
        return
    role = msg.get("role")
    content = msg.get("content")
    if not isinstance(role, str) or role not in VALID_ROLES:
        issues.append(LintIssue(line, field_name, f"invalid role {role!r} (expected one of {VALID_ROLES})"))
    if not isinstance(content, str):
        issues.append(LintIssue(line, field_name, f"content must be a string, got {type(content).__name__}"))


def _validate_prefix(chosen: list[Any], rejected: list[Any], line: int, issues: list[LintIssue]) -> None:
    if not chosen or not rejected:
        issues.append(LintIssue(line, "prefix", "chosen and rejected must be non-empty message lists"))
        return
    if len(chosen) != len(rejected):
        issues.append(LintIssue(line, "prefix", "chosen and rejected must have equal length"))
        return
    for i in range(len(chosen) - 1):
        if not _message_eq(chosen[i], rejected[i]):
            issues.append(LintIssue(line, "prefix", f"chosen[{i}] and rejected[{i}] diverge before the final turn"))
            return
    last_c = chosen[-1]
    last_r = rejected[-1]
    if last_c.get("role") != "assistant" or last_r.get("role") != "assistant":
        issues.append(LintIssue(line, "prefix", "the divergent final turn must be an assistant message"))
    if _message_eq(last_c, last_r):
        issues.append(LintIssue(line, "prefix", "chosen and rejected must differ in the final assistant turn"))


def validate_record(record: Any, line: int) -> list[LintIssue]:
    """Return all lint issues for a single JSONL record (empty list == clean)."""
    issues: list[LintIssue] = []
    if not isinstance(record, dict):
        return [LintIssue(line, "record", f"record must be an object, got {type(record).__name__}")]
    for key in ("prompt", "chosen", "rejected"):
        if key not in record:
            issues.append(LintIssue(line, key, f"missing required key '{key}'"))
    if "prompt" in record and not isinstance(record["prompt"], str):
        issues.append(LintIssue(line, "prompt", f"prompt must be a string, got {type(record['prompt']).__name__}"))
    chosen = record.get("chosen")
    rejected = record.get("rejected")
    if "chosen" in record:
        if not isinstance(chosen, list):
            issues.append(LintIssue(line, "chosen", f"chosen must be a list, got {type(chosen).__name__}"))
        else:
            for i, msg in enumerate(chosen):
                _validate_message(msg, line, f"chosen[{i}]", issues)
    if "rejected" in record:
        if not isinstance(rejected, list):
            issues.append(LintIssue(line, "rejected", f"rejected must be a list, got {type(rejected).__name__}"))
        else:
            for i, msg in enumerate(rejected):
                _validate_message(msg, line, f"rejected[{i}]", issues)
    if isinstance(chosen, list) and isinstance(rejected, list):
        _validate_prefix(chosen, rejected, line, issues)
    return issues


def lint_file(path: Path) -> LintReport:
    """Lint a JSONL dataset file, accumulating one ``LintIssue`` per problem line."""
    report = LintReport(path=path)
    with path.open(encoding="utf-8") as fh:
        for lineno, raw in enumerate(fh, start=1):
            stripped = raw.strip()
            if not stripped:
                continue
            report.total += 1
            try:
                record = json.loads(stripped)
            except json.JSONDecodeError as exc:
                report.issues.append(LintIssue(lineno, "json", f"invalid JSON: {exc}"))
                continue
            report.issues.extend(validate_record(record, lineno))
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Lint a DPO dataset for TRL DPOTrainer compatibility.")
    parser.add_argument("paths", nargs="+", type=Path, help="JSONL dataset file(s) to lint")
    args = parser.parse_args(argv)

    failed = False
    for path in args.paths:
        if not path.exists():
            print(f"MISSING {path}", file=sys.stderr)
            failed = True
            continue
        report = lint_file(path)
        if report.ok:
            print(f"OK   {report.path} ({report.total} records)")
        else:
            failed = True
            print(f"FAIL {report.path} ({report.total} records, {len(report.issues)} issues)")
            for issue in report.issues:
                print(f"  - {issue}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
