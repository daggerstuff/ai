"""Rebuild per-run results files from the RECORD and SUMMARY lines of a job log.

The OVH volume push has failed at job teardown on every attempt so far, which
makes ``ovhai job logs`` the only durable copy of a run. ``ovh_v22_eval/launch.sh``
prints one ``RECORD`` line per graded case and one ``SUMMARY`` line per arm, and
the SUMMARY carries the run name its records belong to. This turns such a log
back into the ``results.jsonl`` files the strict gate grades.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Iterable
from pathlib import Path
from typing import Any

RECORD_PREFIX = "RECORD "
SUMMARY_PREFIX = "SUMMARY "
UNATTRIBUTED = "unattributed"


def payload(line: str) -> str:
    """Strip the ``[job] `` prefix the OVH log puts in front of every line."""
    _, separator, rest = line.partition("[job] ")
    return (rest if separator else line).strip()


def recover(lines: Iterable[str]) -> dict[str, list[dict[str, Any]]]:
    """Group RECORD rows by the run named in the SUMMARY that closes their block."""
    runs: dict[str, list[dict[str, Any]]] = {}
    buffered: list[dict[str, Any]] = []
    for line in lines:
        body = payload(line)
        if body.startswith(RECORD_PREFIX):
            buffered.append(json.loads(body[len(RECORD_PREFIX) :]))
        elif body.startswith(SUMMARY_PREFIX):
            summary: dict[str, Any] = json.loads(body[len(SUMMARY_PREFIX) :])
            run = summary.get("run")
            if run and buffered:
                runs.setdefault(str(run), []).extend(buffered)
            buffered = []
    if buffered:
        # A run that died mid-summary still has readable replies, so keep them
        # rather than dropping the only copy.
        runs.setdefault(UNATTRIBUTED, []).extend(buffered)
    return runs


def write_runs(runs: dict[str, list[dict[str, Any]]], out_root: Path) -> dict[str, dict[str, int]]:
    """Write ``<out_root>/<run>/results.jsonl`` and return per-run pass counts."""
    counts: dict[str, dict[str, int]] = {}
    for run, records in runs.items():
        directory = out_root / run
        directory.mkdir(parents=True, exist_ok=True)
        with (directory / "results.jsonl").open("w", encoding="utf-8") as handle:
            for record in records:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        arms: dict[str, int] = {}
        for record in records:
            arm = str(record.get("arm_id"))
            arms[arm] = arms.get(arm, 0) + int(bool(record.get("passed")))
        counts[run] = arms
    return counts


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--log", type=Path, required=True, help="output of `ovhai job logs`")
    parser.add_argument("--out-dir", type=Path, required=True, help="where to write the run directories")
    args = parser.parse_args(argv)

    runs = recover(args.log.read_text().splitlines())
    if not runs:
        print(f"no RECORD lines in {args.log}", file=sys.stderr)
        return 1
    counts = write_runs(runs, args.out_dir)
    for run, arms in counts.items():
        print(f"{run}: {sum(arms.values())} passed, per arm {arms}")
    if UNATTRIBUTED in counts:
        print("warning: records after the last SUMMARY went to 'unattributed'", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
