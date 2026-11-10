#!/usr/bin/env python3
"""Export a release's train split as an Adaption AI raw-mode upload.

Adaption takes ``prompt``/``completion`` pairs, so a multi-turn release row has
to be unrolled: one datum per assistant turn, with the prompt holding every turn
through the preceding user message and the completion holding the assistant turn
verbatim. That is the run-2 format the account already accepts, and it keeps the
ledger prefix intact so the completion is exactly what the release trains on.

Row selection is deliberately the same deterministic slice the Fireworks export
uses (every routing row up to a cap, every acute-risk client turn, then a
hash-ordered filler sample). Adaption charges nothing for ingest but the run
afterwards is paid per token, so a bounded slice answers the real question —
does the corpus carry its behaviour onto a second trainer — without paying twice
for the same 82k rows W&B already trains on.

The release payload weights every message (user 0, assistant 1). Adaption masks
prompt tokens through its own ``train_on_inputs`` hyperparameter instead of a
per-message weight, so the emitted datums carry role and content only.

Run (from ai/):
  PYTHONPATH=.. python -m training.export_adaption_release --dry-run
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any

from training.export_fireworks_release import (
    SelectionOptions,
    _read_json,
    _sha256,
    select_rows,
)

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MANIFEST = (
    ROOT / "docs/training/execution-2026-11-10/wandb-full-corpus-current-master-20261110-v19.json"
)
DEFAULT_OUT_DIR = ROOT / "ai/training/output/adaption"
DATUM_FLOOR = 10_000  # Adaption effective-row floor for Qwen3.6-35B-A3B
CHAR_PER_TOKEN = 4


def unroll(messages: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], int]:
    """Return one ``{prompt, completion}`` datum per assistant turn, plus skips.

    Arc sessions open with the therapist's ledger-bearing turn, so the first
    assistant message of 11,420 train rows has nothing before it. That turn
    cannot be a completion without a prompt, so it is skipped and counted.
    """
    datums: list[dict[str, Any]] = []
    skipped = 0
    for index, message in enumerate(messages):
        if message.get("role") != "assistant":
            continue
        if index == 0 or messages[index - 1].get("role") != "user":
            skipped += 1
            continue
        prompt = [
            {"role": prior["role"], "content": prior["content"]} for prior in messages[:index]
        ]
        datums.append(
            {
                "prompt": prompt,
                "completion": [{"role": "assistant", "content": message["content"]}],
            }
        )
    return datums, skipped


def _datum_chars(datum: dict[str, Any]) -> int:
    return sum(len(message["content"]) for message in datum["prompt"]) + sum(
        len(message["content"]) for message in datum["completion"]
    )


def export(
    manifest_path: Path,
    out_dir: Path,
    *,
    options: SelectionOptions | None = None,
    dry_run: bool = False,
) -> dict[str, Any]:
    """Write the Adaption upload file and return its identity record."""
    manifest = _read_json(manifest_path)
    train = manifest["partitions"]["train"]
    train_path = ROOT / train["path"]
    if not train_path.is_file():
        raise FileNotFoundError(f"train partition not found: {train_path}")

    selections, counts = select_rows(train_path, options or SelectionOptions())
    if not selections:
        raise ValueError("selection produced no rows")

    datums: list[dict[str, Any]] = []
    skipped = 0
    for selection in selections:
        if selection.payload is None:
            raise ValueError(f"row {selection.row_number} has no emitted payload")
        rows, row_skipped = unroll(selection.payload["messages"])
        datums.extend(rows)
        skipped += row_skipped

    if not datums:
        raise ValueError("unrolling produced no datums")

    chars = sum(_datum_chars(datum) for datum in datums)
    summary: dict[str, Any] = {
        "release_id": manifest["release_id"],
        "source_manifest": str(manifest_path),
        "source_train": str(train_path),
        "source_train_sha256": train["sha256"],
        "source_train_rows": train["row_count"],
        "selection": {
            "routing_rows_available": counts["routing_rows"],
            "routing_rows_kept": counts["routing_kept"],
            "acute_client_rows": counts["acute_client_rows"],
            "sampled_rows": counts["sample"],
            "tool_rows_excluded": counts["tool_rows_excluded"],
            "too_long_excluded": counts["too_long_excluded"],
            "seed": (options or SelectionOptions()).seed,
        },
        "rows": len(selections),
        "datums": len(datums),
        "datums_skipped_no_prompt": skipped,
        "chars": chars,
        "estimated_tokens": round(chars / CHAR_PER_TOKEN),
        "meets_adaption_floor": len(datums) >= DATUM_FLOOR,
        "format": "adaption_prompt_completion_unrolled_v1",
    }
    if dry_run:
        return summary

    out_dir.mkdir(parents=True, exist_ok=True)
    dataset_path = out_dir / "adaption-sft-train.jsonl"
    tmp = dataset_path.with_name(dataset_path.name + ".tmp")
    digest = hashlib.sha256()
    with tmp.open("w", encoding="utf-8") as handle:
        for datum in datums:
            line = json.dumps(datum, ensure_ascii=False, separators=(",", ":")) + "\n"
            digest.update(line.encode())
            handle.write(line)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp, dataset_path)

    summary["dataset_path"] = str(dataset_path)
    summary["dataset_sha256"] = digest.hexdigest()
    summary["dataset_bytes"] = dataset_path.stat().st_size
    summary["source_manifest_sha256"] = _sha256(manifest_path)
    identity_path = out_dir / "adaption-dataset.json"
    identity_path.write_text(json.dumps(summary, indent=1) + "\n", encoding="utf-8")
    return summary


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export the Adaption AI SFT upload for a release.")
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    summary = export(args.manifest, args.out_dir, dry_run=args.dry_run)
    print(json.dumps(summary, indent=1))
    if not summary["meets_adaption_floor"]:
        print(
            f"WARNING: {summary['datums']} datums is below the {DATUM_FLOOR} Adaption floor "
            "for Qwen3.6-35B-A3B"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
