"""Verify that every crisis-family row of a release survives into its tokenized export.

The exporter keeps the crisis family whole at selection time
(``SelectionOptions.keep_replies``), but a datum can still vanish between the
train split and the file the trainer consumes: it may exceed the token budget
after left-truncation, tokenize to fewer than two tokens, or carry no
supervised weight because no assistant turn survived the trim. The audit in
``training.crisis_target_audit`` covers the flagged candidate rows; this
module covers the whole family, so a release is only accepted when every
crisis row the builder appended is present in the export with a nonzero
weight.

Run:

    PYTHONPATH=. python -m training.verify_crisis_survival \\
        --manifest docs/training/execution-2026-11-10/<release>.json \\
        --train /path/to/train.jsonl \\
        --export /path/to/tinker-sft-train.jsonl \\
        --receipt /path/to/tinker-dataset.json \\
        --out /path/to/report.json

Exit code is 0 only when every crisis row tokenizes into a datum that is
present in the export and carries supervised weight. The report never
contains row contents, only counts, hashes and line numbers.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

from transformers import AutoTokenizer

from training.export_tinker_release import Datum, TokenLimits, crisis_reply_surface, trim_to_budget

SCHEMA = "crisis-family-survival-v1"

MIN_DATUM_TOKENS = 2


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _sha256_lines(path: Path) -> Counter[str]:
    counts: Counter[str] = Counter()
    with path.open("rb") as handle:
        for raw in handle:
            if raw.strip():
                counts[hashlib.sha256(raw.rstrip(b"\n")).hexdigest()] += 1
    return counts


def crisis_row_numbers(train_path: Path, surface: frozenset[str]) -> list[int]:
    """Return 1-based line numbers of rows whose assistant content is in the surface."""
    matches: list[int] = []
    with train_path.open("rb") as handle:
        for line_number, raw in enumerate(handle, start=1):
            if not raw.strip():
                continue
            row = json.loads(raw)
            contents = [
                message.get("content") for message in row.get("messages", []) if message.get("role") == "assistant"
            ]
            if any(content in surface for content in contents if content):
                matches.append(line_number)
    return matches


def default_tokenize(
    tokenizer: Any,
    row: Mapping[str, Any],
    limits: TokenLimits,
    *,
    end_token: str,
    template_kwargs: Mapping[str, Any],
) -> Datum | None:
    """Mirror ``tokenize_selections`` for a single row; None means dropped."""
    messages = [dict(message) for message in row["messages"]]
    if messages and messages[0].get("role") == "assistant":
        messages = messages[1:]
        if not messages:
            return None
    messages, ids, weights, trimmed = trim_to_budget(
        tokenizer,
        messages,
        limits.max_tokens,
        end_token=end_token,
        template_kwargs=template_kwargs,
    )
    if len(ids) < MIN_DATUM_TOKENS:
        return None
    if len(ids) > limits.budget(trimmed=trimmed, message_count=len(messages)):
        return None
    targets = weights[1:]
    if not any(targets):
        return None
    return Datum(input_ids=ids[:-1], target_ids=ids[1:], weights=targets)


def audit(
    *,
    train_path: Path,
    export_path: Path,
    receipt_path: Path,
    manifest_path: Path,
    tokenize: Callable[[Mapping[str, Any]], Datum | None] | None = None,
) -> dict[str, Any]:
    """Check that every crisis-family row lands in the export with supervised weight."""
    receipt = json.loads(receipt_path.read_bytes())
    export_sha = _sha256_file(export_path)
    if receipt.get("sha256") != export_sha:
        raise ValueError(f"export sha256 {export_sha} does not match receipt {receipt.get('sha256')}: {export_path}")
    manifest = json.loads(manifest_path.read_bytes())
    surface = crisis_reply_surface(manifest)
    manifest_sha = _sha256_file(manifest_path)
    max_tokens = int(receipt.get("max_tokens", 512))
    # Receipts written before the final-exchange budget existed fall back to it.
    final_exchange_tokens = int(receipt.get("max_final_exchange_tokens", max_tokens))
    end_token = str(receipt.get("end_token", "<|im_end|>"))
    template_kwargs = dict(receipt.get("template_kwargs") or {})

    crisis_lines = crisis_row_numbers(train_path, surface)
    train_sha = _sha256_file(train_path)
    bound_train = receipt.get("source_train_sha256")
    if bound_train and bound_train != train_sha:
        raise ValueError(f"train sha256 {train_sha} does not match receipt source_train_sha256 {bound_train}")

    tokenize_row: Callable[[Mapping[str, Any]], Datum | None]
    if tokenize is not None:
        tokenize_row = tokenize
    else:
        tokenizer = AutoTokenizer.from_pretrained(
            receipt.get("tokenizer") or receipt.get("base_model"),
            fast=True,
            local_files_only=True,
        )

        def tokenize_row(row: Mapping[str, Any]) -> Datum | None:
            return default_tokenize(
                tokenizer,
                row,
                TokenLimits(max_tokens=max_tokens, final_exchange_tokens=final_exchange_tokens),
                end_token=end_token,
                template_kwargs=template_kwargs,
            )

    export_hashes = _sha256_lines(export_path)

    wanted = set(crisis_lines)
    expected: Counter[str] = Counter()
    supervised_tokens = 0
    zero_weight: list[int] = []
    dropped: list[int] = []
    with train_path.open("rb") as handle:
        for line_number, raw in enumerate(handle, start=1):
            if line_number not in wanted:
                continue
            row = json.loads(raw)
            datum = tokenize_row(row)
            if datum is None:
                dropped.append(line_number)
                continue
            if not any(datum.weights):
                zero_weight.append(line_number)
                continue
            expected[hashlib.sha256(datum.line().encode()).hexdigest()] += 1
            supervised_tokens += int(sum(datum.weights))

    missing: list[str] = []
    for digest, needed in expected.items():
        available = export_hashes.get(digest, 0)
        if available < needed:
            missing.extend([digest[:12]] * (needed - available))

    result = "pass" if not (dropped or zero_weight or missing) else "fail"
    return {
        "schema": SCHEMA,
        "source_train": {"path": str(train_path), "sha256": train_sha},
        "export": {
            "path": str(export_path),
            "sha256": export_sha,
            "datums": receipt.get("datums"),
        },
        "manifest": {
            "path": str(manifest_path),
            "sha256": manifest_sha,
            "release_id": manifest.get("release_id"),
        },
        "crisis_surface_replies": len(surface),
        "crisis_rows_in_train": len(crisis_lines),
        "crisis_datums_expected": sum(expected.values()),
        "crisis_datums_present": sum(expected.values()) - len(missing),
        "crisis_supervised_tokens": supervised_tokens,
        "dropped_rows": dropped,
        "zero_weight_rows": zero_weight,
        "missing_datum_hashes": missing,
        "result": result,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--export", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args(argv)

    report = audit(
        train_path=args.train,
        export_path=args.export,
        receipt_path=args.receipt,
        manifest_path=args.manifest,
    )
    rendered = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(rendered, encoding="utf-8")
    sys.stdout.write(rendered)
    return 0 if report["result"] == "pass" else 1


if __name__ == "__main__":
    sys.exit(main())
