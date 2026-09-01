#!/usr/bin/env python3
"""Build clinically-backed SFT examples (Stages 1-2) from local T1_GOLD + streamed corpora.

Hybrid sourcing (Decision 3):
  (a) local ``books_distilled`` T1_GOLD records (46 clinical textbooks) and
  (b) streamed benchmark corpora — ``clinical_redteam`` (DSM-5 personas),
      ``crisis_benchmark`` (JMIR 2026), ``safety_dpo_pairs_10k``.

All remote reads go through ``rclone cat`` (never bulk-downloaded). Records are
normalized to ChatML, gated against sycophantic/cliché assistant turns (step 5),
tagged ``stage1_foundation`` / ``stage2_therapeutic_expertise``, deduped against
the master gold (SHA-256 primary + SHA-1 secondary), and emitted to a staging
JSONL that step 9 consumes for atomic consolidation.

``safety_dpo_pairs_10k``: the ``chosen`` response becomes an SFT example here
(Stage 1); the DPO ``rejected`` pairs themselves are deferred to step 9 (Stage 5).
"""

from __future__ import annotations

import argparse
import json
import logging
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from training.cliche_gate import reject_reason_for_record
from training.stream_staged_edge_assets import (
    _extract_messages,
    _iter_jsonl,
    _looks_placeholder,
    content_hashes,
    load_master_hashes,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("build_clinical_backed_examples")

_AI_ROOT = Path(__file__).resolve().parents[1]
BOOKS_DIR = _AI_ROOT / "data" / "curated" / "books_distilled"
MASTER_GOLD = _AI_ROOT / "data" / "curated" / "sft_chatml" / "train_master_gold.jsonl"
DEFAULT_OUT = _AI_ROOT / "training" / "output" / "nightmare_fuel" / "clinical_backed_staging.jsonl"
DEFAULT_REJECT = _AI_ROOT / "training" / "output" / "nightmare_fuel" / "clinical_backed_rejections.jsonl"

STAGE1 = "stage1_foundation"
STAGE2 = "stage2_therapeutic_expertise"

SYSTEM_PROMPT = (
    "You are an expert clinical psychologist and therapeutic AI assistant. "
    "Provide grounded, authentic, non-sycophantic therapeutic dialogue."
)

# Remote benchmark corpora streamed via rclone cat (S3Streamer).
BENCHMARK_ASSETS: tuple[dict[str, str], ...] = (
    {"name": "clinical_redteam", "remote": "whitebat", "bucket": "whitebat",
     "key": "training/ai-data/raw/clinical_redteam/clinical_redteam.jsonl"},
    {"name": "crisis_benchmark", "remote": "whitebat", "bucket": "whitebat",
     "key": "training/raw/crisis_benchmark/crisis_benchmark.jsonl"},
    {"name": "safety_dpo_pairs_10k", "remote": "gdrive", "bucket": "pixeldata",
     "key": "training/v1/stage3_stress_test/processed/safety_dpo_pairs_10k.jsonl"},
)

# Book-title keywords marking deep clinical/therapeutic content (-> Stage 2).
CLINICAL_BOOK_KEYWORDS: tuple[str, ...] = (
    "cptsd", "ptsd", "trauma", "internal family systems", "dialectical",
    "dbt", "cbt", "dsm", "treating", "therapy", "clinical", "brain energy",
    "psycho logical", "high conflict couple", "addiction",
)

# Stage routing for benchmark sources. safety_dpo's ``chosen`` response is SFT
# grounding (Stage 1); its rejected DPO pairs are handled in step 9 (Stage 5).
BENCHMARK_STAGE: dict[str, str] = {
    "clinical_redteam": STAGE2,
    "crisis_benchmark": STAGE2,
    "safety_dpo_pairs_10k": STAGE1,
}


@dataclass
class _EmitState:
    """Mutable per-run state threaded through the emit path (keeps arity low)."""

    seen: set[str]
    out_path: Path
    reject_path: Path
    summary: dict[str, int] = field(default_factory=dict)


def _book_stage(book_title: str) -> str:
    """Route a clinical book to Stage 2 (therapeutic) or Stage 1 (foundation)."""
    title = book_title.strip().lower()
    return STAGE2 if any(kw in title for kw in CLINICAL_BOOK_KEYWORDS) else STAGE1


def _is_dpo(raw: dict) -> bool:
    return "chosen" in raw and "rejected" in raw and "prompt" in raw


def normalize_book_record(item: dict, path: Path) -> dict | None:
    """Convert a ``{instruction, output, metadata}`` distillation row to ChatML."""
    instruction = str(item.get("instruction") or "").strip()
    output = str(item.get("output") or "").strip()
    if not instruction or not output:
        return None
    metadata = item.get("metadata") or {}
    book_name = str(metadata.get("source_book") or path.stem)
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": instruction},
        {"role": "assistant", "content": output},
    ]
    sha256, sha1 = content_hashes(messages)
    return {
        "messages": messages,
        "source": "clinical_book",
        "task_type": "clinical_literature_distillation",
        "tier": "T1_GOLD",
        "diagnostic_tag": "clinical_modality",
        "stage": _book_stage(book_name),
        "demographic_tags": [],
        "linguistic_style": "spoken_clinical_practitioner",
        "clinical_reviewed": True,
        "provenance": {
            "source_book": book_name,
            "source_type": "clinical_literature",
            "distillation_version": metadata.get("distillation_version"),
        },
        "sha256": sha256,
        "sha1": sha1,
    }


def normalize_benchmark_record(source: str, raw: dict) -> dict | None:
    """Normalize a streamed benchmark record into a stage-tagged ChatML record.

    DPO pairs (``{prompt, chosen, rejected}``) become SFT examples from
    ``prompt -> chosen``; already-ChatML records are preserved with their rich
    metadata and annotated with stage + content hashes.
    """
    stage = BENCHMARK_STAGE.get(source, STAGE1)

    if _is_dpo(raw):
        prompt = str(raw.get("prompt") or "").strip()
        chosen = str(raw.get("chosen") or "").strip()
        if not prompt or not chosen:
            return None
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": chosen},
        ]
        metadata = raw.get("metadata") or {}
        sha256, sha1 = content_hashes(messages)
        return {
            "messages": messages,
            "source": source,
            "task_type": "therapeutic_sft",
            "tier": "benchmark",
            "diagnostic_tag": str(metadata.get("domain") or "therapeutic"),
            "stage": stage,
            "provenance": {"source_asset": source, "pair_type": metadata.get("pair_type")},
            "sha256": sha256,
            "sha1": sha1,
        }

    messages = _extract_messages(raw)
    if not messages:
        return None
    text = "\n".join(f"{m['role']}: {m['content']}" for m in messages).strip().lower()
    if _looks_placeholder(text):
        return None
    sha256, sha1 = content_hashes(messages)
    record: dict[str, Any] = dict(raw)
    record["messages"] = messages
    record["stage"] = stage
    record["tier"] = "benchmark"
    record["sha256"] = sha256
    record["sha1"] = sha1
    return record


def _emit(path: Path, record: dict) -> None:
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def _emit_rejection(path: Path, reason: str, record: dict) -> None:
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(
            {"reason": reason, "sha256": record.get("sha256"),
             "source": record.get("source"), "stage": record.get("stage")},
            ensure_ascii=False,
        ) + "\n")


def _load_existing_hashes(path: Path) -> set[str]:
    """Recover hashes already staged so re-runs stay idempotent."""
    hashes: set[str] = set()
    if not path.exists():
        return hashes
    with path.open() as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line:
                continue
            try:
                raw = json.loads(line)
            except json.JSONDecodeError:
                continue
            h = raw.get("sha256")
            if h:
                hashes.add(h)
    return hashes


def _process(record: dict, family: str, state: _EmitState) -> None:
    """Dedup, cliché-gate, and emit one normalized record."""
    sha = record.get("sha256")
    if sha in state.seen:
        state.summary["duplicates"] += 1
        return
    reason = reject_reason_for_record(record, family=family)
    if reason is not None:
        state.summary["rejected"] += 1
        _emit_rejection(state.reject_path, reason, record)
        return
    state.seen.add(sha)
    _emit(state.out_path, record)
    state.summary["emitted"] += 1
    if record.get("stage") == STAGE2:
        state.summary["stage2"] += 1
    else:
        state.summary["stage1"] += 1


def _iter_book_rows(books_dir: Path, limit: int | None) -> Iterator[tuple[dict, Path]]:
    seen = 0
    for path in sorted(books_dir.glob("*.jsonl")):
        with path.open(encoding="utf-8") as f:
            for raw_line in f:
                line = raw_line.strip()
                if not line:
                    continue
                try:
                    item = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if not isinstance(item, dict):
                    continue
                yield item, path
                seen += 1
                if limit is not None and seen >= limit:
                    return


def build(
    *,
    books_limit: int | None = None,
    benchmark_limit: int | None = None,
    max_master: int | None = None,
    out_path: Path = DEFAULT_OUT,
    reject_path: Path = DEFAULT_REJECT,
) -> dict[str, int]:
    """Build the staging file; return a per-source/per-outcome summary."""
    summary = {"books": 0, "benchmark": 0, "emitted": 0, "rejected": 0,
               "duplicates": 0, "placeholders": 0, "stage1": 0, "stage2": 0}
    out_path.parent.mkdir(parents=True, exist_ok=True)
    reject_path.parent.mkdir(parents=True, exist_ok=True)

    master_hashes = load_master_hashes(MASTER_GOLD, max_master)
    state = _EmitState(
        seen=set(master_hashes) | _load_existing_hashes(out_path),
        out_path=out_path,
        reject_path=reject_path,
        summary=summary,
    )
    logger.info("Loaded %d master hashes + %d staged hashes", len(master_hashes), len(state.seen) - len(master_hashes))

    # (a) Local 46-textbook T1_GOLD records.
    for item, path in _iter_book_rows(BOOKS_DIR, books_limit):
        summary["books"] += 1
        record = normalize_book_record(item, path)
        if record is None:
            summary["placeholders"] += 1
            continue
        _process(record, "clinical_book", state)
    logger.info("books: scanned=%d emitted=%d", summary["books"], summary["emitted"])

    # (b) Streamed benchmark corpora.
    for asset in BENCHMARK_ASSETS:
        name = asset["name"]
        before = summary["emitted"]
        for raw in _iter_jsonl(asset, benchmark_limit):
            summary["benchmark"] += 1
            record = normalize_benchmark_record(name, raw)
            if record is None:
                summary["placeholders"] += 1
                continue
            _process(record, name, state)
        logger.info("%s: emitted=%d", name, summary["emitted"] - before)

    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--reject", type=Path, default=DEFAULT_REJECT)
    parser.add_argument("--books-limit", type=int, default=None, help="cap local book rows (smoke test)")
    parser.add_argument(
        "--benchmark-limit",
        type=int,
        default=None,
        help="cap records per benchmark asset (smoke test)",
    )
    parser.add_argument("--max-master", type=int, default=None, help="cap master hash build (smoke test)")
    args = parser.parse_args()

    summary = build(
        books_limit=args.books_limit,
        benchmark_limit=args.benchmark_limit,
        max_master=args.max_master,
        out_path=args.out,
        reject_path=args.reject,
    )
    logger.info("Build summary:\n%s", json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
