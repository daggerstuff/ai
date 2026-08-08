#!/usr/bin/env python3
"""V7 dataset consolidation with advanced hash deduping.

Combines all data sources into a unified V7 MASTER dataset with:
- Exact dedup via SHA-256 content hash (normalized)
- Near-dedup via MinHash/LSH for O(n) scalable semantic dedup
- Stage-aware conflict resolution (higher-priority stages win on collision)
- Edge-case preservation (P0-1 records bypass near-dedup)
- ChatML normalization to V7 schema with provenance enrichment
- Sharded JSONL output with manifest and stats report

V7 schema fields: messages, source, task_type, diagnostic_tag,
demographic_tags, linguistic_style, clinical_reviewed, provenance.

Usage:
    python -m dataset_pipeline.orchestration.consolidate_v7 \
        --input_dirs ai/data/prepared/pal_inputs ai/data/prepared/safety \
        --output_dir ai/data/prepared/v7_master \
        --jaccard_threshold 0.85 --use_lsh
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import re
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

from dataset_pipeline.processors.minhash_lsh import SemanticDeduplicator

logger = logging.getLogger("consolidate_v7")

# ---------------------------------------------------------------------------
# Stage priority for conflict resolution
# ---------------------------------------------------------------------------
STAGE_PRIORITY: dict[str, int] = {
    "stage4_voice_persona": 5,
    "stage3_edge_stress_test": 4,
    "stage2_therapeutic_expertise": 3,
    "stage1_foundation": 2,
    "supplementary": 1,
}

DEFAULT_SYSTEM_PROMPT = (
    "You are Pixel, a highly empathetic and clinically precise AI therapist. "
    "Respond with warmth, validation, and evidence-based guidance."
)

_VALID_LINGUISTIC_STYLES = {"formal", "informal", "mixed"}
_VALID_TASK_TYPES = {
    "symptom_classification",
    "severity_estimation",
    "therapy_response_generation",
    "risk_assessment",
    "empathy_scoring",
    "dpo_preference",
    "adversarial_safety",
    "voice_training",
    "reasoning_enhancement",
    "personality_balancing",
    "psychology_knowledge",
    "mental_health_conversations",
}

_INST_BOUNDARY = re.compile(r"\[/INST\]")


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------
@dataclass
class ConsolidationStats:
    """Statistics from V7 consolidation run."""

    total_read: int = 0
    total_kept: int = 0
    exact_duplicates: int = 0
    near_duplicates: int = 0
    stage_conflicts_resolved: int = 0
    edge_cases_preserved: int = 0
    reformatted: int = 0
    chatml_failures: int = 0
    lsh_candidates: int = 0
    lsh_comparisons: int = 0
    records_by_source: dict[str, int] = field(default_factory=dict)
    records_by_task_type: dict[str, int] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Hashing
# ---------------------------------------------------------------------------
def compute_primary_hash(record: dict) -> str:
    """SHA-256 of normalized message content (role + content, lowercased)."""
    messages = record.get("messages", [])
    if not messages:
        return hashlib.sha256(b"").hexdigest()
    parts: list[str] = []
    for msg in messages:
        if isinstance(msg, dict):
            parts.append(str(msg.get("role", "")))
            parts.append(_coerce_content(msg.get("content", "")))
    return hashlib.sha256("".join(parts).lower().encode("utf-8")).hexdigest()


def compute_token_set(text: str) -> frozenset[str]:
    """Tokenize text for near-duplicate comparison."""
    return frozenset(text.lower().split())


def jaccard_similarity(a: frozenset[str], b: frozenset[str]) -> float:
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


# ---------------------------------------------------------------------------
# Record inspection helpers
# ---------------------------------------------------------------------------
def _is_edge_case(record: dict) -> bool:
    return record.get("is_training_edge_case", False) is True


def _get_stage(record: dict) -> str:
    metadata = record.get("metadata", {})
    if isinstance(metadata, dict):
        return metadata.get("stage", "supplementary")
    return "supplementary"


def _get_stage_priority(record: dict) -> int:
    return STAGE_PRIORITY.get(_get_stage(record), 1)


def _get_source(record: dict) -> str:
    if record.get("source"):
        return record["source"]
    metadata = record.get("metadata", {})
    if isinstance(metadata, dict):
        return metadata.get("source", metadata.get("source_channel", "unknown"))
    return "unknown"


def _get_task_type(record: dict) -> str:
    if record.get("task_type"):
        return record["task_type"]
    metadata = record.get("metadata", {})
    if isinstance(metadata, dict):
        return metadata.get("task_type", "therapy_response_generation")
    return "therapy_response_generation"


def _coerce_content(content) -> str:
    """Coerce message content to a string, handling lists and other types."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict) and isinstance(item.get("text"), str):
                parts.append(item["text"])
            elif isinstance(item, dict) and isinstance(item.get("content"), str):
                parts.append(item["content"])
        return " ".join(parts)
    return str(content) if content else ""


def _extract_text(record: dict) -> str:
    messages = record.get("messages", [])
    if messages:
        return " ".join(
            _coerce_content(m.get("content", "")) for m in messages if isinstance(m, dict)
        )
    if record.get("prompt") and record.get("chosen") and record.get("rejected"):
        return f"{record['prompt']} {record['chosen']} {record['rejected']}"
    return f"{record.get('instruction', '')} {record.get('output', '')}"


def _verify_chatml(record: dict) -> bool:
    """Check that record has valid ChatML message structure."""
    messages = record.get("messages", [])
    if not messages:
        return False
    for msg in messages:
        if not isinstance(msg, dict):
            return False
        if msg.get("role") not in ("system", "user", "assistant"):
            return False
        content = msg.get("content")
        if not content or not _coerce_content(content).strip():
            return False
    text = _extract_text(record)
    return "[/INST]" not in text or bool(_INST_BOUNDARY.search(text))


# ---------------------------------------------------------------------------
# V7 normalization
# ---------------------------------------------------------------------------
def normalize_to_v7(record: dict, input_source: str) -> tuple[dict | None, bool]:
    """Normalize a record to V7 ChatML schema.

    Returns (normalized_dict | None, was_reformatted).
    """
    # Already has messages — validate and enrich
    if "messages" in record and _verify_chatml(record):
        return _enrich_v7_fields(record, input_source), False

    # Try to reformat instruction/output or DPO pairs
    if record.get("instruction") and record.get("output"):
        record = {
            "messages": [
                {"role": "user", "content": record["instruction"]},
                {"role": "assistant", "content": record["output"]},
            ],
            **{k: v for k, v in record.items() if k not in ("instruction", "output")},
        }
        return _enrich_v7_fields(record, input_source), True

    if record.get("prompt") and record.get("chosen"):
        messages = [{"role": "user", "content": record["prompt"]}]
        if record.get("rejected"):
            messages.append({"role": "assistant", "content": record["chosen"]})
        else:
            messages.append({"role": "assistant", "content": record["chosen"]})
        record = {
            "messages": messages,
            **{k: v for k, v in record.items() if k not in ("prompt", "chosen", "rejected")},
        }
        return _enrich_v7_fields(record, input_source), True

    return None, False


def _enrich_v7_fields(record: dict, input_source: str) -> dict:
    """Ensure all V7 schema fields are present with sensible defaults."""
    # Ensure system message exists
    messages = record.get("messages", [])
    if messages and messages[0].get("role") != "system":
        messages.insert(0, {"role": "system", "content": DEFAULT_SYSTEM_PROMPT})
        record["messages"] = messages

    # Set V7 fields with defaults
    record.setdefault("source", _get_source(record))
    record.setdefault("task_type", _get_task_type(record))
    record.setdefault("diagnostic_tag", None)
    record.setdefault("demographic_tags", [])
    record.setdefault("linguistic_style", "mixed")
    record.setdefault("clinical_reviewed", False)

    # Enforce valid enum values
    if record["linguistic_style"] not in _VALID_LINGUISTIC_STYLES:
        record["linguistic_style"] = "mixed"
    if record["task_type"] not in _VALID_TASK_TYPES:
        record["task_type"] = "therapy_response_generation"

    # Enrich provenance
    prov = record.get("provenance", {})
    if not isinstance(prov, dict):
        prov = {}
    prov.setdefault("source_url", str(input_source))
    prov.setdefault("access_method", "local")
    prov.setdefault("original_format", "jsonl")
    transformations = prov.get("transformations", [])
    if isinstance(transformations, str):
        transformations = [transformations]
    transformations = list(transformations)
    if "v7_normalize" not in transformations:
        transformations.append("v7_normalize")
    prov["transformations"] = transformations
    prov["consolidated_at"] = datetime.now(UTC).isoformat()
    record["provenance"] = prov

    return record


# ---------------------------------------------------------------------------
# Advanced dedup engine
# ---------------------------------------------------------------------------
class V7Deduplicator:
    """Multi-strategy deduplication engine.

    1. Exact dedup via SHA-256 content hash
    2. Stage-aware conflict resolution (higher stage wins on collision)
    3. Near-dedup via Jaccard token similarity (windowed)
    4. Edge-case bypass (P0-1 records skip near-dedup only)
    """

    def __init__(
        self,
        jaccard_threshold: float = 0.85,
        near_dedup_window: int = 5000,
        use_lsh: bool = False,
    ) -> None:
        self.jaccard_threshold = jaccard_threshold
        self.near_dedup_window = near_dedup_window
        self.use_lsh = use_lsh
        # hash -> (record, stage_priority)
        self._hash_map: dict[str, tuple[dict, int]] = {}
        # (token_set, hash) for windowed near-dup comparison
        self._token_sets: list[tuple[frozenset[str], str]] = []
        # LSH-based semantic deduplication
        self._lsh_dedup: SemanticDeduplicator | None = None
        if use_lsh:
            self._lsh_dedup = SemanticDeduplicator(
                jaccard_threshold=jaccard_threshold,
            )
        # Record counter for LSH record IDs
        self._record_counter = 0
        # Edge-case hashes tracked separately
        self._edge_case_hashes: set[str] = set()
        self.stats = ConsolidationStats()

    def process(self, record: dict) -> bool:
        """Process a single record. Returns True if kept, False if dropped."""
        self.stats.total_read += 1

        text = _extract_text(record)
        content_hash = compute_primary_hash(record)
        source = _get_source(record)
        task_type = _get_task_type(record)
        stage_priority = _get_stage_priority(record)
        is_edge = _is_edge_case(record)

        self.stats.records_by_source[source] = self.stats.records_by_source.get(source, 0) + 1
        self.stats.records_by_task_type[task_type] = self.stats.records_by_task_type.get(task_type, 0) + 1

        kept, drop_reason = self._evaluate(record, text, content_hash, stage_priority, is_edge)
        if drop_reason == "replaced":
            pass  # Stage conflict resolved, record replaced — not a new addition
        elif kept:
            self.stats.total_kept += 1
        elif drop_reason == "exact":
            self.stats.exact_duplicates += 1
        elif drop_reason == "near":
            self.stats.near_duplicates += 1
        return kept

    def _evaluate(
        self,
        record: dict,
        text: str,
        content_hash: str,
        stage_priority: int,
        is_edge: bool,
    ) -> tuple[bool, str | None]:
        """Decide whether to keep or drop a record. Returns (kept, drop_reason)."""
        if is_edge:
            self._edge_case_hashes.add(content_hash)
            if content_hash in self._hash_map:
                existing_stage = self._hash_map[content_hash][1]
                if stage_priority > existing_stage:
                    self._hash_map[content_hash] = (record, stage_priority)
                    self.stats.stage_conflicts_resolved += 1
                    self.stats.edge_cases_preserved += 1
                    return True, "replaced"
                return False, "exact"
            self._hash_map[content_hash] = (record, stage_priority)
            self.stats.edge_cases_preserved += 1
            return True, None

        if content_hash in self._hash_map:
            existing_stage = self._hash_map[content_hash][1]
            if stage_priority > existing_stage:
                self._hash_map[content_hash] = (record, stage_priority)
                self.stats.stage_conflicts_resolved += 1
                return True, "replaced"
            return False, "exact"

        if content_hash in self._edge_case_hashes:
            return False, "exact"

        # Near-dedup: LSH (scalable) or Jaccard (windowed)
        # Skip near-dup entirely when window is 0 and not using LSH
        if not self.use_lsh and self.near_dedup_window == 0:
            self._hash_map[content_hash] = (record, stage_priority)
            return True, None

        tokens = compute_token_set(text)

        if self.use_lsh and self._lsh_dedup is not None:
            record_id = f"rec_{self._record_counter}"
            self._record_counter += 1
            result = self._lsh_dedup.add(record_id, tokens)
            self.stats.lsh_candidates += result.candidates_checked
            self.stats.lsh_comparisons += result.candidates_checked
            if result.is_duplicate:
                return False, "near"
        elif self.near_dedup_window > 0:
            window = self._token_sets[-self.near_dedup_window:]
            for seen_tokens, seen_hash in window:
                if seen_hash == content_hash:
                    continue
                if jaccard_similarity(tokens, seen_tokens) >= self.jaccard_threshold:
                    return False, "near"

        # Passed all checks — keep this record
        self._hash_map[content_hash] = (record, stage_priority)
        self._token_sets.append((tokens, content_hash))
        return True, None

    @property
    def kept_records(self) -> list[dict]:
        return [rec for rec, _ in self._hash_map.values()]


# ---------------------------------------------------------------------------
# File I/O
# ---------------------------------------------------------------------------
def _load_jsonl(path: Path) -> list[dict]:
    """Load records from a JSONL file."""
    records: list[dict] = []
    try:
        with path.open(encoding="utf-8", errors="replace") as f:
            for line_no, raw_line in enumerate(f, 1):
                stripped = raw_line.strip()
                if not stripped:
                    continue
                try:
                    records.append(json.loads(stripped))
                except json.JSONDecodeError:
                    logger.warning("Invalid JSON in %s line %d", path, line_no)
    except OSError as exc:
        logger.warning("Cannot read %s: %s", path, exc)
    return records


def _write_jsonl(path: Path, records: list[dict]) -> None:
    """Write records to a JSONL file."""
    with path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def _write_shards(
    records: list[dict],
    output_dir: Path,
    shard_size: int,
) -> int:
    """Write records into sharded JSONL files. Returns shard count."""
    shard_count = 0
    for i in range(0, len(records), shard_size):
        shard = records[i : i + shard_size]
        shard_path = output_dir / f"shard_{shard_count:04d}.jsonl"
        _write_jsonl(shard_path, shard)
        shard_count += 1
    return shard_count


# ---------------------------------------------------------------------------
# Main consolidation pipeline
# ---------------------------------------------------------------------------
def run_consolidation(args: argparse.Namespace) -> None:
    """Run the V7 consolidation pipeline."""
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    run_id = f"v7-consolidate-{datetime.now(UTC).strftime('%Y%m%d-%H%M%S')}"
    logger.info("Starting V7 consolidation run: %s", run_id)

    dedup = V7Deduplicator(
        jaccard_threshold=args.jaccard_threshold,
        near_dedup_window=args.near_dedup_window,
        use_lsh=args.use_lsh,
    )

    # Process each input source
    for input_source in args.input_dirs:
        source_path = Path(input_source)
        if source_path.is_file() and source_path.suffix == ".jsonl":
            files = [source_path]
        elif source_path.is_dir():
            files = sorted(source_path.rglob("*.jsonl"))
            # Exclude report files
            files = [f for f in files if not f.name.endswith(("report.jsonl", "rejection_log.jsonl", "stats.json"))]
        else:
            logger.warning("Source not found: %s", source_path)
            continue

        for jsonl_file in files:
            raw_records = _load_jsonl(jsonl_file)
            logger.info("Loaded %d records from %s", len(raw_records), jsonl_file)

            for raw in raw_records:
                normalized, was_reformatted = normalize_to_v7(raw, str(jsonl_file))
                if normalized is None:
                    dedup.stats.chatml_failures += 1
                    continue
                if was_reformatted:
                    dedup.stats.reformatted += 1
                dedup.process(normalized)

    kept = dedup.kept_records
    stats = dedup.stats

    # Write output
    if args.shard_size > 0:
        shard_count = _write_shards(kept, output_dir, args.shard_size)
    else:
        master_path = output_dir / "MASTER_V7.jsonl"
        _write_jsonl(master_path, kept)
        shard_count = 1

    # Write manifest
    manifest = {
        "run_id": run_id,
        "generated_at": datetime.now(UTC).isoformat(),
        "output_dir": str(output_dir),
        "schema_version": "v7",
        "total_records": stats.total_kept,
        "shard_count": shard_count,
        "jaccard_threshold": args.jaccard_threshold,
        "near_dedup_window": args.near_dedup_window,
    }
    manifest_path = output_dir / "manifest.json"
    with manifest_path.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
        f.write("\n")

    # Write stats
    stats_report = {
        "run_id": run_id,
        "generated_at": datetime.now(UTC).isoformat(),
        "total_read": stats.total_read,
        "total_kept": stats.total_kept,
        "exact_duplicates": stats.exact_duplicates,
        "near_duplicates": stats.near_duplicates,
        "stage_conflicts_resolved": stats.stage_conflicts_resolved,
        "edge_cases_preserved": stats.edge_cases_preserved,
        "reformatted": stats.reformatted,
        "chatml_failures": stats.chatml_failures,
        "lsh_candidates": stats.lsh_candidates,
        "lsh_comparisons": stats.lsh_comparisons,
        "records_by_source": dict(sorted(stats.records_by_source.items())),
        "records_by_task_type": dict(sorted(stats.records_by_task_type.items())),
    }
    stats_path = output_dir / "stats.json"
    with stats_path.open("w", encoding="utf-8") as f:
        json.dump(stats_report, f, indent=2)
        f.write("\n")

    logger.info(
        "V7 consolidation complete: %d in -> %d out "
        "(%d exact dup, %d near dup, %d stage conflicts, %d edge preserved, %d reformatted, %d ChatML fail)",
        stats.total_read,
        stats.total_kept,
        stats.exact_duplicates,
        stats.near_duplicates,
        stats.stage_conflicts_resolved,
        stats.edge_cases_preserved,
        stats.reformatted,
        stats.chatml_failures,
    )
    logger.info("Manifest: %s", manifest_path)
    logger.info("Stats: %s", stats_path)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Consolidate and deduplicate training data into V7 MASTER format.",
    )
    parser.add_argument(
        "--input_dirs",
        nargs="+",
        required=True,
        help="Source directories or files containing JSONL records.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Output directory for V7 MASTER dataset.",
    )
    parser.add_argument(
        "--jaccard_threshold",
        type=float,
        default=0.85,
        help="Jaccard similarity threshold for near-dedup (default: 0.85).",
    )
    parser.add_argument(
        "--near_dedup_window",
        type=int,
        default=5000,
        help="Max prior token sets to compare for near-dedup (default: 5000).",
    )
    parser.add_argument(
        "--use_lsh",
        action="store_true",
        default=False,
        help="Use MinHash/LSH for scalable near-dedup (10k+ records).",
    )
    parser.add_argument(
        "--shard_size",
        type=int,
        default=50000,
        help="Records per output shard (0 = single file, default: 50000).",
    )
    return parser


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    parser = build_parser()
    args = parser.parse_args()
    run_consolidation(args)


if __name__ == "__main__":
    main()
