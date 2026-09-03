#!/usr/bin/env python3
"""CONVERT/HASH HOLD corpus pipeline for individual dataset snapshots.

Streams a corpus from the rclone remote, validates ChatML format, attaches
source-family provenance from the manifest, computes SHA-256 primary hashes,
deduplicates, and writes the result to a HOLD directory that is NEVER merged
into ``train_all``.

Usage::

    uv run python -m ai.pipelines.corpus_hold.convert_hash_hold \\
        --ds-id DS-006 \\
        --remote whitebat \\
        --bucket training \\
        --remote-prefix pixelated-empathy/ingestion/master_work \\
        --output-dir ai/data/hold/DS-006

The script is corpus-scoped: it processes exactly one DS-* identifier and
writes to an isolated HOLD path.  It must not be combined with DS-007 or any
other corpus.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import yaml

from ai.pipelines.data_processing.extractors.s3_streamer import S3Streamer
from ai.pipelines.ingestion_deduplication import compute_primary_hash, deduplicate_records

logger = logging.getLogger("corpus_hold")

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

_MANIFEST_PATH = Path(__file__).resolve().parents[2] / "data" / "source_family_provenance_manifest.yaml"

# Map DS-* aliases to their SRC-* source_id in the provenance manifest.
_DS_TO_SRC = {
    "DS-006": "SRC-002",
    "DS-007": "SRC-002",
}

# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class CorpusConfig:
    """Connection + scope parameters for a single corpus HOLD run."""

    ds_id: str
    remote: str
    bucket: str
    remote_prefix: str
    output_dir: Path
    max_records: int | None = None
    dry_run: bool = False


@dataclass
class ProvenanceInfo:
    """Provenance metadata extracted from the source-family manifest."""

    source_id: str
    source_title: str
    restrictions: list[str]
    policy: str
    scope: str


@dataclass
class HoldManifest:
    """Summary manifest written alongside the held JSONL output."""

    ds_id: str
    source_id: str
    source_title: str
    remote: str
    remote_prefix: str
    output_path: str
    total_records_streamed: int
    valid_records: int
    unique_records: int
    duplicates_removed: int
    stage_conflicts_resolved: int
    records_by_stage: dict[str, int] = field(default_factory=dict)
    restrictions: list[str] = field(default_factory=list)
    policy: str = ""
    scope: str = ""
    processed_at: str = ""


# ---------------------------------------------------------------------------
# Provenance
# ---------------------------------------------------------------------------


def load_provenance(ds_id: str, manifest_path: Path = _MANIFEST_PATH) -> dict:
    """Load the SRC-* provenance entry for a given DS-* alias."""
    with manifest_path.open(encoding="utf-8") as fh:
        manifest = yaml.safe_load(fh)

    sources = manifest.get("sources", manifest)
    target_src = _DS_TO_SRC.get(ds_id)
    if isinstance(sources, dict):
        for src_id, entry in sources.items():
            aliases = entry.get("ds_alias", [])
            if ds_id in aliases or src_id == target_src:
                return entry
    elif isinstance(sources, list):
        for entry in sources:
            aliases = entry.get("ds_alias", [])
            if ds_id in aliases or entry.get("source_id") == target_src:
                return entry

    logger.warning("No provenance entry found for %s — proceeding with minimal metadata", ds_id)
    return {}


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def validate_chatml(record: dict) -> bool:
    """Return ``True`` if *record* is a valid ChatML conversation."""
    messages = record.get("messages")
    if not isinstance(messages, list) or len(messages) == 0:
        return False
    for msg in messages:
        if not isinstance(msg, dict):
            return False
        if "role" not in msg or "content" not in msg:
            return False
        if not isinstance(msg["role"], str) or not isinstance(msg["content"], str):
            return False
    return True


# ---------------------------------------------------------------------------
# Pipeline helpers
# ---------------------------------------------------------------------------


def _stream_and_validate(
    config: CorpusConfig,
    streamer: S3Streamer,
    prov: ProvenanceInfo,
) -> tuple[list[dict], int, int]:
    """Stream JSONL from remote, validate, attach provenance + hash.

    Returns ``(records, total_streamed, valid_count)``.
    """
    # Ensure trailing slash so S3Streamer.list_files joins prefix + filename correctly.
    prefix = config.remote_prefix.rstrip("/") + "/"
    files = streamer.list_files(prefix=prefix, recursive=False)
    jsonl_files = [f for f in files if f.endswith(".jsonl")]
    if not jsonl_files:
        logger.error("No .jsonl files found under %s://%s/%s", config.remote, config.bucket, config.remote_prefix)
        sys.exit(1)

    logger.info("Found %d JSONL file(s): %s", len(jsonl_files), ", ".join(jsonl_files[:5]))

    records: list[dict] = []
    total_streamed = 0
    valid_count = 0

    for file_key in jsonl_files:
        logger.info("Streaming %s ...", file_key)
        for record in streamer.stream_jsonl(file_key):
            total_streamed += 1
            if config.max_records is not None and total_streamed > config.max_records:
                logger.info("Reached --max-records cap (%d), stopping stream", config.max_records)
                break

            if not validate_chatml(record):
                logger.debug("Skipping invalid ChatML record #%d", total_streamed)
                continue

            if "provenance" not in record:
                record["provenance"] = {
                    "source_id": prov.source_id,
                    "ds_alias": config.ds_id,
                    "source_title": prov.source_title,
                    "policy": prov.policy,
                    "scope": prov.scope,
                    "restrictions": prov.restrictions,
                }

            record.setdefault("metadata", {})
            if not record["metadata"].get("primary_hash"):
                record["metadata"]["primary_hash"] = compute_primary_hash(record)

            records.append(record)
            valid_count += 1

        if config.max_records is not None and total_streamed >= config.max_records:
            break

    logger.info("Streamed %d records, %d valid ChatML", total_streamed, valid_count)
    return records, total_streamed, valid_count


def _write_outputs(
    out_jsonl: Path,
    out_manifest: Path,
    deduped: list[dict],
    manifest: HoldManifest,
) -> None:
    """Write deduped JSONL and manifest JSON to disk."""
    out_jsonl.parent.mkdir(parents=True, exist_ok=True)

    with out_jsonl.open("w", encoding="utf-8") as fh:
        for record in deduped:
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")
    logger.info("Wrote %d deduped records to %s", len(deduped), out_jsonl)

    with out_manifest.open("w", encoding="utf-8") as fh:
        json.dump(asdict(manifest), fh, indent=2, ensure_ascii=False)
    logger.info("Wrote manifest to %s", out_manifest)


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------


def process_corpus(config: CorpusConfig) -> HoldManifest:
    """Stream, validate, hash, deduplicate, and write a single corpus to HOLD."""
    prov_dict = load_provenance(config.ds_id)
    prov = ProvenanceInfo(
        source_id=prov_dict.get("source_id", _DS_TO_SRC.get(config.ds_id, "")),
        source_title=prov_dict.get("title", ""),
        restrictions=prov_dict.get("rights_decision", {}).get("restrictions", []),
        policy=prov_dict.get("policy", ""),
        scope=prov_dict.get("scope", ""),
    )

    logger.info("Processing %s (source: %s — %s)", config.ds_id, prov.source_id, prov.source_title)
    logger.info("Remote: %s://%s/%s", config.remote, config.bucket, config.remote_prefix)
    logger.info("Restrictions: %s", ", ".join(prov.restrictions) if prov.restrictions else "(none)")

    streamer = S3Streamer(remote=config.remote, bucket=config.bucket, prefix="")

    records, total_streamed, valid_count = _stream_and_validate(config, streamer, prov)

    deduped, stats = deduplicate_records(records, use_secondary_hash=False)
    logger.info(
        "Dedup: %d unique from %d total (%d removed, %d stage conflicts resolved)",
        stats.unique_records,
        stats.total_records,
        stats.duplicates_removed,
        stats.stage_conflicts_resolved,
    )

    ds_lower = config.ds_id.lower().replace("-", "")
    out_jsonl = config.output_dir / f"{ds_lower}_deduped.jsonl"
    out_manifest = config.output_dir / f"{ds_lower}_manifest.json"

    manifest = HoldManifest(
        ds_id=config.ds_id,
        source_id=prov.source_id,
        source_title=prov.source_title,
        remote=config.remote,
        remote_prefix=config.remote_prefix,
        output_path=str(out_jsonl),
        total_records_streamed=total_streamed,
        valid_records=valid_count,
        unique_records=stats.unique_records,
        duplicates_removed=stats.duplicates_removed,
        stage_conflicts_resolved=stats.stage_conflicts_resolved,
        records_by_stage=dict(stats.records_by_stage) if stats.records_by_stage else {},
        restrictions=prov.restrictions,
        policy=prov.policy,
        scope=prov.scope,
        processed_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    )

    if not config.dry_run:
        _write_outputs(out_jsonl, out_manifest, deduped, manifest)

    return manifest


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="CONVERT/HASH a single corpus to a HOLD directory.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Guardrails:\n"
            "  • Output goes to ai/data/hold/<DS-ID>/ — NEVER into train_all.\n"
            "  • Each invocation processes exactly one DS-* identifier.\n"
            "  • Do not combine with DS-007 or any other corpus.\n"
        ),
    )
    parser.add_argument("--ds-id", required=True, help="Dataset alias (e.g. DS-006)")
    parser.add_argument("--remote", default="whitebat", help="rclone remote name")
    parser.add_argument("--bucket", default="training", help="rclone bucket")
    parser.add_argument(
        "--remote-prefix",
        default="pixelated-empathy/ingestion/master_work",
        help="Path under the bucket where corpus JSONL files live",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Local output directory (default: ai/data/hold/<DS-ID>)",
    )
    parser.add_argument(
        "--max-records",
        type=int,
        default=None,
        help="Cap records streamed (smoke testing). Omit for full run.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Stream and validate but skip writing output")
    parser.add_argument("--verbose", "-v", action="store_true", help="Debug logging")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    output_dir = Path(args.output_dir) if args.output_dir else Path("ai/data/hold") / args.ds_id

    # Safety: refuse to write into train_all or curated splits.
    resolved = output_dir.resolve()
    for forbidden in ("/train_all", "/curated/"):
        if forbidden in str(resolved):
            logger.error("Output dir %s must not be inside %s — aborting", resolved, forbidden)
            sys.exit(2)

    config = CorpusConfig(
        ds_id=args.ds_id,
        remote=args.remote,
        bucket=args.bucket,
        remote_prefix=args.remote_prefix,
        output_dir=output_dir,
        max_records=args.max_records,
        dry_run=args.dry_run,
    )

    manifest = process_corpus(config)

    logger.info(
        "HOLD complete: %s → %d unique records in %s",
        manifest.ds_id,
        manifest.unique_records,
        manifest.output_path,
    )


if __name__ == "__main__":
    main()
