#!/usr/bin/env python3
"""CONVERT/HASH HOLD corpus pipeline for individual dataset snapshots.

Streams a corpus from the rclone remote, validates ChatML format, attaches
source-family provenance from the manifest, computes SHA-256 primary hashes,
deduplicates, and writes the result to a HOLD directory that is NEVER merged
into ``train_all``.

Supports multiple remotes (whitebat, gdrive, upcloud) and file formats
(JSONL, JSON, CSV, AnnoMI conversation format).

Usage::

    uv run python -m ai.pipelines.corpus_hold.convert_hash_hold \\
        --ds-id DS-006

    # Override any pre-configured default
    uv run python -m ai.pipelines.corpus_hold.convert_hash_hold \\
        --ds-id DS-012 \\
        --remote whitebat \\
        --remote-prefix pixelated-empathy/output/cbt_llm_coreissue

The script is corpus-scoped: it processes exactly one DS-* identifier and
writes to an isolated HOLD path.  It must not be combined with DS-007 or any
other corpus.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import re
import sys
import tempfile
import time
from collections.abc import Generator
from dataclasses import asdict, dataclass, field
from pathlib import Path

import yaml

from ai.pipelines.data_processing.extractors.s3_streamer import S3Streamer
from ai.pipelines.ingestion_deduplication import compute_primary_hash

logger = logging.getLogger("corpus_hold")

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

_DATA_DIR = Path(__file__).resolve().parents[2] / "data"
_MANIFEST_PATH = _DATA_DIR / "source_family_provenance_manifest.yaml"
_CLINICAL_MANIFEST_PATH = _DATA_DIR / "source_family_clinical_dialogue_manifest.yaml"

# Map DS-* aliases to their SRC-* source_id.
# DS-006/007 → SRC-002 (provenance manifest)
# DS-008 → SRC-003 (provenance manifest)
# DS-096 → SRC-080 (provenance manifest)
# DS-012 → SRC-015 (clinical dialogue manifest)
# DS-015 → SRC-018 (clinical dialogue manifest)
# DS-060 → SRC-038 (clinical dialogue manifest)
# DS-064 → SRC-042 (clinical dialogue manifest)
# DS-069 → SRC-047 (clinical dialogue manifest)
# burnout-8 → SRC-026 (clinical dialogue manifest)
_DS_TO_SRC = {
    "DS-006": "SRC-002",
    "DS-007": "SRC-002",
    "DS-008": "SRC-003",
    "DS-012": "SRC-015",
    "DS-015": "SRC-018",
    "DS-060": "SRC-038",
    "DS-064": "SRC-042",
    "DS-069": "SRC-047",
    "DS-096": "SRC-080",
    "burnout-8": "SRC-026",
}

# Pre-configured defaults for each corpus.  When --remote/--bucket/--remote-prefix
# are omitted on the CLI, these values are used automatically.
_CORPUS_DEFAULTS: dict[str, dict] = {
    "DS-008": {
        "remote": "whitebat",
        "bucket": "training",
        "remote_prefix": "pixelated-empathy/ingestion/master_work",
        "format": "jsonl",
        "file_filter": "hf_large_corpora_clean.jsonl",
    },
    "DS-012": {
        "remote": "whitebat",
        "bucket": "training",
        "remote_prefix": "pixelated-empathy/output/cbt_llm_coreissue",
        "format": "jsonl",
    },
    "DS-015": {
        "remote": "gdrive",
        "bucket": "",
        "remote_prefix": "pixeldata/archive/gdrive/raw/datasets/CoT_Reasoning_Clinical_Diagnosis_Mental_Health",
        "format": "json",
    },
    "DS-060": {
        "remote": "",
        "bucket": "",
        "remote_prefix": "",
        "format": "doi",
        "doi_url": "https://doi.org/10.5522/04/31587925.v1",
        "note": "DOI-only dataset — no files in remote archives. Requires manual download.",
    },
    "DS-064": {
        "remote": "gdrive",
        "bucket": "",
        "remote_prefix": "pixeldata/archive/gdrive/raw/formatted_annotated_addiction_counseling_csv_SFT",
        "format": "csv",
    },
    "DS-069": {
        "remote": "gdrive",
        "bucket": "",
        "remote_prefix": (
            "pixeldata/archive/local_sync/ai/datasets/tier2_professional/motivational-interviewing-therapy"
        ),
        "format": "annomi",
    },
    "DS-096": {
        "remote": "upcloud",
        "bucket": "crispy",
        "remote_prefix": "training/ai-data/raw/kickflip/joiner",
        "format": "jsonl",
    },
    "burnout-8": {
        "remote": "whitebat",
        "bucket": "training",
        "remote_prefix": "pixelated-empathy/output",
        "format": "jsonl",
        "subdirs": [
            "burnout_buried_scrubs",
            "burnout_final_out",
            "burnout_heading_hills",
            "burnout_how_good_therapists_leave",
            "burnout_i_want_to_quit",
            "burnout_last_attempt",
            "burnout_let_go_working",
            "burnout_therapist_interrupted",
            "burnout_why_counselors_quit",
        ],
    },
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
    file_format: str = "jsonl"
    file_filter: str | None = None
    subdirs: list[str] | None = None
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
    file_format: str = ""
    processed_at: str = ""


# ---------------------------------------------------------------------------
# Provenance
# ---------------------------------------------------------------------------


def _load_manifest(path: Path) -> list[dict]:
    """Load a YAML manifest file and return a list of source entries."""
    with path.open(encoding="utf-8") as fh:
        manifest = yaml.safe_load(fh)
    if isinstance(manifest, dict):
        return (
            list(manifest.get("sources", manifest).values())
            if isinstance(manifest.get("sources", manifest), dict)
            else manifest.get("sources", manifest)
        )
    return manifest if isinstance(manifest, list) else []


def load_provenance(ds_id: str) -> dict:
    """Load the SRC-* provenance entry for a given DS-* alias.

    Searches both the provenance manifest and the clinical dialogue manifest.
    """
    target_src = _DS_TO_SRC.get(ds_id, "")
    for manifest_path in (_MANIFEST_PATH, _CLINICAL_MANIFEST_PATH):
        if not manifest_path.exists():
            continue
        entries = _load_manifest(manifest_path)
        for entry in entries:
            if not isinstance(entry, dict):
                continue
            aliases = entry.get("ds_alias") or []
            if isinstance(aliases, str):
                aliases = [aliases]
            if ds_id in aliases or entry.get("source_id") == target_src:
                return entry

    logger.warning("No provenance entry found for %s — proceeding with minimal metadata", ds_id)
    return {}


# ---------------------------------------------------------------------------
# Format adapters
# ---------------------------------------------------------------------------

_ANNOMI_ROLE_MAP = {"gpt": "assistant", "human": "user", "system": "system"}


def _convert_annomi_to_chatml(record: dict) -> dict | None:
    """Convert AnnoMI format to ChatML.

    AnnoMI uses ``{conversations: [{from: "gpt"|"human", value: str}]}``.
    """
    conversations = record.get("conversations")
    if not isinstance(conversations, list) or len(conversations) == 0:
        return None
    messages = []
    for turn in conversations:
        if not isinstance(turn, dict):
            return None
        role = _ANNOMI_ROLE_MAP.get(turn.get("from", ""), "user")
        content = turn.get("value", "")
        if not isinstance(content, str) or not content:
            return None
        messages.append({"role": role, "content": content})
    if not messages:
        return None
    return {
        "messages": messages,
        "metadata": {"original_format": "annomi", "original_id": str(record.get("id", ""))},
    }


_CSV_CONTEXT_RE = re.compile(r"### Context:\s*(.*?)(?=### Input:)", re.DOTALL)
_CSV_INPUT_RE = re.compile(r"### Input:\s*(.*?)(?=### Response:)", re.DOTALL)
_CSV_RESPONSE_RE = re.compile(r"### Response:\s*(.*)", re.DOTALL)


def _convert_csv_addiction_to_chatml(text: str) -> dict | None:
    """Convert Addiction Stories CSV text to ChatML.

    CSV text column uses ``### Context: ... ### Input: ... ### Response: ...``.
    """
    input_match = _CSV_INPUT_RE.search(text)
    response_match = _CSV_RESPONSE_RE.search(text)
    if not (input_match and response_match):
        return None

    context_match = _CSV_CONTEXT_RE.search(text)
    context = context_match.group(1).strip() if context_match else ""
    user_content = input_match.group(1).strip()
    if context:
        user_content = f"### Context: {context}\n### Input: {user_content}"
    assistant_content = response_match.group(1).strip()

    return {
        "messages": [
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": assistant_content},
        ],
        "metadata": {"original_format": "csv_addiction"},
    }


def _convert_mentalllama_to_chatml(record: dict) -> dict | None:
    """Convert MentalLLaMA JSON record to ChatML.

    MentalLLaMA records typically have ``{question: ..., answer: ...}`` fields.
    """
    question = record.get("question") or record.get("input") or record.get("prompt")
    answer = record.get("answer") or record.get("output") or record.get("response")
    if not (question and answer):
        return None
    return {
        "messages": [
            {"role": "user", "content": str(question)},
            {"role": "assistant", "content": str(answer)},
        ],
        "metadata": {"original_format": "mentalllama"},
    }


def to_chatml(record: dict, file_format: str) -> dict | None:
    """Convert a record to ChatML format based on the corpus file format.

    Returns ``None`` if the record cannot be converted.
    """
    if file_format in ("jsonl", "json"):
        if validate_chatml(record):
            return record
        if file_format == "json":
            return _convert_mentalllama_to_chatml(record)
        return None

    if file_format == "annomi":
        return _convert_annomi_to_chatml(record)

    if file_format == "csv" and isinstance(record, str):
        return _convert_csv_addiction_to_chatml(record)

    return None


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


@dataclass
class StreamStats:
    """Stats collected during streaming dedup."""

    total_streamed: int = 0
    valid_records: int = 0
    unique_records: int = 0
    duplicates_removed: int = 0
    records_by_stage: dict[str, int] = field(default_factory=dict)


def _iter_jsonl_records(
    streamer: S3Streamer,
    config: CorpusConfig,
    prefixes: list[str],
) -> Generator[dict]:
    """Yield parsed JSONL records from one or more remote prefixes."""
    for prefix in prefixes:
        file_keys = _list_jsonl_files(streamer, prefix, config)
        for file_key in file_keys:
            logger.info("Streaming %s ...", file_key)
            try:
                yield from streamer.stream_jsonl(file_key)
            except Exception:
                logger.exception("Error streaming %s — output may be truncated", file_key)
                raise


def _list_jsonl_files(
    streamer: S3Streamer,
    prefix: str,
    config: CorpusConfig,
) -> list[str]:
    """List JSONL files under a prefix, applying optional file filter."""
    list_prefix = prefix.rstrip("/") + "/"
    files = streamer.list_files(prefix=list_prefix, recursive=True)
    jsonl_files = [f for f in files if f.endswith(".jsonl")]

    if config.file_filter:
        jsonl_files = [f for f in jsonl_files if config.file_filter in f]

    if not jsonl_files:
        # For JSON/CSV/AnnoMI formats, also list .json and .csv files
        if config.file_format in ("json", "annomi"):
            jsonl_files = [f for f in files if f.endswith(".json")]
        elif config.file_format == "csv":
            jsonl_files = [f for f in files if f.endswith(".csv")]

    if not jsonl_files:
        logger.error(
            "No matching files found under %s://%s/%s (format: %s)",
            config.remote,
            config.bucket,
            prefix,
            config.file_format,
        )
        sys.exit(1)

    logger.info("Found %d file(s): %s", len(jsonl_files), ", ".join(jsonl_files[:5]))
    return jsonl_files


def _iter_download_records(
    streamer: S3Streamer,
    config: CorpusConfig,
    prefixes: list[str],
) -> Generator[dict]:
    """Download non-JSONL files (JSON/CSV) and yield records from them."""
    for prefix in prefixes:
        file_keys = _list_jsonl_files(streamer, prefix, config)
        for file_key in file_keys:
            logger.info("Downloading %s ...", file_key)
            with tempfile.NamedTemporaryFile(
                suffix=Path(file_key).suffix,
                mode="wb",
                delete=False,
            ) as tmp:
                tmp_path = tmp.name
            try:
                streamer.download_to_file(file_key, tmp_path)

                if config.file_format == "json":
                    # Single JSON file — may be a list of records or a dict
                    with Path(tmp_path).open(encoding="utf-8") as fh:
                        data = json.load(fh)
                    if isinstance(data, list):
                        yield from data
                    elif isinstance(data, dict):
                        yield data
                    else:
                        logger.warning("Unexpected JSON type in %s", file_key)

                elif config.file_format == "csv":
                    # CSV files — read with csv module
                    with Path(tmp_path).open(encoding="utf-8", newline="") as fh:
                        reader = csv.DictReader(fh)
                        for row in reader:
                            yield dict(row)

                elif config.file_format == "annomi":
                    # AnnoMI JSONL — stream as JSONL but convert format
                    with Path(tmp_path).open(encoding="utf-8") as fh:
                        for line in fh:
                            stripped = line.strip()
                            if stripped:
                                yield json.loads(stripped)

            finally:
                Path(tmp_path).unlink(missing_ok=True)


def _iter_records(
    streamer: S3Streamer,
    config: CorpusConfig,
) -> Generator[dict]:
    """Yield raw records from the corpus in their native format.

    Dispatches to the appropriate reader based on ``config.file_format``.
    """
    # Build list of prefixes to search
    if config.subdirs:
        prefixes = [f"{config.remote_prefix.rstrip('/')}/{subdir}" for subdir in config.subdirs]
    else:
        prefixes = [config.remote_prefix]

    if config.file_format in ("jsonl", "annomi"):
        yield from _iter_jsonl_records(streamer, config, prefixes)
    elif config.file_format in ("json", "csv"):
        yield from _iter_download_records(streamer, config, prefixes)
    elif config.file_format == "doi":
        logger.error(
            "DS-%s is DOI-only (no remote files). Download manually from %s",
            config.ds_id,
            _CORPUS_DEFAULTS.get(config.ds_id, {}).get("doi_url", "(unknown URL)"),
        )
        sys.exit(0)
    else:
        logger.error("Unknown file format: %s", config.file_format)
        sys.exit(1)


def _stream_dedup_write(
    config: CorpusConfig,
    streamer: S3Streamer,
    prov: ProvenanceInfo,
    out_jsonl: Path,
) -> StreamStats:
    """Stream records, convert to ChatML, hash, dedup, and write unique records to disk.

    This avoids loading all records into memory — only the set of seen hashes
    (~32 bytes each) is held, making it safe for corpora with millions of records.
    """
    out_jsonl.parent.mkdir(parents=True, exist_ok=True)
    seen_hashes: set[str] = set()
    stats = StreamStats()

    with out_jsonl.open("w", encoding="utf-8") as fh:
        for raw_record in _iter_records(streamer, config):
            stats.total_streamed += 1
            if config.max_records is not None and stats.total_streamed >= config.max_records:
                logger.info("Reached --max-records cap (%d), stopping stream", config.max_records)
                break

            # Convert to ChatML
            record = to_chatml(raw_record, config.file_format)
            if record is None:
                logger.debug("Skipping unconvertible record #%d (format: %s)", stats.total_streamed, config.file_format)
                continue

            if not validate_chatml(record):
                logger.debug("Skipping invalid ChatML record #%d", stats.total_streamed)
                continue

            stats.valid_records += 1

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

            pkey = record["metadata"]["primary_hash"]
            if pkey in seen_hashes:
                stats.duplicates_removed += 1
                continue

            seen_hashes.add(pkey)
            stats.unique_records += 1

            stage = record.get("metadata", {}).get("stage", "supplementary")
            stats.records_by_stage[stage] = stats.records_by_stage.get(stage, 0) + 1

            fh.write(json.dumps(record, ensure_ascii=False) + "\n")

    logger.info(
        "Streamed %d records, %d valid, %d unique (%d duplicates removed)",
        stats.total_streamed,
        stats.valid_records,
        stats.unique_records,
        stats.duplicates_removed,
    )
    return stats


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------


def process_corpus(config: CorpusConfig) -> HoldManifest:
    """Stream, validate, hash, deduplicate, and write a single corpus to HOLD."""
    prov_dict = load_provenance(config.ds_id)
    prov = ProvenanceInfo(
        source_id=prov_dict.get("source_id", _DS_TO_SRC.get(config.ds_id, "")),
        source_title=prov_dict.get("title", ""),
        restrictions=prov_dict.get("rights_decision", {}).get("restrictions", [])
        if isinstance(prov_dict.get("rights_decision"), dict)
        else prov_dict.get("restrictions", []),
        policy=prov_dict.get("policy", ""),
        scope=prov_dict.get("scope", ""),
    )

    logger.info("Processing %s (source: %s — %s)", config.ds_id, prov.source_id, prov.source_title)
    logger.info(
        "Remote: %s://%s/%s (format: %s)",
        config.remote,
        config.bucket,
        config.remote_prefix,
        config.file_format,
    )
    logger.info("Restrictions: %s", ", ".join(prov.restrictions) if prov.restrictions else "(none)")

    # DOI-only datasets cannot be processed remotely
    if config.file_format == "doi":
        doi_url = _CORPUS_DEFAULTS.get(config.ds_id, {}).get("doi_url", "")
        logger.error(
            "%s is a DOI-only dataset (no files in remote archives). "
            "Download manually from %s and place under ai/data/raw/ before running.",
            config.ds_id,
            doi_url,
        )
        return HoldManifest(
            ds_id=config.ds_id,
            source_id=prov.source_id,
            source_title=prov.source_title,
            remote="",
            remote_prefix="",
            output_path="",
            total_records_streamed=0,
            valid_records=0,
            unique_records=0,
            duplicates_removed=0,
            stage_conflicts_resolved=0,
            restrictions=prov.restrictions,
            policy=prov.policy,
            scope=prov.scope,
            file_format="doi",
            processed_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        )

    streamer = S3Streamer(remote=config.remote, bucket=config.bucket, prefix="")

    ds_lower = config.ds_id.lower().replace("-", "")
    out_jsonl = config.output_dir / f"{ds_lower}_deduped.jsonl"
    out_manifest = config.output_dir / f"{ds_lower}_manifest.json"

    if config.dry_run:
        out_jsonl = Path("/dev/null")

    stats = _stream_dedup_write(config, streamer, prov, out_jsonl)
    logger.info("Wrote %d deduped records to %s", stats.unique_records, out_jsonl)

    manifest = HoldManifest(
        ds_id=config.ds_id,
        source_id=prov.source_id,
        source_title=prov.source_title,
        remote=config.remote,
        remote_prefix=config.remote_prefix,
        output_path=str(out_jsonl),
        total_records_streamed=stats.total_streamed,
        valid_records=stats.valid_records,
        unique_records=stats.unique_records,
        duplicates_removed=stats.duplicates_removed,
        stage_conflicts_resolved=0,
        records_by_stage=stats.records_by_stage,
        restrictions=prov.restrictions,
        policy=prov.policy,
        scope=prov.scope,
        file_format=config.file_format,
        processed_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    )

    if not config.dry_run:
        with out_manifest.open("w", encoding="utf-8") as fh:
            json.dump(asdict(manifest), fh, indent=2, ensure_ascii=False)
        logger.info("Wrote manifest to %s", out_manifest)

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
            "  - Output goes to ai/data/hold/<DS-ID>/ -- NEVER into train_all.\n"
            "  - Each invocation processes exactly one DS-* identifier.\n"
            "  - Do not combine with other corpora.\n"
            "\n"
            "Supported DS-* IDs with pre-configured defaults:\n"
            "  DS-006, DS-007 (SRC-002 Master-clean aggregate, whitebat JSONL)\n"
            "  DS-008 (SRC-003 HF cleaned aggregate, whitebat JSONL)\n"
            "  DS-012 (SRC-015 CBT LLM CoreIssue, whitebat JSONL)\n"
            "  DS-015 (SRC-018 MentalLLaMA, gdrive JSON)\n"
            "  DS-060 (SRC-038 UCL psych abuse, DOI-only -- manual download)\n"
            "  DS-064 (SRC-042 Addiction Stories, gdrive CSV)\n"
            "  DS-069 (SRC-047 AnnoMI, gdrive JSONL AnnoMI format)\n"
            "  DS-096 (SRC-080 Kickflip RL, upcloud JSONL)\n"
            "  burnout-8 (SRC-026 Burnout source family, whitebat JSONL multi-dir)\n"
        ),
    )
    parser.add_argument("--ds-id", required=True, help="Dataset alias (e.g. DS-006, burnout-8)")
    parser.add_argument("--remote", default=None, help="rclone remote name (default: from corpus config)")
    parser.add_argument("--bucket", default=None, help="rclone bucket (default: from corpus config)")
    parser.add_argument(
        "--remote-prefix",
        default=None,
        help="Path under the bucket where corpus files live (default: from corpus config)",
    )
    parser.add_argument(
        "--format",
        default=None,
        choices=["jsonl", "json", "csv", "annomi", "doi"],
        help="File format of the corpus (default: from corpus config)",
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

    # Apply corpus defaults, then CLI overrides
    defaults = _CORPUS_DEFAULTS.get(args.ds_id, {})
    remote = args.remote or defaults.get("remote", "whitebat")
    bucket = args.bucket if args.bucket is not None else defaults.get("bucket", "training")
    remote_prefix = args.remote_prefix or defaults.get("remote_prefix", "pixelated-empathy/ingestion/master_work")
    file_format = args.format or defaults.get("format", "jsonl")
    file_filter = defaults.get("file_filter")
    subdirs = defaults.get("subdirs")

    output_dir = Path(args.output_dir) if args.output_dir else Path("ai/data/hold") / args.ds_id

    # Safety: refuse to write into train_all or curated splits.
    resolved = output_dir.resolve()
    for forbidden in ("/train_all", "/curated/"):
        if forbidden in str(resolved):
            logger.error("Output dir %s must not be inside %s -- aborting", resolved, forbidden)
            sys.exit(2)

    config = CorpusConfig(
        ds_id=args.ds_id,
        remote=remote,
        bucket=bucket,
        remote_prefix=remote_prefix,
        output_dir=output_dir,
        file_format=file_format,
        file_filter=file_filter,
        subdirs=subdirs,
        max_records=args.max_records,
        dry_run=args.dry_run,
    )

    manifest = process_corpus(config)

    logger.info(
        "HOLD complete: %s -> %d unique records in %s",
        manifest.ds_id,
        manifest.unique_records,
        manifest.output_path,
    )


if __name__ == "__main__":
    main()
