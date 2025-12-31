#!/usr/bin/env python3
"""Build a CPTSD-tagged ChatML dataset from CPTSD-focused transcript corpora.

Primary target: Tim Fletcher transcripts in S3 (tier4_voice_persona/Tim Fletcher/*.txt)
or local directories.

Outputs:
- ai/training_ready/data/generated/cptsd_transcripts.jsonl
- ai/training_ready/data/generated/cptsd_transcripts_stats.json

Features:
- Streams from S3 or processes local files (memory-efficient)
- Converts transcript text to ChatML format
- Uploads output directly to S3 with --upload-s3 flag
- Scans directories with --input-dir flag
- Progress logging for long-running operations
- Integrates Tim Fletcher voice profile for authentic system prompts

Usage Examples:
    # Extract from default S3 prefix (Tim Fletcher transcripts)
    python build_cptsd_dataset_from_transcripts.py

    # Extract from local directory
    python build_cptsd_dataset_from_transcripts.py \
      --input-dir ~/datasets/gdrive/tier4_voice_persona/Tim\ Fletcher/

    # Process specific files and upload to S3
    python build_cptsd_dataset_from_transcripts.py \
      --source-key /path/to/transcript.txt \
      --upload-s3 --verbose

Notes:
- Uses ai/training_ready/data/s3_manifest.json as the source of truth for bucket/endpoint.
- Does not print transcript content.
- Applies light redaction for obvious PII patterns (emails/phones/urls).
- Uses Tim Fletcher voice profile for system prompts when available.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import re
from collections import Counter
from collections.abc import Iterator
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ai.training_ready.utils.s3_dataset_loader import S3DatasetLoader

# PII redaction patterns
EMAIL_RE = re.compile(r"[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}", re.IGNORECASE)
PHONE_RE = re.compile(r"\b(?:\+?\d{1,3}[-. (]*)?(?:\d{3}[-. )]*)\d{3}[-. ]*\d{4}\b")
URL_RE = re.compile(r"https?://\S+", re.IGNORECASE)
TIMESTAMP_RE = re.compile(r"\b\d{1,2}:\d{2}(?::\d{2})?\b")

# Progress logging interval
PROGRESS_LOG_INTERVAL = 50

# Default S3 prefixes for Tim Fletcher transcripts
DEFAULT_S3_PREFIXES = [
    "datasets/gdrive/tier4_voice_persona/Tim Fletcher/",
]

# Voice profile path (ai/data/tim_fletcher_voice/tim_fletcher_voice_profile.json)
# Script is at ai/training_ready/scripts/, so parents[2] = ai/
VOICE_PROFILE_PATH = Path(__file__).parents[2] / "data" / "tim_fletcher_voice" / "tim_fletcher_voice_profile.json"

logger = logging.getLogger(__name__)


def _load_voice_profile() -> dict[str, Any] | None:
    """Load Tim Fletcher voice profile if available."""
    if VOICE_PROFILE_PATH.exists():
        try:
            with open(VOICE_PROFILE_PATH, encoding="utf-8") as f:
                return json.load(f)
        except (OSError, json.JSONDecodeError) as e:
            logger.warning(f"Failed to load voice profile: {e}")
    return None


def _build_system_prompt(voice_profile: dict[str, Any] | None = None) -> str:
    """Build a system prompt, optionally enhanced with voice profile."""
    base_prompt = (
        "You are a trauma-informed therapeutic AI assistant specializing in Complex PTSD "
        "(CPTSD) and complex trauma. Use a grounded, practical tone. "
        "Explain CPTSD concepts clearly, with compassion and actionable steps. "
        "Do not include personal identifying information."
    )
    
    if not voice_profile:
        return base_prompt
    
    # Extract key elements from voice profile
    empathy_markers = voice_profile.get("empathy_markers", {})
    teaching_patterns = voice_profile.get("teaching_patterns", [])
    transition_phrases = voice_profile.get("transition_phrases", {})
    
    # Build enhanced prompt with voice characteristics
    top_empathy = [k for k, _ in sorted(empathy_markers.items(), key=lambda x: -x[1])[:3]]
    top_transitions = [k for k, _ in sorted(transition_phrases.items(), key=lambda x: -x[1])[:3]]
    
    teaching_tips = []
    for p in teaching_patterns[:3]:
        if isinstance(p, dict) and "pattern" in p:
            teaching_tips.append(p["pattern"])
        elif isinstance(p, str):
            teaching_tips.append(p)
    
    enhanced_prompt = (
        f"{base_prompt}\n\n"
        "Voice characteristics:\n"
        f"- Use empathetic phrases like: {', '.join(top_empathy)}\n"
        f"- Use transition phrases: {', '.join(top_transitions)}\n"
        f"- Structure explanations with: {', '.join(teaching_tips) if teaching_tips else 'First, Second, Third'}\n"
        "- Provide concrete examples and analogies\n"
        "- Acknowledge the difficulty while offering hope and practical steps"
    )
    
    return enhanced_prompt


def _clean_text(text: str) -> str:
    """Clean and redact PII from transcript text."""
    text = URL_RE.sub("[URL]", text)
    text = EMAIL_RE.sub("[EMAIL]", text)
    text = PHONE_RE.sub("[PHONE]", text)
    text = TIMESTAMP_RE.sub("", text)
    # Normalize whitespace
    text = re.sub(r"[\t\r]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r" {2,}", " ", text)
    return text.strip()


def _chunk_text(text: str, *, max_chars: int = 1800, min_chars: int = 500) -> list[str]:
    """Split text into chunks based on paragraphs."""
    # Split on paragraphs, then re-pack
    paras = [p.strip() for p in text.split("\n\n") if p.strip()]
    chunks: list[str] = []
    cur: list[str] = []
    cur_len = 0

    for p in paras:
        if cur_len + len(p) + 2 > max_chars and cur_len >= min_chars:
            chunks.append("\n\n".join(cur).strip())
            cur = []
            cur_len = 0
        cur.append(p)
        cur_len += len(p) + 2

    if cur_len >= min_chars:
        chunks.append("\n\n".join(cur).strip())

    return chunks


def _title_from_path(path: str) -> str:
    """Extract a human-readable title from file path."""
    name = path.split("/")[-1]
    if name.lower().endswith(".txt"):
        name = name[:-4]
    return name.replace("_", " ").strip()


def _content_hash(content: str) -> str:
    """Generate SHA256 hash of content for deduplication."""
    return f"sha256:{hashlib.sha256(content.encode('utf-8')).hexdigest()[:16]}"


def _is_local_path(path: str) -> bool:
    """Check if path is a local filesystem path (not S3)."""
    if path.startswith("s3://"):
        return False
    return (
        path.startswith("/")
        or path.startswith("./")
        or path.startswith("../")
        or path.startswith("~")
        or (len(path) > 1 and path[1] == ":")  # Windows drive letter
    )


def _list_txt_files_in_dir(
    loader: S3DatasetLoader | None,
    *,
    bucket: str,
    input_dir: str,
) -> list[str]:
    """List all .txt files in a directory (S3 prefix or local)."""
    files: list[str] = []
    
    if input_dir.startswith("s3://"):
        # Parse S3 URI
        without_prefix = input_dir[5:]  # Remove s3://
        if "/" in without_prefix:
            s3_bucket, prefix = without_prefix.split("/", 1)
        else:
            s3_bucket = without_prefix
            prefix = ""
        
        if loader is None:
            logger.error("S3DatasetLoader required for S3 paths")
            return files
            
        logger.info(f"Scanning S3 prefix: s3://{s3_bucket}/{prefix}")
        try:
            paginator = loader.s3_client.get_paginator("list_objects_v2")
            pages = paginator.paginate(Bucket=s3_bucket, Prefix=prefix)
            
            for page in pages:
                if "Contents" in page:
                    for obj in page["Contents"]:
                        key = obj["Key"]
                        if key.lower().endswith(".txt"):
                            files.append(f"s3://{s3_bucket}/{key}")
        except Exception as e:
            logger.error(f"Failed to list S3 objects: {e}")
    elif _is_local_path(input_dir):
        # Local directory
        local_dir = Path(input_dir).expanduser()
        if local_dir.is_dir():
            logger.info(f"Scanning local directory: {local_dir}")
            for f in local_dir.rglob("*.txt"):
                files.append(str(f))
        else:
            logger.warning(f"Local directory not found: {input_dir}")
    else:
        # Treat as S3 prefix without s3:// prefix
        if loader is None:
            logger.error("S3DatasetLoader required for S3 paths")
            return files
            
        prefix = input_dir
        logger.info(f"Scanning S3 prefix: s3://{bucket}/{prefix}")
        try:
            paginator = loader.s3_client.get_paginator("list_objects_v2")
            pages = paginator.paginate(Bucket=bucket, Prefix=prefix)
            
            for page in pages:
                if "Contents" in page:
                    for obj in page["Contents"]:
                        key = obj["Key"]
                        if key.lower().endswith(".txt"):
                            files.append(key)
        except Exception as e:
            logger.error(f"Failed to list S3 objects: {e}")
    
    logger.info(f"Found {len(files)} .txt files")
    return files


def _list_transcript_keys_from_manifest(manifest: dict, *, prefix: str) -> list[str]:
    """Extract transcript keys from S3 manifest matching prefix."""
    keys: list[str] = []

    def walk(node: Any) -> None:
        if isinstance(node, dict):
            objs = node.get("objects")
            if isinstance(objs, list):
                for o in objs:
                    k = o.get("key")
                    if isinstance(k, str) and k.startswith(prefix) and k.lower().endswith(".txt"):
                        keys.append(k)
            for v in node.values():
                walk(v)
        elif isinstance(node, list):
            for v in node:
                walk(v)

    walk(manifest.get("categories", {}))
    return sorted(set(keys))


def _load_local_text(path: str) -> str | None:
    """Load text from a local file."""
    try:
        local_path = Path(path).expanduser()
        with open(local_path, encoding="utf-8", errors="replace") as f:
            return f.read()
    except OSError as e:
        logger.warning(f"Failed to read local file {path}: {e}")
        return None


def _load_text(
    loader: S3DatasetLoader | None,
    path: str,
    bucket: str,
) -> str | None:
    """Load text from S3 or local path."""
    if _is_local_path(path):
        return _load_local_text(path)
    
    if loader is None:
        logger.warning(f"S3DatasetLoader required for S3 path: {path}")
        return None
    
    # Handle S3 paths
    if path.startswith("s3://"):
        s3_path = path
    else:
        s3_path = f"s3://{bucket}/{path}"
    
    try:
        return loader.load_text(s3_path)
    except FileNotFoundError:
        logger.warning(f"File not found: {s3_path}")
        return None
    except Exception as e:
        logger.warning(f"Error loading {s3_path}: {e}")
        return None


def _upload_to_s3(
    loader: S3DatasetLoader,
    *,
    local_path: Path,
    s3_key: str,
    bucket: str,
) -> bool:
    """Upload a local file to S3."""
    try:
        logger.info(f"Uploading {local_path} to s3://{bucket}/{s3_key}")
        loader.s3_client.upload_file(str(local_path), bucket, s3_key)
        logger.info(f"✓ Uploaded to s3://{bucket}/{s3_key}")
        return True
    except Exception as e:
        logger.error(f"Failed to upload to S3: {e}")
        return False


def _iter_source_files(
    loader: S3DatasetLoader | None,
    *,
    bucket: str,
    source_keys: list[str],
    input_dir: str | None,
    manifest: dict | None,
    prefix: str,
) -> Iterator[str]:
    """Iterate over all source file paths to process."""
    # Priority: input_dir > source_keys > manifest prefix
    if input_dir:
        yield from _list_txt_files_in_dir(loader, bucket=bucket, input_dir=input_dir)
    elif source_keys:
        yield from source_keys
    elif manifest:
        yield from _list_transcript_keys_from_manifest(manifest, prefix=prefix)
    else:
        # Default: scan default S3 prefixes
        for default_prefix in DEFAULT_S3_PREFIXES:
            yield from _list_txt_files_in_dir(loader, bucket=bucket, input_dir=default_prefix)


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build CPTSD-tagged ChatML dataset from transcript corpora.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s                                    # Use default S3 sources (Tim Fletcher)
  %(prog)s --input-dir ~/datasets/gdrive/tier4_voice_persona/Tim\\ Fletcher/
  %(prog)s --source-key /path/to/transcript.txt --upload-s3 --verbose
  %(prog)s --prefix datasets/gdrive/tier4_voice_persona/Tim\\ Fletcher/ --upload-s3
""",
    )
    parser.add_argument(
        "--manifest",
        default=str(Path(__file__).parents[1] / "data" / "s3_manifest.json"),
        metavar="PATH",
        help="Path to s3_manifest.json for bucket/endpoint config (default: ai/training_ready/data/s3_manifest.json)",
    )
    parser.add_argument(
        "--prefix",
        default="datasets/gdrive/tier4_voice_persona/Tim Fletcher/",
        metavar="PREFIX",
        help="S3 key prefix to pull transcripts from (used with manifest)",
    )
    parser.add_argument(
        "--source-key",
        action="append",
        default=[],
        metavar="PATH",
        help="S3 key or local path to process (repeatable). Supports s3://bucket/key, relative S3 keys, or local paths.",
    )
    parser.add_argument(
        "--input-dir",
        metavar="DIR",
        help="S3 prefix (s3://bucket/prefix/) or local directory to scan for all .txt files. Overrides --source-key and --prefix.",
    )
    parser.add_argument(
        "--output",
        default=str(Path(__file__).parents[1] / "data" / "generated" / "cptsd_transcripts.jsonl"),
        metavar="PATH",
        help="Output JSONL file path (default: ai/training_ready/data/generated/cptsd_transcripts.jsonl)",
    )
    parser.add_argument(
        "--upload-s3",
        action="store_true",
        help="Upload output files to S3 after generation",
    )
    parser.add_argument(
        "--s3-output-prefix",
        default="gdrive/processed/edge_cases/cptsd",
        metavar="PREFIX",
        help="S3 prefix for uploaded output (default: gdrive/processed/edge_cases/cptsd)",
    )
    parser.add_argument(
        "--max-files",
        type=int,
        default=0,
        metavar="N",
        help="Maximum number of files to process (0 = unlimited, default: 0)",
    )
    parser.add_argument(
        "--max-chunks-per-file",
        type=int,
        default=6,
        metavar="N",
        help="Maximum chunks to extract per file (default: 6)",
    )
    parser.add_argument(
        "--no-voice-profile",
        action="store_true",
        dest="no_voice_profile",
        help="Disable Tim Fletcher voice profile (enabled by default)",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose progress logging",
    )
    return parser


def _load_s3_manifest(manifest_path: Path) -> tuple[str, str, dict | None]:
    """Load S3 manifest and return bucket, endpoint, and manifest dict."""
    if not manifest_path.exists():
        logger.warning(f"Manifest not found: {manifest_path}")
        return "pixel-data", "https://s3.us-east-va.io.cloud.ovh.us", None
    
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    bucket = manifest.get("bucket", "pixel-data")
    endpoint = manifest.get("endpoint", "https://s3.us-east-va.io.cloud.ovh.us")
    return bucket, endpoint, manifest


def main() -> int:
    parser = _build_arg_parser()
    args = parser.parse_args()

    # Configure logging based on verbosity
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s - %(levelname)s - %(message)s" if args.verbose else "%(message)s",
    )

    # Load manifest for S3 config
    bucket, endpoint, manifest = _load_s3_manifest(Path(args.manifest))
    
    # Initialize S3 loader if needed (for S3 sources or uploads)
    loader: S3DatasetLoader | None = None
    needs_s3 = (
        args.upload_s3
        or (args.input_dir and not _is_local_path(args.input_dir))
        or (not args.input_dir and not args.source_key)  # Using default S3 sources
        or any(not _is_local_path(k) for k in args.source_key)
    )
    
    if needs_s3:
        try:
            loader = S3DatasetLoader(bucket=bucket, endpoint_url=endpoint)
        except Exception as e:
            logger.error(f"Failed to initialize S3DatasetLoader: {e}")
            if not args.input_dir or not _is_local_path(args.input_dir):
                return 1
            logger.info("Continuing with local files only")
    
    # Load voice profile for enhanced system prompts
    voice_profile = None
    if not args.no_voice_profile:
        voice_profile = _load_voice_profile()
        if voice_profile:
            logger.info("✓ Loaded Tim Fletcher voice profile")
        else:
            logger.info("Voice profile not found, using standard prompts")
    
    system_prompt = _build_system_prompt(voice_profile)
    
    # Collect source files
    source_files = list(_iter_source_files(
        loader,
        bucket=bucket,
        source_keys=args.source_key,
        input_dir=args.input_dir,
        manifest=manifest,
        prefix=args.prefix,
    ))
    
    if args.max_files > 0:
        source_files = source_files[:args.max_files]
    
    if not source_files:
        logger.error("No source files found to process")
        return 1
    
    logger.info(f"Processing {len(source_files)} transcript file(s)")

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    written = 0
    files_processed = 0
    files_skipped = 0
    chunk_hist: Counter[str] = Counter()
    sources_kept: Counter[str] = Counter()
    last_progress_log = 0

    started_at = datetime.now(timezone.utc).isoformat()

    with out_path.open("w", encoding="utf-8") as f:
        for idx, source_path in enumerate(source_files, 1):
            # Determine display path
            if source_path.startswith("s3://"):
                display_path = source_path
            elif _is_local_path(source_path):
                display_path = source_path
            else:
                display_path = f"s3://{bucket}/{source_path}"
            
            if args.verbose:
                logger.info(f"[{idx}/{len(source_files)}] Processing: {display_path}")
            
            # Load text
            raw = _load_text(loader, source_path, bucket)
            if not raw:
                files_skipped += 1
                continue
            
            cleaned = _clean_text(raw)
            if not cleaned:
                files_skipped += 1
                continue

            chunks = _chunk_text(cleaned)
            if not chunks:
                files_skipped += 1
                continue

            title = _title_from_path(source_path)
            files_processed += 1
            file_written = 0

            chunks_to_use = chunks[:args.max_chunks_per_file]
            total_chunks = len(chunks_to_use)
            
            for chunk_index, chunk in enumerate(chunks_to_use):
                record = {
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {
                            "role": "user",
                            "content": f"Teach me about this CPTSD/complex trauma topic: {title}.",
                        },
                        {"role": "assistant", "content": chunk},
                    ],
                    "metadata": {
                        "source_family": "cptsd",
                        "source_key": display_path,
                        "content_hash": _content_hash(chunk),
                        "chunk_index": chunk_index,
                        "total_chunks": total_chunks,
                        "pii_status": "scrubbed",
                        "license_tag": "transcript_corpus",
                        "split": "train",  # will be re-split during final compilation
                        "phase": "stage6_specialized_domains",
                        "provenance": {
                            "original_source": display_path,
                            "processing_pipeline": "build_cptsd_dataset_from_transcripts",
                            "processed_at": datetime.now(timezone.utc).isoformat(),
                            "dedup_status": "unique",
                            "processing_steps": ["text_clean", "pii_redact", "chunk", "chatml_convert"],
                            "voice_profile_used": voice_profile is not None,
                        },
                    },
                }

                f.write(json.dumps(record, ensure_ascii=False) + "\n")
                written += 1
                file_written += 1

            sources_kept[source_path] = file_written
            chunk_hist[str(min(len(chunks), 25))] += 1
            
            # Progress logging
            if idx - last_progress_log >= PROGRESS_LOG_INTERVAL:
                logger.info(f"  Progress: {idx:,}/{len(source_files):,} files, {written:,} examples written")
                last_progress_log = idx

    # Build stats
    stats = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "started_at": started_at,
        "bucket": bucket,
        "endpoint": endpoint,
        "input_dir": args.input_dir,
        "prefix": args.prefix if not args.input_dir else None,
        "source_keys_provided": len(args.source_key) if args.source_key else 0,
        "output": str(out_path),
        "files_discovered": len(source_files),
        "files_processed": files_processed,
        "files_skipped": files_skipped,
        "examples_written": written,
        "max_chunks_per_file": args.max_chunks_per_file,
        "voice_profile_used": voice_profile is not None,
        "chunk_histogram_capped_25": dict(sorted(chunk_hist.items(), key=lambda x: int(x[0]))),
    }

    stats_path = out_path.with_name("cptsd_transcripts_stats.json")
    stats_path.write_text(json.dumps(stats, indent=2, ensure_ascii=False), encoding="utf-8")

    # Summary
    logger.info("")
    logger.info("=" * 60)
    logger.info(f"✓ Generated {written:,} CPTSD training examples")
    logger.info(f"  Files processed: {files_processed:,}")
    logger.info(f"  Files skipped: {files_skipped:,}")
    logger.info(f"  Voice profile: {'enabled' if voice_profile else 'disabled'}")
    logger.info(f"  Output: {out_path}")
    logger.info(f"  Stats: {stats_path}")

    # Upload to S3 if requested
    if args.upload_s3:
        if loader is None:
            logger.error("Cannot upload to S3: S3DatasetLoader not initialized")
            return 1
            
        logger.info("")
        logger.info("Uploading to S3...")
        
        # Upload main output
        output_s3_key = f"{args.s3_output_prefix}/cptsd_transcripts.jsonl"
        success1 = _upload_to_s3(loader, local_path=out_path, s3_key=output_s3_key, bucket=bucket)
        
        # Upload stats
        stats_s3_key = f"{args.s3_output_prefix}/cptsd_transcripts_stats.json"
        success2 = _upload_to_s3(loader, local_path=stats_path, s3_key=stats_s3_key, bucket=bucket)
        
        if success1 and success2:
            logger.info(f"✓ Uploaded to s3://{bucket}/{args.s3_output_prefix}/")
        else:
            logger.error("Some uploads failed")
            return 1

    logger.info("=" * 60)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
