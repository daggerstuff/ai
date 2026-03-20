#!/usr/bin/env python3
"""Build a CPTSD-tagged ChatML dataset from CPTSD-focused transcript corpora.

Supports multiple source authors with per-author voice profiles,
CPTSD topic tagging, crisis detection, and semantic chunking.

Outputs:
- ai/training_ready/data/generated/cptsd_transcripts.jsonl
- ai/training_ready/data/generated/cptsd_transcripts_stats.json

Features:
- Streams from S3 or processes local files (memory-efficient)
- Converts transcript text to ChatML format
- Multi-author voice profile support (--voice-profiles)
- CPTSD topic tagging and crisis detection per chunk
- Improved chunking with semantic paragraph boundaries and overlap
- Uploads output directly to S3 with --upload-s3 flag
- Scans directories with --input-dir flag (supports multiple via comma)
- Progress logging for long-running operations

Usage Examples:
    # Process all local CPTSD sources with per-author voice profiles
    python build_cptsd_dataset_from_transcripts.py \
      --input-dir ai/training/ready_packages/data/cptsd_sources/ \
      --voice-profiles ai/training/ready_packages/data/cptsd_voice_profiles.json

    # Extract from default S3 prefix (Tim Fletcher transcripts)
    python build_cptsd_dataset_from_transcripts.py

    # Extract from local directory
    python build_cptsd_dataset_from_transcripts.py \
      --input-dir ~/datasets/gdrive/tier4_voice_persona/Tim\ Fletcher/

Notes:
- Uses ai/training_ready/data/s3_manifest.json as the
  source of truth for bucket/endpoint.
- Does not print transcript content.
- Applies light redaction for obvious PII patterns (emails/phones/urls).
- Uses voice profiles for system prompts when available.
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

from ai.training.ready_packages.utils.s3_dataset_loader import S3DatasetLoader

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
VOICE_PROFILE_PATH = (
    Path(__file__).parents[2]
    / "data"
    / "tim_fletcher_voice"
    / "tim_fletcher_voice_profile.json"
)

# Multi-author voice profiles
VOICE_PROFILES_PATH = (
    Path(__file__).resolve().parent.parent / "data" / "cptsd_voice_profiles.json"
)

logger = logging.getLogger(__name__)

# Topic tagger (lazy import to avoid hard dep)
_topic_tagger = None


def _get_topic_tagger():
    """Lazy-load the topic tagger."""
    global _topic_tagger
    if _topic_tagger is None:
        try:
            from cptsd_topic_tagger import CPTSDTopicTagger

            _topic_tagger = CPTSDTopicTagger()
        except ImportError:
            # Add script dir to path and retry
            import sys

            sys.path.insert(0, str(Path(__file__).resolve().parent))
            from cptsd_topic_tagger import CPTSDTopicTagger

            _topic_tagger = CPTSDTopicTagger()
    return _topic_tagger


def _load_voice_profile() -> dict[str, Any] | None:
    """Load Tim Fletcher voice profile if available."""
    if VOICE_PROFILE_PATH.exists():
        try:
            with open(VOICE_PROFILE_PATH, encoding="utf-8") as f:
                return json.load(f)
        except (OSError, json.JSONDecodeError) as e:
            logger.warning(f"Failed to load voice profile: {e}")
    return None


def _load_multi_voice_profiles(
    path: str | None = None,
) -> dict[str, dict[str, Any]]:
    """Load per-author voice profiles from cptsd_voice_profiles.json.

    Returns a dict keyed by author slug (e.g. 'heidi_priebe').
    """
    profiles_path = Path(path) if path else VOICE_PROFILES_PATH
    if not profiles_path.exists():
        return {}
    try:
        with open(profiles_path, encoding="utf-8") as f:
            data = json.load(f)
        return data.get("voice_profiles", {})
    except (OSError, json.JSONDecodeError) as e:
        logger.warning("Failed to load multi-profiles: %s", e)
        return {}


def _detect_author(source_path: str) -> str:
    """Detect the author from the source file path."""
    lower = source_path.lower()
    if "heidi priebe" in lower or "heidi_priebe" in lower:
        return "heidi_priebe"
    if "patrick teahan" in lower or "patrick_teahan" in lower:
        return "patrick_teahan"
    if "tim fletcher" in lower or "tim_fletcher" in lower:
        return "tim_fletcher"
    return "crappy_childhood_fairy" if "crappy childhood" in lower else "unknown"


def _build_system_prompt_for_author(
    author: str,
    multi_profiles: dict[str, dict[str, Any]],
    fallback_profile: dict[str, Any] | None = None,
) -> str:
    """Build system prompt using the author's voice profile."""
    if not (profile := multi_profiles.get(author)):
        return _build_system_prompt(fallback_profile)

    voice = profile.get("voice_characteristics", {})
    tone = voice.get("tone", "Compassionate, insightful")
    style = voice.get("style", "Educational")
    approach = voice.get("approach", "Grounded, practical")
    name = profile.get("name", author.replace("_", " ").title())
    specialties = profile.get("specialties", [])
    spec_str = ", ".join(specialties[:3]) if specialties else "complex trauma"

    return (
        f"You are a trauma-informed therapeutic AI assistant "
        f"modeled after {name}'s teaching style. "
        f"Specializing in {spec_str}. "
        f"Tone: {tone}. Style: {style}. Approach: {approach}. "
        f"Explain CPTSD concepts clearly, with compassion "
        f"and actionable steps. "
        f"Do not include personal identifying information."
    )


def _build_system_prompt(voice_profile: dict[str, Any] | None = None) -> str:
    """Build a system prompt, optionally enhanced with voice profile."""
    base_prompt = (
        "You are a trauma-informed therapeutic AI assistant "
        "specializing in Complex PTSD (CPTSD) and complex "
        "trauma. Use a grounded, practical tone. "
        "Explain CPTSD concepts clearly, with compassion "
        "and actionable steps. "
        "Do not include personal identifying information."
    )

    if not voice_profile:
        return base_prompt

    # Extract key elements from voice profile
    empathy_markers = voice_profile.get("empathy_markers", {})
    teaching_patterns = voice_profile.get("teaching_patterns", [])
    transition_phrases = voice_profile.get("transition_phrases", {})

    # Build enhanced prompt with voice characteristics
    top_empathy = [
        k for k, _ in sorted(empathy_markers.items(), key=lambda x: -x[1])[:3]
    ]
    top_transitions = [
        k for k, _ in sorted(transition_phrases.items(), key=lambda x: -x[1])[:3]
    ]

    teaching_tips = []
    for p in teaching_patterns[:3]:
        if isinstance(p, dict) and "pattern" in p:
            teaching_tips.append(p["pattern"])
        elif isinstance(p, str):
            teaching_tips.append(p)

    tips_str = ", ".join(teaching_tips) if teaching_tips else "First, Second, Third"
    return (
        f"{base_prompt}\n\n"
        "Voice characteristics:\n"
        f"- Use empathetic phrases like: {', '.join(top_empathy)}\n"
        f"- Use transition phrases: {', '.join(top_transitions)}\n"
        f"- Structure explanations with: {tips_str}\n"
        "- Provide concrete examples and analogies\n"
        "- Acknowledge the difficulty while offering hope "
        "and practical steps"
    )


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


def _chunk_text(
    text: str,
    *,
    max_chars: int = 1800,
    min_chars: int = 400,
    overlap_ratio: float = 0.1,
) -> list[str]:
    """Split text into chunks with semantic paragraph boundaries and overlap.

    Uses paragraph boundaries for clean splits and adds configurable
    overlap between chunks for context preservation.
    """
    paras = [p.strip() for p in text.split("\n\n") if p.strip()]
    if not paras:
        return []

    chunks: list[str] = []
    cur: list[str] = []
    cur_len = 0

    for p in paras:
        if cur_len + len(p) + 2 > max_chars and cur_len >= min_chars:
            chunks.append("\n\n".join(cur).strip())
            # Keep overlap: last N paragraphs worth ~overlap_ratio of max_chars
            overlap_target = int(max_chars * overlap_ratio)
            overlap_paras: list[str] = []
            overlap_len = 0
            for prev_p in reversed(cur):
                if overlap_len + len(prev_p) > overlap_target:
                    break
                overlap_paras.insert(0, prev_p)
                overlap_len += len(prev_p) + 2
            cur = overlap_paras
            cur_len = overlap_len
        cur.append(p)
        cur_len += len(p) + 2

    # Always include the final chunk if it has content
    if cur_len >= min_chars or (cur and not chunks):
        if final := "\n\n".join(cur).strip():
            chunks.append(final)

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


def _list_s3_txt_keys(
    loader: S3DatasetLoader,
    *,
    s3_bucket: str,
    prefix: str,
    format_key: str = "{key}",
) -> list[str]:
    """List .txt object keys from an S3 prefix."""
    files: list[str] = []
    logger.info("Scanning S3 prefix: s3://%s/%s", s3_bucket, prefix)
    try:
        paginator = loader.s3_client.get_paginator("list_objects_v2")
        pages = paginator.paginate(Bucket=s3_bucket, Prefix=prefix)
        for page in pages:
            for obj in page.get("Contents", []):
                key = obj["Key"]
                if key.lower().endswith(".txt"):
                    files.append(format_key.format(key=key))
    except Exception as e:
        logger.error("Failed to list S3 objects: %s", e)
    return files


def _list_txt_files_in_dir(
    loader: S3DatasetLoader | None,
    *,
    bucket: str,
    input_dir: str,
) -> list[str]:
    """List all .txt files in a directory (S3 prefix or local)."""
    if input_dir.startswith("s3://"):
        without_prefix = input_dir[5:]
        parts = without_prefix.split("/", 1)
        s3_bucket = parts[0]
        prefix = parts[1] if len(parts) > 1 else ""
        if loader is None:
            logger.error("S3DatasetLoader required for S3 paths")
            return []
        files = _list_s3_txt_keys(
            loader,
            s3_bucket=s3_bucket,
            prefix=prefix,
            format_key=f"s3://{s3_bucket}/{{key}}",
        )
    elif _is_local_path(input_dir):
        local_dir = Path(input_dir).expanduser()
        if local_dir.is_dir():
            logger.info("Scanning local directory: %s", local_dir)
            files = [str(f) for f in local_dir.rglob("*.txt")]
        else:
            logger.warning("Local directory not found: %s", input_dir)
            files = []
    else:
        if loader is None:
            logger.error("S3DatasetLoader required for S3 paths")
            return []
        files = _list_s3_txt_keys(
            loader,
            s3_bucket=bucket,
            prefix=input_dir,
        )

    logger.info("Found %d .txt files", len(files))
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
                    if (
                        isinstance(k, str)
                        and k.startswith(prefix)
                        and k.lower().endswith(".txt")
                    ):
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
    s3_path = path if path.startswith("s3://") else f"s3://{bucket}/{path}"

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
            yield from _list_txt_files_in_dir(
                loader, bucket=bucket, input_dir=default_prefix
            )


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
        help=(
            "Path to s3_manifest.json for bucket/endpoint config "
            "(default: ai/training_ready/data/s3_manifest.json)"
        ),
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
        help=(
            "S3 key or local path to process (repeatable). "
            "Supports s3://bucket/key, relative S3 keys, "
            "or local paths."
        ),
    )
    parser.add_argument(
        "--input-dir",
        metavar="DIR",
        help=(
            "S3 prefix (s3://bucket/prefix/) or local directory "
            "to scan for all .txt files. Overrides "
            "--source-key and --prefix."
        ),
    )
    parser.add_argument(
        "--output",
        default=str(
            Path(__file__).parents[1] / "data" / "generated" / "cptsd_transcripts.jsonl"
        ),
        metavar="PATH",
        help=(
            "Output JSONL file path "
            "(default: ai/training_ready/data/generated/"
            "cptsd_transcripts.jsonl)"
        ),
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
        help=(
            "S3 prefix for uploaded output (default: gdrive/processed/edge_cases/cptsd)"
        ),
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
        default=12,
        metavar="N",
        help="Maximum chunks to extract per file (default: 12)",
    )
    parser.add_argument(
        "--no-voice-profile",
        action="store_true",
        dest="no_voice_profile",
        help="Disable voice profile enrichment (enabled by default)",
    )
    parser.add_argument(
        "--voice-profiles",
        metavar="PATH",
        help="Path to multi-author voice profiles JSON (cptsd_voice_profiles.json)",
    )
    parser.add_argument(
        "--no-topic-tagging",
        action="store_true",
        dest="no_topic_tagging",
        help="Disable CPTSD topic tagging (enabled by default)",
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


def _init_s3_loader(
    args: argparse.Namespace,
    bucket: str,
    endpoint: str,
) -> S3DatasetLoader | None:
    """Create an S3DatasetLoader if any source or upload needs S3."""
    needs_s3 = (
        args.upload_s3
        or (args.input_dir and not _is_local_path(args.input_dir))
        or (not args.input_dir and not args.source_key)
        or any(not _is_local_path(k) for k in args.source_key)
    )
    if not needs_s3:
        return None
    try:
        return S3DatasetLoader(bucket=bucket, endpoint_url=endpoint)
    except Exception as e:
        logger.error("Failed to initialize S3DatasetLoader: %s", e)
        if not args.input_dir or not _is_local_path(args.input_dir):
            raise
        logger.info("Continuing with local files only")
        return None


def _init_voice_profiles(
    args: argparse.Namespace,
) -> tuple[dict[str, Any] | None, dict[str, dict[str, Any]]]:
    """Load voice profiles and return (fallback, multi)."""
    if args.no_voice_profile:
        return None, {}

    multi_profiles = _load_multi_voice_profiles(getattr(args, "voice_profiles", None))
    if multi_profiles:
        logger.info(
            "✓ Loaded %d author voice profiles: %s",
            len(multi_profiles),
            ", ".join(multi_profiles.keys()),
        )

    voice_profile = _load_voice_profile()
    if voice_profile and not multi_profiles:
        logger.info("✓ Loaded Tim Fletcher voice profile")
    elif not voice_profile and not multi_profiles:
        logger.info("No voice profiles found, using standard prompts")

    return voice_profile, multi_profiles


def _collect_source_files(
    args: argparse.Namespace,
    loader: S3DatasetLoader | None,
    bucket: str,
    manifest: dict | None,
) -> list[str]:
    """Gather all transcript file paths to process."""
    if args.input_dir:
        input_dirs = [d.strip() for d in args.input_dir.split(",") if d.strip()]
        files: list[str] = []
        for d in input_dirs:
            files.extend(_list_txt_files_in_dir(loader, bucket=bucket, input_dir=d))
    else:
        files = list(
            _iter_source_files(
                loader,
                bucket=bucket,
                source_keys=args.source_key,
                input_dir=None,
                manifest=manifest,
                prefix=args.prefix,
            )
        )

    if args.max_files > 0:
        files = files[: args.max_files]
    return files


def _display_path_for(source_path: str, bucket: str) -> str:
    """Build a human-readable display path."""
    if source_path.startswith("s3://") or _is_local_path(source_path):
        return source_path
    return f"s3://{bucket}/{source_path}"


def _build_record(
    *,
    author_prompt: str,
    title: str,
    chunk: str,
    chunk_index: int,
    total_chunks: int,
    display_path: str,
    author: str,
    topic_data: dict,
    multi_profiles: dict[str, dict[str, Any]],
    voice_profile: dict[str, Any] | None,
) -> dict:
    """Build a single ChatML training record."""
    return {
        "messages": [
            {"role": "system", "content": author_prompt},
            {
                "role": "user",
                "content": (
                    f"Teach me about this CPTSD/complex trauma topic: {title}."
                ),
            },
            {"role": "assistant", "content": chunk},
        ],
        "metadata": {
            "source_family": "cptsd",
            "source_author": author,
            "source_key": display_path,
            "content_hash": _content_hash(chunk),
            "chunk_index": chunk_index,
            "total_chunks": total_chunks,
            "pii_status": "scrubbed",
            "license_tag": "transcript_corpus",
            "split": "train",
            "phase": "stage6_specialized_domains",
            "cptsd_topics": topic_data.get("cptsd_topics", []),
            "topic_scores": topic_data.get("topic_scores", {}),
            "crisis_detected": topic_data.get("crisis_detected", False),
            "crisis_severity": topic_data.get("crisis_severity"),
            "is_training_edge_case": topic_data.get("is_training_edge_case", False),
            "provenance": {
                "original_source": display_path,
                "processing_pipeline": ("build_cptsd_dataset_from_transcripts"),
                "processed_at": (datetime.now(timezone.utc).isoformat()),
                "dedup_status": "unique",
                "processing_steps": [
                    "text_clean",
                    "pii_redact",
                    "chunk",
                    "topic_tag",
                    "chatml_convert",
                ],
                "voice_profile_used": (
                    author
                    if multi_profiles.get(author)
                    else (voice_profile is not None)
                ),
            },
        },
    }


def _load_transcript(
    loader: S3DatasetLoader | None,
    source_path: str,
    bucket: str,
) -> list[str] | None:
    """Load, clean, and chunk a transcript. None if unusable."""
    raw = _load_text(loader, source_path, bucket)
    if not raw:
        return None
    cleaned = _clean_text(raw)
    if not cleaned:
        return None
    chunks = _chunk_text(cleaned)
    return chunks or None


def _process_single_file(
    *,
    chunks: list[str],
    source_path: str,
    display_path: str,
    author_prompt: str,
    topic_tagger: Any,
    multi_profiles: dict[str, dict[str, Any]],
    voice_profile: dict[str, Any] | None,
    max_chunks: int,
    out_file: Any,
) -> int:
    """Process one transcript file. Returns records written."""
    title = _title_from_path(source_path)
    author = _detect_author(source_path)
    chunks_to_use = chunks[:max_chunks]
    total_chunks = len(chunks_to_use)
    file_written = 0

    for ci, chunk in enumerate(chunks_to_use):
        td: dict = topic_tagger.tag(chunk) if topic_tagger else {}
        record = _build_record(
            author_prompt=author_prompt,
            title=title,
            chunk=chunk,
            chunk_index=ci,
            total_chunks=total_chunks,
            display_path=display_path,
            author=author,
            topic_data=td,
            multi_profiles=multi_profiles,
            voice_profile=voice_profile,
        )
        out_file.write(json.dumps(record, ensure_ascii=False) + "\n")
        file_written += 1

    return file_written


def _process_files(
    args: argparse.Namespace,
    source_files: list[str],
    loader: S3DatasetLoader | None,
    bucket: str,
    voice_profile: dict[str, Any] | None,
    multi_profiles: dict[str, dict[str, Any]],
    topic_tagger: Any,
    out_path: Path,
) -> tuple[int, int, int, Counter, Counter]:
    """Process transcript files and write ChatML output.

    Returns (written, files_processed, files_skipped,
             chunk_hist, sources_kept).
    """
    default_prompt = _build_system_prompt(voice_profile)
    written = 0
    files_processed = 0
    files_skipped = 0
    chunk_hist: Counter[str] = Counter()
    sources_kept: Counter[str] = Counter()
    last_progress_log = 0

    with out_path.open("w", encoding="utf-8") as f:
        for idx, source_path in enumerate(source_files, 1):
            display_path = _display_path_for(source_path, bucket)

            if args.verbose:
                logger.info(
                    "[%d/%d] Processing: %s",
                    idx,
                    len(source_files),
                    display_path,
                )

            chunks = _load_transcript(loader, source_path, bucket)
            if not chunks:
                files_skipped += 1
                continue

            files_processed += 1
            author = _detect_author(source_path)
            author_prompt = (
                _build_system_prompt_for_author(author, multi_profiles, voice_profile)
                if multi_profiles
                else default_prompt
            )

            file_written = _process_single_file(
                chunks=chunks,
                source_path=source_path,
                display_path=display_path,
                author_prompt=author_prompt,
                topic_tagger=topic_tagger,
                multi_profiles=multi_profiles,
                voice_profile=voice_profile,
                max_chunks=args.max_chunks_per_file,
                out_file=f,
            )

            written += file_written
            sources_kept[source_path] = file_written
            chunk_hist[str(min(len(chunks), 25))] += 1

            if idx - last_progress_log >= PROGRESS_LOG_INTERVAL:
                logger.info(
                    "  Progress: %s/%s files, %s examples written",
                    f"{idx:,}",
                    f"{len(source_files):,}",
                    f"{written:,}",
                )
                last_progress_log = idx

    return (
        written,
        files_processed,
        files_skipped,
        chunk_hist,
        sources_kept,
    )


def _upload_results(
    loader: S3DatasetLoader,
    *,
    out_path: Path,
    stats_path: Path,
    s3_prefix: str,
    bucket: str,
) -> bool:
    """Upload JSONL and stats to S3. Returns True on success."""
    logger.info("")
    logger.info("Uploading to S3...")

    ok1 = _upload_to_s3(
        loader,
        local_path=out_path,
        s3_key=f"{s3_prefix}/cptsd_transcripts.jsonl",
        bucket=bucket,
    )
    ok2 = _upload_to_s3(
        loader,
        local_path=stats_path,
        s3_key=f"{s3_prefix}/cptsd_transcripts_stats.json",
        bucket=bucket,
    )

    if ok1 and ok2:
        logger.info("✓ Uploaded to s3://%s/%s/", bucket, s3_prefix)
        return True
    logger.error("Some uploads failed")
    return False


def main() -> int:
    """Entry point: parse args, process files, write output."""
    parser = _build_arg_parser()
    args = parser.parse_args()

    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format=(
            "%(asctime)s - %(levelname)s - %(message)s"
            if args.verbose
            else "%(message)s"
        ),
    )

    bucket, endpoint, manifest = _load_s3_manifest(Path(args.manifest))

    try:
        loader = _init_s3_loader(args, bucket, endpoint)
    except Exception:
        return 1

    voice_profile, multi_profiles = _init_voice_profiles(args)
    source_files = _collect_source_files(args, loader, bucket, manifest)

    if not source_files:
        logger.error("No source files found to process")
        return 1

    logger.info("Processing %d transcript file(s)", len(source_files))

    topic_tagger = None
    if not getattr(args, "no_topic_tagging", False):
        try:
            topic_tagger = _get_topic_tagger()
            logger.info("✓ CPTSD topic tagger loaded")
        except Exception as e:
            logger.warning("Topic tagger unavailable: %s", e)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    started_at = datetime.now(timezone.utc).isoformat()

    written, files_processed, files_skipped, chunk_hist, _ = _process_files(
        args,
        source_files,
        loader,
        bucket,
        voice_profile,
        multi_profiles,
        topic_tagger,
        out_path,
    )

    stats = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "started_at": started_at,
        "bucket": bucket,
        "endpoint": endpoint,
        "input_dir": args.input_dir,
        "prefix": None if args.input_dir else args.prefix,
        "source_keys_provided": (len(args.source_key) if args.source_key else 0),
        "output": str(out_path),
        "files_discovered": len(source_files),
        "files_processed": files_processed,
        "files_skipped": files_skipped,
        "examples_written": written,
        "max_chunks_per_file": args.max_chunks_per_file,
        "voice_profile_used": voice_profile is not None,
        "chunk_histogram_capped_25": dict(
            sorted(
                chunk_hist.items(),
                key=lambda x: int(x[0]),
            )
        ),
    }

    stats_path = out_path.with_name("cptsd_transcripts_stats.json")
    stats_path.write_text(
        json.dumps(stats, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    logger.info("")
    logger.info("=" * 60)
    logger.info("✓ Generated %s CPTSD training examples", f"{written:,}")
    logger.info("  Files processed: %s", f"{files_processed:,}")
    logger.info("  Files skipped: %s", f"{files_skipped:,}")
    logger.info(
        "  Voice profile: %s",
        "enabled" if voice_profile else "disabled",
    )
    logger.info("  Output: %s", out_path)
    logger.info("  Stats: %s", stats_path)

    if args.upload_s3:
        if loader is None:
            logger.error("Cannot upload: S3DatasetLoader not initialized")
            return 1
        if not _upload_results(
            loader,
            out_path=out_path,
            stats_path=stats_path,
            s3_prefix=args.s3_output_prefix,
            bucket=bucket,
        ):
            return 1

    logger.info("=" * 60)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
