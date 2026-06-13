#!/usr/bin/env python3
"""YouTube transcript ingestion pipeline for therapeutic AI training.

Reads per-channel transcript directories, converts to JSONL training pairs,
tags German channels, SAFETY FILTERING DISABLED PER USER REQUEST - ALL CONTENT ALLOWED,
and deduplicates against the compiled dataset via SHA-256 hashing.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import re
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from training.provenance import ProvenanceOptions, build_provenance

logger = logging.getLogger("youtube_ingestion")

GERMAN_CHANNELS: frozenset[str] = frozenset(
    {
        "ARTEde",
        "DW Deutsch",
        "Kaltblütig",
        "Klein aber Hannah",
        "SWR Doku",
        "WDR",
        "Y-Kollektiv",
        "ZDF MAGAZIN ROYALE",
        "ZDFheute Nachrichten",
        "rbb Doku",
        "hunds-kompetent Karin Actun",
        "ARTE",
    }
)

DEFAULT_CHUNK_WORDS = 300
DEFAULT_CHUNK_OVERLAP_WORDS = 0
MIN_CHUNK_WORDS = 80
MAX_CHUNK_WORDS = 500
SHORT_PARA_THRESHOLD = 130
LONG_PARA_THRESHOLD = 450
MIN_COMBINED_CHUNK = 100
SEGMENT_OVERLAP = 50
MIN_CHUNKS_FOR_RESCUE = 2

WORD_CHUNK = "word_chunk"
SEMANTIC_CHUNK = "semantic_chunk"
MULTI_PARA_CHUNK = "multi_para_chunk"
SEGMENT_CHUNK = "segment_chunk"


def _content_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _load_compiled_hashes(compiled_path: Path) -> set[str]:
    if not compiled_path.exists():
        logger.warning("Compiled dataset not found at %s — skipping dedup", compiled_path)
        return set()

    hashes: set[str] = set()
    with open(compiled_path, encoding="utf-8") as f:
        for line in f:
            try:
                record = json.loads(line)
                msgs = record.get("messages", [])
                combined = " ".join(m.get("content", "") for m in msgs)
                hashes.add(_content_hash(combined.lower().strip()))
            except (json.JSONDecodeError, KeyError):
                continue
    logger.info("Loaded %d content hashes from compiled dataset", len(hashes))
    return hashes


def _word_count(text: str) -> int:
    return len(re.findall(r"\S+", text))


# Common English abbreviations ending with period that should not trigger sentence splits
_ABBREVIATIONS: frozenset[str] = frozenset({
    "e.g.", "i.e.", "etc.", "vs.", "viz.", "al.",
    "Dr.", "Mr.", "Ms.", "Mrs.", "Prof.", "Sr.", "Jr.",
    "St.", "Ave.", "Blvd.", "Dept.", "Est.",
    "Jan.", "Feb.", "Mar.", "Apr.", "Jun.", "Jul.", "Aug.", "Sep.", "Oct.", "Nov.", "Dec.",
    "approx.", "dept.", "ed.", "esp.", "ex.", "govt.",
    "no.", "vol.", "p.", "pp.",
})


def _split_sentences(text: str) -> list[str]:
    # Split on sentence-ending punctuation followed by whitespace and capital letter,
    # then rejoin fragments that are known abbreviations
    sentences = re.split(r"(?<=[.!?])\s+(?=[A-Z])", text.strip())
    sentences = [s.strip() for s in sentences if s.strip()]
    merged = []
    for s in sentences:
        if merged and any(
            s.lstrip().startswith(abbr[0].upper()) for abbr in _ABBREVIATIONS
        ):
            last_word = merged[-1].split()[-1] if merged[-1].split() else ""
            if last_word.lower().rstrip(".") in {a.rstrip(".").lower() for a in _ABBREVIATIONS}:
                merged[-1] = f"{merged[-1]} {s}"
                continue
        merged.append(s)
    return [m.strip() for m in merged if m.strip()]


def _word_chunks(
    text: str,
    *,
    chunk_words: int = DEFAULT_CHUNK_WORDS,
    overlap_words: int = DEFAULT_CHUNK_OVERLAP_WORDS,
) -> list[dict[str, Any]]:
    """Split a long transcript into fixed-size word windows (fallback for oversized paragraphs)."""
    if chunk_words < MIN_CHUNK_WORDS:
        raise ValueError(f"chunk_words must be at least {MIN_CHUNK_WORDS}")
    if overlap_words < 0 or overlap_words >= chunk_words:
        raise ValueError("overlap_words must be non-negative and smaller than chunk_words")

    words = re.findall(r"\S+", text)
    if not words:
        return []
    if len(words) < MIN_CHUNK_WORDS:
        return [
            {
                "text": " ".join(words),
                "chunk_index": 1,
                "chunk_start_word": 0,
                "chunk_word_count": len(words),
                "pairing_strategy": WORD_CHUNK,
            }
        ]

    step = chunk_words - overlap_words
    chunks: list[dict[str, Any]] = []
    start = 0
    while start < len(words):
        chunk = words[start : start + chunk_words]
        if len(chunk) < MIN_CHUNK_WORDS and chunks:
            chunks[-1]["text"] = f"{chunks[-1]['text']} {' '.join(chunk)}"
            chunks[-1]["chunk_word_count"] += len(chunk)
            break

        chunks.append(
            {
                "text": " ".join(chunk),
                "chunk_index": len(chunks) + 1,
                "chunk_start_word": start,
                "chunk_word_count": len(chunk),
                "pairing_strategy": WORD_CHUNK,
            }
        )
        if start + chunk_words >= len(words):
            break
        start += step

    return chunks


def _append_chunk(
    chunks: list[dict[str, Any]],
    *,
    text: str,
    pairing_strategy: str,
    chunk_start_word: int,
) -> None:
    chunks.append(
        {
            "text": text,
            "chunk_index": len(chunks) + 1,
            "chunk_start_word": chunk_start_word,
            "chunk_word_count": _word_count(text),
            "pairing_strategy": pairing_strategy,
        }
    )


def _hybrid_chunks(
    text: str,
    *,
    chunk_words: int = DEFAULT_CHUNK_WORDS,
    segment_overlap: int = SEGMENT_OVERLAP,
) -> list[dict[str, Any]]:
    """Hybrid multi-paragraph + segmented chunking for maximum recovery.

    Strategy:
    1. Multi-paragraph combine: consecutive short paragraphs (<=130 words)
       are merged into chunks of 100-500 words.
    2. Single-paragraph: medium paragraphs (131-450 words) become one chunk.
    3. Segment: long paragraphs (>450 words) are split on sentence boundaries
       into overlapping segments of ~300 words with 50-word overlap.
    4. Rescue: very short trailing content is appended to previous chunk.

    Returns list of chunk dicts with text, chunk_index, chunk_start_word,
    chunk_word_count, and pairing_strategy.
    """
    if chunk_words < MIN_CHUNK_WORDS:
        raise ValueError(f"chunk_words must be at least {MIN_CHUNK_WORDS}")

    paragraphs = [p.strip() for p in re.split(r"\n{2,}", text) if p.strip()]
    if not paragraphs:
        paragraphs = [text.strip()] if text.strip() else []

    chunks: list[dict[str, Any]] = []
    global_word_offset = 0
    short_buffer: list[str] = []
    buffer_words = 0

    def _flush_buffer() -> None:
        nonlocal buffer_words, global_word_offset
        if not short_buffer:
            return
        merged_text = "\n".join(short_buffer)
        merged_words = _word_count(merged_text)
        _append_chunk(
            chunks,
            text=merged_text,
            pairing_strategy=MULTI_PARA_CHUNK,
            chunk_start_word=global_word_offset,
        )
        global_word_offset += merged_words
        short_buffer.clear()
        buffer_words = 0

    for paragraph in paragraphs:
        para_words = _word_count(paragraph)
        if para_words == 0:
            continue

        # Long paragraph -> segment using sentence boundaries
        if para_words > LONG_PARA_THRESHOLD:
            _flush_buffer()
            sentences = _split_sentences(paragraph)
            if len(sentences) <= 1:
                # No sentence boundaries: word-level fallback
                for sub_chunk in _word_chunks(
                    paragraph,
                    chunk_words=chunk_words,
                    overlap_words=segment_overlap,
                ):
                    sub_chunk["chunk_index"] = len(chunks) + 1
                    sub_chunk["chunk_start_word"] = global_word_offset + sub_chunk["chunk_start_word"]
                    sub_chunk["pairing_strategy"] = SEGMENT_CHUNK
                    chunks.append(sub_chunk)
                global_word_offset += para_words
                continue

            seg_sentences: list[str] = []
            seg_words = 0
            for sentence in sentences:
                s_words = _word_count(sentence)
                if s_words == 0:
                    continue
                if seg_words + s_words > chunk_words and seg_sentences:
                    _append_chunk(
                        chunks,
                        text=" ".join(seg_sentences),
                        pairing_strategy=SEGMENT_CHUNK,
                        chunk_start_word=global_word_offset,
                    )
                    global_word_offset += seg_words
                    # Overlap: keep last sentences for context
                    overlap_sentences = []
                    overlap_words = 0
                    for s in reversed(seg_sentences):
                        sw = _word_count(s)
                        if overlap_words + sw > segment_overlap:
                            break
                        overlap_sentences.insert(0, s)
                        overlap_words += sw
                    seg_sentences = [*overlap_sentences, sentence]
                    seg_words = overlap_words + s_words
                else:
                    seg_sentences.append(sentence)
                    seg_words += s_words

            if seg_sentences:
                _append_chunk(
                    chunks,
                    text=" ".join(seg_sentences),
                    pairing_strategy=SEGMENT_CHUNK,
                    chunk_start_word=global_word_offset,
                )
                global_word_offset += seg_words
            continue

        # Short paragraph -> buffer for multi-paragraph combining
        if para_words <= SHORT_PARA_THRESHOLD:
            short_buffer.append(paragraph)
            buffer_words += para_words
            if buffer_words >= chunk_words:
                _flush_buffer()
            continue

        # Medium paragraph (130-450 words) -> single paragraph chunk
        _flush_buffer()
        _append_chunk(
            chunks,
            text=paragraph,
            pairing_strategy=SEMANTIC_CHUNK,
            chunk_start_word=global_word_offset,
        )
        global_word_offset += para_words

    # Flush any remaining buffer
    _flush_buffer()

    # Rescue: if last chunk is too small, merge with previous
    if len(chunks) >= MIN_CHUNKS_FOR_RESCUE:
        last_chunk = chunks[-1]
        if last_chunk["chunk_word_count"] < MIN_COMBINED_CHUNK:
            prev_chunk = chunks[-2]
            merged_text = f"{prev_chunk['text']}\n{last_chunk['text']}"
            prev_chunk["text"] = merged_text
            prev_chunk["chunk_word_count"] = _word_count(merged_text)
            chunks.pop()

    return chunks


def _transcript_to_pairs(
    text: str,
    channel_name: str,
    *,
    chunk_words: int = DEFAULT_CHUNK_WORDS,
    chunk_overlap_words: int = DEFAULT_CHUNK_OVERLAP_WORDS,
) -> list[dict[str, Any]]:
    """Split a transcript into instruction/output QA pairs."""
    paragraphs = [p.strip() for p in re.split(r"\n{2,}", text) if p.strip()]
    if not paragraphs:
        return []

    pairs: list[dict[str, Any]] = []
    for i in range(0, len(paragraphs) - 1, 2):
        instruction = paragraphs[i]
        output = paragraphs[i + 1] if i + 1 < len(paragraphs) else ""
        if instruction and output:
            pairs.append({"instruction": instruction, "output": output})

    if not pairs and paragraphs:
        transcript_body = "\n\n".join(paragraphs)
        chunks = _hybrid_chunks(
            transcript_body,
            chunk_words=chunk_words,
            segment_overlap=chunk_overlap_words,
        )
        chunk_total = len(chunks)
        for chunk in chunks:
            pairs.append(
                {
                    "instruction": (
                        f"Use this real transcript excerpt from {channel_name} as therapeutic source material."
                    ),
                    "output": chunk["text"],
                    "pairing_strategy": chunk["pairing_strategy"],
                    "chunk_index": chunk["chunk_index"],
                    "chunk_total": chunk_total,
                    "chunk_start_word": chunk["chunk_start_word"],
                    "chunk_word_count": chunk["chunk_word_count"],
                }
            )

    return pairs


def _is_german_channel(channel_name: str, german_channels: frozenset[str]) -> bool:
    return channel_name in german_channels


def ingest_channel(
    channel_dir: Path,
    language: str,
    compiled_hashes: set[str],
    *,
    chunk_words: int = DEFAULT_CHUNK_WORDS,
    chunk_overlap_words: int = DEFAULT_CHUNK_OVERLAP_WORDS,
) -> tuple[list[dict[str, Any]], int, int]:
    """Ingest one channel directory.

    Returns (samples, total_read, skipped_duplicate).
    """
    samples: list[dict[str, Any]] = []
    total_read = 0
    skipped_dup = 0

    if not channel_dir.is_dir():
        return samples, 0, 0

    channel_name = channel_dir.name

    for transcript_file in sorted(channel_dir.glob("*.txt")):
        try:
            text = transcript_file.read_text(encoding="utf-8", errors="replace")
        except OSError as exc:
            logger.warning("Unreadable transcript %s: %s", transcript_file, exc)
            continue

        if not text.strip():
            continue

        pairs = _transcript_to_pairs(
            text,
            channel_name,
            chunk_words=chunk_words,
            chunk_overlap_words=chunk_overlap_words,
        )
        for pair in pairs:
            total_read += 1

            # SAFETY FILTER DISABLED PER USER REQUEST - ALL CONTENT ALLOWED
            # Original safety check removed per user directive:
            # if safety_checker.is_unsafe(full_text, language=language):
            #     skipped_unsafe += 1
            #     continue

            combined_text = f"{pair['instruction']} {pair['output']}"
            content_hash = _content_hash(combined_text.lower().strip())
            if content_hash in compiled_hashes:
                skipped_dup += 1
                continue

            pairing_strategy = pair.get("pairing_strategy", "paragraph_pair")
            provenance_metadata = {
                "channel": channel_name,
                "language": language,
                "transcript_file": transcript_file.name,
                "content_hash": content_hash,
                "pairing_strategy": pairing_strategy,
            }
            if pairing_strategy in {WORD_CHUNK, SEMANTIC_CHUNK}:
                provenance_metadata.update(
                    {
                        "chunk_index": pair["chunk_index"],
                        "chunk_total": pair["chunk_total"],
                        "chunk_start_word": pair["chunk_start_word"],
                        "chunk_word_count": pair["chunk_word_count"],
                    }
                )

            samples.append(
                {
                    "instruction": pair["instruction"],
                    "output": pair["output"],
                    "language": language,
                    "source_channel": channel_name,
                    "provenance": build_provenance(
                        transcript_file.as_posix(),
                        "youtube",
                        options=ProvenanceOptions(
                            license_id="NOASSERTION",
                            transformations=(pairing_strategy, "deduplicated_by_content_hash"),
                        ),
                        metadata=provenance_metadata,
                    ),
                }
            )

    return samples, total_read, skipped_dup


def run_ingestion(args: argparse.Namespace) -> None:
    transcripts_dir = Path(args.transcripts_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    compiled_hashes = _load_compiled_hashes(Path(args.compiled_dataset_dir))
    # safety_checker = MultilingualSafetyChecker  # SAFETY FILTER DISABLED PER USER REQUEST

    german_override = frozenset(args.german_channels.split(",")) if args.german_channels else GERMAN_CHANNELS

    all_samples: list[dict[str, Any]] = []
    channel_stats: list[dict[str, Any]] = []
    total_processed = 0
    total_skipped_dup = 0
    skipped_channels: list[str] = []

    if not transcripts_dir.is_dir():
        logger.error("Transcripts directory not found: %s", transcripts_dir)
        sys.exit(1)

    for channel_dir in sorted(transcripts_dir.iterdir()):
        if not channel_dir.is_dir():
            continue

        channel_name = channel_dir.name
        language = "de" if _is_german_channel(channel_name, german_override) else "en"

        samples, n_read, n_dup = ingest_channel(
            channel_dir,
            language,
            compiled_hashes,
            chunk_words=args.chunk_words,
            chunk_overlap_words=args.chunk_overlap_words,
        )

        if n_read == 0 and not any(channel_dir.glob("*.txt")):
            logger.warning("Channel %s has no transcript files — skipping", channel_name)
            skipped_channels.append(channel_name)
            continue

        all_samples.extend(samples)
        total_processed += n_read
        total_skipped_dup += n_dup

        estimated_tokens = sum(len(s["instruction"].split()) + len(s["output"].split()) for s in samples)
        channel_stats.append(
            {
                "channel": channel_name,
                "path": str(channel_dir),
                "sample_count": len(samples),
                "estimated_tokens": estimated_tokens,
                "language": language,
            }
        )

        channel_output = output_dir / f"{channel_name}.jsonl"
        with open(channel_output, "w", encoding="utf-8") as f:
            for s in samples:
                f.write(json.dumps(s, sort_keys=True) + "\n")

        logger.info(
            "Channel %s: %d samples (%d dup, %d total read), lang=%s",
            channel_name,
            len(samples),
            n_dup,
            n_read,
            language,
        )

    manifest = {
        "generated_at": datetime.now(UTC).isoformat(),
        "channels": channel_stats,
        "totals": {
            "total_samples": len(all_samples),
            "total_channels": len(channel_stats),
            "skipped_channels": len(skipped_channels),
        },
    }
    manifest_path = output_dir / "manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
        f.write("\n")

    report = {
        "generated_at": datetime.now(UTC).isoformat(),
        "processed": total_processed,
        "skipped_duplicate": total_skipped_dup,
        "total_samples": len(all_samples),
        "channels_processed": len(channel_stats),
        "channels_skipped": skipped_channels,
        "chunk_words": args.chunk_words,
        "chunk_overlap_words": args.chunk_overlap_words,
    }
    report_path = output_dir / "processing_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
        f.write("\n")

    logger.info(
        "Ingestion complete: %d samples from %d channels (%d dup)",
        len(all_samples),
        len(channel_stats),
        total_skipped_dup,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Ingest YouTube transcripts into training-ready JSONL.",
    )
    parser.add_argument(
        "--transcripts_dir",
        type=str,
        required=True,
        help="Root directory containing per-channel transcript folders.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Directory for per-channel JSONL outputs, manifest, and report.",
    )
    parser.add_argument(
        "--compiled_dataset_dir",
        type=str,
        required=True,
        help="Path to compiled dataset JSONL for deduplication.",
    )
    parser.add_argument(
        "--german_channels",
        type=str,
        default="",
        help="Comma-separated list of German channel directory names.",
    )
    parser.add_argument(
        "--chunk_words",
        type=int,
        default=DEFAULT_CHUNK_WORDS,
        help="Words per transcript chunk when transcripts do not contain paragraph QA pairs.",
    )
    parser.add_argument(
        "--chunk_overlap_words",
        type=int,
        default=DEFAULT_CHUNK_OVERLAP_WORDS,
        help="Word overlap between transcript chunks.",
    )
    # --safety_checker argument removed per user request - safety filtering disabled
    return parser


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    parser = build_parser()
    args = parser.parse_args()
    run_ingestion(args)


if __name__ == "__main__":
    main()
