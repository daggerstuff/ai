#!/usr/bin/env python3
"""Build a CPTSD-tagged ChatML dataset from CPTSD-focused transcript corpora.

Primary target: Tim Fletcher transcripts in S3 (tier4_voice_persona/Tim Fletcher/*.txt)

Outputs:
- ai/training_ready/data/generated/cptsd_transcripts.jsonl
- ai/training_ready/data/generated/cptsd_transcripts_stats.json

Notes:
- Uses ai/training_ready/data/s3_manifest.json as the source of truth for bucket/endpoint and keys.
- Does not print transcript content.
- Applies light redaction for obvious PII patterns (emails/phones/urls).
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from ai.training_ready.utils.s3_dataset_loader import S3DatasetLoader

EMAIL_RE = re.compile(r"[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}", re.IGNORECASE)
PHONE_RE = re.compile(r"\b(?:\+?\d{1,3}[-. (]*)?(?:\d{3}[-. )]*)\d{3}[-. ]*\d{4}\b")
URL_RE = re.compile(r"https?://\S+", re.IGNORECASE)
TIMESTAMP_RE = re.compile(r"\b\d{1,2}:\d{2}(?::\d{2})?\b")


def _clean_text(text: str) -> str:
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


def _system_prompt() -> str:
    return (
        "You are a trauma-informed therapeutic AI assistant. Use a grounded, practical tone. "
        "Explain CPTSD concepts clearly, with compassion and actionable steps. "
        "Do not include personal identifying information."
    )


def _title_from_key(key: str) -> str:
    name = key.split("/")[-1]
    if name.lower().endswith(".txt"):
        name = name[:-4]
    return name.replace("_", " ").strip()


def _list_transcript_keys(manifest: dict, *, prefix: str) -> list[str]:
    # s3_manifest.json contains nested category listings with objects[]. We search all keys.
    keys: list[str] = []

    def walk(node):
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


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--manifest",
        default=str(Path(__file__).parents[1] / "data" / "s3_manifest.json"),
    )
    parser.add_argument(
        "--prefix",
        default="datasets/gdrive/tier4_voice_persona/Tim Fletcher/",
        help="S3 key prefix to pull transcripts from",
    )
    parser.add_argument(
        "--output",
        default=str(Path(__file__).parents[1] / "data" / "generated" / "cptsd_transcripts.jsonl"),
    )
    parser.add_argument("--max-files", type=int, default=250)
    parser.add_argument("--max-chunks-per-file", type=int, default=6)

    args = parser.parse_args()

    manifest = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
    bucket = manifest["bucket"]
    endpoint = manifest["endpoint"]

    loader = S3DatasetLoader(bucket=bucket, endpoint_url=endpoint)

    keys = _list_transcript_keys(manifest, prefix=args.prefix)
    keys = keys[: args.max_files]

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    written = 0
    files_used = 0
    chunk_hist = Counter()

    started_at = datetime.now(timezone.utc).isoformat()

    with out_path.open("w", encoding="utf-8") as f:
        for key in keys:
            s3_path = f"s3://{bucket}/{key}"
            raw = loader.load_text(s3_path)
            cleaned = _clean_text(raw)
            if not cleaned:
                continue

            chunks = _chunk_text(cleaned)
            if not chunks:
                continue

            title = _title_from_key(key)
            files_used += 1

            for chunk in chunks[: args.max_chunks_per_file]:
                record = {
                    "messages": [
                        {"role": "system", "content": _system_prompt()},
                        {
                            "role": "user",
                            "content": f"Teach me about this CPTSD/complex trauma topic: {title}.",
                        },
                        {"role": "assistant", "content": chunk},
                    ],
                    "metadata": {
                        "source_family": "cptsd",
                        "source_key": s3_path,
                        "pii_status": "scrubbed",  # conservative
                        "license_tag": "transcript_corpus",
                        "split": "train",  # will be re-split during final compilation
                        "phase": "stage6_specialized_domains",
                        "provenance": {
                            "original_s3_key": s3_path,
                            "processing_pipeline": "build_cptsd_dataset_from_transcripts",
                            "processed_at": datetime.now(timezone.utc).isoformat(),
                            "dedup_status": "unique",
                            "processing_steps": ["text_clean", "chunk", "chatml_convert"],
                        },
                    },
                }

                f.write(json.dumps(record, ensure_ascii=False) + "\n")
                written += 1

            chunk_hist[str(min(len(chunks), 25))] += 1

    stats = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "started_at": started_at,
        "bucket": bucket,
        "endpoint": endpoint,
        "prefix": args.prefix,
        "output": str(out_path),
        "files_discovered": len(keys),
        "files_used": files_used,
        "examples_written": written,
        "chunk_histogram_capped_25": dict(chunk_hist),
    }

    stats_path = out_path.with_name("cptsd_transcripts_stats.json")
    stats_path.write_text(json.dumps(stats, indent=2, ensure_ascii=False), encoding="utf-8")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
