#!/usr/bin/env python3
"""Cross-reference our deduped adapter output against HetznerS3 training data.

Checks how many of our freshly-ingested records already exist in the
existing training datasets on HetznerS3.

Handles multiple HetznerS3 formats:
  - MASTER_TRAINING_SET / stage1: user content is JSON {"conversation": [...]}
  - stage2: user content is JSON with uuid/sessionId (chat log)
  - stage3: user content is JSON with prompt_id/category (scenario)
  - stage4: user content is JSON with tool_result
  - supplementary: plain text user content
  - compiled_dataset: standard ChatML, plain text
  - ULTIMATE_FINAL: no messages array, instructions field
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
from pathlib import Path

HETZNER_DIR = Path("ai/data/raw/hetzner")
OUR_DEDUPED = Path("ai/data/raw/deduped/all_deduped.jsonl")
OUTPUT_DIR = Path("ai/data/raw/deduped")


def _normalize(text: str) -> str:
    return " ".join(text.lower().split())


def _hash_text(*parts: str) -> str:
    """Hash normalized text parts."""
    joined = "\n".join(_normalize(p) for p in parts if p)
    return hashlib.sha256(joined.encode()).hexdigest()


def _hash_our_record(record: dict) -> str:
    """Hash our ChatML record (same as dedup script)."""
    parts: list[str] = []
    for msg in record.get("messages", []):
        role = msg.get("role", "")
        if role == "system":
            continue
        content = msg.get("content", "")
        if isinstance(content, (dict, list)):
            content = json.dumps(content)
        parts.append(f"{role}:{_normalize(content)}")
    return hashlib.sha256("\n".join(parts).encode()).hexdigest()


def _extract_hetzner_text(record: dict) -> str:
    """Extract conversational text from a HetznerS3 record (any format).

    Returns normalized user+assistant text concatenated for hashing.
    """
    # Format 1: messages array (MASTER, stage1, compiled, supplementary)
    if "messages" in record:
        msgs = record["messages"]
        parts: list[str] = []
        for msg in msgs:
            role = msg.get("role", "")
            if role == "system":
                continue
            content = msg.get("content", "")
            if isinstance(content, (dict, list)):
                content_str = json.dumps(content)
                # Try to extract conversation from nested JSON
                if isinstance(content, dict) and "conversation" in content:
                    for turn in content["conversation"]:
                        t_role = turn.get("role", "")
                        t_content = turn.get("content", "")
                        parts.append(f"{t_role}:{_normalize(t_content)}")
                elif isinstance(content, dict) and "instructions" in content:
                    parts.append(f"instructions:{_normalize(content['instructions'])}")
                else:
                    parts.append(f"{role}:{_normalize(content_str)}")
            else:
                parts.append(f"{role}:{_normalize(content)}")
        return hashlib.sha256("\n".join(parts).encode()).hexdigest()

    # Format 2: ULTIMATE_FINAL (no messages, has instructions)
    if "instructions" in record:
        parts = [f"instructions:{_normalize(record['instructions'])}"]
        if record.get("category"):
            parts.append(f"category:{_normalize(record['category'])}")
        return hashlib.sha256("\n".join(parts).encode()).hexdigest()

    # Format 3: any other format — hash all string values
    parts = []
    for key, val in record.items():
        if isinstance(val, str) and len(val) > 20:
            parts.append(f"{key}:{_normalize(val)}")
    return hashlib.sha256("\n".join(parts).encode()).hexdigest()


def _find_hetzner_files() -> list[tuple[str, Path]]:
    """Find all JSONL files in hetzner dir."""
    results: list[tuple[str, Path]] = []
    for root, dirs, files in os.walk(HETZNER_DIR):
        for fname in sorted(files):
            if not fname.endswith(".jsonl"):
                continue
            fpath = Path(root) / fname
            if fpath.stat().st_size == 0:
                continue
            rel = fpath.relative_to(HETZNER_DIR)
            results.append((str(rel), fpath))
    return results


def main() -> None:
    # Step 1: Load our deduped hashes
    print("Loading our deduped records...", file=sys.stderr)
    our_hashes: set[str] = set()
    with OUR_DEDUPED.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            our_hashes.add(_hash_our_record(record))
    print(f"  Our deduped: {len(our_hashes):,} unique hashes", file=sys.stderr)

    # Step 2: Hash all HetznerS3 records
    hetzner_files = _find_hetzner_files()
    hetzner_hashes: set[str] = set()
    hetzner_stats: dict[str, dict] = {}

    print(f"\nHashing {len(hetzner_files)} HetznerS3 files...", file=sys.stderr)
    for rel_path, fpath in hetzner_files:
        count = 0
        try:
            with fpath.open(encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        record = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    h = _extract_hetzner_text(record)
                    hetzner_hashes.add(h)
                    count += 1
        except Exception as e:
            print(f"  ERROR reading {rel_path}: {e}", file=sys.stderr)

        hetzner_stats[rel_path] = {"records": count}
        print(f"  {rel_path}: {count:,} records", file=sys.stderr)

    print(f"\n  HetznerS3 total: {len(hetzner_hashes):,} unique hashes", file=sys.stderr)

    # Step 3: Cross-reference
    overlap = our_hashes & hetzner_hashes
    only_ours = our_hashes - hetzner_hashes
    only_hetzner = hetzner_hashes - our_hashes

    print("\nCross-reference results:")
    print(f"  Our unique:     {len(our_hashes):>10,}")
    print(f"  HetznerS3 unique: {len(hetzner_hashes):>10,}")
    print(f"  Overlap:          {len(overlap):>10,} ({len(overlap) / len(our_hashes) * 100:.1f}% of ours)")
    print(f"  Only in ours:     {len(only_ours):>10,}")
    print(f"  Only in Hetzner:  {len(only_hetzner):>10,}")

    # Write report
    report_path = OUTPUT_DIR / "hetzner_overlap_report.txt"
    with report_path.open("w", encoding="utf-8") as f:
        f.write("HetznerS3 Overlap Report\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Our deduped records:    {len(our_hashes):>10,}\n")
        f.write(f"HetznerS3 unique hashes:  {len(hetzner_hashes):>10,}\n")
        f.write(f"Overlap (in both):       {len(overlap):>10,}\n")
        f.write(f"  % of ours in Hetzner:   {len(overlap) / len(our_hashes) * 100:.1f}%\n")
        f.write(f"Only in ours (new):       {len(only_ours):>10,}\n")
        f.write(f"Only in Hetzner:          {len(only_hetzner):>10,}\n\n")
        f.write("HetznerS3 files processed:\n")
        for rel_path, stats in sorted(hetzner_stats.items()):
            f.write(f"  {rel_path:60s} {stats['records']:>10,} records\n")

    print(f"\nReport: {report_path}")


if __name__ == "__main__":
    main()
