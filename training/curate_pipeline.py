#!/usr/bin/env python3
"""
Curated Dataset Pipeline
=========================
Converts raw desloped JSONL (849K records) into tiered, split, training-ready datasets.

Input:  ai/data/raw/deduped/all_desloped.jsonl
Output: ai/data/curated/
  sft_chatml/{train,val,test}.jsonl  — OpenAI chat format, all tiers balanced
  sft_alpaca/{train,val,test}.jsonl      — Alpaca instruction format, 3-msg records
  sft_alpaca_chatml/{train,val,test}.jsonl — ChatML format of the same 3-msg records
  safety/{train,val,test}.jsonl      — Adversarial safety records
  stats.json                         — Distribution statistics

Quality Tiers:
  T1_GOLD   — clinically_reviewed=True or mi_quality=high (~133 records)
  T2_SILVER — Multi-turn therapy dialogues (5+ msgs), non-adversarial (~65K)
  T3_BRONZE — 3-msg classification, not clinically reviewed (downsampled ~785K→~120K)
  T4_SAFETY — task_type=adversarial_safety (~51K)

Balancing:
  T1_GOLD:   keep 100%
  T2_SILVER: keep 100%
  T3_BRONZE: downsample reddit_mental_nlp to ~100K (1-in-6 deterministic)
  T4_SAFETY: keep 100%

Usage:
  uv run python ai/training/curate_pipeline.py [--input PATH] [--output DIR] [--dry-run]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DEFAULT_INPUT = "ai/data/raw/deduped/all_desloped.jsonl"
DEFAULT_OUTPUT = "ai/data/curated"

# System prompts per task type — preserves source/clinical context
SYSTEM_PROMPTS: dict[str, str] = {
    "therapy_response_generation": (
        "You are Pixelated Empathy, an evidence-based clinical AI assistant trained in "
        "motivational interviewing, CBT, DBT, and ACT. Respond therapeutically with "
        "empathic reflection, open questions, and validation. Never give medical advice."
    ),
    "symptom_classification": (
        "You are Pixelated Empathy, a clinical AI assistant. Classify the user's "
        "symptoms and respond with supportive guidance. Do not diagnose — encourage "
        "professional evaluation when indicated."
    ),
    "severity_estimation": (
        "You are Pixelated Empathy, a clinical AI assistant. Estimate symptom "
        "severity based on standardized clinical scales (PHQ-9, GAD-7, BDI, etc.) "
        "and provide appropriate supportive guidance."
    ),
    "risk_assessment": (
        "You are Pixelated Empathy, a clinical AI assistant. Assess risk factors "
        "in the user's message and respond with appropriate safety guidance. "
        "If imminent risk is detected, direct the user to emergency services."
    ),
    "adversarial_safety": (
        "You are a clinical safety assistant. Evaluate the input for risk factors "
        "and respond with appropriate safety guidance. If imminent risk is detected, "
        "direct the user to emergency services."
    ),
}

DEFAULT_SYSTEM_PROMPT = (
    "You are Pixelated Empathy, an evidence-based clinical AI assistant. "
    "Your responses must be empathetic, validating, and grounded in established "
    "therapeutic modalities (such as CBT, DBT, or ACT)."
)

SAFETY_SYSTEM_PROMPT = SYSTEM_PROMPTS["adversarial_safety"]

# Split ratios (train/val/test)
SPLIT_RATIOS = {"train": 0.70, "val": 0.15, "test": 0.15}

# Downsampling: source -> keep fraction (deterministic hash-based)
DOWNSAMPLE_RATES: dict[str, float] = {
    "reddit_mental_nlp": 0.17,  # 597K → ~100K
    "reddit_mental_health_posts": 0.50,  # 88K → ~44K
}

# Sources that are multi-turn therapy dialogues (T2_SILVER candidates)
MULTI_TURN_SOURCES = {
    "annomi",
    "empath",
    "esconv",
    "psy_insight",
    "kokorochat",
    "hope",
    "daic_woz",
    "mental_health_multiagent",
    "psycheval",
    "counseling_conversations",
}

# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class TierStats:
    tier: str
    source_counts: Counter = field(default_factory=Counter)
    task_type_counts: Counter = field(default_factory=Counter)
    total: int = 0
    included: int = 0
    excluded: int = 0


@dataclass
class PipelineStats:
    total_read: int = 0
    total_included: int = 0
    total_excluded: int = 0
    total_deduped: int = 0
    tier_stats: dict[str, TierStats] = field(default_factory=dict)
    split_counts: dict[str, Counter] = field(default_factory=dict)

    def get_or_create_tier(self, tier: str) -> TierStats:
        if tier not in self.tier_stats:
            self.tier_stats[tier] = TierStats(tier=tier)
        return self.tier_stats[tier]

    def to_dict(self) -> dict[str, Any]:
        return {
            "total_read": self.total_read,
            "total_included": self.total_included,
            "total_excluded": self.total_excluded,
            "total_deduped": self.total_deduped,
            "tier_stats": {
                t: {
                    "total": ts.total,
                    "included": ts.included,
                    "excluded": ts.excluded,
                    "source_counts": dict(ts.source_counts.most_common()),
                    "task_type_counts": dict(ts.task_type_counts.most_common()),
                }
                for t, ts in self.tier_stats.items()
            },
            "split_counts": {k: dict(v) for k, v in self.split_counts.items()},
        }


# ---------------------------------------------------------------------------
# Tier classification
# ---------------------------------------------------------------------------


def classify_tier(record: dict[str, Any]) -> str:
    """Classify a record into a quality tier."""
    task_type = record.get("task_type", "")
    clinical_reviewed = record.get("clinical_reviewed", False)
    mi_quality = record.get("mi_quality", "")
    source = record.get("source", "")
    messages = record.get("messages", [])

    # T4_SAFETY: adversarial safety records
    if task_type == "adversarial_safety":
        return "T4_SAFETY"

    # T1_GOLD: clinically reviewed or high MI quality
    if clinical_reviewed or mi_quality == "high":
        return "T1_GOLD"

    # T2_SILVER: multi-turn therapy dialogues (5+ messages, non-adversarial)
    if len(messages) >= 5 and source in MULTI_TURN_SOURCES:
        return "T2_SILVER"

    # T3_BRONZE: everything else (mostly 3-msg classification)
    return "T3_BRONZE"


# ---------------------------------------------------------------------------
# Deduplication
# ---------------------------------------------------------------------------


def content_hash(record: dict[str, Any]) -> str:
    """Hash the conversation content for deduplication."""
    messages = record.get("messages", [])
    text_blob = "||".join(f"{m.get('role', '')}:{m.get('content', '')}" for m in messages)
    return hashlib.md5(text_blob.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# Split assignment
# ---------------------------------------------------------------------------


def assign_split(hash_val: str) -> str:
    """Deterministic split assignment based on content hash."""
    bucket = int(hash_val[:8], 16) % 100
    if bucket < 70:
        return "train"
    elif bucket < 85:
        return "val"
    else:
        return "test"


# ---------------------------------------------------------------------------
# Downsampling
# ---------------------------------------------------------------------------


def should_keep(source: str, hash_val: str) -> bool:
    """Deterministic downsampling for overrepresented sources."""
    rate = DOWNSAMPLE_RATES.get(source, 1.0)
    if rate >= 1.0:
        return True
    threshold = int(rate * 10000)
    bucket = int(hash_val[:8], 16) % 10000
    return bucket < threshold


# ---------------------------------------------------------------------------
# Format conversion
# ---------------------------------------------------------------------------


def to_chatml(record: dict[str, Any], tier: str, system_prompt: str | None = None) -> dict[str, Any]:
    """Convert record to ChatML format with metadata."""
    messages = record.get("messages", [])

    # Ensure system prompt is present
    if system_prompt:
        if not messages or messages[0].get("role") != "system":
            messages = [{"role": "system", "content": system_prompt}] + messages
        else:
            # Replace existing system prompt
            messages[0] = {"role": "system", "content": system_prompt}

    return {
        "messages": messages,
        "source": record.get("source", ""),
        "task_type": record.get("task_type", ""),
        "tier": tier,
        "diagnostic_tag": record.get("diagnostic_tag"),
        "demographic_tags": record.get("demographic_tags", []),
        "linguistic_style": record.get("linguistic_style"),
        "clinical_reviewed": record.get("clinical_reviewed", False),
        "mi_quality": record.get("mi_quality", ""),
    }


def to_alpaca(record: dict[str, Any], tier: str) -> dict[str, Any] | None:
    """Convert single-turn (3-msg) record to Alpaca instruction format."""
    messages = record.get("messages", [])
    non_system = [m for m in messages if m.get("role") != "system"]
    if len(non_system) != 2:
        return None
    if non_system[0].get("role") != "user" or non_system[1].get("role") != "assistant":
        return None

    system_msg = next((m.get("content", "") for m in messages if m.get("role") == "system"), "")
    user_msg = non_system[0].get("content", "")
    assistant_msg = non_system[1].get("content", "")

    if not user_msg.strip() or not assistant_msg.strip():
        return None

    return {
        "instruction": system_msg or "Respond to the following mental health query.",
        "input": user_msg,
        "output": assistant_msg,
        "task_type": record.get("task_type", ""),
        "source": record.get("source", ""),
        "tier": tier,
        "diagnostic_tag": record.get("diagnostic_tag"),
        "demographic_tags": record.get("demographic_tags", []),
        "linguistic_style": record.get("linguistic_style"),
        "clinical_reviewed": record.get("clinical_reviewed", False),
    }


def to_chatml_from_alpaca(alpaca_record: dict[str, Any]) -> dict[str, Any]:
    """Convert an Alpaca instruction-format record to ChatML messages format."""
    messages = [
        {"role": "system", "content": alpaca_record["instruction"]},
        {"role": "user", "content": alpaca_record["input"]},
        {"role": "assistant", "content": alpaca_record["output"]},
    ]
    return {
        "messages": messages,
        "source": alpaca_record.get("source", ""),
        "task_type": alpaca_record.get("task_type", ""),
        "tier": alpaca_record.get("tier", ""),
        "diagnostic_tag": alpaca_record.get("diagnostic_tag"),
        "demographic_tags": alpaca_record.get("demographic_tags", []),
        "linguistic_style": alpaca_record.get("linguistic_style"),
        "clinical_reviewed": alpaca_record.get("clinical_reviewed", False),
    }


def to_safety(record: dict[str, Any], tier: str) -> dict[str, Any]:
    """Convert adversarial record to safety training format."""
    messages = record.get("messages", [])

    # Ensure system prompt is present
    if not messages or messages[0].get("role") != "system":
        messages = [{"role": "system", "content": SAFETY_SYSTEM_PROMPT}] + messages
    else:
        messages[0] = {"role": "system", "content": SAFETY_SYSTEM_PROMPT}

    return {
        "messages": messages,
        "source": record.get("source", ""),
        "task_type": record.get("task_type", "adversarial_safety"),
        "tier": tier,
        "is_safe": True,  # All included records passed deslop — safe responses
        "diagnostic_tag": record.get("diagnostic_tag"),
    }


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------


def run_pipeline(
    input_path: str,
    output_dir: str,
    dry_run: bool = False,
) -> PipelineStats:
    """Run the full curation pipeline."""
    stats = PipelineStats()
    seen_hashes: set[str] = set()

    # Prepare output directories
    if not dry_run:
        for subdir in ["sft_chatml", "sft_alpaca", "sft_alpaca_chatml", "safety"]:
            d = Path(output_dir) / subdir
            d.mkdir(parents=True, exist_ok=True)

    # Open output file handles
    handles: dict[str, dict[str, Any]] = {}
    if not dry_run:
        for subdir in ["sft_chatml", "sft_alpaca", "sft_alpaca_chatml", "safety"]:
            handles[subdir] = {}
            for split in ["train", "val", "test"]:
                path = Path(output_dir) / subdir / f"{split}.jsonl"
                handles[subdir][split] = open(path, "w", encoding="utf-8")

    # Initialize split counters
    for subdir in ["sft_chatml", "sft_alpaca", "sft_alpaca_chatml", "safety"]:
        stats.split_counts[subdir] = Counter()

    print(f"Streaming {input_path}...", file=sys.stderr)
    if dry_run:
        print("DRY RUN — no files will be written", file=sys.stderr)

    with open(input_path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f):
            if not line.strip():
                continue

            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue

            stats.total_read += 1

            # Classify tier
            tier = classify_tier(record)
            tier_stats = stats.get_or_create_tier(tier)
            tier_stats.total += 1
            tier_stats.source_counts[record.get("source", "unknown")] += 1
            tier_stats.task_type_counts[record.get("task_type", "unknown")] += 1

            # Dedup check
            chash = content_hash(record)
            if chash in seen_hashes:
                stats.total_deduped += 1
                tier_stats.excluded += 1
                continue
            seen_hashes.add(chash)

            # Downsampling for T3_BRONZE
            if tier == "T3_BRONZE":
                source = record.get("source", "")
                if not should_keep(source, chash):
                    tier_stats.excluded += 1
                    stats.total_excluded += 1
                    continue

            # Determine split
            split = assign_split(chash)

            # Write to appropriate format(s)
            if dry_run:
                tier_stats.included += 1
                stats.total_included += 1
                continue

            # T4_SAFETY → safety format
            if tier == "T4_SAFETY":
                safety_record = to_safety(record, tier)
                handles["safety"][split].write(json.dumps(safety_record, ensure_ascii=False) + "\n")
                stats.split_counts["safety"][split] += 1

            # All non-safety tiers → ChatML SFT format (T4_SAFETY goes only to safety/)
            if tier != "T4_SAFETY":
                task_type = record.get("task_type", "")
                sys_prompt = SYSTEM_PROMPTS.get(task_type, DEFAULT_SYSTEM_PROMPT)
                chatml_record = to_chatml(record, tier, sys_prompt)
                handles["sft_chatml"][split].write(json.dumps(chatml_record, ensure_ascii=False) + "\n")
                stats.split_counts["sft_chatml"][split] += 1

            # T3_BRONZE 3-msg records → Alpaca instruction + ChatML format
            if tier == "T3_BRONZE":
                alpaca_record = to_alpaca(record, tier)
                if alpaca_record:
                    handles["sft_alpaca"][split].write(json.dumps(alpaca_record, ensure_ascii=False) + "\n")
                    stats.split_counts["sft_alpaca"][split] += 1
                    # Also write ChatML version of the same record
                    alpaca_chatml = to_chatml_from_alpaca(alpaca_record)
                    handles["sft_alpaca_chatml"][split].write(json.dumps(alpaca_chatml, ensure_ascii=False) + "\n")
                    stats.split_counts["sft_alpaca_chatml"][split] += 1

            # T1_GOLD and T2_SILVER 3-msg records → Alpaca + ChatML
            if tier in ("T1_GOLD", "T2_SILVER"):
                alpaca_record = to_alpaca(record, tier)
                if alpaca_record:
                    handles["sft_alpaca"][split].write(json.dumps(alpaca_record, ensure_ascii=False) + "\n")
                    stats.split_counts["sft_alpaca"][split] += 1
                    # Also write ChatML version of the same record
                    alpaca_chatml = to_chatml_from_alpaca(alpaca_record)
                    handles["sft_alpaca_chatml"][split].write(json.dumps(alpaca_chatml, ensure_ascii=False) + "\n")
                    stats.split_counts["sft_alpaca_chatml"][split] += 1

            tier_stats.included += 1
            stats.total_included += 1

            # Progress
            if (line_num + 1) % 100000 == 0:
                print(
                    f"  Processed {line_num + 1:,} records | "
                    f"included={stats.total_included:,} | "
                    f"excluded={stats.total_excluded:,} | "
                    f"deduped={stats.total_deduped:,}",
                    file=sys.stderr,
                )

    # Close file handles
    if not dry_run:
        for subdir in handles:
            for split in handles[subdir]:
                handles[subdir][split].close()

    # Write stats
    if not dry_run:
        stats_path = Path(output_dir) / "stats.json"
        with open(stats_path, "w", encoding="utf-8") as f:
            json.dump(stats.to_dict(), f, indent=2, ensure_ascii=False)

    return stats


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description="Curated dataset pipeline")
    parser.add_argument("--input", default=DEFAULT_INPUT, help="Input JSONL path")
    parser.add_argument("--output", default=DEFAULT_OUTPUT, help="Output directory")
    parser.add_argument("--dry-run", action="store_true", help="Don't write files, just report stats")
    args = parser.parse_args()

    print(f"Input:  {args.input}", file=sys.stderr)
    print(f"Output: {args.output}", file=sys.stderr)
    print(f"Dry run: {args.dry_run}", file=sys.stderr)
    print(file=sys.stderr)

    stats = run_pipeline(args.input, args.output, args.dry_run)

    # Summary
    print("\n" + "=" * 60, file=sys.stderr)
    print("CURATION PIPELINE COMPLETE", file=sys.stderr)
    print("=" * 60, file=sys.stderr)
    print(f"Total read:     {stats.total_read:>10,}", file=sys.stderr)
    print(f"Total included: {stats.total_included:>10,}", file=sys.stderr)
    print(f"Total excluded: {stats.total_excluded:>10,}", file=sys.stderr)
    print(f"Total deduped:  {stats.total_deduped:>10,}", file=sys.stderr)
    print(file=sys.stderr)

    for tier_name in ["T1_GOLD", "T2_SILVER", "T3_BRONZE", "T4_SAFETY"]:
        ts = stats.tier_stats.get(tier_name)
        if ts:
            print(f"  {tier_name}: {ts.included:>8,} included / {ts.total:>8,} total", file=sys.stderr)
            top_sources = ts.source_counts.most_common(5)
            for src, cnt in top_sources:
                print(f"    {src}: {cnt:,}", file=sys.stderr)
            print(file=sys.stderr)

    if not args.dry_run:
        print("Split counts:", file=sys.stderr)
        for subdir, counts in stats.split_counts.items():
            print(f"  {subdir}: {dict(counts)}", file=sys.stderr)


if __name__ == "__main__":
    main()
