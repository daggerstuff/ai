#!/usr/bin/env python3
"""Data audit script for training pipeline inventory.

Scans input directories for JSONL files, categorizes samples by therapeutic
domain, and flags categories below a configurable threshold.
"""

from __future__ import annotations

import argparse
import json
import logging
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path

logger = logging.getLogger("data_audit")

CATEGORY_KEYWORDS: dict[str, list[str]] = {
    "dissociation": ["dissociation"],
    "somatic_therapy": ["somatic"],
    "attachment_disorders": ["attachment"],
    "complicated_grief": ["grief"],
    "eating_disorders": ["eating_disorder"],
    "ocd_intrusive_thoughts": ["ocd", "intrusive"],
    "neurodivergent": ["neurodivergent", "autism", "adhd"],
    "cultural_religious": ["cultural", "religious"],
    "addiction": ["addiction"],
    "personality_disorders": ["personality_disorders", "narcissistic", "narcissist"],
    "cptsd_trauma": ["cptsd", "trauma", "ptsd"],
    "crisis_edge_cases": ["crisis", "edge_case", "nightmare", "stress_test", "edge_crisis"],
    "long_running_therapy": ["long_session", "long_running", "continuity", "long_sessions"],
    "voice_persona": ["voice_persona", "tim_fletcher", "teahan"],
    "roleplay_simulation": ["roleplay", "simulation", "seed_simulation"],
    "dpo_preference": ["dpo", "preference"],
    "cot_reasoning": ["cot", "reasoning"],
    "therapeutic_expertise": ["therapeutic_expertise", "specialized"],
    "safety_guardrails": ["safety", "guardrail", "benchmark"],
    "clinical_literature": ["clinical_literature", "dsm", "knowledge_base"],
    "video_transcripts": ["youtube", "video_transcript", "transcript", "channel"],
    "general_counseling": ["foundation", "counseling", "mental_health", "counseling_conversations"],
}

BROAD_FALLBACK_KEYWORDS: dict[str, list[str]] = {
    "voice_persona": ["voice", "persona"],
    "personality_disorders": ["personality", "disorders"],
    "therapeutic_expertise": ["therapeutic", "expertise"],
    "clinical_literature": ["clinical", "book", "literature"],
}


def _classify_file(filename: str) -> str:
    name_lower = filename.lower()
    for cat, keywords in CATEGORY_KEYWORDS.items():
        if any(kw in name_lower for kw in keywords):
            return cat
    for cat, keywords in BROAD_FALLBACK_KEYWORDS.items():
        if any(kw in name_lower for kw in keywords):
            return cat
    return "uncategorized"


def _extract_categories_from_jsonl(path: Path) -> dict[str, int]:
    """Read JSONL file and count records by their 'category' field."""
    counts: dict[str, int] = {}
    try:
        with open(path, encoding="utf-8", errors="replace") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue
                cat = record.get("category") or record.get("metadata", {}).get("category") or "uncategorized"
                if not cat:
                    cat = "uncategorized"
                counts[cat] = counts.get(cat, 0) + 1
    except OSError:
        logger.warning("Cannot read %s", path)
    return counts


def count_lines(path: Path) -> int:
    count = 0
    try:
        with open(path, encoding="utf-8", errors="replace") as f:
            for _ in f:
                count += 1
    except OSError:
        logger.warning("Cannot read %s", path)
    return count


def run_audit(args: argparse.Namespace) -> None:
    threshold = args.threshold
    input_dirs = [Path(d) for d in args.input_dirs]

    category_counts: dict[str, int] = defaultdict(int)
    category_files: dict[str, list[dict]] = defaultdict(list)
    total_files = 0
    total_samples = 0

    for input_dir in input_dirs:
        if not input_dir.exists():
            logger.warning("Directory not found: %s", input_dir)
            continue
        for jsonl_file in input_dir.rglob("*.jsonl"):
            content_cats = _extract_categories_from_jsonl(jsonl_file)
            n = count_lines(jsonl_file)
            if n == 0:
                continue
            # Prefer content-based categories; fallback to filename classification
            if content_cats:
                for cat, c in content_cats.items():
                    category_counts[cat] += c
                dominant = max(content_cats, key=lambda c: content_cats[c])
                category_files[dominant].append({"path": str(jsonl_file), "samples": n})
            else:
                cat = _classify_file(jsonl_file.name)
                category_counts[cat] += n
                category_files[cat].append({"path": str(jsonl_file), "samples": n})
            total_files += 1
            total_samples += n

    categories = {}
    for cat in sorted(set(list(CATEGORY_KEYWORDS.keys()) + list(category_counts.keys()))):
        c = category_counts.get(cat, 0)
        status = "covered" if c >= threshold else ("partial" if c > 0 else "missing")
        if status in {"missing", "partial"}:
            logger.warning(
                "Category %s: %d samples (below threshold of %d) — status: %s",
                cat, c, threshold, status,
            )
        categories[cat] = {
            "name": cat,
            "sample_count": c,
            "status": status,
            "threshold": threshold,
            "files": category_files.get(cat, []),
        }

    covered = sum(1 for v in categories.values() if v["status"] == "covered")
    partial = sum(1 for v in categories.values() if v["status"] == "partial")
    missing = sum(1 for v in categories.values() if v["status"] == "missing")

    report = {
        "generated_at": datetime.now(UTC).isoformat(),
        "threshold": threshold,
        "input_dirs": [str(d) for d in input_dirs],
        "total_files_scanned": total_files,
        "total_samples": total_samples,
        "categories": categories,
        "summary": {
            "total_categories": len(categories),
            "covered": covered,
            "partial": partial,
            "missing": missing,
        },
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
        f.write("\n")

    logger.info(
        "Audit complete: %d categories (%d covered, %d partial, %d missing), %d total samples",
        len(categories), covered, partial, missing, total_samples,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Audit training data categories and flag gaps.",
    )
    parser.add_argument(
        "--input_dirs",
        nargs="+",
        required=True,
        help="Directories containing JSONL files to audit.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="training/data/data_audit_report.json",
        help="Path for the audit report JSON.",
    )
    parser.add_argument(
        "--threshold",
        type=int,
        default=500,
        help="Minimum sample count per category to be considered covered.",
    )
    return parser


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    parser = build_parser()
    args = parser.parse_args()
    run_audit(args)


if __name__ == "__main__":
    main()
