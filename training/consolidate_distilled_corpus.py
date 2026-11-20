#!/usr/bin/env python3
"""Consolidates distilled clinical books and practitioner personas into the master training corpus."""

import glob
import json
import logging
from pathlib import Path
from typing import Any

_PROJECT_ROOT = Path(__file__).resolve().parents[2]


logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("consolidate_corpus")

BOOKS_DIR = _PROJECT_ROOT / "ai" / "data" / "curated/books_distilled"
PERSONAS_DIR = _PROJECT_ROOT / "ai" / "data" / "curated/youtube_distilled_personas"
MASTER_TRAIN = _PROJECT_ROOT / "ai" / "data" / "curated/sft_chatml/train.jsonl"
CONSOLIDATED_OUT = _PROJECT_ROOT / "ai" / "data" / "curated/sft_chatml/train_master_gold.jsonl"


def main() -> None:
    gold_records: list[dict[str, Any]] = []

    # 1. Ingest YouTube Distilled Personas
    persona_files = glob.glob(str(PERSONAS_DIR / "*.jsonl"))
    logger.info("Found %d persona files in %s", len(persona_files), PERSONAS_DIR)
    for pf in sorted(persona_files):
        with open(pf, encoding="utf-8") as f:
            for line in filter(str.strip, f):
                try:
                    record = json.loads(line)
                    gold_records.append(record)
                except Exception as exc:
                    logger.warning("Error parsing line in %s: %s", pf, exc)

    logger.info("Ingested %d distilled YouTube persona records", len(gold_records))

    # 2. Ingest Books Distilled Pairs
    book_files = glob.glob(str(BOOKS_DIR / "*.jsonl"))
    logger.info("Found %d distilled book files in %s", len(book_files), BOOKS_DIR)
    book_count = 0
    for bf in sorted(book_files):
        with open(bf, encoding="utf-8") as f:
            for line in filter(str.strip, f):
                try:
                    item = json.loads(line)
                    if "messages" in item:
                        gold_records.append(item)
                        book_count += 1
                    elif "instruction" in item and "output" in item:
                        meta = item.get("metadata", {})
                        book_name = meta.get("source_book", Path(bf).stem)
                        record = {
                            "messages": [
                                {
                                    "role": "system",
                                    "content": (
                                        "You are an expert clinical psychologist and therapeutic AI assistant. "
                                        "Provide grounded, authentic, non-sycophantic therapeutic dialogue."
                                    ),
                                },
                                {"role": "user", "content": item["instruction"]},
                                {"role": "assistant", "content": item["output"]},
                            ],
                            "source": f"clinical_book_{book_name[:30]}",
                            "task_type": "clinical_literature_distillation",
                            "tier": "T3_BRONZE",
                            "diagnostic_tag": "clinical_modality",
                            "demographic_tags": [],
                            "linguistic_style": "spoken_clinical_practitioner",
                            "clinical_reviewed": False,
                            "mi_quality": "unassessed",
                            "provenance": {
                                "source_book": book_name,
                                "source_type": "clinical_literature",
                                "review_status": "unreviewed_synthetic",
                            },
                        }
                        gold_records.append(record)
                        book_count += 1
                except Exception as exc:
                    logger.warning("Error parsing line in %s: %s", bf, exc)

    logger.info("Ingested %d distilled clinical book records", book_count)
    logger.info("Total gold proprietary records: %d", len(gold_records))

    # 3. Read Master Base Corpus
    base_count = 0
    with open(CONSOLIDATED_OUT, "w", encoding="utf-8") as fout:
        # Write gold records first (T1_GOLD priority)
        for r in gold_records:
            fout.write(json.dumps(r, ensure_ascii=False) + "\n")

        # Write existing base corpus
        if MASTER_TRAIN.exists():
            with open(MASTER_TRAIN, encoding="utf-8") as fin:
                for line in filter(str.strip, fin):
                    fout.write(line.strip() + "\n")
                    base_count += 1

    total = len(gold_records) + base_count
    logger.info("Consolidation complete!")
    logger.info("  - Gold Proprietary Assets: %d", len(gold_records))
    logger.info("  - Base Curated Corpus: %d", base_count)
    logger.info("  - Final Master Dataset: %d -> %s", total, CONSOLIDATED_OUT)


if __name__ == "__main__":
    main()
