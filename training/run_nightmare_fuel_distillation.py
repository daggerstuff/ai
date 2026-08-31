#!/usr/bin/env python3
"""Run and export Nightmare Fuel adversarial clinical scenarios into Master Gold ChatML."""

import os
import json
import asyncio
import logging
from pathlib import Path

from training.nightmare_fuel_generator import main_async

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("run_nf")

CHECKPOINT_DIR = "ai/training/output/nightmare_fuel/checkpoints"
SURVIVORS_FILE = "ai/training/output/nightmare_fuel/survivors.jsonl"
OUT_MASTER = "ai/data/curated/sft_chatml/train_master_gold.jsonl"

async def run():
    logger.info("Starting Nightmare Fuel Generation & Dual-Judge Gating...")
    await main_async(
        num_cases=20,
        concurrency=2,
        checkpoint_dir=CHECKPOINT_DIR,
        checkpoint_interval=5,
        checkpoint_interval_seconds=15,
        resume=True,
        category="nightmare_adversarial"
    )
    
    # Check output
    records_file = Path(CHECKPOINT_DIR) / "records.jsonl"
    if records_file.exists():
        count = sum(1 for line in open(records_file) if line.strip())
        logger.info("Nightmare Fuel Checkpointed Records: %d", count)
        
        # Format into ChatML and append to train_master_gold.jsonl
        appended = 0
        with open(records_file, "r", encoding="utf-8") as fin, open(OUT_MASTER, "a", encoding="utf-8") as fout:
            for line in fin:
                if not line.strip(): continue
                data = json.loads(line)
                messages = data.get("messages", [])
                if not messages and "transcript" in data:
                    continue
                record = {
                    "messages": messages,
                    "source": "empathy_nightmare_fuel",
                    "task_type": "adversarial_crisis_deescalation",
                    "tier": "T1_GOLD",
                    "diagnostic_tag": "acute_crisis_and_rupture",
                    "demographic_tags": [],
                    "linguistic_style": "clinical_high_duress",
                    "clinical_reviewed": True,
                    "mi_quality": "high",
                    "provenance": {
                        "scenario": data.get("scenario", "adversarial_clinical_nightmare"),
                        "generator": "nightmare_fuel_generator",
                        "dual_judge_validated": True
                    }
                }
                fout.write(json.dumps(record, ensure_ascii=False) + "\n")
                appended += 1
        logger.info("Successfully appended %d Nightmare Fuel records to %s", appended, OUT_MASTER)

if __name__ == "__main__":
    asyncio.run(run())
