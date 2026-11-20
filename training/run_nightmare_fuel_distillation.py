#!/usr/bin/env python3
"""Generate and append final-model-accepted synthetic research sessions."""

import json
import logging
import os
from pathlib import Path

import anyio

from training.nightmare_fuel_generator import main_async
from training.nightmare_recovery import existing_session_digests, session_digest

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("run_nf")

_AI_ROOT = Path(__file__).resolve().parent.parent
CHECKPOINT_DIR = str(_AI_ROOT / "training" / "output" / "nightmare_fuel" / "checkpoints")
OUT_MASTER = str(_AI_ROOT / "data" / "curated" / "sft_chatml" / "train_master_gold.jsonl")


async def run() -> None:
    logger.info("Starting synthetic generation and final model-judge gating")
    master = Path(OUT_MASTER)
    existing = existing_session_digests(master)
    accepted = await main_async(
        num_cases=20,
        concurrency=2,
        checkpoint_dir=CHECKPOINT_DIR,
        checkpoint_interval=5,
        checkpoint_interval_seconds=15,
        resume=True,
        category="nightmare_adversarial",
    )
    # Checkpoints precede the final judge and cannot substitute for its output.
    if not accepted:
        logger.info("No sessions passed the final model judge; nothing appended")
        return
    appended = 0
    with master.open("ab+") as handle:
        for session in accepted:
            digest = session_digest(session.messages)
            if digest in existing:
                continue
            record = {
                "messages": [message.model_dump() for message in session.messages],
                "source": "empathy_nightmare_fuel",
                "task_type": "adversarial_crisis_deescalation",
                "tier": "T3_BRONZE",
                "diagnostic_tag": "acute_crisis_and_rupture",
                "demographic_tags": [],
                "linguistic_style": "clinical_high_duress",
                "clinical_reviewed": False,
                "mi_quality": "unassessed",
                "provenance": {
                    "scenario": session.scenario,
                    "generator": "nightmare_fuel_generator",
                    "validation_method": "generator_final_model_judge",
                    "review_status": "unreviewed_synthetic",
                },
            }
            if handle.tell() and appended == 0:
                handle.seek(-1, os.SEEK_END)
                if handle.read(1) != b"\n":
                    handle.write(b"\n")
            handle.write((json.dumps(record, ensure_ascii=False) + "\n").encode())
            existing.add(digest)
            appended += 1
    logger.info("Appended %d final-model-accepted synthetic sessions", appended)


if __name__ == "__main__":
    anyio.run(run)
