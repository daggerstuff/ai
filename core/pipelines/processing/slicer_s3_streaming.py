#!/usr/bin/env python3
import json
import logging
import subprocess
import time
from pathlib import Path

logger = logging.getLogger(__name__)

FILE_TO_STAGE = {
    # Stage 1
    "mental_health_counseling": "stage1_foundation",
    "additional_specialized": "stage1_foundation",
    "psychology-6k": "stage1_foundation",
    "mental_health_clean": "stage1_foundation",
    "ultimate_final_dataset": "stage1_foundation",

    # Stage 2
    "cot_reasoning": "stage2_therapeutic_expertise",
    "clinical_diagnosis_mental_health": "stage2_therapeutic_expertise",
    "cultural_nuances": "stage2_therapeutic_expertise",

    # Stage 3
    "edge_cases_training_format": "stage3_edge_stress_test",
    "nightmare_fuel": "stage3_edge_stress_test",
    "example_prompts": "stage3_edge_stress_test", # inside ULTIMATE

    # Stage 4
    "tim_fletcher": "stage4_voice_persona",
    "heidi_priebe": "stage4_voice_persona",
    "doctorramani": "stage4_voice_persona",
    "therapy_in_a_nutshell": "stage4_voice_persona",
    "patrick_teahan": "stage4_voice_persona",
    "crappy_childhood_fairy": "stage4_voice_persona",
    "doc_snipes": "stage4_voice_persona",
}

def determine_stage(record: dict) -> str:
    metadata = record.get("metadata", {})
    file_key = metadata.get("file_key", "").lower()

    # Check inner source if file_key is ultimate
    if "ultimate_final_dataset" in file_key:
        # Check messages for inner ghost metadata
        for msg in record.get("messages", []):
            if msg.get("role") == "user":
                content = msg.get("content", "")
                if content.startswith("{") and "_source_file" in content:
                    try:
                        # lightweight check
                        if "example_prompts.jsonl" in content:
                            return "stage3_edge_stress_test"
                    except:
                        pass

    for match, stage in FILE_TO_STAGE.items():
        if match in file_key:
            return stage

    return "stage1_foundation" # Default everything else to foundation

def run_slicer():

    stage_counts = {
        "stage1_foundation": 0,
        "stage2_therapeutic_expertise": 0,
        "stage3_edge_stress_test": 0,
        "stage4_voice_persona": 0
    }

    out_files = {}
    out_dir = Path("ai/data/staged_datasets")
    out_dir.mkdir(parents=True, exist_ok=True)

    for stage in stage_counts:
        out_files[stage] = open(out_dir / f"{stage}.jsonl", "w", encoding="utf-8")

    cmd = ["bash", "-c", "rclone cat HetznerS3:pixeldata/final_dataset/MASTER_TRAINING_SET.jsonl"]

    process = subprocess.Popen(cmd, stdout=subprocess.PIPE, text=True, encoding="utf-8")

    count = 0
    start_time = time.time()
    try:
        for line in process.stdout:
            count += 1
            if count % 500000 == 0:
                time.time() - start_time

            try:
                record = json.loads(line)
            except:
                continue

            stage_id = determine_stage(record)
            out_files[stage_id].write(line)
            stage_counts[stage_id] += 1

    finally:
        for f in out_files.values():
            f.close()

    for stage, _c in stage_counts.items():
        pass

    # Generate enhanced metadata
    metadata = {
        "version": "1.1.0",
        "dact": "DACT-06",
        "description": "Stage-based dataset slices (PIX-249)",
        "total_records": count,
        "stage_details": {
            k: {"stage_id": k, "records_processed": v, "records_written": v} for k, v in stage_counts.items()
        },
        "downstream_ready": True
    }
    with open(out_dir / "dact06_enhanced_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

if __name__ == "__main__":
    run_slicer()
