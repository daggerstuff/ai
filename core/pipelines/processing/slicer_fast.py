#!/usr/bin/env python3
import sys
import logging
import subprocess
from pathlib import Path
import time
import json

logger = logging.getLogger(__name__)

def determine_stage_fast(line: str) -> str:
    # Most records have their file_key near the end
    # We can just check for unique substrings in the line to route them
    # This is extremely fast compared to json.loads()
    
    # Stage 2
    if '"cot_reasoning.json"' in line or "clinical_diagnosis_mental_health" in line or "cultural_nuances" in line:
        return "stage2_therapeutic_expertise"
        
    # Stage 3
    if "edge_cases_training_format" in line or "nightmare_fuel" in line or "example_prompts.jsonl" in line:
        return "stage3_edge_stress_test"
        
    # Stage 4
    if "tim_fletcher" in line or "heidi_priebe" in line or "doctorramani" in line or "therapy_in_a_nutshell" in line or "patrick_teahan" in line or "crappy_childhood_fairy" in line or "doc_snipes" in line:
        return "stage4_voice_persona"
        
    # Stage 1 / Default
    return "stage1_foundation"


def run_slicer():
    print("Initializing fast slicer...")
    
    stage_counts = {
        "stage1_foundation": 0,
        "stage2_therapeutic_expertise": 0,
        "stage3_edge_stress_test": 0,
        "stage4_voice_persona": 0
    }
    
    out_files = {}
    out_dir = Path("ai/data/staged_datasets")
    out_dir.mkdir(parents=True, exist_ok=True)
    
    for stage in stage_counts.keys():
        out_files[stage] = open(out_dir / f"{stage}.jsonl", "w", encoding="utf-8")
        
    print("Streaming from S3...")
    cmd = ["bash", "-c", "rclone cat HetznerS3:pixeldata/final_dataset/MASTER_TRAINING_SET.jsonl"]
    
    process = subprocess.Popen(cmd, stdout=subprocess.PIPE, text=True, encoding='utf-8')
    
    count = 0
    start_time = time.time()
    try:
        for line in process.stdout:
            count += 1
            if count % 500000 == 0:
                elapsed = time.time() - start_time
                print(f"Processed {count} records in {elapsed:.1f}s...")
                
            stage_id = determine_stage_fast(line)
            out_files[stage_id].write(line)
            stage_counts[stage_id] += 1
            
    finally:
        for f in out_files.values():
            f.close()
            
    print(f"Finished processing {count} records in {time.time() - start_time:.1f}s.")
    print("Stage breakdown:")
    for stage, c in stage_counts.items():
        print(f"  {stage}: {c}")
        
    metadata = {
        "version": "1.1.0",
        "dact": "DACT-06",
        "description": "Stage-based dataset slices (Fast)",
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
