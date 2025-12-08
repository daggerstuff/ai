#!/usr/bin/env python3
import sys
import logging
from pathlib import Path

# Add project root to path for imports
project_root = Path(__file__).resolve().parents[1]
sys.path.append(str(project_root))

from ai.dataset_pipeline.orchestration.pipeline_runner import PipelineRunner
from ai.dataset_pipeline.sourcing.academic_sourcing import AcademicSourcingEngine
from ai.dataset_pipeline.therapies.cbt_integration import CBTIntegration
from ai.dataset_pipeline.therapies.dbt_integration import DBTIntegration
from ai.dataset_pipeline.therapies.emdr_integration import EMDRIntegration
from ai.dataset_pipeline.therapies.act_integration import ACTIntegration
from ai.dataset_pipeline.therapies.crisis_expansion import CrisisScenarioExpander # New Stage 3
from ai.dataset_pipeline.simulation.session_simulator import SessionSimulator
from ai.dataset_pipeline.synthesis.dataset_synthesizer import DatasetSynthesizer
from ai.dataset_pipeline.processing.transcript_ingestor import TranscriptIngestor
from ai.dataset_pipeline.alignment.dpo_generator import DPOGenerator
from ai.dataset_pipeline.sourcing.multi_source_ingestor import run_all_ingestors
import ai.training_ready.scripts.upload_to_s3 as s3_uploader

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("MentalHealthPipeline")

# --- Phase 0: External Acquisition (Direct S3 Stream) ---
def run_external_ingestion():
    """Run External Data Ingestion - streams directly to OVH S3."""
    return run_all_ingestors(bucket="pixel-data", prefix="datasets/training_v3/")

# --- Phase 1: Internal Sourcing ---
def run_sourcing():
    """Run sourcing tasks (ArXiv)."""
    engine = AcademicSourcingEngine()
    return engine.run_sourcing_pipeline()

def run_voice_ingestion():
    """Run Voice Data Ingestion."""
    ingestor = TranscriptIngestor()
    output = ingestor.process_batch(batch_size=200)
    return f"Ingested voice data to {output}" if output else "No voice data processed."

# --- Phase 2: Reasoning & Therapy Generation ---
def run_cbt():
    cbt = CBTIntegration()
    data = cbt.generate_batch_content(count=50)  # Increased
    return cbt.export_data(data)

def run_dbt():
    dbt = DBTIntegration()
    data = dbt.generate_batch_content(count=50)  # Increased
    return dbt.export_data(data)

def run_emdr():
    emdr = EMDRIntegration()
    data = emdr.generate_batch_content(count=50)  # Increased
    return emdr.export_data(data)

def run_act():
    act = ACTIntegration()
    data = act.generate_batch_content(count=50)  # Increased
    return act.export_data(data)

# --- Phase 3: Edge Cases ---
def run_crisis_expansion():
    """Run Crisis/Nightmare Fuel Generation."""
    expander = CrisisScenarioExpander()
    scenarios, output = expander.generate_batch(count=20)  # Increased
    return f"Generated {len(scenarios)} nightmare scenarios to {output}"

# --- Phase 4: Simulation ---
def run_simulation():
    """Run Session Simulation (including Journaling)."""
    sim = SessionSimulator()
    data = sim.generate_batch(count=20)  # Increased
    return f"Generated {len(data)} sessions."

# --- Phase 5: Synthesis ---
def run_synthesis():
    """Run Final Synthesis."""
    synth = DatasetSynthesizer()
    dataset = synth.synthesize_dataset(format_type="alpaca")
    splits = synth.split_dataset(dataset)

    results = []
    for split, items in splits.items():
        output = synth.output_path / f"final_{split}.jsonl"
        with open(output, "w") as f:
            import json
            for item in items:
                f.write(json.dumps(item) + "\n")
        results.append(f"{split}: {len(items)}")

    return ", ".join(results)

# --- Phase 6: Alignment ---
def run_dpo_generation():
    """Run DPO Pair Generation."""
    dpo = DPOGenerator()
    voice_file = "ai/training_ready/datasets/stage4_voice/processed_transcripts/voice_training_data_001.json"
    data = dpo.process_voice_data(voice_file)
    if data:
        return dpo.export_dpo(data)
    return "No DPO pairs generated (missing voice data?)"

# --- Phase 7: Deployment ---
def run_s3_upload():
    """Run S3 Upload."""
    try:
        s3_uploader.upload_final_artifacts()
        return "Upload initiated (check logs)."
    except Exception as e:
        return f"Upload failed: {e}"

def main():
    runner = PipelineRunner()

    # Execute Pipeline in Order
    runner.run_stage("External Dataset Ingestion (HF)", run_external_ingestion)

    runner.run_stage("Academic Sourcing (ArXiv)", run_sourcing)
    runner.run_stage("Voice Ingestion (Transcripts)", run_voice_ingestion)

    runner.run_stage("CBT Generation (LLM-Augmented)", run_cbt)
    runner.run_stage("DBT Generation", run_dbt)
    runner.run_stage("EMDR Generation", run_emdr)
    runner.run_stage("ACT Generation", run_act)

    runner.run_stage("Crisis Expansion (Nightmare Fuel)", run_crisis_expansion)

    runner.run_stage("Multi-Turn Simulation (Journaling)", run_simulation)

    runner.run_stage("Final Synthesis (Instruction Tuning)", run_synthesis)

    runner.run_stage("Alignment Data Generation (DPO)", run_dpo_generation)

    runner.run_stage("S3 Upload", run_s3_upload)

    runner.report()

if __name__ == "__main__":
    main()
