#!/usr/bin/env python3
"""
Real End-to-End Pipeline Test for PIX-5 Validation.
This completely runs down the unified processing pipeline locally on dummy data.
"""
import json
import os
import shutil
import sys
from pathlib import Path

workspace_root = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(workspace_root))

from ai.pipelines.orchestrator.unified_preprocessing_pipeline import (
    DataSource,
    ProcessingConfig,
    UnifiedPreprocessingPipeline,
)


def build_dummy_jsonl() -> Path:
    """Builds a temporary JSONL target with deliberately bad data to test filtering, along with good data."""
    test_path = Path("/tmp/e2e_test_source.jsonl")
    test_data = [
        # 1. High Quality, Safe (Should PASS)
        {
            "conversation_id": "test_pass_1",
            "messages": [{"role": "user", "content": "I feel sad."}, {"role": "assistant", "content": "I hear you, let's explore that sadness."}],
            "metadata": {"quality_score": 0.9, "empathy_score": 0.9, "safety_score": 0.9}
        },
        # 2. Low Quality (Should FAIL config threshold)
        {
            "conversation_id": "test_fail_quality",
            "messages": [{"role": "user", "content": "Hi"}, {"role": "assistant", "content": "Hello"}],
            "metadata": {"quality_score": 0.2, "empathy_score": 0.2, "safety_score": 0.9}
        },
        # 3. Crisis Event (Should trigger safety filter)
        {
            "conversation_id": "test_fail_crisis",
            "messages": [{"role": "user", "content": "I want to kill myself right now."}],
            # Note: A real crisis detection service would set the safety_score explicitly.
            # Here we test if our pipeline respects low scores
            "metadata": {"quality_score": 0.9, "empathy_score": 0.9, "safety_score": 0.2}
        },
        # 4. Perfectly valid but duplicate of 1 (Should trigger deduplication if run twice)
        {
            "conversation_id": "test_pass_2",
            "messages": [{"role": "user", "content": "I feel happy."}, {"role": "assistant", "content": "That is wonderful."}],
            "metadata": {"quality_score": 0.95, "empathy_score": 0.95, "safety_score": 0.9}
        }
    ]
    with open(test_path, "w") as f:
        for rec in test_data:
            f.write(json.dumps(rec) + "\n")

    return test_path

def run_e2e_pipeline():
    print("="*60)
    print("PIX-5 REAL END-TO-END PIPELINE AUDIT")
    print("="*60)

    # 1. Setup Data
    test_file = build_dummy_jsonl()
    print(f"✅ Generated mock data at: {test_file}")

    # 2. Configure Pipeline
    config = ProcessingConfig(
        target_quality_threshold=0.7,
        deduplication_enabled=True,
        safety_filtering_enabled=True,
        validation_enabled=True,
        psychology_integration_enabled=False # Keeping isolated for simple E2E logic checking
    )

    class TestPipeline(UnifiedPreprocessingPipeline):
        def discover_data_sources(self) -> None:
            # Override exactly this to bypass scanning anything else.
            self.data_sources.clear()
            self.register_data_source(
                DataSource(
                    name="PIX_5_Audit_E2E",
                    path=str(test_file),
                    format="jsonl",
                    size_bytes=test_file.stat().st_size,
                    source_type="standard" # Use standard so crisis override isn't active
                )
            )

    pipeline = TestPipeline(config)

    # 4. Execution
    print("🚀 Initiating rigorous unified pipeline operations...")
    final_output_path = pipeline.execute_pipeline()

    # Read the final output
    record_ids = []
    with open(final_output_path) as f:
        for line in f:
            r = json.loads(line)
            record_ids.append(r.get("conversation_id"))

    # 5. Validation Assertions
    print(f"\n📊 Results: {len(record_ids)} / 4 records passed.")
    print(f"Expected IDs dropped: test_fail_quality, test_fail_crisis")
    print(f"Actually kept: {record_ids}")

    # Clean up output artifacts to avoid polluting git state
    final_output_dir = Path(final_output_path).parent
    shutil.rmtree(final_output_dir)

    # Expect 1 and 4 to survive.
    assert "test_fail_quality" not in record_ids, "Low quality record was NOT filtered."
    assert "test_fail_crisis" not in record_ids, "Crisis event was NOT filtered."
    assert "test_pass_1" in record_ids, "Valid record 1 was dropped improperly."
    assert "test_pass_2" in record_ids, "Valid record 2 was dropped improperly."

    print("\n✅ Verification PASSED: The components correctly evaluate, filter, and score the payload without hallucinated success metrics.")
    print("="*60)

    if test_file.exists():
        test_file.unlink()

if __name__ == "__main__":
    run_e2e_pipeline()
