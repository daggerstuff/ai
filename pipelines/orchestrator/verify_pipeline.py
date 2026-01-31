#!/usr/bin/env python3
"""
Pipeline Verification Script
Verifies the dataset pipeline can run and produce outputs
"""

import sys
from pathlib import Path

# Add ai to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

def verify_imports():
    """Verify all critical imports work"""
    print("Verifying imports...")
    try:
        from ai.pipelines.orchestrator.orchestration.integrated_training_pipeline import (
            IntegratedTrainingPipeline,
            IntegratedPipelineConfig
        )
        print("✅ IntegratedTrainingPipeline imported")

        from ai.pipelines.orchestrator.ingestion.edge_case_jsonl_loader import EdgeCaseJSONLLoader
        print("✅ EdgeCaseJSONLLoader imported")

        from ai.pipelines.orchestrator.ingestion.dual_persona_loader import DualPersonaLoader
        print("✅ DualPersonaLoader imported")

        from ai.pipelines.orchestrator.ingestion.psychology_knowledge_loader import PsychologyKnowledgeLoader
        print("✅ PsychologyKnowledgeLoader imported")

        from ai.pipelines.orchestrator.ingestion.pixel_voice_loader import PixelVoiceLoader
        print("✅ PixelVoiceLoader imported")

        return True
    except Exception as e:
        print(f"❌ Import error: {e}")
        import traceback
        traceback.print_exc()
        return False


def verify_data_sources():
    """Verify data sources are accessible"""
    print("\nVerifying data sources...")

    try:
        from ai.pipelines.orchestrator.ingestion.edge_case_jsonl_loader import EdgeCaseJSONLLoader
        edge_loader = EdgeCaseJSONLLoader()
        if edge_loader.check_pipeline_output_exists():
            stats = edge_loader.get_statistics()
            print(f"✅ Edge cases: {stats.get('total_examples', 0)} examples")
        else:
            print("⚠️  Edge case data not found (optional)")
    except Exception as e:
        print(f"⚠️  Edge case loader error: {e}")

    try:
        from ai.pipelines.orchestrator.ingestion.dual_persona_loader import DualPersonaLoader
        persona_loader = DualPersonaLoader()
        stats = persona_loader.get_statistics()
        print(f"✅ Dual persona: {stats.get('total_dialogues', 0)} dialogues")
    except Exception as e:
        print(f"⚠️  Dual persona loader error: {e}")

    try:
        from ai.pipelines.orchestrator.ingestion.psychology_knowledge_loader import PsychologyKnowledgeLoader
        psych_loader = PsychologyKnowledgeLoader()
        if psych_loader.check_knowledge_base_exists():
            stats = psych_loader.get_statistics()
            print(f"✅ Psychology knowledge: {stats.get('total_concepts', 0)} concepts")
        else:
            print("⚠️  Psychology knowledge not found (optional)")
    except Exception as e:
        print(f"⚠️  Psychology loader error: {e}")


def test_minimal_pipeline():
    """Test running a minimal pipeline execution"""
    print("\nTesting minimal pipeline execution...")

    try:
        from ai.pipelines.orchestrator.orchestration.integrated_training_pipeline import (
            IntegratedTrainingPipeline,
            IntegratedPipelineConfig
        )

        # Create minimal config
        from ai.pipelines.orchestrator.storage_config import get_dataset_pipeline_output_root
        output_root = get_dataset_pipeline_output_root()

        # Force local storage for this verification run so it works without cloud credentials.
        from ai.pipelines.orchestrator.storage_config import (
            StorageConfig,
            StorageBackend,
            set_storage_config,
        )
        set_storage_config(
            StorageConfig(
                backend=StorageBackend.LOCAL,
                local_base_path=output_root / "data",
            )
        )
        config = IntegratedPipelineConfig(
            target_total_samples=10,  # Very small test
            output_dir=str(output_root / "test_output"),
            output_filename="test_verify.json",
            enable_bias_detection=False,
            enable_quality_validation=False
        )

        print(f"Creating pipeline with target: {config.target_total_samples} samples")
        pipeline = IntegratedTrainingPipeline(config)

        print("Running pipeline...")
        result = pipeline.run()

        print(f"\n✅ Pipeline completed!")
        stats = result.get('statistics')
        if stats:
            print(f"   Total samples: {getattr(stats, 'total_samples', 0)}")
        else:
            print(f"   Total samples: {len(result.get('training_data', []))}")

        output_file = result.get('output_path', result.get('output_file', ''))
        print(f"   Output file: {output_file}")

        # Check if output file exists
        output_path = Path(output_file)
        if output_path.exists():
            print(f"✅ Output file exists: {output_path}")
            file_size = output_path.stat().st_size
            print(f"   File size: {file_size} bytes")
            return True
        else:
            print(f"⚠️  Output file not found: {output_path}")
            return False

    except Exception as e:
        print(f"❌ Pipeline execution error: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    print("=" * 80)
    print("DATASET PIPELINE VERIFICATION")
    print("=" * 80)

    # Step 1: Verify imports
    if not verify_imports():
        print("\n❌ Import verification failed")
        sys.exit(1)

    # Step 2: Verify data sources
    verify_data_sources()

    # Step 3: Test minimal pipeline
    if test_minimal_pipeline():
        print("\n" + "=" * 80)
        print("✅ VERIFICATION COMPLETE - Pipeline is operational")
        print("=" * 80)
        sys.exit(0)
    else:
        print("\n" + "=" * 80)
        print("⚠️  VERIFICATION INCOMPLETE - Pipeline has issues")
        print("=" * 80)
        sys.exit(1)

