#!/usr/bin/env python3
"""
End-to-End Pipeline Test
Tests the complete data integration and training pipeline
"""

import json
import sys
from pathlib import Path

# Ensure the outer workspace root is on sys.path so `ai.*` imports work reliably
workspace_root = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(workspace_root))

from ai.pipelines.orchestrator.storage_config import get_dataset_pipeline_output_root
from ai.pipelines.orchestrator.orchestration.integrated_training_pipeline import (
    IntegratedTrainingPipeline,
    IntegratedPipelineConfig
)
from ai.pipelines.orchestrator.ingestion.edge_case_jsonl_loader import EdgeCaseJSONLLoader
from ai.pipelines.orchestrator.ingestion.dual_persona_loader import DualPersonaLoader
from ai.pipelines.orchestrator.ingestion.psychology_knowledge_loader import PsychologyKnowledgeLoader


def test_individual_loaders():
    """Test each data loader individually"""
    print("=" * 80)
    print("TESTING INDIVIDUAL DATA LOADERS")
    print("=" * 80)
    
    # Test Edge Case Loader
    print("\n1. Testing Edge Case Loader...")
    edge_loader = EdgeCaseJSONLLoader()
    if edge_loader.check_pipeline_output_exists():
        edge_stats = edge_loader.get_statistics()
        print(f"   ✅ Edge cases: {edge_stats['total_examples']} examples")
        print(f"   Categories: {len(edge_stats['categories'])}")
    else:
        print("   ⚠️  Edge case data not found (will be skipped in integration)")
    
    # Test Dual Persona Loader
    print("\n2. Testing Dual Persona Loader...")
    persona_loader = DualPersonaLoader()
    persona_stats = persona_loader.get_statistics()
    print(f"   ✅ Dual persona: {persona_stats['total_dialogues']} dialogues")
    print(f"   Persona pairs: {len(persona_stats['persona_pairs'])}")
    
    # Test Psychology Knowledge Loader
    print("\n3. Testing Psychology Knowledge Loader...")
    psych_loader = PsychologyKnowledgeLoader()
    if psych_loader.check_knowledge_base_exists():
        psych_stats = psych_loader.get_statistics()
        print(f"   ✅ Psychology knowledge: {psych_stats['total_concepts']} concepts")
        print(f"   Categories: {len(psych_stats['categories'])}")
    else:
        print("   ⚠️  Psychology knowledge not found (will be skipped in integration)")
    
    print("\n" + "=" * 80)


def test_integrated_pipeline():
    """Test the complete integrated pipeline"""
    print("\n" + "=" * 80)
    print("TESTING INTEGRATED TRAINING PIPELINE")
    print("=" * 80)
    
    # Create test configuration with smaller target
    output_root = get_dataset_pipeline_output_root()
    config = IntegratedPipelineConfig(
        target_total_samples=100,  # Small test dataset
        output_dir=str(output_root / "test_output"),
        output_filename="test_training_dataset.json",
        enable_bias_detection=False,  # Skip for faster testing
        enable_quality_validation=False
    )
    
    print("\n📋 Configuration:")
    print(f"   Target samples: {config.target_total_samples}")
    print(f"   Edge cases: {config.edge_cases.target_percentage * 100}%")
    print(f"   Pixel voice: {config.pixel_voice.target_percentage * 100}%")
    print(f"   Psychology: {config.psychology_knowledge.target_percentage * 100}%")
    print(f"   Dual persona: {config.dual_persona.target_percentage * 100}%")
    print(f"   Standard: {config.standard_therapeutic.target_percentage * 100}%")
    
    # Run pipeline
    print("\n🚀 Running integrated pipeline...")
    pipeline = IntegratedTrainingPipeline(config)
    
    try:
        result = pipeline.run()
        
        print("\n" + "=" * 80)
        print("PIPELINE RESULTS")
        print("=" * 80)
        
        # Display statistics
        stats = result['statistics']
        print(f"\n✅ Total samples: {stats.total_samples}")
        print(f"⏱️  Integration time: {stats.integration_time:.2f}s")
        
        print(f"\n📊 Samples by source:")
        for source, count in stats.samples_by_source.items():
            percentage = (count / stats.total_samples * 100) if stats.total_samples > 0 else 0
            print(f"   {source}: {count} ({percentage:.1f}%)")
        
        if stats.warnings:
            print(f"\n⚠️  Warnings ({len(stats.warnings)}):")
            for warning in stats.warnings[:5]:
                print(f"   - {warning}")
        
        if stats.errors:
            print(f"\n❌ Errors ({len(stats.errors)}):")
            for error in stats.errors:
                print(f"   - {error}")
        
        # Verify output file
        output_path = Path(result['output_path'])
        if output_path.exists():
            print(f"\n✅ Output file created: {output_path}")
            
            # Load and verify
            with open(output_path, 'r') as f:
                data = json.load(f)
                print(f"   Conversations: {len(data['conversations'])}")
                print(f"   Metadata sources: {data['metadata']['sources']}")
            
            # Show sample
            if data['conversations']:
                sample = data['conversations'][0]
                print(f"\n📝 Sample conversation:")
                print(f"   Source: {sample.get('metadata', {}).get('source', 'unknown')}")
                print(f"   Text: {sample.get('text', '')[:150]}...")
        else:
            print(f"\n❌ Output file not created: {output_path}")
        
        print("\n" + "=" * 80)
        print("✅ END-TO-END TEST COMPLETE")
        print("=" * 80)
        
        return True
        
    except Exception as e:
        print(f"\n❌ Pipeline failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_progress_tracker_integration():
    """Test progress tracker integration"""
    print("\n" + "=" * 80)
    print("TESTING PROGRESS TRACKER INTEGRATION")
    print("=" * 80)
    
    try:
        from pathlib import Path
        import sys
        
        # Add lightning directory to path
        lightning_path = Path(__file__).parent.parent / "lightning"
        sys.path.insert(0, str(lightning_path))
        
        from therapeutic_progress_tracker import TherapeuticProgressTracker
        
        # Create test tracker
        tracker = TherapeuticProgressTracker(db_path=":memory:")
        
        # Log test session
        print("\n📝 Logging test session...")
        tracker.log_session(
            client_id="test_client_001",
            session_id="test_session_001",
            conversation_summary="Test conversation about anxiety",
            emotional_state="negative",
            therapeutic_goals=["Reduce anxiety", "Improve coping skills"],
            progress_notes="Client expressed concerns about work stress",
            therapist_observations="Client appears motivated to change",
            next_session_focus="Explore coping strategies"
        )
        
        # Retrieve sessions
        print("📊 Retrieving sessions...")
        sessions = tracker.get_sessions(client_id="test_client_001", days=7)
        print(f"   Found {len(sessions)} sessions")
        
        # Generate progress report
        print("📈 Generating progress report...")
        report = tracker.generate_progress_report(client_id="test_client_001", days=7)
        print(f"   Total sessions: {report['total_sessions']}")
        print(f"   Emotional trend: {report['emotional_trend']}")
        
        tracker.close()
        
        print("\n✅ Progress tracker integration working")
        return True
        
    except Exception as e:
        print(f"\n❌ Progress tracker test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests"""
    print("\n" + "=" * 80)
    print("FOUNDATION MODEL TRAINING - END-TO-END PIPELINE TEST")
    print("=" * 80)
    
    results = {
        "individual_loaders": False,
        "integrated_pipeline": False,
        "progress_tracker": False
    }
    
    # Test individual loaders
    try:
        test_individual_loaders()
        results["individual_loaders"] = True
    except Exception as e:
        print(f"\n❌ Individual loader tests failed: {e}")
    
    # Test integrated pipeline
    try:
        results["integrated_pipeline"] = test_integrated_pipeline()
    except Exception as e:
        print(f"\n❌ Integrated pipeline test failed: {e}")
    
    # Test progress tracker
    try:
        results["progress_tracker"] = test_progress_tracker_integration()
    except Exception as e:
        print(f"\n❌ Progress tracker test failed: {e}")
    
    # Summary
    print("\n" + "=" * 80)
    print("TEST SUMMARY")
    print("=" * 80)
    
    for test_name, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{status}: {test_name}")
    
    all_passed = all(results.values())
    
    if all_passed:
        print("\n🎉 ALL TESTS PASSED!")
        print("\nNext steps:")
        print("1. Run edge case generator to create actual data:")
        print("   cd ai/pipelines/edge_case/")
        print("   python quick_start.py")
        print("\n2. Run full integrated pipeline:")
        print("   python ai/pipelines/orchestrator/orchestration/integrated_training_pipeline.py")
        print("\n3. Start training on H100:")
        print("   cd ai/lightning/")
        print("   python train_optimized.py")
    else:
        print("\n⚠️  Some tests failed. Review errors above.")
    
    print("=" * 80)
    
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
