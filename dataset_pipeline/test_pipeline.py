"""
Test script for the unified preprocessing pipeline
This directly imports the pipeline module to avoid __init__.py issues
"""

import sys
import os

# Add the dataset_pipeline directory to the path
dataset_pipeline_dir = os.path.join(os.path.dirname(__file__), '.')
sys.path.insert(0, dataset_pipeline_dir)

# Import the module directly
import unified_preprocessing_pipeline

def test_pipeline():
    """Test the unified preprocessing pipeline"""
    print("Testing unified preprocessing pipeline...")
    print(f"Current working directory: {os.getcwd()}")

    try:
        # Create pipeline
        pipeline = unified_preprocessing_pipeline.create_default_pipeline()
        print("✓ Pipeline created successfully")
        print(f"  Data sources: {len(pipeline.data_sources)}")
        print("  Pipeline config:")
        print(f"    Target quality threshold: {pipeline.config.target_quality_threshold}")
        print(f"    Deduplication enabled: {pipeline.config.deduplication_enabled}")
        print(f"    Safety filtering enabled: {pipeline.config.safety_filtering_enabled}")
        print(f"    Psychology integration enabled: {pipeline.config.psychology_integration_enabled}")

        # Discover data sources
        pipeline.discover_data_sources()
        print(f"✓ Discovered {len(pipeline.data_sources)} data sources")

        # Show data source details
        for i, source in enumerate(pipeline.data_sources):
            print(f"  {i+1}. {source.name} ({source.source_type}) - {source.size_bytes} bytes")

        return True

    except Exception as e:
        print(f"✗ Pipeline test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_pipeline()
    exit(0 if success else 1)