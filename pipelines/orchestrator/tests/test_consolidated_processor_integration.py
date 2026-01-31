#!/usr/bin/env python3
"""
Integration test for the consolidated mental health processor.
Tests processing of the actual 86MB merged mental health dataset.
"""

import os
import sys
import tempfile

# Add the project root to the path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from ai.pipelines.orchestrator.consolidated_mental_health_processor import (
    ConsolidatedMentalHealthProcessor,
    ConsolidatedProcessingConfig,
)


def test_consolidated_processor_integration():
    """Test the consolidated processor with the actual dataset."""

    # Check if the dataset file exists
    dataset_path = "ai/merged_mental_health_dataset.jsonl"
    if not os.path.exists(dataset_path):
        return False

    # Get file size
    os.path.getsize(dataset_path) / (1024 * 1024)  # MB

    # Create temporary output directory
    with tempfile.TemporaryDirectory() as temp_dir:

        # Configure processor for small test run
        config = ConsolidatedProcessingConfig(
            input_file=dataset_path,
            output_dir=temp_dir,
            target_conversations=100,  # Small test run
            batch_size=50,
            quality_threshold=0.7,  # Lower threshold for testing
            therapeutic_accuracy_threshold=0.7,
            emotional_authenticity_threshold=0.7
        )

        # Create processor
        processor = ConsolidatedMentalHealthProcessor(config)

        try:
            # Process the dataset
            result = processor.process_consolidated_dataset()

            # Print results


            result["conversation_analysis"]["therapeutic_approaches"]

            result["conversation_analysis"]["emotional_intensity_distribution"]

            result["conversation_analysis"]["conversation_length_stats"]

            # Check output files
            output_file = os.path.join(temp_dir, "consolidated_mental_health_conversations.jsonl")
            report_file = os.path.join(temp_dir, "consolidated_processing_report.json")

            if os.path.exists(output_file):
                os.path.getsize(output_file) / 1024  # KB
            else:
                return False

            if os.path.exists(report_file):
                os.path.getsize(report_file) / 1024  # KB
            else:
                return False

            return True

        except Exception:
            import traceback
            traceback.print_exc()
            return False


def test_sample_conversation_processing():
    """Test processing of a few sample conversations."""

    # Create a small test file
    test_data = [
        {
            "prompt": "You are a helpful mental health counselling assistant, please answer the mental health questions based on the patient's description. I've been feeling anxious lately and can't sleep.",
            "response": "I understand you're experiencing anxiety and sleep difficulties. Let's explore some strategies that might help you manage these symptoms."
        },
        {
            "prompt": "You are a helpful mental health counselling assistant, please answer the mental health questions based on the patient's description. I'm having trouble with my relationships.",
            "response": "Relationship difficulties can be challenging. Let's work together to identify some strategies that might help."
        }
    ]

    with tempfile.TemporaryDirectory() as temp_dir:
        # Create test input file
        test_input = os.path.join(temp_dir, "test_input.jsonl")
        with open(test_input, "w") as f:
            for item in test_data:
                import json
                f.write(json.dumps(item) + "\n")

        # Configure processor
        config = ConsolidatedProcessingConfig(
            input_file=test_input,
            output_dir=temp_dir,
            target_conversations=10,
            batch_size=5,
            quality_threshold=0.5,  # Very low threshold for testing
            therapeutic_accuracy_threshold=0.5,
            emotional_authenticity_threshold=0.5
        )

        # Create processor
        processor = ConsolidatedMentalHealthProcessor(config)

        try:
            # Process the test data
            processor.process_consolidated_dataset()


            return True

        except Exception:
            import traceback
            traceback.print_exc()
            return False


if __name__ == "__main__":

    # Test sample processing first
    sample_success = test_sample_conversation_processing()

    if sample_success:
        # Test with actual dataset
        integration_success = test_consolidated_processor_integration()

        if integration_success:
            sys.exit(0)
        else:
            sys.exit(1)
    else:
        sys.exit(1)
