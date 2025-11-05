"""
Dataset Quality Validation Script
Validates the final integrated dataset quality
"""

import sys
import os
import json
from pathlib import Path
from typing import Dict, List, Any
import hashlib
from datetime import datetime

# Add the project root to the path
project_root = os.path.join(os.path.dirname(__file__), '..', '..')
sys.path.insert(0, project_root)

# Import the modules directly to avoid __init__.py issues
dataset_pipeline_dir = os.path.join(os.path.dirname(__file__), '.')
sys.path.insert(0, dataset_pipeline_dir)

def validate_dataset_quality(dataset_path: str) -> Dict[str, Any]:
    """
    Validate the quality of the integrated dataset

    Args:
        dataset_path: Path to the dataset file (JSONL format)

    Returns:
        Dictionary with validation results
    """
    print(f"Validating dataset quality: {dataset_path}")

    validation_results = {
        "timestamp": "",
        "dataset_path": dataset_path,
        "file_exists": False,
        "file_readable": False,
        "record_count": 0,
        "quality_metrics": {},
        "validation_errors": [],
        "recommendations": []
    }

    try:
        # Check if file exists
        if not os.path.exists(dataset_path):
            validation_results["validation_errors"].append(f"Dataset file not found: {dataset_path}")
            return validation_results

        validation_results["file_exists"] = True

        # Count records and validate format
        record_count = 0
        quality_scores = []
        source_types = {}
        content_lengths = []
        field_coverage = {
            "prompt_id": 0,
            "category": 0,
            "scenario_type": 0,
            "instructions": 0,
            "metadata": 0,
            "_source": 0,
            "messages": 0,
            "text": 0
        }

        with open(dataset_path, 'r') as f:
            for line_num, line in enumerate(f, 1):
                if not line.strip():
                    continue

                try:
                    record = json.loads(line.strip())
                    record_count += 1

                    # Check required fields for different record types
                    for field in field_coverage:
                        if field in record:
                            field_coverage[field] += 1

                    # Track source types
                    source_type = record.get("_source", "unknown")
                    source_types[source_type] = source_types.get(source_type, 0) + 1

                    # Track content length (different fields for different record types)
                    content_length = 0
                    if "instructions" in record:
                        content_length = len(record["instructions"])
                    elif "text" in record:
                        content_length = len(record["text"])
                    elif "messages" in record:
                        for msg in record["messages"]:
                            if "content" in msg:
                                content_length += len(msg["content"])
                    content_lengths.append(content_length)

                    # Sample validation for first 100 records
                    if record_count <= 100:
                        # Validate basic structure - at least one content field should be present
                        has_content = any(field in record for field in ["instructions", "text", "messages"])
                        if not has_content:
                            validation_results["validation_errors"].append(
                                f"Record {record_count} missing content fields (instructions, text, or messages)"
                            )

                        # Validate metadata if present
                        if "metadata" in record:
                            if not isinstance(record["metadata"], dict):
                                validation_results["validation_errors"].append(
                                    f"Record {record_count} has invalid metadata format"
                                )

                    # Stop early for very large files
                    if record_count >= 10000:
                        break

                except json.JSONDecodeError as e:
                    validation_results["validation_errors"].append(
                        f"Invalid JSON at line {line_num}: {str(e)}"
                    )

        validation_results["file_readable"] = True
        validation_results["record_count"] = record_count

        # Calculate quality metrics
        if record_count > 0:
            validation_results["quality_metrics"] = {
                "field_coverage": {
                    field: f"{(count / record_count) * 100:.1f}%"
                    for field, count in field_coverage.items()
                },
                "source_type_distribution": source_types,
                "average_content_length": sum(content_lengths) / len(content_lengths) if content_lengths else 0,
                "min_content_length": min(content_lengths) if content_lengths else 0,
                "max_content_length": max(content_lengths) if content_lengths else 0
            }

        # Generate recommendations
        recommendations = []

        if record_count == 0:
            recommendations.append("Dataset is empty - check data processing pipeline")
        elif record_count < 1000:
            recommendations.append("Dataset is small - consider adding more data sources")

        # Check content length
        avg_length = validation_results["quality_metrics"].get("average_content_length", 0)
        if avg_length < 50:
            recommendations.append(f"Average content length is very short: {avg_length:.1f} characters")
        elif avg_length > 50000:
            recommendations.append(f"Average content length is very long: {avg_length:.1f} characters")

        validation_results["recommendations"] = recommendations

        # Overall validation status
        if len(validation_results["validation_errors"]) == 0:
            validation_results["overall_status"] = "PASSED"
        else:
            validation_results["overall_status"] = "FAILED"

        validation_results["timestamp"] = datetime.now().isoformat()

    except Exception as e:
        validation_results["validation_errors"].append(f"Validation failed: {str(e)}")
        validation_results["overall_status"] = "FAILED"
        import traceback
        validation_results["error_traceback"] = traceback.format_exc()

    return validation_results

def find_latest_dataset() -> str:
    """Find the latest dataset file in the output directory"""
    output_dir = Path("ai/dataset_pipeline/final_output")
    if not output_dir.exists():
        return ""

    # Look for JSONL files
    jsonl_files = list(output_dir.glob("*.jsonl"))
    if not jsonl_files:
        return ""

    # Return the most recent file
    latest_file = max(jsonl_files, key=lambda f: f.stat().st_mtime)
    return str(latest_file)

def main():
    """Main validation function"""
    print("Dataset Quality Validation")
    print("=" * 30)

    # Find the latest dataset
    dataset_path = find_latest_dataset()

    if not dataset_path:
        # Check common dataset locations
        common_paths = [
            "ai/training_data_consolidated/final_datasets/ULTIMATE_FINAL_DATASET.jsonl",
            "ai/dataset_pipeline/final_output/unified_training_dataset.jsonl",
            "ai/training_data_consolidated/datasets/merged_dataset.jsonl"
        ]

        for path in common_paths:
            if os.path.exists(path):
                dataset_path = path
                break

    if not dataset_path:
        print("❌ No dataset file found for validation")
        print("Please specify a dataset path or ensure datasets exist in:")
        print("  - ai/dataset_pipeline/final_output/")
        print("  - ai/training_data_consolidated/final_datasets/")
        return False

    print(f"Validating dataset: {dataset_path}")

    # Validate the dataset
    results = validate_dataset_quality(dataset_path)

    # Display results
    print(f"\nValidation Results:")
    print(f"  Status: {results['overall_status']}")
    print(f"  Records: {results['record_count']:,}")
    print(f"  File readable: {results['file_readable']}")

    if results['quality_metrics']:
        print(f"\nQuality Metrics:")
        metrics = results['quality_metrics']
        if 'field_coverage' in metrics:
            print(f"  Field Coverage:")
            for field, coverage in metrics['field_coverage'].items():
                print(f"    {field}: {coverage}")

        if 'source_type_distribution' in metrics:
            print(f"  Source Types:")
            for source_type, count in metrics['source_type_distribution'].items():
                print(f"    {source_type}: {count:,}")

        avg_length = metrics.get('average_content_length', 0)
        print(f"  Content Length Stats:")
        print(f"    Average: {avg_length:.1f} characters")
        print(f"    Min: {metrics.get('min_content_length', 0):.1f} characters")
        print(f"    Max: {metrics.get('max_content_length', 0):.1f} characters")

    if results['recommendations']:
        print(f"\nRecommendations:")
        for rec in results['recommendations']:
            print(f"  ⚠️  {rec}")

    if results['validation_errors']:
        print(f"\nValidation Errors (showing first 10):")
        for error in results['validation_errors'][:10]:
            print(f"  ❌ {error}")
        if len(results['validation_errors']) > 10:
            print(f"  ... and {len(results['validation_errors']) - 10} more errors")

    # Save results
    output_dir = Path("ai/dataset_pipeline/final_output")
    output_dir.mkdir(exist_ok=True)
    results_file = output_dir / "dataset_validation_report.json"

    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\nValidation report saved to: {results_file}")

    return results['overall_status'] == 'PASSED'

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)