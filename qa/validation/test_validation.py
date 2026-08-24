#!/usr/bin/env python3
"""
Quick test script to verify dataset validation module works correctly
Run this to test the validation system before deploying
"""

import json
import sys
from pathlib import Path

from ai.qa.validation.dataset_validation import DatasetValidator, validate_jsonl_file

# Test cases
TEST_CASES = {
    "valid_crisis": {
        "scenario_id": "crisis_001",
        "edge_case_type": "crisis_intervention",
        "severity_level": "critical",
        "description": "Client expressing suicidal ideation",
        "expected": "PASS",
    },
    "stereotype_cultural": {
        "scenario_id": "bad_001",
        "edge_case_type": "cultural_sensitivity",
        "severity_level": "high",
        "description": "Client from Asian background - naturally hardworking model minority",
        "expected": "FAIL",
    },
    "offensive_generalization": {
        "scenario_id": "bad_002",
        "edge_case_type": "ethical_dilemma",
        "severity_level": "high",
        "description": "All women are too emotional and irrational to make good decisions",
        "expected": "FAIL",
    },
    "problematic_response": {
        "scenario_id": "bad_003",
        "edge_case_type": "boundary_violation",
        "severity_level": "moderate",
        "description": "Therapist: You're overreacting, that's not real trauma",
        "expected": "FAIL",
    },
    "valid_ethical": {
        "scenario_id": "ethical_001",
        "edge_case_type": "ethical_dilemma",
        "severity_level": "high",
        "description": "Conflicting duty to warn vs. confidentiality in threatening situation",
        "expected": "PASS",
    },
}


def test_validation():
    """Test the validation module"""
    try:
        pass
    except ImportError:
        return False

    validator = DatasetValidator(strict_mode=False)

    results = {"passed": 0, "failed": 0, "tests": []}

    for test_name, test_data in TEST_CASES.items():
        expected = test_data.pop("expected")
        result = validator.validate_edge_case(test_data)

        is_valid = result.is_valid
        expected_valid = expected == "PASS"

        test_passed = is_valid == expected_valid

        results["tests"].append(
            {
                "name": test_name,
                "expected": expected,
                "actual": "PASS" if is_valid else "FAIL",
                "passed": test_passed,
            }
        )

        if test_passed:
            results["passed"] += 1
        else:
            results["failed"] += 1

        if not is_valid:
            if result.errors:
                pass
            if result.bias_indicators:
                pass

        if result.warnings:
            pass

    return not results["failed"] > 0


def test_batch_validation():
    """Test batch validation"""
    try:
        pass
    except ImportError:
        return False

    validator = DatasetValidator(strict_mode=False)

    batch = [
        {
            "scenario_id": f"test_{i:03d}",
            "edge_case_type": "crisis_intervention",
            "severity_level": "high",
            "description": f"Test scenario {i}",
        }
        for i in range(5)
    ]

    # Add one bad case
    batch.append(
        {
            "scenario_id": "bad_batch",
            "edge_case_type": "cultural_sensitivity",
            "severity_level": "high",
            "description": "All lazy people lack motivation",
        }
    )

    result = validator.validate_batch(batch)

    if result["bias_summary"]:
        pass

    return result["invalid"] == 1 and result["valid"] == 5


def test_file_validation():
    """Test JSONL file validation"""
    try:
        pass
    except ImportError:
        return False

    # Create test JSONL file
    test_file = Path("/tmp/test_validation.jsonl")
    with open(test_file, "w") as f:
        f.write(
            json.dumps(
                {
                    "scenario_id": "test",
                    "edge_case_type": "crisis",
                    "severity_level": "high",
                    "description": "Test",
                }
            )
            + "\n"
        )
        f.write(
            json.dumps(
                {
                    "scenario_id": "bad",
                    "edge_case_type": "culture",
                    "severity_level": "high",
                    "description": "All asians are model minorities",
                }
            )
            + "\n"
        )

    result = validate_jsonl_file(str(test_file), strict_mode=False)

    test_file.unlink()

    return result.get("invalid", 0) == 1


def main():
    """Run all tests"""

    all_passed = True

    if not test_validation():
        all_passed = False

    if not test_batch_validation():
        all_passed = False

    if not test_file_validation():
        all_passed = False

    if all_passed:
        return 0
    return 1


if __name__ == "__main__":
    sys.exit(main())
