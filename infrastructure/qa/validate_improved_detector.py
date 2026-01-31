#!/usr/bin/env python3
"""
Validation script for improved crisis detector
Tests the enhanced model against the full validation suite
"""

import asyncio
import sys
from pathlib import Path

# Add the qa directory to the path
sys.path.append(str(Path(__file__).parent))

from improved_crisis_detector import enhanced_model_predictor
from safety_accuracy_validator_simple import EnterpriseSafetyAccuracyValidator


async def main():
    """Run full validation with improved crisis detector"""


    # Initialize validator
    validator = EnterpriseSafetyAccuracyValidator()

    # Run validation with improved detector
    result = await validator.validate_safety_accuracy(enhanced_model_predictor)

    # Save results
    _, _ = validator.save_validation_results(result, "improved_validation_results")

    # Print detailed comparison

    # Show next steps
    if result.overall_accuracy >= 95 and result.false_negative_rate < 1:
        # TODO: Add implementation
        pass
    else:
        if result.overall_accuracy < 95:
            # TODO: Add implementation
            pass
        if result.false_negative_rate >= 1:
            # TODO: Add implementation
            pass
        if result.false_positive_rate >= 5:
            # TODO: Add implementation
            pass



if __name__ == "__main__":
    asyncio.run(main())
