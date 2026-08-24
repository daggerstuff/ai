#!/usr/bin/env python3
"""
Validation script for enhanced V2 crisis detector
Tests against full validation suite to measure improvement
"""

import asyncio
import sys
from pathlib import Path

# Add the qa directory to the path
sys.path.append(str(Path(__file__).parent))


from enhanced_crisis_detector_v2_fixed import enhanced_model_predictor_v2_fixed
from safety_accuracy_validator_simple import EnterpriseSafetyAccuracyValidator


async def main():
    """Run full validation with enhanced V2 crisis detector"""

    # Initialize validator
    validator = EnterpriseSafetyAccuracyValidator()

    # Run validation with enhanced V2 detector
    result = await validator.validate_safety_accuracy(enhanced_model_predictor_v2_fixed)

    # Save results
    _json_path, _report_path = validator.save_validation_results(result, "enhanced_v2_validation_results")

    # Print detailed comparison

    # Show next steps
    if result.overall_accuracy >= 95 and result.false_negative_rate < 1:
        pass
    else:
        if result.overall_accuracy < 95:
            pass
        if result.false_negative_rate >= 1:
            pass
        if result.false_positive_rate >= 5:
            pass

        # Calculate progress toward target
        ((result.overall_accuracy - 52.94) / (95 - 52.94)) * 100
        ((84.44 - result.false_negative_rate) / (84.44 - 1)) * 100


if __name__ == "__main__":
    asyncio.run(main())
