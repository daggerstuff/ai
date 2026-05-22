#!/usr/bin/env python3
"""
Validation script for improved crisis detector
Tests the enhanced model against the full validation suite
"""

import asyncio
import logging
import sys
from pathlib import Path

# Add the qa directory to the path
sys.path.append(str(Path(__file__).parent))


from improved_crisis_detector import enhanced_model_predictor
from safety_accuracy_validator_simple import EnterpriseSafetyAccuracyValidator


async def main():
    """Run full validation with improved crisis detector"""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    logger = logging.getLogger(__name__)

    # Initialize validator
    validator = EnterpriseSafetyAccuracyValidator()

    # Run validation with improved detector
    # Gilfoyle v5: Run potentially blocking CPU-bound tasks in a separate thread
    if asyncio.iscoroutinefunction(validator.validate_safety_accuracy):
        result = await validator.validate_safety_accuracy(enhanced_model_predictor)
    else:
        result = await asyncio.to_thread(validator.validate_safety_accuracy, enhanced_model_predictor)

    # Save results
    _, _ = await asyncio.to_thread(validator.save_validation_results, result, "improved_validation_results")

    # Print detailed comparison
    report = await asyncio.to_thread(validator.generate_validation_report, result)
    logger.info("%s", report)

    # Show next steps
    target_reached = result.overall_accuracy >= 95 and result.false_negative_rate < 1 and result.false_positive_rate < 5
    if target_reached:
        logger.info(
            "\nValidation outcome: PASSED\n"
            "- Target accuracy and error-rate gates met.\n"
            "- Next step: promote improved detector into production routing and open clinical readiness review.\n"
            "- Keep monitoring drift with a weekly reevaluation."
        )
        return
    blockers: list[str] = []
    if result.overall_accuracy < 95:
        blockers.append(
            f"Overall accuracy short by {95 - result.overall_accuracy:.2f}% (current: {result.overall_accuracy:.2f}%)."
        )
    if result.false_negative_rate >= 1:
        blockers.append(f"False-negative rate is too high: {result.false_negative_rate:.2f}% (max allowed: 1%).")
    if result.false_positive_rate >= 5:
        blockers.append(f"False-positive rate is too high: {result.false_positive_rate:.2f}% (max allowed: 5%).")

    logger.error("\nValidation outcome: FAILED")
    logger.error("Blocking issues:")
    for blocker in blockers:
        logger.error("- %s", blocker)
    logger.warning("Recommended remediation:")
    logger.warning("- Review false-negative and false-positive cohorts in the generated JSON report.")
    logger.warning("- Tune crisis pattern weights and context filters in improved_crisis_detector.")
    logger.warning("- Re-run this validation and investigate outlier language/demographic buckets.")
    raise SystemExit(1)


if __name__ == "__main__":
    asyncio.run(main())
