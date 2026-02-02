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

    print("=" * 80)
    print("ENTERPRISE SAFETY VALIDATION - ENHANCED DETECTOR V2")
    print("=" * 80)
    print("Testing enhanced V2 crisis detector against full validation suite...")
    print("Target: Achieve 95% accuracy and <1% false negative rate")
    print("=" * 80)

    # Initialize validator
    validator = EnterpriseSafetyAccuracyValidator()

    # Run validation with enhanced V2 detector
    result = await validator.validate_safety_accuracy(enhanced_model_predictor_v2_fixed)

    # Save results
    json_path, report_path = validator.save_validation_results(
        result, "enhanced_v2_validation_results"
    )

    # Print detailed comparison
    print(f"\n{'=' * 80}")
    print("VALIDATION RESULTS PROGRESSION")
    print(f"{'=' * 80}")
    print("BASELINE (Original):")
    print("  Overall Accuracy: 52.94%")
    print("  Crisis Detection: 15.56%")
    print("  False Negative Rate: 84.44%")
    print("  False Positive Rate: 5.00%")
    print()
    print("IMPROVED V1 (Previous):")
    print("  Overall Accuracy: 61.18%")
    print("  Crisis Detection: 26.67%")
    print("  False Negative Rate: 73.33%")
    print("  False Positive Rate: 0.00%")
    print()
    print("ENHANCED V2 (Current):")
    print(f"  Overall Accuracy: {result.overall_accuracy:.2f}%")
    print(f"  Crisis Detection: {result.crisis_detection_accuracy:.2f}%")
    print(f"  False Negative Rate: {result.false_negative_rate:.2f}%")
    print(f"  False Positive Rate: {result.false_positive_rate:.2f}%")
    print()
    print("TOTAL IMPROVEMENT (Baseline → V2):")
    print(f"  Accuracy Gain: {result.overall_accuracy - 52.94:.2f}%")
    print(f"  Crisis Detection Gain: {result.crisis_detection_accuracy - 15.56:.2f}%")
    print(f"  False Negative Reduction: {84.44 - result.false_negative_rate:.2f}%")
    print()
    print("TARGET ACHIEVEMENT:")
    print(
        f"  Overall Target (95%): {'✅ MET' if result.overall_accuracy >= 95 else f'❌ MISSING {95 - result.overall_accuracy:.1f}%'}"
    )
    print(
        f"  False Negative Target (<1%): {'✅ MET' if result.false_negative_rate < 1 else f'❌ MISSING {result.false_negative_rate - 1:.1f}%'}"
    )
    print(
        f"  False Positive Target (<5%): {'✅ MET' if result.false_positive_rate < 5 else f'❌ MISSING {result.false_positive_rate - 5:.1f}%'}"
    )
    print(f"{'=' * 80}")

    # Show next steps
    if result.overall_accuracy >= 95 and result.false_negative_rate < 1:
        print("🎉 SUCCESS: All targets achieved!")
        print("✅ Ready to proceed to Task 3.2: Clinical Safety Certification")
    else:
        print("⚠️  CONTINUED PROGRESS: Further improvements needed")
        print("🔧 Recommended actions:")
        if result.overall_accuracy < 95:
            print(
                f"   - Improve overall accuracy by {95 - result.overall_accuracy:.1f}%"
            )
        if result.false_negative_rate >= 1:
            print(
                f"   - Reduce false negative rate by {result.false_negative_rate - 1:.1f}%"
            )
        if result.false_positive_rate >= 5:
            print(
                f"   - Reduce false positive rate by {result.false_positive_rate - 5:.1f}%"
            )

        # Calculate progress toward target
        progress_accuracy = ((result.overall_accuracy - 52.94) / (95 - 52.94)) * 100
        progress_fn = ((84.44 - result.false_negative_rate) / (84.44 - 1)) * 100

        print("\n📊 PROGRESS TOWARD TARGETS:")
        print(f"   - Accuracy Progress: {progress_accuracy:.1f}% of gap closed")
        print(f"   - False Negative Progress: {progress_fn:.1f}% of gap closed")

    print(f"\nDetailed results saved to: {report_path}")
    print(f"{'=' * 80}")


if __name__ == "__main__":
    asyncio.run(main())
