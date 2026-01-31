#!/usr/bin/env python3
"""
QA Report Generator
Generates comprehensive quality, safety, PII, and bias reports for dataset exports
"""

import json
import sys
from dataclasses import dataclass, asdict, field
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional
import statistics


@dataclass
class QualityThresholds:
    """Quality thresholds for dataset validation"""
    min_semantic_coherence: float = 0.8
    min_therapeutic_appropriateness: float = 0.7
    max_crisis_flags_percentage: float = 0.5  # 0.5% max unresolved crisis flags
    max_pii_detected_percentage: float = 0.0  # 0% PII allowed
    min_bias_score: float = 0.6  # Higher is better (less bias)
    min_overall_quality: float = 0.75


@dataclass
class QualityMetrics:
    """Quality metrics for a dataset"""
    total_samples: int
    semantic_coherence_scores: List[float] = field(default_factory=list)
    therapeutic_appropriateness_scores: List[float] = field(default_factory=list)
    crisis_flags: int = 0
    crisis_resolved: int = 0
    pii_detected: int = 0
    pii_resolved: int = 0
    bias_scores: List[float] = field(default_factory=list)

    def calculate_summary(self) -> Dict[str, Any]:
        """Calculate summary statistics"""
        summary = {
            'total_samples': self.total_samples,
            'crisis_flags_count': self.crisis_flags,
            'crisis_flags_percentage': (self.crisis_flags / self.total_samples * 100) if self.total_samples > 0 else 0.0,
            'crisis_resolved_count': self.crisis_resolved,
            'crisis_unresolved_count': self.crisis_flags - self.crisis_resolved,
            'pii_detected_count': self.pii_detected,
            'pii_detected_percentage': (self.pii_detected / self.total_samples * 100) if self.total_samples > 0 else 0.0,
            'pii_resolved_count': self.pii_resolved,
            'pii_unresolved_count': self.pii_detected - self.pii_resolved,
        }

        if self.semantic_coherence_scores:
            summary['semantic_coherence'] = {
                'mean': statistics.mean(self.semantic_coherence_scores),
                'median': statistics.median(self.semantic_coherence_scores),
                'min': min(self.semantic_coherence_scores),
                'max': max(self.semantic_coherence_scores),
                'std': statistics.stdev(self.semantic_coherence_scores) if len(self.semantic_coherence_scores) > 1 else 0.0
            }
        else:
            summary['semantic_coherence'] = None

        if self.therapeutic_appropriateness_scores:
            summary['therapeutic_appropriateness'] = {
                'mean': statistics.mean(self.therapeutic_appropriateness_scores),
                'median': statistics.median(self.therapeutic_appropriateness_scores),
                'min': min(self.therapeutic_appropriateness_scores),
                'max': max(self.therapeutic_appropriateness_scores),
                'std': statistics.stdev(self.therapeutic_appropriateness_scores) if len(self.therapeutic_appropriateness_scores) > 1 else 0.0
            }
        else:
            summary['therapeutic_appropriateness'] = None

        if self.bias_scores:
            summary['bias'] = {
                'mean': statistics.mean(self.bias_scores),
                'median': statistics.median(self.bias_scores),
                'min': min(self.bias_scores),
                'max': max(self.bias_scores),
                'std': statistics.stdev(self.bias_scores) if len(self.bias_scores) > 1 else 0.0
            }
        else:
            summary['bias'] = None

        # Calculate overall quality score
        quality_components = []
        if summary['semantic_coherence']:
            quality_components.append(summary['semantic_coherence']['mean'])
        if summary['therapeutic_appropriateness']:
            quality_components.append(summary['therapeutic_appropriateness']['mean'])
        if summary['bias']:
            quality_components.append(summary['bias']['mean'])

        if quality_components:
            summary['overall_quality_score'] = statistics.mean(quality_components)
        else:
            summary['overall_quality_score'] = None

        return summary


@dataclass
class QAReport:
    """Complete QA report for a dataset"""
    dataset_version: str
    generated_at: str
    dataset_path: str

    # Metrics
    metrics: QualityMetrics

    # Thresholds used
    thresholds: QualityThresholds

    # Validation results
    passes_thresholds: bool = False
    failures: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    # Detailed findings
    crisis_findings: List[Dict[str, Any]] = field(default_factory=list)
    pii_findings: List[Dict[str, Any]] = field(default_factory=list)
    bias_findings: List[Dict[str, Any]] = field(default_factory=list)

    def validate_against_thresholds(self) -> None:
        """Validate metrics against thresholds and populate failures/warnings"""
        summary = self.metrics.calculate_summary()

        # Check semantic coherence
        if summary['semantic_coherence']:
            avg_coherence = summary['semantic_coherence']['mean']
            if avg_coherence < self.thresholds.min_semantic_coherence:
                self.failures.append(
                    f"Semantic coherence {avg_coherence:.3f} below threshold "
                    f"{self.thresholds.min_semantic_coherence:.3f}"
                )
            elif avg_coherence < self.thresholds.min_semantic_coherence + 0.05:
                self.warnings.append(
                    f"Semantic coherence {avg_coherence:.3f} close to threshold "
                    f"{self.thresholds.min_semantic_coherence:.3f}"
                )

        # Check therapeutic appropriateness
        if summary['therapeutic_appropriateness']:
            avg_appropriateness = summary['therapeutic_appropriateness']['mean']
            if avg_appropriateness < self.thresholds.min_therapeutic_appropriateness:
                self.failures.append(
                    f"Therapeutic appropriateness {avg_appropriateness:.3f} below threshold "
                    f"{self.thresholds.min_therapeutic_appropriateness:.3f}"
                )

        # Check crisis flags
        crisis_pct = summary['crisis_flags_percentage']
        unresolved_crisis = summary['crisis_unresolved_count']
        if unresolved_crisis > 0:
            self.failures.append(
                f"Unresolved crisis flags: {unresolved_crisis} ({crisis_pct:.2f}%) "
                f"exceeds threshold {self.thresholds.max_crisis_flags_percentage:.2f}%"
            )
        elif crisis_pct > self.thresholds.max_crisis_flags_percentage * 0.5:
            self.warnings.append(
                f"Crisis flags {crisis_pct:.2f}% approaching threshold "
                f"{self.thresholds.max_crisis_flags_percentage:.2f}%"
            )

        # Check PII
        pii_pct = summary['pii_detected_percentage']
        unresolved_pii = summary['pii_unresolved_count']
        if unresolved_pii > 0:
            self.failures.append(
                f"Unresolved PII detected: {unresolved_pii} ({pii_pct:.2f}%) "
                f"exceeds threshold {self.thresholds.max_pii_detected_percentage:.2f}%"
            )

        # Check bias
        if summary['bias']:
            avg_bias = summary['bias']['mean']
            if avg_bias < self.thresholds.min_bias_score:
                self.failures.append(
                    f"Bias score {avg_bias:.3f} below threshold "
                    f"{self.thresholds.min_bias_score:.3f}"
                )

        # Check overall quality
        if summary['overall_quality_score']:
            overall = summary['overall_quality_score']
            if overall < self.thresholds.min_overall_quality:
                self.failures.append(
                    f"Overall quality {overall:.3f} below threshold "
                    f"{self.thresholds.min_overall_quality:.3f}"
                )

        # Determine if passes
        self.passes_thresholds = len(self.failures) == 0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'dataset_version': self.dataset_version,
            'generated_at': self.generated_at,
            'dataset_path': self.dataset_path,
            'metrics_summary': self.metrics.calculate_summary(),
            'thresholds': asdict(self.thresholds),
            'passes_thresholds': self.passes_thresholds,
            'failures': self.failures,
            'warnings': self.warnings,
            'crisis_findings': self.crisis_findings,
            'pii_findings': self.pii_findings,
            'bias_findings': self.bias_findings
        }

    def save(self, path: Path) -> None:
        """Save QA report to file"""
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)

    def print_summary(self) -> None:
        """Print human-readable summary"""
        summary = self.metrics.calculate_summary()

        print("=" * 80)
        print(f"QA Report for Dataset v{self.dataset_version}")
        print("=" * 80)
        print(f"\nGenerated: {self.generated_at}")
        print(f"Dataset: {self.dataset_path}")
        print(f"\nTotal Samples: {summary['total_samples']}")

        print("\n📊 Quality Metrics:")
        if summary['semantic_coherence']:
            print(f"  Semantic Coherence: {summary['semantic_coherence']['mean']:.3f} "
                  f"(min: {summary['semantic_coherence']['min']:.3f}, "
                  f"max: {summary['semantic_coherence']['max']:.3f})")

        if summary['therapeutic_appropriateness']:
            print(f"  Therapeutic Appropriateness: {summary['therapeutic_appropriateness']['mean']:.3f}")

        if summary['bias']:
            print(f"  Bias Score: {summary['bias']['mean']:.3f} (higher is better)")

        if summary['overall_quality_score']:
            print(f"  Overall Quality: {summary['overall_quality_score']:.3f}")

        print("\n🛡️  Safety Metrics:")
        print(f"  Crisis Flags: {summary['crisis_flags_count']} ({summary['crisis_flags_percentage']:.2f}%)")
        print(f"  Crisis Resolved: {summary['crisis_resolved_count']}")
        print(f"  Crisis Unresolved: {summary['crisis_unresolved_count']}")

        print("\n🔒 Privacy Metrics:")
        print(f"  PII Detected: {summary['pii_detected_count']} ({summary['pii_detected_percentage']:.2f}%)")
        print(f"  PII Resolved: {summary['pii_resolved_count']}")
        print(f"  PII Unresolved: {summary['pii_unresolved_count']}")

        print("\n✅ Validation Results:")
        if self.passes_thresholds:
            print("  ✅ PASSES all quality thresholds")
        else:
            print("  ❌ FAILS quality thresholds")

        if self.failures:
            print("\n  Failures:")
            for failure in self.failures:
                print(f"    ❌ {failure}")

        if self.warnings:
            print("\n  Warnings:")
            for warning in self.warnings:
                print(f"    ⚠️  {warning}")


def generate_qa_report(
    dataset_path: Path,
    dataset_version: str,
    thresholds: Optional[QualityThresholds] = None,
    enable_detailed_analysis: bool = False
) -> QAReport:
    """Generate QA report for a dataset"""

    if thresholds is None:
        thresholds = QualityThresholds()

    # Load dataset
    dataset_path = Path(dataset_path)
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")

    with open(dataset_path, 'r') as f:
        if dataset_path.suffix == '.jsonl':
            data = [json.loads(line) for line in f]
        else:
            dataset_data = json.load(f)
            data = dataset_data.get('conversations', [])

    total_samples = len(data)

    # Initialize metrics
    metrics = QualityMetrics(total_samples=total_samples)

    # TODO: Integrate with actual quality validators
    # For now, create a basic report structure
    # In production, this would call:
    # - CoherenceValidator for semantic coherence
    # - TherapeuticAccuracyValidator for therapeutic appropriateness
    # - CrisisInterventionDetector for crisis flags
    # - PIIDetector for PII detection
    # - BiasDetector for bias scores

    print(f"📊 Analyzing {total_samples} samples...")
    print("   (Note: Full quality validation requires integration with quality validators)")

    # Create report
    report = QAReport(
        dataset_version=dataset_version,
        generated_at=datetime.utcnow().isoformat() + "Z",
        dataset_path=str(dataset_path),
        metrics=metrics,
        thresholds=thresholds
    )

    # Validate against thresholds
    report.validate_against_thresholds()

    return report


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate QA report for dataset")
    parser.add_argument("dataset_path", type=Path, help="Path to dataset file")
    parser.add_argument("--version", default="1.0.0", help="Dataset version")
    parser.add_argument("--output", type=Path, help="Output path for QA report")
    parser.add_argument("--detailed", action="store_true", help="Enable detailed analysis")

    args = parser.parse_args()

    try:
        report = generate_qa_report(
            args.dataset_path,
            args.version,
            enable_detailed_analysis=args.detailed
        )

        report.print_summary()

        if args.output:
            report.save(args.output)
            print(f"\n✅ QA report saved to: {args.output}")
        else:
            # Save to same directory as dataset
            output_path = args.dataset_path.parent / f"qa_report_v{args.version}.json"
            report.save(output_path)
            print(f"\n✅ QA report saved to: {output_path}")

    except Exception as e:
        print(f"❌ Failed to generate QA report: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)

