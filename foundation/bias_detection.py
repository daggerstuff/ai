"""
Bias Detection and Mitigation Foundation.

Core module for:
- Demographic representation analysis
- Language bias scoring (cultural, gender, racial)
- Fairness metrics calculation
- Bias mitigation techniques (reweighting, augmentation)
"""

from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional


class BiasCategory(str, Enum):
    """Types of bias to detect."""

    GENDER = "gender"
    RACIAL = "racial"
    CULTURAL = "cultural"
    SOCIOECONOMIC = "socioeconomic"
    RELIGIOUS = "religious"
    DISABILITY = "disability"
    AGE = "age"
    SEXUAL_ORIENTATION = "sexual_orientation"


@dataclass
class BiasMetric:
    """Bias measurement result."""

    category: BiasCategory
    severity: float  # 0-1, where 1 = severe bias
    confidence: float  # 0-1, confidence in measurement
    details: Dict
    mitigation_suggestion: Optional[str] = None


class BiasDetector:
    """Detects bias in therapeutic conversations."""

    def __init__(self):
        # Simple patterns for demonstration; real implementation uses transformers
        self.gender_terms = {
            "male": ["man", "boy", "he", "his", "father", "brother", "son"],
            "female": ["woman", "girl", "she", "her", "mother", "sister", "daughter"],
        }
        self.racial_terms = {
            "african_american": ["black", "african american", "afro"],
            "hispanic": ["hispanic", "latino", "spanish"],
            "asian": ["asian", "chinese", "indian", "japanese"],
            "white": ["white", "caucasian", "european"],
        }

    def analyze_text(self, text: str) -> List[BiasMetric]:
        """Analyze conversation for potential biases."""
        metrics = []

        # Gender bias analysis
        gender_metric = self._analyze_gender_representation(text)
        if gender_metric:
            metrics.append(gender_metric)

        # Racial bias analysis
        racial_metric = self._analyze_racial_representation(text)
        if racial_metric:
            metrics.append(racial_metric)

        return metrics

    def _analyze_gender_representation(self, text: str) -> Optional[BiasMetric]:
        """Check for gender bias in representation."""
        text_lower = text.lower()
        male_count = sum(term in text_lower for term in self.gender_terms["male"])
        female_count = sum(term in text_lower for term in self.gender_terms["female"])

        total = male_count + female_count
        if total < 2:
            return None

        male_ratio = male_count / total
        # Bias if one gender >> 60% representation
        if male_ratio > 0.7 or male_ratio < 0.3:
            severity = min(abs(male_ratio - 0.5) * 2, 1.0)
            return BiasMetric(
                category=BiasCategory.GENDER,
                severity=severity,
                confidence=0.4,  # Low confidence with simple pattern matching
                details={
                    "male_ratio": male_ratio,
                    "female_ratio": 1 - male_ratio,
                    "total_gender_terms": total,
                },
                mitigation_suggestion=(
                    "Ensure balanced gender representation in examples"
                ),
            )
        return None

    def _analyze_racial_representation(self, text: str) -> Optional[BiasMetric]:
        """Check for racial bias in representation."""
        text_lower = text.lower()
        racial_counts = {}

        for racial_group, terms in self.racial_terms.items():
            count = sum(term in text_lower for term in terms)
            if count > 0:
                racial_counts[racial_group] = count

        if not racial_counts:
            return None

        total = sum(racial_counts.values())
        max_group = max(racial_counts.values())

        if max_group / total > 0.7:
            severity = (max_group / total - 0.5) * 2
            return BiasMetric(
                category=BiasCategory.RACIAL,
                severity=min(severity, 1.0),
                confidence=0.35,  # Low confidence, needs transformer model
                details=racial_counts,
                mitigation_suggestion=(
                    "Include diverse racial representations in training data"
                ),
            )
        return None

    def calculate_fairness_metrics(
        self, conversations: List, ground_truth_groups: Dict[str, List[str]]
    ) -> Dict:
        """Calculate fairness metrics across demographic groups."""
        # Placeholder for comprehensive fairness metrics
        # Real implementation uses disparate impact, equalized odds, etc.
        return {
            "demographic_parity": None,
            "equalized_odds": None,
            "disparate_impact": None,
            "calibration": None,
        }


class BiasReporter:
    """Generates bias analysis reports."""

    @staticmethod
    def generate_report(metrics: List[BiasMetric]) -> str:
        """Generate human-readable bias report."""
        lines = ["=== Bias Analysis Report ===\n"]

        if not metrics:
            lines.append("✓ No significant biases detected.")
            return "\n".join(lines)

        for metric in sorted(metrics, key=lambda m: m.severity, reverse=True):
            lines.extend(
                [
                    f"{metric.category.value.upper()}",
                    f"  Severity: {metric.severity:.2%}",
                    f"  Confidence: {metric.confidence:.2%}",
                ]
            )
            if metric.details:
                lines.append(f"  Details: {metric.details}")
            if metric.mitigation_suggestion:
                lines.append(f"  → {metric.mitigation_suggestion}")
            lines.append("")

        return "\n".join(lines)


if __name__ == "__main__":
    detector = BiasDetector()

    # Example therapeutic conversation
    sample_text = (
        "A young man came to therapy about his relationship issues. "
        "He felt his mother was too controlling."
    )

    metrics = detector.analyze_text(sample_text)
    report = BiasReporter.generate_report(metrics)
    print(report)
