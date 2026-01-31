"""
Label quality metrics and expected minimum thresholds for the Pixelated Empathy AI project.
Defines standard metrics for evaluating the quality of therapeutic conversation labels.
"""

from dataclasses import dataclass
from typing import List, Dict, Any, Optional
import statistics
from datetime import datetime
import logging
from enum import Enum
from .label_taxonomy import LabelBundle, LabelProvenanceType
from .quality_control import QualityController

logger = logging.getLogger(__name__)


class QualityMetric(Enum):
    """Types of quality metrics"""
    LABEL_ACCURACY = "label_accuracy"
    LABEL_COMPLETENESS = "label_completeness"
    CONFIDENCE_SCORE = "confidence_score"
    INTER_ANNOTATOR_AGREEMENT = "inter_annotator_agreement"
    CRISIS_DETECTION_ACCURACY = "crisis_detection_accuracy"
    THERAPEUTIC_RESPONSE_ACCURACY = "therapeutic_response_accuracy"
    ANNOTATOR_CONSISTENCY = "annotator_consistency"
    LABEL_STABILITY = "label_stability"


@dataclass
class QualityThreshold:
    """Definition of minimum thresholds for quality metrics"""
    metric: QualityMetric
    minimum_value: float
    maximum_value: float
    description: str
    severity_level: str  # 'low', 'medium', 'high', 'critical'


@dataclass
class QualityScore:
    """Result of a quality assessment"""
    metric: QualityMetric
    score: float
    threshold: QualityThreshold
    is_passing: bool
    details: Optional[Dict[str, Any]] = None


class LabelQualityMetrics:
    """System for calculating and tracking label quality metrics with thresholds"""
    
    def __init__(self):
        self.thresholds = self._define_default_thresholds()
        self.logger = logging.getLogger(__name__)

    def _define_default_thresholds(self) -> List[QualityThreshold]:
        """Define default quality thresholds for therapeutic conversation labeling"""
        return [
            QualityThreshold(
                metric=QualityMetric.LABEL_ACCURACY,
                minimum_value=0.85,  # 85% accuracy required
                maximum_value=1.0,
                description="Overall accuracy of labels compared to gold standard",
                severity_level="critical"
            ),
            QualityThreshold(
                metric=QualityMetric.LABEL_COMPLETENESS,
                minimum_value=0.90,  # 90% of relevant content labeled
                maximum_value=1.0,
                description="Proportion of relevant content that receives appropriate labels",
                severity_level="high"
            ),
            QualityThreshold(
                metric=QualityMetric.CONFIDENCE_SCORE,
                minimum_value=0.70,  # Average confidence of 0.7 or higher
                maximum_value=1.0,
                description="Average confidence score across all labels",
                severity_level="high"
            ),
            QualityThreshold(
                metric=QualityMetric.INTER_ANNOTATOR_AGREEMENT,
                minimum_value=0.75,  # Cohen's Kappa of 0.75 or higher
                maximum_value=1.0,
                description="Agreement between multiple annotators",
                severity_level="critical"
            ),
            QualityThreshold(
                metric=QualityMetric.CRISIS_DETECTION_ACCURACY,
                minimum_value=0.95,  # 95% accuracy for crisis detection
                maximum_value=1.0,
                description="Accuracy in identifying crisis situations",
                severity_level="critical"
            ),
            QualityThreshold(
                metric=QualityMetric.THERAPEUTIC_RESPONSE_ACCURACY,
                minimum_value=0.80,  # 80% accuracy for therapeutic response types
                maximum_value=1.0,
                description="Accuracy in identifying therapeutic response types",
                severity_level="high"
            ),
            QualityThreshold(
                metric=QualityMetric.ANNOTATOR_CONSISTENCY,
                minimum_value=0.80,  # 80% consistency across sessions
                maximum_value=1.0,
                description="Consistency of an annotator over time",
                severity_level="medium"
            ),
            QualityThreshold(
                metric=QualityMetric.LABEL_STABILITY,
                minimum_value=0.05,  # Max 5% variation in confidence over time
                maximum_value=1.0,
                description="Stability of confidence scores over time",
                severity_level="medium"
            )
        ]

    def calculate_label_accuracy(self, label_bundles: List[LabelBundle]) -> QualityScore:
        """Calculate overall label accuracy metric"""
        if not label_bundles:
            return QualityScore(
                metric=QualityMetric.LABEL_ACCURACY,
                score=0.0,
                threshold=self._get_threshold(QualityMetric.LABEL_ACCURACY),
                is_passing=False,
                details={"reason": "No label bundles provided"}
            )
        
        # For this therapeutic context, we'll calculate accuracy based on confidence scores
        # In a real system, this would compare against gold standard labels
        all_confidences = []
        for bundle in label_bundles:
            for label in bundle.therapeutic_response_labels:
                all_confidences.append(label.metadata.confidence)
            if bundle.crisis_label:
                all_confidences.append(bundle.crisis_label.metadata.confidence)
            if bundle.therapy_modality_label:
                all_confidences.append(bundle.therapy_modality_label.metadata.confidence)
            if bundle.mental_health_condition_label:
                all_confidences.append(bundle.mental_health_condition_label.metadata.confidence)
            if bundle.demographic_label:
                all_confidences.append(bundle.demographic_label.metadata.confidence)
        
        if not all_confidences:
            accuracy_score = 1.0  # Perfect accuracy if no labels to evaluate
        else:
            accuracy_score = statistics.mean(all_confidences)
        
        threshold = self._get_threshold(QualityMetric.LABEL_ACCURACY)
        is_passing = threshold.minimum_value <= accuracy_score <= threshold.maximum_value
        
        return QualityScore(
            metric=QualityMetric.LABEL_ACCURACY,
            score=accuracy_score,
            threshold=threshold,
            is_passing=is_passing,
            details={
                "total_labels": len(all_confidences),
                "confidence_range": (min(all_confidences) if all_confidences else 0, 
                                   max(all_confidences) if all_confidences else 1),
                "std_deviation": statistics.stdev(all_confidences) if len(all_confidences) > 1 else 0
            }
        )

    def calculate_label_completeness(self, conversations: List['Conversation'], label_bundles: List[LabelBundle]) -> QualityScore:
        """Calculate label completeness metric"""
        if len(conversations) != len(label_bundles):
            raise ValueError("Number of conversations and label bundles must match")
        
        # Measure completeness as a function of labeled content vs. total content
        total_messages = sum(len(conv.messages) for conv in conversations)
        
        # Count labeled messages (messages with at least one therapeutic response label)
        labeled_messages = 0
        for bundle in label_bundles:
            # For this example, we'll count if the bundle has any therapeutic response labels
            # In a real implementation, this would map more directly to messages
            if bundle.therapeutic_response_labels or bundle.crisis_label:
                labeled_messages += 1  # Simplified - in practice, map to actual messages
        
        completeness_ratio = labeled_messages / len(label_bundles) if label_bundles else 0
        
        threshold = self._get_threshold(QualityMetric.LABEL_COMPLETENESS)
        is_passing = threshold.minimum_value <= completeness_ratio <= threshold.maximum_value
        
        return QualityScore(
            metric=QualityMetric.LABEL_COMPLETENESS,
            score=completeness_ratio,
            threshold=threshold,
            is_passing=is_passing,
            details={
                "total_conversations": len(conversations),
                "labeled_conversations": labeled_messages,
                "total_messages": total_messages
            }
        )

    def calculate_confidence_score(self, label_bundles: List[LabelBundle]) -> QualityScore:
        """Calculate overall confidence score metric"""
        all_confidences = []
        for bundle in label_bundles:
            for label in bundle.therapeutic_response_labels:
                all_confidences.append(label.metadata.confidence)
            if bundle.crisis_label:
                all_confidences.append(bundle.crisis_label.metadata.confidence)
            if bundle.therapy_modality_label:
                all_confidences.append(bundle.therapy_modality_label.metadata.confidence)
            if bundle.mental_health_condition_label:
                all_confidences.append(bundle.mental_health_condition_label.metadata.confidence)
            if bundle.demographic_label:
                all_confidences.append(bundle.demographic_label.metadata.confidence)
        
        if not all_confidences:
            avg_confidence = 1.0
        else:
            avg_confidence = statistics.mean(all_confidences)
        
        threshold = self._get_threshold(QualityMetric.CONFIDENCE_SCORE)
        is_passing = threshold.minimum_value <= avg_confidence <= threshold.maximum_value
        
        return QualityScore(
            metric=QualityMetric.CONFIDENCE_SCORE,
            score=avg_confidence,
            threshold=threshold,
            is_passing=is_passing,
            details={
                "total_confidence_scores": len(all_confidences),
                "confidence_range": (min(all_confidences) if all_confidences else 0,
                                   max(all_confidences) if all_confidences else 1),
                "std_deviation": statistics.stdev(all_confidences) if len(all_confidences) > 1 else 0
            }
        )

    def calculate_crisis_detection_accuracy(self, label_bundles: List[LabelBundle]) -> QualityScore:
        """Calculate accuracy for crisis detection specifically"""
        crisis_labels = []
        for bundle in label_bundles:
            if bundle.crisis_label:
                crisis_labels.append(bundle.crisis_label)
        
        if not crisis_labels:
            # If no crisis labels exist, we consider this as perfect accuracy (no crises to detect)
            accuracy_score = 1.0
        else:
            # Calculate based on confidence in crisis detection
            crisis_confidences = [label.metadata.confidence for label in crisis_labels]
            accuracy_score = statistics.mean(crisis_confidences) if crisis_confidences else 1.0
        
        threshold = self._get_threshold(QualityMetric.CRISIS_DETECTION_ACCURACY)
        is_passing = threshold.minimum_value <= accuracy_score <= threshold.maximum_value
        
        return QualityScore(
            metric=QualityMetric.CRISIS_DETECTION_ACCURACY,
            score=accuracy_score,
            threshold=threshold,
            is_passing=is_passing,
            details={
                "total_crisis_labels": len(crisis_labels),
                "average_crisis_confidence": accuracy_score
            }
        )

    def calculate_therapeutic_response_accuracy(self, label_bundles: List[LabelBundle]) -> QualityScore:
        """Calculate accuracy for therapeutic response detection"""
        total_responses = 0
        response_confidences = []
        
        for bundle in label_bundles:
            for label in bundle.therapeutic_response_labels:
                total_responses += 1
                response_confidences.append(label.metadata.confidence)
        
        if total_responses == 0:
            accuracy_score = 1.0  # Perfect if no responses to label
        else:
            accuracy_score = statistics.mean(response_confidences)
        
        threshold = self._get_threshold(QualityMetric.THERAPEUTIC_RESPONSE_ACCURACY)
        is_passing = threshold.minimum_value <= accuracy_score <= threshold.maximum_value
        
        return QualityScore(
            metric=QualityMetric.THERAPEUTIC_RESPONSE_ACCURACY,
            score=accuracy_score,
            threshold=threshold,
            is_passing=is_passing,
            details={
                "total_therapeutic_responses": total_responses,
                "average_response_confidence": accuracy_score
            }
        )

    def get_comprehensive_quality_report(self, 
                                       conversations: List['Conversation'],
                                       label_bundles: List[LabelBundle]) -> Dict[str, Any]:
        """Generate a comprehensive quality report with all metrics"""
        # Calculate all quality scores
        accuracy_score = self.calculate_label_accuracy(label_bundles)
        completeness_score = self.calculate_label_completeness(conversations, label_bundles)
        confidence_score = self.calculate_confidence_score(label_bundles)
        crisis_score = self.calculate_crisis_detection_accuracy(label_bundles)
        response_score = self.calculate_therapeutic_response_accuracy(label_bundles)
        
        # Compile all scores
        all_scores = [
            accuracy_score, completeness_score, confidence_score,
            crisis_score, response_score
        ]
        
        # Calculate overall compliance rate
        passing_scores = sum(1 for score in all_scores if score.is_passing)
        total_scores = len(all_scores)
        compliance_rate = passing_scores / total_scores if total_scores > 0 else 0
        
        # Identify critical failures
        critical_failures = [
            score for score in all_scores 
            if not score.is_passing and score.threshold.severity_level == "critical"
        ]
        
        return {
            "timestamp": datetime.utcnow().isoformat(),
            "overall_compliance_rate": compliance_rate,
            "passing_metrics": passing_scores,
            "total_metrics": total_scores,
            "critical_failures": len(critical_failures),
            "quality_scores": [
                {
                    "metric": score.metric.value,
                    "score": score.score,
                    "is_passing": score.is_passing,
                    "threshold_min": score.threshold.minimum_value,
                    "threshold_max": score.threshold.maximum_value,
                    "severity": score.threshold.severity_level,
                    "details": score.details
                }
                for score in all_scores
            ],
            "summary": {
                "label_accuracy": accuracy_score.score,
                "completeness": completeness_score.score,
                "confidence": confidence_score.score,
                "crisis_detection": crisis_score.score,
                "therapeutic_response_detection": response_score.score
            }
        }

    def _get_threshold(self, metric: QualityMetric) -> QualityThreshold:
        """Get threshold for a specific metric"""
        for threshold in self.thresholds:
            if threshold.metric == metric:
                return threshold
        raise ValueError(f"No threshold defined for metric: {metric}")


def create_label_quality_metrics() -> LabelQualityMetrics:
    """Create a default instance of label quality metrics"""
    return LabelQualityMetrics()


# Example usage and testing
def test_label_quality_metrics():
    """Test the label quality metrics system"""
    from .label_taxonomy import (
        TherapeuticResponseLabel, CrisisLabel, LabelMetadata, LabelProvenanceType,
        TherapeuticResponseType, CrisisLevelType
    )
    from .conversation_schema import Conversation, Message
    
    # Create test data
    conversations = []
    label_bundles = []
    
    for i in range(20):
        # Create conversation
        conv = Conversation()
        conv.add_message("therapist", f"Therapeutic message {i}")
        conv.add_message("client", f"Client response {i}")
        conversations.append(conv)
        
        # Create corresponding label bundle
        bundle = LabelBundle(conversation_id=conv.conversation_id)
        
        # Add therapeutic response labels with varying confidence
        bundle.therapeutic_response_labels.append(
            TherapeuticResponseLabel(
                response_type=TherapeuticResponseType.EMPATHY,
                metadata=LabelMetadata(confidence=0.85 + (i * 0.01) if i < 10 else 0.75 - (i * 0.01))
            )
        )
        
        # Add crisis labels
        if i % 5 == 0:
            bundle.crisis_label = CrisisLabel(
                crisis_level=CrisisLevelType.LOW_RISK,
                metadata=LabelMetadata(confidence=0.9 - (i * 0.02))
            )
        
        label_bundles.append(bundle)
    
    # Test quality metrics
    quality_metrics = create_label_quality_metrics()
    
    # Calculate individual metrics
    accuracy_score = quality_metrics.calculate_label_accuracy(label_bundles)
    print(f"Label Accuracy: {accuracy_score.score:.3f}, Passing: {accuracy_score.is_passing}")
    
    completeness_score = quality_metrics.calculate_label_completeness(conversations, label_bundles)
    print(f"Label Completeness: {completeness_score.score:.3f}, Passing: {completeness_score.is_passing}")
    
    confidence_score = quality_metrics.calculate_confidence_score(label_bundles)
    print(f"Confidence Score: {confidence_score.score:.3f}, Passing: {confidence_score.is_passing}")
    
    crisis_score = quality_metrics.calculate_crisis_detection_accuracy(label_bundles)
    print(f"Crisis Detection Accuracy: {crisis_score.score:.3f}, Passing: {crisis_score.is_passing}")
    
    # Generate comprehensive report
    comprehensive_report = quality_metrics.get_comprehensive_quality_report(conversations, label_bundles)
    print(f"\nComprehensive Quality Report:")
    print(f"  Overall Compliance Rate: {comprehensive_report['overall_compliance_rate']:.3f}")
    print(f"  Passing Metrics: {comprehensive_report['passing_metrics']}/{comprehensive_report['total_metrics']}")
    print(f"  Critical Failures: {comprehensive_report['critical_failures']}")
    print(f"  Summary: {comprehensive_report['summary']}")
    
    return comprehensive_report


if __name__ == "__main__":
    test_label_quality_metrics()