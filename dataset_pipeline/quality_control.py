"""
Quality control system for the labeling pipeline.
Implements inter-annotator agreement metrics and label drift monitoring.
"""

from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Tuple
import statistics
from datetime import datetime, timedelta
import logging
from enum import Enum
import math
from .label_taxonomy import (
    LabelBundle, LabelProvenanceType, TherapeuticResponseType, CrisisLevelType,
    TherapyModalityType, MentalHealthConditionType, DemographicType
)
from .label_versioning import LabelVersionManager, LabelHistory, ProvenanceRecord

logger = logging.getLogger(__name__)


class QualityMetricType(Enum):
    """Types of quality metrics"""
    INTER_ANNOTATOR_AGREEMENT = "inter_annotator_agreement"
    LABEL_DRIFT = "label_drift"
    CONFIDENCE_STABILITY = "confidence_stability"
    ANNOTATOR_CONSISTENCY = "annotator_consistency"
    CATEGORY_BALANCE = "category_balance"


@dataclass
class QualityMetric:
    """A single quality metric result"""
    metric_type: QualityMetricType
    value: float
    threshold: float
    is_passing: bool
    timestamp: str
    details: Optional[Dict[str, Any]] = None


@dataclass
class QualityAlert:
    """Alert for quality issues"""
    alert_id: str
    metric_type: QualityMetricType
    severity: str  # 'low', 'medium', 'high', 'critical'
    message: str
    timestamp: str
    details: Optional[Dict[str, Any]] = None


class QualityController:
    """Main quality control system for the labeling pipeline"""
    
    def __init__(self, 
                 iaa_threshold: float = 0.7,  # Minimum acceptable inter-annotator agreement
                 drift_threshold: float = 0.1,  # Maximum acceptable drift
                 confidence_stability_threshold: float = 0.05,  # Max confidence variation
                 category_balance_threshold: float = 0.15,  # Max imbalance in categories
                 minimum_confidence_threshold: float = 0.6,  # Minimum acceptable confidence
                 minimum_annotation_count: int = 10,  # Minimum annotations per annotator
                 agreement_score_threshold: float = 0.75,  # Minimum agreement for approval
                 label_quality_threshold: float = 0.7):  # Minimum label quality score
        self.iaa_threshold = iaa_threshold
        self.drift_threshold = drift_threshold
        self.confidence_stability_threshold = confidence_stability_threshold
        self.category_balance_threshold = category_balance_threshold
        self.minimum_confidence_threshold = minimum_confidence_threshold
        self.minimum_annotation_count = minimum_annotation_count
        self.agreement_score_threshold = agreement_score_threshold
        self.label_quality_threshold = label_quality_threshold
        self.alerts: List[QualityAlert] = []
        self.logger = logging.getLogger(__name__)

    def calculate_cohens_kappa(self, 
                              rater1_labels: List[str], 
                              rater2_labels: List[str]) -> float:
        """
        Calculate Cohen's Kappa for inter-annotator agreement.
        This is a simplified version for categorical labels.
        """
        if len(rater1_labels) != len(rater2_labels):
            raise ValueError("Rater label lists must have the same length")
        
        if len(rater1_labels) == 0:
            return 1.0  # Perfect agreement if no items
        
        # Calculate observed agreement
        observed_agreement = sum(1 for a, b in zip(rater1_labels, rater2_labels) if a == b) / len(rater1_labels)
        
        # Calculate expected agreement by chance
        unique_labels = set(rater1_labels + rater2_labels)
        expected_agreement = 0.0
        
        for label in unique_labels:
            p_rater1 = rater1_labels.count(label) / len(rater1_labels)
            p_rater2 = rater2_labels.count(label) / len(rater2_labels)
            expected_agreement += p_rater1 * p_rater2
        
        if expected_agreement == 1.0:
            return 1.0  # Perfect chance agreement
        
        # Calculate Cohen's Kappa
        kappa = (observed_agreement - expected_agreement) / (1 - expected_agreement)
        return kappa

    def calculate_fleiss_kappa(self, annotations: List[List[Tuple[str, float]]]) -> float:
        """
        Calculate Fleiss' Kappa for multiple annotators.
        annotations: List of (label, confidence) tuples for each item, for each annotator
        """
        if not annotations:
            return 1.0
        
        n_items = len(annotations)
        if n_items == 0:
            return 1.0
        
        # Get all unique labels across all annotations
        all_labels = set()
        for item_annotations in annotations:
            for label, _ in item_annotations:
                all_labels.add(label)
        
        if len(all_labels) == 0:
            return 1.0
        
        # Calculate agreement matrix
        n_annotators = max(len(item_annotations) for item_annotations in annotations)
        agreement_matrix = {}
        
        for item_annotations in annotations:
            for label, _ in item_annotations:
                agreement_matrix[label] = agreement_matrix.get(label, 0) + 1
        
        # Calculate Fleiss' Kappa
        # This is a simplified implementation focusing on label agreement
        total_annotations = sum(len(item_annotations) for item_annotations in annotations)
        if total_annotations == 0:
            return 1.0
        
        # Calculate P_i (proportion of total annotations for each item)
        p_is = []
        for item_annotations in annotations:
            if len(item_annotations) == 0:
                continue
            # Calculate agreement within this item
            if len(item_annotations) > 1:
                item_agreements = sum(1 for i in range(len(item_annotations)) 
                                    for j in range(i+1, len(item_annotations)) 
                                    if item_annotations[i][0] == item_annotations[j][0])
                max_possible_agreements = len(item_annotations) * (len(item_annotations) - 1) / 2
                if max_possible_agreements > 0:
                    p_is.append(item_agreements / max_possible_agreements)
                else:
                    p_is.append(1.0)
            else:
                p_is.append(1.0)
        
        if not p_is:
            return 1.0
        
        p_bar = sum(p_is) / len(p_is)
        
        # Calculate expected agreement by chance
        label_proportions = {}
        for item_annotations in annotations:
            for label, _ in item_annotations:
                label_proportions[label] = label_proportions.get(label, 0) + 1
        
        for label in label_proportions:
            label_proportions[label] /= total_annotations
        
        pe = sum(p * p for p in label_proportions.values())
        
        if pe == 1.0:
            return 1.0
        
        kappa = (p_bar - pe) / (1 - pe)
        return kappa

    def assess_inter_annotator_agreement(self, 
                                       label_bundles: List[LabelBundle]) -> QualityMetric:
        """Assess agreement between multiple annotators"""
        if len(label_bundles) < 2:
            return QualityMetric(
                metric_type=QualityMetricType.INTER_ANNOTATOR_AGREEMENT,
                value=1.0,  # Perfect agreement if only one annotator
                threshold=self.iaa_threshold,
                is_passing=True,
                timestamp=datetime.utcnow().isoformat(),
                details={"message": "Insufficient annotators for agreement calculation"}
            )
        
        # Group bundles by conversation to compare labels for same content
        conversation_groups = {}
        for bundle in label_bundles:
            conv_id = bundle.conversation_id
            if conv_id not in conversation_groups:
                conversation_groups[conv_id] = []
            conversation_groups[conv_id].append(bundle)
        
        # For each group, compare the therapeutic response labels
        all_annotations = []
        for conv_id, bundles in conversation_groups.items():
            if len(bundles) >= 2:  # Need at least 2 for comparison
                item_annotations = []
                for bundle in bundles:
                    # Extract therapeutic response types as labels
                    for label in bundle.therapeutic_response_labels:
                        item_annotations.append((label.response_type.value, label.metadata.confidence))
                all_annotations.append(item_annotations)
        
        if not all_annotations or max(len(item_annotations) for item_annotations in all_annotations) < 2:
            return QualityMetric(
                metric_type=QualityMetricType.INTER_ANNOTATOR_AGREEMENT,
                value=0.0,
                threshold=self.iaa_threshold,
                is_passing=False,
                timestamp=datetime.utcnow().isoformat(),
                details={"message": "Not enough comparable annotations"}
            )
        
        # Calculate Fleiss' Kappa
        kappa = self.calculate_fleiss_kappa(all_annotations)
        
        is_passing = kappa >= self.iaa_threshold
        details = {
            "kappa": kappa,
            "annotation_groups": len(all_annotations),
            "comparable_items": sum(1 for item in all_annotations if len(item) >= 2)
        }
        
        metric = QualityMetric(
            metric_type=QualityMetricType.INTER_ANNOTATOR_AGREEMENT,
            value=kappa,
            threshold=self.iaa_threshold,
            is_passing=is_passing,
            timestamp=datetime.utcnow().isoformat(),
            details=details
        )
        
        if not is_passing:
            self._create_alert(
                QualityMetricType.INTER_ANNOTATOR_AGREEMENT,
                "high",
                f"Inter-annotator agreement below threshold: {kappa:.3f} < {self.iaa_threshold}"
            )
        
        return metric

    def monitor_label_drift(self, 
                           history: LabelHistory, 
                           window_size: int = 10) -> QualityMetric:
        """Monitor for label drift over time"""
        if len(history.versions) < 2:
            return QualityMetric(
                metric_type=QualityMetricType.LABEL_DRIFT,
                value=0.0,
                threshold=self.drift_threshold,
                is_passing=True,
                timestamp=datetime.utcnow().isoformat(),
                details={"message": "Insufficient versions for drift analysis"}
            )
        
        # Get recent versions within the window
        recent_versions = history.versions[-min(window_size, len(history.versions)):]
        
        if len(recent_versions) < 2:
            return QualityMetric(
                metric_type=QualityMetricType.LABEL_DRIFT,
                value=0.0,
                threshold=self.drift_threshold,
                is_passing=True,
                timestamp=datetime.utcnow().isoformat(),
                details={"message": "Not enough versions in window"}
            )
        
        # Calculate drift based on changes in label distributions and confidence
        drift_score = 0.0
        drift_details = []
        
        for i in range(1, len(recent_versions)):
            prev_version = recent_versions[i-1]
            curr_version = recent_versions[i]
            
            # Calculate semantic difference between versions
            prev_bundle = history.get_version(prev_version.version_number)
            curr_bundle = history.get_version(curr_version.version_number)
            
            if prev_bundle and curr_bundle:
                # Calculate drift as a combination of confidence changes and label changes
                prev_confidence = self._calculate_bundle_confidence(prev_bundle.label_data)
                curr_confidence = self._calculate_bundle_confidence(curr_bundle.label_data)
                
                confidence_drift = abs(prev_confidence - curr_confidence)
                drift_score += confidence_drift
                
                drift_details.append({
                    "version_comparison": f"v{prev_version.version_number} vs v{curr_version.version_number}",
                    "confidence_drift": confidence_drift
                })
        
        avg_drift = drift_score / len(recent_versions) if recent_versions else 0.0
        is_passing = avg_drift <= self.drift_threshold
        
        details = {
            "average_drift": avg_drift,
            "window_size": window_size,
            "drift_details": drift_details
        }
        
        metric = QualityMetric(
            metric_type=QualityMetricType.LABEL_DRIFT,
            value=avg_drift,
            threshold=self.drift_threshold,
            is_passing=is_passing,
            timestamp=datetime.utcnow().isoformat(),
            details=details
        )
        
        if not is_passing:
            self._create_alert(
                QualityMetricType.LABEL_DRIFT,
                "high",
                f"Label drift above threshold: {avg_drift:.3f} > {self.drift_threshold}"
            )
        
        return metric

    def _calculate_bundle_confidence(self, label_data: Dict[str, Any]) -> float:
        """Calculate overall confidence for a label bundle"""
        confidences = []
        
        # Extract confidences from therapeutic response labels
        for label_data in label_data.get("therapeutic_response_labels", []):
            confidences.append(label_data["metadata"]["confidence"])
        
        # Extract confidence from crisis label if present
        crisis_label = label_data.get("crisis_label")
        if crisis_label:
            confidences.append(crisis_label["metadata"]["confidence"])
        
        # Extract confidence from other labels if present
        modality_label = label_data.get("therapy_modality_label")
        if modality_label:
            confidences.append(modality_label["metadata"]["confidence"])
        
        condition_label = label_data.get("mental_health_condition_label")
        if condition_label:
            confidences.append(condition_label["metadata"]["confidence"])
        
        demographic_label = label_data.get("demographic_label")
        if demographic_label:
            confidences.append(demographic_label["metadata"]["confidence"])
        
        return statistics.mean(confidences) if confidences else 1.0

    def assess_confidence_stability(self, history: LabelHistory) -> QualityMetric:
        """Assess stability of confidence scores over time"""
        if len(history.versions) < 2:
            return QualityMetric(
                metric_type=QualityMetricType.CONFIDENCE_STABILITY,
                value=0.0,  # Perfect stability if only one version
                threshold=self.confidence_stability_threshold,
                is_passing=True,
                timestamp=datetime.utcnow().isoformat(),
                details={"message": "Insufficient versions for stability analysis"}
            )
        
        # Extract confidence scores from all versions
        confidence_scores = []
        for version in history.versions:
            bundle = history.get_version(version.version_number)
            if bundle:
                conf = self._calculate_bundle_confidence(bundle.label_data)
                confidence_scores.append(conf)
        
        if len(confidence_scores) < 2:
            return QualityMetric(
                metric_type=QualityMetricType.CONFIDENCE_STABILITY,
                value=0.0,
                threshold=self.confidence_stability_threshold,
                is_passing=True,
                timestamp=datetime.utcnow().isoformat(),
                details={"message": "Insufficient confidence scores for stability analysis"}
            )
        
        # Calculate coefficient of variation as a measure of stability
        mean_conf = statistics.mean(confidence_scores)
        if mean_conf == 0:
            stability_score = 0.0
        else:
            std_dev = statistics.stdev(confidence_scores) if len(confidence_scores) > 1 else 0.0
            stability_score = std_dev / abs(mean_conf) if mean_conf != 0 else float('inf')
        
        is_passing = stability_score <= self.confidence_stability_threshold
        
        details = {
            "coefficient_of_variation": stability_score,
            "mean_confidence": mean_conf,
            "std_deviation": statistics.stdev(confidence_scores) if len(confidence_scores) > 1 else 0,
            "num_versions": len(confidence_scores)
        }
        
        metric = QualityMetric(
            metric_type=QualityMetricType.CONFIDENCE_STABILITY,
            value=stability_score,
            threshold=self.confidence_stability_threshold,
            is_passing=is_passing,
            timestamp=datetime.utcnow().isoformat(),
            details=details
        )
        
        if not is_passing:
            self._create_alert(
                QualityMetricType.CONFIDENCE_STABILITY,
                "medium",
                f"Confidence stability above threshold: {stability_score:.3f} > {self.confidence_stability_threshold}"
            )
        
        return metric

    def assess_category_balance(self, label_bundles: List[LabelBundle]) -> QualityMetric:
        """Assess balance of label categories"""
        if not label_bundles:
            return QualityMetric(
                metric_type=QualityMetricType.CATEGORY_BALANCE,
                value=0.0,
                threshold=self.category_balance_threshold,
                is_passing=True,
                timestamp=datetime.utcnow().isoformat(),
                details={"message": "No label bundles provided"}
            )
        
        # Count occurrences of each therapeutic response type
        category_counts: Dict[str, int] = {}
        total_labels = 0
        
        for bundle in label_bundles:
            for label in bundle.therapeutic_response_labels:
                cat = label.response_type.value
                category_counts[cat] = category_counts.get(cat, 0) + 1
                total_labels += 1
        
        if total_labels == 0:
            return QualityMetric(
                metric_type=QualityMetricType.CATEGORY_BALANCE,
                value=0.0,
                threshold=self.category_balance_threshold,
                is_passing=True,
                timestamp=datetime.utcnow().isoformat(),
                details={"message": "No therapeutic response labels found"}
            )
        
        # Calculate balance as the coefficient of variation
        proportions = [count/total_labels for count in category_counts.values()]
        if len(proportions) <= 1:
            balance_score = 0.0  # Perfectly balanced if only one category
        else:
            mean_prop = statistics.mean(proportions)
            if mean_prop == 0:
                balance_score = 0.0
            else:
                std_dev = statistics.stdev(proportions) if len(proportions) > 1 else 0.0
                balance_score = std_dev / abs(mean_prop) if mean_prop != 0 else float('inf')
        
        is_passing = balance_score <= self.category_balance_threshold
        
        details = {
            "coefficient_of_variation": balance_score,
            "category_counts": category_counts,
            "total_labels": total_labels,
            "num_categories": len(category_counts),
            "proportions": {cat: count/total_labels for cat, count in category_counts.items()}
        }
        
        metric = QualityMetric(
            metric_type=QualityMetricType.CATEGORY_BALANCE,
            value=balance_score,
            threshold=self.category_balance_threshold,
            is_passing=is_passing,
            timestamp=datetime.utcnow().isoformat(),
            details=details
        )
        
        if not is_passing:
            self._create_alert(
                QualityMetricType.CATEGORY_BALANCE,
                "medium",
                f"Category balance above threshold: {balance_score:.3f} > {self.category_balance_threshold}"
            )
        
        return metric

    def perform_quality_assessment(self, 
                                  label_bundles: List[LabelBundle], 
                                  version_manager: Optional[LabelVersionManager] = None) -> List[QualityMetric]:
        """Perform comprehensive quality assessment"""
        metrics = []
        
        # Inter-annotator agreement
        iaa_metric = self.assess_inter_annotator_agreement(label_bundles)
        metrics.append(iaa_metric)
        
        # Category balance
        balance_metric = self.assess_category_balance(label_bundles)
        metrics.append(balance_metric)
        
        # If version manager is provided, assess drift and stability
        if version_manager:
            for bundle in label_bundles:
                history = version_manager.get_history(bundle.label_id)
                if history:
                    # Label drift
                    drift_metric = self.monitor_label_drift(history)
                    metrics.append(drift_metric)
                    
                    # Confidence stability
                    stability_metric = self.assess_confidence_stability(history)
                    metrics.append(stability_metric)
        
        return metrics

    def _create_alert(self, metric_type: QualityMetricType, severity: str, message: str):
        """Create a quality alert"""
        from uuid import uuid4
        alert = QualityAlert(
            alert_id=str(uuid4()),
            metric_type=metric_type,
            severity=severity,
            message=message,
            timestamp=datetime.utcnow().isoformat()
        )
        self.alerts.append(alert)
        self.logger.warning(f"Quality Alert [{severity.upper()}]: {message}")

    def get_quality_report(self) -> Dict[str, Any]:
        """Generate a quality report"""
        passing_metrics = sum(1 for alert in self.alerts if alert.severity in ['low', 'medium'])
        critical_issues = sum(1 for alert in self.alerts if alert.severity in ['high', 'critical'])
        
        return {
            "total_alerts": len(self.alerts),
            "passing_metrics": passing_metrics,
            "critical_issues": critical_issues,
            "recent_alerts": self.alerts[-10:],  # Last 10 alerts
            "timestamp": datetime.utcnow().isoformat()
        }

    def calculate_label_quality_score(self, label_bundle: LabelBundle) -> float:
        """Calculate an overall quality score for a label bundle"""
        scores = []
        
        # Score therapeutic response labels by confidence
        for label in label_bundle.therapeutic_response_labels:
            scores.append(label.metadata.confidence)
        
        # Score crisis label if present
        if label_bundle.crisis_label:
            scores.append(label_bundle.crisis_label.metadata.confidence)
        
        # Score therapy modality label if present
        if label_bundle.therapy_modality_label:
            scores.append(label_bundle.therapy_modality_label.metadata.confidence)
        
        # Score mental health condition label if present
        if label_bundle.mental_health_condition_label:
            scores.append(label_bundle.mental_health_condition_label.metadata.confidence)
        
        # Score demographic label if present
        if label_bundle.demographic_label:
            scores.append(label_bundle.demographic_label.metadata.confidence)
        
        # Return average confidence if there are scores, otherwise perfect score
        return statistics.mean(scores) if scores else 1.0

    def assess_label_quality_metrics(self, label_bundles: List[LabelBundle], 
                                   annotator_performance: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Assess comprehensive label quality metrics with expected minimum thresholds"""
        if not label_bundles:
            return {
                "label_quality_score": 0.0,
                "average_confidence": 0.0,
                "high_confidence_ratio": 0.0,
                "confidence_std_dev": 0.0,
                "minimum_threshold_met": False,
                "metrics_summary": "No labels provided for assessment"
            }
        
        # Calculate overall label quality metrics
        quality_scores = [self.calculate_label_quality_score(bundle) for bundle in label_bundles]
        all_confidences = []
        
        # Collect all confidence scores
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
            return {
                "label_quality_score": 1.0,
                "average_confidence": 1.0,
                "high_confidence_ratio": 1.0,
                "confidence_std_dev": 0.0,
                "minimum_threshold_met": True,
                "metrics_summary": "No confidence scores found, assuming perfect quality"
            }
        
        avg_confidence = statistics.mean(all_confidences)
        confidence_std_dev = statistics.stdev(all_confidences) if len(all_confidences) > 1 else 0
        high_confidence_ratio = sum(1 for conf in all_confidences if conf >= self.minimum_confidence_threshold) / len(all_confidences)
        overall_quality_score = statistics.mean(quality_scores)
        
        # Check against minimum thresholds
        confidence_threshold_met = avg_confidence >= self.minimum_confidence_threshold
        quality_threshold_met = overall_quality_score >= self.label_quality_threshold
        stability_threshold_met = confidence_std_dev <= self.confidence_stability_threshold
        
        minimum_threshold_met = confidence_threshold_met and quality_threshold_met and stability_threshold_met
        
        # Generate alerts if thresholds are not met
        if not confidence_threshold_met:
            self._create_alert(
                QualityMetricType.CONFIDENCE_STABILITY,
                "high",
                f"Average confidence {avg_confidence:.3f} below minimum threshold {self.minimum_confidence_threshold}"
            )
        
        if not quality_threshold_met:
            self._create_alert(
                QualityMetricType.INTER_ANNOTATOR_AGREEMENT,
                "high",
                f"Overall quality score {overall_quality_score:.3f} below minimum threshold {self.label_quality_threshold}"
            )
        
        if not stability_threshold_met:
            self._create_alert(
                QualityMetricType.CONFIDENCE_STABILITY,
                "medium",
                f"Confidence stability {confidence_std_dev:.3f} exceeds maximum threshold {self.confidence_stability_threshold}"
            )
        
        metrics_summary = (
            f"Label Quality: {overall_quality_score:.3f} ({'✓' if quality_threshold_met else '✗'}), "
            f"Average Confidence: {avg_confidence:.3f} ({'✓' if confidence_threshold_met else '✗'}), "
            f"High Confidence Ratio: {high_confidence_ratio:.3f}, "
            f"Confidence Stability: {confidence_std_dev:.3f} ({'✓' if stability_threshold_met else '✗'})"
        )
        
        result = {
            "label_quality_score": overall_quality_score,
            "average_confidence": avg_confidence,
            "high_confidence_ratio": high_confidence_ratio,
            "confidence_std_dev": confidence_std_dev,
            "minimum_threshold_met": minimum_threshold_met,
            "metrics_summary": metrics_summary,
            "total_labels": len(all_confidences),
            "total_bundles": len(label_bundles),
            "thresholds": {
                "minimum_confidence": self.minimum_confidence_threshold,
                "minimum_quality": self.label_quality_threshold,
                "maximum_stability_deviation": self.confidence_stability_threshold
            }
        }
        
        return result

    def generate_quality_report_with_metrics(self, 
                                           label_bundles: List[LabelBundle],
                                           version_manager: Optional[LabelVersionManager] = None) -> Dict[str, Any]:
        """Generate a comprehensive quality report with all metrics"""
        # Get basic quality assessment
        basic_metrics = self.perform_quality_assessment(label_bundles, version_manager)
        
        # Get label quality metrics
        label_quality_metrics = self.assess_label_quality_metrics(label_bundles)
        
        # Get basic quality report
        basic_report = self.get_quality_report()
        
        return {
            "timestamp": datetime.utcnow().isoformat(),
            "basic_quality_metrics": [metric.__dict__ for metric in basic_metrics],
            "label_quality_metrics": label_quality_metrics,
            "quality_alerts": basic_report,
            "annotator_performance": self._assess_annotator_performance(label_bundles)
        }

    def _assess_annotator_performance(self, label_bundles: List[LabelBundle]) -> Dict[str, Any]:
        """Assess performance by annotator if provenance is available"""
        annotator_stats = {}
        
        for bundle in label_bundles:
            # Collect stats from each type of label
            for label in bundle.therapeutic_response_labels:
                if label.metadata.annotator_id:
                    annotator_id = label.metadata.annotator_id
                    if annotator_id not in annotator_stats:
                        annotator_stats[annotator_id] = {
                            "total_labels": 0,
                            "total_confidence": 0,
                            "label_types": Counter()
                        }
                    annotator_stats[annotator_id]["total_labels"] += 1
                    annotator_stats[annotator_id]["total_confidence"] += label.metadata.confidence
                    annotator_stats[annotator_id]["label_types"][label.response_type.value] += 1
            
            # Check crisis label
            if bundle.crisis_label and bundle.crisis_label.metadata.annotator_id:
                annotator_id = bundle.crisis_label.metadata.annotator_id
                if annotator_id not in annotator_stats:
                    annotator_stats[annotator_id] = {
                        "total_labels": 0,
                        "total_confidence": 0,
                        "label_types": Counter()
                    }
                annotator_stats[annotator_id]["total_labels"] += 1
                annotator_stats[annotator_id]["total_confidence"] += bundle.crisis_label.metadata.confidence
            
            # Check other label types similarly...
            if bundle.therapy_modality_label and bundle.therapy_modality_label.metadata.annotator_id:
                annotator_id = bundle.therapy_modality_label.metadata.annotator_id
                if annotator_id not in annotator_stats:
                    annotator_stats[annotator_id] = {
                        "total_labels": 0,
                        "total_confidence": 0,
                        "label_types": Counter()
                    }
                annotator_stats[annotator_id]["total_labels"] += 1
                annotator_stats[annotator_id]["total_confidence"] += bundle.therapy_modality_label.metadata.confidence
        
        # Calculate averages and add annotations counts
        for annotator_id, stats in annotator_stats.items():
            if stats["total_labels"] > 0:
                stats["average_confidence"] = stats["total_confidence"] / stats["total_labels"]
            else:
                stats["average_confidence"] = 0
        
        return annotator_stats


class LabelDriftDetector:
    """Specialized class for detecting label drift over time"""
    
    def __init__(self, window_size: int = 30, min_samples: int = 10):
        self.window_size = window_size
        self.min_samples = min_samples
        self.logger = logging.getLogger(__name__)

    def detect_drift_by_provenance(self, provenance_records: List[ProvenanceRecord]) -> Dict[str, Any]:
        """Detect drift based on provenance changes"""
        if len(provenance_records) < self.min_samples:
            return {
                "drift_detected": False,
                "reason": "Insufficient samples",
                "drift_score": 0.0
            }
        
        # Group by time windows
        time_windows = {}
        for record in provenance_records:
            timestamp = datetime.fromisoformat(record.timestamp.replace('Z', '+00:00'))
            window_key = timestamp.strftime('%Y-%m-%d')
            if window_key not in time_windows:
                time_windows[window_key] = []
            time_windows[window_key].append(record)
        
        # Calculate distribution of provenance types across windows
        window_distributions = {}
        for window, records in time_windows.items():
            dist = {}
            for record in records:
                source = record.source.value
                dist[source] = dist.get(source, 0) + 1
            total = sum(dist.values())
            if total > 0:
                window_distributions[window] = {k: v/total for k, v in dist.items()}
        
        # Compare distributions between windows to detect drift
        if len(window_distributions) < 2:
            return {
                "drift_detected": False,
                "reason": "Insufficient time windows",
                "drift_score": 0.0,
                "distributions": window_distributions
            }
        
        # Calculate drift as difference in distributions
        windows = sorted(window_distributions.keys())
        drift_score = 0.0
        
        for i in range(len(windows)-1):
            dist1 = window_distributions[windows[i]]
            dist2 = window_distributions[windows[i+1]]
            
            # Calculate Jensen-Shannon divergence
            all_keys = set(dist1.keys()).union(set(dist2.keys()))
            diff = sum(abs(dist1.get(k, 0) - dist2.get(k, 0)) for k in all_keys)
            drift_score += diff / len(all_keys) if all_keys else 0.0
        
        avg_drift = drift_score / (len(windows)-1) if len(windows) > 1 else 0.0
        drift_detected = avg_drift > 0.1  # Threshold for drift
        
        return {
            "drift_detected": drift_detected,
            "drift_score": avg_drift,
            "distributions": window_distributions,
            "windows": windows
        }

    def detect_drift_by_confidence(self, histories: List[LabelHistory]) -> Dict[str, Any]:
        """Detect drift based on confidence changes over time"""
        all_confidence_trends = []
        
        for history in histories:
            confidence_over_time = []
            for version in history.versions:
                # Extract average confidence for this version
                bundle = history.get_version(version.version_number)
                if bundle:
                    avg_conf = self._calculate_bundle_confidence(bundle.label_data)
                    confidence_over_time.append({
                        "timestamp": version.timestamp,
                        "confidence": avg_conf
                    })
            
            if len(confidence_over_time) >= 2:
                all_confidence_trends.append(confidence_over_time)
        
        if not all_confidence_trends:
            return {
                "drift_detected": False,
                "reason": "No confidence trends available",
                "drift_score": 0.0
            }
        
        # Calculate trend stability
        overall_drift = 0.0
        for trend in all_confidence_trends:
            if len(trend) >= 2:
                # Calculate average rate of change
                changes = []
                for i in range(1, len(trend)):
                    change = abs(trend[i]["confidence"] - trend[i-1]["confidence"])
                    changes.append(change)
                
                if changes:
                    avg_change = sum(changes) / len(changes)
                    overall_drift += avg_change
        
        avg_drift = overall_drift / len(all_confidence_trends) if all_confidence_trends else 0.0
        drift_detected = avg_drift > 0.1  # Threshold for significant drift
        
        return {
            "drift_detected": drift_detected,
            "drift_score": avg_drift,
            "trend_count": len(all_confidence_trends)
        }
    
    def _calculate_bundle_confidence(self, label_data: Dict[str, Any]) -> float:
        """Calculate overall confidence for a label bundle"""
        confidences = []
        
        # Extract confidences from therapeutic response labels
        for label_data in label_data.get("therapeutic_response_labels", []):
            confidences.append(label_data["metadata"]["confidence"])
        
        # Extract confidence from crisis label if present
        crisis_label = label_data.get("crisis_label")
        if crisis_label:
            confidences.append(crisis_label["metadata"]["confidence"])
        
        # Extract confidence from other labels if present
        modality_label = label_data.get("therapy_modality_label")
        if modality_label:
            confidences.append(modality_label["metadata"]["confidence"])
        
        condition_label = label_data.get("mental_health_condition_label")
        if condition_label:
            confidences.append(condition_label["metadata"]["confidence"])
        
        demographic_label = label_data.get("demographic_label")
        if demographic_label:
            confidences.append(demographic_label["metadata"]["confidence"])
        
        return statistics.mean(confidences) if confidences else 1.0


def create_quality_controller() -> QualityController:
    """Create a default quality controller"""
    return QualityController()


# Example usage
def test_quality_control():
    """Test the quality control system"""
    from .label_taxonomy import (
        TherapeuticResponseLabel, CrisisLabel, LabelMetadata, LabelProvenanceType,
        TherapeuticResponseType, CrisisLevelType
    )
    from .conversation_schema import Conversation
    from .label_versioning import create_version_manager, LabelVersionManager
    
    # Create some test label bundles for the same conversation
    bundles = []
    for i in range(3):
        bundle = LabelBundle(conversation_id=f"test_conv_{i}")
        
        # Add therapeutic response labels
        bundle.therapeutic_response_labels.append(
            TherapeuticResponseLabel(
                response_type=TherapeuticResponseType.EMPATHY,
                metadata=LabelMetadata(
                    confidence=0.8 if i == 0 else 0.85,  # Slightly different for second/third
                    provenance=LabelProvenanceType.HUMAN_EXPERT if i < 2 else LabelProvenanceType.AUTOMATED_MODEL
                )
            )
        )
        
        bundle.therapeutic_response_labels.append(
            TherapeuticResponseLabel(
                response_type=TherapeuticResponseType.REFLECTION,
                metadata=LabelMetadata(
                    confidence=0.9,
                    provenance=LabelProvenanceType.HUMAN_EXPERT if i < 2 else LabelProvenanceType.AUTOMATED_MODEL
                )
            )
        )
        
        bundle.crisis_label = CrisisLabel(
            crisis_level=CrisisLevelType.LOW_RISK,
            metadata=LabelMetadata(
                confidence=0.85,
                provenance=LabelProvenanceType.HUMAN_EXPERT if i < 2 else LabelProvenanceType.AUTOMATED_MODEL
            )
        )
        
        bundles.append(bundle)
    
    # Create quality controller
    qc = create_quality_controller()
    
    # Perform quality assessment
    metrics = qc.perform_quality_assessment(bundles)
    print("Quality Assessment Results:")
    for metric in metrics:
        print(f"  {metric.metric_type.value}: {metric.value:.3f} ({'PASS' if metric.is_passing else 'FAIL'})")
        if metric.details:
            print(f"    Details: {metric.details}")
    
    # Create version manager to test drift and stability
    vm = create_version_manager()
    for bundle in bundles:
        vm.create_initial_version(bundle, f"annotator_{bundle.label_id[:8]}")
    
    # Perform comprehensive assessment with version manager
    comprehensive_metrics = qc.perform_quality_assessment(bundles, vm)
    print("\nComprehensive Assessment Results:")
    for metric in comprehensive_metrics:
        print(f"  {metric.metric_type.value}: {metric.value:.3f} ({'PASS' if metric.is_passing else 'FAIL'})")
    
    # Test drift detector
    if bundles:
        drift_detector = LabelDriftDetector()
        histories = [vm.get_history(bundle.label_id) for bundle in bundles if vm.get_history(bundle.label_id)]
        drift_results = drift_detector.detect_drift_by_confidence(histories)
        print(f"\nDrift Detection Results: {drift_results}")
    
    # Get quality report
    report = qc.get_quality_report()
    print(f"\nQuality Report: {report}")


if __name__ == "__main__":
    test_quality_control()