"""
Tooling for reviewing label distributions and performing edge-case analyses.
Provides visualization and analytical tools for understanding the labeled dataset.
"""

from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Tuple, Set
import json
import logging
from collections import Counter, defaultdict
import statistics
from datetime import datetime
import math
from .label_taxonomy import (
    LabelBundle, TherapeuticResponseType, CrisisLevelType, TherapyModalityType,
    MentalHealthConditionType, DemographicType, LabelProvenanceType
)
from .conversation_schema import Conversation, Message

logger = logging.getLogger(__name__)


@dataclass
class LabelDistributionReport:
    """Report of label distribution across the dataset"""
    total_conversations: int
    therapeutic_response_distribution: Dict[str, int]
    crisis_level_distribution: Dict[str, int]
    therapy_modality_distribution: Dict[str, int]
    mental_health_condition_distribution: Dict[str, int]
    demographic_distribution: Dict[str, int]
    provenance_distribution: Dict[str, int]
    confidence_statistics: Dict[str, Dict[str, float]]  # stats for each label type
    timestamp: str


@dataclass
class EdgeCaseReport:
    """Report of identified edge cases in the dataset"""
    low_confidence_items: List[Dict[str, Any]]
    conflicting_labels: List[Dict[str, Any]]
    rare_categories: List[Dict[str, Any]]
    high_risk_items: List[Dict[str, Any]]
    inconsistent_patterns: List[Dict[str, Any]]
    timestamp: str


class LabelDistributionAnalyzer:
    """Analyzer for examining label distributions across the dataset"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)

    def generate_distribution_report(self, label_bundles: List[LabelBundle]) -> LabelDistributionReport:
        """Generate a comprehensive report on label distributions"""
        if not label_bundles:
            return LabelDistributionReport(
                total_conversations=0,
                therapeutic_response_distribution={},
                crisis_level_distribution={},
                therapy_modality_distribution={},
                mental_health_condition_distribution={},
                demographic_distribution={},
                provenance_distribution={},
                confidence_statistics={},
                timestamp=datetime.utcnow().isoformat()
            )
        
        # Count distributions
        therapeutic_counts = Counter()
        crisis_counts = Counter()
        modality_counts = Counter()
        condition_counts = Counter()
        demographic_counts = Counter()
        
        # Collect confidence scores for statistics
        confidence_scores = {
            'therapeutic_response': [],
            'crisis': [],
            'therapy_modality': [],
            'mental_health_condition': [],
            'demographic': []
        }
        
        # Count provenance types
        provenance_counts = Counter()
        
        for bundle in label_bundles:
            # Count therapeutic responses
            for label in bundle.therapeutic_response_labels:
                therapeutic_counts[label.response_type.value] += 1
                confidence_scores['therapeutic_response'].append(label.metadata.confidence)
                provenance_counts[label.metadata.provenance.value] += 1
            
            # Count crisis levels
            if bundle.crisis_label:
                crisis_counts[bundle.crisis_label.crisis_level.value] += 1
                confidence_scores['crisis'].append(bundle.crisis_label.metadata.confidence)
                provenance_counts[bundle.crisis_label.metadata.provenance.value] += 1
            
            # Count therapy modalities
            if bundle.therapy_modality_label:
                modality_counts[bundle.therapy_modality_label.modality.value] += 1
                confidence_scores['therapy_modality'].append(bundle.therapy_modality_label.metadata.confidence)
                provenance_counts[bundle.therapy_modality_label.metadata.provenance.value] += 1
            
            # Count mental health conditions
            if bundle.mental_health_condition_label:
                for condition in bundle.mental_health_condition_label.conditions:
                    condition_counts[condition.value] += 1
                confidence_scores['mental_health_condition'].append(bundle.mental_health_condition_label.metadata.confidence)
                provenance_counts[bundle.mental_health_condition_label.metadata.provenance.value] += 1
            
            # Count demographics
            if bundle.demographic_label:
                for demo in bundle.demographic_label.demographics:
                    demographic_counts[demo.value] += 1
                confidence_scores['demographic'].append(bundle.demographic_label.metadata.confidence)
                provenance_counts[bundle.demographic_label.metadata.provenance.value] += 1
        
        # Calculate confidence statistics
        confidence_stats = {}
        for label_type, scores in confidence_scores.items():
            if scores:
                confidence_stats[label_type] = {
                    'mean': statistics.mean(scores),
                    'median': statistics.median(scores),
                    'std_dev': statistics.stdev(scores) if len(scores) > 1 else 0,
                    'min': min(scores),
                    'max': max(scores),
                    'count': len(scores)
                }
            else:
                confidence_stats[label_type] = {
                    'mean': 0, 'median': 0, 'std_dev': 0, 'min': 0, 'max': 0, 'count': 0
                }
        
        return LabelDistributionReport(
            total_conversations=len(label_bundles),
            therapeutic_response_distribution=dict(therapeutic_counts),
            crisis_level_distribution=dict(crisis_counts),
            therapy_modality_distribution=dict(modality_counts),
            mental_health_condition_distribution=dict(condition_counts),
            demographic_distribution=dict(demographic_counts),
            provenance_distribution=dict(provenance_counts),
            confidence_statistics=confidence_stats,
            timestamp=datetime.utcnow().isoformat()
        )

    def find_rare_categories(self, label_bundles: List[LabelBundle], 
                           threshold_ratio: float = 0.01) -> Dict[str, List[Dict[str, Any]]]:
        """Find rare categories that appear in less than threshold_ratio of conversations"""
        distribution_report = self.generate_distribution_report(label_bundles)
        total = distribution_report.total_conversations
        
        rare_categories = {}
        
        # Identify rare therapeutic response types
        rare_therapeutic = []
        for response_type, count in distribution_report.therapeutic_response_distribution.items():
            if total > 0 and count / total < threshold_ratio:
                rare_therapeutic.append({
                    "category": response_type,
                    "count": count,
                    "proportion": count / total
                })
        if rare_therapeutic:
            rare_categories["therapeutic_response"] = rare_therapeutic
        
        # Identify rare crisis levels
        rare_crisis = []
        for crisis_level, count in distribution_report.crisis_level_distribution.items():
            if total > 0 and count / total < threshold_ratio:
                rare_crisis.append({
                    "category": crisis_level,
                    "count": count,
                    "proportion": count / total
                })
        if rare_crisis:
            rare_categories["crisis_level"] = rare_crisis
        
        # Identify rare therapy modalities
        rare_modalities = []
        for modality, count in distribution_report.therapy_modality_distribution.items():
            if total > 0 and count / total < threshold_ratio:
                rare_modalities.append({
                    "category": modality,
                    "count": count,
                    "proportion": count / total
                })
        if rare_modalities:
            rare_categories["therapy_modality"] = rare_modalities
        
        # Identify rare mental health conditions
        rare_conditions = []
        for condition, count in distribution_report.mental_health_condition_distribution.items():
            if total > 0 and count / total < threshold_ratio:
                rare_conditions.append({
                    "category": condition,
                    "count": count,
                    "proportion": count / total
                })
        if rare_conditions:
            rare_categories["mental_health_condition"] = rare_conditions
        
        return rare_categories

    def analyze_label_correlations(self, label_bundles: List[LabelBundle]) -> Dict[str, Any]:
        """Analyze correlations between different label types"""
        correlations = {
            'therapeutic_response_crisis': defaultdict(Counter),
            'therapy_modality_crisis': defaultdict(Counter),
            'demographic_crisis': defaultdict(Counter)
        }
        
        for bundle in label_bundles:
            # Get primary crisis level
            crisis_level = "none"
            if bundle.crisis_label:
                crisis_level = bundle.crisis_label.crisis_level.value
            
            # Analyze therapeutic response - crisis correlations
            for label in bundle.therapeutic_response_labels:
                response_type = label.response_type.value
                correlations['therapeutic_response_crisis'][crisis_level][response_type] += 1
            
            # Analyze therapy modality - crisis correlations
            if bundle.therapy_modality_label:
                modality = bundle.therapy_modality_label.modality.value
                correlations['therapy_modality_crisis'][crisis_level][modality] += 1
            
            # Analyze demographic - crisis correlations
            if bundle.demographic_label:
                for demo in bundle.demographic_label.demographics:
                    demo_type = demo.value
                    correlations['demographic_crisis'][crisis_level][demo_type] += 1
        
        # Convert to regular dicts and calculate correlation coefficients
        result = {}
        for key, corr_data in correlations.items():
            result[key] = {}
            for crisis_level, sub_counts in corr_data.items():
                total_for_crisis = sum(sub_counts.values())
                if total_for_crisis > 0:
                    result[key][crisis_level] = {
                        cat: count/total_for_crisis 
                        for cat, count in sub_counts.items()
                    }
        
        return result


class EdgeCaseDetector:
    """Detector for identifying edge cases in the dataset"""
    
    def __init__(self, low_confidence_threshold: float = 0.6):
        self.low_confidence_threshold = low_confidence_threshold
        self.logger = logging.getLogger(__name__)

    def detect_edge_cases(self, 
                         conversations: List[Conversation], 
                         label_bundles: List[LabelBundle]) -> EdgeCaseReport:
        """Detect various types of edge cases in the dataset"""
        paired_data = list(zip(conversations, label_bundles))
        
        # Detect different types of edge cases
        low_confidence_items = self._find_low_confidence_items(paired_data)
        conflicting_labels = self._find_conflicting_labels(paired_data)
        rare_categories = self._find_rare_categories_in_bundles(paired_data)
        high_risk_items = self._find_high_risk_items(paired_data)
        inconsistent_patterns = self._find_inconsistent_patterns(paired_data)
        
        return EdgeCaseReport(
            low_confidence_items=low_confidence_items,
            conflicting_labels=conflicting_labels,
            rare_categories=rare_categories,
            high_risk_items=high_risk_items,
            inconsistent_patterns=inconsistent_patterns,
            timestamp=datetime.utcnow().isoformat()
        )

    def _find_low_confidence_items(self, paired_data: List[Tuple[Conversation, LabelBundle]]) -> List[Dict[str, Any]]:
        """Find items with low confidence scores"""
        low_confidence = []
        
        for conv, bundle in paired_data:
            issues = []
            
            # Check therapeutic response labels
            for i, label in enumerate(bundle.therapeutic_response_labels):
                if label.metadata.confidence < self.low_confidence_threshold:
                    issues.append({
                        "type": "therapeutic_response",
                        "response_type": label.response_type.value,
                        "confidence": label.metadata.confidence,
                        "index": i
                    })
            
            # Check crisis label
            if bundle.crisis_label and bundle.crisis_label.metadata.confidence < self.low_confidence_threshold:
                issues.append({
                    "type": "crisis",
                    "crisis_level": bundle.crisis_label.crisis_level.value,
                    "confidence": bundle.crisis_label.metadata.confidence
                })
            
            # Check therapy modality
            if bundle.therapy_modality_label and bundle.therapy_modality_label.metadata.confidence < self.low_confidence_threshold:
                issues.append({
                    "type": "therapy_modality",
                    "modality": bundle.therapy_modality_label.modality.value,
                    "confidence": bundle.therapy_modality_label.metadata.confidence
                })
            
            # Check mental health condition
            if bundle.mental_health_condition_label and bundle.mental_health_condition_label.metadata.confidence < self.low_confidence_threshold:
                issues.append({
                    "type": "mental_health_condition",
                    "confidence": bundle.mental_health_condition_label.metadata.confidence
                })
            
            # Check demographic
            if bundle.demographic_label and bundle.demographic_label.metadata.confidence < self.low_confidence_threshold:
                issues.append({
                    "type": "demographic",
                    "confidence": bundle.demographic_label.metadata.confidence
                })
            
            if issues:
                low_confidence.append({
                    "conversation_id": conv.conversation_id,
                    "issues": issues
                })
        
        return low_confidence

    def _find_conflicting_labels(self, paired_data: List[Tuple[Conversation, LabelBundle]]) -> List[Dict[str, Any]]:
        """Find potential conflicts in labeling"""
        conflicts = []
        
        for conv, bundle in paired_data:
            # Check for conflicting therapeutic responses that don't make sense together
            response_types = [label.response_type for label in bundle.therapeutic_response_labels]
            
            # Specific conflict detection - example: challenge and support responses might conflict
            if (TherapeuticResponseType.CHALLENGE in response_types and 
                TherapeuticResponseType.SUPPORT in response_types):
                # Check if they have low confidence, which might indicate uncertainty
                challenge_labels = [l for l in bundle.therapeutic_response_labels 
                                  if l.response_type == TherapeuticResponseType.CHALLENGE]
                support_labels = [l for l in bundle.therapeutic_response_labels 
                                if l.response_type == TherapeuticResponseType.SUPPORT]
                
                if (any(l.metadata.confidence < 0.7 for l in challenge_labels) or
                    any(l.metadata.confidence < 0.7 for l in support_labels)):
                    conflicts.append({
                        "conversation_id": conv.conversation_id,
                        "conflict_type": "challenge_support_conflict",
                        "challenges": [l.response_type.value + f"({l.metadata.confidence:.2f})" 
                                     for l in challenge_labels],
                        "support": [l.response_type.value + f"({l.metadata.confidence:.2f})" 
                                  for l in support_labels]
                    })
        
        return conflicts

    def _find_rare_categories_in_bundles(self, paired_data: List[Tuple[Conversation, LabelBundle]]) -> List[Dict[str, Any]]:
        """Find conversations with rare categories"""
        # This would typically use the rare categories from the distribution analyzer
        # For now, we'll look for specific rare patterns
        rare_items = []
        
        for conv, bundle in paired_data:
            rare_categories = []
            
            # Check for very specific or unusual therapeutic responses
            for label in bundle.therapeutic_response_labels:
                # In a real system, you might have domain knowledge about which responses are rare
                if label.response_type.value == "interpretation" and len([l for l in bundle.therapeutic_response_labels 
                                                                         if l.response_type.value == "interpretation"]) == 1:
                    rare_categories.append({
                        "type": "therapeutic_response",
                        "value": label.response_type.value,
                        "confidence": label.metadata.confidence
                    })
            
            # Check for high-risk crisis levels
            if bundle.crisis_label and bundle.crisis_label.crisis_level in [
                CrisisLevelType.HIGH_RISK, CrisisLevelType.IMMEDIATE_RISK]:
                rare_categories.append({
                    "type": "crisis_level",
                    "value": bundle.crisis_label.crisis_level.value,
                    "confidence": bundle.crisis_label.metadata.confidence
                })
            
            if rare_categories:
                rare_items.append({
                    "conversation_id": conv.conversation_id,
                    "rare_categories": rare_categories
                })
        
        return rare_items

    def _find_high_risk_items(self, paired_data: List[Tuple[Conversation, LabelBundle]]) -> List[Dict[str, Any]]:
        """Find high-risk items that need special attention"""
        high_risk = []
        
        for conv, bundle in paired_data:
            risk_factors = []
            
            # High crisis levels
            if bundle.crisis_label:
                if bundle.crisis_label.crisis_level in [CrisisLevelType.HIGH_RISK, CrisisLevelType.IMMEDIATE_RISK]:
                    risk_factors.append({
                        "type": "crisis_level",
                        "level": bundle.crisis_label.crisis_level.value,
                        "confidence": bundle.crisis_label.metadata.confidence
                    })
                
                # High risk probability
                if (bundle.crisis_label.estimated_risk_probability is not None and 
                    bundle.crisis_label.estimated_risk_probability > 0.8):
                    risk_factors.append({
                        "type": "risk_probability",
                        "probability": bundle.crisis_label.estimated_risk_probability
                    })
            
            # Check for high-risk therapeutic approaches with low confidence
            challenge_labels = [l for l in bundle.therapeutic_response_labels 
                              if l.response_type == TherapeuticResponseType.CHALLENGE and l.metadata.confidence < 0.7]
            if challenge_labels:
                risk_factors.append({
                    "type": "low_confidence_challenges",
                    "count": len(challenge_labels),
                    "lowest_confidence": min(l.metadata.confidence for l in challenge_labels)
                })
            
            if risk_factors:
                high_risk.append({
                    "conversation_id": conv.conversation_id,
                    "risk_factors": risk_factors
                })
        
        return high_risk

    def _find_inconsistent_patterns(self, paired_data: List[Tuple[Conversation, LabelBundle]]) -> List[Dict[str, Any]]:
        """Find inconsistent labeling patterns"""
        inconsistencies = []
        
        for conv, bundle in paired_data:
            inconsistencies_found = []
            
            # Check for high confidence with low agreement (if we had multiple annotators)
            # For now, check for unusual confidence patterns
            all_confidences = []
            
            # Collect all confidence scores in the bundle
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
            
            # Check if there's a wide variation in confidence scores
            if len(all_confidences) > 1:
                confidence_range = max(all_confidences) - min(all_confidences)
                if confidence_range > 0.5:  # Large variation
                    inconsistencies_found.append({
                        "type": "confidence_variation",
                        "range": confidence_range,
                        "min": min(all_confidences),
                        "max": max(all_confidences),
                        "mean": statistics.mean(all_confidences)
                    })
            
            # Check for unusual combinations
            if (bundle.crisis_label and 
                bundle.crisis_label.crisis_level in [CrisisLevelType.HIGH_RISK, CrisisLevelType.IMMEDIATE_RISK] and
                bundle.therapeutic_response_labels and
                all(l.metadata.confidence < 0.5 for l in bundle.therapeutic_response_labels)):
                inconsistencies_found.append({
                    "type": "high_risk_low_confidence_responses",
                    "crisis_level": bundle.crisis_label.crisis_level.value,
                    "response_labels_count": len(bundle.therapeutic_response_labels),
                    "lowest_response_confidence": min(l.metadata.confidence for l in bundle.therapeutic_response_labels)
                })
            
            if inconsistencies_found:
                inconsistencies.append({
                    "conversation_id": conv.conversation_id,
                    "inconsistencies": inconsistencies_found
                })
        
        return inconsistencies


class DistributionVisualizer:
    """Helper class to generate various distribution visualizations (text-based for now)"""
    
    def __init__(self):
        pass
    
    def print_distribution_summary(self, report: LabelDistributionReport):
        """Print a text-based summary of the distribution report"""
        print(f"\n=== Label Distribution Summary ===")
        print(f"Total conversations: {report.total_conversations}")
        print(f"Generated at: {report.timestamp}")
        
        print(f"\n--- Therapeutic Response Distribution ---")
        for response_type, count in sorted(report.therapeutic_response_distribution.items(), key=lambda x: x[1], reverse=True):
            percentage = (count / report.total_conversations * 100) if report.total_conversations > 0 else 0
            print(f"  {response_type}: {count} ({percentage:.1f}%)")
        
        print(f"\n--- Crisis Level Distribution ---")
        for crisis_level, count in sorted(report.crisis_level_distribution.items(), key=lambda x: x[1], reverse=True):
            percentage = (count / report.total_conversations * 100) if report.total_conversations > 0 else 0
            print(f"  {crisis_level}: {count} ({percentage:.1f}%)")
        
        print(f"\n--- Therapy Modality Distribution ---")
        for modality, count in sorted(report.therapy_modality_distribution.items(), key=lambda x: x[1], reverse=True):
            percentage = (count / report.total_conversations * 100) if report.total_conversations > 0 else 0
            print(f"  {modality}: {count} ({percentage:.1f}%)")
        
        print(f"\n--- Provenance Distribution ---")
        for provenance, count in sorted(report.provenance_distribution.items(), key=lambda x: x[1], reverse=True):
            percentage = (count / report.total_conversations * 100) if report.total_conversations > 0 else 0
            print(f"  {provenance}: {count} ({percentage:.1f}%)")
        
        print(f"\n--- Confidence Statistics ---")
        for label_type, stats in report.confidence_statistics.items():
            if stats['count'] > 0:
                print(f"  {label_type}: Mean={stats['mean']:.3f}, StdDev={stats['std_dev']:.3f}, Min={stats['min']:.3f}, Max={stats['max']:.3f}")
    
    def print_edge_case_summary(self, report: EdgeCaseReport):
        """Print a text-based summary of the edge case report"""
        print(f"\n=== Edge Case Summary ===")
        print(f"Report generated at: {report.timestamp}")
        
        print(f"\n--- Low Confidence Items ---")
        print(f"Total conversations with low confidence labels: {len(report.low_confidence_items)}")
        for item in report.low_confidence_items[:5]:  # Show first 5
            print(f"  Conversation {item['conversation_id'][:8]}...: {len(item['issues'])} low-confidence issues")
        
        print(f"\n--- Conflicting Labels ---")
        print(f"Total conversations with conflicting labels: {len(report.conflicting_labels)}")
        for item in report.conflicting_labels[:5]:  # Show first 5
            print(f"  Conversation {item['conversation_id'][:8]}...: {item['conflict_type']}")
        
        print(f"\n--- High Risk Items ---")
        print(f"Total high-risk conversations: {len(report.high_risk_items)}")
        for item in report.high_risk_items[:5]:  # Show first 5
            print(f"  Conversation {item['conversation_id'][:8]}...: {len(item['risk_factors'])} risk factors")
        
        print(f"\n--- Inconsistent Patterns ---")
        print(f"Total conversations with inconsistencies: {len(report.inconsistent_patterns)}")
        for item in report.inconsistent_patterns[:5]:  # Show first 5
            print(f"  Conversation {item['conversation_id'][:8]}...: {len(item['inconsistencies'])} inconsistencies")


def create_analyzer() -> LabelDistributionAnalyzer:
    """Create a default label distribution analyzer"""
    return LabelDistributionAnalyzer()


def create_edge_case_detector(low_confidence_threshold: float = 0.6) -> EdgeCaseDetector:
    """Create a default edge case detector"""
    return EdgeCaseDetector(low_confidence_threshold)


# Example usage
def test_label_analysis():
    """Test the label distribution and edge case analysis tools"""
    from .label_taxonomy import (
        TherapeuticResponseLabel, CrisisLabel, LabelMetadata, LabelProvenanceType,
        TherapeuticResponseType, CrisisLevelType, TherapyModalityType
    )
    from .conversation_schema import Conversation, Message
    
    # Create test data
    conversations = []
    label_bundles = []
    
    for i in range(100):
        # Create conversation
        conv = Conversation()
        conv.add_message("therapist", f"Sample conversation {i}")
        conv.add_message("client", f"Client response {i}")
        conversations.append(conv)
        
        # Create label bundle
        bundle = LabelBundle(conversation_id=conv.conversation_id)
        
        # Add therapeutic response labels with varying confidence
        response_type = TherapeuticResponseType.EMPATHY if i % 3 == 0 else \
                       TherapeuticResponseType.REFLECTION if i % 3 == 1 else \
                       TherapeuticResponseType.PROBING
        confidence = 0.9 if i < 80 else 0.4  # Introduce some low confidence labels
        bundle.therapeutic_response_labels.append(
            TherapeuticResponseLabel(
                response_type=response_type,
                metadata=LabelMetadata(
                    confidence=confidence,
                    provenance=LabelProvenanceType.AUTOMATED_MODEL
                )
            )
        )
        
        # Add crisis label
        crisis_level = CrisisLevelType.NO_RISK if i < 95 else CrisisLevelType.HIGH_RISK
        bundle.crisis_label = CrisisLabel(
            crisis_level=crisis_level,
            metadata=LabelMetadata(
                confidence=0.8 if i < 95 else 0.3,
                provenance=LabelProvenanceType.AUTOMATED_MODEL
            )
        )
        
        label_bundles.append(bundle)
    
    print(f"Created {len(conversations)} test conversations with labels")
    
    # Test distribution analysis
    analyzer = create_analyzer()
    dist_report = analyzer.generate_distribution_report(label_bundles)
    
    print(f"Distribution Report Generated:")
    print(f"Total conversations: {dist_report.total_conversations}")
    print(f"Therapeutic response distribution: {dist_report.therapeutic_response_distribution}")
    print(f"Crisis level distribution: {dist_report.crisis_level_distribution}")
    print(f"Provenance distribution: {dist_report.provenance_distribution}")
    print(f"Confidence statistics: {dist_report.confidence_statistics}")
    
    # Find rare categories
    rare_categories = analyzer.find_rare_categories(label_bundles, threshold_ratio=0.1)
    print(f"\nRare categories (threshold 10%): {rare_categories}")
    
    # Test edge case detection
    detector = create_edge_case_detector(low_confidence_threshold=0.7)
    edge_report = detector.detect_edge_cases(conversations, label_bundles)
    
    print(f"\nEdge Case Report Generated:")
    print(f"Low confidence items: {len(edge_report.low_confidence_items)}")
    print(f"Conflicting labels: {len(edge_report.conflicting_labels)}")
    print(f"High risk items: {len(edge_report.high_risk_items)}")
    print(f"Inconsistent patterns: {len(edge_report.inconsistent_patterns)}")
    
    # Print detailed summaries
    visualizer = DistributionVisualizer()
    visualizer.print_distribution_summary(dist_report)
    visualizer.print_edge_case_summary(edge_report)


if __name__ == "__main__":
    test_label_analysis()