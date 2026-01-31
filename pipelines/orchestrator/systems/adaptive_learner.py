#!/usr/bin/env python3
"""
Adaptive Learning Pipeline
Continuously improves dataset quality based on model performance feedback.
"""

import logging
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any

import numpy as np

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AdaptationStrategy(Enum):
    """Adaptation strategies."""

    QUALITY_THRESHOLD_ADJUSTMENT = "quality_threshold_adjustment"
    REBALANCING = "rebalancing"
    FILTERING = "filtering"
    AUGMENTATION = "augmentation"
    PRIORITIZATION = "prioritization"


class PerformanceMetric(Enum):
    """Model performance metrics."""

    ACCURACY = "accuracy"
    F1_SCORE = "f1_score"
    PERPLEXITY = "perplexity"
    BLEU_SCORE = "bleu_score"
    ROUGE_SCORE = "rouge_score"
    THERAPEUTIC_EFFECTIVENESS = "therapeutic_effectiveness"
    SAFETY_COMPLIANCE = "safety_compliance"


@dataclass
class ModelFeedback:
    """Model performance feedback."""

    model_id: str
    dataset_version: str
    performance_metrics: dict[PerformanceMetric, float]
    training_time: float
    convergence_epochs: int
    validation_loss: float
    specific_issues: list[str] = field(default_factory=list)
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class AdaptationAction:
    """Adaptation action taken."""

    action_id: str = ""
    strategy: AdaptationStrategy = AdaptationStrategy.QUALITY_THRESHOLD_ADJUSTMENT
    parameters: dict[str, Any] = field(default_factory=dict)
    rationale: str = ""
    expected_improvement: float = 0.0
    timestamp: datetime = field(default_factory=datetime.now)
    applied: bool = False
    results: dict[str, Any] | None = None


@dataclass
class LearningState:
    """Current learning state."""

    current_quality_threshold: float
    adaptation_rate: float
    performance_history: list[ModelFeedback] = field(default_factory=list)
    adaptation_history: list[AdaptationAction] = field(default_factory=list)
    last_adaptation: datetime | None = None


class AdaptiveLearner:
    """
    Adaptive learning pipeline for continuous dataset improvement.
    """

    def __init__(self):
        """Initialize the adaptive learner."""
        self.learning_state = LearningState(
            current_quality_threshold=0.7, adaptation_rate=0.1
        )

        self.adaptation_strategies = self._initialize_strategies()
        self.performance_callbacks: list[Callable] = []

        # Learning parameters
        self.min_feedback_samples = 3
        self.adaptation_cooldown = timedelta(hours=6)
        self.performance_window = timedelta(days=7)

        # Monitoring
        self.is_learning = False
        self.learning_thread = None

    def _initialize_strategies(self) -> dict[AdaptationStrategy, dict[str, Any]]:
        """Initialize adaptation strategies."""
        return {
            AdaptationStrategy.QUALITY_THRESHOLD_ADJUSTMENT: {
                "min_threshold": 0.5,
                "max_threshold": 0.95,
                "adjustment_factor": 0.05,
                "performance_correlation": 0.8,
            },
            AdaptationStrategy.REBALANCING: {
                "min_samples_per_category": 100,
                "max_imbalance_ratio": 10.0,
                "rebalancing_strength": 0.3,
            },
            AdaptationStrategy.FILTERING: {
                "outlier_threshold": 2.0,
                "consistency_threshold": 0.6,
                "safety_threshold": 0.9,
            },
            AdaptationStrategy.AUGMENTATION: {
                "augmentation_ratio": 0.2,
                "quality_boost_factor": 0.1,
                "diversity_target": 0.8,
            },
            AdaptationStrategy.PRIORITIZATION: {
                "priority_boost": 0.15,
                "effectiveness_weight": 0.4,
                "recency_weight": 0.3,
            },
        }

    def start_adaptive_learning(self):
        """Start the adaptive learning process."""
        if self.is_learning:
            logger.warning("Adaptive learning already started")
            return

        self.is_learning = True
        self.learning_thread = threading.Thread(target=self._learning_loop, daemon=True)
        self.learning_thread.start()

        logger.info("Adaptive learning pipeline started")

    def stop_adaptive_learning(self):
        """Stop the adaptive learning process."""
        self.is_learning = False

        if self.learning_thread:
            self.learning_thread.join(timeout=5.0)

        logger.info("Adaptive learning pipeline stopped")

    def submit_model_feedback(self, feedback: ModelFeedback):
        """Submit model performance feedback."""
        logger.info(f"Received feedback from model {feedback.model_id}")

        # Add to performance history
        self.learning_state.performance_history.append(feedback)

        # Trigger adaptation analysis
        self._analyze_and_adapt(feedback)

        # Notify callbacks
        for callback in self.performance_callbacks:
            try:
                callback(feedback)
            except Exception as e:
                logger.error(f"Error in performance callback: {e}")

    def add_performance_callback(self, callback: Callable[[ModelFeedback], None]):
        """Add callback for performance feedback."""
        self.performance_callbacks.append(callback)

    def _learning_loop(self):
        """Main adaptive learning loop."""
        while self.is_learning:
            try:
                # Periodic adaptation analysis
                self._periodic_adaptation_analysis()

                # Clean up old data
                self._cleanup_old_data()

                # Sleep for a while
                time.sleep(300)  # 5 minutes

            except Exception as e:
                logger.error(f"Error in learning loop: {e}")

    def _analyze_and_adapt(self, feedback: ModelFeedback):
        """Analyze feedback and determine adaptations."""
        logger.info("Analyzing feedback for adaptation opportunities")

        # Check if we have enough data for adaptation
        if len(self.learning_state.performance_history) < self.min_feedback_samples:
            logger.info("Insufficient feedback samples for adaptation")
            return

        # Check cooldown period
        if (
            self.learning_state.last_adaptation
            and datetime.now() - self.learning_state.last_adaptation
            < self.adaptation_cooldown
        ):
            logger.info("Adaptation cooldown period active")
            return

        # Analyze performance trends
        performance_trend = self._analyze_performance_trend()

        # Determine adaptation strategies
        adaptations = self._determine_adaptations(feedback, performance_trend)

        # Apply adaptations
        for adaptation in adaptations:
            self._apply_adaptation(adaptation)

    def _periodic_adaptation_analysis(self):
        """Perform periodic adaptation analysis."""
        if not self.learning_state.performance_history:
            return

        # Analyze long-term trends
        self._analyze_long_term_performance()

        # Check for systematic issues
        systematic_issues = self._identify_systematic_issues()

        # Generate proactive adaptations
        if systematic_issues:
            proactive_adaptations = self._generate_proactive_adaptations(
                systematic_issues
            )

            for adaptation in proactive_adaptations:
                self._apply_adaptation(adaptation)

    def _analyze_performance_trend(self) -> dict[str, float]:
        """Analyze recent performance trends."""
        recent_feedback = self._get_recent_feedback()

        if len(recent_feedback) < 2:
            return {}

        trends = {}

        for metric in PerformanceMetric:
            values = [fb.performance_metrics.get(metric, 0.0) for fb in recent_feedback]

            if len(values) >= 2:
                # Simple linear trend calculation
                x = np.arange(len(values))
                y = np.array(values)

                if len(x) > 1 and np.std(x) > 0:
                    correlation = (
                        np.corrcoef(x, y)[0, 1]
                        if not np.isnan(np.corrcoef(x, y)[0, 1])
                        else 0.0
                    )
                    trends[metric.value] = correlation

        return trends

    def _analyze_long_term_performance(self) -> dict[str, Any]:
        """Analyze long-term performance patterns."""
        cutoff_time = datetime.now() - self.performance_window
        relevant_feedback = [
            fb
            for fb in self.learning_state.performance_history
            if fb.timestamp >= cutoff_time
        ]

        if len(relevant_feedback) < 3:
            return {}

        analysis = {
            "average_performance": {},
            "performance_stability": {},
            "improvement_rate": {},
        }

        for metric in PerformanceMetric:
            values = [
                fb.performance_metrics.get(metric, 0.0) for fb in relevant_feedback
            ]

            if values:
                analysis["average_performance"][metric.value] = np.mean(values)
                analysis["performance_stability"][metric.value] = 1.0 / (
                    1.0 + np.std(values)
                )

                # Calculate improvement rate
                if len(values) >= 2:
                    recent_avg = np.mean(values[-len(values) // 2 :])
                    older_avg = np.mean(values[: len(values) // 2])
                    improvement = (recent_avg - older_avg) / max(older_avg, 0.001)
                    analysis["improvement_rate"][metric.value] = improvement

        return analysis

    def _identify_systematic_issues(self) -> list[str]:
        """Identify systematic performance issues."""
        issues = []
        recent_feedback = self._get_recent_feedback()

        if len(recent_feedback) < 3:
            return issues

        # Check for consistent low performance
        for metric in PerformanceMetric:
            values = [fb.performance_metrics.get(metric, 0.0) for fb in recent_feedback]

            if values and np.mean(values) < 0.6:
                issues.append(f"Consistently low {metric.value}")

        # Check for high variance
        for metric in PerformanceMetric:
            values = [fb.performance_metrics.get(metric, 0.0) for fb in recent_feedback]

            if values and np.std(values) > 0.2:
                issues.append(f"High variance in {metric.value}")

        # Check for convergence issues
        convergence_epochs = [fb.convergence_epochs for fb in recent_feedback]
        if convergence_epochs and np.mean(convergence_epochs) > 50:
            issues.append("Slow convergence")

        # Check for specific issues mentioned in feedback
        all_specific_issues = [
            issue for fb in recent_feedback for issue in fb.specific_issues
        ]
        issue_counts = {}
        for issue in all_specific_issues:
            issue_counts[issue] = issue_counts.get(issue, 0) + 1

        # Add frequently mentioned issues
        for issue, count in issue_counts.items():
            if count >= len(recent_feedback) // 2:
                issues.append(f"Frequent issue: {issue}")

        return issues

    def _determine_adaptations(
        self, feedback: ModelFeedback, performance_trend: dict[str, float]
    ) -> list[AdaptationAction]:
        """Determine appropriate adaptations based on feedback."""
        adaptations = []

        # Quality threshold adjustment
        if self._should_adjust_quality_threshold(feedback, performance_trend):
            adaptation = self._create_quality_threshold_adaptation(
                feedback, performance_trend
            )
            adaptations.append(adaptation)

        # Rebalancing
        if self._should_rebalance(feedback):
            adaptation = self._create_rebalancing_adaptation(feedback)
            adaptations.append(adaptation)

        # Filtering
        if self._should_filter(feedback):
            adaptation = self._create_filtering_adaptation(feedback)
            adaptations.append(adaptation)

        # Augmentation
        if self._should_augment(feedback):
            adaptation = self._create_augmentation_adaptation(feedback)
            adaptations.append(adaptation)

        # Prioritization
        if self._should_prioritize(feedback):
            adaptation = self._create_prioritization_adaptation(feedback)
            adaptations.append(adaptation)

        return adaptations

    def _should_adjust_quality_threshold(
        self, feedback: ModelFeedback, trends: dict[str, float]
    ) -> bool:
        """Determine if quality threshold should be adjusted."""
        # Adjust if performance is consistently low or high
        avg_performance = np.mean(list(feedback.performance_metrics.values()))

        if (
            avg_performance < 0.6
            and self.learning_state.current_quality_threshold > 0.5
        ):
            return True  # Lower threshold
        if (
            avg_performance > 0.9
            and self.learning_state.current_quality_threshold < 0.9
        ):
            return True  # Raise threshold

        return False

    def _should_rebalance(self, feedback: ModelFeedback) -> bool:
        """Determine if dataset should be rebalanced."""
        # Check for class imbalance issues
        return "class_imbalance" in feedback.specific_issues

    def _should_filter(self, feedback: ModelFeedback) -> bool:
        """Determine if dataset should be filtered."""
        # Filter if there are quality or safety issues
        quality_issues = ["low_quality", "inconsistent_data", "safety_violations"]
        return any(issue in feedback.specific_issues for issue in quality_issues)

    def _should_augment(self, feedback: ModelFeedback) -> bool:
        """Determine if dataset should be augmented."""
        # Augment if there's insufficient data or diversity
        diversity_issues = ["insufficient_data", "low_diversity", "overfitting"]
        return any(issue in feedback.specific_issues for issue in diversity_issues)

    def _should_prioritize(self, feedback: ModelFeedback) -> bool:
        """Determine if dataset prioritization should be adjusted."""
        # Prioritize if certain types of data are more effective
        return "effectiveness_imbalance" in feedback.specific_issues

    def _create_quality_threshold_adaptation(
        self, feedback: ModelFeedback, trends: dict[str, float]
    ) -> AdaptationAction:
        """Create quality threshold adaptation."""
        current_threshold = self.learning_state.current_quality_threshold
        avg_performance = np.mean(list(feedback.performance_metrics.values()))

        if avg_performance < 0.6:
            # Lower threshold to include more data
            new_threshold = max(0.5, current_threshold - 0.05)
            rationale = "Lowering quality threshold to include more training data"
        else:
            # Raise threshold to improve quality
            new_threshold = min(0.95, current_threshold + 0.05)
            rationale = "Raising quality threshold to improve data quality"

        return AdaptationAction(
            action_id=f"quality_threshold_{int(time.time())}",
            strategy=AdaptationStrategy.QUALITY_THRESHOLD_ADJUSTMENT,
            parameters={
                "new_threshold": new_threshold,
                "old_threshold": current_threshold,
            },
            rationale=rationale,
            expected_improvement=0.1,
        )

    def _create_rebalancing_adaptation(
        self, feedback: ModelFeedback
    ) -> AdaptationAction:
        """Create rebalancing adaptation."""
        return AdaptationAction(
            action_id=f"rebalancing_{int(time.time())}",
            strategy=AdaptationStrategy.REBALANCING,
            parameters={"target_balance": 0.8, "method": "oversample_minority"},
            rationale="Rebalancing dataset to address class imbalance",
            expected_improvement=0.15,
        )

    def _create_filtering_adaptation(self, feedback: ModelFeedback) -> AdaptationAction:
        """Create filtering adaptation."""
        return AdaptationAction(
            action_id=f"filtering_{int(time.time())}",
            strategy=AdaptationStrategy.FILTERING,
            parameters={"outlier_removal": True, "safety_filtering": True},
            rationale="Filtering dataset to remove low-quality or unsafe content",
            expected_improvement=0.12,
        )

    def _create_augmentation_adaptation(
        self, feedback: ModelFeedback
    ) -> AdaptationAction:
        """Create augmentation adaptation."""
        return AdaptationAction(
            action_id=f"augmentation_{int(time.time())}",
            strategy=AdaptationStrategy.AUGMENTATION,
            parameters={"augmentation_factor": 1.5, "diversity_boost": True},
            rationale="Augmenting dataset to increase diversity and size",
            expected_improvement=0.18,
        )

    def _create_prioritization_adaptation(
        self, feedback: ModelFeedback
    ) -> AdaptationAction:
        """Create prioritization adaptation."""
        return AdaptationAction(
            action_id=f"prioritization_{int(time.time())}",
            strategy=AdaptationStrategy.PRIORITIZATION,
            parameters={"effectiveness_weight": 0.6, "recency_weight": 0.4},
            rationale="Adjusting data prioritization based on effectiveness",
            expected_improvement=0.14,
        )

    def _generate_proactive_adaptations(
        self, issues: list[str]
    ) -> list[AdaptationAction]:
        """Generate proactive adaptations for systematic issues."""
        adaptations = []

        for issue in issues:
            if "low" in issue and "performance" in issue:
                # Proactive quality improvement
                adaptation = AdaptationAction(
                    action_id=f"proactive_quality_{int(time.time())}",
                    strategy=AdaptationStrategy.QUALITY_THRESHOLD_ADJUSTMENT,
                    parameters={"adjustment": 0.1, "direction": "increase"},
                    rationale=f"Proactive adaptation for {issue}",
                    expected_improvement=0.08,
                )
                adaptations.append(adaptation)

            elif "variance" in issue:
                # Proactive filtering
                adaptation = AdaptationAction(
                    action_id=f"proactive_filter_{int(time.time())}",
                    strategy=AdaptationStrategy.FILTERING,
                    parameters={"consistency_filtering": True},
                    rationale=f"Proactive filtering for {issue}",
                    expected_improvement=0.06,
                )
                adaptations.append(adaptation)

        return adaptations

    def _apply_adaptation(self, adaptation: AdaptationAction):
        """Apply an adaptation action."""
        logger.info(f"Applying adaptation: {adaptation.strategy.value}")
        logger.info(f"Rationale: {adaptation.rationale}")

        try:
            if adaptation.strategy == AdaptationStrategy.QUALITY_THRESHOLD_ADJUSTMENT:
                self._apply_quality_threshold_adjustment(adaptation)
            elif adaptation.strategy == AdaptationStrategy.REBALANCING:
                self._apply_rebalancing(adaptation)
            elif adaptation.strategy == AdaptationStrategy.FILTERING:
                self._apply_filtering(adaptation)
            elif adaptation.strategy == AdaptationStrategy.AUGMENTATION:
                self._apply_augmentation(adaptation)
            elif adaptation.strategy == AdaptationStrategy.PRIORITIZATION:
                self._apply_prioritization(adaptation)

            adaptation.applied = True
            self.learning_state.adaptation_history.append(adaptation)
            self.learning_state.last_adaptation = datetime.now()

            logger.info(f"Successfully applied adaptation {adaptation.action_id}")

        except Exception as e:
            logger.error(f"Failed to apply adaptation {adaptation.action_id}: {e}")
            adaptation.results = {"error": str(e)}

    def _apply_quality_threshold_adjustment(self, adaptation: AdaptationAction):
        """Apply quality threshold adjustment."""
        new_threshold = adaptation.parameters.get("new_threshold")
        if new_threshold:
            old_threshold = self.learning_state.current_quality_threshold
            self.learning_state.current_quality_threshold = new_threshold

            adaptation.results = {
                "old_threshold": old_threshold,
                "new_threshold": new_threshold,
                "change": new_threshold - old_threshold,
            }

    def _apply_rebalancing(self, adaptation: AdaptationAction):
        """Apply dataset rebalancing."""
        # This would integrate with the actual dataset balancing system
        adaptation.results = {"rebalancing_applied": True}
        logger.info("Dataset rebalancing triggered")

    def _apply_filtering(self, adaptation: AdaptationAction):
        """Apply dataset filtering."""
        # This would integrate with the actual filtering system
        adaptation.results = {"filtering_applied": True}
        logger.info("Dataset filtering triggered")

    def _apply_augmentation(self, adaptation: AdaptationAction):
        """Apply dataset augmentation."""
        # This would integrate with the actual augmentation system
        adaptation.results = {"augmentation_applied": True}
        logger.info("Dataset augmentation triggered")

    def _apply_prioritization(self, adaptation: AdaptationAction):
        """Apply dataset prioritization."""
        # This would integrate with the actual prioritization system
        adaptation.results = {"prioritization_applied": True}
        logger.info("Dataset prioritization updated")

    def _get_recent_feedback(self, hours: int = 24) -> list[ModelFeedback]:
        """Get recent feedback within specified hours."""
        cutoff_time = datetime.now() - timedelta(hours=hours)
        return [
            fb
            for fb in self.learning_state.performance_history
            if fb.timestamp >= cutoff_time
        ]

    def _cleanup_old_data(self):
        """Clean up old performance data."""
        cutoff_time = datetime.now() - timedelta(days=30)

        # Keep recent performance history
        self.learning_state.performance_history = [
            fb
            for fb in self.learning_state.performance_history
            if fb.timestamp >= cutoff_time
        ]

        # Keep recent adaptation history
        self.learning_state.adaptation_history = [
            ad
            for ad in self.learning_state.adaptation_history
            if ad.timestamp >= cutoff_time
        ]

    def get_learning_summary(self) -> dict[str, Any]:
        """Get adaptive learning summary."""
        recent_feedback = self._get_recent_feedback(hours=168)  # 1 week
        recent_adaptations = [
            ad
            for ad in self.learning_state.adaptation_history
            if ad.timestamp >= datetime.now() - timedelta(days=7)
        ]

        # Calculate adaptation effectiveness
        adaptation_effectiveness = 0.0
        if recent_adaptations:
            successful_adaptations = [ad for ad in recent_adaptations if ad.applied]
            adaptation_effectiveness = len(successful_adaptations) / len(
                recent_adaptations
            )

        # Performance trend
        performance_trend = "stable"
        if len(recent_feedback) >= 2:
            recent_avg = np.mean(
                [
                    np.mean(list(fb.performance_metrics.values()))
                    for fb in recent_feedback[-3:]
                ]
            )
            older_avg = np.mean(
                [
                    np.mean(list(fb.performance_metrics.values()))
                    for fb in recent_feedback[:3]
                ]
            )

            if recent_avg > older_avg + 0.05:
                performance_trend = "improving"
            elif recent_avg < older_avg - 0.05:
                performance_trend = "declining"

        return {
            "learning_active": self.is_learning,
            "current_quality_threshold": self.learning_state.current_quality_threshold,
            "adaptation_rate": self.learning_state.adaptation_rate,
            "recent_feedback_count": len(recent_feedback),
            "recent_adaptations_count": len(recent_adaptations),
            "adaptation_effectiveness": adaptation_effectiveness,
            "performance_trend": performance_trend,
            "last_adaptation": (
                self.learning_state.last_adaptation.isoformat()
                if self.learning_state.last_adaptation
                else None
            ),
        }


def main():
    """Example usage of the AdaptiveLearner."""
    learner = AdaptiveLearner()

    # Start adaptive learning
    learner.start_adaptive_learning()

    try:
        # Simulate model feedback
        feedback1 = ModelFeedback(
            model_id="therapeutic_model_v1",
            dataset_version="v1.0",
            performance_metrics={
                PerformanceMetric.ACCURACY: 0.75,
                PerformanceMetric.F1_SCORE: 0.72,
                PerformanceMetric.THERAPEUTIC_EFFECTIVENESS: 0.68,
                PerformanceMetric.SAFETY_COMPLIANCE: 0.85,
            },
            training_time=120.5,
            convergence_epochs=25,
            validation_loss=0.45,
            specific_issues=["class_imbalance", "low_diversity"],
        )

        feedback2 = ModelFeedback(
            model_id="therapeutic_model_v2",
            dataset_version="v1.1",
            performance_metrics={
                PerformanceMetric.ACCURACY: 0.71,
                PerformanceMetric.F1_SCORE: 0.69,
                PerformanceMetric.THERAPEUTIC_EFFECTIVENESS: 0.65,
                PerformanceMetric.SAFETY_COMPLIANCE: 0.82,
            },
            training_time=135.2,
            convergence_epochs=32,
            validation_loss=0.52,
            specific_issues=["slow_convergence", "overfitting"],
        )

        # Submit feedback
        learner.submit_model_feedback(feedback1)
        time.sleep(1)
        learner.submit_model_feedback(feedback2)

        # Wait for processing
        time.sleep(2)

        # Print learning summary
        learner.get_learning_summary()

    finally:
        # Stop adaptive learning
        learner.stop_adaptive_learning()


if __name__ == "__main__":
    main()
