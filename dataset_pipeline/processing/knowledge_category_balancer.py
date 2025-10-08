"""
Knowledge category balancer for psychology knowledge integration pipeline.

This module ensures balanced representation across different psychology knowledge
categories (DSM-5, PDM-2, Big Five) in generated therapeutic training datasets,
preventing bias and ensuring comprehensive coverage of clinical presentations.
"""

import json
import random
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any

from big_five_processor import BigFiveProcessor, PersonalityFactor
from client_scenario_generator import (
    ClientScenario,
    DemographicCategory,
    ScenarioType,
    SeverityLevel,
)
from conversation_schema import Conversation
from dsm5_parser import DSM5Parser, DSMCategory
from logger import get_logger
from pdm2_parser import PDM2Parser

logger = get_logger("dataset_pipeline.knowledge_category_balancer")


class BalanceStrategy(Enum):
    """Strategies for balancing knowledge categories."""
    EQUAL_DISTRIBUTION = "equal_distribution"  # Equal representation across all categories
    CLINICAL_PREVALENCE = "clinical_prevalence"  # Based on real-world prevalence
    THERAPEUTIC_PRIORITY = "therapeutic_priority"  # Prioritize common therapeutic scenarios
    CUSTOM_WEIGHTS = "custom_weights"  # User-defined category weights


class BalanceMetric(Enum):
    """Metrics for measuring dataset balance."""
    ENTROPY = "entropy"  # Information entropy across categories
    GINI_COEFFICIENT = "gini_coefficient"  # Inequality measure
    CHI_SQUARE = "chi_square"  # Goodness of fit test
    VARIANCE = "variance"  # Distribution variance


@dataclass
class CategoryDistribution:
    """Distribution statistics for a knowledge category."""
    category_name: str
    total_count: int
    percentage: float
    target_percentage: float
    deviation: float
    subcategories: dict[str, int] = field(default_factory=dict)


@dataclass
class BalanceReport:
    """Comprehensive balance analysis report."""
    dataset_id: str
    total_items: int
    balance_score: float  # 0.0 to 1.0, higher is better
    strategy_used: BalanceStrategy
    distributions: dict[str, CategoryDistribution] = field(default_factory=dict)
    recommendations: list[str] = field(default_factory=list)
    metrics: dict[str, float] = field(default_factory=dict)
    generated_at: str | None = None


class KnowledgeCategoryBalancer:
    """
    Comprehensive knowledge category balancer.

    Analyzes and balances representation across DSM-5 disorders, PDM-2 patterns,
    Big Five personality profiles, and other clinical dimensions to ensure
    comprehensive and unbiased therapeutic training datasets.
    """

    def __init__(self, config: dict[str, Any] | None = None):
        """Initialize the knowledge category balancer."""
        self.config = config or {}

        # Initialize psychology knowledge parsers
        self.dsm5_parser = DSM5Parser()
        self.pdm2_parser = PDM2Parser()
        self.big_five_processor = BigFiveProcessor()

        # Initialize balance targets and weights
        self._initialize_balance_targets()

        logger.info("Knowledge Category Balancer initialized")

    def _initialize_balance_targets(self) -> None:
        """Initialize target distributions for different balance strategies."""

        # Equal distribution targets
        self.equal_targets = {
            "dsm5_categories": {
                DSMCategory.ANXIETY.value: 1/8,
                DSMCategory.DEPRESSIVE.value: 1/8,
                DSMCategory.BIPOLAR.value: 1/8,
                DSMCategory.TRAUMA_STRESSOR.value: 1/8,
                DSMCategory.OBSESSIVE_COMPULSIVE.value: 1/8,
                DSMCategory.NEURODEVELOPMENTAL.value: 1/8,
                DSMCategory.PERSONALITY.value: 1/8,
                DSMCategory.SUBSTANCE_RELATED.value: 1/8
            },
            "pdm2_attachment": {
                "secure": 0.25,
                "anxious_preoccupied": 0.25,
                "dismissive_avoidant": 0.25,
                "disorganized": 0.25
            },
            "big_five_factors": {
                PersonalityFactor.OPENNESS.value: 0.2,
                PersonalityFactor.CONSCIENTIOUSNESS.value: 0.2,
                PersonalityFactor.EXTRAVERSION.value: 0.2,
                PersonalityFactor.AGREEABLENESS.value: 0.2,
                PersonalityFactor.NEUROTICISM.value: 0.2
            },
            "severity_levels": {
                SeverityLevel.MILD.value: 0.25,
                SeverityLevel.MODERATE.value: 0.25,
                SeverityLevel.SEVERE.value: 0.25,
                SeverityLevel.CRISIS.value: 0.25
            },
            "scenario_types": {
                ScenarioType.INITIAL_ASSESSMENT.value: 0.25,
                ScenarioType.DIAGNOSTIC_INTERVIEW.value: 0.25,
                ScenarioType.THERAPEUTIC_SESSION.value: 0.25,
                ScenarioType.CRISIS_INTERVENTION.value: 0.25
            }
        }

        # Clinical prevalence-based targets (approximating real-world frequencies)
        self.prevalence_targets = {
            "dsm5_categories": {
                DSMCategory.ANXIETY.value: 0.30,  # Most common
                DSMCategory.DEPRESSIVE.value: 0.25,
                DSMCategory.TRAUMA_STRESSOR.value: 0.15,
                DSMCategory.SUBSTANCE_RELATED.value: 0.10,
                DSMCategory.BIPOLAR.value: 0.08,
                DSMCategory.OBSESSIVE_COMPULSIVE.value: 0.05,
                DSMCategory.PERSONALITY.value: 0.04,
                DSMCategory.NEURODEVELOPMENTAL.value: 0.03
            },
            "severity_levels": {
                SeverityLevel.MILD.value: 0.40,
                SeverityLevel.MODERATE.value: 0.35,
                SeverityLevel.SEVERE.value: 0.20,
                SeverityLevel.CRISIS.value: 0.05
            },
            "scenario_types": {
                ScenarioType.THERAPEUTIC_SESSION.value: 0.50,
                ScenarioType.INITIAL_ASSESSMENT.value: 0.30,
                ScenarioType.DIAGNOSTIC_INTERVIEW.value: 0.15,
                ScenarioType.CRISIS_INTERVENTION.value: 0.05
            }
        }

        # Therapeutic priority targets (emphasizing training value)
        self.therapeutic_targets = {
            "dsm5_categories": {
                DSMCategory.ANXIETY.value: 0.25,
                DSMCategory.DEPRESSIVE.value: 0.25,
                DSMCategory.TRAUMA_STRESSOR.value: 0.20,
                DSMCategory.BIPOLAR.value: 0.10,
                DSMCategory.OBSESSIVE_COMPULSIVE.value: 0.08,
                DSMCategory.PERSONALITY.value: 0.07,
                DSMCategory.SUBSTANCE_RELATED.value: 0.03,
                DSMCategory.NEURODEVELOPMENTAL.value: 0.02
            },
            "severity_levels": {
                SeverityLevel.MODERATE.value: 0.40,
                SeverityLevel.SEVERE.value: 0.30,
                SeverityLevel.MILD.value: 0.20,
                SeverityLevel.CRISIS.value: 0.10
            }
        }

        logger.info("Initialized balance targets for different strategies")

    def analyze_dataset_balance(
        self,
        scenarios: list[ClientScenario],
        conversations: list[Conversation] | None = None,
        strategy: BalanceStrategy = BalanceStrategy.EQUAL_DISTRIBUTION
    ) -> BalanceReport:
        """Analyze the balance of knowledge categories in a dataset."""

        if not scenarios:
            return BalanceReport(
                dataset_id="empty_dataset",
                total_items=0,
                balance_score=0.0,
                strategy_used=strategy,
                generated_at=datetime.now().isoformat()
            )

        # Get target distributions based on strategy
        targets = self._get_target_distributions(strategy)

        # Analyze current distributions
        distributions = self._analyze_current_distributions(scenarios)

        # Calculate balance metrics
        balance_score = self._calculate_balance_score(distributions, targets)
        metrics = self._calculate_balance_metrics(distributions, targets)

        # Generate recommendations
        recommendations = self._generate_balance_recommendations(distributions, targets)

        # Create distribution objects
        distribution_objects = {}
        for category, dist_data in distributions.items():
            if category in targets:
                target_dist = targets[category]
                distribution_objects[category] = CategoryDistribution(
                    category_name=category,
                    total_count=sum(dist_data.values()),
                    percentage=sum(dist_data.values()) / len(scenarios),
                    target_percentage=sum(target_dist.values()) if isinstance(target_dist, dict) else target_dist,
                    deviation=self._calculate_category_deviation(dist_data, target_dist, len(scenarios)),
                    subcategories=dist_data
                )

        report = BalanceReport(
            dataset_id=f"dataset_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            total_items=len(scenarios),
            balance_score=balance_score,
            strategy_used=strategy,
            distributions=distribution_objects,
            recommendations=recommendations,
            metrics=metrics,
            generated_at=datetime.now().isoformat()
        )

        logger.info(f"Analyzed dataset balance: {len(scenarios)} scenarios, score: {balance_score:.2f}")
        return report

    def _get_target_distributions(self, strategy: BalanceStrategy) -> dict[str, dict[str, float]]:
        """Get target distributions based on balance strategy."""
        if strategy == BalanceStrategy.EQUAL_DISTRIBUTION:
            return self.equal_targets
        if strategy == BalanceStrategy.CLINICAL_PREVALENCE:
            return self.prevalence_targets
        if strategy == BalanceStrategy.THERAPEUTIC_PRIORITY:
            return self.therapeutic_targets
        return self.equal_targets  # Default fallback

    def _analyze_current_distributions(self, scenarios: list[ClientScenario]) -> dict[str, dict[str, int]]:
        """Analyze current distributions across knowledge categories."""
        distributions = {
            "dsm5_categories": defaultdict(int),
            "pdm2_attachment": defaultdict(int),
            "big_five_factors": defaultdict(int),
            "severity_levels": defaultdict(int),
            "scenario_types": defaultdict(int),
            "demographic_categories": defaultdict(int)
        }

        for scenario in scenarios:
            # DSM-5 categories
            if scenario.clinical_formulation.dsm5_considerations:
                # Extract category from first DSM-5 consideration
                for consideration in scenario.clinical_formulation.dsm5_considerations:
                    for disorder in self.dsm5_parser.get_disorders():
                        if disorder.name.lower() in consideration.lower():
                            distributions["dsm5_categories"][disorder.category.value] += 1
                            break

            # PDM-2 attachment patterns
            if scenario.clinical_formulation.attachment_style:
                attachment = scenario.clinical_formulation.attachment_style.lower()
                if "secure" in attachment:
                    distributions["pdm2_attachment"]["secure"] += 1
                elif "anxious" in attachment or "preoccupied" in attachment:
                    distributions["pdm2_attachment"]["anxious_preoccupied"] += 1
                elif "avoidant" in attachment or "dismissive" in attachment:
                    distributions["pdm2_attachment"]["dismissive_avoidant"] += 1
                else:
                    distributions["pdm2_attachment"]["disorganized"] += 1

            # Big Five factors (dominant factor)
            personality = scenario.clinical_formulation.personality_profile
            if personality:
                # Find dominant factor
                max_factor = None
                max_value = 0
                for factor, value in personality.items():
                    if isinstance(value, str) and value in ["high", "very high"]:
                        score = 4 if value == "very high" else 3
                        if score > max_value:
                            max_value = score
                            max_factor = factor

                if max_factor:
                    distributions["big_five_factors"][max_factor] += 1

            # Severity levels
            distributions["severity_levels"][scenario.severity_level.value] += 1

            # Scenario types
            distributions["scenario_types"][scenario.scenario_type.value] += 1

            # Demographic categories
            distributions["demographic_categories"][scenario.demographics.occupation] += 1

        # Convert defaultdicts to regular dicts
        return {k: dict(v) for k, v in distributions.items()}

    def _calculate_balance_score(
        self,
        distributions: dict[str, dict[str, int]],
        targets: dict[str, dict[str, float]]
    ) -> float:
        """Calculate overall balance score (0.0 to 1.0, higher is better)."""

        category_scores = []

        for category, dist_data in distributions.items():
            if category not in targets:
                continue

            target_dist = targets[category]
            total_items = sum(dist_data.values())

            if total_items == 0:
                category_scores.append(0.0)
                continue

            # Calculate chi-square goodness of fit
            chi_square = 0.0
            for subcategory, target_prop in target_dist.items():
                observed = dist_data.get(subcategory, 0)
                expected = target_prop * total_items

                if expected > 0:
                    chi_square += ((observed - expected) ** 2) / expected

            # Convert chi-square to score (lower chi-square = better balance)
            # Use exponential decay to map chi-square to 0-1 score
            category_score = max(0.0, 1.0 - (chi_square / (len(target_dist) * 2)))
            category_scores.append(category_score)

        # Return average score across categories
        return sum(category_scores) / len(category_scores) if category_scores else 0.0

    def _calculate_balance_metrics(
        self,
        distributions: dict[str, dict[str, int]],
        targets: dict[str, dict[str, float]]
    ) -> dict[str, float]:
        """Calculate detailed balance metrics."""
        metrics = {}

        for category, dist_data in distributions.items():
            if category not in targets:
                continue

            target_dist = targets[category]
            total_items = sum(dist_data.values())

            if total_items == 0:
                continue

            # Calculate entropy
            import math
            entropy = 0.0
            for count in dist_data.values():
                if count > 0:
                    p = count / total_items
                    entropy -= p * math.log2(p) if p > 0 else 0

            metrics[f"{category}_entropy"] = entropy

            # Calculate Gini coefficient
            sorted_counts = sorted(dist_data.values())
            n = len(sorted_counts)
            gini = 0.0
            for i, count in enumerate(sorted_counts):
                gini += (2 * (i + 1) - n - 1) * count

            if total_items > 0:
                gini = gini / (n * total_items)

            metrics[f"{category}_gini"] = gini

            # Calculate variance from target
            variance = 0.0
            for subcategory, target_prop in target_dist.items():
                observed_prop = dist_data.get(subcategory, 0) / total_items
                variance += (observed_prop - target_prop) ** 2

            metrics[f"{category}_variance"] = variance

        return metrics

    def _calculate_category_deviation(
        self,
        dist_data: dict[str, int],
        target_dist: dict[str, float],
        total_items: int
    ) -> float:
        """Calculate deviation from target distribution for a category."""
        if total_items == 0:
            return 1.0  # Maximum deviation

        total_deviation = 0.0
        for subcategory, target_prop in target_dist.items():
            observed_prop = dist_data.get(subcategory, 0) / total_items
            total_deviation += abs(observed_prop - target_prop)

        return total_deviation / len(target_dist)

    def _generate_balance_recommendations(
        self,
        distributions: dict[str, dict[str, int]],
        targets: dict[str, dict[str, float]]
    ) -> list[str]:
        """Generate recommendations for improving dataset balance."""
        recommendations = []

        for category, dist_data in distributions.items():
            if category not in targets:
                continue

            target_dist = targets[category]
            total_items = sum(dist_data.values())

            if total_items == 0:
                recommendations.append(f"Add content for {category.replace('_', ' ')}")
                continue

            # Find under-represented subcategories
            under_represented = []
            over_represented = []

            for subcategory, target_prop in target_dist.items():
                observed_prop = dist_data.get(subcategory, 0) / total_items
                deviation = observed_prop - target_prop

                if deviation < -0.05:  # More than 5% under-represented
                    under_represented.append((subcategory, abs(deviation)))
                elif deviation > 0.05:  # More than 5% over-represented
                    over_represented.append((subcategory, deviation))

            # Generate specific recommendations
            if under_represented:
                under_represented.sort(key=lambda x: x[1], reverse=True)
                top_under = under_represented[0][0]
                recommendations.append(
                    f"Increase representation of {top_under.replace('_', ' ')} in {category.replace('_', ' ')}"
                )

            if over_represented:
                over_represented.sort(key=lambda x: x[1], reverse=True)
                top_over = over_represented[0][0]
                recommendations.append(
                    f"Reduce over-representation of {top_over.replace('_', ' ')} in {category.replace('_', ' ')}"
                )

        return recommendations

    def generate_balanced_scenarios(
        self,
        target_count: int,
        strategy: BalanceStrategy = BalanceStrategy.EQUAL_DISTRIBUTION,
        existing_scenarios: list[ClientScenario] | None = None
    ) -> list[ClientScenario]:
        """Generate a balanced set of client scenarios."""
        from client_scenario_generator import ClientScenarioGenerator

        scenario_generator = ClientScenarioGenerator()
        targets = self._get_target_distributions(strategy)

        # Calculate target counts for each category
        target_counts = self._calculate_target_counts(target_count, targets)

        # Account for existing scenarios if provided
        if existing_scenarios:
            existing_distributions = self._analyze_current_distributions(existing_scenarios)
            target_counts = self._adjust_targets_for_existing(
                target_counts, existing_distributions, len(existing_scenarios)
            )

        # Generate scenarios to meet targets
        generated_scenarios = []

        # Generate DSM-5 category-based scenarios
        dsm5_targets = target_counts.get("dsm5_categories", {})
        for category_str, count in dsm5_targets.items():
            try:
                category = DSMCategory(category_str)
                disorders = [d for d in self.dsm5_parser.get_disorders() if d.category == category]

                for _ in range(count):
                    if disorders:
                        disorder = random.choice(disorders)
                        scenario = scenario_generator.generate_client_scenario(
                            target_disorder=disorder.name,
                            severity_level=self._select_balanced_severity(targets),
                            scenario_type=self._select_balanced_scenario_type(targets)
                        )
                        generated_scenarios.append(scenario)
            except ValueError:
                continue

        # Fill remaining slots with balanced selection
        remaining_count = target_count - len(generated_scenarios)
        for _ in range(remaining_count):
            scenario = scenario_generator.generate_client_scenario(
                severity_level=self._select_balanced_severity(targets),
                scenario_type=self._select_balanced_scenario_type(targets),
                demographic_category=self._select_balanced_demographic()
            )
            generated_scenarios.append(scenario)

        logger.info(f"Generated {len(generated_scenarios)} balanced scenarios using {strategy.value}")
        return generated_scenarios

    def _calculate_target_counts(
        self,
        total_count: int,
        targets: dict[str, dict[str, float]]
    ) -> dict[str, dict[str, int]]:
        """Calculate target counts for each category."""
        target_counts = {}

        for category, target_dist in targets.items():
            category_counts = {}
            for subcategory, proportion in target_dist.items():
                category_counts[subcategory] = int(total_count * proportion)
            target_counts[category] = category_counts

        return target_counts

    def _adjust_targets_for_existing(
        self,
        target_counts: dict[str, dict[str, int]],
        existing_distributions: dict[str, dict[str, int]],
        existing_total: int
    ) -> dict[str, dict[str, int]]:
        """Adjust target counts based on existing scenario distributions."""
        adjusted_counts = {}

        for category, targets in target_counts.items():
            if category in existing_distributions:
                existing_dist = existing_distributions[category]
                adjusted_category = {}

                for subcategory, target_count in targets.items():
                    existing_count = existing_dist.get(subcategory, 0)
                    needed_count = max(0, target_count - existing_count)
                    adjusted_category[subcategory] = needed_count

                adjusted_counts[category] = adjusted_category
            else:
                adjusted_counts[category] = targets

        return adjusted_counts

    def _select_balanced_severity(self, targets: dict[str, dict[str, float]]) -> SeverityLevel:
        """Select severity level based on target distribution."""
        if "severity_levels" in targets:
            severity_weights = targets["severity_levels"]
            choices = list(severity_weights.keys())
            weights = list(severity_weights.values())
            selected = random.choices(choices, weights=weights)[0]
            return SeverityLevel(selected)
        return random.choice(list(SeverityLevel))

    def _select_balanced_scenario_type(self, targets: dict[str, dict[str, float]]) -> ScenarioType:
        """Select scenario type based on target distribution."""
        if "scenario_types" in targets:
            type_weights = targets["scenario_types"]
            choices = list(type_weights.keys())
            weights = list(type_weights.values())
            selected = random.choices(choices, weights=weights)[0]
            return ScenarioType(selected)
        return random.choice(list(ScenarioType))

    def _select_balanced_demographic(self) -> DemographicCategory:
        """Select demographic category with balanced distribution."""
        return random.choice(list(DemographicCategory))

    def rebalance_existing_dataset(
        self,
        scenarios: list[ClientScenario],
        target_strategy: BalanceStrategy = BalanceStrategy.EQUAL_DISTRIBUTION,
        max_additions: int = 100
    ) -> tuple[list[ClientScenario], BalanceReport]:
        """Rebalance an existing dataset by adding scenarios to under-represented categories."""

        # Analyze current balance
        current_report = self.analyze_dataset_balance(scenarios, strategy=target_strategy)

        # Identify categories that need more representation
        additions_needed = []
        targets = self._get_target_distributions(target_strategy)

        for category, distribution in current_report.distributions.items():
            if distribution.deviation > 0.1:  # More than 10% deviation
                target_dist = targets.get(category, {})
                current_dist = distribution.subcategories
                total_current = sum(current_dist.values())

                for subcategory, target_prop in target_dist.items():
                    current_count = current_dist.get(subcategory, 0)
                    current_prop = current_count / total_current if total_current > 0 else 0

                    if current_prop < target_prop - 0.05:  # 5% under target
                        needed = int((target_prop * len(scenarios)) - current_count)
                        additions_needed.append((category, subcategory, needed))

        # Generate additional scenarios
        additional_scenarios = []
        total_additions = 0

        for category, subcategory, needed in additions_needed:
            if total_additions >= max_additions:
                break

            additions_to_make = min(needed, max_additions - total_additions)

            # Generate scenarios for this specific category/subcategory
            for _ in range(additions_to_make):
                scenario = self._generate_targeted_scenario(category, subcategory)
                if scenario:
                    additional_scenarios.append(scenario)
                    total_additions += 1

        # Combine original and additional scenarios
        balanced_scenarios = scenarios + additional_scenarios

        # Generate final balance report
        final_report = self.analyze_dataset_balance(balanced_scenarios, strategy=target_strategy)

        logger.info(f"Rebalanced dataset: added {len(additional_scenarios)} scenarios, "
                   f"balance score improved from {current_report.balance_score:.2f} to {final_report.balance_score:.2f}")

        return balanced_scenarios, final_report

    def _generate_targeted_scenario(self, category: str, subcategory: str) -> ClientScenario | None:
        """Generate a scenario targeting a specific category and subcategory."""
        from client_scenario_generator import ClientScenarioGenerator

        scenario_generator = ClientScenarioGenerator()

        try:
            if category == "dsm5_categories":
                # Find disorder matching the subcategory
                category_enum = DSMCategory(subcategory)
                disorders = [d for d in self.dsm5_parser.get_disorders() if d.category == category_enum]
                if disorders:
                    disorder = random.choice(disorders)
                    return scenario_generator.generate_client_scenario(target_disorder=disorder.name)

            elif category == "severity_levels":
                severity = SeverityLevel(subcategory)
                return scenario_generator.generate_client_scenario(severity_level=severity)

            elif category == "scenario_types":
                scenario_type = ScenarioType(subcategory)
                return scenario_generator.generate_client_scenario(scenario_type=scenario_type)

            # Default generation
            return scenario_generator.generate_client_scenario()

        except (ValueError, AttributeError):
            # Fallback to default generation
            return scenario_generator.generate_client_scenario()

    def export_balance_report(self, report: BalanceReport, output_path: Path) -> bool:
        """Export balance report to JSON format."""
        try:
            export_data = {
                "balance_report": {
                    "dataset_id": report.dataset_id,
                    "total_items": report.total_items,
                    "balance_score": report.balance_score,
                    "strategy_used": report.strategy_used.value,
                    "distributions": {},
                    "recommendations": report.recommendations,
                    "metrics": report.metrics,
                    "generated_at": report.generated_at
                }
            }

            # Convert distribution objects to dictionaries
            for category, distribution in report.distributions.items():
                export_data["balance_report"]["distributions"][category] = {
                    "category_name": distribution.category_name,
                    "total_count": distribution.total_count,
                    "percentage": distribution.percentage,
                    "target_percentage": distribution.target_percentage,
                    "deviation": distribution.deviation,
                    "subcategories": distribution.subcategories
                }

            # Ensure output directory exists
            output_path.parent.mkdir(parents=True, exist_ok=True)

            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(export_data, f, indent=2, ensure_ascii=False)

            logger.info(f"Exported balance report to {output_path}")
            return True

        except Exception as e:
            logger.error(f"Failed to export balance report: {e}")
            return False

    def get_balance_statistics(self, reports: list[BalanceReport]) -> dict[str, Any]:
        """Get comprehensive statistics across multiple balance reports."""
        if not reports:
            return {}

        stats = {
            "total_reports": len(reports),
            "average_balance_score": sum(r.balance_score for r in reports) / len(reports),
            "score_distribution": {},
            "common_recommendations": {},
            "strategy_usage": {},
            "improvement_trends": []
        }

        # Score distribution
        for report in reports:
            score_range = f"{int(report.balance_score * 10) / 10:.1f}-{int(report.balance_score * 10) / 10 + 0.1:.1f}"
            stats["score_distribution"][score_range] = stats["score_distribution"].get(score_range, 0) + 1

        # Common recommendations
        all_recommendations = []
        for report in reports:
            all_recommendations.extend(report.recommendations)

        recommendation_counts = {}
        for rec in all_recommendations:
            recommendation_counts[rec] = recommendation_counts.get(rec, 0) + 1

        stats["common_recommendations"] = dict(sorted(
            recommendation_counts.items(), key=lambda x: x[1], reverse=True
        )[:5])

        # Strategy usage
        for report in reports:
            strategy = report.strategy_used.value
            stats["strategy_usage"][strategy] = stats["strategy_usage"].get(strategy, 0) + 1

        # Improvement trends (if reports are chronologically ordered)
        if len(reports) > 1:
            scores = [r.balance_score for r in reports]
            for i in range(1, len(scores)):
                improvement = scores[i] - scores[i-1]
                stats["improvement_trends"].append(improvement)

        return stats

    def compare_balance_strategies(
        self,
        scenarios: list[ClientScenario]
    ) -> dict[BalanceStrategy, BalanceReport]:
        """Compare balance reports across different strategies."""
        strategy_reports = {}

        for strategy in BalanceStrategy:
            report = self.analyze_dataset_balance(scenarios, strategy=strategy)
            strategy_reports[strategy] = report

        logger.info(f"Compared {len(strategy_reports)} balance strategies")
        return strategy_reports

    def optimize_dataset_composition(
        self,
        target_size: int,
        strategy: BalanceStrategy = BalanceStrategy.EQUAL_DISTRIBUTION,
        quality_threshold: float = 0.8
    ) -> list[ClientScenario]:
        """Optimize dataset composition for maximum balance and quality."""

        # Generate initial balanced set
        scenarios = self.generate_balanced_scenarios(target_size, strategy)

        # Iteratively improve balance
        max_iterations = 5
        for iteration in range(max_iterations):
            report = self.analyze_dataset_balance(scenarios, strategy=strategy)

            if report.balance_score >= quality_threshold:
                logger.info(f"Achieved target balance score {report.balance_score:.2f} in {iteration + 1} iterations")
                break

            # Rebalance if needed
            scenarios, report = self.rebalance_existing_dataset(
                scenarios, strategy, max_additions=int(target_size * 0.1)
            )

        final_report = self.analyze_dataset_balance(scenarios, strategy=strategy)
        logger.info(f"Optimized dataset: {len(scenarios)} scenarios, balance score: {final_report.balance_score:.2f}")

        return scenarios

    def validate_balance_requirements(
        self,
        scenarios: list[ClientScenario],
        requirements: dict[str, dict[str, float]]
    ) -> tuple[bool, list[str]]:
        """Validate that dataset meets specific balance requirements."""

        distributions = self._analyze_current_distributions(scenarios)
        violations = []

        for category, requirements_dist in requirements.items():
            if category not in distributions:
                violations.append(f"Missing category: {category}")
                continue

            current_dist = distributions[category]
            total_items = sum(current_dist.values())

            for subcategory, min_proportion in requirements_dist.items():
                current_count = current_dist.get(subcategory, 0)
                current_proportion = current_count / total_items if total_items > 0 else 0

                if current_proportion < min_proportion:
                    violations.append(
                        f"{category}.{subcategory}: {current_proportion:.2%} < {min_proportion:.2%} required"
                    )

        is_valid = len(violations) == 0
        return is_valid, violations
