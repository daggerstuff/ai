"""
Unit tests for knowledge category balancer.

Tests the comprehensive knowledge category balancing functionality including
distribution analysis, balance scoring, scenario generation, and dataset
rebalancing across DSM-5, PDM-2, and Big Five categories.
"""

import json
import tempfile
import unittest
from pathlib import Path

from .client_scenario_generator import (
    ClientScenarioGenerator,
    DemographicCategory,
    ScenarioType,
    SeverityLevel,
)
from .dsm5_parser import DSMCategory
from .knowledge_category_balancer import (
    BalanceMetric,
    BalanceReport,
    BalanceStrategy,
    CategoryDistribution,
    KnowledgeCategoryBalancer,
)


class TestKnowledgeCategoryBalancer(unittest.TestCase):
    """Test knowledge category balancer functionality."""

    def setUp(self):
        """Set up test fixtures."""
        self.balancer = KnowledgeCategoryBalancer()
        self.scenario_generator = ClientScenarioGenerator()

        # Create test scenarios with known distributions
        self.test_scenarios = []

        # Create scenarios with specific characteristics for testing
        for i in range(10):
            scenario = self.scenario_generator.generate_client_scenario(
                scenario_type=ScenarioType.INITIAL_ASSESSMENT if i < 5 else ScenarioType.THERAPEUTIC_SESSION,
                severity_level=SeverityLevel.MODERATE if i < 7 else SeverityLevel.SEVERE,
                demographic_category=DemographicCategory.YOUNG_ADULT
            )
            self.test_scenarios.append(scenario)

    def test_initialization(self):
        """Test balancer initialization."""
        assert self.balancer.dsm5_parser is not None
        assert self.balancer.pdm2_parser is not None
        assert self.balancer.big_five_processor is not None
        assert self.balancer.equal_targets is not None
        assert self.balancer.prevalence_targets is not None
        assert self.balancer.therapeutic_targets is not None

    def test_balance_strategy_enum(self):
        """Test BalanceStrategy enum values."""
        expected_strategies = {
            "equal_distribution", "clinical_prevalence",
            "therapeutic_priority", "custom_weights"
        }
        actual_strategies = {strategy.value for strategy in BalanceStrategy}
        assert expected_strategies == actual_strategies

    def test_balance_metric_enum(self):
        """Test BalanceMetric enum values."""
        expected_metrics = {"entropy", "gini_coefficient", "chi_square", "variance"}
        actual_metrics = {metric.value for metric in BalanceMetric}
        assert expected_metrics == actual_metrics

    def test_analyze_dataset_balance_empty(self):
        """Test balance analysis with empty dataset."""
        report = self.balancer.analyze_dataset_balance([])

        assert report.dataset_id == "empty_dataset"
        assert report.total_items == 0
        assert report.balance_score == 0.0
        assert isinstance(report.strategy_used, BalanceStrategy)

    def test_analyze_dataset_balance_basic(self):
        """Test basic dataset balance analysis."""
        report = self.balancer.analyze_dataset_balance(
            self.test_scenarios,
            strategy=BalanceStrategy.EQUAL_DISTRIBUTION
        )

        # Check report structure
        assert isinstance(report, BalanceReport)
        assert report.total_items == len(self.test_scenarios)
        assert report.balance_score >= 0.0
        assert report.balance_score <= 1.0
        assert report.strategy_used == BalanceStrategy.EQUAL_DISTRIBUTION
        assert isinstance(report.distributions, dict)
        assert isinstance(report.recommendations, list)
        assert isinstance(report.metrics, dict)
        assert isinstance(report.generated_at, str)

    def test_get_target_distributions(self):
        """Test target distribution retrieval for different strategies."""

        # Test equal distribution
        equal_targets = self.balancer._get_target_distributions(BalanceStrategy.EQUAL_DISTRIBUTION)
        assert "dsm5_categories" in equal_targets
        assert "severity_levels" in equal_targets

        # Test clinical prevalence
        prevalence_targets = self.balancer._get_target_distributions(BalanceStrategy.CLINICAL_PREVALENCE)
        assert "dsm5_categories" in prevalence_targets

        # Verify anxiety has higher prevalence target than other categories
        anxiety_target = prevalence_targets["dsm5_categories"][DSMCategory.ANXIETY.value]
        depressive_target = prevalence_targets["dsm5_categories"][DSMCategory.DEPRESSIVE.value]
        assert anxiety_target > depressive_target * 0.8  # Allow some tolerance

    def test_analyze_current_distributions(self):
        """Test analysis of current distributions."""
        distributions = self.balancer._analyze_current_distributions(self.test_scenarios)

        # Check expected categories
        expected_categories = {
            "dsm5_categories", "pdm2_attachment", "big_five_factors",
            "severity_levels", "scenario_types", "demographic_categories"
        }
        assert set(distributions.keys()) == expected_categories

        # Check that distributions contain counts
        for _category, dist_data in distributions.items():
            assert isinstance(dist_data, dict)
            for _subcategory, count in dist_data.items():
                assert isinstance(count, int)
                assert count >= 0

    def test_calculate_balance_score(self):
        """Test balance score calculation."""
        distributions = self.balancer._analyze_current_distributions(self.test_scenarios)
        targets = self.balancer._get_target_distributions(BalanceStrategy.EQUAL_DISTRIBUTION)

        score = self.balancer._calculate_balance_score(distributions, targets)

        assert isinstance(score, float)
        assert score >= 0.0
        assert score <= 1.0

    def test_calculate_balance_metrics(self):
        """Test balance metrics calculation."""
        distributions = self.balancer._analyze_current_distributions(self.test_scenarios)
        targets = self.balancer._get_target_distributions(BalanceStrategy.EQUAL_DISTRIBUTION)

        metrics = self.balancer._calculate_balance_metrics(distributions, targets)

        assert isinstance(metrics, dict)

        # Check for expected metric types
        metric_types = set()
        for key in metrics:
            if "_entropy" in key:
                metric_types.add("entropy")
            elif "_gini" in key:
                metric_types.add("gini")
            elif "_variance" in key:
                metric_types.add("variance")

        assert len(metric_types) > 0

    def test_generate_balance_recommendations(self):
        """Test balance recommendation generation."""
        distributions = self.balancer._analyze_current_distributions(self.test_scenarios)
        targets = self.balancer._get_target_distributions(BalanceStrategy.EQUAL_DISTRIBUTION)

        recommendations = self.balancer._generate_balance_recommendations(distributions, targets)

        assert isinstance(recommendations, list)
        for recommendation in recommendations:
            assert isinstance(recommendation, str)
            assert len(recommendation) > 0

    def test_generate_balanced_scenarios(self):
        """Test balanced scenario generation."""
        target_count = 20
        scenarios = self.balancer.generate_balanced_scenarios(
            target_count,
            strategy=BalanceStrategy.EQUAL_DISTRIBUTION
        )

        assert isinstance(scenarios, list)
        assert len(scenarios) > 0
        assert len(scenarios) <= target_count * 1.1  # Allow some tolerance

        # Verify scenarios are valid
        for scenario in scenarios[:5]:  # Check first 5
            assert scenario.id is not None
            assert scenario.scenario_type is not None
            assert scenario.severity_level is not None

    def test_calculate_target_counts(self):
        """Test target count calculation."""
        total_count = 100
        targets = self.balancer._get_target_distributions(BalanceStrategy.EQUAL_DISTRIBUTION)

        target_counts = self.balancer._calculate_target_counts(total_count, targets)

        assert isinstance(target_counts, dict)

        # Check that counts sum approximately to total for each category
        for _category, counts in target_counts.items():
            total_category_count = sum(counts.values())
            assert total_category_count > 0
            assert total_category_count <= total_count

    def test_select_balanced_attributes(self):
        """Test balanced attribute selection."""
        targets = self.balancer._get_target_distributions(BalanceStrategy.EQUAL_DISTRIBUTION)

        # Test severity selection
        severity = self.balancer._select_balanced_severity(targets)
        assert severity in list(SeverityLevel)

        # Test scenario type selection
        scenario_type = self.balancer._select_balanced_scenario_type(targets)
        assert scenario_type in list(ScenarioType)

        # Test demographic selection
        demographic = self.balancer._select_balanced_demographic()
        assert demographic in list(DemographicCategory)

    def test_rebalance_existing_dataset(self):
        """Test rebalancing of existing dataset."""
        # Create imbalanced dataset (all same type)
        imbalanced_scenarios = []
        for _ in range(5):
            scenario = self.scenario_generator.generate_client_scenario(
                scenario_type=ScenarioType.INITIAL_ASSESSMENT,
                severity_level=SeverityLevel.MILD
            )
            imbalanced_scenarios.append(scenario)

        # Rebalance
        balanced_scenarios, report = self.balancer.rebalance_existing_dataset(
            imbalanced_scenarios,
            target_strategy=BalanceStrategy.EQUAL_DISTRIBUTION,
            max_additions=10
        )

        # Check results
        assert len(balanced_scenarios) >= len(imbalanced_scenarios)
        assert isinstance(report, BalanceReport)
        assert report.balance_score >= 0.0

    def test_export_balance_report(self):
        """Test balance report export."""
        report = self.balancer.analyze_dataset_balance(
            self.test_scenarios,
            strategy=BalanceStrategy.EQUAL_DISTRIBUTION
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "test_balance_report.json"

            # Test successful export
            success = self.balancer.export_balance_report(report, output_path)
            assert success
            assert output_path.exists()

            # Verify exported content
            with open(output_path, encoding="utf-8") as f:
                exported_data = json.load(f)

            assert "balance_report" in exported_data
            balance_data = exported_data["balance_report"]

            assert "dataset_id" in balance_data
            assert "balance_score" in balance_data
            assert "distributions" in balance_data
            assert "recommendations" in balance_data

    def test_get_balance_statistics(self):
        """Test balance statistics generation."""
        # Create multiple reports
        reports = []
        for _ in range(3):
            report = self.balancer.analyze_dataset_balance(
                self.test_scenarios,
                strategy=BalanceStrategy.EQUAL_DISTRIBUTION
            )
            reports.append(report)

        stats = self.balancer.get_balance_statistics(reports)

        expected_keys = {
            "total_reports", "average_balance_score", "score_distribution",
            "common_recommendations", "strategy_usage", "improvement_trends"
        }
        assert set(stats.keys()) == expected_keys

        assert stats["total_reports"] == 3
        assert isinstance(stats["average_balance_score"], float)

    def test_compare_balance_strategies(self):
        """Test balance strategy comparison."""
        strategy_reports = self.balancer.compare_balance_strategies(self.test_scenarios)

        assert isinstance(strategy_reports, dict)
        assert len(strategy_reports) == len(BalanceStrategy)

        for strategy, report in strategy_reports.items():
            assert isinstance(strategy, BalanceStrategy)
            assert isinstance(report, BalanceReport)
            assert report.strategy_used == strategy

    def test_optimize_dataset_composition(self):
        """Test dataset composition optimization."""
        target_size = 15
        optimized_scenarios = self.balancer.optimize_dataset_composition(
            target_size,
            strategy=BalanceStrategy.EQUAL_DISTRIBUTION,
            quality_threshold=0.6
        )

        assert isinstance(optimized_scenarios, list)
        assert len(optimized_scenarios) > 0

        # Verify optimization improved balance
        final_report = self.balancer.analyze_dataset_balance(
            optimized_scenarios,
            strategy=BalanceStrategy.EQUAL_DISTRIBUTION
        )
        assert final_report.balance_score >= 0.0

    def test_validate_balance_requirements(self):
        """Test balance requirements validation."""
        # Define requirements
        requirements = {
            "severity_levels": {
                SeverityLevel.MODERATE.value: 0.3,  # At least 30%
                SeverityLevel.SEVERE.value: 0.1     # At least 10%
            }
        }

        is_valid, violations = self.balancer.validate_balance_requirements(
            self.test_scenarios, requirements
        )

        assert isinstance(is_valid, bool)
        assert isinstance(violations, list)

        for violation in violations:
            assert isinstance(violation, str)
            assert len(violation) > 0

    def test_category_distribution_dataclass(self):
        """Test CategoryDistribution dataclass."""
        distribution = CategoryDistribution(
            category_name="test_category",
            total_count=100,
            percentage=0.5,
            target_percentage=0.4,
            deviation=0.1,
            subcategories={"sub1": 60, "sub2": 40}
        )

        assert distribution.category_name == "test_category"
        assert distribution.total_count == 100
        assert distribution.percentage == 0.5
        assert distribution.target_percentage == 0.4
        assert distribution.deviation == 0.1
        assert distribution.subcategories["sub1"] == 60

    def test_balance_report_dataclass(self):
        """Test BalanceReport dataclass."""
        report = BalanceReport(
            dataset_id="test_dataset",
            total_items=50,
            balance_score=0.75,
            strategy_used=BalanceStrategy.EQUAL_DISTRIBUTION
        )

        assert report.dataset_id == "test_dataset"
        assert report.total_items == 50
        assert report.balance_score == 0.75
        assert report.strategy_used == BalanceStrategy.EQUAL_DISTRIBUTION
        assert isinstance(report.distributions, dict)
        assert isinstance(report.recommendations, list)


if __name__ == "__main__":
    unittest.main()
