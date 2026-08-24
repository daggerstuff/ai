#!/usr/bin/env python3
"""
Test suite for CCD Schema Definitions
Tests for Case Conceptualization Diagram (CCD) data structures.
"""

import unittest
from datetime import datetime

from ai.tools.utilities.platform.ccd_schema import (
    CCDConceptualization,
    CCDFactor,
    CCDFactorType,
    CCDFormulation,
    CCDHypothesis,
    CCDIntervention,
    CCDProblem,
)


class TestCCDFactorType(unittest.TestCase):
    """Test CCDFactorType enum"""

    def test_factor_types_exist(self):
        """Test that all expected factor types exist"""
        assert CCDFactorType.PREDISPOSING.value == "predisposing"
        assert CCDFactorType.PRECIPITATING.value == "precipitating"
        assert CCDFactorType.PERPETUATING.value == "perpetuating"
        assert CCDFactorType.PROTECTIVE.value == "protective"


class TestCCDFactor(unittest.TestCase):
    """Test CCDFactor dataclass"""

    def test_factor_creation(self):
        """Test creating a CCD factor"""
        factor = CCDFactor(
            factor_type=CCDFactorType.PREDISPOSING,
            description="Childhood trauma history",
            severity=0.8,
            evidence=["Patient reports abuse at age 8", "Medical records confirm"],
        )

        assert factor.factor_type == CCDFactorType.PREDISPOSING
        assert factor.description == "Childhood trauma history"
        assert factor.severity == 0.8
        assert len(factor.evidence) == 2
        assert "Patient reports abuse at age 8" in factor.evidence

    def test_factor_defaults(self):
        """Test CCD factor with default values"""
        factor = CCDFactor(factor_type=CCDFactorType.PRECIPITATING, description="Job loss")

        assert factor.factor_type == CCDFactorType.PRECIPITATING
        assert factor.description == "Job loss"
        assert factor.severity is None
        assert factor.evidence == []


class TestCCDProblem(unittest.TestCase):
    """Test CCDProblem dataclass"""

    def test_problem_creation(self):
        """Test creating a CCD problem"""
        problem = CCDProblem(
            problem_statement="Persistent depressive symptoms",
            severity=0.7,
            duration="8 months",
            impact_domains=["work", "relationships", "self-care"],
        )

        assert problem.problem_statement == "Persistent depressive symptoms"
        assert problem.severity == 0.7
        assert problem.duration == "8 months"
        assert len(problem.impact_domains) == 3
        assert "work" in problem.impact_domains

    def test_problem_defaults(self):
        """Test CCD problem with default values"""
        problem = CCDProblem(problem_statement="Anxiety symptoms", severity=0.5, duration="3 months")

        assert problem.problem_statement == "Anxiety symptoms"
        assert problem.severity == 0.5
        assert problem.duration == "3 months"
        assert problem.impact_domains == []
        assert problem.factors == []


class TestCCDHypothesis(unittest.TestCase):
    """Test CCDHypothesis dataclass"""

    def test_hypothesis_creation(self):
        """Test creating a CCD hypothesis"""
        hypothesis = CCDHypothesis(
            hypothesis_statement="Depression maintained by negative thought patterns and social withdrawal",
            confidence=0.8,
            supporting_evidence=["Patient reports automatic negative thoughts", "Behavioral avoidance observed"],
            contradicting_evidence=["Some positive social interactions reported"],
            testable_predictions=["If behavioral activation increases, depression should decrease"],
        )

        assert hypothesis.hypothesis_statement == "Depression maintained by negative thought patterns and social withdrawal"
        assert hypothesis.confidence == 0.8
        assert len(hypothesis.supporting_evidence) == 2
        assert len(hypothesis.contradicting_evidence) == 1
        assert len(hypothesis.testable_predictions) == 1

    def test_hypothesis_defaults(self):
        """Test CCD hypothesis with default values"""
        hypothesis = CCDHypothesis(hypothesis_statement="Client has anxiety disorder", confidence=0.6)

        assert hypothesis.hypothesis_statement == "Client has anxiety disorder"
        assert hypothesis.confidence == 0.6
        assert hypothesis.supporting_evidence == []
        assert hypothesis.contradicting_evidence == []
        assert hypothesis.testable_predictions == []


class TestCCDIntervention(unittest.TestCase):
    """Test CCDIntervention dataclass"""

    def test_intervention_creation(self):
        """Test creating a CCD intervention"""
        intervention = CCDIntervention(
            intervention_description="Weekly CBT sessions focusing on cognitive restructuring",
            modality="CBT",
            outcome="partially_effective",
            barriers=["Patient misses sessions due to transportation issues"],
            facilitators=["Strong therapeutic alliance", "Patient motivated to change"],
        )

        assert intervention.intervention_description == "Weekly CBT sessions focusing on cognitive restructuring"
        assert intervention.modality == "CBT"
        assert intervention.outcome == "partially_effective"
        assert len(intervention.barriers) == 1
        assert len(intervention.facilitators) == 2

    def test_intervention_defaults(self):
        """Test CCD intervention with default values"""
        intervention = CCDIntervention(intervention_description="Medication management", modality="Psychopharmacology")

        assert intervention.intervention_description == "Medication management"
        assert intervention.modality == "Psychopharmacology"
        assert intervention.start_date is None
        assert intervention.end_date is None
        assert intervention.outcome is None
        assert intervention.barriers == []
        assert intervention.facilitators == []


class TestCCDFormulation(unittest.TestCase):
    """Test CCDFormulation dataclass"""

    def test_formulation_creation(self):
        """Test creating a CCD formulation"""
        formulation = CCDFormulation(
            summary="Client presents with moderate depression exacerbated by work stress and social isolation",
            strengths=["Good insight", "Strong work ethic when symptomatic"],
            vulnerabilities=["Perfectionism", "Difficulty asking for help"],
            treatment_goals=["Reduce depressive symptoms by 50%", "Increase social engagement"],
            prognosis="guardedly_optimistic",
            confidence=0.75,
        )

        assert formulation.summary == "Client presents with moderate depression exacerbated by work stress and social isolation"
        assert len(formulation.strengths) == 2
        assert len(formulation.vulnerabilities) == 2
        assert len(formulation.treatment_goals) == 2
        assert formulation.prognosis == "guardedly_optimistic"
        assert formulation.confidence == 0.75

    def test_formulation_defaults(self):
        """Test CCD formulation with default values"""
        formulation = CCDFormulation(summary="Test formulation")

        assert formulation.summary == "Test formulation"
        assert formulation.strengths == []
        assert formulation.vulnerabilities == []
        assert formulation.treatment_goals == []
        assert formulation.prognosis == "guardedly_optimistic"
        assert formulation.confidence == 0.7


class TestCCDConceptualization(unittest.TestCase):
    """Test CCDConceptualization dataclass"""

    def setUp(self):
        """Set up test fixtures"""
        self.client_id = "test_client_001"
        self.conceptualization = CCDConceptualization(client_id=self.client_id)

    def test_conceptualization_creation(self):
        """Test creating a CCD conceptualization"""
        assert self.conceptualization.client_id == self.client_id
        assert isinstance(self.conceptualization.timestamp, datetime)
        assert self.conceptualization.problems == []
        assert self.conceptualization.factors == []
        assert self.conceptualization.hypotheses == []
        assert self.conceptualization.interventions == []
        assert self.conceptualization.formulation is None
        assert self.conceptualization.clinician_notes == []
        assert self.conceptualization.revision_history == []

    def test_add_factor(self):
        """Test adding a factor to conceptualization"""
        self.conceptualization.add_factor(
            CCDFactorType.PREDISPOSING,
            "Family history of depression",
            severity=0.6,
            evidence=["Mother diagnosed with depression", "Maternal aunt with anxiety disorder"],
        )

        assert len(self.conceptualization.factors) == 1
        factor = self.conceptualization.factors[0]
        assert factor.factor_type == CCDFactorType.PREDISPOSING
        assert factor.description == "Family history of depression"
        assert factor.severity == 0.6
        assert len(factor.evidence) == 2
        assert len(self.conceptualization.revision_history) == 1

    def test_add_problem(self):
        """Test adding a problem to conceptualization"""
        self.conceptualization.add_problem(
            problem_statement="Chronic anxiety and worry",
            severity=0.8,
            duration="2 years",
            impact_domains=["work", "social life", "physical health"],
        )

        assert len(self.conceptualization.problems) == 1
        problem = self.conceptualization.problems[0]
        assert problem.problem_statement == "Chronic anxiety and worry"
        assert problem.severity == 0.8
        assert problem.duration == "2 years"
        assert len(problem.impact_domains) == 3
        assert len(self.conceptualization.revision_history) == 1

    def test_add_hypothesis(self):
        """Test adding a hypothesis to conceptualization"""
        self.conceptualization.add_hypothesis(
            hypothesis_statement="Anxiety maintained by catastrophic thinking and avoidance behaviors",
            confidence=0.85,
            supporting_evidence=["Patient reports 'what if' thinking", "Avoids social situations"],
            testable_predictions=["If cognitive restructuring reduces catastrophic thinking, anxiety should decrease"],
        )

        assert len(self.conceptualization.hypotheses) == 1
        hypothesis = self.conceptualization.hypotheses[0]
        assert hypothesis.hypothesis_statement == "Anxiety maintained by catastrophic thinking and avoidance behaviors"
        assert hypothesis.confidence == 0.85
        assert len(hypothesis.supporting_evidence) == 2
        assert len(hypothesis.testable_predictions) == 1
        assert len(self.conceptualization.revision_history) == 1

    def test_add_intervention(self):
        """Test adding an intervention to conceptualization"""
        self.conceptualization.add_intervention(
            intervention_description="Weekly ACT sessions",
            modality="ACT",
            outcome="effective",
            barriers=["Initial skepticism about mindfulness"],
            facilitators=["Openness to experiential exercises"],
        )

        assert len(self.conceptualization.interventions) == 1
        intervention = self.conceptualization.interventions[0]
        assert intervention.intervention_description == "Weekly ACT sessions"
        assert intervention.modality == "ACT"
        assert intervention.outcome == "effective"
        assert len(intervention.barriers) == 1
        assert len(intervention.facilitators) == 1
        assert len(self.conceptualization.revision_history) == 1

    def test_set_formulation(self):
        """Test setting a case formulation"""
        self.conceptualization.set_formulation(
            summary="Client presents with anxiety disorder maintained by avoidance and cognitive fusion",
            strengths=["High intelligence", "Willingness to engage in treatment"],
            vulnerabilities=["Experiential avoidance", "Cognitive fusion"],
            treatment_goals=["Increase psychological flexibility", "Reduce avoidance behaviors"],
            prognosis="optimistic",
            confidence=0.8,
        )

        assert self.conceptualization.formulation is not None
        formulation = self.conceptualization.formulation
        assert formulation.summary == "Client presents with anxiety disorder maintained by avoidance and cognitive fusion"
        assert len(formulation.strengths) == 2
        assert len(formulation.vulnerabilities) == 2
        assert len(formulation.treatment_goals) == 2
        assert formulation.prognosis == "optimistic"
        assert formulation.confidence == 0.8
        assert len(self.conceptualization.revision_history) == 1

    def test_get_factors_by_type(self):
        """Test getting factors by type"""
        # Add factors of different types
        self.conceptualization.add_factor(CCDFactorType.PREDISPOSING, "Genetic vulnerability")
        self.conceptualization.add_factor(CCDFactorType.PRECIPITATING, "Recent breakup")
        self.conceptualization.add_factor(CCDFactorType.PREDISPOSING, "Childhood emotional neglect")
        self.conceptualization.add_factor(CCDFactorType.PROTECTIVE, "Strong friendship network")

        predisposing_factors = self.conceptualization.get_factors_by_type(CCDFactorType.PREDISPOSING)
        precipitating_factors = self.conceptualization.get_factors_by_type(CCDFactorType.PRECIPITATING)
        protective_factors = self.conceptualization.get_factors_by_type(CCDFactorType.PROTECTIVE)

        assert len(predisposing_factors) == 2
        assert len(precipitating_factors) == 1
        assert len(protective_factors) == 1

        assert predisposing_factors[0].description == "Genetic vulnerability"
        assert predisposing_factors[1].description == "Childhood emotional neglect"
        assert precipitating_factors[0].description == "Recent breakup"
        assert protective_factors[0].description == "Strong friendship network"

    def test_get_active_problems(self):
        """Test getting active problems (severity > 0.3)"""
        # Add problems with different severities
        self.conceptualization.add_problem("Mild anxiety", 0.2, "1 month")
        self.conceptualization.add_problem("Moderate depression", 0.6, "6 months")
        self.conceptualization.add_problem("Severe insomnia", 0.9, "3 months")
        self.conceptualization.add_problem("Very mild stress", 0.1, "2 weeks")

        active_problems = self.conceptualization.get_active_problems()

        assert len(active_problems) == 2  # Only moderate and severe problems
        severities = [p.severity for p in active_problems]
        assert 0.6 in severities
        assert 0.9 in severities

        # Check that mild and very mild problems are excluded
        problem_statements = [p.problem_statement for p in active_problems]
        assert "Mild anxiety" not in problem_statements
        assert "Very mild stress" not in problem_statements


if __name__ == "__main__":
    unittest.main()
