#!/usr/bin/env python3
"""
Test suite for CCD Schema Definitions
Tests for Case Conceptualization Diagram (CCD) data structures.
"""

import unittest
from datetime import datetime
from typing import List

from ai.platform.ccd_schema import (
    CCDFactorType,
    CCDFactor,
    CCDProblem,
    CCDHypothesis,
    CCDIntervention,
    CCDFormulation,
    CCDConceptualization,
)


class TestCCDFactorType(unittest.TestCase):
    """Test CCDFactorType enum"""

    def test_factor_types_exist(self):
        """Test that all expected factor types exist"""
        self.assertEqual(CCDFactorType.PREDISPOSING.value, "predisposing")
        self.assertEqual(CCDFactorType.PRECIPITATING.value, "precipitating")
        self.assertEqual(CCDFactorType.PERPETUATING.value, "perpetuating")
        self.assertEqual(CCDFactorType.PROTECTIVE.value, "protective")


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

        self.assertEqual(factor.factor_type, CCDFactorType.PREDISPOSING)
        self.assertEqual(factor.description, "Childhood trauma history")
        self.assertEqual(factor.severity, 0.8)
        self.assertEqual(len(factor.evidence), 2)
        self.assertIn("Patient reports abuse at age 8", factor.evidence)

    def test_factor_defaults(self):
        """Test CCD factor with default values"""
        factor = CCDFactor(factor_type=CCDFactorType.PRECIPITATING, description="Job loss")

        self.assertEqual(factor.factor_type, CCDFactorType.PRECIPITATING)
        self.assertEqual(factor.description, "Job loss")
        self.assertIsNone(factor.severity)
        self.assertEqual(factor.evidence, [])


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

        self.assertEqual(problem.problem_statement, "Persistent depressive symptoms")
        self.assertEqual(problem.severity, 0.7)
        self.assertEqual(problem.duration, "8 months")
        self.assertEqual(len(problem.impact_domains), 3)
        self.assertIn("work", problem.impact_domains)

    def test_problem_defaults(self):
        """Test CCD problem with default values"""
        problem = CCDProblem(problem_statement="Anxiety symptoms", severity=0.5, duration="3 months")

        self.assertEqual(problem.problem_statement, "Anxiety symptoms")
        self.assertEqual(problem.severity, 0.5)
        self.assertEqual(problem.duration, "3 months")
        self.assertEqual(problem.impact_domains, [])
        self.assertEqual(problem.factors, [])


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

        self.assertEqual(
            hypothesis.hypothesis_statement, "Depression maintained by negative thought patterns and social withdrawal"
        )
        self.assertEqual(hypothesis.confidence, 0.8)
        self.assertEqual(len(hypothesis.supporting_evidence), 2)
        self.assertEqual(len(hypothesis.contradicting_evidence), 1)
        self.assertEqual(len(hypothesis.testable_predictions), 1)

    def test_hypothesis_defaults(self):
        """Test CCD hypothesis with default values"""
        hypothesis = CCDHypothesis(hypothesis_statement="Client has anxiety disorder", confidence=0.6)

        self.assertEqual(hypothesis.hypothesis_statement, "Client has anxiety disorder")
        self.assertEqual(hypothesis.confidence, 0.6)
        self.assertEqual(hypothesis.supporting_evidence, [])
        self.assertEqual(hypothesis.contradicting_evidence, [])
        self.assertEqual(hypothesis.testable_predictions, [])


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

        self.assertEqual(
            intervention.intervention_description, "Weekly CBT sessions focusing on cognitive restructuring"
        )
        self.assertEqual(intervention.modality, "CBT")
        self.assertEqual(intervention.outcome, "partially_effective")
        self.assertEqual(len(intervention.barriers), 1)
        self.assertEqual(len(intervention.facilitators), 2)

    def test_intervention_defaults(self):
        """Test CCD intervention with default values"""
        intervention = CCDIntervention(intervention_description="Medication management", modality="Psychopharmacology")

        self.assertEqual(intervention.intervention_description, "Medication management")
        self.assertEqual(intervention.modality, "Psychopharmacology")
        self.assertIsNone(intervention.start_date)
        self.assertIsNone(intervention.end_date)
        self.assertIsNone(intervention.outcome)
        self.assertEqual(intervention.barriers, [])
        self.assertEqual(intervention.facilitators, [])


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

        self.assertEqual(
            formulation.summary,
            "Client presents with moderate depression exacerbated by work stress and social isolation",
        )
        self.assertEqual(len(formulation.strengths), 2)
        self.assertEqual(len(formulation.vulnerabilities), 2)
        self.assertEqual(len(formulation.treatment_goals), 2)
        self.assertEqual(formulation.prognosis, "guardedly_optimistic")
        self.assertEqual(formulation.confidence, 0.75)

    def test_formulation_defaults(self):
        """Test CCD formulation with default values"""
        formulation = CCDFormulation(summary="Test formulation")

        self.assertEqual(formulation.summary, "Test formulation")
        self.assertEqual(formulation.strengths, [])
        self.assertEqual(formulation.vulnerabilities, [])
        self.assertEqual(formulation.treatment_goals, [])
        self.assertEqual(formulation.prognosis, "guardedly_optimistic")
        self.assertEqual(formulation.confidence, 0.7)


class TestCCDConceptualization(unittest.TestCase):
    """Test CCDConceptualization dataclass"""

    def setUp(self):
        """Set up test fixtures"""
        self.client_id = "test_client_001"
        self.conceptualization = CCDConceptualization(client_id=self.client_id)

    def test_conceptualization_creation(self):
        """Test creating a CCD conceptualization"""
        self.assertEqual(self.conceptualization.client_id, self.client_id)
        self.assertIsInstance(self.conceptualization.timestamp, datetime)
        self.assertEqual(self.conceptualization.problems, [])
        self.assertEqual(self.conceptualization.factors, [])
        self.assertEqual(self.conceptualization.hypotheses, [])
        self.assertEqual(self.conceptualization.interventions, [])
        self.assertIsNone(self.conceptualization.formulation)
        self.assertEqual(self.conceptualization.clinician_notes, [])
        self.assertEqual(self.conceptualization.revision_history, [])

    def test_add_factor(self):
        """Test adding a factor to conceptualization"""
        self.conceptualization.add_factor(
            CCDFactorType.PREDISPOSING,
            "Family history of depression",
            severity=0.6,
            evidence=["Mother diagnosed with depression", "Maternal aunt with anxiety disorder"],
        )

        self.assertEqual(len(self.conceptualization.factors), 1)
        factor = self.conceptualization.factors[0]
        self.assertEqual(factor.factor_type, CCDFactorType.PREDISPOSING)
        self.assertEqual(factor.description, "Family history of depression")
        self.assertEqual(factor.severity, 0.6)
        self.assertEqual(len(factor.evidence), 2)
        self.assertEqual(len(self.conceptualization.revision_history), 1)

    def test_add_problem(self):
        """Test adding a problem to conceptualization"""
        self.conceptualization.add_problem(
            problem_statement="Chronic anxiety and worry",
            severity=0.8,
            duration="2 years",
            impact_domains=["work", "social life", "physical health"],
        )

        self.assertEqual(len(self.conceptualization.problems), 1)
        problem = self.conceptualization.problems[0]
        self.assertEqual(problem.problem_statement, "Chronic anxiety and worry")
        self.assertEqual(problem.severity, 0.8)
        self.assertEqual(problem.duration, "2 years")
        self.assertEqual(len(problem.impact_domains), 3)
        self.assertEqual(len(self.conceptualization.revision_history), 1)

    def test_add_hypothesis(self):
        """Test adding a hypothesis to conceptualization"""
        self.conceptualization.add_hypothesis(
            hypothesis_statement="Anxiety maintained by catastrophic thinking and avoidance behaviors",
            confidence=0.85,
            supporting_evidence=["Patient reports 'what if' thinking", "Avoids social situations"],
            testable_predictions=["If cognitive restructuring reduces catastrophic thinking, anxiety should decrease"],
        )

        self.assertEqual(len(self.conceptualization.hypotheses), 1)
        hypothesis = self.conceptualization.hypotheses[0]
        self.assertEqual(
            hypothesis.hypothesis_statement, "Anxiety maintained by catastrophic thinking and avoidance behaviors"
        )
        self.assertEqual(hypothesis.confidence, 0.85)
        self.assertEqual(len(hypothesis.supporting_evidence), 2)
        self.assertEqual(len(hypothesis.testable_predictions), 1)
        self.assertEqual(len(self.conceptualization.revision_history), 1)

    def test_add_intervention(self):
        """Test adding an intervention to conceptualization"""
        self.conceptualization.add_intervention(
            intervention_description="Weekly ACT sessions",
            modality="ACT",
            outcome="effective",
            barriers=["Initial skepticism about mindfulness"],
            facilitators=["Openness to experiential exercises"],
        )

        self.assertEqual(len(self.conceptualization.interventions), 1)
        intervention = self.conceptualization.interventions[0]
        self.assertEqual(intervention.intervention_description, "Weekly ACT sessions")
        self.assertEqual(intervention.modality, "ACT")
        self.assertEqual(intervention.outcome, "effective")
        self.assertEqual(len(intervention.barriers), 1)
        self.assertEqual(len(intervention.facilitators), 1)
        self.assertEqual(len(self.conceptualization.revision_history), 1)

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

        self.assertIsNotNone(self.conceptualization.formulation)
        formulation = self.conceptualization.formulation
        self.assertEqual(
            formulation.summary, "Client presents with anxiety disorder maintained by avoidance and cognitive fusion"
        )
        self.assertEqual(len(formulation.strengths), 2)
        self.assertEqual(len(formulation.vulnerabilities), 2)
        self.assertEqual(len(formulation.treatment_goals), 2)
        self.assertEqual(formulation.prognosis, "optimistic")
        self.assertEqual(formulation.confidence, 0.8)
        self.assertEqual(len(self.conceptualization.revision_history), 1)

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

        self.assertEqual(len(predisposing_factors), 2)
        self.assertEqual(len(precipitating_factors), 1)
        self.assertEqual(len(protective_factors), 1)

        self.assertEqual(predisposing_factors[0].description, "Genetic vulnerability")
        self.assertEqual(predisposing_factors[1].description, "Childhood emotional neglect")
        self.assertEqual(precipitating_factors[0].description, "Recent breakup")
        self.assertEqual(protective_factors[0].description, "Strong friendship network")

    def test_get_active_problems(self):
        """Test getting active problems (severity > 0.3)"""
        # Add problems with different severities
        self.conceptualization.add_problem("Mild anxiety", 0.2, "1 month")
        self.conceptualization.add_problem("Moderate depression", 0.6, "6 months")
        self.conceptualization.add_problem("Severe insomnia", 0.9, "3 months")
        self.conceptualization.add_problem("Very mild stress", 0.1, "2 weeks")

        active_problems = self.conceptualization.get_active_problems()

        self.assertEqual(len(active_problems), 2)  # Only moderate and severe problems
        severities = [p.severity for p in active_problems]
        self.assertIn(0.6, severities)
        self.assertIn(0.9, severities)

        # Check that mild and very mild problems are excluded
        problem_statements = [p.problem_statement for p in active_problems]
        self.assertNotIn("Mild anxiety", problem_statements)
        self.assertNotIn("Very mild stress", problem_statements)


if __name__ == "__main__":
    unittest.main()
