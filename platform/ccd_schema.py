#!/usr/bin/env python3
"""
CCD Schema Definitions for Pixelated Empathy
Case Conceptualization Diagram (CCD) data structures for therapeutic case formulation.

CCD refers to Case Conceptualization Diagram, a clinical tool used to organize
and understand a client's presenting problems, history, and treatment needs.
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import List, Optional, Dict, Any


class CCDFactorType(Enum):
    """Types of factors in case conceptualization"""

    PREDISPOSING = "predisposing"  # Background factors that increase vulnerability
    PRECIPITATING = "precipitating"  # Specific events that trigger the problem
    PERPETUATING = "perpetuating"  # Factors that maintain the problem
    PROTECTIVE = "protective"  # Factors that reduce risk or promote resilience


@dataclass
class CCDFactor:
    """Individual factor in case conceptualization"""

    factor_type: CCDFactorType
    description: str
    severity: Optional[float] = None  # 0-1 scale, if applicable
    evidence: List[str] = field(default_factory=list)  # Supporting evidence/examples


@dataclass
class CCDProblem:
    """Problem formulation in CCD"""

    problem_statement: str
    severity: float  # 0-1 scale
    duration: str  # Time period (e.g., "6 months", "2 years")
    impact_domains: List[str] = field(default_factory=list)  # Life areas affected
    factors: List[CCDFactor] = field(default_factory=list)  # Contributing factors


@dataclass
class CCDHypothesis:
    """Hypothesis about client's condition or maintaining mechanisms"""

    hypothesis_statement: str
    confidence: float  # 0-1 scale
    supporting_evidence: List[str] = field(default_factory=list)
    contradicting_evidence: List[str] = field(default_factory=list)
    testable_predictions: List[str] = field(default_factory=list)


@dataclass
class CCDIntervention:
    """Planned or attempted intervention"""

    intervention_description: str
    modality: str  # e.g., "CBT", "DBT", "Psychodynamic"
    start_date: Optional[datetime] = None
    end_date: Optional[datetime] = None
    outcome: Optional[str] = None  # e.g., "effective", "partially_effective", "ineffective"
    barriers: List[str] = field(default_factory=list)
    facilitators: List[str] = field(default_factory=list)


@dataclass
class CCDFormulation:
    """Complete case formulation/summary"""

    summary: str
    strengths: List[str] = field(default_factory=list)
    vulnerabilities: List[str] = field(default_factory=list)
    treatment_goals: List[str] = field(default_factory=list)
    prognosis: str = "guardedly_optimistic"  # optimistic, guarded_optimistic, guarded, poor
    confidence: float = 0.7  # Confidence in formulation (0-1 scale)


@dataclass
class CCDConceptualization:
    """Complete Case Conceptualization Diagram"""

    client_id: str
    timestamp: datetime = field(default_factory=datetime.now)

    # Core CCD components
    problems: List[CCDProblem] = field(default_factory=list)
    factors: List[CCDFactor] = field(default_factory=list)
    hypotheses: List[CCDHypothesis] = field(default_factory=list)
    interventions: List[CCDIntervention] = field(default_factory=list)
    formulation: Optional[CCDFormulation] = None

    # Metadata
    clinician_notes: List[str] = field(default_factory=list)
    revision_history: List[Dict[str, Any]] = field(default_factory=list)

    def add_factor(
        self,
        factor_type: CCDFactorType,
        description: str,
        severity: Optional[float] = None,
        evidence: Optional[List[str]] = None,
    ):
        """Add a factor to the conceptualization"""
        factor = CCDFactor(factor_type=factor_type, description=description, severity=severity, evidence=evidence or [])
        self.factors.append(factor)
        self._log_revision(f"Added {factor_type.value} factor: {description}")

    def add_problem(
        self,
        problem_statement: str,
        severity: float,
        duration: str,
        impact_domains: Optional[List[str]] = None,
        factors: Optional[List[CCDFactor]] = None,
    ):
        """Add a problem to the conceptualization"""
        problem = CCDProblem(
            problem_statement=problem_statement,
            severity=severity,
            duration=duration,
            impact_domains=impact_domains or [],
            factors=factors or [],
        )
        self.problems.append(problem)
        self._log_revision(f"Added problem: {problem_statement}")

    def add_hypothesis(
        self,
        hypothesis_statement: str,
        confidence: float,
        supporting_evidence: Optional[List[str]] = None,
        contradicting_evidence: Optional[List[str]] = None,
        testable_predictions: Optional[List[str]] = None,
    ):
        """Add a hypothesis to the conceptualization"""
        hypothesis = CCDHypothesis(
            hypothesis_statement=hypothesis_statement,
            confidence=confidence,
            supporting_evidence=supporting_evidence or [],
            contradicting_evidence=contradicting_evidence or [],
            testable_predictions=testable_predictions or [],
        )
        self.hypotheses.append(hypothesis)
        self._log_revision(f"Added hypothesis: {hypothesis_statement}")

    def add_intervention(
        self,
        intervention_description: str,
        modality: str,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        outcome: Optional[str] = None,
        barriers: Optional[List[str]] = None,
        facilitators: Optional[List[str]] = None,
    ):
        """Add an intervention to the conceptualization"""
        intervention = CCDIntervention(
            intervention_description=intervention_description,
            modality=modality,
            start_date=start_date,
            end_date=end_date,
            outcome=outcome,
            barriers=barriers or [],
            facilitators=facilitators or [],
        )
        self.interventions.append(intervention)
        self._log_revision(f"Added intervention: {intervention_description}")

    def set_formulation(
        self,
        summary: str,
        strengths: Optional[List[str]] = None,
        vulnerabilities: Optional[List[str]] = None,
        treatment_goals: Optional[List[str]] = None,
        prognosis: str = "guardedly_optimistic",
        confidence: float = 0.7,
    ):
        """Set or update the case formulation"""
        self.formulation = CCDFormulation(
            summary=summary,
            strengths=strengths or [],
            vulnerabilities=vulnerabilities or [],
            treatment_goals=treatment_goals or [],
            prognosis=prognosis,
            confidence=confidence,
        )
        self._log_revision("Updated case formulation")

    def _log_revision(self, description: str):
        """Log a revision to the conceptualization"""
        self.revision_history.append({"timestamp": datetime.now().isoformat(), "description": description})

    def get_factors_by_type(self, factor_type: CCDFactorType) -> List[CCDFactor]:
        """Get all factors of a specific type"""
        return [f for f in self.factors if f.factor_type == factor_type]

    def get_active_problems(self) -> List[CCDProblem]:
        """Get problems with severity above threshold"""
        return [p for p in self.problems if p.severity > 0.3]
