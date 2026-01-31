"""
Assessment Tools and Diagnostic Conversation Templates System

This module provides comprehensive assessment tools and diagnostic conversation templates
for psychology knowledge-based conversation generation. It includes clinical interview
frameworks, structured assessment protocols, and diagnostic conversation generation.

Key Features:
- Clinical interview frameworks (MSE, diagnostic interviews)
- Standardized assessment tools integration
- Diagnostic conversation templates
- Assessment protocol workflows
- Clinical formulation support
- Risk assessment frameworks
- Outcome measurement tools
"""

import json
import random
from dataclasses import asdict, dataclass
from datetime import datetime
from enum import Enum
from typing import Any  # Only import Any for legacy type compatibility

from .client_scenario_generator import ClientScenario, SeverityLevel


class AssessmentType(Enum):
    """Types of clinical assessments."""

    INITIAL_INTAKE = "initial_intake"
    MENTAL_STATUS_EXAM = "mental_status_exam"
    DIAGNOSTIC_INTERVIEW = "diagnostic_interview"
    RISK_ASSESSMENT = "risk_assessment"
    COGNITIVE_ASSESSMENT = "cognitive_assessment"
    PERSONALITY_ASSESSMENT = "personality_assessment"
    TRAUMA_ASSESSMENT = "trauma_assessment"
    SUBSTANCE_USE_ASSESSMENT = "substance_use_assessment"
    FUNCTIONAL_ASSESSMENT = "functional_assessment"
    OUTCOME_MEASUREMENT = "outcome_measurement"


class AssessmentDomain(Enum):
    """Domains covered in clinical assessments."""

    PRESENTING_PROBLEM = "presenting_problem"
    PSYCHIATRIC_HISTORY = "psychiatric_history"
    MEDICAL_HISTORY = "medical_history"
    FAMILY_HISTORY = "family_history"
    SOCIAL_HISTORY = "social_history"
    SUBSTANCE_USE = "substance_use"
    TRAUMA_HISTORY = "trauma_history"
    MENTAL_STATUS = "mental_status"
    COGNITIVE_FUNCTIONING = "cognitive_functioning"
    RISK_FACTORS = "risk_factors"
    PROTECTIVE_FACTORS = "protective_factors"
    FUNCTIONAL_IMPAIRMENT = "functional_impairment"


class InterviewTechnique(Enum):
    """Clinical interview techniques."""

    OPEN_ENDED_EXPLORATION = "open_ended_exploration"
    STRUCTURED_QUESTIONING = "structured_questioning"
    SYMPTOM_INQUIRY = "symptom_inquiry"
    TIMELINE_ESTABLISHMENT = "timeline_establishment"
    SEVERITY_RATING = "severity_rating"
    FUNCTIONAL_IMPACT = "functional_impact"
    COLLATERAL_INFORMATION = "collateral_information"
    BEHAVIORAL_OBSERVATION = "behavioral_observation"


class AssessmentOutcome(Enum):
    """Possible assessment outcomes."""

    DIAGNOSTIC_CRITERIA_MET = "diagnostic_criteria_met"
    DIAGNOSTIC_CRITERIA_PARTIAL = "diagnostic_criteria_partial"
    DIAGNOSTIC_CRITERIA_NOT_MET = "diagnostic_criteria_not_met"
    FURTHER_ASSESSMENT_NEEDED = "further_assessment_needed"
    REFERRAL_RECOMMENDED = "referral_recommended"
    IMMEDIATE_INTERVENTION = "immediate_intervention"


@dataclass
class AssessmentTool:
    """Comprehensive assessment tool definition."""

    id: str
    name: str
    assessment_type: AssessmentType
    domains_covered: list[AssessmentDomain]
    target_population: list[str]
    administration_time: tuple[int, int]  # (min, max) minutes
    required_training: str
    validity_reliability: dict[str, float]
    scoring_method: str
    interpretation_guidelines: list[str]
    clinical_cutoffs: dict[str, Any]
    contraindications: list[str]
    cultural_considerations: list[str]


@dataclass
class AssessmentQuestion:
    """Individual assessment question with context."""

    id: str
    domain: AssessmentDomain
    technique: InterviewTechnique
    question_text: str
    follow_up_questions: list[str]
    clinical_rationale: str
    expected_responses: list[str]
    red_flag_indicators: list[str]
    scoring_criteria: dict[str, Any] | None


@dataclass
class AssessmentProtocol:
    """Structured assessment protocol workflow."""

    id: str
    name: str
    assessment_type: AssessmentType
    target_conditions: list[str]
    protocol_steps: list[str]
    assessment_questions: list[AssessmentQuestion]
    decision_points: list[dict[str, Any]]
    outcome_criteria: dict[AssessmentOutcome, list[str]]
    documentation_requirements: list[str]
    follow_up_recommendations: list[str]


@dataclass
class DiagnosticConversation:
    """Diagnostic conversation with assessment framework."""

    id: str
    assessment_protocol: AssessmentProtocol
    client_scenario: ClientScenario
    conversation_exchanges: list[dict[str, Any]]
    domains_assessed: list[AssessmentDomain]
    techniques_used: list[InterviewTechnique]
    clinical_observations: list[str]
    assessment_findings: dict[str, Any]
    diagnostic_impressions: list[str]
    recommendations: list[str]
    risk_level: str
    follow_up_plan: str


class AssessmentToolsGenerator:
    """
    Comprehensive assessment tools and diagnostic conversation templates system.

    This class provides generation of clinical assessment tools, diagnostic interviews,
    and structured assessment protocols for therapeutic conversation training.
    """

    def __init__(self):
        """Initialize the assessment tools generator."""
        self.assessment_tools = self._initialize_assessment_tools()
        self.assessment_protocols = self._initialize_assessment_protocols()
        self.question_banks = self._initialize_question_banks()

    def _initialize_assessment_tools(self) -> dict[AssessmentType, list[AssessmentTool]]:
        """Initialize comprehensive assessment tools."""
        tools = {
            AssessmentType.MENTAL_STATUS_EXAM: [
                AssessmentTool(
                    id="mse_comprehensive",
                    name="Comprehensive Mental Status Examination",
                    assessment_type=AssessmentType.MENTAL_STATUS_EXAM,
                    domains_covered=[
                        AssessmentDomain.MENTAL_STATUS,
                        AssessmentDomain.COGNITIVE_FUNCTIONING,
                        AssessmentDomain.RISK_FACTORS,
                    ],
                    target_population=["adults", "adolescents"],
                    administration_time=(20, 45),
                    required_training="Clinical interview training",
                    validity_reliability={
                        "inter_rater": 0.85,
                        "test_retest": 0.78,
                    },
                    scoring_method="Clinical observation and structured inquiry",
                    interpretation_guidelines=[
                        "Assess appearance, behavior, speech, mood, affect",
                        "Evaluate thought process, content, perception",
                        "Test cognitive functions: orientation, memory, attention",
                        "Assess insight and judgment",
                    ],
                    clinical_cutoffs={"cognitive_impairment": "2+ domains affected"},
                    contraindications=[
                        "Severe agitation",
                        "Active psychosis without stabilization",
                    ],
                    cultural_considerations=[
                        "Language barriers may affect cognitive testing",
                        "Cultural expressions of distress vary",
                        "Educational background affects cognitive performance",
                    ],
                )
            ]
        }

        # Diagnostic Interview Tools
        tools[AssessmentType.DIAGNOSTIC_INTERVIEW] = [
            AssessmentTool(
                id="structured_diagnostic_interview",
                name="Structured Clinical Interview for DSM-5",
                assessment_type=AssessmentType.DIAGNOSTIC_INTERVIEW,
                domains_covered=[
                    AssessmentDomain.PRESENTING_PROBLEM,
                    AssessmentDomain.PSYCHIATRIC_HISTORY,
                    AssessmentDomain.FUNCTIONAL_IMPAIRMENT,
                ],
                target_population=["adults"],
                administration_time=(60, 120),
                required_training="SCID-5 certification",
                validity_reliability={"diagnostic_accuracy": 0.92, "inter_rater": 0.88},
                scoring_method="Structured diagnostic criteria assessment",
                interpretation_guidelines=[
                    "Follow DSM-5 diagnostic criteria systematically",
                    "Assess onset, duration, severity, impairment",
                    "Rule out medical and substance-induced causes",
                    "Consider differential diagnoses",
                ],
                clinical_cutoffs={"diagnostic_threshold": "Full criteria met"},
                contraindications=["Severe cognitive impairment"],
                cultural_considerations=[
                    "Cultural formulation required",
                    "Consider culture-bound syndromes",
                    "Assess acculturation stress",
                ],
            )
        ]

        # Risk Assessment Tools
        tools[AssessmentType.RISK_ASSESSMENT] = [
            AssessmentTool(
                id="suicide_risk_assessment",
                name="Comprehensive Suicide Risk Assessment",
                assessment_type=AssessmentType.RISK_ASSESSMENT,
                domains_covered=[
                    AssessmentDomain.RISK_FACTORS,
                    AssessmentDomain.PROTECTIVE_FACTORS,
                    AssessmentDomain.MENTAL_STATUS,
                ],
                target_population=["all ages"],
                administration_time=(15, 30),
                required_training="Suicide risk assessment training",
                validity_reliability={"predictive_validity": 0.75, "sensitivity": 0.82},
                scoring_method="Risk factor weighting and clinical judgment",
                interpretation_guidelines=[
                    "Assess ideation, plan, means, intent",
                    "Evaluate risk and protective factors",
                    "Consider previous attempts and family history",
                    "Assess current stressors and support",
                ],
                clinical_cutoffs={
                    "low_risk": "No ideation, good support",
                    "moderate_risk": "Ideation without plan",
                    "high_risk": "Plan with means and intent",
                },
                contraindications=[],
                cultural_considerations=[
                    "Cultural attitudes toward suicide vary",
                    "Religious and spiritual factors important",
                    "Family honor and shame considerations",
                ],
            )
        ]

        return tools

    def _initialize_assessment_protocols(self) -> list[AssessmentProtocol]:
        """Initialize structured assessment protocols."""
        return [
            AssessmentProtocol(
                id="depression_assessment_protocol",
                name="Major Depression Assessment Protocol",
                assessment_type=AssessmentType.DIAGNOSTIC_INTERVIEW,
                target_conditions=["Major Depressive Disorder", "Persistent Depressive Disorder"],
                protocol_steps=[
                    "Establish rapport and explain assessment",
                    "Assess current depressive symptoms",
                    "Evaluate symptom duration and severity",
                    "Assess functional impairment",
                    "Rule out medical causes",
                    "Assess suicide risk",
                    "Formulate diagnostic impression",
                ],
                assessment_questions=[],  # Will be populated separately
                decision_points=[
                    {
                        "point": "symptom_threshold",
                        "criteria": "5+ symptoms for 2+ weeks",
                        "action": "continue_assessment",
                    },
                    {
                        "point": "functional_impairment",
                        "criteria": "significant_impairment_present",
                        "action": "meet_diagnostic_criteria",
                    },
                ],
                outcome_criteria={
                    AssessmentOutcome.DIAGNOSTIC_CRITERIA_MET: [
                        "5+ symptoms present for 2+ weeks",
                        "Significant functional impairment",
                        "Not due to substance or medical condition",
                    ],
                    AssessmentOutcome.FURTHER_ASSESSMENT_NEEDED: [
                        "Unclear symptom timeline",
                        "Possible medical causes",
                        "Substance use complications",
                    ],
                },
                documentation_requirements=[
                    "Symptom inventory with onset dates",
                    "Functional assessment",
                    "Risk assessment",
                    "Diagnostic formulation",
                ],
                follow_up_recommendations=[
                    "Treatment planning session",
                    "Medical evaluation if indicated",
                    "Safety planning if risk present",
                ],
            )
        ]

    def _initialize_question_banks(self) -> dict[AssessmentDomain, list[AssessmentQuestion]]:
        """Initialize comprehensive question banks for assessments."""
        questions = {}

        # Presenting Problem Questions
        questions[AssessmentDomain.PRESENTING_PROBLEM] = [
            AssessmentQuestion(
                id="presenting_problem_main",
                domain=AssessmentDomain.PRESENTING_PROBLEM,
                technique=InterviewTechnique.OPEN_ENDED_EXPLORATION,
                question_text="What brings you in to see me today?",
                follow_up_questions=[
                    "Can you tell me more about that?",
                    "When did you first notice this?",
                    "How has this been affecting your daily life?",
                ],
                clinical_rationale="Establishes primary concerns and client's perspective",
                expected_responses=["symptom description", "life stressors", "relationship issues"],
                red_flag_indicators=["suicidal ideation", "psychosis", "severe impairment"],
                scoring_criteria=None,
            ),
            AssessmentQuestion(
                id="symptom_onset",
                domain=AssessmentDomain.PRESENTING_PROBLEM,
                technique=InterviewTechnique.TIMELINE_ESTABLISHMENT,
                question_text="When did these symptoms first begin?",
                follow_up_questions=[
                    "Was there anything happening in your life around that time?",
                    "Did the symptoms come on gradually or suddenly?",
                    "Have they gotten better or worse since then?",
                ],
                clinical_rationale="Establishes timeline for diagnostic criteria",
                expected_responses=["specific dates", "triggering events", "symptom progression"],
                red_flag_indicators=["sudden onset", "rapid deterioration"],
                scoring_criteria={"duration": "weeks/months/years"},
            ),
        ]

        # Mental Status Questions
        questions[AssessmentDomain.MENTAL_STATUS] = [
            AssessmentQuestion(
                id="mood_assessment",
                domain=AssessmentDomain.MENTAL_STATUS,
                technique=InterviewTechnique.STRUCTURED_QUESTIONING,
                question_text="How would you describe your mood today?",
                follow_up_questions=[
                    "Has your mood been like this for a while?",
                    "What does your mood feel like most days?",
                    "Do you notice your mood changing throughout the day?",
                ],
                clinical_rationale="Assesses current mood state and patterns",
                expected_responses=["mood descriptors", "mood stability", "diurnal variation"],
                red_flag_indicators=["severe depression", "mania", "mood lability"],
                scoring_criteria={"mood_rating": "1-10 scale"},
            ),
            AssessmentQuestion(
                id="cognitive_orientation",
                domain=AssessmentDomain.MENTAL_STATUS,
                technique=InterviewTechnique.STRUCTURED_QUESTIONING,
                question_text="Can you tell me what day it is today?",
                follow_up_questions=[
                    "What month and year is it?",
                    "Where are we right now?",
                    "Do you know who I am?",
                ],
                clinical_rationale="Assesses orientation to time, place, and person",
                expected_responses=["correct orientation", "partial disorientation"],
                red_flag_indicators=["disorientation", "confusion", "memory impairment"],
                scoring_criteria={"orientation_score": "0-4 points"},
            ),
        ]

        # Risk Assessment Questions
        questions[AssessmentDomain.RISK_FACTORS] = [
            AssessmentQuestion(
                id="suicidal_ideation",
                domain=AssessmentDomain.RISK_FACTORS,
                technique=InterviewTechnique.STRUCTURED_QUESTIONING,
                question_text="Have you had any thoughts of hurting yourself or ending your life?",
                follow_up_questions=[
                    "How often do you have these thoughts?",
                    "Do you have a plan for how you would do it?",
                    "Have you ever acted on thoughts like these before?",
                ],
                clinical_rationale="Assesses suicide risk and safety planning needs",
                expected_responses=[
                    "denial",
                    "passive ideation",
                    "active ideation with/without plan",
                ],
                red_flag_indicators=["active plan", "means available", "previous attempts"],
                scoring_criteria={"risk_level": "low/moderate/high"},
            ),
            AssessmentQuestion(
                id="substance_use_screening",
                domain=AssessmentDomain.SUBSTANCE_USE,
                technique=InterviewTechnique.STRUCTURED_QUESTIONING,
                question_text="Do you use alcohol or any other substances?",
                follow_up_questions=[
                    "How often do you drink/use?",
                    "How much do you typically use?",
                    "Has your use caused any problems in your life?",
                ],
                clinical_rationale="Screens for substance use disorders and complications",
                expected_responses=["denial", "social use", "problematic use"],
                red_flag_indicators=["daily use", "tolerance", "withdrawal", "life problems"],
                scoring_criteria={"audit_score": "0-40 points"},
            ),
        ]

        return questions

    def generate_diagnostic_conversation(
        self,
        client_scenario: ClientScenario,
        assessment_type: AssessmentType,
        num_exchanges: int = 10,
    ) -> DiagnosticConversation:
        """
        Generate diagnostic conversation based on assessment protocol.

        Args:
            client_scenario: Client scenario for assessment
            assessment_type: Type of assessment to conduct
            num_exchanges: Number of conversation exchanges

        Returns:
            DiagnosticConversation with structured assessment dialogue
        """
        # Select appropriate protocol
        protocol = self._select_assessment_protocol(client_scenario, assessment_type)

        # Generate conversation exchanges
        exchanges = self._generate_assessment_exchanges(client_scenario, protocol, num_exchanges)

        # Assess domains covered
        domains_assessed = self._identify_domains_assessed(exchanges)

        # Identify techniques used
        techniques_used = self._identify_techniques_used(exchanges)

        # Generate clinical observations
        clinical_observations = self._generate_clinical_observations(exchanges, client_scenario)

        # Generate assessment findings
        assessment_findings = self._generate_assessment_findings(
            exchanges, protocol, client_scenario
        )

        # Generate diagnostic impressions
        diagnostic_impressions = self._generate_diagnostic_impressions(
            assessment_findings, client_scenario
        )

        # Generate recommendations
        recommendations = self._generate_recommendations(assessment_findings, client_scenario)

        # Assess risk level
        risk_level = self._assess_risk_level(assessment_findings, client_scenario)

        # Generate follow-up plan
        follow_up_plan = self._generate_follow_up_plan(assessment_findings, recommendations)

        conversation_id = (
            f"assessment_{assessment_type.value}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        )

        return DiagnosticConversation(
            id=conversation_id,
            assessment_protocol=protocol,
            client_scenario=client_scenario,
            conversation_exchanges=exchanges,
            domains_assessed=domains_assessed,
            techniques_used=techniques_used,
            clinical_observations=clinical_observations,
            assessment_findings=assessment_findings,
            diagnostic_impressions=diagnostic_impressions,
            recommendations=recommendations,
            risk_level=risk_level,
            follow_up_plan=follow_up_plan,
        )

    def _select_assessment_protocol(
        self, client_scenario: ClientScenario, assessment_type: AssessmentType
    ) -> AssessmentProtocol:
        """Select appropriate assessment protocol based on scenario and type."""

        # For now, return the depression protocol as example
        # In full implementation, would select based on presenting concerns
        available_protocols = [
            p for p in self.assessment_protocols if p.assessment_type == assessment_type
        ]

        if available_protocols:
            # Select based on client scenario DSM-5 considerations
            dsm5_considerations = client_scenario.clinical_formulation.dsm5_considerations

            for protocol in available_protocols:
                if any(
                    condition.lower() in " ".join(dsm5_considerations).lower()
                    for condition in protocol.target_conditions
                ):
                    return protocol

            # Return first available if no specific match
            return available_protocols[0]

        # Create fallback protocol
        return self._create_fallback_protocol(assessment_type)

    def _create_fallback_protocol(self, assessment_type: AssessmentType) -> AssessmentProtocol:
        """Create fallback assessment protocol."""
        return AssessmentProtocol(
            id=f"fallback_{assessment_type.value}",
            name=f"General {assessment_type.value.replace('_', ' ').title()} Protocol",
            assessment_type=assessment_type,
            target_conditions=["general mental health concerns"],
            protocol_steps=[
                "Establish rapport",
                "Assess presenting concerns",
                "Conduct systematic inquiry",
                "Formulate impressions",
                "Provide recommendations",
            ],
            assessment_questions=[],
            decision_points=[],
            outcome_criteria={
                AssessmentOutcome.FURTHER_ASSESSMENT_NEEDED: ["Incomplete information"]
            },
            documentation_requirements=["Assessment summary"],
            follow_up_recommendations=["Continue assessment"],
        )

    def _generate_assessment_exchanges(
        self, client_scenario: ClientScenario, protocol: AssessmentProtocol, num_exchanges: int
    ) -> list[dict[str, Any]]:
        """Generate assessment conversation exchanges."""
        exchanges = []

        # Get relevant questions from question banks
        relevant_questions = self._get_relevant_questions(protocol)

        for i in range(num_exchanges):
            if i % 2 == 0:  # Therapist turn
                exchange = self._generate_therapist_assessment_question(
                    client_scenario, protocol, relevant_questions, i // 2
                )
            else:  # Client turn
                exchange = self._generate_client_assessment_response(
                    client_scenario, protocol, i // 2
                )

            exchanges.append(exchange)

        return exchanges

    def _get_relevant_questions(self, protocol: AssessmentProtocol) -> list[AssessmentQuestion]:
        """Get relevant questions for the assessment protocol."""
        relevant_questions = []

        # Get questions from all domains that might be relevant
        for domain_questions in self.question_banks.values():
            relevant_questions.extend(domain_questions)

        return relevant_questions

    def _generate_therapist_assessment_question(
        self,
        client_scenario: ClientScenario,
        protocol: AssessmentProtocol,
        questions: list[AssessmentQuestion],
        exchange_index: int,
    ) -> dict[str, Any]:
        """Generate therapist assessment question."""

        # Select appropriate question based on protocol step
        if exchange_index < len(questions):
            question = questions[exchange_index]
            content = question.question_text
            domain = question.domain
            technique = question.technique
        else:
            # Generate follow-up or clarifying question
            content = "Can you tell me more about how this has been affecting you?"
            domain = AssessmentDomain.FUNCTIONAL_IMPAIRMENT
            technique = InterviewTechnique.OPEN_ENDED_EXPLORATION

        return {
            "speaker": "therapist",
            "content": content,
            "assessment_domain": domain.value,
            "interview_technique": technique.value,
            "clinical_purpose": self._determine_clinical_purpose(domain),
            "timestamp": datetime.now().isoformat(),
        }

    def _generate_client_assessment_response(
        self, client_scenario: ClientScenario, protocol: AssessmentProtocol, exchange_index: int
    ) -> dict[str, Any]:
        """Generate client response to assessment question."""

        # Generate response based on client scenario
        content = self._generate_client_response_content(client_scenario, exchange_index)

        return {
            "speaker": "client",
            "content": content,
            "emotional_state": self._determine_client_emotional_state(
                client_scenario, exchange_index
            ),
            "cooperation_level": self._assess_cooperation_level(client_scenario),
            "information_quality": self._assess_information_quality(
                client_scenario, exchange_index
            ),
            "timestamp": datetime.now().isoformat(),
        }

    def _determine_clinical_purpose(self, domain: AssessmentDomain) -> str:
        """Determine clinical purpose of assessment domain."""
        purposes = {
            AssessmentDomain.PRESENTING_PROBLEM: "Establish chief complaint and current concerns",
            AssessmentDomain.MENTAL_STATUS: "Assess current mental state and cognitive functioning",
            AssessmentDomain.RISK_FACTORS: "Evaluate safety and risk factors",
            AssessmentDomain.PSYCHIATRIC_HISTORY: "Understand psychiatric background and patterns",
            AssessmentDomain.FUNCTIONAL_IMPAIRMENT: "Assess impact on daily functioning",
        }
        return purposes.get(domain, "Gather relevant clinical information")

    def _generate_client_response_content(
        self, client_scenario: ClientScenario, exchange_index: int
    ) -> str:
        """Generate realistic client response content."""

        # Base responses on DSM-5 considerations and severity
        dsm5_considerations = client_scenario.clinical_formulation.dsm5_considerations

        if any("depression" in consideration.lower() for consideration in dsm5_considerations):
            if exchange_index == 0:
                return (
                    "I've been feeling really down lately. Nothing seems to bring me joy anymore."
                )
            if exchange_index == 1:
                return "It started about three months ago, after I lost my job. It's been getting worse."
            return (
                "I can barely get out of bed some days. Work and relationships are suffering."
            )

        if any("anxiety" in consideration.lower() for consideration in dsm5_considerations):
            if exchange_index == 0:
                return "I've been having a lot of anxiety and worry. My heart races and I can't calm down."
            if exchange_index == 1:
                return "The anxiety started about six months ago. It happens almost every day now."
            return "I avoid social situations because I'm afraid of having a panic attack."

        # Generic responses
        responses = [
            "I'm not sure how to explain it, but something doesn't feel right.",
            "It's been going on for a while now, and I'm getting worried.",
            "I just want to feel like myself again.",
        ]
        return responses[exchange_index % len(responses)]

    def _determine_client_emotional_state(
        self, client_scenario: ClientScenario, exchange_index: int
    ) -> str:
        """Determine client's emotional state during assessment."""

        severity = client_scenario.severity_level
        dsm5_considerations = client_scenario.clinical_formulation.dsm5_considerations

        if severity == SeverityLevel.SEVERE:
            return random.choice(["distressed", "overwhelmed", "tearful", "agitated"])
        if any("depression" in consideration.lower() for consideration in dsm5_considerations):
            return random.choice(["sad", "flat", "hopeless", "withdrawn"])
        if any("anxiety" in consideration.lower() for consideration in dsm5_considerations):
            return random.choice(["anxious", "nervous", "restless", "worried"])
        return random.choice(["cooperative", "engaged", "thoughtful", "concerned"])

    def _assess_cooperation_level(self, client_scenario: ClientScenario) -> str:
        """Assess client's cooperation level during assessment."""

        attachment_style = client_scenario.clinical_formulation.attachment_style

        if attachment_style and "secure" in attachment_style.lower():
            return "high"
        if attachment_style and "avoidant" in attachment_style.lower():
            return "guarded"
        if attachment_style and "anxious" in attachment_style.lower():
            return "variable"
        return "moderate"

    def _assess_information_quality(
        self, client_scenario: ClientScenario, exchange_index: int
    ) -> str:
        """Assess quality of information provided by client."""

        if exchange_index == 0:
            return "basic"
        if exchange_index < 3:
            return "developing"
        return "detailed"

    def _identify_domains_assessed(self, exchanges: list[dict[str, Any]]) -> list[AssessmentDomain]:
        """Identify assessment domains covered in conversation."""
        domains = set()

        for exchange in exchanges:
            if exchange.get("speaker") == "therapist" and "assessment_domain" in exchange:
                domain_value = exchange["assessment_domain"]
                for domain in AssessmentDomain:
                    if domain.value == domain_value:
                        domains.add(domain)

        return list(domains)

    def _identify_techniques_used(
        self, exchanges: list[dict[str, Any]]
    ) -> list[InterviewTechnique]:
        """Identify interview techniques used in conversation."""
        techniques = set()

        for exchange in exchanges:
            if exchange.get("speaker") == "therapist" and "interview_technique" in exchange:
                technique_value = exchange["interview_technique"]
                for technique in InterviewTechnique:
                    if technique.value == technique_value:
                        techniques.add(technique)

        return list(techniques)

    def _generate_clinical_observations(
        self, exchanges: list[dict[str, Any]], client_scenario: ClientScenario
    ) -> list[str]:
        """Generate clinical observations from assessment conversation."""
        observations = []

        # Analyze client responses for clinical observations
        client_exchanges = [e for e in exchanges if e.get("speaker") == "client"]

        if client_exchanges:
            # Emotional state observations
            emotional_states = [
                e.get("emotional_state") for e in client_exchanges if e.get("emotional_state")
            ]
            if emotional_states:
                predominant_emotion = max(set(emotional_states), key=emotional_states.count)
                observations.append(
                    f"Client presented with predominantly {predominant_emotion} affect"
                )

            # Cooperation observations
            cooperation_levels = [
                e.get("cooperation_level") for e in client_exchanges if e.get("cooperation_level")
            ]
            if cooperation_levels:
                cooperation = cooperation_levels[0]  # Usually consistent
                observations.append(f"Client demonstrated {cooperation} level of cooperation")

            # Information quality observations
            info_qualities = [
                e.get("information_quality")
                for e in client_exchanges
                if e.get("information_quality")
            ]
            if "detailed" in info_qualities:
                observations.append("Client provided detailed and coherent information")
            elif "basic" in info_qualities:
                observations.append("Client provided limited information initially")

        # Add scenario-specific observations
        if client_scenario.severity_level == SeverityLevel.SEVERE:
            observations.append("Significant distress and impairment evident")

        return observations

    def _generate_assessment_findings(
        self,
        exchanges: list[dict[str, Any]],
        protocol: AssessmentProtocol,
        client_scenario: ClientScenario,
    ) -> dict[str, Any]:
        """Generate assessment findings from conversation."""
        findings = {}

        # Symptom findings based on DSM-5 considerations
        dsm5_considerations = client_scenario.clinical_formulation.dsm5_considerations

        findings["presenting_symptoms"] = dsm5_considerations
        findings["symptom_duration"] = (
            "3+ months" if client_scenario.severity_level != SeverityLevel.MILD else "2-6 weeks"
        )
        findings["functional_impairment"] = self._assess_functional_impairment(client_scenario)
        findings["risk_factors"] = self._identify_risk_factors(client_scenario)
        findings["protective_factors"] = self._identify_protective_factors(client_scenario)

        return findings

    def _assess_functional_impairment(self, client_scenario: ClientScenario) -> str:
        """Assess level of functional impairment."""
        if client_scenario.severity_level == SeverityLevel.SEVERE:
            return "Severe impairment in work, social, and personal functioning"
        if client_scenario.severity_level == SeverityLevel.MODERATE:
            return "Moderate impairment with some areas of functioning preserved"
        return "Mild impairment with most functioning intact"

    def _identify_risk_factors(self, client_scenario: ClientScenario) -> list[str]:
        """Identify risk factors from client scenario."""
        risk_factors = []

        if client_scenario.severity_level == SeverityLevel.SEVERE:
            risk_factors.append("High symptom severity")

        dsm5_considerations = client_scenario.clinical_formulation.dsm5_considerations

        # Check for suicide risk indicators
        if any(
            "suicidal" in consideration.lower() or "suicide" in consideration.lower()
            for consideration in dsm5_considerations
        ):
            risk_factors.append("Potential suicide risk")

        if any(
            "depression" in consideration.lower() or "depressive" in consideration.lower()
            for consideration in dsm5_considerations
        ):
            risk_factors.append("Depressive symptoms")
            # Add suicide risk for severe depression
            if client_scenario.severity_level == SeverityLevel.SEVERE:
                risk_factors.append("Potential suicide risk")

        if any("substance" in consideration.lower() for consideration in dsm5_considerations):
            risk_factors.append("Substance use concerns")

        return risk_factors

    def _identify_protective_factors(self, client_scenario: ClientScenario) -> list[str]:
        """Identify protective factors from client scenario."""
        protective_factors = []

        attachment_style = client_scenario.clinical_formulation.attachment_style
        if attachment_style and "secure" in attachment_style.lower():
            protective_factors.append("Secure attachment style")

        if client_scenario.severity_level != SeverityLevel.SEVERE:
            protective_factors.append("Seeking help voluntarily")

        protective_factors.extend(
            ["Insight into problems", "Motivation for treatment", "No active psychosis"]
        )

        return protective_factors

    def _generate_diagnostic_impressions(
        self, assessment_findings: dict[str, Any], client_scenario: ClientScenario
    ) -> list[str]:
        """Generate diagnostic impressions from assessment findings."""
        impressions = []

        # Primary diagnostic impressions based on DSM-5 considerations
        dsm5_considerations = client_scenario.clinical_formulation.dsm5_considerations

        for consideration in dsm5_considerations:
            if "depression" in consideration.lower():
                impressions.append("Major Depressive Disorder, moderate severity")
            elif "anxiety" in consideration.lower():
                impressions.append("Generalized Anxiety Disorder")
            elif "trauma" in consideration.lower():
                impressions.append("Post-Traumatic Stress Disorder")
            else:
                impressions.append(f"Rule out {consideration}")

        # Add severity specifiers
        if client_scenario.severity_level == SeverityLevel.SEVERE:
            impressions = [imp.replace("moderate", "severe") for imp in impressions]
        elif client_scenario.severity_level == SeverityLevel.MILD:
            impressions = [imp.replace("moderate", "mild") for imp in impressions]

        return impressions

    def _generate_recommendations(
        self, assessment_findings: dict[str, Any], client_scenario: ClientScenario
    ) -> list[str]:
        """Generate treatment recommendations from assessment."""
        recommendations = []

        # Risk-based recommendations (highest priority)
        risk_factors = assessment_findings.get("risk_factors", [])
        if (
            any("suicide" in factor.lower() for factor in risk_factors)
            or client_scenario.severity_level == SeverityLevel.SEVERE
        ):
            recommendations.insert(0, "Safety planning and risk monitoring")
            recommendations.insert(1, "Crisis intervention protocols")

        # Evidence-based treatment recommendations
        dsm5_considerations = client_scenario.clinical_formulation.dsm5_considerations

        if any(
            "depression" in consideration.lower() or "depressive" in consideration.lower()
            for consideration in dsm5_considerations
        ):
            recommendations.extend(
                [
                    "Cognitive Behavioral Therapy (CBT)",
                    "Consider antidepressant medication evaluation",
                    "Regular therapy sessions (weekly initially)",
                ]
            )

        if any("anxiety" in consideration.lower() for consideration in dsm5_considerations):
            recommendations.extend(
                [
                    "Anxiety management techniques",
                    "Exposure therapy if appropriate",
                    "Relaxation and mindfulness training",
                ]
            )

        # General recommendations
        recommendations.extend(
            [
                "Psychoeducation about diagnosis and treatment",
                "Regular progress monitoring",
                "Lifestyle modifications (sleep, exercise, nutrition)",
            ]
        )

        return recommendations

    def _assess_risk_level(
        self, assessment_findings: dict[str, Any], client_scenario: ClientScenario
    ) -> str:
        """Assess overall risk level from assessment."""

        risk_factors = assessment_findings.get("risk_factors", [])

        if (
            "Potential suicide risk" in risk_factors
            or client_scenario.severity_level == SeverityLevel.SEVERE
        ):
            return "high"
        if len(risk_factors) > 2:
            return "moderate"
        return "low"

    def _generate_follow_up_plan(
        self, assessment_findings: dict[str, Any], recommendations: list[str]
    ) -> str:
        """Generate follow-up plan from assessment."""

        plan_components = []

        # Immediate follow-up
        if "Safety planning" in " ".join(recommendations):
            plan_components.append("Schedule follow-up within 1 week for safety monitoring")
        else:
            plan_components.append("Schedule treatment planning session within 2 weeks")

        # Ongoing monitoring
        plan_components.append("Regular therapy sessions as recommended")
        plan_components.append("Monitor symptom progression and treatment response")

        # Additional services
        if "medication evaluation" in " ".join(recommendations).lower():
            plan_components.append("Coordinate with psychiatrist for medication evaluation")

        return "; ".join(plan_components)

    def export_diagnostic_conversations(
        self, conversations: list[DiagnosticConversation], output_file: str
    ) -> dict[str, Any]:
        """Export diagnostic conversations to JSON file."""

        # Convert conversations to JSON-serializable format
        serializable_conversations = []
        for conv in conversations:
            conv_dict = asdict(conv)
            self._convert_enums_to_strings(conv_dict)
            serializable_conversations.append(conv_dict)

        export_data = {
            "metadata": {
                "total_conversations": len(conversations),
                "export_timestamp": datetime.now().isoformat(),
                "assessment_types": list(
                    {conv.assessment_protocol.assessment_type.value for conv in conversations}
                ),
                "domains_covered": list(
                    {domain.value for conv in conversations for domain in conv.domains_assessed}
                ),
                "average_exchanges": sum(len(conv.conversation_exchanges) for conv in conversations)
                / len(conversations)
                if conversations
                else 0,
            },
            "conversations": serializable_conversations,
        }

        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(export_data, f, indent=2, ensure_ascii=False)

        return {
            "exported_conversations": len(conversations),
            "output_file": output_file,
            "assessment_types_covered": len(export_data["metadata"]["assessment_types"]),
            "domains_covered": len(export_data["metadata"]["domains_covered"]),
        }

    def _convert_enums_to_strings(self, obj):
        """Recursively convert enum values to strings for JSON serialization."""
        if isinstance(obj, dict):
            # Handle enum keys in dictionaries
            new_dict = {}
            for key, value in obj.items():
                # Convert enum keys to strings
                new_key = key.value if hasattr(key, "value") else key

                # Convert enum values to strings
                if hasattr(value, "value"):
                    new_dict[new_key] = value.value
                elif isinstance(value, (dict, list)):
                    new_dict[new_key] = value
                    self._convert_enums_to_strings(new_dict[new_key])
                else:
                    new_dict[new_key] = value

            # Replace original dict contents
            obj.clear()
            obj.update(new_dict)

        elif isinstance(obj, list):
            for i, item in enumerate(obj):
                if hasattr(item, "value"):  # Enum object
                    obj[i] = item.value
                elif isinstance(item, (dict, list)):
                    self._convert_enums_to_strings(item)
