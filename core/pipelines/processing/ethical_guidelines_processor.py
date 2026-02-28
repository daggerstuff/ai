"""
Ethical Guidelines and Professional Boundaries Processing System

This module provides comprehensive processing of ethical guidelines and professional
boundaries for psychology knowledge-based conversation generation. It ensures that
all therapeutic conversations adhere to professional standards and ethical principles.

Key Features:
- Professional ethics framework integration
- Boundary-setting conversation templates
- Ethical decision-making support
- Professional standards compliance
- Dual relationship prevention
- Confidentiality and privacy protection
- Informed consent processing
"""

import json
import random
from dataclasses import asdict, dataclass
from datetime import datetime
from enum import Enum
from typing import Any


class EthicalPrinciple(Enum):
    """Core ethical principles in psychology practice."""
    BENEFICENCE = "beneficence"  # Do good
    NON_MALEFICENCE = "non_maleficence"  # Do no harm
    AUTONOMY = "autonomy"  # Respect client self-determination
    JUSTICE = "justice"  # Fair treatment and access
    FIDELITY = "fidelity"  # Trustworthiness and responsibility
    INTEGRITY = "integrity"  # Honesty and accuracy


class ProfessionalBoundary(Enum):
    """Types of professional boundaries in therapeutic relationships."""
    DUAL_RELATIONSHIPS = "dual_relationships"
    PHYSICAL_BOUNDARIES = "physical_boundaries"
    EMOTIONAL_BOUNDARIES = "emotional_boundaries"
    FINANCIAL_BOUNDARIES = "financial_boundaries"
    TIME_BOUNDARIES = "time_boundaries"
    COMMUNICATION_BOUNDARIES = "communication_boundaries"
    GIFT_BOUNDARIES = "gift_boundaries"
    SOCIAL_MEDIA_BOUNDARIES = "social_media_boundaries"


class EthicalDilemmaType(Enum):
    """Types of ethical dilemmas in therapeutic practice."""
    CONFIDENTIALITY_CONFLICT = "confidentiality_conflict"
    DUTY_TO_WARN = "duty_to_warn"
    COMPETENCE_LIMITS = "competence_limits"
    MULTIPLE_RELATIONSHIPS = "multiple_relationships"
    INFORMED_CONSENT = "informed_consent"
    CULTURAL_SENSITIVITY = "cultural_sensitivity"
    SUPERVISION_ISSUES = "supervision_issues"
    TERMINATION_ETHICS = "termination_ethics"


class EthicalSeverity(Enum):
    """Severity levels for ethical concerns."""
    MINOR = "minor"  # Minor boundary crossing
    MODERATE = "moderate"  # Significant ethical concern
    MAJOR = "major"  # Serious ethical violation
    CRITICAL = "critical"  # Immediate intervention required


@dataclass
class EthicalGuideline:
    """Comprehensive ethical guideline definition."""
    id: str
    principle: EthicalPrinciple
    title: str
    description: str
    applicable_situations: list[str]
    boundary_type: ProfessionalBoundary | None
    implementation_steps: list[str]
    warning_signs: list[str]
    corrective_actions: list[str]
    documentation_requirements: list[str]
    supervision_triggers: list[str]
    legal_considerations: list[str]


@dataclass
class EthicalScenario:
    """Ethical dilemma scenario for training."""
    id: str
    dilemma_type: EthicalDilemmaType
    severity: EthicalSeverity
    situation_description: str
    ethical_principles_involved: list[EthicalPrinciple]
    boundary_concerns: list[ProfessionalBoundary]
    stakeholders: list[str]
    potential_consequences: list[str]
    decision_factors: list[str]
    recommended_actions: list[str]
    consultation_needed: bool
    documentation_required: bool


@dataclass
class EthicalConversation:
    """Ethical boundary-setting conversation."""
    id: str
    scenario: EthicalScenario
    conversation_exchanges: list[dict[str, Any]]
    ethical_principles_demonstrated: list[EthicalPrinciple]
    boundaries_addressed: list[ProfessionalBoundary]
    teaching_points: list[str]
    follow_up_actions: list[str]
    supervision_notes: str | None


class EthicalGuidelinesProcessor:
    """
    Comprehensive ethical guidelines and professional boundaries processing system.

    This class provides processing of ethical guidelines, boundary-setting scenarios,
    and ethical decision-making frameworks for therapeutic conversation generation.
    """

    def __init__(self):
        """Initialize the ethical guidelines processor."""
        self.ethical_guidelines = self._initialize_ethical_guidelines()
        self.boundary_frameworks = self._initialize_boundary_frameworks()
        self.ethical_scenarios = self._initialize_ethical_scenarios()

    def _initialize_ethical_guidelines(self) -> dict[EthicalPrinciple, list[EthicalGuideline]]:
        """Initialize comprehensive ethical guidelines."""
        guidelines = {}

        # Beneficence Guidelines
        guidelines[EthicalPrinciple.BENEFICENCE] = [
            EthicalGuideline(
                id="beneficence_client_welfare",
                principle=EthicalPrinciple.BENEFICENCE,
                title="Client Welfare Priority",
                description="Always prioritize client welfare and well-being in all therapeutic decisions",
                applicable_situations=["treatment planning", "intervention selection", "crisis situations"],
                boundary_type=None,
                implementation_steps=[
                    "Assess potential benefits and risks of interventions",
                    "Consider client's best interests in all decisions",
                    "Monitor treatment progress and adjust as needed",
                    "Seek consultation when uncertain about client welfare"
                ],
                warning_signs=[
                    "Client expressing harm to self or others",
                    "Treatment not showing expected progress",
                    "Client requesting potentially harmful interventions"
                ],
                corrective_actions=[
                    "Reassess treatment approach",
                    "Seek immediate consultation",
                    "Consider referral to specialist",
                    "Implement safety measures"
                ],
                documentation_requirements=[
                    "Document decision-making rationale",
                    "Record consultation discussions",
                    "Note safety assessments"
                ],
                supervision_triggers=[
                    "Complex ethical decisions",
                    "Client safety concerns",
                    "Treatment effectiveness questions"
                ],
                legal_considerations=[
                    "Duty of care obligations",
                    "Standard of practice requirements",
                    "Liability for treatment decisions"
                ]
            )
        ]

        # Non-Maleficence Guidelines
        guidelines[EthicalPrinciple.NON_MALEFICENCE] = [
            EthicalGuideline(
                id="non_maleficence_do_no_harm",
                principle=EthicalPrinciple.NON_MALEFICENCE,
                title="Do No Harm Principle",
                description="Avoid actions that could cause harm to clients, colleagues, or the profession",
                applicable_situations=["all therapeutic interactions", "professional relationships", "public statements"],
                boundary_type=None,
                implementation_steps=[
                    "Assess potential for harm in all interventions",
                    "Maintain competence in areas of practice",
                    "Recognize limits of expertise",
                    "Refer when appropriate"
                ],
                warning_signs=[
                    "Client deterioration during treatment",
                    "Practicing outside competence area",
                    "Personal issues affecting judgment"
                ],
                corrective_actions=[
                    "Immediate cessation of harmful practices",
                    "Seek supervision or consultation",
                    "Refer to appropriate specialist",
                    "Personal therapy if needed"
                ],
                documentation_requirements=[
                    "Document harm assessment",
                    "Record referral decisions",
                    "Note competence evaluations"
                ],
                supervision_triggers=[
                    "Any indication of potential harm",
                    "Competence concerns",
                    "Personal issues affecting practice"
                ],
                legal_considerations=[
                    "Malpractice liability",
                    "Standard of care violations",
                    "Duty to refer when incompetent"
                ]
            )
        ]

        # Autonomy Guidelines
        guidelines[EthicalPrinciple.AUTONOMY] = [
            EthicalGuideline(
                id="autonomy_informed_consent",
                principle=EthicalPrinciple.AUTONOMY,
                title="Informed Consent and Client Self-Determination",
                description="Respect client's right to make informed decisions about their treatment",
                applicable_situations=["treatment initiation", "intervention changes", "termination"],
                boundary_type=None,
                implementation_steps=[
                    "Provide clear information about treatment",
                    "Explain risks and benefits",
                    "Respect client's treatment choices",
                    "Support client decision-making capacity"
                ],
                warning_signs=[
                    "Client expressing confusion about treatment",
                    "Pressure to accept specific interventions",
                    "Lack of genuine consent"
                ],
                corrective_actions=[
                    "Re-explain treatment options",
                    "Assess decision-making capacity",
                    "Provide additional time for decisions",
                    "Seek consultation if needed"
                ],
                documentation_requirements=[
                    "Document informed consent process",
                    "Record client's understanding",
                    "Note any capacity concerns"
                ],
                supervision_triggers=[
                    "Capacity concerns",
                    "Complex consent issues",
                    "Client resistance to treatment"
                ],
                legal_considerations=[
                    "Informed consent requirements",
                    "Capacity determination",
                    "Right to refuse treatment"
                ]
            )
        ]

        return guidelines

    def _initialize_boundary_frameworks(self) -> dict[ProfessionalBoundary, dict[str, Any]]:
        """Initialize professional boundary frameworks."""
        return {
            ProfessionalBoundary.DUAL_RELATIONSHIPS: {
                "definition": "Relationships outside the therapeutic context",
                "risk_factors": ["small communities", "professional overlap", "social connections"],
                "prevention_strategies": ["clear role definition", "referral when appropriate", "consultation"],
                "warning_signs": ["role confusion", "boundary blurring", "personal involvement"],
                "intervention_steps": ["immediate consultation", "role clarification", "possible referral"]
            },
            ProfessionalBoundary.PHYSICAL_BOUNDARIES: {
                "definition": "Appropriate physical contact and space",
                "risk_factors": ["cultural differences", "trauma history", "therapeutic modality"],
                "prevention_strategies": ["clear policies", "client education", "cultural sensitivity"],
                "warning_signs": ["inappropriate touch", "invasion of personal space", "discomfort"],
                "intervention_steps": ["immediate cessation", "discussion with client", "documentation"]
            },
            ProfessionalBoundary.EMOTIONAL_BOUNDARIES: {
                "definition": "Appropriate emotional involvement and distance",
                "risk_factors": ["countertransference", "personal issues", "client similarity"],
                "prevention_strategies": ["self-awareness", "supervision", "personal therapy"],
                "warning_signs": ["over-involvement", "emotional fusion", "loss of objectivity"],
                "intervention_steps": ["supervision", "self-reflection", "possible referral"]
            }
        }

    def _initialize_ethical_scenarios(self) -> list[EthicalScenario]:
        """Initialize ethical dilemma scenarios for training."""
        return [
            EthicalScenario(
                id="confidentiality_family_concern",
                dilemma_type=EthicalDilemmaType.CONFIDENTIALITY_CONFLICT,
                severity=EthicalSeverity.MODERATE,
                situation_description="Family member calls requesting information about adult client's progress",
                ethical_principles_involved=[EthicalPrinciple.AUTONOMY, EthicalPrinciple.FIDELITY],
                boundary_concerns=[ProfessionalBoundary.COMMUNICATION_BOUNDARIES],
                stakeholders=["client", "family member", "therapist"],
                potential_consequences=["breach of confidentiality", "family conflict", "trust issues"],
                decision_factors=["client consent", "legal requirements", "therapeutic relationship"],
                recommended_actions=[
                    "Explain confidentiality limits",
                    "Obtain client consent before sharing",
                    "Offer family session if appropriate"
                ],
                consultation_needed=False,
                documentation_required=True
            ),
            EthicalScenario(
                id="duty_to_warn_threat",
                dilemma_type=EthicalDilemmaType.DUTY_TO_WARN,
                severity=EthicalSeverity.MAJOR,
                situation_description="Client expresses specific threat to harm identified person",
                ethical_principles_involved=[EthicalPrinciple.NON_MALEFICENCE, EthicalPrinciple.FIDELITY],
                boundary_concerns=[ProfessionalBoundary.COMMUNICATION_BOUNDARIES],
                stakeholders=["client", "potential victim", "therapist", "authorities"],
                potential_consequences=["breach of confidentiality", "client anger", "legal liability"],
                decision_factors=["imminent danger", "specific threat", "legal requirements"],
                recommended_actions=[
                    "Assess immediacy of threat",
                    "Contact authorities if required",
                    "Warn potential victim",
                    "Document all actions"
                ],
                consultation_needed=True,
                documentation_required=True
            ),
            EthicalScenario(
                id="competence_limits_specialty",
                dilemma_type=EthicalDilemmaType.COMPETENCE_LIMITS,
                severity=EthicalSeverity.MODERATE,
                situation_description="Client presents with specialized issue outside therapist's expertise",
                ethical_principles_involved=[EthicalPrinciple.NON_MALEFICENCE, EthicalPrinciple.BENEFICENCE],
                boundary_concerns=[],
                stakeholders=["client", "therapist", "specialist"],
                potential_consequences=["inadequate treatment", "client harm", "professional liability"],
                decision_factors=["scope of competence", "client needs", "available resources"],
                recommended_actions=[
                    "Acknowledge limitations honestly",
                    "Seek consultation or supervision",
                    "Refer to appropriate specialist",
                    "Provide interim support if safe"
                ],
                consultation_needed=True,
                documentation_required=True
            )
        ]

    def generate_ethical_conversation(
        self,
        scenario: EthicalScenario,
        num_exchanges: int = 6
    ) -> EthicalConversation:
        """
        Generate ethical boundary-setting conversation based on scenario.

        Args:
            scenario: Ethical scenario to address
            num_exchanges: Number of conversation exchanges

        Returns:
            EthicalConversation with boundary-setting dialogue
        """
        exchanges = self._generate_ethical_exchanges(scenario, num_exchanges)

        # Identify principles demonstrated
        principles_demonstrated = self._identify_demonstrated_principles(exchanges, scenario)

        # Identify boundaries addressed
        boundaries_addressed = scenario.boundary_concerns

        # Generate teaching points
        teaching_points = self._generate_teaching_points(scenario, exchanges)

        # Generate follow-up actions
        follow_up_actions = self._generate_follow_up_actions(scenario)

        # Generate supervision notes if needed
        supervision_notes = self._generate_supervision_notes(scenario) if scenario.consultation_needed else None

        conversation_id = f"ethical_{scenario.id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        return EthicalConversation(
            id=conversation_id,
            scenario=scenario,
            conversation_exchanges=exchanges,
            ethical_principles_demonstrated=principles_demonstrated,
            boundaries_addressed=boundaries_addressed,
            teaching_points=teaching_points,
            follow_up_actions=follow_up_actions,
            supervision_notes=supervision_notes
        )

    def _generate_ethical_exchanges(
        self,
        scenario: EthicalScenario,
        num_exchanges: int
    ) -> list[dict[str, Any]]:
        """Generate conversation exchanges for ethical scenario."""
        exchanges = []

        for i in range(num_exchanges):
            if i % 2 == 0:  # Therapist turn
                exchange = self._generate_therapist_ethical_response(scenario, i // 2)
            else:  # Client/other party turn
                exchange = self._generate_client_ethical_response(scenario, i // 2)

            exchanges.append(exchange)

        return exchanges

    def _generate_therapist_ethical_response(
        self,
        scenario: EthicalScenario,
        exchange_index: int
    ) -> dict[str, Any]:
        """Generate therapist response addressing ethical concerns."""

        # Select appropriate ethical principle to demonstrate
        principle = scenario.ethical_principles_involved[exchange_index % len(scenario.ethical_principles_involved)]

        # Generate content based on scenario type and principle
        content = self._generate_ethical_content(scenario, principle, exchange_index)

        return {
            "speaker": "therapist",
            "content": content,
            "ethical_principle": principle.value,
            "boundary_addressed": scenario.boundary_concerns[0].value if scenario.boundary_concerns else None,
            "professional_action": self._determine_professional_action(scenario, exchange_index),
            "timestamp": datetime.now().isoformat()
        }

    def _generate_client_ethical_response(
        self,
        scenario: EthicalScenario,
        exchange_index: int
    ) -> dict[str, Any]:
        """Generate client/other party response in ethical scenario."""

        # Generate response based on scenario type
        content = self._generate_stakeholder_response(scenario, exchange_index)

        return {
            "speaker": self._determine_speaker(scenario, exchange_index),
            "content": content,
            "emotional_state": self._determine_emotional_state(scenario, exchange_index),
            "understanding_level": self._assess_understanding_level(scenario, exchange_index),
            "timestamp": datetime.now().isoformat()
        }

    def _generate_ethical_content(
        self,
        scenario: EthicalScenario,
        principle: EthicalPrinciple,
        exchange_index: int
    ) -> str:
        """Generate ethical content based on principle and scenario."""

        content_templates = {
            EthicalPrinciple.AUTONOMY: [
                "I want to make sure you understand your rights and options in this situation.",
                "It's important that you make an informed decision about how to proceed.",
                "You have the right to choose what feels most comfortable for you."
            ],
            EthicalPrinciple.BENEFICENCE: [
                "My primary concern is what's in your best interest.",
                "Let's think about what approach would be most helpful for you.",
                "I want to make sure we're doing what's most beneficial for your situation."
            ],
            EthicalPrinciple.NON_MALEFICENCE: [
                "I need to be careful that we don't do anything that could be harmful.",
                "Let me explain why I have concerns about this approach.",
                "My professional obligation is to avoid any potential harm."
            ],
            EthicalPrinciple.FIDELITY: [
                "I want to be completely honest with you about the situation.",
                "Trust is fundamental to our therapeutic relationship.",
                "I need to maintain my professional responsibilities while supporting you."
            ],
            EthicalPrinciple.INTEGRITY: [
                "I need to be transparent about my professional obligations.",
                "Honesty requires me to share this information with you.",
                "My professional integrity means I must address this directly."
            ],
            EthicalPrinciple.JUSTICE: [
                "Everyone deserves fair and equal treatment.",
                "I want to ensure you receive the same quality of care as any client.",
                "It's important that we treat this situation fairly for all involved."
            ]
        }

        templates = content_templates.get(principle, [
            "Let me address this professional concern with you.",
            "This situation requires careful ethical consideration.",
            "I need to handle this according to professional standards."
        ])

        base_content = random.choice(templates)

        # Add scenario-specific context
        if scenario.dilemma_type == EthicalDilemmaType.CONFIDENTIALITY_CONFLICT:
            base_content += " Confidentiality is a cornerstone of our therapeutic relationship."
        elif scenario.dilemma_type == EthicalDilemmaType.DUTY_TO_WARN:
            base_content += " I have both legal and ethical obligations to consider."
        elif scenario.dilemma_type == EthicalDilemmaType.COMPETENCE_LIMITS:
            base_content += " I want to ensure you receive the most appropriate care."

        return base_content

    def _generate_stakeholder_response(self, scenario: EthicalScenario, exchange_index: int) -> str:
        """Generate stakeholder response based on scenario type."""

        if scenario.dilemma_type == EthicalDilemmaType.CONFIDENTIALITY_CONFLICT:
            responses = [
                "I'm just worried about them and want to know if they're okay.",
                "Can't you tell me anything? I'm their family.",
                "I understand, but I just need to know they're safe."
            ]
        elif scenario.dilemma_type == EthicalDilemmaType.DUTY_TO_WARN:
            responses = [
                "You can't tell anyone what I said! This is supposed to be confidential.",
                "I was just angry. I didn't really mean it.",
                "What happens if you have to report this?"
            ]
        elif scenario.dilemma_type == EthicalDilemmaType.COMPETENCE_LIMITS:
            responses = [
                "But I've been working with you and I trust you.",
                "Do I really need to see someone else?",
                "I don't want to start over with a new therapist."
            ]
        else:
            responses = [
                "I don't understand why this is a problem.",
                "Can you explain what this means for me?",
                "I'm confused about what happens next."
            ]

        return random.choice(responses)

    def _determine_speaker(self, scenario: EthicalScenario, exchange_index: int) -> str:
        """Determine who is speaking in this exchange."""
        if scenario.dilemma_type == EthicalDilemmaType.CONFIDENTIALITY_CONFLICT:
            return "family_member" if exchange_index == 0 else "client"
        return "client"

    def _determine_professional_action(self, scenario: EthicalScenario, exchange_index: int) -> str:
        """Determine professional action being taken."""
        actions = {
            EthicalDilemmaType.CONFIDENTIALITY_CONFLICT: "boundary_setting",
            EthicalDilemmaType.DUTY_TO_WARN: "risk_assessment",
            EthicalDilemmaType.COMPETENCE_LIMITS: "referral_discussion",
            EthicalDilemmaType.MULTIPLE_RELATIONSHIPS: "boundary_clarification",
            EthicalDilemmaType.INFORMED_CONSENT: "consent_process"
        }
        return actions.get(scenario.dilemma_type, "ethical_guidance")

    def _determine_emotional_state(self, scenario: EthicalScenario, exchange_index: int) -> str:
        """Determine emotional state of speaker."""
        if scenario.severity == EthicalSeverity.MAJOR:
            return random.choice(["anxious", "defensive", "confused"])
        if scenario.severity == EthicalSeverity.MODERATE:
            return random.choice(["concerned", "uncertain", "cooperative"])
        return random.choice(["calm", "understanding", "accepting"])

    def _assess_understanding_level(self, scenario: EthicalScenario, exchange_index: int) -> str:
        """Assess understanding level of ethical issues."""
        if exchange_index == 0:
            return "low"
        if exchange_index == 1:
            return "developing"
        return "good"

    def _identify_demonstrated_principles(
        self,
        exchanges: list[dict[str, Any]],
        scenario: EthicalScenario
    ) -> list[EthicalPrinciple]:
        """Identify ethical principles demonstrated in conversation."""
        demonstrated = set()

        for exchange in exchanges:
            if exchange.get("speaker") == "therapist" and "ethical_principle" in exchange:
                principle_value = exchange["ethical_principle"]
                for principle in EthicalPrinciple:
                    if principle.value == principle_value:
                        demonstrated.add(principle)

        return list(demonstrated)

    def _generate_teaching_points(
        self,
        scenario: EthicalScenario,
        exchanges: list[dict[str, Any]]
    ) -> list[str]:
        """Generate teaching points from ethical conversation."""
        teaching_points = []

        # Add scenario-specific teaching points
        if scenario.dilemma_type == EthicalDilemmaType.CONFIDENTIALITY_CONFLICT:
            teaching_points.extend([
                "Confidentiality protects the therapeutic relationship",
                "Client consent is required before sharing information",
                "Family concerns can be addressed without breaching confidentiality"
            ])
        elif scenario.dilemma_type == EthicalDilemmaType.DUTY_TO_WARN:
            teaching_points.extend([
                "Duty to warn overrides confidentiality in specific circumstances",
                "Imminent danger must be assessed carefully",
                "Documentation is crucial in duty to warn situations"
            ])
        elif scenario.dilemma_type == EthicalDilemmaType.COMPETENCE_LIMITS:
            teaching_points.extend([
                "Practicing within competence protects clients",
                "Referral is an ethical obligation when appropriate",
                "Honesty about limitations builds trust"
            ])

        # Add principle-specific teaching points
        for principle in scenario.ethical_principles_involved:
            if principle == EthicalPrinciple.AUTONOMY:
                teaching_points.append("Respect client's right to self-determination")
            elif principle == EthicalPrinciple.BENEFICENCE:
                teaching_points.append("Always act in client's best interest")
            elif principle == EthicalPrinciple.NON_MALEFICENCE:
                teaching_points.append("Avoid actions that could cause harm")

        return teaching_points

    def _generate_follow_up_actions(self, scenario: EthicalScenario) -> list[str]:
        """Generate follow-up actions for ethical scenario."""
        actions = []

        if scenario.documentation_required:
            actions.append("Document ethical decision-making process")

        if scenario.consultation_needed:
            actions.append("Seek supervision or consultation")

        # Add scenario-specific actions
        actions.extend(scenario.recommended_actions)

        return actions

    def _generate_supervision_notes(self, scenario: EthicalScenario) -> str:
        """Generate supervision notes for ethical scenario."""
        return f"Supervision needed for {scenario.dilemma_type.value} situation. " \
               f"Severity: {scenario.severity.value}. " \
               f"Key considerations: {', '.join(scenario.decision_factors)}."

    def export_ethical_conversations(
        self,
        conversations: list[EthicalConversation],
        output_file: str
    ) -> dict[str, Any]:
        """Export ethical conversations to JSON file."""

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
                "ethical_principles_covered": list({
                    principle.value for conv in conversations
                    for principle in conv.ethical_principles_demonstrated
                }),
                "boundary_types_addressed": list({
                    boundary.value for conv in conversations
                    for boundary in conv.boundaries_addressed
                })
            },
            "conversations": serializable_conversations
        }

        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(export_data, f, indent=2, ensure_ascii=False)

        return {
            "exported_conversations": len(conversations),
            "output_file": output_file,
            "principles_covered": len(export_data["metadata"]["ethical_principles_covered"]),
            "boundaries_addressed": len(export_data["metadata"]["boundary_types_addressed"])
        }

    def _convert_enums_to_strings(self, obj):
        """Recursively convert enum values to strings for JSON serialization."""
        if isinstance(obj, dict):
            for key, value in obj.items():
                if hasattr(value, "value"):  # Enum object
                    obj[key] = value.value
                elif isinstance(value, (dict, list)):
                    self._convert_enums_to_strings(value)
        elif isinstance(obj, list):
            for i, item in enumerate(obj):
                if hasattr(item, "value"):  # Enum object
                    obj[i] = item.value
                elif isinstance(item, (dict, list)):
                    self._convert_enums_to_strings(item)
