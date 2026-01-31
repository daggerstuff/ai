"""
Therapeutic Techniques and Intervention Strategies Integration System

This module provides comprehensive integration of therapeutic techniques and intervention
strategies for psychology knowledge-based conversation generation. It extends beyond
basic therapeutic responses to include specialized protocols, evidence-based frameworks,
and intervention effectiveness tracking.

Key Features:
- Advanced intervention strategy selection
- Evidence-based practice protocols
- Specialized therapeutic frameworks (EMDR, DBT, ACT, etc.)
- Intervention effectiveness tracking
- Protocol-specific conversation generation
- Clinical decision-making support
"""

import json
import random
from dataclasses import asdict, dataclass
from datetime import datetime
from enum import Enum
from typing import Any

from .client_scenario_generator import ClientScenario, ScenarioType, SeverityLevel
from .therapeutic_response_generator import TherapeuticTechnique


class InterventionStrategy(Enum):
    """Advanced intervention strategies beyond basic techniques."""
    CRISIS_STABILIZATION = "crisis_stabilization"
    TRAUMA_PROCESSING = "trauma_processing"
    BEHAVIORAL_MODIFICATION = "behavioral_modification"
    COGNITIVE_RESTRUCTURING_PROTOCOL = "cognitive_restructuring_protocol"
    EMOTIONAL_REGULATION = "emotional_regulation"
    INTERPERSONAL_SKILLS = "interpersonal_skills"
    RELAPSE_PREVENTION = "relapse_prevention"
    EXPOSURE_THERAPY = "exposure_therapy"
    ACCEPTANCE_BASED = "acceptance_based"
    MINDFULNESS_INTEGRATION = "mindfulness_integration"
    FAMILY_SYSTEMS = "family_systems"
    PSYCHOEDUCATION_PROTOCOL = "psychoeducation_protocol"


class EvidenceLevel(Enum):
    """Evidence levels for therapeutic interventions."""
    STRONG = "strong"  # Multiple RCTs, meta-analyses
    MODERATE = "moderate"  # Some RCTs, consistent findings
    EMERGING = "emerging"  # Limited but promising evidence
    EXPERT_CONSENSUS = "expert_consensus"  # Professional guidelines
    THEORETICAL = "theoretical"  # Theory-based, limited empirical support


class SpecializedFramework(Enum):
    """Specialized therapeutic frameworks and protocols."""
    EMDR = "emdr"  # Eye Movement Desensitization and Reprocessing
    DBT = "dbt"  # Dialectical Behavior Therapy
    ACT = "act"  # Acceptance and Commitment Therapy
    IFS = "ifs"  # Internal Family Systems
    SOMATIC_EXPERIENCING = "somatic_experiencing"
    NARRATIVE_THERAPY = "narrative_therapy"
    GESTALT = "gestalt"
    MOTIVATIONAL_INTERVIEWING = "motivational_interviewing"
    SOLUTION_FOCUSED = "solution_focused"
    INTERPERSONAL_THERAPY = "interpersonal_therapy"


@dataclass
class InterventionProtocol:
    """Comprehensive intervention protocol definition."""
    id: str
    name: str
    strategy: InterventionStrategy
    framework: SpecializedFramework | None
    evidence_level: EvidenceLevel
    target_conditions: list[str]
    contraindications: list[str]
    session_structure: list[str]
    key_techniques: list[TherapeuticTechnique]
    expected_outcomes: list[str]
    duration_sessions: tuple[int, int]  # (min, max) sessions
    homework_assignments: list[str]
    progress_indicators: list[str]
    risk_considerations: list[str]


@dataclass
class InterventionSelection:
    """Selected intervention with rationale and implementation details."""
    protocol: InterventionProtocol
    primary_rationale: str
    secondary_considerations: list[str]
    adaptation_notes: list[str]
    monitoring_plan: list[str]
    expected_timeline: str
    success_metrics: list[str]


@dataclass
class ConversationIntervention:
    """Intervention-based conversation with therapeutic context."""
    id: str
    client_scenario: ClientScenario
    intervention_selection: InterventionSelection
    conversation_exchanges: list[dict[str, Any]]
    session_phase: str  # "opening", "working", "closing"
    therapeutic_goals: list[str]
    intervention_fidelity_score: float
    clinical_notes: list[str]
    homework_assigned: str | None
    next_session_plan: str | None


class TherapeuticTechniquesIntegrator:
    """
    Advanced therapeutic techniques and intervention strategies integration system.

    This class provides comprehensive integration of evidence-based therapeutic
    interventions, specialized frameworks, and protocol-driven conversation generation.
    """

    def __init__(self):
        """Initialize the therapeutic techniques integrator."""
        self.intervention_protocols = self._initialize_intervention_protocols()
        self.framework_mappings = self._initialize_framework_mappings()
        self.evidence_base = self._initialize_evidence_base()

    def _initialize_intervention_protocols(self) -> dict[InterventionStrategy, list[InterventionProtocol]]:
        """Initialize comprehensive intervention protocols."""
        protocols = {}

        # Crisis Stabilization Protocols
        protocols[InterventionStrategy.CRISIS_STABILIZATION] = [
            InterventionProtocol(
                id="crisis_safety_planning",
                name="Crisis Safety Planning Protocol",
                strategy=InterventionStrategy.CRISIS_STABILIZATION,
                framework=None,
                evidence_level=EvidenceLevel.STRONG,
                target_conditions=["suicidal ideation", "self-harm", "acute crisis"],
                contraindications=["active psychosis without stabilization"],
                session_structure=[
                    "Immediate safety assessment",
                    "Risk factor identification",
                    "Protective factor enhancement",
                    "Safety plan development",
                    "Support system activation"
                ],
                key_techniques=[
                    TherapeuticTechnique.SAFETY_PLANNING,
                    TherapeuticTechnique.VALIDATION,
                    TherapeuticTechnique.GROUNDING_TECHNIQUES
                ],
                expected_outcomes=["Reduced immediate risk", "Enhanced safety awareness", "Activated support"],
                duration_sessions=(1, 3),
                homework_assignments=["Safety plan review", "Support contact practice"],
                progress_indicators=["Decreased suicidal ideation", "Increased help-seeking"],
                risk_considerations=["Monitor for escalation", "Ensure 24/7 support access"]
            )
        ]

        # Trauma Processing Protocols
        protocols[InterventionStrategy.TRAUMA_PROCESSING] = [
            InterventionProtocol(
                id="emdr_trauma_protocol",
                name="EMDR Trauma Processing Protocol",
                strategy=InterventionStrategy.TRAUMA_PROCESSING,
                framework=SpecializedFramework.EMDR,
                evidence_level=EvidenceLevel.STRONG,
                target_conditions=["PTSD", "trauma", "disturbing memories"],
                contraindications=["active substance abuse", "severe dissociation"],
                session_structure=[
                    "History taking and preparation",
                    "Resource installation",
                    "Target memory processing",
                    "Installation of positive cognition",
                    "Body scan and closure"
                ],
                key_techniques=[
                    TherapeuticTechnique.GROUNDING_TECHNIQUES,
                    TherapeuticTechnique.VALIDATION,
                    TherapeuticTechnique.PSYCHOEDUCATION
                ],
                expected_outcomes=["Reduced trauma symptoms", "Improved emotional regulation"],
                duration_sessions=(8, 16),
                homework_assignments=["Resource practice", "Disturbance log"],
                progress_indicators=["Decreased PTSD symptoms", "Improved daily functioning"],
                risk_considerations=["Monitor for abreaction", "Ensure stabilization"]
            )
        ]

        # Behavioral Modification Protocols
        protocols[InterventionStrategy.BEHAVIORAL_MODIFICATION] = [
            InterventionProtocol(
                id="behavioral_activation_depression",
                name="Behavioral Activation for Depression",
                strategy=InterventionStrategy.BEHAVIORAL_MODIFICATION,
                framework=None,
                evidence_level=EvidenceLevel.STRONG,
                target_conditions=["depression", "behavioral avoidance", "low motivation"],
                contraindications=["severe cognitive impairment"],
                session_structure=[
                    "Activity monitoring",
                    "Value identification",
                    "Activity scheduling",
                    "Barrier problem-solving",
                    "Progress review"
                ],
                key_techniques=[
                    TherapeuticTechnique.BEHAVIORAL_ACTIVATION,
                    TherapeuticTechnique.PSYCHOEDUCATION,
                    TherapeuticTechnique.VALIDATION
                ],
                expected_outcomes=["Increased activity level", "Improved mood", "Enhanced motivation"],
                duration_sessions=(12, 20),
                homework_assignments=["Activity scheduling", "Mood monitoring"],
                progress_indicators=["Increased pleasant activities", "Improved mood ratings"],
                risk_considerations=["Monitor for overwhelm", "Gradual activity increase"]
            )
        ]

        return protocols

    def _initialize_framework_mappings(self) -> dict[SpecializedFramework, dict[str, Any]]:
        """Initialize specialized framework mappings and characteristics."""
        return {
            SpecializedFramework.DBT: {
                "core_modules": ["mindfulness", "distress_tolerance", "emotion_regulation", "interpersonal_effectiveness"],
                "target_populations": ["borderline personality disorder", "emotion_dysregulation"],
                "session_structure": "skills_group_individual_coaching",
                "evidence_level": EvidenceLevel.STRONG
            },
            SpecializedFramework.ACT: {
                "core_processes": ["psychological_flexibility", "values_clarification", "mindfulness", "acceptance"],
                "target_populations": ["anxiety", "depression", "chronic_pain"],
                "session_structure": "experiential_metaphor_based",
                "evidence_level": EvidenceLevel.STRONG
            },
            SpecializedFramework.EMDR: {
                "phases": ["preparation", "assessment", "desensitization", "installation", "body_scan", "closure"],
                "target_populations": ["PTSD", "trauma", "phobias"],
                "session_structure": "structured_bilateral_stimulation",
                "evidence_level": EvidenceLevel.STRONG
            }
        }

    def _initialize_evidence_base(self) -> dict[str, dict[str, Any]]:
        """Initialize evidence base for interventions."""
        return {
            "meta_analyses": {
                "cbt_depression": {"effect_size": 0.68, "studies": 115, "quality": "high"},
                "emdr_ptsd": {"effect_size": 0.88, "studies": 26, "quality": "high"},
                "dbt_bpd": {"effect_size": 0.52, "studies": 13, "quality": "moderate"}
            },
            "practice_guidelines": {
                "apa_ptsd": ["CBT", "EMDR", "exposure_therapy"],
                "nice_depression": ["CBT", "behavioral_activation", "interpersonal_therapy"],
                "samhsa_trauma": ["trauma_informed_care", "evidence_based_treatments"]
            }
        }

    def select_intervention_protocol(
        self,
        client_scenario: ClientScenario,
        preferred_framework: SpecializedFramework | None = None
    ) -> InterventionSelection:
        """
        Select the most appropriate intervention protocol based on client scenario.

        Args:
            client_scenario: Client scenario with clinical formulation
            preferred_framework: Optional preferred therapeutic framework

        Returns:
            InterventionSelection with protocol and implementation details
        """
        # Analyze client needs
        primary_concerns = self._analyze_primary_concerns(client_scenario)
        risk_level = client_scenario.severity_level
        contraindications = self._identify_contraindications(client_scenario)

        # Select appropriate strategy
        strategy = self._select_intervention_strategy(primary_concerns, risk_level)

        # Get available protocols for strategy
        available_protocols = self.intervention_protocols.get(strategy, [])

        # Filter by contraindications and framework preference
        suitable_protocols = self._filter_suitable_protocols(
            available_protocols, contraindications, preferred_framework
        )

        if not suitable_protocols:
            # Fallback to general supportive protocol
            protocol = self._create_fallback_protocol(strategy)
        else:
            # Select best protocol based on evidence and fit
            protocol = self._select_best_protocol(suitable_protocols, client_scenario)

        # Generate implementation details
        return self._create_intervention_selection(protocol, client_scenario)


    def _analyze_primary_concerns(self, client_scenario: ClientScenario) -> list[str]:
        """Analyze primary clinical concerns from scenario."""
        concerns = []

        # Extract from DSM-5 considerations
        dsm5_considerations = client_scenario.clinical_formulation.dsm5_considerations
        for consideration in dsm5_considerations:
            consideration_lower = consideration.lower()
            if "depression" in consideration_lower or "depressive" in consideration_lower:
                concerns.append("depression")
            if "anxiety" in consideration_lower or "anxious" in consideration_lower:
                concerns.append("anxiety")
            if "trauma" in consideration_lower or "ptsd" in consideration_lower:
                concerns.append("trauma")
            if "substance" in consideration_lower or "addiction" in consideration_lower:
                concerns.append("substance_use")
            if "personality" in consideration_lower:
                concerns.append("personality_disorder")

        # Consider scenario type
        if client_scenario.scenario_type == ScenarioType.CRISIS_INTERVENTION:
            concerns.append("crisis")
        elif client_scenario.scenario_type == ScenarioType.THERAPEUTIC_SESSION:
            # Check if trauma-related based on DSM-5 considerations
            if any("trauma" in consideration.lower() or "ptsd" in consideration.lower()
                   for consideration in dsm5_considerations):
                concerns.append("trauma")

        # Consider severity
        if client_scenario.severity_level == SeverityLevel.SEVERE:
            concerns.append("high_acuity")

        return list(set(concerns))  # Remove duplicates

    def _select_intervention_strategy(
        self,
        primary_concerns: list[str],
        risk_level: SeverityLevel
    ) -> InterventionStrategy:
        """Select appropriate intervention strategy based on concerns and risk."""

        # Crisis situations take priority
        if "crisis" in primary_concerns or risk_level == SeverityLevel.SEVERE:
            return InterventionStrategy.CRISIS_STABILIZATION

        # Trauma-specific interventions
        if "trauma" in primary_concerns:
            return InterventionStrategy.TRAUMA_PROCESSING

        # Depression-specific interventions
        if "depression" in primary_concerns:
            return InterventionStrategy.BEHAVIORAL_MODIFICATION

        # Anxiety-specific interventions
        if "anxiety" in primary_concerns:
            return InterventionStrategy.COGNITIVE_RESTRUCTURING_PROTOCOL

        # Emotion regulation for personality disorders
        if "personality_disorder" in primary_concerns:
            return InterventionStrategy.EMOTIONAL_REGULATION

        # Default to cognitive restructuring
        return InterventionStrategy.COGNITIVE_RESTRUCTURING_PROTOCOL

    def _filter_suitable_protocols(
        self,
        protocols: list[InterventionProtocol],
        contraindications: list[str],
        preferred_framework: SpecializedFramework | None
    ) -> list[InterventionProtocol]:
        """Filter protocols based on contraindications and preferences."""
        suitable = []

        for protocol in protocols:
            # Check contraindications
            has_contraindication = any(
                contra.lower() in [c.lower() for c in contraindications]
                for contra in protocol.contraindications
            )

            if has_contraindication:
                continue

            # Check framework preference
            if preferred_framework and protocol.framework != preferred_framework:
                continue

            suitable.append(protocol)

        return suitable

    def _select_best_protocol(
        self,
        protocols: list[InterventionProtocol],
        client_scenario: ClientScenario
    ) -> InterventionProtocol:
        """Select the best protocol based on evidence level and client fit."""

        # Score protocols based on multiple factors
        scored_protocols = []

        for protocol in protocols:
            score = 0

            # Evidence level scoring
            evidence_scores = {
                EvidenceLevel.STRONG: 5,
                EvidenceLevel.MODERATE: 4,
                EvidenceLevel.EMERGING: 3,
                EvidenceLevel.EXPERT_CONSENSUS: 2,
                EvidenceLevel.THEORETICAL: 1
            }
            score += evidence_scores.get(protocol.evidence_level, 0)

            # Target condition match
            dsm5_considerations = client_scenario.clinical_formulation.dsm5_considerations
            for condition in protocol.target_conditions:
                if any(condition.lower() in consideration.lower()
                       for consideration in dsm5_considerations):
                    score += 2

            # Duration appropriateness (prefer shorter for mild cases)
            if client_scenario.severity_level == SeverityLevel.MILD:
                if protocol.duration_sessions[1] <= 12:
                    score += 1

            scored_protocols.append((score, protocol))

        # Return highest scoring protocol
        scored_protocols.sort(key=lambda x: x[0], reverse=True)
        return scored_protocols[0][1] if scored_protocols else protocols[0]

    def _identify_contraindications(self, client_scenario: ClientScenario) -> list[str]:
        """Identify contraindications based on client scenario."""
        contraindications = []

        dsm5_considerations = client_scenario.clinical_formulation.dsm5_considerations

        # Check for psychosis
        if any("psychosis" in consideration.lower() or "psychotic" in consideration.lower()
               for consideration in dsm5_considerations):
            contraindications.append("active psychosis")

        # Check for substance abuse
        if any("substance" in consideration.lower() or "addiction" in consideration.lower()
               for consideration in dsm5_considerations):
            contraindications.append("active substance abuse")

        # Check for cognitive impairment
        if any("cognitive" in consideration.lower() or "dementia" in consideration.lower()
               for consideration in dsm5_considerations):
            contraindications.append("severe cognitive impairment")

        # Check for dissociation
        if any("dissociation" in consideration.lower() or "dissociative" in consideration.lower()
               for consideration in dsm5_considerations):
            contraindications.append("severe dissociation")

        return contraindications

    def _create_fallback_protocol(self, strategy: InterventionStrategy) -> InterventionProtocol:
        """Create a fallback protocol when no suitable protocols are found."""
        return InterventionProtocol(
            id=f"fallback_{strategy.value}",
            name=f"General {strategy.value.replace('_', ' ').title()} Protocol",
            strategy=strategy,
            framework=None,
            evidence_level=EvidenceLevel.EXPERT_CONSENSUS,
            target_conditions=["general mental health concerns"],
            contraindications=[],
            session_structure=[
                "Assessment and rapport building",
                "Problem identification",
                "Intervention planning",
                "Skill building",
                "Progress monitoring"
            ],
            key_techniques=[
                TherapeuticTechnique.ACTIVE_LISTENING,
                TherapeuticTechnique.EMPATHIC_REFLECTION,
                TherapeuticTechnique.VALIDATION
            ],
            expected_outcomes=["Improved coping", "Enhanced insight", "Symptom reduction"],
            duration_sessions=(8, 16),
            homework_assignments=["Self-monitoring", "Skill practice"],
            progress_indicators=["Improved functioning", "Reduced distress"],
            risk_considerations=["Monitor for deterioration", "Assess need for referral"]
        )

    def _create_intervention_selection(
        self,
        protocol: InterventionProtocol,
        client_scenario: ClientScenario
    ) -> InterventionSelection:
        """Create detailed intervention selection with implementation plan."""

        # Generate rationale
        primary_rationale = self._generate_intervention_rationale(protocol, client_scenario)

        # Secondary considerations
        secondary_considerations = self._generate_secondary_considerations(protocol, client_scenario)

        # Adaptation notes
        adaptation_notes = self._generate_adaptation_notes(protocol, client_scenario)

        # Monitoring plan
        monitoring_plan = self._generate_monitoring_plan(protocol, client_scenario)

        # Timeline estimation
        expected_timeline = self._estimate_timeline(protocol, client_scenario)

        # Success metrics
        success_metrics = self._define_success_metrics(protocol, client_scenario)

        return InterventionSelection(
            protocol=protocol,
            primary_rationale=primary_rationale,
            secondary_considerations=secondary_considerations,
            adaptation_notes=adaptation_notes,
            monitoring_plan=monitoring_plan,
            expected_timeline=expected_timeline,
            success_metrics=success_metrics
        )

    def _generate_intervention_rationale(
        self,
        protocol: InterventionProtocol,
        client_scenario: ClientScenario
    ) -> str:
        """Generate clinical rationale for intervention selection."""

        # Base rationale on evidence level
        evidence_rationale = {
            EvidenceLevel.STRONG: "Strong empirical evidence supports this intervention",
            EvidenceLevel.MODERATE: "Moderate evidence base supports this approach",
            EvidenceLevel.EMERGING: "Emerging evidence suggests effectiveness",
            EvidenceLevel.EXPERT_CONSENSUS: "Professional consensus supports this intervention",
            EvidenceLevel.THEORETICAL: "Theoretical framework supports this approach"
        }

        base_rationale = evidence_rationale.get(protocol.evidence_level, "Clinical judgment supports this intervention")

        # Add specific rationale based on client presentation
        dsm5_considerations = client_scenario.clinical_formulation.dsm5_considerations
        specific_rationale = []

        for condition in protocol.target_conditions:
            if any(condition.lower() in consideration.lower() for consideration in dsm5_considerations):
                specific_rationale.append(f"specifically indicated for {condition}")

        if specific_rationale:
            return f"{base_rationale} and is {', '.join(specific_rationale)}."
        return f"{base_rationale} for the presenting concerns."

    def _generate_secondary_considerations(
        self,
        protocol: InterventionProtocol,
        client_scenario: ClientScenario
    ) -> list[str]:
        """Generate secondary considerations for intervention implementation."""
        considerations = []

        # Cultural considerations
        if hasattr(client_scenario, "demographics") and client_scenario.demographics:
            considerations.append("Cultural factors should be integrated into treatment approach")

        # Severity considerations
        if client_scenario.severity_level == SeverityLevel.SEVERE:
            considerations.append("High severity requires careful monitoring and possible adjunct services")
        elif client_scenario.severity_level == SeverityLevel.MILD:
            considerations.append("Mild presentation may benefit from brief intervention approach")

        # Attachment considerations
        attachment_style = client_scenario.clinical_formulation.attachment_style
        if attachment_style and "insecure" in attachment_style.lower():
            considerations.append("Insecure attachment patterns may impact therapeutic alliance")

        # Framework-specific considerations
        if protocol.framework:
            framework_info = self.framework_mappings.get(protocol.framework, {})
            if framework_info:
                considerations.append(f"Requires training in {protocol.framework.value.upper()} methodology")

        return considerations

    def _generate_adaptation_notes(
        self,
        protocol: InterventionProtocol,
        client_scenario: ClientScenario
    ) -> list[str]:
        """Generate adaptation notes for protocol customization."""
        adaptations = []

        # Age-related adaptations
        if hasattr(client_scenario, "demographics") and client_scenario.demographics:
            adaptations.append("Adapt language and examples for client's developmental stage")

        # Severity adaptations
        if client_scenario.severity_level == SeverityLevel.SEVERE:
            adaptations.append("May need to extend duration and increase session frequency")
            adaptations.append("Consider adjunct psychiatric consultation")

        # Scenario-specific adaptations
        if client_scenario.scenario_type == ScenarioType.CRISIS_INTERVENTION:
            adaptations.append("Prioritize safety and stabilization over insight-oriented work")

        # Defense mechanism adaptations
        defense_mechanisms = client_scenario.clinical_formulation.defense_mechanisms
        if defense_mechanisms:
            adaptations.append("Address defense mechanisms gently to maintain therapeutic alliance")

        return adaptations

    def _generate_monitoring_plan(
        self,
        protocol: InterventionProtocol,
        client_scenario: ClientScenario
    ) -> list[str]:
        """Generate monitoring plan for intervention tracking."""
        monitoring = []

        # Standard monitoring
        monitoring.extend([
            "Weekly symptom severity ratings",
            "Session-by-session progress notes",
            "Therapeutic alliance assessment"
        ])

        # Risk-specific monitoring
        if client_scenario.severity_level == SeverityLevel.SEVERE:
            monitoring.extend([
                "Suicide risk assessment each session",
                "Safety plan review and updates",
                "Crisis contact utilization tracking"
            ])

        # Protocol-specific monitoring
        for indicator in protocol.progress_indicators:
            monitoring.append(f"Monitor {indicator.lower()}")

        return monitoring

    def _estimate_timeline(
        self,
        protocol: InterventionProtocol,
        client_scenario: ClientScenario
    ) -> str:
        """Estimate treatment timeline based on protocol and client factors."""
        min_sessions, max_sessions = protocol.duration_sessions

        # Adjust based on severity
        if client_scenario.severity_level == SeverityLevel.SEVERE:
            max_sessions = int(max_sessions * 1.5)
        elif client_scenario.severity_level == SeverityLevel.MILD:
            max_sessions = min(max_sessions, min_sessions + 4)

        return f"{min_sessions}-{max_sessions} sessions over {min_sessions//2}-{max_sessions//2} months"

    def _define_success_metrics(
        self,
        protocol: InterventionProtocol,
        client_scenario: ClientScenario
    ) -> list[str]:
        """Define success metrics for intervention evaluation."""
        metrics = []

        # Protocol-specific outcomes
        metrics.extend(protocol.expected_outcomes)

        # Standardized measures
        dsm5_considerations = client_scenario.clinical_formulation.dsm5_considerations

        if any("depression" in consideration.lower() for consideration in dsm5_considerations):
            metrics.append("PHQ-9 score reduction of 50% or more")

        if any("anxiety" in consideration.lower() for consideration in dsm5_considerations):
            metrics.append("GAD-7 score reduction to subclinical range")

        if any("trauma" in consideration.lower() for consideration in dsm5_considerations):
            metrics.append("PCL-5 score reduction indicating PTSD symptom improvement")

        # Functional outcomes
        metrics.extend([
            "Improved daily functioning",
            "Enhanced quality of life",
            "Increased coping skills utilization"
        ])

        return metrics

    def generate_intervention_conversation(
        self,
        client_scenario: ClientScenario,
        intervention_selection: InterventionSelection,
        session_phase: str = "working",
        num_exchanges: int = 6
    ) -> ConversationIntervention:
        """
        Generate a therapeutic conversation based on selected intervention protocol.

        Args:
            client_scenario: Client scenario with clinical formulation
            intervention_selection: Selected intervention with implementation details
            session_phase: Phase of therapy session ("opening", "working", "closing")
            num_exchanges: Number of conversation exchanges to generate

        Returns:
            ConversationIntervention with protocol-driven therapeutic dialogue
        """
        protocol = intervention_selection.protocol

        # Generate conversation exchanges
        exchanges = self._generate_protocol_exchanges(
            client_scenario, protocol, session_phase, num_exchanges
        )

        # Determine therapeutic goals for this session
        therapeutic_goals = self._determine_session_goals(protocol, session_phase)

        # Calculate intervention fidelity score
        fidelity_score = self._calculate_intervention_fidelity(exchanges, protocol)

        # Generate clinical notes
        clinical_notes = self._generate_clinical_notes(exchanges, protocol, client_scenario)

        # Assign homework if appropriate
        homework = self._assign_homework(protocol, session_phase)

        # Plan next session
        next_session_plan = self._plan_next_session(protocol, session_phase, exchanges)

        # Create conversation ID
        conversation_id = f"intervention_{protocol.id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        return ConversationIntervention(
            id=conversation_id,
            client_scenario=client_scenario,
            intervention_selection=intervention_selection,
            conversation_exchanges=exchanges,
            session_phase=session_phase,
            therapeutic_goals=therapeutic_goals,
            intervention_fidelity_score=fidelity_score,
            clinical_notes=clinical_notes,
            homework_assigned=homework,
            next_session_plan=next_session_plan
        )

    def _generate_protocol_exchanges(
        self,
        client_scenario: ClientScenario,
        protocol: InterventionProtocol,
        session_phase: str,
        num_exchanges: int
    ) -> list[dict[str, Any]]:
        """Generate conversation exchanges following protocol structure."""
        exchanges = []

        # Get session structure for this phase
        if session_phase == "opening":
            structure_elements = protocol.session_structure[:2]
        elif session_phase == "working":
            structure_elements = protocol.session_structure[1:-1]
        else:  # closing
            structure_elements = protocol.session_structure[-2:]

        # Generate exchanges based on structure
        for i in range(num_exchanges):
            if i % 2 == 0:  # Therapist turn
                exchange = self._generate_therapist_exchange(
                    client_scenario, protocol, structure_elements, i // 2
                )
            else:  # Client turn
                exchange = self._generate_client_exchange(
                    client_scenario, protocol, structure_elements, i // 2
                )

            exchanges.append(exchange)

        return exchanges

    def _generate_therapist_exchange(
        self,
        client_scenario: ClientScenario,
        protocol: InterventionProtocol,
        structure_elements: list[str],
        exchange_index: int
    ) -> dict[str, Any]:
        """Generate therapist exchange following protocol guidelines."""

        # Select appropriate technique for this exchange
        technique = random.choice(protocol.key_techniques)

        # Get structure element for this exchange
        structure_element = structure_elements[exchange_index % len(structure_elements)]

        # Generate content based on technique and structure
        content = self._generate_technique_content(
            technique, structure_element, client_scenario, protocol
        )

        return {
            "speaker": "therapist",
            "content": content,
            "technique_used": technique.value,
            "structure_element": structure_element,
            "protocol_adherence": True,
            "timestamp": datetime.now().isoformat()
        }

    def _generate_client_exchange(
        self,
        client_scenario: ClientScenario,
        protocol: InterventionProtocol,
        structure_elements: list[str],
        exchange_index: int
    ) -> dict[str, Any]:
        """Generate client exchange responding to therapeutic intervention."""

        # Generate client response based on scenario and protocol
        content = self._generate_client_response(client_scenario, protocol, exchange_index)

        # Determine client engagement level
        engagement_level = self._determine_client_engagement(client_scenario, protocol)

        return {
            "speaker": "client",
            "content": content,
            "engagement_level": engagement_level,
            "emotional_state": self._determine_emotional_state(client_scenario, exchange_index),
            "timestamp": datetime.now().isoformat()
        }

    def _generate_technique_content(
        self,
        technique: TherapeuticTechnique,
        structure_element: str,
        client_scenario: ClientScenario,
        protocol: InterventionProtocol
    ) -> str:
        """Generate content based on therapeutic technique and protocol structure."""

        technique_templates = {
            TherapeuticTechnique.ACTIVE_LISTENING: [
                "I hear you saying that {concern}. Tell me more about that.",
                "It sounds like {concern} is really important to you.",
                "Help me understand more about {concern}."
            ],
            TherapeuticTechnique.VALIDATION: [
                "That sounds really difficult. Your feelings make complete sense.",
                "Anyone in your situation would feel {emotion}. That's completely understandable.",
                "You're showing real strength by talking about this."
            ],
            TherapeuticTechnique.PSYCHOEDUCATION: [
                "What you're experiencing is actually quite common with {condition}.",
                "Let me explain how {technique} can help with {concern}.",
                "Understanding {concept} can help us work together more effectively."
            ],
            TherapeuticTechnique.SAFETY_PLANNING: [
                "Let's work together to create a plan for when you feel unsafe.",
                "Who are the people you can reach out to when things get difficult?",
                "What are some strategies that have helped you feel safer before?"
            ]
        }

        # Get templates for this technique
        templates = technique_templates.get(technique, [
            "How are you feeling about {concern}?",
            "What thoughts come up for you around {concern}?",
            "Let's explore {concern} together."
        ])

        # Select random template
        template = random.choice(templates)

        # Fill in placeholders
        concern = random.choice(client_scenario.clinical_formulation.dsm5_considerations)
        emotion = random.choice(["overwhelmed", "anxious", "sad", "frustrated", "confused"])
        condition = concern.split()[0] if concern else "this situation"
        concept = structure_element.lower()

        return template.format(
            concern=concern.lower(),
            emotion=emotion,
            condition=condition.lower(),
            technique=protocol.name.lower(),
            concept=concept
        )

    def _generate_client_response(
        self,
        client_scenario: ClientScenario,
        protocol: InterventionProtocol,
        exchange_index: int
    ) -> str:
        """Generate realistic client response based on scenario and protocol."""

        # Response patterns based on attachment style
        attachment_style = client_scenario.clinical_formulation.attachment_style

        if attachment_style and "secure" in attachment_style.lower():
            responses = [
                "I appreciate you asking about that. It's been on my mind a lot.",
                "That makes sense. I hadn't thought about it that way before.",
                "I feel comfortable sharing this with you."
            ]
        elif attachment_style and "avoidant" in attachment_style.lower():
            responses = [
                "I guess that's part of it, but I'm not sure it's that important.",
                "I don't really like talking about feelings much.",
                "Can we focus on practical solutions instead?"
            ]
        elif attachment_style and "anxious" in attachment_style.lower():
            responses = [
                "I'm worried you think I'm being too much. Am I talking too much?",
                "Do you really think this will help? I've tried so many things before.",
                "I just want to make sure I'm doing this right."
            ]
        else:
            responses = [
                "I'm not sure how to answer that.",
                "This is harder than I thought it would be.",
                "I want to get better, but I don't know if I can."
            ]

        # Add severity-based modifications
        if client_scenario.severity_level == SeverityLevel.SEVERE:
            responses.extend([
                "Everything feels overwhelming right now.",
                "I don't know if I can handle this.",
                "Sometimes I wonder if things will ever get better."
            ])

        return random.choice(responses)

    def _determine_session_goals(self, protocol: InterventionProtocol, session_phase: str) -> list[str]:
        """Determine therapeutic goals for this session phase."""
        if session_phase == "opening":
            return ["Establish rapport", "Assess current state", "Set session agenda"]
        if session_phase == "working":
            return protocol.expected_outcomes[:2]  # Focus on primary outcomes
        # closing
        return ["Summarize progress", "Assign homework", "Plan next session"]

    def _calculate_intervention_fidelity(
        self,
        exchanges: list[dict[str, Any]],
        protocol: InterventionProtocol
    ) -> float:
        """Calculate how well the conversation adheres to protocol."""

        # Count protocol-adherent exchanges
        adherent_exchanges = sum(
            1 for exchange in exchanges
            if exchange.get("protocol_adherence", False)
        )

        # Count technique usage
        techniques_used = set()
        for exchange in exchanges:
            if "technique_used" in exchange:
                techniques_used.add(exchange["technique_used"])

        technique_coverage = len(techniques_used) / len(protocol.key_techniques)

        # Calculate overall fidelity
        exchange_fidelity = adherent_exchanges / len(exchanges) if exchanges else 0
        fidelity_score = (exchange_fidelity + technique_coverage) / 2

        return round(fidelity_score, 2)

    def _generate_clinical_notes(
        self,
        exchanges: list[dict[str, Any]],
        protocol: InterventionProtocol,
        client_scenario: ClientScenario
    ) -> list[str]:
        """Generate clinical notes for the session."""
        notes = []

        # Protocol adherence note
        notes.append(f"Session conducted following {protocol.name} protocol")

        # Client engagement note
        client_exchanges = [e for e in exchanges if e.get("speaker") == "client"]
        if client_exchanges:
            avg_engagement = sum(
                {"high": 3, "medium": 2, "low": 1}.get(e.get("engagement_level", "medium"), 2)
                for e in client_exchanges
            ) / len(client_exchanges)

            engagement_level = "high" if avg_engagement > 2.5 else "medium" if avg_engagement > 1.5 else "low"
            notes.append(f"Client demonstrated {engagement_level} engagement throughout session")

        # Technique utilization note
        techniques_used = [e.get("technique_used") for e in exchanges if e.get("technique_used")]
        if techniques_used:
            notes.append(f"Therapeutic techniques utilized: {', '.join(set(techniques_used))}")

        return notes

    def _assign_homework(self, protocol: InterventionProtocol, session_phase: str) -> str | None:
        """Assign homework based on protocol and session phase."""
        if session_phase == "closing" and protocol.homework_assignments:
            return random.choice(protocol.homework_assignments)
        return None

    def _plan_next_session(
        self,
        protocol: InterventionProtocol,
        session_phase: str,
        exchanges: list[dict[str, Any]]
    ) -> str | None:
        """Plan next session based on current progress."""
        if session_phase == "closing":
            structure_elements = protocol.session_structure
            current_index = len([e for e in exchanges if e.get("structure_element")])

            if current_index < len(structure_elements):
                next_element = structure_elements[current_index]
                return f"Continue with {next_element.lower()} in next session"
            return "Review progress and consider transition to next phase of treatment"
        return None

    def _determine_client_engagement(
        self,
        client_scenario: ClientScenario,
        protocol: InterventionProtocol
    ) -> str:
        """Determine client engagement level based on scenario factors."""

        # Base engagement on attachment style
        attachment_style = client_scenario.clinical_formulation.attachment_style

        if attachment_style and "secure" in attachment_style.lower():
            return random.choice(["high", "medium"])
        if attachment_style and "avoidant" in attachment_style.lower():
            return random.choice(["low", "medium"])
        if attachment_style and "anxious" in attachment_style.lower():
            return random.choice(["medium", "high"])
        return "medium"

    def _determine_emotional_state(self, client_scenario: ClientScenario, exchange_index: int) -> str:
        """Determine client emotional state for this exchange."""

        # Emotional progression through session
        if exchange_index == 0:
            return random.choice(["anxious", "guarded", "hopeful"])
        if exchange_index < 3:
            return random.choice(["opening_up", "cautious", "engaged"])
        return random.choice(["reflective", "processing", "insightful"])

    def export_intervention_conversations(
        self,
        conversations: list[ConversationIntervention],
        output_file: str
    ) -> dict[str, Any]:
        """Export intervention conversations to JSON file."""

        # Convert conversations to JSON-serializable format
        serializable_conversations = []
        for conv in conversations:
            conv_dict = asdict(conv)
            # Convert enum values to strings
            self._convert_enums_to_strings(conv_dict)
            serializable_conversations.append(conv_dict)

        export_data = {
            "metadata": {
                "total_conversations": len(conversations),
                "export_timestamp": datetime.now().isoformat(),
                "protocols_used": list({conv.intervention_selection.protocol.id for conv in conversations}),
                "average_fidelity_score": sum(conv.intervention_fidelity_score for conv in conversations) / len(conversations) if conversations else 0
            },
            "conversations": serializable_conversations
        }

        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(export_data, f, indent=2, ensure_ascii=False)

        return {
            "exported_conversations": len(conversations),
            "output_file": output_file,
            "protocols_covered": len(export_data["metadata"]["protocols_used"]),
            "average_fidelity": round(export_data["metadata"]["average_fidelity_score"], 2)
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
