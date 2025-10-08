"""
Edge case integrator for therapeutic training scenarios.
Integrates challenging and edge case scenarios for comprehensive AI training.
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any

from conversation_schema import Conversation, Message
from logger import get_logger

logger = get_logger(__name__)


class EdgeCaseType(Enum):
    """Types of edge cases in therapeutic scenarios."""
    CRISIS_INTERVENTION = "crisis_intervention"
    ETHICAL_DILEMMA = "ethical_dilemma"
    BOUNDARY_VIOLATION = "boundary_violation"
    CULTURAL_SENSITIVITY = "cultural_sensitivity"
    DUAL_RELATIONSHIP = "dual_relationship"
    CONFIDENTIALITY_BREACH = "confidentiality_breach"
    SUICIDAL_IDEATION = "suicidal_ideation"
    SUBSTANCE_ABUSE = "substance_abuse"
    DOMESTIC_VIOLENCE = "domestic_violence"
    CHILD_ABUSE = "child_abuse"
    TREATMENT_RESISTANCE = "treatment_resistance"
    THERAPEUTIC_RUPTURE = "therapeutic_rupture"


class SeverityLevel(Enum):
    """Severity levels for edge cases."""
    LOW = "low"
    MODERATE = "moderate"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class EdgeCaseScenario:
    """Represents an edge case therapeutic scenario."""
    scenario_id: str
    edge_case_type: EdgeCaseType
    severity_level: SeverityLevel
    description: str
    therapeutic_challenges: list[str]
    required_interventions: list[str]
    ethical_considerations: list[str]
    safety_protocols: list[str]
    learning_objectives: list[str]
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class IntegratedEdgeCases:
    """Result of edge case integration."""
    conversations: list[Conversation]
    scenarios: list[EdgeCaseScenario]
    integration_stats: dict[str, Any]
    quality_metrics: dict[str, float]


class EdgeCaseIntegrator:
    """
    Integrates edge case scenarios for comprehensive therapeutic AI training.

    Handles challenging scenarios including crisis intervention, ethical dilemmas,
    boundary issues, and other complex therapeutic situations.
    """

    def __init__(self):
        """Initialize the edge case integrator."""
        self.logger = get_logger(__name__)

        # Edge case patterns and keywords
        self.edge_case_patterns = {
            EdgeCaseType.CRISIS_INTERVENTION: [
                "crisis", "emergency", "urgent", "immediate danger", "suicide",
                "self-harm", "psychotic episode", "manic episode"
            ],
            EdgeCaseType.ETHICAL_DILEMMA: [
                "ethical", "moral dilemma", "conflicting values", "duty to warn",
                "informed consent", "competency"
            ],
            EdgeCaseType.BOUNDARY_VIOLATION: [
                "boundary", "inappropriate", "dual relationship", "gift",
                "personal disclosure", "physical contact"
            ],
            EdgeCaseType.CULTURAL_SENSITIVITY: [
                "cultural", "religious", "ethnic", "minority", "discrimination",
                "bias", "cultural competence"
            ],
            EdgeCaseType.CONFIDENTIALITY_BREACH: [
                "confidentiality", "privacy", "disclosure", "third party",
                "mandated reporting", "court order"
            ],
            EdgeCaseType.TREATMENT_RESISTANCE: [
                "resistant", "non-compliant", "missed appointments", "dropout",
                "ambivalent", "defensive"
            ]
        }

        # Quality thresholds
        self.quality_thresholds = {
            "therapeutic_accuracy": 0.9,  # Higher standard for edge cases
            "ethical_compliance": 0.95,
            "safety_awareness": 0.9,
            "learning_value": 0.8
        }

        # Safety keywords that require special handling
        self.safety_keywords = {
            "suicide", "self-harm", "homicide", "abuse", "violence",
            "danger", "threat", "weapon", "emergency"
        }

        self.logger.info("EdgeCaseIntegrator initialized")

    def integrate_edge_cases(self, data_sources: list[dict[str, Any]]) -> IntegratedEdgeCases:
        """
        Integrate edge cases from multiple data sources.

        Args:
            data_sources: List of data source configurations

        Returns:
            IntegratedEdgeCases with processed scenarios and conversations
        """
        self.logger.info(f"Integrating edge cases from {len(data_sources)} sources")

        all_conversations = []
        all_scenarios = []
        source_stats = {}

        for source_config in data_sources:
            try:
                source_name = source_config.get("name", "unknown")
                source_type = source_config.get("type", "generic")

                self.logger.info(f"Processing edge case source: {source_name}")

                # Process based on source type
                if source_type == "crisis_scenarios":
                    conversations, scenarios = self._process_crisis_scenarios(source_config)
                elif source_type == "ethical_dilemmas":
                    conversations, scenarios = self._process_ethical_dilemmas(source_config)
                elif source_type == "boundary_cases":
                    conversations, scenarios = self._process_boundary_cases(source_config)
                else:
                    conversations, scenarios = self._process_generic_edge_cases(source_config)

                all_conversations.extend(conversations)
                all_scenarios.extend(scenarios)

                source_stats[source_name] = {
                    "conversations": len(conversations),
                    "scenarios": len(scenarios),
                    "source_type": source_type
                }

            except Exception as e:
                self.logger.error(f"Error processing edge case source {source_config}: {e}")

        # Calculate quality metrics
        quality_metrics = self._calculate_quality_metrics(all_conversations, all_scenarios)

        integration_stats = {
            "total_conversations": len(all_conversations),
            "total_scenarios": len(all_scenarios),
            "sources_processed": len(data_sources),
            "source_breakdown": source_stats,
            "edge_case_distribution": self._get_edge_case_distribution(all_scenarios),
            "severity_distribution": self._get_severity_distribution(all_scenarios),
            "processed_at": datetime.now().isoformat()
        }

        self.logger.info(f"Integrated {len(all_conversations)} conversations and {len(all_scenarios)} scenarios")

        return IntegratedEdgeCases(
            conversations=all_conversations,
            scenarios=all_scenarios,
            integration_stats=integration_stats,
            quality_metrics=quality_metrics
        )

    def _process_crisis_scenarios(self, source_config: dict[str, Any]) -> tuple[list[Conversation], list[EdgeCaseScenario]]:
        """Process crisis intervention scenarios."""
        conversations = []
        scenarios = []

        # Example crisis scenarios
        crisis_templates = [
            {
                "description": "Client expressing active suicidal ideation",
                "severity": SeverityLevel.CRITICAL,
                "interventions": ["immediate safety assessment", "crisis intervention", "safety planning"],
                "safety_protocols": ["assess immediate danger", "remove means", "emergency contact"]
            },
            {
                "description": "Client in acute psychotic episode",
                "severity": SeverityLevel.HIGH,
                "interventions": ["reality testing", "grounding techniques", "medical referral"],
                "safety_protocols": ["assess reality contact", "ensure safety", "coordinate care"]
            }
        ]

        for i, template in enumerate(crisis_templates):
            scenario = EdgeCaseScenario(
                scenario_id=f"crisis_{i+1}",
                edge_case_type=EdgeCaseType.CRISIS_INTERVENTION,
                severity_level=template["severity"],
                description=template["description"],
                therapeutic_challenges=["immediate safety", "crisis stabilization"],
                required_interventions=template["interventions"],
                ethical_considerations=["duty to protect", "informed consent"],
                safety_protocols=template["safety_protocols"],
                learning_objectives=["crisis assessment", "safety planning", "intervention skills"]
            )
            scenarios.append(scenario)

            # Create corresponding conversation
            conversation = self._create_crisis_conversation(scenario)
            if conversation:
                conversations.append(conversation)

        return conversations, scenarios

    def _process_ethical_dilemmas(self, source_config: dict[str, Any]) -> tuple[list[Conversation], list[EdgeCaseScenario]]:
        """Process ethical dilemma scenarios."""
        conversations = []
        scenarios = []

        ethical_templates = [
            {
                "description": "Conflicting duty to warn vs. confidentiality",
                "severity": SeverityLevel.HIGH,
                "challenges": ["ethical decision-making", "legal compliance"],
                "considerations": ["client autonomy", "duty to protect", "legal requirements"]
            },
            {
                "description": "Dual relationship boundary issues",
                "severity": SeverityLevel.MODERATE,
                "challenges": ["boundary maintenance", "professional ethics"],
                "considerations": ["power dynamics", "client welfare", "professional standards"]
            }
        ]

        for i, template in enumerate(ethical_templates):
            scenario = EdgeCaseScenario(
                scenario_id=f"ethical_{i+1}",
                edge_case_type=EdgeCaseType.ETHICAL_DILEMMA,
                severity_level=template["severity"],
                description=template["description"],
                therapeutic_challenges=template["challenges"],
                required_interventions=["ethical consultation", "supervision", "documentation"],
                ethical_considerations=template["considerations"],
                safety_protocols=["consult ethics code", "seek supervision"],
                learning_objectives=["ethical reasoning", "decision-making", "consultation skills"]
            )
            scenarios.append(scenario)

            conversation = self._create_ethical_conversation(scenario)
            if conversation:
                conversations.append(conversation)

        return conversations, scenarios

    def _process_boundary_cases(self, source_config: dict[str, Any]) -> tuple[list[Conversation], list[EdgeCaseScenario]]:
        """Process boundary violation scenarios."""
        conversations = []
        scenarios = []

        boundary_templates = [
            {
                "description": "Client requesting personal information",
                "severity": SeverityLevel.LOW,
                "challenges": ["boundary setting", "therapeutic relationship"],
                "protocols": ["clarify boundaries", "explore meaning", "maintain frame"]
            },
            {
                "description": "Inappropriate gift from client",
                "severity": SeverityLevel.MODERATE,
                "challenges": ["boundary maintenance", "client feelings"],
                "protocols": ["decline gracefully", "process meaning", "reinforce boundaries"]
            }
        ]

        for i, template in enumerate(boundary_templates):
            scenario = EdgeCaseScenario(
                scenario_id=f"boundary_{i+1}",
                edge_case_type=EdgeCaseType.BOUNDARY_VIOLATION,
                severity_level=template["severity"],
                description=template["description"],
                therapeutic_challenges=template["challenges"],
                required_interventions=["boundary clarification", "processing", "education"],
                ethical_considerations=["professional boundaries", "client welfare"],
                safety_protocols=template["protocols"],
                learning_objectives=["boundary skills", "therapeutic frame", "processing techniques"]
            )
            scenarios.append(scenario)

            conversation = self._create_boundary_conversation(scenario)
            if conversation:
                conversations.append(conversation)

        return conversations, scenarios

    def _process_generic_edge_cases(self, source_config: dict[str, Any]) -> tuple[list[Conversation], list[EdgeCaseScenario]]:
        """Process generic edge case data."""
        conversations = []
        scenarios = []

        # Create basic edge case scenario
        scenario = EdgeCaseScenario(
            scenario_id="generic_edge_case",
            edge_case_type=EdgeCaseType.TREATMENT_RESISTANCE,
            severity_level=SeverityLevel.MODERATE,
            description="Generic challenging therapeutic scenario",
            therapeutic_challenges=["engagement", "resistance"],
            required_interventions=["motivational interviewing", "rapport building"],
            ethical_considerations=["client autonomy", "beneficence"],
            safety_protocols=["assess motivation", "explore resistance"],
            learning_objectives=["engagement skills", "resistance handling"]
        )
        scenarios.append(scenario)

        return conversations, scenarios

    def _create_crisis_conversation(self, scenario: EdgeCaseScenario) -> Conversation | None:
        """Create a conversation for a crisis scenario."""
        try:
            messages = [
                Message(
                    role="user",
                    content="I'm having thoughts about ending my life. I don't see any way out of this pain.",
                    timestamp=datetime.now()
                ),
                Message(
                    role="assistant",
                    content="I'm very concerned about what you're telling me. Your safety is my top priority right now. Can you tell me if you have any specific plans to hurt yourself?",
                    timestamp=datetime.now()
                ),
                Message(
                    role="user",
                    content="I've been thinking about it, but I don't have a specific plan yet.",
                    timestamp=datetime.now()
                ),
                Message(
                    role="assistant",
                    content="Thank you for being honest with me. That takes courage. Let's work together to keep you safe. Do you have anyone you can stay with tonight, or should we consider other safety options?",
                    timestamp=datetime.now()
                )
            ]

            return Conversation(
                id=f"crisis_conv_{scenario.scenario_id}",
                messages=messages,
                title=f"Crisis Intervention: {scenario.description}",
                metadata={
                    "edge_case_type": scenario.edge_case_type.value,
                    "severity_level": scenario.severity_level.value,
                    "scenario_id": scenario.scenario_id,
                    "safety_critical": True
                },
                tags=["edge_case", "crisis", "safety_critical"]
            )

        except Exception as e:
            self.logger.warning(f"Could not create crisis conversation: {e}")
            return None

    def _create_ethical_conversation(self, scenario: EdgeCaseScenario) -> Conversation | None:
        """Create a conversation for an ethical dilemma."""
        try:
            messages = [
                Message(
                    role="user",
                    content="I need to tell you something, but I'm worried about confidentiality. My partner doesn't know I'm here.",
                    timestamp=datetime.now()
                ),
                Message(
                    role="assistant",
                    content="I understand your concern about confidentiality. Let me explain our confidentiality policy and the few exceptions where I might need to share information to keep you or others safe.",
                    timestamp=datetime.now()
                )
            ]

            return Conversation(
                id=f"ethical_conv_{scenario.scenario_id}",
                messages=messages,
                title=f"Ethical Dilemma: {scenario.description}",
                metadata={
                    "edge_case_type": scenario.edge_case_type.value,
                    "severity_level": scenario.severity_level.value,
                    "scenario_id": scenario.scenario_id,
                    "ethical_complexity": True
                },
                tags=["edge_case", "ethical", "confidentiality"]
            )

        except Exception as e:
            self.logger.warning(f"Could not create ethical conversation: {e}")
            return None

    def _create_boundary_conversation(self, scenario: EdgeCaseScenario) -> Conversation | None:
        """Create a conversation for a boundary scenario."""
        try:
            messages = [
                Message(
                    role="user",
                    content="I brought you a small gift to thank you for all your help. I hope that's okay.",
                    timestamp=datetime.now()
                ),
                Message(
                    role="assistant",
                    content="I really appreciate the thought behind this gesture. However, I'm not able to accept gifts as it's important to maintain clear professional boundaries. Can we talk about what this gift means to you?",
                    timestamp=datetime.now()
                )
            ]

            return Conversation(
                id=f"boundary_conv_{scenario.scenario_id}",
                messages=messages,
                title=f"Boundary Issue: {scenario.description}",
                metadata={
                    "edge_case_type": scenario.edge_case_type.value,
                    "severity_level": scenario.severity_level.value,
                    "scenario_id": scenario.scenario_id,
                    "boundary_focus": True
                },
                tags=["edge_case", "boundary", "professional"]
            )

        except Exception as e:
            self.logger.warning(f"Could not create boundary conversation: {e}")
            return None

    def _calculate_quality_metrics(self, conversations: list[Conversation], scenarios: list[EdgeCaseScenario]) -> dict[str, float]:
        """Calculate quality metrics for integrated edge cases."""
        if not conversations or not scenarios:
            return {"error": "No data to analyze"}

        # Safety coverage
        safety_critical_count = sum(1 for conv in conversations if conv.metadata.get("safety_critical", False))
        safety_coverage = safety_critical_count / len(conversations)

        # Ethical complexity coverage
        ethical_count = sum(1 for conv in conversations if conv.metadata.get("ethical_complexity", False))
        ethical_coverage = ethical_count / len(conversations)

        # Severity distribution balance
        severity_counts = {}
        for scenario in scenarios:
            severity = scenario.severity_level.value
            severity_counts[severity] = severity_counts.get(severity, 0) + 1

        severity_balance = 1.0 - abs(0.25 - (severity_counts.get("critical", 0) / len(scenarios)))

        return {
            "safety_coverage": safety_coverage,
            "ethical_coverage": ethical_coverage,
            "severity_balance": severity_balance,
            "scenario_conversation_ratio": len(conversations) / len(scenarios),
            "overall_quality": (safety_coverage + ethical_coverage + severity_balance) / 3
        }

    def _get_edge_case_distribution(self, scenarios: list[EdgeCaseScenario]) -> dict[str, int]:
        """Get distribution of edge case types."""
        distribution = {}
        for scenario in scenarios:
            case_type = scenario.edge_case_type.value
            distribution[case_type] = distribution.get(case_type, 0) + 1
        return distribution

    def _get_severity_distribution(self, scenarios: list[EdgeCaseScenario]) -> dict[str, int]:
        """Get distribution of severity levels."""
        distribution = {}
        for scenario in scenarios:
            severity = scenario.severity_level.value
            distribution[severity] = distribution.get(severity, 0) + 1
        return distribution


def validate_edge_case_integrator():
    """Validate the EdgeCaseIntegrator functionality."""
    try:
        integrator = EdgeCaseIntegrator()

        # Test basic functionality
        assert hasattr(integrator, "integrate_edge_cases")
        assert hasattr(integrator, "edge_case_patterns")
        assert len(integrator.edge_case_patterns) > 0

        return True

    except Exception:
        return False


if __name__ == "__main__":
    # Run validation
    if validate_edge_case_integrator():
        pass
    else:
        pass
