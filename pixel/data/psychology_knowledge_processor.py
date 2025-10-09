"""
psychology_knowledge_processor.py

Extracts DSM-5 diagnostic criteria and related clinical knowledge for Pixel's psychology knowledge processing pipeline.

The actual DSM-5 data source is not present in the workspace. This module is scaffolded for future integration.
"""

from typing import Any

from ai.pixel.data.therapeutic_conversation_schema import (
    ClinicalContext,
    ConversationRole,
    ConversationTurn,
    TherapeuticConversation,
    TherapeuticModality,
)


class TherapeuticConversationConverter:
    """
    Converts psychology knowledge (DSM-5, PDM-2, etc.) into standardized TherapeuticConversation format.
    """

    @staticmethod
    def dsm5_to_conversations(dsm5_criteria: list[dict]) -> list[TherapeuticConversation]:
        """
        Converts DSM-5 criteria into a list of TherapeuticConversation objects.

"""
        conversations = []
        for disorder in dsm5_criteria:
            disorder_name = disorder.get("name") or disorder.get("disorder")
            if not disorder_name:
                continue
            context = ClinicalContext(
                dsm5_categories=[disorder_name],
                presenting_concerns=[],
                goals=[],
                notes=f"DSM-5 disorder: {disorder_name}"
            )
            turns = []
            for _idx, symptom in enumerate(disorder.get("criteria", []), 1):
                turns.append(ConversationTurn(
                    role=ConversationRole.CLIENT,
                    content=f"I have experienced: {symptom}",
                    clinical_rationale=None
                ))
                turns.append(ConversationTurn(
                    role=ConversationRole.THERAPIST,
                    content=f"In the past {disorder.get('duration', 'period')}, have you experienced: {symptom}?",
                    clinical_rationale="Assessing DSM-5 diagnostic criteria"
                ))
            conversation = TherapeuticConversation(
                title=f"Assessment for {disorder_name}",
                modality=TherapeuticModality.OTHER,
                clinical_context=context,
                turns=turns
            )
            conversations.append(conversation)
        return conversations

    @staticmethod
    def pdm2_to_conversations(pdm2_frameworks: list[dict]) -> list[TherapeuticConversation]:
        """
        Converts PDM-2 frameworks into a list of TherapeuticConversation objects.
        """
        conversations = []
        for fw in pdm2_frameworks:
            name = fw.get("framework")
            features = fw.get("features", [])
            attachment = fw.get("attachment_style", "")
            context = ClinicalContext(
                pdm2_frameworks=[name] if name else [],
                presenting_concerns=features,
                goals=[],
                notes=f"PDM-2 framework: {name}, Attachment: {attachment}"
            )
            turns = []
            for feature in features:
                turns.append(ConversationTurn(
                    role=ConversationRole.CLIENT,
                    content=f"I often experience: {feature}",
                    clinical_rationale=None
                ))
                turns.append(ConversationTurn(
                    role=ConversationRole.THERAPIST,
                    content=f"Can you tell me more about: {feature}?",
                    clinical_rationale="Exploring psychodynamic features"
                ))
            conversation = TherapeuticConversation(
                title=f"Psychodynamic Assessment: {name}",
                modality=TherapeuticModality.PSYCHODYNAMIC,
                clinical_context=context,
                turns=turns
            )
            conversations.append(conversation)
        return conversations


class DSM5CriteriaExtractor:
    """
    Placeholder for DSM-5 diagnostic criteria extraction logic.
    When the raw DSM-5 data is available, implement parsing and structuring here.
    """

    def __init__(self, data_path: str | None = None):
        self.data_path = data_path

    def extract_criteria(self) -> list[dict[str, Any]]:
        """
        Extracts and structures DSM-5 diagnostic criteria.

        Returns:
            List of dictionaries, each representing a disorder and its criteria.
        """
        # Placeholder: Replace with actual extraction logic when data is available
        return self._standardize_criteria(self._get_placeholder_criteria())

    def _get_placeholder_criteria(self) -> list[dict[str, Any]]:
        """
        Returns placeholder DSM-5 criteria for demonstration/testing.
        """
        return [
            {
                "disorder": "Major Depressive Disorder",
                "criteria": [
                    "Depressed mood most of the day, nearly every day",
                    "Markedly diminished interest or pleasure in all, or almost all, activities",
                    "Significant weight loss when not dieting or weight gain",
                    "Insomnia or hypersomnia nearly every day",
                    "Psychomotor agitation or retardation nearly every day",
                    "Fatigue or loss of energy nearly every day",
                    "Feelings of worthlessness or excessive guilt",
                    "Diminished ability to think or concentrate",
                    "Recurrent thoughts of death, suicidal ideation",
                ],
                "minimum_criteria": 5,
                "duration": "2 weeks",
                "exclusions": [
                    "Symptoms not attributable to substance or another medical condition",
                    "Symptoms not better explained by another mental disorder"
                ]
            }
            # Add more disorders as needed
        ]

    def _standardize_criteria(self, raw_criteria: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """
        Converts raw DSM-5 criteria into a standardized format for downstream use.

        Args:
            raw_criteria: List of raw disorder criteria dicts.

        Returns:
            List of standardized disorder dicts.
        """
        standardized = []
        for entry in raw_criteria:
            standardized.append({
                "name": entry.get("disorder"),
                "criteria": entry.get("criteria", []),
                "minimum_criteria": entry.get("minimum_criteria"),
                "duration": entry.get("duration"),
                "exclusions": entry.get("exclusions", []),
                "source": "DSM-5"
            })
        return standardized

    def create_symptom_disorder_map(self, criteria: list[dict[str, Any]]) -> dict[str, list[str]]:
        """
        Creates a mapping from symptom/criterion to the list of disorders it appears in.

        Args:
            criteria: List of standardized disorder dicts.

        Returns:
            Dictionary mapping symptom (str) to list of disorder names (str).
        """
        symptom_map: dict[str, list[str]] = {}
        for disorder in criteria:
            disorder_name = disorder.get("name")
            if not isinstance(disorder_name, str) or not disorder_name:
                continue
            for symptom in disorder.get("criteria", []):
                if symptom not in symptom_map:
                    symptom_map[symptom] = []
                symptom_map[symptom].append(disorder_name)
        return symptom_map

    def build_conversation_templates(self, criteria: list[dict[str, Any]]) -> dict[str, list[str]]:
        """
        Builds diagnostic conversation templates (question lists) for each disorder.

        Args:
            criteria: List of standardized disorder dicts.

        Returns:
            Dictionary mapping disorder name to a list of question templates.
        """
        templates: dict[str, list[str]] = {}
        for disorder in criteria:
            name = disorder.get("name")
            if not isinstance(name, str) or not name:
                continue
            questions = []
            for idx, symptom in enumerate(disorder.get("criteria", []), 1):
                questions.append(
                    f"Q{idx}: In the past {disorder.get('duration', 'period')}, have you experienced: {symptom}?"
                )
            templates[name] = questions
        return templates

    def validate_criteria(self, criteria: list[dict[str, Any]]) -> dict[str, Any]:
        """
        Validates the extracted DSM-5 criteria against expected clinical standards.

        Args:
            criteria: List of standardized disorder dicts.

        Returns:
            Dictionary with validation results and any issues found.
        """
        required_fields = {"name", "criteria", "minimum_criteria", "duration"}
        issues = []
        for disorder in criteria:
            missing = required_fields - disorder.keys()
            if missing:
                issues.append({
                    "disorder": disorder.get("name", "<unknown>"),
                    "missing_fields": list(missing)
                })
            if not isinstance(disorder.get("criteria", []), list) or not disorder.get("criteria"):
                issues.append({
                    "disorder": disorder.get("name", "<unknown>"),
                    "problem": "No criteria listed"
                })
            if not isinstance(disorder.get("minimum_criteria", None), int):
                issues.append({
                    "disorder": disorder.get("name", "<unknown>"),
                    "problem": "minimum_criteria is not an integer"
                })
            if not isinstance(disorder.get("duration", None), str):
                issues.append({
                    "disorder": disorder.get("name", "<unknown>"),
                    "problem": "duration is not a string"
                })
        return {
            "total_disorders": len(criteria),
            "issues_found": len(issues),
            "issues": issues,
            "valid": len(issues) == 0
        }


class PDM2FrameworkExtractor:
    """
    Placeholder for PDM-2 psychodynamic framework and attachment style extraction logic.
    When the raw PDM-2 data is available, implement parsing and structuring here.
    """

    def __init__(self, data_path: str | None = None):
        self.data_path = data_path

    def extract_frameworks(self) -> list[dict[str, Any]]:
        """
        Extracts and structures PDM-2 psychodynamic frameworks.

        Returns:
            List of dictionaries, each representing a psychodynamic framework.
        """
        # Placeholder: Replace with actual extraction logic when data is available
        return self._load_frameworks_from_knowledge_base()

    def _load_frameworks_from_knowledge_base(self) -> list[dict[str, Any]]:
        """
        Simulates loading PDM-2 frameworks from a knowledge base or data file.

        Returns:
            List of dictionaries, each representing a psychodynamic framework.
        """
        # Placeholder: Replace with actual file/database loading logic
        return [
            {
                "framework": "Depressive Personality Organization",
                "features": [
                    "Chronic feelings of sadness or emptiness",
                    "Tendency toward self-criticism and guilt",
                    "Difficulty experiencing pleasure",
                    "Interpersonal withdrawal"
                ],
                "defense_mechanisms": [
                    "Introjection",
                    "Idealization of others",
                    "Devaluation of self"
                ],
                "attachment_style": "anxious"
            }
            # Add more frameworks as needed
        ]

    def extract_attachment_styles(self) -> list[dict[str, Any]]:
        """
        Extracts and structures attachment styles and their features.

        Returns:
            List of dictionaries, each representing an attachment style and its features.
        """
        return [
            {
                "style": "secure",
                "features": [
                    "Comfortable with intimacy and autonomy",
                    "Able to form trusting relationships"
                ]
            },
            {
                "style": "anxious",
                "features": [
                    "Preoccupied with relationships",
                    "Fear of abandonment",
                    "Seeks excessive reassurance"
                ]
            },
            {
                "style": "avoidant",
                "features": [
                    "Discomfort with closeness",
                    "Prefers independence",
                    "Suppresses emotional expression"
                ]
            },
            {
                "style": "disorganized",
                "features": [
                    "Lack of clear attachment strategy",
                    "Erratic or contradictory behaviors",
                    "Difficulty trusting others"
                ]
            }
        ]

    def extract_defense_mechanisms(self) -> dict[str, list[dict[str, str]]]:
        """
        Extracts and categorizes defense mechanisms according to PDM-2, providing examples and brief descriptions.

        Returns:
            dict[str, list[dict[str, str]]]: A dictionary mapping defense mechanism categories
            (e.g., 'mature', 'neurotic', 'immature') to lists of mechanisms with their descriptions.
        """
        # Placeholder/mock data for defense mechanisms
        return {
            "mature": [
                {
                    "mechanism": "sublimation",
                    "description": "Channeling unacceptable impulses into socially acceptable activities."
                },
                {
                    "mechanism": "humor",
                    "description": "Using humor to cope with stress or difficult emotions."
                },
                {
                    "mechanism": "suppression",
                    "description": "Consciously deciding to delay paying attention to a thought or feeling."
                },
            ],
            "neurotic": [
                {
                    "mechanism": "intellectualization",
                    "description": "Using logic and reasoning to block out emotional stress."
                },
                {
                    "mechanism": "repression",
                    "description": "Unconsciously blocking unacceptable thoughts or impulses."
                },
                {
                    "mechanism": "displacement",
                    "description": "Shifting emotional impulses to a safer substitute target."
                },
            ],
            "immature": [
                {
                    "mechanism": "projection",
                    "description": "Attributing ones own unacceptable feelings to others."
                },
                {
                    "mechanism": "denial",
                    "description": "Refusing to accept reality or facts."
                },
                {
                    "mechanism": "acting out",
                    "description": "Performing extreme behaviors to express thoughts or feelings."
                }
            ]
        }

    def build_psychodynamic_conversation_templates(self) -> list[dict[str, str]]:
        """
        Builds psychodynamic conversation templates (question lists) for assessment and intervention.

        Returns:
            list[dict[str, str]]: A list of question templates, each with a 'category' and 'question' field.
        """
        # Placeholder logic: In a real implementation, this would use extracted frameworks, attachment styles,
        # and defense mechanisms to generate tailored questions.
        return [
            {
                "category": "attachment",
                "question": "Can you describe how you typically feel when someone close to you is unavailable?"
            },
            {
                "category": "defense_mechanism",
                "question": "When you feel upset, do you notice yourself using humor or distraction to cope?"
            },
            {
                "category": "personality_organization",
                "question": "How do you usually respond to criticism or perceived failure?"
            },
            {
                "category": "mental_functioning",
                "question": "Can you share how you process strong emotions or stress in your daily life?"
            },
            {
                "category": "interpersonal_patterns",
                "question": "What patterns do you notice in your close relationships over time?"
            },
        ]

    def integrate_attachment_theory(self) -> list[dict[str, str]]:
        """
        Integrates attachment theory into therapeutic conversation templates by adapting questions
        or interventions based on the client's attachment style.

        Returns:
            list[dict[str, str]]: A list of adapted conversation templates, each with 'attachment_style' and 'adapted_question'.
        """
        # Placeholder logic: In a real implementation, this would use assessment results to select/adapt interventions.
        return [
            {
                "attachment_style": "secure",
                "adapted_question": "How do you use your strengths in relationships to support others and yourself?"
            },
            {
                "attachment_style": "anxious",
                "adapted_question": "When you feel uncertain in a relationship, what helps you regain a sense of security?"
            },
            {
                "attachment_style": "avoidant",
                "adapted_question": "How do you balance your need for independence with your desire for closeness?"
            },
            {
                "attachment_style": "disorganized",
                "adapted_question": "What situations make it difficult to trust others, and how do you cope in those moments?"
            },
        ]
