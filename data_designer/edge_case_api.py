"""
API interface for edge case generation that integrates with the scenario generation system.

This module provides a Python API that can be called from TypeScript/JavaScript endpoints
to generate edge case scenarios using NeMo Data Designer.
"""

import logging
from typing import Any, Optional
import json

from ai.data_designer.edge_case_generator import EdgeCaseGenerator, EdgeCaseType
from ai.data_designer.service import NeMoDataDesignerService

logger = logging.getLogger(__name__)


class EdgeCaseAPI:
    """API for generating edge case scenarios."""

    def __init__(self, generator: Optional[EdgeCaseGenerator] = None):
        """
        Initialize the edge case API.

        Args:
            generator: EdgeCaseGenerator instance. If None, creates new.
        """
        self.generator = generator or EdgeCaseGenerator()

    def generate_scenario(
        self,
        edge_case_type: str,
        num_samples: int = 10,
        difficulty_level: str = "advanced",
    ) -> dict[str, Any]:
        """
        Generate edge case scenarios for a specific type.

        Args:
            edge_case_type: Type of edge case (crisis, cultural_complexity, etc.)
            num_samples: Number of scenarios to generate
            difficulty_level: Difficulty level (beginner, intermediate, advanced)

        Returns:
            Dictionary with scenarios formatted for the scenario generation API
        """
        try:
            # Convert string to EdgeCaseType enum
            case_type = EdgeCaseType(edge_case_type.lower())

            # Generate dataset
            result = self.generator.generate_edge_case_dataset(
                edge_case_type=case_type,
                num_samples=num_samples,
                difficulty_level=difficulty_level,
            )

            # Format for scenario API
            scenarios = self._format_for_scenario_api(result)

            return {
                "scenarios": scenarios,
                "metadata": {
                    "edge_case_type": edge_case_type,
                    "difficulty_level": difficulty_level,
                    "num_scenarios": len(scenarios),
                    "source": "nemo_data_designer",
                },
            }
        except ValueError as e:
            logger.error(f"Invalid edge case type: {edge_case_type}")
            raise ValueError(f"Invalid edge case type: {edge_case_type}. Valid types: {[et.value for et in EdgeCaseType]}") from e
        except Exception as e:
            logger.error(f"Failed to generate edge case scenarios: {e}")
            raise

    def generate_multi_type_scenarios(
        self,
        edge_case_types: list[str],
        num_samples_per_type: int = 5,
        difficulty_level: str = "advanced",
    ) -> dict[str, Any]:
        """
        Generate scenarios for multiple edge case types.

        Args:
            edge_case_types: List of edge case types
            num_samples_per_type: Number of scenarios per type
            difficulty_level: Difficulty level

        Returns:
            Dictionary with all scenarios
        """
        try:
            # Convert strings to EdgeCaseType enums
            case_types = [EdgeCaseType(et.lower()) for et in edge_case_types]

            # Generate dataset
            result = self.generator.generate_multi_edge_case_dataset(
                edge_case_types=case_types,
                num_samples_per_type=num_samples_per_type,
                difficulty_level=difficulty_level,
            )

            # Format for scenario API
            scenarios = self._format_for_scenario_api(result)

            return {
                "scenarios": scenarios,
                "metadata": {
                    "edge_case_types": edge_case_types,
                    "difficulty_level": difficulty_level,
                    "num_scenarios": len(scenarios),
                    "source": "nemo_data_designer",
                },
            }
        except ValueError as e:
            logger.error(f"Invalid edge case type in list: {e}")
            raise
        except Exception as e:
            logger.error(f"Failed to generate multi-type scenarios: {e}")
            raise

    def _format_for_scenario_api(self, result: dict[str, Any]) -> list[dict[str, Any]]:
        """
        Format edge case data for the scenario generation API format.

        Args:
            result: Result from edge case generator

        Returns:
            List of formatted scenarios
        """
        scenarios = []
        data = result.get("data", [])

        if not isinstance(data, list):
            logger.warning("Data is not a list, attempting to convert")
            data = [data] if data else []

        for idx, record in enumerate(data):
            if not isinstance(record, dict):
                continue

            # Extract edge case type
            edge_case_type = record.get("edge_case_type", result.get("edge_case_type", "unknown"))

            # Format scenario
            scenario = {
                "id": f"edge_case_{edge_case_type}_{idx+1:04d}",
                "title": self._generate_title(record, edge_case_type),
                "description": self._generate_description(record, edge_case_type),
                "edge_case_type": edge_case_type,
                "difficulty_level": result.get("difficulty_level", "advanced"),
                "clientProfile": self._extract_client_profile(record),
                "edgeCaseDetails": self._extract_edge_case_details(record, edge_case_type),
                "challengeLevel": self._determine_challenge_level(record, edge_case_type),
                "estimatedDuration": self._estimate_duration(edge_case_type),
                "metadata": {
                    "generatedAt": result.get("metadata", {}).get("generated_at", ""),
                    "source": "nemo_data_designer_edge_case_generator",
                },
            }
            scenarios.append(scenario)

        return scenarios

    def _generate_title(self, record: dict[str, Any], edge_case_type: str) -> str:
        """Generate a title for the scenario."""
        case_type_map = {
            "crisis": "Crisis Intervention Scenario",
            "cultural_complexity": "Cultural Complexity Scenario",
            "comorbidity": "Complex Comorbidity Scenario",
            "boundary_violation": "Boundary Management Scenario",
            "trauma_disclosure": "Trauma Disclosure Scenario",
            "substance_abuse": "Substance Use Scenario",
            "ethical_dilemma": "Ethical Dilemma Scenario",
            "rare_diagnosis": "Rare Diagnosis Scenario",
            "multi_generational": "Multi-Generational Family Scenario",
            "systemic_oppression": "Systemic Oppression Scenario",
        }
        base_title = case_type_map.get(edge_case_type, "Edge Case Scenario")
        age = record.get("age", "Unknown")
        return f"{base_title}: Age {age}"

    def _generate_description(self, record: dict[str, Any], edge_case_type: str) -> str:
        """Generate a description for the scenario."""
        descriptions = {
            "crisis": f"Crisis scenario involving {record.get('crisis_type', 'unknown crisis type')} with severity level {record.get('crisis_severity', 'unknown')}",
            "cultural_complexity": f"Cultural complexity scenario with {record.get('cultural_barriers', 'various barriers')} requiring {record.get('cultural_competence_required', 'moderate')} cultural competence",
            "comorbidity": f"Complex case with {record.get('primary_diagnosis', 'primary diagnosis')} and comorbid {record.get('comorbid_conditions', 'conditions')}",
            "boundary_violation": f"Boundary management scenario involving {record.get('boundary_violation_type', 'boundary issue')} with {record.get('violation_severity', 'moderate')} severity",
            "trauma_disclosure": f"Trauma disclosure scenario involving {record.get('trauma_type', 'trauma')} from {record.get('trauma_recency', 'the past')}",
            "substance_abuse": f"Substance use scenario involving {record.get('substance_type', 'substance')} with {record.get('use_frequency', 'regular')} use frequency",
            "ethical_dilemma": f"Ethical dilemma scenario involving {record.get('ethical_dilemma_type', 'ethical issue')} requiring {record.get('ethical_consultation_urgency', 'moderate')} consultation",
            "rare_diagnosis": f"Rare diagnosis scenario: {record.get('rare_diagnosis', 'rare condition')} with {record.get('diagnostic_complexity', 'high')} complexity",
            "multi_generational": f"Multi-generational family scenario with {record.get('family_structure', 'family structure')} and {record.get('generational_conflicts', 'conflicts')}",
            "systemic_oppression": f"Systemic oppression scenario involving {record.get('oppression_type', 'oppression')} with {record.get('systemic_barriers', 'barriers')}",
        }
        return descriptions.get(edge_case_type, "Edge case scenario requiring specialized therapeutic intervention")

    def _extract_client_profile(self, record: dict[str, Any]) -> dict[str, Any]:
        """Extract client profile information from record."""
        return {
            "age": record.get("age", 35),
            "gender": record.get("gender", "unknown"),
            "ethnicity": record.get("ethnicity", "unknown"),
            "background": f"{record.get('ethnicity', 'Unknown')} individual, age {record.get('age', 'unknown')}",
        }

    def _extract_edge_case_details(self, record: dict[str, Any], edge_case_type: str) -> dict[str, Any]:
        """Extract edge case-specific details."""
        details = {}

        # Add type-specific details
        if edge_case_type == "crisis":
            details.update({
                "crisis_type": record.get("crisis_type"),
                "crisis_severity": record.get("crisis_severity"),
                "suicidal_ideation_present": record.get("suicidal_ideation_present"),
                "immediate_risk_score": record.get("immediate_risk_score"),
            })
        elif edge_case_type == "cultural_complexity":
            details.update({
                "primary_language": record.get("primary_language"),
                "immigration_status": record.get("immigration_status"),
                "cultural_barriers": record.get("cultural_barriers"),
                "cultural_competence_required": record.get("cultural_competence_required"),
            })
        elif edge_case_type == "comorbidity":
            details.update({
                "primary_diagnosis": record.get("primary_diagnosis"),
                "comorbid_conditions": record.get("comorbid_conditions"),
                "complexity_score": record.get("complexity_score"),
            })

        # Add common details
        details.update({
            "intervention_complexity": record.get("intervention_complexity", "high"),
            "supervision_required": record.get("supervision_required", "yes"),
            "training_priority": record.get("training_priority", "high"),
        })

        return details

    def _determine_challenge_level(self, record: dict[str, Any], edge_case_type: str) -> str:
        """Determine the challenge level based on record data."""
        # Most edge cases are advanced by nature
        complexity = record.get("intervention_complexity", "high")
        if complexity in ["very_high", "high"]:
            return "advanced"
        elif complexity == "moderate":
            return "intermediate"
        else:
            return "beginner"

    def _estimate_duration(self, edge_case_type: str) -> str:
        """Estimate session duration based on edge case type."""
        duration_map = {
            "crisis": "60-90 minutes",
            "cultural_complexity": "60-75 minutes",
            "comorbidity": "75-90 minutes",
            "boundary_violation": "60-75 minutes",
            "trauma_disclosure": "75-90 minutes",
            "substance_abuse": "60-75 minutes",
            "ethical_dilemma": "60-90 minutes",
            "rare_diagnosis": "75-90 minutes",
            "multi_generational": "90-120 minutes",
            "systemic_oppression": "60-90 minutes",
        }
        return duration_map.get(edge_case_type, "60-75 minutes")

