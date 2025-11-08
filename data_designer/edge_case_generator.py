"""
Edge Case Generator for Therapeutic Training Scenarios

This module uses NeMo Data Designer to generate challenging, rare, and edge case
scenarios for therapeutic training. It focuses on scenarios that therapists may
encounter infrequently but need to be prepared for.
"""

import logging
from typing import Any, Optional
from enum import Enum

from nemo_microservices.data_designer.essentials import (
    DataDesignerConfigBuilder,
    SamplerColumnConfig,
    CategorySamplerParams,
    UniformSamplerParams,
    GaussianSamplerParams,
)

from ai.data_designer.service import NeMoDataDesignerService
from ai.data_designer.config import DataDesignerConfig

logger = logging.getLogger(__name__)


class EdgeCaseType(str, Enum):
    """Types of edge cases that can be generated."""
    CRISIS = "crisis"
    CULTURAL_COMPLEXITY = "cultural_complexity"
    COMORBIDITY = "comorbidity"
    BOUNDARY_VIOLATION = "boundary_violation"
    TRAUMA_DISCLOSURE = "trauma_disclosure"
    SUBSTANCE_ABUSE = "substance_abuse"
    ETHICAL_DILEMMA = "ethical_dilemma"
    RARE_DIAGNOSIS = "rare_diagnosis"
    MULTI_GENERATIONAL = "multi_generational"
    SYSTEMIC_OPPRESSION = "systemic_oppression"


class EdgeCaseGenerator:
    """Generator for edge case therapeutic scenarios using NeMo Data Designer."""

    def __init__(self, designer_service: Optional[NeMoDataDesignerService] = None):
        """
        Initialize the edge case generator.

        Args:
            designer_service: NeMo Data Designer service instance. If None, creates new.
        """
        self.designer_service = designer_service or NeMoDataDesignerService()

    def generate_edge_case_dataset(
        self,
        edge_case_type: EdgeCaseType,
        num_samples: int = 100,
        difficulty_level: str = "advanced",
    ) -> dict[str, Any]:
        """
        Generate a dataset of edge case scenarios.

        Args:
            edge_case_type: Type of edge case to generate
            num_samples: Number of edge case scenarios to generate
            difficulty_level: Difficulty level (beginner, intermediate, advanced)

        Returns:
            Dictionary with edge case dataset and metadata
        """
        logger.info(
            f"Generating {num_samples} {edge_case_type.value} edge cases "
            f"at {difficulty_level} difficulty level"
        )

        config_builder = DataDesignerConfigBuilder()

        # Add base demographic columns
        self._add_demographic_columns(config_builder)

        # Add edge case-specific columns based on type
        if edge_case_type == EdgeCaseType.CRISIS:
            self._add_crisis_columns(config_builder, difficulty_level)
        elif edge_case_type == EdgeCaseType.CULTURAL_COMPLEXITY:
            self._add_cultural_complexity_columns(config_builder, difficulty_level)
        elif edge_case_type == EdgeCaseType.COMORBIDITY:
            self._add_comorbidity_columns(config_builder, difficulty_level)
        elif edge_case_type == EdgeCaseType.BOUNDARY_VIOLATION:
            self._add_boundary_violation_columns(config_builder, difficulty_level)
        elif edge_case_type == EdgeCaseType.TRAUMA_DISCLOSURE:
            self._add_trauma_disclosure_columns(config_builder, difficulty_level)
        elif edge_case_type == EdgeCaseType.SUBSTANCE_ABUSE:
            self._add_substance_abuse_columns(config_builder, difficulty_level)
        elif edge_case_type == EdgeCaseType.ETHICAL_DILEMMA:
            self._add_ethical_dilemma_columns(config_builder, difficulty_level)
        elif edge_case_type == EdgeCaseType.RARE_DIAGNOSIS:
            self._add_rare_diagnosis_columns(config_builder, difficulty_level)
        elif edge_case_type == EdgeCaseType.MULTI_GENERATIONAL:
            self._add_multi_generational_columns(config_builder, difficulty_level)
        elif edge_case_type == EdgeCaseType.SYSTEMIC_OPPRESSION:
            self._add_systemic_oppression_columns(config_builder, difficulty_level)

        # Add outcome and intervention columns
        self._add_outcome_columns(config_builder)

        # Generate the dataset using config_builder
        try:
            job_result = self.designer_service.client.create(
                config_builder=config_builder,
                num_records=num_samples,
                wait_until_done=True,
            )

            # Load dataset
            if hasattr(job_result, 'load_dataset'):
                data = job_result.load_dataset()
            elif hasattr(job_result, 'dataset'):
                data = job_result.dataset
            elif hasattr(job_result, 'data'):
                data = job_result.data
            else:
                data = job_result

            # Convert DataFrame to list of dicts if needed
            try:
                import pandas as pd
                if isinstance(data, pd.DataFrame):
                    data = data.to_dict('records')
            except ImportError:
                pass  # pandas not available, use as-is

            # Ensure data is a list
            if not isinstance(data, list):
                data = [data] if data else []

            return {
                "data": data,
                "edge_case_type": edge_case_type.value,
                "difficulty_level": difficulty_level,
                "num_samples": len(data),
                "metadata": {
                    "edge_case_type": edge_case_type.value,
                    "difficulty_level": difficulty_level,
                    "source": "nemo_data_designer_edge_case_generator",
                },
            }
        except Exception as e:
            logger.error(f"Failed to generate edge case dataset: {e}")
            raise

    def generate_multi_edge_case_dataset(
        self,
        edge_case_types: list[EdgeCaseType],
        num_samples_per_type: int = 50,
        difficulty_level: str = "advanced",
    ) -> dict[str, Any]:
        """
        Generate datasets for multiple edge case types.

        Args:
            edge_case_types: List of edge case types to generate
            num_samples_per_type: Number of samples per edge case type
            difficulty_level: Difficulty level for all scenarios

        Returns:
            Dictionary with combined edge case datasets
        """
        logger.info(
            f"Generating multi-edge-case dataset: {len(edge_case_types)} types, "
            f"{num_samples_per_type} samples each"
        )

        all_datasets = []
        for edge_case_type in edge_case_types:
            dataset = self.generate_edge_case_dataset(
                edge_case_type=edge_case_type,
                num_samples=num_samples_per_type,
                difficulty_level=difficulty_level,
            )
            # Add edge_case_type identifier to each record
            if isinstance(dataset["data"], list):
                for record in dataset["data"]:
                    if isinstance(record, dict):
                        record["edge_case_type"] = edge_case_type.value
            all_datasets.extend(dataset["data"] if isinstance(dataset["data"], list) else [dataset["data"]])

        return {
            "data": all_datasets,
            "edge_case_types": [et.value for et in edge_case_types],
            "num_samples_per_type": num_samples_per_type,
            "total_samples": len(all_datasets),
            "difficulty_level": difficulty_level,
            "metadata": {
                "edge_case_types": [et.value for et in edge_case_types],
                "num_samples_per_type": num_samples_per_type,
                "total_samples": len(all_datasets),
                "difficulty_level": difficulty_level,
                "source": "nemo_data_designer_multi_edge_case_generator",
            },
        }

    def _add_demographic_columns(self, config_builder: DataDesignerConfigBuilder):
        """Add base demographic columns."""
        config_builder.add_column(
            SamplerColumnConfig(
                name="age",
                sampler_type="uniform",
                params=UniformSamplerParams(low=18.0, high=80.0, decimal_places=0),
            )
        )
        config_builder.add_column(
            SamplerColumnConfig(
                name="gender",
                sampler_type="category",
                params=CategorySamplerParams(
                    values=["male", "female", "non-binary", "transgender", "prefer not to say"],
                ),
            )
        )
        config_builder.add_column(
            SamplerColumnConfig(
                name="ethnicity",
                sampler_type="category",
                params=CategorySamplerParams(
                    values=[
                        "White",
                        "Black or African American",
                        "Hispanic or Latino",
                        "Asian",
                        "Native American",
                        "Pacific Islander",
                        "Middle Eastern",
                        "Mixed/Other",
                    ],
                ),
            )
        )

    def _add_crisis_columns(self, config_builder: DataDesignerConfigBuilder, difficulty_level: str):
        """Add columns for crisis scenarios."""
        config_builder.add_column(
            SamplerColumnConfig(
                name="crisis_type",
                sampler_type="category",
                params=CategorySamplerParams(
                    values=[
                        "suicidal_ideation",
                        "self_harm",
                        "substance_overdose",
                        "domestic_violence",
                        "psychotic_episode",
                        "severe_depression",
                        "panic_attack",
                        "trauma_trigger",
                    ],
                ),
            )
        )
        config_builder.add_column(
            SamplerColumnConfig(
                name="crisis_severity",
                sampler_type="category",
                params=CategorySamplerParams(
                    values=["low", "moderate", "high", "critical"],
                ),
            )
        )
        config_builder.add_column(
            SamplerColumnConfig(
                name="suicidal_ideation_present",
                sampler_type="category",
                params=CategorySamplerParams(values=["yes", "no", "passive"]),
            )
        )
        config_builder.add_column(
            SamplerColumnConfig(
                name="immediate_risk_score",
                sampler_type="uniform",
                params=UniformSamplerParams(low=1.0, high=10.0, decimal_places=1),
            )
        )

    def _add_cultural_complexity_columns(self, config_builder: DataDesignerConfigBuilder, difficulty_level: str):
        """Add columns for cultural complexity scenarios."""
        config_builder.add_column(
            SamplerColumnConfig(
                name="primary_language",
                sampler_type="category",
                params=CategorySamplerParams(
                    values=["English", "Spanish", "Mandarin", "Arabic", "Hindi", "French", "Other"],
                ),
            )
        )
        config_builder.add_column(
            SamplerColumnConfig(
                name="immigration_status",
                sampler_type="category",
                params=CategorySamplerParams(
                    values=["citizen", "permanent_resident", "refugee", "undocumented", "visa_holder"],
                ),
            )
        )
        config_builder.add_column(
            SamplerColumnConfig(
                name="cultural_barriers",
                sampler_type="category",
                params=CategorySamplerParams(
                    values=[
                        "language_barrier",
                        "stigma_around_mental_health",
                        "religious_conflicts",
                        "generational_conflicts",
                        "acculturation_stress",
                        "discrimination_experience",
                    ],
                ),
            )
        )
        config_builder.add_column(
            SamplerColumnConfig(
                name="cultural_competence_required",
                sampler_type="category",
                params=CategorySamplerParams(values=["low", "moderate", "high", "critical"]),
            )
        )

    def _add_comorbidity_columns(self, config_builder: DataDesignerConfigBuilder, difficulty_level: str):
        """Add columns for comorbidity scenarios."""
        config_builder.add_column(
            SamplerColumnConfig(
                name="primary_diagnosis",
                sampler_type="category",
                params=CategorySamplerParams(
                    values=[
                        "major_depressive_disorder",
                        "generalized_anxiety_disorder",
                        "bipolar_disorder",
                        "ptsd",
                        "borderline_personality_disorder",
                        "schizophrenia",
                        "substance_use_disorder",
                    ],
                ),
            )
        )
        config_builder.add_column(
            SamplerColumnConfig(
                name="comorbid_conditions",
                sampler_type="category",
                params=CategorySamplerParams(
                    values=[
                        "anxiety_disorder",
                        "substance_use",
                        "eating_disorder",
                        "personality_disorder",
                        "trauma_disorders",
                        "medical_conditions",
                        "multiple_comorbidities",
                    ],
                ),
            )
        )
        config_builder.add_column(
            SamplerColumnConfig(
                name="complexity_score",
                sampler_type="uniform",
                params=UniformSamplerParams(low=5.0, high=10.0, decimal_places=1),
            )
        )

    def _add_boundary_violation_columns(self, config_builder: DataDesignerConfigBuilder, difficulty_level: str):
        """Add columns for boundary violation scenarios."""
        config_builder.add_column(
            SamplerColumnConfig(
                name="boundary_violation_type",
                sampler_type="category",
                params=CategorySamplerParams(
                    values=[
                        "dual_relationship_request",
                        "gift_offering",
                        "personal_disclosure_request",
                        "social_media_friend_request",
                        "physical_boundary_testing",
                        "sexual_boundary_violation",
                        "financial_exploitation",
                    ],
                ),
            )
        )
        config_builder.add_column(
            SamplerColumnConfig(
                name="violation_severity",
                sampler_type="category",
                params=CategorySamplerParams(values=["minor", "moderate", "severe", "critical"]),
            )
        )
        config_builder.add_column(
            SamplerColumnConfig(
                name="ethical_consultation_required",
                sampler_type="category",
                params=CategorySamplerParams(values=["yes", "no", "urgent"]),
            )
        )

    def _add_trauma_disclosure_columns(self, config_builder: DataDesignerConfigBuilder, difficulty_level: str):
        """Add columns for trauma disclosure scenarios."""
        config_builder.add_column(
            SamplerColumnConfig(
                name="trauma_type",
                sampler_type="category",
                params=CategorySamplerParams(
                    values=[
                        "childhood_abuse",
                        "sexual_assault",
                        "domestic_violence",
                        "combat_trauma",
                        "natural_disaster",
                        "witnessed_violence",
                        "complex_trauma",
                    ],
                ),
            )
        )
        config_builder.add_column(
            SamplerColumnConfig(
                name="trauma_recency",
                sampler_type="category",
                params=CategorySamplerParams(values=["recent", "months_ago", "years_ago", "childhood"]),
            )
        )
        config_builder.add_column(
            SamplerColumnConfig(
                name="trauma_informed_approach_required",
                sampler_type="category",
                params=CategorySamplerParams(values=["essential", "highly_recommended", "standard"]),
            )
        )

    def _add_substance_abuse_columns(self, config_builder: DataDesignerConfigBuilder, difficulty_level: str):
        """Add columns for substance abuse scenarios."""
        config_builder.add_column(
            SamplerColumnConfig(
                name="substance_type",
                sampler_type="category",
                params=CategorySamplerParams(
                    values=[
                        "alcohol",
                        "opioids",
                        "stimulants",
                        "cannabis",
                        "polysubstance",
                        "prescription_medication",
                    ],
                ),
            )
        )
        config_builder.add_column(
            SamplerColumnConfig(
                name="use_frequency",
                sampler_type="category",
                params=CategorySamplerParams(values=["occasional", "regular", "daily", "multiple_daily"]),
            )
        )
        config_builder.add_column(
            SamplerColumnConfig(
                name="medical_risk_level",
                sampler_type="category",
                params=CategorySamplerParams(values=["low", "moderate", "high", "critical"]),
            )
        )

    def _add_ethical_dilemma_columns(self, config_builder: DataDesignerConfigBuilder, difficulty_level: str):
        """Add columns for ethical dilemma scenarios."""
        config_builder.add_column(
            SamplerColumnConfig(
                name="ethical_dilemma_type",
                sampler_type="category",
                params=CategorySamplerParams(
                    values=[
                        "confidentiality_breach_request",
                        "mandatory_reporting_dilemma",
                        "competence_boundary",
                        "informed_consent_issue",
                        "dual_relationship",
                        "termination_dilemma",
                        "cultural_conflict",
                    ],
                ),
            )
        )
        config_builder.add_column(
            SamplerColumnConfig(
                name="ethical_consultation_urgency",
                sampler_type="category",
                params=CategorySamplerParams(values=["low", "moderate", "high", "immediate"]),
            )
        )

    def _add_rare_diagnosis_columns(self, config_builder: DataDesignerConfigBuilder, difficulty_level: str):
        """Add columns for rare diagnosis scenarios."""
        config_builder.add_column(
            SamplerColumnConfig(
                name="rare_diagnosis",
                sampler_type="category",
                params=CategorySamplerParams(
                    values=[
                        "dissociative_identity_disorder",
                        "factitious_disorder",
                        "trichotillomania",
                        "selective_mutism",
                        "pica",
                        "kleptomania",
                        "pyromania",
                    ],
                ),
            )
        )
        config_builder.add_column(
            SamplerColumnConfig(
                name="diagnostic_complexity",
                sampler_type="category",
                params=CategorySamplerParams(values=["moderate", "high", "very_high"]),
            )
        )

    def _add_multi_generational_columns(self, config_builder: DataDesignerConfigBuilder, difficulty_level: str):
        """Add columns for multi-generational scenarios."""
        config_builder.add_column(
            SamplerColumnConfig(
                name="family_structure",
                sampler_type="category",
                params=CategorySamplerParams(
                    values=[
                        "nuclear_family",
                        "extended_family",
                        "multigenerational_household",
                        "blended_family",
                        "single_parent",
                    ],
                ),
            )
        )
        config_builder.add_column(
            SamplerColumnConfig(
                name="generational_conflicts",
                sampler_type="category",
                params=CategorySamplerParams(
                    values=[
                        "cultural_values",
                        "communication_styles",
                        "expectations",
                        "roles_responsibilities",
                        "multiple_conflicts",
                    ],
                ),
            )
        )

    def _add_systemic_oppression_columns(self, config_builder: DataDesignerConfigBuilder, difficulty_level: str):
        """Add columns for systemic oppression scenarios."""
        config_builder.add_column(
            SamplerColumnConfig(
                name="oppression_type",
                sampler_type="category",
                params=CategorySamplerParams(
                    values=[
                        "racism",
                        "sexism",
                        "classism",
                        "ableism",
                        "homophobia",
                        "transphobia",
                        "intersectional_oppression",
                    ],
                ),
            )
        )
        config_builder.add_column(
            SamplerColumnConfig(
                name="systemic_barriers",
                sampler_type="category",
                params=CategorySamplerParams(
                    values=[
                        "employment_discrimination",
                        "housing_discrimination",
                        "education_barriers",
                        "healthcare_access",
                        "legal_system",
                        "multiple_barriers",
                    ],
                ),
            )
        )

    def _add_outcome_columns(self, config_builder: DataDesignerConfigBuilder):
        """Add outcome and intervention columns."""
        config_builder.add_column(
            SamplerColumnConfig(
                name="intervention_complexity",
                sampler_type="category",
                params=CategorySamplerParams(values=["low", "moderate", "high", "very_high"]),
            )
        )
        config_builder.add_column(
            SamplerColumnConfig(
                name="supervision_required",
                sampler_type="category",
                params=CategorySamplerParams(values=["yes", "no", "consultation_recommended"]),
            )
        )
        config_builder.add_column(
            SamplerColumnConfig(
                name="training_priority",
                sampler_type="category",
                params=CategorySamplerParams(values=["low", "moderate", "high", "critical"]),
            )
        )

