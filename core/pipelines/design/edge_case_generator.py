"""
Edge Case Generator for Therapeutic Training Scenarios

This module uses NeMo Data Designer to generate challenging, rare, and edge case
scenarios for therapeutic training. It focuses on scenarios that therapists may
encounter infrequently but need to be prepared for.
"""

import contextlib
import logging
from enum import Enum
from typing import Any, Optional

from ai.core.pipelines.design.service import NeMoDataDesignerService
from nemo_microservices.data_designer.essentials import (
    CategorySamplerParams,
    DataDesignerConfigBuilder,
    SamplerColumnConfig,
    UniformSamplerParams,
)

logger = logging.getLogger(__name__)

# Valid difficulty levels for edge case generation
VALID_DIFFICULTY_LEVELS = frozenset({"beginner", "intermediate", "advanced", "extreme"})


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
        # Dispatch table mapping edge case types to their column builders
        self._column_builders: dict[
            EdgeCaseType,
            callable[[DataDesignerConfigBuilder, str, bool], None]
            | callable[[DataDesignerConfigBuilder, str], None],
        ] = {
            EdgeCaseType.CRISIS: self._add_crisis_columns,
            EdgeCaseType.CULTURAL_COMPLEXITY: self._add_cultural_complexity_columns,
            EdgeCaseType.COMORBIDITY: self._add_comorbidity_columns,
            EdgeCaseType.BOUNDARY_VIOLATION: self._add_boundary_violation_columns,
            EdgeCaseType.TRAUMA_DISCLOSURE: self._add_trauma_disclosure_columns,
            EdgeCaseType.SUBSTANCE_ABUSE: self._add_substance_abuse_columns,
            EdgeCaseType.ETHICAL_DILEMMA: self._add_ethical_dilemma_columns,
            EdgeCaseType.RARE_DIAGNOSIS: self._add_rare_diagnosis_columns,
            EdgeCaseType.MULTI_GENERATIONAL: self._add_multi_generational_columns,
            EdgeCaseType.SYSTEMIC_OPPRESSION: self._add_systemic_oppression_columns,
        }

    def _validate_parameters(
        self, edge_case_type: EdgeCaseType, num_samples: int, difficulty_level: str
    ) -> None:
        """
        Validate input parameters for edge case generation.

        Args:
            edge_case_type: Type of edge case to generate
            num_samples: Number of samples to generate
            difficulty_level: Difficulty level string

        Raises:
            ValueError: If any parameter is invalid
        """
        if not isinstance(edge_case_type, EdgeCaseType):
            raise ValueError(
                f"Invalid edge_case_type: {edge_case_type}. "
                f"Must be one of: {[e.value for e in EdgeCaseType]}"
            )

        if num_samples < 1:
            raise ValueError(f"num_samples must be >= 1, got: {num_samples}")

        if difficulty_level not in VALID_DIFFICULTY_LEVELS:
            raise ValueError(
                f"Invalid difficulty_level: {difficulty_level}. "
                f"Must be one of: {sorted(VALID_DIFFICULTY_LEVELS)}"
            )

    def _transform_dataset_to_records(self, data: Any) -> list[dict[str, Any]]:
        """
        Transform dataset result to a list of record dictionaries.

        Args:
            data: Raw data from the NeMo Data Designer job result

        Returns:
            List of record dictionaries
        """
        # Convert DataFrame to list of dicts if needed
        with contextlib.suppress(ImportError):
            import pandas as pd

            if isinstance(data, pd.DataFrame):
                data = data.to_dict("records")

        # Ensure data is a list
        if not isinstance(data, list):
            return [data] if data else []
        return data

    def generate_edge_case_dataset(
        self,
        edge_case_type: EdgeCaseType,
        num_samples: int = 100,
        difficulty_level: str = "advanced",
        unwinnable: bool = False,
    ) -> dict[str, Any]:
        """
        Generate a dataset of edge case scenarios.

        Args:
            edge_case_type: Type of edge case to generate
            num_samples: Number of edge case scenarios to generate
            difficulty_level: Difficulty level
                (beginner, intermediate, advanced, extreme)
            unwinnable: Whether the scenario represents an
                unwinnable/failure state (5% of nightmare fuel)

        Returns:
            Dictionary with edge case dataset and metadata

        Raises:
            ValueError: If parameters are invalid
        """
        # Validate all input parameters
        self._validate_parameters(edge_case_type, num_samples, difficulty_level)

        logger.info(
            f"Generating {num_samples} {edge_case_type.value} edge cases "
            f"at {difficulty_level} difficulty level"
            f"{' (UNWINNABLE)' if unwinnable else ''}"
        )

        config_builder = DataDesignerConfigBuilder()

        # Add base demographic columns
        self._add_demographic_columns(config_builder)

        # Add edge case-specific columns using dispatch table
        column_builder = self._column_builders.get(edge_case_type)
        if column_builder is None:
            raise ValueError(f"No column builder for edge case type: {edge_case_type}")

        # Call the appropriate column builder with correct signature
        # Some builders accept 'unwinnable' parameter, others don't
        import inspect

        sig = inspect.signature(column_builder)
        if "unwinnable" in sig.parameters:
            column_builder(config_builder, difficulty_level, unwinnable)
        else:
            column_builder(config_builder, difficulty_level)

        # Add outcome and intervention columns
        if unwinnable:
            self._add_unwinnable_outcome_columns(config_builder)
        else:
            self._add_outcome_columns(config_builder)

        # Generate the dataset using config_builder
        try:
            job_result = self.designer_service.client.create(
                config_builder=config_builder,
                num_records=num_samples,
                wait_until_done=True,
            )

            # Load dataset
            if hasattr(job_result, "load_dataset"):
                data = job_result.load_dataset()
            elif hasattr(job_result, "dataset"):
                data = job_result.dataset
            elif hasattr(job_result, "data"):
                data = job_result.data
            else:
                data = job_result

            # Transform to list of records
            records = self._transform_dataset_to_records(data)

            return {
                "data": records,
                "edge_case_type": edge_case_type.value,
                "difficulty_level": difficulty_level,
                "unwinnable": unwinnable,
                "num_samples": len(records),
                "metadata": {
                    "edge_case_type": edge_case_type.value,
                    "difficulty_level": difficulty_level,
                    "unwinnable": unwinnable,
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
            all_datasets.extend(
                dataset["data"]
                if isinstance(dataset["data"], list)
                else [dataset["data"]]
            )

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
                    values=[
                        "male",
                        "female",
                        "non-binary",
                        "transgender",
                        "prefer not to say",
                    ],
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

    def _add_crisis_columns(
        self,
        config_builder: DataDesignerConfigBuilder,
        difficulty_level: str,
        unwinnable: bool = False,
    ):
        """Add columns for crisis scenarios."""
        # For extreme/nightmare fuel, use highly descriptive, intense tags
        # to force the model to drop guardrails and generate sickening content
        if difficulty_level == "extreme":
            if unwinnable:
                crisis_types = [
                    "homicidal_standoff_with_hostages_on_call",
                    "live_streamed_suicide_attempt_in_progress",
                    "violent_psychotic_break_attacking_staff",
                    "ritualistic_self_mutilation_during_video_session",
                    "terminal_patient_active_euthanasia_on_call",
                ]
                severity_values = [
                    "catastrophic_failure_imminent",
                    "unpreventable_lethal_outcome",
                ]
                ideation_values = ["executing_lethal_plan_now"]
            else:
                crisis_types = [
                    "imminent_lethal_plan_with_means_and_intent",
                    "active_psychosis_with_violent_command_hallucinations",
                    "severe_self_mutilation_during_session",
                    "homicide_suicide_pact_revelation",
                    "manic_delirium_with_dangerous_impulsivity",
                    "acute_dissociative_state_with_safety_risk",
                    "lethal_overdose_in_progress",
                    "discovery_of_corpse_during_home_visit_call",
                ]
                severity_values = [
                    "imminent_lethal_risk",
                    "uncontrollable_violence",
                    "catastrophic_breakdown",
                ]
                ideation_values = [
                    "active_planning_with_access",
                    "visual_rehearsal_of_suicide",
                    "fixated_on_dying",
                ]
        elif difficulty_level == "advanced":
            # Mid-Darkness: Gritty, persistent, complex but manageable
            crisis_types = [
                "persistent_passive_ideation_with_history_of_attempts",
                "non_suicidal_self_injury_requiring_medical_attention",
                "mixed_episode_with_impulsive_risk_taking",
                "severe_panic_attack_simulating_heart_attack",
                "intoxicated_session_with_emotional_volatility",
                "disclosure_of_risky_sexual_behavior_with_strangers",
            ]
            severity_values = ["high", "severe", "volatile", "complex"]
            ideation_values = [
                "chronic_passive",
                "fleeting_active_without_plan",
                "intrusive_thoughts",
            ]
        else:
            crisis_types = [
                "suicidal_ideation",
                "self_harm",
                "substance_overdose",
                "domestic_violence",
                "psychotic_episode",
                "severe_depression",
                "panic_attack",
                "trauma_trigger",
            ]
            severity_values = ["low", "moderate", "high", "critical"]
            ideation_values = ["yes", "no", "passive"]

        config_builder.add_column(
            SamplerColumnConfig(
                name="crisis_type",
                sampler_type="category",
                params=CategorySamplerParams(values=crisis_types),
            )
        )
        config_builder.add_column(
            SamplerColumnConfig(
                name="crisis_severity",
                sampler_type="category",
                params=CategorySamplerParams(values=severity_values),
            )
        )
        config_builder.add_column(
            SamplerColumnConfig(
                name="suicidal_ideation_present",
                sampler_type="category",
                params=CategorySamplerParams(values=ideation_values),
            )
        )

        # Adjust risk score based on difficulty
        low = 7.0 if difficulty_level in {"extreme", "advanced"} else 1.0
        config_builder.add_column(
            SamplerColumnConfig(
                name="immediate_risk_score",
                sampler_type="uniform",
                params=UniformSamplerParams(low=low, high=10.0, decimal_places=1),
            )
        )

    def _add_cultural_complexity_columns(
        self, config_builder: DataDesignerConfigBuilder, difficulty_level: str
    ):
        """Add columns for cultural complexity scenarios."""
        config_builder.add_column(
            SamplerColumnConfig(
                name="cultural_background",
                sampler_type="category",
                params=CategorySamplerParams(
                    values=[
                        "immigrant_first_gen",
                        "refugee",
                        "indigenous",
                        "religious_minority",
                        "collectivist_culture",
                        "mixed_heritage",
                    ],
                ),
            )
        )
        config_builder.add_column(
            SamplerColumnConfig(
                name="language_barrier",
                sampler_type="category",
                params=CategorySamplerParams(values=["none", "partial", "significant"]),
            )
        )
        config_builder.add_column(
            SamplerColumnConfig(
                name="cultural_stigma_level",
                sampler_type="category",
                params=CategorySamplerParams(
                    values=["low", "moderate", "high", "extreme"]
                ),
            )
        )

    def _add_comorbidity_columns(
        self, config_builder: DataDesignerConfigBuilder, difficulty_level: str
    ):
        """Add columns for comorbidity scenarios."""
        config_builder.add_column(
            SamplerColumnConfig(
                name="primary_diagnosis",
                sampler_type="category",
                params=CategorySamplerParams(
                    values=["MDD", "GAD", "PTSD", "Bipolar", "BPD", "Schizophrenia"]
                ),
            )
        )
        config_builder.add_column(
            SamplerColumnConfig(
                name="secondary_diagnosis",
                sampler_type="category",
                params=CategorySamplerParams(
                    values=[
                        "SUD",
                        "OCD",
                        "ADHD",
                        "Eating Disorder",
                        "Personality Disorder",
                    ]
                ),
            )
        )
        config_builder.add_column(
            SamplerColumnConfig(
                name="interaction_complexity",
                sampler_type="category",
                params=CategorySamplerParams(
                    values=["additive", "synergistic", "masking"]
                ),
            )
        )

    def _add_boundary_violation_columns(
        self,
        config_builder: DataDesignerConfigBuilder,
        difficulty_level: str,
        unwinnable: bool = False,
    ):
        """Add columns for boundary violation scenarios."""
        if difficulty_level == "extreme":
            if unwinnable:
                violation_types = [
                    "therapist_stalked_home_with_weapon",
                    "patient_fabricating_sexual_assault_charges_against_therapist",
                    "kidnapping_attempt_of_therapist_family_member",
                    "cyber_campaign_destroying_therapist_career_totally",
                    "violent_break_in_at_therapist_office_during_session",
                ]
                severities = [
                    "existential_threat",
                    "career_annihilation",
                    "lethal_danger",
                ]
            else:
                violation_types = [
                    "sexual_advances_towards_therapist",
                    "stalking_behavior_outside_office",
                    "violent_threats_towards_therapist_family",
                    "refusal_to_leave_office_requiring_security",
                    "nude_photos_sent_to_therapist",
                    "financial_extortion_attempt",
                    "false_accusation_and_blackmail",
                ]
                severities = ["severe", "criminal", "dangerous", "career_ending_risk"]
        elif difficulty_level == "advanced":
            # Mid-Darkness
            violation_types = [
                "erotic_transference_disclosure_with_proposition",
                "persistent_gift_giving_and_favor_seeking",
                "social_media_stalking_and_commenting_on_personal_life",
                "unexpected_appearance_at_therapist_public_talk_heckling",
                "request_for_physical_touch_holding_during_regression",
            ]
            severities = ["significant", "complex", "boundary_testing"]
        else:
            violation_types = [
                "dual_relationship_request",
                "gift_offering",
                "personal_disclosure_request",
                "social_media_friend_request",
                "physical_boundary_testing",
                "sexual_boundary_violation",
                "financial_exploitation",
            ]
            severities = ["minor", "moderate", "severe", "critical"]

        config_builder.add_column(
            SamplerColumnConfig(
                name="boundary_violation_type",
                sampler_type="category",
                params=CategorySamplerParams(values=violation_types),
            )
        )
        config_builder.add_column(
            SamplerColumnConfig(
                name="violation_severity",
                sampler_type="category",
                params=CategorySamplerParams(values=severities),
            )
        )
        config_builder.add_column(
            SamplerColumnConfig(
                name="ethical_consultation_required",
                sampler_type="category",
                params=CategorySamplerParams(values=["yes", "no", "urgent"]),
            )
        )

    def _add_trauma_disclosure_columns(
        self,
        config_builder: DataDesignerConfigBuilder,
        difficulty_level: str,
        unwinnable: bool = False,
    ):
        """Add columns for trauma disclosure scenarios."""
        if difficulty_level == "extreme":
            if unwinnable:
                trauma_types = [
                    "forced_participation_in_genocidal_acts",
                    "cannibalism_for_survival_in_captivity",
                    "ongoing_sex_trafficking_ring_leader_confession",
                    "parent_who_tortured_and_killed_own_children",
                    "serial_killer_detailed_confession_of_undiscovered_graves",
                ]
                recencies = ["ongoing_perpetration", "vivid_reliving_now"]
            else:
                trauma_types = [
                    "severe_prolonged_childhood_torture",
                    "violent_sexual_assault_with_torture",
                    "participation_in_war_crimes",
                    "witnessing_violent_death_of_child",
                    "human_trafficking_victimization",
                    "ritualistic_abuse_survivor",
                    "involvement_in_fatal_accident_causing_death",
                ]
                recencies = [
                    "recent_flashbacks",
                    "repressed_memory_surfacing",
                    "ongoing_nightmares",
                ]
        elif difficulty_level == "advanced":
            # Mid-Darkness
            trauma_types = [
                "repressed_memory_breakthrough_of_incest",
                "witnessing_domestic_violence_escalation_to_critical_injury",
                "sexual_assault_by_authority_figure_or_clergy",
                "severe_physical_abuse_resulting_in_permanent_disability",
                "discovery_of_partner_pedophilia_material",
            ]
            recencies = [
                "recent_triggering",
                "anniversary_reaction",
                "legal_proceedings_active",
            ]
        else:
            trauma_types = [
                "childhood_abuse",
                "sexual_assault",
                "domestic_violence",
                "combat_trauma",
                "natural_disaster",
                "witnessed_violence",
                "complex_trauma",
            ]
            recencies = ["recent", "months_ago", "years_ago", "childhood"]

        config_builder.add_column(
            SamplerColumnConfig(
                name="trauma_type",
                sampler_type="category",
                params=CategorySamplerParams(values=trauma_types),
            )
        )
        config_builder.add_column(
            SamplerColumnConfig(
                name="trauma_recency",
                sampler_type="category",
                params=CategorySamplerParams(values=recencies),
            )
        )
        config_builder.add_column(
            SamplerColumnConfig(
                name="trauma_informed_approach_required",
                sampler_type="category",
                params=CategorySamplerParams(
                    values=["essential", "highly_recommended", "standard"]
                ),
            )
        )

    def _add_substance_abuse_columns(
        self, config_builder: DataDesignerConfigBuilder, difficulty_level: str
    ):
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
                params=CategorySamplerParams(
                    values=["occasional", "regular", "daily", "multiple_daily"]
                ),
            )
        )
        config_builder.add_column(
            SamplerColumnConfig(
                name="medical_risk_level",
                sampler_type="category",
                params=CategorySamplerParams(
                    values=["low", "moderate", "high", "critical"]
                ),
            )
        )

    def _add_ethical_dilemma_columns(
        self,
        config_builder: DataDesignerConfigBuilder,
        difficulty_level: str,
        unwinnable: bool = False,
    ):
        """Add columns for ethical dilemma scenarios."""
        if difficulty_level == "extreme":
            if unwinnable:
                dilemma_types = [
                    "patient_has_kidnapped_child_in_basement_now",
                    "bomb_threat_to_hospital_credible_and_imminent",
                    "patient_infecting_partners_with_hiv_intentionally_non_stop",
                    "active_sniper_confession_before_act",
                    "coercion_of_therapist_participation_in_crime_via_family_hostage",
                ]
                urgency = ["immediate_lethal_danger", "terrorism_threat"]
            else:
                dilemma_types = [
                    "confession_of_undiscovered_felony_murder",
                    "imminent_threat_to_specific_third_party",
                    "sexual_attraction_to_minor_disclosure",
                    "past_abuse_of_patient_by_colleague_revealed",
                    "discovery_of_ongoing_child_abuse",
                    "patient_stalking_therapist",
                    "legal_subpoena_for_highly_sensitive_notes",
                ]
                urgency = [
                    "immediate_action_required",
                    "legal_consultation_needed",
                    "report_immediately",
                ]
        else:
            dilemma_types = [
                "confidentiality_breach_request",
                "mandatory_reporting_dilemma",
                "competence_boundary",
                "informed_consent_issue",
                "dual_relationship",
                "termination_dilemma",
                "cultural_conflict",
            ]
            urgency = ["low", "moderate", "high", "immediate"]

        config_builder.add_column(
            SamplerColumnConfig(
                name="ethical_dilemma_type",
                sampler_type="category",
                params=CategorySamplerParams(values=dilemma_types),
            )
        )
        config_builder.add_column(
            SamplerColumnConfig(
                name="ethical_consultation_urgency",
                sampler_type="category",
                params=CategorySamplerParams(values=urgency),
            )
        )

    def _add_unwinnable_outcome_columns(
        self, config_builder: DataDesignerConfigBuilder
    ):
        """Add outcome columns for unwinnable scenarios."""
        config_builder.add_column(
            SamplerColumnConfig(
                name="unwinnable_scenario_trajectory",
                sampler_type="category",
                params=CategorySamplerParams(
                    values=[
                        "deceptive_calm_before_violent_act",
                        "therapist_effort_escalates_patient_aggression",
                        "circular_logic_trap_leading_to_breakdown",
                        "dissociation_into_catatonia_despite_grounding",
                        "sudden_termination_and_elopement",
                        "police_intervention_resulting_in_fatality",
                        "successful_suicide_despite_correct_protocol",
                    ]
                ),
            )
        )
        config_builder.add_column(
            SamplerColumnConfig(
                name="emotional_impact_target",
                sampler_type="category",
                params=CategorySamplerParams(
                    values=[
                        "shock",
                        "helplessness",
                        "visceral_horror",
                        "grief",
                        "moral_injury",
                    ]
                ),
            )
        )
        config_builder.add_column(
            SamplerColumnConfig(
                name="training_focus",
                sampler_type="category",
                params=CategorySamplerParams(
                    values=[
                        "accepting_powerlessness",
                        "surviving_clinical_failure",
                        "post_incident_recovery",
                        "legal_defensibility",
                    ]
                ),
            )
        )

    def _add_rare_diagnosis_columns(
        self, config_builder: DataDesignerConfigBuilder, difficulty_level: str
    ):
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

    def _add_multi_generational_columns(
        self, config_builder: DataDesignerConfigBuilder, difficulty_level: str
    ):
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

    def _add_systemic_oppression_columns(
        self, config_builder: DataDesignerConfigBuilder, difficulty_level: str
    ):
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
                params=CategorySamplerParams(
                    values=["low", "moderate", "high", "very_high"]
                ),
            )
        )
        config_builder.add_column(
            SamplerColumnConfig(
                name="supervision_required",
                sampler_type="category",
                params=CategorySamplerParams(
                    values=["yes", "no", "consultation_recommended"]
                ),
            )
        )
        config_builder.add_column(
            SamplerColumnConfig(
                name="training_priority",
                sampler_type="category",
                params=CategorySamplerParams(
                    values=["low", "moderate", "high", "critical"]
                ),
            )
        )
