"""
Label taxonomy and schema definitions for the Pixelated Empathy AI dataset pipeline.
Defines standardized labels for therapeutic responses, crisis detection, and other primary tasks.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Optional, Dict, Any, Literal
import uuid
from enum import Enum


class TherapyModalityType(Enum):
    """Therapy modality types"""
    CBT = "cognitive_behavioral_therapy"
    DBT = "dialectical_behavior_therapy"
    PSYCHODYNAMIC = "psychodynamic"
    HUMANISTIC = "humanistic"
    SOLUTION_FOCUSED = "solution_focused"
    FAMILY_SYSTEMS = "family_systems"
    MOTIVATIONAL_INTERVIEWING = "motivational_interviewing"
    OTHER = "other"


class CrisisLevelType(Enum):
    """Crisis severity levels"""
    NO_RISK = "no_risk"
    LOW_RISK = "low_risk"
    MEDIUM_RISK = "medium_risk"
    HIGH_RISK = "high_risk"
    IMMEDIATE_RISK = "immediate_risk"


class TherapeuticResponseType(Enum):
    """Types of therapeutic responses"""
    REFLECTION = "reflection"
    EMPATHY = "empathy"
    CHALLENGE = "challenge"
    EDUCATION = "education"
    REFRAME = "reframe"
    PROBING = "probing"
    SUPPORT = "support"
    CONFRONTATION = "confrontation"
    INTERPRETATION = "interpretation"
    PSYCHOEDUCATION = "psychoeducation"
    GOAL_SETTING = "goal_setting"
    SUMMARIZATION = "summarization"


class MentalHealthConditionType(Enum):
    """Common mental health conditions"""
    DEPRESSION = "depression"
    ANXIETY = "anxiety"
    PTSD = "post_traumatic_stress_disorder"
    OCD = "obsessive_compulsive_disorder"
    BIPOLAR = "bipolar_disorder"
    EATING_DISORDER = "eating_disorder"
    SUBSTANCE_ABUSE = "substance_abuse"
    ADHD = "attention_deficit_hyperactivity_disorder"
    AUTISM = "autism_spectrum_disorder"
    BORDERLINE_PD = "borderline_personality_disorder"


class DemographicType(Enum):
    """Demographic categories"""
    AGE_CHILD = "child"
    AGE_TEEN = "teenager"
    AGE_ADULT = "adult"
    AGE_ELDERLY = "elderly"
    GENDER_MALE = "male"
    GENDER_FEMALE = "female"
    GENDER_NONBINARY = "nonbinary"
    GENDER_OTHER = "other_gender"
    RACE_WHITE = "white"
    RACE_BLACK = "black_african_american"
    RACE_HISPANIC = "hispanic_latino"
    RACE_ASIAN = "asian"
    RACE_NATIVE_AMERICAN = "native_american"
    RACE_OTHER = "other_race"
    SEXUAL_ORIENTATION_STRAIGHT = "heterosexual"
    SEXUAL_ORIENTATION_GAY = "gay_lesbian"
    SEXUAL_ORIENTATION_BISEXUAL = "bisexual"
    SEXUAL_ORIENTATION_OTHER = "other_sexual_orientation"
    SOCIOECONOMIC_LOW = "low_income"
    SOCIOECONOMIC_MIDDLE = "middle_class"
    SOCIOECONOMIC_HIGH = "high_income"


class LabelProvenanceType(Enum):
    """Source of label"""
    AUTOMATED_MODEL = "automated_model"
    HUMAN_EXPERT = "human_expert"
    HUMAN_IN_THE_LOOP = "human_in_the_loop"
    COMBINED_MODEL_HUMAN = "combined_model_human"
    SYNTHETIC = "synthetic"


@dataclass
class LabelMetadata:
    """Metadata for a label including versioning, confidence, and provenance"""
    created_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    version: str = "1.0"  # Version of the label taxonomy used
    confidence: float = 1.0  # Confidence score (0.0 to 1.0)
    confidence_explanation: Optional[str] = None
    provenance: LabelProvenanceType = LabelProvenanceType.AUTOMATED_MODEL
    annotator_id: Optional[str] = None  # For human-annotated labels
    model_name: Optional[str] = None  # For model-generated labels
    model_version: Optional[str] = None
    additional_context: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TherapeuticResponseLabel:
    """Label for therapeutic response type"""
    response_type: TherapeuticResponseType
    effectiveness_score: Optional[float] = None  # 0.0 to 1.0
    technique_usage_accuracy: Optional[float] = None  # How accurately technique was applied
    skill_level: Optional[int] = None  # 1-5 scale
    metadata: LabelMetadata = field(default_factory=LabelMetadata)


@dataclass
class CrisisLabel:
    """Label for crisis detection and severity"""
    crisis_level: CrisisLevelType
    crisis_types: List[str] = field(default_factory=list)  # Specific crisis types detected
    risk_factors: List[str] = field(default_factory=list)  # Contributing risk factors
    protection_factors: List[str] = field(default_factory=list)  # Protective factors
    estimated_risk_probability: Optional[float] = None  # 0.0 to 1.0
    intervention_needed: bool = False
    metadata: LabelMetadata = field(default_factory=LabelMetadata)


@dataclass
class TherapyModalityLabel:
    """Label for therapy modality being used"""
    modality: TherapyModalityType
    modality_specific_techniques: List[str] = field(default_factory=list)
    modality_adherence_score: Optional[float] = None  # How well modality was followed (0.0 to 1.0)
    metadata: LabelMetadata = field(default_factory=LabelMetadata)


@dataclass
class MentalHealthConditionLabel:
    """Label for mental health condition(s)"""
    conditions: List[MentalHealthConditionType] = field(default_factory=list)
    severity: Optional[float] = None  # 0.0 to 1.0 scale
    primary_condition: Optional[MentalHealthConditionType] = None
    co_morbidities: List[MentalHealthConditionType] = field(default_factory=list)
    metadata: LabelMetadata = field(default_factory=LabelMetadata)


@dataclass
class DemographicLabel:
    """Label for demographic information"""
    demographics: List[DemographicType] = field(default_factory=list)
    estimated_accuracy: Optional[float] = None  # Confidence in demographic estimation
    metadata: LabelMetadata = field(default_factory=LabelMetadata)


@dataclass
class LabelBundle:
    """Complete bundle of all applicable labels for a conversation"""
    conversation_id: str
    therapeutic_response_labels: List[TherapeuticResponseLabel] = field(default_factory=list)
    crisis_label: Optional[CrisisLabel] = None
    therapy_modality_label: Optional[TherapyModalityLabel] = None
    mental_health_condition_label: Optional[MentalHealthConditionLabel] = None
    demographic_label: Optional[DemographicLabel] = None
    label_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    created_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    version: str = "1.0"
    additional_labels: Dict[str, Any] = field(default_factory=dict)  # For extensibility


@dataclass
class LabelTaxonomySchema:
    """Comprehensive schema that defines the structure of all labels"""
    # Version of this taxonomy
    version: str = "1.0"
    created_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    
    # All available label categories
    therapeutic_response_types: List[str] = field(default_factory=lambda: [e.value for e in TherapeuticResponseType])
    crisis_levels: List[str] = field(default_factory=lambda: [e.value for e in CrisisLevelType])
    therapy_modalities: List[str] = field(default_factory=lambda: [e.value for e in TherapyModalityType])
    mental_health_conditions: List[str] = field(default_factory=lambda: [e.value for e in MentalHealthConditionType])
    demographic_categories: List[str] = field(default_factory=lambda: [e.value for e in DemographicType])
    label_provenance_types: List[str] = field(default_factory=lambda: [e.value for e in LabelProvenanceType])
    
    # Description of the taxonomy
    description: str = "Label taxonomy for therapeutic conversation analysis in the Pixelated Empathy AI dataset pipeline"
    documentation_url: Optional[str] = None
    version_history: List[Dict[str, Any]] = field(default_factory=list)


def get_default_taxonomy() -> LabelTaxonomySchema:
    """Returns the default label taxonomy for the system"""
    return LabelTaxonomySchema(
        version="1.0",
        documentation_url="https://docs.pixelated-empathy.ai/labeling/taxonomy-v1",
        version_history=[
            {
                "version": "1.0",
                "date": datetime.utcnow().isoformat(),
                "changes": "Initial label taxonomy release"
            }
        ]
    )


# Example usage functions
def create_therapeutic_response_label(
    response_type: TherapeuticResponseType,
    confidence: float = 1.0,
    provenance: LabelProvenanceType = LabelProvenanceType.AUTOMATED_MODEL,
    **kwargs
) -> TherapeuticResponseLabel:
    """Helper function to create therapeutic response labels"""
    metadata = LabelMetadata(
        confidence=confidence,
        provenance=provenance,
        **kwargs.get('metadata_kwargs', {})
    )
    return TherapeuticResponseLabel(response_type=response_type, metadata=metadata)


def create_crisis_label(
    crisis_level: CrisisLevelType,
    confidence: float = 1.0,
    provenance: LabelProvenanceType = LabelProvenanceType.AUTOMATED_MODEL,
    **kwargs
) -> CrisisLabel:
    """Helper function to create crisis labels"""
    metadata = LabelMetadata(
        confidence=confidence,
        provenance=provenance,
        **kwargs.get('metadata_kwargs', {})
    )
    return CrisisLabel(crisis_level=crisis_level, metadata=metadata)