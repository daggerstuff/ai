"""
Dataset Taxonomy System for Pixelated Empathy AI
Defines comprehensive categorization and metadata structures for therapeutic datasets
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Set, Union
from enum import Enum
from datetime import datetime
import json
import hashlib
from pathlib import Path
import uuid


class DatasetCategory(Enum):
    """Primary dataset categories"""
    CLINICAL = "clinical"
    CONVERSATIONAL = "conversational"
    THERAPEUTIC = "therapeutic"
    SYNTHETIC = "synthetic"
    MULTIMODAL = "multimodal"


class ClinicalSubcategory(Enum):
    """Clinical data subcategories"""
    MENTAL_HEALTH_RECORDS = "mental_health_records"
    THERAPY_SESSIONS = "therapy_sessions"
    CRISIS_INTERVENTION = "crisis_intervention"
    DIAGNOSTIC_ASSESSMENTS = "diagnostic_assessments"
    TREATMENT_PLANS = "treatment_plans"


class ConversationalSubcategory(Enum):
    """Conversational data subcategories"""
    EMPATHETIC_CONVERSATIONS = "empathetic_conversations"
    THERAPEUTIC_CONVERSATIONS = "therapeutic_conversations"
    SUPPORT_INTERACTIONS = "support_interactions"
    COUNSELING_SESSIONS = "counseling_sessions"
    PEER_SUPPORT = "peer_support"


class TherapeuticSubcategory(Enum):
    """Therapeutic approach subcategories"""
    CBT = "cognitive_behavioral_therapy"
    DBT = "dialectical_behavior_therapy"
    PSYCHODYNAMIC = "psychodynamic_therapy"
    HUMANISTIC = "humanistic_therapy"
    FAMILY_THERAPY = "family_therapy"
    GROUP_THERAPY = "group_therapy"


class SyntheticSubcategory(Enum):
    """Synthetic data generation methods"""
    AI_GENERATED = "ai_generated"
    DATA_AUGMENTATION = "data_augmentation"
    BALANCED_SYNTHETIC = "balanced_synthetic"
    PRIVACY_PRESERVING = "privacy_preserving"
    DEMOGRAPHICALLY_BALANCED = "demographically_balanced"


class MultimodalSubcategory(Enum):
    """Multimodal data types"""
    TEXT_AUDIO = "text_audio"
    TEXT_VIDEO = "text_video"
    MULTILINGUAL = "multilingual"
    CULTURAL_CONTEXT = "cultural_context"


class DataFormat(Enum):
    """Supported data formats"""
    JSON = "json"
    JSONL = "jsonl"
    CSV = "csv"
    PARQUET = "parquet"
    HDF5 = "hdf5"
    SQLITE = "sqlite"


class Language(Enum):
    """Supported languages"""
    ENGLISH = "en"
    SPANISH = "es"
    FRENCH = "fr"
    GERMAN = "de"
    CHINESE = "zh"
    JAPANESE = "ja"
    MULTILINGUAL = "multi"


class TherapeuticDomain(Enum):
    """Therapeutic domains and specializations"""
    ANXIETY = "anxiety"
    DEPRESSION = "depression"
    TRAUMA = "trauma"
    ADDICTION = "addiction"
    RELATIONSHIPS = "relationships"
    GRIEF = "grief"
    STRESS = "stress"
    SELF_ESTEEM = "self_esteem"
    ANGER = "anger"
    EATING_DISORDERS = "eating_disorders"


class DemographicGroup(Enum):
    """Demographic groups for bias analysis"""
    AGE_YOUNG = "age_young"
    AGE_MIDDLE = "age_middle"
    AGE_SENIOR = "age_senior"
    GENDER_MALE = "gender_male"
    GENDER_FEMALE = "gender_female"
    GENDER_NONBINARY = "gender_nonbinary"
    ETHNICITY_WHITE = "ethnicity_white"
    ETHNICITY_BLACK = "ethnicity_black"
    ETHNICITY_HISPANIC = "ethnicity_hispanic"
    ETHNICITY_ASIAN = "ethnicity_asian"
    ETHNICITY_OTHER = "ethnicity_other"
    SOCIOECONOMIC_LOW = "socioeconomic_low"
    SOCIOECONOMIC_MIDDLE = "socioeconomic_middle"
    SOCIOECONOMIC_HIGH = "socioeconomic_high"


@dataclass
class DatasetMetadata:
    """Comprehensive dataset metadata"""
    dataset_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    name: str = ""
    description: str = ""
    category: DatasetCategory = DatasetCategory.CONVERSATIONAL
    subcategory: Optional[str] = None
    version: str = "1.0"
    created_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    updated_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    created_by: str = field(default_factory=lambda: "system")
    
    # Content metadata
    language: Language = Language.ENGLISH
    domains: List[TherapeuticDomain] = field(default_factory=list)
    formats: List[DataFormat] = field(default_factory=list)
    record_count: Optional[int] = None
    size_bytes: Optional[int] = None
    avg_record_length: Optional[float] = None
    max_record_length: Optional[int] = None
    
    # Quality metadata
    quality_score: float = 0.0
    completeness_score: float = 0.0
    consistency_score: float = 0.0
    relevance_score: float = 0.0
    
    # Bias and fairness metadata
    demographic_balance: Dict[str, float] = field(default_factory=dict)
    bias_indicators: List[str] = field(default_factory=list)
    fairness_score: float = 0.0
    
    # Privacy and consent metadata
    consent_status: str = "unknown"
    anonymization_level: str = "none"
    pii_detected: bool = False
    privacy_compliance: List[str] = field(default_factory=list)
    
    # Therapeutic metadata
    therapeutic_approaches: List[TherapeuticSubcategory] = field(default_factory=list)
    crisis_content_ratio: float = 0.0
    empathy_indicators: List[str] = field(default_factory=list)
    therapeutic_quality_score: float = 0.0
    
    # Source metadata
    source_type: str = "unknown"
    source_url: Optional[str] = None
    source_license: str = "unknown"
    source_attribution: str = ""
    
    # Technical metadata
    checksum: str = ""
    compression: Optional[str] = None
    encoding: str = "utf-8"
    schema_version: str = "1.0"
    
    # Training metadata
    training_suitability: Dict[str, float] = field(default_factory=dict)
    recommended_styles: List[str] = field(default_factory=list)
    preprocessing_required: List[str] = field(default_factory=list)
    
    # Additional metadata
    tags: List[str] = field(default_factory=list)
    custom_metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        if not self.checksum and self.name:
            self.checksum = self._calculate_checksum()
    
    def _calculate_checksum(self) -> str:
        """Calculate checksum for metadata validation"""
        content = f"{self.name}_{self.version}_{self.category.value}_{self.created_at}"
        return hashlib.sha256(content.encode()).hexdigest()[:16]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert metadata to dictionary"""
        return {
            "dataset_id": self.dataset_id,
            "name": self.name,
            "description": self.description,
            "category": self.category.value,
            "subcategory": self.subcategory,
            "version": self.version,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "created_by": self.created_by,
            "language": self.language.value,
            "domains": [domain.value for domain in self.domains],
            "formats": [format.value for format in self.formats],
            "record_count": self.record_count,
            "size_bytes": self.size_bytes,
            "avg_record_length": self.avg_record_length,
            "max_record_length": self.max_record_length,
            "quality_score": self.quality_score,
            "completeness_score": self.completeness_score,
            "consistency_score": self.consistency_score,
            "relevance_score": self.relevance_score,
            "demographic_balance": self.demographic_balance,
            "bias_indicators": self.bias_indicators,
            "fairness_score": self.fairness_score,
            "consent_status": self.consent_status,
            "anonymization_level": self.anonymization_level,
            "pii_detected": self.pii_detected,
            "privacy_compliance": self.privacy_compliance,
            "therapeutic_approaches": [approach.value for approach in self.therapeutic_approaches],
            "crisis_content_ratio": self.crisis_content_ratio,
            "empathy_indicators": self.empathy_indicators,
            "therapeutic_quality_score": self.therapeutic_quality_score,
            "source_type": self.source_type,
            "source_url": self.source_url,
            "source_license": self.source_license,
            "source_attribution": self.source_attribution,
            "checksum": self.checksum,
            "compression": self.compression,
            "encoding": self.encoding,
            "schema_version": self.schema_version,
            "training_suitability": self.training_suitability,
            "recommended_styles": self.recommended_styles,
            "preprocessing_required": self.preprocessing_required,
            "tags": self.tags,
            "custom_metadata": self.custom_metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'DatasetMetadata':
        """Create metadata from dictionary"""
        # Handle enum conversions
        category = DatasetCategory(data.get('category', 'conversational'))
        language = Language(data.get('language', 'en'))
        
        domains = [TherapeuticDomain(domain) for domain in data.get('domains', [])]
        formats = [DataFormat(format) for format in data.get('formats', [])]
        approaches = [TherapeuticSubcategory(approach) for approach in data.get('therapeutic_approaches', [])]
        
        return cls(
            dataset_id=data.get('dataset_id', str(uuid.uuid4())),
            name=data.get('name', ''),
            description=data.get('description', ''),
            category=category,
            subcategory=data.get('subcategory'),
            version=data.get('version', '1.0'),
            created_at=data.get('created_at', datetime.utcnow().isoformat()),
            updated_at=data.get('updated_at', datetime.utcnow().isoformat()),
            created_by=data.get('created_by', 'system'),
            language=language,
            domains=domains,
            formats=formats,
            record_count=data.get('record_count'),
            size_bytes=data.get('size_bytes'),
            avg_record_length=data.get('avg_record_length'),
            max_record_length=data.get('max_record_length'),
            quality_score=data.get('quality_score', 0.0),
            completeness_score=data.get('completeness_score', 0.0),
            consistency_score=data.get('consistency_score', 0.0),
            relevance_score=data.get('relevance_score', 0.0),
            demographic_balance=data.get('demographic_balance', {}),
            bias_indicators=data.get('bias_indicators', []),
            fairness_score=data.get('fairness_score', 0.0),
            consent_status=data.get('consent_status', 'unknown'),
            anonymization_level=data.get('anonymization_level', 'none'),
            pii_detected=data.get('pii_detected', False),
            privacy_compliance=data.get('privacy_compliance', []),
            therapeutic_approaches=approaches,
            crisis_content_ratio=data.get('crisis_content_ratio', 0.0),
            empathy_indicators=data.get('empathy_indicators', []),
            therapeutic_quality_score=data.get('therapeutic_quality_score', 0.0),
            source_type=data.get('source_type', 'unknown'),
            source_url=data.get('source_url'),
            source_license=data.get('source_license', 'unknown'),
            source_attribution=data.get('source_attribution', ''),
            checksum=data.get('checksum', ''),
            compression=data.get('compression'),
            encoding=data.get('encoding', 'utf-8'),
            schema_version=data.get('schema_version', '1.0'),
            training_suitability=data.get('training_suitability', {}),
            recommended_styles=data.get('recommended_styles', []),
            preprocessing_required=data.get('preprocessing_required', []),
            tags=data.get('tags', []),
            custom_metadata=data.get('custom_metadata', {})
        )


class DatasetTaxonomy:
    """Comprehensive dataset taxonomy management system"""
    
    def __init__(self):
        self.category_hierarchy = self._build_category_hierarchy()
        self.validation_rules = self._build_validation_rules()
        self.recommendation_engine = self._build_recommendation_engine()
    
    def _build_category_hierarchy(self) -> Dict[str, Dict[str, Any]]:
        """Build the complete category hierarchy"""
        return {
            DatasetCategory.CLINICAL.value: {
                "subcategories": [sub.value for sub in ClinicalSubcategory],
                "description": "Clinical and medical data",
                "privacy_level": "maximum",
                "validation_strict": True,
                "therapeutic_domains": list(TherapeuticDomain),
                "required_metadata": ["consent_status", "anonymization_level", "therapeutic_approaches"]
            },
            DatasetCategory.CONVERSATIONAL.value: {
                "subcategories": [sub.value for sub in ConversationalSubcategory],
                "description": "General conversational data",
                "privacy_level": "medium",
                "validation_strict": False,
                "therapeutic_domains": [TherapeuticDomain.ANXIETY, TherapeuticDomain.DEPRESSION, 
                                      TherapeuticDomain.RELATIONSHIPS, TherapeuticDomain.STRESS],
                "required_metadata": ["quality_score", "empathy_indicators"]
            },
            DatasetCategory.THERAPEUTIC.value: {
                "subcategories": [sub.value for sub in TherapeuticSubcategory],
                "description": "Therapy-specific data",
                "privacy_level": "maximum",
                "validation_strict": True,
                "therapeutic_domains": list(TherapeuticDomain),
                "required_metadata": ["therapeutic_approaches", "therapeutic_quality_score", "crisis_content_ratio"]
            },
            DatasetCategory.SYNTHETIC.value: {
                "subcategories": [sub.value for sub in SyntheticSubcategory],
                "description": "Synthetic and generated data",
                "privacy_level": "low",
                "validation_strict": False,
                "therapeutic_domains": list(TherapeuticDomain),
                "required_metadata": ["source_type", "generation_method"]
            },
            DatasetCategory.MULTIMODAL.value: {
                "subcategories": [sub.value for sub in MultimodalSubcategory],
                "description": "Multimodal data",
                "privacy_level": "medium",
                "validation_strict": True,
                "therapeutic_domains": list(TherapeuticDomain),
                "required_metadata": ["modalities", "synchronization_quality"]
            }
        }
    
    def _build_validation_rules(self) -> Dict[str, Dict[str, Any]]:
        """Build validation rules for each category"""
        return {
            DatasetCategory.CLINICAL.value: {
                "min_quality_score": 0.8,
                "max_crisis_content_ratio": 0.3,
                "required_consent": True,
                "anonymization_required": True,
                "bias_threshold": 0.1,
                "fairness_threshold": 0.8
            },
            DatasetCategory.CONVERSATIONAL.value: {
                "min_quality_score": 0.6,
                "max_crisis_content_ratio": 0.2,
                "required_consent": True,
                "anonymization_required": False,
                "bias_threshold": 0.15,
                "fairness_threshold": 0.7
            },
            DatasetCategory.THERAPEUTIC.value: {
                "min_quality_score": 0.85,
                "max_crisis_content_ratio": 0.25,
                "required_consent": True,
                "anonymization_required": True,
                "bias_threshold": 0.08,
                "fairness_threshold": 0.85
            },
            DatasetCategory.SYNTHETIC.value: {
                "min_quality_score": 0.7,
                "max_crisis_content_ratio": 0.15,
                "required_consent": False,
                "anonymization_required": False,
                "bias_threshold": 0.2,
                "fairness_threshold": 0.75
            },
            DatasetCategory.MULTIMODAL.value: {
                "min_quality_score": 0.75,
                "max_crisis_content_ratio": 0.2,
                "required_consent": True,
                "anonymization_required": True,
                "bias_threshold": 0.12,
                "fairness_threshold": 0.8
            }
        }
    
    def _build_recommendation_engine(self) -> Dict[str, List[str]]:
        """Build training style recommendations for each category"""
        return {
            DatasetCategory.CLINICAL.value: [
                "supervised", "transfer_learning", "few_shot", "continual_learning"
            ],
            DatasetCategory.CONVERSATIONAL.value: [
                "self_supervised", "supervised", "transfer_learning", "meta_learning"
            ],
            DatasetCategory.THERAPEUTIC.value: [
                "supervised", "few_shot", "transfer_learning", "reinforcement_learning"
            ],
            DatasetCategory.SYNTHETIC.value: [
                "self_supervised", "supervised", "meta_learning", "continual_learning"
            ],
            DatasetCategory.MULTIMODAL.value: [
                "supervised", "transfer_learning", "self_supervised", "meta_learning"
            ]
        }
    
    def categorize_dataset(self, content_sample: List[Dict[str, Any]], 
                          metadata: DatasetMetadata) -> Dict[str, Any]:
        """Automatically categorize a dataset based on content and metadata"""
        category_scores = {}
        
        # Analyze content patterns
        content_analysis = self._analyze_content_patterns(content_sample)
        
        # Score each category
        for category in DatasetCategory:
            score = self._score_category_fit(category, content_analysis, metadata)
            category_scores[category.value] = score
        
        # Determine best category
        best_category = max(category_scores, key=category_scores.get)
        confidence = category_scores[best_category]
        
        # Recommend subcategories
        recommended_subcategories = self._recommend_subcategories(best_category, content_analysis)
        
        return {
            "category": best_category,
            "confidence": confidence,
            "category_scores": category_scores,
            "recommended_subcategories": recommended_subcategories,
            "analysis_summary": content_analysis
        }
    
    def _analyze_content_patterns(self, content_sample: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze content patterns for categorization"""
        analysis = {
            "clinical_indicators": 0,
            "therapeutic_indicators": 0,
            "conversational_indicators": 0,
            "synthetic_indicators": 0,
            "multimodal_indicators": 0,
            "crisis_indicators": 0,
            "empathy_indicators": 0,
            "technical_indicators": 0
        }
        
        for item in content_sample:
            text_content = self._extract_text_content(item)
            
            # Clinical indicators
            if any(term in text_content.lower() for term in [
                "diagnosis", "treatment", "medication", "symptom", "disorder", "clinical"
            ]):
                analysis["clinical_indicators"] += 1
            
            # Therapeutic indicators
            if any(term in text_content.lower() for term in [
                "therapy", "therapeutic", "cbt", "dbt", "counseling", "session"
            ]):
                analysis["therapeutic_indicators"] += 1
            
            # Conversational indicators
            if any(term in text_content.lower() for term in [
                "conversation", "dialogue", "chat", "talk", "discuss"
            ]):
                analysis["conversational_indicators"] += 1
            
            # Crisis indicators
            if any(term in text_content.lower() for term in [
                "crisis", "emergency", "suicide", "harm", "danger"
            ]):
                analysis["crisis_indicators"] += 1
            
            # Empathy indicators
            if any(term in text_content.lower() for term in [
                "understand", "feel", "empathy", "compassion", "support"
            ]):
                analysis["empathy_indicators"] += 1
        
        # Normalize scores
        total_items = len(content_sample)
        if total_items > 0:
            for key in analysis:
                analysis[key] = analysis[key] / total_items
        
        return analysis
    
    def _extract_text_content(self, item: Dict[str, Any]) -> str:
        """Extract text content from various formats"""
        if isinstance(item, str):
            return item
        elif isinstance(item, dict):
            # Try common text fields
            for field in ["text", "content", "message", "conversation", "transcript"]:
                if field in item and isinstance(item[field], str):
                    return item[field]
                elif field in item and isinstance(item[field], list):
                    return " ".join(str(i) for i in item[field] if isinstance(i, str))
        return ""
    
    def _score_category_fit(self, category: DatasetCategory, 
                           content_analysis: Dict[str, Any], 
                           metadata: DatasetMetadata) -> float:
        """Score how well a category fits the dataset"""
        score = 0.0
        
        if category == DatasetCategory.CLINICAL:
            score += content_analysis["clinical_indicators"] * 0.4
            score += content_analysis["crisis_indicators"] * 0.3
            score += (1.0 if metadata.privacy_compliance else 0.0) * 0.2
            score += (metadata.quality_score or 0.0) * 0.1
        
        elif category == DatasetCategory.CONVERSATIONAL:
            score += content_analysis["conversational_indicators"] * 0.5
            score += content_analysis["empathy_indicators"] * 0.3
            score += (metadata.quality_score or 0.0) * 0.2
        
        elif category == DatasetCategory.THERAPEUTIC:
            score += content_analysis["therapeutic_indicators"] * 0.6
            score += content_analysis["empathy_indicators"] * 0.2
            score += (metadata.therapeutic_quality_score or 0.0) * 0.2
        
        elif category == DatasetCategory.SYNTHETIC:
            # Synthetic data often has different characteristics
            score += (1.0 if metadata.source_type == "synthetic" else 0.1) * 0.7
            score += (metadata.quality_score or 0.0) * 0.3
        
        elif category == DatasetCategory.MULTIMODAL:
            # Check for multimodal indicators
            score += (1.0 if len(metadata.formats) > 1 else 0.0) * 0.5
            score += (1.0 if metadata.language == Language.MULTILINGUAL else 0.0) * 0.3
            score += (metadata.quality_score or 0.0) * 0.2
        
        return min(score, 1.0)  # Cap at 1.0
    
    def _recommend_subcategories(self, category: str, 
                                content_analysis: Dict[str, Any]) -> List[str]:
        """Recommend subcategories based on content analysis"""
        recommendations = []
        
        if category == DatasetCategory.CLINICAL.value:
            if content_analysis["crisis_indicators"] > 0.1:
                recommendations.append(ClinicalSubcategory.CRISIS_INTERVENTION.value)
            if content_analysis["clinical_indicators"] > 0.3:
                recommendations.append(ClinicalSubcategory.MENTAL_HEALTH_RECORDS.value)
            if content_analysis["therapeutic_indicators"] > 0.2:
                recommendations.append(ClinicalSubcategory.THERAPY_SESSIONS.value)
        
        elif category == DatasetCategory.CONVERSATIONAL.value:
            if content_analysis["empathy_indicators"] > 0.4:
                recommendations.append(ConversationalSubcategory.EMPATHETIC_CONVERSATIONS.value)
            if content_analysis["therapeutic_indicators"] > 0.2:
                recommendations.append(ConversationalSubcategory.THERAPEUTIC_CONVERSATIONS.value)
            recommendations.append(ConversationalSubcategory.SUPPORT_INTERACTIONS.value)
        
        elif category == DatasetCategory.THERAPEUTIC.value:
            # Default to CBT as most common
            recommendations.append(TherapeuticSubcategory.CBT.value)
            recommendations.append(TherapeuticSubcategory.DBT.value)
            recommendations.append(TherapeuticSubcategory.HUMANISTIC.value)
        
        return recommendations[:3]  # Return top 3 recommendations
    
    def validate_metadata(self, metadata: DatasetMetadata) -> List[str]:
        """Validate dataset metadata against taxonomy rules"""
        errors = []
        
        # Get validation rules for the category
        category_rules = self.validation_rules.get(metadata.category.value, {})
        
        # Check quality score
        min_quality = category_rules.get("min_quality_score", 0.0)
        if metadata.quality_score < min_quality:
            errors.append(f"Quality score {metadata.quality_score} below minimum {min_quality}")
        
        # Check crisis content ratio
        max_crisis = category_rules.get("max_crisis_content_ratio", 1.0)
        if metadata.crisis_content_ratio > max_crisis:
            errors.append(f"Crisis content ratio {metadata.crisis_content_ratio} exceeds maximum {max_crisis}")
        
        # Check consent requirements
        if category_rules.get("required_consent", False) and metadata.consent_status != "verified":
            errors.append("Consent verification required for this category")
        
        # Check anonymization requirements
        if category_rules.get("anonymization_required", False) and metadata.anonymization_level == "none":
            errors.append("Anonymization required for this category")
        
        # Check bias thresholds
        bias_threshold = category_rules.get("bias_threshold", 1.0)
        if metadata.fairness_score < (1.0 - bias_threshold):
            errors.append(f"Bias indicators exceed threshold for category {metadata.category.value}")
        
        # Check required metadata fields
        required_fields = self.category_hierarchy.get(metadata.category.value, {}).get("required_metadata", [])
        for field in required_fields:
            if field == "therapeutic_approaches" and not metadata.therapeutic_approaches:
                errors.append("Therapeutic approaches required for this category")
            elif field == "consent_status" and metadata.consent_status == "unknown":
                errors.append("Consent status required for this category")
            elif field == "anonymization_level" and metadata.anonymization_level == "none":
                errors.append("Anonymization level required for this category")
        
        return errors
    
    def get_training_recommendations(self, metadata: DatasetMetadata) -> List[str]:
        """Get training style recommendations for a dataset"""
        category = metadata.category.value
        recommendations = self.recommendation_engine.get(category, [])
        
        # Filter based on dataset characteristics
        filtered_recommendations = []
        
        for style in recommendations:
            if self._is_style_suitable(style, metadata):
                filtered_recommendations.append(style)
        
        return filtered_recommendations
    
    def _is_style_suitable(self, style: str, metadata: DatasetMetadata) -> bool:
        """Check if a training style is suitable for the dataset"""
        # Few-shot learning requires high-quality, well-labeled data
        if style == "few_shot" and metadata.quality_score < 0.8:
            return False
        
        # Self-supervised learning works well with large datasets
        if style == "self_supervised" and (metadata.record_count or 0) < 1000:
            return False
        
        # Reinforcement learning requires outcome data
        if style == "reinforcement" and not metadata.custom_metadata.get("has_outcomes", False):
            return False
        
        # Transfer learning requires pre-trained models
        if style == "transfer_learning" and not metadata.custom_metadata.get("has_pretrained_model", False):
            return False
        
        return True
    
    def get_category_info(self, category: str) -> Dict[str, Any]:
        """Get detailed information about a category"""
        return self.category_hierarchy.get(category, {})
    
    def list_all_categories(self) -> List[Dict[str, Any]]:
        """List all available categories with their information"""
        categories = []
        for category in DatasetCategory:
            info = self.get_category_info(category.value)
            categories.append({
                "category": category.value,
                "description": info.get("description", ""),
                "privacy_level": info.get("privacy_level", "unknown"),
                "subcategories": info.get("subcategories", []),
                "therapeutic_domains": [domain.value for domain in info.get("therapeutic_domains", [])]
            })
        return categories


# Example usage and testing
def test_dataset_taxonomy():
    """Test the dataset taxonomy system"""
    print("Testing Dataset Taxonomy System...")
    
    taxonomy = DatasetTaxonomy()
    
    # Test categorization
    sample_content = [
        {"text": "I'm feeling really anxious about my therapy session tomorrow"},
        {"text": "My therapist suggested trying CBT techniques for my depression"},
        {"text": "I learned some mindfulness exercises in today's session"},
        {"conversation": ["How are you feeling today?", "I'm struggling with anxiety"]}
    ]
    
    metadata = DatasetMetadata(
        name="Therapy Conversations Dataset",
        description="Collection of therapeutic conversation examples",
        language=Language.ENGLISH,
        domains=[TherapeuticDomain.ANXIETY, TherapeuticDomain.DEPRESSION],
        formats=[DataFormat.JSON],
        record_count=1000,
        quality_score=0.85,
        therapeutic_approaches=[TherapeuticSubcategory.CBT],
        consent_status="verified",
        anonymization_level="full"
    )
    
    categorization_result = taxonomy.categorize_dataset(sample_content, metadata)
    print(f"Recommended category: {categorization_result['category']}")
    print(f"Confidence: {categorization_result['confidence']}")
    print(f"Recommended subcategories: {categorization_result['recommended_subcategories']}")
    
    # Test validation
    validation_errors = taxonomy.validate_metadata(metadata)
    print(f"Validation errors: {validation_errors}")
    
    # Test training recommendations
    training_recommendations = taxonomy.get_training_recommendations(metadata)
    print(f"Training style recommendations: {training_recommendations}")
    
    # Test category info
    category_info = taxonomy.get_category_info("therapeutic")
    print(f"Therapeutic category info: {category_info}")
    
    print("Dataset taxonomy test completed!")


if __name__ == "__main__":
    test_dataset_taxonomy()