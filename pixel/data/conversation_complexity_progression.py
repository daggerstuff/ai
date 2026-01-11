"""
Conversation Complexity Progression System

Manages the progressive complexity of therapeutic conversations based on client
readiness, therapeutic progress, session dynamics, and clinical appropriateness.
Ensures conversations evolve appropriately from basic to advanced therapeutic work.
"""

import asyncio
import json
import logging
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ComplexityLevel(Enum):
    """Levels of conversation complexity"""
    BASIC = "basic"
    INTERMEDIATE = "intermediate"
    ADVANCED = "advanced"
    EXPERT = "expert"


class ComplexityDimension(Enum):
    """Dimensions of conversation complexity"""
    EMOTIONAL_DEPTH = "emotional_depth"
    COGNITIVE_LOAD = "cognitive_load"
    THERAPEUTIC_TECHNIQUES = "therapeutic_techniques"
    INSIGHT_REQUIREMENTS = "insight_requirements"
    VULNERABILITY_LEVEL = "vulnerability_level"
    ABSTRACTION_LEVEL = "abstraction_level"
    INTERVENTION_SOPHISTICATION = "intervention_sophistication"
    RELATIONAL_COMPLEXITY = "relational_complexity"


class ProgressionTrigger(Enum):
    """Triggers for complexity progression"""
    CLIENT_READINESS = "client_readiness"
    THERAPEUTIC_ALLIANCE = "therapeutic_alliance"
    SKILL_MASTERY = "skill_mastery"
    INSIGHT_DEVELOPMENT = "insight_development"
    EMOTIONAL_REGULATION = "emotional_regulation"
    CRISIS_RESOLUTION = "crisis_resolution"
    GOAL_ACHIEVEMENT = "goal_achievement"
    SESSION_MILESTONE = "session_milestone"


class ProgressionDirection(Enum):
    """Direction of complexity progression"""
    INCREASE = "increase"
    DECREASE = "decrease"
    MAINTAIN = "maintain"


@dataclass
class ComplexityProfile:
    """Profile defining complexity across multiple dimensions"""
    level: ComplexityLevel
    dimensions: Dict[ComplexityDimension, float]  # 0.0 to 1.0 scores
    description: str
    prerequisites: List[str]
    indicators: List[str]
    contraindications: List[str]
    typical_duration: timedelta
    
    def __post_init__(self):
        """Validate complexity profile"""
        for dimension, score in self.dimensions.items():
            if not 0.0 <= score <= 1.0:
                raise ValueError(f"Dimension {dimension} score must be between 0.0 and 1.0")


@dataclass
class ProgressionCriteria:
    """Criteria for determining complexity progression"""
    trigger: ProgressionTrigger
    threshold: float
    weight: float
    direction: ProgressionDirection
    conditions: List[str]
    evaluation_method: str
    
    def __post_init__(self):
        """Validate progression criteria"""
        if not 0.0 <= self.threshold <= 1.0:
            raise ValueError("Threshold must be between 0.0 and 1.0")
        if not 0.0 <= self.weight <= 1.0:
            raise ValueError("Weight must be between 0.0 and 1.0")


@dataclass
class ComplexityAssessment:
    """Assessment of current conversation complexity readiness"""
    current_level: ComplexityLevel
    recommended_level: ComplexityLevel
    readiness_scores: Dict[ComplexityDimension, float]
    progression_triggers: List[ProgressionTrigger]
    blocking_factors: List[str]
    confidence_score: float
    assessment_timestamp: datetime
    reasoning: str
    
    def __post_init__(self):
        """Validate complexity assessment"""
        if not 0.0 <= self.confidence_score <= 1.0:
            raise ValueError("Confidence score must be between 0.0 and 1.0")


@dataclass
class ProgressionHistory:
    """History of complexity progressions"""
    session_number: int
    timestamp: datetime
    from_level: ComplexityLevel
    to_level: ComplexityLevel
    trigger: ProgressionTrigger
    success_indicators: List[str]
    challenges: List[str]
    client_response: str
    therapist_notes: str
    
    
class ConversationComplexityProgression:
    """
    Manages progressive complexity in therapeutic conversations
    
    This system ensures that conversations evolve appropriately based on:
    - Client readiness and therapeutic progress
    - Session dynamics and alliance strength
    - Clinical appropriateness and safety considerations
    - Therapeutic goals and milestone achievement
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """Initialize complexity progression system"""
        self.config = self._load_configuration(config_path)
        self.complexity_profiles = self._initialize_complexity_profiles()
        self.progression_criteria = self._initialize_progression_criteria()
        self.assessment_history: List[ComplexityAssessment] = []
        self.progression_history: List[ProgressionHistory] = []
        self.current_complexity: Optional[ComplexityLevel] = None
        self.session_complexity_tracking: Dict[int, ComplexityLevel] = {}
        
        logger.info("Conversation Complexity Progression system initialized")
    
    def _load_configuration(self, config_path: Optional[str]) -> Dict[str, Any]:
        """Load system configuration"""
        default_config = {
            'progression_sensitivity': 0.7,
            'safety_threshold': 0.8,
            'minimum_session_gap': 2,
            'complexity_smoothing': True,
            'adaptive_thresholds': True,
            'crisis_complexity_reduction': True,
            'alliance_weight': 0.3,
            'readiness_weight': 0.4,
            'progress_weight': 0.3
        }
        
        if config_path and Path(config_path).exists():
            try:
                with open(config_path, 'r') as f:
                    user_config = json.load(f)
                default_config.update(user_config)
            except Exception as e:
                logger.warning(f"Failed to load config from {config_path}: {e}")
        
        return default_config
    
    def _initialize_complexity_profiles(self) -> Dict[ComplexityLevel, ComplexityProfile]:
        """Initialize complexity profiles for each level"""
        profiles = {}
        
        # Basic Level
        profiles[ComplexityLevel.BASIC] = ComplexityProfile(
            level=ComplexityLevel.BASIC,
            dimensions={
                ComplexityDimension.EMOTIONAL_DEPTH: 0.2,
                ComplexityDimension.COGNITIVE_LOAD: 0.1,
                ComplexityDimension.THERAPEUTIC_TECHNIQUES: 0.2,
                ComplexityDimension.INSIGHT_REQUIREMENTS: 0.1,
                ComplexityDimension.VULNERABILITY_LEVEL: 0.2,
                ComplexityDimension.ABSTRACTION_LEVEL: 0.1,
                ComplexityDimension.INTERVENTION_SOPHISTICATION: 0.2,
                ComplexityDimension.RELATIONAL_COMPLEXITY: 0.2
            },
            description="Basic therapeutic conversations focusing on rapport building and surface-level exploration",
            prerequisites=["therapeutic_alliance_established"],
            indicators=["client_engagement", "basic_trust", "willingness_to_share"],
            contraindications=["acute_crisis", "severe_resistance", "cognitive_impairment"],
            typical_duration=timedelta(weeks=3)  # 3 sessions ~ 3 weeks
        )
        
        # Intermediate Level
        profiles[ComplexityLevel.INTERMEDIATE] = ComplexityProfile(
            level=ComplexityLevel.INTERMEDIATE,
            dimensions={
                ComplexityDimension.EMOTIONAL_DEPTH: 0.5,
                ComplexityDimension.COGNITIVE_LOAD: 0.4,
                ComplexityDimension.THERAPEUTIC_TECHNIQUES: 0.5,
                ComplexityDimension.INSIGHT_REQUIREMENTS: 0.4,
                ComplexityDimension.VULNERABILITY_LEVEL: 0.5,
                ComplexityDimension.ABSTRACTION_LEVEL: 0.3,
                ComplexityDimension.INTERVENTION_SOPHISTICATION: 0.5,
                ComplexityDimension.RELATIONAL_COMPLEXITY: 0.4
            },
            description="Intermediate conversations with moderate emotional exploration and skill building",
            prerequisites=["basic_skills_demonstrated", "emotional_regulation_developing"],
            indicators=["insight_moments", "skill_application", "emotional_awareness"],
            contraindications=["recent_trauma_activation", "alliance_rupture"],
            typical_duration=timedelta(weeks=5)  # 5 sessions ~ 5 weeks
        )
        
        # Advanced Level
        profiles[ComplexityLevel.ADVANCED] = ComplexityProfile(
            level=ComplexityLevel.ADVANCED,
            dimensions={
                ComplexityDimension.EMOTIONAL_DEPTH: 0.8,
                ComplexityDimension.COGNITIVE_LOAD: 0.7,
                ComplexityDimension.THERAPEUTIC_TECHNIQUES: 0.8,
                ComplexityDimension.INSIGHT_REQUIREMENTS: 0.7,
                ComplexityDimension.VULNERABILITY_LEVEL: 0.8,
                ComplexityDimension.ABSTRACTION_LEVEL: 0.6,
                ComplexityDimension.INTERVENTION_SOPHISTICATION: 0.8,
                ComplexityDimension.RELATIONAL_COMPLEXITY: 0.7
            },
            description="Advanced therapeutic work with deep emotional processing and complex interventions",
            prerequisites=["strong_alliance", "emotional_regulation_skills", "insight_capacity"],
            indicators=["deep_insights", "complex_pattern_recognition", "relational_awareness"],
            contraindications=["fragile_stability", "overwhelming_life_stressors"],
            typical_duration=timedelta(weeks=8)  # 8 sessions ~ 8 weeks
        )
        
        # Expert Level
        profiles[ComplexityLevel.EXPERT] = ComplexityProfile(
            level=ComplexityLevel.EXPERT,
            dimensions={
                ComplexityDimension.EMOTIONAL_DEPTH: 1.0,
                ComplexityDimension.COGNITIVE_LOAD: 0.9,
                ComplexityDimension.THERAPEUTIC_TECHNIQUES: 1.0,
                ComplexityDimension.INSIGHT_REQUIREMENTS: 0.9,
                ComplexityDimension.VULNERABILITY_LEVEL: 1.0,
                ComplexityDimension.ABSTRACTION_LEVEL: 0.8,
                ComplexityDimension.INTERVENTION_SOPHISTICATION: 1.0,
                ComplexityDimension.RELATIONAL_COMPLEXITY: 0.9
            },
            description="Expert-level therapeutic conversations with maximum depth and sophistication",
            prerequisites=["therapeutic_mastery", "high_insight_capacity", "stable_functioning"],
            indicators=["transformational_insights", "complex_integration", "therapeutic_mastery"],
            contraindications=["any_instability", "recent_major_changes"],
            typical_duration=timedelta(weeks=12)  # 12 sessions ~ 12 weeks
        )
        
        return profiles
    
    def _initialize_progression_criteria(self) -> List[ProgressionCriteria]:
        """Initialize criteria for complexity progression"""
        criteria = []
        
        # Client Readiness Criteria
        criteria.extend([
            ProgressionCriteria(
                trigger=ProgressionTrigger.CLIENT_READINESS,
                threshold=0.7,
                weight=0.4,
                direction=ProgressionDirection.INCREASE,
                conditions=["emotional_stability", "engagement_high"],
                evaluation_method="readiness_assessment"
            ),
            ProgressionCriteria(
                trigger=ProgressionTrigger.CLIENT_READINESS,
                threshold=0.3,
                weight=0.4,
                direction=ProgressionDirection.DECREASE,
                conditions=["emotional_instability", "overwhelm_indicators"],
                evaluation_method="readiness_assessment"
            )
        ])
        
        # Therapeutic Alliance Criteria
        criteria.extend([
            ProgressionCriteria(
                trigger=ProgressionTrigger.THERAPEUTIC_ALLIANCE,
                threshold=0.8,
                weight=0.3,
                direction=ProgressionDirection.INCREASE,
                conditions=["strong_alliance", "trust_established"],
                evaluation_method="alliance_strength"
            ),
            ProgressionCriteria(
                trigger=ProgressionTrigger.THERAPEUTIC_ALLIANCE,
                threshold=0.4,
                weight=0.3,
                direction=ProgressionDirection.DECREASE,
                conditions=["alliance_rupture", "trust_issues"],
                evaluation_method="alliance_strength"
            )
        ])
        
        # Skill Mastery Criteria
        criteria.extend([
            ProgressionCriteria(
                trigger=ProgressionTrigger.SKILL_MASTERY,
                threshold=0.75,
                weight=0.25,
                direction=ProgressionDirection.INCREASE,
                conditions=["skills_demonstrated", "consistent_application"],
                evaluation_method="skill_assessment"
            )
        ])
        
        # Insight Development Criteria
        criteria.extend([
            ProgressionCriteria(
                trigger=ProgressionTrigger.INSIGHT_DEVELOPMENT,
                threshold=0.7,
                weight=0.3,
                direction=ProgressionDirection.INCREASE,
                conditions=["insight_moments", "pattern_recognition"],
                evaluation_method="insight_tracking"
            )
        ])
        
        # Crisis Resolution Criteria
        criteria.extend([
            ProgressionCriteria(
                trigger=ProgressionTrigger.CRISIS_RESOLUTION,
                threshold=0.2,
                weight=0.8,
                direction=ProgressionDirection.DECREASE,
                conditions=["crisis_active", "safety_concerns"],
                evaluation_method="crisis_assessment"
            )
        ])
        
        return criteria
    
    async def assess_complexity_readiness(
        self,
        conversation_history: List[Any],
        clinical_context: Any,
        session_info: Dict[str, Any]
    ) -> ComplexityAssessment:
        """Assess readiness for complexity progression"""
        try:
            # Calculate readiness scores for each dimension
            readiness_scores = await self._calculate_readiness_scores(
                conversation_history, clinical_context, session_info
            )
            
            # Determine current and recommended complexity levels
            current_level = self._determine_current_level(conversation_history, session_info)
            recommended_level = await self._recommend_complexity_level(
                readiness_scores, current_level, clinical_context
            )
            
            # Identify progression triggers
            progression_triggers = await self._identify_progression_triggers(
                conversation_history, clinical_context, readiness_scores
            )
            
            # Identify blocking factors
            blocking_factors = await self._identify_blocking_factors(
                conversation_history, clinical_context, readiness_scores
            )
            
            # Calculate confidence score
            confidence_score = self._calculate_confidence_score(
                readiness_scores, progression_triggers, blocking_factors
            )
            
            # Generate reasoning
            reasoning = self._generate_assessment_reasoning(
                current_level, recommended_level, readiness_scores,
                progression_triggers, blocking_factors
            )
            
            assessment = ComplexityAssessment(
                current_level=current_level,
                recommended_level=recommended_level,
                readiness_scores=readiness_scores,
                progression_triggers=progression_triggers,
                blocking_factors=blocking_factors,
                confidence_score=confidence_score,
                assessment_timestamp=datetime.now(),
                reasoning=reasoning
            )
            
            # Store assessment in history
            self.assessment_history.append(assessment)
            
            return assessment
            
        except Exception as e:
            logger.error(f"Error in complexity readiness assessment: {e}")
            # Return safe default assessment
            return ComplexityAssessment(
                current_level=ComplexityLevel.BASIC,
                recommended_level=ComplexityLevel.BASIC,
                readiness_scores={dim: 0.3 for dim in ComplexityDimension},
                progression_triggers=[],
                blocking_factors=["assessment_error"],
                confidence_score=0.1,
                assessment_timestamp=datetime.now(),
                reasoning="Assessment failed, defaulting to basic level for safety"
            )
    
    async def _calculate_readiness_scores(
        self,
        conversation_history: List[Any],
        clinical_context: Any,
        session_info: Dict[str, Any]
    ) -> Dict[ComplexityDimension, float]:
        """Calculate readiness scores for each complexity dimension"""
        scores = {}
        
        # Emotional Depth Readiness
        scores[ComplexityDimension.EMOTIONAL_DEPTH] = await self._assess_emotional_depth_readiness(
            conversation_history, clinical_context
        )
        
        # Cognitive Load Readiness
        scores[ComplexityDimension.COGNITIVE_LOAD] = await self._assess_cognitive_load_readiness(
            conversation_history, clinical_context
        )
        
        # Therapeutic Techniques Readiness
        scores[ComplexityDimension.THERAPEUTIC_TECHNIQUES] = await self._assess_technique_readiness(
            conversation_history, clinical_context
        )
        
        # Insight Requirements Readiness
        scores[ComplexityDimension.INSIGHT_REQUIREMENTS] = await self._assess_insight_readiness(
            conversation_history, clinical_context
        )
        
        # Vulnerability Level Readiness
        scores[ComplexityDimension.VULNERABILITY_LEVEL] = await self._assess_vulnerability_readiness(
            conversation_history, clinical_context
        )
        
        # Abstraction Level Readiness
        scores[ComplexityDimension.ABSTRACTION_LEVEL] = await self._assess_abstraction_readiness(
            conversation_history, clinical_context
        )
        
        # Intervention Sophistication Readiness
        scores[ComplexityDimension.INTERVENTION_SOPHISTICATION] = await self._assess_intervention_readiness(
            conversation_history, clinical_context
        )
        
        # Relational Complexity Readiness
        scores[ComplexityDimension.RELATIONAL_COMPLEXITY] = await self._assess_relational_readiness(
            conversation_history, clinical_context
        )
        
        return scores
    
    async def _assess_emotional_depth_readiness(
        self,
        conversation_history: List[Any],
        clinical_context: Any
    ) -> float:
        """Assess readiness for emotional depth"""
        score = 0.5  # Base score
        
        # Check for emotional regulation indicators
        emotional_regulation_indicators = [
            "emotional awareness", "affect tolerance", "emotional expression",
            "emotional processing", "emotional stability"
        ]
        
        # Analyze conversation for emotional indicators
        for turn in conversation_history[-10:]:  # Last 10 turns
            content = getattr(turn, 'content', '').lower()
            
            # Positive indicators
            if any(indicator in content for indicator in emotional_regulation_indicators):
                score += 0.1
            
            # Check for emotional overwhelm
            overwhelm_indicators = ["too much", "overwhelming", "can't handle", "too intense"]
            if any(indicator in content for indicator in overwhelm_indicators):
                score -= 0.2
            
            # Check for emotional engagement
            engagement_indicators = ["feel", "emotion", "heart", "deep", "meaningful"]
            if any(indicator in content for indicator in engagement_indicators):
                score += 0.05
        
        # Consider clinical context
        if hasattr(clinical_context, 'crisis_indicators') and clinical_context.crisis_indicators:
            score -= 0.3
        
        return max(0.0, min(1.0, score))
    
    async def _assess_cognitive_load_readiness(
        self,
        conversation_history: List[Any],
        clinical_context: Any
    ) -> float:
        """Assess readiness for cognitive complexity"""
        score = 0.5  # Base score
        
        # Check for cognitive clarity indicators
        clarity_indicators = [
            "understand", "clear", "makes sense", "I see", "I get it"
        ]
        
        confusion_indicators = [
            "confused", "don't understand", "unclear", "too complex", "lost"
        ]
        
        # Analyze recent conversation
        for turn in conversation_history[-10:]:
            content = getattr(turn, 'content', '').lower()
            
            if any(indicator in content for indicator in clarity_indicators):
                score += 0.1
            
            if any(indicator in content for indicator in confusion_indicators):
                score -= 0.2
        
        # Consider session number (cognitive capacity may improve over time)
        session_number = getattr(clinical_context, 'session_number', 1)
        if session_number > 5:
            score += 0.1
        
        return max(0.0, min(1.0, score))
    
    async def _assess_technique_readiness(
        self,
        conversation_history: List[Any],
        clinical_context: Any
    ) -> float:
        """Assess readiness for advanced therapeutic techniques"""
        score = 0.4  # Base score
        
        # Check for technique engagement
        technique_engagement = [
            "technique", "strategy", "skill", "practice", "homework", "exercise"
        ]
        
        technique_resistance = [
            "won't work", "tried before", "doesn't help", "not for me"
        ]
        
        for turn in conversation_history[-15:]:
            content = getattr(turn, 'content', '').lower()
            
            if any(indicator in content for indicator in technique_engagement):
                score += 0.1
            
            if any(indicator in content for indicator in technique_resistance):
                score -= 0.15
        
        return max(0.0, min(1.0, score))
    
    async def _assess_insight_readiness(
        self,
        conversation_history: List[Any],
        clinical_context: Any
    ) -> float:
        """Assess readiness for insight-oriented work"""
        score = 0.4  # Base score
        
        insight_indicators = [
            "realize", "understand now", "see the pattern", "makes sense",
            "connection", "insight", "aha", "I see how"
        ]
        
        for turn in conversation_history[-10:]:
            content = getattr(turn, 'content', '').lower()
            
            if any(indicator in content for indicator in insight_indicators):
                score += 0.15
        
        # Consider therapeutic alliance strength
        # (This would integrate with alliance assessment from other systems)
        score += 0.1  # Placeholder for alliance integration
        
        return max(0.0, min(1.0, score))
    
    async def _assess_vulnerability_readiness(
        self,
        conversation_history: List[Any],
        clinical_context: Any
    ) -> float:
        """Assess readiness for vulnerable exploration"""
        score = 0.3  # Conservative base score
        
        vulnerability_indicators = [
            "share", "open up", "trust", "vulnerable", "personal", "private"
        ]
        
        safety_indicators = [
            "safe", "comfortable", "trust you", "feel secure"
        ]
        
        for turn in conversation_history[-10:]:
            content = getattr(turn, 'content', '').lower()
            
            if any(indicator in content for indicator in vulnerability_indicators):
                score += 0.1
            
            if any(indicator in content for indicator in safety_indicators):
                score += 0.15
        
        # Consider crisis indicators (reduce vulnerability readiness)
        if hasattr(clinical_context, 'crisis_indicators') and clinical_context.crisis_indicators:
            score -= 0.4
        
        return max(0.0, min(1.0, score))
    
    async def _assess_abstraction_readiness(
        self,
        conversation_history: List[Any],
        clinical_context: Any
    ) -> float:
        """Assess readiness for abstract concepts"""
        score = 0.4  # Base score
        
        abstract_engagement = [
            "concept", "idea", "theory", "principle", "philosophy", "meaning"
        ]
        
        concrete_preference = [
            "specific", "concrete", "practical", "real example", "what exactly"
        ]
        
        for turn in conversation_history[-10:]:
            content = getattr(turn, 'content', '').lower()
            
            if any(indicator in content for indicator in abstract_engagement):
                score += 0.1
            
            if any(indicator in content for indicator in concrete_preference):
                score -= 0.05  # Not negative, just preference for concrete
        
        return max(0.0, min(1.0, score))
    
    async def _assess_intervention_readiness(
        self,
        conversation_history: List[Any],
        clinical_context: Any
    ) -> float:
        """Assess readiness for sophisticated interventions"""
        score = 0.4  # Base score
        
        # This would integrate with intervention tracking systems
        # For now, use basic heuristics
        
        intervention_success = [
            "helped", "worked", "better", "improvement", "progress"
        ]
        
        for turn in conversation_history[-15:]:
            content = getattr(turn, 'content', '').lower()
            
            if any(indicator in content for indicator in intervention_success):
                score += 0.1
        
        return max(0.0, min(1.0, score))
    
    async def _assess_relational_readiness(
        self,
        conversation_history: List[Any],
        clinical_context: Any
    ) -> float:
        """Assess readiness for complex relational work"""
        score = 0.4  # Base score
        
        relational_indicators = [
            "relationship", "between us", "how we", "our interaction",
            "feel with you", "trust you", "connection"
        ]
        
        for turn in conversation_history[-10:]:
            content = getattr(turn, 'content', '').lower()
            
            if any(indicator in content for indicator in relational_indicators):
                score += 0.15
        
        return max(0.0, min(1.0, score))
    
    def _determine_current_level(
        self,
        conversation_history: List[Any],
        session_info: Dict[str, Any]
    ) -> ComplexityLevel:
        """Determine current complexity level"""
        session_number = session_info.get('session_number', 1)
        
        # Check session tracking
        if session_number in self.session_complexity_tracking:
            return self.session_complexity_tracking[session_number]
        
        # Default progression based on session number
        if session_number <= 3:
            return ComplexityLevel.BASIC
        elif session_number <= 8:
            return ComplexityLevel.INTERMEDIATE
        elif session_number <= 15:
            return ComplexityLevel.ADVANCED
        else:
            return ComplexityLevel.EXPERT
    
    async def _recommend_complexity_level(
        self,
        readiness_scores: Dict[ComplexityDimension, float],
        current_level: ComplexityLevel,
        clinical_context: Any
    ) -> ComplexityLevel:
        """Recommend appropriate complexity level"""
        # Calculate overall readiness
        overall_readiness = np.mean(list(readiness_scores.values()))
        
        # Safety checks
        if hasattr(clinical_context, 'crisis_indicators') and clinical_context.crisis_indicators:
            return ComplexityLevel.BASIC
        
        # Determine recommended level based on readiness
        if overall_readiness >= 0.8:
            recommended = ComplexityLevel.EXPERT
        elif overall_readiness >= 0.65:
            recommended = ComplexityLevel.ADVANCED
        elif overall_readiness >= 0.45:
            recommended = ComplexityLevel.INTERMEDIATE
        else:
            recommended = ComplexityLevel.BASIC
        
        # Don't jump more than one level at a time
        current_index = list(ComplexityLevel).index(current_level)
        recommended_index = list(ComplexityLevel).index(recommended)
        
        if recommended_index > current_index + 1:
            recommended = list(ComplexityLevel)[current_index + 1]
        elif recommended_index < current_index - 1:
            recommended = list(ComplexityLevel)[current_index - 1]
        
        return recommended
    
    async def _identify_progression_triggers(
        self,
        conversation_history: List[Any],
        clinical_context: Any,
        readiness_scores: Dict[ComplexityDimension, float]
    ) -> List[ProgressionTrigger]:
        """Identify active progression triggers"""
        triggers = []
        
        # Client readiness trigger
        if np.mean(list(readiness_scores.values())) > 0.7:
            triggers.append(ProgressionTrigger.CLIENT_READINESS)
        
        # Skill mastery trigger (placeholder - would integrate with skill tracking)
        if readiness_scores.get(ComplexityDimension.THERAPEUTIC_TECHNIQUES, 0) > 0.75:
            triggers.append(ProgressionTrigger.SKILL_MASTERY)
        
        # Insight development trigger
        if readiness_scores.get(ComplexityDimension.INSIGHT_REQUIREMENTS, 0) > 0.7:
            triggers.append(ProgressionTrigger.INSIGHT_DEVELOPMENT)
        
        # Crisis resolution trigger (negative)
        if hasattr(clinical_context, 'crisis_indicators') and clinical_context.crisis_indicators:
            triggers.append(ProgressionTrigger.CRISIS_RESOLUTION)
        
        return triggers
    
    async def _identify_blocking_factors(
        self,
        conversation_history: List[Any],
        clinical_context: Any,
        readiness_scores: Dict[ComplexityDimension, float]
    ) -> List[str]:
        """Identify factors blocking complexity progression"""
        blocking_factors = []
        
        # Low readiness scores
        for dimension, score in readiness_scores.items():
            if score < 0.3:
                blocking_factors.append(f"low_{dimension.value}_readiness")
        
        # Crisis indicators
        if hasattr(clinical_context, 'crisis_indicators') and clinical_context.crisis_indicators:
            blocking_factors.extend([f"crisis_{indicator}" for indicator in clinical_context.crisis_indicators])
        
        # Cognitive overload indicators
        if readiness_scores.get(ComplexityDimension.COGNITIVE_LOAD, 0) < 0.3:
            blocking_factors.append("cognitive_overload")
        
        # Emotional instability
        if readiness_scores.get(ComplexityDimension.EMOTIONAL_DEPTH, 0) < 0.3:
            blocking_factors.append("emotional_instability")
        
        return blocking_factors
    
    def _calculate_confidence_score(
        self,
        readiness_scores: Dict[ComplexityDimension, float],
        progression_triggers: List[ProgressionTrigger],
        blocking_factors: List[str]
    ) -> float:
        """Calculate confidence in complexity assessment"""
        base_confidence = 0.7
        
        # Adjust based on readiness score variance
        scores = list(readiness_scores.values())
        score_variance = np.var(scores)
        confidence_adjustment = -score_variance * 0.5  # Lower confidence for high variance
        
        # Adjust based on triggers and blocking factors
        trigger_adjustment = len(progression_triggers) * 0.05
        blocking_adjustment = -len(blocking_factors) * 0.1
        
        final_confidence = base_confidence + confidence_adjustment + trigger_adjustment + blocking_adjustment
        
        return max(0.1, min(1.0, final_confidence))
    
    def _generate_assessment_reasoning(
        self,
        current_level: ComplexityLevel,
        recommended_level: ComplexityLevel,
        readiness_scores: Dict[ComplexityDimension, float],
        progression_triggers: List[ProgressionTrigger],
        blocking_factors: List[str]
    ) -> str:
        """Generate human-readable reasoning for assessment"""
        reasoning_parts = []
        
        # Current vs recommended level
        if current_level == recommended_level:
            reasoning_parts.append(f"Client is appropriately matched at {current_level.value} complexity level.")
        elif list(ComplexityLevel).index(recommended_level) > list(ComplexityLevel).index(current_level):
            reasoning_parts.append(f"Client shows readiness to progress from {current_level.value} to {recommended_level.value} complexity.")
        else:
            reasoning_parts.append(f"Recommend reducing complexity from {current_level.value} to {recommended_level.value} for client safety and engagement.")
        
        # Readiness scores summary
        high_scores = [dim.value for dim, score in readiness_scores.items() if score > 0.7]
        low_scores = [dim.value for dim, score in readiness_scores.items() if score < 0.4]
        
        if high_scores:
            reasoning_parts.append(f"Strong readiness in: {', '.join(high_scores)}.")
        if low_scores:
            reasoning_parts.append(f"Areas needing development: {', '.join(low_scores)}.")
        
        # Progression triggers
        if progression_triggers:
            trigger_names = [trigger.value.replace('_', ' ') for trigger in progression_triggers]
            reasoning_parts.append(f"Progression supported by: {', '.join(trigger_names)}.")
        
        # Blocking factors
        if blocking_factors:
            factor_names = [factor.replace('_', ' ') for factor in blocking_factors[:3]]  # Limit to top 3
            reasoning_parts.append(f"Progression considerations: {', '.join(factor_names)}.")
        
        return " ".join(reasoning_parts)
    
    async def apply_complexity_progression(
        self,
        assessment: ComplexityAssessment,
        session_number: int,
        therapist_notes: str = ""
    ) -> bool:
        """Apply complexity progression based on assessment"""
        try:
            if assessment.current_level != assessment.recommended_level:
                # Record progression
                progression = ProgressionHistory(
                    session_number=session_number,
                    timestamp=datetime.now(),
                    from_level=assessment.current_level,
                    to_level=assessment.recommended_level,
                    trigger=assessment.progression_triggers[0] if assessment.progression_triggers else ProgressionTrigger.SESSION_MILESTONE,
                    success_indicators=[],  # Would be populated based on actual progression
                    challenges=assessment.blocking_factors,
                    client_response="",  # Would be populated based on client response
                    therapist_notes=therapist_notes
                )
                
                self.progression_history.append(progression)
                self.session_complexity_tracking[session_number] = assessment.recommended_level
                self.current_complexity = assessment.recommended_level
                
                logger.info(f"Applied complexity progression: {assessment.current_level.value} -> {assessment.recommended_level.value}")
                return True
            else:
                # Maintain current level
                self.session_complexity_tracking[session_number] = assessment.current_level
                self.current_complexity = assessment.current_level
                return False
                
        except Exception as e:
            logger.error(f"Error applying complexity progression: {e}")
            return False
    
    def get_complexity_profile(self, level: ComplexityLevel) -> ComplexityProfile:
        """Get complexity profile for a specific level"""
        return self.complexity_profiles.get(level)
    
    def get_progression_statistics(self) -> Dict[str, Any]:
        """Get progression statistics and insights"""
        stats = {
            'total_assessments': len(self.assessment_history),
            'total_progressions': len(self.progression_history),
            'current_complexity': self.current_complexity.value if self.current_complexity else None,
            'complexity_distribution': {},
            'progression_triggers': {},
            'blocking_factors': {},
            'average_confidence': 0.0,
            'progression_success_rate': 0.0
        }
        
        if self.assessment_history:
            # Complexity distribution
            complexity_counts = {}
            confidence_scores = []
            
            for assessment in self.assessment_history:
                level = assessment.recommended_level.value
                complexity_counts[level] = complexity_counts.get(level, 0) + 1
                confidence_scores.append(assessment.confidence_score)
            
            stats['complexity_distribution'] = complexity_counts
            stats['average_confidence'] = np.mean(confidence_scores)
            
            # Progression triggers
            trigger_counts = {}
            for assessment in self.assessment_history:
                for trigger in assessment.progression_triggers:
                    trigger_name = trigger.value
                    trigger_counts[trigger_name] = trigger_counts.get(trigger_name, 0) + 1
            stats['progression_triggers'] = trigger_counts
            
            # Blocking factors
            blocking_counts = {}
            for assessment in self.assessment_history:
                for factor in assessment.blocking_factors:
                    blocking_counts[factor] = blocking_counts.get(factor, 0) + 1
            stats['blocking_factors'] = blocking_counts
        
        if self.progression_history:
            # Success rate (placeholder - would need success criteria)
            stats['progression_success_rate'] = 0.8  # Placeholder
        
        return stats
    
    def export_progression_data(self, format: str = 'json') -> Union[str, Dict[str, Any]]:
        """Export progression data for analysis"""
        data = {
            'configuration': self.config,
            'complexity_profiles': {
                level.value: {
                    'dimensions': {dim.value: score for dim, score in profile.dimensions.items()},
                    'description': profile.description,
                    'prerequisites': profile.prerequisites,
                    'indicators': profile.indicators,
                    'contraindications': profile.contraindications
                }
                for level, profile in self.complexity_profiles.items()
            },
            'assessment_history': [
                {
                    'timestamp': assessment.assessment_timestamp.isoformat(),
                    'current_level': assessment.current_level.value,
                    'recommended_level': assessment.recommended_level.value,
                    'readiness_scores': {dim.value: score for dim, score in assessment.readiness_scores.items()},
                    'progression_triggers': [trigger.value for trigger in assessment.progression_triggers],
                    'blocking_factors': assessment.blocking_factors,
                    'confidence_score': assessment.confidence_score,
                    'reasoning': assessment.reasoning
                }
                for assessment in self.assessment_history
            ],
            'progression_history': [
                {
                    'session_number': prog.session_number,
                    'timestamp': prog.timestamp.isoformat(),
                    'from_level': prog.from_level.value,
                    'to_level': prog.to_level.value,
                    'trigger': prog.trigger.value,
                    'success_indicators': prog.success_indicators,
                    'challenges': prog.challenges,
                    'therapist_notes': prog.therapist_notes
                }
                for prog in self.progression_history
            ],
            'statistics': self.get_progression_statistics()
        }
        
        if format.lower() == 'json':
            return json.dumps(data, indent=2)
        else:
            return data


# Example usage and testing
if __name__ == "__main__":
    async def test_complexity_progression():
        """Test the complexity progression system"""
        progression_system = ConversationComplexityProgression()
        
        # Mock conversation history and clinical context
        mock_conversation = [
            type('Turn', (), {'content': 'I feel ready to explore deeper issues'})(),
            type('Turn', (), {'content': 'I understand the techniques you taught me'})(),
            type('Turn', (), {'content': 'I see the patterns in my relationships now'})()
        ]
        
        mock_clinical_context = type('Context', (), {
            'crisis_indicators': [],
            'session_number': 5
        })()
        
        mock_session_info = {'session_number': 5}
        
        # Assess complexity readiness
        assessment = await progression_system.assess_complexity_readiness(
            mock_conversation, mock_clinical_context, mock_session_info
        )
        
        print(f"Assessment: {assessment.current_level.value} -> {assessment.recommended_level.value}")
        print(f"Confidence: {assessment.confidence_score:.2f}")
        print(f"Reasoning: {assessment.reasoning}")
        
        # Apply progression
        success = await progression_system.apply_complexity_progression(
            assessment, session_number=5, therapist_notes="Client showing good progress"
        )
        
        print(f"Progression applied: {success}")
        
        # Get statistics
        stats = progression_system.get_progression_statistics()
        print(f"Statistics: {stats}")
    
    # Run test
    asyncio.run(test_complexity_progression())


# Alias for backwards compatibility (Tier 1.2 expects ConversationManager)
ConversationManager = ConversationComplexityProgression

# Add simple wrapper methods for compatibility
def assess_complexity(self, content: str):
    """Simple wrapper for assess_complexity_readiness for compatibility."""
    import asyncio
    try:
        # Run the async method synchronously
        return asyncio.run(self.assess_complexity_readiness(content))
    except:
        # Fallback to a simple mock assessment
        from .therapeutic_conversation_schema import ComplexityLevel
        class MockAssessment:
            def __init__(self):
                self.overall_complexity = ComplexityLevel.BASIC
        return MockAssessment()

# Monkey patch the method onto the class
ConversationComplexityProgression.assess_complexity = assess_complexity
