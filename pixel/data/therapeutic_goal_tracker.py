"""
Therapeutic Goal Tracking System

Tracks therapeutic goals throughout conversations, monitors progress,
identifies milestones, and provides goal-oriented conversation guidance.
Integrates with clinical context and conversation flow for comprehensive
therapeutic progress monitoring.
"""

import asyncio
import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class GoalCategory(Enum):
    """Categories of therapeutic goals"""
    SYMPTOM_REDUCTION = "symptom_reduction"
    SKILL_BUILDING = "skill_building"
    INSIGHT_DEVELOPMENT = "insight_development"
    BEHAVIORAL_CHANGE = "behavioral_change"
    EMOTIONAL_REGULATION = "emotional_regulation"
    RELATIONSHIP_IMPROVEMENT = "relationship_improvement"
    COPING_STRATEGIES = "coping_strategies"
    TRAUMA_PROCESSING = "trauma_processing"
    SELF_AWARENESS = "self_awareness"
    FUNCTIONAL_IMPROVEMENT = "functional_improvement"


class GoalPriority(Enum):
    """Priority levels for therapeutic goals"""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class GoalStatus(Enum):
    """Status of therapeutic goals"""
    NOT_STARTED = "not_started"
    IN_PROGRESS = "in_progress"
    PARTIALLY_ACHIEVED = "partially_achieved"
    ACHIEVED = "achieved"
    ON_HOLD = "on_hold"
    DISCONTINUED = "discontinued"


class ProgressIndicator(Enum):
    """Types of progress indicators"""
    BEHAVIORAL_EVIDENCE = "behavioral_evidence"
    SELF_REPORT = "self_report"
    CLINICAL_OBSERVATION = "clinical_observation"
    SKILL_DEMONSTRATION = "skill_demonstration"
    INSIGHT_EXPRESSION = "insight_expression"
    EMOTIONAL_REGULATION = "emotional_regulation"
    FUNCTIONAL_IMPROVEMENT = "functional_improvement"
    RELATIONSHIP_FEEDBACK = "relationship_feedback"


class MilestoneType(Enum):
    """Types of therapeutic milestones"""
    INITIAL_ENGAGEMENT = "initial_engagement"
    SKILL_ACQUISITION = "skill_acquisition"
    INSIGHT_BREAKTHROUGH = "insight_breakthrough"
    BEHAVIORAL_CHANGE = "behavioral_change"
    SYMPTOM_IMPROVEMENT = "symptom_improvement"
    GOAL_ACHIEVEMENT = "goal_achievement"
    RELAPSE_PREVENTION = "relapse_prevention"
    MAINTENANCE = "maintenance"


@dataclass
class TherapeuticGoal:
    """Individual therapeutic goal with tracking information"""
    goal_id: str
    title: str
    description: str
    category: GoalCategory
    priority: GoalPriority
    target_date: Optional[datetime]
    created_date: datetime
    status: GoalStatus = GoalStatus.NOT_STARTED
    progress_percentage: float = 0.0
    success_criteria: List[str] = field(default_factory=list)
    barriers: List[str] = field(default_factory=list)
    interventions: List[str] = field(default_factory=list)
    milestones: List[str] = field(default_factory=list)
    notes: str = ""
    last_updated: datetime = field(default_factory=datetime.now)
    
    def __post_init__(self):
        """Validate therapeutic goal"""
        if not 0.0 <= self.progress_percentage <= 100.0:
            raise ValueError("Progress percentage must be between 0.0 and 100.0")


@dataclass
class ProgressMeasurement:
    """Measurement of progress toward a goal"""
    measurement_id: str
    goal_id: str
    session_number: int
    timestamp: datetime
    progress_score: float  # 0.0 to 1.0
    indicator_type: ProgressIndicator
    evidence: str
    confidence_level: float  # 0.0 to 1.0
    therapist_notes: str
    client_feedback: str = ""
    
    def __post_init__(self):
        """Validate progress measurement"""
        if not 0.0 <= self.progress_score <= 1.0:
            raise ValueError("Progress score must be between 0.0 and 1.0")
        if not 0.0 <= self.confidence_level <= 1.0:
            raise ValueError("Confidence level must be between 0.0 and 1.0")


@dataclass
class TherapeuticMilestone:
    """Therapeutic milestone achievement"""
    milestone_id: str
    goal_id: str
    milestone_type: MilestoneType
    title: str
    description: str
    achieved_date: datetime
    session_number: int
    evidence: List[str]
    significance_score: float  # 0.0 to 1.0
    client_recognition: bool
    therapist_notes: str
    
    def __post_init__(self):
        """Validate therapeutic milestone"""
        if not 0.0 <= self.significance_score <= 1.0:
            raise ValueError("Significance score must be between 0.0 and 1.0")


@dataclass
class GoalProgressSummary:
    """Summary of goal progress across sessions"""
    goal_id: str
    current_progress: float
    progress_trend: str  # "improving", "stable", "declining"
    recent_milestones: List[TherapeuticMilestone]
    barriers_identified: List[str]
    interventions_used: List[str]
    next_steps: List[str]
    estimated_completion: Optional[datetime]
    confidence_in_achievement: float
    
    def __post_init__(self):
        """Validate goal progress summary"""
        if not 0.0 <= self.current_progress <= 100.0:
            raise ValueError("Current progress must be between 0.0 and 100.0")
        if not 0.0 <= self.confidence_in_achievement <= 1.0:
            raise ValueError("Confidence in achievement must be between 0.0 and 1.0")


class TherapeuticGoalTracker:
    """
    Tracks therapeutic goals throughout conversations
    
    This system provides:
    - Goal creation and management
    - Progress tracking and measurement
    - Milestone identification and celebration
    - Barrier identification and intervention planning
    - Goal-oriented conversation guidance
    - Progress visualization and reporting
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """Initialize therapeutic goal tracker"""
        self.config = self._load_configuration(config_path)
        self.goals: Dict[str, TherapeuticGoal] = {}
        self.progress_measurements: List[ProgressMeasurement] = []
        self.milestones: List[TherapeuticMilestone] = []
        self.goal_templates = self._initialize_goal_templates()
        self.progress_patterns = self._initialize_progress_patterns()
        self.milestone_criteria = self._initialize_milestone_criteria()
        
        logger.info("Therapeutic Goal Tracker initialized")
    
    def _load_configuration(self, config_path: Optional[str]) -> Dict[str, Any]:
        """Load system configuration"""
        default_config = {
            'progress_smoothing_window': 3,
            'milestone_significance_threshold': 0.7,
            'barrier_detection_sensitivity': 0.6,
            'goal_completion_threshold': 0.9,
            'progress_trend_window': 5,
            'automatic_milestone_detection': True,
            'client_self_report_weight': 0.4,
            'clinical_observation_weight': 0.6,
            'goal_review_frequency': 3  # sessions
        }
        
        if config_path and Path(config_path).exists():
            try:
                with open(config_path, 'r') as f:
                    user_config = json.load(f)
                default_config.update(user_config)
            except Exception as e:
                logger.warning(f"Failed to load config from {config_path}: {e}")
        
        return default_config
    
    def _initialize_goal_templates(self) -> Dict[GoalCategory, Dict[str, Any]]:
        """Initialize goal templates for different categories"""
        templates = {}
        
        # Symptom Reduction Goals
        templates[GoalCategory.SYMPTOM_REDUCTION] = {
            'common_goals': [
                "Reduce anxiety symptoms by 50%",
                "Decrease depressive episodes frequency",
                "Manage panic attacks effectively",
                "Reduce intrusive thoughts",
                "Improve sleep quality"
            ],
            'success_criteria': [
                "Measurable symptom reduction",
                "Improved daily functioning",
                "Reduced interference with activities",
                "Client self-report improvement"
            ],
            'typical_interventions': [
                "Cognitive restructuring",
                "Behavioral activation",
                "Exposure therapy",
                "Mindfulness techniques",
                "Medication management"
            ]
        }
        
        # Skill Building Goals
        templates[GoalCategory.SKILL_BUILDING] = {
            'common_goals': [
                "Develop effective coping strategies",
                "Learn communication skills",
                "Build problem-solving abilities",
                "Enhance emotional regulation skills",
                "Develop assertiveness skills"
            ],
            'success_criteria': [
                "Skill demonstration in session",
                "Application in real-life situations",
                "Consistent use of skills",
                "Improved outcomes from skill use"
            ],
            'typical_interventions': [
                "Skills training",
                "Role-playing exercises",
                "Homework assignments",
                "Practice opportunities",
                "Skill generalization"
            ]
        }
        
        # Insight Development Goals
        templates[GoalCategory.INSIGHT_DEVELOPMENT] = {
            'common_goals': [
                "Understand relationship patterns",
                "Recognize emotional triggers",
                "Identify core beliefs",
                "Understand family dynamics impact",
                "Develop self-awareness"
            ],
            'success_criteria': [
                "Verbal expression of insights",
                "Connection between past and present",
                "Recognition of patterns",
                "Application of insights to behavior"
            ],
            'typical_interventions': [
                "Interpretive interventions",
                "Pattern identification",
                "Psychoeducation",
                "Reflective exercises",
                "Journaling"
            ]
        }
        
        # Behavioral Change Goals
        templates[GoalCategory.BEHAVIORAL_CHANGE] = {
            'common_goals': [
                "Establish healthy routines",
                "Reduce avoidance behaviors",
                "Improve social engagement",
                "Develop healthy habits",
                "Change maladaptive behaviors"
            ],
            'success_criteria': [
                "Observable behavior change",
                "Consistency in new behaviors",
                "Reduced problematic behaviors",
                "Improved functioning"
            ],
            'typical_interventions': [
                "Behavioral activation",
                "Exposure exercises",
                "Habit formation techniques",
                "Behavioral experiments",
                "Contingency management"
            ]
        }
        
        # Emotional Regulation Goals
        templates[GoalCategory.EMOTIONAL_REGULATION] = {
            'common_goals': [
                "Manage intense emotions effectively",
                "Develop distress tolerance",
                "Improve emotional awareness",
                "Reduce emotional reactivity",
                "Build emotional resilience"
            ],
            'success_criteria': [
                "Reduced emotional intensity",
                "Improved emotional control",
                "Better emotional expression",
                "Increased emotional vocabulary"
            ],
            'typical_interventions': [
                "Emotion regulation skills",
                "Mindfulness practices",
                "Distress tolerance techniques",
                "Emotional awareness exercises",
                "Grounding techniques"
            ]
        }
        
        # Relationship Improvement Goals
        templates[GoalCategory.RELATIONSHIP_IMPROVEMENT] = {
            'common_goals': [
                "Improve communication with partner",
                "Build healthier boundaries",
                "Develop trust in relationships",
                "Enhance intimacy and connection",
                "Resolve relationship conflicts"
            ],
            'success_criteria': [
                "Improved relationship satisfaction",
                "Better communication patterns",
                "Reduced relationship conflicts",
                "Increased intimacy and connection"
            ],
            'typical_interventions': [
                "Communication skills training",
                "Boundary setting exercises",
                "Couples therapy techniques",
                "Attachment work",
                "Conflict resolution skills"
            ]
        }
        
        return templates
    
    def _initialize_progress_patterns(self) -> Dict[str, Dict[str, Any]]:
        """Initialize patterns for recognizing progress"""
        patterns = {
            'verbal_indicators': {
                'positive_progress': [
                    "I feel better", "I'm improving", "I can handle this now",
                    "I understand now", "I see the difference", "I'm getting better at",
                    "I feel more confident", "I'm managing better", "I notice improvement"
                ],
                'skill_application': [
                    "I used the technique", "I tried what we discussed", "I practiced",
                    "I applied the skill", "I remembered to use", "I implemented"
                ],
                'insight_development': [
                    "I realize", "I understand now", "I see the pattern",
                    "I recognize", "I'm aware that", "I notice"
                ],
                'behavioral_change': [
                    "I did something different", "I changed my approach", "I acted differently",
                    "I made a different choice", "I responded differently"
                ]
            },
            'behavioral_indicators': {
                'engagement_improvement': [
                    "increased_session_attendance", "more_active_participation",
                    "homework_completion", "initiative_taking"
                ],
                'functioning_improvement': [
                    "work_performance", "social_activities", "self_care",
                    "relationship_quality", "daily_activities"
                ]
            },
            'emotional_indicators': {
                'regulation_improvement': [
                    "emotional_stability", "reduced_reactivity", "better_coping",
                    "emotional_awareness", "emotional_expression"
                ]
            }
        }
        
        return patterns
    
    def _initialize_milestone_criteria(self) -> Dict[MilestoneType, Dict[str, Any]]:
        """Initialize criteria for milestone recognition"""
        criteria = {
            MilestoneType.INITIAL_ENGAGEMENT: {
                'indicators': [
                    "regular_attendance", "active_participation", "trust_building",
                    "goal_setting_participation", "therapeutic_alliance"
                ],
                'threshold': 0.6,
                'typical_session': 3
            },
            MilestoneType.SKILL_ACQUISITION: {
                'indicators': [
                    "skill_demonstration", "understanding_concepts", "practice_completion",
                    "skill_application_attempts", "confidence_in_skills"
                ],
                'threshold': 0.7,
                'typical_session': 6
            },
            MilestoneType.INSIGHT_BREAKTHROUGH: {
                'indicators': [
                    "pattern_recognition", "connection_making", "self_awareness",
                    "understanding_development", "aha_moments"
                ],
                'threshold': 0.8,
                'typical_session': 8
            },
            MilestoneType.BEHAVIORAL_CHANGE: {
                'indicators': [
                    "behavior_modification", "new_responses", "habit_formation",
                    "consistent_application", "lifestyle_changes"
                ],
                'threshold': 0.7,
                'typical_session': 10
            },
            MilestoneType.SYMPTOM_IMPROVEMENT: {
                'indicators': [
                    "symptom_reduction", "functional_improvement", "quality_of_life",
                    "distress_reduction", "coping_effectiveness"
                ],
                'threshold': 0.6,
                'typical_session': 8
            }
        }
        
        return criteria
    
    def create_goal(
        self,
        title: str,
        description: str,
        category: GoalCategory,
        priority: GoalPriority = GoalPriority.MEDIUM,
        target_date: Optional[datetime] = None,
        success_criteria: Optional[List[str]] = None
    ) -> str:
        """Create a new therapeutic goal"""
        goal_id = f"goal_{len(self.goals) + 1}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Use template success criteria if not provided
        if success_criteria is None:
            template = self.goal_templates.get(category, {})
            success_criteria = template.get('success_criteria', [])
        
        goal = TherapeuticGoal(
            goal_id=goal_id,
            title=title,
            description=description,
            category=category,
            priority=priority,
            target_date=target_date,
            created_date=datetime.now(),
            success_criteria=success_criteria or []
        )
        
        self.goals[goal_id] = goal
        logger.info(f"Created therapeutic goal: {title} (ID: {goal_id})")
        
        return goal_id
    
    def update_goal_progress(
        self,
        goal_id: str,
        progress_score: float,
        evidence: str,
        indicator_type: ProgressIndicator,
        session_number: int,
        confidence_level: float = 0.8,
        therapist_notes: str = "",
        client_feedback: str = ""
    ) -> bool:
        """Update progress for a specific goal"""
        if goal_id not in self.goals:
            logger.error(f"Goal {goal_id} not found")
            return False
        
        try:
            # Create progress measurement
            measurement_id = f"progress_{len(self.progress_measurements) + 1}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            
            measurement = ProgressMeasurement(
                measurement_id=measurement_id,
                goal_id=goal_id,
                session_number=session_number,
                timestamp=datetime.now(),
                progress_score=progress_score,
                indicator_type=indicator_type,
                evidence=evidence,
                confidence_level=confidence_level,
                therapist_notes=therapist_notes,
                client_feedback=client_feedback
            )
            
            self.progress_measurements.append(measurement)
            
            # Update goal progress
            goal = self.goals[goal_id]
            goal.progress_percentage = self._calculate_overall_progress(goal_id)
            goal.last_updated = datetime.now()
            
            # Update goal status based on progress
            self._update_goal_status(goal_id)
            
            # Check for milestones
            if self.config.get('automatic_milestone_detection', True):
                await self._detect_milestones(goal_id, session_number)
            
            logger.info(f"Updated progress for goal {goal_id}: {progress_score:.2f}")
            return True
            
        except Exception as e:
            logger.error(f"Error updating goal progress: {e}")
            return False
    
    def _calculate_overall_progress(self, goal_id: str) -> float:
        """Calculate overall progress for a goal"""
        goal_measurements = [m for m in self.progress_measurements if m.goal_id == goal_id]
        
        if not goal_measurements:
            return 0.0
        
        # Use smoothing window for recent measurements
        window_size = self.config.get('progress_smoothing_window', 3)
        recent_measurements = sorted(goal_measurements, key=lambda x: x.timestamp)[-window_size:]
        
        # Weight by confidence level and recency
        weighted_scores = []
        total_weight = 0.0
        
        for i, measurement in enumerate(recent_measurements):
            # More recent measurements get higher weight
            recency_weight = (i + 1) / len(recent_measurements)
            confidence_weight = measurement.confidence_level
            combined_weight = recency_weight * confidence_weight
            
            weighted_scores.append(measurement.progress_score * combined_weight)
            total_weight += combined_weight
        
        if total_weight == 0:
            return 0.0
        
        overall_progress = sum(weighted_scores) / total_weight
        return min(100.0, overall_progress * 100.0)  # Convert to percentage
    
    def _update_goal_status(self, goal_id: str):
        """Update goal status based on progress"""
        goal = self.goals[goal_id]
        progress = goal.progress_percentage
        
        completion_threshold = self.config.get('goal_completion_threshold', 0.9) * 100
        
        if progress >= completion_threshold:
            goal.status = GoalStatus.ACHIEVED
        elif progress >= 50.0:
            goal.status = GoalStatus.PARTIALLY_ACHIEVED
        elif progress > 0.0:
            goal.status = GoalStatus.IN_PROGRESS
        else:
            goal.status = GoalStatus.NOT_STARTED
    
    async def _detect_milestones(self, goal_id: str, session_number: int):
        """Detect and record milestones for a goal"""
        self.goals[goal_id]
        goal_measurements = [m for m in self.progress_measurements if m.goal_id == goal_id]
        
        if not goal_measurements:
            return
        
        # Check for different milestone types
        for milestone_type, criteria in self.milestone_criteria.items():
            if await self._check_milestone_criteria(goal_id, milestone_type, criteria, session_number):
                await self._create_milestone(goal_id, milestone_type, session_number, goal_measurements)
    
    async def _check_milestone_criteria(
        self,
        goal_id: str,
        milestone_type: MilestoneType,
        criteria: Dict[str, Any],
        session_number: int
    ) -> bool:
        """Check if milestone criteria are met"""
        # Check if milestone already exists
        existing_milestones = [m for m in self.milestones if m.goal_id == goal_id and m.milestone_type == milestone_type]
        if existing_milestones:
            return False
        
        # Check session timing
        typical_session = criteria.get('typical_session', 5)
        if session_number < typical_session - 2:  # Allow some flexibility
            return False
        
        # Check progress threshold
        goal = self.goals[goal_id]
        threshold = criteria.get('threshold', 0.7) * 100
        
        if goal.progress_percentage >= threshold:
            return True
        
        # Additional criteria checking would go here
        # For now, use basic progress threshold
        return False
    
    async def _create_milestone(
        self,
        goal_id: str,
        milestone_type: MilestoneType,
        session_number: int,
        measurements: List[ProgressMeasurement]
    ):
        """Create a milestone record"""
        goal = self.goals[goal_id]
        milestone_id = f"milestone_{len(self.milestones) + 1}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Generate milestone description
        milestone_descriptions = {
            MilestoneType.INITIAL_ENGAGEMENT: f"Successfully engaged in therapeutic process for {goal.title}",
            MilestoneType.SKILL_ACQUISITION: f"Acquired key skills related to {goal.title}",
            MilestoneType.INSIGHT_BREAKTHROUGH: f"Achieved significant insight regarding {goal.title}",
            MilestoneType.BEHAVIORAL_CHANGE: f"Demonstrated behavioral change toward {goal.title}",
            MilestoneType.SYMPTOM_IMPROVEMENT: f"Showed measurable improvement in {goal.title}"
        }
        
        # Collect evidence from recent measurements
        evidence = [m.evidence for m in measurements[-3:] if m.evidence]
        
        milestone = TherapeuticMilestone(
            milestone_id=milestone_id,
            goal_id=goal_id,
            milestone_type=milestone_type,
            title=f"{milestone_type.value.replace('_', ' ').title()} - {goal.title}",
            description=milestone_descriptions.get(milestone_type, f"Milestone achieved for {goal.title}"),
            achieved_date=datetime.now(),
            session_number=session_number,
            evidence=evidence,
            significance_score=min(1.0, goal.progress_percentage / 100.0),
            client_recognition=False,  # Would be updated based on client feedback
            therapist_notes=f"Automatic milestone detection for {milestone_type.value}"
        )
        
        self.milestones.append(milestone)
        goal.milestones.append(milestone_id)
        
        logger.info(f"Created milestone: {milestone.title}")
    
    async def analyze_conversation_for_progress(
        self,
        conversation_history: List[Any],
        clinical_context: Any,
        session_number: int
    ) -> Dict[str, Any]:
        """Analyze conversation for goal progress indicators"""
        progress_analysis = {
            'detected_progress': {},
            'goal_mentions': {},
            'barriers_identified': [],
            'interventions_suggested': [],
            'milestone_candidates': []
        }
        
        try:
            # Analyze each conversation turn
            for turn in conversation_history:
                content = getattr(turn, 'content', '').lower()
                speaker = getattr(turn, 'speaker', 'unknown')
                
                # Skip therapist turns for progress detection
                if speaker.lower() == 'therapist':
                    continue
                
                # Check for progress indicators
                await self._analyze_turn_for_progress(content, progress_analysis)
                
                # Check for goal mentions
                await self._analyze_turn_for_goal_mentions(content, progress_analysis)
                
                # Check for barriers
                await self._analyze_turn_for_barriers(content, progress_analysis)
            
            # Update goal progress based on analysis
            await self._update_goals_from_analysis(progress_analysis, session_number)
            
            return progress_analysis
            
        except Exception as e:
            logger.error(f"Error analyzing conversation for progress: {e}")
            return progress_analysis
    
    async def _analyze_turn_for_progress(self, content: str, analysis: Dict[str, Any]):
        """Analyze conversation turn for progress indicators"""
        patterns = self.progress_patterns.get('verbal_indicators', {})
        
        # Check positive progress indicators
        positive_indicators = patterns.get('positive_progress', [])
        for indicator in positive_indicators:
            if indicator.lower() in content:
                analysis['detected_progress']['positive_sentiment'] = analysis['detected_progress'].get('positive_sentiment', 0) + 1
        
        # Check skill application indicators
        skill_indicators = patterns.get('skill_application', [])
        for indicator in skill_indicators:
            if indicator.lower() in content:
                analysis['detected_progress']['skill_application'] = analysis['detected_progress'].get('skill_application', 0) + 1
        
        # Check insight development indicators
        insight_indicators = patterns.get('insight_development', [])
        for indicator in insight_indicators:
            if indicator.lower() in content:
                analysis['detected_progress']['insight_development'] = analysis['detected_progress'].get('insight_development', 0) + 1
        
        # Check behavioral change indicators
        behavioral_indicators = patterns.get('behavioral_change', [])
        for indicator in behavioral_indicators:
            if indicator.lower() in content:
                analysis['detected_progress']['behavioral_change'] = analysis['detected_progress'].get('behavioral_change', 0) + 1
    
    async def _analyze_turn_for_goal_mentions(self, content: str, analysis: Dict[str, Any]):
        """Analyze conversation turn for goal mentions"""
        for goal_id, goal in self.goals.items():
            # Check if goal title or keywords are mentioned
            goal_keywords = goal.title.lower().split() + goal.description.lower().split()
            
            for keyword in goal_keywords:
                if len(keyword) > 3 and keyword in content:  # Avoid short words
                    analysis['goal_mentions'][goal_id] = analysis['goal_mentions'].get(goal_id, 0) + 1
    
    async def _analyze_turn_for_barriers(self, content: str, analysis: Dict[str, Any]):
        """Analyze conversation turn for barriers"""
        barrier_indicators = [
            "difficult", "hard", "can't", "unable", "struggling", "stuck",
            "overwhelming", "too much", "not working", "frustrated", "giving up"
        ]
        
        for indicator in barrier_indicators:
            if indicator in content:
                analysis['barriers_identified'].append(f"Client expressed: {indicator}")
    
    async def _update_goals_from_analysis(self, analysis: Dict[str, Any], session_number: int):
        """Update goal progress based on conversation analysis"""
        detected_progress = analysis.get('detected_progress', {})
        goal_mentions = analysis.get('goal_mentions', {})
        
        for goal_id in self.goals.keys():
            progress_score = 0.0
            evidence_parts = []
            
            # Calculate progress based on detected indicators
            if goal_id in goal_mentions:
                mention_count = goal_mentions[goal_id]
                progress_score += min(0.2, mention_count * 0.05)  # Up to 0.2 for mentions
                evidence_parts.append(f"Goal mentioned {mention_count} times")
            
            # Add progress for different types of indicators
            for indicator_type, count in detected_progress.items():
                if count > 0:
                    progress_score += min(0.3, count * 0.1)  # Up to 0.3 per indicator type
                    evidence_parts.append(f"{indicator_type}: {count} instances")
            
            # Only update if there's meaningful progress
            if progress_score > 0.1:
                evidence = "; ".join(evidence_parts)
                await self.update_goal_progress(
                    goal_id=goal_id,
                    progress_score=progress_score,
                    evidence=evidence,
                    indicator_type=ProgressIndicator.SELF_REPORT,
                    session_number=session_number,
                    confidence_level=0.6,  # Moderate confidence for conversation analysis
                    therapist_notes="Progress detected through conversation analysis"
                )
    
    def get_goal_progress_summary(self, goal_id: str) -> Optional[GoalProgressSummary]:
        """Get comprehensive progress summary for a goal"""
        if goal_id not in self.goals:
            return None
        
        goal = self.goals[goal_id]
        goal_measurements = [m for m in self.progress_measurements if m.goal_id == goal_id]
        goal_milestones = [m for m in self.milestones if m.goal_id == goal_id]
        
        # Calculate progress trend
        trend = self._calculate_progress_trend(goal_measurements)
        
        # Identify recent barriers
        barriers = self._identify_recent_barriers(goal_measurements)
        
        # Get interventions used
        interventions = list(set([m.therapist_notes for m in goal_measurements if m.therapist_notes]))
        
        # Generate next steps
        next_steps = self._generate_next_steps(goal, goal_measurements, goal_milestones)
        
        # Estimate completion
        estimated_completion = self._estimate_completion_date(goal, goal_measurements)
        
        # Calculate confidence in achievement
        confidence = self._calculate_achievement_confidence(goal, goal_measurements, goal_milestones)
        
        return GoalProgressSummary(
            goal_id=goal_id,
            current_progress=goal.progress_percentage,
            progress_trend=trend,
            recent_milestones=goal_milestones[-3:],  # Last 3 milestones
            barriers_identified=barriers,
            interventions_used=interventions,
            next_steps=next_steps,
            estimated_completion=estimated_completion,
            confidence_in_achievement=confidence
        )
    
    def _calculate_progress_trend(self, measurements: List[ProgressMeasurement]) -> str:
        """Calculate progress trend from measurements"""
        if len(measurements) < 2:
            return "stable"
        
        # Use recent measurements for trend
        window_size = self.config.get('progress_trend_window', 5)
        recent_measurements = sorted(measurements, key=lambda x: x.timestamp)[-window_size:]
        
        if len(recent_measurements) < 2:
            return "stable"
        
        # Calculate trend slope
        scores = [m.progress_score for m in recent_measurements]
        x = list(range(len(scores)))
        
        # Simple linear regression slope
        n = len(scores)
        slope = (n * sum(x[i] * scores[i] for i in range(n)) - sum(x) * sum(scores)) / (n * sum(x[i]**2 for i in range(n)) - sum(x)**2)
        
        if slope > 0.05:
            return "improving"
        elif slope < -0.05:
            return "declining"
        else:
            return "stable"
    
    def _identify_recent_barriers(self, measurements: List[ProgressMeasurement]) -> List[str]:
        """Identify recent barriers from measurements"""
        barriers = []
        
        # Look for low progress scores or negative evidence
        recent_measurements = sorted(measurements, key=lambda x: x.timestamp)[-5:]
        
        for measurement in recent_measurements:
            if measurement.progress_score < 0.3:
                barriers.append(f"Low progress in session {measurement.session_number}")
            
            # Check evidence for barrier keywords
            evidence = measurement.evidence.lower()
            barrier_keywords = ["difficult", "struggling", "stuck", "frustrated", "overwhelming"]
            
            for keyword in barrier_keywords:
                if keyword in evidence:
                    barriers.append(f"Client reported {keyword} in session {measurement.session_number}")
        
        return list(set(barriers))  # Remove duplicates
    
    def _generate_next_steps(
        self,
        goal: TherapeuticGoal,
        measurements: List[ProgressMeasurement],
        milestones: List[TherapeuticMilestone]
    ) -> List[str]:
        """Generate next steps for goal achievement"""
        next_steps = []
        
        # Based on goal category and current progress
        template = self.goal_templates.get(goal.category, {})
        typical_interventions = template.get('typical_interventions', [])
        
        if goal.progress_percentage < 25:
            next_steps.append("Focus on foundational skill building")
            next_steps.extend(typical_interventions[:2])  # First 2 interventions
        elif goal.progress_percentage < 50:
            next_steps.append("Continue skill development and practice")
            next_steps.extend(typical_interventions[1:3])  # Middle interventions
        elif goal.progress_percentage < 75:
            next_steps.append("Focus on skill application and generalization")
            next_steps.extend(typical_interventions[2:])  # Later interventions
        else:
            next_steps.append("Prepare for goal completion and maintenance")
            next_steps.append("Plan relapse prevention strategies")
        
        return next_steps[:5]  # Limit to 5 steps
    
    def _estimate_completion_date(
        self,
        goal: TherapeuticGoal,
        measurements: List[ProgressMeasurement]
    ) -> Optional[datetime]:
        """Estimate goal completion date"""
        if not measurements or goal.progress_percentage >= 90:
            return None
        
        # Calculate average progress rate
        sorted_measurements = sorted(measurements, key=lambda x: x.timestamp)
        
        if len(sorted_measurements) < 2:
            return None
        
        # Calculate progress per session
        first_measurement = sorted_measurements[0]
        last_measurement = sorted_measurements[-1]
        
        progress_change = last_measurement.progress_score - first_measurement.progress_score
        session_change = last_measurement.session_number - first_measurement.session_number
        
        if session_change == 0 or progress_change <= 0:
            return None
        
        progress_per_session = progress_change / session_change
        remaining_progress = (100 - goal.progress_percentage) / 100.0
        estimated_sessions = remaining_progress / progress_per_session
        
        # Assume weekly sessions
        estimated_weeks = estimated_sessions
        estimated_completion = datetime.now() + timedelta(weeks=estimated_weeks)
        
        return estimated_completion
    
    def _calculate_achievement_confidence(
        self,
        goal: TherapeuticGoal,
        measurements: List[ProgressMeasurement],
        milestones: List[TherapeuticMilestone]
    ) -> float:
        """Calculate confidence in goal achievement"""
        confidence_factors = []
        
        # Current progress factor
        progress_factor = goal.progress_percentage / 100.0
        confidence_factors.append(progress_factor * 0.4)
        
        # Progress trend factor
        trend = self._calculate_progress_trend(measurements)
        if trend == "improving":
            confidence_factors.append(0.3)
        elif trend == "stable":
            confidence_factors.append(0.2)
        else:  # declining
            confidence_factors.append(0.1)
        
        # Milestone achievement factor
        milestone_factor = min(1.0, len(milestones) * 0.2)
        confidence_factors.append(milestone_factor * 0.2)
        
        # Measurement confidence factor
        if measurements:
            avg_confidence = np.mean([m.confidence_level for m in measurements])
            confidence_factors.append(avg_confidence * 0.1)
        
        return min(1.0, sum(confidence_factors))
    
    def get_all_goals_summary(self) -> Dict[str, Any]:
        """Get summary of all goals"""
        summary = {
            'total_goals': len(self.goals),
            'goals_by_status': {},
            'goals_by_category': {},
            'goals_by_priority': {},
            'overall_progress': 0.0,
            'total_milestones': len(self.milestones),
            'recent_progress': []
        }
        
        if not self.goals:
            return summary
        
        # Count by status
        for goal in self.goals.values():
            status = goal.status.value
            summary['goals_by_status'][status] = summary['goals_by_status'].get(status, 0) + 1
            
            category = goal.category.value
            summary['goals_by_category'][category] = summary['goals_by_category'].get(category, 0) + 1
            
            priority = goal.priority.value
            summary['goals_by_priority'][priority] = summary['goals_by_priority'].get(priority, 0) + 1
        
        # Calculate overall progress
        total_progress = sum(goal.progress_percentage for goal in self.goals.values())
        summary['overall_progress'] = total_progress / len(self.goals)
        
        # Recent progress (last 5 measurements)
        recent_measurements = sorted(self.progress_measurements, key=lambda x: x.timestamp)[-5:]
        summary['recent_progress'] = [
            {
                'goal_id': m.goal_id,
                'goal_title': self.goals[m.goal_id].title,
                'progress_score': m.progress_score,
                'session_number': m.session_number,
                'timestamp': m.timestamp.isoformat()
            }
            for m in recent_measurements
        ]
        
        return summary
    
    def export_goal_data(self, format: str = 'json') -> Union[str, Dict[str, Any]]:
        """Export goal tracking data"""
        data = {
            'goals': {
                goal_id: {
                    'title': goal.title,
                    'description': goal.description,
                    'category': goal.category.value,
                    'priority': goal.priority.value,
                    'status': goal.status.value,
                    'progress_percentage': goal.progress_percentage,
                    'created_date': goal.created_date.isoformat(),
                    'target_date': goal.target_date.isoformat() if goal.target_date else None,
                    'success_criteria': goal.success_criteria,
                    'barriers': goal.barriers,
                    'interventions': goal.interventions,
                    'milestones': goal.milestones,
                    'notes': goal.notes,
                    'last_updated': goal.last_updated.isoformat()
                }
                for goal_id, goal in self.goals.items()
            },
            'progress_measurements': [
                {
                    'measurement_id': m.measurement_id,
                    'goal_id': m.goal_id,
                    'session_number': m.session_number,
                    'timestamp': m.timestamp.isoformat(),
                    'progress_score': m.progress_score,
                    'indicator_type': m.indicator_type.value,
                    'evidence': m.evidence,
                    'confidence_level': m.confidence_level,
                    'therapist_notes': m.therapist_notes,
                    'client_feedback': m.client_feedback
                }
                for m in self.progress_measurements
            ],
            'milestones': [
                {
                    'milestone_id': m.milestone_id,
                    'goal_id': m.goal_id,
                    'milestone_type': m.milestone_type.value,
                    'title': m.title,
                    'description': m.description,
                    'achieved_date': m.achieved_date.isoformat(),
                    'session_number': m.session_number,
                    'evidence': m.evidence,
                    'significance_score': m.significance_score,
                    'client_recognition': m.client_recognition,
                    'therapist_notes': m.therapist_notes
                }
                for m in self.milestones
            ],
            'summary': self.get_all_goals_summary()
        }
        
        if format.lower() == 'json':
            return json.dumps(data, indent=2)
        else:
            return data


# Example usage and testing
if __name__ == "__main__":
    async def test_goal_tracker():
        """Test the therapeutic goal tracker"""
        tracker = TherapeuticGoalTracker()
        
        # Create a goal
        goal_id = tracker.create_goal(
            title="Reduce anxiety symptoms",
            description="Decrease frequency and intensity of anxiety episodes",
            category=GoalCategory.SYMPTOM_REDUCTION,
            priority=GoalPriority.HIGH,
            success_criteria=["50% reduction in anxiety episodes", "Improved daily functioning"]
        )
        
        print(f"Created goal: {goal_id}")
        
        # Update progress
        success = await tracker.update_goal_progress(
            goal_id=goal_id,
            progress_score=0.3,
            evidence="Client reported using breathing techniques successfully",
            indicator_type=ProgressIndicator.SELF_REPORT,
            session_number=3,
            therapist_notes="Good progress with coping skills"
        )
        
        print(f"Progress updated: {success}")
        
        # Get progress summary
        summary = tracker.get_goal_progress_summary(goal_id)
        if summary:
            print(f"Progress: {summary.current_progress:.1f}%")
            print(f"Trend: {summary.progress_trend}")
            print(f"Next steps: {summary.next_steps}")
        
        # Get overall summary
        overall_summary = tracker.get_all_goals_summary()
        print(f"Overall summary: {overall_summary}")
    
    # Run test
    asyncio.run(test_goal_tracker())
