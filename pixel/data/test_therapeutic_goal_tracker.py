"""
Unit tests for Therapeutic Goal Tracker

Tests the therapeutic goal tracking system including goal creation,
progress tracking, milestone detection, and goal-oriented analysis.
"""

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import List

import pytest

from .therapeutic_goal_tracker import (
    GoalCategory,
    GoalPriority,
    GoalProgressSummary,
    GoalStatus,
    MilestoneType,
    ProgressIndicator,
    ProgressMeasurement,
    TherapeuticGoalTracker,
    TherapeuticMilestone,
)


@dataclass
class MockConversationTurn:
    content: str
    speaker: str = "client"


@dataclass
class MockClinicalContext:
    primary_diagnosis: str
    session_number: int
    therapeutic_goals: List[str] = None


@pytest.fixture
def goal_tracker():
    """Create therapeutic goal tracker instance"""
    return TherapeuticGoalTracker()


@pytest.fixture
def sample_goal_data():
    """Create sample goal data"""
    return {
        'title': "Reduce anxiety symptoms",
        'description': "Decrease frequency and intensity of anxiety episodes by 50%",
        'category': GoalCategory.SYMPTOM_REDUCTION,
        'priority': GoalPriority.HIGH,
        'success_criteria': ["50% reduction in anxiety episodes", "Improved daily functioning"]
    }


@pytest.fixture
def progress_conversation():
    """Create conversation showing progress"""
    return [
        MockConversationTurn("I feel better this week", "client"),
        MockConversationTurn("I used the breathing technique you taught me", "client"),
        MockConversationTurn("I'm managing my anxiety better now", "client"),
        MockConversationTurn("I see the pattern in my thoughts", "client")
    ]


@pytest.fixture
def barrier_conversation():
    """Create conversation showing barriers"""
    return [
        MockConversationTurn("This is really difficult for me", "client"),
        MockConversationTurn("I'm struggling with the exercises", "client"),
        MockConversationTurn("I feel stuck and frustrated", "client")
    ]


@pytest.fixture
def clinical_context():
    """Create clinical context"""
    return MockClinicalContext(
        primary_diagnosis="Generalized Anxiety Disorder",
        session_number=5,
        therapeutic_goals=["Reduce anxiety", "Improve coping skills"]
    )


class TestTherapeuticGoalTracker:
    """Test cases for TherapeuticGoalTracker"""
    
    def test_initialization(self, goal_tracker):
        """Test tracker initialization"""
        assert goal_tracker.config is not None
        assert goal_tracker.goals == {}
        assert goal_tracker.progress_measurements == []
        assert goal_tracker.milestones == []
        assert goal_tracker.goal_templates is not None
        assert goal_tracker.progress_patterns is not None
        assert goal_tracker.milestone_criteria is not None
    
    def test_goal_templates_initialization(self, goal_tracker):
        """Test goal templates are properly initialized"""
        templates = goal_tracker.goal_templates
        
        # Check all categories are present
        assert GoalCategory.SYMPTOM_REDUCTION in templates
        assert GoalCategory.SKILL_BUILDING in templates
        assert GoalCategory.INSIGHT_DEVELOPMENT in templates
        assert GoalCategory.BEHAVIORAL_CHANGE in templates
        assert GoalCategory.EMOTIONAL_REGULATION in templates
        assert GoalCategory.RELATIONSHIP_IMPROVEMENT in templates
        
        # Check template structure
        for category, template in templates.items():
            assert 'common_goals' in template
            assert 'success_criteria' in template
            assert 'typical_interventions' in template
            assert isinstance(template['common_goals'], list)
            assert isinstance(template['success_criteria'], list)
            assert isinstance(template['typical_interventions'], list)
    
    def test_progress_patterns_initialization(self, goal_tracker):
        """Test progress patterns are properly initialized"""
        patterns = goal_tracker.progress_patterns
        
        assert 'verbal_indicators' in patterns
        assert 'behavioral_indicators' in patterns
        assert 'emotional_indicators' in patterns
        
        verbal = patterns['verbal_indicators']
        assert 'positive_progress' in verbal
        assert 'skill_application' in verbal
        assert 'insight_development' in verbal
        assert 'behavioral_change' in verbal
        
        # Check patterns are lists of strings
        for pattern_type, pattern_list in verbal.items():
            assert isinstance(pattern_list, list)
            assert all(isinstance(item, str) for item in pattern_list)
    
    def test_milestone_criteria_initialization(self, goal_tracker):
        """Test milestone criteria are properly initialized"""
        criteria = goal_tracker.milestone_criteria
        
        # Check all milestone types are present
        assert MilestoneType.INITIAL_ENGAGEMENT in criteria
        assert MilestoneType.SKILL_ACQUISITION in criteria
        assert MilestoneType.INSIGHT_BREAKTHROUGH in criteria
        assert MilestoneType.BEHAVIORAL_CHANGE in criteria
        assert MilestoneType.SYMPTOM_IMPROVEMENT in criteria
        
        # Check criteria structure
        for milestone_type, criterion in criteria.items():
            assert 'indicators' in criterion
            assert 'threshold' in criterion
            assert 'typical_session' in criterion
            assert isinstance(criterion['indicators'], list)
            assert 0.0 <= criterion['threshold'] <= 1.0
            assert isinstance(criterion['typical_session'], int)
    
    def test_create_goal_basic(self, goal_tracker, sample_goal_data):
        """Test basic goal creation"""
        goal_id = goal_tracker.create_goal(**sample_goal_data)
        
        assert goal_id is not None
        assert goal_id in goal_tracker.goals
        
        goal = goal_tracker.goals[goal_id]
        assert goal.title == sample_goal_data['title']
        assert goal.description == sample_goal_data['description']
        assert goal.category == sample_goal_data['category']
        assert goal.priority == sample_goal_data['priority']
        assert goal.status == GoalStatus.NOT_STARTED
        assert goal.progress_percentage == 0.0
        assert goal.success_criteria == sample_goal_data['success_criteria']
        assert isinstance(goal.created_date, datetime)
    
    def test_create_goal_with_target_date(self, goal_tracker):
        """Test goal creation with target date"""
        target_date = datetime.now() + timedelta(weeks=8)
        
        goal_id = goal_tracker.create_goal(
            title="Test Goal",
            description="Test Description",
            category=GoalCategory.SKILL_BUILDING,
            target_date=target_date
        )
        
        goal = goal_tracker.goals[goal_id]
        assert goal.target_date == target_date
    
    def test_create_goal_with_template_criteria(self, goal_tracker):
        """Test goal creation uses template success criteria"""
        goal_id = goal_tracker.create_goal(
            title="Build coping skills",
            description="Develop effective coping strategies",
            category=GoalCategory.SKILL_BUILDING
            # No success_criteria provided - should use template
        )
        
        goal = goal_tracker.goals[goal_id]
        template = goal_tracker.goal_templates[GoalCategory.SKILL_BUILDING]
        assert goal.success_criteria == template['success_criteria']
    
    @pytest.mark.asyncio
    async def test_update_goal_progress_basic(self, goal_tracker, sample_goal_data):
        """Test basic goal progress update"""
        goal_id = goal_tracker.create_goal(**sample_goal_data)
        
        success = await goal_tracker.update_goal_progress(
            goal_id=goal_id,
            progress_score=0.3,
            evidence="Client reported improvement",
            indicator_type=ProgressIndicator.SELF_REPORT,
            session_number=3,
            confidence_level=0.8,
            therapist_notes="Good progress"
        )
        
        assert success is True
        assert len(goal_tracker.progress_measurements) == 1
        
        measurement = goal_tracker.progress_measurements[0]
        assert measurement.goal_id == goal_id
        assert measurement.progress_score == 0.3
        assert measurement.evidence == "Client reported improvement"
        assert measurement.indicator_type == ProgressIndicator.SELF_REPORT
        assert measurement.session_number == 3
        assert measurement.confidence_level == 0.8
        assert measurement.therapist_notes == "Good progress"
        
        # Check goal was updated
        goal = goal_tracker.goals[goal_id]
        assert goal.progress_percentage > 0
        assert goal.status == GoalStatus.IN_PROGRESS
    
    @pytest.mark.asyncio
    async def test_update_goal_progress_nonexistent_goal(self, goal_tracker):
        """Test updating progress for nonexistent goal"""
        success = await goal_tracker.update_goal_progress(
            goal_id="nonexistent",
            progress_score=0.5,
            evidence="Test",
            indicator_type=ProgressIndicator.SELF_REPORT,
            session_number=1
        )
        
        assert success is False
        assert len(goal_tracker.progress_measurements) == 0
    
    def test_calculate_overall_progress_single_measurement(self, goal_tracker, sample_goal_data):
        """Test overall progress calculation with single measurement"""
        goal_id = goal_tracker.create_goal(**sample_goal_data)
        
        # Add a measurement manually
        measurement = ProgressMeasurement(
            measurement_id="test_1",
            goal_id=goal_id,
            session_number=1,
            timestamp=datetime.now(),
            progress_score=0.4,
            indicator_type=ProgressIndicator.SELF_REPORT,
            evidence="Test evidence",
            confidence_level=0.8,
            therapist_notes="Test notes"
        )
        goal_tracker.progress_measurements.append(measurement)
        
        progress = goal_tracker._calculate_overall_progress(goal_id)
        
        assert isinstance(progress, float)
        assert 0.0 <= progress <= 100.0
        # Should be around 40% (0.4 * 100)
        assert 35.0 <= progress <= 45.0
    
    def test_calculate_overall_progress_multiple_measurements(self, goal_tracker, sample_goal_data):
        """Test overall progress calculation with multiple measurements"""
        goal_id = goal_tracker.create_goal(**sample_goal_data)
        
        # Add multiple measurements
        measurements = [
            ProgressMeasurement(
                measurement_id=f"test_{i}",
                goal_id=goal_id,
                session_number=i,
                timestamp=datetime.now() + timedelta(days=i),
                progress_score=0.2 + (i * 0.1),  # Increasing progress
                indicator_type=ProgressIndicator.SELF_REPORT,
                evidence=f"Evidence {i}",
                confidence_level=0.8,
                therapist_notes=f"Notes {i}"
            )
            for i in range(1, 4)
        ]
        
        goal_tracker.progress_measurements.extend(measurements)
        
        progress = goal_tracker._calculate_overall_progress(goal_id)
        
        # Should reflect recent higher scores more
        assert progress > 30.0  # Should be higher than just the average
    
    def test_calculate_overall_progress_no_measurements(self, goal_tracker, sample_goal_data):
        """Test overall progress calculation with no measurements"""
        goal_id = goal_tracker.create_goal(**sample_goal_data)
        
        progress = goal_tracker._calculate_overall_progress(goal_id)
        
        assert progress == 0.0
    
    def test_update_goal_status_not_started(self, goal_tracker, sample_goal_data):
        """Test goal status update for not started"""
        goal_id = goal_tracker.create_goal(**sample_goal_data)
        goal = goal_tracker.goals[goal_id]
        goal.progress_percentage = 0.0
        
        goal_tracker._update_goal_status(goal_id)
        
        assert goal.status == GoalStatus.NOT_STARTED
    
    def test_update_goal_status_in_progress(self, goal_tracker, sample_goal_data):
        """Test goal status update for in progress"""
        goal_id = goal_tracker.create_goal(**sample_goal_data)
        goal = goal_tracker.goals[goal_id]
        goal.progress_percentage = 25.0
        
        goal_tracker._update_goal_status(goal_id)
        
        assert goal.status == GoalStatus.IN_PROGRESS
    
    def test_update_goal_status_partially_achieved(self, goal_tracker, sample_goal_data):
        """Test goal status update for partially achieved"""
        goal_id = goal_tracker.create_goal(**sample_goal_data)
        goal = goal_tracker.goals[goal_id]
        goal.progress_percentage = 60.0
        
        goal_tracker._update_goal_status(goal_id)
        
        assert goal.status == GoalStatus.PARTIALLY_ACHIEVED
    
    def test_update_goal_status_achieved(self, goal_tracker, sample_goal_data):
        """Test goal status update for achieved"""
        goal_id = goal_tracker.create_goal(**sample_goal_data)
        goal = goal_tracker.goals[goal_id]
        goal.progress_percentage = 95.0
        
        goal_tracker._update_goal_status(goal_id)
        
        assert goal.status == GoalStatus.ACHIEVED
    
    @pytest.mark.asyncio
    async def test_check_milestone_criteria_met(self, goal_tracker, sample_goal_data):
        """Test milestone criteria checking when met"""
        goal_id = goal_tracker.create_goal(**sample_goal_data)
        goal = goal_tracker.goals[goal_id]
        goal.progress_percentage = 80.0  # Above threshold
        
        criteria = goal_tracker.milestone_criteria[MilestoneType.SYMPTOM_IMPROVEMENT]
        
        result = await goal_tracker._check_milestone_criteria(
            goal_id, MilestoneType.SYMPTOM_IMPROVEMENT, criteria, 8
        )
        
        assert result is True
    
    @pytest.mark.asyncio
    async def test_check_milestone_criteria_not_met_progress(self, goal_tracker, sample_goal_data):
        """Test milestone criteria checking when progress threshold not met"""
        goal_id = goal_tracker.create_goal(**sample_goal_data)
        goal = goal_tracker.goals[goal_id]
        goal.progress_percentage = 30.0  # Below threshold
        
        criteria = goal_tracker.milestone_criteria[MilestoneType.SYMPTOM_IMPROVEMENT]
        
        result = await goal_tracker._check_milestone_criteria(
            goal_id, MilestoneType.SYMPTOM_IMPROVEMENT, criteria, 8
        )
        
        assert result is False
    
    @pytest.mark.asyncio
    async def test_check_milestone_criteria_not_met_session(self, goal_tracker, sample_goal_data):
        """Test milestone criteria checking when session too early"""
        goal_id = goal_tracker.create_goal(**sample_goal_data)
        goal = goal_tracker.goals[goal_id]
        goal.progress_percentage = 80.0  # Above threshold
        
        criteria = goal_tracker.milestone_criteria[MilestoneType.SYMPTOM_IMPROVEMENT]
        
        result = await goal_tracker._check_milestone_criteria(
            goal_id, MilestoneType.SYMPTOM_IMPROVEMENT, criteria, 2  # Too early
        )
        
        assert result is False
    
    @pytest.mark.asyncio
    async def test_check_milestone_criteria_already_exists(self, goal_tracker, sample_goal_data):
        """Test milestone criteria checking when milestone already exists"""
        goal_id = goal_tracker.create_goal(**sample_goal_data)
        goal = goal_tracker.goals[goal_id]
        goal.progress_percentage = 80.0
        
        # Create existing milestone
        existing_milestone = TherapeuticMilestone(
            milestone_id="existing",
            goal_id=goal_id,
            milestone_type=MilestoneType.SYMPTOM_IMPROVEMENT,
            title="Test Milestone",
            description="Test",
            achieved_date=datetime.now(),
            session_number=5,
            evidence=["test"],
            significance_score=0.8,
            client_recognition=True,
            therapist_notes="Test"
        )
        goal_tracker.milestones.append(existing_milestone)
        
        criteria = goal_tracker.milestone_criteria[MilestoneType.SYMPTOM_IMPROVEMENT]
        
        result = await goal_tracker._check_milestone_criteria(
            goal_id, MilestoneType.SYMPTOM_IMPROVEMENT, criteria, 8
        )
        
        assert result is False
    
    @pytest.mark.asyncio
    async def test_create_milestone(self, goal_tracker, sample_goal_data):
        """Test milestone creation"""
        goal_id = goal_tracker.create_goal(**sample_goal_data)
        
        measurements = [
            ProgressMeasurement(
                measurement_id="test_1",
                goal_id=goal_id,
                session_number=5,
                timestamp=datetime.now(),
                progress_score=0.6,
                indicator_type=ProgressIndicator.SELF_REPORT,
                evidence="Client reported improvement",
                confidence_level=0.8,
                therapist_notes="Good progress"
            )
        ]
        
        await goal_tracker._create_milestone(
            goal_id, MilestoneType.SYMPTOM_IMPROVEMENT, 8, measurements
        )
        
        assert len(goal_tracker.milestones) == 1
        
        milestone = goal_tracker.milestones[0]
        assert milestone.goal_id == goal_id
        assert milestone.milestone_type == MilestoneType.SYMPTOM_IMPROVEMENT
        assert milestone.session_number == 8
        assert len(milestone.evidence) > 0
        assert 0.0 <= milestone.significance_score <= 1.0
        assert isinstance(milestone.achieved_date, datetime)
        
        # Check goal was updated
        goal = goal_tracker.goals[goal_id]
        assert milestone.milestone_id in goal.milestones
    
    @pytest.mark.asyncio
    async def test_analyze_turn_for_progress_positive(self, goal_tracker):
        """Test analyzing conversation turn for positive progress"""
        analysis = {'detected_progress': {}}
        
        await goal_tracker._analyze_turn_for_progress(
            "I feel much better and I'm improving every day", analysis
        )
        
        assert 'positive_sentiment' in analysis['detected_progress']
        assert analysis['detected_progress']['positive_sentiment'] > 0
    
    @pytest.mark.asyncio
    async def test_analyze_turn_for_progress_skill_application(self, goal_tracker):
        """Test analyzing conversation turn for skill application"""
        analysis = {'detected_progress': {}}
        
        await goal_tracker._analyze_turn_for_progress(
            "I used the technique you taught me and I practiced the skills", analysis
        )
        
        assert 'skill_application' in analysis['detected_progress']
        assert analysis['detected_progress']['skill_application'] > 0
    
    @pytest.mark.asyncio
    async def test_analyze_turn_for_progress_insight(self, goal_tracker):
        """Test analyzing conversation turn for insight development"""
        analysis = {'detected_progress': {}}
        
        await goal_tracker._analyze_turn_for_progress(
            "I realize now that I see the pattern in my behavior", analysis
        )
        
        assert 'insight_development' in analysis['detected_progress']
        assert analysis['detected_progress']['insight_development'] > 0
    
    @pytest.mark.asyncio
    async def test_analyze_turn_for_progress_behavioral_change(self, goal_tracker):
        """Test analyzing conversation turn for behavioral change"""
        analysis = {'detected_progress': {}}
        
        await goal_tracker._analyze_turn_for_progress(
            "I did something different this time and I acted differently", analysis
        )
        
        assert 'behavioral_change' in analysis['detected_progress']
        assert analysis['detected_progress']['behavioral_change'] > 0
    
    @pytest.mark.asyncio
    async def test_analyze_turn_for_goal_mentions(self, goal_tracker, sample_goal_data):
        """Test analyzing conversation turn for goal mentions"""
        goal_id = goal_tracker.create_goal(**sample_goal_data)
        analysis = {'goal_mentions': {}}
        
        await goal_tracker._analyze_turn_for_goal_mentions(
            "My anxiety symptoms are getting better", analysis
        )
        
        assert goal_id in analysis['goal_mentions']
        assert analysis['goal_mentions'][goal_id] > 0
    
    @pytest.mark.asyncio
    async def test_analyze_turn_for_barriers(self, goal_tracker):
        """Test analyzing conversation turn for barriers"""
        analysis = {'barriers_identified': []}
        
        await goal_tracker._analyze_turn_for_barriers(
            "This is really difficult and I'm struggling with everything", analysis
        )
        
        assert len(analysis['barriers_identified']) > 0
        assert any("difficult" in barrier for barrier in analysis['barriers_identified'])
        assert any("struggling" in barrier for barrier in analysis['barriers_identified'])
    
    @pytest.mark.asyncio
    async def test_analyze_conversation_for_progress_comprehensive(self, goal_tracker, progress_conversation, clinical_context, sample_goal_data):
        """Test comprehensive conversation analysis for progress"""
        goal_tracker.create_goal(**sample_goal_data)
        
        analysis = await goal_tracker.analyze_conversation_for_progress(
            progress_conversation, clinical_context, 5
        )
        
        assert isinstance(analysis, dict)
        assert 'detected_progress' in analysis
        assert 'goal_mentions' in analysis
        assert 'barriers_identified' in analysis
        assert 'interventions_suggested' in analysis
        assert 'milestone_candidates' in analysis
        
        # Should detect positive progress
        assert len(analysis['detected_progress']) > 0
        
        # Should have updated goal progress
        assert len(goal_tracker.progress_measurements) > 0
    
    @pytest.mark.asyncio
    async def test_analyze_conversation_with_barriers(self, goal_tracker, barrier_conversation, clinical_context, sample_goal_data):
        """Test conversation analysis with barriers"""
        goal_tracker.create_goal(**sample_goal_data)
        
        analysis = await goal_tracker.analyze_conversation_for_progress(
            barrier_conversation, clinical_context, 3
        )
        
        # Should identify barriers
        assert len(analysis['barriers_identified']) > 0
    
    def test_get_goal_progress_summary_existing_goal(self, goal_tracker, sample_goal_data):
        """Test getting progress summary for existing goal"""
        goal_id = goal_tracker.create_goal(**sample_goal_data)
        
        # Add some measurements
        measurements = [
            ProgressMeasurement(
                measurement_id=f"test_{i}",
                goal_id=goal_id,
                session_number=i,
                timestamp=datetime.now() + timedelta(days=i),
                progress_score=0.2 + (i * 0.1),
                indicator_type=ProgressIndicator.SELF_REPORT,
                evidence=f"Evidence {i}",
                confidence_level=0.8,
                therapist_notes=f"Notes {i}"
            )
            for i in range(1, 4)
        ]
        goal_tracker.progress_measurements.extend(measurements)
        
        summary = goal_tracker.get_goal_progress_summary(goal_id)
        
        assert summary is not None
        assert isinstance(summary, GoalProgressSummary)
        assert summary.goal_id == goal_id
        assert isinstance(summary.current_progress, float)
        assert summary.progress_trend in ["improving", "stable", "declining"]
        assert isinstance(summary.recent_milestones, list)
        assert isinstance(summary.barriers_identified, list)
        assert isinstance(summary.interventions_used, list)
        assert isinstance(summary.next_steps, list)
        assert 0.0 <= summary.confidence_in_achievement <= 1.0
    
    def test_get_goal_progress_summary_nonexistent_goal(self, goal_tracker):
        """Test getting progress summary for nonexistent goal"""
        summary = goal_tracker.get_goal_progress_summary("nonexistent")
        
        assert summary is None
    
    def test_calculate_progress_trend_improving(self, goal_tracker):
        """Test progress trend calculation for improving trend"""
        measurements = [
            ProgressMeasurement(
                measurement_id=f"test_{i}",
                goal_id="test_goal",
                session_number=i,
                timestamp=datetime.now() + timedelta(days=i),
                progress_score=0.1 + (i * 0.2),  # Clearly improving
                indicator_type=ProgressIndicator.SELF_REPORT,
                evidence=f"Evidence {i}",
                confidence_level=0.8,
                therapist_notes=f"Notes {i}"
            )
            for i in range(1, 6)
        ]
        
        trend = goal_tracker._calculate_progress_trend(measurements)
        
        assert trend == "improving"
    
    def test_calculate_progress_trend_declining(self, goal_tracker):
        """Test progress trend calculation for declining trend"""
        measurements = [
            ProgressMeasurement(
                measurement_id=f"test_{i}",
                goal_id="test_goal",
                session_number=i,
                timestamp=datetime.now() + timedelta(days=i),
                progress_score=0.8 - (i * 0.15),  # Clearly declining
                indicator_type=ProgressIndicator.SELF_REPORT,
                evidence=f"Evidence {i}",
                confidence_level=0.8,
                therapist_notes=f"Notes {i}"
            )
            for i in range(1, 6)
        ]
        
        trend = goal_tracker._calculate_progress_trend(measurements)
        
        assert trend == "declining"
    
    def test_calculate_progress_trend_stable(self, goal_tracker):
        """Test progress trend calculation for stable trend"""
        measurements = [
            ProgressMeasurement(
                measurement_id=f"test_{i}",
                goal_id="test_goal",
                session_number=i,
                timestamp=datetime.now() + timedelta(days=i),
                progress_score=0.5,  # Stable
                indicator_type=ProgressIndicator.SELF_REPORT,
                evidence=f"Evidence {i}",
                confidence_level=0.8,
                therapist_notes=f"Notes {i}"
            )
            for i in range(1, 6)
        ]
        
        trend = goal_tracker._calculate_progress_trend(measurements)
        
        assert trend == "stable"
    
    def test_calculate_progress_trend_insufficient_data(self, goal_tracker):
        """Test progress trend calculation with insufficient data"""
        measurements = [
            ProgressMeasurement(
                measurement_id="test_1",
                goal_id="test_goal",
                session_number=1,
                timestamp=datetime.now(),
                progress_score=0.5,
                indicator_type=ProgressIndicator.SELF_REPORT,
                evidence="Evidence",
                confidence_level=0.8,
                therapist_notes="Notes"
            )
        ]
        
        trend = goal_tracker._calculate_progress_trend(measurements)
        
        assert trend == "stable"
    
    def test_identify_recent_barriers(self, goal_tracker):
        """Test identifying recent barriers from measurements"""
        measurements = [
            ProgressMeasurement(
                measurement_id="test_1",
                goal_id="test_goal",
                session_number=1,
                timestamp=datetime.now(),
                progress_score=0.1,  # Low progress
                indicator_type=ProgressIndicator.SELF_REPORT,
                evidence="Client reported struggling with exercises",
                confidence_level=0.8,
                therapist_notes="Notes"
            ),
            ProgressMeasurement(
                measurement_id="test_2",
                goal_id="test_goal",
                session_number=2,
                timestamp=datetime.now(),
                progress_score=0.2,
                indicator_type=ProgressIndicator.SELF_REPORT,
                evidence="Client felt frustrated and overwhelmed",
                confidence_level=0.8,
                therapist_notes="Notes"
            )
        ]
        
        barriers = goal_tracker._identify_recent_barriers(measurements)
        
        assert isinstance(barriers, list)
        assert len(barriers) > 0
        assert any("low progress" in barrier.lower() for barrier in barriers)
        assert any("struggling" in barrier.lower() for barrier in barriers)
        assert any("frustrated" in barrier.lower() for barrier in barriers)
    
    def test_generate_next_steps_early_progress(self, goal_tracker, sample_goal_data):
        """Test generating next steps for early progress"""
        goal_id = goal_tracker.create_goal(**sample_goal_data)
        goal = goal_tracker.goals[goal_id]
        goal.progress_percentage = 20.0  # Early progress
        
        next_steps = goal_tracker._generate_next_steps(goal, [], [])
        
        assert isinstance(next_steps, list)
        assert len(next_steps) > 0
        assert any("foundational" in step.lower() for step in next_steps)
    
    def test_generate_next_steps_advanced_progress(self, goal_tracker, sample_goal_data):
        """Test generating next steps for advanced progress"""
        goal_id = goal_tracker.create_goal(**sample_goal_data)
        goal = goal_tracker.goals[goal_id]
        goal.progress_percentage = 80.0  # Advanced progress
        
        next_steps = goal_tracker._generate_next_steps(goal, [], [])
        
        assert isinstance(next_steps, list)
        assert len(next_steps) > 0
        assert any("completion" in step.lower() or "maintenance" in step.lower() for step in next_steps)
    
    def test_estimate_completion_date(self, goal_tracker, sample_goal_data):
        """Test estimating completion date"""
        goal_id = goal_tracker.create_goal(**sample_goal_data)
        goal = goal_tracker.goals[goal_id]
        goal.progress_percentage = 50.0
        
        measurements = [
            ProgressMeasurement(
                measurement_id="test_1",
                goal_id=goal_id,
                session_number=1,
                timestamp=datetime.now() - timedelta(weeks=2),
                progress_score=0.3,
                indicator_type=ProgressIndicator.SELF_REPORT,
                evidence="Evidence",
                confidence_level=0.8,
                therapist_notes="Notes"
            ),
            ProgressMeasurement(
                measurement_id="test_2",
                goal_id=goal_id,
                session_number=3,
                timestamp=datetime.now(),
                progress_score=0.5,
                indicator_type=ProgressIndicator.SELF_REPORT,
                evidence="Evidence",
                confidence_level=0.8,
                therapist_notes="Notes"
            )
        ]
        
        estimated_date = goal_tracker._estimate_completion_date(goal, measurements)
        
        if estimated_date:  # May be None if calculation not possible
            assert isinstance(estimated_date, datetime)
            assert estimated_date > datetime.now()
    
    def test_calculate_achievement_confidence(self, goal_tracker, sample_goal_data):
        """Test calculating achievement confidence"""
        goal_id = goal_tracker.create_goal(**sample_goal_data)
        goal = goal_tracker.goals[goal_id]
        goal.progress_percentage = 70.0
        
        measurements = [
            ProgressMeasurement(
                measurement_id="test_1",
                goal_id=goal_id,
                session_number=1,
                timestamp=datetime.now(),
                progress_score=0.7,
                indicator_type=ProgressIndicator.SELF_REPORT,
                evidence="Evidence",
                confidence_level=0.9,
                therapist_notes="Notes"
            )
        ]
        
        milestones = [
            TherapeuticMilestone(
                milestone_id="milestone_1",
                goal_id=goal_id,
                milestone_type=MilestoneType.SKILL_ACQUISITION,
                title="Test Milestone",
                description="Test",
                achieved_date=datetime.now(),
                session_number=3,
                evidence=["test"],
                significance_score=0.8,
                client_recognition=True,
                therapist_notes="Test"
            )
        ]
        
        confidence = goal_tracker._calculate_achievement_confidence(goal, measurements, milestones)
        
        assert isinstance(confidence, float)
        assert 0.0 <= confidence <= 1.0
        # Should be relatively high given good progress and milestone
        assert confidence > 0.5
    
    def test_get_all_goals_summary_empty(self, goal_tracker):
        """Test getting all goals summary with no goals"""
        summary = goal_tracker.get_all_goals_summary()
        
        assert isinstance(summary, dict)
        assert summary['total_goals'] == 0
        assert summary['goals_by_status'] == {}
        assert summary['goals_by_category'] == {}
        assert summary['goals_by_priority'] == {}
        assert summary['overall_progress'] == 0.0
        assert summary['total_milestones'] == 0
        assert summary['recent_progress'] == []
    
    def test_get_all_goals_summary_with_goals(self, goal_tracker, sample_goal_data):
        """Test getting all goals summary with goals"""
        # Create multiple goals
        goal_id_1 = goal_tracker.create_goal(**sample_goal_data)
        goal_id_2 = goal_tracker.create_goal(
            title="Build coping skills",
            description="Develop effective coping strategies",
            category=GoalCategory.SKILL_BUILDING,
            priority=GoalPriority.MEDIUM
        )
        
        # Set some progress
        goal_tracker.goals[goal_id_1].progress_percentage = 60.0
        goal_tracker.goals[goal_id_1].status = GoalStatus.PARTIALLY_ACHIEVED
        goal_tracker.goals[goal_id_2].progress_percentage = 30.0
        goal_tracker.goals[goal_id_2].status = GoalStatus.IN_PROGRESS
        
        summary = goal_tracker.get_all_goals_summary()
        
        assert summary['total_goals'] == 2
        assert summary['goals_by_status']['partially_achieved'] == 1
        assert summary['goals_by_status']['in_progress'] == 1
        assert summary['goals_by_category']['symptom_reduction'] == 1
        assert summary['goals_by_category']['skill_building'] == 1
        assert summary['goals_by_priority']['high'] == 1
        assert summary['goals_by_priority']['medium'] == 1
        assert summary['overall_progress'] == 45.0  # Average of 60 and 30
    
    def test_export_goal_data_json(self, goal_tracker, sample_goal_data):
        """Test exporting goal data as JSON"""
        goal_id = goal_tracker.create_goal(**sample_goal_data)
        
        json_data = goal_tracker.export_goal_data(format='json')
        
        assert isinstance(json_data, str)
        # Should be valid JSON
        import json
        parsed_data = json.loads(json_data)
        assert 'goals' in parsed_data
        assert 'progress_measurements' in parsed_data
        assert 'milestones' in parsed_data
        assert 'summary' in parsed_data
        
        # Check goal data
        assert goal_id in parsed_data['goals']
        goal_data = parsed_data['goals'][goal_id]
        assert goal_data['title'] == sample_goal_data['title']
        assert goal_data['category'] == sample_goal_data['category'].value
        assert goal_data['priority'] == sample_goal_data['priority'].value
    
    def test_export_goal_data_dict(self, goal_tracker, sample_goal_data):
        """Test exporting goal data as dictionary"""
        goal_id = goal_tracker.create_goal(**sample_goal_data)
        
        dict_data = goal_tracker.export_goal_data(format='dict')
        
        assert isinstance(dict_data, dict)
        assert 'goals' in dict_data
        assert 'progress_measurements' in dict_data
        assert 'milestones' in dict_data
        assert 'summary' in dict_data
        
        # Check goal data structure
        assert goal_id in dict_data['goals']
        goal_data = dict_data['goals'][goal_id]
        assert 'title' in goal_data
        assert 'description' in goal_data
        assert 'category' in goal_data
        assert 'priority' in goal_data
        assert 'status' in goal_data
        assert 'progress_percentage' in goal_data
        assert 'created_date' in goal_data
        assert 'success_criteria' in goal_data


if __name__ == "__main__":
    pytest.main([__file__])
