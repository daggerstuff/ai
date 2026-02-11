"""
Tests for VoiceTrainingOptimizer.
"""

from unittest.mock import Mock

import pytest

from ai.pipelines.orchestrator.conversation_schema import Conversation, Message
from ai.pipelines.orchestrator.voice_training_optimizer import (
    OptimizationResult,
    PersonalityProfile,
    VoiceOptimizationConfig,
    VoiceTrainingOptimizer,
)


class TestVoiceTrainingOptimizer:
    """Test cases for VoiceTrainingOptimizer."""

    @pytest.fixture
    def optimizer(self):
        """Create a VoiceTrainingOptimizer instance for testing."""
        config = VoiceOptimizationConfig(
            min_consistency_threshold=0.8,
            min_authenticity_threshold=0.7,
            batch_size=10,
            max_workers=2
        )
        return VoiceTrainingOptimizer(config=config)

    @pytest.fixture
    def sample_conversations(self):
        """Create sample conversations for testing."""
        conversations = []

        # High empathy conversation
        conv1 = Conversation(
            id="conv_1",
            messages=[
                Message(role="user", content="I'm feeling really sad today"),
                Message(role="assistant", content="I understand how you're feeling. That must be really difficult for you. I'm here to support you.")
            ]
        )
        conversations.append(conv1)

        # Analytical conversation
        conv2 = Conversation(
            id="conv_2",
            messages=[
                Message(role="user", content="Can you help me analyze this problem?"),
                Message(role="assistant", content="Let me think about this carefully. I need to consider all the factors involved.")
            ]
        )
        conversations.append(conv2)

        # Supportive conversation
        conv3 = Conversation(
            id="conv_3",
            messages=[
                Message(role="user", content="I don't think I can do this"),
                Message(role="assistant", content="I believe in you. You're stronger than you think and you can do this.")
            ]
        )
        conversations.append(conv3)

        return conversations

    @pytest.fixture
    def mock_personality_extractor(self):
        """Create a mock personality extractor."""
        extractor = Mock()
        extractor.extract_personality.return_value = {
            "big_five": {
                "openness": 0.7,
                "conscientiousness": 0.8,
                "extraversion": 0.6,
                "agreeableness": 0.9,
                "neuroticism": 0.3
            },
            "confidence": 0.85
        }
        return extractor

    def test_initialization(self):
        """Test VoiceTrainingOptimizer initialization."""
        config = VoiceOptimizationConfig(batch_size=20)
        optimizer = VoiceTrainingOptimizer(config=config)

        assert optimizer.config.batch_size == 20
        assert optimizer.baseline_profile is None
        assert len(optimizer.quality_validators) == 0
        assert len(optimizer.optimization_history) == 0

    def test_register_quality_validator(self, optimizer):
        """Test registering quality validator."""
        def dummy_validator(conversation):
            return {"valid": True, "score": 0.9}

        optimizer.register_quality_validator(dummy_validator)

        assert len(optimizer.quality_validators) == 1
        assert optimizer.quality_validators[0] == dummy_validator

    def test_extract_single_profile(self, optimizer, sample_conversations, mock_personality_extractor):
        """Test extracting single personality profile."""
        optimizer.personality_extractor = mock_personality_extractor

        profile = optimizer._extract_single_profile(sample_conversations[0])

        assert profile is not None
        assert isinstance(profile, PersonalityProfile)
        assert "openness" in profile.big_five_scores
        assert profile.authenticity_score > 0
        assert len(profile.empathy_indicators) > 0
        assert profile.sample_count == 1

    def test_extract_conversation_profiles(self, optimizer, sample_conversations, mock_personality_extractor):
        """Test extracting profiles from multiple conversations."""
        optimizer.personality_extractor = mock_personality_extractor

        profiles = optimizer._extract_conversation_profiles(sample_conversations)

        assert len(profiles) == 3
        assert "conv_1" in profiles
        assert "conv_2" in profiles
        assert "conv_3" in profiles

        for profile in profiles.values():
            assert isinstance(profile, PersonalityProfile)
            assert profile.authenticity_score > 0

    def test_establish_baseline_profile(self, optimizer):
        """Test establishing baseline personality profile."""
        # Create mock profiles
        profiles = {
            "conv_1": PersonalityProfile(
                big_five_scores={"openness": 0.8, "agreeableness": 0.9},
                communication_style={"supportive": 0.8, "formal": 0.3},
                emotional_patterns={"empathetic": 0.9, "positive": 0.7},
                empathy_indicators=["I understand", "that must be hard"],
                authenticity_score=0.85,
                confidence=0.9
            ),
            "conv_2": PersonalityProfile(
                big_five_scores={"openness": 0.7, "agreeableness": 0.8},
                communication_style={"supportive": 0.7, "formal": 0.4},
                emotional_patterns={"empathetic": 0.8, "positive": 0.8},
                empathy_indicators=["I understand", "I feel for you"],
                authenticity_score=0.8,
                confidence=0.85
            )
        }

        baseline = optimizer._establish_baseline_profile(profiles)

        assert isinstance(baseline, PersonalityProfile)
        assert baseline.big_five_scores["openness"] == 0.75  # Average of 0.8 and 0.7
        assert baseline.big_five_scores["agreeableness"] == 0.85  # Average of 0.9 and 0.8
        assert "I understand" in baseline.empathy_indicators
        assert baseline.sample_count == 2
        assert optimizer.baseline_profile == baseline

    def test_calculate_consistency_score(self, optimizer):
        """Test calculating consistency score between profiles."""
        baseline = PersonalityProfile(
            big_five_scores={"openness": 0.8, "agreeableness": 0.9},
            communication_style={"supportive": 0.8},
            emotional_patterns={"empathetic": 0.9}
        )

        # High consistency profile
        high_consistency_profile = PersonalityProfile(
            big_five_scores={"openness": 0.82, "agreeableness": 0.88},
            communication_style={"supportive": 0.78},
            emotional_patterns={"empathetic": 0.92}
        )

        # Low consistency profile
        low_consistency_profile = PersonalityProfile(
            big_five_scores={"openness": 0.3, "agreeableness": 0.4},
            communication_style={"supportive": 0.2},
            emotional_patterns={"empathetic": 0.1}
        )

        high_score = optimizer._calculate_consistency_score(high_consistency_profile, baseline)
        low_score = optimizer._calculate_consistency_score(low_consistency_profile, baseline)

        assert high_score > 0.9
        assert low_score < 0.5
        assert high_score > low_score

    def test_calculate_empathy_score(self, optimizer, sample_conversations):
        """Test calculating empathy score."""
        # High empathy conversation
        high_empathy_conv = sample_conversations[0]  # Contains "I understand", "difficult", "support"

        # Low empathy conversation
        low_empathy_conv = sample_conversations[1]  # More analytical, less empathetic

        high_score = optimizer._calculate_empathy_score(high_empathy_conv)
        low_score = optimizer._calculate_empathy_score(low_empathy_conv)

        assert high_score > low_score
        assert high_score > 0.3  # Should detect empathy indicators
        assert 0 <= low_score <= 1

    def test_calculate_authenticity_score(self, optimizer, sample_conversations):
        """Test calculating authenticity score."""
        conversation = sample_conversations[0]

        score = optimizer._calculate_authenticity_score(conversation)

        assert 0 <= score <= 1
        assert score > 0.5  # Should have reasonable authenticity

    def test_analyze_communication_style(self, optimizer, sample_conversations):
        """Test analyzing communication style."""
        supportive_conv = sample_conversations[2]  # Contains "believe", "support" language

        style = optimizer._analyze_communication_style(supportive_conv)

        assert isinstance(style, dict)
        assert "supportive" in style
        assert "formal" in style
        assert "casual" in style
        assert "analytical" in style

        # Should detect supportive style
        assert style["supportive"] > 0

        # All scores should be between 0 and 1
        for score in style.values():
            assert 0 <= score <= 1

    def test_analyze_emotional_patterns(self, optimizer, sample_conversations):
        """Test analyzing emotional patterns."""
        empathetic_conv = sample_conversations[0]  # Contains empathetic language

        patterns = optimizer._analyze_emotional_patterns(empathetic_conv)

        assert isinstance(patterns, dict)
        assert "empathetic" in patterns
        assert "positive" in patterns
        assert "calm" in patterns
        assert "encouraging" in patterns

        # Should detect empathetic patterns
        assert patterns["empathetic"] > 0

        # All scores should be between 0 and 1
        for score in patterns.values():
            assert 0 <= score <= 1

    def test_extract_empathy_indicators(self, optimizer, sample_conversations):
        """Test extracting empathy indicators."""
        empathetic_conv = sample_conversations[0]

        indicators = optimizer._extract_empathy_indicators(empathetic_conv)

        assert isinstance(indicators, list)
        assert len(indicators) > 0

        # Should find empathy phrases
        found_phrases = " ".join(indicators)
        assert any(phrase in found_phrases for phrase in ["understand", "difficult"])

    def test_filter_by_consistency(self, optimizer, sample_conversations):
        """Test filtering conversations by consistency."""
        # Create mock profiles with different consistency levels
        profiles = {
            "conv_1": PersonalityProfile(
                big_five_scores={"openness": 0.8, "agreeableness": 0.9},
                communication_style={"supportive": 0.8},
                emotional_patterns={"empathetic": 0.9}
            ),
            "conv_2": PersonalityProfile(
                big_five_scores={"openness": 0.2, "agreeableness": 0.1},  # Very different
                communication_style={"supportive": 0.1},
                emotional_patterns={"empathetic": 0.1}
            ),
            "conv_3": PersonalityProfile(
                big_five_scores={"openness": 0.82, "agreeableness": 0.88},  # Similar to baseline
                communication_style={"supportive": 0.78},
                emotional_patterns={"empathetic": 0.92}
            )
        }

        baseline = PersonalityProfile(
            big_five_scores={"openness": 0.8, "agreeableness": 0.9},
            communication_style={"supportive": 0.8},
            emotional_patterns={"empathetic": 0.9}
        )

        filtered = optimizer._filter_by_consistency(sample_conversations, profiles, baseline)

        # Should filter out the inconsistent conversation (conv_2)
        assert len(filtered) < len(sample_conversations)

        # Remaining conversations should have consistency scores in metadata
        for conv in filtered:
            assert "personality_consistency" in conv.meta
            assert conv.meta["personality_consistency"] >= optimizer.config.min_consistency_threshold

    def test_apply_quality_validation(self, optimizer, sample_conversations):
        """Test applying quality validation."""
        # Register validators
        def good_validator(conversation):
            return {"valid": True, "score": 0.9}

        def bad_validator(conversation):
            return {"valid": False, "score": 0.3}

        optimizer.register_quality_validator(good_validator)

        # All conversations should pass
        validated = optimizer._apply_quality_validation(sample_conversations)
        assert len(validated) == len(sample_conversations)

        # Add failing validator
        optimizer.register_quality_validator(bad_validator)

        # No conversations should pass now
        validated = optimizer._apply_quality_validation(sample_conversations)
        assert len(validated) == 0

    def test_optimize_voice_conversations_integration(self, optimizer, sample_conversations, mock_personality_extractor):
        """Test full optimization pipeline integration."""
        optimizer.personality_extractor = mock_personality_extractor

        # Register a permissive validator
        def permissive_validator(conversation):
            return {"valid": True, "score": 0.8}

        optimizer.register_quality_validator(permissive_validator)

        result = optimizer.optimize_voice_conversations(sample_conversations)

        assert isinstance(result, OptimizationResult)
        assert result.success is True
        assert len(result.optimized_conversations) <= len(sample_conversations)
        assert result.total_processed == len(sample_conversations)
        assert result.personality_profile is not None
        assert "overall_quality" in result.quality_metrics
        assert result.processing_time > 0

        # Check that conversations have optimization metadata
        for conv in result.optimized_conversations:
            assert "final_optimization_score" in conv.meta
            assert "personality_consistency" in conv.meta
            assert "empathy_score" in conv.meta
            assert "authenticity_score" in conv.meta

    def test_analyze_personality_consistency(self, optimizer, sample_conversations, mock_personality_extractor):
        """Test personality consistency analysis."""
        optimizer.personality_extractor = mock_personality_extractor

        analysis = optimizer.analyze_personality_consistency(sample_conversations)

        assert isinstance(analysis, dict)
        assert "consistency_score" in analysis
        assert "big_five_consistency" in analysis
        assert "communication_consistency" in analysis
        assert "emotional_consistency" in analysis
        assert "profile_count" in analysis
        assert "analysis" in analysis

        assert 0 <= analysis["consistency_score"] <= 1
        assert analysis["profile_count"] == len(sample_conversations)
        assert isinstance(analysis["analysis"], str)

    def test_get_baseline_profile(self, optimizer):
        """Test getting baseline profile."""
        assert optimizer.get_baseline_profile() is None

        # Set a baseline profile
        baseline = PersonalityProfile(big_five_scores={"openness": 0.8})
        optimizer.baseline_profile = baseline

        assert optimizer.get_baseline_profile() == baseline

    def test_get_optimization_history(self, optimizer):
        """Test getting optimization history."""
        history = optimizer.get_optimization_history()
        assert isinstance(history, list)
        assert len(history) == 0

        # Add a result to history
        result = OptimizationResult(success=True, total_processed=5)
        optimizer.optimization_history.append(result)

        history = optimizer.get_optimization_history()
        assert len(history) == 1
        assert history[0] == result


class TestPersonalityProfile:
    """Test cases for PersonalityProfile."""

    def test_initialization(self):
        """Test PersonalityProfile initialization."""
        profile = PersonalityProfile(
            big_five_scores={"openness": 0.8},
            communication_style={"supportive": 0.9},
            emotional_patterns={"empathetic": 0.7},
            empathy_indicators=["I understand"],
            authenticity_score=0.85,
            sample_count=5,
            confidence=0.9
        )

        assert profile.big_five_scores == {"openness": 0.8}
        assert profile.communication_style == {"supportive": 0.9}
        assert profile.emotional_patterns == {"empathetic": 0.7}
        assert profile.empathy_indicators == ["I understand"]
        assert profile.authenticity_score == 0.85
        assert profile.sample_count == 5
        assert profile.confidence == 0.9


class TestVoiceOptimizationConfig:
    """Test cases for VoiceOptimizationConfig."""

    def test_initialization(self):
        """Test VoiceOptimizationConfig initialization."""
        config = VoiceOptimizationConfig(
            min_consistency_threshold=0.9,
            batch_size=100,
            max_workers=8
        )

        assert config.min_consistency_threshold == 0.9
        assert config.batch_size == 100
        assert config.max_workers == 8
        assert config.enable_cross_validation is True  # Default value


class TestOptimizationResult:
    """Test cases for OptimizationResult."""

    def test_initialization(self):
        """Test OptimizationResult initialization."""
        conversations = [Mock(spec=Conversation)]
        profile = PersonalityProfile()

        result = OptimizationResult(
            success=True,
            optimized_conversations=conversations,
            personality_profile=profile,
            quality_metrics={"score": 0.8},
            filtered_count=5,
            total_processed=10,
            processing_time=1.5
        )

        assert result.success is True
        assert result.optimized_conversations == conversations
        assert result.personality_profile == profile
        assert result.quality_metrics == {"score": 0.8}
        assert result.filtered_count == 5
        assert result.total_processed == 10
        assert result.processing_time == 1.5
        assert result.issues == []  # Default value
