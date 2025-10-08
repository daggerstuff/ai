"""
VoiceTrainingOptimizer orchestration class for personality consistency.
Provides comprehensive optimization and validation for voice-derived training data.
"""

import statistics
from collections import defaultdict
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from conversation_schema import Conversation
from logger import get_logger
from personality_extractor import PersonalityExtractor
from voice_types import OptimizationResult, PersonalityProfile


@dataclass
class VoiceOptimizationConfig:
    """Configuration for voice training optimization."""

    min_consistency_threshold: float = 0.85
    min_authenticity_threshold: float = 0.8
    max_personality_drift: float = 0.15
    empathy_weight: float = 0.3
    communication_weight: float = 0.25
    emotional_weight: float = 0.25
    consistency_weight: float = 0.2
    batch_size: int = 50
    max_workers: int = 4
    enable_cross_validation: bool = True


@dataclass
class OptimizationResult:
    """Result of voice training optimization."""

    success: bool
    optimized_conversations: list[Conversation] = field(default_factory=list)
    personality_profile: PersonalityProfile | None = None
    quality_metrics: dict[str, float] = field(default_factory=dict)
    filtered_count: int = 0
    total_processed: int = 0
    issues: list[str] = field(default_factory=list)
    processing_time: float = 0.0


class VoiceTrainingOptimizer:
    """
    Orchestration class for voice training optimization with personality consistency.

    Features:
    - Comprehensive personality profiling from voice data
    - Cross-sample consistency validation
    - Authenticity scoring and filtering
    - Empathy and communication style analysis
    - Quality-based conversation selection
    - Performance monitoring and optimization
    """

    def __init__(
        self,
        config: VoiceOptimizationConfig | None = None,
        personality_extractor: PersonalityExtractor | None = None,
    ):
        """
        Initialize VoiceTrainingOptimizer.

        Args:
            config: Optimization configuration
            personality_extractor: PersonalityExtractor instance
        """
        self.config = config or VoiceOptimizationConfig()
        self.personality_extractor = personality_extractor or PersonalityExtractor()
        self.logger = get_logger(__name__)

        # Optimization state
        self.baseline_profile: PersonalityProfile | None = None
        self.conversation_profiles: dict[str, PersonalityProfile] = {}
        self.quality_validators: list[Callable] = []

        # Performance tracking
        self.optimization_history: list[OptimizationResult] = []

        self.logger.info("VoiceTrainingOptimizer initialized")

    def register_quality_validator(
        self, validator: Callable[[Conversation], dict[str, Any]]
    ) -> None:
        """Register a quality validation function."""
        self.quality_validators.append(validator)
        self.logger.info("Registered quality validator")

    def optimize_voice_conversations(
        self,
        conversations: list[Conversation],
        source_metadata: dict[str, Any] | None = None,
    ) -> OptimizationResult:
        """
        Optimize voice conversations for personality consistency and quality.

        Args:
            conversations: List of voice-derived conversations
            source_metadata: Optional metadata about the voice source

        Returns:
            OptimizationResult with optimized conversations and metrics
        """
        start_time = datetime.now()

        self.logger.info(
            f"Starting optimization of {len(conversations)} voice conversations"
        )

        # Step 1: Extract personality profiles for all conversations
        conversation_profiles = self._extract_conversation_profiles(conversations)

        # Step 2: Establish baseline personality profile
        baseline_profile = self._establish_baseline_profile(conversation_profiles)

        # Step 3: Filter conversations based on consistency
        consistent_conversations = self._filter_by_consistency(
            conversations, conversation_profiles, baseline_profile
        )

        # Step 4: Score authenticity and empathy
        scored_conversations = self._score_authenticity_and_empathy(
            consistent_conversations
        )

        # Step 5: Apply quality validation
        validated_conversations = self._apply_quality_validation(scored_conversations)

        # Step 6: Final optimization and ranking
        optimized_conversations = self._final_optimization_ranking(
            validated_conversations
        )

        # Calculate metrics
        processing_time = (datetime.now() - start_time).total_seconds()
        quality_metrics = self._calculate_quality_metrics(
            optimized_conversations, baseline_profile
        )

        result = OptimizationResult(
            success=True,
            optimized_conversations=optimized_conversations,
            personality_profile=baseline_profile,
            quality_metrics=quality_metrics,
            filtered_count=len(conversations) - len(optimized_conversations),
            total_processed=len(conversations),
            processing_time=processing_time,
        )

        # Store result for analysis
        self.optimization_history.append(result)

        self.logger.info(
            f"Optimization complete: {len(optimized_conversations)}/{len(conversations)} "
            f"conversations retained (quality score: {quality_metrics.get('overall_quality', 0):.3f})"
        )

        return result

    def analyze_personality_consistency(
        self, conversations: list[Conversation]
    ) -> dict[str, Any]:
        """
        Analyze personality consistency across conversations.

        Args:
            conversations: List of conversations to analyze

        Returns:
            Consistency analysis results
        """
        profiles = self._extract_conversation_profiles(conversations)

        if not profiles:
            return {"consistency_score": 0.0, "analysis": "No valid profiles extracted"}

        # Calculate consistency metrics
        big_five_consistency = self._calculate_big_five_consistency(profiles)
        communication_consistency = self._calculate_communication_consistency(profiles)
        emotional_consistency = self._calculate_emotional_consistency(profiles)

        overall_consistency = (
            big_five_consistency * 0.4
            + communication_consistency * 0.3
            + emotional_consistency * 0.3
        )

        return {
            "consistency_score": overall_consistency,
            "big_five_consistency": big_five_consistency,
            "communication_consistency": communication_consistency,
            "emotional_consistency": emotional_consistency,
            "profile_count": len(profiles),
            "analysis": self._generate_consistency_analysis(profiles),
        }

    def get_baseline_profile(self) -> PersonalityProfile | None:
        """Get the established baseline personality profile."""
        return self.baseline_profile

    def get_optimization_history(self) -> list[OptimizationResult]:
        """Get optimization history for analysis."""
        return self.optimization_history

    # Private methods

    def _extract_conversation_profiles(
        self, conversations: list[Conversation]
    ) -> dict[str, PersonalityProfile]:
        """Extract personality profiles from conversations."""
        profiles = {}

        with ThreadPoolExecutor(max_workers=self.config.max_workers) as executor:
            # Submit profile extraction tasks
            future_to_conv = {
                executor.submit(self._extract_single_profile, conv): conv.id
                or f"conv_{i}"
                for i, conv in enumerate(conversations)
            }

            # Collect results
            for future in as_completed(future_to_conv):
                conv_id = future_to_conv[future]
                try:
                    profile = future.result()
                    if profile:
                        profiles[conv_id] = profile
                except Exception as e:
                    self.logger.warning(f"Failed to extract profile for {conv_id}: {e}")

        self.logger.info(f"Extracted {len(profiles)} personality profiles")
        return profiles

    def _extract_single_profile(
        self, conversation: Conversation
    ) -> PersonalityProfile | None:
        """Extract personality profile from a single conversation."""
        try:
            # Extract text content
            text_content = " ".join([msg.content for msg in conversation.messages])

            # Get personality analysis
            personality_analysis = self.personality_extractor.extract_personality(
                text_content
            )

            # Extract communication style
            communication_style = self._analyze_communication_style(conversation)

            # Extract emotional patterns
            emotional_patterns = self._analyze_emotional_patterns(conversation)

            # Extract empathy indicators
            empathy_indicators = self._extract_empathy_indicators(conversation)

            # Calculate authenticity score
            authenticity_score = self._calculate_authenticity_score(conversation)

            return PersonalityProfile(
                big_five_scores=personality_analysis.get("big_five", {}),
                communication_style=communication_style,
                emotional_patterns=emotional_patterns,
                empathy_indicators=empathy_indicators,
                authenticity_score=authenticity_score,
                sample_count=1,
                confidence=personality_analysis.get("confidence", 0.5),
            )


        except Exception as e:
            self.logger.error(f"Error extracting profile: {e}")
            return None

    def _establish_baseline_profile(
        self, profiles: dict[str, PersonalityProfile]
    ) -> PersonalityProfile:
        """Establish baseline personality profile from all profiles."""
        if not profiles:
            return PersonalityProfile()

        # Aggregate Big Five scores
        big_five_aggregated = defaultdict(list)
        communication_aggregated = defaultdict(list)
        emotional_aggregated = defaultdict(list)
        all_empathy_indicators = []
        authenticity_scores = []

        for profile in profiles.values():
            # Collect Big Five scores
            for trait, score in profile.big_five_scores.items():
                big_five_aggregated[trait].append(score)

            # Collect communication style scores
            for style, score in profile.communication_style.items():
                communication_aggregated[style].append(score)

            # Collect emotional patterns
            for emotion, score in profile.emotional_patterns.items():
                emotional_aggregated[emotion].append(score)

            # Collect empathy indicators
            all_empathy_indicators.extend(profile.empathy_indicators)

            # Collect authenticity scores
            authenticity_scores.append(profile.authenticity_score)

        # Calculate baseline averages
        baseline_big_five = {
            trait: statistics.mean(scores)
            for trait, scores in big_five_aggregated.items()
        }
        baseline_communication = {
            style: statistics.mean(scores)
            for style, scores in communication_aggregated.items()
        }
        baseline_emotional = {
            emotion: statistics.mean(scores)
            for emotion, scores in emotional_aggregated.items()
        }

        # Get most common empathy indicators
        empathy_counts = defaultdict(int)
        for indicator in all_empathy_indicators:
            empathy_counts[indicator] += 1

        common_empathy_indicators = [
            indicator
            for indicator, count in empathy_counts.items()
            if count >= len(profiles) * 0.3  # Present in at least 30% of profiles
        ]

        baseline_profile = PersonalityProfile(
            big_five_scores=baseline_big_five,
            communication_style=baseline_communication,
            emotional_patterns=baseline_emotional,
            empathy_indicators=common_empathy_indicators,
            authenticity_score=(
                statistics.mean(authenticity_scores) if authenticity_scores else 0.0
            ),
            sample_count=len(profiles),
            confidence=statistics.mean([p.confidence for p in profiles.values()]),
        )

        self.baseline_profile = baseline_profile
        return baseline_profile

    def _filter_by_consistency(
        self,
        conversations: list[Conversation],
        profiles: dict[str, PersonalityProfile],
        baseline: PersonalityProfile,
    ) -> list[Conversation]:
        """Filter conversations based on personality consistency."""
        consistent_conversations = []

        for i, conversation in enumerate(conversations):
            conv_id = conversation.id or f"conv_{i}"
            profile = profiles.get(conv_id)

            if not profile:
                continue

            # Calculate consistency score
            consistency_score = self._calculate_consistency_score(profile, baseline)

            if consistency_score >= self.config.min_consistency_threshold:
                # Add consistency score to conversation metadata
                if not conversation.meta:
                    conversation.meta = {}
                conversation.meta["personality_consistency"] = consistency_score
                consistent_conversations.append(conversation)

        self.logger.info(
            f"Filtered to {len(consistent_conversations)} consistent conversations"
        )
        return consistent_conversations

    def _score_authenticity_and_empathy(
        self, conversations: list[Conversation]
    ) -> list[Conversation]:
        """Score conversations for authenticity and empathy."""
        scored_conversations = []

        for conversation in conversations:
            # Calculate empathy score
            empathy_score = self._calculate_empathy_score(conversation)

            # Get authenticity score (already calculated in profile extraction)
            authenticity_score = conversation.meta.get("authenticity_score", 0.0)

            # Calculate combined score
            combined_score = (
                empathy_score * self.config.empathy_weight
                + authenticity_score * (1 - self.config.empathy_weight)
            )

            # Add scores to metadata
            if not conversation.meta:
                conversation.meta = {}
            conversation.meta["empathy_score"] = empathy_score
            conversation.meta["authenticity_score"] = authenticity_score
            conversation.meta["combined_quality_score"] = combined_score

            # Filter by minimum thresholds
            if (
                authenticity_score >= self.config.min_authenticity_threshold
                and empathy_score >= 0.5
            ):  # Minimum empathy threshold
                scored_conversations.append(conversation)

        self.logger.info(
            f"Scored and filtered to {len(scored_conversations)} conversations"
        )
        return scored_conversations

    def _apply_quality_validation(
        self, conversations: list[Conversation]
    ) -> list[Conversation]:
        """Apply registered quality validators."""
        if not self.quality_validators:
            return conversations

        validated_conversations = []

        for conversation in conversations:
            passed_validation = True
            validation_scores = []

            for validator in self.quality_validators:
                try:
                    result = validator(conversation)
                    if isinstance(result, dict):
                        if not result.get("valid", True):
                            passed_validation = False
                            break
                        validation_scores.append(result.get("score", 1.0))
                    elif isinstance(result, bool):
                        if not result:
                            passed_validation = False
                            break
                        validation_scores.append(1.0)
                except Exception as e:
                    self.logger.warning(f"Validator failed: {e}")
                    passed_validation = False
                    break

            if passed_validation:
                # Add validation score to metadata
                if not conversation.meta:
                    conversation.meta = {}
                conversation.meta["validation_score"] = (
                    statistics.mean(validation_scores) if validation_scores else 1.0
                )
                validated_conversations.append(conversation)

        self.logger.info(f"Validated {len(validated_conversations)} conversations")
        return validated_conversations

    def _final_optimization_ranking(
        self, conversations: list[Conversation]
    ) -> list[Conversation]:
        """Final optimization and ranking of conversations."""
        # Calculate final scores for ranking
        for conversation in conversations:
            meta = conversation.meta or {}

            final_score = (
                meta.get("personality_consistency", 0.0)
                * self.config.consistency_weight
                + meta.get("empathy_score", 0.0) * self.config.empathy_weight
                + meta.get("authenticity_score", 0.0)
                * (1 - self.config.consistency_weight - self.config.empathy_weight)
                + meta.get("validation_score", 1.0) * 0.1
            )

            conversation.meta["final_optimization_score"] = final_score

        # Sort by final score (descending)
        return sorted(
            conversations,
            key=lambda c: c.meta.get("final_optimization_score", 0.0),
            reverse=True,
        )


    def _analyze_communication_style(
        self, conversation: Conversation
    ) -> dict[str, float]:
        """Analyze communication style from conversation."""
        text_content = " ".join([msg.content for msg in conversation.messages]).lower()

        # Communication style indicators
        styles = {
            "formal": len(
                [
                    w
                    for w in ["please", "thank you", "would you", "could you"]
                    if w in text_content
                ]
            )
            / 10,
            "casual": len(
                [w for w in ["yeah", "ok", "cool", "awesome"] if w in text_content]
            )
            / 10,
            "supportive": len(
                [
                    w
                    for w in ["understand", "support", "help", "care"]
                    if w in text_content
                ]
            )
            / 10,
            "analytical": len(
                [
                    w
                    for w in ["analyze", "consider", "think", "reason"]
                    if w in text_content
                ]
            )
            / 10,
        }

        # Normalize scores to 0-1 range
        return {style: min(1.0, score) for style, score in styles.items()}

    def _analyze_emotional_patterns(
        self, conversation: Conversation
    ) -> dict[str, float]:
        """Analyze emotional patterns from conversation."""
        text_content = " ".join([msg.content for msg in conversation.messages]).lower()

        # Emotional pattern indicators
        emotions = {
            "positive": len(
                [
                    w
                    for w in ["happy", "good", "great", "wonderful", "excited"]
                    if w in text_content
                ]
            )
            / 10,
            "empathetic": len(
                [
                    w
                    for w in ["sorry", "understand", "feel", "difficult"]
                    if w in text_content
                ]
            )
            / 10,
            "calm": len(
                [
                    w
                    for w in ["calm", "peaceful", "relax", "breathe"]
                    if w in text_content
                ]
            )
            / 10,
            "encouraging": len(
                [
                    w
                    for w in ["can do", "believe", "strong", "capable"]
                    if w in text_content
                ]
            )
            / 10,
        }

        # Normalize scores to 0-1 range
        return {emotion: min(1.0, score) for emotion, score in emotions.items()}

    def _extract_empathy_indicators(self, conversation: Conversation) -> list[str]:
        """Extract empathy indicators from conversation."""
        text_content = " ".join([msg.content for msg in conversation.messages]).lower()

        empathy_phrases = [
            "i understand",
            "that must be",
            "i can imagine",
            "sounds difficult",
            "i hear you",
            "that's hard",
            "you're not alone",
            "i feel for you",
        ]

        found_indicators = []
        for phrase in empathy_phrases:
            if phrase in text_content:
                found_indicators.append(phrase)

        return found_indicators

    def _calculate_authenticity_score(self, conversation: Conversation) -> float:
        """Calculate authenticity score for conversation."""
        # This would typically involve more sophisticated analysis
        # For now, use simple heuristics

        text_content = " ".join([msg.content for msg in conversation.messages])

        # Authenticity indicators
        authenticity_score = 0.5  # Base score

        # Natural language patterns
        if any(filler in text_content.lower() for filler in ["um", "uh", "you know"]):
            authenticity_score += 0.1

        # Personal pronouns usage
        personal_pronouns = len(
            [
                w
                for w in text_content.lower().split()
                if w in ["i", "me", "my", "myself"]
            ]
        )
        if personal_pronouns > 0:
            authenticity_score += min(0.2, personal_pronouns / 20)

        # Emotional expressions
        if any(
            emotion in text_content.lower()
            for emotion in ["feel", "feeling", "emotion", "heart"]
        ):
            authenticity_score += 0.1

        # Conversational flow
        if len(conversation.messages) > 2:
            authenticity_score += 0.1

        return min(1.0, authenticity_score)

    def _calculate_consistency_score(
        self, profile: PersonalityProfile, baseline: PersonalityProfile
    ) -> float:
        """Calculate consistency score between profile and baseline."""
        scores = []

        # Big Five consistency
        for trait in baseline.big_five_scores:
            if trait in profile.big_five_scores:
                diff = abs(
                    profile.big_five_scores[trait] - baseline.big_five_scores[trait]
                )
                consistency = max(0.0, 1.0 - diff)
                scores.append(consistency)

        # Communication style consistency
        for style in baseline.communication_style:
            if style in profile.communication_style:
                diff = abs(
                    profile.communication_style[style]
                    - baseline.communication_style[style]
                )
                consistency = max(0.0, 1.0 - diff)
                scores.append(consistency)

        # Emotional pattern consistency
        for emotion in baseline.emotional_patterns:
            if emotion in profile.emotional_patterns:
                diff = abs(
                    profile.emotional_patterns[emotion]
                    - baseline.emotional_patterns[emotion]
                )
                consistency = max(0.0, 1.0 - diff)
                scores.append(consistency)

        return statistics.mean(scores) if scores else 0.0

    def _calculate_empathy_score(self, conversation: Conversation) -> float:
        """Calculate empathy score for conversation."""
        empathy_indicators = self._extract_empathy_indicators(conversation)

        # Base score from number of empathy indicators
        base_score = min(1.0, len(empathy_indicators) / 5)

        # Bonus for variety of empathy expressions
        variety_bonus = min(0.2, len(set(empathy_indicators)) / 10)

        return min(1.0, base_score + variety_bonus)

    def _calculate_quality_metrics(
        self, conversations: list[Conversation], baseline: PersonalityProfile
    ) -> dict[str, float]:
        """Calculate overall quality metrics."""
        if not conversations:
            return {"overall_quality": 0.0}

        # Collect all scores
        consistency_scores = []
        empathy_scores = []
        authenticity_scores = []
        final_scores = []

        for conv in conversations:
            meta = conv.meta or {}
            consistency_scores.append(meta.get("personality_consistency", 0.0))
            empathy_scores.append(meta.get("empathy_score", 0.0))
            authenticity_scores.append(meta.get("authenticity_score", 0.0))
            final_scores.append(meta.get("final_optimization_score", 0.0))

        return {
            "overall_quality": statistics.mean(final_scores),
            "average_consistency": statistics.mean(consistency_scores),
            "average_empathy": statistics.mean(empathy_scores),
            "average_authenticity": statistics.mean(authenticity_scores),
            "quality_variance": (
                statistics.stdev(final_scores) if len(final_scores) > 1 else 0.0
            ),
            "baseline_confidence": baseline.confidence,
        }

    def _calculate_big_five_consistency(
        self, profiles: dict[str, PersonalityProfile]
    ) -> float:
        """Calculate Big Five consistency across profiles."""
        if len(profiles) < 2:
            return 1.0

        trait_scores = defaultdict(list)
        for profile in profiles.values():
            for trait, score in profile.big_five_scores.items():
                trait_scores[trait].append(score)

        consistencies = []
        for trait, scores in trait_scores.items():
            if len(scores) > 1:
                variance = statistics.stdev(scores)
                consistency = max(0.0, 1.0 - variance)
                consistencies.append(consistency)

        return statistics.mean(consistencies) if consistencies else 1.0

    def _calculate_communication_consistency(
        self, profiles: dict[str, PersonalityProfile]
    ) -> float:
        """Calculate communication style consistency."""
        if len(profiles) < 2:
            return 1.0

        style_scores = defaultdict(list)
        for profile in profiles.values():
            for style, score in profile.communication_style.items():
                style_scores[style].append(score)

        consistencies = []
        for style, scores in style_scores.items():
            if len(scores) > 1:
                variance = statistics.stdev(scores)
                consistency = max(0.0, 1.0 - variance)
                consistencies.append(consistency)

        return statistics.mean(consistencies) if consistencies else 1.0

    def _calculate_emotional_consistency(
        self, profiles: dict[str, PersonalityProfile]
    ) -> float:
        """Calculate emotional pattern consistency."""
        if len(profiles) < 2:
            return 1.0

        emotion_scores = defaultdict(list)
        for profile in profiles.values():
            for emotion, score in profile.emotional_patterns.items():
                emotion_scores[emotion].append(score)

        consistencies = []
        for emotion, scores in emotion_scores.items():
            if len(scores) > 1:
                variance = statistics.stdev(scores)
                consistency = max(0.0, 1.0 - variance)
                consistencies.append(consistency)

        return statistics.mean(consistencies) if consistencies else 1.0

    def _generate_consistency_analysis(
        self, profiles: dict[str, PersonalityProfile]
    ) -> str:
        """Generate human-readable consistency analysis."""
        if not profiles:
            return "No profiles available for analysis"

        profile_count = len(profiles)

        # Analyze Big Five consistency
        big_five_consistency = self._calculate_big_five_consistency(profiles)

        # Analyze empathy indicators
        all_empathy = []
        for profile in profiles.values():
            all_empathy.extend(profile.empathy_indicators)

        unique_empathy = len(set(all_empathy))

        analysis_parts = [
            f"Analyzed {profile_count} personality profiles",
            f"Big Five consistency: {big_five_consistency:.2f}",
            f"Unique empathy indicators: {unique_empathy}",
        ]

        if big_five_consistency >= 0.8:
            analysis_parts.append("High personality consistency detected")
        elif big_five_consistency >= 0.6:
            analysis_parts.append("Moderate personality consistency")
        else:
            analysis_parts.append("Low personality consistency - may need filtering")

        return ". ".join(analysis_parts)
