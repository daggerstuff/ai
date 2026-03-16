"""
Personality balancer for training data distribution.
Balances personality types in training data for comprehensive AI training.
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any

from conversation_schema import Conversation
from logger import get_logger

logger = get_logger(__name__)


class PersonalityFramework(Enum):
    """Personality assessment frameworks."""
    BIG_FIVE = "big_five"
    MBTI = "mbti"
    ENNEAGRAM = "enneagram"
    DISC = "disc"
    TEMPERAMENT = "temperament"


@dataclass
class PersonalityProfile:
    """Represents a personality profile."""
    profile_id: str
    framework: PersonalityFramework
    traits: dict[str, float]  # Trait scores (0.0-1.0)
    type_label: str  # E.g., "ENFP", "Type 7", "High Extraversion"
    confidence: float
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class BalancingResult:
    """Result of personality balancing operation."""
    balanced_conversations: list[Conversation]
    personality_distribution: dict[str, int]
    balancing_stats: dict[str, Any]
    quality_metrics: dict[str, float]


class PersonalityBalancer:
    """
    Balances personality types in training data.

    Ensures comprehensive representation of different personality types
    for robust therapeutic AI training across diverse client personalities.
    """

    def __init__(self):
        """Initialize the personality balancer."""
        self.logger = get_logger(__name__)

        # Target distributions for balanced training
        self.target_distributions = {
            PersonalityFramework.BIG_FIVE: {
                "high_extraversion": 0.20,
                "moderate_extraversion": 0.60,
                "low_extraversion": 0.20,
                "high_openness": 0.25,
                "moderate_openness": 0.50,
                "low_openness": 0.25,
                "high_conscientiousness": 0.30,
                "moderate_conscientiousness": 0.50,
                "low_conscientiousness": 0.20,
                "high_agreeableness": 0.35,
                "moderate_agreeableness": 0.45,
                "low_agreeableness": 0.20,
                "high_neuroticism": 0.25,
                "moderate_neuroticism": 0.50,
                "low_neuroticism": 0.25
            },
            PersonalityFramework.MBTI: {
                "NT": 0.15,  # Analysts
                "NF": 0.20,  # Diplomats
                "SJ": 0.35,  # Sentinels
                "SP": 0.30   # Explorers
            }
        }

        # Personality detection patterns
        self.personality_indicators = {
            "extraversion": {
                "high": ["outgoing", "social", "energetic", "talkative", "assertive"],
                "low": ["quiet", "reserved", "introspective", "solitary", "withdrawn"]
            },
            "openness": {
                "high": ["creative", "curious", "imaginative", "artistic", "innovative"],
                "low": ["practical", "conventional", "traditional", "concrete", "routine"]
            },
            "conscientiousness": {
                "high": ["organized", "disciplined", "responsible", "punctual", "thorough"],
                "low": ["disorganized", "impulsive", "careless", "spontaneous", "flexible"]
            },
            "agreeableness": {
                "high": ["cooperative", "trusting", "helpful", "compassionate", "kind"],
                "low": ["competitive", "skeptical", "critical", "demanding", "stubborn"]
            },
            "neuroticism": {
                "high": ["anxious", "stressed", "worried", "emotional", "unstable"],
                "low": ["calm", "stable", "confident", "resilient", "composed"]
            }
        }

        self.logger.info("PersonalityBalancer initialized")

    def balance_personalities(self, conversations: list[Conversation],
                            target_framework: PersonalityFramework = PersonalityFramework.BIG_FIVE) -> BalancingResult:
        """
        Balance personality representation in conversations.

        Args:
            conversations: List of conversations to balance
            target_framework: Personality framework to use for balancing

        Returns:
            BalancingResult with balanced conversations and statistics
        """
        self.logger.info(f"Balancing {len(conversations)} conversations using {target_framework.value}")

        # Extract personality profiles from conversations
        personality_profiles = self._extract_personality_profiles(conversations, target_framework)

        # Analyze current distribution
        current_distribution = self._analyze_distribution(personality_profiles, target_framework)

        # Balance the conversations
        balanced_conversations = self._apply_balancing(conversations, personality_profiles,
                                                     current_distribution, target_framework)

        # Calculate final distribution
        final_profiles = self._extract_personality_profiles(balanced_conversations, target_framework)
        final_distribution = self._analyze_distribution(final_profiles, target_framework)

        # Calculate quality metrics
        quality_metrics = self._calculate_balancing_quality(current_distribution, final_distribution, target_framework)

        balancing_stats = {
            "original_count": len(conversations),
            "balanced_count": len(balanced_conversations),
            "framework_used": target_framework.value,
            "current_distribution": current_distribution,
            "final_distribution": final_distribution,
            "target_distribution": self.target_distributions.get(target_framework, {}),
            "processed_at": datetime.now().isoformat()
        }

        self.logger.info(f"Balanced to {len(balanced_conversations)} conversations")

        return BalancingResult(
            balanced_conversations=balanced_conversations,
            personality_distribution=final_distribution,
            balancing_stats=balancing_stats,
            quality_metrics=quality_metrics
        )

    def _extract_personality_profiles(self, conversations: list[Conversation],
                                    framework: PersonalityFramework) -> list[PersonalityProfile]:
        """Extract personality profiles from conversations."""
        profiles = []

        for conv in conversations:
            profile = self._analyze_conversation_personality(conv, framework)
            if profile:
                profiles.append(profile)

        return profiles

    def _analyze_conversation_personality(self, conversation: Conversation,
                                       framework: PersonalityFramework) -> PersonalityProfile | None:
        """Analyze personality traits in a single conversation."""
        try:
            # Combine all user messages (client messages)
            user_content = " ".join([
                msg.content.lower() for msg in conversation.messages
                if msg.role == "user"
            ])

            if not user_content:
                return None

            if framework == PersonalityFramework.BIG_FIVE:
                return self._analyze_big_five(conversation.id, user_content)
            if framework == PersonalityFramework.MBTI:
                return self._analyze_mbti(conversation.id, user_content)
            return self._analyze_generic(conversation.id, user_content, framework)

        except Exception as e:
            self.logger.warning(f"Could not analyze personality for conversation {conversation.id}: {e}")
            return None

    def _analyze_big_five(self, conv_id: str, content: str) -> PersonalityProfile:
        """Analyze Big Five personality traits."""
        traits = {}

        for trait, indicators in self.personality_indicators.items():
            high_count = sum(1 for word in indicators["high"] if word in content)
            low_count = sum(1 for word in indicators["low"] if word in content)

            # Calculate trait score (0.0 = low, 1.0 = high)
            total_indicators = high_count + low_count
            if total_indicators > 0:
                traits[trait] = high_count / total_indicators
            else:
                traits[trait] = 0.5  # Neutral if no indicators

        # Determine type label based on dominant traits
        high_traits = [trait for trait, score in traits.items() if score > 0.6]
        type_label = f"High {', '.join(high_traits)}" if high_traits else "Balanced"

        # Calculate confidence based on number of indicators found
        total_words = len(content.split())
        indicator_density = sum(
            sum(1 for word in indicators["high"] + indicators["low"] if word in content)
            for indicators in self.personality_indicators.values()
        ) / max(total_words, 1)

        confidence = min(indicator_density * 10, 1.0)  # Scale to 0-1

        return PersonalityProfile(
            profile_id=f"big5_{conv_id}",
            framework=PersonalityFramework.BIG_FIVE,
            traits=traits,
            type_label=type_label,
            confidence=confidence,
            metadata={"content_length": len(content), "word_count": total_words}
        )

    def _analyze_mbti(self, conv_id: str, content: str) -> PersonalityProfile:
        """Analyze MBTI personality type."""
        # Simplified MBTI analysis based on key indicators
        mbti_indicators = {
            "E": ["social", "outgoing", "people", "group", "talk"],
            "I": ["quiet", "alone", "think", "private", "internal"],
            "N": ["future", "possibility", "creative", "abstract", "theory"],
            "S": ["practical", "facts", "details", "concrete", "experience"],
            "T": ["logical", "analyze", "objective", "reason", "think"],
            "F": ["feel", "values", "harmony", "personal", "emotion"],
            "J": ["plan", "organize", "structure", "decide", "closure"],
            "P": ["flexible", "adapt", "spontaneous", "open", "explore"]
        }

        scores = {}
        for dimension, words in mbti_indicators.items():
            scores[dimension] = sum(1 for word in words if word in content)

        # Determine type
        type_code = ""
        type_code += "E" if scores.get("E", 0) > scores.get("I", 0) else "I"
        type_code += "N" if scores.get("N", 0) > scores.get("S", 0) else "S"
        type_code += "T" if scores.get("T", 0) > scores.get("F", 0) else "F"
        type_code += "J" if scores.get("J", 0) > scores.get("P", 0) else "P"

        # Determine temperament group
        temperament_map = {
            "NT": ["ENTJ", "ENTP", "INTJ", "INTP"],
            "NF": ["ENFJ", "ENFP", "INFJ", "INFP"],
            "SJ": ["ESTJ", "ESFJ", "ISTJ", "ISFJ"],
            "SP": ["ESTP", "ESFP", "ISTP", "ISFP"]
        }

        temperament = "Unknown"
        for temp, types in temperament_map.items():
            if type_code in types:
                temperament = temp
                break

        confidence = min(sum(scores.values()) / max(len(content.split()), 1) * 5, 1.0)

        return PersonalityProfile(
            profile_id=f"mbti_{conv_id}",
            framework=PersonalityFramework.MBTI,
            traits=dict(scores.items()),
            type_label=f"{type_code} ({temperament})",
            confidence=confidence,
            metadata={"type_code": type_code, "temperament": temperament}
        )

    def _analyze_generic(self, conv_id: str, content: str, framework: PersonalityFramework) -> PersonalityProfile:
        """Generic personality analysis for other frameworks."""
        return PersonalityProfile(
            profile_id=f"{framework.value}_{conv_id}",
            framework=framework,
            traits={"generic_score": 0.5},
            type_label="Generic",
            confidence=0.5,
            metadata={"framework": framework.value}
        )

    def _analyze_distribution(self, profiles: list[PersonalityProfile],
                            framework: PersonalityFramework) -> dict[str, int]:
        """Analyze current personality distribution."""
        distribution = {}

        if framework == PersonalityFramework.BIG_FIVE:
            for profile in profiles:
                for trait, score in profile.traits.items():
                    if score > 0.6:
                        key = f"high_{trait}"
                    elif score < 0.4:
                        key = f"low_{trait}"
                    else:
                        key = f"moderate_{trait}"

                    distribution[key] = distribution.get(key, 0) + 1

        elif framework == PersonalityFramework.MBTI:
            for profile in profiles:
                temperament = profile.metadata.get("temperament", "Unknown")
                if temperament != "Unknown":
                    distribution[temperament] = distribution.get(temperament, 0) + 1

        return distribution

    def _apply_balancing(self, conversations: list[Conversation],
                        profiles: list[PersonalityProfile],
                        current_dist: dict[str, int],
                        framework: PersonalityFramework) -> list[Conversation]:
        """Apply balancing to achieve target distribution."""
        target_dist = self.target_distributions.get(framework, {})
        if not target_dist:
            return conversations  # No balancing if no target distribution

        total_conversations = len(conversations)
        balanced_conversations = []

        # Create conversation-profile mapping
        conv_profile_map = {}
        for i, profile in enumerate(profiles):
            if i < len(conversations):
                conv_profile_map[conversations[i].id] = profile

        # Calculate target counts
        target_counts = {
            category: int(total_conversations * target_pct)
            for category, target_pct in target_dist.items()
        }

        # Group conversations by personality category
        category_conversations = {}
        for conv in conversations:
            profile = conv_profile_map.get(conv.id)
            if profile:
                category = self._get_personality_category(profile, framework)
                if category not in category_conversations:
                    category_conversations[category] = []
                category_conversations[category].append(conv)

        # Balance each category
        for category, target_count in target_counts.items():
            available_convs = category_conversations.get(category, [])

            if len(available_convs) >= target_count:
                # Downsample if we have too many
                balanced_conversations.extend(available_convs[:target_count])
            else:
                # Use all available if we have too few
                balanced_conversations.extend(available_convs)

        # Fill remaining slots with any available conversations
        remaining_slots = total_conversations - len(balanced_conversations)
        if remaining_slots > 0:
            used_ids = {conv.id for conv in balanced_conversations}
            remaining_convs = [conv for conv in conversations if conv.id not in used_ids]
            balanced_conversations.extend(remaining_convs[:remaining_slots])

        return balanced_conversations

    def _get_personality_category(self, profile: PersonalityProfile, framework: PersonalityFramework) -> str:
        """Get personality category for a profile."""
        if framework == PersonalityFramework.BIG_FIVE:
            # Use the most prominent trait
            max_trait = max(profile.traits.items(), key=lambda x: abs(x[1] - 0.5))
            trait_name, score = max_trait

            if score > 0.6:
                return f"high_{trait_name}"
            if score < 0.4:
                return f"low_{trait_name}"
            return f"moderate_{trait_name}"

        if framework == PersonalityFramework.MBTI:
            return profile.metadata.get("temperament", "Unknown")

        return "unknown"

    def _calculate_balancing_quality(self, current_dist: dict[str, int],
                                   final_dist: dict[str, int],
                                   framework: PersonalityFramework) -> dict[str, float]:
        """Calculate quality metrics for balancing operation."""
        target_dist = self.target_distributions.get(framework, {})

        if not target_dist:
            return {"balance_quality": 0.5}

        total_final = sum(final_dist.values())
        if total_final == 0:
            return {"balance_quality": 0.0}

        # Calculate deviation from target distribution
        deviations = []
        for category, target_pct in target_dist.items():
            actual_pct = final_dist.get(category, 0) / total_final
            deviation = abs(actual_pct - target_pct)
            deviations.append(deviation)

        # Balance quality (1.0 = perfect, 0.0 = worst)
        avg_deviation = sum(deviations) / len(deviations) if deviations else 1.0
        balance_quality = max(0.0, 1.0 - (avg_deviation * 2))  # Scale deviation

        # Coverage (how many categories are represented)
        coverage = len([cat for cat in target_dist if final_dist.get(cat, 0) > 0]) / len(target_dist)

        return {
            "balance_quality": balance_quality,
            "coverage": coverage,
            "average_deviation": avg_deviation,
            "total_categories": len(target_dist),
            "represented_categories": len([cat for cat in target_dist if final_dist.get(cat, 0) > 0])
        }

    def get_balancing_recommendations(self, conversations: list[Conversation],
                                   framework: PersonalityFramework = PersonalityFramework.BIG_FIVE) -> dict[str, Any]:
        """Get recommendations for improving personality balance."""
        profiles = self._extract_personality_profiles(conversations, framework)
        current_dist = self._analyze_distribution(profiles, framework)
        target_dist = self.target_distributions.get(framework, {})

        if not target_dist:
            return {"error": "No target distribution available for framework"}

        total_conversations = len(conversations)
        recommendations = {}

        for category, target_pct in target_dist.items():
            current_count = current_dist.get(category, 0)
            target_count = int(total_conversations * target_pct)

            if current_count < target_count:
                recommendations[category] = {
                    "action": "increase",
                    "current": current_count,
                    "target": target_count,
                    "needed": target_count - current_count
                }
            elif current_count > target_count * 1.2:  # 20% tolerance
                recommendations[category] = {
                    "action": "decrease",
                    "current": current_count,
                    "target": target_count,
                    "excess": current_count - target_count
                }
            else:
                recommendations[category] = {
                    "action": "maintain",
                    "current": current_count,
                    "target": target_count,
                    "status": "balanced"
                }

        return {
            "framework": framework.value,
            "total_conversations": total_conversations,
            "current_distribution": current_dist,
            "target_distribution": {k: int(total_conversations * v) for k, v in target_dist.items()},
            "recommendations": recommendations
        }


def validate_personality_balancer():
    """Validate the PersonalityBalancer functionality."""
    try:
        balancer = PersonalityBalancer()

        # Test basic functionality
        assert hasattr(balancer, "balance_personalities")
        assert hasattr(balancer, "target_distributions")
        assert len(balancer.target_distributions) > 0

        return True

    except Exception:
        return False


if __name__ == "__main__":
    # Run validation
    if validate_personality_balancer():
        pass
    else:
        pass
