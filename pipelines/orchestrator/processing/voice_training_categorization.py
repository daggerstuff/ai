"""
Voice Training Categorization

Creates voice data categorization for training ratio allocation (Task 3.13).
Categorizes voice-derived conversations for optimal training data distribution.
"""

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
from logger import get_logger

logger = get_logger(__name__)

@dataclass
class VoiceTrainingCategory:
    """Voice training data category."""
    category_id: str
    category_name: str
    description: str
    target_ratio: float
    current_count: int = 0
    quality_threshold: float = 0.6
    priority_level: int = 1  # 1=high, 2=medium, 3=low

@dataclass
class VoiceCategorization:
    """Voice data categorization result."""
    conversation_id: str
    assigned_categories: list[str]
    primary_category: str
    category_confidence: float
    training_suitability: str  # "excellent", "good", "acceptable", "poor"
    allocation_recommendation: str

class VoiceTrainingCategorization:
    """Creates voice data categorization for training ratio allocation."""

    def __init__(self, voice_data_path: str = "./voice_data", output_dir: str = "./training_categories"):
        self.voice_data_path = Path(voice_data_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)

        self.logger = get_logger(__name__)

        # Training categories with target ratios
        self.training_categories = {
            "therapeutic_dialogue": VoiceTrainingCategory(
                category_id="therapeutic_dialogue",
                category_name="Therapeutic Dialogue",
                description="Client-therapist conversations for therapeutic AI training",
                target_ratio=0.35,  # 35% of training data
                quality_threshold=0.7,
                priority_level=1
            ),
            "counseling_sessions": VoiceTrainingCategory(
                category_id="counseling_sessions",
                category_name="Counseling Sessions",
                description="Professional counseling conversations",
                target_ratio=0.25,  # 25% of training data
                quality_threshold=0.65,
                priority_level=1
            ),
            "support_conversations": VoiceTrainingCategory(
                category_id="support_conversations",
                category_name="Support Conversations",
                description="Peer support and group therapy conversations",
                target_ratio=0.20,  # 20% of training data
                quality_threshold=0.6,
                priority_level=2
            ),
            "educational_content": VoiceTrainingCategory(
                category_id="educational_content",
                category_name="Educational Content",
                description="Mental health education and psychoeducation",
                target_ratio=0.10,  # 10% of training data
                quality_threshold=0.7,
                priority_level=2
            ),
            "crisis_intervention": VoiceTrainingCategory(
                category_id="crisis_intervention",
                category_name="Crisis Intervention",
                description="Crisis support and intervention conversations",
                target_ratio=0.05,  # 5% of training data
                quality_threshold=0.8,
                priority_level=1
            ),
            "general_wellness": VoiceTrainingCategory(
                category_id="general_wellness",
                category_name="General Wellness",
                description="General mental wellness and self-care conversations",
                target_ratio=0.05,  # 5% of training data
                quality_threshold=0.55,
                priority_level=3
            )
        }

        # Category identification patterns
        self.category_patterns = {
            "therapeutic_dialogue": {
                "keywords": ["therapy", "therapist", "therapeutic", "treatment", "session", "clinical"],
                "speaker_roles": ["client", "therapist", "patient", "clinician"],
                "content_indicators": ["feelings", "thoughts", "coping", "symptoms", "diagnosis"]
            },
            "counseling_sessions": {
                "keywords": ["counseling", "counselor", "guidance", "advice", "support", "help"],
                "speaker_roles": ["counselee", "counselor", "client", "advisor"],
                "content_indicators": ["problem", "solution", "guidance", "decision", "choice"]
            },
            "support_conversations": {
                "keywords": ["support", "group", "peer", "sharing", "community", "together"],
                "speaker_roles": ["participant", "member", "facilitator", "peer"],
                "content_indicators": ["experience", "sharing", "understanding", "empathy", "connection"]
            },
            "educational_content": {
                "keywords": ["education", "learning", "information", "explanation", "teaching", "knowledge"],
                "speaker_roles": ["educator", "student", "learner", "instructor"],
                "content_indicators": ["learn", "understand", "explain", "information", "knowledge"]
            },
            "crisis_intervention": {
                "keywords": ["crisis", "emergency", "urgent", "immediate", "help", "danger"],
                "speaker_roles": ["crisis_counselor", "caller", "responder", "operator"],
                "content_indicators": ["crisis", "emergency", "suicide", "harm", "danger", "immediate"]
            },
            "general_wellness": {
                "keywords": ["wellness", "wellbeing", "health", "lifestyle", "self-care", "mindfulness"],
                "speaker_roles": ["coach", "participant", "guide", "individual"],
                "content_indicators": ["wellness", "healthy", "balance", "self-care", "mindfulness"]
            }
        }

        logger.info("VoiceTrainingCategorization initialized")

    def categorize_voice_conversation(self, voice_conversation_data: dict[str, Any]) -> VoiceCategorization:
        """Categorize a voice conversation for training allocation."""

        try:
            conversation_id = voice_conversation_data.get("conversation_id", f"voice_cat_{datetime.now().strftime('%Y%m%d_%H%M%S')}")

            # Extract features for categorization
            conversation_features = self._extract_categorization_features(voice_conversation_data)

            # Score against each category
            category_scores = self._score_categories(conversation_features)

            # Determine primary category and confidence
            primary_category, category_confidence = self._determine_primary_category(category_scores)

            # Get all applicable categories (above threshold)
            assigned_categories = self._get_assigned_categories(category_scores, threshold=0.3)

            # Assess training suitability
            training_suitability = self._assess_training_suitability(conversation_features, primary_category)

            # Generate allocation recommendation
            allocation_recommendation = self._generate_allocation_recommendation(
                primary_category, category_confidence, training_suitability
            )

            categorization = VoiceCategorization(
                conversation_id=conversation_id,
                assigned_categories=assigned_categories,
                primary_category=primary_category,
                category_confidence=category_confidence,
                training_suitability=training_suitability,
                allocation_recommendation=allocation_recommendation
            )

            logger.info(f"Voice conversation categorized: {conversation_id} -> {primary_category} ({category_confidence:.2f})")
            return categorization

        except Exception as e:
            logger.error(f"Voice conversation categorization failed: {e}")
            return VoiceCategorization(
                conversation_id=f"failed_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                assigned_categories=[],
                primary_category="unknown",
                category_confidence=0.0,
                training_suitability="poor",
                allocation_recommendation="exclude_from_training"
            )

    def batch_categorize_conversations(self, voice_conversations: list[dict[str, Any]]) -> dict[str, Any]:
        """Categorize multiple voice conversations and optimize training allocation."""
        start_time = datetime.now()

        categorizations = []
        category_counts = dict.fromkeys(self.training_categories.keys(), 0)

        # Categorize each conversation
        for conversation_data in voice_conversations:
            categorization = self.categorize_voice_conversation(conversation_data)
            categorizations.append(categorization)

            # Update category counts
            if categorization.primary_category in category_counts:
                category_counts[categorization.primary_category] += 1

        # Calculate current ratios
        total_conversations = len(categorizations)
        current_ratios = {cat_id: count / total_conversations for cat_id, count in category_counts.items()}

        # Generate training allocation plan
        allocation_plan = self._generate_training_allocation_plan(categorizations, current_ratios)

        # Calculate batch statistics
        batch_stats = self._calculate_categorization_statistics(categorizations)

        # Save batch results
        output_path = self._save_batch_categorization_results(categorizations, allocation_plan, batch_stats)

        return {
            "success": True,
            "conversations_categorized": len(voice_conversations),
            "category_distribution": category_counts,
            "current_ratios": current_ratios,
            "target_ratios": {cat_id: cat.target_ratio for cat_id, cat in self.training_categories.items()},
            "allocation_plan": allocation_plan,
            "batch_statistics": batch_stats,
            "categorizations": categorizations,
            "output_path": str(output_path),
            "processing_time": (datetime.now() - start_time).total_seconds()
        }

    def _extract_categorization_features(self, voice_conversation_data: dict[str, Any]) -> dict[str, Any]:
        """Extract features for categorization from voice conversation data."""

        # Extract text content
        transcription = voice_conversation_data.get("transcription", "")
        messages = voice_conversation_data.get("messages", [])

        # Combine all text content
        all_text = transcription
        for msg in messages:
            if isinstance(msg, dict) and "content" in msg:
                all_text += " " + msg["content"]

        all_text = all_text.lower()

        # Extract speaker information
        speakers = []
        for msg in messages:
            if isinstance(msg, dict) and "role" in msg:
                speakers.append(msg["role"])

        unique_speakers = list(set(speakers))

        # Extract conversation metadata
        metadata = voice_conversation_data.get("metadata", {})

        return {
            "text_content": all_text,
            "word_count": len(all_text.split()),
            "speakers": unique_speakers,
            "speaker_count": len(unique_speakers),
            "message_count": len(messages),
            "conversation_length": voice_conversation_data.get("duration", 0),
            "metadata": metadata,
            "quality_score": voice_conversation_data.get("quality_score", 0.5)
        }

    def _score_categories(self, conversation_features: dict[str, Any]) -> dict[str, float]:
        """Score conversation against each training category."""

        text_content = conversation_features["text_content"]
        speakers = conversation_features["speakers"]
        metadata = conversation_features["metadata"]

        category_scores = {}

        for category_id, patterns in self.category_patterns.items():
            score = 0.0

            # Keyword matching (40% of score)
            keyword_matches = sum(1 for keyword in patterns["keywords"] if keyword in text_content)
            keyword_score = min(1.0, keyword_matches / len(patterns["keywords"]))
            score += keyword_score * 0.4

            # Speaker role matching (30% of score)
            speaker_matches = sum(1 for speaker in speakers if any(role in speaker for role in patterns["speaker_roles"]))
            speaker_score = min(1.0, speaker_matches / max(1, len(speakers)))
            score += speaker_score * 0.3

            # Content indicator matching (20% of score)
            content_matches = sum(1 for indicator in patterns["content_indicators"] if indicator in text_content)
            content_score = min(1.0, content_matches / len(patterns["content_indicators"]))
            score += content_score * 0.2

            # Metadata alignment (10% of score)
            metadata_score = 0.5  # Default neutral score
            if "session_type" in metadata:
                session_type = metadata["session_type"].lower()
                if category_id in session_type or any(keyword in session_type for keyword in patterns["keywords"]):
                    metadata_score = 1.0
            score += metadata_score * 0.1

            category_scores[category_id] = score

        return category_scores

    def _determine_primary_category(self, category_scores: dict[str, float]) -> tuple[str, float]:
        """Determine primary category and confidence level."""

        if not category_scores:
            return "unknown", 0.0

        # Find highest scoring category
        primary_category = max(category_scores.items(), key=lambda x: x[1])
        category_name, confidence = primary_category

        # Adjust confidence based on score separation
        sorted_scores = sorted(category_scores.values(), reverse=True)
        if len(sorted_scores) > 1:
            score_separation = sorted_scores[0] - sorted_scores[1]
            confidence = confidence * (0.5 + score_separation * 0.5)  # Boost confidence with clear separation

        return category_name, confidence

    def _get_assigned_categories(self, category_scores: dict[str, float], threshold: float = 0.3) -> list[str]:
        """Get all categories above the threshold."""
        return [category for category, score in category_scores.items() if score >= threshold]

    def _assess_training_suitability(self, conversation_features: dict[str, Any], primary_category: str) -> str:
        """Assess suitability for training based on quality and category requirements."""

        quality_score = conversation_features["quality_score"]
        word_count = conversation_features["word_count"]
        speaker_count = conversation_features["speaker_count"]

        # Get category requirements
        category_info = self.training_categories.get(primary_category)
        if not category_info:
            return "poor"

        quality_threshold = category_info.quality_threshold

        # Base assessment on quality
        if quality_score >= quality_threshold + 0.2:
            base_suitability = "excellent"
        elif quality_score >= quality_threshold:
            base_suitability = "good"
        elif quality_score >= quality_threshold - 0.1:
            base_suitability = "acceptable"
        else:
            base_suitability = "poor"

        # Adjust based on conversation characteristics
        if word_count < 50:  # Too short
            base_suitability = "poor"
        elif speaker_count < 2:  # Not enough interaction
            if base_suitability in ["excellent", "good"]:
                base_suitability = "acceptable"

        return base_suitability

    def _generate_allocation_recommendation(self, primary_category: str,
                                         category_confidence: float,
                                         training_suitability: str) -> str:
        """Generate training allocation recommendation."""

        if training_suitability == "poor":
            return "exclude_from_training"

        if category_confidence < 0.4:
            return "manual_review_required"

        category_info = self.training_categories.get(primary_category)
        if not category_info:
            return "exclude_from_training"

        # Check priority and suitability
        if category_info.priority_level == 1 and training_suitability in ["excellent", "good"]:
            return "high_priority_training"
        if category_info.priority_level <= 2 and training_suitability in ["excellent", "good", "acceptable"]:
            return "standard_training"
        if training_suitability == "acceptable":
            return "supplementary_training"
        return "exclude_from_training"

    def _generate_training_allocation_plan(self, categorizations: list[VoiceCategorization],
                                         current_ratios: dict[str, float]) -> dict[str, Any]:
        """Generate optimized training allocation plan."""

        # Calculate ratio deviations
        ratio_deviations = {}
        for category_id, target_ratio in [(cat_id, cat.target_ratio) for cat_id, cat in self.training_categories.items()]:
            current_ratio = current_ratios.get(category_id, 0.0)
            deviation = target_ratio - current_ratio
            ratio_deviations[category_id] = deviation

        # Identify categories that need more data
        underrepresented = {cat_id: dev for cat_id, dev in ratio_deviations.items() if dev > 0.05}
        overrepresented = {cat_id: abs(dev) for cat_id, dev in ratio_deviations.items() if dev < -0.05}

        # Generate recommendations
        recommendations = []

        if underrepresented:
            for category_id, deficit in underrepresented.items():
                category_name = self.training_categories[category_id].category_name
                recommendations.append(f"Increase {category_name} data by {deficit:.1%}")

        if overrepresented:
            for category_id, excess in overrepresented.items():
                category_name = self.training_categories[category_id].category_name
                recommendations.append(f"Reduce {category_name} data by {excess:.1%}")

        # Calculate training set composition
        training_suitable = [c for c in categorizations if c.training_suitability in ["excellent", "good", "acceptable"]]
        training_composition = {}

        for category_id in self.training_categories:
            category_conversations = [c for c in training_suitable if c.primary_category == category_id]
            training_composition[category_id] = {
                "count": len(category_conversations),
                "excellent": len([c for c in category_conversations if c.training_suitability == "excellent"]),
                "good": len([c for c in category_conversations if c.training_suitability == "good"]),
                "acceptable": len([c for c in category_conversations if c.training_suitability == "acceptable"])
            }

        return {
            "current_ratios": current_ratios,
            "target_ratios": {cat_id: cat.target_ratio for cat_id, cat in self.training_categories.items()},
            "ratio_deviations": ratio_deviations,
            "underrepresented_categories": underrepresented,
            "overrepresented_categories": overrepresented,
            "recommendations": recommendations,
            "training_composition": training_composition,
            "total_training_suitable": len(training_suitable)
        }

    def _calculate_categorization_statistics(self, categorizations: list[VoiceCategorization]) -> dict[str, Any]:
        """Calculate statistics for categorization batch."""

        if not categorizations:
            return {}

        # Confidence statistics
        confidences = [c.category_confidence for c in categorizations]

        # Training suitability distribution
        suitability_counts = {}
        for suitability in ["excellent", "good", "acceptable", "poor"]:
            suitability_counts[suitability] = sum(1 for c in categorizations if c.training_suitability == suitability)

        # Category assignment statistics
        category_assignments = {}
        for category_id in self.training_categories:
            category_assignments[category_id] = sum(1 for c in categorizations if c.primary_category == category_id)

        return {
            "total_conversations": len(categorizations),
            "average_confidence": sum(confidences) / len(confidences),
            "confidence_std": np.std(confidences),
            "min_confidence": min(confidences),
            "max_confidence": max(confidences),
            "suitability_distribution": suitability_counts,
            "category_distribution": category_assignments,
            "high_confidence_count": sum(1 for c in confidences if c >= 0.7),
            "low_confidence_count": sum(1 for c in confidences if c < 0.4)
        }

    def _save_batch_categorization_results(self, categorizations: list[VoiceCategorization],
                                         allocation_plan: dict[str, Any],
                                         batch_stats: dict[str, Any]) -> Path:
        """Save batch categorization results."""
        output_file = self.output_dir / f"voice_training_categorization_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

        # Convert categorizations to serializable format
        categorizations_data = []
        for categorization in categorizations:
            categorization_dict = {
                "conversation_id": categorization.conversation_id,
                "assigned_categories": categorization.assigned_categories,
                "primary_category": categorization.primary_category,
                "category_confidence": categorization.category_confidence,
                "training_suitability": categorization.training_suitability,
                "allocation_recommendation": categorization.allocation_recommendation
            }
            categorizations_data.append(categorization_dict)

        output_data = {
            "batch_info": {
                "categorization_type": "voice_training_categorization",
                "processed_at": datetime.now().isoformat(),
                "categorizer_version": "1.0"
            },
            "training_categories": {
                cat_id: {
                    "category_name": cat.category_name,
                    "description": cat.description,
                    "target_ratio": cat.target_ratio,
                    "quality_threshold": cat.quality_threshold,
                    "priority_level": cat.priority_level
                }
                for cat_id, cat in self.training_categories.items()
            },
            "allocation_plan": allocation_plan,
            "batch_statistics": batch_stats,
            "categorizations": categorizations_data
        }

        with open(output_file, "w") as f:
            json.dump(output_data, f, indent=2)

        logger.info(f"Batch categorization results saved: {output_file}")
        return output_file


# Example usage
if __name__ == "__main__":
    # Initialize categorizer
    categorizer = VoiceTrainingCategorization()

    # Mock voice conversation data
    mock_conversations = [
        {
            "conversation_id": "voice_001",
            "transcription": "I've been feeling anxious lately. Can you help me understand what's happening?",
            "messages": [
                {"role": "client", "content": "I've been feeling anxious lately."},
                {"role": "therapist", "content": "Can you tell me more about when you notice these feelings?"}
            ],
            "quality_score": 0.85,
            "duration": 180
        },
        {
            "conversation_id": "voice_002",
            "transcription": "Welcome to our support group. Let's share our experiences today.",
            "messages": [
                {"role": "facilitator", "content": "Welcome to our support group."},
                {"role": "participant", "content": "I'd like to share my experience with anxiety."}
            ],
            "quality_score": 0.75,
            "duration": 240
        }
    ]

    # Batch categorization
    result = categorizer.batch_categorize_conversations(mock_conversations)


    # Individual categorization example
    individual_result = categorizer.categorize_voice_conversation(mock_conversations[0])
