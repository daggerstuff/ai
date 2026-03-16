#!/usr/bin/env python3
"""
Intelligent Data Fusion Engine for Task 6.2
Fuses multiple conversation datasets with intelligent deduplication and quality optimization.
"""

import hashlib
import logging
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Any

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataSource(Enum):
    """Types of data sources."""

    PRIORITY = "priority"
    COT = "cot"
    REDDIT = "reddit"
    SYNTHETIC = "synthetic"
    EXTERNAL = "external"


class FusionStrategy(Enum):
    """Data fusion strategies."""

    QUALITY_WEIGHTED = "quality_weighted"
    SOURCE_PRIORITY = "source_priority"
    DIVERSITY_MAXIMIZING = "diversity_maximizing"
    BALANCED = "balanced"


@dataclass
class ConversationMetadata:
    """Metadata for conversation in fusion process."""

    conversation_id: str
    source: DataSource
    quality_score: float
    length: int
    topic: str
    condition: str | None = None
    therapeutic_approach: str | None = None
    complexity_level: str = "medium"
    uniqueness_score: float = 0.0
    fusion_priority: int = 1


@dataclass
class FusionResult:
    """Result of data fusion process."""

    fused_conversations: list[dict[str, Any]]
    source_distribution: dict[str, int]
    quality_distribution: dict[str, int]
    deduplication_stats: dict[str, int]
    fusion_metadata: dict[str, Any]
    total_conversations: int
    fusion_quality_score: float


class DataFusionEngine:
    """
    Intelligent data fusion engine for conversation datasets.
    """

    def __init__(self):
        """Initialize the data fusion engine."""
        self.conversations: dict[str, dict[str, Any]] = {}
        self.metadata: dict[str, ConversationMetadata] = {}
        self.similarity_threshold = 0.85
        self.quality_threshold = 0.6
        self.fusion_strategies = {
            FusionStrategy.QUALITY_WEIGHTED: self._quality_weighted_fusion,
            FusionStrategy.SOURCE_PRIORITY: self._source_priority_fusion,
            FusionStrategy.DIVERSITY_MAXIMIZING: self._diversity_maximizing_fusion,
            FusionStrategy.BALANCED: self._balanced_fusion,
        }

        logger.info("DataFusionEngine initialized")

    def add_dataset(
        self,
        conversations: list[dict[str, Any]],
        source: DataSource,
        source_metadata: dict[str, Any] | None = None,
    ) -> int:
        """Add a dataset to the fusion engine."""
        added_count = 0

        for conv in conversations:
            conv_id = self._generate_conversation_id(conv, source)

            # Skip if already exists
            if conv_id in self.conversations:
                continue

            # Calculate metadata
            metadata = self._extract_metadata(conv, source, source_metadata)

            self.conversations[conv_id] = conv
            self.metadata[conv_id] = metadata
            added_count += 1

        logger.info(f"Added {added_count} conversations from {source.value} source")
        return added_count

    def fuse_datasets(
        self,
        strategy: FusionStrategy = FusionStrategy.BALANCED,
        target_size: int | None = None,
        quality_threshold: float | None = None,
    ) -> FusionResult:
        """Fuse datasets using specified strategy."""
        logger.info(f"Starting data fusion with {strategy.value} strategy")

        if quality_threshold is not None:
            self.quality_threshold = quality_threshold

        # Step 1: Deduplicate conversations
        dedup_stats = self._deduplicate_conversations()

        # Step 2: Calculate uniqueness scores
        self._calculate_uniqueness_scores()

        # Step 3: Apply fusion strategy
        fusion_func = self.fusion_strategies[strategy]
        selected_conversations = fusion_func(target_size)

        # Step 4: Generate fusion result
        result = self._generate_fusion_result(selected_conversations, dedup_stats)

        logger.info(
            f"Fusion completed: {result.total_conversations} conversations selected"
        )
        return result

    def _generate_conversation_id(
        self, conversation: dict[str, Any], source: DataSource
    ) -> str:
        """Generate unique conversation ID."""
        # Extract content from messages or content field
        content = ""
        if "messages" in conversation:
            # Extract text from messages
            messages = conversation["messages"]
            if isinstance(messages, list):
                content = " ".join(
                    str(msg.get("content", ""))
                    for msg in messages
                    if isinstance(msg, dict)
                )
            else:
                content = str(messages)
        elif "content" in conversation:
            content = conversation["content"]
            if isinstance(content, list):
                content = " ".join(str(turn) for turn in content)

        # Create hash from content and source
        hash_input = f"{source.value}_{content[:500]}"
        return hashlib.md5(hash_input.encode()).hexdigest()[:16]

    def _extract_metadata(
        self,
        conversation: dict[str, Any],
        source: DataSource,
        source_metadata: dict[str, Any] | None = None,
    ) -> ConversationMetadata:
        """Extract metadata from conversation."""
        # Extract content from messages or content field
        content = ""
        if "messages" in conversation:
            messages = conversation["messages"]
            if isinstance(messages, list):
                content = " ".join(
                    str(msg.get("content", ""))
                    for msg in messages
                    if isinstance(msg, dict)
                )
            else:
                content = str(messages)
        elif "content" in conversation:
            content = conversation["content"]
            if isinstance(content, list):
                content = " ".join(str(turn) for turn in content)

        # Calculate basic metrics
        length = len(content.split()) if content else 0
        quality_score = self._estimate_quality_score(conversation, source)
        topic = self._extract_topic(conversation)
        condition = self._extract_condition(conversation)
        therapeutic_approach = self._extract_therapeutic_approach(conversation)
        complexity_level = self._assess_complexity(conversation)

        # Determine fusion priority based on source
        priority_map = {
            DataSource.PRIORITY: 1,
            DataSource.COT: 2,
            DataSource.SYNTHETIC: 3,
            DataSource.REDDIT: 4,
            DataSource.EXTERNAL: 5,
        }

        return ConversationMetadata(
            conversation_id=self._generate_conversation_id(conversation, source),
            source=source,
            quality_score=quality_score,
            length=length,
            topic=topic,
            condition=condition,
            therapeutic_approach=therapeutic_approach,
            complexity_level=complexity_level,
            fusion_priority=priority_map.get(source, 5),
        )

    def _estimate_quality_score(
        self, conversation: dict[str, Any], source: DataSource
    ) -> float:
        """Estimate quality score for conversation."""
        content = conversation.get("content", "")
        if isinstance(content, list):
            content = " ".join(str(turn) for turn in content)

        score = 0.5  # Base score

        # Source-based scoring
        source_scores = {
            DataSource.PRIORITY: 0.9,
            DataSource.COT: 0.8,
            DataSource.SYNTHETIC: 0.7,
            DataSource.REDDIT: 0.6,
            DataSource.EXTERNAL: 0.5,
        }
        score = source_scores.get(source, 0.5)

        # Content-based adjustments
        if content:
            word_count = len(content.split())

            # Length bonus/penalty
            if 50 <= word_count <= 500:
                score += 0.1
            elif word_count < 20 or word_count > 1000:
                score -= 0.1

            # Therapeutic content indicators
            therapeutic_terms = [
                "feel",
                "emotion",
                "therapy",
                "counseling",
                "support",
                "help",
                "understand",
                "cope",
                "stress",
                "anxiety",
                "depression",
            ]

            term_count = sum(1 for term in therapeutic_terms if term in content.lower())
            score += min(0.2, term_count * 0.02)

            # Quality indicators
            if any(
                indicator in content.lower()
                for indicator in [
                    "how are you feeling",
                    "tell me more",
                    "that sounds difficult",
                ]
            ):
                score += 0.1

        return min(1.0, max(0.0, score))

    def _extract_topic(self, conversation: dict[str, Any]) -> str:
        """Extract topic from conversation."""
        content = conversation.get("content", "")
        if isinstance(content, list):
            content = " ".join(str(turn) for turn in content)

        content_lower = content.lower()

        # Topic keywords
        topics = {
            "anxiety": ["anxiety", "anxious", "worry", "panic", "fear"],
            "depression": ["depression", "depressed", "sad", "hopeless", "down"],
            "relationships": [
                "relationship",
                "partner",
                "marriage",
                "family",
                "friend",
            ],
            "work": ["work", "job", "career", "boss", "workplace"],
            "trauma": ["trauma", "abuse", "ptsd", "flashback", "trigger"],
            "addiction": ["addiction", "substance", "alcohol", "drugs", "recovery"],
            "general": [],
        }

        for topic, keywords in topics.items():
            if any(keyword in content_lower for keyword in keywords):
                return topic

        return "general"

    def _extract_condition(self, conversation: dict[str, Any]) -> str | None:
        """Extract mental health condition from conversation."""
        content = conversation.get("content", "")
        if isinstance(content, list):
            content = " ".join(str(turn) for turn in content)

        content_lower = content.lower()

        conditions = {
            "anxiety_disorder": [
                "anxiety disorder",
                "generalized anxiety",
                "social anxiety",
            ],
            "depression": [
                "major depression",
                "clinical depression",
                "depressive disorder",
            ],
            "bipolar": ["bipolar", "manic", "mania"],
            "ptsd": ["ptsd", "post-traumatic stress"],
            "ocd": ["ocd", "obsessive compulsive"],
            "adhd": ["adhd", "attention deficit"],
        }

        for condition, keywords in conditions.items():
            if any(keyword in content_lower for keyword in keywords):
                return condition

        return None

    def _extract_therapeutic_approach(
        self, conversation: dict[str, Any]
    ) -> str | None:
        """Extract therapeutic approach from conversation."""
        content = conversation.get("content", "")
        if isinstance(content, list):
            content = " ".join(str(turn) for turn in content)

        content_lower = content.lower()

        approaches = {
            "cbt": [
                "cognitive behavioral",
                "cbt",
                "thought challenging",
                "behavioral activation",
            ],
            "dbt": ["dbt", "dialectical", "mindfulness", "distress tolerance"],
            "psychodynamic": ["psychodynamic", "unconscious", "childhood", "insight"],
            "humanistic": [
                "humanistic",
                "person-centered",
                "unconditional positive regard",
            ],
            "solution_focused": ["solution focused", "goals", "strengths", "resources"],
        }

        for approach, keywords in approaches.items():
            if any(keyword in content_lower for keyword in keywords):
                return approach

        return None

    def _assess_complexity(self, conversation: dict[str, Any]) -> str:
        """Assess conversation complexity."""
        content = conversation.get("content", "")
        if isinstance(content, list):
            content = " ".join(str(turn) for turn in content)

        if not content:
            return "low"

        word_count = len(content.split())
        sentence_count = len([s for s in content.split(".") if s.strip()])

        # Complexity indicators
        complex_terms = [
            "however",
            "nevertheless",
            "furthermore",
            "consequently",
            "psychological",
            "therapeutic",
            "intervention",
            "assessment",
        ]

        complex_term_count = sum(1 for term in complex_terms if term in content.lower())

        # Calculate complexity score
        complexity_score = 0

        if word_count > 200:
            complexity_score += 1
        if sentence_count > 10:
            complexity_score += 1
        if complex_term_count > 2:
            complexity_score += 1

        if complexity_score >= 2:
            return "high"
        if complexity_score == 1:
            return "medium"
        return "low"

    def _deduplicate_conversations(self) -> dict[str, int]:
        """Remove duplicate conversations."""
        logger.info("Starting deduplication process")

        original_count = len(self.conversations)
        duplicates_removed = 0
        near_duplicates_removed = 0

        # Create content hashes for exact duplicate detection
        content_hashes: dict[str, str] = {}
        exact_duplicates: set[str] = set()

        for conv_id, conv in self.conversations.items():
            content = conv.get("content", "")
            if isinstance(content, list):
                content = " ".join(str(turn) for turn in content)

            content_hash = hashlib.md5(content.encode()).hexdigest()

            if content_hash in content_hashes:
                # Exact duplicate found - keep higher quality one
                existing_id = content_hashes[content_hash]
                existing_quality = self.metadata[existing_id].quality_score
                current_quality = self.metadata[conv_id].quality_score

                if current_quality > existing_quality:
                    exact_duplicates.add(existing_id)
                    content_hashes[content_hash] = conv_id
                else:
                    exact_duplicates.add(conv_id)
            else:
                content_hashes[content_hash] = conv_id

        # Remove exact duplicates
        for dup_id in exact_duplicates:
            del self.conversations[dup_id]
            del self.metadata[dup_id]
            duplicates_removed += 1

        # Near-duplicate detection (simplified)
        remaining_ids = list(self.conversations.keys())
        near_duplicates: set[str] = set()

        for i, id1 in enumerate(remaining_ids):
            if id1 in near_duplicates:
                continue

            content1 = self.conversations[id1].get("content", "")
            if isinstance(content1, list):
                content1 = " ".join(str(turn) for turn in content1)

            for id2 in remaining_ids[i + 1 :]:
                if id2 in near_duplicates:
                    continue

                content2 = self.conversations[id2].get("content", "")
                if isinstance(content2, list):
                    content2 = " ".join(str(turn) for turn in content2)

                similarity = self._calculate_similarity(content1, content2)

                if similarity > self.similarity_threshold:
                    # Near duplicate found - keep higher quality one
                    quality1 = self.metadata[id1].quality_score
                    quality2 = self.metadata[id2].quality_score

                    if quality2 > quality1:
                        near_duplicates.add(id1)
                        break
                    near_duplicates.add(id2)

        # Remove near duplicates
        for dup_id in near_duplicates:
            if dup_id in self.conversations:
                del self.conversations[dup_id]
                del self.metadata[dup_id]
                near_duplicates_removed += 1

        final_count = len(self.conversations)

        logger.info(
            f"Deduplication completed: {original_count} -> {final_count} conversations"
        )

        return {
            "original_count": original_count,
            "final_count": final_count,
            "exact_duplicates_removed": duplicates_removed,
            "near_duplicates_removed": near_duplicates_removed,
            "total_removed": duplicates_removed + near_duplicates_removed,
        }

    def _calculate_similarity(self, content1: str, content2: str) -> float:
        """Calculate similarity between two pieces of content."""
        if not content1 or not content2:
            return 0.0

        # Simple word-based similarity
        words1 = set(content1.lower().split())
        words2 = set(content2.lower().split())

        if not words1 or not words2:
            return 0.0

        intersection = len(words1.intersection(words2))
        union = len(words1.union(words2))

        return intersection / union if union > 0 else 0.0

    def _calculate_uniqueness_scores(self):
        """Calculate uniqueness scores for all conversations."""
        logger.info("Calculating uniqueness scores")

        conv_ids = list(self.conversations.keys())

        for _i, conv_id in enumerate(conv_ids):
            content = self.conversations[conv_id].get("content", "")
            if isinstance(content, list):
                content = " ".join(str(turn) for turn in content)

            similarities = []

            # Compare with a sample of other conversations
            sample_size = min(100, len(conv_ids) - 1)
            other_ids = [id for id in conv_ids if id != conv_id]

            if len(other_ids) > sample_size:
                import random

                other_ids = random.sample(other_ids, sample_size)

            for other_id in other_ids:
                other_content = self.conversations[other_id].get("content", "")
                if isinstance(other_content, list):
                    other_content = " ".join(str(turn) for turn in other_content)

                similarity = self._calculate_similarity(content, other_content)
                similarities.append(similarity)

            # Uniqueness is inverse of average similarity
            avg_similarity = (
                sum(similarities) / len(similarities) if similarities else 0
            )
            uniqueness_score = 1.0 - avg_similarity

            self.metadata[conv_id].uniqueness_score = uniqueness_score

    def _quality_weighted_fusion(self, target_size: int | None) -> list[str]:
        """Select conversations based on quality weighting."""
        # Filter by quality threshold
        qualified_ids = [
            conv_id
            for conv_id, metadata in self.metadata.items()
            if metadata.quality_score >= self.quality_threshold
        ]

        # Sort by quality score (descending)
        qualified_ids.sort(key=lambda x: self.metadata[x].quality_score, reverse=True)

        if target_size and len(qualified_ids) > target_size:
            return qualified_ids[:target_size]

        return qualified_ids

    def _source_priority_fusion(self, target_size: int | None) -> list[str]:
        """Select conversations based on source priority."""
        # Sort by fusion priority (ascending - lower is better)
        all_ids = list(self.metadata.keys())
        all_ids.sort(
            key=lambda x: (
                self.metadata[x].fusion_priority,
                -self.metadata[
                    x
                ].quality_score,  # Secondary sort by quality (descending)
            )
        )

        if target_size and len(all_ids) > target_size:
            return all_ids[:target_size]

        return all_ids

    def _diversity_maximizing_fusion(self, target_size: int | None) -> list[str]:
        """Select conversations to maximize diversity."""
        selected_ids = []
        remaining_ids = list(self.metadata.keys())

        # Start with highest quality conversation
        if remaining_ids:
            best_id = max(remaining_ids, key=lambda x: self.metadata[x].quality_score)
            selected_ids.append(best_id)
            remaining_ids.remove(best_id)

        # Iteratively select most diverse conversations
        while remaining_ids and (not target_size or len(selected_ids) < target_size):
            best_candidate = None
            best_diversity_score = -1

            for candidate_id in remaining_ids:
                # Calculate diversity score (combination of uniqueness and quality)
                uniqueness = self.metadata[candidate_id].uniqueness_score
                quality = self.metadata[candidate_id].quality_score
                diversity_score = (uniqueness * 0.7) + (quality * 0.3)

                if diversity_score > best_diversity_score:
                    best_diversity_score = diversity_score
                    best_candidate = candidate_id

            if best_candidate:
                selected_ids.append(best_candidate)
                remaining_ids.remove(best_candidate)
            else:
                break

        return selected_ids

    def _balanced_fusion(self, target_size: int | None) -> list[str]:
        """Select conversations using balanced approach."""
        # Combine quality, source priority, and uniqueness
        scored_ids = []

        for conv_id, metadata in self.metadata.items():
            # Weighted score combining multiple factors
            quality_weight = 0.4
            priority_weight = 0.3  # Lower priority number is better
            uniqueness_weight = 0.3

            priority_score = (
                1.0 - (metadata.fusion_priority - 1) / 4
            )  # Normalize to 0-1

            combined_score = (
                metadata.quality_score * quality_weight
                + priority_score * priority_weight
                + metadata.uniqueness_score * uniqueness_weight
            )

            scored_ids.append((conv_id, combined_score))

        # Sort by combined score (descending)
        scored_ids.sort(key=lambda x: x[1], reverse=True)

        selected_ids = [conv_id for conv_id, score in scored_ids]

        if target_size and len(selected_ids) > target_size:
            return selected_ids[:target_size]

        return selected_ids

    def _generate_fusion_result(
        self, selected_ids: list[str], dedup_stats: dict[str, int]
    ) -> FusionResult:
        """Generate fusion result."""
        # Collect selected conversations
        fused_conversations = [self.conversations[conv_id] for conv_id in selected_ids]

        # Calculate distributions
        source_distribution = {}
        quality_distribution = {"high": 0, "medium": 0, "low": 0}

        for conv_id in selected_ids:
            metadata = self.metadata[conv_id]

            # Source distribution
            source = metadata.source.value
            source_distribution[source] = source_distribution.get(source, 0) + 1

            # Quality distribution
            if metadata.quality_score >= 0.8:
                quality_distribution["high"] += 1
            elif metadata.quality_score >= 0.6:
                quality_distribution["medium"] += 1
            else:
                quality_distribution["low"] += 1

        # Calculate overall fusion quality
        total_quality = sum(
            self.metadata[conv_id].quality_score for conv_id in selected_ids
        )
        fusion_quality_score = (
            total_quality / len(selected_ids) if selected_ids else 0.0
        )

        return FusionResult(
            fused_conversations=fused_conversations,
            source_distribution=source_distribution,
            quality_distribution=quality_distribution,
            deduplication_stats=dedup_stats,
            fusion_metadata={
                "fusion_timestamp": datetime.now().isoformat(),
                "similarity_threshold": self.similarity_threshold,
                "quality_threshold": self.quality_threshold,
                "total_input_conversations": dedup_stats["original_count"],
            },
            total_conversations=len(selected_ids),
            fusion_quality_score=fusion_quality_score,
        )

    def get_fusion_summary(self) -> dict[str, Any]:
        """Get summary of current fusion state."""
        if not self.metadata:
            return {"status": "empty", "conversations": 0}

        source_counts = {}
        quality_scores = []

        for metadata in self.metadata.values():
            source = metadata.source.value
            source_counts[source] = source_counts.get(source, 0) + 1
            quality_scores.append(metadata.quality_score)

        avg_quality = sum(quality_scores) / len(quality_scores)

        return {
            "status": "loaded",
            "total_conversations": len(self.conversations),
            "source_distribution": source_counts,
            "average_quality": avg_quality,
            "quality_threshold": self.quality_threshold,
            "similarity_threshold": self.similarity_threshold,
        }


def main():
    """Test the data fusion engine."""
    engine = DataFusionEngine()

    # Create test datasets
    priority_conversations = [
        {
            "content": "I'm feeling anxious about my upcoming presentation. Can you help me?",
            "quality": 0.9,
        },
        {
            "content": "I've been struggling with depression for months. What should I do?",
            "quality": 0.85,
        },
    ]

    reddit_conversations = [
        {"content": "Anyone else feel anxious about presentations?", "quality": 0.6},
        {"content": "Depression is hard. Looking for advice.", "quality": 0.65},
        {
            "content": "I'm feeling anxious about my upcoming presentation. Can you help me?",
            "quality": 0.6,
        },  # Duplicate
    ]

    cot_conversations = [
        {
            "content": "Let's think through your anxiety step by step. First, what specifically worries you?",
            "quality": 0.8,
        },
        {
            "content": "Depression affects many people. Let's explore your feelings together.",
            "quality": 0.75,
        },
    ]

    # Add datasets
    engine.add_dataset(priority_conversations, DataSource.PRIORITY)
    engine.add_dataset(reddit_conversations, DataSource.REDDIT)
    engine.add_dataset(cot_conversations, DataSource.COT)

    engine.get_fusion_summary()

    # Test different fusion strategies
    strategies = [
        FusionStrategy.BALANCED,
        FusionStrategy.QUALITY_WEIGHTED,
        FusionStrategy.SOURCE_PRIORITY,
        FusionStrategy.DIVERSITY_MAXIMIZING,
    ]

    for strategy in strategies:
        engine.fuse_datasets(strategy=strategy, target_size=4)



if __name__ == "__main__":
    main()
