"""
Data deduplication and similarity detection system for conversation datasets.

This module provides comprehensive deduplication capabilities using multiple
similarity metrics including content similarity, semantic similarity, and
structural similarity to identify and remove duplicate conversations.
"""

import difflib
import hashlib
import logging
import re
from collections import defaultdict
from dataclasses import dataclass
from typing import Any

from conversation_schema import Conversation

# Set up logging
logger = logging.getLogger(__name__)


@dataclass
class SimilarityMetrics:
    """Container for similarity assessment metrics."""
    content_similarity: float
    semantic_similarity: float
    structural_similarity: float
    overall_similarity: float
    details: dict[str, Any]


@dataclass
class DuplicationResult:
    """Result of deduplication analysis."""
    original_count: int
    unique_count: int
    duplicates_removed: int
    duplicate_groups: list[list[str]]  # Groups of duplicate conversation IDs
    similarity_distribution: dict[str, int]
    processing_time: float
    details: dict[str, Any]


class ConversationDeduplicator:
    """
    Comprehensive conversation deduplication system.

    Uses multiple similarity metrics to identify duplicates:
    - Content similarity (exact and fuzzy text matching)
    - Semantic similarity (meaning-based comparison)
    - Structural similarity (conversation flow and patterns)
    """

    def __init__(self, similarity_threshold: float = 0.85,
                 content_weight: float = 0.4,
                 semantic_weight: float = 0.4,
                 structural_weight: float = 0.2):
        """
        Initialize the deduplication system.

        Args:
            similarity_threshold: Threshold for considering conversations as duplicates
            content_weight: Weight for content similarity in overall score
            semantic_weight: Weight for semantic similarity in overall score
            structural_weight: Weight for structural similarity in overall score
        """
        self.similarity_threshold = similarity_threshold
        self.content_weight = content_weight
        self.semantic_weight = semantic_weight
        self.structural_weight = structural_weight

        # Normalize weights
        total_weight = content_weight + semantic_weight + structural_weight
        self.content_weight /= total_weight
        self.semantic_weight /= total_weight
        self.structural_weight /= total_weight

        logger.info(f"Deduplicator initialized with threshold {similarity_threshold}")

    def deduplicate_conversations(self, conversations: list[Conversation]) -> tuple[list[Conversation], DuplicationResult]:
        """
        Remove duplicate conversations from the dataset.

        Args:
            conversations: List of conversations to deduplicate

        Returns:
            Tuple of (unique_conversations, deduplication_result)
        """
        import time
        start_time = time.time()

        logger.info(f"Starting deduplication of {len(conversations)} conversations")

        # Step 1: Quick hash-based exact duplicate detection
        exact_duplicates = self._find_exact_duplicates(conversations)

        # Step 2: Remove exact duplicates and get unique conversations
        unique_by_hash = self._remove_exact_duplicates(conversations, exact_duplicates)

        # Step 3: Similarity-based duplicate detection on remaining conversations
        similarity_duplicates = self._find_similarity_duplicates(unique_by_hash)

        # Step 4: Remove similarity-based duplicates
        final_unique = self._remove_similarity_duplicates(unique_by_hash, similarity_duplicates)

        # Generate comprehensive result
        processing_time = time.time() - start_time

        all_duplicate_groups = list(exact_duplicates.values()) + similarity_duplicates

        result = DuplicationResult(
            original_count=len(conversations),
            unique_count=len(final_unique),
            duplicates_removed=len(conversations) - len(final_unique),
            duplicate_groups=all_duplicate_groups,
            similarity_distribution=self._calculate_similarity_distribution(conversations),
            processing_time=processing_time,
            details={
                "exact_duplicate_groups": len(exact_duplicates),
                "similarity_duplicate_groups": len(similarity_duplicates),
                "exact_duplicates_removed": len(conversations) - len(unique_by_hash),
                "similarity_duplicates_removed": len(unique_by_hash) - len(final_unique)
            }
        )

        logger.info(f"Deduplication completed: {len(conversations)} -> {len(final_unique)} conversations")

        return final_unique, result

    def calculate_similarity(self, conv1: Conversation, conv2: Conversation) -> SimilarityMetrics:
        """
        Calculate comprehensive similarity between two conversations.

        Args:
            conv1: First conversation
            conv2: Second conversation

        Returns:
            SimilarityMetrics with detailed similarity scores
        """
        # Content similarity
        content_sim = self._calculate_content_similarity(conv1, conv2)

        # Semantic similarity
        semantic_sim = self._calculate_semantic_similarity(conv1, conv2)

        # Structural similarity
        structural_sim = self._calculate_structural_similarity(conv1, conv2)

        # Overall weighted similarity
        overall_sim = (
            content_sim * self.content_weight +
            semantic_sim * self.semantic_weight +
            structural_sim * self.structural_weight
        )

        return SimilarityMetrics(
            content_similarity=content_sim,
            semantic_similarity=semantic_sim,
            structural_similarity=structural_sim,
            overall_similarity=overall_sim,
            details={
                "content_weight": self.content_weight,
                "semantic_weight": self.semantic_weight,
                "structural_weight": self.structural_weight,
                "threshold": self.similarity_threshold
            }
        )

    def _find_exact_duplicates(self, conversations: list[Conversation]) -> dict[str, list[str]]:
        """Find exact duplicates using content hashing."""
        hash_groups = defaultdict(list)

        for conv in conversations:
            content_hash = self._generate_content_hash(conv)
            hash_groups[content_hash].append(conv.id)

        # Return only groups with duplicates
        return {h: ids for h, ids in hash_groups.items() if len(ids) > 1}

    def _remove_exact_duplicates(self, conversations: list[Conversation],
                                exact_duplicates: dict[str, list[str]]) -> list[Conversation]:
        """Remove exact duplicates, keeping the first occurrence."""
        if not exact_duplicates:
            return conversations

        # Create set of IDs to keep (first in each duplicate group)
        ids_to_keep = set()
        duplicate_ids = set()

        for group in exact_duplicates.values():
            ids_to_keep.add(group[0])  # Keep first occurrence
            duplicate_ids.update(group[1:])  # Mark rest as duplicates

        # Keep conversations not in duplicate groups + first occurrence of each group
        return [conv for conv in conversations
                if conv.id not in duplicate_ids or conv.id in ids_to_keep]

    def _find_similarity_duplicates(self, conversations: list[Conversation]) -> list[list[str]]:
        """Find similarity-based duplicates using comprehensive similarity metrics."""
        duplicate_groups = []
        processed_ids = set()

        for i, conv1 in enumerate(conversations):
            if conv1.id in processed_ids:
                continue

            current_group = [conv1.id]

            for _j, conv2 in enumerate(conversations[i+1:], i+1):
                if conv2.id in processed_ids:
                    continue

                similarity = self.calculate_similarity(conv1, conv2)

                if similarity.overall_similarity >= self.similarity_threshold:
                    current_group.append(conv2.id)
                    processed_ids.add(conv2.id)

            if len(current_group) > 1:
                duplicate_groups.append(current_group)
                processed_ids.update(current_group)

        return duplicate_groups

    def _remove_similarity_duplicates(self, conversations: list[Conversation],
                                    similarity_duplicates: list[list[str]]) -> list[Conversation]:
        """Remove similarity-based duplicates, keeping the first occurrence."""
        if not similarity_duplicates:
            return conversations

        # Create set of IDs to remove (all but first in each group)
        ids_to_remove = set()

        for group in similarity_duplicates:
            ids_to_remove.update(group[1:])  # Remove all but first

        return [conv for conv in conversations if conv.id not in ids_to_remove]

    def _generate_content_hash(self, conversation: Conversation) -> str:
        """Generate hash for exact duplicate detection."""
        # Normalize content for hashing
        normalized_content = []

        for message in conversation.messages:
            # Normalize whitespace and case
            normalized = re.sub(r"\s+", " ", message.content.lower().strip())
            normalized_content.append(f"{message.role}:{normalized}")

        content_str = "|".join(normalized_content)
        return hashlib.sha256(content_str.encode()).hexdigest()

    def _calculate_content_similarity(self, conv1: Conversation, conv2: Conversation) -> float:
        """Calculate content-based similarity using text comparison."""
        # Extract all content
        content1 = " ".join(msg.content for msg in conv1.messages)
        content2 = " ".join(msg.content for msg in conv2.messages)

        # Normalize content
        content1 = re.sub(r"\s+", " ", content1.lower().strip())
        content2 = re.sub(r"\s+", " ", content2.lower().strip())

        return difflib.SequenceMatcher(None, content1, content2).ratio()

    def _calculate_semantic_similarity(self, conv1: Conversation, conv2: Conversation) -> float:
        """Calculate semantic similarity using word overlap and patterns."""
        # Extract words from both conversations
        words1 = set()
        words2 = set()

        for msg in conv1.messages:
            words = re.findall(r"\b\w+\b", msg.content.lower())
            words1.update(words)

        for msg in conv2.messages:
            words = re.findall(r"\b\w+\b", msg.content.lower())
            words2.update(words)

        # Calculate Jaccard similarity
        intersection = len(words1 & words2)
        union = len(words1 | words2)

        jaccard_similarity_score = intersection / union if union > 0 else 0.0

        # Additional semantic checks
        semantic_patterns = [
            "anxiety", "depression", "stress", "therapy", "counseling",
            "mental health", "emotional", "psychological", "treatment"
        ]

        patterns1 = sum(
            any(pattern in msg.content.lower() for msg in conv1.messages)
            for pattern in semantic_patterns
        )
        patterns2 = sum(
            any(pattern in msg.content.lower() for msg in conv2.messages)
            for pattern in semantic_patterns
        )

        pattern_similarity = 1.0 - abs(patterns1 - patterns2) / max(patterns1 + patterns2, 1)

        # Weighted combination
        return 0.7 * jaccard_similarity_score + 0.3 * pattern_similarity

    def _calculate_structural_similarity(self, conv1: Conversation, conv2: Conversation) -> float:
        """Calculate structural similarity based on conversation patterns."""
        # Message count similarity
        count_diff = abs(len(conv1.messages) - len(conv2.messages))
        max_count = max(len(conv1.messages), len(conv2.messages))
        count_similarity = 1.0 - (count_diff / max_count) if max_count > 0 else 1.0

        # Role pattern similarity
        pattern1 = [msg.role for msg in conv1.messages]
        pattern2 = [msg.role for msg in conv2.messages]

        pattern_similarity = difflib.SequenceMatcher(None, pattern1, pattern2).ratio()

        # Message length pattern similarity
        lengths1 = [len(msg.content) for msg in conv1.messages]
        lengths2 = [len(msg.content) for msg in conv2.messages]

        # Normalize lengths to categories
        def categorize_length(length):
            if length < 20: return "short"
            if length < 100: return "medium"
            return "long"

        length_pattern1 = [categorize_length(l) for l in lengths1]
        length_pattern2 = [categorize_length(l) for l in lengths2]

        length_similarity = difflib.SequenceMatcher(None, length_pattern1, length_pattern2).ratio()

        # Weighted combination
        return 0.4 * count_similarity + 0.4 * pattern_similarity + 0.2 * length_similarity

    def _calculate_similarity_distribution(self, conversations: list[Conversation]) -> dict[str, int]:
        """Calculate distribution of similarity scores for analysis."""
        distribution = {
            "very_high": 0,  # 0.9+
            "high": 0,       # 0.8-0.9
            "medium": 0,     # 0.6-0.8
            "low": 0,        # 0.4-0.6
            "very_low": 0    # <0.4
        }

        # Sample a subset for performance
        sample_size = min(100, len(conversations))
        sample_conversations = conversations[:sample_size]

        comparisons = 0
        for i, conv1 in enumerate(sample_conversations):
            for conv2 in sample_conversations[i+1:]:
                similarity = self.calculate_similarity(conv1, conv2)
                score = similarity.overall_similarity

                if score >= 0.9:
                    distribution["very_high"] += 1
                elif score >= 0.8:
                    distribution["high"] += 1
                elif score >= 0.6:
                    distribution["medium"] += 1
                elif score >= 0.4:
                    distribution["low"] += 1
                else:
                    distribution["very_low"] += 1

                comparisons += 1

        return distribution


def jaccard_similarity(a: str, b: str) -> float:
    """
    Computes Jaccard similarity between two strings.
    """
    set_a = set(a.lower().split())
    set_b = set(b.lower().split())
    if not set_a and not set_b:
        return 1.0
    if not set_a or not set_b:
        return 0.0
    return len(set_a & set_b) / len(set_a | set_b)


def conversation_text(conv: Conversation) -> str:
    """
    Concatenates all message contents in a conversation for similarity comparison.
    """
    return " ".join([msg.content for msg in conv.messages])


# Enhanced backward compatibility function
def deduplicate_conversations(
    conversations: list[Conversation], similarity_threshold: float = 0.9
) -> tuple[list[Conversation], list[tuple[int, int, float]]]:
    """
    Enhanced deduplication function with backward compatibility.

    Now uses the comprehensive ConversationDeduplicator system while maintaining
    the original function signature for backward compatibility.

    Args:
        conversations: List of Conversation objects.
        similarity_threshold: Float between 0.0 and 1.0. Pairs above this are considered duplicates.

    Returns:
        Tuple:
            - List of unique conversations
            - List of (idx1, idx2, similarity) for removed duplicates
    """
    # Use the new comprehensive deduplication system
    deduplicator = ConversationDeduplicator(similarity_threshold=similarity_threshold)
    unique_conversations, result = deduplicator.deduplicate_conversations(conversations)

    # Convert to old format for backward compatibility
    duplicates = []
    original_ids = [conv.id for conv in conversations]

    for group in result.duplicate_groups:
        for duplicate_id in group[1:]:  # Skip first (kept) conversation
            # Find original indices
            try:
                kept_idx = original_ids.index(group[0])
                duplicate_idx = original_ids.index(duplicate_id)
                # Use a default similarity score since we don't track individual scores
                duplicates.append((kept_idx, duplicate_idx, similarity_threshold))
            except ValueError:
                # Handle case where ID not found (shouldn't happen but be safe)
                continue

    return unique_conversations, duplicates


# Legacy function for simple Jaccard similarity (kept for compatibility)
def deduplicate_conversations_simple(
    conversations: list[Conversation], similarity_threshold: float = 0.9
) -> tuple[list[Conversation], list[tuple[int, int, float]]]:
    """
    Original simple deduplication using only Jaccard similarity.
    Kept for backward compatibility and performance comparison.
    """
    unique = []
    seen: list[str] = []
    duplicates = []
    for idx, conv in enumerate(conversations):
        text = conversation_text(conv)
        is_duplicate = False
        for sidx, stext in enumerate(seen):
            sim = jaccard_similarity(text, stext)
            if sim >= similarity_threshold:
                duplicates.append((sidx, idx, sim))
                is_duplicate = True
                break
        if not is_duplicate:
            unique.append(conv)
            seen.append(text)
    return unique, duplicates

# Alias for compatibility
Deduplicator = ConversationDeduplicator
