#!/usr/bin/env python3
"""
Task 6.2: Intelligent Data Fusion Algorithms

This module implements sophisticated algorithms to merge multi-source therapeutic
conversations while preserving quality, removing duplicates, and maintaining
therapeutic coherence across the 6-tier ecosystem.

Strategic Goal: Intelligently fuse 2.59M+ conversations from 50+ datasets
into a coherent, high-quality therapeutic training corpus.
"""

import hashlib
import json
import logging
import statistics
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from difflib import SequenceMatcher
from typing import Any

# Import our ecosystem components
from distributed_architecture import DataTier


@dataclass
class ConversationSignature:
    """Unique signature for conversation deduplication and similarity."""
    content_hash: str
    semantic_hash: str
    length_category: str  # short, medium, long
    topic_category: str
    quality_tier: str
    participant_count: int
    turn_count: int


@dataclass
class FusionCandidate:
    """Represents a conversation candidate for fusion."""
    conversation_id: str
    source_dataset: str
    tier: DataTier
    content: dict[str, Any]
    signature: ConversationSignature
    quality_score: float
    fusion_priority: float
    metadata: dict[str, Any] = None


@dataclass
class FusionResult:
    """Result of the data fusion process."""
    fused_conversation: dict[str, Any]
    source_conversations: list[str]
    fusion_method: str
    quality_improvement: float
    confidence_score: float
    metadata: dict[str, Any] = None


class ConversationSignatureGenerator:
    """Generates unique signatures for conversations to enable intelligent fusion."""

    def __init__(self):
        self.topic_keywords = {
            "anxiety": ["anxious", "worry", "panic", "nervous", "fear", "stress"],
            "depression": ["depressed", "sad", "hopeless", "empty", "down", "worthless"],
            "trauma": ["trauma", "abuse", "ptsd", "flashback", "trigger", "dissociation"],
            "relationships": ["relationship", "partner", "marriage", "family", "conflict"],
            "therapy": ["therapy", "counseling", "treatment", "session", "therapeutic"],
            "coping": ["coping", "manage", "handle", "deal with", "strategy", "skill"],
            "emotions": ["feeling", "emotion", "mood", "affect", "emotional"],
            "behavioral": ["behavior", "habit", "pattern", "action", "change"]
        }

        self.quality_indicators = {
            "high": ["insight", "awareness", "understanding", "growth", "progress"],
            "medium": ["help", "support", "better", "improve", "work on"],
            "low": ["okay", "fine", "whatever", "dunno", "maybe"]
        }

    def generate_signature(self, conversation: dict[str, Any]) -> ConversationSignature:
        """Generate a comprehensive signature for a conversation."""
        # Extract text content
        text_content = self._extract_text_content(conversation)

        # Generate content hash (exact duplicate detection)
        content_hash = hashlib.md5(text_content.encode()).hexdigest()

        # Generate semantic hash (similar content detection)
        semantic_hash = self._generate_semantic_hash(text_content)

        # Categorize length
        word_count = len(text_content.split())
        if word_count < 50:
            length_category = "short"
        elif word_count < 200:
            length_category = "medium"
        else:
            length_category = "long"

        # Identify topic category
        topic_category = self._identify_topic_category(text_content)

        # Determine quality tier
        quality_tier = self._determine_quality_tier(text_content)

        # Count participants and turns
        messages = conversation.get("messages", [])
        participant_count = len({msg.get("role", "unknown") for msg in messages})
        turn_count = len(messages)

        return ConversationSignature(
            content_hash=content_hash,
            semantic_hash=semantic_hash,
            length_category=length_category,
            topic_category=topic_category,
            quality_tier=quality_tier,
            participant_count=participant_count,
            turn_count=turn_count
        )

    def _extract_text_content(self, conversation: dict[str, Any]) -> str:
        """Extract all text content from conversation."""
        text_parts = []

        if "messages" in conversation:
            for message in conversation["messages"]:
                if isinstance(message, dict) and "content" in message:
                    text_parts.append(message["content"])
        elif "conversations" in conversation:
            for conv in conversation["conversations"]:
                if isinstance(conv, dict):
                    for _key, value in conv.items():
                        if isinstance(value, str):
                            text_parts.append(value)

        return " ".join(text_parts).lower()

    def _generate_semantic_hash(self, text: str) -> str:
        """Generate semantic hash for similar content detection."""
        # Remove common words and normalize
        common_words = {"the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for", "of", "with", "by", "is", "are", "was", "were", "be", "been", "have", "has", "had", "do", "does", "did", "will", "would", "could", "should", "may", "might", "can", "i", "you", "he", "she", "it", "we", "they", "me", "him", "her", "us", "them", "my", "your", "his", "its", "our", "their"}

        words = [word for word in text.split() if word not in common_words and len(word) > 2]

        # Sort words to make hash order-independent
        words.sort()

        # Take first 50 words for semantic similarity
        semantic_content = " ".join(words[:50])

        return hashlib.md5(semantic_content.encode()).hexdigest()

    def _identify_topic_category(self, text: str) -> str:
        """Identify the primary topic category of the conversation."""
        topic_scores = {}

        for topic, keywords in self.topic_keywords.items():
            score = sum(1 for keyword in keywords if keyword in text)
            if score > 0:
                topic_scores[topic] = score

        if topic_scores:
            return max(topic_scores, key=topic_scores.get)
        return "general"

    def _determine_quality_tier(self, text: str) -> str:
        """Determine quality tier based on content indicators."""
        quality_scores = {}

        for tier, indicators in self.quality_indicators.items():
            score = sum(1 for indicator in indicators if indicator in text)
            if score > 0:
                quality_scores[tier] = score

        if quality_scores:
            return max(quality_scores, key=quality_scores.get)
        return "medium"


class IntelligentDataFusionEngine:
    """Main engine for intelligent data fusion across the ecosystem."""

    def __init__(self, config: dict[str, Any] | None = None):
        self.config = config or self._default_config()
        self.signature_generator = ConversationSignatureGenerator()

        # Fusion state
        self.conversation_signatures: dict[str, ConversationSignature] = {}
        self.fusion_candidates: dict[str, FusionCandidate] = {}
        self.duplicate_groups: list[list[str]] = []
        self.fusion_results: list[FusionResult] = []

        # Statistics
        self.fusion_stats = {
            "total_conversations": 0,
            "exact_duplicates": 0,
            "semantic_duplicates": 0,
            "fused_conversations": 0,
            "quality_improvements": 0,
            "processing_time": 0.0
        }

        # Setup logging
        self.logger = logging.getLogger(__name__)

    def _default_config(self) -> dict[str, Any]:
        """Default configuration for data fusion."""
        return {
            "similarity_threshold": 0.85,
            "quality_improvement_threshold": 0.1,
            "max_fusion_candidates": 5,
            "enable_semantic_fusion": True,
            "enable_quality_enhancement": True,
            "preserve_source_metadata": True,
            "tier_priority_weights": {
                "priority": 1.0,
                "professional": 0.8,
                "cot_reasoning": 0.6,
                "reddit": 0.4,
                "research": 0.3,
                "knowledge_base": 0.2
            }
        }

    def add_conversation_for_fusion(self, conversation_id: str, conversation: dict[str, Any],
                                  source_dataset: str, tier: DataTier, quality_score: float | None = None):
        """Add a conversation as a candidate for fusion."""
        # Generate signature
        signature = self.signature_generator.generate_signature(conversation)

        # Calculate fusion priority based on tier and quality
        tier_weight = self.config["tier_priority_weights"].get(tier.value, 0.5)
        quality_weight = quality_score or 0.5
        fusion_priority = tier_weight * 0.6 + quality_weight * 0.4

        # Create fusion candidate
        candidate = FusionCandidate(
            conversation_id=conversation_id,
            source_dataset=source_dataset,
            tier=tier,
            content=conversation,
            signature=signature,
            quality_score=quality_score or 0.5,
            fusion_priority=fusion_priority,
            metadata={
                "added_at": datetime.now().isoformat(),
                "word_count": len(signature.semantic_hash),
                "topic": signature.topic_category
            }
        )

        self.fusion_candidates[conversation_id] = candidate
        self.conversation_signatures[conversation_id] = signature
        self.fusion_stats["total_conversations"] += 1

        self.logger.debug(f"Added fusion candidate: {conversation_id} (tier: {tier.value}, priority: {fusion_priority:.3f})")

    def detect_duplicates(self) -> list[list[str]]:
        """Detect duplicate and similar conversations."""
        duplicate_groups = []
        processed_ids = set()

        # Group by exact content hash first
        content_hash_groups = defaultdict(list)
        for conv_id, signature in self.conversation_signatures.items():
            content_hash_groups[signature.content_hash].append(conv_id)

        # Process exact duplicates
        for _hash_key, conv_ids in content_hash_groups.items():
            if len(conv_ids) > 1:
                duplicate_groups.append(conv_ids)
                processed_ids.update(conv_ids)
                self.fusion_stats["exact_duplicates"] += len(conv_ids) - 1

        # Group by semantic hash for similar content
        if self.config["enable_semantic_fusion"]:
            semantic_hash_groups = defaultdict(list)
            for conv_id, signature in self.conversation_signatures.items():
                if conv_id not in processed_ids:
                    semantic_hash_groups[signature.semantic_hash].append(conv_id)

            # Process semantic duplicates
            for _hash_key, conv_ids in semantic_hash_groups.items():
                if len(conv_ids) > 1:
                    # Verify semantic similarity
                    if self._verify_semantic_similarity(conv_ids):
                        duplicate_groups.append(conv_ids)
                        processed_ids.update(conv_ids)
                        self.fusion_stats["semantic_duplicates"] += len(conv_ids) - 1

        self.duplicate_groups = duplicate_groups
        self.logger.info(f"Detected {len(duplicate_groups)} duplicate groups")

        return duplicate_groups

    def _verify_semantic_similarity(self, conv_ids: list[str]) -> bool:
        """Verify that conversations are semantically similar enough to fuse."""
        if len(conv_ids) < 2:
            return False

        # Compare text similarity between conversations
        conversations = [self.fusion_candidates[conv_id].content for conv_id in conv_ids]
        texts = [self.signature_generator._extract_text_content(conv) for conv in conversations]

        # Calculate pairwise similarity
        similarities = []
        for i in range(len(texts)):
            for j in range(i + 1, len(texts)):
                similarity = SequenceMatcher(None, texts[i], texts[j]).ratio()
                similarities.append(similarity)

        # Check if average similarity exceeds threshold
        avg_similarity = statistics.mean(similarities) if similarities else 0
        return avg_similarity >= self.config["similarity_threshold"]

    def fuse_duplicate_groups(self) -> list[FusionResult]:
        """Fuse duplicate conversation groups into high-quality conversations."""
        fusion_results = []

        for group in self.duplicate_groups:
            if len(group) < 2:
                continue

            # Get candidates for this group
            candidates = [self.fusion_candidates[conv_id] for conv_id in group]

            # Sort by fusion priority (highest first)
            candidates.sort(key=lambda c: c.fusion_priority, reverse=True)

            # Perform fusion
            fusion_result = self._fuse_conversation_group(candidates)

            if fusion_result:
                fusion_results.append(fusion_result)
                self.fusion_stats["fused_conversations"] += 1

                if fusion_result.quality_improvement > 0:
                    self.fusion_stats["quality_improvements"] += 1

        self.fusion_results = fusion_results
        self.logger.info(f"Fused {len(fusion_results)} conversation groups")

        return fusion_results

    def _fuse_conversation_group(self, candidates: list[FusionCandidate]) -> FusionResult | None:
        """Fuse a group of similar conversations into a single high-quality conversation."""
        if not candidates:
            return None

        # Use highest priority candidate as base
        base_candidate = candidates[0]
        base_conversation = base_candidate.content.copy()

        # Enhance with content from other candidates
        enhanced_conversation = self._enhance_conversation(base_conversation, candidates[1:])

        # Calculate quality improvement
        original_quality = base_candidate.quality_score
        enhanced_quality = self._estimate_conversation_quality(enhanced_conversation)
        quality_improvement = enhanced_quality - original_quality

        # Calculate confidence score
        confidence_score = self._calculate_fusion_confidence(candidates)

        # Create fusion result
        return FusionResult(
            fused_conversation=enhanced_conversation,
            source_conversations=[c.conversation_id for c in candidates],
            fusion_method="intelligent_enhancement",
            quality_improvement=quality_improvement,
            confidence_score=confidence_score,
            metadata={
                "base_source": base_candidate.source_dataset,
                "base_tier": base_candidate.tier.value,
                "fusion_timestamp": datetime.now().isoformat(),
                "candidate_count": len(candidates),
                "original_quality": original_quality,
                "enhanced_quality": enhanced_quality
            }
        )


    def _enhance_conversation(self, base_conversation: dict[str, Any],
                            enhancement_candidates: list[FusionCandidate]) -> dict[str, Any]:
        """Enhance base conversation with content from other candidates."""
        enhanced = base_conversation.copy()

        # Enhance messages if available
        if "messages" in enhanced and enhancement_candidates:
            base_messages = enhanced["messages"]

            # Look for additional context or better phrasing in candidates
            for candidate in enhancement_candidates:
                candidate_messages = candidate.content.get("messages", [])

                # Enhance messages with better quality content
                enhanced_messages = self._merge_message_sequences(base_messages, candidate_messages)
                enhanced["messages"] = enhanced_messages

        # Add metadata about fusion
        if "metadata" not in enhanced:
            enhanced["metadata"] = {}

        enhanced["metadata"]["fusion_info"] = {
            "is_fused": True,
            "source_count": len(enhancement_candidates) + 1,
            "fusion_method": "intelligent_enhancement",
            "enhancement_sources": [c.source_dataset for c in enhancement_candidates]
        }

        return enhanced

    def _merge_message_sequences(self, base_messages: list[dict], candidate_messages: list[dict]) -> list[dict]:
        """Merge message sequences to create enhanced conversation flow."""
        # For now, use base messages but could implement more sophisticated merging
        # This could include:
        # - Adding missing therapeutic responses
        # - Enhancing emotional depth
        # - Improving conversation flow
        # - Adding clinical insights

        enhanced_messages = base_messages.copy()

        # Simple enhancement: if candidate has more detailed responses, use them
        if len(candidate_messages) > len(base_messages):
            for i in range(len(base_messages), len(candidate_messages)):
                if i < len(candidate_messages):
                    enhanced_messages.append(candidate_messages[i])

        return enhanced_messages

    def _estimate_conversation_quality(self, conversation: dict[str, Any]) -> float:
        """Estimate quality score for a conversation."""
        # Simple quality estimation based on content characteristics
        text = self.signature_generator._extract_text_content(conversation)

        quality_factors = []

        # Length factor (moderate length is better)
        word_count = len(text.split())
        if 50 <= word_count <= 300:
            quality_factors.append(0.8)
        elif 30 <= word_count <= 500:
            quality_factors.append(0.6)
        else:
            quality_factors.append(0.4)

        # Therapeutic content factor
        therapeutic_words = ["therapy", "counseling", "feeling", "emotion", "help", "support", "understand", "insight"]
        therapeutic_count = sum(1 for word in therapeutic_words if word in text)
        quality_factors.append(min(therapeutic_count / 5.0, 1.0))

        # Conversation structure factor
        messages = conversation.get("messages", [])
        if len(messages) >= 4:  # Good back-and-forth
            quality_factors.append(0.8)
        elif len(messages) >= 2:
            quality_factors.append(0.6)
        else:
            quality_factors.append(0.3)

        return statistics.mean(quality_factors) if quality_factors else 0.5

    def _calculate_fusion_confidence(self, candidates: list[FusionCandidate]) -> float:
        """Calculate confidence score for fusion result."""
        factors = []

        # Tier consistency (higher confidence if from similar tiers)
        tiers = [c.tier for c in candidates]
        tier_consistency = len(set(tiers)) / len(tiers)  # Lower is better
        factors.append(1.0 - tier_consistency)

        # Quality consistency
        qualities = [c.quality_score for c in candidates]
        quality_variance = statistics.variance(qualities) if len(qualities) > 1 else 0
        quality_consistency = max(0, 1.0 - quality_variance)
        factors.append(quality_consistency)

        # Source diversity (moderate diversity is good)
        sources = [c.source_dataset for c in candidates]
        source_diversity = len(set(sources)) / len(sources)
        factors.append(min(source_diversity * 2, 1.0))  # Cap at 1.0

        return statistics.mean(factors)

    def get_fusion_statistics(self) -> dict[str, Any]:
        """Get comprehensive fusion statistics."""
        return {
            "processing_stats": self.fusion_stats.copy(),
            "duplicate_analysis": {
                "total_groups": len(self.duplicate_groups),
                "exact_duplicate_groups": len([g for g in self.duplicate_groups if len(g) > 1]),
                "average_group_size": statistics.mean([len(g) for g in self.duplicate_groups]) if self.duplicate_groups else 0
            },
            "fusion_quality": {
                "total_fusions": len(self.fusion_results),
                "quality_improvements": self.fusion_stats["quality_improvements"],
                "average_confidence": statistics.mean([r.confidence_score for r in self.fusion_results]) if self.fusion_results else 0,
                "average_quality_gain": statistics.mean([r.quality_improvement for r in self.fusion_results]) if self.fusion_results else 0
            },
            "tier_distribution": {
                tier.value: len([c for c in self.fusion_candidates.values() if c.tier == tier])
                for tier in DataTier
            }
        }

    def export_fused_conversations(self, output_path: str) -> bool:
        """Export fused conversations to file."""
        try:
            fused_conversations = []

            for result in self.fusion_results:
                conversation = result.fused_conversation.copy()
                conversation["fusion_metadata"] = result.metadata
                fused_conversations.append(conversation)

            with open(output_path, "w") as f:
                for conv in fused_conversations:
                    f.write(json.dumps(conv) + "\n")

            self.logger.info(f"Exported {len(fused_conversations)} fused conversations to {output_path}")
            return True

        except Exception as e:
            self.logger.error(f"Error exporting fused conversations: {e}")
            return False


# Example usage and testing
def main():
    """Example usage of the intelligent data fusion engine."""

    # Create fusion engine
    fusion_engine = IntelligentDataFusionEngine()

    # Sample conversations for testing
    sample_conversations = [
        {
            "id": "conv_1",
            "messages": [
                {"role": "client", "content": "I feel anxious about my job interview tomorrow."},
                {"role": "therapist", "content": "I understand that interviews can be anxiety-provoking. What specifically about the interview is making you feel anxious?"}
            ]
        },
        {
            "id": "conv_2",
            "messages": [
                {"role": "client", "content": "I feel anxious about my job interview tomorrow."},
                {"role": "therapist", "content": "I hear that you're feeling anxious about your upcoming interview. Can you tell me more about what aspects of the interview are causing you the most worry?"}
            ]
        },
        {
            "id": "conv_3",
            "messages": [
                {"role": "client", "content": "I've been feeling depressed lately."},
                {"role": "therapist", "content": "I'm sorry to hear you're struggling with depression. How long have you been experiencing these feelings?"}
            ]
        }
    ]

    # Add conversations for fusion
    for i, conv in enumerate(sample_conversations):
        tier = DataTier.TIER_1_PRIORITY if i < 2 else DataTier.TIER_2_PROFESSIONAL
        fusion_engine.add_conversation_for_fusion(
            conv["id"], conv, f"sample_dataset_{i}", tier, 0.7 + i * 0.1
        )

    # Detect duplicates
    fusion_engine.detect_duplicates()

    # Fuse duplicates
    fusion_engine.fuse_duplicate_groups()

    # Get statistics
    fusion_engine.get_fusion_statistics()

    # Export results
    fusion_engine.export_fused_conversations("fused_conversations.jsonl")


if __name__ == "__main__":
    main()
