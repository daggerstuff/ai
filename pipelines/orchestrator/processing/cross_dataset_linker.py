#!/usr/bin/env python3
"""
Cross-Dataset Conversation Linking for Task 6.5
Links related conversations across different datasets for comprehensive analysis.
"""

import hashlib
import logging
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class LinkType(Enum):
    """Types of conversation links."""
    EXACT_DUPLICATE = "exact_duplicate"
    NEAR_DUPLICATE = "near_duplicate"
    THEMATIC_SIMILAR = "thematic_similar"
    TEMPORAL_RELATED = "temporal_related"
    USER_RELATED = "user_related"
    TOPIC_RELATED = "topic_related"
    RESPONSE_PATTERN = "response_pattern"


class DatasetSource(Enum):
    """Dataset sources for linking."""
    PRIORITY = "priority"
    COT = "cot"
    REDDIT = "reddit"
    SYNTHETIC = "synthetic"
    EXTERNAL = "external"


@dataclass
class ConversationLink:
    """Link between conversations."""
    link_id: str
    source_conversation_id: str
    target_conversation_id: str
    source_dataset: DatasetSource
    target_dataset: DatasetSource
    link_type: LinkType
    similarity_score: float
    confidence: float
    link_metadata: dict[str, Any]
    created_at: datetime = field(default_factory=datetime.now)


@dataclass
class LinkingResult:
    """Result of cross-dataset linking."""
    total_conversations: int
    total_links: int
    links_by_type: dict[LinkType, int]
    links_by_dataset_pair: dict[tuple[str, str], int]
    duplicate_clusters: list[list[str]]
    similarity_distribution: dict[str, int]
    linking_metadata: dict[str, Any]


class CrossDatasetLinker:
    """
    Cross-dataset conversation linking system.
    """

    def __init__(self):
        """Initialize the cross-dataset linker."""
        self.conversations: dict[str, dict[str, Any]] = {}
        self.links: list[ConversationLink] = []
        self.similarity_threshold = 0.8
        self.topic_keywords = self._load_topic_keywords()

        logger.info("CrossDatasetLinker initialized")

    def _load_topic_keywords(self) -> dict[str, list[str]]:
        """Load topic keywords for thematic linking."""
        return {
            "anxiety": [
                "anxiety", "anxious", "worry", "panic", "fear", "nervous",
                "stress", "overwhelmed", "restless", "tense"
            ],
            "depression": [
                "depression", "depressed", "sad", "down", "hopeless", "empty",
                "worthless", "tired", "unmotivated", "despair"
            ],
            "relationships": [
                "relationship", "partner", "boyfriend", "girlfriend", "marriage",
                "divorce", "breakup", "dating", "love", "family"
            ],
            "work": [
                "work", "job", "career", "boss", "colleague", "office",
                "workplace", "employment", "salary", "promotion"
            ],
            "therapy": [
                "therapy", "therapist", "counseling", "treatment", "session",
                "therapeutic", "healing", "recovery", "support"
            ],
            "trauma": [
                "trauma", "abuse", "ptsd", "flashback", "trigger", "survivor",
                "assault", "violence", "accident", "loss"
            ]
        }

    def add_conversations(self, conversations: list[dict[str, Any]],
                         dataset_source: DatasetSource) -> int:
        """Add conversations from a dataset."""
        added_count = 0

        for conv in conversations:
            conv_id = self._generate_conversation_id(conv, dataset_source)

            # Add dataset source to conversation metadata
            conv_with_metadata = conv.copy()
            conv_with_metadata["dataset_source"] = dataset_source.value
            conv_with_metadata["conversation_id"] = conv_id

            self.conversations[conv_id] = conv_with_metadata
            added_count += 1

        logger.info(f"Added {added_count} conversations from {dataset_source.value}")
        return added_count

    def link_conversations(self) -> LinkingResult:
        """Link conversations across datasets."""
        logger.info("Starting cross-dataset conversation linking")

        self.links = []
        conversation_ids = list(self.conversations.keys())

        # Find different types of links
        self._find_duplicate_links(conversation_ids)
        self._find_thematic_links(conversation_ids)
        self._find_temporal_links(conversation_ids)
        self._find_response_pattern_links(conversation_ids)

        # Generate linking result
        result = self._generate_linking_result()

        logger.info(f"Linking completed: {len(self.links)} links found")
        return result

    def _generate_conversation_id(self, conversation: dict[str, Any],
                                 dataset_source: DatasetSource) -> str:
        """Generate unique conversation ID."""
        content = self._extract_content(conversation)
        hash_input = f"{dataset_source.value}_{content[:200]}"
        return hashlib.md5(hash_input.encode()).hexdigest()[:16]

    def _extract_content(self, conversation: dict[str, Any]) -> str:
        """Extract content from conversation."""
        content = conversation.get("content", "")
        if isinstance(content, list):
            content = " ".join(str(turn) for turn in content)
        elif isinstance(content, dict) and "turns" in conversation:
            turns = conversation["turns"]
            content = " ".join(turn.get("content", "") for turn in turns)
        return content.lower()

    def _find_duplicate_links(self, conversation_ids: list[str]):
        """Find duplicate and near-duplicate conversations."""
        logger.info("Finding duplicate links")

        for i, conv_id1 in enumerate(conversation_ids):
            conv1 = self.conversations[conv_id1]
            content1 = self._extract_content(conv1)

            for conv_id2 in conversation_ids[i+1:]:
                conv2 = self.conversations[conv_id2]
                content2 = self._extract_content(conv2)

                # Skip if same dataset
                if conv1["dataset_source"] == conv2["dataset_source"]:
                    continue

                # Calculate similarity
                similarity = self._calculate_text_similarity(content1, content2)

                if similarity >= 0.95:
                    # Exact duplicate
                    link = self._create_link(
                        conv_id1, conv_id2, LinkType.EXACT_DUPLICATE,
                        similarity, 0.95, {"similarity_method": "text"}
                    )
                    self.links.append(link)

                elif similarity >= self.similarity_threshold:
                    # Near duplicate
                    link = self._create_link(
                        conv_id1, conv_id2, LinkType.NEAR_DUPLICATE,
                        similarity, 0.8, {"similarity_method": "text"}
                    )
                    self.links.append(link)

    def _find_thematic_links(self, conversation_ids: list[str]):
        """Find thematically similar conversations."""
        logger.info("Finding thematic links")

        # Group conversations by topic
        topic_groups = {}
        for conv_id in conversation_ids:
            conv = self.conversations[conv_id]
            content = self._extract_content(conv)
            topics = self._identify_topics(content)

            for topic in topics:
                if topic not in topic_groups:
                    topic_groups[topic] = []
                topic_groups[topic].append(conv_id)

        # Create links within topic groups
        for topic, conv_ids in topic_groups.items():
            if len(conv_ids) < 2:
                continue

            for i, conv_id1 in enumerate(conv_ids):
                conv1 = self.conversations[conv_id1]

                for conv_id2 in conv_ids[i+1:]:
                    conv2 = self.conversations[conv_id2]

                    # Skip if same dataset
                    if conv1["dataset_source"] == conv2["dataset_source"]:
                        continue

                    # Skip if already linked
                    if self._already_linked(conv_id1, conv_id2):
                        continue

                    # Calculate thematic similarity
                    similarity = self._calculate_thematic_similarity(conv1, conv2, topic)

                    if similarity >= 0.6:
                        link = self._create_link(
                            conv_id1, conv_id2, LinkType.THEMATIC_SIMILAR,
                            similarity, 0.7, {"topic": topic}
                        )
                        self.links.append(link)

    def _find_temporal_links(self, conversation_ids: list[str]):
        """Find temporally related conversations."""
        logger.info("Finding temporal links")

        # Extract temporal patterns
        temporal_patterns = {}
        for conv_id in conversation_ids:
            conv = self.conversations[conv_id]
            content = self._extract_content(conv)
            patterns = self._extract_temporal_patterns(content)

            for pattern in patterns:
                if pattern not in temporal_patterns:
                    temporal_patterns[pattern] = []
                temporal_patterns[pattern].append(conv_id)

        # Create temporal links
        for pattern, conv_ids in temporal_patterns.items():
            if len(conv_ids) < 2:
                continue

            for i, conv_id1 in enumerate(conv_ids):
                conv1 = self.conversations[conv_id1]

                for conv_id2 in conv_ids[i+1:]:
                    conv2 = self.conversations[conv_id2]

                    # Skip if same dataset
                    if conv1["dataset_source"] == conv2["dataset_source"]:
                        continue

                    # Skip if already linked
                    if self._already_linked(conv_id1, conv_id2):
                        continue

                    link = self._create_link(
                        conv_id1, conv_id2, LinkType.TEMPORAL_RELATED,
                        0.7, 0.6, {"temporal_pattern": pattern}
                    )
                    self.links.append(link)

    def _find_response_pattern_links(self, conversation_ids: list[str]):
        """Find conversations with similar response patterns."""
        logger.info("Finding response pattern links")

        # Extract response patterns
        response_patterns = {}
        for conv_id in conversation_ids:
            conv = self.conversations[conv_id]
            patterns = self._extract_response_patterns(conv)

            for pattern in patterns:
                if pattern not in response_patterns:
                    response_patterns[pattern] = []
                response_patterns[pattern].append(conv_id)

        # Create response pattern links
        for pattern, conv_ids in response_patterns.items():
            if len(conv_ids) < 2:
                continue

            for i, conv_id1 in enumerate(conv_ids):
                conv1 = self.conversations[conv_id1]

                for conv_id2 in conv_ids[i+1:]:
                    conv2 = self.conversations[conv_id2]

                    # Skip if same dataset
                    if conv1["dataset_source"] == conv2["dataset_source"]:
                        continue

                    # Skip if already linked
                    if self._already_linked(conv_id1, conv_id2):
                        continue

                    link = self._create_link(
                        conv_id1, conv_id2, LinkType.RESPONSE_PATTERN,
                        0.6, 0.5, {"response_pattern": pattern}
                    )
                    self.links.append(link)

    def _calculate_text_similarity(self, text1: str, text2: str) -> float:
        """Calculate text similarity between two pieces of content."""
        if not text1 or not text2:
            return 0.0

        # Simple word-based similarity
        words1 = set(text1.split())
        words2 = set(text2.split())

        if not words1 or not words2:
            return 0.0

        intersection = len(words1.intersection(words2))
        union = len(words1.union(words2))

        return intersection / union if union > 0 else 0.0

    def _identify_topics(self, content: str) -> list[str]:
        """Identify topics in content."""
        topics = []

        for topic, keywords in self.topic_keywords.items():
            if any(keyword in content for keyword in keywords):
                topics.append(topic)

        return topics

    def _calculate_thematic_similarity(self, conv1: dict[str, Any],
                                     conv2: dict[str, Any], topic: str) -> float:
        """Calculate thematic similarity for a specific topic."""
        content1 = self._extract_content(conv1)
        content2 = self._extract_content(conv2)

        # Count topic-related keywords
        keywords = self.topic_keywords.get(topic, [])

        count1 = sum(1 for keyword in keywords if keyword in content1)
        count2 = sum(1 for keyword in keywords if keyword in content2)

        # Normalize by content length
        len1 = len(content1.split())
        len2 = len(content2.split())

        if len1 == 0 or len2 == 0:
            return 0.0

        density1 = count1 / len1
        density2 = count2 / len2

        # Calculate similarity based on keyword density
        avg_density = (density1 + density2) / 2
        return min(1.0, avg_density * 10)  # Scale up for better discrimination

    def _extract_temporal_patterns(self, content: str) -> list[str]:
        """Extract temporal patterns from content."""
        patterns = []

        temporal_markers = {
            "recent": ["recently", "lately", "these days", "now"],
            "past": ["used to", "before", "previously", "in the past"],
            "future": ["will", "going to", "planning", "hope to"],
            "ongoing": ["still", "continue", "keep", "always"],
            "frequency": ["daily", "weekly", "often", "sometimes", "rarely"]
        }

        for pattern_type, markers in temporal_markers.items():
            if any(marker in content for marker in markers):
                patterns.append(pattern_type)

        return patterns

    def _extract_response_patterns(self, conversation: dict[str, Any]) -> list[str]:
        """Extract response patterns from conversation."""
        patterns = []
        content = self._extract_content(conversation)

        response_indicators = {
            "empathetic": ["i understand", "that sounds", "i can see"],
            "questioning": ["what do you", "how do you", "tell me more"],
            "supportive": ["you're not alone", "it's okay", "you can"],
            "educational": ["it's important", "research shows", "studies indicate"],
            "directive": ["you should", "try to", "i recommend"]
        }

        for pattern_type, indicators in response_indicators.items():
            if any(indicator in content for indicator in indicators):
                patterns.append(pattern_type)

        return patterns

    def _create_link(self, source_id: str, target_id: str, link_type: LinkType,
                    similarity: float, confidence: float,
                    metadata: dict[str, Any]) -> ConversationLink:
        """Create a conversation link."""
        source_conv = self.conversations[source_id]
        target_conv = self.conversations[target_id]

        link_id = hashlib.md5(f"{source_id}_{target_id}_{link_type.value}".encode()).hexdigest()[:12]

        return ConversationLink(
            link_id=link_id,
            source_conversation_id=source_id,
            target_conversation_id=target_id,
            source_dataset=DatasetSource(source_conv["dataset_source"]),
            target_dataset=DatasetSource(target_conv["dataset_source"]),
            link_type=link_type,
            similarity_score=similarity,
            confidence=confidence,
            link_metadata=metadata
        )

    def _already_linked(self, conv_id1: str, conv_id2: str) -> bool:
        """Check if two conversations are already linked."""
        for link in self.links:
            if ((link.source_conversation_id == conv_id1 and link.target_conversation_id == conv_id2) or
                (link.source_conversation_id == conv_id2 and link.target_conversation_id == conv_id1)):
                return True
        return False

    def _generate_linking_result(self) -> LinkingResult:
        """Generate linking result summary."""
        # Count links by type
        links_by_type = {}
        for link in self.links:
            links_by_type[link.link_type] = links_by_type.get(link.link_type, 0) + 1

        # Count links by dataset pair
        links_by_dataset_pair = {}
        for link in self.links:
            pair = (link.source_dataset.value, link.target_dataset.value)
            # Normalize pair order
            pair = tuple(sorted(pair))
            links_by_dataset_pair[pair] = links_by_dataset_pair.get(pair, 0) + 1

        # Find duplicate clusters
        duplicate_clusters = self._find_duplicate_clusters()

        # Similarity distribution
        similarity_distribution = self._calculate_similarity_distribution()

        return LinkingResult(
            total_conversations=len(self.conversations),
            total_links=len(self.links),
            links_by_type=links_by_type,
            links_by_dataset_pair=links_by_dataset_pair,
            duplicate_clusters=duplicate_clusters,
            similarity_distribution=similarity_distribution,
            linking_metadata={
                "similarity_threshold": self.similarity_threshold,
                "linking_timestamp": datetime.now().isoformat(),
                "datasets_processed": len({conv["dataset_source"] for conv in self.conversations.values()})
            }
        )

    def _find_duplicate_clusters(self) -> list[list[str]]:
        """Find clusters of duplicate conversations."""
        clusters = []
        processed = set()

        for link in self.links:
            if link.link_type in [LinkType.EXACT_DUPLICATE, LinkType.NEAR_DUPLICATE]:
                source_id = link.source_conversation_id
                target_id = link.target_conversation_id

                if source_id in processed or target_id in processed:
                    continue

                # Find all conversations linked to these
                cluster = {source_id, target_id}

                # Expand cluster
                changed = True
                while changed:
                    changed = False
                    for other_link in self.links:
                        if other_link.link_type in [LinkType.EXACT_DUPLICATE, LinkType.NEAR_DUPLICATE]:
                            if (other_link.source_conversation_id in cluster and
                                other_link.target_conversation_id not in cluster):
                                cluster.add(other_link.target_conversation_id)
                                changed = True
                            elif (other_link.target_conversation_id in cluster and
                                  other_link.source_conversation_id not in cluster):
                                cluster.add(other_link.source_conversation_id)
                                changed = True

                clusters.append(list(cluster))
                processed.update(cluster)

        return clusters

    def _calculate_similarity_distribution(self) -> dict[str, int]:
        """Calculate distribution of similarity scores."""
        distribution = {
            "very_high": 0,  # 0.9+
            "high": 0,       # 0.8-0.9
            "medium": 0,     # 0.6-0.8
            "low": 0         # <0.6
        }

        for link in self.links:
            score = link.similarity_score
            if score >= 0.9:
                distribution["very_high"] += 1
            elif score >= 0.8:
                distribution["high"] += 1
            elif score >= 0.6:
                distribution["medium"] += 1
            else:
                distribution["low"] += 1

        return distribution

    def get_links_for_conversation(self, conversation_id: str) -> list[ConversationLink]:
        """Get all links for a specific conversation."""
        links = []
        for link in self.links:
            if (conversation_id in (link.source_conversation_id, link.target_conversation_id)):
                links.append(link)
        return links

    def get_linking_summary(self) -> dict[str, Any]:
        """Get summary of linking results."""
        if not self.links:
            return {"status": "no_links", "total_conversations": len(self.conversations)}

        result = self._generate_linking_result()

        return {
            "total_conversations": result.total_conversations,
            "total_links": result.total_links,
            "link_density": result.total_links / result.total_conversations if result.total_conversations > 0 else 0,
            "links_by_type": {lt.value: count for lt, count in result.links_by_type.items()},
            "duplicate_clusters": len(result.duplicate_clusters),
            "similarity_distribution": result.similarity_distribution,
            "cross_dataset_pairs": len(result.links_by_dataset_pair)
        }


def main():
    """Test the cross-dataset linker."""
    linker = CrossDatasetLinker()

    # Test conversations from different datasets
    priority_conversations = [
        {"content": "I'm feeling anxious about my upcoming presentation. Can you help me?"},
        {"content": "I've been struggling with depression for months. What should I do?"},
    ]

    reddit_conversations = [
        {"content": "Anyone else feel anxious about presentations? Looking for advice."},
        {"content": "Depression is hard. Has anyone found good coping strategies?"},
        {"content": "Work stress is overwhelming me lately. Need support."},
    ]

    cot_conversations = [
        {"content": "Let's think through your presentation anxiety step by step. What specifically worries you?"},
        {"content": "Depression affects many people. Let's explore your feelings together."},
    ]

    # Add conversations
    linker.add_conversations(priority_conversations, DatasetSource.PRIORITY)
    linker.add_conversations(reddit_conversations, DatasetSource.REDDIT)
    linker.add_conversations(cot_conversations, DatasetSource.COT)


    # Link conversations
    linker.link_conversations()


    # Show some example links
    for _i, _link in enumerate(linker.links[:3]):
        pass

    # Get summary
    linker.get_linking_summary()


if __name__ == "__main__":
    main()
