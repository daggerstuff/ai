"""
Stage-aware deduplication for training data per MasterTrainingPlan.md.

Implements the master plan requirements:
- Primary hash: sha256(lowercase(concat(messages.role + messages.content)))
- Secondary hash: sha1(conversation_id + stage + source + crisis_intensity)
- Conflict resolution: stage4 > stage3 > stage2 > stage1 > supplementary
"""

import hashlib
import logging
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List

from conversation_schema import Conversation

logger = logging.getLogger(__name__)

# Stage priority for conflict resolution (higher = more important)
STAGE_PRIORITY = {
    "stage4_voice_persona": 5,          # Highest priority
    "stage3_edge_stress_test": 4,
    "stage2_therapeutic_expertise": 3,
    "stage1_foundation": 2,
    "supplementary": 1,                 # Lowest priority
}

# Reverse mapping for quick lookup
STAGE_PRIORITY_DEFAULT = 1  # Default for unknown stages


@dataclass
class DeduplicationResult:
    """Result of stage-aware deduplication."""
    original_count: int
    unique_count: int
    duplicates_removed: int
    stage_distribution: Dict[str, int]
    dedup_rate: float
    details: dict


def compute_primary_hash(conversation: Conversation) -> str:
    """
    Primary hash: sha256(lowercase(concat(messages.role + messages.content)))

    Per MasterTrainingPlan.md specification.
    """
    content_parts = []
    for msg in conversation.messages:
        # Concatenate role and content for each message
        role = getattr(msg, 'role', 'unknown')
        msg_content = getattr(msg, 'content', '')
        content_parts.append(f"{role}{msg_content}")

    # Join all parts and lowercase
    full_content = "".join(content_parts).lower()

    # SHA256 hash
    return hashlib.sha256(full_content.encode()).hexdigest()


def compute_secondary_hash(conversation: Conversation) -> str:
    """
    Secondary hash: sha1(conversation_id + stage + source + crisis_intensity)

    Per MasterTrainingPlan.md specification.
    """
    metadata = getattr(conversation, 'metadata', {}) or {}

    # Extract stage from metadata
    stage = metadata.get('stage', 'unknown')

    # Extract source from metadata
    source = metadata.get('source', 'unknown')

    # Extract crisis_intensity from metadata
    crisis_intensity = metadata.get('crisis_intensity', 'unknown')

    # Concatenate for hash input
    hash_input = f"{conversation.conversation_id}{stage}{source}{crisis_intensity}"

    # SHA1 hash
    return hashlib.sha1(hash_input.encode()).hexdigest()


def get_stage_priority(conversation: Conversation) -> int:
    """
    Get stage priority for conflict resolution.

    Higher priority wins when duplicate hash found.
    """
    metadata = getattr(conversation, 'metadata', {}) or {}
    stage = metadata.get('stage', 'supplementary')

    return STAGE_PRIORITY.get(stage, STAGE_PRIORITY_DEFAULT)


def get_stage_name(conversation: Conversation) -> str:
    """Extract stage name from conversation metadata."""
    metadata = getattr(conversation, 'metadata', {}) or {}
    return metadata.get('stage', 'supplementary')


def resolve_conflict(conversations: List[Conversation]) -> Conversation:
    """
    When duplicate hash found, keep highest priority stage.

    Per MasterTrainingPlan.md:
    stage4_voice_persona > stage3_edge_stress_test > stage2_therapeutic_expertise > stage1_foundation > supplementary
    """
    if not conversations:
        raise ValueError("Cannot resolve conflict with no conversations")

    if len(conversations) == 1:
        return conversations[0]

    # Find conversation with highest stage priority
    return max(
        conversations,
        key=lambda c: get_stage_priority(c)
    )


def deduplicate_conversations(
    conversations: List[Conversation]
) -> tuple[List[Conversation], DeduplicationResult]:
    """
    Deduplicate conversations using stage-aware hashing.

    Args:
        conversations: List of conversations to deduplicate
        use_secondary_hash: If True, use secondary hash for additional deduplication

    Returns:
        Tuple of (unique_conversations, deduplication_result)
    """
    if not conversations:
        return [], DeduplicationResult(
            original_count=0,
            unique_count=0,
            duplicates_removed=0,
            stage_distribution={},
            dedup_rate=0.0,
            details={}
        )

    logger.info(f"Starting stage-aware deduplication of {len(conversations)} conversations")

    # Group conversations by primary hash
    hash_groups: Dict[str, List[Conversation]] = defaultdict(list)

    for conv in conversations:
        primary_hash = compute_primary_hash(conv)
        hash_groups[primary_hash].append(conv)

    # Resolve conflicts by stage priority
    unique_conversations = []
    stage_counts: Dict[str, int] = defaultdict(int)
    duplicates_removed = 0

    for primary_hash, group in hash_groups.items():
        if len(group) == 1:
            # No conflict, keep the conversation
            unique_conversations.append(group[0])
            stage_name = get_stage_name(group[0])
            stage_counts[stage_name] += 1
        else:
            # Conflict: resolve by stage priority
            winner = resolve_conflict(group)
            unique_conversations.append(winner)
            stage_name = get_stage_name(winner)
            stage_counts[stage_name] += 1
            duplicates_removed += len(group) - 1

            logger.debug(
                f"Duplicate resolved: kept {winner.conversation_id} "
                f"(stage={stage_name}, priority={get_stage_priority(winner)}) "
                f"over {len(group) - 1} others"
            )

    # Calculate deduplication rate
    dedup_rate = duplicates_removed / len(conversations) if conversations else 0.0

    result = DeduplicationResult(
        original_count=len(conversations),
        unique_count=len(unique_conversations),
        duplicates_removed=duplicates_removed,
        stage_distribution=dict(stage_counts),
        dedup_rate=dedup_rate,
        details={
            "primary_hash_groups": len(hash_groups),
            "conflicts_resolved": len([g for g in hash_groups.values() if len(g) > 1]),
        }
    )

    logger.info(
        f"Deduplication complete: {len(conversations)} -> {len(unique_conversations)} "
        f"({duplicates_removed} removed, {dedup_rate:.2%} dedup rate)"
    )

    return unique_conversations, result


def deduplicate_with_secondary_hash(
    conversations: List[Conversation]
) -> tuple[List[Conversation], DeduplicationResult, Dict[str, List[Conversation]]]:
    """
    Deduplicate using both primary and secondary hashes.

    This is a two-pass approach:
    1. First pass: Group by primary hash
    2. Second pass: Within each primary group, subgroup by secondary hash
    3. Resolve conflicts by stage priority within each secondary group

    Returns:
        Tuple of (unique_conversations, deduplication_result, conflict_details)
    """
    if not conversations:
        return [], DeduplicationResult(
            original_count=0,
            unique_count=0,
            duplicates_removed=0,
            stage_distribution={},
            dedup_rate=0.0,
            details={}
        ), {}

    logger.info(f"Starting two-pass stage-aware deduplication of {len(conversations)} conversations")

    # First pass: Group by primary hash
    primary_groups: Dict[str, List[Conversation]] = defaultdict(list)
    for conv in conversations:
        primary_hash = compute_primary_hash(conv)
        primary_groups[primary_hash].append(conv)

    # Second pass: Within each primary group, resolve by stage priority
    unique_conversations = []
    stage_counts: Dict[str, int] = defaultdict(int)
    duplicates_removed = 0
    conflict_details: Dict[str, List[Conversation]] = {}

    for primary_hash, group in primary_groups.items():
        if len(group) == 1:
            # No conflict at primary level
            unique_conversations.append(group[0])
            stage_name = get_stage_name(group[0])
            stage_counts[stage_name] += 1
        else:
            # Multiple conversations with same primary hash
            # Sub-group by secondary hash for finer deduplication
            secondary_groups: Dict[str, List[Conversation]] = defaultdict(list)
            for conv in group:
                secondary_hash = compute_secondary_hash(conv)
                secondary_groups[secondary_hash].append(conv)

            # For each secondary group, resolve by stage priority
            for secondary_hash, sub_group in secondary_groups.items():
                if len(sub_group) == 1:
                    unique_conversations.append(sub_group[0])
                    stage_name = get_stage_name(sub_group[0])
                    stage_counts[stage_name] += 1
                else:
                    # Conflict: resolve by stage priority
                    winner = resolve_conflict(sub_group)
                    unique_conversations.append(winner)
                    stage_name = get_stage_name(winner)
                    stage_counts[stage_name] += 1
                    duplicates_removed += len(sub_group) - 1

                    # Track conflict details
                    conflict_key = f"{primary_hash[:8]}:{secondary_hash[:8]}"
                    conflict_details[conflict_key] = sub_group

                    logger.debug(
                        f"Secondary conflict resolved: kept {winner.conversation_id} "
                        f"(stage={stage_name}, priority={get_stage_priority(winner)})"
                    )

    dedup_rate = duplicates_removed / len(conversations) if conversations else 0.0

    result = DeduplicationResult(
        original_count=len(conversations),
        unique_count=len(unique_conversations),
        duplicates_removed=duplicates_removed,
        stage_distribution=dict(stage_counts),
        dedup_rate=dedup_rate,
        details={
            "primary_hash_groups": len(primary_groups),
            "secondary_hash_groups": sum(len(g) for g in secondary_groups.values()) if secondary_groups else 0,
            "conflicts_resolved": len(conflict_details),
        }
    )

    logger.info(
        f"Two-pass deduplication complete: {len(conversations)} -> {len(unique_conversations)} "
        f"({duplicates_removed} removed, {dedup_rate:.2%} dedup rate)"
    )

    return unique_conversations, result, conflict_details


def validate_deduplication(conversations: List[Conversation], target_dedup_rate: float = 0.01) -> bool:
    """
    Validate that deduplication rate is within acceptable bounds.

    Per MasterTrainingPlan.md: target dedup rate < 1%

    Args:
        conversations: List of conversations to validate
        target_dedup_rate: Maximum acceptable deduplication rate (default 1%)

    Returns:
        True if deduplication rate is acceptable, False otherwise
    """
    if len(conversations) < 2:
        return True

    _, result = deduplicate_conversations(conversations)

    if result.dedup_rate > target_dedup_rate:
        logger.warning(
            f"Deduplication rate {result.dedup_rate:.2%} exceeds target {target_dedup_rate:.2%}"
        )
        return False

    logger.info(
        f"Deduplication rate {result.dedup_rate:.2%} within target {target_dedup_rate:.2%}"
    )
    return True


__all__ = [
    "compute_primary_hash",
    "compute_secondary_hash",
    "get_stage_priority",
    "get_stage_name",
    "resolve_conflict",
    "deduplicate_conversations",
    "deduplicate_with_secondary_hash",
    "validate_deduplication",
    "DeduplicationResult",
    "STAGE_PRIORITY",
]
