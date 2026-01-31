"""
Comprehensive conversation coherence assessment system for the dataset pipeline.
Evaluates multiple dimensions of conversational coherence including logical flow,
contextual consistency, semantic coherence, and dialogue structure.
"""

import re
from dataclasses import dataclass
from typing import Any

from conversation_schema import Conversation, Message
from logger import get_logger

logger = get_logger(__name__)


@dataclass
class CoherenceMetrics:
    """Container for coherence assessment metrics."""
    overall_score: float
    logical_flow_score: float
    contextual_consistency_score: float
    semantic_coherence_score: float
    dialogue_structure_score: float
    turn_taking_score: float
    topic_continuity_score: float
    response_relevance_score: float
    issues: list[str]
    details: dict[str, Any]


class ConversationCoherenceAssessor:
    """
    Comprehensive conversation coherence assessment system.

    Evaluates conversations across multiple dimensions:
    - Logical flow and progression
    - Contextual consistency
    - Semantic coherence
    - Dialogue structure
    - Turn-taking patterns
    - Topic continuity
    - Response relevance
    """

    def __init__(self, config: dict[str, Any] | None = None):
        """Initialize the coherence assessor with configuration."""
        self.config = config or {}
        self.weights = self.config.get("weights", {
            "logical_flow": 0.20,
            "contextual_consistency": 0.18,
            "semantic_coherence": 0.16,
            "dialogue_structure": 0.14,
            "turn_taking": 0.12,
            "topic_continuity": 0.12,
            "response_relevance": 0.08
        })

        # Thresholds for quality assessment
        self.thresholds = self.config.get("thresholds", {
            "excellent": 0.85,
            "good": 0.70,
            "acceptable": 0.55,
            "poor": 0.40
        })

        # Common discourse markers and transition words
        self.discourse_markers = {
            "continuation": ["and", "also", "furthermore", "moreover", "additionally"],
            "contrast": ["but", "however", "nevertheless", "on the other hand", "although"],
            "causation": ["because", "therefore", "thus", "consequently", "as a result"],
            "temporal": ["then", "next", "afterwards", "meanwhile", "previously"],
            "clarification": ["in other words", "that is", "specifically", "for example"]
        }

    def assess_conversation_coherence(self, conversation: Conversation) -> CoherenceMetrics:
        """
        Perform comprehensive coherence assessment of a conversation.

        Args:
            conversation: The conversation to assess

        Returns:
            CoherenceMetrics object with detailed assessment results
        """
        logger.info(f"Assessing coherence for conversation {conversation.id}")

        if not conversation.messages or len(conversation.messages) < 2:
            return self._create_minimal_assessment(conversation, "Insufficient messages for coherence assessment")

        # Assess individual dimensions
        logical_flow = self._assess_logical_flow(conversation)
        contextual_consistency = self._assess_contextual_consistency(conversation)
        semantic_coherence = self._assess_semantic_coherence(conversation)
        dialogue_structure = self._assess_dialogue_structure(conversation)
        turn_taking = self._assess_turn_taking(conversation)
        topic_continuity = self._assess_topic_continuity(conversation)
        response_relevance = self._assess_response_relevance(conversation)

        # Calculate weighted overall score
        overall_score = (
            logical_flow["score"] * self.weights["logical_flow"] +
            contextual_consistency["score"] * self.weights["contextual_consistency"] +
            semantic_coherence["score"] * self.weights["semantic_coherence"] +
            dialogue_structure["score"] * self.weights["dialogue_structure"] +
            turn_taking["score"] * self.weights["turn_taking"] +
            topic_continuity["score"] * self.weights["topic_continuity"] +
            response_relevance["score"] * self.weights["response_relevance"]
        )

        # Collect all issues
        all_issues = []
        all_issues.extend(logical_flow.get("issues", []))
        all_issues.extend(contextual_consistency.get("issues", []))
        all_issues.extend(semantic_coherence.get("issues", []))
        all_issues.extend(dialogue_structure.get("issues", []))
        all_issues.extend(turn_taking.get("issues", []))
        all_issues.extend(topic_continuity.get("issues", []))
        all_issues.extend(response_relevance.get("issues", []))

        # Compile detailed results
        details = {
            "conversation_length": len(conversation.messages),
            "unique_roles": len({msg.role for msg in conversation.messages}),
            "logical_flow_details": logical_flow.get("details", {}),
            "contextual_consistency_details": contextual_consistency.get("details", {}),
            "semantic_coherence_details": semantic_coherence.get("details", {}),
            "dialogue_structure_details": dialogue_structure.get("details", {}),
            "turn_taking_details": turn_taking.get("details", {}),
            "topic_continuity_details": topic_continuity.get("details", {}),
            "response_relevance_details": response_relevance.get("details", {}),
            "quality_level": self._determine_quality_level(overall_score)
        }

        return CoherenceMetrics(
            overall_score=overall_score,
            logical_flow_score=logical_flow["score"],
            contextual_consistency_score=contextual_consistency["score"],
            semantic_coherence_score=semantic_coherence["score"],
            dialogue_structure_score=dialogue_structure["score"],
            turn_taking_score=turn_taking["score"],
            topic_continuity_score=topic_continuity["score"],
            response_relevance_score=response_relevance["score"],
            issues=all_issues,
            details=details
        )

    def _assess_logical_flow(self, conversation: Conversation) -> dict[str, Any]:
        """Assess the logical flow and progression of the conversation."""
        score = 1.0
        issues = []
        details = {}

        messages = conversation.messages

        # Check for logical progression
        discourse_marker_usage = 0

        for i in range(1, len(messages)):
            messages[i-1]
            curr_msg = messages[i]

            # Check for discourse markers that indicate logical flow
            curr_content_lower = curr_msg.content.lower()
            for _category, markers in self.discourse_markers.items():
                for marker in markers:
                    if marker in curr_content_lower:
                        discourse_marker_usage += 1
                        break

            # Simple heuristic: very short responses might indicate poor flow
            if len(curr_msg.content.strip()) < 10 and i > 1:
                issues.append(f"Very short response at turn {i}: '{curr_msg.content.strip()}'")
                score -= 0.15

        # Penalize lack of discourse markers in longer conversations
        if len(messages) > 6 and discourse_marker_usage == 0:
            issues.append("No discourse markers found in extended conversation")
            score -= 0.1

        details.update({
            "discourse_marker_count": discourse_marker_usage,
            "average_message_length": sum(len(msg.content) for msg in messages) / len(messages)
        })

        return {
            "score": max(0.0, min(1.0, score)),
            "issues": issues,
            "details": details
        }

    def _assess_contextual_consistency(self, conversation: Conversation) -> dict[str, Any]:
        """Assess contextual consistency throughout the conversation."""
        score = 1.0
        issues = []
        details = {}

        messages = conversation.messages

        # Track mentioned entities and concepts
        mentioned_entities = set()
        entity_consistency_violations = 0

        # Simple entity extraction (names, places, etc.)
        entity_pattern = r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b"

        for _i, message in enumerate(messages):
            entities_in_message = set(re.findall(entity_pattern, message.content))

            # Check for contradictory information (simplified)
            for entity in entities_in_message:
                if entity in mentioned_entities:
                    # Entity mentioned before - check for consistency
                    # This is a simplified check; real implementation would be more sophisticated
                    pass
                else:
                    mentioned_entities.add(entity)

        # Check for pronoun consistency
        pronoun_issues = self._check_pronoun_consistency(messages)
        issues.extend(pronoun_issues)
        score -= len(pronoun_issues) * 0.05

        details.update({
            "unique_entities_mentioned": len(mentioned_entities),
            "entity_consistency_violations": entity_consistency_violations
        })

        return {
            "score": max(0.0, min(1.0, score)),
            "issues": issues,
            "details": details
        }

    def _check_pronoun_consistency(self, messages: list[Message]) -> list[str]:
        """Check for pronoun consistency issues."""
        issues = []

        # Track pronoun usage patterns
        pronoun_contexts = {}

        for i, message in enumerate(messages):
            content = message.content.lower()

            # Simple pronoun detection
            pronouns = ["he", "she", "they", "it", "him", "her", "them"]
            for pronoun in pronouns:
                if f" {pronoun} " in f" {content} ":
                    if pronoun not in pronoun_contexts:
                        pronoun_contexts[pronoun] = []
                    pronoun_contexts[pronoun].append(i)

        # Check for potential inconsistencies (simplified)
        if "he" in pronoun_contexts and "she" in pronoun_contexts:
            if len(pronoun_contexts["he"]) > 1 and len(pronoun_contexts["she"]) > 1:
                issues.append("Potential pronoun inconsistency: both 'he' and 'she' used frequently")

        return issues

    def _assess_semantic_coherence(self, conversation: Conversation) -> dict[str, Any]:
        """Assess semantic coherence and meaning consistency."""
        score = 1.0
        issues = []
        details = {}

        messages = conversation.messages

        # Check for semantic consistency
        word_overlap_scores = []

        for i in range(1, len(messages)):
            prev_content = messages[i-1].content.lower().split()
            curr_content = messages[i].content.lower().split()

            # Calculate word overlap between consecutive messages
            if prev_content and curr_content:
                overlap = len(set(prev_content) & set(curr_content))
                total_unique = len(set(prev_content) | set(curr_content))
                overlap_score = overlap / total_unique if total_unique > 0 else 0
                word_overlap_scores.append(overlap_score)

        # Very low overlap might indicate poor semantic coherence
        if word_overlap_scores:
            avg_overlap = sum(word_overlap_scores) / len(word_overlap_scores)
            if avg_overlap < 0.05:
                issues.append("Very low semantic overlap between consecutive messages")
                score -= 0.2
            elif avg_overlap < 0.1:
                issues.append("Low semantic overlap between consecutive messages")
                score -= 0.1

        # Check for contradictory statements (simplified)
        contradiction_indicators = ["no", "not", "never", "wrong", "incorrect", "false"]
        contradiction_count = 0

        for i in range(1, len(messages)):
            curr_content = messages[i].content.lower()
            for indicator in contradiction_indicators:
                if indicator in curr_content:
                    contradiction_count += 1
                    break

        if contradiction_count > len(messages) * 0.3:
            issues.append("High frequency of contradiction indicators")
            score -= 0.15

        details.update({
            "average_word_overlap": sum(word_overlap_scores) / len(word_overlap_scores) if word_overlap_scores else 0,
            "contradiction_indicator_count": contradiction_count
        })

        return {
            "score": max(0.0, min(1.0, score)),
            "issues": issues,
            "details": details
        }

    def _assess_dialogue_structure(self, conversation: Conversation) -> dict[str, Any]:
        """Assess the structural quality of the dialogue."""
        score = 1.0
        issues = []
        details = {}

        messages = conversation.messages

        # Check role alternation
        roles = [msg.role for msg in messages]
        role_alternation_violations = 0

        for i in range(1, len(roles)):
            if roles[i] == roles[i-1]:
                role_alternation_violations += 1

        if role_alternation_violations > 0:
            issues.append(f"Role alternation violations: {role_alternation_violations}")
            score -= role_alternation_violations * 0.2

        # Check for appropriate conversation length
        if len(messages) < 4:
            issues.append("Very short conversation")
            score -= 0.2
        elif len(messages) > 50:
            issues.append("Unusually long conversation")
            score -= 0.1

        # Check for empty or very short messages
        empty_messages = sum(1 for msg in messages if not msg.content.strip())
        very_short_messages = sum(1 for msg in messages if len(msg.content.strip()) < 3)

        if empty_messages > 0:
            issues.append(f"Empty messages found: {empty_messages}")
            score -= empty_messages * 0.2

        if very_short_messages > len(messages) * 0.3:
            issues.append("High proportion of very short messages")
            score -= 0.15

        details.update({
            "role_alternation_violations": role_alternation_violations,
            "empty_messages": empty_messages,
            "very_short_messages": very_short_messages,
            "conversation_length": len(messages)
        })

        return {
            "score": max(0.0, min(1.0, score)),
            "issues": issues,
            "details": details
        }

    def _assess_turn_taking(self, conversation: Conversation) -> dict[str, Any]:
        """Assess turn-taking patterns and conversational dynamics."""
        score = 1.0
        issues = []
        details = {}

        messages = conversation.messages
        roles = [msg.role for msg in messages]

        # Analyze turn-taking patterns
        role_counts = {}
        for role in roles:
            role_counts[role] = role_counts.get(role, 0) + 1

        # Check for balanced participation
        if len(role_counts) > 1:
            max_count = max(role_counts.values())
            min_count = min(role_counts.values())

            if max_count > min_count * 3:
                issues.append("Highly imbalanced participation between roles")
                score -= 0.3
            elif max_count > min_count * 2:
                issues.append("Moderately imbalanced participation between roles")
                score -= 0.2

        # Check for response length imbalance (one role consistently much shorter)
        role_lengths = {}
        for msg in messages:
            if msg.role not in role_lengths:
                role_lengths[msg.role] = []
            role_lengths[msg.role].append(len(msg.content))

        if len(role_lengths) > 1:
            avg_lengths = {role: sum(lengths) / len(lengths) for role, lengths in role_lengths.items()}
            max_avg = max(avg_lengths.values())
            min_avg = min(avg_lengths.values())

            if max_avg > min_avg * 5 and min_avg < 20:  # One role much shorter on average
                issues.append("Significant response length imbalance between roles")
                score -= 0.2

        # Check for appropriate response lengths
        response_lengths = [len(msg.content) for msg in messages]
        if response_lengths:
            avg_length = sum(response_lengths) / len(response_lengths)
            length_variance = sum((length - avg_length) ** 2 for length in response_lengths) / len(response_lengths)

            if length_variance > avg_length ** 2:
                issues.append("High variance in response lengths")
                score -= 0.1

        details.update({
            "role_distribution": role_counts,
            "average_response_length": sum(response_lengths) / len(response_lengths) if response_lengths else 0,
            "response_length_variance": length_variance if response_lengths else 0
        })

        return {
            "score": max(0.0, min(1.0, score)),
            "issues": issues,
            "details": details
        }

    def _assess_topic_continuity(self, conversation: Conversation) -> dict[str, Any]:
        """Assess topic continuity and thematic coherence."""
        score = 1.0
        issues = []
        details = {}

        messages = conversation.messages

        # Simple topic continuity assessment using keyword overlap
        topic_shifts = 0
        keyword_overlaps = []

        for i in range(1, len(messages)):
            prev_words = set(messages[i-1].content.lower().split())
            curr_words = set(messages[i].content.lower().split())

            # Remove common stop words for better topic analysis
            stop_words = {"the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for", "of", "with", "by", "is", "are", "was", "were", "be", "been", "have", "has", "had", "do", "does", "did", "will", "would", "could", "should", "may", "might", "can", "i", "you", "he", "she", "it", "we", "they", "me", "him", "her", "us", "them", "my", "your", "his", "its", "our", "their"}

            prev_content_words = prev_words - stop_words
            curr_content_words = curr_words - stop_words

            if prev_content_words and curr_content_words:
                overlap = len(prev_content_words & curr_content_words)
                total = len(prev_content_words | curr_content_words)
                overlap_ratio = overlap / total if total > 0 else 0
                keyword_overlaps.append(overlap_ratio)

                # Detect potential topic shifts
                if overlap_ratio < 0.1 and len(prev_content_words) > 2 and len(curr_content_words) > 2:
                    topic_shifts += 1

        # Assess topic continuity
        if keyword_overlaps:
            avg_overlap = sum(keyword_overlaps) / len(keyword_overlaps)
            if avg_overlap < 0.15:
                issues.append("Low topic continuity - frequent topic shifts")
                score -= 0.2
            elif avg_overlap < 0.25:
                issues.append("Moderate topic continuity issues")
                score -= 0.1

        if topic_shifts > len(messages) * 0.4:
            issues.append(f"High number of topic shifts: {topic_shifts}")
            score -= 0.15

        details.update({
            "topic_shifts": topic_shifts,
            "average_keyword_overlap": sum(keyword_overlaps) / len(keyword_overlaps) if keyword_overlaps else 0,
            "keyword_overlap_scores": keyword_overlaps
        })

        return {
            "score": max(0.0, min(1.0, score)),
            "issues": issues,
            "details": details
        }

    def _assess_response_relevance(self, conversation: Conversation) -> dict[str, Any]:
        """Assess the relevance of responses to previous messages."""
        score = 1.0
        issues = []
        details = {}

        messages = conversation.messages
        irrelevant_responses = 0

        for i in range(1, len(messages)):
            prev_msg = messages[i-1]
            curr_msg = messages[i]

            # Simple relevance check based on content similarity
            set(prev_msg.content.lower().split())
            set(curr_msg.content.lower().split())

            # Check for question-answer patterns
            any(word in curr_msg.content.lower() for word in ["yes", "no", "maybe", "i think", "i believe", "because"])

            # Check for acknowledgment patterns
            is_acknowledgment = any(word in curr_msg.content.lower() for word in ["okay", "ok", "i see", "understood", "thanks", "thank you"])

            # Very short responses to long questions might be irrelevant
            if len(prev_msg.content) > 100 and len(curr_msg.content) < 20 and not is_acknowledgment:
                irrelevant_responses += 1
                issues.append(f"Potentially irrelevant short response at turn {i}")

            # Generic responses that don't address specific content
            generic_responses = ["ok", "yes", "no", "maybe", "i don't know", "sure", "alright", "i see"]
            if curr_msg.content.lower().strip() in generic_responses and len(prev_msg.content) > 20:
                irrelevant_responses += 1
                issues.append(f"Generic response to detailed message at turn {i}")

            # Check for very short responses that might be irrelevant
            if len(curr_msg.content.strip()) < 5 and len(prev_msg.content) > 30:
                irrelevant_responses += 1
                issues.append(f"Very short response to substantial message at turn {i}")

        if irrelevant_responses > 0:
            score -= irrelevant_responses * 0.2

        details.update({
            "irrelevant_responses": irrelevant_responses,
            "relevance_ratio": 1 - (irrelevant_responses / max(1, len(messages) - 1))
        })

        return {
            "score": max(0.0, min(1.0, score)),
            "issues": issues,
            "details": details
        }

    def _create_minimal_assessment(self, conversation: Conversation, reason: str) -> CoherenceMetrics:
        """Create a minimal assessment for conversations that can't be properly assessed."""
        return CoherenceMetrics(
            overall_score=0.0,
            logical_flow_score=0.0,
            contextual_consistency_score=0.0,
            semantic_coherence_score=0.0,
            dialogue_structure_score=0.0,
            turn_taking_score=0.0,
            topic_continuity_score=0.0,
            response_relevance_score=0.0,
            issues=[reason],
            details={"conversation_id": conversation.id, "message_count": len(conversation.messages)}
        )

    def _determine_quality_level(self, score: float) -> str:
        """Determine quality level based on score."""
        if score >= self.thresholds["excellent"]:
            return "excellent"
        if score >= self.thresholds["good"]:
            return "good"
        if score >= self.thresholds["acceptable"]:
            return "acceptable"
        if score >= self.thresholds["poor"]:
            return "poor"
        return "very_poor"


# Convenience function for backward compatibility
def assess_coherence(conversation: Conversation) -> dict[str, Any]:
    """
    Legacy function for backward compatibility.
    Use ConversationCoherenceAssessor for more comprehensive assessment.
    """
    assessor = ConversationCoherenceAssessor()
    metrics = assessor.assess_conversation_coherence(conversation)

    return {
        "score": metrics.overall_score,
        "issues": metrics.issues,
        "details": metrics.details
    }
