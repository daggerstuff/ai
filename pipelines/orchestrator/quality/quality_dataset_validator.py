"""
Quality Dataset Validation

Specialized validation for quality enhancement datasets with content quality scoring,
conversation coherence analysis, and therapeutic effectiveness assessment.
"""

import json
import re
import statistics
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from dataset_validator import DatasetValidator, ValidationResult
from logger import get_logger

logger = get_logger(__name__)


@dataclass
class QualityValidationResult(ValidationResult):
    """Extended validation result for quality datasets."""

    content_quality_score: float = 0.0
    coherence_score: float = 0.0
    therapeutic_effectiveness_score: float = 0.0
    conversation_length_stats: dict[str, float] = None
    quality_distribution: dict[str, int] = None
    coherence_issues: list[str] = None


class QualityDatasetValidator(DatasetValidator):
    """Specialized validator for quality enhancement datasets."""

    def __init__(self):
        super().__init__()
        self.logger = get_logger(__name__)

        # Quality validation rules
        self.quality_rules = {
            "min_content_quality_score": 0.75,
            "min_coherence_score": 0.8,
            "min_therapeutic_effectiveness": 0.7,
            "min_conversation_length": 50,  # words
            "max_conversation_length": 2000,  # words
            "quality_tiers": ["priority", "research", "archive"],
            "min_priority_percentage": 0.2,  # 20% should be priority quality
        }

        # Quality indicators
        self.quality_indicators = {
            "high_quality_markers": [
                r"\b(?:evidence.based|research.backed|clinically.proven)\b",
                r"\b(?:therapeutic|professional|structured|systematic)\b",
                r"\b(?:empathetic|compassionate|understanding|supportive)\b",
            ],
            "low_quality_markers": [
                r"\b(?:um|uh|like|you know|whatever|dunno)\b",
                r"\b(?:lol|omg|wtf|idk|tbh|smh)\b",
                r"[.]{3,}|[!]{2,}|[?]{2,}",  # Excessive punctuation
            ],
            "coherence_markers": [
                r"\b(?:first|second|third|next|then|finally|therefore)\b",
                r"\b(?:because|since|however|although|moreover)\b",
                r"\b(?:in conclusion|to summarize|overall|in summary)\b",
            ],
            "therapeutic_markers": [
                r"\b(?:validate|acknowledge|reflect|explore|process)\b",
                r"\b(?:coping|healing|growth|insight|awareness)\b",
                r"\b(?:feelings|emotions|thoughts|experiences|perspective)\b",
            ],
        }

        # Conversation quality patterns
        self.conversation_patterns = {
            "question_answer_flow": r"(?:[?].*?[.])|(?:what|how|why|when|where.*?[.])",
            "emotional_support": r"\b(?:understand|feel|sorry|support|here for you)\b",
            "professional_language": r"\b(?:assessment|intervention|treatment|diagnosis|therapy)\b",
            "personal_disclosure": r"\b(?:I feel|I think|I believe|in my experience)\b",
        }

        logger.info("QualityDatasetValidator initialized")

    def validate_quality_dataset(self, dataset_path: str) -> QualityValidationResult:
        """Validate quality dataset with specialized checks."""
        logger.info(f"Validating quality dataset: {dataset_path}")

        # Run base validation
        base_result = self.validate_dataset(dataset_path, "quality")

        # Perform quality specific validation
        content_quality = self._calculate_content_quality_score(dataset_path)
        coherence_score = self._calculate_coherence_score(dataset_path)
        therapeutic_score = self._calculate_therapeutic_effectiveness_score(
            dataset_path
        )
        length_stats = self._analyze_conversation_length_stats(dataset_path)
        quality_dist = self._analyze_quality_distribution(dataset_path)
        coherence_issues = self._detect_coherence_issues(dataset_path)

        # Create extended result
        result = QualityValidationResult(
            dataset_name=base_result.dataset_name,
            is_valid=base_result.is_valid
            and self._meets_quality_requirements(
                content_quality, coherence_score, therapeutic_score, quality_dist
            ),
            file_count=base_result.file_count,
            total_size=base_result.total_size,
            format_compliance=base_result.format_compliance,
            integrity_check=base_result.integrity_check,
            quality_score=base_result.quality_score * content_quality * coherence_score,
            issues=base_result.issues,
            validation_timestamp=base_result.validation_timestamp,
            content_quality_score=content_quality,
            coherence_score=coherence_score,
            therapeutic_effectiveness_score=therapeutic_score,
            conversation_length_stats=length_stats,
            quality_distribution=quality_dist,
            coherence_issues=coherence_issues,
        )

        # Add quality specific issues
        if content_quality < self.quality_rules["min_content_quality_score"]:
            result.issues.append(f"Content quality too low: {content_quality:.2f}")
        if coherence_score < self.quality_rules["min_coherence_score"]:
            result.issues.append(f"Coherence score too low: {coherence_score:.2f}")
        if therapeutic_score < self.quality_rules["min_therapeutic_effectiveness"]:
            result.issues.append(
                f"Therapeutic effectiveness too low: {therapeutic_score:.2f}"
            )
        if coherence_issues:
            result.issues.extend(
                [f"Coherence issue: {issue}" for issue in coherence_issues[:3]]
            )

        logger.info(
            f"Quality validation completed: {'VALID' if result.is_valid else 'INVALID'}"
        )
        return result

    def _calculate_content_quality_score(self, dataset_path: str) -> float:
        """Calculate overall content quality score."""
        try:
            quality_scores = []

            for file_path in Path(dataset_path).rglob("*.json"):
                if file_path.is_file():
                    try:
                        with open(file_path, encoding="utf-8") as f:
                            data = json.load(f)

                        conversations = self._extract_conversations(data)
                        for conversation in conversations:
                            score = self._score_conversation_quality(conversation)
                            quality_scores.append(score)

                    except Exception:
                        continue

            return statistics.mean(quality_scores) if quality_scores else 0.0

        except Exception as e:
            logger.error(f"Content quality scoring failed: {e}")
            return 0.0

    def _calculate_coherence_score(self, dataset_path: str) -> float:
        """Calculate conversation coherence score."""
        try:
            coherence_scores = []

            for file_path in Path(dataset_path).rglob("*.json"):
                if file_path.is_file():
                    try:
                        with open(file_path, encoding="utf-8") as f:
                            data = json.load(f)

                        conversations = self._extract_conversations(data)
                        for conversation in conversations:
                            score = self._score_conversation_coherence(conversation)
                            coherence_scores.append(score)

                    except Exception:
                        continue

            return statistics.mean(coherence_scores) if coherence_scores else 0.0

        except Exception as e:
            logger.error(f"Coherence scoring failed: {e}")
            return 0.0

    def _calculate_therapeutic_effectiveness_score(self, dataset_path: str) -> float:
        """Calculate therapeutic effectiveness score."""
        try:
            therapeutic_scores = []

            for file_path in Path(dataset_path).rglob("*.json"):
                if file_path.is_file():
                    try:
                        with open(file_path, encoding="utf-8") as f:
                            data = json.load(f)

                        conversations = self._extract_conversations(data)
                        for conversation in conversations:
                            score = self._score_therapeutic_effectiveness(conversation)
                            therapeutic_scores.append(score)

                    except Exception:
                        continue

            return statistics.mean(therapeutic_scores) if therapeutic_scores else 0.0

        except Exception as e:
            logger.error(f"Therapeutic effectiveness scoring failed: {e}")
            return 0.0

    def _score_conversation_quality(self, conversation: dict) -> float:
        """Score individual conversation quality."""
        text = self._extract_text_from_conversation(conversation)

        # Initialize score
        quality_score = 0.5  # Base score

        # Check for high quality markers
        high_quality_count = sum(
            len(re.findall(pattern, text, re.IGNORECASE))
            for pattern in self.quality_indicators["high_quality_markers"]
        )

        # Check for low quality markers
        low_quality_count = sum(
            len(re.findall(pattern, text, re.IGNORECASE))
            for pattern in self.quality_indicators["low_quality_markers"]
        )

        # Adjust score based on markers
        quality_score += high_quality_count * 0.1
        quality_score -= low_quality_count * 0.05

        # Check conversation length
        word_count = len(text.split())
        if (
            self.quality_rules["min_conversation_length"]
            <= word_count
            <= self.quality_rules["max_conversation_length"]
        ):
            quality_score += 0.1
        else:
            quality_score -= 0.1

        # Check for proper grammar and structure
        sentence_count = len(re.findall(r"[.!?]+", text))
        if sentence_count > 0:
            avg_sentence_length = word_count / sentence_count
            if 10 <= avg_sentence_length <= 25:  # Good sentence length
                quality_score += 0.1

        return max(0.0, min(1.0, quality_score))

    def _score_conversation_coherence(self, conversation: dict) -> float:
        """Score conversation coherence."""
        text = self._extract_text_from_conversation(conversation)

        # Initialize coherence score
        coherence_score = 0.5

        # Check for coherence markers
        coherence_markers = sum(
            len(re.findall(pattern, text, re.IGNORECASE))
            for pattern in self.quality_indicators["coherence_markers"]
        )

        # Check conversation flow patterns
        flow_patterns = sum(
            len(re.findall(pattern, text, re.IGNORECASE))
            for pattern in self.conversation_patterns.values()
        )

        # Adjust score
        coherence_score += coherence_markers * 0.1
        coherence_score += flow_patterns * 0.05

        # Check for topic consistency (simple heuristic)
        if "messages" in conversation:
            messages = conversation["messages"]
            if len(messages) >= 2:
                # Check if messages relate to each other
                first_half = " ".join(
                    msg.get("content", "") for msg in messages[: len(messages) // 2]
                )
                second_half = " ".join(
                    msg.get("content", "") for msg in messages[len(messages) // 2 :]
                )

                # Simple word overlap check
                first_words = set(first_half.lower().split())
                second_words = set(second_half.lower().split())
                overlap = len(first_words & second_words) / max(
                    len(first_words | second_words), 1
                )

                coherence_score += overlap * 0.2

        return max(0.0, min(1.0, coherence_score))

    def _score_therapeutic_effectiveness(self, conversation: dict) -> float:
        """Score therapeutic effectiveness."""
        text = self._extract_text_from_conversation(conversation)

        # Initialize therapeutic score
        therapeutic_score = 0.3

        # Check for therapeutic markers
        therapeutic_markers = sum(
            len(re.findall(pattern, text, re.IGNORECASE))
            for pattern in self.quality_indicators["therapeutic_markers"]
        )

        # Check for emotional support patterns
        emotional_support = len(
            re.findall(
                self.conversation_patterns["emotional_support"], text, re.IGNORECASE
            )
        )

        # Check for professional language
        professional_language = len(
            re.findall(
                self.conversation_patterns["professional_language"], text, re.IGNORECASE
            )
        )

        # Adjust score
        therapeutic_score += therapeutic_markers * 0.1
        therapeutic_score += emotional_support * 0.15
        therapeutic_score += professional_language * 0.1

        # Check for question-answer therapeutic flow
        qa_flow = len(
            re.findall(
                self.conversation_patterns["question_answer_flow"], text, re.IGNORECASE
            )
        )
        therapeutic_score += qa_flow * 0.05

        return max(0.0, min(1.0, therapeutic_score))

    def _analyze_conversation_length_stats(self, dataset_path: str) -> dict[str, float]:
        """Analyze conversation length statistics."""
        lengths = []

        try:
            for file_path in Path(dataset_path).rglob("*.json"):
                if file_path.is_file():
                    try:
                        with open(file_path, encoding="utf-8") as f:
                            data = json.load(f)

                        conversations = self._extract_conversations(data)
                        for conversation in conversations:
                            text = self._extract_text_from_conversation(conversation)
                            word_count = len(text.split())
                            lengths.append(word_count)

                    except Exception:
                        continue

        except Exception as e:
            logger.error(f"Length statistics analysis failed: {e}")

        if lengths:
            return {
                "mean": statistics.mean(lengths),
                "median": statistics.median(lengths),
                "min": min(lengths),
                "max": max(lengths),
                "std_dev": statistics.stdev(lengths) if len(lengths) > 1 else 0,
            }
        return {"mean": 0, "median": 0, "min": 0, "max": 0, "std_dev": 0}

    def _analyze_quality_distribution(self, dataset_path: str) -> dict[str, int]:
        """Analyze quality tier distribution."""
        quality_counts = {"priority": 0, "research": 0, "archive": 0}

        try:
            for file_path in Path(dataset_path).rglob("*.json"):
                if file_path.is_file():
                    try:
                        with open(file_path, encoding="utf-8") as f:
                            data = json.load(f)

                        conversations = self._extract_conversations(data)
                        for conversation in conversations:
                            quality_score = self._score_conversation_quality(
                                conversation
                            )

                            # Classify into quality tiers
                            if quality_score >= 0.8:
                                quality_counts["priority"] += 1
                            elif quality_score >= 0.6:
                                quality_counts["research"] += 1
                            else:
                                quality_counts["archive"] += 1

                    except Exception:
                        continue

        except Exception as e:
            logger.error(f"Quality distribution analysis failed: {e}")

        return quality_counts

    def _detect_coherence_issues(self, dataset_path: str) -> list[str]:
        """Detect specific coherence issues."""
        issues = []

        try:
            for file_path in Path(dataset_path).rglob("*.json"):
                if file_path.is_file():
                    try:
                        with open(file_path, encoding="utf-8") as f:
                            data = json.load(f)

                        conversations = self._extract_conversations(data)
                        for i, conversation in enumerate(conversations):
                            coherence_score = self._score_conversation_coherence(
                                conversation
                            )

                            if coherence_score < 0.5:
                                issues.append(
                                    f"Low coherence in conversation {i} of {file_path.name}"
                                )

                            # Check for specific issues
                            text = self._extract_text_from_conversation(conversation)
                            if len(text.split()) < 10:
                                issues.append(
                                    f"Very short conversation {i} in {file_path.name}"
                                )

                    except Exception:
                        continue

        except Exception as e:
            logger.error(f"Coherence issue detection failed: {e}")

        return issues[:10]  # Limit to first 10 issues

    def _extract_conversations(self, data) -> list[dict]:
        """Extract conversations from JSON data."""
        conversations = []

        if isinstance(data, list):
            conversations.extend(data)
        elif isinstance(data, dict):
            if "conversations" in data:
                conversations.extend(data["conversations"])
            elif "messages" in data:
                conversations.append(data)
            else:
                conversations.append(data)

        return conversations

    def _extract_text_from_conversation(self, conversation: dict) -> str:
        """Extract text content from a conversation."""
        text_parts = []

        if "messages" in conversation:
            for message in conversation["messages"]:
                if isinstance(message, dict) and "content" in message:
                    text_parts.append(message["content"])
        elif "content" in conversation:
            text_parts.append(conversation["content"])
        elif "text" in conversation:
            text_parts.append(conversation["text"])

        return " ".join(text_parts)

    def _meets_quality_requirements(
        self,
        content_quality: float,
        coherence_score: float,
        therapeutic_score: float,
        quality_dist: dict[str, int],
    ) -> bool:
        """Check if dataset meets quality requirements."""
        total_conversations = sum(quality_dist.values())
        priority_percentage = quality_dist["priority"] / max(total_conversations, 1)

        return (
            content_quality >= self.quality_rules["min_content_quality_score"]
            and coherence_score >= self.quality_rules["min_coherence_score"]
            and therapeutic_score >= self.quality_rules["min_therapeutic_effectiveness"]
            and priority_percentage >= self.quality_rules["min_priority_percentage"]
        )

    def generate_quality_report(
        self,
        results: list[QualityValidationResult],
        output_path: str = "quality_validation_report.json",
    ) -> str:
        """Generate specialized quality validation report."""
        report = {
            "report_type": "Quality Dataset Validation",
            "report_timestamp": datetime.now().isoformat(),
            "summary": {
                "total_datasets": len(results),
                "valid_datasets": sum(1 for r in results if r.is_valid),
                "average_content_quality": (
                    sum(r.content_quality_score for r in results) / len(results)
                    if results
                    else 0
                ),
                "average_coherence_score": (
                    sum(r.coherence_score for r in results) / len(results)
                    if results
                    else 0
                ),
                "average_therapeutic_score": (
                    sum(r.therapeutic_effectiveness_score for r in results)
                    / len(results)
                    if results
                    else 0
                ),
                "total_coherence_issues": sum(
                    len(r.coherence_issues) for r in results if r.coherence_issues
                ),
            },
            "detailed_results": [
                {
                    "dataset_name": r.dataset_name,
                    "is_valid": r.is_valid,
                    "content_quality_score": r.content_quality_score,
                    "coherence_score": r.coherence_score,
                    "therapeutic_effectiveness_score": r.therapeutic_effectiveness_score,
                    "conversation_length_stats": r.conversation_length_stats,
                    "quality_distribution": r.quality_distribution,
                    "coherence_issues": r.coherence_issues,
                    "issues": r.issues,
                }
                for r in results
            ],
        }

        with open(output_path, "w") as f:
            json.dump(report, f, indent=2)

        logger.info(f"Quality validation report generated: {output_path}")
        return output_path


# Example usage
if __name__ == "__main__":
    validator = QualityDatasetValidator()

    # Test validation
    result = validator.validate_quality_dataset("./test_quality_dataset")
