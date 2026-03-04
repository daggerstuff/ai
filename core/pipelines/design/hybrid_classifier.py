"""
Hybrid Taxonomy Classifier - Phase 2

Combines keyword-based (fast) and LLM-based (accurate) classification for
optimal performance and cost-efficiency.

Strategy:
1. Try keyword-based classification first (fast, free)
2. If confidence is high (≥0.80), use keyword result
3. If confidence is low, fall back to NVIDIA NIM GLM4.7 LLM classification
4. Cache LLM results to avoid redundant API calls
"""

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

from ai.core.pipelines.design.context_detector import ContextDetector
from ai.core.pipelines.design.llm_classifier import (
    LLMClassificationConfig,
    LLMTaxonomyClassifier,
)
from ai.core.pipelines.design.taxonomy_classifier import (
    CategoryClassification,
    TaxonomyClassifier,
    TherapeuticCategory,
)

logger = logging.getLogger(__name__)


@dataclass
class HybridClassificationStats:
    """Statistics for hybrid classification run."""

    total_records: int = 0
    keyword_classified: int = 0
    llm_classified: int = 0
    low_confidence: int = 0
    avg_keyword_confidence: float = 0.0
    avg_llm_confidence: float = 0.0
    avg_overall_confidence: float = 0.0
    categories: Dict[str, int] = None
    llm_api_calls: int = 0
    estimated_cost: float = 0.0  # USD

    def __post_init__(self):
        if self.categories is None:
            self.categories = {cat.value: 0 for cat in TherapeuticCategory}


class HybridTaxonomyClassifier:
    """
    Hybrid classifier combining keyword-based and LLM-based classification.

    Optimizes for both speed and accuracy by using:
    - Keyword-based for clear-cut cases (fast, free)
    - LLM-based for ambiguous cases (slower, costs money)
    """

    def __init__(
        self,
        keyword_confidence_threshold: float = 0.80,
        final_confidence_threshold: float = 0.70,
        llm_config: Optional[LLMClassificationConfig] = None,
        enable_llm: bool = True,
        cache_llm_results: bool = True,
    ):
        """
        Initialize hybrid classifier.

        Args:
            keyword_confidence_threshold: Min confidence to accept keyword result (0-1)
            final_confidence_threshold: Min confidence to classify record (0-1)
            llm_config: Configuration for LLM classifier
            enable_llm: Whether to use LLM for low-confidence cases
            cache_llm_results: Whether to cache LLM results
        """
        self.keyword_classifier = TaxonomyClassifier()
        self.llm_classifier = (
            LLMTaxonomyClassifier(config=llm_config) if enable_llm else None
        )
        self.context_detector = ContextDetector()

        self.keyword_confidence_threshold = keyword_confidence_threshold
        self.final_confidence_threshold = final_confidence_threshold
        self.enable_llm = enable_llm
        self.cache_llm_results = cache_llm_results

        # LLM result cache (text_hash -> classification)
        self.llm_cache: Dict[str, CategoryClassification] = {}

    def _get_text_hash(self, text: str) -> str:
        """Get hash of text for caching."""
        import hashlib

        return hashlib.md5(text.encode()).hexdigest()

    def classify_record(self, record: Dict[str, Any]) -> CategoryClassification:
        """
        Classify a single record using hybrid approach.

        Args:
            record: Conversation record with messages

        Returns:
            Classification result
        """
        # Extract conversation text
        text = self.keyword_classifier._extract_conversation_text(record)

        if not text:
            return CategoryClassification(
                category=TherapeuticCategory.THERAPEUTIC_CONVERSATION,
                confidence=0.30,
                reasoning="Empty conversation",
                keywords_detected=[],
            )

        # Step 1: Check context for educational/theoretical content
        conversation_text = self.keyword_classifier._extract_conversation_text(record)
        context = self.context_detector.detect_context(conversation_text)

        # Step 2: Try keyword-based classification
        keyword_result = self.keyword_classifier.classify_record(record)

        # Step 3: Override if educational/theoretical context detected
        if not context.is_therapeutic and context.confidence >= 0.7:
            # Educational/theoretical - downgrade crisis/trauma even if keywords matched
            if keyword_result.category in [
                TherapeuticCategory.CRISIS_SUPPORT,
                TherapeuticCategory.TRAUMA_PROCESSING,
            ]:
                logger.info(
                    f"Educational context detected - downgrading "
                    f"{keyword_result.category.value}"
                )
                keyword_result = CategoryClassification(
                    category=TherapeuticCategory.THERAPEUTIC_CONVERSATION,
                    confidence=0.50,
                    reasoning=(
                        f"Educational/theoretical context: "
                        f"{', '.join(context.indicators[:3])}"
                    ),
                    keywords_detected=keyword_result.keywords_detected,
                )

        # Step 3b: Detect resolved/past-tense crisis or mental health language
        # "I used to have suicidal thoughts but I'm better now" → NOT crisis
        text_lower = text.lower()
        resolved_patterns = [
            "used to",
            "years ago",
            "in the past",
            "much better now",
            "i'm better now",
            "worked through it",
            "fully processed",
            "no longer",
            "overcame",
            "recovered from",
            "got over",
            "moved past",
            "behind me",
        ]
        active_crisis_patterns = [
            "right now",
            "currently",
            "today",
            "can't take it",
            "need help now",
            "i want to",
            "i'm going to",
        ]
        has_resolved = any(p in text_lower for p in resolved_patterns)
        has_active = any(p in text_lower for p in active_crisis_patterns)

        if has_resolved and not has_active:
            if keyword_result.category in [
                TherapeuticCategory.CRISIS_SUPPORT,
                TherapeuticCategory.MENTAL_HEALTH_SUPPORT,
            ]:
                logger.info(
                    f"Resolved/past context detected - downgrading "
                    f"{keyword_result.category.value} to "
                    f"therapeutic_conversation"
                )
                keyword_result = CategoryClassification(
                    category=(TherapeuticCategory.THERAPEUTIC_CONVERSATION),
                    confidence=0.85,
                    reasoning=(
                        "Resolved/past issue: crisis/mental health "
                        "language present but in past tense context"
                    ),
                    keywords_detected=(keyword_result.keywords_detected),
                )

        # Step 4: If high confidence, use keyword result
        if keyword_result.confidence >= self.keyword_confidence_threshold:
            keyword_result.reasoning = (
                f"Keyword (high confidence): {keyword_result.reasoning}"
            )
            return keyword_result

        # Step 2: Fall back to LLM if enabled
        if self.enable_llm and self.llm_classifier:
            # Check cache first
            text_hash = self._get_text_hash(text)

            if self.cache_llm_results and text_hash in self.llm_cache:
                logger.debug("Using cached LLM result")
                cached_result = self.llm_cache[text_hash]
                cached_result.reasoning = f"LLM (cached): {cached_result.reasoning}"
                return cached_result

            # Get LLM classification
            llm_result = self.llm_classifier.classify(text)

            # Cache result
            if self.cache_llm_results:
                self.llm_cache[text_hash] = llm_result

            return llm_result

        # No LLM available, return keyword result
        keyword_result.reasoning = (
            f"Keyword (low confidence, no LLM): {keyword_result.reasoning}"
        )
        return keyword_result

    def classify_file(
        self, input_path: Path, output_path: Path, max_records: Optional[int] = None
    ) -> HybridClassificationStats:
        """
        Classify all records in a JSONL file using hybrid approach.

        Args:
            input_path: Input JSONL file
            output_path: Output JSONL file
            max_records: Maximum records to process (for testing)

        Returns:
            Classification statistics
        """
        logger.info(f"Hybrid classification: {input_path.name}")

        stats = HybridClassificationStats()

        keyword_confidences = []
        llm_confidences = []
        all_confidences = []

        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(input_path, "r") as infile, open(output_path, "w") as outfile:
            for line_num, line in enumerate(infile, 1):
                if not line.strip():
                    continue

                if max_records and line_num > max_records:
                    logger.info(f"Reached max_records limit: {max_records}")
                    break

                try:
                    record = json.loads(line)
                    stats.total_records += 1

                    # Classify using hybrid approach
                    classification = self.classify_record(record)
                    all_confidences.append(classification.confidence)

                    # Track which method was used
                    if "LLM" in classification.reasoning:
                        stats.llm_classified += 1
                        llm_confidences.append(classification.confidence)
                        stats.llm_api_calls += 1
                    else:
                        stats.keyword_classified += 1
                        keyword_confidences.append(classification.confidence)

                    # Update record metadata
                    if "metadata" not in record:
                        record["metadata"] = {}

                    if classification.confidence >= self.final_confidence_threshold:
                        record["metadata"]["category"] = classification.category.value
                        record["metadata"][
                            "category_confidence"
                        ] = classification.confidence
                        record["metadata"][
                            "category_reasoning"
                        ] = classification.reasoning
                        record["metadata"]["classification_method"] = (
                            "llm" if "LLM" in classification.reasoning else "keyword"
                        )
                        stats.categories[classification.category.value] += 1
                    else:
                        stats.low_confidence += 1
                        record["metadata"]["category"] = "uncategorized"
                        record["metadata"][
                            "category_confidence"
                        ] = classification.confidence

                    # Write updated record
                    outfile.write(json.dumps(record) + "\n")

                    # Progress logging
                    if stats.total_records % 100 == 0:
                        logger.info(
                            f"Processed {stats.total_records} records "
                            f"(KW: {stats.keyword_classified}, LLM: {stats.llm_classified})"
                        )

                except json.JSONDecodeError:
                    logger.warning(f"Invalid JSON at line {line_num}")
                    continue

        # Calculate statistics
        if keyword_confidences:
            stats.avg_keyword_confidence = sum(keyword_confidences) / len(
                keyword_confidences
            )
        if llm_confidences:
            stats.avg_llm_confidence = sum(llm_confidences) / len(llm_confidences)
        if all_confidences:
            stats.avg_overall_confidence = sum(all_confidences) / len(all_confidences)

        # Estimate cost (NVIDIA NIM GLM4.7: ~$0.20/1M tokens for both input and output)
        # Assume ~1000 tokens input, ~100 tokens output per conversation
        input_cost = stats.llm_api_calls * 1000 * 0.20 / 1_000_000
        output_cost = stats.llm_api_calls * 100 * 0.20 / 1_000_000
        stats.estimated_cost = input_cost + output_cost

        self._log_stats(stats)

        return stats

    def _log_stats(self, stats: HybridClassificationStats):
        """Log classification statistics."""
        logger.info("\n" + "=" * 80)
        logger.info("📊 HYBRID CLASSIFICATION RESULTS")
        logger.info("=" * 80)
        logger.info(f"Total records: {stats.total_records:,}")
        logger.info(
            f"Keyword classified: {stats.keyword_classified:,} "
            f"({stats.keyword_classified / stats.total_records * 100:.1f}%)"
        )
        logger.info(
            f"LLM classified: {stats.llm_classified:,} "
            f"({stats.llm_classified / stats.total_records * 100:.1f}%)"
        )
        logger.info(f"Low confidence: {stats.low_confidence:,}")
        logger.info(f"LLM API calls: {stats.llm_api_calls:,}")
        logger.info(f"Estimated cost: ${stats.estimated_cost:.4f} USD")
        logger.info("\nConfidence Averages:")
        logger.info(f"  Keyword: {stats.avg_keyword_confidence:.2%}")
        logger.info(f"  LLM: {stats.avg_llm_confidence:.2%}")
        logger.info(f"  Overall: {stats.avg_overall_confidence:.2%}")
        logger.info("\nCategories:")
        for cat, count in sorted(stats.categories.items(), key=lambda x: -x[1]):
            if count > 0:
                pct = (
                    (count / stats.total_records * 100)
                    if stats.total_records > 0
                    else 0
                )
                logger.info(f"  • {cat}: {count:,} ({pct:.1f}%)")
        logger.info("=" * 80)


def main():
    """Example usage of hybrid classifier."""
    import argparse

    parser = argparse.ArgumentParser(description="Hybrid taxonomy classification")
    parser.add_argument("input", type=Path, help="Input JSONL file")
    parser.add_argument("output", type=Path, help="Output JSONL file")
    parser.add_argument(
        "--keyword-threshold",
        type=float,
        default=0.80,
        help="Keyword confidence threshold (default: 0.80)",
    )
    parser.add_argument(
        "--final-threshold",
        type=float,
        default=0.70,
        help="Final confidence threshold (default: 0.70)",
    )
    parser.add_argument(
        "--max-records", type=int, help="Max records to process (for testing)"
    )
    parser.add_argument("--no-llm", action="store_true", help="Disable LLM fallback")
    parser.add_argument(
        "--model", type=str, default="z-ai/glm4.7", help="NVIDIA NIM model"
    )

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    llm_config = LLMClassificationConfig(model=args.model)

    classifier = HybridTaxonomyClassifier(
        keyword_confidence_threshold=args.keyword_threshold,
        final_confidence_threshold=args.final_threshold,
        llm_config=llm_config,
        enable_llm=not args.no_llm,
    )

    stats = classifier.classify_file(
        args.input, args.output, max_records=args.max_records
    )

    # Print final summary
    print("\n" + "=" * 80)
    print("✅ CLASSIFICATION COMPLETE")
    print("=" * 80)
    print(f"Output written to: {args.output}")
    print(f"Total processed: {stats.total_records:,}")
    print(f"Keyword: {stats.keyword_classified:,} | LLM: {stats.llm_classified:,}")
    print(f"Cost: ${stats.estimated_cost:.4f} USD")
    print("=" * 80)


if __name__ == "__main__":
    main()
