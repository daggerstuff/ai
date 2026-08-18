#!/usr/bin/env python3
"""Clinical validity enhancement pipeline for therapeutic training data.

This pipeline enhances the clinical quality of training data by:
1. Scoring existing data using ClinicalValidityScorer
2. Identifying samples below clinical validity threshold
3. Applying enhancement techniques to improve clinical validity
4. Outputting enhanced training data suitable for therapeutic AI training
"""

import json
import logging
from pathlib import Path
from typing import Any

from .clinical_validity_scorer import ClinicalValidityScorer

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger("clinical_validity_enhancer")


class ClinicalValidityEnhancer:
    """Enhance clinical validity of therapeutic training data.

    This enhancer takes scored data and applies targeted improvements
    to samples that fall below clinical validity thresholds.
    """

    def __init__(self, min_clinical_validity_score: float = 0.6, enhancement_threshold: float = 0.5):
        """Initialize the clinical validity enhancer.

        Args:
            min_clinical_validity_score: Minimum acceptable clinical validity score (0-1)
            enhancement_threshold: Score below which samples get enhanced (0-1)
        """
        self.scorer = ClinicalValidityScorer()
        self.min_clinical_validity_score = min_clinical_validity_score
        self.enhancement_threshold = enhancement_threshold
        logger.info(
            f"Initialized ClinicalValidityEnhancer with "
            f"min_score={min_clinical_validity_score}, "
            f"enhancement_threshold={enhancement_threshold}"
        )

    def load_jsonl(self, file_path: Path) -> list[dict[str, Any]]:
        """Load JSONL file.

        Args:
            file_path: Path to JSONL file

        Returns:
            List of conversation records
        """
        records = []
        if not file_path.exists():
            logger.warning(f"File not found: {file_path}")
            return records

        with open(file_path, encoding="utf-8") as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                    records.append(record)
                except json.JSONDecodeError as e:
                    logger.warning(f"Invalid JSON on line {line_num}: {e}")

        logger.info(f"Loaded {len(records)} records from {file_path}")
        return records

    def save_jsonl(self, records: list[dict[str, Any]], file_path: Path) -> None:
        """Save records to JSONL file.

        Args:
            records: List of conversation records
            file_path: Output file path
        """
        file_path.parent.mkdir(parents=True, exist_ok=True)

        with open(file_path, "w", encoding="utf-8") as f:
            for record in records:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

        logger.info(f"Saved {len(records)} records to {file_path}")

    def score_record(self, record: dict[str, Any]) -> float | None:
        """Score a single record for clinical validity.

        Args:
            record: Conversation record in messages format

        Returns:
            Clinical validity score (0-1) or None if scoring fails
        """
        try:
            # Extract conversation text for scoring
            conversation_text = self._extract_conversation_text(record)
            if not conversation_text:
                return None

            # Score using the clinical validity scorer
            score_result = self.scorer.score(conversation_text)

            if isinstance(score_result, dict) and "overall_score" in score_result:
                return float(score_result["overall_score"])
            if isinstance(score_result, (int, float)):
                return float(score_result)
            logger.warning(f"Unexpected score format: {score_result}")
            return None

        except Exception as e:
            logger.warning(f"Failed to score record: {e}")
            return None

    def _extract_conversation_text(self, record: dict[str, Any]) -> str:
        """Extract plain text conversation from record.

        Args:
            record: Conversation record

        Returns:
            Plain text conversation string
        """
        if "messages" not in record or not isinstance(record["messages"], list):
            return ""

        conversation_parts = []
        for message in record["messages"]:
            if isinstance(message, dict) and "content" in message:
                role = message.get("role", "unknown")
                content = message["content"].strip()
                if content:
                    conversation_parts.append(f"{role}: {content}")

        return "\n".join(conversation_parts)

    def enhance_record(self, record: dict[str, Any], score: float) -> dict[str, Any]:
        """Enhance a record to improve clinical validity.

        Args:
            record: Original conversation record
            score: Current clinical validity score (0-1)

        Returns:
            Enhanced conversation record
        """
        # Create a copy to avoid modifying original
        enhanced_record = record.copy()

        if "messages" in enhanced_record:
            enhanced_messages = []
            for message in enhanced_record["messages"]:
                if isinstance(message, dict):
                    enhanced_message = message.copy()

                    # Apply enhancement techniques based on message role
                    if message.get("role") == "assistant":
                        enhanced_message = self._enhance_therapist_response(enhanced_message, score)
                    elif message.get("role") == "user":
                        enhanced_message = self._enhance_user_message(enhanced_message, score)

                    enhanced_messages.append(enhanced_message)
                else:
                    enhanced_messages.append(message)

            enhanced_record["messages"] = enhanced_messages

        # Add enhancement metadata
        if "metadata" not in enhanced_record:
            enhanced_record["metadata"] = {}

        enhanced_record["metadata"].update(
            {
                "clinical_validity_enhanced": True,
                "original_score": score,
                "enhancement_threshold": self.enhancement_threshold,
                "enhancement_applied": score < self.enhancement_threshold,
            }
        )

        return enhanced_record

    def _enhance_therapist_response(self, message: dict[str, Any], score: float) -> dict[str, Any]:
        """Enhance therapist response to improve clinical validity.

        Args:
            message: Therapist message dictionary
            score: Current clinical validity score

        Returns:
            Enhanced therapist message
        """
        if "content" not in message or not isinstance(message["content"], str):
            return message

        content = message["content"]

        # Apply enhancement strategies based on score deficits
        # Lower scores get more aggressive enhancements

        enhancement_notes = []

        # Strategy 1: Add open-ended questions for exploration
        if score < 0.4 and not content.strip().endswith("?") and len(content) > 10:
            # Add a gentle open-ended question if not already present
            if not any(word in content.lower() for word in ["how", "what", "tell me", "describe", "explore"]):
                content = content.rstrip(".") + ". How does that feel for you?"
                enhancement_notes.append("Added open-ended question")

        # Strategy 2: Add reflective statements
        if score < 0.5 and not any(
            phrase in content.lower()
            for phrase in ["it sounds like", "i hear", "what i'm hearing", "so you're saying", "it seems like"]
        ):
            # Prepend a reflective statement
            content = "It sounds like you're dealing with something important. " + content
            enhancement_notes.append("Added reflective statement")

        # Strategy 3: Normalize and validate experience
        if score < 0.6:
            if not any(phrase in content.lower() for phrase in ["makes sense", "understandable", "normal", "common"]):
                # Add validation
                content = content + " What you're experiencing makes complete sense."
                enhancement_notes.append("Added validation statement")

        # Strategy 4: Encourage elaboration
        if score < 0.3 and len(content.split()) < 15:  # Very brief response
            content = content + " Would you be willing to tell me more about that?"
            enhancement_notes.append("Added invitation to elaborate")

        if enhancement_notes:
            logger.debug(f"Enhanced therapist response: {', '.join(enhancement_notes)}")
            message["content"] = content

        return message

    def _enhance_user_message(self, message: dict[str, Any], score: float) -> dict[str, Any]:
        """Enhance user message to improve clinical validity context.

        Args:
            message: User message dictionary
            score: Current clinical validity score

        Returns:
            Enhanced user message
        """
        # User messages typically don't need enhancement for clinical validity
        # as they represent the client's authentic experience
        # However, we can ensure they're properly formatted

        if "content" not in message or not isinstance(message["content"], str):
            return message

        content = message["content"].strip()
        if not content:
            # Empty messages don't contribute to training
            message["content"] = "I'm not sure how to express this right now."

        return message

    def process_file(self, input_path: Path, output_path: Path, stats_path: Path | None = None) -> dict[str, Any]:
        """Process a JSONL file to enhance clinical validity.

        Args:
            input_path: Path to input JSONL file
            output_path: Path to output JSONL file
            stats_path: Optional path to save processing statistics

        Returns:
            Dictionary with processing statistics
        """
        # Load input data
        records = self.load_jsonl(input_path)
        if not records:
            logger.warning(f"No records to process from {input_path}")
            return {"processed": 0, "enhanced": 0, "errors": 0}

        # Process each record
        enhanced_records = []
        stats = {
            "total_records": len(records),
            "processed": 0,
            "enhanced": 0,
            "scored": 0,
            "errors": 0,
            "score_distribution": {
                "excellent": 0,  # 0.8-1.0
                "good": 0,  # 0.6-0.79
                "fair": 0,  # 0.4-0.59
                "poor": 0,  # 0.0-0.39
            },
        }

        for i, record in enumerate(records):
            try:
                # Score the record
                score = self.score_record(record)

                if score is not None:
                    stats["scored"] += 1

                    if score >= 0.8:
                        stats["score_distribution"]["excellent"] += 1
                    elif score >= 0.6:
                        stats["score_distribution"]["good"] += 1
                    elif score >= 0.4:
                        stats["score_distribution"]["fair"] += 1
                    else:
                        stats["score_distribution"]["poor"] += 1

                    # Enhance if below threshold
                    if score < self.enhancement_threshold:
                        enhanced_record = self.enhance_record(record, score)
                        enhanced_records.append(enhanced_record)
                        stats["enhanced"] += 1
                        logger.debug(f"Enhanced record {i} (score: {score:.2f})")
                    else:
                        enhanced_records.append(record)

                    stats["processed"] += 1
                else:
                    # If we can't score, pass through unchanged
                    enhanced_records.append(record)
                    stats["processed"] += 1
                    logger.debug(f"Could not score record {i}, passing through")

            except Exception as e:
                logger.error(f"Error processing record {i}: {e}")
                stats["errors"] += 1
                # Pass through original record on error
                enhanced_records.append(record)

        # Save enhanced data
        self.save_jsonl(enhanced_records, output_path)

        # Save stats if requested
        if stats_path:
            stats_path.parent.mkdir(parents=True, exist_ok=True)
            with open(stats_path, "w", encoding="utf-8") as f:
                json.dump(stats, f, indent=2)
            logger.info(f"Saved processing statistics to {stats_path}")

        # Log summary
        logger.info(
            f"Processing complete: {stats['processed']} records processed, "
            f"{stats['enhanced']} enhanced ({stats['enhanced'] / max(stats['processed'], 1) * 100:.1f}%)"
        )

        return stats


def main():
    """Command-line interface for clinical validity enhancer."""
    import argparse

    parser = argparse.ArgumentParser(description="Enhance clinical validity of therapeutic training data")
    parser.add_argument("--input", type=str, required=True, help="Input JSONL file path")
    parser.add_argument("--output", type=str, required=True, help="Output JSONL file path")
    parser.add_argument("--stats", type=str, default=None, help="Optional statistics output file path")
    parser.add_argument(
        "--min-score", type=float, default=0.6, help="Minimum acceptable clinical validity score (default: 0.6)"
    )
    parser.add_argument(
        "--enhancement-threshold",
        type=float,
        default=0.5,
        help="Score below which samples get enhanced (default: 0.5)",
    )
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable verbose logging")

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    enhancer = ClinicalValidityEnhancer(
        min_clinical_validity_score=args.min_score, enhancement_threshold=args.enhancement_threshold
    )

    stats = enhancer.process_file(
        input_path=Path(args.input), output_path=Path(args.output), stats_path=Path(args.stats) if args.stats else None
    )

    # Print summary
    if stats["processed"] > 0:
        pass

    for _category, _count in stats["score_distribution"].items():
        pass


if __name__ == "__main__":
    main()
