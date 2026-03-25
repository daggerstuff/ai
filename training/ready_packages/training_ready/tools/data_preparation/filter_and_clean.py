#!/usr/bin/env python3
"""
Data Filtering and Cleaning Pipeline

Filters datasets by quality thresholds per stage, removes PII, duplicates,
and malformed conversations while preserving edge case intensity.
"""

import json
import re
import sys
import hashlib
from pathlib import Path
from typing import Dict, List, Any, Optional, Set
from datetime import datetime
import logging

# Add project root to path for imports
project_root = Path(__file__).parent.parent.parent.parent.parent
sys.path.insert(0, str(project_root))

try:
    from ai.pipelines.orchestrator.schemas.conversation_schema import Conversation, Message
    from ai.pipelines.orchestrator.configs.stages import STAGE1_ID, STAGE2_ID, STAGE3_ID, STAGE4_ID, get_stage_config
except ImportError as e:
    logging.error(f"Failed to import pipeline modules: {e}")
    logging.error(f"Project root: {project_root}")
    sys.exit(1)

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class DataFilter:
    """Filters and cleans datasets"""

    def __init__(self, processing_report_path: Path):
        self.processing_report_path = processing_report_path
        self.processing_report = self._load_processing_report()

        # PII patterns (from unified_preprocessing_pipeline.py)
        self.pii_patterns = [
            re.compile(r"\b\d{3}-\d{3}-\d{4}\b"),  # Phone: 123-456-7890
            re.compile(r"\b\(?\d{3}\)?\s*\d{3}[-.\s]?\d{4}\b"),  # Phone: (123) 456-7890
            re.compile(r"\b\d{9}\b"),  # SSN-like: 123456789
            re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}"),  # Email
            re.compile(r"\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b"),  # Dates that might be SSN
        ]

        # Stage-specific quality thresholds
        self.stage_thresholds = {
            STAGE1_ID: {"min_empathy": 0.55, "min_safety": 0.7},
            STAGE2_ID: {"min_empathy": 0.5, "min_safety": 0.68},
            STAGE3_ID: {"min_empathy": 0.35, "min_safety": 0.55},  # Lower for edge cases
            STAGE4_ID: {"min_empathy": 0.6, "min_safety": 0.75},
        }

        self.seen_hashes: Set[str] = set()
        self.stats = {
            "total_processed": 0,
            "pii_removed": 0,
            "duplicates_removed": 0,
            "malformed_removed": 0,
            "quality_filtered": 0,
            "stage_filtered": {stage: 0 for stage in self.stage_thresholds},
            "final_count": 0,
        }

    def _load_processing_report(self) -> Dict[str, Any]:
        """Load processing report"""
        if not self.processing_report_path.exists():
            return {"datasets": []}
        with open(self.processing_report_path, "r") as f:
            return json.load(f)

    def remove_pii(self, text: str) -> tuple[str, bool]:
        """Remove PII from text, return cleaned text and whether PII was found"""
        cleaned = text
        pii_found = False

        for pattern in self.pii_patterns:
            matches = pattern.findall(cleaned)
            if matches:
                pii_found = True
                # Replace with generic placeholder
                if pattern == self.pii_patterns[3]:  # Email
                    cleaned = pattern.sub("[EMAIL_REDACTED]", cleaned)
                elif pattern in self.pii_patterns[:3]:  # Phone/SSN
                    cleaned = pattern.sub("[PHONE_REDACTED]", cleaned)
                else:
                    cleaned = pattern.sub("[REDACTED]", cleaned)

        return cleaned, pii_found

    def hash_conversation(self, conversation: Dict[str, Any]) -> str:
        """Create hash of conversation content for deduplication"""
        # Extract message content for hashing
        messages = conversation.get("messages", [])
        content_parts = []
        for msg in messages:
            role = msg.get("role", "")
            content = msg.get("content", "")
            content_parts.append(f"{role}:{content}")

        content_str = "|".join(content_parts)
        return hashlib.md5(content_str.encode()).hexdigest()

    def is_duplicate(self, conversation: Dict[str, Any]) -> bool:
        """Check if conversation is a duplicate"""
        conv_hash = self.hash_conversation(conversation)
        if conv_hash in self.seen_hashes:
            return True
        self.seen_hashes.add(conv_hash)
        return False

    def is_malformed(self, conversation: Dict[str, Any]) -> bool:
        """Check if conversation is malformed"""
        # Must have messages
        messages = conversation.get("messages", [])
        if not messages or len(messages) < 2:
            return True

        # Each message must have role and content
        for msg in messages:
            if not isinstance(msg, dict):
                return True
            if "role" not in msg or "content" not in msg:
                return True
            if not isinstance(msg["content"], str) or len(msg["content"].strip()) < 1:
                return True

        return False

    def meets_quality_threshold(self, conversation: Dict[str, Any], stage: str) -> bool:
        """Check if conversation meets quality thresholds for stage"""
        thresholds = self.stage_thresholds.get(stage, self.stage_thresholds[STAGE1_ID])

        metadata = conversation.get("metadata", {})
        empathy = metadata.get("empathy_score", 1.0)  # Default to passing if not present
        safety = metadata.get("safety_score", 1.0)

        # For edge cases (stage 3), allow crisis override
        if stage == STAGE3_ID:
            crisis_intensity = metadata.get("crisis_intensity", "").lower()
            if crisis_intensity in ["high", "very_high"]:
                # Lower safety threshold for high-intensity crisis scenarios
                return safety >= 0.45

        return empathy >= thresholds["min_empathy"] and safety >= thresholds["min_safety"]

    def clean_conversation(self, conversation: Dict[str, Any], stage: str) -> Optional[Dict[str, Any]]:
        """Clean a single conversation"""
        self.stats["total_processed"] += 1

        # Check if malformed
        if self.is_malformed(conversation):
            self.stats["malformed_removed"] += 1
            return None

        # Remove PII from messages
        messages = conversation.get("messages", [])
        pii_found = False
        for msg in messages:
            content = msg.get("content", "")
            cleaned_content, found = self.remove_pii(content)
            if found:
                pii_found = True
                msg["content"] = cleaned_content

        if pii_found:
            self.stats["pii_removed"] += 1

        # Check for duplicates
        if self.is_duplicate(conversation):
            self.stats["duplicates_removed"] += 1
            return None

        # Check quality thresholds
        if not self.meets_quality_threshold(conversation, stage):
            self.stats["quality_filtered"] += 1
            self.stats["stage_filtered"][stage] += 1
            return None

        self.stats["final_count"] += 1
        return conversation

    def filter_dataset(self, input_path: Path, output_path: Path, stage: str) -> Dict[str, Any]:
        """Filter a single dataset file"""
        logger.info(f"Filtering {input_path.name} (stage: {stage})...")

        filtered_conversations = []

        try:
            with open(input_path, "r", encoding="utf-8") as f:
                for line_num, line in enumerate(f, 1):
                    if not line.strip():
                        continue

                    try:
                        conversation = json.loads(line)
                        cleaned = self.clean_conversation(conversation, stage)
                        if cleaned:
                            filtered_conversations.append(cleaned)
                    except json.JSONDecodeError as e:
                        logger.warning(f"Invalid JSON on line {line_num}: {e}")
                        self.stats["malformed_removed"] += 1
                        continue
                    except Exception as e:
                        logger.warning(f"Error processing line {line_num}: {e}")
                        continue

            # Write filtered output
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, "w", encoding="utf-8") as f:
                for conv in filtered_conversations:
                    f.write(json.dumps(conv, ensure_ascii=False) + "\n")

            logger.info(f"  ✅ Filtered: {len(filtered_conversations)}/{self.stats['total_processed']} conversations")

            return {
                "input_path": str(input_path),
                "output_path": str(output_path),
                "stage": stage,
                "input_count": self.stats["total_processed"],
                "output_count": len(filtered_conversations),
                "removed": {
                    "pii": self.stats["pii_removed"],
                    "duplicates": self.stats["duplicates_removed"],
                    "malformed": self.stats["malformed_removed"],
                    "quality": self.stats["quality_filtered"],
                },
            }
        except Exception as e:
            logger.error(f"Error filtering {input_path}: {e}")
            return {
                "input_path": str(input_path),
                "error": str(e),
            }

    def filter_all_datasets(self, output_dir: Path) -> Dict[str, Any]:
        """Filter all datasets from processing report"""
        logger.info("🧹 Starting filtering and cleaning pipeline...")

        results = []

        for dataset in self.processing_report.get("datasets", []):
            input_path = Path(dataset["path"])
            if not input_path.exists():
                logger.warning(f"Input file not found: {input_path}")
                continue

            stage = dataset.get("stage", STAGE1_ID)
            output_path = output_dir / stage / f"{input_path.stem}_filtered.jsonl"

            result = self.filter_dataset(input_path, output_path, stage)
            results.append(result)

        return {
            "timestamp": datetime.now().isoformat(),
            "total_datasets": len(results),
            "stats": self.stats,
            "results": results,
        }


def main():
    """Main function"""
    base_path = Path.cwd()
    processing_report_path = base_path / "ai" / "training_ready" / "scripts" / "output" / "processing_report.json"
    output_dir = base_path / "ai" / "training_ready" / "datasets" / "filtered"

    if not processing_report_path.exists():
        logger.error(f"Processing report not found: {processing_report_path}")
        logger.info("Please run process_all_datasets.py first")
        return 1

    logger.info("🧹 Starting data filtering and cleaning...")

    filterer = DataFilter(processing_report_path)
    report = filterer.filter_all_datasets(output_dir)

    # Save report
    report_path = base_path / "ai" / "training_ready" / "scripts" / "output" / "filtering_report.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)

    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)

    logger.info(f"\n📊 Filtering Summary:")
    logger.info(f"  Total processed: {report['stats']['total_processed']:,}")
    logger.info(f"  PII removed: {report['stats']['pii_removed']:,}")
    logger.info(f"  Duplicates removed: {report['stats']['duplicates_removed']:,}")
    logger.info(f"  Malformed removed: {report['stats']['malformed_removed']:,}")
    logger.info(f"  Quality filtered: {report['stats']['quality_filtered']:,}")
    logger.info(f"  Final count: {report['stats']['final_count']:,}")
    logger.info(f"\n💾 Report saved to: {report_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())

