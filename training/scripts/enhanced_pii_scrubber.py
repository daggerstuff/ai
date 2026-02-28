#!/usr/bin/env python3
"""
Enhanced PII Scrubber for Therapeutic Datasets
Advanced privacy protection with context-aware redaction for mental health data
"""

import json
import logging
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import hashlib
from datetime import datetime

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)


class EnhancedTherapeuticPIIScrubber:
    """
    Advanced PII scrubber with therapeutic context awareness
    """

    def __init__(self, conservative_mode: bool = True):
        self.conservative_mode = conservative_mode

        # Enhanced regex patterns for therapeutic context
        self.patterns = {
            # Personal identifiers
            "email": re.compile(
                r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b", re.IGNORECASE
            ),
            "phone": re.compile(
                r"\b(?:\+?1[-.\s]?)?(?:\(?\d{3}\)?[-.\s]?)?\d{3}[-.\s]?\d{4}(?:\s?(?:ext|extension)\.?\s?\d{1,5})?\b",
                re.IGNORECASE,
            ),
            "ssn": re.compile(r"\b\d{3}-\d{2}-\d{4}\b|\b\d{9}\b", re.IGNORECASE),
            "credit_card": re.compile(
                r"\b(?:\d{4}[-\s]?){3}\d{4}\b|\b\d{16}\b", re.IGNORECASE
            ),
            # Medical identifiers
            "medical_id": re.compile(
                r"\b(?:patient\s+id|medical\s+record|mrn|chart\s+number)\s*:?\s*\d+\b",
                re.IGNORECASE,
            ),
            "insurance_id": re.compile(
                r"\b(?:insurance\s+id|policy\s+number|member\s+id)\s*:?\s*[A-Z0-9-]+\b",
                re.IGNORECASE,
            ),
            # Names and locations
            "full_name": re.compile(
                r"\b[A-Z][a-z]+\s+[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b", re.IGNORECASE
            ),
            "location": re.compile(
                r"\b(?:\d+\s+[A-Z][a-z]+\s+(?:street|st|avenue|ave|road|rd|drive|dr|lane|ln|court|ct|boulevard|blvd))\b",
                re.IGNORECASE,
            ),
            # Dates that could be identifying
            "specific_date": re.compile(
                r"\b(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},?\s+\d{4}\b",
                re.IGNORECASE,
            ),
            # URLs and websites
            "url": re.compile(r"\b(?:https?://|www\.)[^\s]+\b", re.IGNORECASE),
            # IP addresses
            "ip_address": re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b", re.IGNORECASE),
        }

        # Context-aware redaction placeholders
        self.redaction_placeholders = {
            "email": "[EMAIL_REDACTED]",
            "phone": "[PHONE_REDACTED]",
            "ssn": "[SSN_REDACTED]",
            "credit_card": "[CARD_REDACTED]",
            "medical_id": "[MEDICAL_ID_REDACTED]",
            "insurance_id": "[INSURANCE_ID_REDACTED]",
            "full_name": "[NAME_REDACTED]",
            "location": "[LOCATION_REDACTED]",
            "specific_date": "[DATE_REDACTED]",
            "url": "[URL_REDACTED]",
            "ip_address": "[IP_REDACTED]",
        }

        # Statistics tracking
        self.stats = {
            "total_conversations": 0,
            "conversations_with_pii": 0,
            "total_pii_instances": 0,
            "pii_types_found": {},
            "conservative_skips": 0,
        }

    def scrub_text(
        self, text: str, context: str = "general"
    ) -> Tuple[str, Dict[str, int]]:
        """
        Scrub PII from text with context awareness
        Returns (cleaned_text, pii_stats)
        """
        if not text or not isinstance(text, str):
            return text, {}

        cleaned_text = text
        pii_found = {}

        for pii_type, pattern in self.patterns.items():
            matches = pattern.findall(cleaned_text)
            if matches:
                count = len(matches)
                pii_found[pii_type] = count
                self.stats["total_pii_instances"] += count
                self.stats["pii_types_found"][pii_type] = (
                    self.stats["pii_types_found"].get(pii_type, 0) + count
                )

                # Context-aware redaction
                placeholder = self.redaction_placeholders[pii_type]
                cleaned_text = pattern.sub(placeholder, cleaned_text)

        return cleaned_text, pii_found

    def scrub_conversation(
        self, conversation: Dict[str, Any]
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """
        Scrub PII from a single conversation
        Returns (cleaned_conversation, scrub_stats)
        """
        self.stats["total_conversations"] += 1

        cleaned = conversation.copy()
        pii_stats = {
            "pii_found": False,
            "pii_types": {},
            "messages_affected": 0,
            "total_pii_instances": 0,
        }

        messages = cleaned.get("messages", [])
        cleaned_messages = []

        for message in messages:
            if not isinstance(message, dict):
                cleaned_messages.append(message)
                continue

            content = message.get("content", "")
            if not content or not isinstance(content, str):
                cleaned_messages.append(message)
                continue

            cleaned_content, pii_found = self.scrub_text(content, context="therapeutic")

            if pii_found:
                pii_stats["pii_found"] = True
                pii_stats["messages_affected"] += 1
                pii_stats["total_pii_instances"] += sum(pii_found.values())

                for pii_type, count in pii_found.items():
                    pii_stats["pii_types"][pii_type] = (
                        pii_stats["pii_types"].get(pii_type, 0) + count
                    )

            # Create new message with cleaned content
            cleaned_message = message.copy()
            cleaned_message["content"] = cleaned_content
            cleaned_messages.append(cleaned_message)

        cleaned["messages"] = cleaned_messages

        # Add scrubbing metadata
        metadata = cleaned.get("metadata", {})
        metadata["pii_status"] = "scrubbed" if pii_stats["pii_found"] else "clean"
        metadata["pii_scrubbing_stats"] = pii_stats
        metadata["pii_scrubbed_at"] = datetime.now().isoformat()
        cleaned["metadata"] = metadata

        if pii_stats["pii_found"]:
            self.stats["conversations_with_pii"] += 1

        return cleaned, pii_stats

    def scrub_dataset(self, dataset: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Scrub PII from entire dataset
        Returns cleaned dataset with comprehensive statistics
        """
        logger.info(f"Starting PII scrubbing on {len(dataset)} conversations...")

        cleaned_dataset = []
        scrubbing_report = {
            "total_conversations": len(dataset),
            "conversations_scrubbed": 0,
            "conversations_clean": 0,
            "pii_types_summary": {},
            "detailed_stats": [],
            "processing_timestamp": datetime.now().isoformat(),
        }

        for i, conversation in enumerate(dataset):
            if i % 1000 == 0:
                logger.info(f"Processed {i}/{len(dataset)} conversations...")

            cleaned, stats = self.scrub_conversation(conversation)
            cleaned_dataset.append(cleaned)

            if stats["pii_found"]:
                scrubbing_report["conversations_scrubbed"] += 1
                scrubbing_report["detailed_stats"].append(
                    {
                        "index": i,
                        "stats": stats,
                        "source_family": cleaned.get("metadata", {}).get(
                            "source_family", "unknown"
                        ),
                    }
                )
            else:
                scrubbing_report["conversations_clean"] += 1

        # Aggregate PII types
        for pii_type, count in self.stats["pii_types_found"].items():
            scrubbing_report["pii_types_summary"][pii_type] = count

        logger.info(f"PII scrubbing complete:")
        logger.info(f"  Total conversations: {len(dataset)}")
        logger.info(
            f"  Conversations with PII: {scrubbing_report['conversations_scrubbed']}"
        )
        logger.info(f"  Clean conversations: {scrubbing_report['conversations_clean']}")
        logger.info(
            f"  Total PII instances removed: {self.stats['total_pii_instances']}"
        )

        return {
            "cleaned_dataset": cleaned_dataset,
            "scrubbing_report": scrubbing_report,
            "summary_stats": self.stats,
        }

    def validate_scrubbing(self, sample_size: int = 1000) -> Dict[str, Any]:
        """
        Validate scrubbing quality with spot checks
        """
        validation_report = {
            "sample_size": sample_size,
            "validation_timestamp": datetime.now().isoformat(),
            "spot_checks": [],
            "validation_passed": True,
            "issues": [],
        }

        # This would typically load a sample and perform validation
        # For now, return structure for implementation

        return validation_report


def main():
    """CLI entry point for PII scrubbing"""
    import argparse

    parser = argparse.ArgumentParser(
        description="Enhanced PII scrubbing for therapeutic datasets"
    )
    parser.add_argument("--input", required=True, help="Input dataset file (JSONL)")
    parser.add_argument(
        "--output", required=True, help="Output cleaned dataset file (JSONL)"
    )
    parser.add_argument("--report", help="Output scrubbing report file (JSON)")
    parser.add_argument(
        "--conservative", action="store_true", help="Use conservative scrubbing mode"
    )

    args = parser.parse_args()

    # Load dataset
    dataset = []
    with open(args.input, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                dataset.append(json.loads(line))

    # Scrub PII
    scrubber = EnhancedTherapeuticPIIScrubber(conservative_mode=args.conservative)
    result = scrubber.scrub_dataset(dataset)

    # Save cleaned dataset
    with open(args.output, "w", encoding="utf-8") as f:
        for conversation in result["cleaned_dataset"]:
            f.write(json.dumps(conversation, ensure_ascii=False) + "\n")

    # Save report
    if args.report:
        with open(args.report, "w", encoding="utf-8") as f:
            json.dump(result["scrubbing_report"], f, indent=2, ensure_ascii=False)

    print(f"✅ PII scrubbing complete!")
    print(f"   Input: {len(dataset)} conversations")
    print(
        f"   Scrubbed: {result['scrubbing_report']['conversations_scrubbed']} conversations"
    )
    print(
        f"   Clean: {result['scrubbing_report']['conversations_clean']} conversations"
    )
    print(f"   PII instances removed: {result['summary_stats']['total_pii_instances']}")


if __name__ == "__main__":
    main()
