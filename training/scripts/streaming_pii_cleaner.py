#!/usr/bin/env python3
"""
Streaming PII Cleaner for Large Datasets
Processes GB-scale datasets without loading into memory
"""

import json
import logging
import re
import sys
from pathlib import Path
from typing import Dict, Any, Generator, Tuple
import hashlib
from datetime import datetime

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)


class StreamingPIICleaner:
    """
    Memory-efficient PII cleaning for large datasets
    """

    def __init__(self, chunk_size: int = 10000):
        self.chunk_size = chunk_size

        # Comprehensive PII patterns
        self.patterns = {
            "email": re.compile(
                r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b", re.IGNORECASE
            ),
            "phone": re.compile(
                r"\b(?:\+?1[-.\s]?)?(?:\(?\d{3}\)?[-.\s]?)?\d{3}[-.\s]?\d{4}\b",
                re.IGNORECASE,
            ),
            "ssn": re.compile(r"\b\d{3}-\d{2}-\d{4}\b|\b\d{9}\b", re.IGNORECASE),
            "credit_card": re.compile(
                r"\b(?:\d{4}[-\s]?){3}\d{4}\b|\b\d{16}\b", re.IGNORECASE
            ),
            "medical_id": re.compile(
                r"\b(?:patient\s+id|medical\s+record|mrn|chart\s+number)\s*:?\s*\d+\b",
                re.IGNORECASE,
            ),
            "full_name": re.compile(
                r"\b[A-Z][a-z]+\s+[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b", re.IGNORECASE
            ),
            "location": re.compile(
                r"\b(?:\d+\s+[A-Z][a-z]+\s+(?:street|st|avenue|ave|road|rd))\b",
                re.IGNORECASE,
            ),
            "url": re.compile(r"\b(?:https?://|www\.)[^\s]+\b", re.IGNORECASE),
            "ip_address": re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b", re.IGNORECASE),
        }

        self.redaction_placeholders = {
            "email": "[EMAIL]",
            "phone": "[PHONE]",
            "ssn": "[SSN]",
            "credit_card": "[CARD]",
            "medical_id": "[ID]",
            "full_name": "[NAME]",
            "location": "[LOCATION]",
            "url": "[URL]",
            "ip_address": "[IP]",
        }

        self.stats = {
            "total_processed": 0,
            "with_pii": 0,
            "total_pii_removed": 0,
            "pii_types": {},
            "errors": 0,
        }

    def clean_message(self, text: str) -> Tuple[str, Dict[str, int]]:
        """Clean PII from a single message text"""
        if not text or not isinstance(text, str):
            return text, {}

        cleaned = text
        pii_found = {}

        for pii_type, pattern in self.patterns.items():
            matches = pattern.findall(cleaned)
            if matches:
                count = len(matches)
                pii_found[pii_type] = count
                self.stats["total_pii_removed"] += count
                self.stats["pii_types"][pii_type] = (
                    self.stats["pii_types"].get(pii_type, 0) + count
                )
                cleaned = pattern.sub(self.redaction_placeholders[pii_type], cleaned)

        return cleaned, pii_found

    def clean_conversation(
        self, conversation: Dict[str, Any]
    ) -> Tuple[Dict[str, Any], bool]:
        """Clean a single conversation"""
        try:
            cleaned = conversation.copy()
            has_pii = False

            messages = cleaned.get("messages", [])
            cleaned_messages = []

            for message in messages:
                if not isinstance(message, dict):
                    cleaned_messages.append(message)
                    continue

                content = message.get("content", "")
                if isinstance(content, str) and content:
                    cleaned_content, pii_found = self.clean_message(content)
                    if pii_found:
                        has_pii = True

                    new_message = message.copy()
                    new_message["content"] = cleaned_content
                    cleaned_messages.append(new_message)
                else:
                    cleaned_messages.append(message)

            cleaned["messages"] = cleaned_messages

            # Update metadata
            metadata = cleaned.get("metadata", {})
            metadata["pii_cleaned"] = has_pii
            metadata["cleaned_at"] = datetime.now().isoformat()
            cleaned["metadata"] = metadata

            return cleaned, has_pii

        except Exception as e:
            logger.error(f"Error processing conversation: {e}")
            self.stats["errors"] += 1
            return conversation, False

    def process_file_streaming(
        self, input_path: Path, output_path: Path
    ) -> Dict[str, Any]:
        """
        Process large file without loading into memory
        Returns processing statistics
        """
        logger.info(f"Starting streaming PII cleaning: {input_path}")
        logger.info(f"Output: {output_path}")

        processed = 0
        with_pii = 0

        with (
            open(input_path, "r", encoding="utf-8") as infile,
            open(output_path, "w", encoding="utf-8") as outfile,
        ):
            for line_num, line in enumerate(infile, 1):
                if not line.strip():
                    continue

                try:
                    conversation = json.loads(line.strip())
                    cleaned, has_pii = self.clean_conversation(conversation)

                    outfile.write(json.dumps(cleaned, ensure_ascii=False) + "\n")

                    processed += 1
                    if has_pii:
                        with_pii += 1

                    if processed % self.chunk_size == 0:
                        logger.info(f"Processed {processed} conversations...")

                except json.JSONDecodeError as e:
                    logger.warning(f"Skipping invalid JSON on line {line_num}: {e}")
                    self.stats["errors"] += 1
                    continue
                except Exception as e:
                    logger.error(f"Error on line {line_num}: {e}")
                    self.stats["errors"] += 1
                    continue

        self.stats["total_processed"] = processed
        self.stats["with_pii"] = with_pii

        return self.stats

    def validate_cleaning(
        self, input_path: Path, sample_size: int = 100
    ) -> Dict[str, Any]:
        """Validate cleaning quality with spot checks"""
        validation = {
            "sample_size": sample_size,
            "samples_checked": 0,
            "issues": [],
            "validation_passed": True,
        }

        checked = 0
        with open(input_path, "r", encoding="utf-8") as f:
            for line in f:
                if checked >= sample_size:
                    break

                try:
                    conv = json.loads(line)
                    messages = conv.get("messages", [])

                    # Check for remaining PII patterns
                    all_text = " ".join([m.get("content", "") for m in messages])

                    for pii_type, pattern in self.patterns.items():
                        if pattern.search(all_text):
                            validation["issues"].append(
                                f"Found remaining {pii_type} in sample {checked}"
                            )
                            validation["validation_passed"] = False

                    checked += 1

                except Exception as e:
                    validation["issues"].append(f"Validation error: {e}")

        validation["samples_checked"] = checked
        return validation


def main():
    """CLI entry point"""
    import argparse

    parser = argparse.ArgumentParser(
        description="Streaming PII cleaner for large datasets"
    )
    parser.add_argument("--input", required=True, help="Input JSONL file")
    parser.add_argument("--output", required=True, help="Output cleaned JSONL file")
    parser.add_argument(
        "--chunk-size", type=int, default=10000, help="Progress logging interval"
    )
    parser.add_argument(
        "--validate", action="store_true", help="Run validation after processing"
    )

    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)

    if not input_path.exists():
        print(f"Error: Input file {input_path} not found")
        sys.exit(1)

    # Create output directory
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Process the file
    cleaner = StreamingPIICleaner(chunk_size=args.chunk_size)
    stats = cleaner.process_file_streaming(input_path, output_path)

    # Print results
    print("\n" + "=" * 60)
    print("PII CLEANING COMPLETE")
    print("=" * 60)
    print(f"Input file: {input_path}")
    print(f"Output file: {output_path}")
    print(f"Total processed: {stats['total_processed']:,}")
    print(f"With PII removed: {stats['with_pii']:,}")
    print(f"Total PII instances: {stats['total_pii_removed']:,}")
    print(f"Errors: {stats['errors']}")

    if stats["pii_types"]:
        print("\nPII types removed:")
        for pii_type, count in sorted(stats["pii_types"].items()):
            print(f"  {pii_type}: {count:,}")

    # Validation
    if args.validate:
        print("\nRunning validation...")
        validation = cleaner.validate_cleaning(output_path)
        print(
            f"Validation: {'PASSED' if validation['validation_passed'] else 'FAILED'}"
        )
        if validation["issues"]:
            for issue in validation["issues"]:
                print(f"  {issue}")

    # Save stats
    stats_path = output_path.with_suffix(".stats.json")
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)

    print(f"\nStats saved to: {stats_path}")


if __name__ == "__main__":
    main()
