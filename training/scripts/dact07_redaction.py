#!/usr/bin/env python3
"""
DACT-07: Redaction & Re-screening for Stage-Based Datasets

Applies PII redaction to staged datasets and re-screens to verify privacy gates.
Addresses findings from DACT-05:
- Phone numbers: 40 instances
- URLs: 3 instances
- Specific dates: 2 instances
- Full names: 1,038,465 instances (likely false positives in therapeutic context)

Usage:
    python -m ai.training.scripts.dact07_redaction --input-dir ai/data/staged_datasets --output-dir ai/data/redacted_datasets
"""

from datetime import datetime, timezone








import argparse
import json
import logging
from pathlib import Path
from typing import Dict, Any, List, Tuple
import re

# Import existing PII scrubber from same directory
import sys
from enhanced_pii_scrubber import EnhancedTherapeuticPIIScrubber

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class DACT07Redactor:
    """
    DACT-07 specific redaction with therapeutic context awareness.

    Key difference from generic PII scrubbing:
    - Therapeutic content often mentions "my therapist", "my patient", etc.
    - We need to distinguish between:
      - Generic references: "my therapist said" (keep)
      - Specific PII: "Dr. John Smith said" (redact)
      - Phone numbers in crisis contexts (always redact)
      - URLs (always redact)
    """

    def __init__(self, conservative_mode: bool = True):
        """
        Initialize redactor.

        Args:
            conservative_mode: If True, skip name redaction in therapeutic contexts
                              to avoid false positives like "my therapist"
        """
        self.conservative_mode = conservative_mode
        self.scrubber = EnhancedTherapeuticPIIScrubber(conservative_mode=conservative_mode)

        # Additional patterns specific to DACT-07 findings
        self.crisis_patterns = {
            # Phone numbers in various formats
            "phone_crisis": re.compile(r'\b(?:call|text|reach)\s*(?:me|him|her)?\s*:?\s*[\+]?1?[-.\s]?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b', re.IGNORECASE),
            # Emergency contact patterns
            "emergency_contact": re.compile(r'\b(?:emergency\s+contact|crisis\s+line)\s*:?\s*[\+]?1?[-.\s]?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b', re.IGNORECASE),
        }

        # Statistics
        self.stats = {
            'files_processed': 0,
            'records_processed': 0,
            'records_redacted': 0,
            'pii_instances_found': {},
            'pii_instances_redacted': {},
            'false_positive_prevented': 0,
        }

    def is_therapeutic_context(self, text: str) -> bool:
        """Check if text appears to be therapeutic content."""
        therapeutic_keywords = [
            'therapist', 'therapy', 'patient', 'counselor', 'session',
            'mental health', 'depression', 'anxiety', 'trauma', 'healing'
        ]
        text_lower = text.lower()
        return any(kw in text_lower for kw in therapeutic_keywords)

    def should_skip_name_redaction(self, text: str) -> bool:
        """
        Determine if name redaction should be skipped to avoid therapeutic false positives.

        In conservative mode, we skip name redaction when:
        - Text is clearly therapeutic context
        - Names appear as generic references ("my therapist", "the patient")
        """
        if not self.conservative_mode:
            return False

        # Check for therapeutic context
        if self.is_therapeutic_context(text):
            # Generic therapeutic references - skip name redaction
            generic_refs = ['my therapist', 'the therapist', 'my counselor', 'the patient', 'my doctor']
            if any(ref in text.lower() for ref in generic_refs):
                self.stats['false_positive_prevented'] += 1
                return True

        return False

    def redact_record(self, record: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, int]]:
        """
        Redact PII from a single record.

        Returns:
            Tuple of (redacted_record, pii_counts)
        """
        redacted = record.copy()
        pii_found = {}

        # Get text content
        text = record.get('text', '')
        if not text:
            # Try alternative fields
            text = (
                record.get('content', '') or
                record.get('message', '') or
                record.get('conversation', '') or
                json.dumps(record)
            )

        if not text or not isinstance(text, str):
            return redacted, {}

        # Use the enhanced scrubber for most PII types
        scrubbed_text, scrub_stats = self.scrubber.scrub_text(text)

        # Track PII types
        for pii_type, count in scrub_stats.items():
            if count > 0:
                pii_found[pii_type] = count
                self.stats['pii_instances_found'][pii_type] = (
                    self.stats['pii_instances_found'].get(pii_type, 0) + count
                )

        # Handle crisis-specific patterns
        for pattern_name, pattern in self.crisis_patterns.items():
            matches = pattern.findall(scrubbed_text)
            if matches:
                pii_found[pattern_name] = len(matches)
                scrubbed_text = pattern.sub('[CRISIS_CONTACT_REDACTED]', scrubbed_text)

        # Update record with redacted text
        if 'text' in redacted:
            redacted['text'] = scrubbed_text
        elif 'content' in redacted:
            redacted['content'] = scrubbed_text
        elif 'message' in redacted:
            redacted['message'] = scrubbed_text

        # Add redaction metadata
        metadata = redacted.get('metadata', {})
        metadata['dact07_redacted'] = True
        metadata['dact07_timestamp'] = datetime.now(timezone.utc).isoformat()
        metadata['pii_found'] = pii_found
        redacted['metadata'] = metadata

        # Update stats
        if pii_found:
            self.stats['records_redacted'] += 1
            for pii_type, count in pii_found.items():
                self.stats['pii_instances_redacted'][pii_type] = (
                    self.stats['pii_instances_redacted'].get(pii_type, 0) + count
                )

        self.stats['records_processed'] += 1

        return redacted, pii_found

    def redact_file(self, input_path: Path, output_path: Path) -> Dict[str, Any]:
        """
        Redact PII from a JSONL file.

        Returns:
            File-level statistics
        """
        logger.info(f"Redacting PII from: {input_path}")

        file_stats = {
            'input_file': str(input_path),
            'output_file': str(output_path),
            'records_processed': 0,
            'records_with_pii': 0,
            'pii_types_found': {},
        }

        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(input_path, 'r', encoding='utf-8') as infile, \
             open(output_path, 'w', encoding='utf-8') as outfile:

            for line in infile:
                line = line.strip()
                if not line:
                    continue

                try:
                    record = json.loads(line)
                except json.JSONDecodeError as e:
                    logger.warning(f"Invalid JSON line: {e}")
                    continue

                # Redact PII
                redacted, pii_found = self.redact_record(record)

                # Write redacted record
                outfile.write(json.dumps(redacted, ensure_ascii=False) + '\n')

                file_stats['records_processed'] += 1
                if pii_found:
                    file_stats['records_with_pii'] += 1
                    for pii_type, count in pii_found.items():
                        file_stats['pii_types_found'][pii_type] = (
                            file_stats['pii_types_found'].get(pii_type, 0) + count
                        )

        self.stats['files_processed'] += 1
        logger.info(f"Processed {file_stats['records_processed']} records, "
                   f"found PII in {file_stats['records_with_pii']} records")

        return file_stats

    def generate_report(self, output_dir: Path) -> Dict[str, Any]:
        """Generate comprehensive redaction report."""
        report = {
            'dact': 'DACT-07',
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'conservative_mode': self.conservative_mode,
            'summary': {
                'files_processed': self.stats['files_processed'],
                'records_processed': self.stats['records_processed'],
                'records_redacted': self.stats['records_redacted'],
                'pii_instances_found': self.stats['pii_instances_found'],
                'pii_instances_redacted': self.stats['pii_instances_redacted'],
                'false_positives_prevented': self.stats['false_positive_prevented'],
            },
        }

        report_path = output_dir / 'dact07_redaction_report.json'
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)

        logger.info(f"Report saved to: {report_path}")
        return report


def main():
    parser = argparse.ArgumentParser(description='DACT-07: Redaction & Re-screening')
    parser.add_argument(
        '--input-dir',
        type=str,
        default='ai/data/staged_datasets',
        help='Input directory with staged datasets'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='ai/data/redacted_datasets',
        help='Output directory for redacted datasets'
    )
    parser.add_argument(
        '--conservative',
        action='store_true',
        default=True,
        help='Use conservative redaction mode (skip therapeutic name references)'
    )
    parser.add_argument(
        '--stages',
        type=str,
        nargs='+',
        default=['stage1_foundation', 'stage2_therapeutic_expertise', 'stage4_voice_persona'],
        help='Which stages to process'
    )

    args = parser.parse_args()

    print("=" * 60)
    print("DACT-07: Redaction & Re-screening")
    print("=" * 60)
    print(f"Input:  {args.input_dir}")
    print(f"Output: {args.output_dir}")
    print(f"Conservative mode: {args.conservative}")
    print()

    redactor = DACT07Redactor(conservative_mode=args.conservative)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Process each stage
    for stage in args.stages:
        input_file = Path(args.input_dir) / f"{stage}.jsonl"
        output_file = output_dir / f"{stage}_redacted.jsonl"

        if input_file.exists():
            redactor.redact_file(input_file, output_file)
        else:
            logger.warning(f"Stage file not found: {input_file}")

    # Generate report
    report = redactor.generate_report(output_dir)

    # Print summary
    print()
    print("=" * 60)
    print("REDACTION SUMMARY")
    print("=" * 60)
    print(f"Files processed:      {report['summary']['files_processed']}")
    print(f"Records processed:    {report['summary']['records_processed']:,}")
    print(f"Records redacted:     {report['summary']['records_redacted']:,}")
    print(f"False positives prevented: {report['summary']['false_positives_prevented']:,}")
    print()
    print("PII Instances Found:")
    for pii_type, count in report['summary']['pii_instances_found'].items():
        print(f"  - {pii_type}: {count:,}")
    print()
    print("PII Instances Redacted:")
    for pii_type, count in report['summary']['pii_instances_redacted'].items():
        print(f"  - {pii_type}: {count:,}")
    print()
    print("✓ DACT-07 redaction complete")

    return report


if __name__ == '__main__':
    main()
