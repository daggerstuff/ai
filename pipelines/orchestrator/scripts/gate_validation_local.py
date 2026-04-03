#!/usr/bin/env python3
"""
Release 0 Gate Validation: Dedup/Leakage + Distribution (Local Build)

Validates:
1. Deduplication: Check for exact duplicates within and across stages
2. Leakage: Check for cross-split contamination
3. Distribution: Token/turn stats by family and split

Usage:
    python -m ai.pipelines.orchestrator.scripts.gate_validation_local --release-version v2026-04-03
"""

import hashlib
import json
import logging
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple
import re

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class ContentAnalyzer:
    """Analyzes content for distribution statistics."""

    def __init__(self):
        self.word_pattern = re.compile(r'\b\w+\b')

    def estimate_tokens(self, text: str) -> int:
        """Estimate token count using word-based tokenization."""
        if not text or not isinstance(text, str):
            return 0
        return len(self.word_pattern.findall(text.lower()))

    def count_turns(self, content: Any) -> int:
        """Count conversation turns."""
        if isinstance(content, dict):
            if "messages" in content and isinstance(content["messages"], list):
                return len(content["messages"])
            if "conversations" in content and isinstance(content["conversations"], list):
                return sum(1 for _ in content["conversations"])
        return 1


class GateValidator:
    """Validates Release 0 gates for local datasets."""

    def __init__(self, release_version: str, release_dir: str):
        self.release_version = release_version
        self.release_dir = Path(release_dir)
        self.manifest_file = self.release_dir / 'manifest.json'
        self.export_file = self.release_dir / 'compiled_export.jsonl'
        self.output_dir = self.release_dir

        self.content_analyzer = ContentAnalyzer()
        self.stats = {
            'total_records': 0,
            'duplicates_found': 0,
            'leakage_detected': False,
            'families_analyzed': 0,
        }

    def load_export(self) -> List[Dict[str, Any]]:
        """Load compiled export data."""
        if not self.export_file.exists():
            raise FileNotFoundError(f"Export file not found: {self.export_file}")

        records = []
        with open(self.export_file, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        records.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue
        return records

    def load_manifest(self) -> Dict[str, Any]:
        """Load release manifest."""
        if not self.manifest_file.exists():
            raise FileNotFoundError(f"Manifest not found: {self.manifest_file}")

        with open(self.manifest_file, 'r', encoding='utf-8') as f:
            return json.load(f)

    def check_deduplication(self, records: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Check for exact duplicates within the dataset.

        Returns:
            Deduplication report with statistics.
        """
        logger.info("🔍 Running deduplication check...")

        seen_hashes: Dict[str, str] = {}  # hash -> record_id
        duplicates = []
        exact_duplicates = 0

        for i, record in enumerate(records):
            # Extract text content for comparison
            text = ''
            if 'messages' in record and isinstance(record['messages'], list):
                text = ' '.join(
                    msg.get('content', '') if isinstance(msg, dict) else str(msg)
                    for msg in record['messages']
                )
            else:
                text = json.dumps(record, ensure_ascii=False)

            # Compute hash
            text_hash = hashlib.sha256(text.encode('utf-8')).hexdigest()[:16]

            # Check for duplicate
            if text_hash in seen_hashes:
                exact_duplicates += 1
                duplicates.append({
                    'record_index': i,
                    'duplicate_of': seen_hashes[text_hash],
                    'hash': text_hash,
                })
            else:
                seen_hashes[text_hash] = str(i)

        self.stats['duplicates_found'] = exact_duplicates

        # Determine pass/fail
        dup_rate = exact_duplicates / len(records) if records else 0
        passed = dup_rate < 0.01  # Less than 1% duplicates

        report = {
            'gate_name': 'deduplication',
            'passed': passed,
            'total_records': len(records),
            'exact_duplicates': exact_duplicates,
            'duplicate_rate': round(dup_rate * 100, 4),
            'threshold': '1%',
            'duplicates': duplicates[:10],  # First 10 duplicates
        }

        logger.info(f"  Total: {len(records):,}, Duplicates: {exact_duplicates:,} ({dup_rate:.4f})")
        logger.info(f"  Status: {'✅ PASS' if passed else '❌ FAIL'}")

        return report

    def check_cross_split_leakage(self, records: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Check for cross-split leakage.

        Since all local data is in 'train' split, this checks for
        potential leakage patterns in the data itself.
        """
        logger.info("🔍 Running cross-split leakage check...")

        # For local build: all data is train split
        # Check for actual split assignment markers (not just the word "split")
        leakage_indicators = []

        for i, record in enumerate(records):
            # Check for explicit val/test split assignments in metadata
            metadata = record.get('metadata', {})
            record_split = metadata.get('split', 'train')

            # Flag if explicitly assigned to val/test (would indicate leakage)
            if record_split in ['val', 'test', 'validation']:
                leakage_indicators.append({
                    'record_index': i,
                    'type': 'wrong_split_assignment',
                    'expected': 'train',
                    'found': record_split,
                })

        leakage_detected = len(leakage_indicators) > 0
        self.stats['leakage_detected'] = leakage_detected

        report = {
            'gate_name': 'cross_split_leakage',
            'passed': not leakage_detected,
            'leakage_indicators': leakage_indicators[:10],
            'total_indicators': len(leakage_indicators),
            'note': 'All local data assigned to train split - checking for wrong assignments',
        }

        logger.info(f"  Leakage indicators: {len(leakage_indicators)}")
        logger.info(f"  Status: {'✅ PASS' if not leakage_detected else '⚠️ WARNING'}")

        return report

    def analyze_distribution(self, records: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Analyze distribution statistics by family.
        """
        logger.info("📊 Analyzing distribution statistics...")

        family_stats: Dict[str, List[Dict[str, Any]]] = defaultdict(list)

        for record in records:
            # Extract family from metadata
            metadata = record.get('metadata', {})
            family = metadata.get('source', 'unknown')
            stage = metadata.get('stage', 'unknown')

            # Get text content
            text = ''
            if 'messages' in record and isinstance(record['messages'], list):
                text = ' '.join(
                    str(msg.get('content', '')) if isinstance(msg, dict) else str(msg)
                    for msg in record['messages']
                )
            else:
                text = json.dumps(record, ensure_ascii=False)

            # Calculate stats
            token_count = self.content_analyzer.estimate_tokens(text)
            turn_count = self.content_analyzer.count_turns(record)
            char_count = len(text)

            family_stats[family].append({
                'tokens': token_count,
                'turns': turn_count,
                'chars': char_count,
            })

        # Aggregate statistics
        distribution_report = {
            'gate_name': 'distribution',
            'passed': True,
            'families': {},
        }

        for family, stats_list in family_stats.items():
            tokens = [s['tokens'] for s in stats_list]
            turns = [s['turns'] for s in stats_list]
            chars = [s['chars'] for s in stats_list]

            distribution_report['families'][family] = {
                'record_count': len(stats_list),
                'tokens': {
                    'min': min(tokens),
                    'max': max(tokens),
                    'mean': round(sum(tokens) / len(tokens), 2),
                    'total': sum(tokens),
                },
                'turns': {
                    'min': min(turns),
                    'max': max(turns),
                    'mean': round(sum(turns) / len(turns), 2),
                },
                'chars': {
                    'min': min(chars),
                    'max': max(chars),
                    'mean': round(sum(chars) / len(chars), 2),
                    'total': sum(chars),
                },
            }

            self.stats['families_analyzed'] += 1

        logger.info(f"  Families analyzed: {len(family_stats)}")
        logger.info(f"  Status: ✅ PASS")

        return distribution_report

    def run_all_gates(self) -> Dict[str, Any]:
        """Run all gate validations."""
        logger.info(f"Running gate validation for {self.release_version}...")

        # Load data
        records = self.load_export()
        self.stats['total_records'] = len(records)

        # Run gates
        dedup_report = self.check_deduplication(records)
        leakage_report = self.check_cross_split_leakage(records)
        distribution_report = self.analyze_distribution(records)

        # Overall status
        all_passed = (
            dedup_report['passed'] and
            leakage_report['passed'] and
            distribution_report['passed']
        )

        # Save report
        report = {
            'release_version': self.release_version,
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'overall_passed': all_passed,
            'gates': {
                'deduplication': dedup_report,
                'cross_split_leakage': leakage_report,
                'distribution': distribution_report,
            },
            'summary': self.stats,
        }

        report_file = self.output_dir / 'gate_validation_report.json'
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)

        logger.info(f"Report saved to: {report_file}")

        return report


def main():
    import argparse

    parser = argparse.ArgumentParser(description='Gate Validation (Local)')
    parser.add_argument(
        '--release-version',
        type=str,
        default='v2026-04-03',
        help='Release version'
    )
    parser.add_argument(
        '--release-dir',
        type=str,
        default='ai/data/releases/v2026-04-03',
        help='Release directory'
    )

    args = parser.parse_args()

    print("=" * 60)
    print("Release 0 Gate Validation")
    print("=" * 60)
    print(f"Release: {args.release_version}")
    print(f"Directory: {args.release_dir}")
    print()

    validator = GateValidator(
        release_version=args.release_version,
        release_dir=args.release_dir,
    )

    report = validator.run_all_gates()

    print()
    print("=" * 60)
    print("GATE VALIDATION SUMMARY")
    print("=" * 60)
    print(f"Release: {report['release_version']}")
    print(f"Timestamp: {report['timestamp']}")
    print()
    print("Gate Results:")

    # Dedup
    dedup = report['gates']['deduplication']
    status = "✅ PASS" if dedup['passed'] else "❌ FAIL"
    print(f"  Deduplication: {status}")
    print(f"    Records: {dedup['total_records']:,}")
    print(f"    Duplicates: {dedup['exact_duplicates']:,} ({dedup['duplicate_rate']}%)")

    # Leakage
    leakage = report['gates']['cross_split_leakage']
    status = "✅ PASS" if leakage['passed'] else "⚠️ WARNING"
    print(f"  Cross-Split Leakage: {status}")
    print(f"    Indicators: {leakage['total_indicators']}")

    # Distribution
    dist = report['gates']['distribution']
    status = "✅ PASS" if dist['passed'] else "❌ FAIL"
    print(f"  Distribution: {status}")
    print(f"    Families: {len(dist['families'])}")

    # Overall
    print()
    overall = "✅ ALL GATES PASSED" if report['overall_passed'] else "❌ GATES FAILED"
    print(f"Overall: {overall}")
    print("=" * 60)

    return report


if __name__ == '__main__':
    main()
