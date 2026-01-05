#!/usr/bin/env python3
"""
Distribution Re

Implements Issue 6: Release 0: Record distribution stats by family and split

This script produces token/turn/length distribution stats by dataset family
and split, with a regression-friendly format.
"""

import json
import re
import statistics
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple

import boto3
from botocore.exceptions import ClientError

# Add the dataset_pipeline to Python path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from storage_config import StorageConfig, get_storage_config


class ContentAnalyzer:
    """Analyzes content for distribution statistics"""

    def __init__(self):
        # Simple tokenization patterns
        self.word_pattern = re.compile(r"\b\w+\b")
        self.sentence_pattern = re.compile(r"[.!?]+")

    def estimate_tokens(self, text: str) -> int:
        """Estimate token count using simple word-based tokenization"""
        if not text or not isinstance(text, str):
            return 0

        # Simple word-based token estimation
        words = self.word_pattern.findall(text.lower())
        return len(words)

    def count_turns(self, content: Any) -> int:
        """Count conversation turns in structured content"""
        turn_count = 0

        def count_turns_recursive(obj):
            nonlocal turn_count

            if isinstance(obj, dict):
                # Look for conversation structure indicators
                if "messages" in obj and isinstance(obj["messages"], list):
                    turn_count += len(obj["messages"])
                elif "conversations" in obj and isinstance(obj["conversations"], list):
                    for conv in obj["conversations"]:
                        count_turns_recursive(conv)
                elif "input" in obj and "output" in obj:
                    turn_count += 2  # Input + output = 2 turns
                elif "human" in obj and "assistant" in obj:
                    turn_count += 2  # Human + assistant = 2 turns
                else:
                    # Recursively check other dict values
                    for value in obj.values():
                        if isinstance(value, (dict, list)):
                            count_turns_recursive(value)
            elif isinstance(obj, list):
                for item in obj:
                    count_turns_recursive(item)

        count_turns_recursive(content)

        # If no structured turns found, estimate from text length
        if turn_count == 0:
            text_content = self.extract_all_text(content)
            if text_content:
                # Rough heuristic: one turn per 100 words
                word_count = len(self.word_pattern.findall(text_content))
                turn_count = max(1, word_count // 100)

        return turn_count

    def extract_all_text(self, content: Any) -> str:
        """Extract all text content from structured data"""
        text_parts = []

        def extract_text_recursive(obj):
            if isinstance(obj, str):
                text_parts.append(obj)
            elif isinstance(obj, dict):
                for value in obj.values():
                    if isinstance(value, (str, dict, list)):
                        extract_text_recursive(value)
            elif isinstance(obj, list):
                for item in obj:
                    extract_text_recursive(item)

        extract_text_recursive(content)
        return " ".join(text_parts)

    def calculate_content_length(self, text: str) -> int:
        """Calculate character length of content"""
        if not text or not isinstance(text, str):
            return 0
        return len(text.strip())

    def analyze_content_sample(self, content_sample: str) -> Dict[str, Any]:
        """Analyze a content sample and return statistics"""
        if not content_sample:
            return {
                "tokens": 0,
                "turns": 0,
                "characters": 0,
                "words": 0,
                "sentences": 0,
            }

        try:
            # Try to parse as JSON first
            parsed_content = json.loads(content_sample)

            # Extract text and analyze structure
            all_text = self.extract_all_text(parsed_content)
            tokens = self.estimate_tokens(all_text)
            turns = self.count_turns(parsed_content)

        except json.JSONDecodeError:
            # Treat as plain text
            all_text = content_sample
            tokens = self.estimate_tokens(all_text)
            turns = 1  # Assume single turn for plain text

        # Calculate additional metrics
        characters = self.calculate_content_length(all_text)
        words = len(self.word_pattern.findall(all_text.lower()))
        sentences = len(self.sentence_pattern.findall(all_text))

        return {
            "tokens": tokens,
            "turns": turns,
            "characters": characters,
            "words": words,
            "sentences": sentences,
        }


class DistributionAnalyzer:
    """Analyzes distribution statistics for dataset families and splits"""

    def __init__(self, storage_config: StorageConfig):
        self.config = storage_config
        self.s3_client = None
        self.content_analyzer = ContentAnalyzer()
        self._init_s3_client()

    def _init_s3_client(self):
        """Initialize S3 client"""
        if self.config.backend != self.config.backend.S3:
            raise ValueError("S3 backend required for distribution analysis")

        try:
            self.s3_client = boto3.client(
                "s3",
                endpoint_url=self.config.s3_endpoint_url,
                aws_access_key_id=self.config.s3_access_key_id,
                aws_secret_access_key=self.config.s3_secret_access_key,
                region_name=self.config.s3_region or "us-east-1",
            )

            self.s3_client.head_bucket(Bucket=self.config.s3_bucket)
            print(f"✓ Connected to S3 bucket: {self.config.s3_bucket}")

        except Exception as e:
            raise ValueError(f"Failed to connect to S3: {e}")

    def load_manifest(self, release_version: str) -> Dict[str, Any]:
        """Load release manifest from S3"""
        manifest_key = (
            f"{self.config.exports_prefix}/releases/{release_version}/manifest.json"
        )

        try:
            response = self.s3_client.get_object(
                Bucket=self.config.s3_bucket, Key=manifest_key
            )
            manifest = json.loads(response["Body"].read())
            print(f"✓ Loaded manifest: {manifest_key}")
            return manifest
        except ClientError as e:
            raise ValueError(f"Failed to load manifest {manifest_key}: {e}")

    def sample_file_content(self, s3_key: str, sample_size: int = 4096) -> str:
        """Sample content from an S3 file for analysis"""
        try:
            # Get sample of file content
            response = self.s3_client.get_object(
                Bucket=self.config.s3_bucket,
                Key=s3_key,
                Range=f"bytes=0-{sample_size - 1}",
            )

            content = response["Body"].read().decode("utf-8", errors="ignore")
            return content

        except ClientError as e:
            print(f"⚠️  Failed to sample {s3_key}: {e}")
            return ""

    def calculate_statistics(self, values: List[float]) -> Dict[str, float]:
        """Calculate statistical measures for a list of values"""
        if not values:
            return {
                "count": 0,
                "mean": 0.0,
                "median": 0.0,
                "std_dev": 0.0,
                "min": 0.0,
                "max": 0.0,
                "p25": 0.0,
                "p75": 0.0,
                "p95": 0.0,
            }

        sorted_values = sorted(values)
        n = len(values)

        return {
            "count": n,
            "mean": statistics.mean(values),
            "median": statistics.median(values),
            "std_dev": statistics.stdev(values) if n > 1 else 0.0,
            "min": min(values),
            "max": max(values),
            "p25": sorted_values[int(n * 0.25)] if n > 0 else 0.0,
            "p75": sorted_values[int(n * 0.75)] if n > 0 else 0.0,
            "p95": sorted_values[int(n * 0.95)] if n > 0 else 0.0,
        }

    def analyze_family_distribution(
        self,
        family_name: str,
        files: List[Dict[str, Any]],
        sample_percentage: float = 0.2,
    ) -> Dict[str, Any]:
        """Analyze distribution statistics for a dataset family"""
        print(f"  Analyzing {family_name}: {len(files)} files")

        # Group files by split
        files_by_split = defaultdict(list)
        for file_info in files:
            split = file_info.get("split", "unknown")
            files_by_split[split].append(file_info)

        family_results = {
            "family": family_name,
            "total_files": len(files),
            "splits": {},
            "overall": {},
        }

        # Collect all metrics for overall statistics
        all_metrics = {
            "tokens": [],
            "turns": [],
            "characters": [],
            "words": [],
            "sentences": [],
        }

        # Analyze each split
        for split, split_files in files_by_split.items():
            print(f"    Analyzing {split}: {len(split_files)} files")

            # Sample files for analysis
            sample_count = max(1, int(len(split_files) * sample_percentage))
            sampled_files = split_files[:sample_count]

            split_metrics = {
                "tokens": [],
                "turns": [],
                "characters": [],
                "words": [],
                "sentences": [],
            }

            # Analyze sampled files
            for file_info in sampled_files:
                s3_key = file_info["key"]

                # Sample file content
                content_sample = self.sample_file_content(s3_key)

                if content_sample:
                    # Analyze content
                    analysis = self.content_analyzer.analyze_content_sample(
                        content_sample
                    )

                    # Collect metrics
                    for metric, value in analysis.items():
                        if metric in split_metrics:
                            split_metrics[metric].append(value)
                            all_metrics[metric].append(value)

            # Calculate statistics for this split
            split_stats = {}
            for metric, values in split_metrics.items():
                split_stats[metric] = self.calculate_statistics(values)

            family_results["splits"][split] = {
                "file_count": len(split_files),
                "sampled_files": len(sampled_files),
                "statistics": split_stats,
            }

        # Calculate overall statistics
        overall_stats = {}
        for metric, values in all_metrics.items():
            overall_stats[metric] = self.calculate_statistics(values)

        family_results["overall"] = {
            "total_sampled": sum(
                len(all_metrics["tokens"]) for _ in [1]
            ),  # Count of all samples
            "statistics": overall_stats,
        }

        return family_results

    def define_expected_ranges(
        self,
    ) -> Dict[str, Dict[str, Dict[str, Tuple[float, float]]]]:
        """Define expected ranges for different dataset families and metrics"""
        return {
            "professional_therapeutic": {
                "tokens": {
                    "mean": (50, 500),  # 50-500 tokens per conversation
                    "max": (100, 2000),  # Max 100-2000 tokens
                },
                "turns": {
                    "mean": (2, 20),  # 2-20 turns per conversation
                    "max": (5, 50),  # Max 5-50 turns
                },
            },
            "cot_reasoning": {
                "tokens": {
                    "mean": (100, 1000),  # Longer reasoning chains
                    "max": (200, 3000),
                },
                "turns": {
                    "mean": (1, 10),  # Usually single reasoning chain
                    "max": (1, 20),
                },
            },
            "edge_cases": {
                "tokens": {
                    "mean": (20, 300),  # Crisis scenarios can be brief
                    "max": (50, 1000),
                },
                "turns": {
                    "mean": (2, 15),  # Crisis conversations
                    "max": (3, 30),
                },
            },
            "voice_persona": {
                "tokens": {
                    "mean": (100, 800),  # Teaching/instructional content
                    "max": (200, 2000),
                },
                "turns": {
                    "mean": (1, 5),  # Often monologue-style
                    "max": (1, 15),
                },
            },
        }

    def check_distribution_thresholds(
        self, family_results: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Check if distribution statistics fall within expected ranges"""
        expected_ranges = self.define_expected_ranges()
        family_name = family_results["family"]

        threshold_results = {
            "family": family_name,
            "within_expected_ranges": True,
            "warnings": [],
            "errors": [],
        }

        # Get expected ranges for this family
        family_ranges = expected_ranges.get(family_name, {})

        if not family_ranges:
            threshold_results["warnings"].append(
                f"No expected ranges defined for family {family_name}"
            )
            return threshold_results

        # Check overall statistics against expected ranges
        overall_stats = family_results["overall"]["statistics"]

        for metric, ranges in family_ranges.items():
            if metric not in overall_stats:
                continue

            metric_stats = overall_stats[metric]

            for stat_type, (min_val, max_val) in ranges.items():
                if stat_type not in metric_stats:
                    continue

                actual_value = metric_stats[stat_type]

                if actual_value < min_val or actual_value > max_val:
                    threshold_results["within_expected_ranges"] = False
                    threshold_results["errors"].append(
                        f"{metric}.{stat_type}: {actual_value:.1f} outside range [{min_val}, {max_val}]"
                    )
                elif actual_value < min_val * 1.2 or actual_value > max_val * 0.8:
                    threshold_results["warnings"].append(
                        f"{metric}.{stat_type}: {actual_value:.1f} near range boundary [{min_val}, {max_val}]"
                    )

        return threshold_results

    def run_distribution_gate(self, release_version: str) -> Dict[str, Any]:
        """Run distribution analysis gate for a release"""
        print(f"📊 Running distribution analysis gate for {release_version}...")

        # Load manifest
        manifest = self.load_manifest(release_version)

        gate_results = {
            "gate_name": "distribution_analysis",
            "passed": True,
            "timestamp": datetime.utcnow().isoformat(),
            "release_version": release_version,
            "family_results": {},
            "summary": {
                "total_families": len(manifest["families"]),
                "families_within_ranges": 0,
                "families_with_warnings": 0,
                "families_with_errors": 0,
            },
        }

        # Analyze each family
        for family_name, family_data in manifest["families"].items():
            files = family_data["files"]

            # Analyze distribution
            family_analysis = self.analyze_family_distribution(family_name, files)

            # Check thresholds
            threshold_check = self.check_distribution_thresholds(family_analysis)

            # Combine results
            family_result = {**family_analysis, "threshold_check": threshold_check}

            gate_results["family_results"][family_name] = family_result

            # Update summary
            if threshold_check["within_expected_ranges"]:
                gate_results["summary"]["families_within_ranges"] += 1
            else:
                gate_results["passed"] = False
                gate_results["summary"]["families_with_errors"] += 1

            if threshold_check["warnings"]:
                gate_results["summary"]["families_with_warnings"] += 1

        return gate_results

    def save_gate_report(
        self, gate_results: Dict[str, Any], release_version: str
    ) -> str:
        """Save gate results to S3"""
        report_key = f"{self.config.exports_prefix}/releases/{release_version}/gates/distribution_report.json"

        try:
            self.s3_client.put_object(
                Bucket=self.config.s3_bucket,
                Key=report_key,
                Body=json.dumps(gate_results, indent=2),
                ContentType="application/json",
            )

            report_url = f"s3://{self.config.s3_bucket}/{report_key}"
            print(f"✓ Distribution report saved: {report_url}")
            return report_url

        except ClientError as e:
            raise ValueError(f"Failed to save distribution report: {e}")

    def print_distribution_summary(self, results: Dict[str, Any]):
        """Print human-readable distribution summary"""
        print("\n" + "=" * 60)
        print("📊 DISTRIBUTION ANALYSIS SUMMARY")
        print("=" * 60)

        print(f"Release Version: {results['release_version']}")
        print(f"Analysis Time: {results['timestamp']}")

        gate_status = "✅ PASSED" if results["passed"] else "❌ FAILED"
        print(f"Gate Status: {gate_status}")

        summary = results["summary"]
        print("\n📈 SUMMARY STATISTICS:")
        print(f"  Total Families: {summary['total_families']}")
        print(f"  Within Expected Ranges: {summary['families_within_ranges']}")
        print(f"  With Warnings: {summary['families_with_warnings']}")
        print(f"  With Errors: {summary['families_with_errors']}")

        print("\n📋 FAMILY DETAILS:")
        for family_name, family_result in results["family_results"].items():
            threshold_check = family_result["threshold_check"]

            if threshold_check["within_expected_ranges"]:
                status_icon = "✅"
            elif threshold_check["errors"]:
                status_icon = "❌"
            else:
                status_icon = "⚠️"

            print(f"  {status_icon} {family_name}")
            print(f"    Files: {family_result['total_files']}")

            # Show key statistics
            overall_stats = family_result["overall"]["statistics"]
            if "tokens" in overall_stats:
                tokens = overall_stats["tokens"]
                print(f"    Tokens: mean={tokens['mean']:.1f}, max={tokens['max']:.1f}")

            if "turns" in overall_stats:
                turns = overall_stats["turns"]
                print(f"    Turns: mean={turns['mean']:.1f}, max={turns['max']:.1f}")

            # Show warnings/errors
            if threshold_check["warnings"]:
                for warning in threshold_check["warnings"][:2]:  # Limit output
                    print(f"    ⚠️  {warning}")

            if threshold_check["errors"]:
                for error in threshold_check["errors"][:2]:  # Limit output
                    print(f"    ❌ {error}")

        if not results["passed"]:
            print("\n🚨 DISTRIBUTION ISSUES DETECTED:")
            print("  Some families have statistics outside expected ranges.")
            print("  Review the full report for detailed analysis.")

        print("\n" + "=" * 60)


def main():
    """Main entry point"""
    print("🚀 Starting Distribution Analysis Gate...")

    # Parse command line arguments
    if len(sys.argv) < 2:
        print("Usage: python distribution_gate.py <release_version>")
        print("Example: python distribution_gate.py v2025-01-02")
        sys.exit(1)

    release_version = sys.argv[1]

    # Load storage configuration
    config = get_storage_config()

    # Validate S3 configuration
    is_valid, error_msg = config.validate()
    if not is_valid:
        print(f"❌ Storage configuration error: {error_msg}")
        sys.exit(1)

    if config.backend != config.backend.S3:
        print("❌ S3 backend required. Set DATASET_STORAGE_BACKEND=s3")
        sys.exit(1)

    try:
        # Create analyzer and run gate
        analyzer = DistributionAnalyzer(config)
        results = analyzer.run_distribution_gate(release_version)

        # Save report
        report_url = analyzer.save_gate_report(results, release_version)

        # Print results
        analyzer.print_distribution_summary(results)

        print(f"\n📄 Full report: {report_url}")

        # Exit with appropriate code
        if results["passed"]:
            print(f"\n✅ Distribution gate passed for {release_version}!")
            sys.exit(0)
        else:
            print(f"\n⚠️  Distribution gate has warnings for {release_version}")
            sys.exit(1)  # Non-blocking warnings

    except Exception as e:
        print(f"❌ Distribution analysis failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
