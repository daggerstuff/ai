#!/usr/bin/env python3
"""
Release 0 Quality Gates Execution Framework
Implements privacy, provenance, deduplication, and PII detection gates
"""

import hashlib
import json
import re
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Set


@dataclass
class GateResult:
    """Result from a quality gate check"""

    gate_name: str
    status: str  # PASS, FAIL, WARNING
    conversations_processed: int
    issues_found: int
    issues: List[Dict[str, Any]]
    execution_time_seconds: float
    timestamp: str


class PIIDetector:
    """Detect and flag PII in conversations"""

    # Common PII patterns
    PATTERNS = {
        "email": r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b",
        "phone": r"\b\d{3}[-.]?\d{3}[-.]?\d{4}\b",
        "ssn": r"\b\d{3}-\d{2}-\d{4}\b",
        "credit_card": r"\b\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}\b",
        "address": (
            r"\b\d+\s+[\w\s]+(?:street|st|avenue|ave|road|rd|lane|ln|drive|dr|"
            r"court|ct|boulevard|blvd)\b"
        ),
        "name": r"\b(?:my name is|I am|I\'m)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)\b",
    }

    def __init__(self, strict_mode: bool = True, confidence_threshold: float = 0.85):
        self.strict_mode = strict_mode
        self.confidence_threshold = confidence_threshold
        self.compiled_patterns = {
            name: re.compile(pattern, re.IGNORECASE)
            for name, pattern in self.PATTERNS.items()
        }

    def scan_conversation(self, conversation: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Scan a conversation for PII"""
        issues = []

        # Extract all text content
        if "messages" in conversation:
            text_content = [msg.get("content", "") for msg in conversation["messages"]]
        elif "prompt" in conversation:
            text_content = [
                conversation.get("prompt", ""),
                conversation.get("response", ""),
            ]
        else:
            text_content = []

        # Scan each piece of content
        for idx, text in enumerate(text_content):
            for pii_type, pattern in self.compiled_patterns.items():
                if matches := pattern.findall(text):
                    issues.append(
                        {
                            "type": "PII_DETECTED",
                            "pii_category": pii_type,
                            "location": f"message_{idx}",
                            "matches": len(matches),
                            "severity": "CRITICAL" if self.strict_mode else "WARNING",
                            "sample": matches[0][:50] if matches else None,
                        }
                    )

        return issues


class ProvenanceValidator:
    """Validate dataset provenance and licensing"""

    REQUIRED_FIELDS = ["source", "dataset_family", "quality"]
    VALID_LICENSES = [
        "CC0",
        "CC-BY",
        "CC-BY-SA",
        "MIT",
        "Apache-2.0",
        "proprietary-approved",
    ]

    def validate_conversation(
        self, conversation: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Validate provenance metadata"""
        issues = []
        metadata = conversation.get("metadata", {})

        # Check required fields
        issues.extend(
            {
                "type": "MISSING_PROVENANCE",
                "field": field,
                "severity": "WARNING",
            }
            for field in self.REQUIRED_FIELDS
            if field not in metadata and field not in conversation
        )

        # Validate license if present
        license_field = metadata.get("license") or conversation.get("license")
        if license_field and license_field not in self.VALID_LICENSES:
            issues.append(
                {
                    "type": "INVALID_LICENSE",
                    "license": license_field,
                    "severity": "WARNING",
                }
            )

        # Check for source URL or citation
        if not metadata.get("source_url") and not conversation.get("source"):
            issues.append({"type": "MISSING_SOURCE", "severity": "WARNING"})

        return issues


class DeduplicationEngine:
    """Detect duplicate conversations"""

    def __init__(self, similarity_threshold: float = 0.9):
        self.similarity_threshold = similarity_threshold
        self.seen_hashes: Set[str] = set()
        self.fuzzy_hashes: Dict[str, List[str]] = {}

    def hash_conversation(self, conversation: Dict[str, Any]) -> str:
        """Generate hash of conversation content"""
        # Extract text content
        if "messages" in conversation:
            text_parts = [msg.get("content", "") for msg in conversation["messages"]]
        elif "prompt" in conversation:
            text_parts = [
                conversation.get("prompt", ""),
                conversation.get("response", ""),
            ]
        else:
            text_parts = []

        # Normalize and hash
        normalized = " ".join(text_parts).lower().strip()
        return hashlib.sha256(normalized.encode()).hexdigest()

    def check_duplicate(self, conversation: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Check if conversation is a duplicate"""
        issues = []
        conv_hash = self.hash_conversation(conversation)

        # Exact duplicate check
        if conv_hash in self.seen_hashes:
            issues.append(
                {"type": "EXACT_DUPLICATE", "hash": conv_hash, "severity": "CRITICAL"}
            )
        else:
            self.seen_hashes.add(conv_hash)

        return issues


class BiasDetector:
    """Detect potential bias in conversations"""

    BIAS_KEYWORDS = {
        "gender": [
            "he always",
            "she always",
            "men are",
            "women are",
            "boys should",
            "girls should",
        ],
        "race": ["all [racial term]", "those people", "they all"],
        "age": ["too old", "too young", "millennials are", "boomers are"],
        "disability": ["handicapped", "retarded", "crazy", "insane", "psycho"],
    }

    def __init__(self, categories: List[str], threshold: float = 0.7):
        self.categories = categories
        self.threshold = threshold

    def scan_conversation(self, conversation: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Scan for bias indicators"""
        issues = []

        # Extract text
        if "messages" in conversation:
            text_content = [msg.get("content", "") for msg in conversation["messages"]]
        else:
            text_content = []

        # Check for bias keywords
        for category in self.categories:
            if category in self.BIAS_KEYWORDS:
                for keyword in self.BIAS_KEYWORDS[category]:
                    if matches := [
                        {
                            "type": "BIAS_DETECTED",
                            "category": category,
                            "keyword": keyword,
                            "location": f"message_{idx}",
                            "severity": "WARNING",
                        }
                        for idx, text in enumerate(text_content)
                        if keyword.lower() in text.lower()
                    ]:
                        issues.extend(matches)

        return issues


class QualityGateRunner:
    """Execute all quality gates on Release 0 datasets"""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.pii_detector = PIIDetector(
            strict_mode=config["quality_gates"]["pii_detection"]["strict_mode"],
            confidence_threshold=config["quality_gates"]["pii_detection"][
                "confidence_threshold"
            ],
        )
        self.provenance_validator = ProvenanceValidator()
        self.deduplication_engine = DeduplicationEngine(
            similarity_threshold=config["quality_gates"]["deduplication"][
                "similarity_threshold"
            ]
        )
        self.bias_detector = BiasDetector(
            categories=config["quality_gates"]["bias_detection"]["categories"],
            threshold=config["quality_gates"]["bias_detection"]["threshold"],
        )

    def run_all_gates(
        self, conversations: List[Dict[str, Any]]
    ) -> Dict[str, GateResult]:
        """Run all quality gates on conversations"""
        results = {}

        # PII Detection
        if self.config["quality_gates"]["pii_detection"]["enabled"]:
            start_time = datetime.now()
            pii_issues = [
                issue
                for conv in conversations
                for issue in self.pii_detector.scan_conversation(conv)
            ]

            results["pii_detection"] = GateResult(
                gate_name="PII Detection",
                status="FAIL" if pii_issues else "PASS",
                conversations_processed=len(conversations),
                issues_found=len(pii_issues),
                issues=pii_issues[:100],  # Limit to first 100
                execution_time_seconds=(datetime.now() - start_time).total_seconds(),
                timestamp=datetime.now(timezone.utc).isoformat(),
            )

        # Provenance Validation
        if self.config["quality_gates"]["provenance_validation"]["enabled"]:
            start_time = datetime.now()
            provenance_issues = [
                issue
                for conv in conversations
                for issue in self.provenance_validator.validate_conversation(conv)
            ]

            results["provenance_validation"] = GateResult(
                gate_name="Provenance Validation",
                status="WARNING" if provenance_issues else "PASS",
                conversations_processed=len(conversations),
                issues_found=len(provenance_issues),
                issues=provenance_issues[:100],
                execution_time_seconds=(datetime.now() - start_time).total_seconds(),
                timestamp=datetime.now(timezone.utc).isoformat(),
            )

        # Deduplication
        if self.config["quality_gates"]["deduplication"]["enabled"]:
            start_time = datetime.now()
            dedup_issues = [
                issue
                for conv in conversations
                for issue in self.deduplication_engine.check_duplicate(conv)
            ]

            results["deduplication"] = GateResult(
                gate_name="Deduplication",
                status="WARNING" if dedup_issues else "PASS",
                conversations_processed=len(conversations),
                issues_found=len(dedup_issues),
                issues=dedup_issues[:100],
                execution_time_seconds=(datetime.now() - start_time).total_seconds(),
                timestamp=datetime.now(timezone.utc).isoformat(),
            )

        # Bias Detection
        if self.config["quality_gates"]["bias_detection"]["enabled"]:
            start_time = datetime.now()
            bias_issues = [
                issue
                for conv in conversations
                for issue in self.bias_detector.scan_conversation(conv)
            ]

            results["bias_detection"] = GateResult(
                gate_name="Bias Detection",
                status="WARNING" if bias_issues else "PASS",
                conversations_processed=len(conversations),
                issues_found=len(bias_issues),
                issues=bias_issues[:100],
                execution_time_seconds=(datetime.now() - start_time).total_seconds(),
                timestamp=datetime.now(timezone.utc).isoformat(),
            )

        return results

    def generate_report(self, results: Dict[str, GateResult], output_path: Path):
        """Generate quality gates report"""
        report = {
            "release": "v2026-01-07",
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "gates": {name: asdict(result) for name, result in results.items()},
            "summary": {
                "total_gates": len(results),
                "passed": len([r for r in results.values() if r.status == "PASS"]),
                "warnings": len([r for r in results.values() if r.status == "WARNING"]),
                "failed": len([r for r in results.values() if r.status == "FAIL"]),
                "total_issues": sum(r.issues_found for r in results.values()),
            },
        }

        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(report, f, indent=2)

        print(f"\n{'=' * 60}")
        print("RELEASE 0 QUALITY GATES REPORT")
        print(f"{'=' * 60}")
        print(f"Total Gates Run: {report['summary']['total_gates']}")
        print(f"✅ Passed: {report['summary']['passed']}")
        print(f"⚠️  Warnings: {report['summary']['warnings']}")
        print(f"❌ Failed: {report['summary']['failed']}")
        print(f"Total Issues Found: {report['summary']['total_issues']}")
        print(f"{'=' * 60}\n")
        print(f"Full report: {output_path}")

        return report


def main():
    """Execute Release 0 quality gates"""
    # Load routing config
    config_path = (
        Path(__file__).parent.parent
        / "training_ready"
        / "config"
        / "release_0_routing_config.json"
    )
    with open(config_path) as f:
        config = json.load(f)

    print("Release 0 Quality Gates Execution Framework")
    print("=" * 60)
    print(f"Configuration: {config_path}")
    print(f"Release: {config['release']}")
    print(f"Families: {config['metadata']['families_included']}")
    print("=" * 60)

    # For demonstration, create sample conversations
    sample_conversations = [
        {
            "messages": [
                {"role": "user", "content": "I feel anxious about my job"},
                {
                    "role": "assistant",
                    "content": (
                        "Can you tell me more about what specifically "
                        "makes you anxious?"
                    ),
                },
            ],
            "metadata": {
                "source": "priority_dataset",
                "dataset_family": "priority",
                "quality": "high",
            },
        }
    ]

    # Run gates
    runner = QualityGateRunner(config)
    results = runner.run_all_gates(sample_conversations)

    # Generate report
    report_path = (
        Path(__file__).parent.parent
        / "training_ready"
        / "reports"
        / "release_0_quality_gates_report.json"
    )
    runner.generate_report(results, report_path)


if __name__ == "__main__":
    main()
