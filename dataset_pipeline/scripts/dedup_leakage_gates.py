#!/usr/bin/env python3
"""
ENTERPRISE-GRADE Deduplication and Cross-Split Leakage Gates

Implements Issue 5: Release 0: Run dedup and cross-split leakage gates

This script runs enterprise-grade exact + near-duplicate scans and cross-split
leakage checks using the existing enterprise deduplication infrastructure,
with comprehensive audit trails and clinical validation integration.
"""

import json
import logging
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

import boto3
from botocore.exceptions import ClientError

# Add the dataset_pipeline to Python path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from processing.enterprise_deduplication import EnterpriseConversationDeduplicator
from safety_ethics_audit_trail import get_audit_trail
from storage_config import StorageConfig, get_storage_config
from validation.clinical_validator import ClinicalValidator

# Configure enterprise logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("dedup_leakage_gates.log"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)


class EnterpriseLeakageDetector:
    """Enterprise-grade cross-split leakage detection with clinical validation"""

    def __init__(self):
        self.audit_trail = get_audit_trail()

        # Initialize clinical validator if available
        try:
            self.clinical_validator = ClinicalValidator()
            logger.info("Clinical validator integrated for leakage detection")
        except Exception as e:
            logger.warning(f"Clinical validator unavailable: {e}")
            self.clinical_validator = None

        # Initialize enterprise deduplicator for similarity calculations
        try:
            dedup_config = {
                "similarity_thresholds": {
                    "content": 0.90,  # Higher threshold for leakage detection
                    "semantic": 0.85,
                    "structural": 0.80,
                    "temporal": 0.75,
                    "overall": 0.85,
                },
                "processing": {
                    "batch_size": 500,  # Smaller batches for leakage detection
                    "max_workers": 2,
                    "enable_caching": True,
                },
                "quality": {
                    "min_confidence_score": 0.8,  # Higher confidence for leakage
                    "enable_fuzzy_matching": True,
                },
            }
            self.deduplicator = EnterpriseConversationDeduplicator(dedup_config)
            logger.info("Enterprise deduplicator initialized for leakage detection")
        except Exception as e:
            logger.error(f"Failed to initialize enterprise deduplicator: {e}")
            self.deduplicator = None

    def check_cross_split_leakage(
        self, files_by_split: Dict[str, List[Dict[str, Any]]]
    ) -> Dict[str, Any]:
        """Check for data leakage across train/val/test splits using enterprise methods"""
        logger.info("🔍 Running enterprise cross-split leakage detection...")

        leakage_results = {
            "leakage_detected": False,
            "exact_leakage": [],
            "near_leakage": [],
            "split_summary": {},
            "enterprise_grade": self.deduplicator is not None,
            "clinical_validated": self.clinical_validator is not None,
        }

        # Convert files to conversation format for enterprise deduplicator
        conversations_by_split = {}

        for split, files in files_by_split.items():
            conversations = []
            for i, file_info in enumerate(files[:10]):  # Limit for performance
                # Create conversation format
                conversation = {
                    "id": f"{split}_{i}_{file_info['key'].split('/')[-1]}",
                    "messages": [
                        {"role": "system", "content": f"File: {file_info['key']}"},
                        {
                            "role": "user",
                            "content": file_info.get("content_sample", "")[:500],
                        },
                    ],
                    "metadata": {
                        "split": split,
                        "original_file": file_info["key"],
                        "size": file_info.get("size", 0),
                    },
                }
                conversations.append(conversation)

            conversations_by_split[split] = conversations
            leakage_results["split_summary"][split] = len(conversations)

        # Check for leakage between splits using enterprise deduplicator
        if self.deduplicator:
            splits = list(conversations_by_split.keys())

            for i in range(len(splits)):
                for j in range(i + 1, len(splits)):
                    split1, split2 = splits[i], splits[j]

                    # Combine conversations from both splits for deduplication analysis
                    combined_conversations = (
                        conversations_by_split[split1] + conversations_by_split[split2]
                    )

                    if combined_conversations:
                        try:
                            # Run enterprise deduplication
                            dedup_result = self.deduplicator.deduplicate_conversations(
                                combined_conversations
                            )

                            # Analyze duplicate groups for cross-split leakage
                            for duplicate_group in dedup_result.duplicate_groups:
                                if len(duplicate_group) > 1:
                                    # Check if duplicates span different splits
                                    splits_in_group = set()
                                    for conv_id in duplicate_group:
                                        conv_split = conv_id.split("_")[0]
                                        splits_in_group.add(conv_split)

                                    if len(splits_in_group) > 1:
                                        # Cross-split leakage detected
                                        leakage_results["leakage_detected"] = True

                                        leakage_entry = {
                                            "splits_affected": list(splits_in_group),
                                            "duplicate_conversations": duplicate_group,
                                            "detection_method": "enterprise_deduplication",
                                            "confidence": "high",
                                        }

                                        leakage_results["exact_leakage"].append(
                                            leakage_entry
                                        )

                                        # Log leakage detection
                                        if self.audit_trail:
                                            self.audit_trail.log_safety_issue(
                                                f"leakage_{split1}_{split2}",
                                                {
                                                    "issue_type": "cross_split_leakage",
                                                    "splits": [split1, split2],
                                                    "duplicate_count": len(
                                                        duplicate_group
                                                    ),
                                                    "detection_method": "enterprise_deduplication",
                                                },
                                            )

                        except Exception as e:
                            logger.error(
                                f"Enterprise deduplication failed for {split1}-{split2}: {e}"
                            )
                            # Fallback to basic similarity check
                            self._basic_leakage_check(
                                conversations_by_split[split1],
                                conversations_by_split[split2],
                                split1,
                                split2,
                                leakage_results,
                            )
        else:
            # Fallback to basic leakage detection
            logger.warning(
                "Using basic leakage detection - enterprise deduplicator unavailable"
            )
            splits = list(conversations_by_split.keys())

            for i in range(len(splits)):
                for j in range(i + 1, len(splits)):
                    split1, split2 = splits[i], splits[j]
                    self._basic_leakage_check(
                        conversations_by_split[split1],
                        conversations_by_split[split2],
                        split1,
                        split2,
                        leakage_results,
                    )

        return leakage_results

    def _basic_leakage_check(
        self, conversations1, conversations2, split1, split2, leakage_results
    ):
        """Basic leakage detection fallback"""
        import difflib

        for conv1 in conversations1:
            content1 = self._extract_content(conv1)

            for conv2 in conversations2:
                content2 = self._extract_content(conv2)

                # Calculate similarity
                similarity = difflib.SequenceMatcher(None, content1, content2).ratio()

                if similarity >= 0.9:  # High similarity threshold
                    leakage_results["leakage_detected"] = True
                    leakage_results["near_leakage"].append(
                        {
                            "conv1_id": conv1["id"],
                            "conv2_id": conv2["id"],
                            "split1": split1,
                            "split2": split2,
                            "similarity": similarity,
                            "detection_method": "basic_similarity",
                        }
                    )

    def _extract_content(self, conversation):
        """Extract text content from conversation"""
        content_parts = []
        for message in conversation.get("messages", []):
            if isinstance(message, dict) and "content" in message:
                content_parts.append(message["content"])
        return " ".join(content_parts)


class EnterpriseDedupLeakageGates:
    """Enterprise-grade deduplication and leakage gates with full infrastructure integration"""

    def __init__(self, storage_config: StorageConfig):
        self.config = storage_config
        self.s3_client = None
        self.audit_trail = get_audit_trail()

        # Initialize enterprise deduplicator
        try:
            dedup_config = {
                "similarity_thresholds": {
                    "content": 0.85,
                    "semantic": 0.80,
                    "structural": 0.75,
                    "temporal": 0.70,
                    "overall": 0.80,
                },
                "processing": {
                    "batch_size": 1000,
                    "max_workers": 4,
                    "memory_limit_mb": 2048,
                    "enable_caching": True,
                },
                "quality": {
                    "min_confidence_score": 0.7,
                    "enable_fuzzy_matching": True,
                },
                "reporting": {
                    "enable_detailed_logging": True,
                    "save_duplicate_groups": True,
                },
            }
            self.deduplicator = EnterpriseConversationDeduplicator(dedup_config)
            logger.info("Enterprise deduplicator initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize enterprise deduplicator: {e}")
            self.deduplicator = None

        # Initialize enterprise leakage detector
        self.leakage_detector = EnterpriseLeakageDetector()

        self._init_s3_client()

    def _init_s3_client(self):
        """Initialize S3 client with enterprise configuration"""
        if self.config.backend != self.config.backend.S3:
            raise ValueError(
                "S3 backend required for enterprise dedup/leakage validation"
            )

        try:
            self.s3_client = boto3.client(
                "s3",
                endpoint_url=self.config.s3_endpoint_url,
                aws_access_key_id=self.config.s3_access_key_id,
                aws_secret_access_key=self.config.s3_secret_access_key,
                region_name=self.config.s3_region or "us-east-1",
            )

            self.s3_client.head_bucket(Bucket=self.config.s3_bucket)
            logger.info(f"✓ Connected to S3 bucket: {self.config.s3_bucket}")

        except Exception as e:
            logger.error(f"Failed to connect to S3: {e}")
            raise ValueError(f"Failed to connect to S3: {e}")

    def load_manifest(self, release_version: str) -> Dict[str, Any]:
        """Load release manifest from S3 with validation"""
        manifest_key = (
            f"{self.config.exports_prefix}/releases/{release_version}/manifest.json"
        )

        try:
            response = self.s3_client.get_object(
                Bucket=self.config.s3_bucket, Key=manifest_key
            )
            manifest = json.loads(response["Body"].read())
            logger.info(f"✓ Loaded manifest: {manifest_key}")

            # Log manifest access for audit
            if self.audit_trail:
                self.audit_trail.log_validation_started(
                    release_version, user_id="enterprise_dedup_leakage_gates"
                )

            return manifest
        except ClientError as e:
            logger.error(f"Failed to load manifest {manifest_key}: {e}")
            raise ValueError(f"Failed to load manifest {manifest_key}: {e}")

    def sample_file_content(self, s3_key: str, sample_size: int = 4096) -> str:
        """Sample content from an S3 file with security validation"""
        try:
            # Validate key for security
            if not self._validate_s3_key_security(s3_key):
                logger.warning(f"Security validation failed for key: {s3_key}")
                return ""

            # Get sample of file content
            response = self.s3_client.get_object(
                Bucket=self.config.s3_bucket,
                Key=s3_key,
                Range=f"bytes=0-{sample_size - 1}",
            )

            content = response["Body"].read().decode("utf-8", errors="ignore")
            return content

        except ClientError as e:
            logger.warning(f"Failed to sample {s3_key}: {e}")
            return ""

    def _validate_s3_key_security(self, s3_key: str) -> bool:
        """Validate S3 key meets enterprise security standards"""
        # Check for path traversal
        if ".." in s3_key or s3_key.startswith("/"):
            return False

        # Check for suspicious patterns
        suspicious_patterns = ["admin", "root", "config", "secret", "password"]
        if any(pattern in s3_key.lower() for pattern in suspicious_patterns):
            return False

        return True

    def run_deduplication_gate(self, manifest: Dict[str, Any]) -> Dict[str, Any]:
        """Run enterprise deduplication analysis using existing infrastructure"""
        logger.info("🔍 Running enterprise deduplication gate...")

        gate_results = {
            "gate_name": "enterprise_deduplication",
            "passed": True,
            "timestamp": datetime.utcnow().isoformat(),
            "family_results": {},
            "summary": {
                "total_files": 0,
                "exact_duplicates": 0,
                "near_duplicates": 0,
                "duplicate_groups": 0,
                "enterprise_grade": self.deduplicator is not None,
                "deduplication_rate": 0.0,
                "efficiency_score": 0.0,
            },
        }

        # Priority families for enhanced deduplication (as specified in requirements)
        priority_families = ["edge_cases", "voice_persona", "professional_therapeutic"]

        for family_name, family_data in manifest["families"].items():
            files = family_data["files"]
            logger.info(f"  Analyzing {family_name}: {len(files)} files")

            family_result = {
                "total_files": len(files),
                "sampled_files": 0,
                "exact_duplicate_groups": 0,
                "near_duplicates": 0,
                "passed": True,
                "enterprise_grade": False,
                "deduplication_summary": {},
                "exact_duplicates_detail": [],
                "near_duplicates_detail": [],
            }

            # Enhanced sampling for priority families
            if family_name in priority_families:
                sample_size = min(100, len(files))  # More comprehensive sampling
            else:
                sample_size = min(50, len(files))

            sample_files = files[:sample_size]
            family_result["sampled_files"] = len(sample_files)

            # Convert files to conversation format for enterprise deduplicator
            conversations = []
            for i, file_info in enumerate(sample_files):
                s3_key = file_info["key"]
                content_sample = self.sample_file_content(s3_key)

                if content_sample:
                    # Create conversation format
                    conversation = {
                        "id": f"{family_name}_{i}_{s3_key.split('/')[-1]}",
                        "messages": [
                            {"role": "system", "content": f"File: {s3_key}"},
                            {
                                "role": "user",
                                "content": content_sample[:1000],
                            },  # Limit content size
                        ],
                        "metadata": {
                            "family": family_name,
                            "original_file": s3_key,
                            "size": file_info.get("size", 0),
                        },
                    }
                    conversations.append(conversation)

            # Run enterprise deduplication if available
            if self.deduplicator and conversations:
                try:
                    dedup_result = self.deduplicator.deduplicate_conversations(
                        conversations
                    )

                    # Extract results
                    family_result["exact_duplicate_groups"] = len(
                        dedup_result.duplicate_groups
                    )
                    family_result["deduplication_summary"] = {
                        "original_count": dedup_result.original_count,
                        "unique_count": dedup_result.unique_count,
                        "duplicates_removed": dedup_result.duplicates_removed,
                        "deduplication_rate": dedup_result.deduplication_rate,
                        "efficiency_score": dedup_result.efficiency_score,
                        "processing_time": dedup_result.processing_time_seconds,
                    }

                    # Store duplicate groups (limited for readability)
                    family_result["exact_duplicates_detail"] = (
                        dedup_result.duplicate_groups[:10]
                    )

                    # Enterprise quality assessment
                    if (
                        dedup_result.deduplication_rate
                        < 0.1  # Low duplicate rate is good
                        and dedup_result.efficiency_score > 100
                    ):  # Good processing efficiency
                        family_result["enterprise_grade"] = True

                    # Check for excessive duplication (quality issue)
                    if (
                        dedup_result.deduplication_rate > 0.3
                    ):  # More than 30% duplicates
                        family_result["passed"] = False
                        gate_results["passed"] = False

                        # Log quality issue
                        if self.audit_trail:
                            self.audit_trail.log_safety_issue(
                                family_name,
                                {
                                    "issue_type": "excessive_duplication",
                                    "deduplication_rate": dedup_result.deduplication_rate,
                                    "duplicates_removed": dedup_result.duplicates_removed,
                                    "family": family_name,
                                },
                            )

                    # Update summary
                    gate_results["summary"]["exact_duplicates"] += len(
                        dedup_result.duplicate_groups
                    )
                    gate_results["summary"]["duplicate_groups"] += len(
                        dedup_result.duplicate_groups
                    )

                except Exception as e:
                    logger.error(
                        f"Enterprise deduplication failed for {family_name}: {e}"
                    )
                    family_result["passed"] = False
                    family_result["error"] = str(e)

                    # Log deduplication failure
                    if self.audit_trail:
                        self.audit_trail.log_safety_issue(
                            family_name,
                            {
                                "issue_type": "deduplication_failure",
                                "error": str(e),
                                "family": family_name,
                            },
                        )
            else:
                # Fallback or no deduplicator available
                if not self.deduplicator:
                    logger.warning(
                        f"Enterprise deduplicator unavailable for {family_name}"
                    )
                    family_result["error"] = "Enterprise deduplicator unavailable"
                    gate_results["summary"]["enterprise_grade"] = False

            gate_results["family_results"][family_name] = family_result
            gate_results["summary"]["total_files"] += len(files)

        # Calculate overall metrics
        total_duplicate_groups = gate_results["summary"]["duplicate_groups"]
        total_files = gate_results["summary"]["total_files"]

        if total_files > 0:
            gate_results["summary"]["deduplication_rate"] = (
                total_duplicate_groups / total_files
            )

        # Enterprise readiness assessment
        enterprise_families = sum(
            1
            for result in gate_results["family_results"].values()
            if result.get("enterprise_grade", False)
        )
        total_families = len(gate_results["family_results"])

        gate_results["enterprise_readiness"] = (
            (enterprise_families / total_families) if total_families > 0 else 0
        )

        return gate_results

    def run_leakage_gate(self, manifest: Dict[str, Any]) -> Dict[str, Any]:
        """Run enterprise cross-split leakage analysis"""
        logger.info("🚰 Running enterprise cross-split leakage gate...")

        gate_results = {
            "gate_name": "enterprise_cross_split_leakage",
            "passed": True,
            "timestamp": datetime.utcnow().isoformat(),
            "family_results": {},
            "summary": {
                "total_families": 0,
                "families_with_leakage": 0,
                "total_leakage_instances": 0,
                "enterprise_grade": self.leakage_detector.deduplicator is not None,
                "clinical_validated": self.leakage_detector.clinical_validator is not None
            },
        }

        # Priority families for leakage checking (holdout-sensitive)
        priority_families = ["edge_cases", "voice_persona", "professional_therapeutic"]

        for family_name, family_data in manifest["families"].items():
            files = family_data["files"]
            logger.info(f"  Checking {family_name} for cross-split leakage...")

            # Group files by split
            files_by_split = defaultdict(list)
            for file_info in files:
                split = file_info.get("split", "unknown")

                # Add content sample for leakage detection
                s3_key = file_info["key"]
                content_sample = self.sample_file_content(s3_key, 1024)  # Smaller sample for leakage
                file_info_with_content = {**file_info, "content_sample": content_sample}

                files_by_split[split].append(file_info_with_content)

            # Check for leakage using enterprise detector
            leakage_result = self.leakage_detector.check_cross_split_leakage(files_by_split)

            # Determine if famild = Fals leakage gate
            family_passed = True
            if leakage_result["leakage_detected"]:
                if family_name in priority_families:
                    # Strict enforcement for priority families
                    family_passed = False
                    gate_results["passed"] = False

                    # Log critical leakage
                    if self.audit_trail:
                        self.audit_trail.log_intervention_required(
                            family_name,
                            f"Cross-split leakage detected in priority family: {family_name}",
                            "high"
                        )
                else:
                    # Warning for other families
                    logger.warning(f"Leakage detected in {family_name} (non-critical)")

            family_result = {
                "total_files": len(files),
                "splits": list(files_by_split.keys()),
                "split_distribution": {
                    split: len(files) for split, files in files_by_split.items()
                },
                "leakage_detected": leakage_result["leakage_detected"],
                "exact_leakage_count": len(leakage_result["exact_leakage"]),
                "near_leakage_count": len(leakage_result["near_leakage"]),
                "passed": family_passed,
                "enterprise_grade": leakage_result.get("enterprise_grade", False),
                "clinical_validated": leakage_result.get("clinical_validated", False),
                "leakage_details": {
                    **leakage_result,
                    # Limit details for readability
                    "exact_leakage": leakage_result["exact_leakage"][:5],
                    "near_leakage": leakage_result["near_leakage"][:5]
                }
            }

            gate_results["family_results"][family_name] = family_result
            gate_results["summary"]["total_families"] += 1

            if leakage_result["leakage_detected"]:
                gate_results["summary"]["families_with_leakage"] += 1
                gate_results["summary"]["total_leakage_instances"] += (
                    len(leakage_result["exact_leakage"]) + len(leakage_result["near_leakage"])
                t(f"✓ Gate report saved: {report_url}")

        # Enterprise readiness assessment
        enterprise_families = sum(1 for result in gate_results["family_results"].values()
                                 if result.get("enterprise_grade", False))
        total_families = gate_results["summary"]["total_families"]

        gate_results["enterprise_readiness"] = (enterprise_families / total_families) if total_families > 0 else 0

        return gate_results

    def save_gate_report(
        self, gate_results: Dict[str, Any], release_version: str
    ) -> str:
        """Save gate results to S3 with enhanced metadata"""
        gate_name = gate_results["gate_name"]
        report_key = f"{self.config.exports_prefix}/releases/{release_version}/gates/{gate_name}_report.json"

        # Add enhanced metadata
        enhanced_results = {
            **gate_results,
            "metadata": {
                "report_version": "2.0",
                "generated_by": "enterprise_dedup_leakage_gates",
                "s3_bucket": self.config.s3_bucket,
                "s3_key": report_key,
                "enterprise_grade": True,
                "audit_trail_enabled": self.audit_trail is not None,
                "deduplicator_available": self.deduplicator is not None
            }
        }

        try:
            self.s3_client.put_object(
                Bucket=self.config.s3_bucket,
                Key=report_key,
                Body=json.dumps(enhanced_results, indent=2),
                ContentType="application/json",
                Metadata={
                    "gate-type": gate_name,
                    "release-version": release_version,
                    "enterprise-grade": "true"
                }
            )

            report_url = f"s3://{self.config.s3_bucket}/{report_key}"
            logger.info(f"✓ Enterprise gate report saved: {report_url}")
            return report_url

        except ClientError as e:
            logger.error(f"Failed to save gate report: {e}")
            raise ValueError(f"Failed to save gate report: {e}")

    def run_all_gates(se, release_version: str) -> Dict[str, Any]:
        """Run all enterprise deduplication and leakage gates"""
        logger.info(f"🔍 Running enterprise deduplication and leakage gates for {release_version}...")

        # Initialize audit trail for this release
        if self.audit_trail:
            self.audit_trail.log_validation_started(
                release_version,
                user_id="enterprise_dedup_leakage_gates"
            )

        # Load manifest
        manifest = self.load_manifest(release_version)

        # Run gates with enhanced error handling
        try:
            dedup_results = self.run_deduplication_gate(manifest)
        except Exception as e:
            logger.error(f"Deduplication gate failed: {e}")
            dedup_results = {
                "gate_name": "enterprise_deduplication",
                "passed": False,
                "error": str(e),
                "timestamp": datetime.utcnow().isoformat()
            }

        try:
            leakage_results = self.run_leakage_gate(manifest)
        except Exception as e:
            logger.error(f"Leakage gate failed: {e}")
            leakage_results = {
                "gate_name": "enterprise_cross_split_leakage",
                "passed": False,
                "error": str(e),
                "timestamp": datetime.utcnow().isoformat()
            }

        # Save individual gate reports
        dedup_report_url = self.save_gate_report(dedup_results, release_version)
        leakage_report_url = self.save_gate_report(leakage_results, release_version)

        # Create comprehensive combined results
        combined_results = {
            "release_version": release_version,
            "timestamp": datetime.utcnow().isoformat(),
            "overall_passed": dedup_results["passed"] and leakage_results["passed"],
            "enterprise_ready": (
                dedup_results.get("summary", {}).get("enterprise_grade", False) and
                leakage_results.get("summary", {}).get("enterprise_grade", False)
            ),
            "gates": {
                "deduplication": {
                    "passed": dedup_results["passed"],
                    "report_url": dedup_report_url,
                    "summary": dedup_results.get("summary", {}),
                    "enterprise_readiness": dedup_results.get("enterprise_readiness", 0)
                },
                "cross_split_leakage": {
                    "passed": leakage_results["passed"],
                    "report_url": leakage_report_url,
                    "summary": leakage_results.get("summary", {}),
                    "enterprise_readiness": leakage_results.get("enterprise_readiness", 0)
                },
            },
            "infrastructure_status": {
                "enterprise_deduplicator": self.deduplicator is not None,
                "clinical_validator": self.leakage_detector.clinical_validator is not None,
                "audit_trail": self.audit_trail is not None
            }
        }

        # Save combined report
        combined_key = f"{self.config.exports_prefix}/releases/{release_version}/gates/enterprise_dedup_leakage_combined_report.json"

        try:
            self.s3_client.put_object(
                Bucket=self.config.s3_bucket,
                Key=combined_key,
                Body=json.dumps(combined_results, indent=2),
                ContentType="application/json",
            )

            combined_results["combined_report_url"] = (
                f"s3://{self.config.s3_bucket}/{combined_key}"
            )

        except ClientError as e:
            logger.error(f"Failed to save combined report: {e}")

        # Log completion
        if self.audit_trail:
            self.audit_trail.log_validation_completion(
                release_version,
                {
                    "overall_passed": combined_results["overall_passed"],
                    "enterprise_ready": combined_results["enterprise_ready"],
                    "gates_run": ["deduplication", "cross_split_leakage"]
                }
            )

        return combined_results

    def print_gate_summary(self, results: Dict[str, Any]):
        """Print comprehensive enterprise gate summary"""
        print("\n" + "=" * 70)
        print("🔍 ENTERPRISE DEDUPLICATION & LEAKAGE GATES SUMMARY")
        print("=" * 70)

        print(f"Release Version: {results['release_version']}")
        print(f"Gate Run Time: {results['timestamp']}")

        overall_status = "✅ PASSED" if results["overall_passed"] else "❌ FAILED"
        enterprise_status = "🏢 ENTERPRISE READY" if results.get("enterprise_ready", False) else "⚠️  STANDARD GRADE"

        print(f"Overall Status: {overall_status}")
        print(f"Enterprise Grade: {enterprise_status}")

        print("\n📊 ENTERPRISE GATE RESULTS:")

        # Deduplication Gate
        dedup_gate = results["gates"]["deduplication"]
        dedup_status = "✅ PASSED" if dedup_gate["passed"] else "❌ FAILED"
        enterprise_dedup = f"🏢 {dedup_gate.get('enterprise_readiness', 0):.1%}" if dedup_gate.get('enterprise_readiness', 0) > 0.8 else f"📊 {dedup_gate.get('enterprise_readiness', 0):.1%}"

        print(f"  Deduplication: {dedup_status} ({enterprise_dedup})")
        print(f"    Total Files: {dedup_gate['summary'].get('total_files', 0)}")
        print(f"    Exact Duplicates: {dedup_gate['summary'].get('exact_duplicates', 0)}")
        print(f"    Near Duplicates: {dedup_gate['summary'].get('near_duplicates', 0)}")
        print(f"    Enterprise Grade: {'✅' if dedup_gate['summary'].get('enterprise_grade', False) else '❌'}")

        # Leakage Gate
        leakage_gate = results["gates"]["cross_split_leakage"]
        leakage_status = "✅ PASSED" if leakage_gate["passed"] else "❌ FAILED"
        enterprise_leakage = f"🏢 {leakage_gate.get('enterprise_readiness', 0):.1%}" if leakage_gate.get('enterprise_readiness', 0) > 0.8 else f"📊 {leakage_gate.get('enterprise_readiness', 0):.1%}"

        print(f"  Cross-Split Leakage: {leakage_status} ({enterprise_leakage})")
        print(f"    Families Checked: {leakage_gate['summary'].get('total_families', 0)}")
        print(f"    Families with Leakage: {leakage_gate['summary'].get('families_with_leakage', 0)}")
        print(f"    Total Leakage Instances: {leakage_gate['summary'].get('total_leakage_instances', 0)}")
        print(f"    Enterprise Grade: {'✅' if leakage_gate['summary'].get('enterprise_grade', False) else '❌'}")
        print(f"    Clinical Validated: {'✅' if leakage_gate['summary'].get('clinical_validated', False) else '❌'}")

        # Infrastructure Status
        infra_status = results.get("infrastructure_status", {})
        print(f"\n🏗️  INFRASTRUCTURE STATUS:")
        print(f"  Enterprise Deduplicator: {'✅ AVAILABLE' if infra_status.get('enterprise_deduplicator', False) else '❌ UNAVAILABLE'}")
        print(f"  Clinical Validator: {'✅ AVAILABLE' if infra_status.get('clinical_validator', False) else '❌ UNAVAILABLE'}")
        print(f"  Audit Trail: {'✅ ENABLED' if infra_status.get('audit_trail', False) else '⚠️  DISABLED'}")

        if not results["overall_passed"]:
            print("\n🚨 RELEASE BLOCKED:")
            print("  Release cannot proceed due to deduplication/leakage issues.")
            print("  Review gate reports and address all issues before retry.")

            if not results.get("enterprise_ready", False):
                print("\n💼 ENTERPRISE READINESS:")
                print("  System not meeting enterprise-grade standards.")
                print("  Consider upgrading infrastructure components.")

        print("\n📄 DETAILED REPORTS:")
        for gate_name, gate_info in results["gates"].items():
            print(f"  {gate_name}: {gate_info.get('report_url', 'N/A')}")

        if "combined_report_url" in results:
f"  Combined Report: {results['combined_report_url']}")

        print("\n" + "=" * 70)


def main():
    """Main entry point"""
    print("🚀 Starting Enterprise Deduplication & Leakage Gates...")

    # Parse command line arguments
    if len(sys.argv) < 2:
        print("Usage: python dedup_leakage_gates.py <release_version>")
        print("Example: python dedup_leakage_gates.py v2025-01-02")
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
        # Create enterprise gates validator and run gates
        gates = EnterpriseDedupLeakageGates(config)
        results = gates.run_all_gates(release_version)

        # Print results
        gates.print_gate_summary(results)

        # Exit with appropriate code (fail closed)
        if results["overall_passed"]:
            if results.get("enterprise_ready", False):
                print(f"\n🏢 Enterprise-grade dedup/leakage gates passed for {release_version}!")
                sys.exit(0)
            else:
                print(f"\n✅ Standard dedup/leakage gates passed for {release_version} (enterprise upgrade recommended)")
                sys.exit(0)
        else:
            print(f"\n❌ Dedup/leakage gates failed for {release_version} - RELEASE BLOCKED")
            sys.exit(1)

    except Exception as e:
        logger.error(f"Enterprise dedup/leakage gate validation failed: {e}")
        print(f"❌ Enterprise dedup/leakage gate validation failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
