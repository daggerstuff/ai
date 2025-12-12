#!/usr/bin/env python3
"""
Dataset Inventory Audit - Maps S3 objects to dataset families and generates coverage report
"""

import json
import logging
import re
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)


@dataclass
class DatasetFamily:
    """Represents a required dataset family"""

    name: str
    description: str
    required: bool = True
    s3_evidence: list[str] = field(default_factory=list)
    status: str = "missing"  # missing, partial, present
    notes: str = ""


@dataclass
class S3Object:
    """Represents an S3 object"""

    key: str
    size: int
    size_formatted: str
    last_modified: str | None = None
    etag: str | None = None


class DatasetInventoryAuditor:
    """Audits S3 manifest against required dataset families"""

    # Required dataset families based on user requirements
    REQUIRED_FAMILIES = {
        "edge_case_generator": DatasetFamily(
            name="edge_case_generator",
            description="Edge case generator outputs (nightmare fuel scenarios)",
            required=True,
        ),
        "edge_case_resulting_chats": DatasetFamily(
            name="edge_case_resulting_chats",
            description="Resulting chats from edge case generator",
            required=True,
        ),
        "edge_case_synthetic": DatasetFamily(
            name="edge_case_synthetic", description="Synthetic edge case dataset", required=True
        ),
        "long_running_therapy": DatasetFamily(
            name="long_running_therapy",
            description="Long-running therapy sessions (actual therapy sessions)",
            required=True,
        ),
        "mental_health_datasets": DatasetFamily(
            name="mental_health_datasets", description="All mental health datasets", required=True
        ),
        "video_transcripts": DatasetFamily(
            name="video_transcripts",
            description="Video transcripts (YouTube, podcasts, etc.)",
            required=True,
        ),
        "voice_persona": DatasetFamily(
            name="voice_persona",
            description="Personality/voice datasets from video transcripts",
            required=True,
        ),
        "safety_guardrails_annihilator": DatasetFamily(
            name="safety_guardrails_annihilator",
            description="Unrestricted content for robust handling of nasty content",
            required=True,
        ),
        "sarcasm": DatasetFamily(name="sarcasm", description="Sarcasm datasets", required=True),
        "cptsd": DatasetFamily(name="cptsd", description="CPTSD-specific datasets", required=True),
        "addiction": DatasetFamily(
            name="addiction", description="Addiction-specific datasets", required=True
        ),
        "experimental": DatasetFamily(
            name="experimental", description="Experimental datasets", required=True
        ),
        "roleplay_simulator": DatasetFamily(
            name="roleplay_simulator",
            description="Roleplaying guide/training simulator designer datasets",
            required=True,
        ),
        "dpo_preference": DatasetFamily(
            name="dpo_preference", description="DPO/preference training datasets", required=True
        ),
    }

    # Keywords to match S3 keys to families
    KEYWORD_MAPPINGS = {
        "edge_case_generator": [
            "edge_case",
            "edge-case",
            "edgecase",
            "crisis",
            "nightmare",
            "edge_case_pipeline",
            "edge_case_loader",
            "edge_cases_training",
        ],
        "edge_case_resulting_chats": ["edge_case.*chat", "resulting.*chat", "edge.*conversation"],
        "edge_case_synthetic": ["synthetic.*edge", "edge.*synthetic"],
        "long_running_therapy": [
            "long.*session",
            "long.*therapy",
            "extended.*session",
            "multi.*session",
            "ongoing.*therapy",
        ],
        "mental_health_datasets": [
            "mental_health",
            "mental-health",
            "counseling",
            "therapy",
            "therapeutic",
            "psychology",
            "psych",
            "counsel",
        ],
        "video_transcripts": [
            "transcript",
            "youtube",
            "video",
            "podcast",
            "tim_fletcher",
            "tim fletcher",
            "crappy.*childhood",
            "doc.*snipes",
            "heidi.*priebe",
        ],
        "voice_persona": [
            "voice",
            "persona",
            "personality",
            "tim_fletcher_voice",
            "pixel_voice",
            "voice_profile",
        ],
        "safety_guardrails_annihilator": [
            "unrestricted",
            "guardrail",
            "jailbreak",
            "adversarial",
            "reddit.*mental",
            "reddit_mental_health",
        ],
        "sarcasm": ["sarcasm", "sarcastic", "irony"],
        "cptsd": [
            "cptsd",
            "c-ptsd",
            "complex.*ptsd",
            "complex.*trauma",
            "complextrauma",
            "tim.*fletcher",
        ],
        "addiction": ["addiction", "addiction_counseling", "substance", "recovery"],
        "experimental": ["experimental", "experiment", "research", "pilot"],
        "roleplay_simulator": [
            "roleplay",
            "role-play",
            "simulator",
            "training.*simulator",
            "designer",
            "scenario.*generator",
        ],
        "dpo_preference": ["dpo", "preference", "rlhf", "human.*preference", "anthropic.*hh"],
    }

    def __init__(self, manifest_path: str | Path, registry_path: str | Path | None = None):
        self.manifest_path = Path(manifest_path)
        self.registry_path = Path(registry_path) if registry_path else None
        self.manifest_data: dict[str, Any] = {}
        self.registry_data: dict[str, Any] = {}
        self.s3_objects: list[S3Object] = []
        self.family_matches: dict[str, list[S3Object]] = defaultdict(list)

    def load_manifest(self) -> None:
        """Load S3 manifest JSON"""
        logger.info(f"Loading S3 manifest from {self.manifest_path}")
        with open(self.manifest_path, encoding="utf-8") as f:
            self.manifest_data = json.load(f)

        # Extract all S3 objects
        self._extract_s3_objects()
        logger.info(f"Loaded {len(self.s3_objects)} S3 objects")

    def _extract_s3_objects(self) -> None:
        """Extract S3 objects from manifest structure"""

        def extract_from_category(category_data: dict, prefix: str = "") -> None:
            if "objects" in category_data:
                for obj in category_data["objects"]:
                    if "key" in obj:
                        s3_obj = S3Object(
                            key=obj["key"],
                            size=obj.get("size", 0),
                            size_formatted=obj.get("size_formatted", "0 B"),
                            last_modified=obj.get("last_modified"),
                            etag=obj.get("etag"),
                        )
                        self.s3_objects.append(s3_obj)

            # Recursively process nested categories
            for key, value in category_data.items():
                if isinstance(value, dict) and key != "objects":
                    extract_from_category(value, f"{prefix}/{key}")

        if "categories" in self.manifest_data:
            for category_name, category_data in self.manifest_data["categories"].items():
                extract_from_category(category_data, category_name)

    def load_registry(self) -> None:
        """Load dataset registry if available"""
        if not self.registry_path or not self.registry_path.exists():
            logger.warning("Dataset registry not found, skipping")
            return

        logger.info(f"Loading dataset registry from {self.registry_path}")
        with open(self.registry_path, encoding="utf-8") as f:
            self.registry_data = json.load(f)

    def match_objects_to_families(self) -> None:
        """Match S3 objects to dataset families using keywords"""
        logger.info("Matching S3 objects to dataset families...")

        for s3_obj in self.s3_objects:
            key_lower = s3_obj.key.lower()

            # Skip non-data files
            if any(key_lower.endswith(ext) for ext in [".lock", ".md", ".txt", ".py", ".sh"]):
                continue

            # Match against each family
            for family_name, keywords in self.KEYWORD_MAPPINGS.items():
                for keyword in keywords:
                    pattern = keyword.replace(".*", ".*").replace("*", ".*")
                    if re.search(pattern, key_lower, re.IGNORECASE):
                        self.family_matches[family_name].append(s3_obj)
                        self.REQUIRED_FAMILIES[family_name].s3_evidence.append(s3_obj.key)
                        break

    def assess_family_status(self) -> None:
        """Assess status of each required family"""
        for family_name, family in self.REQUIRED_FAMILIES.items():
            evidence_count = len(family.s3_evidence)

            if evidence_count == 0:
                family.status = "missing"
                family.notes = "No S3 objects found matching this family"
            elif evidence_count < 3:
                family.status = "partial"
                family.notes = f"Found {evidence_count} matching object(s) - may need more"
            else:
                family.status = "present"
                family.notes = f"Found {evidence_count} matching object(s)"

    def generate_coverage_report(self) -> dict[str, Any]:
        """Generate comprehensive coverage report"""
        report = {
            "generated_at": datetime.now().isoformat(),
            "manifest_path": str(self.manifest_path),
            "total_s3_objects": len(self.s3_objects),
            "families": {},
            "summary": {
                "total_families": len(self.REQUIRED_FAMILIES),
                "present": 0,
                "partial": 0,
                "missing": 0,
            },
        }

        for family_name, family in self.REQUIRED_FAMILIES.items():
            family_report = {
                "name": family.name,
                "description": family.description,
                "required": family.required,
                "status": family.status,
                "evidence_count": len(family.s3_evidence),
                "s3_evidence": family.s3_evidence[:10],  # First 10 for brevity
                "total_evidence": len(family.s3_evidence),
                "notes": family.notes,
            }
            report["families"][family_name] = family_report
            report["summary"][family.status] += 1

        return report

    def run_audit(self) -> dict[str, Any]:
        """Run complete audit"""
        logger.info("Starting dataset inventory audit...")

        self.load_manifest()
        if self.registry_path:
            self.load_registry()

        self.match_objects_to_families()
        self.assess_family_status()

        report = self.generate_coverage_report()

        logger.info("Audit complete")
        logger.info(
            f"Summary: {report['summary']['present']} present, "
            f"{report['summary']['partial']} partial, "
            f"{report['summary']['missing']} missing"
        )

        return report


def main():
    """Main entry point"""
    project_root = Path(__file__).parents[3]
    manifest_path = project_root / "ai" / "training_ready" / "data" / "s3_manifest.json"
    registry_path = project_root / "ai" / "data" / "dataset_registry.json"

    auditor = DatasetInventoryAuditor(manifest_path, registry_path)
    report = auditor.run_audit()

    # Save report
    output_path = project_root / "ai" / "training_ready" / "data" / "dataset_coverage_report.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    logger.info(f"Coverage report saved to {output_path}")

    # Print summary
    print("\n" + "=" * 60)
    print("DATASET COVERAGE SUMMARY")
    print("=" * 60)
    for family_name, family_data in report["families"].items():
        status_icon = {"present": "✅", "partial": "⚠️", "missing": "❌"}.get(
            family_data["status"], "❓"
        )

        print(
            f"{status_icon} {family_name:30s} {family_data['status']:10s} "
            f"({family_data['evidence_count']} objects)"
        )

    print("=" * 60)
    print(
        f"Total: {report['summary']['present']} present, "
        f"{report['summary']['partial']} partial, "
        f"{report['summary']['missing']} missing"
    )
    print("=" * 60)


if __name__ == "__main__":
    main()
