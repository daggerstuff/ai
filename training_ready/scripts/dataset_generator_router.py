#!/usr/bin/env python3
"""
Dataset Generator Router - Ensures all required dataset families are routable and can be loaded
"""

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)


@dataclass
class DatasetGenerator:
    """Represents a dataset generator/loader"""

    family_name: str
    loader_type: str  # 's3_direct', 'pipeline', 'synthetic', 'missing'
    s3_path: str | None = None
    pipeline_path: str | None = None
    status: str = "unknown"  # 'available', 'needs_generation', 'missing'
    notes: str = ""


class DatasetGeneratorRouter:
    """Routes dataset family requests to appropriate generators/loaders"""

    def __init__(self, coverage_report_path: Path, registry_path: Path | None = None):
        self.coverage_report_path = coverage_report_path
        self.registry_path = registry_path
        self.coverage_data: dict[str, Any] = {}
        self.registry_data: dict[str, Any] = {}
        self.generators: dict[str, DatasetGenerator] = {}

    def load_coverage_report(self) -> None:
        """Load coverage report"""
        with open(self.coverage_report_path, encoding="utf-8") as f:
            self.coverage_data = json.load(f)

    def load_registry(self) -> None:
        """Load dataset registry if available"""
        if self.registry_path and self.registry_path.exists():
            with open(self.registry_path, encoding="utf-8") as f:
                self.registry_data = json.load(f)

    def map_families_to_generators(self) -> None:
        """Map dataset families to generators/loaders"""
        families = self.coverage_data.get("families", {})

        for family_name, family_data in families.items():
            status = family_data.get("status", "missing")
            evidence = family_data.get("s3_evidence", [])

            generator = DatasetGenerator(
                family_name=family_name, loader_type="unknown", status=status
            )

            # Route based on family type and available evidence
            if status == "present" and evidence:
                # Has S3 data - use S3 direct loader
                generator.loader_type = "s3_direct"
                generator.s3_path = evidence[0] if evidence else None
                generator.status = "available"
                generator.notes = f"Available in S3: {len(evidence)} objects"

            elif status == "partial" and evidence:
                # Partial - may need more data
                generator.loader_type = "s3_direct"
                generator.s3_path = evidence[0] if evidence else None
                generator.status = "available"
                generator.notes = f"Partial coverage: {len(evidence)} objects - may need expansion"

            elif family_name == "edge_case_generator":
                # Edge case generator - check registry for pipeline
                generator.loader_type = "pipeline"
                generator.pipeline_path = "ai/pipelines/edge_case_pipeline_standalone"
                if self.registry_data.get("edge_case_sources", {}).get("edge_case_generator"):
                    generator.status = "available"
                    generator.notes = "Edge case generator pipeline exists"
                else:
                    generator.status = "needs_generation"
                    generator.notes = "Edge case generator pipeline needs to be run"

            elif family_name == "edge_case_resulting_chats":
                # Resulting chats from edge case generator
                generator.loader_type = "pipeline"
                generator.pipeline_path = "ai/pipelines/edge_case_pipeline_standalone"
                generator.status = "needs_generation"
                generator.notes = "Run edge case generator to produce resulting chats"

            elif family_name == "edge_case_synthetic":
                # Synthetic edge cases
                generator.loader_type = "synthetic"
                generator.status = "needs_generation"
                generator.notes = "Synthetic edge case dataset needs to be generated"

            elif family_name == "long_running_therapy":
                # Long-running therapy sessions
                generator.loader_type = "s3_direct"
                generator.status = "needs_generation"
                generator.notes = (
                    "Need to identify/extract long-running therapy sessions from existing datasets"
                )

            elif family_name == "cptsd":
                # CPTSD - check if Tim Fletcher transcripts cover this
                if any("tim" in str(e).lower() and "fletcher" in str(e).lower() for e in evidence):
                    generator.loader_type = "s3_direct"
                    generator.status = "available"
                    generator.notes = (
                        "CPTSD content available in Tim Fletcher transcripts - needs proper tagging"
                    )
                else:
                    generator.loader_type = "s3_direct"
                    generator.status = "needs_generation"
                    generator.notes = (
                        "CPTSD datasets need to be identified/tagged from existing content"
                    )

            elif family_name == "sarcasm":
                # Sarcasm - has partial coverage
                if evidence:
                    generator.loader_type = "s3_direct"
                    generator.s3_path = evidence[0]
                    generator.status = "available"
                    generator.notes = f"Partial sarcasm dataset available: {evidence[0]}"
                else:
                    generator.loader_type = "synthetic"
                    generator.status = "needs_generation"
                    generator.notes = "Sarcasm dataset needs to be generated/curated"

            elif family_name == "roleplay_simulator":
                # Roleplay/simulator - has partial coverage
                if evidence:
                    generator.loader_type = "s3_direct"
                    generator.s3_path = evidence[0]
                    generator.status = "available"
                    generator.notes = f"Roleplay datasets available: {len(evidence)} objects"
                else:
                    generator.loader_type = "synthetic"
                    generator.status = "needs_generation"
                    generator.notes = "Roleplay/simulator datasets need expansion"

            else:
                # Missing - mark as needs generation
                generator.loader_type = "missing"
                generator.status = "needs_generation"
                generator.notes = "Dataset family missing - needs to be generated or located"

            self.generators[family_name] = generator

    def generate_routing_config(self) -> dict[str, Any]:
        """Generate routing configuration for all dataset families"""
        routing_config = {"generated_at": self.coverage_data.get("generated_at"), "families": {}}

        for family_name, generator in self.generators.items():
            routing_config["families"][family_name] = {
                "family_name": generator.family_name,
                "loader_type": generator.loader_type,
                "status": generator.status,
                "s3_path": generator.s3_path,
                "pipeline_path": generator.pipeline_path,
                "notes": generator.notes,
                "action_required": "none" if generator.status == "available" else "generate",
            }

        return routing_config

    def generate_loader_code(self, output_dir: Path) -> None:
        """Generate loader code snippets for missing generators"""
        output_dir.mkdir(parents=True, exist_ok=True)

        loader_code = {
            "edge_case_resulting_chats": '''
def load_edge_case_resulting_chats():
    """Load resulting chats from edge case generator"""
    from ai.training_ready.utils.s3_dataset_loader import S3DatasetLoader

    loader = S3DatasetLoader()
    # Load from edge case pipeline output
    s3_path = "s3://pixel-data/gdrive/processed/edge_cases/resulting_chats.jsonl"

    if loader.object_exists(s3_path):
        return list(loader.stream_jsonl(s3_path))
    else:
        # Run edge case generator pipeline first
        raise FileNotFoundError("Edge case resulting chats not found. Run edge case generator pipeline.")
''',
            "edge_case_synthetic": '''
def load_edge_case_synthetic():
    """Load synthetic edge case dataset"""
    from ai.training_ready.utils.s3_dataset_loader import S3DatasetLoader

    loader = S3DatasetLoader()
    s3_path = "s3://pixel-data/gdrive/processed/edge_cases/synthetic.jsonl"

    if loader.object_exists(s3_path):
        return list(loader.stream_jsonl(s3_path))
    else:
        # Generate synthetic edge cases
        raise FileNotFoundError("Synthetic edge cases not found. Generate using edge case generator.")
''',
            "long_running_therapy": '''
def load_long_running_therapy():
    """Load long-running therapy sessions"""
    from ai.training_ready.utils.s3_dataset_loader import S3DatasetLoader

    loader = S3DatasetLoader()
    # Filter existing therapy datasets for long sessions (>20 turns)
    s3_paths = [
        "s3://pixel-data/gdrive/processed/professional_therapeutic/therapist_sft/...",
        "s3://pixel-data/gdrive/processed/professional_therapeutic/psych8k/..."
    ]

    long_sessions = []
    for s3_path in s3_paths:
        if loader.object_exists(s3_path):
            for conv in loader.stream_jsonl(s3_path):
                if len(conv.get('messages', [])) > 20:
                    long_sessions.append(conv)

    return long_sessions
''',
            "cptsd": '''
def load_cptsd_datasets():
    """Load CPTSD-specific datasets"""
    from ai.training_ready.utils.s3_dataset_loader import S3DatasetLoader

    loader = S3DatasetLoader()
    # Tim Fletcher transcripts are CPTSD-focused
    s3_paths = [
        "s3://pixel-data/gdrive/processed/voice_persona/tim_fletcher/...",
        "s3://pixel-data/gdrive/processed/edge_cases/cptsd/..."
    ]

    cptsd_data = []
    for s3_path in s3_paths:
        if loader.object_exists(s3_path):
            cptsd_data.extend(list(loader.stream_jsonl(s3_path)))

    return cptsd_data
''',
        }

        for family_name, code in loader_code.items():
            if self.generators.get(family_name, DatasetGenerator("", "")).status != "available":
                code_file = output_dir / f"{family_name}_loader.py"
                with open(code_file, "w") as f:
                    f.write(code)
                logger.info(f"Generated loader code for {family_name}: {code_file}")


def main():
    """Main entry point"""
    project_root = Path(__file__).parents[3]
    coverage_report_path = (
        project_root / "ai" / "training_ready" / "data" / "dataset_coverage_report.json"
    )
    registry_path = project_root / "ai" / "data" / "dataset_registry.json"

    router = DatasetGeneratorRouter(coverage_report_path, registry_path)
    router.load_coverage_report()
    router.load_registry()
    router.map_families_to_generators()

    routing_config = router.generate_routing_config()

    # Save routing config
    output_path = project_root / "ai" / "training_ready" / "data" / "dataset_routing_config.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(routing_config, f, indent=2, ensure_ascii=False)

    logger.info(f"Routing config saved to {output_path}")

    # Generate loader code for missing generators
    loader_dir = project_root / "ai" / "training_ready" / "scripts" / "generated_loaders"
    router.generate_loader_code(loader_dir)

    # Print summary
    print("\n" + "=" * 60)
    print("DATASET GENERATOR ROUTING SUMMARY")
    print("=" * 60)
    for family_name, generator in router.generators.items():
        status_icon = {"available": "✅", "needs_generation": "⚠️", "missing": "❌"}.get(
            generator.status, "❓"
        )

        print(f"{status_icon} {family_name:30s} {generator.loader_type:15s} {generator.status}")
        if generator.notes:
            print(f"   └─ {generator.notes}")
    print("=" * 60)


if __name__ == "__main__":
    main()
