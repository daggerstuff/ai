#!/usr/bin/env python3
"""
Unified Data Processing Pipeline Orchestrator

Processes all sourced datasets through unified preprocessing pipeline.
Converts to standard training format (JSONL) with proper conversation structure.
"""

import json
import sys
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime
import logging

# Add project root to path for imports
project_root = Path(__file__).parent.parent.parent.parent.parent
sys.path.insert(0, str(project_root))

try:
    from ai.dataset_pipeline.unified_preprocessing_pipeline import UnifiedPreprocessingPipeline, ProcessingConfig, DataSource, StagePolicy
    from ai.dataset_pipeline.orchestration.integrated_training_pipeline import IntegratedTrainingPipeline, IntegratedPipelineConfig
    from ai.dataset_pipeline.configs.stages import get_all_stages, STAGE1_ID, STAGE2_ID, STAGE3_ID, STAGE4_ID
except ImportError as e:
    logging.error(f"Failed to import pipeline modules: {e}")
    logging.error(f"Project root: {project_root}")
    logging.error(f"Python path: {sys.path[:3]}")
    sys.exit(1)

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class DatasetProcessor:
    """Processes datasets through unified preprocessing pipeline"""

    def __init__(self, manifest_path: Path, sourcing_report_path: Path):
        self.manifest_path = manifest_path
        self.sourcing_report_path = sourcing_report_path
        self.manifest = self._load_manifest()
        self.sourcing_report = self._load_sourcing_report()
        self.processor = UnifiedPreprocessingPipeline(
            ProcessingConfig(
                target_quality_threshold=0.8,
                deduplication_enabled=True,
                validation_enabled=True,
                safety_filtering_enabled=True,
            )
        )
        self.processed_datasets: List[Dict[str, Any]] = []

    def _load_manifest(self) -> Dict[str, Any]:
        """Load training manifest"""
        with open(self.manifest_path, "r") as f:
            return json.load(f)

    def _load_sourcing_report(self) -> Dict[str, Any]:
        """Load sourcing report"""
        if not self.sourcing_report_path.exists():
            return {"results": []}
        with open(self.sourcing_report_path, "r") as f:
            return json.load(f)

    def determine_stage(self, dataset_info: Dict[str, Any]) -> str:
        """Determine stage for dataset based on manifest and heuristics"""
        stage = dataset_info.get("stage")
        if stage and stage.startswith("stage"):
            return stage

        # Use heuristics from manifest
        path = dataset_info.get("path", "").lower()
        name = dataset_info.get("name", "").lower()

        if any(x in path or x in name for x in ["edge", "crisis", "trauma", "suicidality"]):
            return STAGE3_ID
        if any(x in path or x in name for x in ["cot", "reasoning", "therapeutic_expertise"]):
            return STAGE2_ID
        if any(x in path or x in name for x in ["voice", "persona", "wayfarer", "tim_fletcher"]):
            return STAGE4_ID
        return STAGE1_ID

    def process_dataset(self, sourcing_result: Dict[str, Any], dataset_info: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Process a single dataset through pipeline"""
        if not sourcing_result.get("success"):
            logger.warning(f"Skipping {sourcing_result['name']}: {sourcing_result.get('error')}")
            return None

        file_path = Path(sourcing_result["path"])
        if not file_path.exists():
            logger.warning(f"File not found: {file_path}")
            return None

        stage = self.determine_stage(dataset_info)

        logger.info(f"Processing {sourcing_result['name']} -> {stage}")

        # Create data source
        source = DataSource(
            name=sourcing_result["name"],
            path=str(file_path),
            format=file_path.suffix[1:] if file_path.suffix else "jsonl",
            size_bytes=sourcing_result.get("size_bytes", 0),
            source_type=sourcing_result.get("source_type", "unknown"),
            stage=stage,
        )

        # Register with processor
        self.processor.register_data_source(source)

        return {
            "name": sourcing_result["name"],
            "path": str(file_path),
            "stage": stage,
            "source_type": sourcing_result.get("source_type"),
            "size_bytes": sourcing_result.get("size_bytes", 0),
        }

    def process_all_datasets(self) -> Dict[str, Any]:
        """Process all successfully sourced datasets"""
        logger.info("🔄 Processing datasets through unified preprocessing pipeline...")

        # Map sourcing results to manifest datasets
        manifest_datasets = {d["name"]: d for d in self.manifest.get("datasets", [])}

        for sourcing_result in self.sourcing_report.get("results", []):
            if not sourcing_result.get("success"):
                continue

            dataset_info = manifest_datasets.get(sourcing_result["name"], {})
            processed = self.process_dataset(sourcing_result, dataset_info)
            if processed:
                self.processed_datasets.append(processed)

        logger.info(f"  ✅ Processed {len(self.processed_datasets)} datasets")

        return {
            "processed_count": len(self.processed_datasets),
            "datasets": self.processed_datasets,
        }

    def generate_processing_report(self) -> Dict[str, Any]:
        """Generate processing report"""
        return {
            "timestamp": datetime.now().isoformat(),
            "processed_datasets": len(self.processed_datasets),
            "by_stage": {
                STAGE1_ID: sum(1 for d in self.processed_datasets if d.get("stage") == STAGE1_ID),
                STAGE2_ID: sum(1 for d in self.processed_datasets if d.get("stage") == STAGE2_ID),
                STAGE3_ID: sum(1 for d in self.processed_datasets if d.get("stage") == STAGE3_ID),
                STAGE4_ID: sum(1 for d in self.processed_datasets if d.get("stage") == STAGE4_ID),
            },
            "datasets": self.processed_datasets,
        }


def main():
    """Main function"""
    base_path = Path.cwd()
    manifest_path = base_path / "ai" / "training_ready" / "TRAINING_MANIFEST.json"
    sourcing_report_path = base_path / "ai" / "training_ready" / "scripts" / "output" / "sourcing_report.json"

    if not manifest_path.exists():
        logger.error(f"Manifest not found: {manifest_path}")
        return 1

    if not sourcing_report_path.exists():
        logger.error(f"Sourcing report not found: {sourcing_report_path}")
        logger.info("Please run source_datasets.py first")
        return 1

    logger.info("🔄 Starting dataset processing pipeline...")

    processor = DatasetProcessor(manifest_path, sourcing_report_path)
    processor.process_all_datasets()

    # Generate report
    report = processor.generate_processing_report()
    report_path = base_path / "ai" / "training_ready" / "scripts" / "output" / "processing_report.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)

    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)

    logger.info(f"\n📊 Processing Summary:")
    logger.info(f"  Processed: {report['processed_datasets']} datasets")
    logger.info(f"  Stage 1: {report['by_stage'][STAGE1_ID]}")
    logger.info(f"  Stage 2: {report['by_stage'][STAGE2_ID]}")
    logger.info(f"  Stage 3: {report['by_stage'][STAGE3_ID]}")
    logger.info(f"  Stage 4: {report['by_stage'][STAGE4_ID]}")
    logger.info(f"\n💾 Report saved to: {report_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())

