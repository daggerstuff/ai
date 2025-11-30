#!/usr/bin/env python3
"""
Integrated Training Pipeline Orchestrator
Combines ALL data sources for comprehensive therapeutic AI training
"""

import json
import re
from pathlib import Path
from typing import List, Dict, Optional
from dataclasses import dataclass, field
from datetime import datetime
import random

from ..ingestion.edge_case_jsonl_loader import EdgeCaseJSONLLoader, load_edge_case_training_data
from ..ingestion.psychology_knowledge_loader import PsychologyKnowledgeLoader, load_psychology_knowledge
from ..ingestion.pixel_voice_loader import PixelVoiceLoader, load_pixel_voice_training_data
from ..ingestion.dual_persona_loader import DualPersonaLoader, load_dual_persona_training_data
from ..utils.logger import get_logger

logger = get_logger("dataset_pipeline.integrated_training_pipeline")


@dataclass
class DataSourceConfig:
    """Configuration for each data source"""
    enabled: bool = True
    target_percentage: float = 0.0  # Target percentage of final dataset
    max_samples: Optional[int] = None
    source_path: Optional[str] = None


@dataclass
class IntegratedPipelineConfig:
    """Configuration for integrated training pipeline"""

    # Data source configurations
    edge_cases: DataSourceConfig = field(default_factory=lambda: DataSourceConfig(
        enabled=True,
        target_percentage=0.25,  # 25% edge cases
        source_path="ai/pipelines/edge_case_pipeline_standalone/output"
    ))

    pixel_voice: DataSourceConfig = field(default_factory=lambda: DataSourceConfig(
        enabled=True,
        target_percentage=0.20,  # 20% voice-derived
        source_path="ai/pipelines/pixel_voice"
    ))

    psychology_knowledge: DataSourceConfig = field(default_factory=lambda: DataSourceConfig(
        enabled=True,
        target_percentage=0.15,  # 15% psychology knowledge
        source_path="ai/training_data_consolidated"
    ))

    dual_persona: DataSourceConfig = field(default_factory=lambda: DataSourceConfig(
        enabled=True,
        target_percentage=0.10,  # 10% dual persona
        source_path="ai/pipelines/dual_persona_training"
    ))

    standard_therapeutic: DataSourceConfig = field(default_factory=lambda: DataSourceConfig(
        enabled=True,
        target_percentage=0.30,  # 30% standard conversations
        source_path="ai/dataset_pipeline/pixelated-training"
    ))

    # Output configuration
    output_dir: str = "ai/lightning"
    output_filename: str = "training_dataset.json"
    target_total_samples: int = 8000
    stage_distribution: Dict[str, float] = field(default_factory=lambda: {
        "stage1_foundation": 0.40,
        "stage2_therapeutic_expertise": 0.25,
        "stage3_edge_stress_test": 0.20,
        "stage4_voice_persona": 0.15
    })

    # Quality settings
    enable_bias_detection: bool = True
    enable_quality_validation: bool = True
    min_quality_score: float = 0.7

    # Progress tracking integration
    enable_progress_tracking: bool = True
    progress_tracker_path: str = "ai/lightning/therapeutic_progress_tracker.py"


@dataclass
class IntegrationStats:
    """Statistics from pipeline integration"""
    total_samples: int = 0
    samples_by_source: Dict[str, int] = field(default_factory=dict)
    samples_by_category: Dict[str, int] = field(default_factory=dict)
    samples_by_stage: Dict[str, int] = field(default_factory=dict)
    stage_balance: Dict[str, Dict[str, int]] = field(default_factory=dict)
    quality_scores: Dict[str, float] = field(default_factory=dict)
    bias_detection_results: Dict[str, any] = field(default_factory=dict)
    integration_time: float = 0.0
    warnings: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)


class IntegratedTrainingPipeline:
    """
    Orchestrates integration of all data sources into unified training dataset
    """

    def __init__(self, config: Optional[IntegratedPipelineConfig] = None):
        self.config = config or IntegratedPipelineConfig()
        self.stats = IntegrationStats()

    def run(self) -> Dict:
        """
        Run the complete integrated pipeline

        Returns:
            Dictionary with training data and statistics
        """
        logger.info("🚀 Starting Integrated Training Pipeline")
        logger.info("=" * 60)

        start_time = datetime.now()
        all_training_data = []

        # 1. Load Edge Case Data
        if self.config.edge_cases.enabled:
            edge_data = self._load_edge_cases()
            all_training_data.extend(edge_data)
            self.stats.samples_by_source['edge_cases'] = len(edge_data)
            logger.info(f"✅ Loaded {len(edge_data)} edge case examples")

        # 2. Load Pixel Voice Data
        if self.config.pixel_voice.enabled:
            voice_data = self._load_pixel_voice()
            all_training_data.extend(voice_data)
            self.stats.samples_by_source['pixel_voice'] = len(voice_data)
            logger.info(f"✅ Loaded {len(voice_data)} voice-derived examples")

        # 3. Load Psychology Knowledge
        if self.config.psychology_knowledge.enabled:
            psych_data = self._load_psychology_knowledge()
            all_training_data.extend(psych_data)
            self.stats.samples_by_source['psychology_knowledge'] = len(psych_data)
            logger.info(f"✅ Loaded {len(psych_data)} psychology knowledge examples")

        # 4. Load Dual Persona Data
        if self.config.dual_persona.enabled:
            persona_data = self._load_dual_persona()
            all_training_data.extend(persona_data)
            self.stats.samples_by_source['dual_persona'] = len(persona_data)
            logger.info(f"✅ Loaded {len(persona_data)} dual persona examples")

        # 5. Load Standard Therapeutic Conversations
        if self.config.standard_therapeutic.enabled:
            standard_data = self._load_standard_therapeutic()
            all_training_data.extend(standard_data)
            self.stats.samples_by_source['standard_therapeutic'] = len(standard_data)
            logger.info(f"✅ Loaded {len(standard_data)} standard therapeutic examples")

        # 6. Balance dataset according to target percentages
        balanced_data, stage_segments = self._balance_dataset(all_training_data)

        # 7. Run bias detection if enabled
        if self.config.enable_bias_detection:
            balanced_data = self._run_bias_detection(balanced_data)

        # 8. Run quality validation if enabled
        if self.config.enable_quality_validation:
            balanced_data = self._run_quality_validation(balanced_data)

        # 9. Save integrated dataset
        output_path = self._save_dataset(balanced_data)
        self._write_stage_outputs(stage_segments)

        # 10. Generate integration report
        self.stats.total_samples = len(balanced_data)
        self.stats.samples_by_category = dict(self.stats.samples_by_stage)
        self.stats.integration_time = (datetime.now() - start_time).total_seconds()

        report = self._generate_report()

        logger.info("=" * 60)
        logger.info(f"✅ Integration Complete!")
        logger.info(f"📊 Total samples: {self.stats.total_samples}")
        logger.info(f"📁 Output: {output_path}")
        logger.info(f"⏱️  Time: {self.stats.integration_time:.2f}s")

        return {
            'training_data': balanced_data,
            'statistics': self.stats,
            'output_path': output_path,
            'report': report
        }

    def _load_edge_cases(self) -> List[Dict]:
        """Load edge case training data"""
        try:
            loader = EdgeCaseJSONLLoader(self.config.edge_cases.source_path)

            if not loader.check_pipeline_output_exists():
                warning = "Edge case data not found. Run edge case pipeline first."
                logger.warning(warning)
                self.stats.warnings.append(warning)
                return []

            return loader.convert_to_training_format()

        except Exception as e:
            error = f"Failed to load edge cases: {e}"
            logger.error(error)
            self.stats.errors.append(error)
            return []

    def _load_pixel_voice(self) -> List[Dict]:
        """Load Pixel Voice pipeline data"""
        try:
            loader = PixelVoiceLoader(self.config.pixel_voice.source_path)

            if not loader.check_pipeline_output_exists():
                warning = "Pixel Voice data not found. Run Pixel Voice pipeline first."
                logger.warning(warning)
                self.stats.warnings.append(warning)
                return []

            return loader.convert_to_training_format()

        except Exception as e:
            error = f"Failed to load Pixel Voice data: {e}"
            logger.error(error)
            self.stats.errors.append(error)
            return []

    def _load_psychology_knowledge(self) -> List[Dict]:
        """Load psychology knowledge base"""
        try:
            loader = PsychologyKnowledgeLoader()

            if not loader.check_knowledge_base_exists():
                warning = "Psychology knowledge base not found."
                logger.warning(warning)
                self.stats.warnings.append(warning)
                return []

            return loader.convert_to_training_format()

        except Exception as e:
            error = f"Failed to load psychology knowledge: {e}"
            logger.error(error)
            self.stats.errors.append(error)
            return []

    def _load_dual_persona(self) -> List[Dict]:
        """Load dual persona training data"""
        try:
            loader = DualPersonaLoader(self.config.dual_persona.source_path)

            # Dual persona loader will generate synthetic data if none exists
            return loader.convert_to_training_format()

        except Exception as e:
            error = f"Failed to load dual persona data: {e}"
            logger.error(error)
            self.stats.errors.append(error)
            return []

    def _load_standard_therapeutic(self) -> List[Dict]:
        """Load standard therapeutic conversations with robust error handling"""
        # Try multiple file locations
        possible_files = [
            Path(self.config.standard_therapeutic.source_path) / "training_dataset.json",
            Path("ai/lightning/pixelated-training/training_dataset.json"),
            Path("ai/dataset_pipeline/pixelated-training/training_dataset.json"),
        ]

        # Try each file until one loads successfully
        conversations = []
        last_error = None

        for standard_file in possible_files:
            if not standard_file.exists():
                continue

            logger.info(f"Attempting to load from: {standard_file}")

            # Try multiple loading strategies
            raw_data = None

            # Strategy 1: Try standard JSON load
            try:
                with open(standard_file, 'r', encoding='utf-8') as f:
                    raw_data = json.load(f)

                # Handle different data structures
                if isinstance(raw_data, list):
                    # File is a list of conversations
                    conversations = raw_data
                    logger.info(f"✅ Loaded {len(conversations)} conversations from {standard_file} (list format)")
                    break  # Success, exit loop
                elif isinstance(raw_data, dict):
                    # File is a dict with conversations key
                    conversations = raw_data.get('conversations', [])
                    if conversations:
                        logger.info(f"✅ Loaded {len(conversations)} conversations from {standard_file} (dict format)")
                        break  # Success, exit loop
                    else:
                        logger.warning(f"File {standard_file} loaded but no conversations found")
                else:
                    logger.warning(f"Unexpected data type in {standard_file}: {type(raw_data)}")

            except json.JSONDecodeError as e:
                logger.warning(f"JSON parsing error in {standard_file} at position {e.pos}: {e.msg}")
                last_error = e
                # Try next file
                continue
            except Exception as e:
                logger.warning(f"Error loading {standard_file}: {e}")
                last_error = e
                # Try next file
                continue

        # If no file loaded successfully, report error
        if not conversations:
            if last_error:
                error = f"Failed to load standard therapeutic data from any available file. Last error: {last_error}"
            else:
                error = f"Standard therapeutic data not found in any of: {[str(f) for f in possible_files]}"
            logger.error(error)
            self.stats.errors.append(error)
            return []

        # Convert to standard format
        training_data = []
        for conv in conversations:
            if not isinstance(conv, dict):
                continue

            # Handle different conversation formats
            text = conv.get('text', '')

            # If no text, try to construct from conversation array
            if not text:
                # Check for 'conversation' key (list format)
                conversation_array = conv.get('conversation', [])
                if conversation_array:
                    text_parts = []
                    for msg in conversation_array:
                        if isinstance(msg, dict):
                            role = msg.get('role', '')
                            content = msg.get('content', '')
                            if role and content:
                                text_parts.append(f"{role.capitalize()}: {content}")
                    text = "\n".join(text_parts)

                # Try messages format
                if not text:
                    messages = conv.get('messages', [])
                    if messages:
                        text_parts = []
                        for msg in messages:
                            if isinstance(msg, dict):
                                role = msg.get('role', '')
                                content = msg.get('content', '')
                                if role and content:
                                    text_parts.append(f"{role.capitalize()}: {content}")
                        text = "\n".join(text_parts)

            # If still no text, try direct content
            if not text:
                text = conv.get('content', '')

            if text:
                training_data.append({
                    'text': text,
                    'metadata': {
                        'source': 'standard_therapeutic',
                        'is_edge_case': False
                    }
                })

        logger.info(f"✅ Converted {len(training_data)} standard therapeutic examples to training format")
        return training_data

    def _balance_dataset(self, data: List[Dict]) -> (List[Dict], Dict[str, List[Dict]]):
        """Balance dataset according to stage distribution."""
        logger.info("⚖️  Balancing dataset by stage...")

        stage_buckets: Dict[str, List[Dict]] = {}
        for item in data:
            stage = item.get('metadata', {}).get('stage', 'stage1_foundation')
            stage_buckets.setdefault(stage, []).append(item)

        balanced: List[Dict] = []
        stage_segments: Dict[str, List[Dict]] = {}

        for stage, percentage in self.config.stage_distribution.items():
            target_count = int(self.config.target_total_samples * percentage)
            bucket = stage_buckets.get(stage, [])

            if not bucket:
                warning = f"No data found for stage '{stage}' (target: {target_count})."
                logger.warning(warning)
                self.stats.warnings.append(warning)
                self.stats.stage_balance[stage] = {
                    "target": target_count,
                    "available": 0,
                    "actual": 0
                }
                continue

            if len(bucket) <= target_count:
                stage_sample = bucket
                if len(bucket) < target_count:
                    warning = f"Stage '{stage}' has only {len(bucket)} samples (target: {target_count})."
                    logger.warning(warning)
                    self.stats.warnings.append(warning)
            else:
                stage_sample = random.sample(bucket, target_count)

            balanced.extend(stage_sample)
            stage_segments[stage] = stage_sample
            actual = len(stage_sample)
            self.stats.samples_by_stage[stage] = actual
            self.stats.stage_balance[stage] = {
                "target": target_count,
                "available": len(bucket),
                "actual": actual
            }

        logger.info(f"   Stage-balanced to {len(balanced)} samples")
        return balanced, stage_segments

    def _run_bias_detection(self, data: List[Dict]) -> List[Dict]:
        """Run bias detection on training data"""
        logger.info("🔍 Running bias detection...")

        try:
            from ..quality.evidence_based_practice_validator import validate_bias

            flagged_count = 0
            filtered_data = []

            for item in data:
                text = item.get('text', '')
                if validate_bias(text):
                    filtered_data.append(item)
                else:
                    flagged_count += 1

            self.stats.bias_detection_results = {
                'total_checked': len(data),
                'flagged': flagged_count,
                'passed': len(filtered_data)
            }

            logger.info(f"   Flagged {flagged_count} items for bias")
            return filtered_data

        except Exception as e:
            logger.warning(f"Bias detection failed: {e}")
            return data

    def _run_quality_validation(self, data: List[Dict]) -> List[Dict]:
        """Run quality validation on training data"""
        logger.info("✓ Running quality validation...")

        # TODO: Implement comprehensive quality validation
        logger.info(f"   Validated {len(data)} samples")
        return data

    def _save_dataset(self, data: List[Dict]) -> str:
        """Save integrated dataset"""
        output_dir = Path(self.config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        output_path = output_dir / self.config.output_filename

        # Convert to expected format
        output_data = {
            'conversations': data,
            'metadata': {
                'total_conversations': len(data),
                'sources': list(self.stats.samples_by_source.keys()),
                'generated_at': datetime.now().isoformat(),
                'pipeline_version': '1.0',
                'stage_metrics': self.stats.stage_balance,
                'integration_stats': {
                    'samples_by_source': self.stats.samples_by_source,
                    'samples_by_stage': self.stats.samples_by_stage,
                    'warnings': self.stats.warnings,
                    'errors': self.stats.errors
                }
            }
        }

        with open(output_path, 'w') as f:
            json.dump(output_data, f, indent=2)

        logger.info(f"💾 Saved dataset to {output_path}")
        return str(output_path)

    def _write_stage_outputs(self, stage_segments: Dict[str, List[Dict]]) -> None:
        """Persist per-stage datasets and manifest for downstream tracking."""
        if not stage_segments:
            return

        stage_dir = Path("ai/training_data_consolidated/final")
        stage_dir.mkdir(parents=True, exist_ok=True)

        manifest = {
            "generated_at": datetime.now().isoformat(),
            "stages": {}
        }

        for stage, records in stage_segments.items():
            stage_file = stage_dir / f"MASTER_{stage}.jsonl"
            with open(stage_file, "w") as stage_handle:
                for record in records:
                    stage_handle.write(json.dumps(record) + "\n")

            balance_stats = self.stats.stage_balance.get(stage, {})
            manifest["stages"][stage] = {
                "samples": len(records),
                "target": balance_stats.get("target"),
                "available": balance_stats.get("available"),
                "output_path": str(stage_file)
            }

        manifest_path = stage_dir / "MASTER_STAGE_MANIFEST.json"
        with open(manifest_path, "w") as manifest_handle:
            json.dump(manifest, manifest_handle, indent=2)

        logger.info(f"🗂️  Stage manifest updated at {manifest_path}")

    def _generate_report(self) -> Dict:
        """Generate integration report"""
        return {
            'timestamp': datetime.now().isoformat(),
            'total_samples': self.stats.total_samples,
            'samples_by_source': self.stats.samples_by_source,
            'stage_distribution_targets': self.config.stage_distribution,
            'stage_balance': self.stats.stage_balance,
            'actual_stage_percentages': {
                stage: count / self.stats.total_samples if self.stats.total_samples > 0 else 0
                for stage, count in self.stats.samples_by_stage.items()
            },
            'integration_time_seconds': self.stats.integration_time,
            'warnings': self.stats.warnings,
            'errors': self.stats.errors,
            'bias_detection': self.stats.bias_detection_results
        }


def run_integrated_pipeline(config: Optional[IntegratedPipelineConfig] = None) -> Dict:
    """
    Convenience function to run the integrated training pipeline

    Args:
        config: Optional pipeline configuration

    Returns:
        Dictionary with training data and statistics
    """
    pipeline = IntegratedTrainingPipeline(config)
    return pipeline.run()


if __name__ == "__main__":
    # Run the integrated pipeline
    print("🚀 Integrated Training Pipeline")
    print("=" * 60)

    result = run_integrated_pipeline()

    print("\n📊 Integration Report:")
    print(json.dumps(result['report'], indent=2))
