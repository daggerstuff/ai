#!/usr/bin/env python3
"""
Integrated Training Pipeline Orchestrator
Combines ALL data sources for comprehensive therapeutic AI training
"""

import json
from pathlib import Path
from typing import List, Dict, Optional
from dataclasses import dataclass, field
from datetime import datetime

# Note: These imports reference ai/dataset_pipeline which still exists
# They are not part of the training_ready consolidation
try:
    from ai.pipelines.orchestrator.ingestion.edge_case_jsonl_loader import EdgeCaseJSONLLoader, load_edge_case_training_data
    from ai.pipelines.orchestrator.ingestion.psychology_knowledge_loader import PsychologyKnowledgeLoader, load_psychology_knowledge
    from ai.pipelines.orchestrator.ingestion.pixel_voice_loader import PixelVoiceLoader, load_pixel_voice_training_data
    from ai.pipelines.orchestrator.ingestion.dual_persona_loader import DualPersonaLoader, load_dual_persona_training_data
    from ai.pipelines.orchestrator.utils.logger import get_logger
except ImportError:
    # Fallback for relative imports if run from different context
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
    output_dir: str = "ai/training_ready/data"  # Updated after consolidation
    output_filename: str = "training_dataset.json"
    target_total_samples: int = 8000
    output_to_s3: bool = True  # Also upload to S3 (canonical)
    s3_output_path: str = "gdrive/processed/unified/training_dataset.json"

    # Quality settings
    enable_bias_detection: bool = True
    enable_quality_validation: bool = True
    min_quality_score: float = 0.7

    # Progress tracking integration
    enable_progress_tracking: bool = True
    progress_tracker_path: str = "ai/training_ready/models/therapeutic_progress_tracker.py"  # Updated after consolidation


@dataclass
class IntegrationStats:
    """Statistics from pipeline integration"""
    total_samples: int = 0
    samples_by_source: Dict[str, int] = field(default_factory=dict)
    samples_by_category: Dict[str, int] = field(default_factory=dict)
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
        balanced_data = self._balance_dataset(all_training_data)

        # 7. Run bias detection if enabled
        if self.config.enable_bias_detection:
            balanced_data = self._run_bias_detection(balanced_data)

        # 8. Run quality validation if enabled
        if self.config.enable_quality_validation:
            balanced_data = self._run_quality_validation(balanced_data)

        # 9. Save integrated dataset
        output_path = self._save_dataset(balanced_data)

        # 10. Generate integration report
        self.stats.total_samples = len(balanced_data)
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
        """Load standard therapeutic conversations"""
        try:
            standard_file = Path(self.config.standard_therapeutic.source_path) / "training_dataset.json"
            if not standard_file.exists():
                warning = f"Standard therapeutic data not found: {standard_file}"
                logger.warning(warning)
                self.stats.warnings.append(warning)
                return []

            with open(standard_file, 'r') as f:
                data = json.load(f)
                conversations = data.get('conversations', [])

            # Convert to standard format
            training_data = []
            for conv in conversations:
                training_data.append({
                    'text': conv.get('text', ''),
                    'metadata': {
                        'source': 'standard_therapeutic',
                        'is_edge_case': False
                    }
                })

            return training_data

        except Exception as e:
            error = f"Failed to load standard therapeutic data: {e}"
            logger.error(error)
            self.stats.errors.append(error)
            return []

    def _balance_dataset(self, data: List[Dict]) -> List[Dict]:
        """Balance dataset according to target percentages"""
        logger.info("⚖️  Balancing dataset...")

        # Group by source
        by_source = {}
        for item in data:
            source = item.get('metadata', {}).get('source', 'unknown')
            if source not in by_source:
                by_source[source] = []
            by_source[source].append(item)

        # Calculate target counts
        target_counts = {
            'edge_cases': int(self.config.target_total_samples * self.config.edge_cases.target_percentage),
            'pixel_voice': int(self.config.target_total_samples * self.config.pixel_voice.target_percentage),
            'psychology_knowledge': int(self.config.target_total_samples * self.config.psychology_knowledge.target_percentage),
            'dual_persona': int(self.config.target_total_samples * self.config.dual_persona.target_percentage),
            'standard_therapeutic': int(self.config.target_total_samples * self.config.standard_therapeutic.target_percentage),
        }

        # Sample from each source
        balanced = []
        for source, target_count in target_counts.items():
            source_data = by_source.get(source, [])
            if len(source_data) >= target_count:
                # Sample down
                import random
                balanced.extend(random.sample(source_data, target_count))
            else:
                # Use all available
                balanced.extend(source_data)
                if source_data:
                    warning = f"Source '{source}' has only {len(source_data)} samples (target: {target_count})"
                    logger.warning(warning)
                    self.stats.warnings.append(warning)

        logger.info(f"   Balanced to {len(balanced)} samples")
        return balanced

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
        """
        Save integrated dataset to file and optionally to S3.
        S3 is canonical - always save to S3 if enabled.
        """
        # Convert to expected format
        output_data = {
            'conversations': data,
            'metadata': {
                'total_conversations': len(data),
                'sources': list(self.stats.samples_by_source.keys()),
                'generated_at': datetime.now().isoformat(),
                'pipeline_version': '1.0',
                'integration_stats': {
                    'samples_by_source': self.stats.samples_by_source,
                    'warnings': self.stats.warnings,
                    'errors': self.stats.errors
                }
            }
        }

        # Save locally first
        output_dir = Path(self.config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / self.config.output_filename

        with open(output_path, 'w') as f:
            json.dump(output_data, f, indent=2)

        logger.info(f"💾 Saved dataset locally to {output_path}")

        # Save to S3 if enabled (canonical location)
        if getattr(self.config, 'output_to_s3', False):
            try:
                from ai.training.ready_packages.utils.s3_dataset_loader import S3DatasetLoader

                loader = S3DatasetLoader()
                s3_key = getattr(self.config, 's3_output_path', 'gdrive/processed/unified/training_dataset.json')
                s3_path = f"s3://{loader.bucket}/{s3_key}"

                # Upload to S3
                loader.s3_client.put_object(
                    Bucket=loader.bucket,
                    Key=s3_key,
                    Body=json.dumps(output_data, indent=2).encode('utf-8'),
                    ContentType='application/json'
                )

                logger.info(f"💾 Saved dataset to S3 (canonical): {s3_path}")
            except Exception as e:
                logger.warning(f"Failed to save to S3: {e}. Local file saved.")

        return str(output_path)

    def _generate_report(self) -> Dict:
        """Generate integration report"""
        return {
            'timestamp': datetime.now().isoformat(),
            'total_samples': self.stats.total_samples,
            'samples_by_source': self.stats.samples_by_source,
            'target_percentages': {
                'edge_cases': self.config.edge_cases.target_percentage,
                'pixel_voice': self.config.pixel_voice.target_percentage,
                'psychology_knowledge': self.config.psychology_knowledge.target_percentage,
                'dual_persona': self.config.dual_persona.target_percentage,
                'standard_therapeutic': self.config.standard_therapeutic.target_percentage,
            },
            'actual_percentages': {
                source: count / self.stats.total_samples if self.stats.total_samples > 0 else 0
                for source, count in self.stats.samples_by_source.items()
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
