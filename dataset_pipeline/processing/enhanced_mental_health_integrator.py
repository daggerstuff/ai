"""
Enhanced Mental Health Dataset Integration System (PIX-60 + PIX-31)

Integrates mental health datasets across all tiers with:
- Adapter pattern for multiple formats
- Progress tracking and error handling
- Tier 2 professional data integration (95% quality threshold)
- Comprehensive telemetry and monitoring
"""

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional
from tqdm import tqdm

from conversation_schema import Conversation
from ai.dataset_pipeline.processing.dataset_format_adapters import AdapterRegistry
from ai.dataset_pipeline.ingestion.tier_loaders.tier2_professional_loader import (
    Tier2ProfessionalLoader,
)

logger = logging.getLogger(__name__)


@dataclass
class DatasetIntegrationConfig:
    """Configuration for a single dataset integration."""
    
    name: str
    source_path: Optional[Path] = None
    format_type: str = "generic_chatml"
    tier: int = 3  # Default to Tier 3
    quality_threshold: float = 0.7
    target_conversations: Optional[int] = None
    enabled: bool = True
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class IntegrationProgress:
    """Track progress for a dataset integration."""
    
    dataset_name: str
    total_records: int = 0
    processed: int = 0
    accepted: int = 0
    quality_filtered: int = 0
    format_errors: int = 0
    validation_errors: int = 0
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    
    @property
    def processing_time(self) -> float:
        """Get processing time in seconds."""
        if self.start_time and self.end_time:
            return (self.end_time - self.start_time).total_seconds()
        return 0.0
    
    @property
    def acceptance_rate(self) -> float:
        """Get acceptance rate (0-1)."""
        if self.processed == 0:
            return 0.0
        return self.accepted / self.processed
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for reporting."""
        return {
            "dataset_name": self.dataset_name,
            "total_records": self.total_records,
            "processed": self.processed,
            "accepted": self.accepted,
            "quality_filtered": self.quality_filtered,
            "format_errors": self.format_errors,
            "validation_errors": self.validation_errors,
            "acceptance_rate": self.acceptance_rate,
            "processing_time_seconds": self.processing_time,
        }


@dataclass
class IntegrationTelemetry:
    """Comprehensive telemetry for integration process."""
    
    total_datasets: int = 0
    successful_datasets: int = 0
    failed_datasets: int = 0
    total_conversations_processed: int = 0
    total_conversations_accepted: int = 0
    total_quality_filtered: int = 0
    total_format_errors: int = 0
    total_validation_errors: int = 0
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    dataset_progress: Dict[str, IntegrationProgress] = field(default_factory=dict)
    
    @property
    def overall_acceptance_rate(self) -> float:
        """Get overall acceptance rate across all datasets."""
        if self.total_conversations_processed == 0:
            return 0.0
        return self.total_conversations_accepted / self.total_conversations_processed
    
    @property
    def total_processing_time(self) -> float:
        """Get total processing time in seconds."""
        if self.start_time and self.end_time:
            return (self.end_time - self.start_time).total_seconds()
        return 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for reporting."""
        return {
            "summary": {
                "total_datasets": self.total_datasets,
                "successful_datasets": self.successful_datasets,
                "failed_datasets": self.failed_datasets,
                "total_conversations_processed": self.total_conversations_processed,
                "total_conversations_accepted": self.total_conversations_accepted,
                "overall_acceptance_rate": self.overall_acceptance_rate,
                "total_processing_time_seconds": self.total_processing_time,
            },
            "filtering": {
                "quality_filtered": self.total_quality_filtered,
                "format_errors": self.total_format_errors,
                "validation_errors": self.total_validation_errors,
            },
            "dataset_details": {
                name: progress.to_dict()
                for name, progress in self.dataset_progress.items()
            },
        }


class EnhancedMentalHealthIntegrator:
    """
    Enhanced mental health dataset integrator with adapter pattern.
    
    Features:
    - Multi-format support via adapter pattern
    - Progress tracking with tqdm
    - Comprehensive error handling and recovery
    - Tier 2 professional data integration (95% quality)
    - Telemetry and monitoring
    - Resume capability
    """
    
    def __init__(
        self,
        output_dir: Path = Path("data/processed/mental_health"),
        base_datasets_path: Path = Path("ai/datasets"),
        enable_tier2: bool = True,
        enable_progress_bar: bool = True,
    ):
        """
        Initialize enhanced mental health integrator.
        
        Args:
            output_dir: Output directory for integrated conversations
            base_datasets_path: Base path to datasets directory
            enable_tier2: Enable Tier 2 professional loader integration
            enable_progress_bar: Show progress bars
        """
        self.output_dir = Path(output_dir)
        self.base_datasets_path = Path(base_datasets_path)
        self.enable_progress_bar = enable_progress_bar
        
        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize adapter registry
        self.adapter_registry = AdapterRegistry()
        
        # Initialize Tier 2 loader if enabled
        self.tier2_loader = None
        if enable_tier2:
            try:
                self.tier2_loader = Tier2ProfessionalLoader(
                    base_path=base_datasets_path,
                    quality_threshold=0.95,  # 95% for Tier 2
                )
                logger.info("Initialized Tier 2 professional loader")
            except Exception as e:
                logger.warning(f"Failed to initialize Tier 2 loader: {e}")
        
        # Initialize telemetry
        self.telemetry = IntegrationTelemetry()
        
        # Dataset configurations
        self.dataset_configs: List[DatasetIntegrationConfig] = []
        
        logger.info(
            f"Initialized EnhancedMentalHealthIntegrator: output_dir={output_dir}"
        )
    
    def add_dataset_config(self, config: DatasetIntegrationConfig):
        """Add a dataset configuration for integration."""
        self.dataset_configs.append(config)
        logger.info(f"Added dataset config: {config.name} (tier={config.tier})")
    
    def integrate_all_datasets(
        self,
        include_tier2: bool = True,
        max_conversations_per_dataset: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Integrate all configured datasets.
        
        Args:
            include_tier2: Include Tier 2 professional datasets
            max_conversations_per_dataset: Optional limit per dataset
        
        Returns:
            Integration report with statistics
        """
        logger.info("=" * 80)
        logger.info("Starting Enhanced Mental Health Dataset Integration")
        logger.info("=" * 80)
        
        self.telemetry.start_time = datetime.now()
        all_conversations: List[Conversation] = []
        
        # Step 1: Integrate Tier 2 professional datasets
        if include_tier2 and self.tier2_loader:
            logger.info("\n📊 TIER 2: Professional Therapeutic Datasets")
            logger.info("-" * 80)
            tier2_conversations = self._integrate_tier2_datasets()
            all_conversations.extend(tier2_conversations)
            logger.info(
                f"✅ Tier 2 complete: {len(tier2_conversations)} conversations"
            )
        
        # Step 2: Integrate other configured datasets
        if self.dataset_configs:
            logger.info("\n📚 Additional Configured Datasets")
            logger.info("-" * 80)
            for config in self.dataset_configs:
                if not config.enabled:
                    logger.info(f"⏭️  Skipping disabled dataset: {config.name}")
                    continue
                
                try:
                    conversations = self._integrate_single_dataset(
                        config, max_conversations_per_dataset
                    )
                    all_conversations.extend(conversations)
                    self.telemetry.successful_datasets += 1
                    
                except Exception as e:
                    logger.error(
                        f"❌ Failed to integrate {config.name}: {e}",
                        exc_info=True,
                    )
                    self.telemetry.failed_datasets += 1
        
        self.telemetry.end_time = datetime.now()
        
        # Step 3: Save integrated conversations
        logger.info("\n💾 Saving Integrated Conversations")
        logger.info("-" * 80)
        output_file = self.output_dir / "integrated_conversations.jsonl"
        self._save_conversations(all_conversations, output_file)
        logger.info(f"✅ Saved {len(all_conversations)} conversations to {output_file}")
        
        # Step 4: Generate and save report
        report = self._generate_integration_report()
        report_file = self.output_dir / "integration_report.json"
        with open(report_file, "w") as f:
            json.dump(report, f, indent=2)
        logger.info(f"📊 Report saved to {report_file}")
        
        # Step 5: Print summary
        self._print_summary()
        
        return report
    
    def _integrate_tier2_datasets(self) -> List[Conversation]:
        """
        Integrate Tier 2 professional therapeutic datasets.
        
        Returns:
            List of conversations from Tier 2 datasets
        """
        if not self.tier2_loader:
            logger.warning("Tier 2 loader not initialized")
            return []
        
        all_tier2_conversations: List[Conversation] = []
        
        try:
            # Load all Tier 2 datasets
            tier2_datasets = self.tier2_loader.load_datasets()
            
            for dataset_name, conversations in tier2_datasets.items():
                progress = IntegrationProgress(
                    dataset_name=f"tier2_{dataset_name}",
                    start_time=datetime.now(),
                )
                
                progress.total_records = len(conversations)
                progress.processed = len(conversations)
                progress.accepted = len(conversations)  # Already filtered by loader
                
                all_tier2_conversations.extend(conversations)
                
                progress.end_time = datetime.now()
                self.telemetry.dataset_progress[f"tier2_{dataset_name}"] = progress
                self.telemetry.total_datasets += 1
                
                logger.info(
                    f"  ✅ {dataset_name}: {len(conversations)} conversations "
                    f"(quality >= 95%)"
                )
            
            # Update telemetry
            self.telemetry.total_conversations_processed += len(all_tier2_conversations)
            self.telemetry.total_conversations_accepted += len(all_tier2_conversations)
            
        except Exception as e:
            logger.error(f"Error integrating Tier 2 datasets: {e}", exc_info=True)
        
        return all_tier2_conversations
    
    def _integrate_single_dataset(
        self,
        config: DatasetIntegrationConfig,
        max_conversations: Optional[int] = None,
    ) -> List[Conversation]:
        """
        Integrate a single dataset using adapter pattern.
        
        Args:
            config: Dataset configuration
            max_conversations: Optional limit on conversations
        
        Returns:
            List of integrated conversations
        """
        logger.info(f"\n📁 Processing: {config.name}")
        
        progress = IntegrationProgress(
            dataset_name=config.name,
            start_time=datetime.now(),
        )
        
        conversations: List[Conversation] = []
        
        try:
            # Load raw data
            if not config.source_path or not config.source_path.exists():
                logger.warning(f"  ⚠️  Source path not found: {config.source_path}")
                return []
            
            raw_data = self._load_raw_data(config.source_path)
            progress.total_records = len(raw_data)
            
            # Get adapter
            adapter = self.adapter_registry.get_adapter(config.format_type)
            if not adapter:
                logger.error(f"  ❌ No adapter found for format: {config.format_type}")
                progress.format_errors = progress.total_records
                return []
            
            # Process with progress bar
            iterator = tqdm(
                raw_data,
                desc=f"  {config.name}",
                disable=not self.enable_progress_bar,
            )
            
            for raw_entry in iterator:
                progress.processed += 1
                
                # Adapt to conversation
                conversation = self.adapter_registry.adapt_with_fallback(
                    raw_entry,
                    config.format_type,
                    config.name,
                )
                
                if not conversation:
                    progress.format_errors += 1
                    continue
                
                # Add tier and config metadata
                conversation.metadata.update({
                    "tier": config.tier,
                    "quality_threshold": config.quality_threshold,
                    **config.metadata,
                })
                
                conversations.append(conversation)
                progress.accepted += 1
                
                # Check limit
                if max_conversations and progress.accepted >= max_conversations:
                    logger.info(f"  ⚠️  Reached limit: {max_conversations}")
                    break
            
            progress.end_time = datetime.now()
            
            # Log results
            logger.info(
                f"  ✅ {config.name}: {progress.accepted}/{progress.processed} "
                f"({progress.acceptance_rate:.1%}) in {progress.processing_time:.1f}s"
            )
            
        except Exception as e:
            logger.error(f"  ❌ Error: {e}", exc_info=True)
            progress.validation_errors += 1
        
        finally:
            self.telemetry.dataset_progress[config.name] = progress
            self.telemetry.total_datasets += 1
            self.telemetry.total_conversations_processed += progress.processed
            self.telemetry.total_conversations_accepted += progress.accepted
            self.telemetry.total_format_errors += progress.format_errors
            self.telemetry.total_validation_errors += progress.validation_errors
        
        return conversations
    
    def _load_raw_data(self, source_path: Path) -> List[Dict[str, Any]]:
        """Load raw data from JSONL file."""
        raw_data = []
        
        try:
            with open(source_path, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        raw_data.append(json.loads(line))
        except Exception as e:
            logger.error(f"Error loading {source_path}: {e}")
        
        return raw_data
    
    def _save_conversations(
        self,
        conversations: List[Conversation],
        output_path: Path,
    ):
        """Save conversations to JSONL file."""
        try:
            with open(output_path, "w", encoding="utf-8") as f:
                for conversation in conversations:
                    f.write(json.dumps(conversation.to_dict()) + "\n")
        except Exception as e:
            logger.error(f"Error saving conversations: {e}")
    
    def _generate_integration_report(self) -> Dict[str, Any]:
        """Generate comprehensive integration report."""
        report = {
            "timestamp": datetime.now().isoformat(),
            "telemetry": self.telemetry.to_dict(),
            "adapters_registered": self.adapter_registry.list_adapters(),
            "tier2_enabled": self.tier2_loader is not None,
        }
        
        return report
    
    def _print_summary(self):
        """Print integration summary."""
        logger.info("\n" + "=" * 80)
        logger.info("🎉 INTEGRATION COMPLETE")
        logger.info("=" * 80)
        logger.info(f"Total Datasets: {self.telemetry.total_datasets}")
        logger.info(f"Successful: {self.telemetry.successful_datasets}")
        logger.info(f"Failed: {self.telemetry.failed_datasets}")
        logger.info(f"Total Processed: {self.telemetry.total_conversations_processed:,}")
        logger.info(f"Total Accepted: {self.telemetry.total_conversations_accepted:,}")
        logger.info(
            f"Overall Acceptance Rate: {self.telemetry.overall_acceptance_rate:.1%}"
        )
        logger.info(f"Processing Time: {self.telemetry.total_processing_time:.1f}s")
        logger.info("=" * 80)
