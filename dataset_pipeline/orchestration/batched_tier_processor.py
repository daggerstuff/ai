"""
Batched Tier Processor

Extension of TierProcessor to support "hotswapping" of datasets
(Load -> Process -> Offload -> Delete), allowing large-scale processing on
systems with limited storage (e.g. VPS).
"""

import json
import logging
import os
import shutil
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

# Hack to fix import paths for conversation_schema
# This assumes we are running from the workspace root or similar
# We attempt to find 'ai/dataset_pipeline/schemas' and add it to path
current_dir = Path(os.getcwd())
schemas_dir = current_dir / "ai" / "dataset_pipeline" / "schemas"
if schemas_dir.exists():
    sys.path.append(str(schemas_dir))

from ai.dataset_pipeline.orchestration.tier_processor import (  # noqa: E402
    TierProcessor,
)
from ai.dataset_pipeline.processing.personality_adapter import (  # noqa: E402
    CommunicationStyle,
    PersonalityAdapter,
)
from ai.dataset_pipeline.storage_config import (  # noqa: E402
    StorageBackend,
    get_storage_config,
)

logger = logging.getLogger(__name__)


class BatchedTierProcessor(TierProcessor):
    """
    Processor that handles one tier at a time, protecting memory and disk space.
    """

    def __init__(self, target_persona: Optional[str] = None, **kwargs):
        super().__init__(**kwargs)
        self.storage_config = get_storage_config()
        self.output_dir = self.storage_config.local_base_path / "processed_batches"
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Initialize personality adapter if a target persona is specified
        self.target_persona = target_persona
        if target_persona:
            self.personality_adapter = PersonalityAdapter()
            logger.info(
                f"BatchedProcessor initialized with target persona: {target_persona}"
            )

    def process_and_offload_all_tiers(self, cleanup_raw: bool = True):
        """
        Process all enabled tiers sequentially, offloading results to disk/S3
        and cleaning up raw data to save space ("hotswapping").
        """
        logger.info("Starting Batched Processing (Hotswap Mode)")

        results_manifest = {}

        # Iterate through tiers 1 to 6 (if enabled)
        for tier_num in sorted(self.tier_loaders.keys()):
            logger.info(f"=== Starting Batch for Tier {tier_num} ===")

            # 1. Load & Process (Downloads raw data if needed)
            datasets = self.process_tier(tier_num)

            if not datasets:
                logger.warning(f"No data found for Tier {tier_num}. Skipping.")
                continue

            # 2. Apply Personality Adaptation (if configured)
            if self.target_persona:
                self._apply_persona_to_datasets(datasets)

            # 3. Offload (Save Processed Data)
            output_file = self._offload_tier_data(tier_num, datasets)
            results_manifest[tier_num] = str(output_file)

            # 4. Memory Cleanup
            self._clear_tier_memory(tier_num)

            # 5. Disk Cleanup (Delete Raw Data)
            if cleanup_raw:
                self._cleanup_tier_raw_data(tier_num)

            logger.info(f"=== Completed Batch for Tier {tier_num} ===")

        logger.info("All tiers processed and offloaded.")
        return results_manifest

    # Mapping for descriptive filenames
    TIER_NAMES = {
        1: "priority_curated",
        2: "professional_therapeutic",
        3: "clinical_cot",
        4: "social_context",
        5: "academic_research",
        6: "knowledge_base",
    }

    def _apply_persona_to_datasets(self, datasets: Dict[str, List[Any]]):
        """Apply the target persona to all conversations in the datasets."""
        logger.info(f"Applying persona '{self.target_persona}' to batch...")

        style = None
        if self.target_persona == "dark_humor":
            style = CommunicationStyle.DARK_HUMOR

        if not style:
            logger.warning(
                f"Unknown persona '{self.target_persona}'. Skipping adaptation."
            )
            return

        for dataset_name, conversations in datasets.items():
            for conv in conversations:
                # We need to simulate an adaptation context
                # For now, we force the style on the assistant's responses

                # Create a dummy adaptation object with the target style
                # We reuse the adapter's logic but force the style
                adaptation = self.personality_adapter._get_default_adaptation()
                adaptation.communication_style = style

                for msg in conv.messages:
                    if msg.role == "assistant":
                        result = self.personality_adapter.adapt_response(
                            msg.content, adaptation
                        )
                        msg.content = result.adapted_response

                        # Add metadata about adaptation
                        if not conv.metadata.get("adaptation_applied"):
                            conv.metadata["adaptation_applied"] = []
                        conv.metadata["adaptation_applied"].append(self.target_persona)

    def _offload_tier_data(self, tier_num: int, datasets: Dict[str, List[Any]]) -> Path:
        """Save processed tier data to a JSONL file and return path."""
        # Generate descriptive filename
        tier_name = self.TIER_NAMES.get(tier_num, "generic")
        persona_suffix = f"_{self.target_persona}" if self.target_persona else ""
        filename = f"pixelated_tier{tier_num}_{tier_name}{persona_suffix}.jsonl"

        output_file = self.output_dir / filename
        logger.info(f"Offloading Tier {tier_num} data to {output_file}...")

        count = 0
        try:
            with open(output_file, "w", encoding="utf-8") as f:
                for dataset_name, conversations in datasets.items():
                    for conv in conversations:
                        # Add tracking of source dataset if not present
                        if "dataset_name" not in conv.metadata:
                            conv.metadata["dataset_name"] = dataset_name

                        f.write(json.dumps(conv.to_dict()) + "\n")
                        count += 1

            logger.info(f"Successfully wrote {count} conversations to {output_file}")

            # Upload to S3 if a bucket is configured (regardless of backend mode setting)
            # This ensures we always backup to S3 if possible
            if self.storage_config.s3_bucket:
                s3_key = f"processed/{filename}"
                self._upload_to_s3(output_file, s3_key)
            elif self.storage_config.backend == StorageBackend.S3:
                # Fallback if backend says S3 but bucket might be missing from this check?
                # (Should be caught by valid check, but good for safety)
                logger.warning("StorageBackend is S3 but s3_bucket seems missing?")

            return output_file

        except Exception as e:
            logger.error(f"Failed to offload Tier {tier_num} data: {e}")
            raise

    def _clear_tier_memory(self, tier_num: int):
        """Clear the in-memory processed datasets for a tier."""
        if tier_num in self.processed_datasets:
            del self.processed_datasets[tier_num]

        # Also force garbage collection if needed, though Python assumes auto
        import gc

        gc.collect()
        logger.info(f"Cleared memory for Tier {tier_num}")

    def _cleanup_tier_raw_data(self, tier_num: int):
        """
        Delete the raw downloaded files for this tier.
        WARNING: This is destructive.
        """
        loader = self.tier_loaders.get(tier_num)
        if not loader or not loader.base_path:
            logger.warning(f"Cannot cleanup Tier {tier_num}: No base path found.")
            return

        target_dir = loader.base_path

        if str(target_dir).strip() in ["/", "/home", "/usr", "."]:
            logger.error(f"Refusing to delete unsafe path: {target_dir}")
            return

        if target_dir.exists():
            logger.info(f"Cleaning up raw data at {target_dir}...")
            try:
                # Remove specific large files or the whole directory?
                # Loaders usually own their specific directory.
                # Use shutil.rmtree for directories
                if target_dir.is_dir():
                    # Instead of deleting the folder (which might break mount points),
                    # let's delete contents.
                    for item in target_dir.iterdir():
                        if item.is_file():
                            item.unlink()
                        elif item.is_dir():
                            shutil.rmtree(item)
                else:
                    target_dir.unlink()

                logger.info(f"Cleaned up raw data for Tier {tier_num}")
            except Exception as e:
                logger.error(f"Failed to cleanup raw data at {target_dir}: {e}")

    def _upload_to_s3(self, local_file: Path, s3_key: str):
        """Upload file to S3 if configured."""
        try:
            import boto3

            s3 = boto3.client(
                "s3",
                endpoint_url=self.storage_config.s3_endpoint_url,
                aws_access_key_id=self.storage_config.s3_access_key_id,
                aws_secret_access_key=self.storage_config.s3_secret_access_key,
                region_name=self.storage_config.s3_region,
            )
            bucket = self.storage_config.s3_bucket
            if bucket:
                logger.info(f"Uploading to S3: s3://{bucket}/{s3_key}")
                s3.upload_file(str(local_file), bucket, s3_key)
        except Exception as e:
            logger.warning(f"Failed to upload to S3: {e}")


if __name__ == "__main__":
    import argparse
    import sys

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    parser = argparse.ArgumentParser(description="Run Batched Tier Processing")
    parser.add_argument(
        "--persona",
        type=str,
        help="Target persona to apply (e.g., 'dark_humor')",
        default=None,
    )
    parser.add_argument(
        "--no-cleanup",
        action="store_true",
        help="Disable cleanup of raw data (requires more disk space)",
    )
    args = parser.parse_args()

    # Initialize processor with ALL tiers enabled
    processor = BatchedTierProcessor(
        target_persona=args.persona,
        enable_tier_1=True,
        enable_tier_2=True,
        enable_tier_3=True,
        enable_tier_4=True,
        enable_tier_5=True,
        enable_tier_6=True,
    )

    logger.info(
        f"Initialized BatchedTierProcessor for ALL TIERS (1-6). "
        f"Persona: {args.persona or 'None'}"
    )

    # Validate configuration
    if not processor.storage_config.validate()[0]:
        logger.warning(
            "Storage configuration validation failed. "
            "Check environment variables if using S3/GCS."
        )

    # Run the hotswap processing loop
    try:
        manifest = processor.process_and_offload_all_tiers(
            cleanup_raw=not args.no_cleanup
        )
        logger.info("Processing Complete. Results Manifest:")
        logger.info(json.dumps(manifest, indent=2))
    except Exception as e:
        logger.error(f"Fatal error during batch processing: {e}")
        sys.exit(1)
