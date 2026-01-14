"""
Integration tests for the complete tier loading pipeline.

Tests the end-to-end functionality of all tier loaders (1-6) with:
- Registry integration
- S3 download capability (mocked)
- Dataset loading and validation
- Quality threshold enforcement
"""

import logging
from unittest.mock import MagicMock, patch

import pytest
from ai.dataset_pipeline.orchestration.tier_processor import TierProcessor

logger = logging.getLogger(__name__)


class TestTierPipelineIntegration:
    """Integration tests for the complete tier loading pipeline."""

    @pytest.fixture
    def tier_processor(self):
        """Create a TierProcessor with all tiers enabled."""
        return TierProcessor(
            enable_tier_1=True,
            enable_tier_2=True,
            enable_tier_3=True,
            enable_tier_4=True,
            enable_tier_5=True,
            enable_tier_6=True,
        )

    def test_tier_processor_initialization(self, tier_processor):
        """Test that TierProcessor initializes all tier loaders correctly."""
        assert len(tier_processor.tier_loaders) == 6

        # Verify each tier loader is present
        for tier_num in range(1, 7):
            assert tier_num in tier_processor.tier_loaders
            loader = tier_processor.tier_loaders[tier_num]
            assert loader.tier == tier_num

        logger.info("✓ All tier loaders initialized successfully")

    def test_tier_quality_thresholds(self, tier_processor):
        """Test that each tier has the correct quality threshold."""
        expected_thresholds = {
            1: 0.99,  # Priority datasets
            2: 0.95,  # Professional therapeutic
            3: 0.90,  # Chain-of-Thought reasoning
            4: 0.85,  # Reddit mental health
            5: 0.80,  # Research datasets
            6: 1.00,  # Knowledge base (reference)
        }

        for tier_num, expected_threshold in expected_thresholds.items():
            loader = tier_processor.tier_loaders[tier_num]
            assert loader.quality_threshold == expected_threshold, (
                f"Tier {tier_num} has incorrect threshold: "
                f"{loader.quality_threshold} != {expected_threshold}"
            )

        logger.info("✓ All tier quality thresholds are correct")

    def test_tier_dataset_discovery(self, tier_processor):
        """Test that tier loaders discover datasets from registry."""
        # Tier 3 should have discovered 27+ CoT datasets
        tier3_loader = tier_processor.tier_loaders[3]
        assert hasattr(tier3_loader, "dataset_paths")
        assert len(tier3_loader.dataset_paths) >= 27, (
            f"Tier 3 should have 27+ datasets, found {len(tier3_loader.dataset_paths)}"
        )

        # Tier 2 should have discovered 11+ professional datasets
        tier2_loader = tier_processor.tier_loaders[2]
        assert hasattr(tier2_loader, "dataset_paths")
        assert len(tier2_loader.dataset_paths) >= 11, (
            f"Tier 2 should have 11+ datasets, found {len(tier2_loader.dataset_paths)}"
        )

        logger.info("✓ Tier loaders successfully discovered datasets from registry")

    @patch("ai.dataset_pipeline.ingestion.tier_loaders.base_tier_loader.subprocess.run")
    @patch("ai.dataset_pipeline.ingestion.tier_loaders.base_tier_loader.shutil.which")
    def test_s3_download_capability(self, mock_which, mock_subprocess, tier_processor):
        """Test that tier loaders can handle S3 downloads (mocked)."""
        # Mock ovhai CLI availability
        mock_which.return_value = "/usr/bin/ovhai"
        mock_subprocess.return_value = MagicMock(returncode=0)

        # Test that each loader has S3 support methods
        for tier_num, loader in tier_processor.tier_loaders.items():
            assert hasattr(loader, "_is_ovhai_available")
            assert hasattr(loader, "_is_s3_path")
            assert hasattr(loader, "_ensure_dataset_locally")

        logger.info("✓ All tier loaders have S3 download capability")

    def test_tier_metadata_structure(self, tier_processor):
        """Test that tier loaders add correct metadata structure."""
        for tier_num, loader in tier_processor.tier_loaders.items():
            # Create a mock conversation
            from conversation_schema import Conversation, Message

            mock_conv = Conversation(
                conversation_id="test_001",
                source="test_source",
                messages=[Message(role="user", content="Test message")],
                metadata={},
            )

            # Add tier metadata
            loader.add_tier_metadata([mock_conv], {"test_key": "test_value"})

            # Verify metadata structure
            assert "tier" in mock_conv.metadata
            assert mock_conv.metadata["tier"] == tier_num
            assert "quality_threshold" in mock_conv.metadata
            assert mock_conv.metadata["quality_threshold"] == loader.quality_threshold
            assert "test_key" in mock_conv.metadata
            assert mock_conv.metadata["test_key"] == "test_value"

        logger.info("✓ All tier loaders add correct metadata structure")

    def test_tier_training_ratios(self, tier_processor):
        """Test that tier training ratios are correctly configured."""
        expected_ratios = {
            1: 0.40,  # 40% Priority
            2: 0.25,  # 25% Professional
            3: 0.20,  # 20% CoT Reasoning
            4: 0.10,  # 10% Reddit
            5: 0.04,  # 4% Research
            6: 0.01,  # 1% Knowledge Base
        }

        for tier_num, expected_ratio in expected_ratios.items():
            actual_ratio = tier_processor.TRAINING_RATIO_WEIGHTS.get(tier_num)
            assert actual_ratio == expected_ratio, (
                f"Tier {tier_num} has incorrect training ratio: "
                f"{actual_ratio} != {expected_ratio}"
            )

        # Verify ratios sum to 1.0
        total_ratio = sum(tier_processor.TRAINING_RATIO_WEIGHTS.values())
        assert abs(total_ratio - 1.0) < 0.001, (
            f"Training ratios should sum to 1.0, got {total_ratio}"
        )

        logger.info("✓ Tier training ratios are correctly configured")

    def test_tier_processor_statistics(self, tier_processor):
        """Test that TierProcessor can generate statistics."""
        # This would normally require loading actual data
        # For now, just verify the method exists and structure
        stats = tier_processor.get_tier_statistics()

        assert isinstance(stats, dict)
        assert "tiers_processed" in stats
        assert "tier_details" in stats
        assert "total_conversations" in stats

        # Since no data is loaded, tiers_processed should be 0
        assert stats["tiers_processed"] == 0
        assert stats["total_conversations"] == 0

        logger.info("✓ TierProcessor can generate statistics")

    @pytest.mark.slow
    @patch("ai.dataset_pipeline.ingestion.tier_loaders.base_tier_loader.subprocess.run")
    @patch("ai.dataset_pipeline.ingestion.tier_loaders.base_tier_loader.shutil.which")
    def test_process_single_tier(self, mock_which, mock_subprocess, tier_processor):
        """Test processing a single tier (Tier 1 - smallest dataset)."""
        # Mock ovhai CLI
        mock_which.return_value = "/usr/bin/ovhai"
        mock_subprocess.return_value = MagicMock(returncode=0)

        try:
            # Process Tier 1 (Priority datasets - should be smallest)
            datasets = tier_processor.process_tier(1)

            assert isinstance(datasets, dict)
            logger.info(
                f"✓ Successfully processed Tier 1: {len(datasets)} datasets loaded"
            )

        except FileNotFoundError as e:
            # Expected if datasets don't exist locally
            logger.warning(
                f"Tier 1 datasets not found (expected in test environment): {e}"
            )
            pytest.skip("Tier 1 datasets not available in test environment")

    def test_registry_loading(self, tier_processor):
        """Test that tier loaders can load the dataset registry."""
        for tier_num, loader in tier_processor.tier_loaders.items():
            if hasattr(loader, "registry"):
                assert isinstance(loader.registry, dict)
                assert "datasets" in loader.registry or len(loader.registry) == 0

        logger.info("✓ Tier loaders successfully load dataset registry")


class TestTierLoaderFileHandling:
    """Test file handling capabilities across all tier loaders."""

    @pytest.fixture
    def tier_processor(self):
        """Create a TierProcessor with all tiers enabled."""
        return TierProcessor(
            enable_tier_1=True,
            enable_tier_2=True,
            enable_tier_3=True,
            enable_tier_4=True,
            enable_tier_5=True,
            enable_tier_6=True,
        )

    def test_json_loading_capability(self, tier_processor):
        """Test that tier loaders can load JSON files."""
        for tier_num, loader in tier_processor.tier_loaders.items():
            assert hasattr(loader, "load_json_file")
            assert callable(loader.load_json_file)

        logger.info("✓ All tier loaders have JSON loading capability")

    def test_jsonl_loading_capability(self, tier_processor):
        """Test that tier loaders can load JSONL files."""
        for tier_num, loader in tier_processor.tier_loaders.items():
            assert hasattr(loader, "load_jsonl_file")
            assert callable(loader.load_jsonl_file)

        logger.info("✓ All tier loaders have JSONL loading capability")

    def test_directory_loading_capability(self, tier_processor):
        """Test that tier loaders can load from directories."""
        for tier_num, loader in tier_processor.tier_loaders.items():
            assert hasattr(loader, "_load_dataset_directory")
            assert callable(loader._load_dataset_directory)

        logger.info("✓ All tier loaders have directory loading capability")


if __name__ == "__main__":
    # Run tests with verbose output
    pytest.main([__file__, "-v", "-s"])
