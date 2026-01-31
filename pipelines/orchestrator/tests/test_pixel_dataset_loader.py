"""
Unit tests for PixelDatasetLoader orchestration class.

Tests the comprehensive dataset loading orchestration functionality including
dataset registration, concurrent loading, progress tracking, caching, and
error handling across multiple data sources.
"""

import tempfile
import unittest
from datetime import datetime
from pathlib import Path
from unittest.mock import patch

from .conversation_schema import Conversation, Message
from .pixel_dataset_loader import DatasetInfo, LoadingProgress, PixelDatasetLoader


class TestPixelDatasetLoader(unittest.TestCase):
    """Test PixelDatasetLoader orchestration functionality."""

    def setUp(self):
        """Set up test fixtures."""
        self.config = {}
        self.loader = PixelDatasetLoader(self.config)

        # Create test conversations
        self.test_conversations = [
            Conversation(
                id=f"test_conv_{i}",
                messages=[
                    Message(role="user", content=f"Test message {i}"),
                    Message(role="assistant", content=f"Test response {i}")
                ],
                context={},
                source="test"
            )
            for i in range(5)
        ]

    def test_initialization(self):
        """Test loader initialization."""
        assert self.loader.config is not None
        assert isinstance(self.loader.datasets, dict)
        assert len(self.loader.datasets) == 0
        assert self.loader.loading_progress is None
        assert isinstance(self.loader.progress_callbacks, list)
        assert isinstance(self.loader.dataset_callbacks, list)

    def test_register_dataset(self):
        """Test dataset registration."""
        self.loader.register_dataset(
            name="test_dataset",
            source_type="huggingface",
            source_path="test/dataset",
            target_conversations=1000,
            priority=1,
            metadata={"category": "test"}
        )

        assert "test_dataset" in self.loader.datasets
        dataset_info = self.loader.datasets["test_dataset"]

        assert dataset_info.name == "test_dataset"
        assert dataset_info.source_type == "huggingface"
        assert dataset_info.source_path == "test/dataset"
        assert dataset_info.target_conversations == 1000
        assert dataset_info.priority == 1
        assert dataset_info.status == "pending"
        assert dataset_info.progress == 0.0
        assert dataset_info.conversations_loaded == 0
        assert dataset_info.metadata["category"] == "test"

    def test_register_huggingface_datasets(self):
        """Test HuggingFace dataset registration."""
        self.loader.register_huggingface_datasets()

        # Check that datasets were registered
        assert len(self.loader.datasets) > 0

        # Check specific datasets
        expected_datasets = [
            "mental_health_counseling",
            "psych8k",
            "psychology_10k",
            "clinical_diagnosis_cot"
        ]

        for dataset_name in expected_datasets:
            assert dataset_name in self.loader.datasets
            dataset_info = self.loader.datasets[dataset_name]
            assert dataset_info.source_type == "huggingface"
            assert dataset_info.target_conversations > 0

    def test_register_local_datasets(self):
        """Test local dataset registration."""
        # Create temporary directories to simulate local datasets
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            # Create test directories
            (temp_path / "data" / "mental_health").mkdir(parents=True)
            (temp_path / "data" / "edge_cases").mkdir(parents=True)

            # Patch the Path.exists method to return True for our test paths
            with patch("pathlib.Path.exists") as mock_exists:
                mock_exists.return_value = True
                self.loader.register_local_datasets()

            # Check that local datasets were registered
            local_datasets = [d for d in self.loader.datasets.values() if d.source_type == "local"]
            assert len(local_datasets) > 0

    def test_register_generated_datasets(self):
        """Test generated dataset registration."""
        self.loader.register_generated_datasets()

        # Check that generated datasets were registered
        generated_datasets = [d for d in self.loader.datasets.values() if d.source_type == "generated"]
        assert len(generated_datasets) > 0

        # Check specific datasets
        expected_datasets = ["client_scenarios", "therapeutic_responses", "balanced_conversations"]

        for dataset_name in expected_datasets:
            assert dataset_name in self.loader.datasets
            dataset_info = self.loader.datasets[dataset_name]
            assert dataset_info.source_type == "generated"

    def test_get_dataset_category(self):
        """Test dataset category determination."""
        test_cases = [
            ("mental_health_counseling", "mental_health"),
            ("clinical_diagnosis_cot", "reasoning"),
            ("hercules_personality", "personality"),
            ("gutenberg_quality", "quality"),
            ("unknown_dataset", "general")
        ]

        for dataset_name, expected_category in test_cases:
            category = self.loader._get_dataset_category(dataset_name)
            assert category == expected_category

    def test_progress_callbacks(self):
        """Test progress callback functionality."""
        callback_called = False
        received_progress = None

        def test_callback(progress: LoadingProgress):
            nonlocal callback_called, received_progress
            callback_called = True
            received_progress = progress

        self.loader.add_progress_callback(test_callback)

        # Create test progress
        test_progress = LoadingProgress(
            total_datasets=5,
            completed_datasets=2,
            failed_datasets=0,
            total_conversations=1000,
            loaded_conversations=400,
            start_time=self.loader.loading_progress.start_time if self.loader.loading_progress else None
        )

        self.loader.loading_progress = test_progress
        self.loader._notify_progress()

        assert callback_called
        assert received_progress == test_progress

    def test_dataset_callbacks(self):
        """Test dataset callback functionality."""
        callback_called = False
        received_dataset = None

        def test_callback(dataset_info: DatasetInfo):
            nonlocal callback_called, received_dataset
            callback_called = True
            received_dataset = dataset_info

        self.loader.add_dataset_callback(test_callback)

        # Create test dataset info
        test_dataset = DatasetInfo(
            name="test_dataset",
            source_type="test",
            source_path="test/path",
            target_conversations=100
        )

        self.loader._notify_dataset_update(test_dataset)

        assert callback_called
        assert received_dataset == test_dataset

    def test_get_loading_order(self):
        """Test dataset loading order calculation."""
        # Register datasets with different priorities
        self.loader.register_dataset("high_priority_small", "test", "path1", 100, priority=1)
        self.loader.register_dataset("high_priority_large", "test", "path2", 1000, priority=1)
        self.loader.register_dataset("low_priority", "test", "path3", 500, priority=3)
        self.loader.register_dataset("medium_priority", "test", "path4", 200, priority=2)

        loading_order = self.loader.get_loading_order()

        # Should be ordered by priority first, then by size (descending) within same priority
        expected_order = [
            "high_priority_large",   # Priority 1, size 1000
            "high_priority_small",   # Priority 1, size 100
            "medium_priority",       # Priority 2, size 200
            "low_priority"           # Priority 3, size 500
        ]

        assert loading_order == expected_order

    def test_get_dataset_status(self):
        """Test dataset status retrieval."""
        self.loader.register_dataset("test_dataset", "test", "path", 100)

        status = self.loader.get_dataset_status("test_dataset")
        assert status is not None
        assert status.name == "test_dataset"

        # Test non-existent dataset
        status = self.loader.get_dataset_status("non_existent")
        assert status is None

    @patch("ai.pipelines.orchestrator.pixel_dataset_loader.save_json")
    @patch("ai.pipelines.orchestrator.pixel_dataset_loader.load_json")
    def test_caching_functionality(self, mock_load_json, mock_save_json):
        """Test conversation caching and loading."""
        # Test caching
        test_conversations = self.test_conversations[:2]
        self.loader._cache_conversations("test_dataset", test_conversations)

        mock_save_json.assert_called_once()

        # Test loading from cache
        mock_load_json.return_value = [conv.to_dict() for conv in test_conversations]

        with patch("pathlib.Path.exists", return_value=True):
            cached_conversations = self.loader._load_cached_conversations("test_dataset")

        assert cached_conversations is not None
        assert len(cached_conversations) == 2
        mock_load_json.assert_called_once()

    def test_update_overall_progress(self):
        """Test overall progress calculation."""
        # Initialize progress
        self.loader.loading_progress = LoadingProgress(
            total_datasets=4,
            completed_datasets=2,
            failed_datasets=0,
            total_conversations=1000,
            loaded_conversations=300,
            start_time=datetime.now()
        )

        self.loader._update_overall_progress()

        # Check progress calculation
        expected_dataset_progress = 2 / 4  # 0.5
        expected_conversation_progress = 300 / 1000  # 0.3
        expected_overall = (expected_dataset_progress * 0.7) + (expected_conversation_progress * 0.3)

        self.assertAlmostEqual(self.loader.loading_progress.overall_progress, expected_overall, places=2)
        assert self.loader.loading_progress.estimated_completion is not None

    @patch("ai.pipelines.orchestrator.pixel_dataset_loader.save_json")
    def test_export_loading_report(self, mock_save_json):
        """Test loading report export."""
        # Set up test data
        self.loader.register_dataset("test_dataset", "test", "path", 100)
        dataset_info = self.loader.datasets["test_dataset"]
        dataset_info.status = "completed"
        dataset_info.conversations_loaded = 95
        dataset_info.progress = 1.0

        self.loader.loading_progress = LoadingProgress(
            total_datasets=1,
            completed_datasets=1,
            failed_datasets=0,
            total_conversations=100,
            loaded_conversations=95,
            start_time=datetime.now(),
            end_time=datetime.now()
        )

        # Test export
        output_path = Path("test_report.json")
        result = self.loader.export_loading_report(output_path)

        assert result
        mock_save_json.assert_called_once()

        # Check report structure
        call_args = mock_save_json.call_args
        report_data = call_args[0][0]

        assert "loading_summary" in report_data
        assert "dataset_details" in report_data
        assert "generated_at" in report_data

        # Check summary data
        summary = report_data["loading_summary"]
        assert summary["total_datasets"] == 1
        assert summary["completed_datasets"] == 1
        assert summary["failed_datasets"] == 0
        assert summary["total_conversations"] == 95


if __name__ == "__main__":
    unittest.main()
