"""
Unit tests for pipeline orchestrator.

Tests the comprehensive automated pipeline orchestration functionality including
stage execution, dataset loading coordination, quality monitoring integration,
error recovery, and performance optimization.
"""

import asyncio
import tempfile
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

from .conversation_schema import Conversation, Message
from .pipeline_orchestrator import (
    ExecutionMode,
    PipelineConfig,
    PipelineMetrics,
    PipelineOrchestrator,
    PipelineStage,
)


class TestPipelineOrchestrator(unittest.TestCase):
    """Test pipeline orchestrator functionality."""

    def setUp(self):
        """Set up test fixtures."""
        # Create temporary directories for testing
        self.temp_dir = Path(tempfile.mkdtemp())

        # Create test configuration
        self.config = PipelineConfig(
            execution_mode=ExecutionMode.SEQUENTIAL,
            max_concurrent_datasets=2,
            quality_threshold=0.6,
            error_tolerance=0.2,
            retry_attempts=2,
            retry_delay=0.1,  # Short delay for testing
            output_directory=self.temp_dir / "output",
            cache_directory=self.temp_dir / "cache",
            report_directory=self.temp_dir / "reports"
        )

        self.orchestrator = PipelineOrchestrator(self.config)

        # Create test conversations
        self.test_conversations = [
            Conversation(
                id=f"test_conv_{i}",
                messages=[
                    Message(role="client", content=f"I need help with anxiety issue {i}"),
                    Message(role="therapist", content=f"I understand your concern about anxiety {i}. Let's work through this together.")
                ],
                context={"quality": "high"},
                source="test"
            )
            for i in range(3)
        ]

    def tearDown(self):
        """Clean up test fixtures."""
        # Clean up temporary directory
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_initialization(self):
        """Test orchestrator initialization."""
        assert self.orchestrator.config is not None
        assert self.orchestrator.dataset_loader is not None
        assert self.orchestrator.acquisition_monitor is not None
        assert isinstance(self.orchestrator.metrics, PipelineMetrics)
        assert self.orchestrator.current_stage == PipelineStage.INITIALIZATION

        # Check that directories were created
        assert self.config.output_directory.exists()
        assert self.config.cache_directory.exists()
        assert self.config.report_directory.exists()

    def test_pipeline_config(self):
        """Test pipeline configuration."""
        config = PipelineConfig(
            execution_mode=ExecutionMode.CONCURRENT,
            max_concurrent_datasets=5,
            quality_threshold=0.8
        )

        assert config.execution_mode == ExecutionMode.CONCURRENT
        assert config.max_concurrent_datasets == 5
        assert config.quality_threshold == 0.8
        assert config.error_tolerance == 0.1  # Default value

    def test_stage_callbacks(self):
        """Test pipeline stage callback functionality."""
        callback_called = False
        received_stage = None

        def test_callback(stage):
            nonlocal callback_called, received_stage
            callback_called = True
            received_stage = stage

        self.orchestrator.add_stage_callback(test_callback)

        # Execute a stage
        asyncio.run(self.orchestrator._execute_stage(PipelineStage.INITIALIZATION))

        assert callback_called
        assert received_stage == PipelineStage.INITIALIZATION

    def test_progress_callbacks(self):
        """Test progress callback functionality."""
        callback_called = False
        received_metrics = None

        def test_callback(metrics):
            nonlocal callback_called, received_metrics
            callback_called = True
            received_metrics = metrics

        self.orchestrator.add_progress_callback(test_callback)

        # Manually trigger progress callback (would normally be called during execution)
        for callback in self.orchestrator.progress_callbacks:
            callback(self.orchestrator.metrics)

        assert callback_called
        assert received_metrics == self.orchestrator.metrics

    def test_error_callbacks(self):
        """Test error callback functionality."""
        callback_called = False
        received_context = None
        received_error = None

        def test_callback(context, error):
            nonlocal callback_called, received_context, received_error
            callback_called = True
            received_context = context
            received_error = error

        self.orchestrator.add_error_callback(test_callback)

        # Log an error
        test_error = Exception("Test error")
        self.orchestrator._log_error("test_context", test_error)

        assert callback_called
        assert received_context == "test_context"
        assert received_error == test_error

    @patch("ai.dataset_pipeline.pipeline_orchestrator.PixelDatasetLoader")
    @patch("ai.dataset_pipeline.pipeline_orchestrator.AcquisitionMonitor")
    def test_initialization_stage(self, mock_monitor, mock_loader):
        """Test pipeline initialization stage."""

        async def run_test():
            await self.orchestrator._initialize_pipeline()

            # Check that metrics were reset
            assert self.orchestrator.metrics.total_datasets == 0
            assert self.orchestrator.metrics.completed_datasets == 0
            assert self.orchestrator.metrics.failed_datasets == 0

            # Check that state was cleared
            assert len(self.orchestrator.failed_datasets) == 0
            assert len(self.orchestrator.retry_counts) == 0
            assert len(self.orchestrator.error_log) == 0

            # Check execution context
            assert "start_time" in self.orchestrator.execution_context
            assert "config" in self.orchestrator.execution_context

        asyncio.run(run_test())

    @patch("ai.dataset_pipeline.pipeline_orchestrator.PixelDatasetLoader")
    def test_dataset_registration(self, mock_loader_class):
        """Test dataset registration stage."""

        # Mock the dataset loader instance
        mock_loader = MagicMock()
        mock_loader.datasets = {
            "hf_dataset1": MagicMock(source_type="huggingface"),
            "hf_dataset2": MagicMock(source_type="huggingface"),
            "local_dataset1": MagicMock(source_type="local")
        }
        self.orchestrator.dataset_loader = mock_loader

        async def run_test():
            await self.orchestrator._register_datasets(
                include_huggingface=True,
                include_local=True,
                include_generated=False,
                custom_datasets=None
            )

            # Check that registration methods were called
            mock_loader.register_huggingface_datasets.assert_called_once()
            mock_loader.register_local_datasets.assert_called_once()

            # Check that metrics were updated
            assert self.orchestrator.metrics.total_datasets == 3

        asyncio.run(run_test())

    def test_quality_monitoring_setup(self):
        """Test quality monitoring setup."""

        async def run_test():
            await self.orchestrator._setup_quality_monitoring()

            # Check that callbacks were registered
            assert len(self.orchestrator.acquisition_monitor.metric_callbacks) > 0
            assert len(self.orchestrator.acquisition_monitor.alert_callbacks) > 0
            assert len(self.orchestrator.acquisition_monitor.stats_callbacks) > 0

        asyncio.run(run_test())

    @patch("ai.dataset_pipeline.pipeline_orchestrator.PixelDatasetLoader")
    def test_sequential_loading(self, mock_loader_class):
        """Test sequential dataset loading."""

        # Mock the dataset loader
        mock_loader = MagicMock()
        mock_loader.get_loading_order.return_value = ["dataset1", "dataset2"]
        mock_loader.load_all_datasets = AsyncMock(return_value={
            "dataset1": self.test_conversations[:2],
            "dataset2": self.test_conversations[2:]
        })
        self.orchestrator.dataset_loader = mock_loader

        async def run_test():
            result = await self.orchestrator._execute_sequential_loading()

            assert "dataset1" in result
            assert "dataset2" in result
            assert len(result["dataset1"]) == 2
            assert len(result["dataset2"]) == 1

        asyncio.run(run_test())

    @patch("ai.dataset_pipeline.pipeline_orchestrator.PixelDatasetLoader")
    def test_concurrent_loading(self, mock_loader_class):
        """Test concurrent dataset loading."""

        # Mock the dataset loader
        mock_loader = MagicMock()
        mock_loader.datasets = {"dataset1": MagicMock(), "dataset2": MagicMock()}
        mock_loader.load_all_datasets = AsyncMock(return_value={
            "dataset1": self.test_conversations[:2],
            "dataset2": self.test_conversations[2:]
        })
        self.orchestrator.dataset_loader = mock_loader

        async def run_test():
            result = await self.orchestrator._execute_concurrent_loading()

            assert "dataset1" in result
            assert "dataset2" in result

            # Check that concurrent loading was called with correct parameters
            mock_loader.load_all_datasets.assert_called_with(
                max_concurrent=self.config.max_concurrent_datasets
            )

        asyncio.run(run_test())

    def test_quality_validation(self):
        """Test quality validation of loaded datasets."""

        # Mock the acquisition monitor to return quality scores
        self.orchestrator.acquisition_monitor.process_conversation = MagicMock(
            return_value={"quality_score": 0.8}  # Above threshold
        )

        test_datasets = {
            "dataset1": self.test_conversations[:2],
            "dataset2": self.test_conversations[2:]
        }

        async def run_test():
            result = await self.orchestrator._validate_quality(test_datasets)

            # All conversations should be accepted (quality score 0.8 > threshold 0.6)
            assert len(result["dataset1"]) == 2
            assert len(result["dataset2"]) == 1

            # Check metrics were updated
            assert self.orchestrator.metrics.total_conversations == 3
            assert self.orchestrator.metrics.accepted_conversations == 3
            assert self.orchestrator.metrics.rejected_conversations == 0

        asyncio.run(run_test())

    def test_quality_validation_with_rejection(self):
        """Test quality validation with conversation rejection."""

        # Mock the acquisition monitor to return low quality scores
        self.orchestrator.acquisition_monitor.process_conversation = MagicMock(
            return_value={"quality_score": 0.4}  # Below threshold
        )

        test_datasets = {
            "dataset1": self.test_conversations[:2]
        }

        async def run_test():
            result = await self.orchestrator._validate_quality(test_datasets)

            # All conversations should be rejected (quality score 0.4 < threshold 0.6)
            assert len(result["dataset1"]) == 0

            # Check metrics were updated
            assert self.orchestrator.metrics.total_conversations == 2
            assert self.orchestrator.metrics.accepted_conversations == 0
            assert self.orchestrator.metrics.rejected_conversations == 2

        asyncio.run(run_test())

    def test_performance_score_calculation(self):
        """Test performance score calculation."""

        # Set up test metrics
        self.orchestrator.metrics.total_datasets = 10
        self.orchestrator.metrics.completed_datasets = 9  # 90% completion
        self.orchestrator.metrics.quality_score = 0.8
        self.orchestrator.metrics.processing_rate = 5.0  # 50% of excellent rate (10)
        self.orchestrator.metrics.error_rate = 0.05  # 5% error rate

        performance_score = self.orchestrator._calculate_performance_score()

        # Expected: (0.9 * 0.3) + (0.8 * 0.3) + (0.5 * 0.2) + (0.95 * 0.2) = 0.8
        expected_score = (0.9 * 0.3) + (0.8 * 0.3) + (0.5 * 0.2) + (0.95 * 0.2)
        self.assertAlmostEqual(performance_score, expected_score, places=2)

    def test_recommendations_generation(self):
        """Test recommendation generation based on metrics."""

        # Set up metrics that should trigger recommendations
        self.orchestrator.metrics.total_datasets = 10
        self.orchestrator.metrics.completed_datasets = 7  # 70% completion (< 90%)
        self.orchestrator.metrics.quality_score = 0.7  # Below 0.8
        self.orchestrator.metrics.processing_rate = 3.0  # Below 5.0
        self.orchestrator.metrics.error_rate = 0.08  # Above 0.05

        recommendations = self.orchestrator._generate_recommendations()

        # Should generate multiple recommendations
        assert len(recommendations) > 0

        # Check for specific recommendation types
        recommendation_text = " ".join(recommendations).lower()
        assert "failed datasets" in recommendation_text
        assert "quality" in recommendation_text
        assert "performance" in recommendation_text
        assert "error rate" in recommendation_text

    def test_error_logging(self):
        """Test error logging functionality."""

        test_error = Exception("Test error message")
        context = "test_context"

        initial_error_count = self.orchestrator.metrics.total_errors

        self.orchestrator._log_error(context, test_error)

        # Check that error was logged
        assert len(self.orchestrator.error_log) == 1
        assert self.orchestrator.error_log[0]["type"] == "error"
        assert self.orchestrator.error_log[0]["context"] == context
        assert self.orchestrator.error_log[0]["message"] == str(test_error)

        # Check that error count was incremented
        assert self.orchestrator.metrics.total_errors == initial_error_count + 1

    def test_execution_report_generation(self):
        """Test execution report generation."""

        # Set up some test data
        self.orchestrator.metrics.total_datasets = 5
        self.orchestrator.metrics.completed_datasets = 4
        self.orchestrator.metrics.total_conversations = 100
        self.orchestrator.metrics.accepted_conversations = 85

        report = self.orchestrator._generate_execution_report()

        # Check report structure
        assert "execution_summary" in report
        assert "dataset_metrics" in report
        assert "conversation_metrics" in report
        assert "quality_metrics" in report
        assert "performance_metrics" in report
        assert "configuration" in report

        # Check specific values
        assert report["dataset_metrics"]["total_datasets"] == 5
        assert report["dataset_metrics"]["completed_datasets"] == 4
        assert report["conversation_metrics"]["total_conversations"] == 100
        assert report["conversation_metrics"]["accepted_conversations"] == 85

    def test_get_current_metrics(self):
        """Test getting current pipeline metrics."""

        metrics = self.orchestrator.get_current_metrics()

        assert isinstance(metrics, PipelineMetrics)
        assert metrics == self.orchestrator.metrics

    def test_get_execution_context(self):
        """Test getting execution context."""

        # Set up some context
        self.orchestrator.execution_context = {"test_key": "test_value"}

        context = self.orchestrator.get_execution_context()

        assert isinstance(context, dict)
        assert context["test_key"] == "test_value"

        # Should be a copy, not the original
        context["new_key"] = "new_value"
        assert "new_key" not in self.orchestrator.execution_context


if __name__ == "__main__":
    unittest.main()
