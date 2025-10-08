"""
Automated pipeline orchestration system for systematic data loading.

This module provides comprehensive orchestration of the entire dataset acquisition
pipeline, coordinating between dataset loading, quality monitoring, error recovery,
and performance optimization for systematic, high-quality data processing.
"""

import asyncio
import time
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any

from acquisition_monitor import AcquisitionMonitor
from conversation_schema import Conversation
from logger import get_logger
from pixel_dataset_loader import PixelDatasetLoader

from utils import read_json, write_json

logger = get_logger("dataset_pipeline.pipeline_orchestrator")


class PipelineStage(Enum):
    """Pipeline execution stages."""

    INITIALIZATION = "initialization"
    DATASET_REGISTRATION = "dataset_registration"
    QUALITY_SETUP = "quality_setup"
    LOADING_EXECUTION = "loading_execution"
    QUALITY_VALIDATION = "quality_validation"
    ERROR_RECOVERY = "error_recovery"
    FINALIZATION = "finalization"
    COMPLETED = "completed"
    FAILED = "failed"


class ExecutionMode(Enum):
    """Pipeline execution modes."""

    SEQUENTIAL = "sequential"  # Load datasets one by one
    CONCURRENT = "concurrent"  # Load multiple datasets simultaneously
    ADAPTIVE = "adaptive"  # Adjust concurrency based on performance
    PRIORITY_BASED = "priority_based"  # Load by priority with smart scheduling


@dataclass
class PipelineConfig:
    """Configuration for pipeline orchestration."""

    execution_mode: ExecutionMode = ExecutionMode.ADAPTIVE
    max_concurrent_datasets: int = 3
    quality_threshold: float = 0.7
    error_tolerance: float = 0.1  # 10% error rate tolerance
    retry_attempts: int = 3
    retry_delay: float = 5.0  # seconds
    enable_auto_recovery: bool = True
    enable_performance_optimization: bool = True
    checkpoint_interval: int = 100  # conversations
    output_directory: Path = Path("data/processed")
    cache_directory: Path = Path("cache/pipeline")
    report_directory: Path = Path("reports")


@dataclass
class PipelineMetrics:
    """Comprehensive pipeline execution metrics."""

    start_time: datetime
    end_time: datetime | None = None
    total_datasets: int = 0
    completed_datasets: int = 0
    failed_datasets: int = 0
    total_conversations: int = 0
    accepted_conversations: int = 0
    rejected_conversations: int = 0
    total_errors: int = 0
    retry_count: int = 0
    current_stage: PipelineStage = PipelineStage.INITIALIZATION
    processing_rate: float = 0.0  # conversations per second
    quality_score: float = 0.0
    error_rate: float = 0.0
    performance_score: float = 0.0


@dataclass
class ExecutionResult:
    """Result of pipeline execution."""

    success: bool
    metrics: PipelineMetrics
    datasets: dict[str, list[Conversation]]
    errors: list[str]
    warnings: list[str]
    recommendations: list[str]
    execution_report: dict[str, Any]


class PipelineOrchestrator:
    """
    Comprehensive automated pipeline orchestration system.

    Coordinates dataset loading, quality monitoring, error recovery,
    and performance optimization for systematic data processing.
    """

    def __init__(self, config: PipelineConfig | None = None):
        """Initialize the pipeline orchestrator."""
        self.config = config or PipelineConfig()

        # Core components
        self.dataset_loader = PixelDatasetLoader()
        self.acquisition_monitor = AcquisitionMonitor()

        # Pipeline state
        self.metrics = PipelineMetrics(start_time=datetime.now())
        self.current_stage = PipelineStage.INITIALIZATION
        self.execution_context: dict[str, Any] = {}
        self.checkpoints: list[dict[str, Any]] = []

        # Error tracking and recovery
        self.failed_datasets: set[str] = set()
        self.retry_counts: dict[str, int] = {}
        self.error_log: list[dict[str, Any]] = []

        # Performance tracking
        self.stage_timings: dict[PipelineStage, float] = {}
        self.dataset_performance: dict[str, dict[str, float]] = {}

        # Callbacks for pipeline events
        self.stage_callbacks: list[Callable[[PipelineStage], None]] = []
        self.progress_callbacks: list[Callable[[PipelineMetrics], None]] = []
        self.error_callbacks: list[Callable[[str, Exception], None]] = []

        # Ensure directories exist
        self.config.output_directory.mkdir(parents=True, exist_ok=True)
        self.config.cache_directory.mkdir(parents=True, exist_ok=True)
        self.config.report_directory.mkdir(parents=True, exist_ok=True)

        logger.info("Pipeline Orchestrator initialized")

    async def execute_pipeline(
        self,
        include_huggingface: bool = True,
        include_local: bool = True,
        include_generated: bool = True,
        custom_datasets: list[dict[str, Any]] | None = None,
    ) -> ExecutionResult:
        """Execute the complete data loading pipeline."""

        logger.info("Starting automated pipeline execution")
        self.metrics.start_time = datetime.now()

        try:
            # Stage 1: Initialization
            await self._execute_stage(PipelineStage.INITIALIZATION)
            await self._initialize_pipeline()

            # Stage 2: Dataset Registration
            await self._execute_stage(PipelineStage.DATASET_REGISTRATION)
            await self._register_datasets(
                include_huggingface, include_local, include_generated, custom_datasets
            )

            # Stage 3: Quality Setup
            await self._execute_stage(PipelineStage.QUALITY_SETUP)
            await self._setup_quality_monitoring()

            # Stage 4: Loading Execution
            await self._execute_stage(PipelineStage.LOADING_EXECUTION)
            datasets = await self._execute_loading()

            # Stage 5: Quality Validation
            await self._execute_stage(PipelineStage.QUALITY_VALIDATION)
            validated_datasets = await self._validate_quality(datasets)

            # Stage 6: Error Recovery (if needed)
            if self.failed_datasets and self.config.enable_auto_recovery:
                await self._execute_stage(PipelineStage.ERROR_RECOVERY)
                recovered_datasets = await self._execute_error_recovery()
                validated_datasets.update(recovered_datasets)

            # Stage 7: Finalization
            await self._execute_stage(PipelineStage.FINALIZATION)
            result = await self._finalize_pipeline(validated_datasets)

            await self._execute_stage(PipelineStage.COMPLETED)
            logger.info("Pipeline execution completed successfully")

            return result

        except Exception as e:
            await self._execute_stage(PipelineStage.FAILED)
            logger.error(f"Pipeline execution failed: {e}")

            return ExecutionResult(
                success=False,
                metrics=self.metrics,
                datasets={},
                errors=[str(e)],
                warnings=[],
                recommendations=[
                    "Review error logs and retry with adjusted configuration"
                ],
                execution_report=self._generate_execution_report(),
            )

    async def _execute_stage(self, stage: PipelineStage) -> None:
        """Execute a pipeline stage with timing and callbacks."""

        stage_start = time.time()
        self.current_stage = stage
        self.metrics.current_stage = stage

        logger.info(f"Executing pipeline stage: {stage.value}")

        # Notify stage callbacks
        for callback in self.stage_callbacks:
            try:
                callback(stage)
            except Exception as e:
                logger.error(f"Error in stage callback: {e}")

        # Record stage timing
        if stage != PipelineStage.INITIALIZATION:
            # Calculate previous stage duration
            prev_stages = list(PipelineStage)
            if stage in prev_stages:
                prev_index = prev_stages.index(stage) - 1
                if prev_index >= 0:
                    prev_stage = prev_stages[prev_index]
                    if prev_stage in self.stage_timings:
                        duration = stage_start - self.stage_timings[prev_stage]
                        logger.debug(
                            f"Stage {prev_stage.value} completed in {duration:.2f}s"
                        )

        self.stage_timings[stage] = stage_start

    async def _initialize_pipeline(self) -> None:
        """Initialize pipeline components and state."""

        # Reset metrics
        self.metrics.total_datasets = 0
        self.metrics.completed_datasets = 0
        self.metrics.failed_datasets = 0

        # Clear previous state
        self.failed_datasets.clear()
        self.retry_counts.clear()
        self.error_log.clear()

        # Initialize execution context
        self.execution_context = {
            "start_time": self.metrics.start_time.isoformat(),
            "config": {
                "execution_mode": self.config.execution_mode.value,
                "max_concurrent_datasets": self.config.max_concurrent_datasets,
                "quality_threshold": self.config.quality_threshold,
                "error_tolerance": self.config.error_tolerance,
            },
        }

        logger.info("Pipeline initialization completed")

    async def _register_datasets(
        self,
        include_huggingface: bool,
        include_local: bool,
        include_generated: bool,
        custom_datasets: list[dict[str, Any]] | None,
    ) -> None:
        """Register datasets for loading."""

        registration_count = 0

        # Register HuggingFace datasets
        if include_huggingface:
            self.dataset_loader.register_huggingface_datasets()
            hf_count = len(
                [
                    d
                    for d in self.dataset_loader.datasets.values()
                    if d.source_type == "huggingface"
                ]
            )
            registration_count += hf_count
            logger.info(f"Registered {hf_count} HuggingFace datasets")

        # Register local datasets
        if include_local:
            self.dataset_loader.register_local_datasets()
            local_count = len(
                [
                    d
                    for d in self.dataset_loader.datasets.values()
                    if d.source_type == "local"
                ]
            )
            registration_count += local_count
            logger.info(f"Registered {local_count} local datasets")

        # Register generated datasets
        if include_generated:
            self.dataset_loader.register_generated_datasets()
            gen_count = len(
                [
                    d
                    for d in self.dataset_loader.datasets.values()
                    if d.source_type == "generated"
                ]
            )
            registration_count += gen_count
            logger.info(f"Registered {gen_count} generated datasets")

        # Register custom datasets
        if custom_datasets:
            for dataset_config in custom_datasets:
                self.dataset_loader.register_dataset(**dataset_config)
                registration_count += 1
            logger.info(f"Registered {len(custom_datasets)} custom datasets")

        self.metrics.total_datasets = registration_count
        logger.info(f"Total datasets registered: {registration_count}")

    async def _setup_quality_monitoring(self) -> None:
        """Setup quality monitoring and callbacks."""

        # Setup monitoring callbacks
        def metric_callback(metric):
            # Update pipeline metrics based on quality metrics
            if metric.metric_type == MetricType.QUALITY_SCORE:
                # Update running quality average
                current_total = (
                    self.metrics.quality_score * self.metrics.total_conversations
                )
                new_total = current_total + metric.value
                self.metrics.total_conversations += 1
                self.metrics.quality_score = (
                    new_total / self.metrics.total_conversations
                )

        def alert_callback(alert):
            # Log alerts and potentially trigger recovery actions
            self.error_log.append(
                {
                    "type": "alert",
                    "level": alert.level.value,
                    "message": alert.message,
                    "dataset": alert.dataset_name,
                    "timestamp": alert.timestamp.isoformat(),
                }
            )

            if alert.level in [AlertLevel.ERROR, AlertLevel.CRITICAL]:
                logger.warning(f"Quality alert: {alert.message}")

        def stats_callback(stats):
            # Update dataset performance tracking
            if stats.dataset_name not in self.dataset_performance:
                self.dataset_performance[stats.dataset_name] = {}

            self.dataset_performance[stats.dataset_name].update(
                {
                    "conversations_processed": stats.conversations_processed,
                    "acceptance_rate": stats.conversations_accepted
                    / max(stats.conversations_processed, 1),
                    "average_quality": stats.average_quality_score,
                    "processing_rate": stats.current_rate,
                    "error_rate": stats.error_count
                    / max(stats.conversations_processed, 1),
                }
            )

        # Register callbacks
        self.acquisition_monitor.add_metric_callback(metric_callback)
        self.acquisition_monitor.add_alert_callback(alert_callback)
        self.acquisition_monitor.add_stats_callback(stats_callback)

        logger.info("Quality monitoring setup completed")

    async def _execute_loading(self) -> dict[str, list[Conversation]]:
        """Execute dataset loading with the configured execution mode."""

        if self.config.execution_mode == ExecutionMode.SEQUENTIAL:
            return await self._execute_sequential_loading()
        if self.config.execution_mode == ExecutionMode.CONCURRENT:
            return await self._execute_concurrent_loading()
        if self.config.execution_mode == ExecutionMode.ADAPTIVE:
            return await self._execute_adaptive_loading()
        if self.config.execution_mode == ExecutionMode.PRIORITY_BASED:
            return await self._execute_priority_based_loading()
        raise ValueError(f"Unknown execution mode: {self.config.execution_mode}")

    async def _execute_sequential_loading(self) -> dict[str, list[Conversation]]:
        """Execute sequential dataset loading."""

        datasets = {}
        loading_order = self.dataset_loader.get_loading_order()

        for dataset_name in loading_order:
            try:
                logger.info(f"Loading dataset: {dataset_name}")
                self.acquisition_monitor.start_monitoring(dataset_name)

                # Load single dataset
                result = await self.dataset_loader.load_all_datasets(max_concurrent=1)

                if dataset_name in result:
                    datasets[dataset_name] = result[dataset_name]
                    self.metrics.completed_datasets += 1
                    logger.info(
                        f"Successfully loaded {len(result[dataset_name])} conversations from {dataset_name}"
                    )
                else:
                    self.failed_datasets.add(dataset_name)
                    self.metrics.failed_datasets += 1
                    logger.error(f"Failed to load dataset: {dataset_name}")

                self.acquisition_monitor.stop_monitoring(dataset_name)

            except Exception as e:
                self.failed_datasets.add(dataset_name)
                self.metrics.failed_datasets += 1
                self._log_error(dataset_name, e)

        return datasets

    async def _execute_concurrent_loading(self) -> dict[str, list[Conversation]]:
        """Execute concurrent dataset loading."""

        logger.info(
            f"Loading datasets concurrently (max {self.config.max_concurrent_datasets})"
        )

        # Start monitoring for all datasets
        for dataset_name in self.dataset_loader.datasets:
            self.acquisition_monitor.start_monitoring(dataset_name)

        try:
            # Load all datasets concurrently
            datasets = await self.dataset_loader.load_all_datasets(
                max_concurrent=self.config.max_concurrent_datasets
            )

            # Update metrics
            self.metrics.completed_datasets = len(datasets)
            self.metrics.failed_datasets = len(self.dataset_loader.datasets) - len(
                datasets
            )

            # Stop monitoring
            for dataset_name in self.dataset_loader.datasets:
                self.acquisition_monitor.stop_monitoring(dataset_name)

            return datasets

        except Exception as e:
            logger.error(f"Concurrent loading failed: {e}")
            self._log_error("concurrent_loading", e)
            return {}

    async def _execute_adaptive_loading(self) -> dict[str, list[Conversation]]:
        """Execute adaptive loading with dynamic concurrency adjustment."""

        # Start with moderate concurrency
        current_concurrency = min(2, self.config.max_concurrent_datasets)
        datasets = {}

        # Monitor performance and adjust concurrency
        performance_window = []

        loading_order = self.dataset_loader.get_loading_order()

        for i in range(0, len(loading_order), current_concurrency):
            batch = loading_order[i : i + current_concurrency]
            batch_start = time.time()

            logger.info(
                f"Loading batch of {len(batch)} datasets with concurrency {current_concurrency}"
            )

            # Start monitoring
            for dataset_name in batch:
                self.acquisition_monitor.start_monitoring(dataset_name)

            # Load batch
            try:
                # Create temporary loader for this batch
                batch_loader = PixelDatasetLoader()
                for dataset_name in batch:
                    dataset_info = self.dataset_loader.datasets[dataset_name]
                    batch_loader.register_dataset(
                        dataset_name,
                        dataset_info.source_type,
                        dataset_info.source_path,
                        dataset_info.target_conversations,
                        dataset_info.priority,
                    )

                batch_result = await batch_loader.load_all_datasets(
                    max_concurrent=current_concurrency
                )
                datasets.update(batch_result)

                # Calculate performance
                batch_time = time.time() - batch_start
                conversations_loaded = sum(
                    len(convs) for convs in batch_result.values()
                )
                batch_rate = conversations_loaded / batch_time if batch_time > 0 else 0

                performance_window.append(batch_rate)
                if len(performance_window) > 3:
                    performance_window.pop(0)

                # Adjust concurrency based on performance
                if len(performance_window) >= 2:
                    recent_avg = sum(performance_window[-2:]) / 2
                    if recent_avg > sum(performance_window[:-2]) / max(
                        len(performance_window) - 2, 1
                    ):
                        # Performance improving, increase concurrency
                        current_concurrency = min(
                            current_concurrency + 1, self.config.max_concurrent_datasets
                        )
                    else:
                        # Performance declining, decrease concurrency
                        current_concurrency = max(current_concurrency - 1, 1)

                logger.info(
                    f"Batch completed in {batch_time:.2f}s, rate: {batch_rate:.2f} conv/s, "
                    f"next concurrency: {current_concurrency}"
                )

            except Exception as e:
                logger.error(f"Batch loading failed: {e}")
                for dataset_name in batch:
                    self.failed_datasets.add(dataset_name)

            # Stop monitoring
            for dataset_name in batch:
                self.acquisition_monitor.stop_monitoring(dataset_name)

        return datasets

    async def _execute_priority_based_loading(self) -> dict[str, list[Conversation]]:
        """Execute priority-based loading with smart scheduling."""

        # Group datasets by priority
        priority_groups = {}
        for dataset_name, dataset_info in self.dataset_loader.datasets.items():
            priority = dataset_info.priority
            if priority not in priority_groups:
                priority_groups[priority] = []
            priority_groups[priority].append(dataset_name)

        datasets = {}

        # Process each priority group
        for priority in sorted(priority_groups.keys()):
            group = priority_groups[priority]
            logger.info(f"Loading priority {priority} datasets: {group}")

            # Determine concurrency for this priority group
            group_concurrency = min(len(group), self.config.max_concurrent_datasets)

            # Start monitoring
            for dataset_name in group:
                self.acquisition_monitor.start_monitoring(dataset_name)

            try:
                # Create loader for this priority group
                group_loader = PixelDatasetLoader()
                for dataset_name in group:
                    dataset_info = self.dataset_loader.datasets[dataset_name]
                    group_loader.register_dataset(
                        dataset_name,
                        dataset_info.source_type,
                        dataset_info.source_path,
                        dataset_info.target_conversations,
                        dataset_info.priority,
                    )

                group_result = await group_loader.load_all_datasets(
                    max_concurrent=group_concurrency
                )
                datasets.update(group_result)

                logger.info(
                    f"Priority {priority} group completed: {len(group_result)} datasets loaded"
                )

            except Exception as e:
                logger.error(f"Priority {priority} group failed: {e}")
                for dataset_name in group:
                    self.failed_datasets.add(dataset_name)

            # Stop monitoring
            for dataset_name in group:
                self.acquisition_monitor.stop_monitoring(dataset_name)

        return datasets

    async def _validate_quality(
        self, datasets: dict[str, list[Conversation]]
    ) -> dict[str, list[Conversation]]:
        """Validate quality of loaded datasets and filter conversations."""

        validated_datasets = {}
        total_conversations = 0
        accepted_conversations = 0

        for dataset_name, conversations in datasets.items():
            logger.info(f"Validating quality for dataset: {dataset_name}")

            validated_conversations = []

            for conversation in conversations:
                # Process conversation through quality monitor
                quality_scores = self.acquisition_monitor.process_conversation(
                    conversation, dataset_name, 0.1  # Placeholder processing time
                )

                # Check if conversation meets quality threshold
                if (
                    quality_scores.get("quality_score", 0)
                    >= self.config.quality_threshold
                ):
                    validated_conversations.append(conversation)
                    accepted_conversations += 1

                total_conversations += 1

            validated_datasets[dataset_name] = validated_conversations

            acceptance_rate = (
                len(validated_conversations) / len(conversations)
                if conversations
                else 0
            )
            logger.info(
                f"Dataset {dataset_name}: {len(validated_conversations)}/{len(conversations)} "
                f"conversations accepted ({acceptance_rate:.1%})"
            )

        # Update metrics
        self.metrics.total_conversations = total_conversations
        self.metrics.accepted_conversations = accepted_conversations
        self.metrics.rejected_conversations = (
            total_conversations - accepted_conversations
        )

        if total_conversations > 0:
            self.metrics.error_rate = (
                self.metrics.rejected_conversations / total_conversations
            )

        return validated_datasets

    async def _execute_error_recovery(self) -> dict[str, list[Conversation]]:
        """Execute error recovery for failed datasets."""

        recovered_datasets = {}

        for dataset_name in list(self.failed_datasets):
            retry_count = self.retry_counts.get(dataset_name, 0)

            if retry_count < self.config.retry_attempts:
                logger.info(
                    f"Attempting recovery for dataset: {dataset_name} (attempt {retry_count + 1})"
                )

                try:
                    # Wait before retry
                    await asyncio.sleep(self.config.retry_delay)

                    # Create single dataset loader
                    recovery_loader = PixelDatasetLoader()
                    dataset_info = self.dataset_loader.datasets[dataset_name]
                    recovery_loader.register_dataset(
                        dataset_name,
                        dataset_info.source_type,
                        dataset_info.source_path,
                        dataset_info.target_conversations,
                        dataset_info.priority,
                    )

                    # Attempt recovery
                    result = await recovery_loader.load_all_datasets(max_concurrent=1)

                    if result.get(dataset_name):
                        recovered_datasets[dataset_name] = result[dataset_name]
                        self.failed_datasets.remove(dataset_name)
                        self.metrics.completed_datasets += 1
                        self.metrics.failed_datasets -= 1
                        logger.info(f"Successfully recovered dataset: {dataset_name}")
                    else:
                        self.retry_counts[dataset_name] = retry_count + 1
                        self.metrics.retry_count += 1

                except Exception as e:
                    self.retry_counts[dataset_name] = retry_count + 1
                    self.metrics.retry_count += 1
                    self._log_error(f"{dataset_name}_recovery", e)
            else:
                logger.error(f"Dataset {dataset_name} exceeded maximum retry attempts")

        return recovered_datasets

    async def _finalize_pipeline(
        self, datasets: dict[str, list[Conversation]]
    ) -> ExecutionResult:
        """Finalize pipeline execution and generate results."""

        self.metrics.end_time = datetime.now()

        # Calculate final metrics
        total_time = (self.metrics.end_time - self.metrics.start_time).total_seconds()
        total_conversations = sum(len(convs) for convs in datasets.values())

        if total_time > 0:
            self.metrics.processing_rate = total_conversations / total_time

        # Calculate performance score
        self.metrics.performance_score = self._calculate_performance_score()

        # Generate recommendations
        recommendations = self._generate_recommendations()

        # Save datasets to output directory
        await self._save_datasets(datasets)

        # Generate execution report
        execution_report = self._generate_execution_report()

        # Save execution report
        report_path = (
            self.config.report_directory
            / f"pipeline_execution_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        )
        write_json(str(report_path), execution_report)

        # Determine success
        success = (
            self.metrics.failed_datasets == 0
            and self.metrics.error_rate <= self.config.error_tolerance
            and self.metrics.quality_score >= self.config.quality_threshold
        )

        return ExecutionResult(
            success=success,
            metrics=self.metrics,
            datasets=datasets,
            errors=[
                error["message"] for error in self.error_log if error["type"] == "error"
            ],
            warnings=[
                error["message"]
                for error in self.error_log
                if error["type"] == "warning"
            ],
            recommendations=recommendations,
            execution_report=execution_report,
        )

    def _calculate_performance_score(self) -> float:
        """Calculate overall performance score."""

        # Factors: completion rate, quality score, processing rate, error rate
        completion_rate = self.metrics.completed_datasets / max(
            self.metrics.total_datasets, 1
        )
        quality_factor = self.metrics.quality_score

        # Normalize processing rate (assume 10 conv/s is excellent)
        rate_factor = min(self.metrics.processing_rate / 10.0, 1.0)

        # Error penalty
        error_penalty = 1.0 - self.metrics.error_rate

        # Weighted average
        return (
            completion_rate * 0.3
            + quality_factor * 0.3
            + rate_factor * 0.2
            + error_penalty * 0.2
        )


    def _generate_recommendations(self) -> list[str]:
        """Generate recommendations based on execution metrics."""

        recommendations = []

        # Completion rate recommendations
        if self.metrics.completed_datasets / max(self.metrics.total_datasets, 1) < 0.9:
            recommendations.append(
                "Consider investigating failed datasets and improving error handling"
            )

        # Quality recommendations
        if self.metrics.quality_score < 0.8:
            recommendations.append(
                "Review quality thresholds and consider improving data sources"
            )

        # Performance recommendations
        if self.metrics.processing_rate < 5.0:
            recommendations.append(
                "Consider optimizing processing pipeline or increasing concurrency"
            )

        # Error rate recommendations
        if self.metrics.error_rate > 0.05:
            recommendations.append(
                "High error rate detected - review error logs and improve data validation"
            )

        # Execution mode recommendations
        if (
            self.config.execution_mode == ExecutionMode.SEQUENTIAL
            and self.metrics.total_datasets > 5
        ):
            recommendations.append(
                "Consider using concurrent or adaptive execution mode for better performance"
            )

        return recommendations

    async def _save_datasets(self, datasets: dict[str, list[Conversation]]) -> None:
        """Save processed datasets to output directory."""

        for dataset_name, conversations in datasets.items():
            if conversations:
                output_path = (
                    self.config.output_directory / f"{dataset_name}_processed.json"
                )

                # Convert conversations to serializable format
                serializable_data = {
                    "dataset_name": dataset_name,
                    "conversation_count": len(conversations),
                    "processed_at": datetime.now().isoformat(),
                    "conversations": [conv.to_dict() for conv in conversations],
                }

                write_json(str(output_path), serializable_data)
                logger.info(
                    f"Saved {len(conversations)} conversations to {output_path}"
                )

    def _generate_execution_report(self) -> dict[str, Any]:
        """Generate comprehensive execution report."""

        return {
            "execution_summary": {
                "start_time": self.metrics.start_time.isoformat(),
                "end_time": (
                    self.metrics.end_time.isoformat() if self.metrics.end_time else None
                ),
                "total_duration": (
                    str(self.metrics.end_time - self.metrics.start_time)
                    if self.metrics.end_time
                    else None
                ),
                "current_stage": self.metrics.current_stage.value,
                "success": self.metrics.failed_datasets == 0,
            },
            "dataset_metrics": {
                "total_datasets": self.metrics.total_datasets,
                "completed_datasets": self.metrics.completed_datasets,
                "failed_datasets": self.metrics.failed_datasets,
                "completion_rate": self.metrics.completed_datasets
                / max(self.metrics.total_datasets, 1),
            },
            "conversation_metrics": {
                "total_conversations": self.metrics.total_conversations,
                "accepted_conversations": self.metrics.accepted_conversations,
                "rejected_conversations": self.metrics.rejected_conversations,
                "acceptance_rate": self.metrics.accepted_conversations
                / max(self.metrics.total_conversations, 1),
            },
            "quality_metrics": {
                "average_quality_score": self.metrics.quality_score,
                "quality_threshold": self.config.quality_threshold,
                "quality_threshold_met": self.metrics.quality_score
                >= self.config.quality_threshold,
            },
            "performance_metrics": {
                "processing_rate": self.metrics.processing_rate,
                "performance_score": self.metrics.performance_score,
                "error_rate": self.metrics.error_rate,
                "retry_count": self.metrics.retry_count,
            },
            "dataset_performance": self.dataset_performance,
            "stage_timings": {
                stage.value: timing for stage, timing in self.stage_timings.items()
            },
            "error_log": self.error_log,
            "configuration": {
                "execution_mode": self.config.execution_mode.value,
                "max_concurrent_datasets": self.config.max_concurrent_datasets,
                "quality_threshold": self.config.quality_threshold,
                "error_tolerance": self.config.error_tolerance,
                "retry_attempts": self.config.retry_attempts,
            },
        }

    def _log_error(self, context: str, error: Exception) -> None:
        """Log an error with context."""

        error_entry = {
            "type": "error",
            "context": context,
            "message": str(error),
            "timestamp": datetime.now().isoformat(),
        }

        self.error_log.append(error_entry)
        self.metrics.total_errors += 1

        # Notify error callbacks
        for callback in self.error_callbacks:
            try:
                callback(context, error)
            except Exception as e:
                logger.error(f"Error in error callback: {e}")

    def add_stage_callback(self, callback: Callable[[PipelineStage], None]) -> None:
        """Add callback for pipeline stage changes."""
        self.stage_callbacks.append(callback)

    def add_progress_callback(
        self, callback: Callable[[PipelineMetrics], None]
    ) -> None:
        """Add callback for progress updates."""
        self.progress_callbacks.append(callback)

    def add_error_callback(self, callback: Callable[[str, Exception], None]) -> None:
        """Add callback for error notifications."""
        self.error_callbacks.append(callback)

    def get_current_metrics(self) -> PipelineMetrics:
        """Get current pipeline metrics."""
        return self.metrics

    def get_execution_context(self) -> dict[str, Any]:
        """Get current execution context."""
        return self.execution_context.copy()

    def shutdown(self) -> None:
        """Shutdown the pipeline orchestrator."""

        # Shutdown monitoring
        self.acquisition_monitor.shutdown()

        logger.info("Pipeline Orchestrator shutdown complete")

    async def _save_datasets(self, datasets: dict[str, list[Conversation]]) -> None:
        """Save processed datasets to output directory."""

        for dataset_name, conversations in datasets.items():
            if conversations:
                output_path = (
                    self.config.output_directory / f"{dataset_name}_processed.json"
                )

                # Convert conversations to serializable format
                serializable_data = {
                    "dataset_name": dataset_name,
                    "conversation_count": len(conversations),
                    "processed_at": datetime.now().isoformat(),
                    "conversations": [conv.to_dict() for conv in conversations],
                }

                write_json(str(output_path), serializable_data)
                logger.info(
                    f"Saved {len(conversations)} conversations to {output_path}"
                )

    def _generate_execution_report(self) -> dict[str, Any]:
        """Generate comprehensive execution report."""

        return {
            "execution_summary": {
                "start_time": self.metrics.start_time.isoformat(),
                "end_time": (
                    self.metrics.end_time.isoformat() if self.metrics.end_time else None
                ),
                "total_duration": (
                    str(self.metrics.end_time - self.metrics.start_time)
                    if self.metrics.end_time
                    else None
                ),
                "current_stage": self.metrics.current_stage.value,
                "success": self.metrics.failed_datasets == 0,
            },
            "dataset_metrics": {
                "total_datasets": self.metrics.total_datasets,
                "completed_datasets": self.metrics.completed_datasets,
                "failed_datasets": self.metrics.failed_datasets,
                "completion_rate": self.metrics.completed_datasets
                / max(self.metrics.total_datasets, 1),
            },
            "conversation_metrics": {
                "total_conversations": self.metrics.total_conversations,
                "accepted_conversations": self.metrics.accepted_conversations,
                "rejected_conversations": self.metrics.rejected_conversations,
                "acceptance_rate": self.metrics.accepted_conversations
                / max(self.metrics.total_conversations, 1),
            },
            "quality_metrics": {
                "average_quality_score": self.metrics.quality_score,
                "quality_threshold": self.config.quality_threshold,
                "quality_threshold_met": self.metrics.quality_score
                >= self.config.quality_threshold,
            },
            "performance_metrics": {
                "processing_rate": self.metrics.processing_rate,
                "performance_score": self.metrics.performance_score,
                "error_rate": self.metrics.error_rate,
                "retry_count": self.metrics.retry_count,
            },
            "dataset_performance": self.dataset_performance,
            "stage_timings": {
                stage.value: timing for stage, timing in self.stage_timings.items()
            },
            "error_log": self.error_log,
            "configuration": {
                "execution_mode": self.config.execution_mode.value,
                "max_concurrent_datasets": self.config.max_concurrent_datasets,
                "quality_threshold": self.config.quality_threshold,
                "error_tolerance": self.config.error_tolerance,
                "retry_attempts": self.config.retry_attempts,
            },
        }

    def _log_error(self, context: str, error: Exception) -> None:
        """Log an error with context."""

        error_entry = {
            "type": "error",
            "context": context,
            "message": str(error),
            "timestamp": datetime.now().isoformat(),
        }

        self.error_log.append(error_entry)
        self.metrics.total_errors += 1

        # Notify error callbacks
        for callback in self.error_callbacks:
            try:
                callback(context, error)
            except Exception as e:
                logger.error(f"Error in error callback: {e}")

    def add_stage_callback(self, callback: Callable[[PipelineStage], None]) -> None:
        """Add callback for pipeline stage changes."""
        self.stage_callbacks.append(callback)

    def add_progress_callback(
        self, callback: Callable[[PipelineMetrics], None]
    ) -> None:
        """Add callback for progress updates."""
        self.progress_callbacks.append(callback)

    def add_error_callback(self, callback: Callable[[str, Exception], None]) -> None:
        """Add callback for error notifications."""
        self.error_callbacks.append(callback)

    def get_current_metrics(self) -> PipelineMetrics:
        """Get current pipeline metrics."""
        return self.metrics

    def get_execution_context(self) -> dict[str, Any]:
        """Get current execution context."""
        return self.execution_context.copy()

    async def create_checkpoint(self) -> str:
        """Create a checkpoint of current pipeline state."""

        checkpoint_id = f"checkpoint_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        checkpoint_data = {
            "checkpoint_id": checkpoint_id,
            "timestamp": datetime.now().isoformat(),
            "metrics": {
                "completed_datasets": self.metrics.completed_datasets,
                "failed_datasets": self.metrics.failed_datasets,
                "total_conversations": self.metrics.total_conversations,
                "current_stage": self.metrics.current_stage.value,
            },
            "failed_datasets": list(self.failed_datasets),
            "retry_counts": self.retry_counts.copy(),
            "execution_context": self.execution_context.copy(),
        }

        checkpoint_path = self.config.cache_directory / f"{checkpoint_id}.json"
        write_json(str(checkpoint_path), checkpoint_data)

        self.checkpoints.append(checkpoint_data)
        logger.info(f"Created checkpoint: {checkpoint_id}")

        return checkpoint_id

    async def restore_checkpoint(self, checkpoint_id: str) -> bool:
        """Restore pipeline state from checkpoint."""

        try:
            checkpoint_path = self.config.cache_directory / f"{checkpoint_id}.json"
            checkpoint_data = read_json(str(checkpoint_path))

            # Restore metrics
            self.metrics.completed_datasets = checkpoint_data["metrics"][
                "completed_datasets"
            ]
            self.metrics.failed_datasets = checkpoint_data["metrics"]["failed_datasets"]
            self.metrics.total_conversations = checkpoint_data["metrics"][
                "total_conversations"
            ]
            self.current_stage = PipelineStage(
                checkpoint_data["metrics"]["current_stage"]
            )

            # Restore state
            self.failed_datasets = set(checkpoint_data["failed_datasets"])
            self.retry_counts = checkpoint_data["retry_counts"]
            self.execution_context = checkpoint_data["execution_context"]

            logger.info(f"Restored checkpoint: {checkpoint_id}")
            return True

        except Exception as e:
            logger.error(f"Failed to restore checkpoint {checkpoint_id}: {e}")
            return False

    def shutdown(self) -> None:
        """Shutdown the pipeline orchestrator."""

        # Shutdown monitoring
        self.acquisition_monitor.shutdown()

        logger.info("Pipeline Orchestrator shutdown complete")
