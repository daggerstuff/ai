"""
Voice data optimization pipeline with systematic consistency validation.
Provides comprehensive pipeline for optimizing voice-derived training data.
"""

import asyncio
import json
import statistics
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from ai.dataset_pipeline.conversation_schema import Conversation
from ai.dataset_pipeline.logger import get_logger
from ai.dataset_pipeline.personality_extractor import PersonalityExtractor
from ai.dataset_pipeline.voice_training_optimizer import PersonalityProfile, VoiceTrainingOptimizer


@dataclass
class PipelineStage:
    """Configuration for a pipeline stage."""
    name: str
    processor: Callable
    enabled: bool = True
    config: dict[str, Any] = field(default_factory=dict)
    dependencies: list[str] = field(default_factory=list)


@dataclass
class ValidationRule:
    """Validation rule for consistency checking."""
    name: str
    validator: Callable
    threshold: float
    weight: float = 1.0
    critical: bool = False
    description: str = ""


@dataclass
class PipelineMetrics:
    """Comprehensive pipeline metrics."""
    total_input_conversations: int = 0
    conversations_processed: int = 0
    conversations_passed: int = 0
    conversations_failed: int = 0
    stage_metrics: dict[str, dict[str, float]] = field(default_factory=dict)
    validation_scores: dict[str, float] = field(default_factory=dict)
    processing_time: float = 0.0
    quality_improvement: float = 0.0
    consistency_improvement: float = 0.0


@dataclass
class PipelineResult:
    """Result of voice optimization pipeline."""
    success: bool
    optimized_conversations: list[Conversation] = field(default_factory=list)
    baseline_profile: PersonalityProfile | None = None
    metrics: PipelineMetrics | None = None
    stage_results: dict[str, Any] = field(default_factory=dict)
    validation_report: dict[str, Any] = field(default_factory=dict)
    issues: list[str] = field(default_factory=list)


class VoiceOptimizationPipeline:
    """
    Comprehensive voice data optimization pipeline.

    Features:
    - Multi-stage processing with dependency management
    - Systematic consistency validation
    - Quality improvement tracking
    - Configurable validation rules
    - Performance monitoring and optimization
    - Batch and streaming processing support
    """

    def __init__(
        self,
        voice_optimizer: VoiceTrainingOptimizer | None = None,
        personality_extractor: PersonalityExtractor | None = None
    ):
        """
        Initialize VoiceOptimizationPipeline.

        Args:
            voice_optimizer: VoiceTrainingOptimizer instance
            personality_extractor: PersonalityExtractor instance
        """
        self.voice_optimizer = voice_optimizer or VoiceTrainingOptimizer()
        self.personality_extractor = personality_extractor or PersonalityExtractor()
        self.logger = get_logger(__name__)

        # Pipeline configuration
        self.stages: list[PipelineStage] = []
        self.validation_rules: list[ValidationRule] = []

        # Initialize default pipeline
        self._initialize_default_pipeline()
        self._initialize_default_validation_rules()

        # Processing state
        self.current_metrics = PipelineMetrics()
        self.processing_history: list[PipelineResult] = []

        self.logger.info("VoiceOptimizationPipeline initialized with default configuration")

    def add_stage(self, stage: PipelineStage) -> None:
        """Add a processing stage to the pipeline."""
        self.stages.append(stage)
        self.logger.info(f"Added pipeline stage: {stage.name}")

    def add_validation_rule(self, rule: ValidationRule) -> None:
        """Add a validation rule to the pipeline."""
        self.validation_rules.append(rule)
        self.logger.info(f"Added validation rule: {rule.name}")

    def process_conversations(
        self,
        conversations: list[Conversation],
        source_metadata: dict[str, Any] | None = None
    ) -> PipelineResult:
        """
        Process conversations through the optimization pipeline.

        Args:
            conversations: List of voice-derived conversations
            source_metadata: Optional metadata about the voice source

        Returns:
            PipelineResult with optimized conversations and metrics
        """
        start_time = datetime.now()

        self.logger.info(f"Starting pipeline processing of {len(conversations)} conversations")

        # Initialize metrics
        self.current_metrics = PipelineMetrics(
            total_input_conversations=len(conversations)
        )

        # Process through stages
        current_conversations = conversations.copy()
        stage_results = {}

        try:
            # Execute pipeline stages in order
            for stage in self._get_ordered_stages():
                if not stage.enabled:
                    continue

                self.logger.info(f"Executing stage: {stage.name}")

                stage_start = datetime.now()
                stage_result = self._execute_stage(stage, current_conversations, source_metadata)
                stage_time = (datetime.now() - stage_start).total_seconds()

                # Update conversations and metrics
                current_conversations = stage_result.get("conversations", current_conversations)
                stage_results[stage.name] = stage_result

                # Record stage metrics
                self.current_metrics.stage_metrics[stage.name] = {
                    "processing_time": stage_time,
                    "input_count": stage_result.get("input_count", 0),
                    "output_count": stage_result.get("output_count", 0),
                    "quality_score": stage_result.get("quality_score", 0.0)
                }

                self.logger.info(f"Stage {stage.name} completed: "
                               f"{len(current_conversations)} conversations remaining")

            # Perform systematic validation
            validation_report = self._perform_systematic_validation(current_conversations)

            # Calculate final metrics
            processing_time = (datetime.now() - start_time).total_seconds()
            self.current_metrics.conversations_processed = len(conversations)
            self.current_metrics.conversations_passed = len(current_conversations)
            self.current_metrics.conversations_failed = len(conversations) - len(current_conversations)
            self.current_metrics.processing_time = processing_time

            # Calculate quality improvement
            if conversations:
                initial_quality = self._calculate_average_quality(conversations)
                final_quality = self._calculate_average_quality(current_conversations)
                self.current_metrics.quality_improvement = final_quality - initial_quality

            # Extract baseline profile
            baseline_profile = self.voice_optimizer.get_baseline_profile()

            result = PipelineResult(
                success=True,
                optimized_conversations=current_conversations,
                baseline_profile=baseline_profile,
                metrics=self.current_metrics,
                stage_results=stage_results,
                validation_report=validation_report
            )

            # Store result for analysis
            self.processing_history.append(result)

            self.logger.info(f"Pipeline processing complete: {len(current_conversations)}/{len(conversations)} "
                           f"conversations optimized (quality improvement: "
                           f"{self.current_metrics.quality_improvement:.3f})")

            return result

        except Exception as e:
            self.logger.error(f"Pipeline processing failed: {e}")
            return PipelineResult(
                success=False,
                issues=[str(e)],
                metrics=self.current_metrics
            )

    async def process_conversations_async(
        self,
        conversations: list[Conversation],
        source_metadata: dict[str, Any] | None = None
    ) -> PipelineResult:
        """Asynchronously process conversations through the pipeline."""
        loop = asyncio.get_event_loop()

        with ThreadPoolExecutor(max_workers=4) as executor:
            future = loop.run_in_executor(
                executor,
                self.process_conversations,
                conversations,
                source_metadata
            )
            return await future

    def validate_consistency(
        self,
        conversations: list[Conversation]
    ) -> dict[str, Any]:
        """
        Perform systematic consistency validation on conversations.

        Args:
            conversations: List of conversations to validate

        Returns:
            Comprehensive validation report
        """
        return self._perform_systematic_validation(conversations)

    def get_pipeline_metrics(self) -> PipelineMetrics:
        """Get current pipeline metrics."""
        return self.current_metrics

    def get_processing_history(self) -> list[PipelineResult]:
        """Get processing history for analysis."""
        return self.processing_history

    def export_pipeline_config(self, file_path: str) -> None:
        """Export pipeline configuration to file."""
        config = {
            "stages": [
                {
                    "name": stage.name,
                    "enabled": stage.enabled,
                    "config": stage.config,
                    "dependencies": stage.dependencies
                }
                for stage in self.stages
            ],
            "validation_rules": [
                {
                    "name": rule.name,
                    "threshold": rule.threshold,
                    "weight": rule.weight,
                    "critical": rule.critical,
                    "description": rule.description
                }
                for rule in self.validation_rules
            ]
        }

        with open(file_path, "w") as f:
            json.dump(config, f, indent=2)

        self.logger.info(f"Pipeline configuration exported to {file_path}")

    def import_pipeline_config(self, file_path: str) -> None:
        """Import pipeline configuration from file."""
        with open(file_path) as f:
            config = json.load(f)

        # Clear existing configuration
        self.stages.clear()
        self.validation_rules.clear()

        # Import stages (processors need to be registered separately)
        for stage_config in config.get("stages", []):
            # Note: This is a simplified import - actual processors need to be registered
            self.logger.warning(f"Stage {stage_config['name']} imported but processor needs registration")

        # Import validation rules (validators need to be registered separately)
        for rule_config in config.get("validation_rules", []):
            self.logger.warning(f"Validation rule {rule_config['name']} imported but validator needs registration")

        self.logger.info(f"Pipeline configuration imported from {file_path}")

    # Private methods

    def _initialize_default_pipeline(self) -> None:
        """Initialize default pipeline stages."""
        # Stage 1: Personality profiling
        self.add_stage(PipelineStage(
            name="personality_profiling",
            processor=self._stage_personality_profiling,
            config={"include_detailed_analysis": True}
        ))

        # Stage 2: Consistency filtering
        self.add_stage(PipelineStage(
            name="consistency_filtering",
            processor=self._stage_consistency_filtering,
            config={"min_consistency_threshold": 0.8},
            dependencies=["personality_profiling"]
        ))

        # Stage 3: Quality optimization
        self.add_stage(PipelineStage(
            name="quality_optimization",
            processor=self._stage_quality_optimization,
            config={"enable_empathy_scoring": True},
            dependencies=["consistency_filtering"]
        ))

        # Stage 4: Final validation
        self.add_stage(PipelineStage(
            name="final_validation",
            processor=self._stage_final_validation,
            dependencies=["quality_optimization"]
        ))

    def _initialize_default_validation_rules(self) -> None:
        """Initialize default validation rules."""
        # Personality consistency validation
        self.add_validation_rule(ValidationRule(
            name="personality_consistency",
            validator=self._validate_personality_consistency,
            threshold=0.8,
            weight=0.3,
            critical=True,
            description="Validates consistency of personality traits across conversations"
        ))

        # Empathy consistency validation
        self.add_validation_rule(ValidationRule(
            name="empathy_consistency",
            validator=self._validate_empathy_consistency,
            threshold=0.7,
            weight=0.25,
            description="Validates consistency of empathy expressions"
        ))

        # Authenticity validation
        self.add_validation_rule(ValidationRule(
            name="authenticity_validation",
            validator=self._validate_authenticity,
            threshold=0.75,
            weight=0.25,
            description="Validates authenticity of voice-derived content"
        ))

        # Communication style consistency
        self.add_validation_rule(ValidationRule(
            name="communication_style_consistency",
            validator=self._validate_communication_style,
            threshold=0.8,
            weight=0.2,
            description="Validates consistency of communication patterns"
        ))

    def _get_ordered_stages(self) -> list[PipelineStage]:
        """Get stages ordered by dependencies."""
        ordered_stages = []
        processed_stages = set()

        def can_process_stage(stage: PipelineStage) -> bool:
            return all(dep in processed_stages for dep in stage.dependencies)

        remaining_stages = self.stages.copy()

        while remaining_stages:
            # Find stages that can be processed
            ready_stages = [stage for stage in remaining_stages if can_process_stage(stage)]

            if not ready_stages:
                # Circular dependency or missing dependency
                self.logger.warning("Circular dependency detected in pipeline stages")
                break

            # Process ready stages
            for stage in ready_stages:
                ordered_stages.append(stage)
                processed_stages.add(stage.name)
                remaining_stages.remove(stage)

        return ordered_stages

    def _execute_stage(
        self,
        stage: PipelineStage,
        conversations: list[Conversation],
        source_metadata: dict[str, Any] | None
    ) -> dict[str, Any]:
        """Execute a pipeline stage."""
        try:
            return stage.processor(conversations, stage.config, source_metadata)
        except Exception as e:
            self.logger.error(f"Stage {stage.name} failed: {e}")
            return {
                "conversations": conversations,
                "input_count": len(conversations),
                "output_count": len(conversations),
                "quality_score": 0.0,
                "error": str(e)
            }

    # Default stage processors

    def _stage_personality_profiling(
        self,
        conversations: list[Conversation],
        config: dict[str, Any],
        source_metadata: dict[str, Any] | None
    ) -> dict[str, Any]:
        """Stage 1: Personality profiling."""
        processed_conversations = []

        for conversation in conversations:
            # Extract personality profile
            text_content = " ".join([msg.content for msg in conversation.messages])
            personality_analysis = self.personality_extractor.extract_personality(text_content)

            # Add personality metadata
            if not conversation.meta:
                conversation.meta = {}
            conversation.meta["personality_analysis"] = {
                "big_five_scores": personality_analysis.big_five_scores,
                "empathy_markers": personality_analysis.empathy_markers,
                "authenticity_indicators": personality_analysis.authenticity_indicators,
                "confidence": personality_analysis.confidence
            }

            processed_conversations.append(conversation)

        return {
            "conversations": processed_conversations,
            "input_count": len(conversations),
            "output_count": len(processed_conversations),
            "quality_score": statistics.mean([
                conv.meta.get("personality_analysis", {}).get("confidence", 0.0)
                for conv in processed_conversations
            ]) if processed_conversations else 0.0
        }

    def _stage_consistency_filtering(
        self,
        conversations: list[Conversation],
        config: dict[str, Any],
        source_metadata: dict[str, Any] | None
    ) -> dict[str, Any]:
        """Stage 2: Consistency filtering."""
        min_threshold = config.get("min_consistency_threshold", 0.8)

        # Use voice optimizer for consistency filtering
        optimization_result = self.voice_optimizer.optimize_voice_conversations(
            conversations, source_metadata
        )

        # Filter by consistency threshold
        consistent_conversations = [
            conv for conv in optimization_result.optimized_conversations
            if conv.meta.get("personality_consistency", 0.0) >= min_threshold
        ]

        return {
            "conversations": consistent_conversations,
            "input_count": len(conversations),
            "output_count": len(consistent_conversations),
            "quality_score": statistics.mean([
                conv.meta.get("personality_consistency", 0.0)
                for conv in consistent_conversations
            ]) if consistent_conversations else 0.0,
            "optimization_result": optimization_result
        }

    def _stage_quality_optimization(
        self,
        conversations: list[Conversation],
        config: dict[str, Any],
        source_metadata: dict[str, Any] | None
    ) -> dict[str, Any]:
        """Stage 3: Quality optimization."""
        optimized_conversations = []

        for conversation in conversations:
            # Calculate quality scores
            empathy_score = conversation.meta.get("empathy_score", 0.0)
            authenticity_score = conversation.meta.get("authenticity_score", 0.0)
            consistency_score = conversation.meta.get("personality_consistency", 0.0)

            # Calculate combined quality score
            quality_score = (
                empathy_score * 0.4 +
                authenticity_score * 0.3 +
                consistency_score * 0.3
            )

            # Add quality metadata
            conversation.meta["combined_quality_score"] = quality_score

            # Filter by quality threshold
            if quality_score >= 0.7:  # Configurable threshold
                optimized_conversations.append(conversation)

        return {
            "conversations": optimized_conversations,
            "input_count": len(conversations),
            "output_count": len(optimized_conversations),
            "quality_score": statistics.mean([
                conv.meta.get("combined_quality_score", 0.0)
                for conv in optimized_conversations
            ]) if optimized_conversations else 0.0
        }

    def _stage_final_validation(
        self,
        conversations: list[Conversation],
        config: dict[str, Any],
        source_metadata: dict[str, Any] | None
    ) -> dict[str, Any]:
        """Stage 4: Final validation."""
        validated_conversations = []
        validation_scores = []

        for conversation in conversations:
            # Apply all validation rules
            validation_passed = True
            conversation_validation_scores = []

            for rule in self.validation_rules:
                try:
                    score = rule.validator(conversation)
                    conversation_validation_scores.append(score * rule.weight)

                    if rule.critical and score < rule.threshold:
                        validation_passed = False
                        break

                except Exception as e:
                    self.logger.warning(f"Validation rule {rule.name} failed: {e}")
                    if rule.critical:
                        validation_passed = False
                        break

            if validation_passed:
                # Add validation metadata
                conversation.meta["validation_score"] = statistics.mean(conversation_validation_scores)
                validated_conversations.append(conversation)
                validation_scores.append(conversation.meta["validation_score"])

        return {
            "conversations": validated_conversations,
            "input_count": len(conversations),
            "output_count": len(validated_conversations),
            "quality_score": statistics.mean(validation_scores) if validation_scores else 0.0
        }

    def _perform_systematic_validation(
        self,
        conversations: list[Conversation]
    ) -> dict[str, Any]:
        """Perform systematic validation on conversations."""
        validation_report = {
            "total_conversations": len(conversations),
            "validation_results": {},
            "overall_score": 0.0,
            "passed_validations": 0,
            "failed_validations": 0
        }

        if not conversations:
            return validation_report

        # Apply each validation rule
        rule_scores = []

        for rule in self.validation_rules:
            rule_results = {
                "passed": 0,
                "failed": 0,
                "scores": [],
                "average_score": 0.0
            }

            for conversation in conversations:
                try:
                    score = rule.validator(conversation)
                    rule_results["scores"].append(score)

                    if score >= rule.threshold:
                        rule_results["passed"] += 1
                    else:
                        rule_results["failed"] += 1

                except Exception as e:
                    self.logger.warning(f"Validation {rule.name} failed for conversation: {e}")
                    rule_results["failed"] += 1
                    rule_results["scores"].append(0.0)

            # Calculate rule statistics
            if rule_results["scores"]:
                rule_results["average_score"] = statistics.mean(rule_results["scores"])
                rule_scores.append(rule_results["average_score"] * rule.weight)

            validation_report["validation_results"][rule.name] = rule_results

        # Calculate overall validation score
        if rule_scores:
            validation_report["overall_score"] = statistics.mean(rule_scores)

        # Count overall pass/fail
        validation_report["passed_validations"] = sum(
            result["passed"] for result in validation_report["validation_results"].values()
        )
        validation_report["failed_validations"] = sum(
            result["failed"] for result in validation_report["validation_results"].values()
        )

        return validation_report

    def _calculate_average_quality(self, conversations: list[Conversation]) -> float:
        """Calculate average quality score for conversations."""
        if not conversations:
            return 0.0

        quality_scores = []

        for conversation in conversations:
            # Try to get existing quality score
            quality_score = conversation.meta.get("combined_quality_score")

            if quality_score is None:
                # Calculate basic quality score
                empathy_score = conversation.meta.get("empathy_score", 0.0)
                authenticity_score = conversation.meta.get("authenticity_score", 0.0)
                consistency_score = conversation.meta.get("personality_consistency", 0.0)

                quality_score = (empathy_score + authenticity_score + consistency_score) / 3

            quality_scores.append(quality_score)

        return statistics.mean(quality_scores)

    # Default validation rule implementations

    def _validate_personality_consistency(self, conversation: Conversation) -> float:
        """Validate personality consistency for a conversation."""
        return conversation.meta.get("personality_consistency", 0.0)

    def _validate_empathy_consistency(self, conversation: Conversation) -> float:
        """Validate empathy consistency for a conversation."""
        return conversation.meta.get("empathy_score", 0.0)

    def _validate_authenticity(self, conversation: Conversation) -> float:
        """Validate authenticity for a conversation."""
        return conversation.meta.get("authenticity_score", 0.0)

    def _validate_communication_style(self, conversation: Conversation) -> float:
        """Validate communication style consistency for a conversation."""
        # This would typically involve more sophisticated analysis
        # For now, use a simple heuristic based on available metadata
        personality_analysis = conversation.meta.get("personality_analysis", {})
        return personality_analysis.get("confidence", 0.0)

        # Communication style consistency correlates with analysis confidence
