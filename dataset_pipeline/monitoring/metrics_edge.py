#!/usr/bin/env python3
"""
Edge Training Metrics and Monitoring
Tracks crisis response accuracy, empathy scores, edge scenario success rates,
and resource utilization for edge training as described in the expanded brief.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any, Union
from datetime import datetime
from enum import Enum
import json
from pathlib import Path

from ..types.edge_categories import EdgeCategory, IntensityLevel
from ..storage_config import get_dataset_pipeline_output_root
from ..utils.logger import get_logger

logger = get_logger("dataset_pipeline.monitoring.metrics_edge")


class MetricType(Enum):
    """Types of metrics tracked"""
    TRAINING_LOSS = "training_loss"
    ACCURACY = "accuracy"
    PERPLEXITY = "perplexity"
    THROUGHPUT = "throughput"
    CRISIS_RESPONSE_ACCURACY = "crisis_response_accuracy"
    EMPATHY_SCORE = "empathy_score"
    EDGE_SCENARIO_SUCCESS_RATE = "edge_scenario_success_rate"
    GPU_UTILIZATION = "gpu_utilization"
    GPU_MEMORY = "gpu_memory"
    CPU_UTILIZATION = "cpu_utilization"
    MEMORY_USAGE = "memory_usage"
    TRAINING_HOURS = "training_hours"


@dataclass
class TrainingMetrics:
    """Core training performance metrics"""
    loss: float
    accuracy: float
    perplexity: Optional[float] = None
    throughput: Optional[float] = None  # examples/second
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())


@dataclass
class CrisisResponseMetrics:
    """Crisis response accuracy metrics"""
    crisis_response_accuracy: float  # 0.0 to 1.0
    by_category: Dict[str, float] = field(default_factory=dict)  # EdgeCategory -> accuracy
    by_intensity: Dict[str, float] = field(default_factory=dict)  # IntensityLevel -> accuracy
    total_crisis_examples: int = 0
    correctly_handled: int = 0
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())


@dataclass
class EmpathyMetrics:
    """Empathy score tracking metrics"""
    empathy_score: float  # 0.0 to 1.0
    empathy_by_stage: Dict[str, float] = field(default_factory=dict)  # stage_id -> score
    empathy_by_tone: Dict[str, float] = field(default_factory=dict)  # tone -> score
    total_examples: int = 0
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())


@dataclass
class EdgeScenarioMetrics:
    """Edge scenario success rate metrics"""
    edge_scenario_success_rate: float  # 0.0 to 1.0
    by_category: Dict[str, float] = field(default_factory=dict)  # EdgeCategory -> success rate
    nightmare_fuel_success_rate: float = 0.0  # Overall nightmare scenario success
    total_edge_scenarios: int = 0
    successful_scenarios: int = 0
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())


@dataclass
class ResourceMetrics:
    """Resource utilization metrics"""
    gpu_utilization: float  # 0.0 to 100.0
    gpu_memory_used_gb: float
    gpu_memory_total_gb: float
    cpu_utilization: float  # 0.0 to 100.0
    memory_usage_gb: float
    memory_total_gb: float
    multi_gpu_coordination: Optional[Dict[str, Any]] = None
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())


@dataclass
class EdgeTrainingMetrics:
    """Comprehensive edge training metrics report"""
    run_id: str
    training_metrics: TrainingMetrics
    crisis_metrics: CrisisResponseMetrics
    empathy_metrics: EmpathyMetrics
    edge_scenario_metrics: EdgeScenarioMetrics
    resource_metrics: ResourceMetrics
    training_hours: float
    total_examples: int
    edge_examples: int
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize metrics to dictionary"""
        return {
            "run_id": self.run_id,
            "training_metrics": {
                "loss": self.training_metrics.loss,
                "accuracy": self.training_metrics.accuracy,
                "perplexity": self.training_metrics.perplexity,
                "throughput": self.training_metrics.throughput,
                "timestamp": self.training_metrics.timestamp,
            },
            "crisis_metrics": {
                "crisis_response_accuracy": self.crisis_metrics.crisis_response_accuracy,
                "by_category": self.crisis_metrics.by_category,
                "by_intensity": self.crisis_metrics.by_intensity,
                "total_crisis_examples": self.crisis_metrics.total_crisis_examples,
                "correctly_handled": self.crisis_metrics.correctly_handled,
                "timestamp": self.crisis_metrics.timestamp,
            },
            "empathy_metrics": {
                "empathy_score": self.empathy_metrics.empathy_score,
                "empathy_by_stage": self.empathy_metrics.empathy_by_stage,
                "empathy_by_tone": self.empathy_metrics.empathy_by_tone,
                "total_examples": self.empathy_metrics.total_examples,
                "timestamp": self.empathy_metrics.timestamp,
            },
            "edge_scenario_metrics": {
                "edge_scenario_success_rate": self.edge_scenario_metrics.edge_scenario_success_rate,
                "by_category": self.edge_scenario_metrics.by_category,
                "nightmare_fuel_success_rate": self.edge_scenario_metrics.nightmare_fuel_success_rate,
                "total_edge_scenarios": self.edge_scenario_metrics.total_edge_scenarios,
                "successful_scenarios": self.edge_scenario_metrics.successful_scenarios,
                "timestamp": self.edge_scenario_metrics.timestamp,
            },
            "resource_metrics": {
                "gpu_utilization": self.resource_metrics.gpu_utilization,
                "gpu_memory_used_gb": self.resource_metrics.gpu_memory_used_gb,
                "gpu_memory_total_gb": self.resource_metrics.gpu_memory_total_gb,
                "cpu_utilization": self.resource_metrics.cpu_utilization,
                "memory_usage_gb": self.resource_metrics.memory_usage_gb,
                "memory_total_gb": self.resource_metrics.memory_total_gb,
                "multi_gpu_coordination": self.resource_metrics.multi_gpu_coordination,
                "timestamp": self.resource_metrics.timestamp,
            },
            "training_hours": self.training_hours,
            "total_examples": self.total_examples,
            "edge_examples": self.edge_examples,
            "metadata": self.metadata,
        }


class EdgeMetricsCollector:
    """
    Collects and tracks edge training metrics.
    Supports real-time monitoring and historical analysis.
    """

    def __init__(self, output_dir: Optional[Union[str, Path]] = None):
        """
        Initialize metrics collector.
        
        Args:
            output_dir: Directory to save metrics reports
        """
        self.output_dir = Path(output_dir) if output_dir else get_dataset_pipeline_output_root() / "monitoring" / "metrics"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.metrics_history: List[EdgeTrainingMetrics] = []

    def record_training_step(
        self,
        run_id: str,
        loss: float,
        accuracy: float,
        perplexity: Optional[float] = None,
        throughput: Optional[float] = None,
    ) -> TrainingMetrics:
        """
        Record a training step.
        
        Args:
            run_id: Training run identifier
            loss: Training loss
            accuracy: Training accuracy
            perplexity: Optional perplexity
            throughput: Optional throughput (examples/second)
        
        Returns:
            TrainingMetrics object
        """
        return TrainingMetrics(
            loss=loss,
            accuracy=accuracy,
            perplexity=perplexity,
            throughput=throughput,
        )

    def record_crisis_response(
        self,
        run_id: str,
        category: EdgeCategory,
        intensity: IntensityLevel,
        correctly_handled: bool,
    ) -> None:
        """
        Record a crisis response evaluation.
        
        Args:
            run_id: Training run identifier
            category: Edge category
            intensity: Intensity level
            correctly_handled: Whether the response was correct
        """
        logger.debug(
            f"Crisis response: {category.value}/{intensity.value} - "
            f"{'correct' if correctly_handled else 'incorrect'}"
        )

    def calculate_crisis_metrics(
        self,
        crisis_evaluations: List[Dict[str, Any]],
    ) -> CrisisResponseMetrics:
        """
        Calculate crisis response metrics from evaluations.
        
        Args:
            crisis_evaluations: List of evaluation results
        
        Returns:
            CrisisResponseMetrics
        """
        if not crisis_evaluations:
            return CrisisResponseMetrics(crisis_response_accuracy=0.0)
        
        total = len(crisis_evaluations)
        correct = sum(1 for e in crisis_evaluations if e.get("correctly_handled", False))
        accuracy = correct / total if total > 0 else 0.0
        
        # Calculate by category
        by_category = {}
        for cat in EdgeCategory:
            cat_evals = [e for e in crisis_evaluations if e.get("category") == cat.value]
            if cat_evals:
                cat_correct = sum(1 for e in cat_evals if e.get("correctly_handled", False))
                by_category[cat.value] = cat_correct / len(cat_evals)
        
        # Calculate by intensity
        by_intensity = {}
        for intensity in IntensityLevel:
            int_evals = [e for e in crisis_evaluations if e.get("intensity") == intensity.value]
            if int_evals:
                int_correct = sum(1 for e in int_evals if e.get("correctly_handled", False))
                by_intensity[intensity.value] = int_correct / len(int_evals)
        
        return CrisisResponseMetrics(
            crisis_response_accuracy=accuracy,
            by_category=by_category,
            by_intensity=by_intensity,
            total_crisis_examples=total,
            correctly_handled=correct,
        )

    def calculate_empathy_metrics(
        self,
        empathy_scores: List[Dict[str, Any]],
    ) -> EmpathyMetrics:
        """
        Calculate empathy metrics from scores.
        
        Args:
            empathy_scores: List of empathy score records
        
        Returns:
            EmpathyMetrics
        """
        if not empathy_scores:
            return EmpathyMetrics(empathy_score=0.0)
        
        total = len(empathy_scores)
        avg_empathy = sum(s.get("score", 0.0) for s in empathy_scores) / total if total > 0 else 0.0
        
        # Calculate by stage
        by_stage = {}
        for stage_id in ["stage1_foundation", "stage2_therapeutic_expertise", "stage3_edge_stress_test", "stage4_voice_persona"]:
            stage_scores = [s for s in empathy_scores if s.get("stage") == stage_id]
            if stage_scores:
                by_stage[stage_id] = sum(s.get("score", 0.0) for s in stage_scores) / len(stage_scores)
        
        # Calculate by tone
        by_tone = {}
        for tone in ["FOUNDATION", "CLINICAL", "CRISIS_DIRECT"]:
            tone_scores = [s for s in empathy_scores if s.get("tone") == tone]
            if tone_scores:
                by_tone[tone] = sum(s.get("score", 0.0) for s in tone_scores) / len(tone_scores)
        
        return EmpathyMetrics(
            empathy_score=avg_empathy,
            empathy_by_stage=by_stage,
            empathy_by_tone=by_tone,
            total_examples=total,
        )

    def calculate_edge_scenario_metrics(
        self,
        scenario_results: List[Dict[str, Any]],
    ) -> EdgeScenarioMetrics:
        """
        Calculate edge scenario success rate metrics.
        
        Args:
            scenario_results: List of scenario evaluation results
        
        Returns:
            EdgeScenarioMetrics
        """
        if not scenario_results:
            return EdgeScenarioMetrics(edge_scenario_success_rate=0.0)
        
        total = len(scenario_results)
        successful = sum(1 for r in scenario_results if r.get("success", False))
        success_rate = successful / total if total > 0 else 0.0
        
        # Calculate by category
        by_category = {}
        for cat in EdgeCategory:
            cat_results = [r for r in scenario_results if r.get("category") == cat.value]
            if cat_results:
                cat_success = sum(1 for r in cat_results if r.get("success", False))
                by_category[cat.value] = cat_success / len(cat_results)
        
        return EdgeScenarioMetrics(
            edge_scenario_success_rate=success_rate,
            by_category=by_category,
            nightmare_fuel_success_rate=success_rate,  # Simplified - would be more nuanced
            total_edge_scenarios=total,
            successful_scenarios=successful,
        )

    def collect_resource_metrics(
        self,
        gpu_utilization: float,
        gpu_memory_used_gb: float,
        gpu_memory_total_gb: float,
        cpu_utilization: float,
        memory_usage_gb: float,
        memory_total_gb: float,
        multi_gpu_info: Optional[Dict[str, Any]] = None,
    ) -> ResourceMetrics:
        """
        Collect resource utilization metrics.
        
        Args:
            gpu_utilization: GPU utilization percentage
            gpu_memory_used_gb: GPU memory used in GB
            gpu_memory_total_gb: Total GPU memory in GB
            cpu_utilization: CPU utilization percentage
            memory_usage_gb: System memory used in GB
            memory_total_gb: Total system memory in GB
            multi_gpu_info: Optional multi-GPU coordination info
        
        Returns:
            ResourceMetrics
        """
        return ResourceMetrics(
            gpu_utilization=gpu_utilization,
            gpu_memory_used_gb=gpu_memory_used_gb,
            gpu_memory_total_gb=gpu_memory_total_gb,
            cpu_utilization=cpu_utilization,
            memory_usage_gb=memory_usage_gb,
            memory_total_gb=memory_total_gb,
            multi_gpu_coordination=multi_gpu_info,
        )

    def generate_metrics_report(
        self,
        run_id: str,
        training_metrics: TrainingMetrics,
        crisis_metrics: CrisisResponseMetrics,
        empathy_metrics: EmpathyMetrics,
        edge_scenario_metrics: EdgeScenarioMetrics,
        resource_metrics: ResourceMetrics,
        training_hours: float,
        total_examples: int,
        edge_examples: int,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> EdgeTrainingMetrics:
        """
        Generate comprehensive metrics report.
        
        Args:
            run_id: Training run identifier
            training_metrics: Training performance metrics
            crisis_metrics: Crisis response metrics
            empathy_metrics: Empathy metrics
            edge_scenario_metrics: Edge scenario metrics
            resource_metrics: Resource utilization metrics
            training_hours: Total training hours
            total_examples: Total training examples
            edge_examples: Number of edge examples
            metadata: Optional additional metadata
        
        Returns:
            EdgeTrainingMetrics report
        """
        report = EdgeTrainingMetrics(
            run_id=run_id,
            training_metrics=training_metrics,
            crisis_metrics=crisis_metrics,
            empathy_metrics=empathy_metrics,
            edge_scenario_metrics=edge_scenario_metrics,
            resource_metrics=resource_metrics,
            training_hours=training_hours,
            total_examples=total_examples,
            edge_examples=edge_examples,
            metadata=metadata or {},
        )
        
        # Save to history
        self.metrics_history.append(report)
        
        # Save to file
        self._save_report(report)
        
        return report

    def _save_report(self, report: EdgeTrainingMetrics):
        """Save metrics report to file"""
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        filename = f"edge_metrics_{report.run_id}_{timestamp}.json"
        filepath = self.output_dir / filename
        
        with open(filepath, 'w') as f:
            json.dump(report.to_dict(), f, indent=2)
        
        logger.info(f"Saved metrics report to {filepath}")

    def export_metrics_summary(
        self,
        output_path: Optional[Union[str, Path]] = None,
    ) -> Dict[str, Any]:
        """
        Export summary of all collected metrics.
        
        Args:
            output_path: Optional path to save summary
        
        Returns:
            Summary dictionary
        """
        if not self.metrics_history:
            return {"message": "No metrics collected yet"}
        
        # Aggregate metrics
        avg_crisis_accuracy = sum(
            m.crisis_metrics.crisis_response_accuracy
            for m in self.metrics_history
        ) / len(self.metrics_history)
        
        avg_empathy = sum(
            m.empathy_metrics.empathy_score
            for m in self.metrics_history
        ) / len(self.metrics_history)
        
        avg_edge_success = sum(
            m.edge_scenario_metrics.edge_scenario_success_rate
            for m in self.metrics_history
        ) / len(self.metrics_history)
        
        summary = {
            "total_runs": len(self.metrics_history),
            "average_crisis_response_accuracy": avg_crisis_accuracy,
            "average_empathy_score": avg_empathy,
            "average_edge_scenario_success_rate": avg_edge_success,
            "runs": [m.to_dict() for m in self.metrics_history],
        }
        
        if output_path:
            with open(output_path, 'w') as f:
                json.dump(summary, f, indent=2)
            logger.info(f"Exported metrics summary to {output_path}")
        
        return summary
