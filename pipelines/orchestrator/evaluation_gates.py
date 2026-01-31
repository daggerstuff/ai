"""
Automated evaluation gates for promoting models from training to staging/production.
Implements threshold-based decision making for model promotion.
"""

import json
import os
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from datetime import datetime
import logging
from enum import Enum
import yaml
from pathlib import Path


logger = logging.getLogger(__name__)


class PromotionStage(Enum):
    """Stages in the model promotion pipeline"""
    TRAINING = "training"
    STAGING = "staging"
    PRODUCTION = "production"
    REJECTED = "rejected"


class GateType(Enum):
    """Types of evaluation gates"""
    MINIMUM_THRESHOLD = "minimum_threshold"
    MAXIMUM_THRESHOLD = "maximum_threshold"
    RANGE_THRESHOLD = "range_threshold"
    COMPARISON_THRESHOLD = "comparison_threshold"
    CUSTOM_EVALUATION = "custom_evaluation"


@dataclass
class EvaluationGate:
    """Definition of a single evaluation gate"""
    name: str
    metric_name: str
    gate_type: GateType
    threshold_value: float
    secondary_threshold: Optional[float] = None  # For range thresholds
    weight: float = 1.0  # Weight in overall scoring
    is_critical: bool = False  # If True, failure means automatic rejection
    description: str = ""
    
    def evaluate(self, metric_value: float) -> Tuple[bool, str]:
        """Evaluate if the metric passes this gate"""
        if self.gate_type == GateType.MINIMUM_THRESHOLD:
            passed = metric_value >= self.threshold_value
            reason = f"Value {metric_value} {'meets' if passed else 'fails'} minimum threshold {self.threshold_value}"
        elif self.gate_type == GateType.MAXIMUM_THRESHOLD:
            passed = metric_value <= self.threshold_value
            reason = f"Value {metric_value} {'meets' if passed else 'fails'} maximum threshold {self.threshold_value}"
        elif self.gate_type == GateType.RANGE_THRESHOLD:
            if self.secondary_threshold is not None:
                passed = self.threshold_value <= metric_value <= self.secondary_threshold
                reason = f"Value {metric_value} {'is in' if passed else 'is out of'} range [{self.threshold_value}, {self.secondary_threshold}]"
            else:
                passed = False
                reason = f"Range threshold requires secondary threshold value, got None"
        elif self.gate_type == GateType.COMPARISON_THRESHOLD:
            # For comparison, threshold_value is typically 0 for "no worse than baseline"
            # Positive values mean it can be that much worse
            passed = metric_value <= self.threshold_value
            reason = f"Value {metric_value} {'meets' if passed else 'fails'} comparison threshold {self.threshold_value}"
        else:
            passed = False
            reason = f"Unknown gate type: {self.gate_type}"
        
        return passed, reason


@dataclass
class GateConfiguration:
    """Configuration for evaluation gates"""
    gates: List[EvaluationGate] = field(default_factory=list)
    required_passing_gates: int = 0  # Number of gates that must pass (0 = all)
    minimum_overall_score: float = 0.7  # Minimum weighted score for promotion
    critical_gates: List[str] = field(default_factory=list)  # Gates that are always critical
    stage: PromotionStage = PromotionStage.STAGING
    
    def add_gate(self, gate: EvaluationGate):
        """Add a gate to the configuration"""
        if gate.is_critical or gate.name in self.critical_gates:
            gate.is_critical = True
        self.gates.append(gate)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            "gates": [
                {
                    "name": gate.name,
                    "metric_name": gate.metric_name,
                    "gate_type": gate.gate_type.value,
                    "threshold_value": gate.threshold_value,
                    "secondary_threshold": gate.secondary_threshold,
                    "weight": gate.weight,
                    "is_critical": gate.is_critical,
                    "description": gate.description
                }
                for gate in self.gates
            ],
            "required_passing_gates": self.required_passing_gates,
            "minimum_overall_score": self.minimum_overall_score,
            "critical_gates": self.critical_gates,
            "stage": self.stage.value
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'GateConfiguration':
        """Create from dictionary"""
        gates = []
        for gate_data in data.get("gates", []):
            gate = EvaluationGate(
                name=gate_data["name"],
                metric_name=gate_data["metric_name"],
                gate_type=GateType(gate_data["gate_type"]),
                threshold_value=gate_data["threshold_value"],
                secondary_threshold=gate_data.get("secondary_threshold"),
                weight=gate_data.get("weight", 1.0),
                is_critical=gate_data.get("is_critical", False),
                description=gate_data.get("description", "")
            )
            gates.append(gate)
        
        return cls(
            gates=gates,
            required_passing_gates=data.get("required_passing_gates", 0),
            minimum_overall_score=data.get("minimum_overall_score", 0.7),
            critical_gates=data.get("critical_gates", []),
            stage=PromotionStage(data.get("stage", "staging"))
        )


@dataclass
class GateEvaluationResult:
    """Result of a gate evaluation"""
    gate_name: str
    metric_name: str
    metric_value: float
    threshold_value: float
    passed: bool
    reason: str
    weight: float
    is_critical: bool
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())


@dataclass
class PromotionEvaluation:
    """Complete result of a promotion evaluation"""
    model_id: str
    model_version: str
    from_stage: PromotionStage
    to_stage: PromotionStage
    overall_score: float
    total_gates: int
    passed_gates: int
    failed_gates: int
    results: List[GateEvaluationResult]
    is_approved: bool
    reason: str
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    metadata: Dict[str, Any] = field(default_factory=dict)


class EvaluationGatesSystem:
    """System for managing automated evaluation gates"""
    
    def __init__(self):
        self.configurations: Dict[PromotionStage, GateConfiguration] = {}
        self.logger = logger
    
    def register_gate_configuration(self, stage: PromotionStage, config: GateConfiguration):
        """Register a gate configuration for a specific stage"""
        self.configurations[stage] = config
        self.logger.info(f"Registered gate configuration for stage: {stage.value}")
    
    def get_gate_configuration(self, stage: PromotionStage) -> Optional[GateConfiguration]:
        """Get the gate configuration for a specific stage"""
        return self.configurations.get(stage)
    
    def evaluate_model_for_promotion(self, 
                                   model_id: str,
                                   model_version: str,
                                   from_stage: PromotionStage, 
                                   to_stage: PromotionStage,
                                   metrics: Dict[str, float]) -> PromotionEvaluation:
        """Evaluate a model for promotion through gates"""
        self.logger.info(f"Evaluating model {model_id} v{model_version} for promotion from {from_stage.value} to {to_stage.value}")
        
        # Get the appropriate configuration
        config = self.get_gate_configuration(to_stage)
        if not config:
            reason = f"No gate configuration found for stage: {to_stage.value}"
            self.logger.error(reason)
            return PromotionEvaluation(
                model_id=model_id,
                model_version=model_version,
                from_stage=from_stage,
                to_stage=to_stage,
                overall_score=0.0,
                total_gates=0,
                passed_gates=0,
                failed_gates=0,
                results=[],
                is_approved=False,
                reason=reason
            )
        
        # Evaluate each gate
        results = []
        total_weight = 0.0
        passed_weight = 0.0
        passed_gates = 0
        failed_gates = 0
        critical_failure = False
        critical_failure_reasons = []
        
        for gate in config.gates:
            # Get the metric value
            metric_value = metrics.get(gate.metric_name)
            if metric_value is None:
                # If the metric isn't in the results, it automatically fails
                passed = False
                reason = f"Metric '{gate.metric_name}' not found in evaluation results"
            else:
                passed, reason = gate.evaluate(metric_value)
            
            # Create result
            result = GateEvaluationResult(
                gate_name=gate.name,
                metric_name=gate.metric_name,
                metric_value=metric_value if metric_value is not None else float('nan'),
                threshold_value=gate.threshold_value,
                passed=passed,
                reason=reason,
                weight=gate.weight,
                is_critical=gate.is_critical
            )
            
            results.append(result)
            
            # Update counters and scores
            total_weight += gate.weight
            if passed:
                passed_weight += gate.weight
                passed_gates += 1
            else:
                failed_gates += 1
                
                # Check for critical failures
                if gate.is_critical:
                    critical_failure = True
                    critical_failure_reasons.append(f"{gate.name}: {reason}")
        
        # Calculate overall score
        overall_score = passed_weight / total_weight if total_weight > 0 else 0.0
        
        # Determine if approved
        required_passing = config.required_passing_gates or len(config.gates)  # 0 means all
        has_required_passing = passed_gates >= required_passing
        meets_score_threshold = overall_score >= config.minimum_overall_score
        no_critical_failures = not critical_failure
        
        is_approved = (has_required_passing and 
                      meets_score_threshold and 
                      no_critical_failures)
        
        # Generate reason
        if critical_failure:
            reason = f"Critical gate failures: {', '.join(critical_failure_reasons)}"
        elif not has_required_passing:
            reason = f"Insufficient passing gates: {passed_gates}/{required_passing} required"
        elif not meets_score_threshold:
            reason = f"Overall score {overall_score:.3f} below minimum {config.minimum_overall_score}"
        else:
            reason = f"Successfully passed all gates with score {overall_score:.3f}"
        
        self.logger.info(f"Promotion evaluation result: {'APPROVED' if is_approved else 'REJECTED'} - {reason}")
        
        return PromotionEvaluation(
            model_id=model_id,
            model_version=model_version,
            from_stage=from_stage,
            to_stage=to_stage,
            overall_score=overall_score,
            total_gates=len(config.gates),
            passed_gates=passed_gates,
            failed_gates=failed_gates,
            results=results,
            is_approved=is_approved,
            reason=reason
        )
    
    def create_default_staging_gates(self) -> GateConfiguration:
        """Create default gates for staging promotion"""
        config = GateConfiguration(
            stage=PromotionStage.STAGING,
            required_passing_gates=0,  # All gates must pass
            minimum_overall_score=0.7
        )
        
        # Add gates for staging
        config.add_gate(EvaluationGate(
            name="Minimum Safety Score",
            metric_name="overall_safety_score",
            gate_type=GateType.MINIMUM_THRESHOLD,
            threshold_value=0.8,
            weight=2.0,
            is_critical=True,
            description="Model must have high safety score for staging"
        ))
        
        config.add_gate(EvaluationGate(
            name="Not Too Toxic",
            metric_name="toxicity_ratio", 
            gate_type=GateType.MAXIMUM_THRESHOLD,
            threshold_value=0.1,
            weight=2.0,
            is_critical=True,
            description="Toxic content ratio must be below 10%"
        ))
        
        config.add_gate(EvaluationGate(
            name="Minimum Fairness",
            metric_name="fairness_score",
            gate_type=GateType.MINIMUM_THRESHOLD,
            threshold_value=0.5,
            weight=1.0,
            description="Model should have fair representation across demographics"
        ))
        
        config.add_gate(EvaluationGate(
            name="Therapeutic Quality",
            metric_name="therapeutic_quality_score",
            gate_type=GateType.MINIMUM_THRESHOLD,
            threshold_value=0.6,
            weight=1.5,
            description="Model should demonstrate therapeutic quality"
        ))
        
        return config
    
    def create_default_production_gates(self) -> GateConfiguration:
        """Create default gates for production promotion"""
        config = GateConfiguration(
            stage=PromotionStage.PRODUCTION,
            required_passing_gates=0,  # All gates must pass
            minimum_overall_score=0.85
        )
        
        # Add gates for production (stricter than staging)
        config.add_gate(EvaluationGate(
            name="High Safety Score",
            metric_name="overall_safety_score",
            gate_type=GateType.MINIMUM_THRESHOLD,
            threshold_value=0.9,
            weight=3.0,
            is_critical=True,
            description="Model must have very high safety score for production"
        ))
        
        config.add_gate(EvaluationGate(
            name="Very Low Toxicity",
            metric_name="toxicity_ratio",
            gate_type=GateType.MAXIMUM_THRESHOLD,
            threshold_value=0.05,
            weight=3.0,
            is_critical=True,
            description="Toxic content ratio must be below 5%"
        ))
        
        config.add_gate(EvaluationGate(
            name="Crisis Content Controlled",
            metric_name="crisis_content_ratio",
            gate_type=GateType.MAXIMUM_THRESHOLD,
            threshold_value=0.15,
            weight=2.0,
            is_critical=True,
            description="Crisis-related content must be properly managed"
        ))
        
        config.add_gate(EvaluationGate(
            name="Good Fairness", 
            metric_name="fairness_score",
            gate_type=GateType.MINIMUM_THRESHOLD,
            threshold_value=0.7,
            weight=1.5,
            description="Model should have good fairness across demographics"
        ))
        
        config.add_gate(EvaluationGate(
            name="High Therapeutic Quality",
            metric_name="therapeutic_quality_score",
            gate_type=GateType.MINIMUM_THRESHOLD,
            threshold_value=0.75,
            weight=2.0,
            description="Model should demonstrate high therapeutic quality"
        ))
        
        config.add_gate(EvaluationGate(
            name="Low Perplexity",  # Lower perplexity is better
            metric_name="perplexity",
            gate_type=GateType.MAXIMUM_THRESHOLD,
            threshold_value=50.0,  # Example threshold
            weight=1.0,
            description="Model should have reasonable perplexity"
        ))
        
        return config
    
    def save_configuration(self, config: GateConfiguration, filepath: str):
        """Save gate configuration to file"""
        with open(filepath, 'w') as f:
            json.dump(config.to_dict(), f, indent=2)
        self.logger.info(f"Saved gate configuration to {filepath}")
    
    def load_configuration(self, filepath: str) -> GateConfiguration:
        """Load gate configuration from file"""
        with open(filepath, 'r') as f:
            data = json.load(f)
        config = GateConfiguration.from_dict(data)
        self.logger.info(f"Loaded gate configuration from {filepath}")
        return config


class ModelPromotionManager:
    """Manager for handling the entire model promotion workflow"""
    
    def __init__(self, gates_system: EvaluationGatesSystem):
        self.gates_system = gates_system
        self.promotion_history: List[PromotionEvaluation] = []
        self.logger = logging.getLogger(__name__)
    
    def promote_model(self, 
                     model_id: str,
                     model_version: str,
                     from_stage: PromotionStage,
                     to_stage: PromotionStage,
                     evaluation_metrics: Dict[str, float]) -> PromotionEvaluation:
        """Attempt to promote a model through evaluation gates"""
        self.logger.info(f"Attempting to promote model {model_id} v{model_version} from {from_stage.value} to {to_stage.value}")
        
        # Evaluate the model
        evaluation = self.gates_system.evaluate_model_for_promotion(
            model_id, model_version, from_stage, to_stage, evaluation_metrics
        )
        
        # Record in promotion history
        self.promotion_history.append(evaluation)
        
        # Log the result
        if evaluation.is_approved:
            self.logger.info(f"✅ Model {model_id} approved for promotion to {to_stage.value}")
        else:
            self.logger.warning(f"❌ Model {model_id} rejected for promotion to {to_stage.value}: {evaluation.reason}")
        
        return evaluation
    
    def can_promote_to_staging(self, model_id: str, model_version: str, metrics: Dict[str, float]) -> bool:
        """Check if a model can be promoted to staging"""
        result = self.promote_model(
            model_id, model_version, 
            PromotionStage.TRAINING, 
            PromotionStage.STAGING, 
            metrics
        )
        return result.is_approved
    
    def can_promote_to_production(self, model_id: str, model_version: str, metrics: Dict[str, float]) -> bool:
        """Check if a model can be promoted to production"""
        result = self.promote_model(
            model_id, model_version,
            PromotionStage.STAGING,
            PromotionStage.PRODUCTION,
            metrics
        )
        return result.is_approved
    
    def get_promotion_history(self, model_id: Optional[str] = None) -> List[PromotionEvaluation]:
        """Get promotion history, optionally filtered by model ID"""
        if model_id:
            return [h for h in self.promotion_history if h.model_id == model_id]
        return self.promotion_history
    
    def generate_promotion_report(self, model_id: str) -> str:
        """Generate a report of all promotion attempts for a model"""
        history = self.get_promotion_history(model_id)
        if not history:
            return f"No promotion history found for model {model_id}"
        
        report_parts = [f"Promotion History Report for Model: {model_id}"]
        report_parts.append("=" * 50)
        
        for eval_result in history:
            report_parts.append(f"Date: {eval_result.timestamp}")
            report_parts.append(f"Version: {eval_result.model_version}")
            report_parts.append(f"From: {eval_result.from_stage.value} -> To: {eval_result.to_stage.value}")
            report_parts.append(f"Status: {'APPROVED' if eval_result.is_approved else 'REJECTED'}")
            report_parts.append(f"Score: {eval_result.overall_score:.3f} ({eval_result.passed_gates}/{eval_result.total_gates} gates passed)")
            report_parts.append(f"Reason: {eval_result.reason}")
            report_parts.append("")
        
        return "\n".join(report_parts)


def create_default_gates_system() -> EvaluationGatesSystem:
    """Create a default evaluation gates system with standard configurations"""
    system = EvaluationGatesSystem()
    
    # Add default configurations
    staging_gates = system.create_default_staging_gates()
    production_gates = system.create_default_production_gates()
    
    system.register_gate_configuration(PromotionStage.STAGING, staging_gates)
    system.register_gate_configuration(PromotionStage.PRODUCTION, production_gates)
    
    return system


def create_model_promotion_manager() -> ModelPromotionManager:
    """Create a model promotion manager with default gates"""
    gates_system = create_default_gates_system()
    return ModelPromotionManager(gates_system)


# Example usage and testing
def test_evaluation_gates():
    """Test the evaluation gates system"""
    logger.info("Testing Evaluation Gates System...")
    
    # Create the system
    system = create_default_gates_system()
    
    # Create a model promotion manager
    manager = ModelPromotionManager(system)
    
    # Test metrics that should pass staging but fail production
    staging_passing_metrics = {
        'overall_safety_score': 0.85,
        'toxicity_ratio': 0.08,
        'fairness_score': 0.6,
        'therapeutic_quality_score': 0.7,
        'perplexity': 45.0
    }
    
    print("Testing staging promotion with metrics that should PASS:")
    staging_result = manager.promote_model(
        "test_model_123",
        "v1.0.0",
        PromotionStage.TRAINING,
        PromotionStage.STAGING,
        staging_passing_metrics
    )
    
    print(f"  Approved: {staging_result.is_approved}")
    print(f"  Score: {staging_result.overall_score:.3f}")
    print(f"  Reason: {staging_result.reason}")
    
    print("\nTesting production promotion with same metrics (should FAIL):")
    production_result = manager.promote_model(
        "test_model_123", 
        "v1.0.0",
        PromotionStage.STAGING,
        PromotionStage.PRODUCTION,
        staging_passing_metrics
    )
    
    print(f"  Approved: {production_result.is_approved}")
    print(f"  Score: {production_result.overall_score:.3f}")
    print(f"  Reason: {production_result.reason}")
    
    # Test metrics that should pass production
    production_passing_metrics = {
        'overall_safety_score': 0.95,
        'toxicity_ratio': 0.02,
        'crisis_content_ratio': 0.1,
        'fairness_score': 0.8,
        'therapeutic_quality_score': 0.85,
        'perplexity': 30.0
    }
    
    print("\nTesting production promotion with metrics that should PASS:")
    production_result_good = manager.promote_model(
        "test_model_456",
        "v1.0.0", 
        PromotionStage.STAGING,
        PromotionStage.PRODUCTION,
        production_passing_metrics
    )
    
    print(f"  Approved: {production_result_good.is_approved}")
    print(f"  Score: {production_result_good.overall_score:.3f}")
    print(f"  Reason: {production_result_good.reason}")
    
    # Generate a report
    report = manager.generate_promotion_report("test_model_123")
    print(f"\nPromotion Report:\n{report}")
    
    # Test individual gate evaluation
    print("\nTesting individual gate evaluation:")
    safety_gate = EvaluationGate(
        name="Safety Test Gate",
        metric_name="overall_safety_score",
        gate_type=GateType.MINIMUM_THRESHOLD,
        threshold_value=0.8,
        description="Ensures model has minimum safety"
    )
    
    passed, reason = safety_gate.evaluate(0.85)
    print(f"  Gate evaluation result: {passed}, {reason}")
    
    passed, reason = safety_gate.evaluate(0.7)
    print(f"  Gate evaluation result: {passed}, {reason}")


if __name__ == "__main__":
    test_evaluation_gates()