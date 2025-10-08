"""
Deployment management system for A/B testing and canary deployments.
Enables safe rollout of new model versions with traffic splitting and metrics collection.
"""

import json
import time
import random
import hashlib
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum
import logging
from pathlib import Path
import numpy as np


logger = logging.getLogger(__name__)


class DeploymentStrategy(Enum):
    """Types of deployment strategies"""
    DIRECT = "direct"          # Immediate full rollout
    CANARY = "canary"           # Gradual rollout to increasing percentages
    AB_TEST = "ab_test"         # A/B testing between versions
    SHADOW = "shadow"           # Shadow traffic to new version
    ROLLOUT = "rollout"         # Scheduled gradual rollout


class DeploymentStatus(Enum):
    """Status of a deployment"""
    PENDING = "pending"
    ACTIVE = "active"
    PAUSED = "paused"
    FAILED = "failed"
    COMPLETED = "completed"
    ROLLED_BACK = "rolled_back"


@dataclass
class DeploymentConfig:
    """Configuration for a deployment"""
    strategy: DeploymentStrategy
    traffic_percentage: float = 0.0  # Current traffic percentage (0-100)
    target_percentage: float = 100.0  # Target traffic percentage
    ramp_up_interval_minutes: int = 30  # Time between traffic increases for canary
    ramp_up_increment: float = 10.0  # Percentage increase per interval for canary
    ab_test_ratio: float = 50.0  # For A/B testing, percentage to new version (50/50 split)
    shadow_sampling_rate: float = 0.1  # For shadow deployments, sample rate (0-1)
    evaluation_metrics: List[str] = field(default_factory=lambda: ["latency", "error_rate", "user_satisfaction"])
    success_criteria: Dict[str, Any] = field(default_factory=dict)  # Metric thresholds for success
    rollback_criteria: Dict[str, Any] = field(default_factory=dict)  # Metric thresholds for rollback
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class DeploymentVersion:
    """Represents a specific version of a model/service"""
    version_id: str
    model_name: str
    model_path: str
    model_type: str  # pytorch, tensorflow, onnx, llm
    created_at: str
    hash_checksum: Optional[str] = None
    size_bytes: Optional[int] = None
    performance_baseline: Optional[Dict[str, float]] = None  # Baseline metrics
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class Deployment:
    """Represents a deployment operation"""
    deployment_id: str
    model_name: str
    strategy: DeploymentStrategy
    current_version: DeploymentVersion
    new_version: Optional[DeploymentVersion] = None
    config: DeploymentConfig = field(default_factory=DeploymentConfig)
    status: DeploymentStatus = DeploymentStatus.PENDING
    created_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    started_at: Optional[str] = None
    completed_at: Optional[str] = None
    last_updated: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    traffic_shift_history: List[Dict[str, Any]] = field(default_factory=list)
    metrics_history: List[Dict[str, Any]] = field(default_factory=list)
    alerts: List[Dict[str, Any]] = field(default_factory=list)
    rollback_info: Optional[Dict[str, Any]] = None
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class TrafficRoutingRule:
    """Rule for routing traffic to specific versions"""
    rule_id: str
    model_name: str
    condition: str  # Condition for applying this rule (e.g., "user_id_hash < 0.1")
    target_version: str  # Version ID to route to
    priority: int = 100  # Lower number = higher priority
    enabled: bool = True
    created_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    expires_at: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class ExperimentResult:
    """Results from an A/B test or canary deployment"""
    experiment_id: str
    model_name: str
    control_version: str
    treatment_version: str
    start_time: str
    end_time: Optional[str] = None
    metrics: Dict[str, Dict[str, float]] = field(default_factory=dict)  # version -> metric -> value
    statistical_significance: Optional[Dict[str, float]] = None  # p-values for metrics
    winner: Optional[str] = None  # Winning version based on primary metric
    confidence_level: Optional[float] = None
    sample_size: Optional[int] = None
    metadata: Optional[Dict[str, Any]] = None


class DeploymentManager:
    """Manages model deployments with A/B testing and canary capabilities"""
    
    def __init__(self, config_path: Optional[str] = None):
        self.deployments: Dict[str, Deployment] = {}
        self.traffic_rules: Dict[str, List[TrafficRoutingRule]] = {}
        self.experiment_results: Dict[str, ExperimentResult] = {}
        self.config_path = config_path
        self.logger = logging.getLogger(__name__)
        self._load_config()
    
    def _load_config(self):
        """Load deployment configuration from file"""
        if self.config_path and Path(self.config_path).exists():
            try:
                with open(self.config_path, 'r') as f:
                    config = json.load(f)
                # Restore deployments from config
                for dep_id, dep_data in config.get('deployments', {}).items():
                    self.deployments[dep_id] = self._dict_to_deployment(dep_data)
                for model_name, rules_data in config.get('traffic_rules', {}).items():
                    self.traffic_rules[model_name] = [self._dict_to_traffic_rule(rule_data) for rule_data in rules_data]
                self.logger.info(f"Loaded {len(self.deployments)} deployments from config")
            except Exception as e:
                self.logger.error(f"Failed to load deployment config: {e}")
    
    def _save_config(self):
        """Save deployment configuration to file"""
        if self.config_path:
            try:
                config_data = {
                    'deployments': {dep_id: self._deployment_to_dict(dep) for dep_id, dep in self.deployments.items()},
                    'traffic_rules': {model_name: [self._traffic_rule_to_dict(rule) for rule in rules] 
                                    for model_name, rules in self.traffic_rules.items()},
                    'experiment_results': {exp_id: self._experiment_result_to_dict(exp) 
                                          for exp_id, exp in self.experiment_results.items()}
                }
                with open(self.config_path, 'w') as f:
                    json.dump(config_data, f, indent=2)
            except Exception as e:
                self.logger.error(f"Failed to save deployment config: {e}")
    
    def _dict_to_deployment(self, data: Dict[str, Any]) -> Deployment:
        """Convert dictionary to Deployment object"""
        # This is a simplified implementation - in practice, you'd need proper deserialization
        config_data = data.get('config', {})
        config = DeploymentConfig(
            strategy=DeploymentStrategy(config_data.get('strategy', 'direct')),
            traffic_percentage=config_data.get('traffic_percentage', 0.0),
            target_percentage=config_data.get('target_percentage', 100.0),
            ramp_up_interval_minutes=config_data.get('ramp_up_interval_minutes', 30),
            ramp_up_increment=config_data.get('ramp_up_increment', 10.0),
            ab_test_ratio=config_data.get('ab_test_ratio', 50.0),
            shadow_sampling_rate=config_data.get('shadow_sampling_rate', 0.1),
            evaluation_metrics=config_data.get('evaluation_metrics', ["latency", "error_rate"]),
            success_criteria=config_data.get('success_criteria', {}),
            rollback_criteria=config_data.get('rollback_criteria', {}),
            metadata=config_data.get('metadata')
        )
        
        current_version_data = data.get('current_version', {})
        current_version = DeploymentVersion(
            version_id=current_version_data.get('version_id', ''),
            model_name=current_version_data.get('model_name', ''),
            model_path=current_version_data.get('model_path', ''),
            model_type=current_version_data.get('model_type', ''),
            created_at=current_version_data.get('created_at', ''),
            hash_checksum=current_version_data.get('hash_checksum'),
            size_bytes=current_version_data.get('size_bytes'),
            performance_baseline=current_version_data.get('performance_baseline'),
            metadata=current_version_data.get('metadata')
        )
        
        new_version_data = data.get('new_version')
        new_version = None
        if new_version_data:
            new_version = DeploymentVersion(
                version_id=new_version_data.get('version_id', ''),
                model_name=new_version_data.get('model_name', ''),
                model_path=new_version_data.get('model_path', ''),
                model_type=new_version_data.get('model_type', ''),
                created_at=new_version_data.get('created_at', ''),
                hash_checksum=new_version_data.get('hash_checksum'),
                size_bytes=new_version_data.get('size_bytes'),
                performance_baseline=new_version_data.get('performance_baseline'),
                metadata=new_version_data.get('metadata')
            )
        
        return Deployment(
            deployment_id=data.get('deployment_id', ''),
            model_name=data.get('model_name', ''),
            strategy=DeploymentStrategy(data.get('strategy', 'direct')),
            current_version=current_version,
            new_version=new_version,
            config=config,
            status=DeploymentStatus(data.get('status', 'pending')),
            created_at=data.get('created_at', ''),
            started_at=data.get('started_at'),
            completed_at=data.get('completed_at'),
            last_updated=data.get('last_updated', ''),
            traffic_shift_history=data.get('traffic_shift_history', []),
            metrics_history=data.get('metrics_history', []),
            alerts=data.get('alerts', []),
            rollback_info=data.get('rollback_info'),
            metadata=data.get('metadata')
        )
    
    def _deployment_to_dict(self, deployment: Deployment) -> Dict[str, Any]:
        """Convert Deployment object to dictionary"""
        return {
            'deployment_id': deployment.deployment_id,
            'model_name': deployment.model_name,
            'strategy': deployment.strategy.value,
            'current_version': {
                'version_id': deployment.current_version.version_id,
                'model_name': deployment.current_version.model_name,
                'model_path': deployment.current_version.model_path,
                'model_type': deployment.current_version.model_type,
                'created_at': deployment.current_version.created_at,
                'hash_checksum': deployment.current_version.hash_checksum,
                'size_bytes': deployment.current_version.size_bytes,
                'performance_baseline': deployment.current_version.performance_baseline,
                'metadata': deployment.current_version.metadata
            },
            'new_version': {
                'version_id': deployment.new_version.version_id if deployment.new_version else None,
                'model_name': deployment.new_version.model_name if deployment.new_version else None,
                'model_path': deployment.new_version.model_path if deployment.new_version else None,
                'model_type': deployment.new_version.model_type if deployment.new_version else None,
                'created_at': deployment.new_version.created_at if deployment.new_version else None,
                'hash_checksum': deployment.new_version.hash_checksum if deployment.new_version else None,
                'size_bytes': deployment.new_version.size_bytes if deployment.new_version else None,
                'performance_baseline': deployment.new_version.performance_baseline if deployment.new_version else None,
                'metadata': deployment.new_version.metadata if deployment.new_version else None
            } if deployment.new_version else None,
            'config': {
                'strategy': deployment.config.strategy.value,
                'traffic_percentage': deployment.config.traffic_percentage,
                'target_percentage': deployment.config.target_percentage,
                'ramp_up_interval_minutes': deployment.config.ramp_up_interval_minutes,
                'ramp_up_increment': deployment.config.ramp_up_increment,
                'ab_test_ratio': deployment.config.ab_test_ratio,
                'shadow_sampling_rate': deployment.config.shadow_sampling_rate,
                'evaluation_metrics': deployment.config.evaluation_metrics,
                'success_criteria': deployment.config.success_criteria,
                'rollback_criteria': deployment.config.rollback_criteria,
                'metadata': deployment.config.metadata
            },
            'status': deployment.status.value,
            'created_at': deployment.created_at,
            'started_at': deployment.started_at,
            'completed_at': deployment.completed_at,
            'last_updated': deployment.last_updated,
            'traffic_shift_history': deployment.traffic_shift_history,
            'metrics_history': deployment.metrics_history,
            'alerts': deployment.alerts,
            'rollback_info': deployment.rollback_info,
            'metadata': deployment.metadata
        }
    
    def _dict_to_traffic_rule(self, data: Dict[str, Any]) -> TrafficRoutingRule:
        """Convert dictionary to TrafficRoutingRule"""
        return TrafficRoutingRule(
            rule_id=data.get('rule_id', ''),
            model_name=data.get('model_name', ''),
            condition=data.get('condition', ''),
            target_version=data.get('target_version', ''),
            priority=data.get('priority', 100),
            enabled=data.get('enabled', True),
            created_at=data.get('created_at', ''),
            expires_at=data.get('expires_at'),
            metadata=data.get('metadata')
        )
    
    def _traffic_rule_to_dict(self, rule: TrafficRoutingRule) -> Dict[str, Any]:
        """Convert TrafficRoutingRule to dictionary"""
        return {
            'rule_id': rule.rule_id,
            'model_name': rule.model_name,
            'condition': rule.condition,
            'target_version': rule.target_version,
            'priority': rule.priority,
            'enabled': rule.enabled,
            'created_at': rule.created_at,
            'expires_at': rule.expires_at,
            'metadata': rule.metadata
        }
    
    def _experiment_result_to_dict(self, result: ExperimentResult) -> Dict[str, Any]:
        """Convert ExperimentResult to dictionary"""
        return {
            'experiment_id': result.experiment_id,
            'model_name': result.model_name,
            'control_version': result.control_version,
            'treatment_version': result.treatment_version,
            'start_time': result.start_time,
            'end_time': result.end_time,
            'metrics': result.metrics,
            'statistical_significance': result.statistical_significance,
            'winner': result.winner,
            'confidence_level': result.confidence_level,
            'sample_size': result.sample_size,
            'metadata': result.metadata
        }
    
    def create_deployment(self,
                         model_name: str,
                         new_version: DeploymentVersion,
                         strategy: DeploymentStrategy,
                         config: Optional[DeploymentConfig] = None) -> str:
        """Create a new deployment"""
        deployment_id = f"deploy_{model_name}_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}_{hashlib.md5(model_name.encode()).hexdigest()[:8]}"
        
        if not config:
            config = DeploymentConfig(strategy=strategy)
        
        # Get current version (this would typically come from the model registry)
        current_version = DeploymentVersion(
            version_id="current",
            model_name=model_name,
            model_path="",  # Would be populated from registry
            model_type="pytorch",  # Would be populated from registry
            created_at=datetime.utcnow().isoformat()
        )
        
        deployment = Deployment(
            deployment_id=deployment_id,
            model_name=model_name,
            strategy=strategy,
            current_version=current_version,
            new_version=new_version,
            config=config,
            status=DeploymentStatus.PENDING
        )
        
        self.deployments[deployment_id] = deployment
        self._save_config()
        
        self.logger.info(f"Created deployment {deployment_id} for model {model_name} using strategy {strategy.value}")
        return deployment_id
    
    def start_deployment(self, deployment_id: str) -> bool:
        """Start a deployment"""
        if deployment_id not in self.deployments:
            self.logger.error(f"Deployment {deployment_id} not found")
            return False
        
        deployment = self.deployments[deployment_id]
        if deployment.status != DeploymentStatus.PENDING:
            self.logger.error(f"Deployment {deployment_id} is not in PENDING status")
            return False
        
        deployment.status = DeploymentStatus.ACTIVE
        deployment.started_at = datetime.utcnow().isoformat()
        deployment.last_updated = datetime.utcnow().isoformat()
        
        # Set initial traffic percentage
        if deployment.strategy == DeploymentStrategy.AB_TEST:
            deployment.config.traffic_percentage = deployment.config.ab_test_ratio
        elif deployment.strategy == DeploymentStrategy.CANARY:
            deployment.config.traffic_percentage = deployment.config.ramp_up_increment
        elif deployment.strategy == DeploymentStrategy.SHADOW:
            deployment.config.traffic_percentage = 100.0  # Shadow gets 100% traffic
        else:  # DIRECT
            deployment.config.traffic_percentage = 100.0
        
        self._setup_traffic_routing(deployment)
        self._save_config()
        
        self.logger.info(f"Started deployment {deployment_id} for model {deployment.model_name}")
        return True
    
    def _setup_traffic_routing(self, deployment: Deployment):
        """Set up traffic routing rules for the deployment"""
        model_name = deployment.model_name
        
        # Clear existing rules for this model
        self.traffic_rules[model_name] = []
        
        # Add routing rules based on deployment strategy
        if deployment.strategy == DeploymentStrategy.AB_TEST:
            # A/B test routing - split traffic based on user hash
            control_rule = TrafficRoutingRule(
                rule_id=f"ab_control_{deployment.deployment_id}",
                model_name=model_name,
                condition=f"user_id_hash < {(100 - deployment.config.ab_test_ratio)/100}",
                target_version=deployment.current_version.version_id,
                priority=1
            )
            treatment_rule = TrafficRoutingRule(
                rule_id=f"ab_treatment_{deployment.deployment_id}",
                model_name=model_name,
                condition=f"user_id_hash >= {(100 - deployment.config.ab_test_ratio)/100}",
                target_version=deployment.new_version.version_id if deployment.new_version else deployment.current_version.version_id,
                priority=2
            )
            self.traffic_rules[model_name].extend([control_rule, treatment_rule])
            
        elif deployment.strategy == DeploymentStrategy.CANARY:
            # Canary routing - gradually shift traffic to new version
            canary_rule = TrafficRoutingRule(
                rule_id=f"canary_{deployment.deployment_id}",
                model_name=model_name,
                condition=f"user_id_hash < {deployment.config.traffic_percentage/100}",
                target_version=deployment.new_version.version_id if deployment.new_version else deployment.current_version.version_id,
                priority=1
            )
            baseline_rule = TrafficRoutingRule(
                rule_id=f"baseline_{deployment.deployment_id}",
                model_name=model_name,
                condition="true",  # Catch-all
                target_version=deployment.current_version.version_id,
                priority=100
            )
            self.traffic_rules[model_name].extend([canary_rule, baseline_rule])
            
        elif deployment.strategy == DeploymentStrategy.SHADOW:
            # Shadow deployment - route to current version but also shadow to new
            main_rule = TrafficRoutingRule(
                rule_id=f"shadow_main_{deployment.deployment_id}",
                model_name=model_name,
                condition="true",
                target_version=deployment.current_version.version_id,
                priority=1
            )
            shadow_rule = TrafficRoutingRule(
                rule_id=f"shadow_secondary_{deployment.deployment_id}",
                model_name=model_name,
                condition=f"random_sampling < {deployment.config.shadow_sampling_rate}",
                target_version=deployment.new_version.version_id if deployment.new_version else deployment.current_version.version_id,
                priority=2
            )
            self.traffic_rules[model_name].extend([main_rule, shadow_rule])
            
        else:  # DIRECT or other strategies
            # Route all traffic to new version
            direct_rule = TrafficRoutingRule(
                rule_id=f"direct_{deployment.deployment_id}",
                model_name=model_name,
                condition="true",
                target_version=deployment.new_version.version_id if deployment.new_version else deployment.current_version.version_id,
                priority=1
            )
            self.traffic_rules[model_name].append(direct_rule)
    
    def advance_canary_deployment(self, deployment_id: str) -> bool:
        """Advance a canary deployment by one increment"""
        if deployment_id not in self.deployments:
            self.logger.error(f"Deployment {deployment_id} not found")
            return False
        
        deployment = self.deployments[deployment_id]
        if deployment.strategy != DeploymentStrategy.CANARY:
            self.logger.error(f"Deployment {deployment_id} is not a canary deployment")
            return False
        
        if deployment.status != DeploymentStatus.ACTIVE:
            self.logger.error(f"Deployment {deployment_id} is not active")
            return False
        
        # Check if we've reached target percentage
        if deployment.config.traffic_percentage >= deployment.config.target_percentage:
            self.logger.info(f"Deployment {deployment_id} has reached target percentage")
            return True
        
        # Advance by increment
        old_percentage = deployment.config.traffic_percentage
        deployment.config.traffic_percentage = min(
            deployment.config.traffic_percentage + deployment.config.ramp_up_increment,
            deployment.config.target_percentage
        )
        
        # Record the traffic shift
        deployment.traffic_shift_history.append({
            "timestamp": datetime.utcnow().isoformat(),
            "from_percentage": old_percentage,
            "to_percentage": deployment.config.traffic_percentage,
            "reason": "scheduled_advance"
        })
        
        # Update routing rules
        self._setup_traffic_routing(deployment)
        deployment.last_updated = datetime.utcnow().isoformat()
        
        # Log the advancement
        self.logger.info(f"Advanced canary deployment {deployment_id} from {old_percentage}% to {deployment.config.traffic_percentage}%")
        
        # Check if we've completed
        if deployment.config.traffic_percentage >= deployment.config.target_percentage:
            deployment.status = DeploymentStatus.COMPLETED
            deployment.completed_at = datetime.utcnow().isoformat()
            self.logger.info(f"Canary deployment {deployment_id} completed")
        
        self._save_config()
        return True
    
    def evaluate_deployment(self, deployment_id: str, metrics: Dict[str, float]) -> Dict[str, Any]:
        """Evaluate deployment based on metrics"""
        if deployment_id not in self.deployments:
            self.logger.error(f"Deployment {deployment_id} not found")
            return {"status": "error", "message": "Deployment not found"}
        
        deployment = self.deployments[deployment_id]
        
        # Record metrics
        metrics_record = {
            "timestamp": datetime.utcnow().isoformat(),
            "metrics": metrics,
            "traffic_percentage": deployment.config.traffic_percentage
        }
        deployment.metrics_history.append(metrics_record)
        deployment.last_updated = datetime.utcnow().isoformat()
        
        # Check success criteria
        success_violations = []
        rollback_triggers = []
        
        # Check success criteria
        for metric_name, threshold in deployment.config.success_criteria.items():
            if metric_name in metrics:
                actual_value = metrics[metric_name]
                if isinstance(threshold, dict):
                    # Handle range thresholds
                    min_val = threshold.get("min", float("-inf"))
                    max_val = threshold.get("max", float("inf"))
                    if not (min_val <= actual_value <= max_val):
                        success_violations.append(f"{metric_name}: {actual_value} not in range [{min_val}, {max_val}]")
                else:
                    # Handle simple threshold (assuming lower is better for error rates, higher is better for others)
                    if "error" in metric_name.lower() or "latency" in metric_name.lower():
                        if actual_value > threshold:
                            success_violations.append(f"{metric_name}: {actual_value} > {threshold}")
                    else:
                        if actual_value < threshold:
                            success_violations.append(f"{metric_name}: {actual_value} < {threshold}")
        
        # Check rollback criteria
        for metric_name, threshold in deployment.config.rollback_criteria.items():
            if metric_name in metrics:
                actual_value = metrics[metric_name]
                if isinstance(threshold, dict):
                    # Handle range thresholds
                    min_val = threshold.get("min", float("-inf"))
                    max_val = threshold.get("max", float("inf"))
                    if not (min_val <= actual_value <= max_val):
                        rollback_triggers.append(f"{metric_name}: {actual_value} not in range [{min_val}, {max_val}]")
                else:
                    # Handle simple threshold
                    if "error" in metric_name.lower() or "latency" in metric_name.lower():
                        if actual_value > threshold:
                            rollback_triggers.append(f"{metric_name}: {actual_value} > {threshold}")
                    else:
                        if actual_value < threshold:
                            rollback_triggers.append(f"{metric_name}: {actual_value} < {threshold}")
        
        evaluation_result = {
            "deployment_id": deployment_id,
            "timestamp": datetime.utcnow().isoformat(),
            "metrics": metrics,
            "success_violations": success_violations,
            "rollback_triggers": rollback_triggers,
            "traffic_percentage": deployment.config.traffic_percentage,
            "should_rollback": len(rollback_triggers) > 0,
            "should_pause": len(success_violations) > 0 and len(rollback_triggers) == 0
        }
        
        # Trigger alerts if needed
        if rollback_triggers:
            self._trigger_alert(deployment, "rollback_triggered", rollback_triggers)
        elif success_violations:
            self._trigger_alert(deployment, "success_violation", success_violations)
        
        self._save_config()
        return evaluation_result
    
    def _trigger_alert(self, deployment: Deployment, alert_type: str, details: List[str]):
        """Trigger an alert for a deployment"""
        alert = {
            "timestamp": datetime.utcnow().isoformat(),
            "type": alert_type,
            "details": details,
            "resolved": False
        }
        deployment.alerts.append(alert)
        self.logger.warning(f"Alert triggered for deployment {deployment.deployment_id}: {alert_type} - {details}")
    
    def rollback_deployment(self, deployment_id: str) -> bool:
        """Rollback a deployment to the previous version"""
        if deployment_id not in self.deployments:
            self.logger.error(f"Deployment {deployment_id} not found")
            return False
        
        deployment = self.deployments[deployment_id]
        if deployment.status in [DeploymentStatus.ROLLED_BACK, DeploymentStatus.FAILED]:
            self.logger.error(f"Deployment {deployment_id} cannot be rolled back (already {deployment.status.value})")
            return False
        
        old_status = deployment.status
        deployment.status = DeploymentStatus.ROLLED_BACK
        deployment.completed_at = datetime.utcnow().isoformat()
        deployment.last_updated = datetime.utcnow().isoformat()
        deployment.rollback_info = {
            "rollback_timestamp": datetime.utcnow().isoformat(),
            "previous_status": old_status.value,
            "reason": "manual_rollback" if deployment.status != DeploymentStatus.ACTIVE else "automatic_rollback"
        }
        
        # Update routing to point all traffic back to current version
        deployment.config.traffic_percentage = 100.0
        self._setup_traffic_routing(deployment)
        
        self.logger.info(f"Rolled back deployment {deployment_id} from {old_status.value} to rolled_back")
        self._save_config()
        return True
    
    def pause_deployment(self, deployment_id: str) -> bool:
        """Pause a deployment"""
        if deployment_id not in self.deployments:
            self.logger.error(f"Deployment {deployment_id} not found")
            return False
        
        deployment = self.deployments[deployment_id]
        if deployment.status != DeploymentStatus.ACTIVE:
            self.logger.error(f"Deployment {deployment_id} is not active")
            return False
        
        deployment.status = DeploymentStatus.PAUSED
        deployment.last_updated = datetime.utcnow().isoformat()
        
        self.logger.info(f"Paused deployment {deployment_id}")
        self._save_config()
        return True
    
    def resume_deployment(self, deployment_id: str) -> bool:
        """Resume a paused deployment"""
        if deployment_id not in self.deployments:
            self.logger.error(f"Deployment {deployment_id} not found")
            return False
        
        deployment = self.deployments[deployment_id]
        if deployment.status != DeploymentStatus.PAUSED:
            self.logger.error(f"Deployment {deployment_id} is not paused")
            return False
        
        deployment.status = DeploymentStatus.ACTIVE
        deployment.last_updated = datetime.utcnow().isoformat()
        
        self.logger.info(f"Resumed deployment {deployment_id}")
        self._save_config()
        return True
    
    def get_route_version(self, model_name: str, context: Dict[str, Any]) -> str:
        """Get the version to route a request to based on current rules"""
        if model_name not in self.traffic_rules:
            # No specific rules, return current version
            return "current"
        
        rules = self.traffic_rules[model_name]
        
        # Sort rules by priority
        rules.sort(key=lambda r: r.priority)
        
        # Evaluate rules in order
        for rule in rules:
            if not rule.enabled:
                continue
            
            # Check expiration
            if rule.expires_at:
                expire_time = datetime.fromisoformat(rule.expires_at.replace('Z', '+00:00'))
                if datetime.utcnow() > expire_time:
                    continue
            
            # Evaluate condition (simplified evaluation)
            if self._evaluate_condition(rule.condition, context):
                return rule.target_version
        
        # No matching rule, return current version
        return "current"
    
    def _evaluate_condition(self, condition: str, context: Dict[str, Any]) -> bool:
        """Evaluate a routing condition against request context"""
        # This is a simplified condition evaluator
        # In practice, you'd want a more robust expression evaluator
        
        try:
            # Handle common conditions
            if condition == "true":
                return True
            elif condition == "false":
                return False
            
            # Handle user_id_hash conditions
            if "user_id_hash" in condition:
                user_id = context.get("user_id", "")
                if user_id:
                    # Simple hash-based routing
                    user_hash = int(hashlib.md5(user_id.encode()).hexdigest(), 16) % 1000000
                    user_hash_normalized = user_hash / 1000000.0
                    
                    # Parse comparison (e.g., "user_id_hash < 0.1")
                    if "<" in condition:
                        _, threshold_str = condition.split("<")
                        threshold = float(threshold_str.strip())
                        return user_hash_normalized < threshold
                    elif ">" in condition:
                        _, threshold_str = condition.split(">")
                        threshold = float(threshold_str.strip())
                        return user_hash_normalized > threshold
            
            # Handle random sampling conditions
            if "random_sampling" in condition:
                if "<" in condition:
                    _, threshold_str = condition.split("<")
                    threshold = float(threshold_str.strip())
                    return random.random() < threshold
            
            # Default to false for unrecognized conditions
            return False
            
        except Exception as e:
            self.logger.error(f"Error evaluating condition '{condition}': {e}")
            return False
    
    def complete_experiment(self, deployment_id: str, results: ExperimentResult) -> bool:
        """Mark an experiment as complete and record results"""
        if deployment_id not in self.deployments:
            self.logger.error(f"Deployment {deployment_id} not found")
            return False
        
        deployment = self.deployments[deployment_id]
        deployment.status = DeploymentStatus.COMPLETED
        deployment.completed_at = datetime.utcnow().isoformat()
        deployment.last_updated = datetime.utcnow().isoformat()
        
        # Store experiment results
        self.experiment_results[results.experiment_id] = results
        
        self.logger.info(f"Completed experiment for deployment {deployment_id}")
        self._save_config()
        return True
    
    def get_deployment_status(self, deployment_id: str) -> Optional[Dict[str, Any]]:
        """Get the status of a deployment"""
        if deployment_id not in self.deployments:
            return None
        
        deployment = self.deployments[deployment_id]
        return {
            "deployment_id": deployment.deployment_id,
            "model_name": deployment.model_name,
            "strategy": deployment.strategy.value,
            "status": deployment.status.value,
            "traffic_percentage": deployment.config.traffic_percentage,
            "target_percentage": deployment.config.target_percentage,
            "current_version": deployment.current_version.version_id,
            "new_version": deployment.new_version.version_id if deployment.new_version else None,
            "created_at": deployment.created_at,
            "started_at": deployment.started_at,
            "completed_at": deployment.completed_at,
            "last_updated": deployment.last_updated,
            "alerts": len([a for a in deployment.alerts if not a.get("resolved", False)]),
            "metrics_history_count": len(deployment.metrics_history)
        }
    
    def list_deployments(self, model_name: Optional[str] = None) -> List[Dict[str, Any]]:
        """List all deployments, optionally filtered by model name"""
        deployments = []
        
        for deployment in self.deployments.values():
            if model_name and deployment.model_name != model_name:
                continue
            
            deployments.append(self.get_deployment_status(deployment.deployment_id))
        
        return deployments


# Global deployment manager instance
deployment_manager = DeploymentManager("./deployment_config.json")


# Example usage and testing
def test_deployment_manager():
    """Test the deployment manager functionality"""
    logger.info("Testing Deployment Manager...")
    
    # Create a test version
    test_version = DeploymentVersion(
        version_id="v2.0.0",
        model_name="therapy_model",
        model_path="/models/therapy_model_v2",
        model_type="pytorch",
        created_at=datetime.utcnow().isoformat(),
        size_bytes=1024*1024*500,  # 500MB
        performance_baseline={
            "latency_ms": 150.0,
            "throughput_reqs_per_sec": 10.0
        }
    )
    
    # Test different deployment strategies
    
    # 1. Direct deployment
    print("Testing Direct Deployment...")
    direct_config = DeploymentConfig(
        strategy=DeploymentStrategy.DIRECT,
        success_criteria={"latency_ms": 200.0, "error_rate": 0.01},
        rollback_criteria={"latency_ms": 500.0, "error_rate": 0.05}
    )
    
    direct_deployment_id = deployment_manager.create_deployment(
        model_name="therapy_model",
        new_version=test_version,
        strategy=DeploymentStrategy.DIRECT,
        config=direct_config
    )
    
    print(f"Created direct deployment: {direct_deployment_id}")
    
    # Start the deployment
    success = deployment_manager.start_deployment(direct_deployment_id)
    print(f"Direct deployment started: {success}")
    
    # Get deployment status
    status = deployment_manager.get_deployment_status(direct_deployment_id)
    print(f"Direct deployment status: {status}")
    
    # 2. Canary deployment
    print("\nTesting Canary Deployment...")
    canary_config = DeploymentConfig(
        strategy=DeploymentStrategy.CANARY,
        traffic_percentage=0.0,
        target_percentage=100.0,
        ramp_up_interval_minutes=5,
        ramp_up_increment=20.0,
        success_criteria={"latency_ms": 200.0, "error_rate": 0.01},
        rollback_criteria={"latency_ms": 500.0, "error_rate": 0.05}
    )
    
    canary_deployment_id = deployment_manager.create_deployment(
        model_name="therapy_model",
        new_version=test_version,
        strategy=DeploymentStrategy.CANARY,
        config=canary_config
    )
    
    print(f"Created canary deployment: {canary_deployment_id}")
    
    # Start the deployment
    success = deployment_manager.start_deployment(canary_deployment_id)
    print(f"Canary deployment started: {success}")
    
    # Advance the canary
    success = deployment_manager.advance_canary_deployment(canary_deployment_id)
    print(f"Canary deployment advanced: {success}")
    
    # 3. A/B Testing
    print("\nTesting A/B Testing...")
    ab_config = DeploymentConfig(
        strategy=DeploymentStrategy.AB_TEST,
        ab_test_ratio=30.0,  # 30% to new version
        success_criteria={"latency_ms": 200.0, "user_satisfaction": 0.8},
        rollback_criteria={"latency_ms": 500.0, "error_rate": 0.05}
    )
    
    ab_deployment_id = deployment_manager.create_deployment(
        model_name="therapy_model",
        new_version=test_version,
        strategy=DeploymentStrategy.AB_TEST,
        config=ab_config
    )
    
    print(f"Created A/B test deployment: {ab_deployment_id}")
    
    # Start the deployment
    success = deployment_manager.start_deployment(ab_deployment_id)
    print(f"A/B test deployment started: {success}")
    
    # Test routing decisions
    print("\nTesting Routing Decisions...")
    
    # Test with different user contexts
    test_contexts = [
        {"user_id": "user_123", "session_id": "sess_456"},
        {"user_id": "user_789", "session_id": "sess_012"},
        {"user_id": "user_345", "session_id": "sess_678"}
    ]
    
    for i, context in enumerate(test_contexts):
        version = deployment_manager.get_route_version("therapy_model", context)
        print(f"Context {i+1}: User {context['user_id']} routed to version {version}")
    
    # Test deployment evaluation
    print("\nTesting Deployment Evaluation...")
    test_metrics = {
        "latency_ms": 180.0,
        "error_rate": 0.005,
        "throughput_reqs_per_sec": 12.0,
        "user_satisfaction": 0.85
    }
    
    evaluation_result = deployment_manager.evaluate_deployment(direct_deployment_id, test_metrics)
    print(f"Evaluation result: {evaluation_result}")
    
    # List all deployments
    print("\nListing all deployments...")
    all_deployments = deployment_manager.list_deployments()
    print(f"Total deployments: {len(all_deployments)}")
    for dep in all_deployments:
        print(f"  - {dep['deployment_id']}: {dep['status']} ({dep['traffic_percentage']}%)")
    
    print("Deployment manager tests completed!")


if __name__ == "__main__":
    test_deployment_manager()