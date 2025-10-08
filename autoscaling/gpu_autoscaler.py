"""
GPU autoscaling and cost management system for Pixelated Empathy AI project.
Implements dynamic scaling based on demand and provides cost optimization.
"""

import json
import time
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum
import numpy as np
import psutil
import torch


logger = logging.getLogger(__name__)


class ScalingStrategy(Enum):
    """Scaling strategies for GPU resources"""
    REACTIVE = "reactive"      # Scale based on current load
    PREDICTIVE = "predictive"   # Scale based on predicted load
    HYBRID = "hybrid"          # Combination of reactive and predictive
    SCHEDULED = "scheduled"     # Scale based on predefined schedule


class InstanceType(Enum):
    """Supported GPU instance types"""
    CPU_SMALL = "cpu_small"           # No GPU, small CPU
    CPU_MEDIUM = "cpu_medium"         # No GPU, medium CPU
    CPU_LARGE = "cpu_large"           # No GPU, large CPU
    GPU_T4 = "gpu_t4"                 # NVIDIA T4 (16GB)
    GPU_A10 = "gpu_a10"               # NVIDIA A10 (24GB)
    GPU_A100 = "gpu_a100"             # NVIDIA A100 (40GB or 80GB)
    GPU_H100 = "gpu_h100"             # NVIDIA H100 (80GB)
    GPU_L4 = "gpu_l4"                 # NVIDIA L4 (24GB)
    MULTI_GPU = "multi_gpu"           # Multiple GPUs


@dataclass
class CostModel:
    """Cost model for different instance types"""
    hourly_rate: float
    startup_cost: float = 0.0      # One-time startup cost
    shutdown_cost: float = 0.0     # Cost for scaling down
    idle_cost_multiplier: float = 1.0  # Multiplier for idle instances
    data_transfer_cost_per_gb: float = 0.01  # Cost per GB transferred
    storage_cost_per_gb_hour: float = 0.0001  # Storage cost per GB per hour


@dataclass
class ResourceUsage:
    """Current resource usage metrics"""
    cpu_percent: float
    memory_percent: float
    gpu_memory_percent: Optional[float] = None
    gpu_utilization_percent: Optional[float] = None
    network_in_mbps: float = 0.0
    network_out_mbps: float = 0.0
    disk_io_read_mbps: float = 0.0
    disk_io_write_mbps: float = 0.0
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())


@dataclass
class ScalingPolicy:
    """Policy for scaling decisions"""
    min_instances: int = 1
    max_instances: int = 10
    scale_up_threshold: float = 70.0    # CPU/GPU utilization percentage to trigger scale-up
    scale_down_threshold: float = 30.0  # CPU/GPU utilization percentage to trigger scale-down
    scale_up_factor: float = 1.5         # Multiply current instances by this factor when scaling up
    scale_down_factor: float = 0.7      # Multiply current instances by this factor when scaling down
    cooldown_period_minutes: int = 5    # Minimum time between scaling actions
    predictive_horizon_minutes: int = 30 # Time horizon for predictive scaling
    target_response_time_ms: float = 100.0  # Target response time in milliseconds


@dataclass
class ScalingDecision:
    """Decision from the autoscaler"""
    action: str  # scale_up, scale_down, maintain
    current_instances: int
    target_instances: int
    reason: str
    confidence: float
    estimated_cost_impact: float
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())


@dataclass
class CostForecast:
    """Forecast of future costs"""
    hourly_forecast: float
    daily_forecast: float
    weekly_forecast: float
    monthly_forecast: float
    projected_savings: float
    cost_drivers: List[Dict[str, Any]]
    forecast_period_hours: int = 24


class GPUCostModel:
    """Cost model for GPU instances"""
    
    # Hourly rates for different instance types (example rates)
    INSTANCE_COSTS = {
        InstanceType.CPU_SMALL: CostModel(hourly_rate=0.05, startup_cost=0.0, idle_cost_multiplier=0.1),
        InstanceType.CPU_MEDIUM: CostModel(hourly_rate=0.15, startup_cost=0.0, idle_cost_multiplier=0.1),
        InstanceType.CPU_LARGE: CostModel(hourly_rate=0.30, startup_cost=0.0, idle_cost_multiplier=0.1),
        InstanceType.GPU_T4: CostModel(hourly_rate=0.526, startup_cost=0.5, shutdown_cost=0.1, idle_cost_multiplier=0.3),
        InstanceType.GPU_A10: CostModel(hourly_rate=0.752, startup_cost=1.0, shutdown_cost=0.2, idle_cost_multiplier=0.3),
        InstanceType.GPU_A100: CostModel(hourly_rate=3.06, startup_cost=2.0, shutdown_cost=0.5, idle_cost_multiplier=0.4),
        InstanceType.GPU_H100: CostModel(hourly_rate=5.00, startup_cost=3.0, shutdown_cost=0.8, idle_cost_multiplier=0.4),
        InstanceType.GPU_L4: CostModel(hourly_rate=0.60, startup_cost=0.8, shutdown_cost=0.2, idle_cost_multiplier=0.3),
        InstanceType.MULTI_GPU: CostModel(hourly_rate=10.0, startup_cost=5.0, shutdown_cost=1.0, idle_cost_multiplier=0.5)
    }
    
    # Data transfer costs (per GB)
    DATA_TRANSFER_COST = {
        "intra_region": 0.01,
        "inter_region": 0.02,
        "internet_egress": 0.09,
        "cdn": 0.02
    }
    
    # Storage costs (per GB-hour)
    STORAGE_COST = {
        "ssd": 0.0001,
        "hdd": 0.00005,
        "cold_storage": 0.00001
    }


class Autoscaler:
    """Main autoscaling system"""
    
    def __init__(self, 
                 model_name: str,
                 scaling_strategy: ScalingStrategy = ScalingStrategy.REACTIVE,
                 instance_type: InstanceType = InstanceType.GPU_T4,
                 scaling_policy: Optional[ScalingPolicy] = None):
        self.model_name = model_name
        self.scaling_strategy = scaling_strategy
        self.instance_type = instance_type
        self.scaling_policy = scaling_policy or ScalingPolicy()
        self.current_instances = self.scaling_policy.min_instances
        self.last_scaling_action = datetime.utcnow() - timedelta(minutes=self.scaling_policy.cooldown_period_minutes)
        self.usage_history: List[ResourceUsage] = []
        self.scaling_decisions: List[ScalingDecision] = []
        self.cost_history: List[Dict[str, Any]] = []
        self.logger = logging.getLogger(__name__)
        
        # Predictive modeling components
        self.request_forecaster = RequestForecaster()
        self.cost_forecaster = CostForecaster()
    
    def get_current_resource_usage(self) -> ResourceUsage:
        """Get current resource usage metrics"""
        cpu_percent = psutil.cpu_percent()
        memory = psutil.virtual_memory()
        memory_percent = memory.percent
        
        gpu_memory_percent = None
        gpu_utilization_percent = None
        
        # Get GPU metrics if available
        if torch.cuda.is_available():
            try:
                gpu_memory_percent = (torch.cuda.memory_allocated() / torch.cuda.get_device_properties(0).total_memory) * 100
                # Approximate GPU utilization (this is a simplification)
                gpu_utilization_percent = torch.cuda.utilization() if hasattr(torch.cuda, 'utilization') else 0
            except Exception as e:
                self.logger.warning(f"Could not get GPU metrics: {e}")
        
        usage = ResourceUsage(
            cpu_percent=cpu_percent,
            memory_percent=memory_percent,
            gpu_memory_percent=gpu_memory_percent,
            gpu_utilization_percent=gpu_utilization_percent,
            timestamp=datetime.utcnow().isoformat()
        )
        
        # Store in history (keep last 100 entries)
        self.usage_history.append(usage)
        if len(self.usage_history) > 100:
            self.usage_history = self.usage_history[-100:]
        
        return usage
    
    def make_scaling_decision(self, 
                             current_load: Optional[float] = None,
                             incoming_requests: Optional[int] = None) -> ScalingDecision:
        """Make a scaling decision based on current metrics"""
        current_time = datetime.utcnow()
        
        # Check cooldown period
        if (current_time - self.last_scaling_action).total_seconds() < (self.scaling_policy.cooldown_period_minutes * 60):
            return ScalingDecision(
                action="maintain",
                current_instances=self.current_instances,
                target_instances=self.current_instances,
                reason="Cooldown period active",
                confidence=0.9,
                estimated_cost_impact=0.0
            )
        
        # Get current usage
        usage = self.get_current_resource_usage()
        
        # Determine utilization metric to use
        utilization_metric = self._get_utilization_metric(usage)
        
        # Base scaling decision on current utilization
        if utilization_metric >= self.scaling_policy.scale_up_threshold:
            # Scale up
            target_instances = min(
                int(self.current_instances * self.scaling_policy.scale_up_factor),
                self.scaling_policy.max_instances
            )
            
            decision = ScalingDecision(
                action="scale_up",
                current_instances=self.current_instances,
                target_instances=target_instances,
                reason=f"Utilization ({utilization_metric:.1f}%) above scale-up threshold ({self.scaling_policy.scale_up_threshold}%)",
                confidence=0.8,
                estimated_cost_impact=self._estimate_cost_impact(self.current_instances, target_instances)
            )
            
        elif utilization_metric <= self.scaling_policy.scale_down_threshold:
            # Scale down
            target_instances = max(
                int(self.current_instances * self.scaling_policy.scale_down_factor),
                self.scaling_policy.min_instances
            )
            
            decision = ScalingDecision(
                action="scale_down",
                current_instances=self.current_instances,
                target_instances=target_instances,
                reason=f"Utilization ({utilization_metric:.1f}%) below scale-down threshold ({self.scaling_policy.scale_down_threshold}%)",
                confidence=0.7,
                estimated_cost_impact=self._estimate_cost_impact(self.current_instances, target_instances)
            )
            
        else:
            # Maintain current instances
            decision = ScalingDecision(
                action="maintain",
                current_instances=self.current_instances,
                target_instances=self.current_instances,
                reason=f"Utilization ({utilization_metric:.1f}%) within normal range",
                confidence=0.95,
                estimated_cost_impact=0.0
            )
        
        # Apply predictive adjustments if using predictive or hybrid strategy
        if self.scaling_strategy in [ScalingStrategy.PREDICTIVE, ScalingStrategy.HYBRID]:
            predicted_load = self.request_forecaster.predict_future_load(
                self.usage_history, 
                horizon_minutes=self.scaling_policy.predictive_horizon_minutes
            )
            
            if predicted_load > utilization_metric * 1.2:  # Predicted load is 20% higher than current
                # Adjust scaling decision upwards
                target_instances = min(
                    int(target_instances * 1.2),
                    self.scaling_policy.max_instances
                )
                decision.target_instances = target_instances
                decision.reason += f" | Predictive adjustment for expected load increase"
                decision.confidence *= 0.9  # Slightly lower confidence for predictions
        
        # Record the decision
        self.scaling_decisions.append(decision)
        if len(self.scaling_decisions) > 1000:
            self.scaling_decisions = self.scaling_decisions[-1000:]
        
        # Update last scaling action time if we're actually scaling
        if decision.action != "maintain":
            self.last_scaling_action = current_time
            self.current_instances = decision.target_instances
        
        self.logger.info(f"Scaling decision: {decision.action} from {decision.current_instances} to {decision.target_instances} instances")
        
        return decision
    
    def _get_utilization_metric(self, usage: ResourceUsage) -> float:
        """Get the primary utilization metric for scaling decisions"""
        # Prioritize GPU metrics if available, otherwise use CPU
        if usage.gpu_utilization_percent is not None:
            return usage.gpu_utilization_percent
        elif usage.gpu_memory_percent is not None:
            return usage.gpu_memory_percent
        else:
            return usage.cpu_percent
    
    def _estimate_cost_impact(self, current_instances: int, target_instances: int) -> float:
        """Estimate the cost impact of a scaling decision"""
        cost_model = GPUCostModel.INSTANCE_COSTS[self.instance_type]
        
        # Calculate hourly cost difference
        current_hourly_cost = current_instances * cost_model.hourly_rate
        target_hourly_cost = target_instances * cost_model.hourly_rate
        hourly_cost_difference = target_hourly_cost - current_hourly_cost
        
        # Add startup/shutdown costs
        if target_instances > current_instances:
            # Scaling up
            startup_cost = (target_instances - current_instances) * cost_model.startup_cost
            cost_impact = hourly_cost_difference + startup_cost
        elif target_instances < current_instances:
            # Scaling down
            shutdown_cost = (current_instances - target_instances) * cost_model.shutdown_cost
            cost_impact = hourly_cost_difference - shutdown_cost
        else:
            # Maintaining
            cost_impact = 0.0
        
        return cost_impact
    
    def get_cost_forecast(self, hours_ahead: int = 24) -> CostForecast:
        """Get cost forecast for the specified period"""
        current_usage = self.get_current_resource_usage()
        utilization = self._get_utilization_metric(current_usage)
        
        # Calculate base costs
        cost_model = GPUCostModel.INSTANCE_COSTS[self.instance_type]
        hourly_base_cost = self.current_instances * cost_model.hourly_rate
        
        # Adjust for current utilization
        if utilization < 20:
            # Low utilization - idle costs
            hourly_cost = hourly_base_cost * cost_model.idle_cost_multiplier
        elif utilization > 80:
            # High utilization - full costs
            hourly_cost = hourly_base_cost
        else:
            # Normal utilization - proportional costs
            hourly_cost = hourly_base_cost * (0.5 + (utilization / 100) * 0.5)
        
        # Calculate forecasts
        hourly_forecast = hourly_cost
        daily_forecast = hourly_forecast * 24
        weekly_forecast = daily_forecast * 7
        monthly_forecast = daily_forecast * 30
        
        # Estimate potential savings from optimization
        projected_savings = 0.0
        if utilization < 30:
            # Could potentially scale down
            potential_savings = hourly_base_cost * 0.3  # Rough estimate
            projected_savings = potential_savings * hours_ahead
        
        cost_drivers = self._identify_cost_drivers(utilization, hourly_base_cost)
        
        return CostForecast(
            hourly_forecast=hourly_forecast,
            daily_forecast=daily_forecast,
            weekly_forecast=weekly_forecast,
            monthly_forecast=monthly_forecast,
            projected_savings=projected_savings,
            cost_drivers=cost_drivers,
            forecast_period_hours=hours_ahead
        )
    
    def _identify_cost_drivers(self, utilization: float, hourly_base_cost: float) -> List[Dict[str, Any]]:
        """Identify the main drivers of costs"""
        drivers = []
        
        if utilization < 20:
            drivers.append({
                "driver": "idle_capacity",
                "impact": "high",
                "description": f"Low utilization ({utilization:.1f}%) leading to wasted capacity costs",
                "savings_potential": hourly_base_cost * 0.4
            })
        elif utilization > 80:
            drivers.append({
                "driver": "high_utilization",
                "impact": "medium",
                "description": f"High utilization ({utilization:.1f}%) may require scaling",
                "savings_potential": 0.0
            })
        
        # Add instance type cost driver
        drivers.append({
            "driver": "instance_type",
            "impact": "high" if self.instance_type in [InstanceType.GPU_A100, InstanceType.GPU_H100, InstanceType.MULTI_GPU] else "medium",
            "description": f"Using {self.instance_type.value} instances",
            "savings_potential": hourly_base_cost * 0.3 if self.instance_type in [InstanceType.GPU_A100, InstanceType.GPU_H100, InstanceType.MULTI_GPU] else 0.0
        })
        
        return drivers
    
    def optimize_instance_allocation(self) -> Dict[str, Any]:
        """Optimize instance allocation based on workload patterns"""
        if len(self.usage_history) < 10:
            return {
                "recommendation": "insufficient_data",
                "message": "Need more usage data to make optimization recommendations"
            }
        
        # Analyze usage patterns
        recent_usage = self.usage_history[-24:]  # Last 24 entries (if sampled hourly)
        utilizations = [self._get_utilization_metric(usage) for usage in recent_usage]
        
        avg_utilization = np.mean(utilizations) if utilizations else 0
        max_utilization = np.max(utilizations) if utilizations else 0
        min_utilization = np.min(utilizations) if utilizations else 0
        
        recommendation = {
            "current_instances": self.current_instances,
            "avg_utilization": avg_utilization,
            "max_utilization": max_utilization,
            "min_utilization": min_utilization
        }
        
        # Make recommendations based on patterns
        if avg_utilization < 30:
            recommendation["recommendation"] = "scale_down"
            recommendation["recommended_instances"] = max(1, int(self.current_instances * 0.7))
            recommendation["reason"] = f"Average utilization {avg_utilization:.1f}% is low"
        elif avg_utilization > 70:
            recommendation["recommendation"] = "scale_up"
            recommendation["recommended_instances"] = min(self.scaling_policy.max_instances, int(self.current_instances * 1.3))
            recommendation["reason"] = f"Average utilization {avg_utilization:.1f}% is high"
        else:
            recommendation["recommendation"] = "maintain"
            recommendation["recommended_instances"] = self.current_instances
            recommendation["reason"] = f"Utilization {avg_utilization:.1f}% is within optimal range"
        
        return recommendation
    
    def get_scaling_history(self, limit: int = 50) -> List[ScalingDecision]:
        """Get recent scaling decisions"""
        return self.scaling_decisions[-limit:] if self.scaling_decisions else []


class RequestForecaster:
    """Forecasts future request loads for predictive scaling"""
    
    def __init__(self):
        self.model = None  # In a real implementation, this would be an ML model
        self.logger = logging.getLogger(__name__)
    
    def predict_future_load(self, 
                          usage_history: List[ResourceUsage], 
                          horizon_minutes: int = 30) -> float:
        """Predict future load based on historical usage"""
        if not usage_history:
            return 50.0  # Default moderate utilization
        
        # Simple forecasting using exponential smoothing
        recent_utils = []
        for usage in usage_history[-10:]:  # Last 10 data points
            util = 0
            if usage.gpu_utilization_percent is not None:
                util = usage.gpu_utilization_percent
            elif usage.gpu_memory_percent is not None:
                util = usage.gpu_memory_percent
            else:
                util = usage.cpu_percent
            recent_utils.append(util)
        
        if not recent_utils:
            return 50.0
        
        # Exponential smoothing with alpha = 0.3
        alpha = 0.3
        smoothed = recent_utils[0]
        for util in recent_utils[1:]:
            smoothed = alpha * util + (1 - alpha) * smoothed
        
        # Add a small trend component
        if len(recent_utils) > 1:
            trend = recent_utils[-1] - recent_utils[0]
            forecast = smoothed + (trend / len(recent_utils)) * (horizon_minutes / 5)  # Assuming 5-min intervals
        else:
            forecast = smoothed
        
        # Clamp forecast to reasonable bounds
        forecast = max(0.0, min(100.0, forecast))
        
        return forecast
    
    def train_model(self, historical_data: List[Dict[str, Any]]):
        """Train the forecasting model (placeholder)"""
        # In a real implementation, you would train an ML model here
        self.logger.info("Training request forecasting model with historical data")
        pass


class CostForecaster:
    """Forecasts future costs and identifies optimization opportunities"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def forecast_costs(self, 
                      autoscaler: Autoscaler,
                      hours_ahead: int = 24) -> CostForecast:
        """Forecast costs for the specified period"""
        return autoscaler.get_cost_forecast(hours_ahead)
    
    def identify_optimization_opportunities(self, 
                                           autoscaler: Autoscaler) -> List[Dict[str, Any]]:
        """Identify opportunities to reduce costs"""
        current_usage = autoscaler.get_current_resource_usage()
        utilization = autoscaler._get_utilization_metric(current_usage)
        cost_model = GPUCostModel.INSTANCE_COSTS[autoscaler.instance_type]
        
        opportunities = []
        
        # Right-sizing opportunity
        if utilization < 30 and autoscaler.instance_type not in [InstanceType.CPU_SMALL, InstanceType.GPU_L4]:
            opportunities.append({
                "opportunity": "right_sizing",
                "description": "Consider using smaller instances for current workload",
                "potential_savings": cost_model.hourly_rate * 0.4,
                "confidence": 0.8,
                "implementation": "Switch to smaller GPU instances or CPU instances if workload permits"
            })
        
        # Spot instance opportunity
        opportunities.append({
            "opportunity": "spot_instances",
            "description": "Consider using spot/preemptible instances for cost savings",
            "potential_savings": cost_model.hourly_rate * 0.6,
            "confidence": 0.7,
            "implementation": "Use spot instances with fallback to on-demand for non-critical workloads"
        })
        
        # Reserved instance opportunity
        if autoscaler.current_instances >= 5:  # Threshold for reserved instances
            opportunities.append({
                "opportunity": "reserved_instances",
                "description": "Consider reserved instances for steady-state workloads",
                "potential_savings": cost_model.hourly_rate * 0.3,
                "confidence": 0.9,
                "implementation": "Purchase reserved instances for consistently running workloads"
            })
        
        return opportunities


class MultiModelAutoscaler:
    """Manages autoscaling for multiple models"""
    
    def __init__(self):
        self.autoscalers: Dict[str, Autoscaler] = {}
        self.logger = logging.getLogger(__name__)
    
    def register_model(self, 
                      model_name: str,
                      instance_type: InstanceType = InstanceType.GPU_T4,
                      scaling_policy: Optional[ScalingPolicy] = None) -> Autoscaler:
        """Register a model for autoscaling"""
        autoscaler = Autoscaler(
            model_name=model_name,
            instance_type=instance_type,
            scaling_policy=scaling_policy
        )
        self.autoscalers[model_name] = autoscaler
        self.logger.info(f"Registered model {model_name} for autoscaling")
        return autoscaler
    
    def get_autoscaler(self, model_name: str) -> Optional[Autoscaler]:
        """Get autoscaler for a specific model"""
        return self.autoscalers.get(model_name)
    
    def make_all_scaling_decisions(self) -> Dict[str, ScalingDecision]:
        """Make scaling decisions for all registered models"""
        decisions = {}
        for model_name, autoscaler in self.autoscalers.items():
            try:
                decision = autoscaler.make_scaling_decision()
                decisions[model_name] = decision
            except Exception as e:
                self.logger.error(f"Failed to make scaling decision for {model_name}: {e}")
        return decisions
    
    def get_total_cost_forecast(self, hours_ahead: int = 24) -> CostForecast:
        """Get combined cost forecast for all models"""
        total_hourly = 0.0
        total_daily = 0.0
        total_weekly = 0.0
        total_monthly = 0.0
        total_savings = 0.0
        all_cost_drivers = []
        
        for autoscaler in self.autoscalers.values():
            try:
                forecast = autoscaler.get_cost_forecast(hours_ahead)
                total_hourly += forecast.hourly_forecast
                total_daily += forecast.daily_forecast
                total_weekly += forecast.weekly_forecast
                total_monthly += forecast.monthly_forecast
                total_savings += forecast.projected_savings
                all_cost_drivers.extend(forecast.cost_drivers)
            except Exception as e:
                self.logger.error(f"Failed to get cost forecast for {autoscaler.model_name}: {e}")
        
        return CostForecast(
            hourly_forecast=total_hourly,
            daily_forecast=total_daily,
            weekly_forecast=total_weekly,
            monthly_forecast=total_monthly,
            projected_savings=total_savings,
            cost_drivers=all_cost_drivers,
            forecast_period_hours=hours_ahead
        )


# Global autoscaler instance
multi_model_autoscaler = MultiModelAutoscaler()


# Utility functions for API integration
def register_model_for_autoscaling(model_name: str,
                                 instance_type: InstanceType = InstanceType.GPU_T4,
                                 scaling_policy: Optional[ScalingPolicy] = None) -> Autoscaler:
    """Register a model for autoscaling"""
    return multi_model_autoscaler.register_model(model_name, instance_type, scaling_policy)


def get_autoscaling_decision(model_name: str,
                           current_load: Optional[float] = None,
                           incoming_requests: Optional[int] = None) -> Optional[ScalingDecision]:
    """Get autoscaling decision for a model"""
    autoscaler = multi_model_autoscaler.get_autoscaler(model_name)
    if autoscaler:
        return autoscaler.make_scaling_decision(current_load, incoming_requests)
    return None


def get_cost_optimization_recommendations(model_name: str) -> List[Dict[str, Any]]:
    """Get cost optimization recommendations for a model"""
    autoscaler = multi_model_autoscaler.get_autoscaler(model_name)
    if autoscaler:
        forecaster = CostForecaster()
        return forecaster.identify_optimization_opportunities(autoscaler)
    return []


# Example usage and testing
def test_autoscaling_system():
    """Test the autoscaling system"""
    logger.info("Testing GPU Autoscaling System...")
    
    # Create a scaling policy
    policy = ScalingPolicy(
        min_instances=1,
        max_instances=5,
        scale_up_threshold=70.0,
        scale_down_threshold=30.0,
        scale_up_factor=1.5,
        scale_down_factor=0.7,
        cooldown_period_minutes=2,  # Short for testing
        target_response_time_ms=100.0
    )
    
    # Register a model
    model_name = "therapy_model_v1"
    autoscaler = register_model_for_autoscaling(
        model_name,
        instance_type=InstanceType.GPU_T4,
        scaling_policy=policy
    )
    
    print(f"Registered model {model_name} for autoscaling")
    
    # Test current resource usage
    print("\nTesting Resource Usage Collection...")
    usage = autoscaler.get_current_resource_usage()
    print(f"Current CPU: {usage.cpu_percent:.1f}%")
    print(f"Current Memory: {usage.memory_percent:.1f}%")
    if usage.gpu_memory_percent:
        print(f"GPU Memory: {usage.gpu_memory_percent:.1f}%")
    if usage.gpu_utilization_percent:
        print(f"GPU Utilization: {usage.gpu_utilization_percent:.1f}%")
    
    # Test scaling decisions under different conditions
    print("\nTesting Scaling Decisions...")
    
    # Low utilization scenario
    print("Low utilization scenario (< 30%)")
    low_util_decision = autoscaler.make_scaling_decision(current_load=25.0)
    print(f"Decision: {low_util_decision.action} ({low_util_decision.reason})")
    
    # High utilization scenario
    print("High utilization scenario (> 70%)")
    high_util_decision = autoscaler.make_scaling_decision(current_load=85.0)
    print(f"Decision: {high_util_decision.action} ({high_util_decision.reason})")
    
    # Normal utilization scenario
    print("Normal utilization scenario (30-70%)")
    normal_util_decision = autoscaler.make_scaling_decision(current_load=50.0)
    print(f"Decision: {normal_util_decision.action} ({normal_util_decision.reason})")
    
    # Test cost forecasting
    print("\nTesting Cost Forecasting...")
    cost_forecast = autoscaler.get_cost_forecast(hours_ahead=24)
    print(f"Hourly forecast: ${cost_forecast.hourly_forecast:.2f}")
    print(f"Daily forecast: ${cost_forecast.daily_forecast:.2f}")
    print(f"Weekly forecast: ${cost_forecast.weekly_forecast:.2f}")
    print(f"Monthly forecast: ${cost_forecast.monthly_forecast:.2f}")
    print(f"Projected savings: ${cost_forecast.projected_savings:.2f}")
    
    # Test optimization recommendations
    print("\nTesting Optimization Recommendations...")
    recommendations = get_cost_optimization_recommendations(model_name)
    print(f"Found {len(recommendations)} optimization opportunities:")
    for rec in recommendations[:3]:  # Show top 3
        print(f"  - {rec.get('opportunity', 'N/A')}: {rec.get('description', 'N/A')}")
        print(f"    Potential savings: ${rec.get('potential_savings', 0):.2f}")
    
    # Test instance optimization
    print("\nTesting Instance Allocation Optimization...")
    optimization = autoscaler.optimize_instance_allocation()
    print(f"Optimization recommendation: {optimization.get('recommendation', 'N/A')}")
    print(f"Reason: {optimization.get('reason', 'N/A')}")
    print(f"Recommended instances: {optimization.get('recommended_instances', 'N/A')}")
    
    # Test multi-model autoscaler
    print("\nTesting Multi-Model Autoscaler...")
    
    # Register another model
    model2_name = "crisis_detection_model"
    autoscaler2 = register_model_for_autoscaling(
        model2_name,
        instance_type=InstanceType.GPU_A10,
        scaling_policy=ScalingPolicy(min_instances=2, max_instances=8)
    )
    
    print(f"Registered second model {model2_name}")
    
    # Make scaling decisions for all models
    all_decisions = multi_model_autoscaler.make_all_scaling_decisions()
    print(f"Made scaling decisions for {len(all_decisions)} models")
    
    # Get total cost forecast
    total_forecast = multi_model_autoscaler.get_total_cost_forecast(hours_ahead=24)
    print(f"Total hourly cost for all models: ${total_forecast.hourly_forecast:.2f}")
    print(f"Total daily cost for all models: ${total_forecast.daily_forecast:.2f}")
    print(f"Total monthly cost for all models: ${total_forecast.monthly_forecast:.2f}")
    
    # Test scaling history
    print("\nTesting Scaling History...")
    history = autoscaler.get_scaling_history(limit=5)
    print(f"Retrieved {len(history)} recent scaling decisions")
    if history:
        latest = history[-1]
        print(f"Most recent decision: {latest.action} at {latest.timestamp}")
    
    print("\nGPU autoscaling system tests completed!")


if __name__ == "__main__":
    test_autoscaling_system()