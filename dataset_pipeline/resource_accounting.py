"""
Cost and resource accounting system for tracking and controlling spend during training runs.
Implements monitoring and alerts for resource usage and budget management.
"""

import json
import os
import time
import psutil
import torch
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from datetime import datetime
import logging
from pathlib import Path
import threading
from dataclasses import asdict
import atexit


logger = logging.getLogger(__name__)


@dataclass
class ResourceUsage:
    """Represents resource usage at a point in time"""
    timestamp: str
    cpu_percent: float
    memory_used_gb: float
    memory_total_gb: float
    gpu_memory_used_gb: Optional[float] = None
    gpu_memory_total_gb: Optional[float] = None
    gpu_utilization_percent: Optional[float] = None
    network_io_mb: Optional[float] = None
    disk_io_mb: Optional[float] = None
    training_step: Optional[int] = None


@dataclass
class CostEstimation:
    """Cost estimation for a training run"""
    duration_hours: float
    estimated_cost_usd: float
    compute_cost_usd: float
    storage_cost_usd: float
    network_cost_usd: Optional[float] = None
    gpu_cost_usd: Optional[float] = None
    cpu_cost_usd: Optional[float] = None


@dataclass
class BudgetLimits:
    """Budget limits for training runs"""
    max_cost_usd: float
    max_runtime_hours: float
    max_gpu_memory_gb: float
    max_system_memory_gb: float
    max_network_gb: Optional[float] = None
    notification_threshold: float = 0.8  # Percentage at which to send alerts


@dataclass
class ResourceReport:
    """Complete resource and cost report"""
    run_id: str
    start_time: str
    end_time: Optional[str] = None
    total_runtime_hours: Optional[float] = None
    peak_resources: ResourceUsage = None
    average_resources: ResourceUsage = None
    cost_estimation: Optional[CostEstimation] = None
    budget_limits: Optional[BudgetLimits] = None
    exceeded_limits: List[str] = field(default_factory=list)
    resource_usage_history: List[ResourceUsage] = field(default_factory=list)
    metadata: Optional[Dict[str, Any]] = None


class ResourceMonitor:
    """Real-time resource monitoring system"""
    
    def __init__(self, run_id: str, budget_limits: Optional[BudgetLimits] = None):
        self.run_id = run_id
        self.budget_limits = budget_limits
        self.start_time = time.time()
        self.is_monitoring = False
        self.monitoring_thread = None
        self.resource_history: List[ResourceUsage] = []
        self.peak_usage = None
        self.logger = logging.getLogger(__name__)
        
        # Initialize GPU monitoring if available
        self.gpu_available = torch.cuda.is_available()
        self.gpu_count = torch.cuda.device_count() if self.gpu_available else 0
        
        # Setup atexit handler to ensure monitoring stops
        atexit.register(self.stop_monitoring)
    
    def start_monitoring(self, interval: float = 30.0) -> bool:
        """Start resource monitoring in a separate thread"""
        if self.is_monitoring:
            self.logger.warning(f"Resource monitoring already running for run {self.run_id}")
            return False
        
        self.is_monitoring = True
        self.monitoring_thread = threading.Thread(
            target=self._monitor_loop,
            args=(interval,),
            daemon=True
        )
        self.monitoring_thread.start()
        
        self.logger.info(f"Started resource monitoring for run {self.run_id}")
        return True
    
    def stop_monitoring(self):
        """Stop resource monitoring"""
        if not self.is_monitoring:
            return
        
        self.is_monitoring = False
        if self.monitoring_thread and self.monitoring_thread.is_alive():
            self.monitoring_thread.join(timeout=2.0)  # Wait up to 2 seconds
        
        self.logger.info(f"Stopped resource monitoring for run {self.run_id}")
    
    def _monitor_loop(self, interval: float):
        """Main monitoring loop running in separate thread"""
        while self.is_monitoring:
            try:
                usage = self._collect_resource_usage()
                self.resource_history.append(usage)
                
                # Update peak usage
                self._update_peak_usage(usage)
                
                # Check limits
                exceeded = self._check_limits(usage)
                if exceeded:
                    self._handle_limit_exceeded(exceeded)
                
                time.sleep(interval)
            except Exception as e:
                self.logger.error(f"Error in resource monitoring: {e}")
                break
    
    def _collect_resource_usage(self) -> ResourceUsage:
        """Collect current resource usage"""
        current_time = datetime.utcnow().isoformat()
        
        # CPU and memory usage
        cpu_percent = psutil.cpu_percent()
        memory_info = psutil.virtual_memory()
        memory_used_gb = memory_info.used / (1024**3)
        memory_total_gb = memory_info.total / (1024**3)
        
        # GPU usage if available
        gpu_memory_used_gb = None
        gpu_memory_total_gb = None
        gpu_utilization_percent = None
        
        if self.gpu_available and self.gpu_count > 0:
            try:
                # Try to get GPU info using torch (more reliable)
                if torch.cuda.is_available():
                    gpu_memory_used_gb = torch.cuda.memory_allocated() / (1024**3)
                    gpu_memory_total_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
                    # GPU utilization is harder to get with PyTorch
                    gpu_utilization_percent = torch.cuda.utilization() if hasattr(torch.cuda, 'utilization') else 0
            except Exception as e:
                self.logger.warning(f"Could not get GPU usage: {e}")
        
        return ResourceUsage(
            timestamp=current_time,
            cpu_percent=cpu_percent,
            memory_used_gb=memory_used_gb,
            memory_total_gb=memory_total_gb,
            gpu_memory_used_gb=gpu_memory_used_gb,
            gpu_memory_total_gb=gpu_memory_total_gb,
            gpu_utilization_percent=gpu_utilization_percent
        )
    
    def _update_peak_usage(self, usage: ResourceUsage):
        """Update peak resource usage"""
        if self.peak_usage is None:
            self.peak_usage = usage
        else:
            # Update peak values
            if usage.memory_used_gb > self.peak_usage.memory_used_gb:
                self.peak_usage.memory_used_gb = usage.memory_used_gb
            if usage.cpu_percent > self.peak_usage.cpu_percent:
                self.peak_usage.cpu_percent = usage.cpu_percent
            if (usage.gpu_memory_used_gb is not None and 
                self.peak_usage.gpu_memory_used_gb is not None and
                usage.gpu_memory_used_gb > self.peak_usage.gpu_memory_used_gb):
                self.peak_usage.gpu_memory_used_gb = usage.gpu_memory_used_gb
            if (usage.gpu_utilization_percent is not None and 
                self.peak_usage.gpu_utilization_percent is not None and
                usage.gpu_utilization_percent > self.peak_usage.gpu_utilization_percent):
                self.peak_usage.gpu_utilization_percent = usage.gpu_utilization_percent
    
    def _check_limits(self, usage: ResourceUsage) -> List[str]:
        """Check if current usage exceeds budget limits"""
        if not self.budget_limits:
            return []
        
        exceeded = []
        
        # Check GPU memory
        if (self.budget_limits.max_gpu_memory_gb and 
            usage.gpu_memory_used_gb and 
            usage.gpu_memory_used_gb > self.budget_limits.max_gpu_memory_gb):
            exceeded.append(f"GPU memory ({usage.gpu_memory_used_gb:.2f}GB > {self.budget_limits.max_gpu_memory_gb}GB)")
        
        # Check system memory
        if (self.budget_limits.max_system_memory_gb and 
            usage.memory_used_gb > self.budget_limits.max_system_memory_gb):
            exceeded.append(f"System memory ({usage.memory_used_gb:.2f}GB > {self.budget_limits.max_system_memory_gb}GB)")
        
        # Check runtime (if budget has max time limit)
        current_runtime = (time.time() - self.start_time) / 3600
        if (self.budget_limits.max_runtime_hours and 
            current_runtime > self.budget_limits.max_runtime_hours):
            exceeded.append(f"Runtime ({current_runtime:.2f}h > {self.budget_limits.max_runtime_hours}h)")
        
        return exceeded
    
    def _handle_limit_exceeded(self, exceeded: List[str]):
        """Handle when limits are exceeded"""
        for limit in exceeded:
            self.logger.warning(f"Limit exceeded for run {self.run_id}: {limit}")
    
    def get_current_usage(self) -> ResourceUsage:
        """Get the latest resource usage"""
        if self.resource_history:
            return self.resource_history[-1]
        else:
            return self._collect_resource_usage()
    
    def get_average_usage(self) -> ResourceUsage:
        """Calculate average resource usage over the run"""
        if not self.resource_history:
            return self._collect_resource_usage()
        
        # Calculate averages
        cpu_percent = sum(u.cpu_percent for u in self.resource_history) / len(self.resource_history)
        memory_used_gb = sum(u.memory_used_gb for u in self.resource_history) / len(self.resource_history)
        
        # For GPU, only average if we have values
        gpu_memory_used_gb = None
        if any(u.gpu_memory_used_gb is not None for u in self.resource_history):
            gpu_values = [u.gpu_memory_used_gb for u in self.resource_history if u.gpu_memory_used_gb is not None]
            if gpu_values:
                gpu_memory_used_gb = sum(gpu_values) / len(gpu_values)
        
        gpu_utilization_percent = None
        if any(u.gpu_utilization_percent is not None for u in self.resource_history):
            gpu_values = [u.gpu_utilization_percent for u in self.resource_history if u.gpu_utilization_percent is not None]
            if gpu_values:
                gpu_utilization_percent = sum(gpu_values) / len(gpu_values)
        
        return ResourceUsage(
            timestamp=datetime.utcnow().isoformat(),
            cpu_percent=cpu_percent,
            memory_used_gb=memory_used_gb,
            memory_total_gb=self.resource_history[0].memory_total_gb,
            gpu_memory_used_gb=gpu_memory_used_gb,
            gpu_memory_total_gb=self.resource_history[0].gpu_memory_total_gb if self.resource_history else None,
            gpu_utilization_percent=gpu_utilization_percent
        )


class CostCalculator:
    """Calculator for estimating training costs"""
    
    def __init__(self):
        # Default pricing (these would come from cloud provider APIs in real implementation)
        self.default_pricing = {
            # AWS pricing examples (per hour)
            'cpu_instance': {
                'm5.large': 0.096,    # $0.096 per hour
                'm5.xlarge': 0.192,   # $0.192 per hour
            },
            'gpu_instance': {
                'g4dn.xlarge': 0.526,  # $0.526 per hour (1x T4)
                'g4dn.2xlarge': 0.752, # $0.752 per hour (1x T4)
                'p3.2xlarge': 3.06,    # $3.06 per hour (1x V100)
            },
            'storage': {
                'gp2_per_gb_month': 0.10,  # $0.10 per GB per month
                'gp3_per_gb_month': 0.08,  # $0.08 per GB per month
            },
            'network': {
                'ingress_per_gb': 0.0,    # Usually free
                'egress_per_gb': 0.09,    # $0.09 per GB (first 10TB)
            }
        }
        self.logger = logging.getLogger(__name__)
    
    def calculate_cost(self, 
                      runtime_hours: float,
                      resources_used: ResourceUsage,
                      instance_type: str = 'g4dn.xlarge',
                      storage_gb: float = 100.0) -> CostEstimation:
        """Calculate estimated cost based on resource usage"""
        # Determine if it's a GPU or CPU instance
        if any(gpu_type in instance_type for gpu_type in ['g4', 'p3', 'p4', 'g5']):
            # GPU instance
            compute_rate = self.default_pricing['gpu_instance'].get(instance_type, 1.0)
        else:
            # CPU instance
            compute_rate = self.default_pricing['cpu_instance'].get(instance_type, 0.1)
        
        # Calculate costs
        compute_cost = compute_rate * runtime_hours
        storage_cost = (self.default_pricing['storage']['gp3_per_gb_month'] / 730) * storage_gb * runtime_hours  # Convert monthly to hourly
        estimated_cost = compute_cost + storage_cost
        
        # Calculate GPU-specific costs if applicable
        gpu_cost = compute_cost if 'g4' in instance_type or 'p3' in instance_type else 0
        cpu_cost = 0 if gpu_cost > 0 else compute_cost
        
        return CostEstimation(
            duration_hours=runtime_hours,
            estimated_cost_usd=estimated_cost,
            compute_cost_usd=compute_cost,
            storage_cost_usd=storage_cost,
            gpu_cost_usd=gpu_cost,
            cpu_cost_usd=cpu_cost
        )
    
    def estimate_remaining_budget(self, 
                                current_cost: float, 
                                budget_limit: float) -> Tuple[float, float, str]:
        """Estimate remaining budget and time"""
        remaining = budget_limit - current_cost
        percentage_used = (current_cost / budget_limit) * 100 if budget_limit > 0 else 0
        
        status = "ok"
        if percentage_used >= 90:
            status = "critical"
        elif percentage_used >= 75:
            status = "warning"
        elif percentage_used >= 50:
            status = "caution"
        
        return remaining, percentage_used, status


class ResourceManager:
    """Main resource management system"""
    
    def __init__(self):
        self.active_monitors: Dict[str, ResourceMonitor] = {}
        self.cost_calculator = CostCalculator()
        self.logger = logging.getLogger(__name__)
    
    def start_run_monitoring(self, 
                           run_id: str, 
                           budget_limits: Optional[BudgetLimits] = None,
                           interval: float = 30.0) -> bool:
        """Start monitoring for a training run"""
        if run_id in self.active_monitors:
            self.logger.warning(f"Run {run_id} is already being monitored")
            return False
        
        monitor = ResourceMonitor(run_id, budget_limits)
        self.active_monitors[run_id] = monitor
        
        return monitor.start_monitoring(interval)
    
    def stop_run_monitoring(self, run_id: str) -> bool:
        """Stop monitoring for a training run"""
        if run_id not in self.active_monitors:
            return False
        
        monitor = self.active_monitors[run_id]
        monitor.stop_monitoring()
        del self.active_monitors[run_id]
        return True
    
    def get_run_report(self, run_id: str, instance_type: str = 'g4dn.xlarge') -> Optional[ResourceReport]:
        """Generate comprehensive resource report for a run"""
        if run_id not in self.active_monitors:
            self.logger.warning(f"No monitor found for run {run_id}")
            return None
        
        monitor = self.active_monitors[run_id]
        current_time = time.time()
        runtime_hours = (current_time - monitor.start_time) / 3600
        
        # Calculate average and peak usage
        if monitor.resource_history:
            average_usage = monitor.get_average_usage()
            peak_usage = monitor.peak_usage
        else:
            average_usage = peak_usage = monitor.get_current_usage()
        
        # Calculate costs
        cost_estimation = self.cost_calculator.calculate_cost(
            runtime_hours=runtime_hours,
            resources_used=average_usage,
            instance_type=instance_type
        )
        
        # Determine exceeded limits
        exceeded_limits = []
        if monitor.budget_limits:
            if runtime_hours > monitor.budget_limits.max_runtime_hours:
                exceeded_limits.append(f"Runtime exceeded: {runtime_hours:.2f}h > {monitor.budget_limits.max_runtime_hours}h")
            
            if (monitor.budget_limits.max_cost_usd and 
                cost_estimation.estimated_cost_usd > monitor.budget_limits.max_cost_usd):
                exceeded_limits.append(f"Cost exceeded: ${cost_estimation.estimated_cost_usd:.2f} > ${monitor.budget_limits.max_cost_usd}")
        
        return ResourceReport(
            run_id=run_id,
            start_time=datetime.fromtimestamp(monitor.start_time).isoformat(),
            end_time=datetime.utcnow().isoformat(),
            total_runtime_hours=runtime_hours,
            peak_resources=peak_usage,
            average_resources=average_usage,
            cost_estimation=cost_estimation,
            budget_limits=monitor.budget_limits,
            exceeded_limits=exceeded_limits,
            resource_usage_history=monitor.resource_history.copy()
        )
    
    def get_all_active_runs(self) -> List[str]:
        """Get list of all actively monitored runs"""
        return list(self.active_monitors.keys())
    
    def check_budget_alerts(self, run_id: str) -> Optional[Tuple[float, str]]:
        """Check if a run should trigger budget alerts"""
        if run_id not in self.active_monitors:
            return None
        
        monitor = self.active_monitors[run_id]
        if not monitor.budget_limits:
            return None
        
        # Calculate current cost
        current_runtime = (time.time() - monitor.start_time) / 3600
        current_usage = monitor.get_current_usage()
        current_cost = self.cost_calculator.calculate_cost(
            runtime_hours=current_runtime,
            resources_used=current_usage
        ).estimated_cost_usd
        
        # Check if we've exceeded notification threshold
        if monitor.budget_limits.max_cost_usd > 0:
            usage_percentage = (current_cost / monitor.budget_limits.max_cost_usd) * 100
            if usage_percentage >= monitor.budget_limits.notification_threshold * 100:
                return current_cost, f"Budget usage reached {usage_percentage:.1f}% threshold"
        
        return None
    
    def save_resource_report(self, report: ResourceReport, filepath: str):
        """Save resource report to file"""
        report_dict = {
            'run_id': report.run_id,
            'start_time': report.start_time,
            'end_time': report.end_time,
            'total_runtime_hours': report.total_runtime_hours,
            'peak_resources': asdict(report.peak_resources) if report.peak_resources else {},
            'average_resources': asdict(report.average_resources) if report.average_resources else {},
            'cost_estimation': asdict(report.cost_estimation) if report.cost_estimation else {},
            'budget_limits': asdict(report.budget_limits) if report.budget_limits else {},
            'exceeded_limits': report.exceeded_limits,
            'resource_usage_history': [asdict(usage) for usage in report.resource_usage_history],
            'metadata': report.metadata
        }
        
        with open(filepath, 'w') as f:
            json.dump(report_dict, f, indent=2)
        
        self.logger.info(f"Saved resource report to {filepath}")
    
    def generate_cost_report(self, run_id: str) -> str:
        """Generate a human-readable cost report"""
        report = self.get_run_report(run_id)
        if not report or not report.cost_estimation:
            return f"No cost information available for run {run_id}"
        
        cost_est = report.cost_estimation
        
        report_lines = [
            f"=== Resource and Cost Report for Run: {run_id} ===",
            f"Runtime: {cost_est.duration_hours:.2f} hours",
            f"Estimated Total Cost: ${cost_est.estimated_cost_usd:.2f}",
            "",
            "Cost Breakdown:",
            f"  Compute: ${cost_est.compute_cost_usd:.2f}",
            f"  Storage: ${cost_est.storage_cost_usd:.2f}",
            f"  GPU Cost: ${cost_est.gpu_cost_usd or 0:.2f}",
            f"  CPU Cost: ${cost_est.cpu_cost_usd or 0:.2f}",
            "",
            f"Peak Memory Usage: {report.peak_resources.memory_used_gb:.2f}GB" if report.peak_resources else "",
            f"Average CPU Usage: {report.average_resources.cpu_percent:.1f}%" if report.average_resources else "",
            f"Peak GPU Memory Usage: {report.peak_resources.gpu_memory_used_gb:.2f}GB" if report.peak_resources and report.peak_resources.gpu_memory_used_gb else "",
        ]
        
        if report.exceeded_limits:
            report_lines.extend([
                "",
                "Budget Limit Exceeded:",
            ])
            for limit in report.exceeded_limits:
                report_lines.append(f"  - {limit}")
        
        return "\n".join([line for line in report_lines if line.strip()])


class BudgetManager:
    """Manager for handling overall budget allocation and tracking"""
    
    def __init__(self, total_budget_usd: float):
        self.total_budget_usd = total_budget_usd
        self.spent_budget_usd = 0.0
        self.run_budgets: Dict[str, float] = {}  # run_id -> allocated_budget
        self.run_costs: Dict[str, float] = {}    # run_id -> actual_cost
        self.logger = logging.getLogger(__name__)
    
    def allocate_budget(self, run_id: str, amount_usd: float) -> bool:
        """Allocate budget for a specific run"""
        remaining_budget = self.total_budget_usd - self.spent_budget_usd
        
        if amount_usd > remaining_budget:
            self.logger.error(f"Insufficient budget: requested ${amount_usd}, available ${remaining_budget}")
            return False
        
        self.run_budgets[run_id] = amount_usd
        self.logger.info(f"Allocated ${amount_usd} for run {run_id}")
        return True
    
    def track_cost(self, run_id: str, cost_usd: float):
        """Track actual cost for a run"""
        self.run_costs[run_id] = cost_usd
        self.spent_budget_usd = sum(self.run_costs.values())
        
        # Check overall budget
        if self.spent_budget_usd > self.total_budget_usd:
            self.logger.warning(f"Total spent exceeds budget: ${self.spent_budget_usd} > ${self.total_budget_usd}")
    
    def get_remaining_budget(self) -> float:
        """Get remaining budget"""
        return self.total_budget_usd - self.spent_budget_usd
    
    def is_overall_budget_exceeded(self) -> bool:
        """Check if overall budget is exceeded"""
        return self.spent_budget_usd > self.total_budget_usd
    
    def get_run_remaining_budget(self, run_id: str) -> float:
        """Get remaining budget for a specific run"""
        allocated = self.run_budgets.get(run_id, 0)
        spent = self.run_costs.get(run_id, 0)
        return allocated - spent


def create_resource_manager() -> ResourceManager:
    """Create a default resource manager"""
    return ResourceManager()


def create_budget_manager(total_budget: float) -> BudgetManager:
    """Create a default budget manager"""
    return BudgetManager(total_budget)


# Integration functions for training runs
def setup_resource_monitoring(run_id: str, 
                            budget_limits: Optional[BudgetLimits] = None) -> ResourceManager:
    """Setup resource monitoring for a training run"""
    rm = create_resource_manager()
    rm.start_run_monitoring(run_id, budget_limits)
    return rm


def get_training_cost_estimate(resources: ResourceUsage, 
                             runtime_hours: float,
                             instance_type: str = 'g4dn.xlarge') -> CostEstimation:
    """Get cost estimate for training run"""
    calculator = CostCalculator()
    return calculator.calculate_cost(runtime_hours, resources, instance_type)


# Example usage and testing
def test_resource_accounting_system():
    """Test the resource accounting system"""
    logger.info("Testing Resource Accounting System...")
    
    # Create resource manager
    rm = create_resource_manager()
    
    # Define budget limits
    budget_limits = BudgetLimits(
        max_cost_usd=100.0,
        max_runtime_hours=10.0,
        max_gpu_memory_gb=16.0,
        max_system_memory_gb=32.0,
        notification_threshold=0.8
    )
    
    # Start monitoring for a test run
    run_id = "test_run_123"
    rm.start_run_monitoring(run_id, budget_limits, interval=5.0)  # Check every 5 seconds for testing
    
    print(f"Started monitoring for run: {run_id}")
    
    # Simulate some training activity
    import time
    time.sleep(7)  # Wait for a few monitoring cycles
    
    # Get current resource usage
    current_usage = rm.active_monitors[run_id].get_current_usage()
    print(f"Current resource usage: CPU {current_usage.cpu_percent}%, Memory {current_usage.memory_used_gb:.2f}GB")
    
    # Check for budget alerts
    alert = rm.check_budget_alerts(run_id)
    if alert:
        print(f"Budget alert: {alert[1]} (Cost: ${alert[0]:.2f})")
    
    # Generate and print report
    report = rm.get_run_report(run_id)
    if report:
        print(f"\nRun report generated:")
        print(f"  Runtime: {report.total_runtime_hours:.2f} hours")
        print(f"  Peak memory: {report.peak_resources.memory_used_gb:.2f}GB")
        print(f"  Estimated cost: ${report.cost_estimation.estimated_cost_usd:.2f}" if report.cost_estimation else "  Cost: Not available")
    
    # Generate cost report
    cost_report = rm.generate_cost_report(run_id)
    print(f"\nCost Report:\n{cost_report}")
    
    # Save report
    rm.save_resource_report(report, f"./test_resource_report_{run_id}.json")
    print(f"Resource report saved to ./test_resource_report_{run_id}.json")
    
    # Stop monitoring
    rm.stop_run_monitoring(run_id)
    print(f"Stopped monitoring for run: {run_id}")
    
    # Test budget manager
    print(f"\nTesting Budget Manager...")
    budget_manager = create_budget_manager(1000.0)  # $1000 total budget
    
    # Allocate budget for runs
    success1 = budget_manager.allocate_budget("run_a", 300.0)
    success2 = budget_manager.allocate_budget("run_b", 400.0)
    success3 = budget_manager.allocate_budget("run_c", 500.0)  # This should fail
    
    print(f"Budget allocation results: Run A: {success1}, Run B: {success2}, Run C: {success3}")
    print(f"Remaining budget: ${budget_manager.get_remaining_budget():.2f}")
    
    # Track costs
    budget_manager.track_cost("run_a", 250.0)
    budget_manager.track_cost("run_b", 100.0)
    print(f"Tracked costs - Run A: $250, Run B: $100")
    print(f"Remaining budget: ${budget_manager.get_remaining_budget():.2f}")
    print(f"Run A remaining: ${budget_manager.get_run_remaining_budget('run_a'):.2f}")
    print(f"Run B remaining: ${budget_manager.get_run_remaining_budget('run_b'):.2f}")
    
    print("Resource accounting system test completed!")


if __name__ == "__main__":
    test_resource_accounting_system()