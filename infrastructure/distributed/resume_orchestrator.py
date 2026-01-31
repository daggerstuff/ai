#!/usr/bin/env python3
"""
Resume Orchestrator for Pixelated Empathy AI
Manages coordination and orchestration of multiple resumable processes
"""

import asyncio
import json
import logging
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable, Set
from dataclasses import dataclass, field, asdict
from enum import Enum
import uuid
import threading
from collections import defaultdict, deque

from auto_resume_engine import (
    AutoResumeEngine, InterruptionContext, InterruptionType, 
    ResumeStrategy, ResumeConfiguration
)
from checkpoint_system import CheckpointManager, CheckpointType, ProcessingState

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ProcessPriority(Enum):
    """Priority levels for process resumption"""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"

class DependencyType(Enum):
    """Types of process dependencies"""
    SEQUENTIAL = "sequential"      # Must complete before dependent can start
    PARALLEL = "parallel"          # Can run concurrently
    RESOURCE = "resource"          # Shares resources, may need coordination
    DATA = "data"                  # Depends on data output from another process

@dataclass
class ProcessDependency:
    """Represents a dependency between processes"""
    dependent_process: str
    dependency_process: str
    dependency_type: DependencyType
    required_completion_percent: float = 100.0
    timeout_minutes: int = 60
    
    def __post_init__(self):
        if isinstance(self.dependency_type, str):
            self.dependency_type = DependencyType(self.dependency_type)

@dataclass
class ProcessMetrics:
    """Metrics for a resumable process"""
    process_id: str
    total_resumes: int = 0
    successful_resumes: int = 0
    failed_resumes: int = 0
    avg_resume_time_seconds: float = 0.0
    last_resume_time: Optional[datetime] = None
    interruption_frequency: float = 0.0  # interruptions per hour
    reliability_score: float = 100.0
    
    def update_resume_success(self, resume_time_seconds: float):
        """Update metrics for successful resume"""
        self.total_resumes += 1
        self.successful_resumes += 1
        self.last_resume_time = datetime.utcnow()
        
        # Update average resume time
        if self.avg_resume_time_seconds == 0:
            self.avg_resume_time_seconds = resume_time_seconds
        else:
            self.avg_resume_time_seconds = (
                (self.avg_resume_time_seconds * (self.successful_resumes - 1) + resume_time_seconds) 
                / self.successful_resumes
            )
        
        # Update reliability score
        self.reliability_score = (self.successful_resumes / self.total_resumes) * 100
    
    def update_resume_failure(self):
        """Update metrics for failed resume"""
        self.total_resumes += 1
        self.failed_resumes += 1
        self.reliability_score = (self.successful_resumes / self.total_resumes) * 100

class ResumeOrchestrator:
    """Orchestrates resumption of multiple processes with dependencies and priorities"""
    
    def __init__(self, checkpoint_manager: CheckpointManager, 
                 resume_config: ResumeConfiguration = None):
        self.checkpoint_manager = checkpoint_manager
        self.resume_config = resume_config or ResumeConfiguration()
        self.resume_engine = AutoResumeEngine(checkpoint_manager, self.resume_config)
        
        # Process management
        self.registered_processes: Dict[str, Dict[str, Any]] = {}
        self.process_dependencies: List[ProcessDependency] = []
        self.process_priorities: Dict[str, ProcessPriority] = {}
        self.process_metrics: Dict[str, ProcessMetrics] = {}
        
        # Resume coordination
        self.resume_queue = deque()
        self.active_resumes: Set[str] = set()
        self.resume_locks: Dict[str, asyncio.Lock] = {}
        self.dependency_graph: Dict[str, List[str]] = defaultdict(list)
        
        # Monitoring
        self.orchestrator_active = False
        self.orchestrator_thread = None
        self.resume_history: List[Dict[str, Any]] = []
        
        # Resource management
        self.max_concurrent_resumes = 5
        self.resource_pools: Dict[str, int] = {
            "cpu_intensive": 2,
            "memory_intensive": 2,
            "io_intensive": 3
        }
        self.resource_usage: Dict[str, int] = defaultdict(int)
    
    def start(self):
        """Start the resume orchestrator"""
        
        self.resume_engine.start()
        self.orchestrator_active = True
        
        # Start orchestration thread
        self.orchestrator_thread = threading.Thread(target=self._orchestration_loop, daemon=True)
        self.orchestrator_thread.start()
        
        logger.info("Resume orchestrator started")
    
    def stop(self):
        """Stop the resume orchestrator"""
        
        self.orchestrator_active = False
        if self.orchestrator_thread:
            self.orchestrator_thread.join(timeout=10)
        
        self.resume_engine.stop()
        logger.info("Resume orchestrator stopped")
    
    def register_resumable_process(self, process_id: str, task_id: str,
                                 resume_handler: Callable, 
                                 priority: ProcessPriority = ProcessPriority.MEDIUM,
                                 resource_requirements: Dict[str, int] = None,
                                 metadata: Dict[str, Any] = None):
        """Register a process for orchestrated resumption"""
        
        # Register with resume engine
        self.resume_engine.register_process(process_id, task_id, resume_handler, metadata)
        
        # Store orchestrator-specific information
        self.registered_processes[process_id] = {
            "task_id": task_id,
            "resume_handler": resume_handler,
            "priority": priority,
            "resource_requirements": resource_requirements or {},
            "metadata": metadata or {},
            "registered_at": datetime.utcnow()
        }
        
        self.process_priorities[process_id] = priority
        self.process_metrics[process_id] = ProcessMetrics(process_id=process_id)
        self.resume_locks[process_id] = asyncio.Lock()
        
        logger.info(f"Registered resumable process {process_id} with priority {priority.value}")
    
    def add_process_dependency(self, dependent_process: str, dependency_process: str,
                             dependency_type: DependencyType = DependencyType.SEQUENTIAL,
                             required_completion_percent: float = 100.0,
                             timeout_minutes: int = 60):
        """Add a dependency between processes"""
        
        dependency = ProcessDependency(
            dependent_process=dependent_process,
            dependency_process=dependency_process,
            dependency_type=dependency_type,
            required_completion_percent=required_completion_percent,
            timeout_minutes=timeout_minutes
        )
        
        self.process_dependencies.append(dependency)
        self.dependency_graph[dependency_process].append(dependent_process)
        
        logger.info(f"Added dependency: {dependent_process} depends on {dependency_process}")
    
    async def request_resume(self, process_id: str, 
                           interruption_context: InterruptionContext = None,
                           force_immediate: bool = False) -> bool:
        """Request resumption of a specific process"""
        
        if process_id not in self.registered_processes:
            logger.error(f"Process {process_id} not registered for resumption")
            return False
        
        # Check if already in queue or active
        if process_id in self.active_resumes:
            logger.info(f"Process {process_id} already being resumed")
            return True
        
        # Add to resume queue with priority
        priority = self.process_priorities.get(process_id, ProcessPriority.MEDIUM)
        
        resume_request = {
            "process_id": process_id,
            "priority": priority,
            "interruption_context": interruption_context,
            "requested_at": datetime.utcnow(),
            "force_immediate": force_immediate
        }
        
        # Insert based on priority
        if force_immediate or priority == ProcessPriority.CRITICAL:
            self.resume_queue.appendleft(resume_request)
        else:
            # Find insertion point based on priority
            inserted = False
            for i, existing_request in enumerate(self.resume_queue):
                if self._compare_priority(priority, existing_request["priority"]) > 0:
                    self.resume_queue.insert(i, resume_request)
                    inserted = True
                    break
            
            if not inserted:
                self.resume_queue.append(resume_request)
        
        logger.info(f"Queued resume request for {process_id} with priority {priority.value}")
        return True
    
    async def resume_process_with_dependencies(self, process_id: str,
                                             interruption_context: InterruptionContext = None) -> bool:
        """Resume a process considering its dependencies"""
        
        # Check dependencies
        if not await self._check_dependencies_satisfied(process_id):
            logger.info(f"Dependencies not satisfied for {process_id}, waiting...")
            return False
        
        # Check resource availability
        if not self._check_resource_availability(process_id):
            logger.info(f"Resources not available for {process_id}, waiting...")
            return False
        
        # Acquire resources
        self._acquire_resources(process_id)
        
        try:
            # Mark as active
            self.active_resumes.add(process_id)
            
            # Record resume start time
            resume_start_time = time.time()
            
            # Perform actual resume
            success = await self.resume_engine.resume_process(process_id, interruption_context)
            
            # Calculate resume time
            resume_time = time.time() - resume_start_time
            
            # Update metrics
            if success:
                self.process_metrics[process_id].update_resume_success(resume_time)
                logger.info(f"Successfully resumed {process_id} in {resume_time:.2f} seconds")
            else:
                self.process_metrics[process_id].update_resume_failure()
                logger.error(f"Failed to resume {process_id}")
            
            # Record in history
            self.resume_history.append({
                "process_id": process_id,
                "success": success,
                "resume_time_seconds": resume_time,
                "timestamp": datetime.utcnow().isoformat(),
                "interruption_type": interruption_context.interruption_type.value if interruption_context else None
            })
            
            # Keep only last 1000 entries
            if len(self.resume_history) > 1000:
                self.resume_history = self.resume_history[-1000:]
            
            return success
            
        finally:
            # Release resources and cleanup
            self._release_resources(process_id)
            self.active_resumes.discard(process_id)
    
    def _orchestration_loop(self):
        """Main orchestration loop running in background thread"""
        
        while self.orchestrator_active:
            try:
                # Process resume queue
                if self.resume_queue and len(self.active_resumes) < self.max_concurrent_resumes:
                    # Get next resume request
                    resume_request = self.resume_queue.popleft()
                    process_id = resume_request["process_id"]
                    
                    # Schedule resume in async context
                    asyncio.run_coroutine_threadsafe(
                        self.resume_process_with_dependencies(
                            process_id, 
                            resume_request.get("interruption_context")
                        ),
                        asyncio.get_event_loop()
                    )
                
                # Check for dependency updates
                self._check_dependency_updates()
                
                # Clean up old resume history
                self._cleanup_old_data()
                
                # Sleep before next iteration
                time.sleep(1)
                
            except Exception as e:
                logger.error(f"Orchestration loop error: {e}")
                time.sleep(5)
    
    async def _check_dependencies_satisfied(self, process_id: str) -> bool:
        """Check if all dependencies for a process are satisfied"""
        
        for dependency in self.process_dependencies:
            if dependency.dependent_process == process_id:
                # Check if dependency process has completed required percentage
                dependency_state = self.checkpoint_manager.active_processes.get(
                    dependency.dependency_process
                )
                
                if dependency_state:
                    completion_percent = dependency_state.progress_percentage
                    if completion_percent < dependency.required_completion_percent:
                        logger.debug(f"Dependency {dependency.dependency_process} only {completion_percent:.1f}% complete")
                        return False
                else:
                    # Check if dependency has completed checkpoints
                    checkpoints = self.checkpoint_manager.storage.list_checkpoints(
                        process_id=dependency.dependency_process
                    )
                    
                    completion_checkpoints = [
                        c for c in checkpoints 
                        if "completion" in c.description.lower()
                    ]
                    
                    if not completion_checkpoints:
                        logger.debug(f"Dependency {dependency.dependency_process} not completed")
                        return False
        
        return True
    
    def _check_resource_availability(self, process_id: str) -> bool:
        """Check if required resources are available"""
        
        if process_id not in self.registered_processes:
            return True
        
        requirements = self.registered_processes[process_id].get("resource_requirements", {})
        
        for resource_type, required_amount in requirements.items():
            available = self.resource_pools.get(resource_type, 0) - self.resource_usage.get(resource_type, 0)
            if available < required_amount:
                return False
        
        return True
    
    def _acquire_resources(self, process_id: str):
        """Acquire resources for a process"""
        
        if process_id not in self.registered_processes:
            return
        
        requirements = self.registered_processes[process_id].get("resource_requirements", {})
        
        for resource_type, required_amount in requirements.items():
            self.resource_usage[resource_type] += required_amount
        
        logger.debug(f"Acquired resources for {process_id}: {requirements}")
    
    def _release_resources(self, process_id: str):
        """Release resources for a process"""
        
        if process_id not in self.registered_processes:
            return
        
        requirements = self.registered_processes[process_id].get("resource_requirements", {})
        
        for resource_type, required_amount in requirements.items():
            self.resource_usage[resource_type] = max(0, self.resource_usage[resource_type] - required_amount)
        
        logger.debug(f"Released resources for {process_id}: {requirements}")
    
    def _compare_priority(self, priority1: ProcessPriority, priority2: ProcessPriority) -> int:
        """Compare two priorities (returns positive if priority1 > priority2)"""
        
        priority_values = {
            ProcessPriority.CRITICAL: 4,
            ProcessPriority.HIGH: 3,
            ProcessPriority.MEDIUM: 2,
            ProcessPriority.LOW: 1
        }
        
        return priority_values[priority1] - priority_values[priority2]
    
    def _check_dependency_updates(self):
        """Check for dependency updates and trigger resumes if needed"""
        
        # Check if any waiting processes can now be resumed
        for process_id in list(self.registered_processes.keys()):
            if (process_id not in self.active_resumes and 
                process_id not in [req["process_id"] for req in self.resume_queue]):
                
                # Check if this process has unsatisfied dependencies that might now be satisfied
                has_dependencies = any(
                    dep.dependent_process == process_id 
                    for dep in self.process_dependencies
                )
                
                if has_dependencies:
                    # Check if we should queue this process for resume
                    asyncio.run_coroutine_threadsafe(
                        self._maybe_queue_dependent_process(process_id),
                        asyncio.get_event_loop()
                    )
    
    async def _maybe_queue_dependent_process(self, process_id: str):
        """Maybe queue a dependent process if its dependencies are now satisfied"""
        
        if await self._check_dependencies_satisfied(process_id):
            # Check if process needs resumption (has incomplete checkpoints)
            recovered_state = self.checkpoint_manager.recover_process(process_id)
            if recovered_state and recovered_state.progress_percentage < 100:
                await self.request_resume(process_id)
    
    def _cleanup_old_data(self):
        """Clean up old data to prevent memory leaks"""
        
        # Clean up old resume history (keep last 24 hours)
        cutoff_time = datetime.utcnow() - timedelta(hours=24)
        self.resume_history = [
            entry for entry in self.resume_history
            if datetime.fromisoformat(entry["timestamp"]) > cutoff_time
        ]
    
    def get_orchestrator_status(self) -> Dict[str, Any]:
        """Get comprehensive orchestrator status"""
        
        # Calculate overall metrics
        total_processes = len(self.registered_processes)
        active_resumes = len(self.active_resumes)
        queued_resumes = len(self.resume_queue)
        
        # Calculate success rates
        recent_history = [
            entry for entry in self.resume_history
            if datetime.fromisoformat(entry["timestamp"]) > datetime.utcnow() - timedelta(hours=1)
        ]
        
        successful_resumes = sum(1 for entry in recent_history if entry["success"])
        total_recent_resumes = len(recent_history)
        success_rate = (successful_resumes / total_recent_resumes * 100) if total_recent_resumes > 0 else 0
        
        # Resource utilization
        resource_utilization = {}
        for resource_type, pool_size in self.resource_pools.items():
            used = self.resource_usage.get(resource_type, 0)
            utilization = (used / pool_size * 100) if pool_size > 0 else 0
            resource_utilization[resource_type] = {
                "used": used,
                "total": pool_size,
                "utilization_percent": round(utilization, 1)
            }
        
        return {
            "orchestrator_active": self.orchestrator_active,
            "total_registered_processes": total_processes,
            "active_resumes": active_resumes,
            "queued_resumes": queued_resumes,
            "max_concurrent_resumes": self.max_concurrent_resumes,
            "success_rate_percent": round(success_rate, 2),
            "resource_utilization": resource_utilization,
            "process_metrics": {
                pid: asdict(metrics) for pid, metrics in self.process_metrics.items()
            },
            "dependency_count": len(self.process_dependencies),
            "resume_history_count": len(self.resume_history)
        }
    
    def get_process_status(self, process_id: str) -> Optional[Dict[str, Any]]:
        """Get detailed status for a specific process"""
        
        if process_id not in self.registered_processes:
            return None
        
        process_info = self.registered_processes[process_id]
        metrics = self.process_metrics.get(process_id)
        
        # Check current state
        current_state = self.checkpoint_manager.active_processes.get(process_id)
        
        # Find dependencies
        dependencies = [
            dep for dep in self.process_dependencies
            if dep.dependent_process == process_id or dep.dependency_process == process_id
        ]
        
        return {
            "process_id": process_id,
            "task_id": process_info["task_id"],
            "priority": process_info["priority"].value,
            "registered_at": process_info["registered_at"].isoformat(),
            "is_active": process_id in self.active_resumes,
            "is_queued": process_id in [req["process_id"] for req in self.resume_queue],
            "current_progress": current_state.progress_percentage if current_state else None,
            "metrics": asdict(metrics) if metrics else None,
            "dependencies": [asdict(dep) for dep in dependencies],
            "resource_requirements": process_info.get("resource_requirements", {}),
            "metadata": process_info.get("metadata", {})
        }

# Example usage
async def example_orchestrated_resume():
    """Example of using the resume orchestrator"""
    
    from checkpoint_system import CheckpointManager
    
    # Initialize systems
    checkpoint_manager = CheckpointManager()
    orchestrator = ResumeOrchestrator(checkpoint_manager)
    
    # Example resume handlers
    async def data_processor_handler(resume_point: Dict[str, Any], 
                                   interruption_context: InterruptionContext = None):
        print(f"Resuming data processor from step {resume_point['resume_step']}")
        # Simulate data processing resume
        await asyncio.sleep(2)
        return True
    
    async def model_trainer_handler(resume_point: Dict[str, Any], 
                                  interruption_context: InterruptionContext = None):
        print(f"Resuming model trainer from step {resume_point['resume_step']}")
        # Simulate model training resume
        await asyncio.sleep(3)
        return True
    
    try:
        # Start orchestrator
        orchestrator.start()
        
        # Register processes with different priorities and resource requirements
        orchestrator.register_resumable_process(
            process_id="data_processor_001",
            task_id="data_processing",
            resume_handler=data_processor_handler,
            priority=ProcessPriority.HIGH,
            resource_requirements={"cpu_intensive": 1, "io_intensive": 1}
        )
        
        orchestrator.register_resumable_process(
            process_id="model_trainer_001", 
            task_id="model_training",
            resume_handler=model_trainer_handler,
            priority=ProcessPriority.CRITICAL,
            resource_requirements={"cpu_intensive": 2, "memory_intensive": 1}
        )
        
        # Add dependency (model trainer depends on data processor)
        orchestrator.add_process_dependency(
            dependent_process="model_trainer_001",
            dependency_process="data_processor_001",
            dependency_type=DependencyType.SEQUENTIAL,
            required_completion_percent=80.0
        )
        
        # Request resumes
        await orchestrator.request_resume("data_processor_001")
        await orchestrator.request_resume("model_trainer_001")
        
        # Wait for processing
        await asyncio.sleep(10)
        
        # Get status
        status = orchestrator.get_orchestrator_status()
        print(f"Orchestrator status: {json.dumps(status, indent=2, default=str)}")
        
    finally:
        orchestrator.stop()

if __name__ == "__main__":
    asyncio.run(example_orchestrated_resume())
