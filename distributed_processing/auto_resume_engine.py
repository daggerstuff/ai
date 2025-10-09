#!/usr/bin/env python3
"""
Automatic Resume Engine for Pixelated Empathy AI
Provides seamless recovery and resumption of interrupted processing operations
"""

import asyncio
import json
import logging
import os
import signal
import sys
import time
import threading
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable, Union
from dataclasses import dataclass, field
from enum import Enum
import uuid
import psutil

from checkpoint_system import (
    CheckpointManager, CheckpointType, CheckpointStatus, 
    ProcessingState, CheckpointMetadata
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class InterruptionType(Enum):
    """Types of processing interruptions"""
    SYSTEM_SHUTDOWN = "system_shutdown"
    PROCESS_CRASH = "process_crash"
    MEMORY_EXHAUSTION = "memory_exhaustion"
    TIMEOUT = "timeout"
    USER_TERMINATION = "user_termination"
    NETWORK_FAILURE = "network_failure"
    DEPENDENCY_FAILURE = "dependency_failure"
    RESOURCE_EXHAUSTION = "resource_exhaustion"
    UNKNOWN = "unknown"

class ResumeStrategy(Enum):
    """Resume strategies for different scenarios"""
    EXACT_CONTINUATION = "exact_continuation"
    ROLLBACK_AND_RETRY = "rollback_and_retry"
    PARTIAL_RESTART = "partial_restart"
    FULL_RESTART = "full_restart"
    SKIP_AND_CONTINUE = "skip_and_continue"
    MANUAL_INTERVENTION = "manual_intervention"

@dataclass
class InterruptionContext:
    """Context information about an interruption"""
    interruption_id: str
    process_id: str
    task_id: str
    interruption_type: InterruptionType
    timestamp: datetime
    last_checkpoint_id: Optional[str] = None
    error_details: Optional[str] = None
    system_state: Dict[str, Any] = field(default_factory=dict)
    recovery_attempts: int = 0
    max_recovery_attempts: int = 3
    
    def __post_init__(self):
        if isinstance(self.timestamp, str):
            self.timestamp = datetime.fromisoformat(self.timestamp)
        if isinstance(self.interruption_type, str):
            self.interruption_type = InterruptionType(self.interruption_type)

@dataclass
class ResumeConfiguration:
    """Configuration for resume behavior"""
    auto_resume_enabled: bool = True
    max_resume_attempts: int = 3
    resume_delay_seconds: int = 5
    rollback_steps: int = 1
    timeout_threshold_minutes: int = 30
    memory_threshold_percent: float = 90.0
    cpu_threshold_percent: float = 95.0
    enable_graceful_shutdown: bool = True
    enable_crash_detection: bool = True
    enable_resource_monitoring: bool = True
    resume_strategies: Dict[InterruptionType, ResumeStrategy] = field(default_factory=dict)
    
    def __post_init__(self):
        # Set default resume strategies if not provided
        if not self.resume_strategies:
            self.resume_strategies = {
                InterruptionType.SYSTEM_SHUTDOWN: ResumeStrategy.EXACT_CONTINUATION,
                InterruptionType.PROCESS_CRASH: ResumeStrategy.ROLLBACK_AND_RETRY,
                InterruptionType.MEMORY_EXHAUSTION: ResumeStrategy.PARTIAL_RESTART,
                InterruptionType.TIMEOUT: ResumeStrategy.ROLLBACK_AND_RETRY,
                InterruptionType.USER_TERMINATION: ResumeStrategy.EXACT_CONTINUATION,
                InterruptionType.NETWORK_FAILURE: ResumeStrategy.ROLLBACK_AND_RETRY,
                InterruptionType.DEPENDENCY_FAILURE: ResumeStrategy.ROLLBACK_AND_RETRY,
                InterruptionType.RESOURCE_EXHAUSTION: ResumeStrategy.PARTIAL_RESTART,
                InterruptionType.UNKNOWN: ResumeStrategy.ROLLBACK_AND_RETRY
            }

class ProcessingInterruptionDetector:
    """Detects various types of processing interruptions"""
    
    def __init__(self, config: ResumeConfiguration):
        self.config = config
        self.monitoring_active = False
        self.monitor_thread = None
        self.interruption_callbacks: List[Callable] = []
        self.last_heartbeat = {}
        self.resource_history = []
    
    def start_monitoring(self):
        """Start interruption monitoring"""
        
        if self.monitoring_active:
            return
        
        self.monitoring_active = True
        
        # Set up signal handlers for graceful shutdown
        if self.config.enable_graceful_shutdown:
            signal.signal(signal.SIGTERM, self._handle_shutdown_signal)
            signal.signal(signal.SIGINT, self._handle_shutdown_signal)
        
        # Start resource monitoring thread
        if self.config.enable_resource_monitoring:
            self.monitor_thread = threading.Thread(target=self._monitor_resources, daemon=True)
            self.monitor_thread.start()
        
        logger.info("Started interruption monitoring")
    
    def stop_monitoring(self):
        """Stop interruption monitoring"""
        
        self.monitoring_active = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=5)
        
        logger.info("Stopped interruption monitoring")
    
    def register_process_heartbeat(self, process_id: str):
        """Register a heartbeat for a process"""
        
        self.last_heartbeat[process_id] = datetime.utcnow()
    
    def add_interruption_callback(self, callback: Callable):
        """Add callback for interruption detection"""
        
        self.interruption_callbacks.append(callback)
    
    def _handle_shutdown_signal(self, signum, frame):
        """Handle shutdown signals"""
        
        logger.info(f"Received shutdown signal {signum}")
        
        # Trigger interruption callbacks
        interruption = InterruptionContext(
            interruption_id=str(uuid.uuid4()),
            process_id="system",
            task_id="shutdown",
            interruption_type=InterruptionType.SYSTEM_SHUTDOWN,
            timestamp=datetime.utcnow(),
            error_details=f"Received signal {signum}"
        )
        
        self._trigger_interruption_callbacks(interruption)
    
    def _monitor_resources(self):
        """Monitor system resources for potential issues"""
        
        while self.monitoring_active:
            try:
                # Get current resource usage
                cpu_percent = psutil.cpu_percent(interval=1)
                memory = psutil.virtual_memory()
                
                # Store resource history
                resource_data = {
                    "timestamp": datetime.utcnow(),
                    "cpu_percent": cpu_percent,
                    "memory_percent": memory.percent,
                    "memory_available": memory.available
                }
                
                self.resource_history.append(resource_data)
                
                # Keep only last hour of data
                cutoff_time = datetime.utcnow() - timedelta(hours=1)
                self.resource_history = [
                    r for r in self.resource_history 
                    if r["timestamp"] > cutoff_time
                ]
                
                # Check for resource exhaustion
                if memory.percent > self.config.memory_threshold_percent:
                    logger.warning(f"High memory usage detected: {memory.percent:.1f}%")
                    
                    interruption = InterruptionContext(
                        interruption_id=str(uuid.uuid4()),
                        process_id="system",
                        task_id="resource_monitoring",
                        interruption_type=InterruptionType.MEMORY_EXHAUSTION,
                        timestamp=datetime.utcnow(),
                        error_details=f"Memory usage: {memory.percent:.1f}%",
                        system_state=resource_data
                    )
                    
                    self._trigger_interruption_callbacks(interruption)
                
                if cpu_percent > self.config.cpu_threshold_percent:
                    logger.warning(f"High CPU usage detected: {cpu_percent:.1f}%")
                
                # Check for process timeouts
                self._check_process_timeouts()
                
                # Sleep until next check
                time.sleep(30)  # Check every 30 seconds
                
            except Exception as e:
                logger.error(f"Resource monitoring error: {e}")
                time.sleep(60)
    
    def _check_process_timeouts(self):
        """Check for process timeouts based on heartbeats"""
        
        timeout_threshold = timedelta(minutes=self.config.timeout_threshold_minutes)
        current_time = datetime.utcnow()
        
        for process_id, last_heartbeat in list(self.last_heartbeat.items()):
            if current_time - last_heartbeat > timeout_threshold:
                logger.warning(f"Process timeout detected: {process_id}")
                
                interruption = InterruptionContext(
                    interruption_id=str(uuid.uuid4()),
                    process_id=process_id,
                    task_id="timeout_detection",
                    interruption_type=InterruptionType.TIMEOUT,
                    timestamp=current_time,
                    error_details=f"No heartbeat for {timeout_threshold}"
                )
                
                self._trigger_interruption_callbacks(interruption)
                
                # Remove from tracking
                del self.last_heartbeat[process_id]
    
    def _trigger_interruption_callbacks(self, interruption: InterruptionContext):
        """Trigger all registered interruption callbacks"""
        
        for callback in self.interruption_callbacks:
            try:
                callback(interruption)
            except Exception as e:
                logger.error(f"Interruption callback error: {e}")

class AutoResumeEngine:
    """Main engine for automatic resume functionality"""
    
    def __init__(self, checkpoint_manager: CheckpointManager, 
                 config: ResumeConfiguration = None):
        self.checkpoint_manager = checkpoint_manager
        self.config = config or ResumeConfiguration()
        self.detector = ProcessingInterruptionDetector(self.config)
        self.resume_handlers: Dict[str, Callable] = {}
        self.active_processes: Dict[str, Dict[str, Any]] = {}
        self.interruption_history: List[InterruptionContext] = []
        
        # Register for interruption detection
        self.detector.add_interruption_callback(self._handle_interruption)
    
    def start(self):
        """Start the auto-resume engine"""
        
        self.detector.start_monitoring()
        
        # Check for existing processes that need resumption
        asyncio.create_task(self._check_for_resumable_processes())
        
        logger.info("Auto-resume engine started")
    
    def stop(self):
        """Stop the auto-resume engine"""
        
        self.detector.stop_monitoring()
        logger.info("Auto-resume engine stopped")
    
    def register_process(self, process_id: str, task_id: str, 
                        resume_handler: Callable, metadata: Dict[str, Any] = None):
        """Register a process for automatic resumption"""
        
        self.active_processes[process_id] = {
            "task_id": task_id,
            "resume_handler": resume_handler,
            "metadata": metadata or {},
            "registered_at": datetime.utcnow(),
            "last_heartbeat": datetime.utcnow()
        }
        
        self.resume_handlers[process_id] = resume_handler
        
        logger.info(f"Registered process {process_id} for auto-resume")
    
    def unregister_process(self, process_id: str):
        """Unregister a process from automatic resumption"""
        
        if process_id in self.active_processes:
            del self.active_processes[process_id]
        
        if process_id in self.resume_handlers:
            del self.resume_handlers[process_id]
        
        logger.info(f"Unregistered process {process_id} from auto-resume")
    
    def heartbeat(self, process_id: str, metadata: Dict[str, Any] = None):
        """Send heartbeat for a process"""
        
        if process_id in self.active_processes:
            self.active_processes[process_id]["last_heartbeat"] = datetime.utcnow()
            if metadata:
                self.active_processes[process_id]["metadata"].update(metadata)
        
        # Also register with detector
        self.detector.register_process_heartbeat(process_id)
    
    async def resume_process(self, process_id: str, 
                           interruption_context: InterruptionContext = None) -> bool:
        """Resume a specific process"""
        
        try:
            # Get the latest checkpoint for the process
            recovered_state = self.checkpoint_manager.recover_process(process_id)
            
            if not recovered_state:
                logger.error(f"No checkpoint found for process {process_id}")
                return False
            
            # Determine resume strategy
            strategy = self._determine_resume_strategy(interruption_context)
            
            # Apply resume strategy
            resume_point = self._apply_resume_strategy(recovered_state, strategy, interruption_context)
            
            # Get resume handler
            if process_id not in self.resume_handlers:
                logger.error(f"No resume handler registered for process {process_id}")
                return False
            
            resume_handler = self.resume_handlers[process_id]
            
            # Execute resume
            logger.info(f"Resuming process {process_id} using strategy {strategy.value}")
            
            success = await resume_handler(resume_point, interruption_context)
            
            if success:
                logger.info(f"Successfully resumed process {process_id}")
                
                # Update process registration
                if process_id in self.active_processes:
                    self.active_processes[process_id]["last_resume"] = datetime.utcnow()
                    self.active_processes[process_id]["resume_count"] = \
                        self.active_processes[process_id].get("resume_count", 0) + 1
                
                return True
            else:
                logger.error(f"Failed to resume process {process_id}")
                return False
                
        except Exception as e:
            logger.error(f"Error resuming process {process_id}: {e}")
            return False
    
    def _handle_interruption(self, interruption: InterruptionContext):
        """Handle detected interruption"""
        
        logger.info(f"Handling interruption: {interruption.interruption_type.value} for {interruption.process_id}")
        
        # Store interruption in history
        self.interruption_history.append(interruption)
        
        # Keep only last 100 interruptions
        if len(self.interruption_history) > 100:
            self.interruption_history = self.interruption_history[-100:]
        
        # If auto-resume is enabled, attempt resume
        if self.config.auto_resume_enabled:
            # Schedule resume after delay
            asyncio.create_task(self._schedule_resume(interruption))
    
    async def _schedule_resume(self, interruption: InterruptionContext):
        """Schedule resume after delay"""
        
        # Wait for resume delay
        await asyncio.sleep(self.config.resume_delay_seconds)
        
        # Check if we should attempt resume
        if interruption.recovery_attempts >= self.config.max_resume_attempts:
            logger.error(f"Max resume attempts reached for {interruption.process_id}")
            return
        
        # Increment recovery attempts
        interruption.recovery_attempts += 1
        
        # Attempt resume
        success = await self.resume_process(interruption.process_id, interruption)
        
        if not success and interruption.recovery_attempts < self.config.max_resume_attempts:
            # Schedule another attempt with exponential backoff
            delay = self.config.resume_delay_seconds * (2 ** interruption.recovery_attempts)
            logger.info(f"Scheduling retry for {interruption.process_id} in {delay} seconds")
            
            await asyncio.sleep(delay)
            await self._schedule_resume(interruption)
    
    def _determine_resume_strategy(self, interruption_context: InterruptionContext = None) -> ResumeStrategy:
        """Determine the appropriate resume strategy"""
        
        if not interruption_context:
            return ResumeStrategy.EXACT_CONTINUATION
        
        # Get strategy from configuration
        strategy = self.config.resume_strategies.get(
            interruption_context.interruption_type,
            ResumeStrategy.ROLLBACK_AND_RETRY
        )
        
        # Adjust strategy based on recovery attempts
        if interruption_context.recovery_attempts > 1:
            if strategy == ResumeStrategy.EXACT_CONTINUATION:
                strategy = ResumeStrategy.ROLLBACK_AND_RETRY
            elif strategy == ResumeStrategy.ROLLBACK_AND_RETRY:
                strategy = ResumeStrategy.PARTIAL_RESTART
        
        return strategy
    
    def _apply_resume_strategy(self, state: ProcessingState, strategy: ResumeStrategy,
                             interruption_context: InterruptionContext = None) -> Dict[str, Any]:
        """Apply resume strategy to determine resume point"""
        
        resume_point = {
            "process_id": state.process_id,
            "task_id": state.task_id,
            "strategy": strategy,
            "original_state": state
        }
        
        if strategy == ResumeStrategy.EXACT_CONTINUATION:
            # Resume from exact point
            resume_point.update({
                "resume_step": state.completed_steps,
                "current_step": state.current_step,
                "processed_items": state.processed_items.copy(),
                "failed_items": state.failed_items.copy()
            })
        
        elif strategy == ResumeStrategy.ROLLBACK_AND_RETRY:
            # Rollback a few steps and retry
            rollback_steps = min(self.config.rollback_steps, state.completed_steps)
            resume_step = max(0, state.completed_steps - rollback_steps)
            
            resume_point.update({
                "resume_step": resume_step,
                "current_step": f"Rolled back {rollback_steps} steps",
                "processed_items": state.processed_items[:-rollback_steps] if rollback_steps > 0 else state.processed_items.copy(),
                "failed_items": state.failed_items.copy()
            })
        
        elif strategy == ResumeStrategy.PARTIAL_RESTART:
            # Restart from a safe checkpoint (e.g., last 10% of progress)
            safe_point = max(0, int(state.completed_steps * 0.9))
            
            resume_point.update({
                "resume_step": safe_point,
                "current_step": f"Partial restart from step {safe_point}",
                "processed_items": state.processed_items[:safe_point] if safe_point > 0 else [],
                "failed_items": []  # Clear failed items for fresh start
            })
        
        elif strategy == ResumeStrategy.FULL_RESTART:
            # Complete restart
            resume_point.update({
                "resume_step": 0,
                "current_step": "Full restart",
                "processed_items": [],
                "failed_items": []
            })
        
        elif strategy == ResumeStrategy.SKIP_AND_CONTINUE:
            # Skip problematic section and continue
            skip_steps = 1
            resume_step = min(state.total_steps, state.completed_steps + skip_steps)
            
            resume_point.update({
                "resume_step": resume_step,
                "current_step": f"Skipped {skip_steps} steps and continuing",
                "processed_items": state.processed_items.copy(),
                "failed_items": state.failed_items.copy()
            })
        
        else:  # MANUAL_INTERVENTION
            resume_point.update({
                "resume_step": state.completed_steps,
                "current_step": "Awaiting manual intervention",
                "processed_items": state.processed_items.copy(),
                "failed_items": state.failed_items.copy(),
                "requires_manual_intervention": True
            })
        
        return resume_point
    
    async def _check_for_resumable_processes(self):
        """Check for processes that can be resumed on startup"""
        
        # Get all active checkpoints
        active_checkpoints = self.checkpoint_manager.storage.list_checkpoints(
            status=CheckpointStatus.ACTIVE
        )
        
        # Group by process_id
        process_checkpoints = {}
        for checkpoint in active_checkpoints:
            if checkpoint.process_id not in process_checkpoints:
                process_checkpoints[checkpoint.process_id] = []
            process_checkpoints[checkpoint.process_id].append(checkpoint)
        
        # Check each process for resumability
        for process_id, checkpoints in process_checkpoints.items():
            # Find latest processing state checkpoint
            processing_checkpoints = [
                c for c in checkpoints 
                if c.checkpoint_type == CheckpointType.PROCESSING_STATE
            ]
            
            if processing_checkpoints:
                # Sort by creation time
                processing_checkpoints.sort(key=lambda x: x.created_at, reverse=True)
                latest_checkpoint = processing_checkpoints[0]
                
                # Check if process was interrupted (no completion checkpoint)
                completion_checkpoints = [
                    c for c in checkpoints 
                    if "completion" in c.description.lower()
                ]
                
                if not completion_checkpoints:
                    logger.info(f"Found potentially interrupted process: {process_id}")
                    
                    # Create interruption context for startup resume
                    interruption = InterruptionContext(
                        interruption_id=str(uuid.uuid4()),
                        process_id=process_id,
                        task_id=latest_checkpoint.task_id,
                        interruption_type=InterruptionType.SYSTEM_SHUTDOWN,
                        timestamp=datetime.utcnow(),
                        last_checkpoint_id=latest_checkpoint.checkpoint_id,
                        error_details="Process found incomplete on startup"
                    )
                    
                    # Only resume if we have a handler registered
                    if process_id in self.resume_handlers:
                        await self.resume_process(process_id, interruption)
                    else:
                        logger.warning(f"No resume handler for process {process_id}, skipping auto-resume")
    
    def get_resume_statistics(self) -> Dict[str, Any]:
        """Get statistics about resume operations"""
        
        total_interruptions = len(self.interruption_history)
        
        # Count by interruption type
        interruption_counts = {}
        for interruption in self.interruption_history:
            int_type = interruption.interruption_type.value
            interruption_counts[int_type] = interruption_counts.get(int_type, 0) + 1
        
        # Count successful resumes
        successful_resumes = sum(
            1 for process_info in self.active_processes.values()
            if process_info.get("resume_count", 0) > 0
        )
        
        # Calculate success rate
        total_resume_attempts = sum(
            interruption.recovery_attempts for interruption in self.interruption_history
        )
        
        success_rate = (successful_resumes / total_resume_attempts * 100) if total_resume_attempts > 0 else 0
        
        return {
            "total_interruptions": total_interruptions,
            "interruption_types": interruption_counts,
            "successful_resumes": successful_resumes,
            "total_resume_attempts": total_resume_attempts,
            "success_rate_percent": round(success_rate, 2),
            "active_processes": len(self.active_processes),
            "registered_handlers": len(self.resume_handlers)
        }

# Example usage and testing
async def example_auto_resume():
    """Example of using the auto-resume engine"""
    
    from checkpoint_system import CheckpointManager
    
    # Initialize systems
    checkpoint_manager = CheckpointManager()
    
    config = ResumeConfiguration(
        auto_resume_enabled=True,
        max_resume_attempts=3,
        resume_delay_seconds=2
    )
    
    resume_engine = AutoResumeEngine(checkpoint_manager, config)
    
    # Example resume handler
    async def example_resume_handler(resume_point: Dict[str, Any], 
                                   interruption_context: InterruptionContext = None):
        """Example resume handler for a processing operation"""
        
        print(f"Resuming process {resume_point['process_id']} from step {resume_point['resume_step']}")
        print(f"Strategy: {resume_point['strategy'].value}")
        
        if interruption_context:
            print(f"Interruption type: {interruption_context.interruption_type.value}")
        
        # Simulate resuming processing
        original_state = resume_point["original_state"]
        resume_step = resume_point["resume_step"]
        
        # Continue processing from resume point
        for step in range(resume_step, original_state.total_steps + 1):
            checkpoint_manager.update_process_progress(
                process_id=original_state.process_id,
                completed_steps=step,
                current_step=f"Resumed step {step}",
                metadata={"resumed": True, "resume_step": resume_step}
            )
            
            # Send heartbeat
            resume_engine.heartbeat(original_state.process_id, {"current_step": step})
            
            await asyncio.sleep(0.1)  # Simulate work
        
        # Complete process
        checkpoint_manager.complete_process(
            original_state.process_id,
            {"resumed_successfully": True}
        )
        
        return True
    
    try:
        # Start the resume engine
        resume_engine.start()
        
        # Register a process
        process_id = "example_resumable_process"
        task_id = "example_task"
        
        # Register process for auto-resume
        resume_engine.register_process(
            process_id=process_id,
            task_id=task_id,
            resume_handler=example_resume_handler,
            metadata={"description": "Example resumable process"}
        )
        
        # Create a process that will be "interrupted"
        state = checkpoint_manager.register_process(
            process_id=process_id,
            task_id=task_id,
            total_steps=20,
            description="Example process for resume testing"
        )
        
        # Simulate some progress
        for step in range(0, 10):
            checkpoint_manager.update_process_progress(
                process_id=process_id,
                completed_steps=step,
                current_step=f"Step {step}"
            )
            
            resume_engine.heartbeat(process_id)
            await asyncio.sleep(0.1)
        
        # Simulate interruption
        print("Simulating process interruption...")
        
        interruption = InterruptionContext(
            interruption_id=str(uuid.uuid4()),
            process_id=process_id,
            task_id=task_id,
            interruption_type=InterruptionType.PROCESS_CRASH,
            timestamp=datetime.utcnow(),
            error_details="Simulated crash for testing"
        )
        
        # Trigger interruption handling
        resume_engine._handle_interruption(interruption)
        
        # Wait for resume to complete
        await asyncio.sleep(5)
        
        # Get statistics
        stats = resume_engine.get_resume_statistics()
        print(f"Resume statistics: {json.dumps(stats, indent=2)}")
        
    finally:
        resume_engine.stop()
        checkpoint_manager.stop_background_tasks()

if __name__ == "__main__":
    asyncio.run(example_auto_resume())
