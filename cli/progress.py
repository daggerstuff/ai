"""
Progress tracking and indicators for CLI operations.

This module provides real-time progress indicators, status updates, and progress
tracking for long-running operations like pipeline execution, file processing,
and data transformations.
"""

import time
import threading
import sys
from typing import Optional, Dict, Any, Callable
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum

from rich.console import Console
from rich.progress import (
    Progress, SpinnerColumn, BarColumn, TextColumn, 
    TimeElapsedColumn, TimeRemainingColumn, MofNCompleteColumn
)
from rich.table import Table
from rich.panel import Panel
from rich.live import Live
from rich.layout import Layout
from rich.text import Text


class ProgressStatus(Enum):
    """Progress status enumeration"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class ProgressStep:
    """Individual progress step information"""
    name: str
    description: str
    status: ProgressStatus = ProgressStatus.PENDING
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ProgressReport:
    """Complete progress report for an operation"""
    operation_id: str
    operation_name: str
    total_steps: int
    completed_steps: int = 0
    failed_steps: int = 0
    start_time: datetime = field(default_factory=datetime.now)
    end_time: Optional[datetime] = None
    steps: list[ProgressStep] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


class ProgressTracker:
    """Real-time progress tracking with visual indicators"""
    
    def __init__(self, config: Optional[Any] = None):
        self.config = config
        self.console = Console()
        self._current_progress: Optional[Progress] = None
        self._current_task_id: Optional[str] = None
        self._progress_reports: Dict[str, ProgressReport] = {}
        self._lock = threading.Lock()
        
    def create_progress_bar(
        self,
        total: int,
        description: str = "Processing",
        transient: bool = False
    ) -> tuple[Progress, str]:
        """Create a new progress bar"""
        progress = Progress(
            SpinnerColumn(),
            TextColumn("[bold blue]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            MofNCompleteColumn(),
            TimeElapsedColumn(),
            TimeRemainingColumn(),
            console=self.console,
            transient=transient
        )
        
        task_id = progress.add_task(description, total=total)
        return progress, task_id
    
    def track_operation(
        self,
        operation_name: str,
        total_steps: int,
        operation_id: Optional[str] = None
    ) -> str:
        """Start tracking a new operation"""
        if operation_id is None:
            operation_id = f"{operation_name}_{int(time.time())}"
            
        report = ProgressReport(
            operation_id=operation_id,
            operation_name=operation_name,
            total_steps=total_steps
        )
        
        with self._lock:
            self._progress_reports[operation_id] = report
            
        return operation_id
    
    def update_step(
        self,
        operation_id: str,
        step_name: str,
        status: ProgressStatus,
        description: Optional[str] = None,
        error: Optional[str] = None
    ):
        """Update the status of a specific step"""
        with self._lock:
            if operation_id not in self._progress_reports:
                return
                
            report = self._progress_reports[operation_id]
            
            # Find or create step
            step = next((s for s in report.steps if s.name == step_name), None)
            if step is None:
                step = ProgressStep(
                    name=step_name,
                    description=description or step_name
                )
                report.steps.append(step)
            
            # Update step status
            step.status = status
            if status == ProgressStatus.RUNNING and step.start_time is None:
                step.start_time = datetime.now()
            elif status in [ProgressStatus.COMPLETED, ProgressStatus.FAILED, ProgressStatus.CANCELLED]:
                step.end_time = datetime.now()
                if status == ProgressStatus.COMPLETED:
                    report.completed_steps += 1
                elif status == ProgressStatus.FAILED:
                    report.failed_steps += 1
                if error:
                    step.error = error
    
    def complete_operation(self, operation_id: str, success: bool = True):
        """Mark an operation as complete"""
        with self._lock:
            if operation_id not in self._progress_reports:
                return
                
            report = self._progress_reports[operation_id]
            report.end_time = datetime.now()
    
    def get_progress_report(self, operation_id: str) -> Optional[ProgressReport]:
        """Get progress report for an operation"""
        with self._lock:
            return self._progress_reports.get(operation_id)
    
    def display_progress_report(self, operation_id: str):
        """Display a formatted progress report"""
        report = self.get_progress_report(operation_id)
        if not report:
            self.console.print(f"[red]No progress report found for operation: {operation_id}[/red]")
            return
        
        # Create summary table
        table = Table(title=f"Progress Report: {report.operation_name}")
        table.add_column("Step", style="cyan")
        table.add_column("Status", style="magenta")
        table.add_column("Duration", style="green")
        table.add_column("Description", style="white")
        
        for step in report.steps:
            status_color = {
                ProgressStatus.PENDING: "yellow",
                ProgressStatus.RUNNING: "blue",
                ProgressStatus.COMPLETED: "green",
                ProgressStatus.FAILED: "red",
                ProgressStatus.CANCELLED: "dim"
            }.get(step.status, "white")
            
            duration = ""
            if step.start_time and step.end_time:
                duration = str(step.end_time - step.start_time)
            elif step.start_time:
                duration = f"{datetime.now() - step.start_time} (running)"
            
            status_text = f"[{status_color}]{step.status.value}[/{status_color}]"
            if step.error:
                status_text += f"\n[red]Error: {step.error}[/red]"
            
            table.add_row(
                step.name,
                status_text,
                duration,
                step.description
            )
        
        # Add summary
        total_duration = ""
        if report.end_time:
            total_duration = str(report.end_time - report.start_time)
        else:
            total_duration = f"{datetime.now() - report.start_time} (ongoing)"
        
        summary = Panel(
            f"Total Steps: {report.total_steps}\n"
            f"Completed: {report.completed_steps}\n"
            f"Failed: {report.failed_steps}\n"
            f"Duration: {total_duration}",
            title="Summary"
        )
        
        self.console.print(table)
        self.console.print(summary)
    
    def create_spinner(self, text: str = "Processing..."):
        """Create a simple spinner for indeterminate operations"""
        return self.console.status(text)
    
    def track_file_processing(
        self,
        files: list,
        operation_name: str = "File Processing"
    ) -> str:
        """Track file processing with progress bar"""
        operation_id = self.track_operation(operation_name, len(files))
        
        with self.create_progress_bar(len(files), f"Processing {len(files)} files") as (progress, task_id):
            for i, file_path in enumerate(files):
                self.update_step(
                    operation_id,
                    f"file_{i}",
                    ProgressStatus.RUNNING,
                    f"Processing {file_path}"
                )
                
                # Simulate processing time
                time.sleep(0.1)
                
                self.update_step(
                    operation_id,
                    f"file_{i}",
                    ProgressStatus.COMPLETED,
                    f"Processed {file_path}"
                )
                
                progress.update(task_id, advance=1)
        
        self.complete_operation(operation_id)
        return operation_id
    
    def track_pipeline_execution(
        self,
        pipeline_config: Dict[str, Any],
        operation_name: str = "Pipeline Execution"
    ) -> str:
        """Track pipeline execution with detailed progress"""
        steps = pipeline_config.get('steps', [])
        operation_id = self.track_operation(operation_name, len(steps))
        
        for i, step_config in enumerate(steps):
            step_name = step_config.get('name', f'step_{i}')
            step_description = step_config.get('description', step_name)
            
            self.update_step(
                operation_id,
                step_name,
                ProgressStatus.RUNNING,
                step_description
            )
            
            # Simulate step execution
            time.sleep(0.5)
            
            # Simulate success/failure
            success = True  # In real implementation, this would be actual result
            self.update_step(
                operation_id,
                step_name,
                ProgressStatus.COMPLETED if success else ProgressStatus.FAILED,
                step_description
            )
        
        self.complete_operation(operation_id)
        return operation_id
    
    def display_live_progress(self, operation_id: str):
        """Display live progress updates"""
        def update_display():
            while True:
                report = self.get_progress_report(operation_id)
                if not report or report.end_time:
                    break
                
                # Clear previous output
                self.console.clear()
                
                # Create progress layout
                layout = Layout()
                
                # Header
                header = Panel(
                    f"Operation: {report.operation_name}\n"
                    f"Progress: {report.completed_steps}/{report.total_steps}",
                    title="Live Progress"
                )
                
                # Progress steps
                steps_table = Table()
                steps_table.add_column("Step", style="cyan")
                steps_table.add_column("Status", style="magenta")
                steps_table.add_column("Start Time", style="green")
                
                for step in report.steps:
                    status_color = {
                        ProgressStatus.PENDING: "yellow",
                        ProgressStatus.RUNNING: "blue",
                        ProgressStatus.COMPLETED: "green",
                        ProgressStatus.FAILED: "red"
                    }.get(step.status, "white")
                    
                    steps_table.add_row(
                        step.name,
                        f"[{status_color}]{step.status.value}[/{status_color}]",
                        step.start_time.strftime("%H:%M:%S") if step.start_time else "N/A"
                    )
                
                layout.split_column(
                    Layout(header, size=3),
                    Layout(steps_table, size=10)
                )
                
                with Live(layout, refresh_per_second=4, console=self.console):
                    time.sleep(1)
        
        # Run in separate thread
        thread = threading.Thread(target=update_display)
        thread.daemon = True
        thread.start()
    
    def create_progress_callback(self, operation_id: str, total: int) -> Callable[[int], None]:
        """Create a callback function for updating progress"""
        def callback(completed: int):
            progress = min(completed / total * 100, 100)
            self.console.print(f"Progress: {progress:.1f}% ({completed}/{total})")
        
        return callback


# Convenience functions for common progress patterns
def track_with_spinner(text: str, func: Callable, *args, **kwargs):
    """Execute a function with a spinner indicator"""
    tracker = ProgressTracker()
    with tracker.create_spinner(text):
        return func(*args, **kwargs)


def create_progress_bar(total: int, description: str = "Processing"):
    """Create a simple progress bar"""
    tracker = ProgressTracker()
    return tracker.create_progress_bar(total, description)


def display_operation_summary(operation_id: str, config: Optional[Any] = None):
    """Display a summary of a completed operation"""
    tracker = ProgressTracker(config)
    tracker.display_progress_report(operation_id)