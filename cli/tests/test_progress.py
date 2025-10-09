"""
Tests for progress tracking and indicators.
"""

import pytest
import time
import threading
from unittest.mock import patch, MagicMock
from datetime import datetime, timedelta

from cli.progress import (
    ProgressTracker, ProgressStatus, ProgressStep, ProgressReport,
    track_with_spinner, create_progress_bar, display_operation_summary
)


class TestProgressTracker:
    """Test cases for progress tracking functionality"""
    
    def test_progress_tracker_initialization(self, mock_config):
        """Test progress tracker initialization"""
        tracker = ProgressTracker(mock_config)
        
        assert tracker.config == mock_config
        assert tracker._progress_reports == {}
        assert tracker._current_progress is None
        assert tracker._current_task_id is None
    
    def test_track_operation_creation(self, mock_config):
        """Test operation tracking creation"""
        tracker = ProgressTracker(mock_config)
        
        operation_id = tracker.track_operation("Test Operation", 5)
        
        assert operation_id.startswith("Test Operation_")
        assert operation_id in tracker._progress_reports
        assert tracker._progress_reports[operation_id].operation_name == "Test Operation"
        assert tracker._progress_reports[operation_id].total_steps == 5
        assert tracker._progress_reports[operation_id].completed_steps == 0
    
    def test_update_step_status_changes(self, mock_config):
        """Test step status updates"""
        tracker = ProgressTracker(mock_config)
        operation_id = tracker.track_operation("Test Operation", 3)
        
        # Test PENDING to RUNNING
        tracker.update_step(operation_id, "step1", ProgressStatus.RUNNING, "Running step 1")
        step = tracker._progress_reports[operation_id].steps[0]
        assert step.status == ProgressStatus.RUNNING
        assert step.start_time is not None
        assert step.end_time is None
        
        # Test RUNNING to COMPLETED
        tracker.update_step(operation_id, "step1", ProgressStatus.COMPLETED, "Completed step 1")
        step = tracker._progress_reports[operation_id].steps[0]
        assert step.status == ProgressStatus.COMPLETED
        assert step.end_time is not None
        assert tracker._progress_reports[operation_id].completed_steps == 1
    
    def test_update_step_failure_handling(self, mock_config):
        """Test step failure handling"""
        tracker = ProgressTracker(mock_config)
        operation_id = tracker.track_operation("Test Operation", 2)
        
        # Complete first step
        tracker.update_step(operation_id, "step1", ProgressStatus.COMPLETED, "Step 1 completed")
        
        # Fail second step
        tracker.update_step(operation_id, "step2", ProgressStatus.FAILED, "Step 2 failed", "Error details")
        step = tracker._progress_reports[operation_id].steps[1]
        
        assert step.status == ProgressStatus.FAILED
        assert step.error == "Error details"
        assert tracker._progress_reports[operation_id].failed_steps == 1
    
    def test_complete_operation(self, mock_config):
        """Test operation completion"""
        tracker = ProgressTracker(mock_config)
        operation_id = tracker.track_operation("Test Operation", 2)
        
        # Complete some steps
        tracker.update_step(operation_id, "step1", ProgressStatus.COMPLETED)
        tracker.update_step(operation_id, "step2", ProgressStatus.COMPLETED)
        
        # Complete operation
        tracker.complete_operation(operation_id)
        
        report = tracker._progress_reports[operation_id]
        assert report.end_time is not None
        assert report.completed_steps == 2
        assert report.failed_steps == 0
    
    def test_get_progress_report(self, mock_config):
        """Test getting progress report"""
        tracker = ProgressTracker(mock_config)
        operation_id = tracker.track_operation("Test Operation", 1)
        
        tracker.update_step(operation_id, "step1", ProgressStatus.COMPLETED)
        tracker.complete_operation(operation_id)
        
        report = tracker.get_progress_report(operation_id)
        
        assert report is not None
        assert report.operation_name == "Test Operation"
        assert report.completed_steps == 1
    
    def test_get_progress_report_nonexistent(self, mock_config):
        """Test getting progress report for non-existent operation"""
        tracker = ProgressTracker(mock_config)
        
        report = tracker.get_progress_report("nonexistent")
        
        assert report is None
    
    def test_create_progress_bar(self, mock_config):
        """Test progress bar creation"""
        tracker = ProgressTracker(mock_config)
        
        progress, task_id = tracker.create_progress_bar(100, "Test Progress")
        
        assert progress is not None
        assert task_id is not None
        assert hasattr(progress, 'update')
    
    def test_create_spinner(self, mock_config):
        """Test spinner creation"""
        tracker = ProgressTracker(mock_config)
        
        spinner = tracker.create_spinner("Testing spinner")
        
        assert spinner is not None
        assert hasattr(spinner, '__enter__')
        assert hasattr(spinner, '__exit__')
    
    def test_track_file_processing(self, mock_config):
        """Test file processing tracking"""
        tracker = ProgressTracker(mock_config)
        
        files = ["file1.txt", "file2.txt", "file3.txt"]
        operation_id = tracker.track_file_processing(files, "File Processing Test")
        
        report = tracker.get_progress_report(operation_id)
        
        assert report is not None
        assert report.operation_name == "File Processing Test"
        assert len(report.steps) == 3
        assert all(step.status == ProgressStatus.COMPLETED for step in report.steps)
    
    def test_track_pipeline_execution(self, mock_config):
        """Test pipeline execution tracking"""
        tracker = ProgressTracker(mock_config)
        
        pipeline_config = {
            'steps': [
                {'name': 'preprocess', 'description': 'Preprocessing data'},
                {'name': 'analyze', 'description': 'Analyzing data'},
                {'name': 'generate', 'description': 'Generating output'}
            ]
        }
        
        operation_id = tracker.track_pipeline_execution(pipeline_config, "Pipeline Test")
        
        report = tracker.get_progress_report(operation_id)
        
        assert report is not None
        assert report.operation_name == "Pipeline Test"
        assert len(report.steps) == 3
        assert all(step.status == ProgressStatus.COMPLETED for step in report.steps)
    
    def test_display_progress_report(self, mock_config):
        """Test progress report display"""
        tracker = ProgressTracker(mock_config)
        
        # Create a completed operation
        operation_id = tracker.track_operation("Display Test", 2)
        tracker.update_step(operation_id, "step1", ProgressStatus.COMPLETED, "Step 1")
        tracker.update_step(operation_id, "step2", ProgressStatus.COMPLETED, "Step 2")
        tracker.complete_operation(operation_id)
        
        # Should not raise any exceptions
        tracker.display_progress_report(operation_id)
        
        # Test with non-existent operation
        tracker.display_progress_report("nonexistent")  # Should handle gracefully
    
    def test_display_live_progress(self, mock_config):
        """Test live progress display"""
        tracker = ProgressTracker(mock_config)
        
        operation_id = tracker.track_operation("Live Progress Test", 3)
        
        # Start live display in background
        tracker.display_live_progress(operation_id)
        
        # Simulate some progress updates
        time.sleep(0.1)
        tracker.update_step(operation_id, "step1", ProgressStatus.COMPLETED)
        time.sleep(0.1)
        tracker.update_step(operation_id, "step2", ProgressStatus.RUNNING)
        time.sleep(0.1)
        tracker.update_step(operation_id, "step2", ProgressStatus.COMPLETED)
        
        tracker.complete_operation(operation_id)
        
        # Should complete without errors
        assert True
    
    def test_create_progress_callback(self, mock_config):
        """Test progress callback creation"""
        tracker = ProgressTracker(mock_config)
        operation_id = tracker.track_operation("Callback Test", 100)
        
        callback = tracker.create_progress_callback(operation_id, 100)
        
        # Test callback with different progress values
        callback(25)  # 25%
        callback(50)  # 50%
        callback(75)  # 75%
        callback(100)  # 100%
        
        # Should complete without errors
        assert True
    
    def test_thread_safety(self, mock_config):
        """Test thread safety of progress tracking"""
        tracker = ProgressTracker(mock_config)
        
        operation_id = tracker.track_operation("Thread Safety Test", 10)
        
        def update_progress():
            for i in range(5):
                tracker.update_step(operation_id, f"step{i}", ProgressStatus.COMPLETED)
                time.sleep(0.01)
        
        # Create multiple threads
        threads = []
        for _ in range(3):
            thread = threading.Thread(target=update_progress)
            threads.append(thread)
            thread.start()
        
        # Wait for all threads to complete
        for thread in threads:
            thread.join()
        
        report = tracker.get_progress_report(operation_id)
        
        # Should have handled concurrent updates safely
        assert report is not None
        assert report.completed_steps >= 5  # At least 5 steps completed


class TestProgressConvenienceFunctions:
    """Test convenience functions for progress tracking"""
    
    def test_track_with_spinner(self):
        """Test track with spinner function"""
        def test_function():
            return "test_result"
        
        result = track_with_spinner("Testing spinner", test_function)
        
        assert result == "test_result"
    
    def test_create_progress_bar(self):
        """Test create progress bar function"""
        progress, task_id = create_progress_bar(50, "Test Progress")
        
        assert progress is not None
        assert task_id is not None
        
        # Test progress updates
        progress.update(task_id, advance=25)
        progress.update(task_id, advance=25)
        
        progress.stop()
    
    def test_display_operation_summary(self, mock_config):
        """Test display operation summary function"""
        tracker = ProgressTracker(mock_config)
        operation_id = tracker.track_operation("Summary Test", 2)
        
        tracker.update_step(operation_id, "step1", ProgressStatus.COMPLETED, "Step 1")
        tracker.update_step(operation_id, "step2", ProgressStatus.COMPLETED, "Step 2")
        tracker.complete_operation(operation_id)
        
        # Should not raise any exceptions
        display_operation_summary(operation_id, mock_config)


class TestProgressStatus:
    """Test progress status enumeration"""
    
    def test_progress_status_values(self):
        """Test progress status enum values"""
        assert ProgressStatus.PENDING.value == "pending"
        assert ProgressStatus.RUNNING.value == "running"
        assert ProgressStatus.COMPLETED.value == "completed"
        assert ProgressStatus.FAILED.value == "failed"
        assert ProgressStatus.CANCELLED.value == "cancelled"
    
    def test_progress_status_comparison(self):
        """Test progress status comparison"""
        assert ProgressStatus.PENDING != ProgressStatus.RUNNING
        assert ProgressStatus.COMPLETED != ProgressStatus.FAILED
        assert ProgressStatus.RUNNING == ProgressStatus.RUNNING


class TestProgressDataClasses:
    """Test progress data classes"""
    
    def test_progress_step_creation(self):
        """Test ProgressStep creation"""
        step = ProgressStep(
            name="test_step",
            description="Test step description",
            status=ProgressStatus.RUNNING
        )
        
        assert step.name == "test_step"
        assert step.description == "Test step description"
        assert step.status == ProgressStatus.RUNNING
        assert step.start_time is None
        assert step.end_time is None
        assert step.error is None
        assert step.metadata == {}
    
    def test_progress_report_creation(self):
        """Test ProgressReport creation"""
        report = ProgressReport(
            operation_id="test_op_123",
            operation_name="Test Operation",
            total_steps=5
        )
        
        assert report.operation_id == "test_op_123"
        assert report.operation_name == "Test Operation"
        assert report.total_steps == 5
        assert report.completed_steps == 0
        assert report.failed_steps == 0
        assert report.start_time is not None
        assert report.end_time is None
        assert report.steps == []
        assert report.metadata == {}


class TestProgressPerformance:
    """Performance tests for progress tracking"""
    
    def test_large_number_of_operations(self, mock_config):
        """Test handling large number of operations"""
        tracker = ProgressTracker(mock_config)
        
        # Create many operations
        operation_ids = []
        for i in range(100):
            operation_id = tracker.track_operation(f"Bulk Test {i}", 10)
            operation_ids.append(operation_id)
        
        # Update all operations
        for operation_id in operation_ids:
            for j in range(5):
                tracker.update_step(operation_id, f"step{j}", ProgressStatus.COMPLETED)
            tracker.complete_operation(operation_id)
        
        # Verify all operations are tracked
        assert len(tracker._progress_reports) == 100
        
        # Verify random operation
        random_report = tracker.get_progress_report(operation_ids[50])
        assert random_report is not None
        assert random_report.completed_steps == 5
    
    def test_rapid_status_updates(self, mock_config):
        """Test rapid status updates"""
        tracker = ProgressTracker(mock_config)
        operation_id = tracker.track_operation("Rapid Updates Test", 1000)
        
        start_time = time.time()
        
        # Perform rapid updates
        for i in range(100):
            tracker.update_step(operation_id, f"step{i}", ProgressStatus.COMPLETED)
        
        end_time = time.time()
        
        # Should complete quickly (under 1 second for 100 updates)
        assert (end_time - start_time) < 1.0
        
        report = tracker.get_progress_report(operation_id)
        assert report.completed_steps == 100
    
    def test_memory_efficiency(self, mock_config):
        """Test memory efficiency with large operations"""
        tracker = ProgressTracker(mock_config)
        
        # Create operation with many steps
        operation_id = tracker.track_operation("Memory Test", 1000)
        
        # Add many steps with metadata
        for i in range(1000):
            tracker.update_step(
                operation_id,
                f"step{i}",
                ProgressStatus.COMPLETED,
                f"Step {i} completed",
                metadata={"index": i, "data": f"step_data_{i}"}
            )
        
        tracker.complete_operation(operation_id)
        
        report = tracker.get_progress_report(operation_id)
        
        # Should handle large amount of data
        assert report is not None
        assert len(report.steps) == 1000
        assert report.completed_steps == 1000


# Integration tests
class TestProgressIntegration:
    """Integration tests for progress tracking"""
    
    def test_progress_with_pipeline_integration(self, mock_config, mock_pipeline_manager):
        """Test progress tracking with pipeline manager integration"""
        tracker = ProgressTracker(mock_config)
        
        # Simulate pipeline execution with progress tracking
        operation_id = tracker.track_operation("Pipeline Integration Test", 3)
        
        # Mock pipeline steps
        steps = [
            {"name": "data_loading", "description": "Loading data"},
            {"name": "preprocessing", "description": "Preprocessing data"},
            {"name": "analysis", "description": "Analyzing data"}
        ]
        
        for step_config in steps:
            tracker.update_step(
                operation_id,
                step_config["name"],
                ProgressStatus.RUNNING,
                step_config["description"]
            )
            
            # Simulate work
            time.sleep(0.01)
            
            tracker.update_step(
                operation_id,
                step_config["name"],
                ProgressStatus.COMPLETED,
                step_config["description"]
            )
        
        tracker.complete_operation(operation_id)
        
        report = tracker.get_progress_report(operation_id)
        
        assert report is not None
        assert report.completed_steps == 3
        assert len(report.steps) == 3
        assert all(step.status == ProgressStatus.COMPLETED for step in report.steps)
    
    def test_progress_with_auth_integration(self, mock_config, mock_auth_manager):
        """Test progress tracking with authentication integration"""
        tracker = ProgressTracker(mock_config)
        
        operation_id = tracker.track_operation("Auth Integration Test", 2)
        
        # Simulate authentication steps
        tracker.update_step(operation_id, "login", ProgressStatus.RUNNING, "Logging in")
        # Simulate successful login
        tracker.update_step(operation_id, "login", ProgressStatus.COMPLETED, "Login successful")
        
        tracker.update_step(operation_id, "token_validation", ProgressStatus.RUNNING, "Validating token")
        # Simulate successful validation
        tracker.update_step(operation_id, "token_validation", ProgressStatus.COMPLETED, "Token valid")
        
        tracker.complete_operation(operation_id)
        
        report = tracker.get_progress_report(operation_id)
        
        assert report is not None
        assert report.completed_steps == 2
        assert all(step.status == ProgressStatus.COMPLETED for step in report.steps)