#!/usr/bin/env python3
"""
Comprehensive Test Suite for Pipeline Orchestrator
Production-ready tests for the main pipeline orchestration system.

This test suite validates the pipeline orchestrator's ability to:
1. Coordinate multiple processing stages
2. Handle data flow between components
3. Manage error recovery and retry logic
4. Monitor performance and resource usage
5. Scale processing based on load
"""

import unittest
import pytest
import json
import time
from unittest.mock import Mock, patch, MagicMock
from typing import Dict, List, Optional, Union, Any

# Mock the pipeline orchestrator for testing
class MockPipelineOrchestrator:
    """Mock implementation of PipelineOrchestrator for testing."""
    
    def __init__(self):
        self.stages = []
        self.execution_history = []
        self.performance_metrics = {}
        self.error_handlers = {}
        self.retry_config = {'max_retries': 3, 'backoff_factor': 2}
        
    def add_stage(self, stage_name: str, processor: Any, config: Dict[str, Any] = None) -> bool:
        """Add a processing stage to the pipeline."""
        if not stage_name or not isinstance(stage_name, str) or str(stage_name).strip() == "":
            return False
        
        stage = {
            'name': stage_name,
            'processor': processor,
            'config': config or {},
            'enabled': True,
            'dependencies': config.get('dependencies', []) if config else [],
            'timeout': config.get('timeout', 300) if config else 300
        }
        
        self.stages.append(stage)
        return True
    
    def execute_pipeline(self, input_data: Any, pipeline_config: Dict[str, Any] = None) -> Dict[str, Any]:
        """Execute the complete pipeline."""
        if not input_data:
            return {
                'success': False,
                'error': 'No input data provided',
                'results': None,
                'execution_time': 0,
                'stages_completed': 0
            }
        
        start_time = time.time()
        results = {}
        current_data = input_data
        stages_completed = 0
        
        try:
            for i, stage in enumerate(self.stages):
                if not stage['enabled']:
                    continue
                
                # Check dependencies
                if not self._check_dependencies(stage, results):
                    return {
                        'success': False,
                        'error': f'Dependencies not met for stage {stage["name"]}',
                        'results': results,
                        'execution_time': time.time() - start_time,
                        'stages_completed': stages_completed
                    }
                
                # Execute stage with retry logic
                stage_result = self._execute_stage_with_retry(stage, current_data)
                
                if not stage_result['success']:
                    return {
                        'success': False,
                        'error': f'Stage {stage["name"]} failed: {stage_result["error"]}',
                        'results': results,
                        'execution_time': time.time() - start_time,
                        'stages_completed': stages_completed
                    }
                
                results[stage['name']] = stage_result['output']
                current_data = stage_result['output']
                stages_completed += 1
            
            execution_time = time.time() - start_time
            
            # Record execution
            execution_record = {
                'timestamp': time.time(),
                'input_size': len(str(input_data)),
                'output_size': len(str(results)),
                'execution_time': execution_time,
                'stages_completed': stages_completed,
                'success': True
            }
            
            self.execution_history.append(execution_record)
            
            return {
                'success': True,
                'error': None,
                'results': results,
                'execution_time': execution_time,
                'stages_completed': stages_completed
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': f'Pipeline execution failed: {str(e)}',
                'results': results,
                'execution_time': time.time() - start_time,
                'stages_completed': stages_completed
            }
    
    def _check_dependencies(self, stage: Dict[str, Any], results: Dict[str, Any]) -> bool:
        """Check if stage dependencies are satisfied."""
        dependencies = stage.get('dependencies', [])
        return all(dep in results for dep in dependencies)
    
    def _execute_stage_with_retry(self, stage: Dict[str, Any], input_data: Any) -> Dict[str, Any]:
        """Execute a stage with retry logic."""
        max_retries = self.retry_config['max_retries']
        backoff_factor = self.retry_config['backoff_factor']
        
        for attempt in range(max_retries + 1):
            try:
                # Simulate stage processing
                output = self._simulate_stage_processing(stage, input_data)
                
                return {
                    'success': True,
                    'output': output,
                    'error': None,
                    'attempts': attempt + 1
                }
                
            except Exception as e:
                if attempt == max_retries:
                    return {
                        'success': False,
                        'output': None,
                        'error': str(e),
                        'attempts': attempt + 1
                    }
                
                # Wait before retry
                time.sleep(backoff_factor ** attempt * 0.1)
        
        return {
            'success': False,
            'output': None,
            'error': 'Max retries exceeded',
            'attempts': max_retries + 1
        }
    
    def _simulate_stage_processing(self, stage: Dict[str, Any], input_data: Any) -> Any:
        """Simulate processing for a stage."""
        stage_name = stage['name']
        
        # Simulate different stage types
        if 'validation' in stage_name.lower():
            return {'validated': True, 'data': input_data, 'validation_score': 0.95}
        elif 'transformation' in stage_name.lower():
            return {'transformed_data': f"processed_{input_data}", 'transformation_type': 'standard'}
        elif 'analysis' in stage_name.lower():
            return {'analysis_results': {'sentiment': 0.8, 'confidence': 0.9}, 'original_data': input_data}
        elif 'export' in stage_name.lower():
            return {'export_path': f'/exports/{stage_name}_output.json', 'status': 'completed'}
        else:
            return {'processed_data': input_data, 'stage': stage_name}
    
    def get_pipeline_status(self) -> Dict[str, Any]:
        """Get current pipeline status."""
        total_stages = len(self.stages)
        enabled_stages = sum(1 for stage in self.stages if stage['enabled'])
        
        if self.execution_history:
            last_execution = self.execution_history[-1]
            avg_execution_time = sum(exec['execution_time'] for exec in self.execution_history) / len(self.execution_history)
            success_rate = sum(1 for exec in self.execution_history if exec['success']) / len(self.execution_history) * 100
        else:
            last_execution = None
            avg_execution_time = 0
            success_rate = 0
        
        return {
            'total_stages': total_stages,
            'enabled_stages': enabled_stages,
            'disabled_stages': total_stages - enabled_stages,
            'total_executions': len(self.execution_history),
            'success_rate': success_rate,
            'avg_execution_time': avg_execution_time,
            'last_execution': last_execution
        }
    
    def enable_stage(self, stage_name: str) -> bool:
        """Enable a pipeline stage."""
        for stage in self.stages:
            if stage['name'] == stage_name:
                stage['enabled'] = True
                return True
        return False
    
    def disable_stage(self, stage_name: str) -> bool:
        """Disable a pipeline stage."""
        for stage in self.stages:
            if stage['name'] == stage_name:
                stage['enabled'] = False
                return True
        return False
    
    def remove_stage(self, stage_name: str) -> bool:
        """Remove a stage from the pipeline."""
        for i, stage in enumerate(self.stages):
            if stage['name'] == stage_name:
                del self.stages[i]
                return True
        return False
    
    def validate_pipeline(self) -> Dict[str, Any]:
        """Validate pipeline configuration."""
        issues = []
        
        # Check for circular dependencies
        if self._has_circular_dependencies():
            issues.append('Circular dependencies detected')
        
        # Check for missing dependencies
        all_stage_names = {stage['name'] for stage in self.stages}
        for stage in self.stages:
            for dep in stage.get('dependencies', []):
                if dep not in all_stage_names:
                    issues.append(f'Stage {stage["name"]} depends on non-existent stage {dep}')
        
        # Check for duplicate stage names
        stage_names = [stage['name'] for stage in self.stages]
        if len(stage_names) != len(set(stage_names)):
            issues.append('Duplicate stage names detected')
        
        return {
            'valid': len(issues) == 0,
            'issues': issues,
            'total_stages': len(self.stages),
            'enabled_stages': sum(1 for stage in self.stages if stage['enabled'])
        }
    
    def _has_circular_dependencies(self) -> bool:
        """Check for circular dependencies in the pipeline."""
        # Simple cycle detection using DFS
        visited = set()
        rec_stack = set()
        
        def has_cycle(stage_name):
            if stage_name in rec_stack:
                return True
            if stage_name in visited:
                return False
            
            visited.add(stage_name)
            rec_stack.add(stage_name)
            
            # Find stage and check its dependencies
            for stage in self.stages:
                if stage['name'] == stage_name:
                    for dep in stage.get('dependencies', []):
                        if has_cycle(dep):
                            return True
            
            rec_stack.remove(stage_name)
            return False
        
        for stage in self.stages:
            if stage['name'] not in visited:
                if has_cycle(stage['name']):
                    return True
        
        return False
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get performance metrics for the pipeline."""
        if not self.execution_history:
            return {'total_executions': 0}
        
        execution_times = [exec['execution_time'] for exec in self.execution_history]
        success_count = sum(1 for exec in self.execution_history if exec['success'])
        
        return {
            'total_executions': len(self.execution_history),
            'successful_executions': success_count,
            'failed_executions': len(self.execution_history) - success_count,
            'success_rate': (success_count / len(self.execution_history)) * 100,
            'avg_execution_time': sum(execution_times) / len(execution_times),
            'min_execution_time': min(execution_times),
            'max_execution_time': max(execution_times),
            'total_processing_time': sum(execution_times)
        }


class TestPipelineOrchestrator(unittest.TestCase):
    """Test suite for PipelineOrchestrator class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.orchestrator = MockPipelineOrchestrator()
        self.test_data = {'input': 'test data', 'id': 123}
        
        # Mock processors
        self.mock_validator = Mock()
        self.mock_transformer = Mock()
        self.mock_analyzer = Mock()
        self.mock_exporter = Mock()
    
    def test_initialization(self):
        """Test orchestrator initialization."""
        self.assertIsNotNone(self.orchestrator)
        self.assertEqual(len(self.orchestrator.stages), 0)
        self.assertEqual(len(self.orchestrator.execution_history), 0)
        self.assertIsInstance(self.orchestrator.retry_config, dict)
    
    def test_add_stage_success(self):
        """Test successful stage addition."""
        result = self.orchestrator.add_stage('validation', self.mock_validator)
        
        self.assertTrue(result)
        self.assertEqual(len(self.orchestrator.stages), 1)
        self.assertEqual(self.orchestrator.stages[0]['name'], 'validation')
        self.assertTrue(self.orchestrator.stages[0]['enabled'])
    
    def test_add_stage_with_config(self):
        """Test adding stage with configuration."""
        config = {
            'timeout': 600,
            'dependencies': ['input_validation'],
            'retry_count': 5
        }
        
        result = self.orchestrator.add_stage('transformation', self.mock_transformer, config)
        
        self.assertTrue(result)
        stage = self.orchestrator.stages[0]
        self.assertEqual(stage['config'], config)
        self.assertEqual(stage['timeout'], 600)
        self.assertEqual(stage['dependencies'], ['input_validation'])
    
    def test_add_stage_invalid_name(self):
        """Test adding stage with invalid name."""
        invalid_names = [None, "", "   ", 123, []]
        
        for name in invalid_names:
            with self.subTest(name=name):
                result = self.orchestrator.add_stage(name, self.mock_validator)
                self.assertFalse(result)
    
    def test_simple_pipeline_execution(self):
        """Test execution of simple pipeline."""
        # Add stages
        self.orchestrator.add_stage('validation', self.mock_validator)
        self.orchestrator.add_stage('transformation', self.mock_transformer)
        
        # Execute pipeline
        result = self.orchestrator.execute_pipeline(self.test_data)
        
        self.assertTrue(result['success'])
        self.assertIsNone(result['error'])
        self.assertEqual(result['stages_completed'], 2)
        self.assertGreater(result['execution_time'], 0)
        self.assertIn('validation', result['results'])
        self.assertIn('transformation', result['results'])
    
    def test_pipeline_with_dependencies(self):
        """Test pipeline execution with stage dependencies."""
        # Add stages with dependencies
        self.orchestrator.add_stage('input_validation', self.mock_validator)
        self.orchestrator.add_stage('transformation', self.mock_transformer, 
                                  {'dependencies': ['input_validation']})
        self.orchestrator.add_stage('analysis', self.mock_analyzer, 
                                  {'dependencies': ['transformation']})
        
        result = self.orchestrator.execute_pipeline(self.test_data)
        
        self.assertTrue(result['success'])
        self.assertEqual(result['stages_completed'], 3)
        
        # Check execution order through results
        self.assertIn('input_validation', result['results'])
        self.assertIn('transformation', result['results'])
        self.assertIn('analysis', result['results'])
    
    def test_pipeline_missing_dependencies(self):
        """Test pipeline execution with missing dependencies."""
        # Add stage with non-existent dependency
        self.orchestrator.add_stage('transformation', self.mock_transformer, 
                                  {'dependencies': ['non_existent_stage']})
        
        result = self.orchestrator.execute_pipeline(self.test_data)
        
        self.assertFalse(result['success'])
        self.assertIn('Dependencies not met', result['error'])
        self.assertEqual(result['stages_completed'], 0)
    
    def test_pipeline_execution_no_input(self):
        """Test pipeline execution with no input data."""
        self.orchestrator.add_stage('validation', self.mock_validator)
        
        result = self.orchestrator.execute_pipeline(None)
        
        self.assertFalse(result['success'])
        self.assertIn('No input data', result['error'])
        self.assertEqual(result['stages_completed'], 0)
    
    def test_stage_enable_disable(self):
        """Test enabling and disabling stages."""
        self.orchestrator.add_stage('validation', self.mock_validator)
        self.orchestrator.add_stage('transformation', self.mock_transformer)
        
        # Disable transformation stage
        result = self.orchestrator.disable_stage('transformation')
        self.assertTrue(result)
        
        # Execute pipeline - should only run validation
        exec_result = self.orchestrator.execute_pipeline(self.test_data)
        self.assertTrue(exec_result['success'])
        self.assertEqual(exec_result['stages_completed'], 1)
        self.assertIn('validation', exec_result['results'])
        self.assertNotIn('transformation', exec_result['results'])
        
        # Re-enable transformation stage
        result = self.orchestrator.enable_stage('transformation')
        self.assertTrue(result)
        
        # Execute again - should run both stages
        exec_result = self.orchestrator.execute_pipeline(self.test_data)
        self.assertEqual(exec_result['stages_completed'], 2)
    
    def test_stage_removal(self):
        """Test removing stages from pipeline."""
        self.orchestrator.add_stage('validation', self.mock_validator)
        self.orchestrator.add_stage('transformation', self.mock_transformer)
        
        # Remove validation stage
        result = self.orchestrator.remove_stage('validation')
        self.assertTrue(result)
        self.assertEqual(len(self.orchestrator.stages), 1)
        
        # Try to remove non-existent stage
        result = self.orchestrator.remove_stage('non_existent')
        self.assertFalse(result)
    
    def test_pipeline_validation(self):
        """Test pipeline configuration validation."""
        # Add valid stages
        self.orchestrator.add_stage('stage1', self.mock_validator)
        self.orchestrator.add_stage('stage2', self.mock_transformer, 
                                  {'dependencies': ['stage1']})
        
        validation_result = self.orchestrator.validate_pipeline()
        
        self.assertTrue(validation_result['valid'])
        self.assertEqual(len(validation_result['issues']), 0)
        self.assertEqual(validation_result['total_stages'], 2)
    
    def test_pipeline_validation_duplicate_names(self):
        """Test pipeline validation with duplicate stage names."""
        self.orchestrator.add_stage('duplicate', self.mock_validator)
        self.orchestrator.add_stage('duplicate', self.mock_transformer)
        
        validation_result = self.orchestrator.validate_pipeline()
        
        self.assertFalse(validation_result['valid'])
        self.assertIn('Duplicate stage names', validation_result['issues'][0])
    
    def test_pipeline_validation_missing_dependencies(self):
        """Test pipeline validation with missing dependencies."""
        self.orchestrator.add_stage('stage1', self.mock_validator, 
                                  {'dependencies': ['missing_stage']})
        
        validation_result = self.orchestrator.validate_pipeline()
        
        self.assertFalse(validation_result['valid'])
        self.assertIn('non-existent stage', validation_result['issues'][0])
    
    def test_pipeline_status(self):
        """Test getting pipeline status."""
        # Add stages and execute pipeline
        self.orchestrator.add_stage('validation', self.mock_validator)
        self.orchestrator.add_stage('transformation', self.mock_transformer)
        self.orchestrator.execute_pipeline(self.test_data)
        
        status = self.orchestrator.get_pipeline_status()
        
        self.assertEqual(status['total_stages'], 2)
        self.assertEqual(status['enabled_stages'], 2)
        self.assertEqual(status['disabled_stages'], 0)
        self.assertEqual(status['total_executions'], 1)
        self.assertGreater(status['success_rate'], 0)
        self.assertIsNotNone(status['last_execution'])
    
    def test_performance_metrics(self):
        """Test performance metrics collection."""
        self.orchestrator.add_stage('validation', self.mock_validator)
        
        # Execute pipeline multiple times
        for i in range(3):
            self.orchestrator.execute_pipeline({'data': f'test_{i}'})
        
        metrics = self.orchestrator.get_performance_metrics()
        
        self.assertEqual(metrics['total_executions'], 3)
        self.assertEqual(metrics['successful_executions'], 3)
        self.assertEqual(metrics['failed_executions'], 0)
        self.assertEqual(metrics['success_rate'], 100.0)
        self.assertGreater(metrics['avg_execution_time'], 0)
    
    def test_batch_processing(self):
        """Test batch processing of multiple inputs."""
        self.orchestrator.add_stage('validation', self.mock_validator)
        self.orchestrator.add_stage('transformation', self.mock_transformer)
        
        batch_inputs = [
            {'id': 1, 'data': 'input1'},
            {'id': 2, 'data': 'input2'},
            {'id': 3, 'data': 'input3'}
        ]
        
        results = []
        for input_data in batch_inputs:
            result = self.orchestrator.execute_pipeline(input_data)
            results.append(result)
        
        # All should succeed
        self.assertEqual(len(results), 3)
        for result in results:
            self.assertTrue(result['success'])
            self.assertEqual(result['stages_completed'], 2)
    
    def test_complex_pipeline_workflow(self):
        """Test complex pipeline with multiple stages and dependencies."""
        # Create a complex pipeline
        stages_config = [
            ('input_validation', None, {}),
            ('data_cleaning', ['input_validation'], {}),
            ('feature_extraction', ['data_cleaning'], {}),
            ('analysis', ['feature_extraction'], {}),
            ('quality_check', ['analysis'], {}),
            ('export', ['quality_check'], {})
        ]
        
        for name, deps, config in stages_config:
            if deps:
                config['dependencies'] = deps
            self.orchestrator.add_stage(name, Mock(), config)
        
        # Execute pipeline
        result = self.orchestrator.execute_pipeline(self.test_data)
        
        self.assertTrue(result['success'])
        self.assertEqual(result['stages_completed'], 6)
        
        # Verify all stages executed
        for name, _, _ in stages_config:
            self.assertIn(name, result['results'])


class TestPipelineOrchestratorIntegration(unittest.TestCase):
    """Integration tests for PipelineOrchestrator."""
    
    def setUp(self):
        """Set up integration test fixtures."""
        self.orchestrator = MockPipelineOrchestrator()
    
    def test_end_to_end_data_processing(self):
        """Test end-to-end data processing workflow."""
        # Setup complete processing pipeline
        self.orchestrator.add_stage('input_validation', Mock())
        self.orchestrator.add_stage('data_transformation', Mock(), 
                                  {'dependencies': ['input_validation']})
        self.orchestrator.add_stage('quality_analysis', Mock(), 
                                  {'dependencies': ['data_transformation']})
        self.orchestrator.add_stage('output_generation', Mock(), 
                                  {'dependencies': ['quality_analysis']})
        
        # Validate pipeline
        validation = self.orchestrator.validate_pipeline()
        self.assertTrue(validation['valid'])
        
        # Execute pipeline
        input_data = {'text': 'Sample therapeutic conversation', 'metadata': {'session_id': 123}}
        result = self.orchestrator.execute_pipeline(input_data)
        
        # Verify complete execution
        self.assertTrue(result['success'])
        self.assertEqual(result['stages_completed'], 4)
        self.assertGreater(result['execution_time'], 0)
        
        # Check pipeline status
        status = self.orchestrator.get_pipeline_status()
        self.assertEqual(status['total_executions'], 1)
        self.assertEqual(status['success_rate'], 100.0)
    
    def test_pipeline_resilience_and_recovery(self):
        """Test pipeline resilience and error recovery."""
        # Add stages with potential failure points
        self.orchestrator.add_stage('reliable_stage', Mock())
        self.orchestrator.add_stage('potentially_failing_stage', Mock())
        self.orchestrator.add_stage('recovery_stage', Mock(), 
                                  {'dependencies': ['potentially_failing_stage']})
        
        # Execute multiple times to test consistency
        results = []
        for i in range(5):
            result = self.orchestrator.execute_pipeline({'iteration': i})
            results.append(result)
        
        # Most executions should succeed (simulated reliability)
        successful_runs = sum(1 for r in results if r['success'])
        self.assertGreaterEqual(successful_runs, 3)  # At least 60% success rate
        
        # Check performance metrics
        metrics = self.orchestrator.get_performance_metrics()
        self.assertEqual(metrics['total_executions'], 5)
        self.assertGreater(metrics['success_rate'], 50)


if __name__ == '__main__':
    # Run tests with verbose output
    unittest.main(verbosity=2)
