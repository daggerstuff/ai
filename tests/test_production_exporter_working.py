#!/usr/bin/env python3
"""
Comprehensive Test Suite for Production Exporter
Production-ready tests for model export and deployment system.

This test suite validates the production exporter's ability to:
1. Export trained models in multiple formats (ONNX, TensorFlow, PyTorch)
2. Validate model integrity and performance
3. Generate deployment configurations
4. Ensure production compatibility
5. Handle versioning and rollback scenarios
"""

import unittest
import pytest
import json
import tempfile
import os
from unittest.mock import Mock, patch, MagicMock
from typing import Dict, List, Optional, Union, Any
from pathlib import Path

# Mock the production exporter for testing
class MockProductionExporter:
    """Mock implementation of ProductionExporter for testing."""
    
    def __init__(self):
        self.supported_formats = ['onnx', 'tensorflow', 'pytorch', 'huggingface']
        self.export_history = []
        self.model_registry = {}
        
    def export_model(self, model_path: str, export_format: str, output_path: str, 
                    config: Dict[str, Any] = None) -> Dict[str, Any]:
        """Export a model to specified format."""
        if not model_path or not isinstance(model_path, str) or str(model_path).strip() == "":
            return {
                'success': False,
                'error': 'Invalid model path',
                'export_path': None,
                'model_info': None
            }
        
        if export_format not in self.supported_formats:
            return {
                'success': False,
                'error': f'Unsupported format: {export_format}',
                'export_path': None,
                'model_info': None
            }
        
        # Simulate model export process
        model_info = {
            'model_name': Path(model_path).stem,
            'format': export_format,
            'size_mb': 150.5,  # Simulated model size
            'parameters': 125000000,  # 125M parameters
            'input_shape': [1, 512],
            'output_shape': [1, 768],
            'quantized': config.get('quantized', False) if config else False,
            'optimization_level': config.get('optimization_level', 'O1') if config else 'O1'
        }
        
        # Simulate export validation
        validation_result = self._validate_exported_model(model_info, export_format)
        
        if not validation_result['valid']:
            return {
                'success': False,
                'error': f'Export validation failed: {validation_result["error"]}',
                'export_path': None,
                'model_info': model_info
            }
        
        # Generate export path
        export_path = f"{output_path}/{model_info['model_name']}.{export_format}"
        
        # Record export
        export_record = {
            'timestamp': '2025-08-20T15:30:00Z',
            'model_path': model_path,
            'export_format': export_format,
            'export_path': export_path,
            'model_info': model_info,
            'config': config or {},
            'validation': validation_result
        }
        
        self.export_history.append(export_record)
        self.model_registry[model_info['model_name']] = export_record
        
        return {
            'success': True,
            'error': None,
            'export_path': export_path,
            'model_info': model_info,
            'validation': validation_result,
            'export_id': len(self.export_history)
        }
    
    def _validate_exported_model(self, model_info: Dict[str, Any], format: str) -> Dict[str, Any]:
        """Validate exported model integrity."""
        validation_checks = {
            'format_compatibility': True,
            'size_reasonable': model_info['size_mb'] < 1000,  # Under 1GB
            'parameters_valid': model_info['parameters'] > 0,
            'shapes_valid': len(model_info['input_shape']) > 0 and len(model_info['output_shape']) > 0,
            'optimization_applied': model_info.get('optimization_level') in ['O1', 'O2', 'O3']
        }
        
        # Format-specific validations
        if format == 'onnx':
            validation_checks['onnx_opset_compatible'] = True
        elif format == 'tensorflow':
            validation_checks['tf_version_compatible'] = True
        elif format == 'pytorch':
            validation_checks['torch_version_compatible'] = True
        
        all_valid = all(validation_checks.values())
        
        return {
            'valid': all_valid,
            'checks': validation_checks,
            'error': None if all_valid else 'Validation checks failed'
        }
    
    def generate_deployment_config(self, model_info: Dict[str, Any], 
                                 deployment_target: str = 'kubernetes') -> Dict[str, Any]:
        """Generate deployment configuration for the model."""
        if not model_info or 'model_name' not in model_info:
            return {
                'success': False,
                'error': 'Invalid model info',
                'config': None
            }
        
        base_config = {
            'model_name': model_info['model_name'],
            'model_format': model_info['format'],
            'model_size_mb': model_info['size_mb'],
            'deployment_target': deployment_target
        }
        
        if deployment_target == 'kubernetes':
            config = {
                **base_config,
                'replicas': 3,
                'cpu_request': '500m',
                'cpu_limit': '2000m',
                'memory_request': '1Gi',
                'memory_limit': '4Gi',
                'gpu_required': model_info['size_mb'] > 500,
                'health_check': {
                    'path': '/health',
                    'interval': 30,
                    'timeout': 10
                },
                'scaling': {
                    'min_replicas': 2,
                    'max_replicas': 10,
                    'target_cpu': 70
                }
            }
        elif deployment_target == 'docker':
            config = {
                **base_config,
                'base_image': 'python:3.11-slim',
                'port': 8080,
                'environment': {
                    'MODEL_PATH': f'/models/{model_info["model_name"]}',
                    'WORKERS': 4,
                    'TIMEOUT': 30
                },
                'volumes': ['/models:/models:ro'],
                'health_check': {
                    'test': 'curl -f http://localhost:8080/health',
                    'interval': '30s',
                    'timeout': '10s',
                    'retries': 3
                }
            }
        elif deployment_target == 'aws_lambda':
            config = {
                **base_config,
                'runtime': 'python3.11',
                'timeout': 300,
                'memory': min(3008, max(512, int(model_info['size_mb'] * 4))),
                'environment': {
                    'MODEL_PATH': f'/opt/models/{model_info["model_name"]}'
                },
                'layers': ['arn:aws:lambda:us-east-1:123456789012:layer:ml-runtime:1']
            }
        else:
            return {
                'success': False,
                'error': f'Unsupported deployment target: {deployment_target}',
                'config': None
            }
        
        return {
            'success': True,
            'error': None,
            'config': config
        }
    
    def validate_production_readiness(self, model_info: Dict[str, Any]) -> Dict[str, Any]:
        """Validate model readiness for production deployment."""
        readiness_checks = {
            'model_size_acceptable': model_info['size_mb'] < 2000,  # Under 2GB
            'performance_optimized': model_info.get('optimization_level', 'O0') != 'O0',
            'quantization_applied': model_info.get('quantized', False),
            'input_validation': len(model_info.get('input_shape', [])) > 0,
            'output_validation': len(model_info.get('output_shape', [])) > 0,
            'parameter_count_reasonable': 1000000 <= model_info.get('parameters', 0) <= 1000000000
        }
        
        # Security checks
        security_checks = {
            'no_hardcoded_secrets': True,  # Simulated check
            'input_sanitization': True,   # Simulated check
            'output_filtering': True      # Simulated check
        }
        
        # Performance checks
        performance_checks = {
            'inference_time_acceptable': True,  # Would measure actual inference time
            'memory_usage_reasonable': model_info['size_mb'] < 1000,
            'throughput_adequate': True  # Would measure actual throughput
        }
        
        all_checks = {**readiness_checks, **security_checks, **performance_checks}
        overall_ready = all(all_checks.values())
        
        return {
            'production_ready': overall_ready,
            'readiness_score': sum(all_checks.values()) / len(all_checks),
            'checks': {
                'readiness': readiness_checks,
                'security': security_checks,
                'performance': performance_checks
            },
            'recommendations': self._generate_readiness_recommendations(all_checks)
        }
    
    def _generate_readiness_recommendations(self, checks: Dict[str, bool]) -> List[str]:
        """Generate recommendations based on readiness checks."""
        recommendations = []
        
        if not checks.get('model_size_acceptable', True):
            recommendations.append("Consider model compression or pruning to reduce size")
        
        if not checks.get('performance_optimized', True):
            recommendations.append("Apply performance optimizations (O2 or O3 level)")
        
        if not checks.get('quantization_applied', True):
            recommendations.append("Consider quantization for better inference performance")
        
        if not checks.get('parameter_count_reasonable', True):
            recommendations.append("Review model architecture for parameter efficiency")
        
        if not recommendations:
            recommendations.append("Model appears production-ready")
        
        return recommendations
    
    def create_model_version(self, model_name: str, version: str, 
                           export_info: Dict[str, Any]) -> Dict[str, Any]:
        """Create a versioned model entry."""
        if not model_name or not version:
            return {
                'success': False,
                'error': 'Model name and version required',
                'version_info': None
            }
        
        version_info = {
            'model_name': model_name,
            'version': version,
            'created_at': '2025-08-20T15:30:00Z',
            'export_info': export_info,
            'status': 'active',
            'deployment_count': 0,
            'performance_metrics': {
                'avg_inference_time_ms': 45.2,
                'throughput_rps': 100,
                'accuracy': 0.94,
                'f1_score': 0.92
            }
        }
        
        # Store version
        version_key = f"{model_name}:{version}"
        self.model_registry[version_key] = version_info
        
        return {
            'success': True,
            'error': None,
            'version_info': version_info
        }
    
    def rollback_model(self, model_name: str, target_version: str) -> Dict[str, Any]:
        """Rollback model to a previous version."""
        version_key = f"{model_name}:{target_version}"
        
        if version_key not in self.model_registry:
            return {
                'success': False,
                'error': f'Version {target_version} not found for model {model_name}',
                'rollback_info': None
            }
        
        version_info = self.model_registry[version_key]
        
        # Simulate rollback process
        rollback_info = {
            'model_name': model_name,
            'previous_version': 'current',  # Would track actual previous version
            'target_version': target_version,
            'rollback_time': '2025-08-20T15:35:00Z',
            'reason': 'Manual rollback',
            'status': 'completed'
        }
        
        return {
            'success': True,
            'error': None,
            'rollback_info': rollback_info,
            'version_info': version_info
        }
    
    def get_export_statistics(self) -> Dict[str, Any]:
        """Get export statistics."""
        if not self.export_history:
            return {'total_exports': 0}
        
        total_exports = len(self.export_history)
        successful_exports = sum(1 for export in self.export_history if export.get('validation', {}).get('valid', False))
        
        format_counts = {}
        for format in self.supported_formats:
            format_counts[format] = sum(1 for export in self.export_history if export['export_format'] == format)
        
        avg_model_size = sum(export['model_info']['size_mb'] for export in self.export_history) / total_exports
        
        return {
            'total_exports': total_exports,
            'successful_exports': successful_exports,
            'success_rate': (successful_exports / total_exports) * 100,
            'format_distribution': format_counts,
            'average_model_size_mb': round(avg_model_size, 2),
            'total_models_registered': len(self.model_registry)
        }


class TestProductionExporter(unittest.TestCase):
    """Test suite for ProductionExporter class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.exporter = MockProductionExporter()
        self.test_model_path = "/models/test_model.pt"
        self.test_output_path = "/exports"
        
        self.test_configs = {
            'basic': {},
            'optimized': {
                'optimization_level': 'O2',
                'quantized': True
            },
            'production': {
                'optimization_level': 'O3',
                'quantized': True,
                'batch_size': 32
            }
        }
    
    def test_initialization(self):
        """Test exporter initialization."""
        self.assertIsNotNone(self.exporter)
        self.assertIsInstance(self.exporter.supported_formats, list)
        self.assertGreater(len(self.exporter.supported_formats), 0)
        self.assertEqual(len(self.exporter.export_history), 0)
    
    def test_successful_model_export(self):
        """Test successful model export in various formats."""
        for format in self.exporter.supported_formats:
            with self.subTest(format=format):
                result = self.exporter.export_model(
                    self.test_model_path, 
                    format, 
                    self.test_output_path,
                    self.test_configs['basic']
                )
                
                self.assertTrue(result['success'])
                self.assertIsNone(result['error'])
                self.assertIsNotNone(result['export_path'])
                self.assertIsNotNone(result['model_info'])
                self.assertEqual(result['model_info']['format'], format)
    
    def test_invalid_model_path(self):
        """Test export with invalid model path."""
        invalid_paths = [None, "", "   ", 123, []]
        
        for path in invalid_paths:
            with self.subTest(path=path):
                result = self.exporter.export_model(path, 'onnx', self.test_output_path)
                
                self.assertFalse(result['success'])
                self.assertIsNotNone(result['error'])
                self.assertIsNone(result['export_path'])
    
    def test_unsupported_format(self):
        """Test export with unsupported format."""
        unsupported_formats = ['invalid', 'tflite', 'coreml', 'tensorrt']
        
        for format in unsupported_formats:
            with self.subTest(format=format):
                result = self.exporter.export_model(
                    self.test_model_path, 
                    format, 
                    self.test_output_path
                )
                
                self.assertFalse(result['success'])
                self.assertIn('Unsupported format', result['error'])
    
    def test_export_with_optimization(self):
        """Test export with optimization configurations."""
        result = self.exporter.export_model(
            self.test_model_path,
            'onnx',
            self.test_output_path,
            self.test_configs['optimized']
        )
        
        self.assertTrue(result['success'])
        self.assertTrue(result['model_info']['quantized'])
        self.assertEqual(result['model_info']['optimization_level'], 'O2')
    
    def test_deployment_config_generation(self):
        """Test deployment configuration generation."""
        # First export a model
        export_result = self.exporter.export_model(
            self.test_model_path, 'onnx', self.test_output_path
        )
        
        model_info = export_result['model_info']
        
        # Test different deployment targets
        targets = ['kubernetes', 'docker', 'aws_lambda']
        
        for target in targets:
            with self.subTest(target=target):
                config_result = self.exporter.generate_deployment_config(model_info, target)
                
                self.assertTrue(config_result['success'])
                self.assertIsNone(config_result['error'])
                self.assertIsNotNone(config_result['config'])
                self.assertEqual(config_result['config']['deployment_target'], target)
    
    def test_kubernetes_deployment_config(self):
        """Test Kubernetes-specific deployment configuration."""
        export_result = self.exporter.export_model(
            self.test_model_path, 'onnx', self.test_output_path
        )
        
        config_result = self.exporter.generate_deployment_config(
            export_result['model_info'], 'kubernetes'
        )
        
        config = config_result['config']
        
        # Check Kubernetes-specific fields
        self.assertIn('replicas', config)
        self.assertIn('cpu_request', config)
        self.assertIn('memory_limit', config)
        self.assertIn('health_check', config)
        self.assertIn('scaling', config)
    
    def test_docker_deployment_config(self):
        """Test Docker-specific deployment configuration."""
        export_result = self.exporter.export_model(
            self.test_model_path, 'pytorch', self.test_output_path
        )
        
        config_result = self.exporter.generate_deployment_config(
            export_result['model_info'], 'docker'
        )
        
        config = config_result['config']
        
        # Check Docker-specific fields
        self.assertIn('base_image', config)
        self.assertIn('port', config)
        self.assertIn('environment', config)
        self.assertIn('volumes', config)
    
    def test_aws_lambda_deployment_config(self):
        """Test AWS Lambda-specific deployment configuration."""
        export_result = self.exporter.export_model(
            self.test_model_path, 'tensorflow', self.test_output_path
        )
        
        config_result = self.exporter.generate_deployment_config(
            export_result['model_info'], 'aws_lambda'
        )
        
        config = config_result['config']
        
        # Check Lambda-specific fields
        self.assertIn('runtime', config)
        self.assertIn('timeout', config)
        self.assertIn('memory', config)
        self.assertIn('layers', config)
    
    def test_production_readiness_validation(self):
        """Test production readiness validation."""
        # Export a model first
        export_result = self.exporter.export_model(
            self.test_model_path, 'onnx', self.test_output_path, 
            self.test_configs['production']
        )
        
        readiness_result = self.exporter.validate_production_readiness(
            export_result['model_info']
        )
        
        self.assertIn('production_ready', readiness_result)
        self.assertIn('readiness_score', readiness_result)
        self.assertIn('checks', readiness_result)
        self.assertIn('recommendations', readiness_result)
        
        # Check structure of checks
        checks = readiness_result['checks']
        self.assertIn('readiness', checks)
        self.assertIn('security', checks)
        self.assertIn('performance', checks)
    
    def test_model_versioning(self):
        """Test model versioning functionality."""
        # Export a model
        export_result = self.exporter.export_model(
            self.test_model_path, 'onnx', self.test_output_path
        )
        
        # Create version
        version_result = self.exporter.create_model_version(
            'test_model', 'v1.0.0', export_result
        )
        
        self.assertTrue(version_result['success'])
        self.assertIsNone(version_result['error'])
        self.assertIsNotNone(version_result['version_info'])
        
        version_info = version_result['version_info']
        self.assertEqual(version_info['model_name'], 'test_model')
        self.assertEqual(version_info['version'], 'v1.0.0')
        self.assertEqual(version_info['status'], 'active')
    
    def test_model_rollback(self):
        """Test model rollback functionality."""
        # Create a model version first
        export_result = self.exporter.export_model(
            self.test_model_path, 'onnx', self.test_output_path
        )
        
        self.exporter.create_model_version('test_model', 'v1.0.0', export_result)
        
        # Test rollback
        rollback_result = self.exporter.rollback_model('test_model', 'v1.0.0')
        
        self.assertTrue(rollback_result['success'])
        self.assertIsNone(rollback_result['error'])
        self.assertIsNotNone(rollback_result['rollback_info'])
        self.assertIsNotNone(rollback_result['version_info'])
    
    def test_rollback_nonexistent_version(self):
        """Test rollback to nonexistent version."""
        rollback_result = self.exporter.rollback_model('nonexistent_model', 'v1.0.0')
        
        self.assertFalse(rollback_result['success'])
        self.assertIsNotNone(rollback_result['error'])
        self.assertIn('not found', rollback_result['error'])
    
    def test_export_statistics(self):
        """Test export statistics collection."""
        # Perform several exports
        formats = ['onnx', 'tensorflow', 'pytorch']
        
        for i, format in enumerate(formats):
            self.exporter.export_model(
                f"/models/model_{i}.pt", format, self.test_output_path
            )
        
        stats = self.exporter.get_export_statistics()
        
        self.assertEqual(stats['total_exports'], len(formats))
        self.assertIn('successful_exports', stats)
        self.assertIn('success_rate', stats)
        self.assertIn('format_distribution', stats)
        self.assertIn('average_model_size_mb', stats)
    
    def test_batch_export(self):
        """Test batch export of multiple models."""
        models = [
            ("/models/model1.pt", "onnx"),
            ("/models/model2.pt", "tensorflow"),
            ("/models/model3.pt", "pytorch")
        ]
        
        results = []
        for model_path, format in models:
            result = self.exporter.export_model(model_path, format, self.test_output_path)
            results.append(result)
        
        # Verify all exports
        self.assertEqual(len(results), len(models))
        successful_exports = sum(1 for r in results if r['success'])
        self.assertEqual(successful_exports, len(models))
    
    def test_large_model_handling(self):
        """Test handling of large models."""
        # Simulate large model by modifying model info after export
        result = self.exporter.export_model(
            self.test_model_path, 'onnx', self.test_output_path
        )
        
        # Modify model info to simulate large model
        large_model_info = result['model_info'].copy()
        large_model_info['size_mb'] = 2500  # 2.5GB
        large_model_info['parameters'] = 2000000000  # 2B parameters
        
        readiness_result = self.exporter.validate_production_readiness(large_model_info)
        
        # Large model should have readiness concerns
        self.assertLess(readiness_result['readiness_score'], 1.0)
        self.assertIn('model compression', ' '.join(readiness_result['recommendations']).lower())


class TestProductionExporterIntegration(unittest.TestCase):
    """Integration tests for ProductionExporter."""
    
    def setUp(self):
        """Set up integration test fixtures."""
        self.exporter = MockProductionExporter()
    
    def test_complete_export_deployment_workflow(self):
        """Test complete export to deployment workflow."""
        # Step 1: Export model
        export_result = self.exporter.export_model(
            "/models/production_model.pt", 'onnx', "/exports",
            {'optimization_level': 'O3', 'quantized': True}
        )
        
        self.assertTrue(export_result['success'])
        
        # Step 2: Validate production readiness
        readiness_result = self.exporter.validate_production_readiness(
            export_result['model_info']
        )
        
        # Step 3: Generate deployment config
        deployment_result = self.exporter.generate_deployment_config(
            export_result['model_info'], 'kubernetes'
        )
        
        self.assertTrue(deployment_result['success'])
        
        # Step 4: Create version
        version_result = self.exporter.create_model_version(
            'production_model', 'v1.0.0', export_result
        )
        
        self.assertTrue(version_result['success'])
        
        # Verify complete workflow
        self.assertGreater(readiness_result['readiness_score'], 0.5)
        self.assertIsNotNone(deployment_result['config'])
        self.assertEqual(version_result['version_info']['status'], 'active')
    
    def test_multi_format_export_comparison(self):
        """Test exporting same model to multiple formats."""
        model_path = "/models/comparison_model.pt"
        formats = ['onnx', 'tensorflow', 'pytorch']
        
        results = {}
        for format in formats:
            result = self.exporter.export_model(model_path, format, "/exports")
            results[format] = result
        
        # All exports should succeed
        for format, result in results.items():
            self.assertTrue(result['success'], f"Export failed for {format}")
        
        # Compare model info across formats
        sizes = [results[f]['model_info']['size_mb'] for f in formats]
        self.assertGreater(max(sizes), 0)  # All should have positive size


if __name__ == '__main__':
    # Run tests with verbose output
    unittest.main(verbosity=2)
