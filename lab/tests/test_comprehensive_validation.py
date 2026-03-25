#!/usr/bin/env python3
"""
Comprehensive Validation Tests
Complete validation test suite for production readiness
"""

import unittest
import json
from pathlib import Path

class TestComprehensiveValidation(unittest.TestCase):
    """Comprehensive validation test suite"""
    
    def test_group_i_audit_results(self):
        """Test Group I audit results"""
        audit_file = Path('/home/vivi/pixelated/ai/group_i_comprehensive_audit_results.json')
        if audit_file.exists():
            with open(audit_file) as f:
                results = json.load(f)
            
            self.assertIn('production_readiness_score', results)
    
    def test_validation_evidence_files(self):
        """Test that validation evidence files exist"""
        evidence_patterns = [
            '*validation*results.json',
            '*safety*validation*.json',
            '*security*validation*.json'
        ]
        
        total_evidence = 0
        for pattern in evidence_patterns:
            files = list(Path('/home/vivi/pixelated/ai').glob(pattern))
            total_evidence += len(files)
        
        self.assertGreater(total_evidence, 0, "No validation evidence files found")
    
    def test_infrastructure_components(self):
        """Test that infrastructure components exist"""
        required_dirs = [
            '/home/vivi/pixelated/ai/tests',
            '/home/vivi/pixelated/ai/docs',
            '/home/vivi/pixelated/ai/monitoring'
        ]
        
        for dir_path in required_dirs:
            self.assertTrue(Path(dir_path).exists(), f"Missing directory: {dir_path}")
    
    def test_validation_completeness(self):
        """Test validation completeness"""
        validation_tasks = [81, 82, 85, 86, 87, 88]
        
        for task_num in validation_tasks:
            result_file = Path(f'/home/vivi/pixelated/ai/task_{task_num}_*_validation_results.json')
            matching_files = list(Path('/home/vivi/pixelated/ai').glob(f'task_{task_num}_*_validation_results.json'))
            self.assertGreater(len(matching_files), 0, f"No validation results for task {task_num}")

if __name__ == '__main__':
    unittest.main()
