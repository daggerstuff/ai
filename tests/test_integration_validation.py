#!/usr/bin/env python3
"""
Integration Validation Tests
Tests for integration between validation components
"""

import unittest
import json
from pathlib import Path

class TestIntegrationValidation(unittest.TestCase):
    """Test integration validation components"""
    
    def test_all_validation_results_exist(self):
        """Test that all validation result files exist"""
        required_files = [
            'task_81_safety_validation_results.json',
            'task_82_test_coverage_validation_results.json',
            'task_85_security_validation_results.json',
            'task_86_documentation_validation_results.json',
            'task_87_compliance_validation_results.json',
            'task_88_usability_validation_results.json'
        ]
        
        for filename in required_files:
            file_path = Path(f'/home/vivi/pixelated/ai/{filename}')
            self.assertTrue(file_path.exists(), f"Missing validation file: {filename}")
    
    def test_validation_scores_meet_thresholds(self):
        """Test that validation scores meet production thresholds"""
        validation_files = {
            'task_81_safety_validation_results.json': ('overall_safety_score', 90.0),
            'task_85_security_validation_results.json': ('overall_security_score', 90.0),
        }
        
        for filename, (score_key, threshold) in validation_files.items():
            file_path = Path(f'/home/vivi/pixelated/ai/{filename}')
            if file_path.exists():
                with open(file_path) as f:
                    results = json.load(f)
                
                if score_key in results:
                    self.assertGreaterEqual(results[score_key], threshold,
                                          f"{filename} score {results[score_key]} below threshold {threshold}")
    
    def test_production_readiness_flags(self):
        """Test that production readiness flags are set correctly"""
        validation_files = [
            'task_81_safety_validation_results.json',
            'task_85_security_validation_results.json'
        ]
        
        for filename in validation_files:
            file_path = Path(f'/home/vivi/pixelated/ai/{filename}')
            if file_path.exists():
                with open(file_path) as f:
                    results = json.load(f)
                
                if 'production_ready' in results:
                    self.assertTrue(results['production_ready'],
                                  f"{filename} not marked as production ready")

if __name__ == '__main__':
    unittest.main()
