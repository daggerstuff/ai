from unittest.mock import Mock, patch, MagicMock
import pytest
#!/usr/bin/env python3
"""
Test Suite for Quality Comparison System (Task 5.6.2.5)

Comprehensive testing of quality comparison, benchmarking, and reporting
functionality with statistical validation and performance testing.
"""

import pandas as pd
import numpy as np
import sqlite3
import tempfile
import json
import shutil
from .datetime import datetime, timedelta
from .pathlib import Path
import sys
import os

# Add the monitoring directory to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def run_comprehensive_test():
    """Run comprehensive test suite and generate report."""
    print("🧪 Running Quality Comparison System Test Suite...")
    
    # Create temporary database for testing
    db_fd, db_path = tempfile.mkstemp(suffix='.db')
    os.close(db_fd)
    
    try:
        # Setup test database with comparison data
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # Create tables
        cursor.execute("""
        CREATE TABLE conversations (
            id INTEGER PRIMARY KEY,
            tier TEXT,
            dataset_name TEXT,
            created_at TEXT,
            conversation_length INTEGER
        )
        """)
        
        cursor.execute("""
        CREATE TABLE quality_metrics (
            id INTEGER PRIMARY KEY,
            conversation_id INTEGER,
            therapeutic_accuracy REAL,
            conversation_coherence REAL,
            emotional_authenticity REAL,
            clinical_compliance REAL,
            personality_consistency REAL,
            language_quality REAL,
            safety_score REAL,
            overall_quality REAL,
            validated_at TEXT,
            FOREIGN KEY (conversation_id) REFERENCES conversations (id)
        )
        """)
        
        # Generate test data with clear differences between tiers and datasets
        base_date = datetime.now() - timedelta(days=30)
        conversations_data = []
        quality_data = []
        
        # Create different quality patterns for different tiers and datasets
        tier_patterns = {
            'priority_1': {'mean': 0.85, 'std': 0.08},    # High quality
            'priority_2': {'mean': 0.75, 'std': 0.10},    # Medium quality
            'priority_3': {'mean': 0.65, 'std': 0.12},    # Lower quality
        }
        
        dataset_patterns = {
            'high_quality_dataset': {'mean': 0.80, 'std': 0.09},
            'medium_quality_dataset': {'mean': 0.70, 'std': 0.11},
            'low_quality_dataset': {'mean': 0.60, 'std': 0.13},
        }
        
        for i in range(300):  # 300 conversations
            tier = ['priority_1', 'priority_2', 'priority_3'][i % 3]
            dataset = ['high_quality_dataset', 'medium_quality_dataset', 'low_quality_dataset'][i % 3]
            conversation_date = base_date + timedelta(days=i % 30)
            
            # Generate quality scores based on tier and dataset patterns
            tier_pattern = tier_patterns[tier]
            dataset_pattern = dataset_patterns[dataset]
            
            # Combine tier and dataset effects
            base_quality = (tier_pattern['mean'] + dataset_pattern['mean']) / 2
            noise = np.random.normal(0, (tier_pattern['std'] + dataset_pattern['std']) / 2)
            overall_quality = max(0.1, min(0.95, base_quality + noise))
            
            # Generate component scores with some correlation
            therapeutic_accuracy = max(0.1, min(0.95, overall_quality + np.random.normal(0, 0.05)))
            conversation_coherence = max(0.1, min(0.95, overall_quality + np.random.normal(0, 0.05)))
            emotional_authenticity = max(0.1, min(0.95, overall_quality + np.random.normal(0, 0.05)))
            clinical_compliance = max(0.1, min(0.95, overall_quality + np.random.normal(0, 0.05)))
            personality_consistency = max(0.1, min(0.95, overall_quality + np.random.normal(0, 0.05)))
            language_quality = max(0.1, min(0.95, overall_quality + np.random.normal(0, 0.05)))
            safety_score = max(0.1, min(0.95, overall_quality + np.random.normal(0, 0.05)))
            
            conversations_data.append((
                i + 1, tier, dataset, 
                conversation_date.isoformat(), 
                150 + (i % 100)
            ))
            
            quality_data.append((
                i + 1, i + 1,
                therapeutic_accuracy, conversation_coherence, emotional_authenticity,
                clinical_compliance, personality_consistency, language_quality, safety_score,
                overall_quality,
                (conversation_date + timedelta(minutes=15)).isoformat()
            ))
        
        cursor.executemany(
            "INSERT INTO conversations (id, tier, dataset_name, created_at, conversation_length) VALUES (?, ?, ?, ?, ?)",
            conversations_data
        )
        
        cursor.executemany(
            """INSERT INTO quality_metrics 
            (id, conversation_id, therapeutic_accuracy, conversation_coherence, 
             emotional_authenticity, clinical_compliance, personality_consistency, 
             language_quality, safety_score, overall_quality, validated_at) 
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            quality_data
        )
        
        conn.commit()
        conn.close()
        
        # Import and test the quality comparator
        from .quality_comparator import QualityComparator
        from .quality_comparison_reporter import QualityComparisonReporter
        
        comparator = QualityComparator(db_path=db_path)
        
        test_results = {
            'total_tests': 0,
            'passed_tests': 0,
            'failed_tests': 0,
            'test_details': []
        }
        
        # Test 1: Data Loading
        try:
            df = comparator.load_comparison_data(days_back=30)
            assert not df.empty, "DataFrame should not be empty"
            # Account for timing precision - should be close to 300 but may vary slightly due to date filtering
            assert len(df) >= 290, f"Expected at least 290 records, got {len(df)}"
            assert len(df) <= 300, f"Expected at most 300 records, got {len(df)}"
            # Verify required columns exist
            required_columns = ['tier', 'dataset_name', 'overall_quality', 'therapeutic_accuracy']
            for col in required_columns:
                assert col in df.columns, f"Missing required column: {col}"
            # Verify no null values in critical columns
            assert df['overall_quality'].notna().all(), "Found null values in overall_quality"
            test_results['passed_tests'] += 1
            test_results['test_details'].append(f"✅ Data Loading: PASSED ({len(df)} records loaded)")
        except Exception as e:
            test_results['failed_tests'] += 1
            test_results['test_details'].append(f"❌ Data Loading: FAILED - {e}")
        test_results['total_tests'] += 1
        
        # Test 2: Tier Comparisons
        try:
            tier_comparisons = comparator.compare_tiers(df)
            assert len(tier_comparisons) > 0  # Should have tier comparisons
            assert all(hasattr(c, 'effect_size') for c in tier_comparisons)
            test_results['passed_tests'] += 1
            test_results['test_details'].append("✅ Tier Comparisons: PASSED")
        except Exception as e:
            test_results['failed_tests'] += 1
            test_results['test_details'].append(f"❌ Tier Comparisons: FAILED - {e}")
        test_results['total_tests'] += 1
        
        # Test 3: Dataset Comparisons
        try:
            dataset_comparisons = comparator.compare_datasets(df)
            assert len(dataset_comparisons) > 0  # Should have dataset comparisons
            assert all(hasattr(c, 'statistical_tests') for c in dataset_comparisons)
            test_results['passed_tests'] += 1
            test_results['test_details'].append("✅ Dataset Comparisons: PASSED")
        except Exception as e:
            test_results['failed_tests'] += 1
            test_results['test_details'].append(f"❌ Dataset Comparisons: FAILED - {e}")
        test_results['total_tests'] += 1
        
        # Test 4: Component Comparisons
        try:
            component_comparisons = comparator.compare_components(df)
            assert len(component_comparisons) > 0  # Should have component comparisons
            test_results['passed_tests'] += 1
            test_results['test_details'].append("✅ Component Comparisons: PASSED")
        except Exception as e:
            test_results['failed_tests'] += 1
            test_results['test_details'].append(f"❌ Component Comparisons: FAILED - {e}")
        test_results['total_tests'] += 1
        
        # Test 5: Benchmark Analysis
        try:
            benchmark_analyses = comparator.perform_benchmark_analysis(df)
            assert len(benchmark_analyses) > 0  # Should have benchmark analyses
            assert all(hasattr(b, 'performance_gap') for b in benchmark_analyses)
            test_results['passed_tests'] += 1
            test_results['test_details'].append("✅ Benchmark Analysis: PASSED")
        except Exception as e:
            test_results['failed_tests'] += 1
            test_results['test_details'].append(f"❌ Benchmark Analysis: FAILED - {e}")
        test_results['total_tests'] += 1
        
        # Test 6: Performance Rankings
        try:
            performance_rankings = comparator.calculate_performance_rankings(df)
            assert 'tiers' in performance_rankings
            assert 'datasets' in performance_rankings
            assert 'components' in performance_rankings
            test_results['passed_tests'] += 1
            test_results['test_details'].append("✅ Performance Rankings: PASSED")
        except Exception as e:
            test_results['failed_tests'] += 1
            test_results['test_details'].append(f"❌ Performance Rankings: FAILED - {e}")
        test_results['total_tests'] += 1
        
        # Test 7: Reporter Initialization
        try:
            temp_dir = tempfile.mkdtemp()
            reporter = QualityComparisonReporter(output_dir=temp_dir)
            assert reporter.comparator is not None
            test_results['passed_tests'] += 1
            test_results['test_details'].append("✅ Reporter Initialization: PASSED")
        except Exception as e:
            test_results['failed_tests'] += 1
            test_results['test_details'].append(f"❌ Reporter Initialization: FAILED - {e}")
        test_results['total_tests'] += 1
        
        # Test 8: Comprehensive Report Generation
        try:
            # Mock the comparator to use our test database
            reporter.comparator = comparator
            
            report = reporter.generate_comprehensive_report(days_back=30)
            assert report is not None
            assert len(report.executive_summary) > 0
            assert len(report.tier_comparisons) > 0
            test_results['passed_tests'] += 1
            test_results['test_details'].append("✅ Comprehensive Report Generation: PASSED")
        except Exception as e:
            test_results['failed_tests'] += 1
            test_results['test_details'].append(f"❌ Comprehensive Report Generation: FAILED - {e}")
        test_results['total_tests'] += 1
        
        # Test 9: Report Saving
        try:
            json_path = reporter.save_report(report, format='json')
            assert Path(json_path).exists()
            test_results['passed_tests'] += 1
            test_results['test_details'].append("✅ Report Saving: PASSED")
            
            # Cleanup
            shutil.rmtree(temp_dir)
        except Exception as e:
            test_results['failed_tests'] += 1
            test_results['test_details'].append(f"❌ Report Saving: FAILED - {e}")
        test_results['total_tests'] += 1
        
        # Test 10: Visualization Creation
        try:
            visualizations = reporter.create_comparison_visualizations(report)
            assert isinstance(visualizations, dict)
            assert len(visualizations) > 0
            test_results['passed_tests'] += 1
            test_results['test_details'].append("✅ Visualization Creation: PASSED")
        except Exception as e:
            test_results['failed_tests'] += 1
            test_results['test_details'].append(f"❌ Visualization Creation: FAILED - {e}")
        test_results['total_tests'] += 1
        
        # Generate test report
        success_rate = (test_results['passed_tests'] / test_results['total_tests']) * 100
        
        report_data = {
            'test_suite': 'Quality Comparison System',
            'task': '5.6.2.5',
            'timestamp': datetime.now().isoformat(),
            'results': test_results,
            'success_rate': success_rate,
            'status': 'PASSED' if success_rate >= 80 else 'FAILED',
            'comparison_analysis_sample': {
                'tier_comparisons': len(tier_comparisons) if 'tier_comparisons' in locals() else 0,
                'dataset_comparisons': len(dataset_comparisons) if 'dataset_comparisons' in locals() else 0,
                'component_comparisons': len(component_comparisons) if 'component_comparisons' in locals() else 0,
                'benchmark_analyses': len(benchmark_analyses) if 'benchmark_analyses' in locals() else 0,
                'performance_rankings_dimensions': len(performance_rankings) if 'performance_rankings' in locals() else 0
            },
            'report_sample': {
                'tier_comparisons': len(report.tier_comparisons) if 'report' in locals() else 0,
                'dataset_comparisons': len(report.dataset_comparisons) if 'report' in locals() else 0,
                'component_comparisons': len(report.component_comparisons) if 'report' in locals() else 0,
                'benchmark_analyses': len(report.benchmark_analyses) if 'report' in locals() else 0,
                'executive_summary_items': len(report.executive_summary) if 'report' in locals() else 0,
                'action_items': len(report.action_items) if 'report' in locals() else 0
            }
        }
        
        # Save test report
        report_path = Path(__file__).parent / "quality_comparison_test_report.json"
        with open(report_path, 'w') as f:
            json.dump(report_data, f, indent=2)
        
        print(f"\n📊 Test Results Summary:")
        print(f"Total Tests: {test_results['total_tests']}")
        print(f"Passed: {test_results['passed_tests']}")
        print(f"Failed: {test_results['failed_tests']}")
        print(f"Success Rate: {success_rate:.1f}%")
        print(f"Status: {report_data['status']}")
        
        print(f"\n📋 Test Details:")
        for detail in test_results['test_details']:
            print(f"  {detail}")
        
        print(f"\n📈 Comparison Analysis Sample:")
        sample = report_data['comparison_analysis_sample']
        print(f"  Tier Comparisons: {sample['tier_comparisons']}")
        print(f"  Dataset Comparisons: {sample['dataset_comparisons']}")
        print(f"  Component Comparisons: {sample['component_comparisons']}")
        print(f"  Benchmark Analyses: {sample['benchmark_analyses']}")
        print(f"  Performance Ranking Dimensions: {sample['performance_rankings_dimensions']}")
        
        print(f"\n📊 Report Sample:")
        report_sample = report_data['report_sample']
        print(f"  Tier Comparisons: {report_sample['tier_comparisons']}")
        print(f"  Dataset Comparisons: {report_sample['dataset_comparisons']}")
        print(f"  Component Comparisons: {report_sample['component_comparisons']}")
        print(f"  Benchmark Analyses: {report_sample['benchmark_analyses']}")
        print(f"  Executive Summary Items: {report_sample['executive_summary_items']}")
        print(f"  Action Items: {report_sample['action_items']}")
        
        print(f"\n📁 Test report saved to: {report_path}")
        
        return report_data
        
    finally:
        # Cleanup
        if os.path.exists(db_path):
            os.unlink(db_path)

if __name__ == "__main__":
    run_comprehensive_test()
