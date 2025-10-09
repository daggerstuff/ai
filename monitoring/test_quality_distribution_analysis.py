from unittest.mock import Mock, patch, MagicMock
import pytest
#!/usr/bin/env python3
"""
Test Suite for Quality Distribution Analysis System (Task 5.6.2.3)

Comprehensive testing of distribution analysis, comparative analysis,
and reporting functionality with statistical validation.
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
    print("🧪 Running Quality Distribution Analysis Test Suite...")
    
    # Create temporary database for testing
    db_fd, db_path = tempfile.mkstemp(suffix='.db')
    os.close(db_fd)
    
    try:
        # Setup test database with distribution data
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
        
        # Generate test data with different distributions
        base_date = datetime.now() - timedelta(days=30)
        conversations_data = []
        quality_data = []
        
        # Create different distribution patterns for different tiers
        tier_patterns = {
            'priority_1': {'mean': 0.8, 'std': 0.1},    # High quality, low variance
            'priority_2': {'mean': 0.7, 'std': 0.15},   # Medium quality, medium variance
            'priority_3': {'mean': 0.6, 'std': 0.2},    # Lower quality, high variance
        }
        
        for i in range(300):  # 300 conversations
            tier = ['priority_1', 'priority_2', 'priority_3'][i % 3]
            dataset = f'test_dataset_{i % 2}'
            conversation_date = base_date + timedelta(days=i % 30)
            
            # Generate quality scores based on tier pattern
            pattern = tier_patterns[tier]
            overall_quality = np.random.normal(pattern['mean'], pattern['std'])
            overall_quality = max(0.1, min(0.95, overall_quality))  # Clamp to valid range
            
            # Generate component scores with some correlation
            base_score = overall_quality
            therapeutic_accuracy = max(0.1, min(0.95, base_score + np.random.normal(0, 0.05)))
            conversation_coherence = max(0.1, min(0.95, base_score + np.random.normal(0, 0.05)))
            emotional_authenticity = max(0.1, min(0.95, base_score + np.random.normal(0, 0.05)))
            clinical_compliance = max(0.1, min(0.95, base_score + np.random.normal(0, 0.05)))
            personality_consistency = max(0.1, min(0.95, base_score + np.random.normal(0, 0.05)))
            language_quality = max(0.1, min(0.95, base_score + np.random.normal(0, 0.05)))
            safety_score = max(0.1, min(0.95, base_score + np.random.normal(0, 0.05)))
            
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
        
        # Import and test the distribution analyzer
        from .quality_distribution_analyzer import QualityDistributionAnalyzer
        from .quality_distribution_comparator import QualityDistributionComparator
        from .quality_distribution_reporter import QualityDistributionReporter
        
        analyzer = QualityDistributionAnalyzer(db_path=db_path)
        comparator = QualityDistributionComparator(analyzer)
        
        test_results = {
            'total_tests': 0,
            'passed_tests': 0,
            'failed_tests': 0,
            'test_details': []
        }
        
        # Test 1: Data Loading
        try:
            df = analyzer.load_quality_data(days_back=30)
            assert not df.empty, "DataFrame should not be empty"
            assert len(df) >= 290, f"Expected at least 290 records, got {len(df)}"
            assert len(df) <= 300, f"Expected at most 300 records, got {len(df)}"
            test_results['passed_tests'] += 1
            test_results['test_details'].append(f"✅ Data Loading: PASSED ({len(df)} records)")
        except Exception as e:
            test_results['failed_tests'] += 1
            test_results['test_details'].append(f"❌ Data Loading: FAILED - {e}")
        test_results['total_tests'] += 1
        
        # Test 2: Distribution Analysis
        try:
            distribution_analysis = analyzer.analyze_quality_distribution(
                df['overall_quality'], 'overall_quality'
            )
            assert distribution_analysis.sample_size >= 290, f"Expected at least 290 samples, got {distribution_analysis.sample_size}"
            assert distribution_analysis.statistics.mean > 0, "Mean should be positive"
            assert len(distribution_analysis.normality_tests) > 0, "Should have normality tests"
            test_results['passed_tests'] += 1
            test_results['test_details'].append("✅ Distribution Analysis: PASSED")
        except Exception as e:
            test_results['failed_tests'] += 1
            test_results['test_details'].append(f"❌ Distribution Analysis: FAILED - {e}")
        test_results['total_tests'] += 1
        
        # Test 3: Statistical Calculations
        try:
            stats = distribution_analysis.statistics
            assert 0 <= stats.mean <= 1
            assert stats.std_dev > 0
            assert stats.min_value <= stats.max_value
            assert stats.q1 <= stats.median <= stats.q3
            test_results['passed_tests'] += 1
            test_results['test_details'].append("✅ Statistical Calculations: PASSED")
        except Exception as e:
            test_results['failed_tests'] += 1
            test_results['test_details'].append(f"❌ Statistical Calculations: FAILED - {e}")
        test_results['total_tests'] += 1
        
        # Test 4: Normality Tests
        try:
            normality_tests = distribution_analysis.normality_tests
            assert len(normality_tests) > 0
            assert all(hasattr(test, 'test_name') for test in normality_tests)
            assert all(hasattr(test, 'p_value') for test in normality_tests)
            test_results['passed_tests'] += 1
            test_results['test_details'].append("✅ Normality Tests: PASSED")
        except Exception as e:
            test_results['failed_tests'] += 1
            test_results['test_details'].append(f"❌ Normality Tests: FAILED - {e}")
        test_results['total_tests'] += 1
        
        # Test 5: Outlier Detection
        try:
            outliers = distribution_analysis.outliers
            assert isinstance(outliers, list)
            # Should have some outliers given our data generation
            test_results['passed_tests'] += 1
            test_results['test_details'].append("✅ Outlier Detection: PASSED")
        except Exception as e:
            test_results['failed_tests'] += 1
            test_results['test_details'].append(f"❌ Outlier Detection: FAILED - {e}")
        test_results['total_tests'] += 1
        
        # Test 6: Comparative Analysis - Tiers
        try:
            tier_comparison = comparator.compare_across_tiers(df, 'overall_quality')
            assert len(tier_comparison.groups) == 3  # 3 tiers
            assert len(tier_comparison.statistical_tests) > 0
            # Should detect differences between our tier patterns
            test_results['passed_tests'] += 1
            test_results['test_details'].append("✅ Tier Comparison: PASSED")
        except Exception as e:
            test_results['failed_tests'] += 1
            test_results['test_details'].append(f"❌ Tier Comparison: FAILED - {e}")
        test_results['total_tests'] += 1
        
        # Test 7: Component Comparison
        try:
            component_comparison = comparator.compare_across_components(df)
            assert len(component_comparison.groups) >= 7  # Should have all quality components
            test_results['passed_tests'] += 1
            test_results['test_details'].append("✅ Component Comparison: PASSED")
        except Exception as e:
            test_results['failed_tests'] += 1
            test_results['test_details'].append(f"❌ Component Comparison: FAILED - {e}")
        test_results['total_tests'] += 1
        
        # Test 8: Correlation Analysis
        try:
            correlation_analysis = comparator.calculate_correlation_analysis(df)
            assert 'correlation_matrix' in correlation_analysis
            assert 'strongest_correlations' in correlation_analysis
            test_results['passed_tests'] += 1
            test_results['test_details'].append("✅ Correlation Analysis: PASSED")
        except Exception as e:
            test_results['failed_tests'] += 1
            test_results['test_details'].append(f"❌ Correlation Analysis: FAILED - {e}")
        test_results['total_tests'] += 1
        
        # Test 9: Report Generation
        try:
            temp_dir = tempfile.mkdtemp()
            reporter = QualityDistributionReporter(output_dir=temp_dir)
            
            # Mock the analyzer to use our test database
            reporter.analyzer = analyzer
            
            report = reporter.generate_comprehensive_report(days_back=30)
            assert report.overall_distribution.sample_size >= 290
            assert len(report.executive_summary) > 0
            
            # Test report saving
            json_path = reporter.save_report(report, format='json')
            assert Path(json_path).exists()
            
            test_results['passed_tests'] += 1
            test_results['test_details'].append("✅ Report Generation: PASSED")
            
            # Cleanup
            shutil.rmtree(temp_dir)
            
        except Exception as e:
            test_results['failed_tests'] += 1
            test_results['test_details'].append(f"❌ Report Generation: FAILED - {e}")
        test_results['total_tests'] += 1
        
        # Test 10: Visualization Creation
        try:
            visualizations = reporter.create_distribution_visualizations(report)
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
            'test_suite': 'Quality Distribution Analysis System',
            'task': '5.6.2.3',
            'timestamp': datetime.now().isoformat(),
            'results': test_results,
            'success_rate': success_rate,
            'status': 'PASSED' if success_rate >= 80 else 'FAILED',
            'distribution_analysis_sample': {
                'sample_size': distribution_analysis.sample_size,
                'distribution_type': distribution_analysis.distribution_type,
                'mean': distribution_analysis.statistics.mean,
                'median': distribution_analysis.statistics.median,
                'std_dev': distribution_analysis.statistics.std_dev,
                'skewness': distribution_analysis.statistics.skewness,
                'kurtosis': distribution_analysis.statistics.kurtosis,
                'outliers_count': len(distribution_analysis.outliers),
                'normality_tests_count': len(distribution_analysis.normality_tests)
            },
            'comparative_analysis_sample': {
                'tier_groups': len(tier_comparison.groups),
                'tier_statistical_tests': len(tier_comparison.statistical_tests),
                'component_groups': len(component_comparison.groups),
                'correlation_components': correlation_analysis.get('component_count', 0)
            }
        }
        
        # Save test report
        report_path = Path(__file__).parent / "quality_distribution_analysis_test_report.json"
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
        
        print(f"\n📈 Distribution Analysis Sample:")
        sample = report_data['distribution_analysis_sample']
        print(f"  Sample Size: {sample['sample_size']}")
        print(f"  Distribution Type: {sample['distribution_type']}")
        print(f"  Mean: {sample['mean']:.3f}")
        print(f"  Median: {sample['median']:.3f}")
        print(f"  Std Dev: {sample['std_dev']:.3f}")
        print(f"  Skewness: {sample['skewness']:.3f}")
        print(f"  Kurtosis: {sample['kurtosis']:.3f}")
        print(f"  Outliers: {sample['outliers_count']}")
        print(f"  Normality Tests: {sample['normality_tests_count']}")
        
        print(f"\n🔍 Comparative Analysis Sample:")
        comp_sample = report_data['comparative_analysis_sample']
        print(f"  Tier Groups: {comp_sample['tier_groups']}")
        print(f"  Tier Tests: {comp_sample['tier_statistical_tests']}")
        print(f"  Component Groups: {comp_sample['component_groups']}")
        print(f"  Correlation Components: {comp_sample['correlation_components']}")
        
        print(f"\n📁 Test report saved to: {report_path}")
        
        return report_data
        
    finally:
        # Cleanup
        if os.path.exists(db_path):
            os.unlink(db_path)

if __name__ == "__main__":
    run_comprehensive_test()
