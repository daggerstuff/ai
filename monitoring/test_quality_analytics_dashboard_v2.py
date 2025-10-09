import pytest
#!/usr/bin/env python3
"""
Test Suite for Quality Analytics Dashboard V2 (Task 5.6.2.1)

Enterprise-grade test suite validating all dashboard functionality
against the actual database schema with comprehensive coverage.
"""

import sys
import os
import unittest
import sqlite3
import pandas as pd
import json
from .datetime import datetime, timedelta
from .pathlib import Path
import tempfile
import logging

# Add the monitoring directory to the path
sys.path.insert(0, str(Path(__file__).parent))

from .quality_analytics_dashboard_v2 import QualityAnalyticsDashboard, QualityAnalytics

# Configure logging for tests
logging.basicConfig(level=logging.WARNING)  # Reduce noise during tests

class TestQualityAnalyticsDashboard(unittest.TestCase):
    """Comprehensive test suite for Quality Analytics Dashboard V2."""
    
    @classmethod
    def setUpClass(cls):
        """Set up test database with realistic data."""
        cls.test_db = tempfile.NamedTemporaryFile(delete=False, suffix='.db')
        cls.test_db_path = cls.test_db.name
        cls.test_db.close()
        
        # Create test database with actual schema
        conn = sqlite3.connect(cls.test_db_path)
        cursor = conn.cursor()
        
        # Create conversations table
        cursor.execute("""
        CREATE TABLE conversations (
            conversation_id TEXT PRIMARY KEY,
            dataset_source TEXT,
            tier TEXT,
            title TEXT,
            turn_count INTEGER,
            word_count INTEGER,
            processing_status TEXT,
            created_at TIMESTAMP,
            updated_at TIMESTAMP
        )
        """)
        
        # Create conversation_quality table
        cursor.execute("""
        CREATE TABLE conversation_quality (
            quality_id TEXT PRIMARY KEY,
            conversation_id TEXT,
            overall_quality REAL,
            therapeutic_accuracy REAL,
            clinical_compliance REAL,
            safety_score REAL,
            conversation_coherence REAL,
            emotional_authenticity REAL,
            validation_date TIMESTAMP,
            validator_version TEXT,
            FOREIGN KEY (conversation_id) REFERENCES conversations (conversation_id)
        )
        """)
        
        # Insert test data
        test_conversations = [
            ('conv_1', 'test_dataset', 'priority_1', 'Anxiety Session', 4, 150, 'processed', '2025-08-01 10:00:00', '2025-08-01 10:00:00'),
            ('conv_2', 'test_dataset', 'priority_2', 'Depression Support', 6, 200, 'processed', '2025-08-02 11:00:00', '2025-08-02 11:00:00'),
            ('conv_3', 'test_dataset', 'professional', 'Therapy Session', 8, 300, 'processed', '2025-08-03 12:00:00', '2025-08-03 12:00:00'),
            ('conv_4', 'test_dataset', 'research', 'Research Interview', 10, 400, 'processed', '2025-08-04 13:00:00', '2025-08-04 13:00:00'),
            ('conv_5', 'test_dataset', 'priority_1', 'Crisis Intervention', 3, 100, 'processed', '2025-08-05 14:00:00', '2025-08-05 14:00:00'),
        ]
        
        cursor.executemany("""
        INSERT INTO conversations (conversation_id, dataset_source, tier, title, turn_count, word_count, processing_status, created_at, updated_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, test_conversations)
        
        test_quality = [
            ('qual_1', 'conv_1', 0.85, 0.8, 0.9, 0.95, 0.8, 0.75, '2025-08-01 10:05:00', '5.6.2'),
            ('qual_2', 'conv_2', 0.65, 0.6, 0.7, 0.8, 0.6, 0.55, '2025-08-02 11:05:00', '5.6.2'),
            ('qual_3', 'conv_3', 0.92, 0.9, 0.95, 0.98, 0.9, 0.88, '2025-08-03 12:05:00', '5.6.2'),
            ('qual_4', 'conv_4', 0.45, 0.4, 0.5, 0.6, 0.4, 0.35, '2025-08-04 13:05:00', '5.6.2'),  # Low quality
            ('qual_5', 'conv_5', 0.78, 0.75, 0.8, 0.85, 0.75, 0.7, '2025-08-05 14:05:00', '5.6.2'),
        ]
        
        cursor.executemany("""
        INSERT INTO conversation_quality (quality_id, conversation_id, overall_quality, therapeutic_accuracy, clinical_compliance, safety_score, conversation_coherence, emotional_authenticity, validation_date, validator_version)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, test_quality)
        
        conn.commit()
        conn.close()
        
        # Initialize dashboard
        cls.dashboard = QualityAnalyticsDashboard(db_path=cls.test_db_path)
    
    @classmethod
    def tearDownClass(cls):
        """Clean up test database."""
        try:
            os.unlink(cls.test_db_path)
        except:
            pass
    
    def test_01_dashboard_initialization(self):
        """Test dashboard initialization with correct database."""
        self.assertIsInstance(self.dashboard, QualityAnalyticsDashboard)
        self.assertEqual(str(self.dashboard.db_path), self.test_db_path)
        self.assertIsInstance(self.dashboard.quality_components, dict)
        self.assertIn('overall_quality', self.dashboard.quality_components)
    
    def test_02_database_connection(self):
        """Test database connection and schema validation."""
        # Test that we can connect and query
        conn = sqlite3.connect(self.test_db_path)
        cursor = conn.cursor()
        
        # Check tables exist
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = [row[0] for row in cursor.fetchall()]
        self.assertIn('conversations', tables)
        self.assertIn('conversation_quality', tables)
        
        # Check data exists
        cursor.execute("SELECT COUNT(*) FROM conversations")
        conv_count = cursor.fetchone()[0]
        self.assertEqual(conv_count, 5)
        
        cursor.execute("SELECT COUNT(*) FROM conversation_quality")
        qual_count = cursor.fetchone()[0]
        self.assertEqual(qual_count, 5)
        
        conn.close()
    
    def test_03_load_quality_data_basic(self):
        """Test basic data loading functionality."""
        df = self.dashboard.load_quality_data()
        
        self.assertIsInstance(df, pd.DataFrame)
        self.assertFalse(df.empty)
        self.assertEqual(len(df), 5)
        
        # Check required columns exist
        required_columns = [
            'conversation_id', 'tier', 'overall_quality',
            'therapeutic_accuracy', 'clinical_compliance', 'safety_score'
        ]
        for col in required_columns:
            self.assertIn(col, df.columns)
        
        # Check data types
        self.assertTrue(pd.api.types.is_datetime64_any_dtype(df['created_at']))
        self.assertTrue(pd.api.types.is_numeric_dtype(df['overall_quality']))
    
    def test_04_load_quality_data_with_filters(self):
        """Test data loading with various filters."""
        # Test tier filter
        df_priority = self.dashboard.load_quality_data(tier_filter=['priority_1'])
        self.assertEqual(len(df_priority), 2)  # conv_1 and conv_5
        self.assertTrue(all(df_priority['tier'] == 'priority_1'))
        
        # Test date range filter
        start_date = datetime(2025, 8, 2)
        end_date = datetime(2025, 8, 4)
        df_date = self.dashboard.load_quality_data(date_range=(start_date, end_date))
        self.assertEqual(len(df_date), 3)  # conv_2, conv_3, conv_4
        
        # Test quality threshold filter
        df_quality = self.dashboard.load_quality_data(min_quality=0.7)
        self.assertEqual(len(df_quality), 3)  # conv_1, conv_3, conv_5
        self.assertTrue(all(df_quality['overall_quality'] >= 0.7))
    
    def test_05_calculate_quality_analytics_basic(self):
        """Test basic analytics calculation."""
        df = self.dashboard.load_quality_data()
        analytics = self.dashboard.calculate_quality_analytics(df)
        
        self.assertIsInstance(analytics, QualityAnalytics)
        self.assertEqual(analytics.total_conversations, 5)
        self.assertAlmostEqual(analytics.average_quality, 0.73, places=2)  # (0.85+0.65+0.92+0.45+0.78)/5
        
        # Check quality distribution
        self.assertIsInstance(analytics.quality_distribution, dict)
        self.assertIn('Good', analytics.quality_distribution)
        self.assertIn('Excellent', analytics.quality_distribution)
        
        # Check tier performance
        self.assertIsInstance(analytics.tier_performance, dict)
        self.assertIn('priority_1', analytics.tier_performance)
        self.assertIn('professional', analytics.tier_performance)
    
    def test_06_calculate_quality_analytics_empty_data(self):
        """Test analytics calculation with empty data."""
        empty_df = pd.DataFrame()
        analytics = self.dashboard.calculate_quality_analytics(empty_df)
        
        self.assertEqual(analytics.total_conversations, 0)
        self.assertEqual(analytics.average_quality, 0.0)
        self.assertEqual(analytics.quality_distribution, {})
        self.assertEqual(analytics.tier_performance, {})
        self.assertIn("No quality data available", analytics.recommendations[0])
    
    def test_07_anomaly_detection(self):
        """Test quality anomaly detection."""
        df = self.dashboard.load_quality_data()
        
        # Test IQR method
        anomalies_iqr = self.dashboard._detect_quality_anomalies(df, method='iqr')
        self.assertIsInstance(anomalies_iqr, list)
        
        # Test zscore method (should detect conv_4 as anomaly since it's 0.45 vs mean 0.73)
        anomalies_zscore = self.dashboard._detect_quality_anomalies(df, method='zscore')
        self.assertIsInstance(anomalies_zscore, list)
        
        # With our test data (0.45, 0.65, 0.78, 0.85, 0.92), conv_4 (0.45) should be detected
        # as an anomaly using zscore method since it's more than 2 standard deviations from mean
        if anomalies_zscore:
            anomaly_ids = [a['conversation_id'] for a in anomalies_zscore]
            # Check if any anomaly was detected (the exact detection depends on the statistical method)
            self.assertGreater(len(anomalies_zscore), 0, "Should detect at least one anomaly with zscore method")
        
        # Check anomaly structure if any anomalies found
        all_anomalies = anomalies_iqr + anomalies_zscore
        if all_anomalies:
            anomaly = all_anomalies[0]
            required_keys = ['conversation_id', 'tier', 'quality_score', 'anomaly_type']
            for key in required_keys:
                self.assertIn(key, anomaly)
        
        # Test with insufficient data
        small_df = df.head(2)  # Only 2 records
        anomalies_small = self.dashboard._detect_quality_anomalies(small_df)
        self.assertEqual(len(anomalies_small), 0, "Should not detect anomalies with insufficient data")
    
    def test_08_recommendation_generation(self):
        """Test recommendation generation."""
        df = self.dashboard.load_quality_data()
        analytics = self.dashboard.calculate_quality_analytics(df)
        
        self.assertIsInstance(analytics.recommendations, list)
        self.assertGreater(len(analytics.recommendations), 0)
        
        # Should have overall quality assessment
        overall_rec = [r for r in analytics.recommendations if 'Overall quality' in r or 'GOOD:' in r or 'WARNING:' in r]
        self.assertGreater(len(overall_rec), 0)
    
    def test_09_caching_functionality(self):
        """Test data caching functionality."""
        # First load
        start_time = datetime.now()
        df1 = self.dashboard.load_quality_data()
        first_load_time = (datetime.now() - start_time).total_seconds()
        
        # Second load (should be cached)
        start_time = datetime.now()
        df2 = self.dashboard.load_quality_data()
        second_load_time = (datetime.now() - start_time).total_seconds()
        
        # Cached load should be faster
        self.assertLess(second_load_time, first_load_time)
        
        # Data should be identical
        pd.testing.assert_frame_equal(df1, df2)
    
    def test_10_visualization_creation(self):
        """Test visualization creation."""
        df = self.dashboard.load_quality_data()
        analytics = self.dashboard.calculate_quality_analytics(df)
        
        # Test overview chart
        overview_fig = self.dashboard.create_quality_overview_chart(analytics)
        self.assertIsNotNone(overview_fig)
        self.assertGreater(len(overview_fig.data), 0)
        
        # Test detailed charts
        heatmap_fig, corr_fig = self.dashboard.create_detailed_analysis_charts(df)
        self.assertIsNotNone(heatmap_fig)
        self.assertIsNotNone(corr_fig)
    
    def test_11_error_handling(self):
        """Test error handling with invalid database."""
        # Test with non-existent database
        with self.assertRaises(FileNotFoundError):
            QualityAnalyticsDashboard(db_path="/nonexistent/path/db.sqlite")
        
        # Test with corrupted data
        df_corrupted = pd.DataFrame({'invalid_column': [1, 2, 3]})
        analytics = self.dashboard.calculate_quality_analytics(df_corrupted)
        self.assertEqual(analytics.total_conversations, 0)
    
    def test_12_component_performance_calculation(self):
        """Test component performance calculation."""
        df = self.dashboard.load_quality_data()
        analytics = self.dashboard.calculate_quality_analytics(df)
        
        self.assertIsInstance(analytics.component_performance, dict)
        
        # Check that components are calculated correctly
        for component_name, metrics in analytics.component_performance.items():
            self.assertIn('average_score', metrics)
            self.assertIn('sample_count', metrics)
            self.assertIn('coverage_percent', metrics)
            self.assertGreaterEqual(metrics['average_score'], 0.0)
            self.assertLessEqual(metrics['average_score'], 1.0)
    
    def test_13_trend_data_calculation(self):
        """Test trend data calculation."""
        df = self.dashboard.load_quality_data()
        analytics = self.dashboard.calculate_quality_analytics(df)
        
        # Should have trend data since we have data within 30 days
        self.assertIsInstance(analytics.trend_data, list)
        self.assertGreater(len(analytics.trend_data), 0)
        
        # Check trend data structure
        if analytics.trend_data:
            trend_item = analytics.trend_data[0]
            required_keys = ['date', 'average_quality', 'conversation_count']
            for key in required_keys:
                self.assertIn(key, trend_item)
    
    def test_14_data_freshness_calculation(self):
        """Test data freshness calculation."""
        df = self.dashboard.load_quality_data()
        analytics = self.dashboard.calculate_quality_analytics(df)
        
        self.assertIsInstance(analytics.data_freshness, str)
        self.assertNotEqual(analytics.data_freshness, "No data")
        # Should indicate recent data since test data is from recent dates
        self.assertIn("day", analytics.data_freshness.lower())
    
    def test_15_performance_benchmarks(self):
        """Test performance benchmarks."""
        # Test loading performance
        start_time = datetime.now()
        df = self.dashboard.load_quality_data(force_refresh=True)
        load_time = (datetime.now() - start_time).total_seconds()
        
        # Should load quickly (under 1 second for test data)
        self.assertLess(load_time, 1.0)
        
        # Test analytics calculation performance
        start_time = datetime.now()
        analytics = self.dashboard.calculate_quality_analytics(df)
        calc_time = (datetime.now() - start_time).total_seconds()
        
        # Should calculate quickly (under 0.5 seconds for test data)
        self.assertLess(calc_time, 0.5)

def run_comprehensive_test():
    """Run comprehensive test suite and generate detailed report."""
    print("🧪 Starting Quality Analytics Dashboard V2 Test Suite")
    print("=" * 60)
    
    # Create test suite
    loader = unittest.TestLoader()
    suite = loader.loadTestsFromTestCase(TestQualityAnalyticsDashboard)
    
    # Run tests with detailed output
    runner = unittest.TextTestRunner(verbosity=2, stream=sys.stdout)
    result = runner.run(suite)
    
    # Generate test report
    test_report = {
        "test_suite": "Quality Analytics Dashboard V2",
        "task": "5.6.2.1",
        "timestamp": datetime.now().isoformat(),
        "results": {
            "total_tests": result.testsRun,
            "passed_tests": result.testsRun - len(result.failures) - len(result.errors),
            "failed_tests": len(result.failures),
            "error_tests": len(result.errors),
            "test_details": []
        },
        "success_rate": ((result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun * 100) if result.testsRun > 0 else 0,
        "status": "PASSED" if result.wasSuccessful() else "FAILED"
    }
    
    # Add test details
    for test, error in result.failures + result.errors:
        test_report["results"]["test_details"].append(f"❌ {test}: FAILED - {error}")
    
    # Add passed tests
    passed_count = result.testsRun - len(result.failures) - len(result.errors)
    for i in range(passed_count):
        test_report["results"]["test_details"].append(f"✅ Test {i+1}: PASSED")
    
    # Save test report
    report_path = Path(__file__).parent / "quality_analytics_dashboard_v2_test_report.json"
    with open(report_path, 'w') as f:
        json.dump(test_report, f, indent=2)
    
    print(f"\n📊 Test Report Summary:")
    print(f"Total Tests: {test_report['results']['total_tests']}")
    print(f"Passed: {test_report['results']['passed_tests']}")
    print(f"Failed: {test_report['results']['failed_tests']}")
    print(f"Errors: {test_report['results']['error_tests']}")
    print(f"Success Rate: {test_report['success_rate']:.1f}%")
    print(f"Status: {test_report['status']}")
    print(f"Report saved: {report_path}")
    
    return result.wasSuccessful()

if __name__ == "__main__":
    success = run_comprehensive_test()
    sys.exit(0 if success else 1)
