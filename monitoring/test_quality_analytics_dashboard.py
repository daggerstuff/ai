#!/usr/bin/env python3
"""
Test Suite for Quality Analytics Dashboard (Task 5.6.2.1)

Comprehensive testing of the quality analytics dashboard functionality
including data loading, analytics calculation, and visualization generation.
"""

import pytest
import pandas as pd
import sqlite3
import tempfile
import json
from .datetime import datetime, timedelta
from .pathlib import Path
import sys
import os

# Add the monitoring directory to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from .quality_analytics_dashboard import QualityAnalyticsDashboard, QualityAnalytics

class TestQualityAnalyticsDashboard:
    """Test suite for Quality Analytics Dashboard."""
    
    @pytest.fixture
    def sample_db(self):
        """Create a sample database with test data."""
        # Create temporary database
        db_fd, db_path = tempfile.mkstemp(suffix='.db')
        os.close(db_fd)
        
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
        
        # Insert sample data
        conversations_data = [
            (1, 'priority_1', 'test_dataset_1', '2024-08-01 10:00:00', 150),
            (2, 'priority_2', 'test_dataset_2', '2024-08-02 11:00:00', 200),
            (3, 'priority_3', 'test_dataset_3', '2024-08-03 12:00:00', 180),
            (4, 'professional', 'test_dataset_4', '2024-08-04 13:00:00', 220),
            (5, 'research', 'test_dataset_5', '2024-08-05 14:00:00', 160),
        ]
        
        cursor.executemany(
            "INSERT INTO conversations (id, tier, dataset_name, created_at, conversation_length) VALUES (?, ?, ?, ?, ?)",
            conversations_data
        )
        
        quality_data = [
            (1, 1, 0.8, 0.75, 0.82, 0.78, 0.76, 0.79, 0.85, 0.79, '2024-08-01 10:30:00'),
            (2, 2, 0.7, 0.68, 0.72, 0.69, 0.71, 0.73, 0.75, 0.71, '2024-08-02 11:30:00'),
            (3, 3, 0.6, 0.58, 0.62, 0.59, 0.61, 0.63, 0.65, 0.61, '2024-08-03 12:30:00'),
            (4, 4, 0.9, 0.88, 0.92, 0.89, 0.87, 0.91, 0.93, 0.90, '2024-08-04 13:30:00'),
            (5, 5, 0.3, 0.28, 0.32, 0.29, 0.31, 0.33, 0.35, 0.31, '2024-08-05 14:30:00'),  # Anomaly
        ]
        
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
        
        yield db_path
        
        # Cleanup
        os.unlink(db_path)
    
    @pytest.fixture
    def dashboard(self, sample_db):
        """Create dashboard instance with sample database."""
        return QualityAnalyticsDashboard(db_path=sample_db)
    
    def test_dashboard_initialization(self, sample_db):
        """Test dashboard initialization."""
        dashboard = QualityAnalyticsDashboard(db_path=sample_db)
        
        assert dashboard.db_path == sample_db
        assert dashboard.cache_duration == 300
        assert dashboard.quality_thresholds['excellent'] == 0.8
        assert len(dashboard.color_schemes['quality_levels']) == 4
    
    def test_load_quality_data(self, dashboard):
        """Test quality data loading from database."""
        df = dashboard.load_quality_data()
        
        assert not df.empty
        assert len(df) == 5
        assert 'overall_quality' in df.columns
        assert 'tier' in df.columns
        assert 'dataset_name' in df.columns
        
        # Check data types
        assert pd.api.types.is_datetime64_any_dtype(df['created_at'])
        assert pd.api.types.is_datetime64_any_dtype(df['validated_at'])
    
    def test_load_quality_data_caching(self, dashboard):
        """Test data caching functionality."""
        # First load
        df1 = dashboard.load_quality_data()
        cache_time1 = dashboard._last_cache_time
        
        # Second load (should use cache)
        df2 = dashboard.load_quality_data()
        cache_time2 = dashboard._last_cache_time
        
        assert cache_time1 == cache_time2
        assert df1.equals(df2)
        
        # Force refresh
        df3 = dashboard.load_quality_data(force_refresh=True)
        cache_time3 = dashboard._last_cache_time
        
        assert cache_time3 > cache_time1
    
    def test_calculate_quality_analytics(self, dashboard):
        """Test quality analytics calculation."""
        df = dashboard.load_quality_data()
        analytics = dashboard.calculate_quality_analytics(df)
        
        assert isinstance(analytics, QualityAnalytics)
        assert analytics.total_conversations == 5
        assert 0 <= analytics.average_quality <= 1
        assert len(analytics.quality_distribution) > 0
        assert len(analytics.tier_performance) > 0
        assert len(analytics.recommendations) > 0
    
    def test_calculate_quality_analytics_empty_data(self, dashboard):
        """Test analytics calculation with empty data."""
        empty_df = pd.DataFrame()
        analytics = dashboard.calculate_quality_analytics(empty_df)
        
        assert analytics.total_conversations == 0
        assert analytics.average_quality == 0.0
        assert len(analytics.quality_distribution) == 0
        assert len(analytics.tier_performance) == 0
        assert len(analytics.recommendations) == 1
        assert "No quality data available" in analytics.recommendations[0]
    
    def test_quality_distribution_calculation(self, dashboard):
        """Test quality distribution calculation."""
        df = dashboard.load_quality_data()
        analytics = dashboard.calculate_quality_analytics(df)
        
        # Check that quality distribution adds up to total conversations
        total_distributed = sum(analytics.quality_distribution.values())
        assert total_distributed == analytics.total_conversations
        
        # Check that all quality levels are represented
        expected_levels = ['Poor', 'Fair', 'Good', 'Excellent']
        for level in analytics.quality_distribution.keys():
            assert level in expected_levels
    
    def test_tier_performance_calculation(self, dashboard):
        """Test tier performance calculation."""
        df = dashboard.load_quality_data()
        analytics = dashboard.calculate_quality_analytics(df)
        
        # Check that all tiers are represented
        expected_tiers = ['priority_1', 'priority_2', 'priority_3', 'professional', 'research']
        for tier in analytics.tier_performance.keys():
            assert tier in expected_tiers
        
        # Check that performance values are valid
        for performance in analytics.tier_performance.values():
            assert 0 <= performance <= 1
    
    def test_anomaly_detection(self, dashboard):
        """Test anomaly detection functionality."""
        df = dashboard.load_quality_data()
        analytics = dashboard.calculate_quality_analytics(df)
        
        # Should detect the low-quality conversation (0.31) as an anomaly
        assert len(analytics.anomalies) > 0
        
        # Check anomaly structure
        anomaly = analytics.anomalies[0]
        assert 'id' in anomaly
        assert 'dataset' in anomaly
        assert 'tier' in anomaly
        assert 'quality' in anomaly
        assert 'date' in anomaly
        
        # The anomaly should be the lowest quality conversation
        assert anomaly['quality'] == 0.31
    
    def test_trend_data_calculation(self, dashboard):
        """Test trend data calculation."""
        df = dashboard.load_quality_data()
        analytics = dashboard.calculate_quality_analytics(df)
        
        # Should have trend data for recent conversations
        assert len(analytics.trend_data) > 0
        
        # Check trend data structure
        trend_point = analytics.trend_data[0]
        assert 'date' in trend_point
        assert 'quality' in trend_point
        assert isinstance(trend_point['quality'], float)
    
    def test_recommendations_generation(self, dashboard):
        """Test quality recommendations generation."""
        df = dashboard.load_quality_data()
        analytics = dashboard.calculate_quality_analytics(df)
        
        assert len(analytics.recommendations) > 0
        
        # Check that recommendations are strings
        for recommendation in analytics.recommendations:
            assert isinstance(recommendation, str)
            assert len(recommendation) > 0
    
    def test_create_quality_overview_chart(self, dashboard):
        """Test quality overview chart creation."""
        df = dashboard.load_quality_data()
        analytics = dashboard.calculate_quality_analytics(df)
        
        fig = dashboard.create_quality_overview_chart(analytics)
        
        assert fig is not None
        assert hasattr(fig, 'data')
        assert len(fig.data) > 0
    
    def test_create_detailed_quality_heatmap(self, dashboard):
        """Test detailed quality heatmap creation."""
        df = dashboard.load_quality_data()
        
        fig = dashboard.create_detailed_quality_heatmap(df)
        
        assert fig is not None
        assert hasattr(fig, 'data')
    
    def test_create_detailed_quality_heatmap_empty_data(self, dashboard):
        """Test heatmap creation with empty data."""
        empty_df = pd.DataFrame()
        
        fig = dashboard.create_detailed_quality_heatmap(empty_df)
        
        assert fig is not None
    
    def test_create_anomaly_detection_chart(self, dashboard):
        """Test anomaly detection chart creation."""
        df = dashboard.load_quality_data()
        analytics = dashboard.calculate_quality_analytics(df)
        
        fig = dashboard.create_anomaly_detection_chart(analytics)
        
        assert fig is not None
        assert hasattr(fig, 'data')
    
    def test_create_anomaly_detection_chart_no_anomalies(self, dashboard):
        """Test anomaly chart creation with no anomalies."""
        analytics = QualityAnalytics(
            total_conversations=0,
            average_quality=0.0,
            quality_distribution={},
            tier_performance={},
            trend_data=[],
            anomalies=[],  # No anomalies
            recommendations=[]
        )
        
        fig = dashboard.create_anomaly_detection_chart(analytics)
        
        assert fig is not None
    
    def test_quality_thresholds(self, dashboard):
        """Test quality threshold definitions."""
        thresholds = dashboard.quality_thresholds
        
        assert thresholds['excellent'] == 0.8
        assert thresholds['good'] == 0.6
        assert thresholds['fair'] == 0.4
        assert thresholds['poor'] == 0.0
        
        # Check ordering
        assert thresholds['excellent'] > thresholds['good']
        assert thresholds['good'] > thresholds['fair']
        assert thresholds['fair'] > thresholds['poor']
    
    def test_color_schemes(self, dashboard):
        """Test color scheme definitions."""
        colors = dashboard.color_schemes
        
        assert 'quality_levels' in colors
        assert 'tiers' in colors
        assert 'trends' in colors
        assert 'anomalies' in colors
        
        # Check that color lists have appropriate lengths
        assert len(colors['quality_levels']) == 4
        assert len(colors['tiers']) == 5
    
    def test_data_filtering_by_tier(self, dashboard):
        """Test data filtering by tier."""
        df = dashboard.load_quality_data()
        
        # Filter for specific tier
        priority_1_df = df[df['tier'] == 'priority_1']
        assert len(priority_1_df) == 1
        assert priority_1_df.iloc[0]['tier'] == 'priority_1'
    
    def test_data_filtering_by_date(self, dashboard):
        """Test data filtering by date."""
        df = dashboard.load_quality_data()
        
        # Filter for specific date range
        start_date = datetime(2024, 8, 2)
        end_date = datetime(2024, 8, 4)
        
        filtered_df = df[
            (df['created_at'] >= start_date) & 
            (df['created_at'] <= end_date)
        ]
        
        assert len(filtered_df) == 3  # Should include conversations 2, 3, 4
    
    def test_data_filtering_by_quality_threshold(self, dashboard):
        """Test data filtering by quality threshold."""
        df = dashboard.load_quality_data()
        
        # Filter for quality >= 0.7
        high_quality_df = df[df['overall_quality'] >= 0.7]
        assert len(high_quality_df) == 3  # Should include conversations 1, 2, 4
        
        # Filter for quality >= 0.9
        excellent_quality_df = df[df['overall_quality'] >= 0.9]
        assert len(excellent_quality_df) == 1  # Should include only conversation 4

def run_comprehensive_test():
    """Run comprehensive test suite and generate report."""
    print("🧪 Running Quality Analytics Dashboard Test Suite...")
    
    # Create temporary database for testing
    db_fd, db_path = tempfile.mkstemp(suffix='.db')
    os.close(db_fd)
    
    try:
        # Setup test database
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # Create tables and insert test data
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
        
        # Insert comprehensive test data
        conversations_data = []
        quality_data = []
        
        for i in range(1, 101):  # 100 test conversations
            tier = ['priority_1', 'priority_2', 'priority_3', 'professional', 'research'][i % 5]
            dataset = f'test_dataset_{i % 10}'
            created_at = (datetime.now() - timedelta(days=i % 30)).strftime('%Y-%m-%d %H:%M:%S')
            length = 100 + (i * 10)
            
            conversations_data.append((i, tier, dataset, created_at, length))
            
            # Generate quality scores with some variation
            base_quality = 0.5 + (i % 50) / 100  # Range from 0.5 to 1.0
            therapeutic_accuracy = min(1.0, base_quality + (i % 10) / 100)
            conversation_coherence = min(1.0, base_quality + (i % 8) / 100)
            emotional_authenticity = min(1.0, base_quality + (i % 12) / 100)
            clinical_compliance = min(1.0, base_quality + (i % 6) / 100)
            personality_consistency = min(1.0, base_quality + (i % 9) / 100)
            language_quality = min(1.0, base_quality + (i % 7) / 100)
            safety_score = min(1.0, base_quality + (i % 5) / 100)
            overall_quality = (therapeutic_accuracy + conversation_coherence + emotional_authenticity + 
                             clinical_compliance + personality_consistency + language_quality + safety_score) / 7
            
            validated_at = (datetime.now() - timedelta(days=i % 30, hours=1)).strftime('%Y-%m-%d %H:%M:%S')
            
            quality_data.append((
                i, i, therapeutic_accuracy, conversation_coherence, emotional_authenticity,
                clinical_compliance, personality_consistency, language_quality, safety_score,
                overall_quality, validated_at
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
        
        # Run tests
        dashboard = QualityAnalyticsDashboard(db_path=db_path)
        
        test_results = {
            'total_tests': 0,
            'passed_tests': 0,
            'failed_tests': 0,
            'test_details': []
        }
        
        # Test 1: Data Loading
        try:
            df = dashboard.load_quality_data()
            assert not df.empty
            assert len(df) == 100
            test_results['passed_tests'] += 1
            test_results['test_details'].append("✅ Data Loading: PASSED")
        except Exception as e:
            test_results['failed_tests'] += 1
            test_results['test_details'].append(f"❌ Data Loading: FAILED - {e}")
        test_results['total_tests'] += 1
        
        # Test 2: Analytics Calculation
        try:
            analytics = dashboard.calculate_quality_analytics(df)
            assert analytics.total_conversations == 100
            assert 0 <= analytics.average_quality <= 1
            test_results['passed_tests'] += 1
            test_results['test_details'].append("✅ Analytics Calculation: PASSED")
        except Exception as e:
            test_results['failed_tests'] += 1
            test_results['test_details'].append(f"❌ Analytics Calculation: FAILED - {e}")
        test_results['total_tests'] += 1
        
        # Test 3: Visualization Creation
        try:
            overview_chart = dashboard.create_quality_overview_chart(analytics)
            heatmap_chart = dashboard.create_detailed_quality_heatmap(df)
            anomaly_chart = dashboard.create_anomaly_detection_chart(analytics)
            
            assert overview_chart is not None
            assert heatmap_chart is not None
            assert anomaly_chart is not None
            
            test_results['passed_tests'] += 1
            test_results['test_details'].append("✅ Visualization Creation: PASSED")
        except Exception as e:
            test_results['failed_tests'] += 1
            test_results['test_details'].append(f"❌ Visualization Creation: FAILED - {e}")
        test_results['total_tests'] += 1
        
        # Test 4: Quality Distribution
        try:
            total_distributed = sum(analytics.quality_distribution.values())
            assert total_distributed == analytics.total_conversations
            test_results['passed_tests'] += 1
            test_results['test_details'].append("✅ Quality Distribution: PASSED")
        except Exception as e:
            test_results['failed_tests'] += 1
            test_results['test_details'].append(f"❌ Quality Distribution: FAILED - {e}")
        test_results['total_tests'] += 1
        
        # Test 5: Tier Performance
        try:
            assert len(analytics.tier_performance) > 0
            for performance in analytics.tier_performance.values():
                assert 0 <= performance <= 1
            test_results['passed_tests'] += 1
            test_results['test_details'].append("✅ Tier Performance: PASSED")
        except Exception as e:
            test_results['failed_tests'] += 1
            test_results['test_details'].append(f"❌ Tier Performance: FAILED - {e}")
        test_results['total_tests'] += 1
        
        # Generate test report
        success_rate = (test_results['passed_tests'] / test_results['total_tests']) * 100
        
        report = {
            'test_suite': 'Quality Analytics Dashboard',
            'task': '5.6.2.1',
            'timestamp': datetime.now().isoformat(),
            'results': test_results,
            'success_rate': success_rate,
            'status': 'PASSED' if success_rate >= 80 else 'FAILED',
            'analytics_sample': {
                'total_conversations': analytics.total_conversations,
                'average_quality': analytics.average_quality,
                'quality_distribution': analytics.quality_distribution,
                'tier_performance': analytics.tier_performance,
                'anomalies_detected': len(analytics.anomalies),
                'recommendations_count': len(analytics.recommendations)
            }
        }
        
        # Save test report
        report_path = Path(__file__).parent / "quality_analytics_dashboard_test_report.json"
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"\n📊 Test Results Summary:")
        print(f"Total Tests: {test_results['total_tests']}")
        print(f"Passed: {test_results['passed_tests']}")
        print(f"Failed: {test_results['failed_tests']}")
        print(f"Success Rate: {success_rate:.1f}%")
        print(f"Status: {report['status']}")
        
        print(f"\n📋 Test Details:")
        for detail in test_results['test_details']:
            print(f"  {detail}")
        
        print(f"\n📁 Test report saved to: {report_path}")
        
        return report
        
    finally:
        # Cleanup
        if os.path.exists(db_path):
            os.unlink(db_path)

if __name__ == "__main__":
    run_comprehensive_test()
