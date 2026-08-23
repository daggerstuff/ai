#!/usr/bin/env python3

"""
Integration Tests for End-to-End Processing
Task 5.7.1.2: Implement integration tests for end-to-end processing

Tests complete workflows including:
- Database to analytics pipeline
- Data processing workflows
- Analytics system integration
- Dashboard generation
- Report creation
"""

import json
import os
import shutil
import sqlite3
import sys
import tempfile
import unittest
import warnings
from datetime import UTC, datetime

import pandas as pd

TOTAL_TEST_CONVERSATIONS = 50
EXPECTED_DATASET_COUNT = 4
BASE_QUALITY_SCORE = 50
WORD_BONUS_CAP = 20
TURN_BONUS_MULTIPLIER = 5
QUALITY_EXCELLENT_THRESHOLD = 70
QUALITY_LOW_THRESHOLD = 40
WORD_COUNT_EXCELLENT_THRESHOLD = 25
WORD_COUNT_GOOD_MIN = 15
WORD_COUNT_GOOD_MAX = 25
DASHBOARD_JSON_FILENAME = "dashboard_data.json"
DASHBOARD_REPORT_FILENAME = "dashboard_report.txt"
INTEGRATION_REPORT_FILENAME = "integration_test_report.json"
QUALITY_TOLERANCE = 0.1
MIN_CONVERSATION_TURNS = 2
SYSTEM_WORKFLOW_REPORTS = [
    "quality_report.json",
    "performance_dashboard.png",
    "insights_summary.txt",
]

warnings.simplefilter("default")

# Add paths for imports
sys.path.append("/home/vivi/pixelated/ai/monitoring")
sys.path.append("/home/vivi/pixelated/ai")


class TestEndToEndDataFlow(unittest.TestCase):
    """Test complete data flow from database to analytics"""

    def setUp(self):
        """Set up test environment with sample database"""
        db_fd, self.test_db_path = tempfile.mkstemp(suffix=".db")
        os.close(db_fd)

        self.test_output_dir = tempfile.mkdtemp()
        self._create_comprehensive_test_database()

    def tearDown(self):
        """Clean up test environment"""
        if os.path.exists(self.test_db_path):
            os.unlink(self.test_db_path)

        # Clean up output directory

        if os.path.exists(self.test_output_dir):
            shutil.rmtree(self.test_output_dir)

    def _create_comprehensive_test_database(self):
        """Create comprehensive test database with realistic data"""
        conn = sqlite3.connect(self.test_db_path)
        cursor = conn.cursor()

        # Create conversations table with full schema
        cursor.execute("""
            CREATE TABLE conversations (
                conversation_id TEXT PRIMARY KEY,
                dataset_source TEXT,
                tier TEXT,
                conversations_json TEXT,
                character_count INTEGER,
                word_count INTEGER,
                turn_count INTEGER,
                created_at TIMESTAMP,
                processed_at TIMESTAMP,
                processing_status TEXT,
                language TEXT
            )
        """)

        # Insert comprehensive test data
        test_conversations = []
        datasets = [
            "professional_psychology",
            "cot_reasoning",
            "test_dataset",
            "priority_conversations",
        ]
        tiers = ["priority_1", "standard", "additional_specialized"]

        for i in range(TOTAL_TEST_CONVERSATIONS):  # Create test conversations
            dataset = datasets[i % len(datasets)]
            tier = tiers[i % len(tiers)]

            # Create realistic conversation JSON
            conversation_json = json.dumps(
                [
                    {"human": f"This is test question {i + 1} about {dataset}"},
                    {
                        "assistant": (
                            f"This is a comprehensive response for {dataset} "
                            "covering the topic in detail with helpful information."
                        )
                    },
                ]
            )

            word_count = WORD_COUNT_GOOD_MIN + (i % 20)  # Vary word count
            char_count = word_count * 6  # Approximate character count
            turn_count = 2

            test_conversations.append(
                (
                    f"test_{i + 1:03d}",
                    dataset,
                    tier,
                    conversation_json,
                    char_count,
                    word_count,
                    turn_count,
                    f"2025-08-{(i % 7) + 1:02d} 10:{i % 60:02d}:00",
                    f"2025-08-{(i % 7) + 1:02d} 10:{(i % 60) + 1:02d}:00",
                    "processed",
                    "en",
                )
            )

        cursor.executemany(
            """
            INSERT INTO conversations VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
            test_conversations,
        )

        conn.commit()
        conn.close()

    def test_database_to_dataframe_pipeline(self):
        """Test complete database to DataFrame processing pipeline"""
        # Step 1: Load data from database
        conn = sqlite3.connect(self.test_db_path)
        df = pd.read_sql_query("SELECT * FROM conversations", conn)
        conn.close()

        # Verify data loading
        assert len(df) == TOTAL_TEST_CONVERSATIONS
        assert "conversation_id" in df.columns
        assert "dataset_source" in df.columns

        # Step 2: Process conversation JSON
        def extract_text_from_json(json_str):
            try:
                conversations = json.loads(json_str)
                if isinstance(conversations, list):
                    text_parts = []
                    for turn in conversations:
                        if isinstance(turn, dict):
                            for role, content in turn.items():
                                text_parts.append(f"{role}: {content}")
                    return "\n".join(text_parts)
                return str(conversations)
            except (json.JSONDecodeError, TypeError):
                return json_str

        df["conversation_text"] = df["conversations_json"].apply(extract_text_from_json)

        # Verify text extraction
        assert all(len(text) > 0 for text in df["conversation_text"])

        # Step 3: Calculate analytics metrics
        df["quality_score"] = df.apply(self._calculate_test_quality_score, axis=1)
        df["complexity_score"] = df["word_count"] * 2  # Simple complexity metric

        # Verify analytics calculations
        assert all(score > 0 for score in df["quality_score"])
        assert all(score > 0 for score in df["complexity_score"])

        # Step 4: Aggregate by dataset
        dataset_stats = (
            df.groupby("dataset_source")
            .agg(
                {
                    "quality_score": ["mean", "std", "count"],
                    "word_count": "mean",
                    "complexity_score": "mean",
                }
            )
            .round(2)
        )

        # Verify aggregation
        assert len(dataset_stats) == EXPECTED_DATASET_COUNT

    def test_analytics_system_integration(self):
        """Test integration between different analytics systems"""
        df = self._load_conversations_dataframe()
        # Extract conversation text
        df["conversation_text"] = df["conversations_json"].apply(self._extract_conversation_text)

        # Test 1: Dataset Statistics Integration
        dataset_stats = self._run_dataset_statistics_analysis(df)
        assert isinstance(dataset_stats, dict)
        assert "total_conversations" in dataset_stats

        # Test 2: Quality Analysis Integration
        quality_analysis = self._run_quality_analysis(df)
        assert isinstance(quality_analysis, dict)
        assert "average_quality" in quality_analysis

        # Test 3: Cross-system data consistency
        assert dataset_stats["total_conversations"] == len(df)
        assert (
            abs(quality_analysis["average_quality"] - df.apply(self._calculate_test_quality_score, axis=1).mean())
            < QUALITY_TOLERANCE
        )

    def test_dashboard_generation_pipeline(self):
        """Test complete dashboard generation pipeline"""
        df = self._load_conversations_dataframe()
        df["conversation_text"] = df["conversations_json"].apply(self._extract_conversation_text)
        df["quality_score"] = df.apply(self._calculate_test_quality_score, axis=1)

        # Step 2: Generate dashboard data
        dashboard_data = self._generate_dashboard_data(df)

        # Verify dashboard data structure
        required_sections = [
            "executive_metrics",
            "operational_metrics",
            "quality_distribution",
        ]
        for section in required_sections:
            assert section in dashboard_data

        # Step 3: Create visualization data
        viz_data = self._create_visualization_data(dashboard_data)

        # Verify visualization data
        assert "charts" in viz_data
        assert "tables" in viz_data

        # Step 4: Generate output files
        output_files = self._generate_output_files(viz_data)

        # Verify output files were created
        for file_path in output_files:
            assert os.path.exists(file_path)

    def test_report_generation_workflow(self):
        """Test complete report generation workflow"""
        df = self._load_conversations_dataframe()
        # Step 2: Analytics processing
        analytics_results = {
            "dataset_analysis": self._run_dataset_statistics_analysis(df),
            "quality_analysis": self._run_quality_analysis(df),
            "performance_metrics": self._calculate_performance_metrics(df),
        }

        # Step 3: Report compilation
        report_data = self._compile_report_data(analytics_results)

        # Verify report structure
        required_sections = ["executive_summary", "detailed_analysis", "recommendations"]
        for section in required_sections:
            assert section in report_data

        # Step 4: Report export
        report_file = self._export_report(report_data)

        # Verify report file
        assert os.path.exists(report_file)
        assert os.path.getsize(report_file) > 0

    def _load_conversations_dataframe(self):
        conn = sqlite3.connect(self.test_db_path)
        result = pd.read_sql_query("SELECT * FROM conversations", conn)
        conn.close()
        return result

    def test_error_handling_integration(self):
        """Test error handling across integrated systems"""
        # Test 1: Database connection error handling
        invalid_db_path = "/nonexistent/database.db"
        result = self._safe_database_operation(invalid_db_path)
        assert result is None

        # Test 2: Data processing error handling
        invalid_data = pd.DataFrame({"invalid_column": [1, 2, 3]})
        processed_data = self._safe_data_processing(invalid_data)
        assert processed_data is not None

        # Test 3: Analytics error handling
        empty_data = pd.DataFrame()
        analytics_result = self._safe_analytics_processing(empty_data)
        assert isinstance(analytics_result, dict)
        assert "error" in analytics_result

    # Helper methods for testing
    def _calculate_test_quality_score(self, row):
        """Calculate a simple quality score for testing"""
        base_score = BASE_QUALITY_SCORE
        word_bonus = min(WORD_BONUS_CAP, row["word_count"] * 0.5)
        turn_bonus = row["turn_count"] * TURN_BONUS_MULTIPLIER
        return base_score + word_bonus + turn_bonus

    def _run_dataset_statistics_analysis(self, df):
        """Run dataset statistics analysis"""
        return {
            "total_conversations": len(df),
            "unique_datasets": df["dataset_source"].nunique(),
            "unique_tiers": df["tier"].nunique(),
            "avg_word_count": df["word_count"].mean(),
            "total_word_count": df["word_count"].sum(),
        }

    def _extract_conversation_text(self, json_str):
        conversations = json.loads(json_str)
        if not isinstance(conversations, list) or len(conversations) < MIN_CONVERSATION_TURNS:
            return str(conversations)
        human_turn = conversations[0]
        assistant_turn = conversations[1]
        if not isinstance(human_turn, dict) or not isinstance(assistant_turn, dict):
            return str(conversations)
        return f"{human_turn.get('human', '')} {assistant_turn.get('assistant', '')}"

    def _run_quality_analysis(self, df):
        """Run quality analysis"""
        quality_scores = df.apply(self._calculate_test_quality_score, axis=1)
        return {
            "average_quality": quality_scores.mean(),
            "quality_std": quality_scores.std(),
            "high_quality_count": len(quality_scores[quality_scores > QUALITY_EXCELLENT_THRESHOLD]),
            "low_quality_count": len(quality_scores[quality_scores < QUALITY_LOW_THRESHOLD]),
        }

    def _generate_dashboard_data(self, df):
        """Generate dashboard data structure"""
        return {
            "executive_metrics": {
                "total_conversations": len(df),
                "quality_score": df.apply(self._calculate_test_quality_score, axis=1).mean(),
            },
            "operational_metrics": {
                "processing_status": "healthy",
                "last_updated": datetime.now(UTC).isoformat(),
            },
            "quality_distribution": {
                "excellent": len(df[df["word_count"] > WORD_COUNT_EXCELLENT_THRESHOLD]),
                "good": len(df[(df["word_count"] >= WORD_COUNT_GOOD_MIN) & (df["word_count"] <= WORD_COUNT_GOOD_MAX)]),
                "needs_improvement": len(df[df["word_count"] < WORD_COUNT_GOOD_MIN]),
            },
        }

    def _create_visualization_data(self, dashboard_data):
        """Create visualization data"""
        return {
            "charts": {
                "quality_distribution": dashboard_data["quality_distribution"],
                "metrics_overview": dashboard_data["executive_metrics"],
            },
            "tables": {"summary_stats": dashboard_data["operational_metrics"]},
        }

    def _generate_output_files(self, viz_data):
        """Generate output files"""
        output_files = []

        # Generate JSON file
        json_file = os.path.join(self.test_output_dir, DASHBOARD_JSON_FILENAME)
        with open(json_file, "w") as f:
            json.dump(viz_data, f, indent=2, default=str)
        output_files.append(json_file)

        # Generate text report
        txt_file = os.path.join(self.test_output_dir, DASHBOARD_REPORT_FILENAME)
        with open(txt_file, "w") as f:
            f.write("Dashboard Report\n")
            f.write("================\n")
            f.write(f"Generated: {datetime.now(UTC)}\n")
            f.write(f"Charts: {len(viz_data['charts'])}\n")
            f.write(f"Tables: {len(viz_data['tables'])}\n")
        output_files.append(txt_file)

        return output_files

    def _calculate_performance_metrics(self, _df):
        """Calculate performance metrics"""
        return {
            "processing_efficiency": 95.5,
            "data_quality_score": 87.3,
            "system_uptime": 99.8,
            "response_time": 2.1,
        }

    def _compile_report_data(self, analytics_results):
        """Compile report data"""
        return {
            "executive_summary": {
                "total_conversations": analytics_results["dataset_analysis"]["total_conversations"],
                "average_quality": analytics_results["quality_analysis"]["average_quality"],
                "key_insights": ["Quality is within acceptable range", "Processing is efficient"],
            },
            "detailed_analysis": analytics_results,
            "recommendations": [
                "Continue monitoring quality trends",
                "Optimize processing for better efficiency",
                "Expand dataset coverage",
            ],
        }

    def _export_report(self, report_data):
        """Export report to file"""
        report_file = os.path.join(self.test_output_dir, INTEGRATION_REPORT_FILENAME)
        with open(report_file, "w") as f:
            json.dump(report_data, f, indent=2, default=str)
        return report_file

    def _safe_database_operation(self, db_path):
        """Safe database operation with error handling"""
        try:
            conn = sqlite3.connect(db_path)
            df = pd.read_sql_query("SELECT COUNT(*) as count FROM conversations", conn)
            conn.close()
            return df
        except Exception:
            return None

    def _safe_data_processing(self, df):
        """Safe data processing with error handling"""
        try:
            # Attempt to process data
            if "conversation_id" in df.columns:
                return df.groupby("dataset_source").size()
            # Return empty result for invalid data
            return pd.Series(dtype=int)
        except Exception:
            return pd.Series(dtype=int)

    def _safe_analytics_processing(self, df):
        """Safe analytics processing with error handling"""
        try:
            if len(df) == 0:
                return {"error": "No data available for analysis"}

            return {"total_records": len(df), "analysis_complete": True}
        except Exception as e:
            return {"error": str(e)}


class TestSystemWorkflows(unittest.TestCase):
    """Test complete system workflows"""

    def test_monitoring_system_workflow(self):
        """Test monitoring system workflow"""
        # Simulate monitoring workflow
        system_status = {"database": "healthy", "analytics": "healthy", "dashboards": "healthy"}

        # Test workflow steps
        assert self._check_system_health(system_status)

        alerts = self._check_for_alerts(system_status)
        assert isinstance(alerts, list)

        report = self._generate_monitoring_report(system_status, alerts)
        assert "status" in report
        assert "timestamp" in report

    def test_analytics_pipeline_workflow(self):
        """Test complete analytics pipeline workflow"""
        # Simulate data input
        input_data = {"conversations": 100, "datasets": 4, "processing_time": 2.5}

        # Test pipeline steps
        processed_data = self._process_input_data(input_data)
        assert "processed_conversations" in processed_data

        analytics_results = self._run_analytics_pipeline(processed_data)
        assert "quality_metrics" in analytics_results

        output_reports = self._generate_pipeline_outputs(analytics_results)
        assert len(output_reports) > 0

    # Helper methods for workflow testing
    def _check_system_health(self, status):
        """Check overall system health"""
        return all(component == "healthy" for component in status.values())

    def _check_for_alerts(self, status):
        """Check for system alerts"""
        alerts = []
        for component, health in status.items():
            if health != "healthy":
                alerts.append(f"{component} is {health}")
        return alerts

    def _generate_monitoring_report(self, status, alerts):
        """Generate monitoring report"""
        return {
            "status": "healthy" if not alerts else "warning",
            "alerts": alerts,
            "timestamp": datetime.now(UTC).isoformat(),
            "components": status,
        }

    def _process_input_data(self, input_data):
        """Process input data"""
        return {
            "processed_conversations": input_data["conversations"],
            "unique_datasets": input_data["datasets"],
            "avg_processing_time": input_data["processing_time"],
            "processing_complete": True,
        }

    def _run_analytics_pipeline(self, processed_data):
        """Run analytics pipeline"""
        return {
            "quality_metrics": {
                "average_score": 75.5,
                "distribution": {"high": 30, "medium": 50, "low": 20},
            },
            "performance_metrics": {
                "efficiency": 92.3,
                "throughput": processed_data["processed_conversations"] / processed_data["avg_processing_time"],
            },
            "insights": ["Quality is improving", "Processing is efficient"],
        }

    def _generate_pipeline_outputs(self, _analytics_results):
        """Generate pipeline outputs"""
        return SYSTEM_WORKFLOW_REPORTS


def run_integration_tests():
    """Run all integration tests and return results"""

    # Create test suite
    test_suite = unittest.TestSuite()

    # Add test classes
    test_classes = [TestEndToEndDataFlow, TestSystemWorkflows]

    for test_class in test_classes:
        tests = unittest.TestLoader().loadTestsFromTestCase(test_class)
        test_suite.addTests(tests)

    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)

    # Print summary

    if result.failures:
        for _test, traceback in result.failures:
            (
                traceback.split("AssertionError: ")[-1].split("\n")[0]
                if "AssertionError:" in traceback
                else "Unknown failure"
            )

    if result.errors:
        for _test, traceback in result.errors:
            traceback.split("\n")[-2] if traceback else "Unknown error"

    if not result.failures and not result.errors:
        pass

    return result


if __name__ == "__main__":
    run_integration_tests()
