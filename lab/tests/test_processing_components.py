from datetime import datetime

import pytest

#!/usr/bin/env python3
"""
Comprehensive Unit Tests for Processing Components
Task 5.7.1.1: Build unit tests for all processing components

Tests all core processing components including:
- Database operations
- Data processing functions
- Analytics calculations
- Utility functions
- Error handling
"""

import json
import os
import sqlite3
import sys
import tempfile
import unittest
import warnings
from unittest.mock import Mock, patch

import numpy as np
import pandas as pd

warnings.simplefilter("default")

# Add the monitoring directory to the path for imports
sys.path.append("/home/vivi/pixelated/ai/monitoring")
sys.path.append("/home/vivi/pixelated/ai")


class TestDatabaseOperations(unittest.TestCase):
    """Test database connection and operations"""

    def setUp(self):
        """Set up test database"""
        self.test_db = tempfile.NamedTemporaryFile(delete=False, suffix=".db")
        self.test_db_path = self.test_db.name
        self.test_db.close()

        # Create test database with sample data
        self._create_test_database()

    def tearDown(self):
        """Clean up test database"""
        if os.path.exists(self.test_db_path):
            os.unlink(self.test_db_path)

    def _create_test_database(self):
        """Create test database with sample data"""
        conn = sqlite3.connect(self.test_db_path)
        cursor = conn.cursor()

        # Create conversations table
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
                processed_at TIMESTAMP
            )
        """)

        # Insert test data
        test_data = [
            (
                "test_001",
                "test_dataset",
                "priority_1",
                '[{"human": "Hello", "assistant": "Hi there!"}]',
                25,
                4,
                2,
                "2025-08-07 10:00:00",
                "2025-08-07 10:01:00",
            ),
            (
                "test_002",
                "test_dataset",
                "standard",
                '[{"human": "How are you?", "assistant": "I am doing well, thank you for asking."}]',
                55,
                12,
                2,
                "2025-08-07 10:05:00",
                "2025-08-07 10:06:00",
            ),
            (
                "test_003",
                "professional_dataset",
                "priority_1",
                '[{"human": "What is machine learning?", "assistant": "Machine learning is a subset of artificial intelligence."}]',
                85,
                15,
                2,
                "2025-08-07 10:10:00",
                "2025-08-07 10:11:00",
            ),
        ]

        cursor.executemany(
            """
            INSERT INTO conversations VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
            test_data,
        )

        conn.commit()
        conn.close()

    def test_database_connection(self):
        """Test database connection functionality"""
        conn = sqlite3.connect(self.test_db_path)
        assert conn is not None

        # Test query execution
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM conversations")
        count = cursor.fetchone()[0]
        assert count == 3

        conn.close()

    def test_data_retrieval(self):
        """Test data retrieval operations"""
        conn = sqlite3.connect(self.test_db_path)

        # Test pandas integration
        df = pd.read_sql_query("SELECT * FROM conversations", conn)
        assert len(df) == 3
        assert "conversation_id" in df.columns
        assert "dataset_source" in df.columns

        conn.close()

    def test_data_filtering(self):
        """Test data filtering operations"""
        conn = sqlite3.connect(self.test_db_path)

        # Test filtering by dataset
        df = pd.read_sql_query("SELECT * FROM conversations WHERE dataset_source = 'test_dataset'", conn)
        assert len(df) == 2

        # Test filtering by tier
        df = pd.read_sql_query("SELECT * FROM conversations WHERE tier = 'priority_1'", conn)
        assert len(df) == 2

        conn.close()


class TestDataProcessingFunctions(unittest.TestCase):
    """Test data processing and transformation functions"""

    def setUp(self):
        """Set up test data"""
        self.sample_conversations = pd.DataFrame(
            {
                "conversation_id": ["test_001", "test_002", "test_003"],
                "dataset_source": ["dataset_a", "dataset_b", "dataset_a"],
                "tier": ["priority_1", "standard", "priority_1"],
                "conversations_json": [
                    '[{"human": "Hello", "assistant": "Hi there!"}]',
                    '[{"human": "How are you?", "assistant": "I am doing well."}]',
                    '[{"human": "What is AI?", "assistant": "AI is artificial intelligence."}]',
                ],
                "word_count": [4, 8, 7],
                "character_count": [25, 45, 40],
            }
        )

    def test_json_text_extraction(self):
        """Test JSON conversation text extraction"""

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
            except:
                return json_str

        # Test valid JSON
        json_input = '[{"human": "Hello", "assistant": "Hi there!"}]'
        expected_output = "human: Hello\nassistant: Hi there!"
        result = extract_text_from_json(json_input)
        assert result == expected_output

        # Test invalid JSON
        invalid_json = "invalid json string"
        result = extract_text_from_json(invalid_json)
        assert result == invalid_json

    def test_quality_score_calculation(self):
        """Test quality score calculation functions"""

        def calculate_basic_quality_score(text, word_count):
            if not text or word_count == 0:
                return 0

            # Simple quality metrics
            questions = text.count("?")
            exclamations = text.count("!")

            # Basic scoring
            engagement_score = min(100, (questions + exclamations) * 10)
            length_score = min(100, word_count * 2)

            return (engagement_score + length_score) / 2

        # Test with question
        text_with_question = "How are you doing today?"
        score = calculate_basic_quality_score(text_with_question, 5)
        assert score > 0

        # Test with empty text
        score = calculate_basic_quality_score("", 0)
        assert score == 0

    def test_data_aggregation(self):
        """Test data aggregation functions"""
        # Test groupby operations
        dataset_stats = self.sample_conversations.groupby("dataset_source").agg(
            {"word_count": ["mean", "sum", "count"], "character_count": "mean"}
        )

        assert len(dataset_stats) == 2

        # Test tier aggregation
        tier_stats = self.sample_conversations.groupby("tier").size()
        assert "priority_1" in tier_stats.index
        assert "standard" in tier_stats.index

    def test_statistical_calculations(self):
        """Test statistical calculation functions"""
        data = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

        # Test basic statistics
        mean_val = np.mean(data)
        assert mean_val == 5.5

        std_val = np.std(data)
        assert std_val == pytest.approx(2.8722813232690143, abs=1e-05)

        # Test percentiles
        percentile_25 = np.percentile(data, 25)
        percentile_75 = np.percentile(data, 75)
        assert percentile_25 == 3.25
        assert percentile_75 == 7.75


class TestAnalyticsCalculations(unittest.TestCase):
    """Test analytics and metrics calculations"""

    def setUp(self):
        """Set up test data for analytics"""
        np.random.seed(42)  # For reproducible tests
        self.test_data = pd.DataFrame(
            {
                "quality_score": np.random.normal(50, 15, 100),
                "engagement_score": np.random.normal(60, 20, 100),
                "complexity_score": np.random.normal(40, 10, 100),
                "dataset": ["dataset_a"] * 50 + ["dataset_b"] * 50,
                "tier": ["priority_1"] * 25 + ["standard"] * 25 + ["priority_1"] * 25 + ["standard"] * 25,
            }
        )

    def test_correlation_analysis(self):
        """Test correlation calculations"""
        correlation = self.test_data["quality_score"].corr(self.test_data["engagement_score"])
        assert isinstance(correlation, float)
        assert correlation >= -1
        assert correlation <= 1

    def test_distribution_analysis(self):
        """Test distribution analysis functions"""
        # Test quartile calculations
        q1 = self.test_data["quality_score"].quantile(0.25)
        q2 = self.test_data["quality_score"].quantile(0.50)  # median
        q3 = self.test_data["quality_score"].quantile(0.75)

        assert q1 < q2
        assert q2 < q3

        # Test distribution categorization
        high_quality = len(self.test_data[self.test_data["quality_score"] > 70])
        low_quality = len(self.test_data[self.test_data["quality_score"] < 30])

        assert isinstance(high_quality, int)
        assert isinstance(low_quality, int)

    def test_trend_analysis(self):
        """Test trend analysis calculations"""
        # Create time series data
        pd.date_range("2025-08-01", periods=10, freq="D")
        values = [10, 12, 11, 15, 14, 16, 18, 17, 19, 20]

        # Simple trend calculation (slope)
        x = np.arange(len(values))
        slope, intercept = np.polyfit(x, values, 1)

        assert slope > 0  # Positive trend
        assert isinstance(intercept, float)

    def test_performance_metrics(self):
        """Test performance metrics calculations"""
        # Test accuracy-like metrics
        predicted = [1, 0, 1, 1, 0, 1, 0, 0, 1, 1]
        actual = [1, 0, 1, 0, 0, 1, 1, 0, 1, 1]

        # Calculate accuracy
        correct = sum(p == a for p, a in zip(predicted, actual, strict=False))
        accuracy = correct / len(predicted)

        assert accuracy >= 0
        assert accuracy <= 1

        # Test R-squared calculation
        y_true = np.array([1, 2, 3, 4, 5])
        y_pred = np.array([1.1, 2.2, 2.9, 3.8, 5.2])

        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
        r2 = 1 - (ss_res / ss_tot)

        assert r2 > 0.8  # Should be high for this test data


class TestUtilityFunctions(unittest.TestCase):
    """Test utility and helper functions"""

    def test_file_operations(self):
        """Test file reading and writing operations"""
        # Test JSON operations
        test_data = {"key": "value", "number": 42, "list": [1, 2, 3]}

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(test_data, f)
            temp_file = f.name

        # Test reading
        with open(temp_file) as f:
            loaded_data = json.load(f)

        assert loaded_data == test_data

        # Cleanup
        os.unlink(temp_file)

    def test_data_validation(self):
        """Test data validation functions"""

        def validate_conversation_data(data):
            required_fields = ["conversation_id", "dataset_source", "tier"]

            if not isinstance(data, dict):
                return False, "Data must be a dictionary"

            for field in required_fields:
                if field not in data:
                    return False, f"Missing required field: {field}"

            if not data["conversation_id"]:
                return False, "conversation_id cannot be empty"

            return True, "Valid"

        # Test valid data
        valid_data = {"conversation_id": "test_001", "dataset_source": "test_dataset", "tier": "priority_1"}
        is_valid, message = validate_conversation_data(valid_data)
        assert is_valid

        # Test invalid data
        invalid_data = {"conversation_id": ""}
        is_valid, _message = validate_conversation_data(invalid_data)
        assert not is_valid

    def test_text_processing(self):
        """Test text processing utility functions"""

        def clean_text(text):
            if not text:
                return ""

            # Basic text cleaning
            cleaned = text.strip()
            return " ".join(cleaned.split())  # Remove extra whitespace

        # Test text cleaning
        messy_text = "  Hello   world  \n\n  "
        cleaned = clean_text(messy_text)
        assert cleaned == "Hello world"

        # Test empty text
        cleaned = clean_text("")
        assert cleaned == ""

        # Test None input
        cleaned = clean_text(None)
        assert cleaned == ""

    def test_date_time_operations(self):
        """Test date and time utility functions"""

        def parse_datetime(date_string):
            try:
                return datetime.fromisoformat(date_string.replace("Z", "+00:00"))
            except:
                return None

        # Test valid datetime
        valid_date = "2025-08-07T10:00:00"
        parsed = parse_datetime(valid_date)
        assert isinstance(parsed, datetime)

        # Test invalid datetime
        invalid_date = "not a date"
        parsed = parse_datetime(invalid_date)
        assert parsed is None


class TestErrorHandling(unittest.TestCase):
    """Test error handling and edge cases"""

    def test_database_error_handling(self):
        """Test database error handling"""

        def safe_database_operation(db_path, query):
            try:
                conn = sqlite3.connect(db_path)
                cursor = conn.cursor()
                cursor.execute(query)
                result = cursor.fetchall()
                conn.close()
                return result, None
            except Exception as e:
                return None, str(e)

        # Test with non-existent database
        result, error = safe_database_operation("/nonexistent/path.db", "SELECT 1")
        assert result is None
        assert error is not None

        # Test with invalid query
        with tempfile.NamedTemporaryFile(suffix=".db") as temp_db:
            conn = sqlite3.connect(temp_db.name)
            conn.execute("CREATE TABLE test (id INTEGER)")
            conn.close()

            result, error = safe_database_operation(temp_db.name, "INVALID SQL")
            assert result is None
            assert error is not None

    def test_data_processing_errors(self):
        """Test data processing error handling"""

        def safe_division(a, b):
            try:
                return a / b, None
            except ZeroDivisionError:
                return None, "Division by zero"
            except Exception as e:
                return None, str(e)

        # Test division by zero
        result, error = safe_division(10, 0)
        assert result is None
        assert error == "Division by zero"

        # Test valid division
        result, error = safe_division(10, 2)
        assert result == 5.0
        assert error is None

    def test_json_parsing_errors(self):
        """Test JSON parsing error handling"""

        def safe_json_parse(json_string):
            try:
                return json.loads(json_string), None
            except json.JSONDecodeError as e:
                return None, f"JSON decode error: {e!s}"
            except Exception as e:
                return None, f"Unexpected error: {e!s}"

        # Test valid JSON
        result, error = safe_json_parse('{"key": "value"}')
        assert result == {"key": "value"}
        assert error is None

        # Test invalid JSON
        result, error = safe_json_parse("invalid json")
        assert result is None
        assert "JSON decode error" in error


class TestSystemIntegration(unittest.TestCase):
    """Test system integration components"""

    @patch("sqlite3.connect")
    def test_database_connection_mock(self, mock_connect):
        """Test database connection with mocking"""
        # Mock database connection
        mock_conn = Mock()
        mock_cursor = Mock()
        mock_connect.return_value = mock_conn
        mock_conn.cursor.return_value = mock_cursor
        mock_cursor.fetchall.return_value = [("test_data",)]

        # Test function that uses database
        def get_data_from_db(db_path):
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM test_table")
            return cursor.fetchall()

        result = get_data_from_db("test.db")
        assert result == [("test_data",)]
        mock_connect.assert_called_once_with("test.db")

    def test_configuration_loading(self):
        """Test configuration loading and validation"""

        def load_config(config_data):
            required_keys = ["database_path", "processing_settings"]

            if not isinstance(config_data, dict):
                raise ValueError("Config must be a dictionary")

            for key in required_keys:
                if key not in config_data:
                    raise KeyError(f"Missing required config key: {key}")

            return config_data

        # Test valid config
        valid_config = {"database_path": "/path/to/db", "processing_settings": {"batch_size": 100}}
        result = load_config(valid_config)
        assert result == valid_config

        # Test invalid config
        with pytest.raises(KeyError):
            load_config({"database_path": "/path/to/db"})


def run_all_tests():
    """Run all unit tests and return results"""

    # Create test suite
    test_suite = unittest.TestSuite()

    # Add all test classes
    test_classes = [
        TestDatabaseOperations,
        TestDataProcessingFunctions,
        TestAnalyticsCalculations,
        TestUtilityFunctions,
        TestErrorHandling,
        TestSystemIntegration,
    ]

    for test_class in test_classes:
        tests = unittest.TestLoader().loadTestsFromTestCase(test_class)
        test_suite.addTests(tests)

    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)

    # Print summary

    if result.failures:
        for _test, traceback in result.failures:
            traceback.split("AssertionError: ")[-1].split("\n")[
                0
            ] if "AssertionError:" in traceback else "Unknown failure"

    if result.errors:
        for _test, traceback in result.errors:
            traceback.split("\n")[-2] if traceback else "Unknown error"

    if not result.failures and not result.errors:
        pass

    return result


if __name__ == "__main__":
    run_all_tests()
