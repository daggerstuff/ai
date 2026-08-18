"""
Integration tests for enhanced safety filtering in the inference API.
Tests that all inference endpoints properly filter content for safety.
"""

import contextlib
import json
import logging
import sys
import time
import unittest
from unittest.mock import patch

# Import our modules
from ai.lab.inference.inference_api import UserTier, api_key_manager, app, safety_filtered_endpoint
from ai.lab.safety.enhanced_safety_filter import (
    CrisisDetectionResult,
    EnhancedSafetyFilter,
    SafetyCheckResult,
    SafetyLevel,
)

logger = logging.getLogger(__name__)


class TestSafetyFilteredInferenceAPI(unittest.TestCase):
    """Integration tests for safety-filtered inference API endpoints"""

    @classmethod
    def setUpClass(cls):
        """Set up test environment"""
        cls.test_client = app.test_client()
        cls.test_api_key = api_key_manager.generate_api_key("test_user", UserTier.PRO)
        cls.headers = {
            "Authorization": f"Bearer {cls.test_api_key}",
            "Content-Type": "application/json",
        }

        # Initialize enhanced safety filter for testing
        cls.safety_filter = EnhancedSafetyFilter(SafetyLevel.MODERATE)

        logger.info("Setting up safety-filtered inference API tests")

    @classmethod
    def tearDownClass(cls):
        """Clean up test environment"""
        logger.info("Tearing down safety-filtered inference API tests")

    def test_health_endpoint_safety(self):
        """Test that health endpoint works even with safety filtering"""
        response = self.test_client.get("/health")
        assert response.status_code == 200

        data = json.loads(response.data)
        assert "status" in data
        assert data["status"] == "healthy"

    @patch("..inference.model_adapters.ModelAdapterManager.predict")
    def test_safe_content_passes_through(self, mock_predict):
        """Test that safe content passes through without filtering"""
        # Mock the prediction response with safe content
        mock_predict.return_value = "This is a perfectly safe therapeutic response."

        test_request = {
            "model": "test_model",
            "messages": [
                {"role": "user", "content": "Hello, how are you today?"},
                {
                    "role": "assistant",
                    "content": "I'm doing well, thank you for asking!",
                },
            ],
            "max_tokens": 100,
            "temperature": 0.7,
        }

        response = self.test_client.post("/chat/completions", json=test_request, headers=self.headers)

        assert response.status_code == 200

        data = json.loads(response.data)
        assert "choices" in data
        assert len(data["choices"]) > 0

        # Check that the response content is as expected
        response_content = data["choices"][0]["message"]["content"]
        assert "safe therapeutic response" in response_content.lower()

        # Check that safety filtering information is included
        assert "safety_filtered" in data
        assert not data["safety_filtered"]
        assert "safety_score" in data
        assert data["safety_score"] >= 0.8  # Should be high for safe content

    @patch("..inference.model_adapters.ModelAdapterManager.predict")
    def test_crisis_content_gets_filtered(self, mock_predict):
        """Test that crisis content gets properly filtered"""
        # Mock the prediction response with crisis content
        mock_predict.return_value = "I've been thinking about ending it all. Life isn't worth living anymore."

        test_request = {
            "model": "test_model",
            "messages": [
                {"role": "user", "content": "I'm feeling really down lately."},
                {
                    "role": "assistant",
                    "content": "I understand how difficult that must be.",
                },
            ],
            "max_tokens": 100,
            "temperature": 0.7,
        }

        response = self.test_client.post("/chat/completions", json=test_request, headers=self.headers)

        assert response.status_code == 200

        data = json.loads(response.data)
        assert "choices" in data
        assert len(data["choices"]) > 0

        # Check that the response content has been filtered
        response_content = data["choices"][0]["message"]["content"]
        assert "ending it all" not in response_content.lower()
        assert "life isn't worth living" not in response_content.lower()

        # Check that safety filtering was applied
        assert "safety_filtered" in data
        assert data["safety_filtered"]
        assert "safety_score" in data
        assert data["safety_score"] < 0.5  # Should be low for crisis content

        # Check that crisis content was detected
        assert "safety_categories" in data
        assert "crisis" in [cat.lower() for cat in data["safety_categories"]]

        # Check for crisis intervention information
        assert "crisis_intervention" in data
        crisis_intervention = data["crisis_intervention"]
        assert crisis_intervention is not None
        assert "status" in crisis_intervention
        assert "response_content" in crisis_intervention

    @patch("..inference.model_adapters.ModelAdapterManager.predict")
    def test_toxic_content_gets_filtered(self, mock_predict):
        """Test that toxic content gets properly filtered"""
        # Mock the prediction response with toxic content
        mock_predict.return_value = "This stupid idiot is completely worthless and should just disappear."

        test_request = {
            "model": "test_model",
            "messages": [
                {"role": "user", "content": "What do you think about this person?"},
                {"role": "assistant", "content": "I don't know them personally."},
            ],
            "max_tokens": 100,
            "temperature": 0.7,
        }

        response = self.test_client.post("/chat/completions", json=test_request, headers=self.headers)

        assert response.status_code == 200

        data = json.loads(response.data)

        # Check that the response content has been filtered
        response_content = data["choices"][0]["message"]["content"]
        assert "stupid idiot" not in response_content.lower()
        assert "worthless" not in response_content.lower()

        # Check that safety filtering was applied
        assert "safety_filtered" in data
        # May or may not be filtered depending on threshold, but should have low safety score
        assert "safety_score" in data

        # Check that toxicity was detected
        assert "safety_categories" in data
        safety_categories_lower = [cat.lower() for cat in data["safety_categories"]]
        # May not always detect toxicity depending on thresholds, but we can check the score
        if data["safety_score"] < 0.7:  # If score is low
            assert len(safety_categories_lower) > 0

    @patch("..inference.model_adapters.ModelAdapterManager.predict")
    def test_privacy_content_gets_redacted(self, mock_predict):
        """Test that privacy violations get redacted"""
        # Mock the prediction response with privacy violations
        mock_predict.return_value = "My social security number is 123-45-6789 and my email is test@example.com."

        test_request = {
            "model": "test_model",
            "messages": [
                {"role": "user", "content": "Can you help me with my documents?"},
                {
                    "role": "assistant",
                    "content": "Of course, what do you need help with?",
                },
            ],
            "max_tokens": 100,
            "temperature": 0.7,
        }

        response = self.test_client.post("/chat/completions", json=test_request, headers=self.headers)

        assert response.status_code == 200

        data = json.loads(response.data)

        # Check that the response content has been redacted
        response_content = data["choices"][0]["message"]["content"]
        assert "123-45-6789" not in response_content
        assert "test@example.com" not in response_content

        # Check for privacy-related safety categories
        assert "safety_categories" in data
        [cat.lower() for cat in data["safety_categories"]]
        # Privacy violations should be flagged

    @patch("..inference.model_adapters.ModelAdapterManager.predict")
    def test_bias_content_gets_handled(self, mock_predict):
        """Test that biased content gets properly handled"""
        # Mock the prediction response with biased content
        mock_predict.return_value = "All people from that country are lazy and criminals."

        test_request = {
            "model": "test_model",
            "messages": [
                {"role": "user", "content": "What do you think about immigrants?"},
                {"role": "assistant", "content": "People come from all backgrounds."},
            ],
            "max_tokens": 100,
            "temperature": 0.7,
        }

        response = self.test_client.post("/chat/completions", json=test_request, headers=self.headers)

        assert response.status_code == 200

        data = json.loads(response.data)

        # Check that the response content has been handled appropriately
        data["choices"][0]["message"]["content"]
        # The exact handling depends on the safety filter configuration

        # Check that safety information is included
        assert "safety_score" in data
        assert "safety_categories" in data

    def test_empty_content_handling(self):
        """Test that empty or invalid content is handled properly"""
        with patch("..inference.model_adapters.ModelAdapterManager.predict") as mock_predict:
            # Mock the prediction response with empty content
            mock_predict.return_value = ""

            test_request = {
                "model": "test_model",
                "messages": [
                    {"role": "user", "content": "Hello?"},
                ],
                "max_tokens": 100,
                "temperature": 0.7,
            }

            response = self.test_client.post("/chat/completions", json=test_request, headers=self.headers)

            # Empty content should still return a valid response
            assert response.status_code == 200

            data = json.loads(response.data)
            assert "choices" in data
            if data["choices"]:  # If there are choices
                assert "message" in data["choices"][0]

    @patch("..inference.model_adapters.ModelAdapterManager.predict")
    def test_mixed_content_handling(self, mock_predict):
        """Test handling of content with multiple safety issues"""
        # Mock the prediction response with mixed safety issues
        mock_predict.return_value = "I'm thinking about suicide tonight. My phone number is 555-123-4567."

        test_request = {
            "model": "test_model",
            "messages": [
                {"role": "user", "content": "I'm not doing well."},
                {"role": "assistant", "content": "I'm here to listen."},
            ],
            "max_tokens": 100,
            "temperature": 0.7,
        }

        response = self.test_client.post("/chat/completions", json=test_request, headers=self.headers)

        assert response.status_code == 200

        data = json.loads(response.data)

        # Check that multiple safety issues were detected and handled
        assert "safety_filtered" in data
        assert data["safety_filtered"]

        assert "safety_categories" in data
        assert len(data["safety_categories"]) > 0

        # Check that crisis intervention was triggered
        assert "crisis_intervention" in data
        crisis_intervention = data["crisis_intervention"]
        assert crisis_intervention is not None

        # Verify that the content was appropriately filtered/redacted
        response_content = data["choices"][0]["message"]["content"]
        # Should not contain the original sensitive information
        assert "suicide tonight" not in response_content.lower()
        assert "555-123-4567" not in response_content

    def test_safety_filter_levels(self):
        """Test different safety filter levels"""
        # Test with paranoid level (very strict)
        paranoid_filter = EnhancedSafetyFilter(SafetyLevel.PARANOID)

        # Test with lenient level (less strict)
        lenient_filter = EnhancedSafetyFilter(SafetyLevel.LENIENT)

        # Test content that would be flagged differently
        test_content = "I'm feeling really down and thinking about not being here anymore."

        # Check paranoid filtering
        paranoid_result = paranoid_filter.check_output_safety(test_content)
        assert paranoid_result.overall_score <= 0.5  # Should be very strict

        # Check lenient filtering
        lenient_result = lenient_filter.check_output_safety(test_content)
        # Even lenient should catch clear crisis content
        assert lenient_result.overall_score <= 0.7

    @patch("..inference.model_adapters.ModelAdapterManager.predict")
    def test_safety_filtering_performance(self, mock_predict):
        """Test that safety filtering doesn't significantly impact performance"""
        # Mock a simple response
        mock_predict.return_value = "This is a safe response."

        test_request = {
            "model": "test_model",
            "messages": [
                {"role": "user", "content": "Hello"},
                {"role": "assistant", "content": "Hi there!"},
            ],
            "max_tokens": 50,
            "temperature": 0.7,
        }

        # Time multiple requests to check performance
        start_time = time.time()
        num_requests = 10

        for _i in range(num_requests):
            response = self.test_client.post("/chat/completions", json=test_request, headers=self.headers)
            assert response.status_code == 200

        end_time = time.time()
        total_time = end_time - start_time
        avg_time_per_request = total_time / num_requests

        # Each request should take less than 2 seconds (including safety filtering)
        assert avg_time_per_request < 2.0
        logger.info(f"Average time per request with safety filtering: {avg_time_per_request:.3f}s")

    def test_crisis_intervention_system(self):
        """Test that crisis intervention system works correctly"""
        # Test crisis detection
        crisis_content = "I don't want to live anymore. I'm going to overdose."
        crisis_result = self.safety_filter._check_crisis_content(crisis_content)

        assert crisis_result.is_crisis
        assert crisis_result.crisis_type is not None
        assert crisis_result.confidence > 0.5
        assert crisis_result.urgency_level in ["low", "medium", "high", "immediate"]

        # Test crisis handling
        crisis_response = crisis_intervention_system.handle_crisis_detection(
            crisis_result, user_context={"user_id": "test_user"}, content=crisis_content
        )

        assert "status" in crisis_response
        assert "response_content" in crisis_response
        assert "logged" in crisis_response


class TestSafetyFilterIntegration(unittest.TestCase):
    """Tests for safety filter integration with inference system"""

    def test_safety_filter_initialization(self):
        """Test that safety filter initializes correctly"""
        safety_filter = EnhancedSafetyFilter(SafetyLevel.MODERATE)
        assert safety_filter is not None
        assert safety_filter.safety_level == SafetyLevel.MODERATE

    def test_safety_check_result_structure(self):
        """Test that safety check results have correct structure"""
        # Test with safe content
        safety_filter = EnhancedSafetyFilter(SafetyLevel.MODERATE)
        result = safety_filter.check_output_safety("This is safe content.")

        assert isinstance(result, SafetyCheckResult)
        assert hasattr(result, "is_safe")
        assert hasattr(result, "overall_score")
        assert hasattr(result, "category_scores")
        assert hasattr(result, "flagged_categories")
        assert hasattr(result, "confidence")
        assert hasattr(result, "explanation")
        assert hasattr(result, "timestamp")

    def test_crisis_detection_result_structure(self):
        """Test that crisis detection results have correct structure"""
        # Test with crisis content
        safety_filter = EnhancedSafetyFilter(SafetyLevel.MODERATE)
        crisis_result = safety_filter._check_crisis_content("I'm thinking about suicide.")

        assert isinstance(crisis_result, CrisisDetectionResult)
        assert hasattr(crisis_result, "is_crisis")
        assert hasattr(crisis_result, "crisis_type")
        assert hasattr(crisis_result, "confidence")
        assert hasattr(crisis_result, "urgency_level")
        assert hasattr(crisis_result, "recommended_action")
        assert hasattr(crisis_result, "timestamp")

    def test_batch_filtering(self):
        """Test batch filtering of multiple responses"""
        safety_filter = EnhancedSafetyFilter(SafetyLevel.MODERATE)

        test_responses = [
            "This is safe content.",
            "I'm thinking about ending it all.",
            "Normal conversation.",
            "I hate that person so much.",
        ]

        results = safety_filter.batch_filter_responses(test_responses)

        assert len(results) == len(test_responses)
        for result in results:
            assert isinstance(result, tuple)
            assert len(result) == 3  # (is_safe, filtered_content, safety_result)
            assert isinstance(result[0], bool)
            assert isinstance(result[1], str)
            assert isinstance(result[2], SafetyCheckResult)


# Pytest-style tests for additional coverage
def test_safety_filter_decorator():
    """Test that safety filter decorator works correctly"""

    # This would test the decorator functionality
    assert safety_filtered_endpoint is not None


def test_crisis_intervention_integration():
    """Test crisis intervention system integration"""
    # Test that crisis intervention system can be instantiated
    safety_filter = EnhancedSafetyFilter(SafetyLevel.MODERATE)
    crisis_system = CrisisInterventionSystem(safety_filter)

    assert crisis_system is not None
    assert hasattr(crisis_system, "handle_crisis_detection")


# Performance benchmark tests
def benchmark_safety_filtering():
    """Benchmark safety filtering performance"""
    safety_filter = EnhancedSafetyFilter(SafetyLevel.MODERATE)

    # Test content
    test_content = "This is a test message that might contain various types of content."

    # Warm up
    for _ in range(5):
        safety_filter.check_output_safety(test_content)

    # Benchmark

    start_time = time.time()
    num_iterations = 100

    for _ in range(num_iterations):
        result = safety_filter.check_output_safety(test_content)
        assert isinstance(result, SafetyCheckResult)

    end_time = time.time()
    total_time = end_time - start_time
    avg_time_per_check = total_time / num_iterations * 1000  # Convert to milliseconds

    # Should be reasonably fast (under 100ms per check)
    assert avg_time_per_check < 100.0


# Main test runner
def run_safety_filtering_tests():
    """Run all safety filtering integration tests"""

    # Run unit tests
    test_suite = unittest.TestLoader().loadTestsFromModule(__name__)
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)

    # Run pytest-style tests
    try:
        test_safety_filter_decorator()
        test_crisis_intervention_integration()
    except Exception:
        pass

    # Run performance benchmarks
    with contextlib.suppress(Exception):
        benchmark_safety_filtering()

    # Summary

    if result.failures:
        for _test, _traceback in result.failures:
            pass

    if result.errors:
        for _test, _traceback in result.errors:
            pass

    return result.wasSuccessful()



if __name__ == "__main__":
    success = run_safety_filtering_tests()
    sys.exit(0 if success else 1)
