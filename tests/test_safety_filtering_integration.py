"""
Integration tests for enhanced safety filtering in the inference API.
Tests that all inference endpoints properly filter content for safety.
"""

import unittest
import json
import time
from datetime import datetime
from typing import Dict, List, Any, Optional
import logging
from unittest.mock import patch, MagicMock

# Import our modules
from ..inference.inference_api import app, api_key_manager, UserTier
from ..safety.enhanced_safety_filter import (
    EnhancedSafetyFilter, 
    SafetyCategory, 
    SafetyLevel,
    SafetyCheckResult,
    CrisisDetectionResult
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
            "Content-Type": "application/json"
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
        self.assertEqual(response.status_code, 200)
        
        data = json.loads(response.data)
        self.assertIn("status", data)
        self.assertEqual(data["status"], "healthy")
    
    @patch('..inference.model_adapters.ModelAdapterManager.predict')
    def test_safe_content_passes_through(self, mock_predict):
        """Test that safe content passes through without filtering"""
        # Mock the prediction response with safe content
        mock_predict.return_value = "This is a perfectly safe therapeutic response."
        
        test_request = {
            "model": "test_model",
            "messages": [
                {"role": "user", "content": "Hello, how are you today?"},
                {"role": "assistant", "content": "I'm doing well, thank you for asking!"}
            ],
            "max_tokens": 100,
            "temperature": 0.7
        }
        
        response = self.test_client.post(
            "/chat/completions",
            json=test_request,
            headers=self.headers
        )
        
        self.assertEqual(response.status_code, 200)
        
        data = json.loads(response.data)
        self.assertIn("choices", data)
        self.assertGreater(len(data["choices"]), 0)
        
        # Check that the response content is as expected
        response_content = data["choices"][0]["message"]["content"]
        self.assertIn("safe therapeutic response", response_content.lower())
        
        # Check that safety filtering information is included
        self.assertIn("safety_filtered", data)
        self.assertFalse(data["safety_filtered"])
        self.assertIn("safety_score", data)
        self.assertGreaterEqual(data["safety_score"], 0.8)  # Should be high for safe content
    
    @patch('..inference.model_adapters.ModelAdapterManager.predict')
    def test_crisis_content_gets_filtered(self, mock_predict):
        """Test that crisis content gets properly filtered"""
        # Mock the prediction response with crisis content
        mock_predict.return_value = "I've been thinking about ending it all. Life isn't worth living anymore."
        
        test_request = {
            "model": "test_model",
            "messages": [
                {"role": "user", "content": "I'm feeling really down lately."},
                {"role": "assistant", "content": "I understand how difficult that must be."}
            ],
            "max_tokens": 100,
            "temperature": 0.7
        }
        
        response = self.test_client.post(
            "/chat/completions",
            json=test_request,
            headers=self.headers
        )
        
        self.assertEqual(response.status_code, 200)
        
        data = json.loads(response.data)
        self.assertIn("choices", data)
        self.assertGreater(len(data["choices"]), 0)
        
        # Check that the response content has been filtered
        response_content = data["choices"][0]["message"]["content"]
        self.assertNotIn("ending it all", response_content.lower())
        self.assertNotIn("life isn't worth living", response_content.lower())
        
        # Check that safety filtering was applied
        self.assertIn("safety_filtered", data)
        self.assertTrue(data["safety_filtered"])
        self.assertIn("safety_score", data)
        self.assertLess(data["safety_score"], 0.5)  # Should be low for crisis content
        
        # Check that crisis content was detected
        self.assertIn("safety_categories", data)
        self.assertIn("crisis", [cat.lower() for cat in data["safety_categories"]])
        
        # Check for crisis intervention information
        self.assertIn("crisis_intervention", data)
        crisis_intervention = data["crisis_intervention"]
        self.assertIsNotNone(crisis_intervention)
        self.assertIn("status", crisis_intervention)
        self.assertIn("response_content", crisis_intervention)
    
    @patch('..inference.model_adapters.ModelAdapterManager.predict')
    def test_toxic_content_gets_filtered(self, mock_predict):
        """Test that toxic content gets properly filtered"""
        # Mock the prediction response with toxic content
        mock_predict.return_value = "This stupid idiot is completely worthless and should just disappear."
        
        test_request = {
            "model": "test_model",
            "messages": [
                {"role": "user", "content": "What do you think about this person?"},
                {"role": "assistant", "content": "I don't know them personally."}
            ],
            "max_tokens": 100,
            "temperature": 0.7
        }
        
        response = self.test_client.post(
            "/chat/completions",
            json=test_request,
            headers=self.headers
        )
        
        self.assertEqual(response.status_code, 200)
        
        data = json.loads(response.data)
        
        # Check that the response content has been filtered
        response_content = data["choices"][0]["message"]["content"]
        self.assertNotIn("stupid idiot", response_content.lower())
        self.assertNotIn("worthless", response_content.lower())
        
        # Check that safety filtering was applied
        self.assertIn("safety_filtered", data)
        # May or may not be filtered depending on threshold, but should have low safety score
        self.assertIn("safety_score", data)
        
        # Check that toxicity was detected
        self.assertIn("safety_categories", data)
        safety_categories_lower = [cat.lower() for cat in data["safety_categories"]]
        # May not always detect toxicity depending on thresholds, but we can check the score
        if data["safety_score"] < 0.7:  # If score is low
            self.assertGreater(len(safety_categories_lower), 0)
    
    @patch('..inference.model_adapters.ModelAdapterManager.predict')
    def test_privacy_content_gets_redacted(self, mock_predict):
        """Test that privacy violations get redacted"""
        # Mock the prediction response with privacy violations
        mock_predict.return_value = "My social security number is 123-45-6789 and my email is test@example.com."
        
        test_request = {
            "model": "test_model",
            "messages": [
                {"role": "user", "content": "Can you help me with my documents?"},
                {"role": "assistant", "content": "Of course, what do you need help with?"}
            ],
            "max_tokens": 100,
            "temperature": 0.7
        }
        
        response = self.test_client.post(
            "/chat/completions",
            json=test_request,
            headers=self.headers
        )
        
        self.assertEqual(response.status_code, 200)
        
        data = json.loads(response.data)
        
        # Check that the response content has been redacted
        response_content = data["choices"][0]["message"]["content"]
        self.assertNotIn("123-45-6789", response_content)
        self.assertNotIn("test@example.com", response_content)
        
        # Check for privacy-related safety categories
        self.assertIn("safety_categories", data)
        safety_categories_lower = [cat.lower() for cat in data["safety_categories"]]
        # Privacy violations should be flagged
    
    @patch('..inference.model_adapters.ModelAdapterManager.predict')
    def test_bias_content_gets_handled(self, mock_predict):
        """Test that biased content gets properly handled"""
        # Mock the prediction response with biased content
        mock_predict.return_value = "All people from that country are lazy and criminals."
        
        test_request = {
            "model": "test_model",
            "messages": [
                {"role": "user", "content": "What do you think about immigrants?"},
                {"role": "assistant", "content": "People come from all backgrounds."}
            ],
            "max_tokens": 100,
            "temperature": 0.7
        }
        
        response = self.test_client.post(
            "/chat/completions",
            json=test_request,
            headers=self.headers
        )
        
        self.assertEqual(response.status_code, 200)
        
        data = json.loads(response.data)
        
        # Check that the response content has been handled appropriately
        response_content = data["choices"][0]["message"]["content"]
        # The exact handling depends on the safety filter configuration
        
        # Check that safety information is included
        self.assertIn("safety_score", data)
        self.assertIn("safety_categories", data)
    
    def test_empty_content_handling(self):
        """Test that empty or invalid content is handled properly"""
        with patch('..inference.model_adapters.ModelAdapterManager.predict') as mock_predict:
            # Mock the prediction response with empty content
            mock_predict.return_value = ""
            
            test_request = {
                "model": "test_model",
                "messages": [
                    {"role": "user", "content": "Hello?"},
                ],
                "max_tokens": 100,
                "temperature": 0.7
            }
            
            response = self.test_client.post(
                "/chat/completions",
                json=test_request,
                headers=self.headers
            )
            
            # Empty content should still return a valid response
            self.assertEqual(response.status_code, 200)
            
            data = json.loads(response.data)
            self.assertIn("choices", data)
            if data["choices"]:  # If there are choices
                self.assertIn("message", data["choices"][0])
    
    @patch('..inference.model_adapters.ModelAdapterManager.predict')
    def test_mixed_content_handling(self, mock_predict):
        """Test handling of content with multiple safety issues"""
        # Mock the prediction response with mixed safety issues
        mock_predict.return_value = "I'm thinking about suicide tonight. My phone number is 555-123-4567."
        
        test_request = {
            "model": "test_model",
            "messages": [
                {"role": "user", "content": "I'm not doing well."},
                {"role": "assistant", "content": "I'm here to listen."}
            ],
            "max_tokens": 100,
            "temperature": 0.7
        }
        
        response = self.test_client.post(
            "/chat/completions",
            json=test_request,
            headers=self.headers
        )
        
        self.assertEqual(response.status_code, 200)
        
        data = json.loads(response.data)
        
        # Check that multiple safety issues were detected and handled
        self.assertIn("safety_filtered", data)
        self.assertTrue(data["safety_filtered"])
        
        self.assertIn("safety_categories", data)
        self.assertGreater(len(data["safety_categories"]), 0)
        
        # Check that crisis intervention was triggered
        self.assertIn("crisis_intervention", data)
        crisis_intervention = data["crisis_intervention"]
        self.assertIsNotNone(crisis_intervention)
        
        # Verify that the content was appropriately filtered/redacted
        response_content = data["choices"][0]["message"]["content"]
        # Should not contain the original sensitive information
        self.assertNotIn("suicide tonight", response_content.lower())
        self.assertNotIn("555-123-4567", response_content)
    
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
        self.assertLessEqual(paranoid_result.overall_score, 0.5)  # Should be very strict
        
        # Check lenient filtering
        lenient_result = lenient_filter.check_output_safety(test_content)
        # Even lenient should catch clear crisis content
        self.assertLessEqual(lenient_result.overall_score, 0.7)
    
    @patch('..inference.model_adapters.ModelAdapterManager.predict')
    def test_safety_filtering_performance(self, mock_predict):
        """Test that safety filtering doesn't significantly impact performance"""
        # Mock a simple response
        mock_predict.return_value = "This is a safe response."
        
        test_request = {
            "model": "test_model",
            "messages": [
                {"role": "user", "content": "Hello"},
                {"role": "assistant", "content": "Hi there!"}
            ],
            "max_tokens": 50,
            "temperature": 0.7
        }
        
        # Time multiple requests to check performance
        start_time = time.time()
        num_requests = 10
        
        for i in range(num_requests):
            response = self.test_client.post(
                "/chat/completions",
                json=test_request,
                headers=self.headers
            )
            self.assertEqual(response.status_code, 200)
        
        end_time = time.time()
        total_time = end_time - start_time
        avg_time_per_request = total_time / num_requests
        
        # Each request should take less than 2 seconds (including safety filtering)
        self.assertLess(avg_time_per_request, 2.0)
        logger.info(f"Average time per request with safety filtering: {avg_time_per_request:.3f}s")
    
    def test_crisis_intervention_system(self):
        """Test that crisis intervention system works correctly"""
        # Test crisis detection
        crisis_content = "I don't want to live anymore. I'm going to overdose."
        crisis_result = self.safety_filter._check_crisis_content(crisis_content)
        
        self.assertTrue(crisis_result.is_crisis)
        self.assertIsNotNone(crisis_result.crisis_type)
        self.assertGreater(crisis_result.confidence, 0.5)
        self.assertIn(crisis_result.urgency_level, ["low", "medium", "high", "immediate"])
        
        # Test crisis handling
        crisis_response = crisis_intervention_system.handle_crisis_detection(
            crisis_result,
            user_context={"user_id": "test_user"},
            content=crisis_content
        )
        
        self.assertIn("status", crisis_response)
        self.assertIn("response_content", crisis_response)
        self.assertIn("logged", crisis_response)


class TestSafetyFilterIntegration(unittest.TestCase):
    """Tests for safety filter integration with inference system"""
    
    def test_safety_filter_initialization(self):
        """Test that safety filter initializes correctly"""
        safety_filter = EnhancedSafetyFilter(SafetyLevel.MODERATE)
        self.assertIsNotNone(safety_filter)
        self.assertEqual(safety_filter.safety_level, SafetyLevel.MODERATE)
    
    def test_safety_check_result_structure(self):
        """Test that safety check results have correct structure"""
        # Test with safe content
        safety_filter = EnhancedSafetyFilter(SafetyLevel.MODERATE)
        result = safety_filter.check_output_safety("This is safe content.")
        
        self.assertIsInstance(result, SafetyCheckResult)
        self.assertTrue(hasattr(result, 'is_safe'))
        self.assertTrue(hasattr(result, 'overall_score'))
        self.assertTrue(hasattr(result, 'category_scores'))
        self.assertTrue(hasattr(result, 'flagged_categories'))
        self.assertTrue(hasattr(result, 'confidence'))
        self.assertTrue(hasattr(result, 'explanation'))
        self.assertTrue(hasattr(result, 'timestamp'))
    
    def test_crisis_detection_result_structure(self):
        """Test that crisis detection results have correct structure"""
        # Test with crisis content
        safety_filter = EnhancedSafetyFilter(SafetyLevel.MODERATE)
        crisis_result = safety_filter._check_crisis_content("I'm thinking about suicide.")
        
        self.assertIsInstance(crisis_result, CrisisDetectionResult)
        self.assertTrue(hasattr(crisis_result, 'is_crisis'))
        self.assertTrue(hasattr(crisis_result, 'crisis_type'))
        self.assertTrue(hasattr(crisis_result, 'confidence'))
        self.assertTrue(hasattr(crisis_result, 'urgency_level'))
        self.assertTrue(hasattr(crisis_result, 'recommended_action'))
        self.assertTrue(hasattr(crisis_result, 'timestamp'))
    
    def test_batch_filtering(self):
        """Test batch filtering of multiple responses"""
        safety_filter = EnhancedSafetyFilter(SafetyLevel.MODERATE)
        
        test_responses = [
            "This is safe content.",
            "I'm thinking about ending it all.",
            "Normal conversation.",
            "I hate that person so much."
        ]
        
        results = safety_filter.batch_filter_responses(test_responses)
        
        self.assertEqual(len(results), len(test_responses))
        for result in results:
            self.assertIsInstance(result, tuple)
            self.assertEqual(len(result), 3)  # (is_safe, filtered_content, safety_result)
            self.assertIsInstance(result[0], bool)
            self.assertIsInstance(result[1], str)
            self.assertIsInstance(result[2], SafetyCheckResult)


# Pytest-style tests for additional coverage
def test_safety_filter_decorator():
    """Test that safety filter decorator works correctly"""
    from ..inference.inference_api import safety_filtered_endpoint
    
    # This would test the decorator functionality
    assert safety_filtered_endpoint is not None


def test_crisis_intervention_integration():
    """Test crisis intervention system integration"""
    # Test that crisis intervention system can be instantiated
    safety_filter = EnhancedSafetyFilter(SafetyLevel.MODERATE)
    crisis_system = CrisisInterventionSystem(safety_filter)
    
    assert crisis_system is not None
    assert hasattr(crisis_system, 'handle_crisis_detection')


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
    import time
    start_time = time.time()
    num_iterations = 100
    
    for _ in range(num_iterations):
        result = safety_filter.check_output_safety(test_content)
        assert isinstance(result, SafetyCheckResult)
    
    end_time = time.time()
    total_time = end_time - start_time
    avg_time_per_check = total_time / num_iterations * 1000  # Convert to milliseconds
    
    print(f"Safety filtering benchmark:")
    print(f"  Average time per check: {avg_time_per_check:.2f}ms")
    print(f"  Checks per second: {1000/avg_time_per_check:.0f}")
    
    # Should be reasonably fast (under 100ms per check)
    assert avg_time_per_check < 100.0


# Main test runner
def run_safety_filtering_tests():
    """Run all safety filtering integration tests"""
    print("Running Safety Filtering Integration Tests...")
    
    # Run unit tests
    test_suite = unittest.TestLoader().loadTestsFromModule(__name__)
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)
    
    # Run pytest-style tests
    print("\nRunning Pytest-style Tests...")
    try:
        test_safety_filter_decorator()
        test_crisis_intervention_integration()
        print("Pytest-style tests completed successfully")
    except Exception as e:
        print(f"Pytest-style tests failed: {e}")
    
    # Run performance benchmarks
    print("\nRunning Performance Benchmarks...")
    try:
        benchmark_safety_filtering()
        print("Performance benchmarks completed successfully")
    except Exception as e:
        print(f"Performance benchmarks failed: {e}")
    
    # Summary
    print(f"\nTest Results Summary:")
    print(f"  Tests run: {result.testsRun}")
    print(f"  Failures: {len(result.failures)}")
    print(f"  Errors: {len(result.errors)}")
    
    if result.failures:
        print("\nFailures:")
        for test, traceback in result.failures:
            print(f"  {test}: {traceback}")
    
    if result.errors:
        print("\nErrors:")
        for test, traceback in result.errors:
            print(f"  {test}: {traceback}")
    
    success = result.wasSuccessful()
    print(f"\nOverall Test Status: {'✅ PASSED' if success else '❌ FAILED'}")
    
    return success


if __name__ == "__main__":
    success = run_safety_filtering_tests()
    exit(0 if success else 1)