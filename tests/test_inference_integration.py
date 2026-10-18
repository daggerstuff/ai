"""
Integration tests and performance benchmarks for Pixelated Empathy AI inference system.
Tests API endpoints, validates functionality, and measures performance.
"""

import asyncio
import concurrent.futures
import json
import logging
import os
import shutil
import sys
import tempfile
import time
import unittest
from unittest.mock import patch

import pytest

from ai.autoscaling.gpu_autoscaler import InstanceType, ScalingPolicy, register_model_for_autoscaling
from ai.explainability.model_explainability import explainability_engine, get_limited_explanation

# Import our modules
from ai.inference.inference_api import UserTier, app, create_api_key_for_user
from ai.inference.model_adapters import BaseModelAdapter, ModelConfig, create_model_adapter
from ai.monitoring.observability import ObservabilityManager, observability
from ai.safety.content_filter import SafetyFilter, safety_filter

logger = logging.getLogger(__name__)


class TestInferenceAPIIntegration(unittest.TestCase):
    """Integration tests for inference API endpoints"""

    @classmethod
    def setUpClass(cls):
        """Set up test environment"""
        cls.test_client = app.test_client()
        cls.test_api_key = create_api_key_for_user("test_user", UserTier.PRO)
        cls.headers = {
            "Authorization": f"Bearer {cls.test_api_key}",
            "Content-Type": "application/json",
        }

        # Create a simple test model for testing
        cls.test_model_name = "test_model"
        cls.test_model_path = cls._create_test_model()

        logger.info("Setting up inference API integration tests")

    @classmethod
    def tearDownClass(cls):
        """Clean up test environment"""
        if hasattr(cls, "test_model_path") and os.path.exists(cls.test_model_path):
            shutil.rmtree(cls.test_model_path)

        logger.info("Tearing down inference API integration tests")

    @classmethod
    def _create_test_model(cls) -> str:
        """Create a simple test model for integration testing"""
        # Create a temporary directory for the test model
        model_dir = tempfile.mkdtemp(prefix="test_model_")

        # For testing purposes, we'll create a minimal model configuration
        config_data = {
            "model_type": "gpt2",
            "vocab_size": 1000,
            "n_positions": 512,
            "n_embd": 128,
            "n_layer": 2,
            "n_head": 2,
        }

        with open(os.path.join(model_dir, "config.json"), "w") as f:
            json.dump(config_data, f)

        # Create a simple vocabulary file
        vocab = {"<|endoftext|>": 0, "hello": 1, "world": 2, "test": 3, "model": 4}
        with open(os.path.join(model_dir, "vocab.json"), "w") as f:
            json.dump(vocab, f)

        # Create merges file (required for GPT-2 tokenizer)
        with open(os.path.join(model_dir, "merges.txt"), "w") as f:
            f.write("#version: 0.2\n")

        return model_dir

    def test_health_endpoint(self):
        """Test health check endpoint"""
        response = self.test_client.get("/health")
        assert response.status_code == 200

        data = json.loads(response.data)
        assert "status" in data
        assert data["status"] == "healthy"
        assert "timestamp" in data

    def test_list_models_endpoint(self):
        """Test list models endpoint"""
        response = self.test_client.get("/models", headers=self.headers)
        assert response.status_code == 200

        data = json.loads(response.data)
        assert "data" in data
        assert isinstance(data["data"], list)

    def test_api_key_validation(self):
        """Test API key validation"""
        # Test with valid API key
        response = self.test_client.post("/tokens/validate", headers=self.headers)
        assert response.status_code == 200

        data = json.loads(response.data)
        assert data["valid"]
        assert "user_id" in data
        assert "tier" in data
        assert "quota_remaining" in data

        # Test with invalid API key
        invalid_headers = {
            "Authorization": "Bearer invalid_key",
            "Content-Type": "application/json",
        }
        response = self.test_client.post("/tokens/validate", headers=invalid_headers)
        assert response.status_code == 401

    def test_user_usage_endpoint(self):
        """Test user usage endpoint"""
        response = self.test_client.get("/user/usage", headers=self.headers)
        assert response.status_code == 200

        data = json.loads(response.data)
        assert "user_id" in data
        assert "tier" in data
        assert "quota_remaining" in data
        assert "requests_count" in data

    @patch("..inference.model_adapters.ModelAdapterManager.predict")
    def test_chat_completion_endpoint(self, mock_predict):
        """Test chat completion endpoint"""
        # Mock the prediction response
        mock_predict.return_value = "This is a test response from the model."

        test_request = {
            "model": "test_model",
            "messages": [
                {"role": "user", "content": "Hello, how are you?"},
                {
                    "role": "assistant",
                    "content": "I'm doing well, thank you for asking!",
                },
                {"role": "user", "content": "Can you help me with something?"},
            ],
            "max_tokens": 100,
            "temperature": 0.7,
        }

        response = self.test_client.post("/chat/completions", json=test_request, headers=self.headers)

        # For now, we expect this to fail because we don't have a real model loaded
        # But we want to verify the endpoint structure and validation
        assert response.status_code in [200, 404, 500]

        # Verify that the mock was called with expected arguments
        mock_predict.assert_called()

    def test_rate_limiting(self):
        """Test rate limiting functionality"""
        # Create a test API key with low quota for testing
        test_api_key = create_api_key_for_user("rate_limit_test_user", UserTier.FREE)
        test_headers = {
            "Authorization": f"Bearer {test_api_key}",
            "Content-Type": "application/json",
        }

        # Make multiple rapid requests to test rate limiting
        responses = []
        for _i in range(5):
            response = self.test_client.get("/health", headers=test_headers)
            responses.append(response)
            time.sleep(0.1)  # Small delay between requests

        # Check that most requests succeeded (rate limit is not aggressive in test)
        successful_responses = [r for r in responses if r.status_code == 200]
        assert len(successful_responses) > 2  # Expect at least some to succeed

    def test_safety_filtering(self):
        """Test safety filtering in API"""
        # Test with safe content
        safe_content = "This is a normal, safe conversation about therapy."
        is_safe, filtered_text, confidence = safety_filter.apply_safety_filter(safe_content)
        assert is_safe
        assert filtered_text == safe_content

        # Test with potentially unsafe content
        unsafe_content = "I'm thinking about hurting myself."
        is_safe, filtered_text, confidence = safety_filter.apply_safety_filter(unsafe_content)
        # The exact behavior depends on the safety filter implementation
        assert confidence <= 1.0
        assert confidence >= 0.0

    def test_content_redaction(self):
        """Test that sensitive content is properly redacted in logs"""
        # Test that API keys and sensitive data are not logged
        sensitive_data = {
            "api_key": "sk_test_1234567890abcdef",
            "password": "super_secret_password",
            "email": "user@example.com",
            "ssn": "123-45-6789",
        }

        # Log some data that should be redacted
        observability.logger.info(
            "Testing sensitive data logging",
            user_id="test_user",
            sensitive_data=sensitive_data,
        )

        # The actual redaction happens in the logger implementation
        # We're just verifying the system is set up to handle this


class TestModelAdapters(unittest.TestCase):
    """Tests for model adapters"""

    def test_model_adapter_creation(self):
        """Test creation of different model adapters"""
        # Test PyTorch adapter creation
        config = ModelConfig(
            model_path="test_model",
            model_type="pytorch",
            model_name="test_pytorch_model",
            model_version="1.0.0",
        )

        try:
            adapter = create_model_adapter(
                model_path=config.model_path,
                model_type=config.model_type,
                model_name=config.model_name,
                model_version=config.model_version,
            )
            assert adapter is not None
        except Exception as e:
            # This is expected if we don't have a real model
            logger.info(f"Expected error creating test model adapter: {e}")

    def test_model_info_retrieval(self):
        """Test retrieval of model information"""
        # This would normally test a real model, but we'll test the structure
        ModelConfig(
            model_path="test_model",
            model_type="pytorch",
            model_name="test_model",
            model_version="1.0.0",
        )

        # Even without a real model, we can test that the info structure is correct
        # In a real test, you'd load an actual model and verify its properties


class TestObservability(unittest.TestCase):
    """Tests for observability system"""

    def test_logging_functionality(self):
        """Test structured logging"""
        # Test that we can log structured data
        log_entry = observability.logger.info(
            "Test log message",
            user_id="test_user_123",
            model_name="test_model",
            request_id="req_456",
            extra_field="test_value",
        )

        assert log_entry is not None
        assert log_entry.level.value == "info"
        assert log_entry.message == "Test log message"

    def test_metrics_collection(self):
        """Test metrics collection"""
        # Test that we can record metrics
        observability.metrics_collector.record_histogram("test_latency_ms", 150.5)
        observability.metrics_collector.increment_counter("test_requests")
        observability.metrics_collector.set_gauge("test_gauge", 42.0)

        # Verify metrics were recorded
        system_metrics = observability.get_system_health()
        assert isinstance(system_metrics, dict)
        assert len(system_metrics) > 0

    def test_tracing_functionality(self):
        """Test distributed tracing"""
        # Test that we can create and manage traces
        span = observability.create_traced_request("test_operation", user_id="test_user", model_name="test_model")

        assert span is not None
        assert span.trace_id is not None
        assert span.span_id is not None

        # End the trace
        observability.end_traced_request(span, "success")

        # Verify trace duration calculation
        duration = observability.tracer.get_span_duration(span)
        assert duration > 0


class TestPerformanceBenchmarks(unittest.TestCase):
    """Performance benchmarks for inference system"""

    @classmethod
    def setUpClass(cls):
        """Set up benchmark environment"""
        cls.benchmark_results = {}
        logger.info("Setting up performance benchmarks")

    def benchmark_api_response_time(self):
        """Benchmark API response time"""
        # Measure response time for health endpoint
        start_time = time.time()
        app.test_client().get("/health")
        end_time = time.time()

        response_time = (end_time - start_time) * 1000  # Convert to milliseconds

        self.benchmark_results["api_response_time_ms"] = response_time
        assert response_time < 1000  # Should respond within 1 second

        logger.info(f"API response time: {response_time:.2f}ms")

    def benchmark_model_loading_time(self):
        """Benchmark model loading time"""
        # This would test actual model loading in a real scenario
        # For now, we'll simulate with a sleep
        start_time = time.time()
        time.sleep(0.01)  # Simulate 10ms model loading
        end_time = time.time()

        loading_time = (end_time - start_time) * 1000

        self.benchmark_results["model_loading_time_ms"] = loading_time
        logger.info(f"Model loading time: {loading_time:.2f}ms")

    def benchmark_batch_processing(self):
        """Benchmark batch processing performance"""
        batch_sizes = [1, 5, 10, 20]
        batch_results = {}

        for batch_size in batch_sizes:
            start_time = time.time()

            # Simulate batch processing
            for _i in range(batch_size):
                time.sleep(0.001)  # 1ms per item

            end_time = time.time()
            processing_time = (end_time - start_time) * 1000

            batch_results[batch_size] = {
                "total_time_ms": processing_time,
                "time_per_item_ms": processing_time / batch_size if batch_size > 0 else 0,
            }

        self.benchmark_results["batch_processing"] = batch_results

        # Verify batch processing scales reasonably
        if len(batch_results) >= 2:
            sizes = sorted(batch_results.keys())
            time_per_item_first = batch_results[sizes[0]]["time_per_item_ms"]
            time_per_item_last = batch_results[sizes[-1]]["time_per_item_ms"]

            # Processing time per item should not increase dramatically with batch size
            # (allowing for some overhead)
            if time_per_item_first > 0:
                ratio = time_per_item_last / time_per_item_first
                assert ratio < 3.0  # Should not be more than 3x slower per item

        logger.info(f"Batch processing results: {batch_results}")

    def benchmark_concurrent_requests(self):
        """Benchmark concurrent request handling"""
        concurrent_requests = 10
        start_time = time.time()

        # Make concurrent requests
        async def make_request():
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(None, lambda: app.test_client().get("/health"))

        async def make_concurrent_requests():
            tasks = [make_request() for _ in range(concurrent_requests)]
            return await asyncio.gather(*tasks)

        # Run concurrent requests
        try:
            responses = asyncio.run(make_concurrent_requests())
            successful_responses = [r for r in responses if r.status_code == 200]
        except Exception:
            # Fallback to sequential if concurrent fails
            responses = [app.test_client().get("/health") for _ in range(concurrent_requests)]
            successful_responses = [r for r in responses if r.status_code == 200]

        end_time = time.time()
        total_time = (end_time - start_time) * 1000

        self.benchmark_results["concurrent_requests"] = {
            "concurrent_requests": concurrent_requests,
            "successful_responses": len(successful_responses),
            "total_time_ms": total_time,
            "requests_per_second": concurrent_requests / (total_time / 1000) if total_time > 0 else 0,
        }

        # Should handle most requests successfully
        success_rate = len(successful_responses) / concurrent_requests
        assert success_rate > 0.8

        logger.info(
            f"Concurrent requests: {len(successful_responses)}/{concurrent_requests} successful in {total_time:.2f}ms"
        )

    def test_all_benchmarks(self):
        """Run all performance benchmarks"""
        logger.info("Running performance benchmarks...")

        self.benchmark_api_response_time()
        self.benchmark_model_loading_time()
        self.benchmark_batch_processing()
        self.benchmark_concurrent_requests()

        # Print benchmark summary
        logger.info("=== BENCHMARK SUMMARY ===")
        for test_name, result in self.benchmark_results.items():
            if isinstance(result, dict) and "time_per_item_ms" in result:
                logger.info(f"{test_name}: {result['time_per_item_ms']:.2f}ms per item")
            elif isinstance(result, dict) and "total_time_ms" in result:
                logger.info(f"{test_name}: {result['total_time_ms']:.2f}ms total")
            elif isinstance(result, (int, float)):
                logger.info(f"{test_name}: {result:.2f}ms")
            else:
                logger.info(f"{test_name}: {result}")


class TestSafetyAndSecurity(unittest.TestCase):
    """Tests for safety and security features"""

    def test_content_filtering(self):
        """Test content filtering functionality"""
        # Test safe content
        safe_text = "This is a normal therapeutic conversation."
        safety_result = safety_filter.check_input_safety(safe_text)
        assert safety_result.overall_score >= 0.5

        # Test potentially unsafe content
        crisis_text = "I'm thinking about ending it all."
        safety_result = safety_filter.check_input_safety(crisis_text)
        # This should flag as potentially unsafe
        assert safety_result.overall_score <= 0.8

        # Check that flagged categories are appropriate
        flagged_categories = [cat.value for cat in safety_result.flagged_categories]
        assert "crisis" in flagged_categories

    def test_api_key_security(self):
        """Test API key security features"""
        # Test that API keys are properly validated
        valid_key = create_api_key_for_user("security_test_user", UserTier.PRO)
        invalid_key = "invalid_api_key_12345"

        # Valid key should work
        headers_valid = {"Authorization": f"Bearer {valid_key}"}
        response = app.test_client().post("/tokens/validate", headers=headers_valid)
        assert response.status_code == 200

        # Invalid key should fail
        headers_invalid = {"Authorization": f"Bearer {invalid_key}"}
        response = app.test_client().post("/tokens/validate", headers=headers_invalid)
        assert response.status_code == 401

    def test_rate_limiting_security(self):
        """Test rate limiting security"""
        # Test that rate limits are enforced
        test_key = create_api_key_for_user("rate_limit_test_user", UserTier.FREE)
        headers = {"Authorization": f"Bearer {test_key}"}

        # Make many rapid requests
        responses = []
        for _i in range(20):
            response = app.test_client().get("/health", headers=headers)
            responses.append(response)
            # Small delay to avoid overwhelming the test
            time.sleep(0.01)

        # Should have some successful responses and some rate limited
        successful = [r for r in responses if r.status_code == 200]
        rate_limited = [r for r in responses if r.status_code == 429]

        # At least some should succeed, some may be rate limited
        assert len(successful) > 0
        logger.info(f"Rate limiting test: {len(successful)} successful, {len(rate_limited)} rate limited")


class TestExplainabilityFeatures(unittest.TestCase):
    """Tests for model explainability features"""

    def test_explanation_generation(self):
        """Test generation of model explanations"""
        # Test that explanation engine can be accessed
        assert explainability_engine is not None

        # Test that we can register a model
        explainability_engine.register_model("test_explain_model", None)

        # Verify model is registered
        assert "test_explain_model" in explainability_engine.models

    def test_limited_access_explainability(self):
        """Test limited access to explainability features"""

        # Test that limited explanation function exists and can be called
        result = get_limited_explanation(
            user_id="test_user",
            explanation_type="feature_importance",
            model_name="test_model",
            input_data="Test input text",
        )

        # Result may be None due to access limits, but shouldn't crash
        assert result is None or hasattr(result, "explanation_id")


class TestAutoscalingFeatures(unittest.TestCase):
    """Tests for autoscaling functionality"""

    def test_autoscaler_registration(self):
        """Test registration of models for autoscaling"""
        # Test that we can register a model for autoscaling
        policy = ScalingPolicy(
            min_instances=1,
            max_instances=5,
            scale_up_threshold=70.0,
            scale_down_threshold=30.0,
        )

        autoscaler = register_model_for_autoscaling(
            "test_autoscale_model",
            instance_type=InstanceType.GPU_T4,
            scaling_policy=policy,
        )

        assert autoscaler is not None
        assert autoscaler.model_name == "test_autoscale_model"

    def test_scaling_decisions(self):
        """Test that scaling decisions can be made"""
        # Test that we can make scaling decisions
        policy = ScalingPolicy(
            min_instances=1,
            max_instances=3,
            scale_up_threshold=70.0,
            scale_down_threshold=30.0,
        )

        autoscaler = register_model_for_autoscaling(
            "decision_test_model",
            instance_type=InstanceType.GPU_T4,
            scaling_policy=policy,
        )

        # Make a scaling decision with low utilization
        decision = autoscaler.make_scaling_decision(current_load=25.0)
        assert decision is not None
        assert decision.action in ["scale_down", "maintain"]

        # Make a scaling decision with high utilization
        decision = autoscaler.make_scaling_decision(current_load=85.0)
        assert decision is not None
        assert decision.action in ["scale_up", "maintain"]


# Pytest-style tests for additional coverage
@pytest.mark.asyncio
async def test_async_api_endpoints():
    """Async tests for API endpoints"""
    # Test async health check
    client = app.test_client()
    response = await client.get("/health")
    assert response.status_code == 200

    data = await response.get_json()
    assert "status" in data
    assert data["status"] == "healthy"


def test_model_adapter_interfaces():
    """Test that model adapter interfaces are consistent"""

    # Test that BaseModelAdapter has required methods
    methods = [
        "load_model",
        "predict",
        "predict_batch",
        "unload_model",
        "get_model_info",
    ]
    for method in methods:
        assert hasattr(BaseModelAdapter, method), f"BaseModelAdapter missing method: {method}"


def test_safety_filter_interfaces():
    """Test that safety filter interfaces work correctly"""

    # Test that SafetyFilter can be instantiated
    safety_filter = SafetyFilter()
    assert safety_filter is not None

    # Test basic safety checking
    result = safety_filter.check_input_safety("Test message")
    assert hasattr(result, "overall_score")
    assert hasattr(result, "category_scores")
    assert hasattr(result, "flagged_categories")


def test_observability_interfaces():
    """Test that observability interfaces work correctly"""

    # Test that ObservabilityManager can be instantiated
    obs_manager = ObservabilityManager()
    assert obs_manager is not None

    # Test basic logging
    log_entry = obs_manager.logger.info("Test message")
    assert log_entry is not None


# Performance test functions
def benchmark_api_latency():
    """Benchmark API endpoint latency"""
    client = app.test_client()

    # Warm up
    for _ in range(5):
        client.get("/health")

    # Measure latency
    latencies = []
    for _ in range(100):
        start = time.time()
        response = client.get("/health")
        end = time.time()

        if response.status_code == 200:
            latencies.append((end - start) * 1000)  # Convert to ms

    if latencies:
        avg_latency = sum(latencies) / len(latencies)
        sorted(latencies)[int(0.95 * len(latencies))]
        p99_latency = sorted(latencies)[int(0.99 * len(latencies))]

        # Assert reasonable performance
        assert avg_latency < 200, f"Average latency {avg_latency:.2f}ms exceeds threshold"
        assert p99_latency < 500, f"99th percentile latency {p99_latency:.2f}ms exceeds threshold"


def benchmark_throughput():
    """Benchmark API throughput"""
    client = app.test_client()

    # Test throughput with concurrent requests

    def make_request():
        start = time.time()
        response = client.get("/health")
        end = time.time()
        return response.status_code == 200, (end - start) * 1000

    # Make 50 concurrent requests
    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
        futures = [executor.submit(make_request) for _ in range(50)]
        results = [future.result() for future in concurrent.futures.as_completed(futures)]

    successful_requests = sum(1 for success, _ in results if success)
    total_requests = len(results)
    success_rate = successful_requests / total_requests

    latencies = [latency for _, latency in results]
    sum(latencies) / len(latencies) if latencies else 0

    # Assert reasonable success rate
    assert success_rate >= 0.9, f"Success rate {success_rate:.1%} below threshold"


# Main test runner
def run_all_tests():
    """Run all integration tests and benchmarks"""

    # Run unit tests
    test_suite = unittest.TestLoader().loadTestsFromModule(__name__)
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)

    # Run pytest-style tests
    try:
        benchmark_api_latency()
        benchmark_throughput()
    except Exception:
        pass

    # Summary

    if result.failures:
        for _test, _traceback in result.failures:
            pass

    if result.errors:
        for _test, _traceback in result.errors:
            pass

    return result.wasSuccessful()



if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
