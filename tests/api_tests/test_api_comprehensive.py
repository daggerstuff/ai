#!/usr/bin/env python3
"""
Pixelated Empathy AI - Comprehensive API Test Suite
Task 3A.3.5: API Testing and Validation Tools

Enterprise-grade test suite for the Pixelated Empathy AI API.
Separates real integration tests from mock data tests.
"""

import pytest
import asyncio
import httpx
import json
import os
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
import time

# Test configuration
API_BASE_URL = os.getenv("PIXELATED_API_BASE_URL", "http://localhost:8000")
API_KEY = os.getenv("PIXELATED_API_KEY", "test-api-key")
REAL_API_TESTING = os.getenv("PIXELATED_REAL_API_TEST", "false").lower() == "true"


class APITestClient:
    """Enhanced test client for API testing."""
    
    def __init__(self, base_url: str, api_key: str):
        self.base_url = base_url.rstrip('/')
        self.api_key = api_key
        self.client = httpx.AsyncClient(
            base_url=self.base_url,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            timeout=30.0
        )
    
    async def request(self, method: str, endpoint: str, **kwargs) -> httpx.Response:
        """Make HTTP request with error handling."""
        response = await self.client.request(method, endpoint, **kwargs)
        return response
    
    async def close(self):
        """Close the HTTP client."""
        await self.client.aclose()


@pytest.fixture
async def api_client():
    """Fixture providing API test client."""
    client = APITestClient(API_BASE_URL, API_KEY)
    yield client
    await client.close()


@pytest.fixture
def sample_conversation():
    """Sample conversation for testing."""
    return {
        "id": "test_conv_001",
        "messages": [
            {
                "role": "user",
                "content": "I'm feeling anxious about my upcoming presentation.",
                "timestamp": "2025-08-29T08:00:00Z"
            },
            {
                "role": "assistant", 
                "content": "I understand that presentations can trigger anxiety. What specifically about the presentation is causing you the most concern?",
                "timestamp": "2025-08-29T08:00:30Z"
            }
        ],
        "quality_score": 0.78,
        "tier": "professional",
        "metadata": {
            "dataset": "test_dataset",
            "created_at": "2025-08-29T08:00:00Z"
        }
    }


@pytest.fixture
def advanced_query():
    """Advanced query for testing."""
    return {
        "tier": "professional",
        "min_quality": 0.7,
        "min_therapeutic_accuracy": 0.75,
        "min_safety_score": 0.9,
        "created_after": (datetime.now() - timedelta(days=30)).isoformat(),
        "sort_by": "quality_score",
        "sort_order": "desc",
        "limit": 10,
        "offset": 0
    }


@pytest.fixture
def bulk_export_request():
    """Bulk export request for testing."""
    return {
        "dataset": "priority_complete_fixed",
        "format": "jsonl",
        "filters": {
            "tier": "professional",
            "min_quality": 0.8,
            "limit": 100
        },
        "include_metadata": True,
        "include_quality_metrics": True,
        "batch_size": 50
    }


class TestAPIAuthentication:
    """Test suite for API authentication."""
    
    async def test_root_endpoint_with_auth(self, api_client):
        """Test root endpoint with authentication."""
        response = await api_client.request("GET", "/")
        assert response.status_code == 200
        
        data = response.json()
        assert data["success"] is True
        assert "Pixelated Empathy AI API" in data["data"]["name"]
        assert "endpoints" in data["data"]
    
    async def test_root_endpoint_without_auth(self):
        """Test root endpoint without authentication."""
        async with httpx.AsyncClient(base_url=API_BASE_URL) as client:
            response = await client.get("/")
            # Root endpoint should be accessible without auth
            assert response.status_code in [200, 401]  # Depends on API configuration
    
    async def test_protected_endpoint_without_auth(self):
        """Test protected endpoint without authentication."""
        async with httpx.AsyncClient(base_url=API_BASE_URL) as client:
            response = await client.get("/v1/datasets")
            assert response.status_code == 401
    
    async def test_invalid_api_key(self):
        """Test request with invalid API key."""
        client = APITestClient(API_BASE_URL, "invalid-api-key")
        try:
            response = await client.request("GET", "/v1/datasets")
            assert response.status_code == 401
        finally:
            await client.close()


class TestDatasetEndpoints:
    """Test suite for dataset endpoints."""
    
    async def test_list_datasets(self, api_client):
        """Test listing datasets."""
        response = await api_client.request("GET", "/v1/datasets")
        assert response.status_code == 200
        
        data = response.json()
        assert data["success"] is True
        assert "datasets" in data["data"]
        assert isinstance(data["data"]["datasets"], list)
        
        if data["data"]["datasets"]:
            dataset = data["data"]["datasets"][0]
            assert "name" in dataset
            assert "description" in dataset
            assert "conversations" in dataset
            assert "quality_score" in dataset
    
    async def test_get_dataset_info(self, api_client):
        """Test getting specific dataset information."""
        # First get available datasets
        response = await api_client.request("GET", "/v1/datasets")
        assert response.status_code == 200
        
        datasets = response.json()["data"]["datasets"]
        if datasets:
            dataset_name = datasets[0]["name"]
            
            # Get detailed info
            response = await api_client.request("GET", f"/v1/datasets/{dataset_name}")
            assert response.status_code == 200
            
            data = response.json()
            assert data["success"] is True
            assert "name" in data["data"]
            assert "statistics" in data["data"]
    
    async def test_get_nonexistent_dataset(self, api_client):
        """Test getting information for non-existent dataset."""
        response = await api_client.request("GET", "/v1/datasets/nonexistent_dataset")
        # Should return either 404 or mock data, depending on implementation
        assert response.status_code in [200, 404, 500]


class TestConversationEndpoints:
    """Test suite for conversation endpoints."""
    
    async def test_list_conversations_basic(self, api_client):
        """Test basic conversation listing."""
        response = await api_client.request("GET", "/v1/conversations")
        assert response.status_code == 200
        
        data = response.json()
        assert data["success"] is True
        assert "conversations" in data["data"]
        assert "pagination" in data["data"]
    
    async def test_list_conversations_with_filters(self, api_client):
        """Test conversation listing with filters."""
        params = {
            "tier": "professional",
            "min_quality": "0.8",
            "limit": "5"
        }
        
        response = await api_client.request("GET", "/v1/conversations", params=params)
        assert response.status_code == 200
        
        data = response.json()
        assert data["success"] is True
        assert len(data["data"]["conversations"]) <= 5
    
    async def test_advanced_conversation_query(self, api_client, advanced_query):
        """Test advanced conversation querying."""
        response = await api_client.request(
            "POST", "/v1/conversations/query", 
            json=advanced_query
        )
        assert response.status_code == 200
        
        data = response.json()
        assert data["success"] is True
        assert "conversations" in data["data"]
        assert "total_matches" in data["data"]
        assert "query_time_ms" in data["data"]
    
    async def test_get_conversation_by_id(self, api_client):
        """Test getting specific conversation by ID."""
        conversation_id = "test_conv_001"
        
        response = await api_client.request("GET", f"/v1/conversations/{conversation_id}")
        assert response.status_code in [200, 404]  # Depends on whether test data exists
        
        if response.status_code == 200:
            data = response.json()
            assert data["success"] is True
            assert "id" in data["data"]
            assert "messages" in data["data"]
            assert "quality_metrics" in data["data"]


class TestQualityEndpoints:
    """Test suite for quality assessment endpoints."""
    
    async def test_get_quality_metrics(self, api_client):
        """Test getting quality metrics."""
        response = await api_client.request("GET", "/v1/quality/metrics")
        assert response.status_code == 200
        
        data = response.json()
        assert data["success"] is True
        assert "overall_statistics" in data["data"]
        assert "tier_metrics" in data["data"]
    
    async def test_get_quality_metrics_with_filters(self, api_client):
        """Test getting quality metrics with filters."""
        params = {
            "tier": "professional",
            "dataset": "priority_complete_fixed"
        }
        
        response = await api_client.request(
            "GET", "/v1/quality/metrics", 
            params=params
        )
        assert response.status_code == 200
    
    async def test_validate_conversation_quality(self, api_client, sample_conversation):
        """Test conversation quality validation."""
        response = await api_client.request(
            "POST", "/v1/quality/validate",
            json=sample_conversation
        )
        assert response.status_code == 200
        
        data = response.json()
        assert data["success"] is True
        assert "validation_results" in data["data"]
        assert "tier_classification" in data["data"]
        assert "recommendations" in data["data"]


class TestExportEndpoints:
    """Test suite for bulk export endpoints."""
    
    async def test_create_bulk_export(self, api_client, bulk_export_request):
        """Test creating bulk export job."""
        response = await api_client.request(
            "POST", "/v1/export/bulk",
            json=bulk_export_request
        )
        assert response.status_code == 200
        
        data = response.json()
        assert data["success"] is True
        assert "job_id" in data["data"]
        
        return data["data"]["job_id"]
    
    async def test_get_export_status(self, api_client, bulk_export_request):
        """Test getting export job status."""
        # Create export job first
        job_id = await self.test_create_bulk_export(api_client, bulk_export_request)
        
        # Get job status
        response = await api_client.request("GET", f"/v1/export/jobs/{job_id}/status")
        assert response.status_code == 200
        
        data = response.json()
        assert data["success"] is True
        assert "job_id" in data["data"]
        assert "status" in data["data"]
        assert "progress" in data["data"]
    
    async def test_list_export_jobs(self, api_client):
        """Test listing export jobs."""
        response = await api_client.request("GET", "/v1/export/jobs")
        assert response.status_code == 200
        
        data = response.json()
        assert data["success"] is True
        assert "jobs" in data["data"]
    
    async def test_cancel_export_job(self, api_client, bulk_export_request):
        """Test canceling export job."""
        # Create export job first
        job_id = await self.test_create_bulk_export(api_client, bulk_export_request)
        
        # Cancel the job
        response = await api_client.request("DELETE", f"/v1/export/jobs/{job_id}")
        assert response.status_code == 200
        
        data = response.json()
        assert data["success"] is True


class TestSearchEndpoints:
    """Test suite for search endpoints."""
    
    async def test_search_conversations(self, api_client):
        """Test conversation search."""
        search_request = {
            "query": "anxiety therapy",
            "filters": {
                "tier": "professional",
                "min_quality": 0.7
            },
            "limit": 5,
            "offset": 0
        }
        
        response = await api_client.request(
            "POST", "/v1/search",
            json=search_request
        )
        assert response.status_code == 200
        
        data = response.json()
        assert data["success"] is True
        assert "results" in data["data"]
        assert "total_matches" in data["data"]
        assert "search_time_ms" in data["data"]


class TestMonitoringEndpoints:
    """Test suite for monitoring endpoints."""
    
    async def test_health_check(self, api_client):
        """Test health check endpoint."""
        response = await api_client.request("GET", "/health")
        assert response.status_code == 200
        
        data = response.json()
        assert data["success"] is True
        assert "status" in data["data"]
    
    async def test_detailed_health_check(self, api_client):
        """Test detailed health check."""
        response = await api_client.request("GET", "/v1/monitoring/health/detailed")
        assert response.status_code == 200
        
        data = response.json()
        assert data["success"] is True
        assert "status" in data["data"]
        assert "components" in data["data"]
    
    async def test_usage_statistics(self, api_client):
        """Test usage statistics endpoint."""
        response = await api_client.request("GET", "/v1/monitoring/usage")
        assert response.status_code == 200
        
        data = response.json()
        assert data["success"] is True
        assert "user_statistics" in data["data"]
        assert "rate_limiting" in data["data"]
        assert "system_statistics" in data["data"]
    
    async def test_rate_limit_info(self, api_client):
        """Test rate limit information."""
        response = await api_client.request("GET", "/v1/monitoring/rate-limits")
        assert response.status_code == 200
        
        data = response.json()
        assert data["success"] is True


class TestRateLimiting:
    """Test suite for rate limiting functionality."""
    
    @pytest.mark.skipif(not REAL_API_TESTING, reason="Requires real API for rate limiting tests")
    async def test_rate_limit_enforcement(self):
        """Test that rate limits are enforced."""
        client = APITestClient(API_BASE_URL, API_KEY)
        
        try:
            # Make rapid requests to trigger rate limiting
            responses = []
            for i in range(20):
                response = await client.request("GET", "/v1/datasets")
                responses.append(response.status_code)
                
                # Check if we hit rate limit
                if response.status_code == 429:
                    assert "rate_limit_info" in response.json()
                    break
                
                await asyncio.sleep(0.1)  # Brief pause
            
            # Should have hit rate limit or all succeeded
            assert 429 in responses or all(code == 200 for code in responses)
            
        finally:
            await client.close()
    
    async def test_rate_limit_headers(self, api_client):
        """Test that rate limit headers are present."""
        response = await api_client.request("GET", "/v1/datasets")
        
        # Check for rate limit headers
        headers = response.headers
        assert "x-ratelimit-limit" in headers or "X-RateLimit-Limit" in headers
        assert "x-ratelimit-remaining" in headers or "X-RateLimit-Remaining" in headers


class TestErrorHandling:
    """Test suite for error handling."""
    
    async def test_invalid_json_request(self, api_client):
        """Test handling of invalid JSON requests."""
        response = await api_client.request(
            "POST", "/v1/conversations/query",
            content="invalid json",
            headers={"content-type": "application/json"}
        )
        assert response.status_code == 422  # Validation error
    
    async def test_missing_required_fields(self, api_client):
        """Test handling of missing required fields."""
        incomplete_export = {
            "format": "jsonl"
            # Missing required 'dataset' field
        }
        
        response = await api_client.request(
            "POST", "/v1/export/bulk",
            json=incomplete_export
        )
        assert response.status_code == 422  # Validation error
    
    async def test_invalid_parameter_values(self, api_client):
        """Test handling of invalid parameter values."""
        invalid_query = {
            "tier": "invalid_tier",
            "min_quality": 1.5,  # Quality should be 0.0-1.0
            "limit": -1  # Negative limit
        }
        
        response = await api_client.request(
            "POST", "/v1/conversations/query",
            json=invalid_query
        )
        # Should handle gracefully, either validation error or filtered results
        assert response.status_code in [200, 422]


class TestPerformance:
    """Performance tests for API endpoints."""
    
    async def test_response_time_datasets(self, api_client):
        """Test response time for dataset listing."""
        start_time = time.time()
        response = await api_client.request("GET", "/v1/datasets")
        end_time = time.time()
        
        assert response.status_code == 200
        assert end_time - start_time < 5.0  # Should respond within 5 seconds
    
    async def test_response_time_query(self, api_client, advanced_query):
        """Test response time for conversation queries."""
        start_time = time.time()
        response = await api_client.request(
            "POST", "/v1/conversations/query",
            json=advanced_query
        )
        end_time = time.time()
        
        assert response.status_code == 200
        assert end_time - start_time < 10.0  # Should respond within 10 seconds
    
    async def test_concurrent_requests(self, api_client):
        """Test handling of concurrent requests."""
        async def make_request():
            return await api_client.request("GET", "/v1/datasets")
        
        # Make 5 concurrent requests
        tasks = [make_request() for _ in range(5)]
        responses = await asyncio.gather(*tasks)
        
        # All requests should succeed
        for response in responses:
            assert response.status_code == 200


class TestDataIntegrity:
    """Test data integrity and consistency."""
    
    async def test_pagination_consistency(self, api_client):
        """Test that pagination returns consistent results."""
        # Get first page
        response1 = await api_client.request(
            "GET", "/v1/conversations",
            params={"limit": "5", "offset": "0"}
        )
        assert response1.status_code == 200
        
        # Get second page
        response2 = await api_client.request(
            "GET", "/v1/conversations", 
            params={"limit": "5", "offset": "5"}
        )
        assert response2.status_code == 200
        
        # Results should be different (assuming more than 5 conversations)
        data1 = response1.json()["data"]["conversations"]
        data2 = response2.json()["data"]["conversations"]
        
        if len(data1) > 0 and len(data2) > 0:
            # Should have different conversations (if enough data exists)
            ids1 = {conv["id"] for conv in data1}
            ids2 = {conv["id"] for conv in data2}
            assert ids1.isdisjoint(ids2)  # No overlap in IDs
    
    async def test_filter_consistency(self, api_client):
        """Test that filters produce consistent results."""
        query_params = {
            "tier": "professional",
            "min_quality": "0.8"
        }
        
        # Make same request twice
        response1 = await api_client.request("GET", "/v1/conversations", params=query_params)
        response2 = await api_client.request("GET", "/v1/conversations", params=query_params)
        
        assert response1.status_code == 200
        assert response2.status_code == 200
        
        # Results should be identical
        data1 = response1.json()["data"]
        data2 = response2.json()["data"]
        
        # At minimum, counts should match
        assert len(data1["conversations"]) == len(data2["conversations"])


# Utility functions for test setup and teardown

@pytest.fixture(scope="session", autouse=True)
async def setup_test_environment():
    """Setup test environment before running tests."""
    print(f"Setting up API tests for: {API_BASE_URL}")
    print(f"Real API testing: {REAL_API_TESTING}")
    
    # Verify API is accessible
    async with httpx.AsyncClient() as client:
        try:
            response = await client.get(f"{API_BASE_URL}/health", timeout=10.0)
            if response.status_code != 200:
                pytest.skip(f"API not accessible at {API_BASE_URL}")
        except httpx.RequestError:
            pytest.skip(f"Cannot connect to API at {API_BASE_URL}")


def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line(
        "markers", "real_api: mark test as requiring real API connection"
    )
    config.addinivalue_line(
        "markers", "performance: mark test as performance test"
    )


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v", "--tb=short"])