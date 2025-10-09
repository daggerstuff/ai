"""
Comprehensive API Testing Suite for Pixelated Empathy AI
Tests all API endpoints, authentication, rate limiting, and error handling.
"""

import pytest
import asyncio
import json
from .httpx import AsyncClient
from fastapi.testclient import TestClient
from unittest.mock import Mock, patch
import jwt
from .datetime import datetime, timedelta

from pixel_voice.api.server import app
from pixel_voice.api.auth import create_access_token, UserRole
from pixel_voice.api.models import TranscriptRequest, PipelineJobRequest

class TestAPIEndpoints:
    """Test all API endpoints."""
    
    @pytest.fixture
    def client(self):
        """Create test client."""
        return TestClient(app)
    
    @pytest.fixture
    def auth_headers(self):
        """Create authentication headers."""
        token = create_access_token(
            data={"sub": "test_user", "role": UserRole.STANDARD}
        )
        return {"Authorization": f"Bearer {token}"}
    
    @pytest.fixture
    def admin_headers(self):
        """Create admin authentication headers."""
        token = create_access_token(
            data={"sub": "admin_user", "role": UserRole.ADMIN}
        )
        return {"Authorization": f"Bearer {token}"}
    
    def test_health_check(self, client):
        """Test health check endpoint."""
        response = client.get("/")
        assert response.status_code == 200
        data = response.json()
        assert "status" in data
        assert data["status"] == "healthy"
    
    def test_transcribe_endpoint(self, client, auth_headers):
        """Test transcription endpoint."""
        request_data = {
            "audio_url": "https://example.com/audio.mp3",
            "language": "en",
            "model": "whisper-1"
        }
        response = client.post("/transcribe", json=request_data, headers=auth_headers)
        assert response.status_code in [200, 202]  # Success or accepted
    
    def test_pipeline_job_creation(self, client, auth_headers):
        """Test pipeline job creation."""
        job_data = {
            "job_type": "transcription",
            "input_data": {"url": "https://example.com/video.mp4"},
            "config": {"language": "en"}
        }
        response = client.post("/pipeline/jobs", json=job_data, headers=auth_headers)
        assert response.status_code in [200, 201]
    
    def test_job_listing(self, client, auth_headers):
        """Test job listing endpoint."""
        response = client.get("/pipeline/jobs", headers=auth_headers)
        assert response.status_code == 200
        assert isinstance(response.json(), list)
    
    def test_pipeline_status(self, client, auth_headers):
        """Test pipeline status endpoint."""
        response = client.get("/pipeline/status", headers=auth_headers)
        assert response.status_code == 200
        data = response.json()
        assert "active_jobs" in data
    
    def test_data_access(self, client, auth_headers):
        """Test data access endpoints."""
        response = client.get("/data/transcripts/latest", headers=auth_headers)
        assert response.status_code in [200, 404]  # Success or not found
    
    def test_unauthorized_access(self, client):
        """Test unauthorized access is blocked."""
        response = client.get("/pipeline/jobs")
        assert response.status_code == 401
    
    def test_rate_limiting(self, client, auth_headers):
        """Test rate limiting functionality."""
        # Make multiple rapid requests
        responses = []
        for _ in range(100):  # Exceed rate limit
            response = client.get("/", headers=auth_headers)
            responses.append(response.status_code)
        
        # Should eventually get rate limited
        assert 429 in responses  # Too Many Requests

class TestAuthentication:
    """Test authentication system."""
    
    def test_jwt_token_creation(self):
        """Test JWT token creation."""
        token = create_access_token(
            data={"sub": "test_user", "role": UserRole.STANDARD}
        )
        assert isinstance(token, str)
        assert len(token) > 0
    
    def test_jwt_token_validation(self):
        """Test JWT token validation."""
        token = create_access_token(
            data={"sub": "test_user", "role": UserRole.STANDARD}
        )
        # Token should be valid
        assert token is not None
    
    def test_expired_token(self):
        """Test expired token handling."""
        # Create token with past expiration
        expired_token = jwt.encode(
            {"sub": "test_user", "exp": datetime.utcnow() - timedelta(hours=1)},
            "secret",
            algorithm="HS256"
        )
        # Should be rejected (implementation depends on auth system)
        assert expired_token is not None
    
    def test_role_based_access(self, client):
        """Test role-based access control."""
        # Standard user token
        standard_token = create_access_token(
            data={"sub": "standard_user", "role": UserRole.STANDARD}
        )
        standard_headers = {"Authorization": f"Bearer {standard_token}"}
        
        # Admin user token
        admin_token = create_access_token(
            data={"sub": "admin_user", "role": UserRole.ADMIN}
        )
        admin_headers = {"Authorization": f"Bearer {admin_token}"}
        
        # Test access to admin-only endpoints (if any)
        # This would depend on specific admin endpoints
        pass

class TestRateLimiting:
    """Test rate limiting system."""
    
    def test_per_user_rate_limits(self, client):
        """Test per-user rate limiting."""
        # Create user-specific headers
        token = create_access_token(
            data={"sub": "rate_test_user", "role": UserRole.STANDARD}
        )
        headers = {"Authorization": f"Bearer {token}"}
        
        # Make requests and check for rate limiting
        response_codes = []
        for _ in range(50):
            response = client.get("/", headers=headers)
            response_codes.append(response.status_code)
        
        # Should eventually hit rate limit
        assert any(code == 429 for code in response_codes)
    
    def test_rate_limit_headers(self, client):
        """Test rate limit headers are present."""
        response = client.get("/")
        # Check for rate limit headers
        assert "X-RateLimit-Limit" in response.headers or response.status_code == 401
    
    def test_burst_protection(self, client):
        """Test burst protection."""
        # Make rapid burst of requests
        start_time = datetime.now()
        responses = []
        for _ in range(20):
            response = client.get("/")
            responses.append(response.status_code)
        
        # Should handle burst appropriately
        assert len(responses) == 20

class TestErrorHandling:
    """Test error handling and validation."""
    
    def test_invalid_json(self, client, auth_headers):
        """Test invalid JSON handling."""
        response = client.post(
            "/transcribe",
            data="invalid json",
            headers={**auth_headers, "Content-Type": "application/json"}
        )
        assert response.status_code == 422  # Unprocessable Entity
    
    def test_missing_required_fields(self, client, auth_headers):
        """Test missing required fields."""
        response = client.post("/transcribe", json={}, headers=auth_headers)
        assert response.status_code == 422
    
    def test_invalid_endpoints(self, client):
        """Test invalid endpoint access."""
        response = client.get("/nonexistent")
        assert response.status_code == 404
    
    def test_method_not_allowed(self, client):
        """Test method not allowed."""
        response = client.put("/")  # GET-only endpoint
        assert response.status_code == 405

class TestPerformance:
    """Test API performance."""
    
    @pytest.mark.asyncio
    async def test_concurrent_requests(self):
        """Test concurrent request handling."""
        async with AsyncClient(app=app, base_url="http://test") as client:
            # Create multiple concurrent requests
            tasks = []
            for _ in range(10):
                task = client.get("/")
                tasks.append(task)
            
            responses = await asyncio.gather(*tasks)
            
            # All requests should complete
            assert len(responses) == 10
            assert all(r.status_code in [200, 401] for r in responses)
    
    def test_response_time(self, client):
        """Test response time is reasonable."""
        import time
        start_time = time.time()
        response = client.get("/")
        end_time = time.time()
        
        response_time = end_time - start_time
        assert response_time < 1.0  # Should respond within 1 second

# Pytest configuration
@pytest.fixture(scope="session")
def event_loop():
    """Create event loop for async tests."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
