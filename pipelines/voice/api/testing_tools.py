"""
API Testing Tools for Pixelated Empathy AI
Comprehensive testing utilities and tools for API validation.
"""

import requests
import asyncio
import aiohttp
import json
import time
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from datetime import datetime
import concurrent.futures
import statistics

@dataclass
class TestResult:
    """Test result data structure."""
    endpoint: str
    method: str
    status_code: int
    response_time: float
    success: bool
    error: Optional[str] = None
    response_data: Optional[Dict] = None

class APITestClient:
    """Comprehensive API testing client."""
    
    def __init__(self, base_url: str, api_key: str = None, jwt_token: str = None):
        self.base_url = base_url.rstrip('/')
        self.session = requests.Session()
        
        # Set authentication headers
        if api_key:
            self.session.headers['X-API-Key'] = api_key
        if jwt_token:
            self.session.headers['Authorization'] = f'Bearer {jwt_token}'
    
    def test_endpoint(self, endpoint: str, method: str = 'GET', 
                     data: Dict = None, expected_status: int = 200) -> TestResult:
        """Test a single API endpoint."""
        url = f"{self.base_url}{endpoint}"
        start_time = time.time()
        
        try:
            if method.upper() == 'GET':
                response = self.session.get(url)
            elif method.upper() == 'POST':
                response = self.session.post(url, json=data)
            elif method.upper() == 'PUT':
                response = self.session.put(url, json=data)
            elif method.upper() == 'DELETE':
                response = self.session.delete(url)
            else:
                raise ValueError(f"Unsupported method: {method}")
            
            response_time = time.time() - start_time
            success = response.status_code == expected_status
            
            try:
                response_data = response.json()
            except:
                response_data = {"raw": response.text}
            
            return TestResult(
                endpoint=endpoint,
                method=method,
                status_code=response.status_code,
                response_time=response_time,
                success=success,
                response_data=response_data
            )
            
        except Exception as e:
            response_time = time.time() - start_time
            return TestResult(
                endpoint=endpoint,
                method=method,
                status_code=0,
                response_time=response_time,
                success=False,
                error=str(e)
            )
    
    def load_test(self, endpoint: str, concurrent_users: int = 10, 
                  duration_seconds: int = 60) -> Dict[str, Any]:
        """Perform load testing on an endpoint."""
        results = []
        start_time = time.time()
        
        def make_request():
            return self.test_endpoint(endpoint)
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=concurrent_users) as executor:
            futures = []
            
            while time.time() - start_time < duration_seconds:
                future = executor.submit(make_request)
                futures.append(future)
                time.sleep(0.1)  # Small delay between requests
            
            # Collect results
            for future in concurrent.futures.as_completed(futures):
                try:
                    result = future.result()
                    results.append(result)
                except Exception as e:
                    print(f"Request failed: {e}")
        
        # Analyze results
        response_times = [r.response_time for r in results]
        success_count = sum(1 for r in results if r.success)
        
        return {
            "endpoint": endpoint,
            "total_requests": len(results),
            "successful_requests": success_count,
            "failed_requests": len(results) - success_count,
            "success_rate": success_count / len(results) if results else 0,
            "avg_response_time": statistics.mean(response_times) if response_times else 0,
            "min_response_time": min(response_times) if response_times else 0,
            "max_response_time": max(response_times) if response_times else 0,
            "p95_response_time": statistics.quantiles(response_times, n=20)[18] if len(response_times) > 20 else 0,
            "requests_per_second": len(results) / duration_seconds
        }

class APITestSuite:
    """Comprehensive API test suite."""
    
    def __init__(self, base_url: str):
        self.base_url = base_url
        self.test_client = APITestClient(base_url)
    
    def run_health_checks(self) -> List[TestResult]:
        """Run health check tests."""
        tests = [
            ("/health", "GET", 200),
            ("/health/detailed", "GET", 200),
            ("/health/live", "GET", 200),
            ("/health/ready", "GET", 200)
        ]
        
        results = []
        for endpoint, method, expected_status in tests:
            result = self.test_client.test_endpoint(endpoint, method, expected_status=expected_status)
            results.append(result)
        
        return results
    
    def run_authentication_tests(self) -> List[TestResult]:
        """Run authentication tests."""
        results = []
        
        # Test without authentication
        result = self.test_client.test_endpoint("/conversation", "POST", expected_status=401)
        results.append(result)
        
        # Test with invalid token
        invalid_client = APITestClient(self.base_url, jwt_token="invalid_token")
        result = invalid_client.test_endpoint("/conversation", "POST", expected_status=401)
        results.append(result)
        
        return results
    
    def run_rate_limit_tests(self) -> List[TestResult]:
        """Run rate limiting tests."""
        results = []
        
        # Make rapid requests to trigger rate limiting
        for i in range(150):  # Exceed typical rate limit
            result = self.test_client.test_endpoint("/health")
            results.append(result)
            
            # Check if we hit rate limit
            if result.status_code == 429:
                break
        
        return results
    
    def run_comprehensive_tests(self) -> Dict[str, Any]:
        """Run comprehensive test suite."""
        test_results = {
            "timestamp": datetime.now().isoformat(),
            "base_url": self.base_url,
            "tests": {}
        }
        
        # Health checks
        print("Running health check tests...")
        health_results = self.run_health_checks()
        test_results["tests"]["health_checks"] = {
            "results": [result.__dict__ for result in health_results],
            "success_rate": sum(1 for r in health_results if r.success) / len(health_results)
        }
        
        # Authentication tests
        print("Running authentication tests...")
        auth_results = self.run_authentication_tests()
        test_results["tests"]["authentication"] = {
            "results": [result.__dict__ for result in auth_results],
            "success_rate": sum(1 for r in auth_results if r.success) / len(auth_results)
        }
        
        # Rate limiting tests
        print("Running rate limiting tests...")
        rate_limit_results = self.run_rate_limit_tests()
        test_results["tests"]["rate_limiting"] = {
            "results": [result.__dict__ for result in rate_limit_results],
            "rate_limit_triggered": any(r.status_code == 429 for r in rate_limit_results)
        }
        
        return test_results

# CLI tool for API testing
def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='API Testing Tools')
    parser.add_argument('--base-url', required=True, help='Base URL of the API')
    parser.add_argument('--api-key', help='API key for authentication')
    parser.add_argument('--jwt-token', help='JWT token for authentication')
    parser.add_argument('--test-type', choices=['health', 'auth', 'rate-limit', 'load', 'comprehensive'], 
                       default='comprehensive', help='Type of test to run')
    parser.add_argument('--endpoint', help='Specific endpoint to test')
    parser.add_argument('--concurrent-users', type=int, default=10, help='Concurrent users for load testing')
    parser.add_argument('--duration', type=int, default=60, help='Duration for load testing (seconds)')
    parser.add_argument('--output', help='Output file for results')
    
    args = parser.parse_args()
    
    # Initialize test suite
    test_suite = APITestSuite(args.base_url)
    
    if args.api_key or args.jwt_token:
        test_suite.test_client = APITestClient(args.base_url, args.api_key, args.jwt_token)
    
    # Run tests based on type
    if args.test_type == 'health':
        results = test_suite.run_health_checks()
        output = {"health_checks": [r.__dict__ for r in results]}
    elif args.test_type == 'auth':
        results = test_suite.run_authentication_tests()
        output = {"authentication_tests": [r.__dict__ for r in results]}
    elif args.test_type == 'rate-limit':
        results = test_suite.run_rate_limit_tests()
        output = {"rate_limit_tests": [r.__dict__ for r in results]}
    elif args.test_type == 'load':
        if not args.endpoint:
            print("Error: --endpoint required for load testing")
            return
        results = test_suite.test_client.load_test(args.endpoint, args.concurrent_users, args.duration)
        output = {"load_test": results}
    else:  # comprehensive
        output = test_suite.run_comprehensive_tests()
    
    # Output results
    if args.output:
        with open(args.output, 'w') as f:
            json.dump(output, f, indent=2)
        print(f"Results saved to {args.output}")
    else:
        print(json.dumps(output, indent=2))

if __name__ == "__main__":
    main()
