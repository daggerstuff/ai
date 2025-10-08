#!/usr/bin/env python3
"""
Pixelated Empathy AI - Python Client Library
Task 51: Complete API Documentation

Official Python client for the Pixelated Empathy AI API.
"""

import requests
import json
import time
from typing import Dict, List, Any, Optional, Union, Iterator
from dataclasses import dataclass
from datetime import datetime
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class APIResponse:
    """Standard API response wrapper."""
    success: bool
    data: Any
    message: str
    timestamp: str
    error: Optional[Dict[str, Any]] = None

class PixelatedEmpathyAPIError(Exception):
    """Base exception for API errors."""
    def __init__(self, message: str, error_code: str = None, status_code: int = None):
        self.message = message
        self.error_code = error_code
        self.status_code = status_code
        super().__init__(self.message)

class RateLimitError(PixelatedEmpathyAPIError):
    """Exception raised when rate limit is exceeded."""
    def __init__(self, retry_after: int):
        self.retry_after = retry_after
        super().__init__(f"Rate limit exceeded. Retry after {retry_after} seconds.")

class PixelatedEmpathyAPI:
    """
    Official Python client for the Pixelated Empathy AI API.
    
    Provides access to 2.59M+ therapeutic conversations with enterprise-grade
    quality validation, real-time processing, and advanced search capabilities.
    """
    
    def __init__(self, 
                 api_key: str, 
                 base_url: str = "https://api.pixelatedempathy.com/v1",
                 timeout: int = 30,
                 max_retries: int = 3):
        """
        Initialize the API client.
        
        Args:
            api_key: Your API key from https://api.pixelatedempathy.com
            base_url: API base URL (default: production)
            timeout: Request timeout in seconds
            max_retries: Maximum number of retries for failed requests
        """
        self.api_key = api_key
        self.base_url = base_url.rstrip('/')
        self.timeout = timeout
        self.max_retries = max_retries
        
        self.session = requests.Session()
        self.session.headers.update({
            'Authorization': f'Bearer {api_key}',
            'Content-Type': 'application/json',
            'User-Agent': 'PixelatedEmpathyAPI-Python/1.0.0'
        })
    
    def _make_request(self, method: str, endpoint: str, **kwargs) -> APIResponse:
        """Make an HTTP request with error handling and retries."""
        url = f"{self.base_url}{endpoint}"
        
        for attempt in range(self.max_retries + 1):
            try:
                response = self.session.request(
                    method, url, timeout=self.timeout, **kwargs
                )
                
                # Handle rate limiting
                if response.status_code == 429:
                    retry_after = int(response.headers.get('Retry-After', 60))
                    if attempt < self.max_retries:
                        logger.warning(f"Rate limited. Retrying after {retry_after} seconds...")
                        time.sleep(retry_after)
                        continue
                    else:
                        raise RateLimitError(retry_after)
                
                # Parse response
                try:
                    data = response.json()
                except json.JSONDecodeError:
                    data = {"success": False, "message": "Invalid JSON response"}
                
                # Handle API errors
                if not response.ok:
                    error_message = data.get('error', {}).get('message', 'Unknown error')
                    error_code = data.get('error', {}).get('code', 'UNKNOWN_ERROR')
                    raise PixelatedEmpathyAPIError(
                        error_message, error_code, response.status_code
                    )
                
                return APIResponse(
                    success=data.get('success', False),
                    data=data.get('data'),
                    message=data.get('message', ''),
                    timestamp=data.get('timestamp', ''),
                    error=data.get('error')
                )
                
            except requests.exceptions.RequestException as e:
                if attempt < self.max_retries:
                    logger.warning(f"Request failed (attempt {attempt + 1}): {e}")
                    time.sleep(2 ** attempt)  # Exponential backoff
                    continue
                else:
                    raise PixelatedEmpathyAPIError(f"Request failed: {e}")
    
    # Dataset methods
    def list_datasets(self) -> List[Dict[str, Any]]:
        """
        List all available datasets.
        
        Returns:
            List of dataset information dictionaries
        """
        response = self._make_request('GET', '/datasets')
        return response.data.get('datasets', [])
    
    def get_dataset_info(self, dataset_name: str) -> Dict[str, Any]:
        """
        Get detailed information about a specific dataset.
        
        Args:
            dataset_name: Name of the dataset
            
        Returns:
            Dataset information dictionary
        """
        response = self._make_request('GET', f'/datasets/{dataset_name}')
        return response.data
    
    # Conversation methods
    def get_conversations(self, 
                         dataset: Optional[str] = None,
                         tier: Optional[str] = None,
                         min_quality: Optional[float] = None,
                         limit: int = 100,
                         offset: int = 0) -> Dict[str, Any]:
        """
        Get conversations with optional filtering.
        
        Args:
            dataset: Filter by dataset name
            tier: Filter by quality tier (basic, standard, professional, clinical, research)
            min_quality: Minimum quality score (0.0-1.0)
            limit: Maximum number of results (1-1000)
            offset: Offset for pagination
            
        Returns:
            Dictionary with conversations list and pagination info
        """
        params = {'limit': limit, 'offset': offset}
        if dataset:
            params['dataset'] = dataset
        if tier:
            params['tier'] = tier
        if min_quality is not None:
            params['min_quality'] = min_quality
        
        response = self._make_request('GET', '/conversations', params=params)
        return response.data
    
    def get_conversation(self, conversation_id: str) -> Dict[str, Any]:
        """
        Get a specific conversation by ID.
        
        Args:
            conversation_id: Unique conversation identifier
            
        Returns:
            Conversation details dictionary
        """
        response = self._make_request('GET', f'/conversations/{conversation_id}')
        return response.data
    
    def iter_conversations(self, 
                          dataset: Optional[str] = None,
                          tier: Optional[str] = None,
                          min_quality: Optional[float] = None,
                          batch_size: int = 100) -> Iterator[Dict[str, Any]]:
        """
        Iterate through all conversations with automatic pagination.
        
        Args:
            dataset: Filter by dataset name
            tier: Filter by quality tier
            min_quality: Minimum quality score
            batch_size: Number of conversations to fetch per request
            
        Yields:
            Individual conversation dictionaries
        """
        offset = 0
        while True:
            batch = self.get_conversations(
                dataset=dataset, tier=tier, min_quality=min_quality,
                limit=batch_size, offset=offset
            )
            
            conversations = batch.get('conversations', [])
            if not conversations:
                break
            
            for conversation in conversations:
                yield conversation
            
            offset += len(conversations)
            
            # Check if we've reached the end
            if len(conversations) < batch_size:
                break
    
    # Quality methods
    def get_quality_metrics(self, 
                           dataset: Optional[str] = None,
                           tier: Optional[str] = None) -> Dict[str, Any]:
        """
        Get quality metrics for datasets or tiers.
        
        Args:
            dataset: Filter by dataset name
            tier: Filter by quality tier
            
        Returns:
            Quality metrics dictionary
        """
        params = {}
        if dataset:
            params['dataset'] = dataset
        if tier:
            params['tier'] = tier
        
        response = self._make_request('GET', '/quality/metrics', params=params)
        return response.data
    
    def validate_conversation_quality(self, conversation: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate the quality of a conversation using NLP-based assessment.
        
        Args:
            conversation: Conversation dictionary with id, messages, etc.
            
        Returns:
            Quality validation results dictionary
        """
        response = self._make_request('POST', '/quality/validate', json=conversation)
        return response.data
    
    # Processing methods
    def submit_processing_job(self, 
                             dataset_name: str,
                             processing_type: str,
                             parameters: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Submit a processing job for dataset analysis or export.
        
        Args:
            dataset_name: Name of the dataset to process
            processing_type: Type of processing (quality_validation, export, analysis)
            parameters: Processing parameters dictionary
            
        Returns:
            Job information dictionary
        """
        job_data = {
            'dataset_name': dataset_name,
            'processing_type': processing_type,
            'parameters': parameters or {}
        }
        
        response = self._make_request('POST', '/processing/submit', json=job_data)
        return response.data
    
    def get_job_status(self, job_id: str) -> Dict[str, Any]:
        """
        Get the status of a processing job.
        
        Args:
            job_id: Unique job identifier
            
        Returns:
            Job status dictionary
        """
        response = self._make_request('GET', f'/processing/jobs/{job_id}')
        return response.data
    
    def wait_for_job(self, job_id: str, 
                     poll_interval: int = 30,
                     timeout: int = 3600) -> Dict[str, Any]:
        """
        Wait for a processing job to complete.
        
        Args:
            job_id: Unique job identifier
            poll_interval: Seconds between status checks
            timeout: Maximum time to wait in seconds
            
        Returns:
            Final job status dictionary
        """
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            status = self.get_job_status(job_id)
            
            if status['status'] in ['completed', 'failed', 'cancelled']:
                return status
            
            logger.info(f"Job {job_id} status: {status['status']} "
                       f"({status.get('progress', 0)}%)")
            time.sleep(poll_interval)
        
        raise PixelatedEmpathyAPIError(f"Job {job_id} did not complete within {timeout} seconds")
    
    # Search methods
    def search_conversations(self, 
                           query: str,
                           filters: Dict[str, Any] = None,
                           limit: int = 100,
                           offset: int = 0) -> Dict[str, Any]:
        """
        Search conversations using advanced filters and full-text search.
        
        Args:
            query: Search query string
            filters: Search filters dictionary
            limit: Maximum number of results
            offset: Offset for pagination
            
        Returns:
            Search results dictionary
        """
        search_data = {
            'query': query,
            'filters': filters or {},
            'limit': limit,
            'offset': offset
        }
        
        response = self._make_request('POST', '/search', json=search_data)
        return response.data
    
    # Statistics methods
    def get_statistics_overview(self) -> Dict[str, Any]:
        """
        Get comprehensive statistics about the API and datasets.
        
        Returns:
            Statistics overview dictionary
        """
        response = self._make_request('GET', '/statistics/overview')
        return response.data
    
    # Export methods
    def export_data(self, 
                   dataset: str,
                   format: str = 'jsonl',
                   tier: Optional[str] = None,
                   min_quality: Optional[float] = None) -> Dict[str, Any]:
        """
        Export data in specified format with optional filtering.
        
        Args:
            dataset: Dataset name to export
            format: Export format (jsonl, csv, parquet, huggingface, openai)
            tier: Filter by quality tier
            min_quality: Minimum quality score
            
        Returns:
            Export information dictionary
        """
        export_data = {
            'dataset': dataset,
            'format': format
        }
        if tier:
            export_data['tier'] = tier
        if min_quality is not None:
            export_data['min_quality'] = min_quality
        
        response = self._make_request('POST', '/export', data=export_data)
        return response.data
    
    # Utility methods
    def health_check(self) -> bool:
        """
        Check if the API is healthy.
        
        Returns:
            True if API is healthy, False otherwise
        """
        try:
            response = self._make_request('GET', '/health')
            return response.success
        except Exception:
            return False

# Example usage
if __name__ == "__main__":
    # Initialize client
    api = PixelatedEmpathyAPI("your_api_key_here")
    
    # Check API health
    if not api.health_check():
        print("API is not available")
        exit(1)
    
    # List datasets
    datasets = api.list_datasets()
    print(f"Available datasets: {len(datasets)}")
    for dataset in datasets:
        print(f"  - {dataset['name']}: {dataset['conversations']} conversations")
    
    # Get professional conversations
    conversations = api.get_conversations(tier="professional", limit=5)
    print(f"\nFound {len(conversations['conversations'])} professional conversations")
    
    # Search for anxiety-related conversations
    search_results = api.search_conversations(
        "anxiety therapy techniques",
        filters={"tier": "professional", "min_quality": 0.7}
    )
    print(f"\nFound {search_results['total_matches']} matching conversations")
    
    # Get quality metrics
    metrics = api.get_quality_metrics()
    print(f"\nOverall quality metrics:")
    print(f"  Average quality: {metrics['overall_statistics']['average_quality']}")
    print(f"  Total conversations: {metrics['overall_statistics']['total_conversations']}")
    
    # Example conversation quality validation
    sample_conversation = {
        "id": "test_conv_001",
        "messages": [
            {"role": "user", "content": "I'm feeling anxious about my job interview tomorrow."},
            {"role": "assistant", "content": "It's completely natural to feel anxious before an important interview. Can you tell me what specific aspects are making you most worried?"}
        ],
        "quality_score": 0.0,
        "tier": "unknown"
    }
    
    validation_result = api.validate_conversation_quality(sample_conversation)
    print(f"\nConversation quality validation:")
    print(f"  Overall quality: {validation_result['validation_results']['overall_quality']}")
    print(f"  Tier classification: {validation_result['tier_classification']}")
