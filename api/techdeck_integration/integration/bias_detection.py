"""
Bias Detection Integration Module for TechDeck-Python Pipeline Integration.

This module provides integration with the bias detection service for real-time
bias monitoring and validation of dataset processing operations.
"""

import asyncio
import json
import logging
from typing import Dict, List, Optional, Any
from datetime import datetime
import aiohttp
from dataclasses import dataclass
from enum import Enum

from ..error_handling.custom_errors import (
    BiasDetectionError,
    ServiceUnavailableError,
    ValidationError
)
from ..utils.logger import get_logger

logger = get_logger(__name__)


class BiasType(Enum):
    """Types of bias that can be detected."""
    GENDER = "gender"
    RACIAL = "racial"
    AGE = "age"
    SOCIOECONOMIC = "socioeconomic"
    GEOGRAPHIC = "geographic"
    DISABILITY = "disability"
    RELIGIOUS = "religious"
    SEXUAL_ORIENTATION = "sexual_orientation"


@dataclass
class BiasMetrics:
    """Bias detection metrics for a dataset or operation."""
    bias_type: BiasType
    score: float  # 0.0 to 1.0, higher indicates more bias
    severity: str  # "low", "medium", "high", "critical"
    affected_groups: List[str]
    confidence: float  # 0.0 to 1.0
    recommendations: List[str]
    metadata: Dict[str, Any]


@dataclass
class BiasDetectionResult:
    """Complete bias detection result."""
    operation_id: str
    dataset_id: str
    overall_bias_score: float
    bias_metrics: List[BiasMetrics]
    compliance_status: bool
    timestamp: datetime
    processing_time_ms: float
    warnings: List[str]


class BiasDetectionClient:
    """Client for interacting with the bias detection service."""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the bias detection client.
        
        Args:
            config: Configuration dictionary containing:
                - service_url: URL of the bias detection service
                - api_key: API key for authentication
                - timeout: Request timeout in seconds
                - max_retries: Maximum number of retry attempts
                - retry_delay: Delay between retries in seconds
        """
        self.service_url = config.get('service_url', 'http://localhost:8080')
        self.api_key = config.get('api_key', '')
        self.timeout = config.get('timeout', 30)
        self.max_retries = config.get('max_retries', 3)
        self.retry_delay = config.get('retry_delay', 1.0)
        self.session = None
        
        logger.info(f"Initialized BiasDetectionClient with URL: {self.service_url}")
    
    async def __aenter__(self):
        """Async context manager entry."""
        await self._ensure_session()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.close()
    
    async def _ensure_session(self):
        """Ensure aiohttp session is created."""
        if self.session is None or self.session.closed:
            timeout = aiohttp.ClientTimeout(total=self.timeout)
            headers = {
                'Authorization': f'Bearer {self.api_key}',
                'Content-Type': 'application/json',
                'User-Agent': 'TechDeck-Integration/1.0'
            }
            self.session = aiohttp.ClientSession(
                timeout=timeout,
                headers=headers
            )
    
    async def close(self):
        """Close the aiohttp session."""
        if self.session and not self.session.closed:
            await self.session.close()
    
    async def detect_bias(
        self,
        dataset_data: Dict[str, Any],
        operation_id: str,
        dataset_id: str
    ) -> BiasDetectionResult:
        """
        Detect bias in dataset data.
        
        Args:
            dataset_data: Dataset data to analyze
            operation_id: Unique operation identifier
            dataset_id: Dataset identifier
            
        Returns:
            BiasDetectionResult with bias analysis
            
        Raises:
            BiasDetectionError: If bias detection fails
            ServiceUnavailableError: If service is unavailable
        """
        start_time = datetime.utcnow()
        
        try:
            logger.info(f"Starting bias detection for operation {operation_id}")
            
            # Prepare request payload
            payload = {
                'dataset_data': dataset_data,
                'operation_id': operation_id,
                'dataset_id': dataset_id,
                'timestamp': start_time.isoformat(),
                'context': 'techdeck_pipeline_integration'
            }
            
            # Make request with retries
            result = await self._make_request_with_retry(
                'POST',
                f'{self.service_url}/api/v1/bias/detect',
                payload
            )
            
            # Parse response
            bias_result = self._parse_bias_detection_result(result, operation_id, dataset_id)
            
            processing_time = (datetime.utcnow() - start_time).total_seconds() * 1000
            
            logger.info(
                f"Bias detection completed for operation {operation_id} "
                f"in {processing_time:.2f}ms, overall score: {bias_result.overall_bias_score}"
            )
            
            return bias_result
            
        except Exception as e:
            logger.error(f"Bias detection failed for operation {operation_id}: {str(e)}")
            raise BiasDetectionError(f"Bias detection failed: {str(e)}") from e
    
    async def validate_bias_threshold(
        self,
        bias_result: BiasDetectionResult,
        threshold: float = 0.7
    ) -> bool:
        """
        Validate if bias scores are within acceptable thresholds.
        
        Args:
            bias_result: Bias detection result to validate
            threshold: Maximum acceptable bias score
            
        Returns:
            True if within threshold, False otherwise
            
        Raises:
            ValidationError: If validation fails
        """
        try:
            if bias_result.overall_bias_score > threshold:
                logger.warning(
                    f"Bias threshold exceeded for operation {bias_result.operation_id}: "
                    f"{bias_result.overall_bias_score} > {threshold}"
                )
                return False
            
            # Check individual bias types
            for metric in bias_result.bias_metrics:
                if metric.score > threshold:
                    logger.warning(
                        f"Specific bias threshold exceeded: {metric.bias_type.value} "
                        f"score {metric.score} > {threshold}"
                    )
                    return False
            
            logger.info(f"Bias validation passed for operation {bias_result.operation_id}")
            return True
            
        except Exception as e:
            logger.error(f"Bias validation failed: {str(e)}")
            raise ValidationError(f"Bias validation failed: {str(e)}") from e
    
    async def get_bias_recommendations(
        self,
        bias_result: BiasDetectionResult
    ) -> List[str]:
        """
        Get bias mitigation recommendations.
        
        Args:
            bias_result: Bias detection result
            
        Returns:
            List of recommendations for bias mitigation
        """
        recommendations = []
        
        try:
            # Collect all recommendations from bias metrics
            for metric in bias_result.bias_metrics:
                recommendations.extend(metric.recommendations)
            
            # Add general recommendations based on overall score
            if bias_result.overall_bias_score > 0.8:
                recommendations.append(
                    "High overall bias detected. Consider comprehensive data rebalancing."
                )
            elif bias_result.overall_bias_score > 0.6:
                recommendations.append(
                    "Moderate bias detected. Review data collection and sampling methods."
                )
            
            # Remove duplicates while preserving order
            seen = set()
            unique_recommendations = []
            for rec in recommendations:
                if rec not in seen:
                    seen.add(rec)
                    unique_recommendations.append(rec)
            
            return unique_recommendations
            
        except Exception as e:
            logger.error(f"Failed to get bias recommendations: {str(e)}")
            return ["Unable to generate bias mitigation recommendations"]
    
    async def _make_request_with_retry(
        self,
        method: str,
        url: str,
        payload: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Make HTTP request with retry logic.
        
        Args:
            method: HTTP method
            url: Request URL
            payload: Request payload
            
        Returns:
            Response data
            
        Raises:
            ServiceUnavailableError: If service is unavailable after retries
        """
        await self._ensure_session()
        
        for attempt in range(self.max_retries + 1):
            try:
                async with self.session.request(method, url, json=payload) as response:
                    if response.status == 200:
                        return await response.json()
                    elif response.status == 503:
                        if attempt < self.max_retries:
                            logger.warning(
                                f"Service unavailable (attempt {attempt + 1}), "
                                f"retrying in {self.retry_delay}s"
                            )
                            await asyncio.sleep(self.retry_delay)
                            continue
                        else:
                            raise ServiceUnavailableError(
                                "Bias detection service unavailable after all retries"
                            )
                    else:
                        error_text = await response.text()
                        raise BiasDetectionError(
                            f"Bias detection service returned {response.status}: {error_text}"
                        )
                        
            except aiohttp.ClientError as e:
                if attempt < self.max_retries:
                    logger.warning(
                        f"Request failed (attempt {attempt + 1}): {str(e)}, "
                        f"retrying in {self.retry_delay}s"
                    )
                    await asyncio.sleep(self.retry_delay)
                    continue
                else:
                    raise ServiceUnavailableError(
                        f"Failed to connect to bias detection service after {self.max_retries} retries: {str(e)}"
                    )
        
        raise ServiceUnavailableError("All retry attempts exhausted")
    
    def _parse_bias_detection_result(
        self,
        result: Dict[str, Any],
        operation_id: str,
        dataset_id: str
    ) -> BiasDetectionResult:
        """
        Parse bias detection API response into BiasDetectionResult.
        
        Args:
            result: API response data
            operation_id: Operation identifier
            dataset_id: Dataset identifier
            
        Returns:
            Parsed BiasDetectionResult
        """
        try:
            bias_metrics = []
            for metric_data in result.get('bias_metrics', []):
                bias_type = BiasType(metric_data['bias_type'])
                metric = BiasMetrics(
                    bias_type=bias_type,
                    score=float(metric_data['score']),
                    severity=metric_data['severity'],
                    affected_groups=metric_data.get('affected_groups', []),
                    confidence=float(metric_data.get('confidence', 0.0)),
                    recommendations=metric_data.get('recommendations', []),
                    metadata=metric_data.get('metadata', {})
                )
                bias_metrics.append(metric)
            
            return BiasDetectionResult(
                operation_id=operation_id,
                dataset_id=dataset_id,
                overall_bias_score=float(result.get('overall_bias_score', 0.0)),
                bias_metrics=bias_metrics,
                compliance_status=bool(result.get('compliance_status', True)),
                timestamp=datetime.fromisoformat(result.get('timestamp', datetime.utcnow().isoformat())),
                processing_time_ms=float(result.get('processing_time_ms', 0.0)),
                warnings=result.get('warnings', [])
            )
            
        except (KeyError, ValueError, TypeError) as e:
            logger.error(f"Failed to parse bias detection result: {str(e)}")
            raise BiasDetectionError(f"Invalid bias detection response format: {str(e)}") from e


class BiasDetectionManager:
    """Manager for bias detection operations with caching and monitoring."""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the bias detection manager.
        
        Args:
            config: Configuration dictionary
        """
        self.client = BiasDetectionClient(config)
        self.cache_ttl = config.get('bias_detection_cache_ttl', 3600)  # 1 hour default
        self.enabled = config.get('bias_detection_enabled', True)
        self.default_threshold = config.get('bias_threshold', 0.7)
        
        logger.info(f"Initialized BiasDetectionManager (enabled: {self.enabled})")
    
    async def analyze_dataset(
        self,
        dataset_data: Dict[str, Any],
        operation_id: str,
        dataset_id: str,
        validate_threshold: Optional[float] = None
    ) -> BiasDetectionResult:
        """
        Analyze dataset for bias with optional threshold validation.
        
        Args:
            dataset_data: Dataset data to analyze
            operation_id: Unique operation identifier
            dataset_id: Dataset identifier
            validate_threshold: Optional threshold for validation
            
        Returns:
            BiasDetectionResult with analysis and validation
            
        Raises:
            BiasDetectionError: If bias detection fails
            ValidationError: If validation fails and threshold is exceeded
        """
        if not self.enabled:
            logger.info(f"Bias detection disabled, skipping analysis for {operation_id}")
            return self._create_dummy_result(operation_id, dataset_id)
        
        threshold = validate_threshold or self.default_threshold
        
        try:
            # Perform bias detection
            result = await self.client.detect_bias(
                dataset_data, operation_id, dataset_id
            )
            
            # Validate against threshold if requested
            if validate_threshold is not None:
                is_valid = await self.client.validate_bias_threshold(result, threshold)
                if not is_valid:
                    raise ValidationError(
                        f"Bias threshold {threshold} exceeded: "
                        f"overall score {result.overall_bias_score}"
                    )
            
            return result
            
        except Exception as e:
            logger.error(f"Dataset bias analysis failed for {operation_id}: {str(e)}")
            raise
    
    async def get_bias_summary(
        self,
        bias_result: BiasDetectionResult
    ) -> Dict[str, Any]:
        """
        Get a summary of bias detection results.
        
        Args:
            bias_result: Bias detection result
            
        Returns:
            Summary dictionary with key metrics and recommendations
        """
        try:
            recommendations = await self.client.get_bias_recommendations(bias_result)
            
            summary = {
                'operation_id': bias_result.operation_id,
                'dataset_id': bias_result.dataset_id,
                'overall_bias_score': bias_result.overall_bias_score,
                'compliance_status': bias_result.compliance_status,
                'bias_types_detected': [
                    metric.bias_type.value for metric in bias_result.bias_metrics
                    if metric.score > 0.1  # Only significant bias
                ],
                'high_risk_bias_types': [
                    metric.bias_type.value for metric in bias_result.bias_metrics
                    if metric.severity in ['high', 'critical']
                ],
                'recommendations': recommendations[:5],  # Top 5 recommendations
                'processing_time_ms': bias_result.processing_time_ms,
                'timestamp': bias_result.timestamp.isoformat()
            }
            
            return summary
            
        except Exception as e:
            logger.error(f"Failed to generate bias summary: {str(e)}")
            return {
                'error': f"Failed to generate bias summary: {str(e)}",
                'operation_id': bias_result.operation_id,
                'dataset_id': bias_result.dataset_id
            }
    
    def _create_dummy_result(
        self,
        operation_id: str,
        dataset_id: str
    ) -> BiasDetectionResult:
        """Create a dummy result when bias detection is disabled."""
        return BiasDetectionResult(
            operation_id=operation_id,
            dataset_id=dataset_id,
            overall_bias_score=0.0,
            bias_metrics=[],
            compliance_status=True,
            timestamp=datetime.utcnow(),
            processing_time_ms=0.0,
            warnings=["Bias detection is disabled"]
        )


# Convenience functions for direct usage
async def detect_bias_in_dataset(
    dataset_data: Dict[str, Any],
    operation_id: str,
    dataset_id: str,
    config: Dict[str, Any]
) -> BiasDetectionResult:
    """
    Convenience function to detect bias in dataset data.
    
    Args:
        dataset_data: Dataset data to analyze
        operation_id: Unique operation identifier
        dataset_id: Dataset identifier
        config: Configuration dictionary
        
    Returns:
        BiasDetectionResult with analysis
    """
    manager = BiasDetectionManager(config)
    try:
        return await manager.analyze_dataset(dataset_data, operation_id, dataset_id)
    finally:
        await manager.client.close()


async def validate_bias_compliance(
    bias_result: BiasDetectionResult,
    threshold: float = 0.7
) -> bool:
    """
    Validate bias compliance against threshold.
    
    Args:
        bias_result: Bias detection result to validate
        threshold: Maximum acceptable bias score
        
    Returns:
        True if compliant, False otherwise
    """
    client = BiasDetectionClient({'service_url': 'http://localhost:8080'})
    try:
        return await client.validate_bias_threshold(bias_result, threshold)
    finally:
        await client.close()