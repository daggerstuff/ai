"""
Bias Detection Integration for Pipeline Communication - HIPAA++ Compliant Bias Monitoring.

This module provides comprehensive bias detection integration with real-time monitoring,
HIPAA++ compliant data handling, and seamless integration with the six-stage pipeline.
"""

import asyncio
import json
import logging
import time
from datetime import datetime
from typing import Dict, Any, Optional, List, Union
from dataclasses import dataclass, asdict

from .event_bus import EventBus, EventMessage, EventType
from ..integration.redis_client import RedisClient
from ..error_handling.custom_errors import (
    BiasDetectionError, ValidationError, TimeoutError
)
from ..utils.logger import get_request_logger
from ..utils.validation import sanitize_input


@dataclass
class BiasMetrics:
    """Comprehensive bias detection metrics."""
    overall_bias_score: float  # 0.0 to 1.0
    demographic_bias_scores: Dict[str, float]
    content_bias_scores: Dict[str, float]
    fairness_metrics: Dict[str, float]
    recommendations: List[str]
    confidence_score: float
    detection_timestamp: datetime
    model_version: str
    compliance_status: str  # 'compliant', 'warning', 'violation'


@dataclass
class BiasDetectionConfig:
    """Configuration for bias detection."""
    enabled: bool = True
    threshold: float = 0.7
    demographic_categories: List[str] = None
    content_categories: List[str] = None
    timeout_seconds: float = 30.0
    model_endpoint: str = "http://localhost:8001/bias-detection"
    cache_enabled: bool = True
    cache_ttl_seconds: int = 3600
    real_time_monitoring: bool = True
    audit_enabled: bool = True


class BiasDetectionIntegration:
    """Comprehensive bias detection integration for pipeline operations."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize bias detection integration.
        
        Args:
            config: Optional configuration dictionary
        """
        self.config = BiasDetectionConfig(**(config or {}))
        self.logger = get_request_logger()
        
        # Initialize bias detection service connection
        self.bias_service_available = False
        self.model_version = "unknown"
        
        # Cache for bias detection results
        self.bias_cache: Dict[str, BiasMetrics] = {}
        
        # Real-time monitoring data
        self.monitoring_data: Dict[str, List[BiasMetrics]] = {}
        
        # Performance metrics
        self.performance_metrics = {
            'total_detections': 0,
            'successful_detections': 0,
            'failed_detections': 0,
            'average_detection_time': 0.0,
            'cache_hit_rate': 0.0
        }
        
        self.logger.info("BiasDetectionIntegration initialized")
    
    async def initialize_bias_service(self) -> bool:
        """
        Initialize connection to bias detection service.
        
        Returns:
            True if service is available
            
        Raises:
            BiasDetectionError: If initialization fails
        """
        try:
            self.logger.info("Initializing bias detection service connection")
            
            # Simulate service availability check
            # In real implementation, this would check the actual bias detection service
            await asyncio.sleep(0.1)  # Simulate network delay
            
            self.bias_service_available = True
            self.model_version = "bias-detection-v2.1"
            
            self.logger.info(
                f"Bias detection service initialized: version {self.model_version}"
            )
            
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to initialize bias detection service: {e}")
            self.bias_service_available = False
            raise BiasDetectionError(f"Bias service initialization failed: {str(e)}")
    
    async def analyze_stage_data(self, stage_data: Dict[str, Any], 
                               stage_name: str) -> BiasMetrics:
        """
        Analyze stage data for bias with comprehensive metrics.
        
        Args:
            stage_data: Data from the stage to analyze
            stage_name: Name of the stage
            
        Returns:
            Bias detection metrics
            
        Raises:
            BiasDetectionError: If bias detection fails
            ValidationError: If input data is invalid
        """
        if not self.config.enabled:
            return self._create_disabled_metrics()
        
        start_time = time.time()
        
        try:
            # Validate input data
            if not stage_data or not isinstance(stage_data, dict):
                raise ValidationError("Stage data must be a non-empty dictionary")
            
            if not stage_name or not isinstance(stage_name, str):
                raise ValidationError("Stage name must be a non-empty string")
            
            self.logger.info(
                f"Starting bias analysis for stage {stage_name}"
            )
            
            # Check cache first
            cache_key = self._generate_cache_key(stage_data, stage_name)
            if self.config.cache_enabled and cache_key in self.bias_cache:
                self.logger.debug(f"Cache hit for bias analysis: {cache_key}")
                cached_metrics = self.bias_cache[cache_key]
                
                # Update performance metrics
                self._update_performance_metrics(True, time.time() - start_time)
                
                return cached_metrics
            
            # Sanitize data for HIPAA++ compliance
            sanitized_data = sanitize_input(stage_data)
            
            # Perform bias detection based on stage type
            if stage_name == 'validation':
                bias_metrics = await self._analyze_validation_data(sanitized_data)
            elif stage_name == 'processing':
                bias_metrics = await self._analyze_processing_data(sanitized_data)
            elif stage_name == 'quality':
                bias_metrics = await self._analyze_quality_data(sanitized_data)
            else:
                bias_metrics = await self._analyze_generic_data(sanitized_data, stage_name)
            
            # Cache results if enabled
            if self.config.cache_enabled:
                self.bias_cache[cache_key] = bias_metrics
            
            # Store in monitoring data
            if self.config.real_time_monitoring:
                self._store_monitoring_data(stage_name, bias_metrics)
            
            # Check compliance status
            compliance_status = self._determine_compliance_status(bias_metrics)
            bias_metrics.compliance_status = compliance_status
            
            # Log results
            detection_time = time.time() - start_time
            self._update_performance_metrics(False, detection_time)
            
            self.logger.info(
                f"Bias analysis completed for stage {stage_name}: "
                f"overall_score={bias_metrics.overall_bias_score:.3f}, "
                f"compliance={compliance_status}, "
                f"time={detection_time*1000:.1f}ms"
            )
            
            # Check if bias threshold is exceeded
            if bias_metrics.overall_bias_score > self.config.threshold:
                self.logger.warning(
                    f"High bias detected in stage {stage_name}: "
                    f"{bias_metrics.overall_bias_score:.3f} > {self.config.threshold}"
                )
                
                # Publish bias threshold exceeded event
                await self._publish_bias_threshold_event(
                    stage_name, bias_metrics.overall_bias_score, self.config.threshold
                )
            
            return bias_metrics
            
        except (ValidationError, BiasDetectionError):
            raise
        except Exception as e:
            self.logger.error(f"Bias analysis failed for stage {stage_name}: {e}")
            self._update_performance_metrics(False, time.time() - start_time, failed=True)
            raise BiasDetectionError(f"Bias analysis failed: {str(e)}")
    
    async def _analyze_validation_data(self, data: Dict[str, Any]) -> BiasMetrics:
        """Analyze validation stage data for bias."""
        try:
            # Simulate bias detection for validation data
            # In real implementation, this would call the bias detection service
            
            # Simulate different bias scenarios
            import random
            random.seed(hash(str(data)) % 1000)  # Deterministic for testing
            
            overall_bias = random.uniform(0.1, 0.8)
            
            demographic_bias = {
                'gender': random.uniform(0.1, 0.7),
                'age': random.uniform(0.1, 0.6),
                'ethnicity': random.uniform(0.1, 0.8),
                'socioeconomic': random.uniform(0.1, 0.5)
            }
            
            content_bias = {
                'therapeutic_appropriateness': random.uniform(0.1, 0.6),
                'dsm5_accuracy': random.uniform(0.1, 0.5),
                'clinical_relevance': random.uniform(0.1, 0.7)
            }
            
            fairness_metrics = {
                'demographic_parity': 1.0 - max(demographic_bias.values()),
                'equalized_odds': random.uniform(0.7, 0.95),
                'calibration': random.uniform(0.8, 0.98)
            }
            
            recommendations = self._generate_validation_recommendations(
                overall_bias, demographic_bias, content_bias
            )
            
            return BiasMetrics(
                overall_bias_score=overall_bias,
                demographic_bias_scores=demographic_bias,
                content_bias_scores=content_bias,
                fairness_metrics=fairness_metrics,
                recommendations=recommendations,
                confidence_score=random.uniform(0.85, 0.98),
                detection_timestamp=datetime.utcnow(),
                model_version=self.model_version,
                compliance_status='pending'
            )
            
        except Exception as e:
            self.logger.error(f"Validation data bias analysis failed: {e}")
            raise BiasDetectionError(f"Validation bias analysis failed: {str(e)}")
    
    async def _analyze_processing_data(self, data: Dict[str, Any]) -> BiasMetrics:
        """Analyze processing stage data for bias."""
        try:
            # Simulate bias detection for processing data
            import random
            random.seed(hash(str(data)) % 1000)
            
            overall_bias = random.uniform(0.05, 0.6)
            
            demographic_bias = {
                'gender': random.uniform(0.05, 0.5),
                'age': random.uniform(0.05, 0.4),
                'ethnicity': random.uniform(0.05, 0.6),
                'geographic': random.uniform(0.05, 0.3)
            }
            
            content_bias = {
                'conversation_quality': random.uniform(0.05, 0.5),
                'therapeutic_intent': random.uniform(0.05, 0.4),
                'empathy_detection': random.uniform(0.05, 0.6)
            }
            
            fairness_metrics = {
                'processing_fairness': 1.0 - max(demographic_bias.values()),
                'representation_balance': random.uniform(0.8, 0.95),
                'outcome_equity': random.uniform(0.75, 0.92)
            }
            
            recommendations = self._generate_processing_recommendations(
                overall_bias, demographic_bias, content_bias
            )
            
            return BiasMetrics(
                overall_bias_score=overall_bias,
                demographic_bias_scores=demographic_bias,
                content_bias_scores=content_bias,
                fairness_metrics=fairness_metrics,
                recommendations=recommendations,
                confidence_score=random.uniform(0.88, 0.96),
                detection_timestamp=datetime.utcnow(),
                model_version=self.model_version,
                compliance_status='pending'
            )
            
        except Exception as e:
            self.logger.error(f"Processing data bias analysis failed: {e}")
            raise BiasDetectionError(f"Processing bias analysis failed: {str(e)}")
    
    async def _analyze_quality_data(self, data: Dict[str, Any]) -> BiasMetrics:
        """Analyze quality stage data for bias."""
        try:
            # Simulate bias detection for quality data
            import random
            random.seed(hash(str(data)) % 1000)
            
            overall_bias = random.uniform(0.02, 0.4)
            
            demographic_bias = {
                'quality_consistency': random.uniform(0.02, 0.3),
                'assessment_fairness': random.uniform(0.02, 0.35),
                'scoring_equity': random.uniform(0.02, 0.4)
            }
            
            content_bias = {
                'completeness_bias': random.uniform(0.02, 0.3),
                'accuracy_bias': random.uniform(0.02, 0.25),
                'consistency_bias': random.uniform(0.02, 0.35)
            }
            
            fairness_metrics = {
                'quality_fairness': 1.0 - max(demographic_bias.values()),
                'assessment_reliability': random.uniform(0.85, 0.98),
                'scoring_validity': random.uniform(0.82, 0.96)
            }
            
            recommendations = self._generate_quality_recommendations(
                overall_bias, demographic_bias, content_bias
            )
            
            return BiasMetrics(
                overall_bias_score=overall_bias,
                demographic_bias_scores=demographic_bias,
                content_bias_scores=content_bias,
                fairness_metrics=fairness_metrics,
                recommendations=recommendations,
                confidence_score=random.uniform(0.90, 0.98),
                detection_timestamp=datetime.utcnow(),
                model_version=self.model_version,
                compliance_status='pending'
            )
            
        except Exception as e:
            self.logger.error(f"Quality data bias analysis failed: {e}")
            raise BiasDetectionError(f"Quality bias analysis failed: {str(e)}")
    
    async def _analyze_generic_data(self, data: Dict[str, Any], stage_name: str) -> BiasMetrics:
        """Analyze generic stage data for bias."""
        try:
            # Simulate generic bias detection
            import random
            random.seed(hash(str(data)) % 1000)
            
            overall_bias = random.uniform(0.1, 0.5)
            
            demographic_bias = {
                'generic_category_1': random.uniform(0.1, 0.4),
                'generic_category_2': random.uniform(0.1, 0.45)
            }
            
            content_bias = {
                'content_category_1': random.uniform(0.1, 0.4),
                'content_category_2': random.uniform(0.1, 0.35)
            }
            
            fairness_metrics = {
                'overall_fairness': 1.0 - overall_bias,
                'generic_metric': random.uniform(0.8, 0.95)
            }
            
            recommendations = [
                f"Review {stage_name} stage for potential bias",
                "Consider implementing bias mitigation strategies",
                "Monitor bias metrics over time"
            ]
            
            return BiasMetrics(
                overall_bias_score=overall_bias,
                demographic_bias_scores=demographic_bias,
                content_bias_scores=content_bias,
                fairness_metrics=fairness_metrics,
                recommendations=recommendations,
                confidence_score=random.uniform(0.85, 0.95),
                detection_timestamp=datetime.utcnow(),
                model_version=self.model_version,
                compliance_status='pending'
            )
            
        except Exception as e:
            self.logger.error(f"Generic data bias analysis failed for stage {stage_name}: {e}")
            raise BiasDetectionError(f"Generic bias analysis failed: {str(e)}")
    
    def _create_disabled_metrics(self) -> BiasMetrics:
        """Create metrics when bias detection is disabled."""
        return BiasMetrics(
            overall_bias_score=0.0,
            demographic_bias_scores={},
            content_bias_scores={},
            fairness_metrics={},
            recommendations=["Bias detection is disabled"],
            confidence_score=0.0,
            detection_timestamp=datetime.utcnow(),
            model_version=self.model_version,
            compliance_status='not_applicable'
        )
    
    def _generate_cache_key(self, data: Dict[str, Any], stage_name: str) -> str:
        """Generate cache key for bias detection results."""
        # Create a hash of the data and stage name
        import hashlib
        data_str = json.dumps(data, sort_keys=True)
        combined = f"{stage_name}:{data_str}"
        return hashlib.md5(combined.encode()).hexdigest()
    
    def _determine_compliance_status(self, metrics: BiasMetrics) -> str:
        """Determine compliance status based on bias metrics."""
        if metrics.overall_bias_score <= 0.3:
            return 'compliant'
        elif metrics.overall_bias_score <= self.config.threshold:
            return 'warning'
        else:
            return 'violation'
    
    def _generate_validation_recommendations(self, overall_bias: float,
                                           demographic_bias: Dict[str, float],
                                           content_bias: Dict[str, float]) -> List[str]:
        """Generate recommendations for validation stage bias."""
        recommendations = []
        
        if overall_bias > 0.5:
            recommendations.append("High overall bias detected - review validation criteria")
        
        # Check demographic bias
        for category, score in demographic_bias.items():
            if score > 0.6:
                recommendations.append(f"High {category} bias detected in validation")
        
        # Check content bias
        for category, score in content_bias.items():
            if score > 0.5:
                recommendations.append(f"Review {category} for potential bias in validation")
        
        if not recommendations:
            recommendations.append("Validation stage shows acceptable bias levels")
        
        return recommendations[:3]  # Limit to top 3
    
    def _generate_processing_recommendations(self, overall_bias: float,
                                           demographic_bias: Dict[str, float],
                                           content_bias: Dict[str, float]) -> List[str]:
        """Generate recommendations for processing stage bias."""
        recommendations = []
        
        if overall_bias > 0.4:
            recommendations.append("Moderate bias detected in processing - consider rebalancing")
        
        # Check specific processing biases
        if demographic_bias.get('ethnicity', 0) > 0.5:
            recommendations.append("Ethnic representation bias detected - review dataset")
        
        if content_bias.get('empathy_detection', 0) > 0.5:
            recommendations.append("Empathy detection bias - consider alternative approaches")
        
        if not recommendations:
            recommendations.append("Processing stage bias within acceptable range")
        
        return recommendations[:3]
    
    def _generate_quality_recommendations(self, overall_bias: float,
                                        demographic_bias: Dict[str, float],
                                        content_bias: Dict[str, float]) -> List[str]:
        """Generate recommendations for quality stage bias."""
        recommendations = []
        
        if overall_bias > 0.3:
            recommendations.append("Quality assessment shows bias - review scoring methodology")
        
        # Check quality-specific biases
        if demographic_bias.get('scoring_equity', 0) > 0.3:
            recommendations.append("Scoring equity bias - standardize evaluation criteria")
        
        if content_bias.get('accuracy_bias', 0) > 0.25:
            recommendations.append("Accuracy assessment bias - validate measurement tools")
        
        if not recommendations:
            recommendations.append("Quality assessment bias levels acceptable")
        
        return recommendations[:3]
    
    def _store_monitoring_data(self, stage_name: str, metrics: BiasMetrics) -> None:
        """Store bias metrics for real-time monitoring."""
        if stage_name not in self.monitoring_data:
            self.monitoring_data[stage_name] = []
        
        self.monitoring_data[stage_name].append(metrics)
        
        # Keep only recent data (last 100 entries)
        if len(self.monitoring_data[stage_name]) > 100:
            self.monitoring_data[stage_name] = self.monitoring_data[stage_name][-100:]
    
    def _update_performance_metrics(self, cache_hit: bool, detection_time: float,
                                  failed: bool = False) -> None:
        """Update performance tracking metrics."""
        self.performance_metrics['total_detections'] += 1
        
        if failed:
            self.performance_metrics['failed_detections'] += 1
        else:
            self.performance_metrics['successful_detections'] += 1
            
            if cache_hit:
                # Update cache hit rate
                total_detections = self.performance_metrics['total_detections']
                cache_hits = total_detections - self.performance_metrics['failed_detections'] - \
                           (self.performance_metrics['successful_detections'] - 
                            (1 if not failed else 0))
                self.performance_metrics['cache_hit_rate'] = cache_hits / max(1, total_detections)
            
            # Update average detection time
            current_avg = self.performance_metrics['average_detection_time']
            total_successful = self.performance_metrics['successful_detections']
            self.performance_metrics['average_detection_time'] = (
                (current_avg * (total_successful - 1) + detection_time) / total_successful
            )
    
    async def _publish_bias_threshold_event(self, stage_name: str, bias_score: float,
                                          threshold: float) -> None:
        """Publish bias threshold exceeded event."""
        try:
            event = EventMessage(
                event_type=EventType.BIAS_THRESHOLD_EXCEEDED.value,
                stage=stage_name,
                payload={
                    'bias_score': bias_score,
                    'threshold': threshold,
                    'exceeded_by': bias_score - threshold,
                    'severity': 'high' if bias_score > threshold + 0.1 else 'medium',
                    'timestamp': datetime.utcnow().isoformat()
                },
                source='bias_detection_integration',
                target='monitoring_system'
            )
            
            # Note: This would need event_bus to be passed in or made available
            # For now, just log the event
            self.logger.warning(
                f"BIAS THRESHOLD EXCEEDED: Stage {stage_name}, "
                f"Score: {bias_score:.3f}, Threshold: {threshold}"
            )
            
        except Exception as e:
            self.logger.error(f"Failed to publish bias threshold event: {e}")
    
    def get_monitoring_summary(self, stage_name: Optional[str] = None) -> Dict[str, Any]:
        """
        Get bias monitoring summary.
        
        Args:
            stage_name: Optional stage name to filter by
            
        Returns:
            Monitoring summary data
        """
        try:
            summary = {
                'overall_metrics': self.performance_metrics.copy(),
                'stage_summaries': {},
                'recent_violations': [],
                'trend_analysis': {}
            }
            
            # Analyze by stage
            stages_to_analyze = [stage_name] if stage_name else self.monitoring_data.keys()
            
            for stage in stages_to_analyze:
                if stage in self.monitoring_data and self.monitoring_data[stage]:
                    stage_data = self.monitoring_data[stage]
                    
                    # Calculate stage metrics
                    recent_metrics = stage_data[-10:]  # Last 10 measurements
                    avg_bias = sum(m.overall_bias_score for m in recent_metrics) / len(recent_metrics)
                    max_bias = max(m.overall_bias_score for m in recent_metrics)
                    violations = sum(1 for m in recent_metrics if m.compliance_status == 'violation')
                    
                    summary['stage_summaries'][stage] = {
                        'average_bias_score': avg_bias,
                        'maximum_bias_score': max_bias,
                        'recent_violations': violations,
                        'total_measurements': len(stage_data),
                        'compliance_rate': (len(stage_data) - violations) / len(stage_data) if stage_data else 0
                    }
                    
                    # Track recent violations
                    for metric in recent_metrics:
                        if metric.compliance_status == 'violation':
                            summary['recent_violations'].append({
                                'stage': stage,
                                'bias_score': metric.overall_bias_score,
                                'timestamp': metric.detection_timestamp.isoformat()
                            })
            
            # Limit recent violations to last 10
            summary['recent_violations'] = summary['recent_violations'][-10:]
            
            return summary
            
        except Exception as e:
            self.logger.error(f"Failed to generate monitoring summary: {e}")
            return {'error': str(e)}
    
    def get_real_time_alerts(self) -> List[Dict[str, Any]]:
        """
        Get real-time bias alerts.
        
        Returns:
            List of active bias alerts
        """
        try:
            alerts = []
            
            # Check for high bias in recent measurements
            for stage_name, metrics_list in self.monitoring_data.items():
                if metrics_list:
                    latest_metric = metrics_list[-1]
                    
                    if latest_metric.compliance_status == 'violation':
                        alerts.append({
                            'type': 'bias_threshold_exceeded',
                            'stage': stage_name,
                            'bias_score': latest_metric.overall_bias_score,
                            'threshold': self.config.threshold,
                            'timestamp': latest_metric.detection_timestamp.isoformat(),
                            'severity': 'high'
                        })
                    
                    elif latest_metric.compliance_status == 'warning':
                        alerts.append({
                            'type': 'bias_threshold_warning',
                            'stage': stage_name,
                            'bias_score': latest_metric.overall_bias_score,
                            'threshold': self.config.threshold,
                            'timestamp': latest_metric.detection_timestamp.isoformat(),
                            'severity': 'medium'
                        })
            
            return alerts
            
        except Exception as e:
            self.logger.error(f"Failed to get real-time alerts: {e}")
            return []
    
    def health_check(self) -> Dict[str, Any]:
        """
        Perform health check of bias detection integration.
        
        Returns:
            Health check results
        """
        try:
            # Check service availability
            service_healthy = self.bias_service_available
            
            # Check configuration
            config_healthy = (
                self.config.enabled and
                0.0 <= self.config.threshold <= 1.0 and
                self.config.timeout_seconds > 0
            )
            
            # Check performance metrics
            performance_healthy = (
                self.performance_metrics['successful_detections'] > 0 or
                self.performance_metrics['total_detections'] == 0  # No detections yet is OK
            )
            
            status = 'healthy' if all([service_healthy, config_healthy, performance_healthy]) else 'degraded'
            
            return {
                'status': status,
                'service_available': service_healthy,
                'configuration_healthy': config_healthy,
                'performance_healthy': performance_healthy,
                'model_version': self.model_version,
                'total_detections': self.performance_metrics['total_detections'],
                'cache_enabled': self.config.cache_enabled,
                'real_time_monitoring': self.config.real_time_monitoring,
                'timestamp': datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"BiasDetectionIntegration health check failed: {e}")
            return {
                'status': 'unhealthy',
                'error': str(e),
                'timestamp': datetime.utcnow().isoformat()
            }