"""
Analytics API routes for TechDeck-Python Pipeline Integration.

This module implements REST API endpoints for analytics operations,
including usage metrics, performance tracking, and data insights.
"""

import logging
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List
from flask import Blueprint, request, jsonify, g
from werkzeug.exceptions import BadRequest, NotFound

from ..utils.validation import validate_analytics_request, sanitize_input
from ..utils.logger import get_request_logger
from ..error_handling.custom_errors import (
    ValidationError, ResourceNotFoundError, AnalyticsError
)
from ..integration.redis_client import RedisClient
from ..auth.decorators import require_auth, require_role

# Initialize blueprint
analytics_bp = Blueprint('analytics', __name__)
logger = logging.getLogger(__name__)


@analytics_bp.route('/usage', methods=['GET'])
@require_auth
def get_usage_analytics():
    """
    Get usage analytics and metrics.
    
    Query Parameters:
        start_date (str): Start date for analytics period (ISO format)
        end_date (str): End date for analytics period (ISO format)
        metric_type (str): Type of metrics (api, pipeline, storage)
        granularity (str): Time granularity (hour, day, week)
        
    Returns:
        Usage analytics data with trends and statistics
        
    Raises:
        ValidationError: If query parameters are invalid
        AnalyticsError: If analytics retrieval fails
    """
    request_logger = get_request_logger(g.request_id)
    request_logger.info("Retrieving usage analytics")
    
    try:
        # Get query parameters
        start_date_str = request.args.get('start_date')
        end_date_str = request.args.get('end_date')
        metric_type = request.args.get('metric_type', 'api')
        granularity = request.args.get('granularity', 'day')
        
        # Parse and validate dates
        if start_date_str:
            start_date = datetime.fromisoformat(start_date_str.replace('Z', '+00:00'))
        else:
            start_date = datetime.utcnow() - timedelta(days=30)
        
        if end_date_str:
            end_date = datetime.fromisoformat(end_date_str.replace('Z', '+00:00'))
        else:
            end_date = datetime.utcnow()
        
        if start_date >= end_date:
            raise ValidationError("Start date must be before end date")
        
        # Sanitize inputs
        metric_type = sanitize_input(metric_type)
        granularity = sanitize_input(granularity)
        
        request_logger.info(f"Retrieving {metric_type} usage analytics from {start_date} to {end_date}")
        
        # Get Redis client from app context
        redis_client = g.app.redis_client
        
        # Retrieve usage analytics
        analytics_data = _get_usage_metrics(redis_client, start_date, end_date, metric_type, granularity)
        
        # Log successful retrieval
        request_logger.info(f"Retrieved usage analytics with {len(analytics_data.get('metrics', []))} data points")
        
        return jsonify({
            'success': True,
            'data': {
                'metrics': analytics_data.get('metrics', []),
                'summary': analytics_data.get('summary', {}),
                'trends': analytics_data.get('trends', {}),
                'period': {
                    'start': start_date.isoformat(),
                    'end': end_date.isoformat()
                },
                'timestamp': datetime.utcnow().isoformat()
            }
        }), 200
        
    except ValueError as e:
        raise ValidationError(f"Invalid date format: {str(e)}")
    except ValidationError:
        raise
    except Exception as e:
        request_logger.error(f"Error retrieving usage analytics: {e}")
        raise AnalyticsError(f"Failed to retrieve usage analytics: {str(e)}")


@analytics_bp.route('/performance', methods=['GET'])
@require_auth
def get_performance_analytics():
    """
    Get performance analytics and metrics.
    
    Query Parameters:
        start_date (str): Start date for analytics period
        end_date (str): End date for analytics period
        metric_type (str): Type of performance metrics (response_time, throughput, error_rate)
        service (str): Specific service to analyze (api, pipeline, database)
        
    Returns:
        Performance analytics data with benchmarks and trends
        
    Raises:
        ValidationError: If query parameters are invalid
        AnalyticsError: If analytics retrieval fails
    """
    request_logger = get_request_logger(g.request_id)
    request_logger.info("Retrieving performance analytics")
    
    try:
        # Get query parameters
        start_date_str = request.args.get('start_date')
        end_date_str = request.args.get('end_date')
        metric_type = request.args.get('metric_type', 'response_time')
        service = request.args.get('service', 'api')
        
        # Parse and validate dates
        if start_date_str:
            start_date = datetime.fromisoformat(start_date_str.replace('Z', '+00:00'))
        else:
            start_date = datetime.utcnow() - timedelta(days=7)
        
        if end_date_str:
            end_date = datetime.fromisoformat(end_date_str.replace('Z', '+00:00'))
        else:
            end_date = datetime.utcnow()
        
        if start_date >= end_date:
            raise ValidationError("Start date must be before end date")
        
        # Sanitize inputs
        metric_type = sanitize_input(metric_type)
        service = sanitize_input(service)
        
        request_logger.info(f"Retrieving {service} {metric_type} performance analytics")
        
        # Get Redis client from app context
        redis_client = g.app.redis_client
        
        # Retrieve performance analytics
        performance_data = _get_performance_metrics(redis_client, start_date, end_date, metric_type, service)
        
        # Log successful retrieval
        request_logger.info(f"Retrieved performance analytics with {len(performance_data.get('metrics', []))} data points")
        
        return jsonify({
            'success': True,
            'data': {
                'metrics': performance_data.get('metrics', []),
                'benchmarks': performance_data.get('benchmarks', {}),
                'percentiles': performance_data.get('percentiles', {}),
                'service': service,
                'metric_type': metric_type,
                'timestamp': datetime.utcnow().isoformat()
            }
        }), 200
        
    except ValueError as e:
        raise ValidationError(f"Invalid date format: {str(e)}")
    except ValidationError:
        raise
    except Exception as e:
        request_logger.error(f"Error retrieving performance analytics: {e}")
        raise AnalyticsError(f"Failed to retrieve performance analytics: {str(e)}")


@analytics_bp.route('/pipeline', methods=['GET'])
@require_auth
def get_pipeline_analytics():
    """
    Get pipeline execution analytics and statistics.
    
    Query Parameters:
        pipeline_id (str): Optional pipeline ID to filter by
        status (str): Optional status filter (running, completed, failed)
        start_date (str): Start date for analytics period
        end_date (str): End date for analytics period
        
    Returns:
        Pipeline execution analytics with success rates and timing
        
    Raises:
        ValidationError: If query parameters are invalid
        AnalyticsError: If analytics retrieval fails
    """
    request_logger = get_request_logger(g.request_id)
    request_logger.info("Retrieving pipeline analytics")
    
    try:
        # Get query parameters
        pipeline_id = request.args.get('pipeline_id')
        status = request.args.get('status')
        start_date_str = request.args.get('start_date')
        end_date_str = request.args.get('end_date')
        
        # Parse and validate dates
        if start_date_str:
            start_date = datetime.fromisoformat(start_date_str.replace('Z', '+00:00'))
        else:
            start_date = datetime.utcnow() - timedelta(days=30)
        
        if end_date_str:
            end_date = datetime.fromisoformat(end_date_str.replace('Z', '+00:00'))
        else:
            end_date = datetime.utcnow()
        
        if start_date >= end_date:
            raise ValidationError("Start date must be before end date")
        
        # Sanitize inputs
        pipeline_id = sanitize_input(pipeline_id) if pipeline_id else None
        status = sanitize_input(status) if status else None
        
        request_logger.info(f"Retrieving pipeline analytics for pipeline: {pipeline_id}")
        
        # Get Redis client from app context
        redis_client = g.app.redis_client
        
        # Retrieve pipeline analytics
        pipeline_data = _get_pipeline_metrics(redis_client, pipeline_id, status, start_date, end_date)
        
        # Log successful retrieval
        request_logger.info(f"Retrieved pipeline analytics with {len(pipeline_data.get('executions', []))} executions")
        
        return jsonify({
            'success': True,
            'data': {
                'executions': pipeline_data.get('executions', []),
                'statistics': pipeline_data.get('statistics', {}),
                'success_rate': pipeline_data.get('success_rate', 0),
                'average_duration': pipeline_data.get('average_duration', 0),
                'pipeline_id': pipeline_id,
                'timestamp': datetime.utcnow().isoformat()
            }
        }), 200
        
    except ValueError as e:
        raise ValidationError(f"Invalid date format: {str(e)}")
    except ValidationError:
        raise
    except Exception as e:
        request_logger.error(f"Error retrieving pipeline analytics: {e}")
        raise AnalyticsError(f"Failed to retrieve pipeline analytics: {str(e)}")


@analytics_bp.route('/datasets', methods=['GET'])
@require_auth
def get_dataset_analytics():
    """
    Get dataset usage and quality analytics.
    
    Query Parameters:
        dataset_id (str): Optional dataset ID to filter by
        metric_type (str): Type of dataset metrics (usage, quality, storage)
        start_date (str): Start date for analytics period
        end_date (str): End date for analytics period
        
    Returns:
        Dataset analytics with usage patterns and quality metrics
        
    Raises:
        ValidationError: If query parameters are invalid
        AnalyticsError: If analytics retrieval fails
    """
    request_logger = get_request_logger(g.request_id)
    request_logger.info("Retrieving dataset analytics")
    
    try:
        # Get query parameters
        dataset_id = request.args.get('dataset_id')
        metric_type = request.args.get('metric_type', 'usage')
        start_date_str = request.args.get('start_date')
        end_date_str = request.args.get('end_date')
        
        # Parse and validate dates
        if start_date_str:
            start_date = datetime.fromisoformat(start_date_str.replace('Z', '+00:00'))
        else:
            start_date = datetime.utcnow() - timedelta(days=30)
        
        if end_date_str:
            end_date = datetime.fromisoformat(end_date_str.replace('Z', '+00:00'))
        else:
            end_date = datetime.utcnow()
        
        if start_date >= end_date:
            raise ValidationError("Start date must be before end date")
        
        # Sanitize inputs
        dataset_id = sanitize_input(dataset_id) if dataset_id else None
        metric_type = sanitize_input(metric_type)
        
        request_logger.info(f"Retrieving {metric_type} analytics for dataset: {dataset_id}")
        
        # Get Redis client from app context
        redis_client = g.app.redis_client
        
        # Retrieve dataset analytics
        dataset_data = _get_dataset_metrics(redis_client, dataset_id, metric_type, start_date, end_date)
        
        # Log successful retrieval
        request_logger.info(f"Retrieved dataset analytics with {len(dataset_data.get('datasets', []))} datasets")
        
        return jsonify({
            'success': True,
            'data': {
                'datasets': dataset_data.get('datasets', []),
                'summary': dataset_data.get('summary', {}),
                'trends': dataset_data.get('trends', {}),
                'metric_type': metric_type,
                'timestamp': datetime.utcnow().isoformat()
            }
        }), 200
        
    except ValueError as e:
        raise ValidationError(f"Invalid date format: {str(e)}")
    except ValidationError:
        raise
    except Exception as e:
        request_logger.error(f"Error retrieving dataset analytics: {e}")
        raise AnalyticsError(f"Failed to retrieve dataset analytics: {str(e)}")


@analytics_bp.route('/real-time', methods=['GET'])
@require_auth
def get_real_time_analytics():
    """
    Get real-time analytics and current system metrics.
    
    Query Parameters:
        metric_types (str): Comma-separated list of metric types to include
        refresh_rate (int): Refresh rate in seconds (default: 60)
        
    Returns:
        Real-time system metrics and current status
        
    Raises:
        ValidationError: If query parameters are invalid
        AnalyticsError: If analytics retrieval fails
    """
    request_logger = get_request_logger(g.request_id)
    request_logger.info("Retrieving real-time analytics")
    
    try:
        # Get query parameters
        metric_types_str = request.args.get('metric_types', 'system,pipeline,api')
        refresh_rate = int(request.args.get('refresh_rate', 60))
        
        # Validate refresh rate
        if refresh_rate < 10 or refresh_rate > 3600:
            raise ValidationError("Refresh rate must be between 10 and 3600 seconds")
        
        # Parse metric types
        metric_types = [mt.strip() for mt in metric_types_str.split(',')]
        valid_types = {'system', 'pipeline', 'api', 'database', 'storage'}
        
        for mt in metric_types:
            if mt not in valid_types:
                raise ValidationError(f"Invalid metric type: {mt}")
        
        request_logger.info(f"Retrieving real-time analytics for types: {metric_types}")
        
        # Get Redis client from app context
        redis_client = g.app.redis_client
        
        # Retrieve real-time metrics
        real_time_data = _get_real_time_metrics(redis_client, metric_types)
        
        # Log successful retrieval
        request_logger.info(f"Retrieved real-time analytics for {len(metric_types)} metric types")
        
        return jsonify({
            'success': True,
            'data': {
                'metrics': real_time_data.get('metrics', {}),
                'status': real_time_data.get('status', {}),
                'alerts': real_time_data.get('alerts', []),
                'refresh_rate': refresh_rate,
                'timestamp': datetime.utcnow().isoformat()
            }
        }), 200
        
    except ValueError as e:
        raise ValidationError(f"Invalid parameter: {str(e)}")
    except ValidationError:
        raise
    except Exception as e:
        request_logger.error(f"Error retrieving real-time analytics: {e}")
        raise AnalyticsError(f"Failed to retrieve real-time analytics: {str(e)}")


@analytics_bp.route('/export', methods=['POST'])
@require_auth
def export_analytics():
    """
    Export analytics data in various formats.
    
    Request Body:
        {
            "export_type": "string",    # Type of export (csv, json, pdf)
            "metric_type": "string",    # Type of metrics to export
            "start_date": "string",     # Start date for export period
            "end_date": "string",       # End date for export period
            "filters": {...},           # Optional filters
            "options": {...}            # Export-specific options
        }
        
    Returns:
        Export task information and download link
        
    Raises:
        ValidationError: If request data is invalid
        AnalyticsError: If export fails
    """
    request_logger = get_request_logger(g.request_id)
    request_logger.info("Creating analytics export")
    
    try:
        # Get request data
        request_data = request.get_json()
        if not request_data:
            raise ValidationError("Request body is required")
        
        # Validate request structure
        validation_result = validate_analytics_request(request_data, 'export')
        if not validation_result['valid']:
            raise ValidationError(validation_result['errors'])
        
        # Extract parameters
        export_type = request_data.get('export_type', 'json')
        metric_type = request_data.get('metric_type', 'usage')
        start_date_str = request_data.get('start_date')
        end_date_str = request_data.get('end_date')
        filters = request_data.get('filters', {})
        options = request_data.get('options', {})
        
        # Parse dates
        if start_date_str:
            start_date = datetime.fromisoformat(start_date_str.replace('Z', '+00:00'))
        else:
            start_date = datetime.utcnow() - timedelta(days=30)
        
        if end_date_str:
            end_date = datetime.fromisoformat(end_date_str.replace('Z', '+00:00'))
        else:
            end_date = datetime.utcnow()
        
        # Sanitize inputs
        export_type = sanitize_input(export_type)
        metric_type = sanitize_input(metric_type)
        
        request_logger.info(f"Creating {export_type} export for {metric_type} analytics")
        
        # Get Redis client from app context
        redis_client = g.app.redis_client
        
        # Create export task
        export_task = _create_export_task(redis_client, export_type, metric_type, start_date, end_date, filters, options)
        
        # Log export creation
        request_logger.info(f"Created export task: {export_task.get('task_id')}")
        
        return jsonify({
            'success': True,
            'data': {
                'task_id': export_task.get('task_id'),
                'status': export_task.get('status'),
                'export_type': export_type,
                'estimated_size': export_task.get('estimated_size'),
                'download_url': export_task.get('download_url'),
                'created_at': datetime.utcnow().isoformat()
            }
        }), 201
        
    except (ValidationError, ValueError) as e:
        raise ValidationError(f"Invalid request: {str(e)}")
    except ValidationError:
        raise
    except Exception as e:
        request_logger.error(f"Error creating analytics export: {e}")
        raise AnalyticsError(f"Failed to create analytics export: {str(e)}")


@analytics_bp.route('/export/<task_id>/status', methods=['GET'])
@require_auth
def get_export_status(task_id: str):
    """
    Get the status of an analytics export task.
    
    Args:
        task_id: Unique identifier for the export task
        
    Returns:
        Export task status and progress information
        
    Raises:
        ResourceNotFoundError: If task is not found
        ValidationError: If task_id is invalid
    """
    request_logger = get_request_logger(g.request_id)
    request_logger.info(f"Retrieving export status: {task_id}")
    
    try:
        # Validate task ID
        if not task_id or not isinstance(task_id, str):
            raise ValidationError("Invalid task ID format")
        
        # Sanitize input
        task_id = sanitize_input(task_id)
        
        # Get Redis client from app context
        redis_client = g.app.redis_client
        
        # Retrieve export task status
        task_status = _get_export_task_status(redis_client, task_id)
        
        if not task_status:
            raise ResourceNotFoundError(f"Export task '{task_id}' not found")
        
        # Log successful retrieval
        request_logger.info(f"Retrieved export status for task: {task_id}")
        
        return jsonify({
            'success': True,
            'data': task_status,
            'timestamp': datetime.utcnow().isoformat()
        }), 200
        
    except (ValidationError, ResourceNotFoundError):
        raise
    except Exception as e:
        request_logger.error(f"Error retrieving export status: {e}")
        raise AnalyticsError(f"Failed to retrieve export status: {str(e)}")


# Helper Functions
def _get_usage_metrics(redis_client: RedisClient, start_date: datetime, end_date: datetime, metric_type: str, granularity: str) -> Dict[str, Any]:
    """Retrieve usage metrics from Redis or database."""
    try:
        # Generate placeholder usage metrics
        metrics = []
        current_date = start_date
        
        while current_date <= end_date:
            # Generate sample data based on metric type
            if metric_type == 'api':
                value = 150 + (current_date.day % 50)  # Simulate daily variation
            elif metric_type == 'pipeline':
                value = 25 + (current_date.day % 15)
            elif metric_type == 'storage':
                value = 1024 + (current_date.day * 10)  # MB
            else:
                value = 100
            
            metrics.append({
                'timestamp': current_date.isoformat(),
                'value': value,
                'metric_type': metric_type
            })
            
            # Increment based on granularity
            if granularity == 'hour':
                current_date += timedelta(hours=1)
            elif granularity == 'day':
                current_date += timedelta(days=1)
            elif granularity == 'week':
                current_date += timedelta(weeks=1)
            else:
                current_date += timedelta(days=1)
        
        # Calculate summary statistics
        values = [m['value'] for m in metrics]
        summary = {
            'total': sum(values),
            'average': sum(values) / len(values) if values else 0,
            'max': max(values) if values else 0,
            'min': min(values) if values else 0,
            'count': len(values)
        }
        
        # Calculate trends
        if len(values) > 1:
            trend = 'increasing' if values[-1] > values[0] else 'decreasing'
            trend_percentage = ((values[-1] - values[0]) / values[0] * 100) if values[0] != 0 else 0
        else:
            trend = 'stable'
            trend_percentage = 0
        
        trends = {
            'direction': trend,
            'percentage_change': trend_percentage
        }
        
        return {
            'metrics': metrics,
            'summary': summary,
            'trends': trends
        }
        
    except Exception as e:
        logger.error(f"Error retrieving usage metrics: {e}")
        raise AnalyticsError(f"Failed to retrieve usage metrics: {str(e)}")


def _get_performance_metrics(redis_client: RedisClient, start_date: datetime, end_date: datetime, metric_type: str, service: str) -> Dict[str, Any]:
    """Retrieve performance metrics from Redis or database."""
    try:
        # Generate placeholder performance metrics
        metrics = []
        current_date = start_date
        
        while current_date <= end_date:
            # Generate sample performance data
            if metric_type == 'response_time':
                base_time = 45 if service == 'api' else 120  # ms
                value = base_time + (current_date.hour % 20)  # Simulate hourly variation
            elif metric_type == 'throughput':
                value = 100 + (current_date.hour * 5)  # requests per second
            elif metric_type == 'error_rate':
                value = 0.5 + (current_date.hour % 3) * 0.1  # percentage
            else:
                value = 50
            
            metrics.append({
                'timestamp': current_date.isoformat(),
                'value': value,
                'metric_type': metric_type,
                'service': service
            })
            
            current_date += timedelta(hours=1)
        
        # Calculate benchmarks and percentiles
        values = [m['value'] for m in metrics]
        
        benchmarks = {
            'target': 50 if metric_type == 'response_time' else 100,
            'acceptable': 100 if metric_type == 'response_time' else 80,
            'critical': 200 if metric_type == 'response_time' else 50
        }
        
        # Calculate percentiles
        if values:
            sorted_values = sorted(values)
            n = len(sorted_values)
            percentiles = {
                'p50': sorted_values[n // 2],
                'p90': sorted_values[int(n * 0.9)],
                'p95': sorted_values[int(n * 0.95)],
                'p99': sorted_values[int(n * 0.99)]
            }
        else:
            percentiles = {'p50': 0, 'p90': 0, 'p95': 0, 'p99': 0}
        
        return {
            'metrics': metrics,
            'benchmarks': benchmarks,
            'percentiles': percentiles
        }
        
    except Exception as e:
        logger.error(f"Error retrieving performance metrics: {e}")
        raise AnalyticsError(f"Failed to retrieve performance metrics: {str(e)}")


def _get_pipeline_metrics(redis_client: RedisClient, pipeline_id: Optional[str], status: Optional[str], start_date: datetime, end_date: datetime) -> Dict[str, Any]:
    """Retrieve pipeline execution metrics."""
    try:
        # Generate placeholder pipeline metrics
        executions = []
        
        # Generate sample executions
        for i in range(20):
            execution_date = start_date + timedelta(days=i)
            if execution_date > end_date:
                break
            
            execution_status = status if status else ('completed' if i % 5 != 0 else 'failed')
            
            executions.append({
                'execution_id': f'exec_{i+1:03d}',
                'pipeline_id': pipeline_id or f'pipeline_{(i % 3) + 1}',
                'status': execution_status,
                'start_time': execution_date.isoformat(),
                'duration_seconds': 300 + (i * 10),  # 5-7 minutes
                'stage_progress': {
                    'data_ingestion': 'completed',
                    'preprocessing': 'completed',
                    'transformation': 'completed' if execution_status == 'completed' else 'failed',
                    'validation': 'pending' if execution_status == 'running' else 'completed'
                }
            })
        
        # Filter by status if specified
        if status:
            executions = [e for e in executions if e['status'] == status]
        
        # Calculate statistics
        total_executions = len(executions)
        completed_executions = len([e for e in executions if e['status'] == 'completed'])
        failed_executions = len([e for e in executions if e['status'] == 'failed'])
        
        success_rate = (completed_executions / total_executions * 100) if total_executions > 0 else 0
        
        durations = [e['duration_seconds'] for e in executions if e['status'] == 'completed']
        average_duration = sum(durations) / len(durations) if durations else 0
        
        statistics = {
            'total_executions': total_executions,
            'completed_executions': completed_executions,
            'failed_executions': failed_executions,
            'success_rate': success_rate,
            'average_duration_seconds': average_duration
        }
        
        return {
            'executions': executions,
            'statistics': statistics,
            'success_rate': success_rate,
            'average_duration': average_duration
        }
        
    except Exception as e:
        logger.error(f"Error retrieving pipeline metrics: {e}")
        raise AnalyticsError(f"Failed to retrieve pipeline metrics: {str(e)}")


def _get_dataset_metrics(redis_client: RedisClient, dataset_id: Optional[str], metric_type: str, start_date: datetime, end_date: datetime) -> Dict[str, Any]:
    """Retrieve dataset usage and quality metrics."""
    try:
        # Generate placeholder dataset metrics
        datasets = []
        
        # Sample datasets
        sample_datasets = [
            {'id': 'dataset_001', 'name': 'Healthcare Patients', 'size_mb': 512, 'record_count': 10000},
            {'id': 'dataset_002', 'name': 'Clinical Trials', 'size_mb': 256, 'record_count': 5000},
            {'id': 'dataset_003', 'name': 'Mental Health Sessions', 'size_mb': 1024, 'record_count': 25000}
        ]
        
        for ds in sample_datasets:
            if dataset_id and ds['id'] != dataset_id:
                continue
            
            if metric_type == 'usage':
                metrics = {
                    'access_count': 150 + (hash(ds['id']) % 100),
                    'last_accessed': (start_date + timedelta(days=hash(ds['id']) % 30)).isoformat(),
                    'unique_users': 25 + (hash(ds['id']) % 20)
                }
            elif metric_type == 'quality':
                metrics = {
                    'completeness_score': 0.95,
                    'accuracy_score': 0.88,
                    'consistency_score': 0.92,
                    'overall_quality': 0.91
                }
            elif metric_type == 'storage':
                metrics = {
                    'size_mb': ds['size_mb'],
                    'record_count': ds['record_count'],
                    'growth_rate_mb_per_day': 5 + (hash(ds['id']) % 10),
                    'compression_ratio': 0.7
                }
            else:
                metrics = {}
            
            datasets.append({
                'dataset_id': ds['id'],
                'name': ds['name'],
                'metric_type': metric_type,
                'metrics': metrics,
                'last_updated': datetime.utcnow().isoformat()
            })
        
        # Calculate summary
        if datasets:
            if metric_type == 'usage':
                total_accesses = sum(d['metrics']['access_count'] for d in datasets)
                summary = {'total_accesses': total_accesses, 'active_datasets': len(datasets)}
            elif metric_type == 'quality':
                avg_quality = sum(d['metrics']['overall_quality'] for d in datasets) / len(datasets)
                summary = {'average_quality_score': avg_quality, 'datasets_analyzed': len(datasets)}
            elif metric_type == 'storage':
                total_size = sum(d['metrics']['size_mb'] for d in datasets)
                summary = {'total_size_mb': total_size, 'total_records': sum(d['metrics']['record_count'] for d in datasets)}
            else:
                summary = {}
        else:
            summary = {}
        
        # Calculate trends (placeholder)
        trends = {
            'usage_trend': 'increasing',
            'quality_trend': 'stable',
            'storage_trend': 'increasing'
        }
        
        return {
            'datasets': datasets,
            'summary': summary,
            'trends': trends
        }
        
    except Exception as e:
        logger.error(f"Error retrieving dataset metrics: {e}")
        raise AnalyticsError(f"Failed to retrieve dataset metrics: {str(e)}")


def _get_real_time_metrics(redis_client: RedisClient, metric_types: List[str]) -> Dict[str, Any]:
    """Retrieve real-time system metrics."""
    try:
        metrics = {}
        status = {}
        alerts = []
        
        # System metrics
        if 'system' in metric_types:
            metrics['system'] = {
                'cpu_usage_percent': 45.2,
                'memory_usage_percent': 67.8,
                'disk_usage_percent': 72.1,
                'network_io_mb_per_sec': 12.5
            }
            status['system'] = 'healthy' if metrics['system']['cpu_usage_percent'] < 80 else 'warning'
        
        # API metrics
        if 'api' in metric_types:
            metrics['api'] = {
                'requests_per_second': 25,
                'active_connections': 150,
                'error_rate_percent': 0.5,
                'average_response_time_ms': 45
            }
            status['api'] = 'healthy' if metrics['api']['error_rate_percent'] < 1.0 else 'warning'
        
        # Pipeline metrics
        if 'pipeline' in metric_types:
            metrics['pipeline'] = {
                'active_executions': 3,
                'queued_executions': 5,
                'completed_today': 47,
                'failed_today': 2
            }
            status['pipeline'] = 'healthy' if metrics['pipeline']['failed_today'] < 5 else 'warning'
        
        # Database metrics
        if 'database' in metric_types:
            metrics['database'] = {
                'connections': 25,
                'queries_per_second': 150,
                'average_query_time_ms': 12,
                'cache_hit_ratio': 0.95
            }
            status['database'] = 'healthy' if metrics['database']['cache_hit_ratio'] > 0.9 else 'warning'
        
        # Storage metrics
        if 'storage' in metric_types:
            metrics['storage'] = {
                'total_space_gb': 1000,
                'used_space_gb': 650,
                'free_space_gb': 350,
                'iops': 2500
            }
            usage_percent = (metrics['storage']['used_space_gb'] / metrics['storage']['total_space_gb']) * 100
            status['storage'] = 'healthy' if usage_percent < 85 else 'warning'
        
        # Generate alerts based on status
        for component, component_status in status.items():
            if component_status != 'healthy':
                alerts.append({
                    'component': component,
                    'status': component_status,
                    'message': f"{component.capitalize()} is experiencing {component_status} conditions",
                    'severity': 'warning' if component_status == 'warning' else 'critical'
                })
        
        return {
            'metrics': metrics,
            'status': status,
            'alerts': alerts
        }
        
    except Exception as e:
        logger.error(f"Error retrieving real-time metrics: {e}")
        raise AnalyticsError(f"Failed to retrieve real-time metrics: {str(e)}")


def _create_export_task(redis_client: RedisClient, export_type: str, metric_type: str, start_date: datetime, end_date: datetime, filters: Dict[str, Any], options: Dict[str, Any]) -> Dict[str, Any]:
    """Create an analytics export task."""
    try:
        # Generate task ID
        task_id = f"export_{datetime.utcnow().timestamp()}_{hash(str(filters)) % 10000}"
        
        # Create export task
        export_task = {
            'task_id': task_id,
            'status': 'queued',
            'export_type': export_type,
            'metric_type': metric_type,
            'start_date': start_date.isoformat(),
            'end_date': end_date.isoformat(),
            'filters': filters,
            'options': options,
            'progress': 0,
            'estimated_size': '2.5 MB',  # Placeholder
            'created_at': datetime.utcnow().isoformat(),
            'download_url': f"/api/v1/analytics/export/{task_id}/download"
        }
        
        # Store task in Redis
        redis_client.setex(f'export:task:{task_id}', 86400, export_task)  # 24 hour expiry
        
        # Simulate async processing (in real implementation, this would queue a background job)
        export_task['status'] = 'processing'
        export_task['progress'] = 25
        redis_client.setex(f'export:task:{task_id}', 86400, export_task)
        
        return export_task
        
    except Exception as e:
        logger.error(f"Error creating export task: {e}")
        raise AnalyticsError(f"Failed to create export task: {str(e)}")


def _get_export_task_status(redis_client: RedisClient, task_id: str) -> Optional[Dict[str, Any]]:
    """Retrieve export task status from Redis."""
    try:
        return redis_client.get(f'export:task:{task_id}')
    except Exception as e:
        logger.error(f"Error retrieving export task status: {e}")
        return None


def _get_dataset_context(redis_client: RedisClient, dataset_id: str) -> Optional[Dict[str, Any]]:
    """Retrieve dataset context information."""
    try:
        # Placeholder dataset context
        contexts = {
            'dataset_123': {
                'expected_fields': ['patient_id', 'age', 'gender', 'diagnosis'],
                'data_types': {'patient_id': 'string', 'age': 'integer', 'gender': 'string', 'diagnosis': 'string'},
                'required_fields': ['patient_id', 'age'],
                'quality_thresholds': {'completeness': 0.9, 'accuracy': 0.85}
            }
        }
        
        return contexts.get(dataset_id)
        
    except Exception as e:
        logger.error(f"Error retrieving dataset context: {e}")
        return None