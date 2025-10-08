"""
Pipeline Orchestration API Routes.

This module provides REST API endpoints for pipeline management,
execution control, and progress tracking integration.
"""

import uuid
from datetime import datetime
from typing import Dict, Any, Optional, List
from flask import Blueprint, request, jsonify, current_app

from ..auth.middleware import require_auth, require_role
from ..error_handling.custom_errors import (
    ValidationError,
    PipelineNotFoundError,
    PipelineExecutionError,
    AuthenticationError,
    RateLimitExceededError
)
from ..utils.logger import get_logger
from ..utils.validation import validate_pipeline_config, validate_stage_config
from ..services.pipeline_service import PipelineService
from ..integration.redis_client import RedisClient


logger = get_logger(__name__)
pipeline_bp = Blueprint('pipeline', __name__, url_prefix='/api/v1/pipeline')


@pipeline_bp.route('/configs', methods=['GET'])
@require_auth
def list_pipeline_configs() -> Dict[str, Any]:
    """
    List available pipeline configurations.
    
    Query Parameters:
        page (int): Page number (default: 1)
        limit (int): Items per page (default: 20, max: 100)
        search (str): Search term for pipeline name or description
        stage (str): Filter by stage type
        
    Returns:
        {
            "success": true,
            "data": {
                "configs": [...],
                "pagination": {
                    "page": 1,
                    "limit": 20,
                    "total": 50,
                    "pages": 3
                }
            }
        }
    """
    try:
        # Parse query parameters
        page = int(request.args.get('page', 1))
        limit = min(int(request.args.get('limit', 20)), 100)
        search = request.args.get('search', '')
        stage = request.args.get('stage')
        
        current_user = request.user
        
        logger.info(
            f"Listing pipeline configs for user {current_user['id']}",
            extra={
                'user_id': current_user['id'],
                'page': page,
                'limit': limit,
                'search': search,
                'stage': stage
            }
        )
        
        # Initialize pipeline service
        pipeline_service = PipelineService(current_app.config, current_app.redis_client)
        
        # Get pipeline configs
        result = pipeline_service.list_configs(
            user_id=current_user['id'],
            page=page,
            limit=limit,
            search=search,
            stage=stage
        )
        
        logger.info(
            f"Successfully retrieved {len(result['configs'])} pipeline configs",
            extra={'user_id': current_user['id'], 'total': result['pagination']['total']}
        )
        
        return jsonify({
            'success': True,
            'data': result,
            'meta': {
                'timestamp': datetime.utcnow().isoformat()
            }
        }), 200
        
    except ValueError as e:
        logger.warning(f"Invalid pagination parameters: {e}")
        raise ValidationError(f"Invalid pagination parameters: {e}")
    except Exception as e:
        logger.error(f"Error listing pipeline configs: {e}")
        raise


@pipeline_bp.route('/configs/<config_id>', methods=['GET'])
@require_auth
def get_pipeline_config(config_id: str) -> Dict[str, Any]:
    """
    Get detailed pipeline configuration.
    
    Args:
        config_id: UUID of the pipeline configuration
        
    Returns:
        {
            "success": true,
            "data": {
                "config": {
                    "id": "uuid",
                    "name": "Standard Pipeline",
                    "description": "Standard 6-stage pipeline",
                    "stages": [...],
                    "parameters": {...},
                    "created_at": "2024-01-01T00:00:00Z"
                }
            }
        }
    """
    try:
        current_user = request.user
        
        logger.info(
            f"Retrieving pipeline config {config_id}",
            extra={'user_id': current_user['id'], 'config_id': config_id}
        )
        
        # Initialize pipeline service
        pipeline_service = PipelineService(current_app.config, current_app.redis_client)
        
        # Get pipeline config
        config = pipeline_service.get_config(config_id, current_user['id'])
        
        if not config:
            logger.warning(
                f"Pipeline config {config_id} not found for user {current_user['id']}",
                extra={'user_id': current_user['id'], 'config_id': config_id}
            )
            raise PipelineNotFoundError(f"Pipeline config {config_id} not found")
        
        logger.info(
            f"Successfully retrieved pipeline config {config_id}",
            extra={'user_id': current_user['id'], 'config_id': config_id}
        )
        
        return jsonify({
            'success': True,
            'data': {'config': config},
            'meta': {
                'timestamp': datetime.utcnow().isoformat()
            }
        }), 200
        
    except PipelineNotFoundError:
        raise
    except Exception as e:
        logger.error(f"Error retrieving pipeline config {config_id}: {e}")
        raise


@pipeline_bp.route('/configs', methods=['POST'])
@require_auth
@require_role(['admin', 'developer'])
def create_pipeline_config() -> Dict[str, Any]:
    """
    Create a new pipeline configuration.
    
    Request Body:
        {
            "name": "Custom Pipeline",
            "description": "Custom pipeline configuration",
            "stages": [
                {
                    "name": "data_ingestion",
                    "type": "ingestion",
                    "config": {...}
                }
            ],
            "parameters": {
                "batch_size": 1000,
                "timeout": 300
            }
        }
        
    Returns:
        {
            "success": true,
            "data": {
                "config": {
                    "id": "uuid",
                    "name": "Custom Pipeline",
                    "description": "Custom pipeline configuration",
                    "stages": [...],
                    "created_at": "2024-01-01T00:00:00Z"
                }
            }
        }
    """
    try:
        current_user = request.user
        
        # Validate request body
        if not request.is_json:
            raise ValidationError("Content-Type must be application/json")
        
        data = request.get_json()
        
        # Validate required fields
        required_fields = ['name', 'stages']
        for field in required_fields:
            if field not in data or not data[field]:
                raise ValidationError(f"Missing required field: {field}")
        
        # Validate pipeline configuration
        validate_pipeline_config(data)
        
        logger.info(
            f"Creating pipeline config for user {current_user['id']}",
            extra={
                'user_id': current_user['id'],
                'config_name': data['name']
            }
        )
        
        # Initialize pipeline service
        pipeline_service = PipelineService(current_app.config, current_app.redis_client)
        
        # Create pipeline config
        config = pipeline_service.create_config(
            user_id=current_user['id'],
            name=data['name'],
            description=data.get('description', ''),
            stages=data['stages'],
            parameters=data.get('parameters', {})
        )
        
        logger.info(
            f"Successfully created pipeline config {config['id']}",
            extra={'user_id': current_user['id'], 'config_id': config['id']}
        )
        
        return jsonify({
            'success': True,
            'data': {'config': config},
            'meta': {
                'timestamp': datetime.utcnow().isoformat()
            }
        }), 201
        
    except ValidationError:
        raise
    except Exception as e:
        logger.error(f"Error creating pipeline config: {e}")
        raise


@pipeline_bp.route('/configs/<config_id>', methods=['PUT'])
@require_auth
@require_role(['admin', 'developer'])
def update_pipeline_config(config_id: str) -> Dict[str, Any]:
    """
    Update pipeline configuration.
    
    Args:
        config_id: UUID of the pipeline configuration
        
    Request Body:
        {
            "name": "Updated Pipeline",
            "description": "Updated description",
            "stages": [...],
            "parameters": {...}
        }
        
    Returns:
        {
            "success": true,
            "data": {
                "config": {
                    "id": "uuid",
                    "name": "Updated Pipeline",
                    "updated_at": "2024-01-01T00:00:00Z"
                }
            }
        }
    """
    try:
        current_user = request.user
        
        # Validate request body
        if not request.is_json:
            raise ValidationError("Content-Type must be application/json")
        
        data = request.get_json()
        
        logger.info(
            f"Updating pipeline config {config_id}",
            extra={
                'user_id': current_user['id'],
                'config_id': config_id,
                'update_data': data
            }
        )
        
        # Initialize pipeline service
        pipeline_service = PipelineService(current_app.config, current_app.redis_client)
        
        # Update pipeline config
        config = pipeline_service.update_config(
            config_id=config_id,
            user_id=current_user['id'],
            update_data=data
        )
        
        if not config:
            logger.warning(
                f"Pipeline config {config_id} not found for update",
                extra={'user_id': current_user['id'], 'config_id': config_id}
            )
            raise PipelineNotFoundError(f"Pipeline config {config_id} not found")
        
        logger.info(
            f"Successfully updated pipeline config {config_id}",
            extra={'user_id': current_user['id'], 'config_id': config_id}
        )
        
        return jsonify({
            'success': True,
            'data': {'config': config},
            'meta': {
                'timestamp': datetime.utcnow().isoformat()
            }
        }), 200
        
    except ValidationError:
        raise
    except PipelineNotFoundError:
        raise
    except Exception as e:
        logger.error(f"Error updating pipeline config {config_id}: {e}")
        raise


@pipeline_bp.route('/configs/<config_id>', methods=['DELETE'])
@require_auth
@require_role(['admin', 'developer'])
def delete_pipeline_config(config_id: str) -> Dict[str, Any]:
    """
    Delete pipeline configuration.
    
    Args:
        config_id: UUID of the pipeline configuration
        
    Returns:
        {
            "success": true,
            "data": {
                "message": "Pipeline config deleted successfully"
            }
        }
    """
    try:
        current_user = request.user
        
        logger.info(
            f"Deleting pipeline config {config_id}",
            extra={'user_id': current_user['id'], 'config_id': config_id}
        )
        
        # Initialize pipeline service
        pipeline_service = PipelineService(current_app.config, current_app.redis_client)
        
        # Delete pipeline config
        success = pipeline_service.delete_config(config_id, current_user['id'])
        
        if not success:
            logger.warning(
                f"Pipeline config {config_id} not found for deletion",
                extra={'user_id': current_user['id'], 'config_id': config_id}
            )
            raise PipelineNotFoundError(f"Pipeline config {config_id} not found")
        
        logger.info(
            f"Successfully deleted pipeline config {config_id}",
            extra={'user_id': current_user['id'], 'config_id': config_id}
        )
        
        return jsonify({
            'success': True,
            'data': {'message': 'Pipeline config deleted successfully'},
            'meta': {
                'timestamp': datetime.utcnow().isoformat()
            }
        }), 200
        
    except PipelineNotFoundError:
        raise
    except Exception as e:
        logger.error(f"Error deleting pipeline config {config_id}: {e}")
        raise


@pipeline_bp.route('/execute', methods=['POST'])
@require_auth
def execute_pipeline() -> Dict[str, Any]:
    """
    Execute a pipeline with specified configuration and dataset.
    
    Request Body:
        {
            "config_id": "uuid",
            "dataset_id": "uuid",
            "parameters": {
                "batch_size": 1000,
                "timeout": 300
            },
            "priority": "normal"
        }
        
    Returns:
        {
            "success": true,
            "data": {
                "execution": {
                    "id": "uuid",
                    "config_id": "uuid",
                    "dataset_id": "uuid",
                    "status": "queued",
                    "progress": 0,
                    "created_at": "2024-01-01T00:00:00Z"
                }
            }
        }
    """
    try:
        current_user = request.user
        
        # Validate request body
        if not request.is_json:
            raise ValidationError("Content-Type must be application/json")
        
        data = request.get_json()
        
        # Validate required fields
        required_fields = ['config_id', 'dataset_id']
        for field in required_fields:
            if field not in data or not data[field]:
                raise ValidationError(f"Missing required field: {field}")
        
        logger.info(
            f"Executing pipeline for user {current_user['id']}",
            extra={
                'user_id': current_user['id'],
                'config_id': data['config_id'],
                'dataset_id': data['dataset_id']
            }
        )
        
        # Initialize pipeline service
        pipeline_service = PipelineService(current_app.config, current_app.redis_client)
        
        # Execute pipeline
        execution = pipeline_service.execute_pipeline(
            user_id=current_user['id'],
            config_id=data['config_id'],
            dataset_id=data['dataset_id'],
            parameters=data.get('parameters', {}),
            priority=data.get('priority', 'normal')
        )
        
        logger.info(
            f"Successfully started pipeline execution {execution['id']}",
            extra={
                'user_id': current_user['id'],
                'execution_id': execution['id'],
                'config_id': data['config_id'],
                'dataset_id': data['dataset_id']
            }
        )
        
        return jsonify({
            'success': True,
            'data': {'execution': execution},
            'meta': {
                'timestamp': datetime.utcnow().isoformat()
            }
        }), 201
        
    except ValidationError:
        raise
    except Exception as e:
        logger.error(f"Error executing pipeline: {e}")
        raise


@pipeline_bp.route('/executions/<execution_id>', methods=['GET'])
@require_auth
def get_pipeline_execution(execution_id: str) -> Dict[str, Any]:
    """
    Get pipeline execution status and progress.
    
    Args:
        execution_id: UUID of the pipeline execution
        
    Returns:
        {
            "success": true,
            "data": {
                "execution": {
                    "id": "uuid",
                    "config_id": "uuid",
                    "dataset_id": "uuid",
                    "status": "running",
                    "progress": 45,
                    "current_stage": "data_cleaning",
                    "stages": [...],
                    "results": {...},
                    "error": null,
                    "started_at": "2024-01-01T00:00:00Z",
                    "updated_at": "2024-01-01T00:00:00Z"
                }
            }
        }
    """
    try:
        current_user = request.user
        
        logger.info(
            f"Retrieving pipeline execution {execution_id}",
            extra={'user_id': current_user['id'], 'execution_id': execution_id}
        )
        
        # Initialize pipeline service
        pipeline_service = PipelineService(current_app.config, current_app.redis_client)
        
        # Get pipeline execution
        execution = pipeline_service.get_execution(execution_id, current_user['id'])
        
        if not execution:
            logger.warning(
                f"Pipeline execution {execution_id} not found for user {current_user['id']}",
                extra={'user_id': current_user['id'], 'execution_id': execution_id}
            )
            raise PipelineNotFoundError(f"Pipeline execution {execution_id} not found")
        
        logger.info(
            f"Successfully retrieved pipeline execution {execution_id}",
            extra={
                'user_id': current_user['id'],
                'execution_id': execution_id,
                'status': execution['status'],
                'progress': execution['progress']
            }
        )
        
        return jsonify({
            'success': True,
            'data': {'execution': execution},
            'meta': {
                'timestamp': datetime.utcnow().isoformat()
            }
        }), 200
        
    except PipelineNotFoundError:
        raise
    except Exception as e:
        logger.error(f"Error retrieving pipeline execution {execution_id}: {e}")
        raise


@pipeline_bp.route('/executions/<execution_id>/cancel', methods=['POST'])
@require_auth
def cancel_pipeline_execution(execution_id: str) -> Dict[str, Any]:
    """
    Cancel a running pipeline execution.
    
    Args:
        execution_id: UUID of the pipeline execution
        
    Returns:
        {
            "success": true,
            "data": {
                "execution": {
                    "id": "uuid",
                    "status": "cancelled",
                    "cancelled_at": "2024-01-01T00:00:00Z"
                }
            }
        }
    """
    try:
        current_user = request.user
        
        logger.info(
            f"Cancelling pipeline execution {execution_id}",
            extra={'user_id': current_user['id'], 'execution_id': execution_id}
        )
        
        # Initialize pipeline service
        pipeline_service = PipelineService(current_app.config, current_app.redis_client)
        
        # Cancel pipeline execution
        execution = pipeline_service.cancel_execution(execution_id, current_user['id'])
        
        if not execution:
            logger.warning(
                f"Pipeline execution {execution_id} not found for cancellation",
                extra={'user_id': current_user['id'], 'execution_id': execution_id}
            )
            raise PipelineNotFoundError(f"Pipeline execution {execution_id} not found")
        
        logger.info(
            f"Successfully cancelled pipeline execution {execution_id}",
            extra={
                'user_id': current_user['id'],
                'execution_id': execution_id,
                'status': execution['status']
            }
        )
        
        return jsonify({
            'success': True,
            'data': {'execution': execution},
            'meta': {
                'timestamp': datetime.utcnow().isoformat()
            }
        }), 200
        
    except PipelineNotFoundError:
        raise
    except Exception as e:
        logger.error(f"Error cancelling pipeline execution {execution_id}: {e}")
        raise


@pipeline_bp.route('/executions', methods=['GET'])
@require_auth
def list_pipeline_executions() -> Dict[str, Any]:
    """
    List pipeline executions with filtering and pagination.
    
    Query Parameters:
        page (int): Page number (default: 1)
        limit (int): Items per page (default: 20, max: 100)
        status (str): Filter by execution status
        config_id (str): Filter by pipeline config ID
        dataset_id (str): Filter by dataset ID
        
    Returns:
        {
            "success": true,
            "data": {
                "executions": [...],
                "pagination": {
                    "page": 1,
                    "limit": 20,
                    "total": 100,
                    "pages": 5
                }
            }
        }
    """
    try:
        # Parse query parameters
        page = int(request.args.get('page', 1))
        limit = min(int(request.args.get('limit', 20)), 100)
        status = request.args.get('status')
        config_id = request.args.get('config_id')
        dataset_id = request.args.get('dataset_id')
        
        current_user = request.user
        
        logger.info(
            f"Listing pipeline executions for user {current_user['id']}",
            extra={
                'user_id': current_user['id'],
                'page': page,
                'limit': limit,
                'status': status,
                'config_id': config_id,
                'dataset_id': dataset_id
            }
        )
        
        # Initialize pipeline service
        pipeline_service = PipelineService(current_app.config, current_app.redis_client)
        
        # Get pipeline executions
        result = pipeline_service.list_executions(
            user_id=current_user['id'],
            page=page,
            limit=limit,
            status=status,
            config_id=config_id,
            dataset_id=dataset_id
        )
        
        logger.info(
            f"Successfully retrieved {len(result['executions'])} pipeline executions",
            extra={'user_id': current_user['id'], 'total': result['pagination']['total']}
        )
        
        return jsonify({
            'success': True,
            'data': result,
            'meta': {
                'timestamp': datetime.utcnow().isoformat()
            }
        }), 200
        
    except ValueError as e:
        logger.warning(f"Invalid pagination parameters: {e}")
        raise ValidationError(f"Invalid pagination parameters: {e}")
    except Exception as e:
        logger.error(f"Error listing pipeline executions: {e}")
        raise


@pipeline_bp.route('/stages', methods=['GET'])
@require_auth
def get_pipeline_stages() -> Dict[str, Any]:
    """
    Get available pipeline stages and their configurations.
    
    Returns:
        {
            "success": true,
            "data": {
                "stages": [
                    {
                        "name": "data_ingestion",
                        "type": "ingestion",
                        "description": "Data ingestion stage",
                        "config_schema": {...},
                        "required_fields": ["input_format", "validation_rules"]
                    }
                ]
            }
        }
    """
    try:
        current_user = request.user
        
        logger.info(
            f"Retrieving pipeline stages for user {current_user['id']}",
            extra={'user_id': current_user['id']}
        )
        
        # Initialize pipeline service
        pipeline_service = PipelineService(current_app.config, current_app.redis_client)
        
        # Get pipeline stages
        stages = pipeline_service.get_available_stages()
        
        logger.info(
            f"Successfully retrieved {len(stages)} pipeline stages",
            extra={'user_id': current_user['id']}
        )
        
        return jsonify({
            'success': True,
            'data': {'stages': stages},
            'meta': {
                'timestamp': datetime.utcnow().isoformat()
            }
        }), 200
        
    except Exception as e:
        logger.error(f"Error retrieving pipeline stages: {e}")
        raise


@pipeline_bp.route('/health', methods=['GET'])
def get_pipeline_health() -> Dict[str, Any]:
    """
    Get pipeline system health status.
    
    Returns:
        {
            "success": true,
            "data": {
                "health": {
                    "status": "healthy",
                    "queue_size": 5,
                    "active_executions": 3,
                    "failed_executions_24h": 2,
                    "average_execution_time": 120.5,
                    "last_health_check": "2024-01-01T00:00:00Z"
                }
            }
        }
    """
    try:
        logger.info("Retrieving pipeline system health")
        
        # Initialize pipeline service
        pipeline_service = PipelineService(current_app.config, current_app.redis_client)
        
        # Get health status
        health = pipeline_service.get_system_health()
        
        logger.info(
            "Successfully retrieved pipeline system health",
            extra={'status': health['status']}
        )
        
        return jsonify({
            'success': True,
            'data': {'health': health},
            'meta': {
                'timestamp': datetime.utcnow().isoformat()
            }
        }), 200
        
    except Exception as e:
        logger.error(f"Error retrieving pipeline health: {e}")
        raise