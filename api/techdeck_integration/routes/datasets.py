"""
Dataset Management API Routes.

This module provides REST API endpoints for dataset CRUD operations,
file upload handling, and dataset metadata management.
"""

import os
import uuid
from datetime import datetime
from typing import Dict, Any, Optional, List
from flask import Blueprint, request, jsonify, current_app
from werkzeug.utils import secure_filename
from werkzeug.datastructures import FileStorage

from ..auth.middleware import require_auth, require_role
from ..error_handling.custom_errors import (
    ValidationError, 
    FileUploadError, 
    DatasetNotFoundError,
    AuthenticationError,
    RateLimitExceededError
)
from ..utils.logger import get_logger
from ..utils.validation import validate_dataset_metadata, validate_file_upload
from ..services.dataset_service import DatasetService
from ..integration.redis_client import RedisClient


logger = get_logger(__name__)
datasets_bp = Blueprint('datasets', __name__, url_prefix='/api/v1/datasets')


@datasets_bp.route('', methods=['GET'])
@require_auth
def list_datasets() -> Dict[str, Any]:
    """
    List all datasets with pagination and filtering.
    
    Query Parameters:
        page (int): Page number (default: 1)
        limit (int): Items per page (default: 20, max: 100)
        search (str): Search term for dataset name or description
        format (str): Filter by file format (csv, json, jsonl, parquet)
        status (str): Filter by processing status
        
    Returns:
        {
            "success": true,
            "data": {
                "datasets": [...],
                "pagination": {
                    "page": 1,
                    "limit": 20,
                    "total": 100,
                    "pages": 5
                }
            },
            "meta": {
                "timestamp": "2024-01-01T00:00:00Z"
            }
        }
    """
    try:
        # Parse query parameters
        page = int(request.args.get('page', 1))
        limit = min(int(request.args.get('limit', 20)), 100)  # Max 100 per page
        search = request.args.get('search', '')
        file_format = request.args.get('format')
        status = request.args.get('status')
        
        # Get current user from request context
        current_user = request.user
        
        logger.info(
            f"Listing datasets for user {current_user['id']}",
            extra={
                'user_id': current_user['id'],
                'page': page,
                'limit': limit,
                'search': search,
                'format': file_format,
                'status': status
            }
        )
        
        # Initialize dataset service
        dataset_service = DatasetService(current_app.config, current_app.redis_client)
        
        # Get datasets with pagination
        result = dataset_service.list_datasets(
            user_id=current_user['id'],
            page=page,
            limit=limit,
            search=search,
            file_format=file_format,
            status=status
        )
        
        logger.info(
            f"Successfully retrieved {len(result['datasets'])} datasets",
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
        logger.error(f"Error listing datasets: {e}")
        raise


@datasets_bp.route('/<dataset_id>', methods=['GET'])
@require_auth
def get_dataset(dataset_id: str) -> Dict[str, Any]:
    """
    Get detailed information about a specific dataset.
    
    Args:
        dataset_id: UUID of the dataset
        
    Returns:
        {
            "success": true,
            "data": {
                "dataset": {
                    "id": "uuid",
                    "name": "Dataset Name",
                    "description": "Dataset description",
                    "format": "csv",
                    "size": 1024,
                    "rows": 1000,
                    "columns": 10,
                    "status": "processed",
                    "created_at": "2024-01-01T00:00:00Z",
                    "updated_at": "2024-01-01T00:00:00Z",
                    "metadata": {...}
                }
            }
        }
    """
    try:
        current_user = request.user
        
        logger.info(
            f"Retrieving dataset {dataset_id}",
            extra={'user_id': current_user['id'], 'dataset_id': dataset_id}
        )
        
        # Initialize dataset service
        dataset_service = DatasetService(current_app.config, current_app.redis_client)
        
        # Get dataset by ID
        dataset = dataset_service.get_dataset(dataset_id, current_user['id'])
        
        if not dataset:
            logger.warning(
                f"Dataset {dataset_id} not found for user {current_user['id']}",
                extra={'user_id': current_user['id'], 'dataset_id': dataset_id}
            )
            raise DatasetNotFoundError(f"Dataset {dataset_id} not found")
        
        logger.info(
            f"Successfully retrieved dataset {dataset_id}",
            extra={'user_id': current_user['id'], 'dataset_id': dataset_id}
        )
        
        return jsonify({
            'success': True,
            'data': {'dataset': dataset},
            'meta': {
                'timestamp': datetime.utcnow().isoformat()
            }
        }), 200
        
    except DatasetNotFoundError:
        raise
    except Exception as e:
        logger.error(f"Error retrieving dataset {dataset_id}: {e}")
        raise


@datasets_bp.route('', methods=['POST'])
@require_auth
def create_dataset() -> Dict[str, Any]:
    """
    Create a new dataset with metadata.
    
    Request Body:
        {
            "name": "Dataset Name",
            "description": "Dataset description",
            "format": "csv",
            "metadata": {
                "source": "user_upload",
                "tags": ["tag1", "tag2"]
            }
        }
        
    Returns:
        {
            "success": true,
            "data": {
                "dataset": {
                    "id": "uuid",
                    "name": "Dataset Name",
                    "description": "Dataset description",
                    "format": "csv",
                    "status": "pending",
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
        required_fields = ['name', 'format']
        for field in required_fields:
            if field not in data or not data[field]:
                raise ValidationError(f"Missing required field: {field}")
        
        # Validate dataset metadata
        validate_dataset_metadata(data)
        
        logger.info(
            f"Creating dataset for user {current_user['id']}",
            extra={
                'user_id': current_user['id'],
                'dataset_name': data['name'],
                'format': data['format']
            }
        )
        
        # Initialize dataset service
        dataset_service = DatasetService(current_app.config, current_app.redis_client)
        
        # Create dataset
        dataset = dataset_service.create_dataset(
            user_id=current_user['id'],
            name=data['name'],
            description=data.get('description', ''),
            format=data['format'],
            metadata=data.get('metadata', {})
        )
        
        logger.info(
            f"Successfully created dataset {dataset['id']}",
            extra={'user_id': current_user['id'], 'dataset_id': dataset['id']}
        )
        
        return jsonify({
            'success': True,
            'data': {'dataset': dataset},
            'meta': {
                'timestamp': datetime.utcnow().isoformat()
            }
        }), 201
        
    except ValidationError:
        raise
    except Exception as e:
        logger.error(f"Error creating dataset: {e}")
        raise


@datasets_bp.route('/<dataset_id>/upload', methods=['POST'])
@require_auth
def upload_dataset_file(dataset_id: str) -> Dict[str, Any]:
    """
    Upload a file for an existing dataset.
    
    Args:
        dataset_id: UUID of the dataset
        
    Request:
        Multipart form data with file upload
        
    Returns:
        {
            "success": true,
            "data": {
                "upload": {
                    "id": "upload_uuid",
                    "dataset_id": "dataset_uuid",
                    "filename": "data.csv",
                    "size": 1024,
                    "status": "uploaded",
                    "uploaded_at": "2024-01-01T00:00:00Z"
                }
            }
        }
    """
    try:
        current_user = request.user
        
        # Check if file is present in request
        if 'file' not in request.files:
            raise ValidationError("No file provided in request")
        
        file = request.files['file']
        
        if file.filename == '':
            raise ValidationError("No file selected")
        
        # Validate file upload
        validate_file_upload(file, current_app.config)
        
        logger.info(
            f"Uploading file for dataset {dataset_id}",
            extra={
                'user_id': current_user['id'],
                'dataset_id': dataset_id,
                'filename': file.filename
            }
        )
        
        # Initialize dataset service
        dataset_service = DatasetService(current_app.config, current_app.redis_client)
        
        # Upload file
        upload_result = dataset_service.upload_file(
            dataset_id=dataset_id,
            user_id=current_user['id'],
            file=file
        )
        
        logger.info(
            f"Successfully uploaded file for dataset {dataset_id}",
            extra={
                'user_id': current_user['id'],
                'dataset_id': dataset_id,
                'upload_id': upload_result['id']
            }
        )
        
        return jsonify({
            'success': True,
            'data': {'upload': upload_result},
            'meta': {
                'timestamp': datetime.utcnow().isoformat()
            }
        }), 201
        
    except ValidationError:
        raise
    except Exception as e:
        logger.error(f"Error uploading file for dataset {dataset_id}: {e}")
        raise


@datasets_bp.route('/<dataset_id>', methods=['PUT'])
@require_auth
def update_dataset(dataset_id: str) -> Dict[str, Any]:
    """
    Update dataset metadata.
    
    Args:
        dataset_id: UUID of the dataset
        
    Request Body:
        {
            "name": "Updated Dataset Name",
            "description": "Updated description",
            "metadata": {
                "tags": ["new_tag"]
            }
        }
        
    Returns:
        {
            "success": true,
            "data": {
                "dataset": {
                    "id": "uuid",
                    "name": "Updated Dataset Name",
                    "description": "Updated description",
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
            f"Updating dataset {dataset_id}",
            extra={
                'user_id': current_user['id'],
                'dataset_id': dataset_id,
                'update_data': data
            }
        )
        
        # Initialize dataset service
        dataset_service = DatasetService(current_app.config, current_app.redis_client)
        
        # Update dataset
        dataset = dataset_service.update_dataset(
            dataset_id=dataset_id,
            user_id=current_user['id'],
            update_data=data
        )
        
        if not dataset:
            logger.warning(
                f"Dataset {dataset_id} not found for update",
                extra={'user_id': current_user['id'], 'dataset_id': dataset_id}
            )
            raise DatasetNotFoundError(f"Dataset {dataset_id} not found")
        
        logger.info(
            f"Successfully updated dataset {dataset_id}",
            extra={'user_id': current_user['id'], 'dataset_id': dataset_id}
        )
        
        return jsonify({
            'success': True,
            'data': {'dataset': dataset},
            'meta': {
                'timestamp': datetime.utcnow().isoformat()
            }
        }), 200
        
    except ValidationError:
        raise
    except DatasetNotFoundError:
        raise
    except Exception as e:
        logger.error(f"Error updating dataset {dataset_id}: {e}")
        raise


@datasets_bp.route('/<dataset_id>', methods=['DELETE'])
@require_auth
def delete_dataset(dataset_id: str) -> Dict[str, Any]:
    """
    Delete a dataset and its associated files.
    
    Args:
        dataset_id: UUID of the dataset
        
    Returns:
        {
            "success": true,
            "data": {
                "message": "Dataset deleted successfully"
            }
        }
    """
    try:
        current_user = request.user
        
        logger.info(
            f"Deleting dataset {dataset_id}",
            extra={'user_id': current_user['id'], 'dataset_id': dataset_id}
        )
        
        # Initialize dataset service
        dataset_service = DatasetService(current_app.config, current_app.redis_client)
        
        # Delete dataset
        success = dataset_service.delete_dataset(dataset_id, current_user['id'])
        
        if not success:
            logger.warning(
                f"Dataset {dataset_id} not found for deletion",
                extra={'user_id': current_user['id'], 'dataset_id': dataset_id}
            )
            raise DatasetNotFoundError(f"Dataset {dataset_id} not found")
        
        logger.info(
            f"Successfully deleted dataset {dataset_id}",
            extra={'user_id': current_user['id'], 'dataset_id': dataset_id}
        )
        
        return jsonify({
            'success': True,
            'data': {'message': 'Dataset deleted successfully'},
            'meta': {
                'timestamp': datetime.utcnow().isoformat()
            }
        }), 200
        
    except DatasetNotFoundError:
        raise
    except Exception as e:
        logger.error(f"Error deleting dataset {dataset_id}: {e}")
        raise


@datasets_bp.route('/<dataset_id>/download', methods=['GET'])
@require_auth
def download_dataset(dataset_id: str) -> Any:
    """
    Download dataset file.
    
    Args:
        dataset_id: UUID of the dataset
        
    Returns:
        File download response
    """
    try:
        current_user = request.user
        
        logger.info(
            f"Downloading dataset {dataset_id}",
            extra={'user_id': current_user['id'], 'dataset_id': dataset_id}
        )
        
        # Initialize dataset service
        dataset_service = DatasetService(current_app.config, current_app.redis_client)
        
        # Get dataset file
        file_data = dataset_service.get_dataset_file(dataset_id, current_user['id'])
        
        if not file_data:
            logger.warning(
                f"Dataset file {dataset_id} not found",
                extra={'user_id': current_user['id'], 'dataset_id': dataset_id}
            )
            raise DatasetNotFoundError(f"Dataset file {dataset_id} not found")
        
        logger.info(
            f"Successfully prepared dataset {dataset_id} for download",
            extra={'user_id': current_user['id'], 'dataset_id': dataset_id}
        )
        
        # Return file response
        return file_data
        
    except DatasetNotFoundError:
        raise
    except Exception as e:
        logger.error(f"Error downloading dataset {dataset_id}: {e}")
        raise


@datasets_bp.route('/<dataset_id>/validate', methods=['POST'])
@require_auth
def validate_dataset(dataset_id: str) -> Dict[str, Any]:
    """
    Validate dataset integrity and format.
    
    Args:
        dataset_id: UUID of the dataset
        
    Returns:
        {
            "success": true,
            "data": {
                "validation": {
                    "valid": true,
                    "errors": [],
                    "warnings": ["Column 'age' has missing values"],
                    "stats": {
                        "rows": 1000,
                        "columns": 10,
                        "missing_values": 50
                    }
                }
            }
        }
    """
    try:
        current_user = request.user
        
        logger.info(
            f"Validating dataset {dataset_id}",
            extra={'user_id': current_user['id'], 'dataset_id': dataset_id}
        )
        
        # Initialize dataset service
        dataset_service = DatasetService(current_app.config, current_app.redis_client)
        
        # Validate dataset
        validation_result = dataset_service.validate_dataset(dataset_id, current_user['id'])
        
        logger.info(
            f"Successfully validated dataset {dataset_id}",
            extra={
                'user_id': current_user['id'],
                'dataset_id': dataset_id,
                'valid': validation_result['valid']
            }
        )
        
        return jsonify({
            'success': True,
            'data': {'validation': validation_result},
            'meta': {
                'timestamp': datetime.utcnow().isoformat()
            }
        }), 200
        
    except Exception as e:
        logger.error(f"Error validating dataset {dataset_id}: {e}")
        raise


@datasets_bp.route('/formats', methods=['GET'])
@require_auth
def get_supported_formats() -> Dict[str, Any]:
    """
    Get list of supported dataset formats.
    
    Returns:
        {
            "success": true,
            "data": {
                "formats": [
                    {
                        "format": "csv",
                        "description": "Comma-separated values",
                        "max_size_mb": 100,
                        "mime_type": "text/csv"
                    },
                    {
                        "format": "json",
                        "description": "JavaScript Object Notation",
                        "max_size_mb": 50,
                        "mime_type": "application/json"
                    }
                ]
            }
        }
    """
    try:
        current_user = request.user
        
        logger.info(
            "Retrieving supported dataset formats",
            extra={'user_id': current_user['id']}
        )
        
        # Get supported formats from config
        formats = current_app.config.get('SUPPORTED_DATASET_FORMATS', [])
        
        logger.info(
            f"Successfully retrieved {len(formats)} supported formats",
            extra={'user_id': current_user['id']}
        )
        
        return jsonify({
            'success': True,
            'data': {'formats': formats},
            'meta': {
                'timestamp': datetime.utcnow().isoformat()
            }
        }), 200
        
    except Exception as e:
        logger.error(f"Error retrieving supported formats: {e}")
        raise