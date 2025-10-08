"""
Dataset management service for TechDeck-Python Pipeline Integration.

This module implements business logic for dataset operations including
CRUD operations, file processing, metadata management, and validation.
"""

import logging
import os
import uuid
import hashlib
from datetime import datetime
from typing import Dict, Any, Optional, List, Tuple
from werkzeug.datastructures import FileStorage

from ..utils.validation import validate_dataset_metadata, sanitize_input
from ..utils.logger import get_request_logger
from ..utils.file_handler import FileHandler
from ..error_handling.custom_errors import (
    ValidationError, ResourceNotFoundError, FileProcessingError, StorageError
)
from ..integration.redis_client import RedisClient


class DatasetService:
    """Service for dataset management operations."""
    
    def __init__(self, redis_client: RedisClient, file_handler: FileHandler):
        self.redis_client = redis_client
        self.file_handler = file_handler
        self.logger = logging.getLogger(__name__)
        
    def create_dataset(self, user_id: str, name: str, description: str,
                      file_data: Optional[FileStorage] = None,
                      metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Create a new dataset with optional file upload.
        
        Args:
            user_id: ID of the user creating the dataset
            name: Dataset name
            description: Dataset description
            file_data: Optional file data
            metadata: Optional dataset metadata
            
        Returns:
            Created dataset information
            
        Raises:
            ValidationError: If input data is invalid
            FileProcessingError: If file processing fails
            StorageError: If storage operation fails
        """
        request_logger = get_request_logger()
        request_logger.info(f"Creating dataset '{name}' for user {user_id}")
        
        try:
            # Validate inputs
            if not name or len(name.strip()) == 0:
                raise ValidationError("Dataset name is required")
            
            name = sanitize_input(name.strip())
            description = sanitize_input(description.strip()) if description else ""
            
            if metadata:
                validate_dataset_metadata(metadata)
                metadata = sanitize_input(metadata)
            
            # Generate dataset ID
            dataset_id = str(uuid.uuid4())
            
            # Process file if provided
            file_info = None
            if file_data:
                file_info = self._process_uploaded_file(file_data, dataset_id, user_id)
            
            # Create dataset record
            dataset_record = {
                'id': dataset_id,
                'user_id': user_id,
                'name': name,
                'description': description,
                'file_info': file_info,
                'metadata': metadata or {},
                'status': 'active',
                'created_at': datetime.utcnow().isoformat(),
                'updated_at': datetime.utcnow().isoformat(),
                'version': 1,
                'tags': [],
                'permissions': {
                    'read': True,
                    'write': True,
                    'delete': True,
                    'share': False
                }
            }
            
            # Store in Redis
            dataset_key = f"dataset:{dataset_id}"
            self.redis_client.set_json(dataset_key, dataset_record, ex=86400)  # 24 hours
            
            # Add to user's dataset list
            user_datasets_key = f"user_datasets:{user_id}"
            self.redis_client.sadd(user_datasets_key, dataset_id)
            self.redis_client.expire(user_datasets_key, 86400)
            
            request_logger.info(f"Dataset created successfully: {dataset_id}")
            
            return dataset_record
            
        except ValidationError:
            raise
        except Exception as e:
            request_logger.error(f"Error creating dataset: {e}")
            raise StorageError(f"Failed to create dataset: {str(e)}")
    
    def get_dataset(self, dataset_id: str, user_id: str) -> Dict[str, Any]:
        """
        Retrieve a dataset by ID.
        
        Args:
            dataset_id: Dataset ID
            user_id: User ID for access verification
            
        Returns:
            Dataset information
            
        Raises:
            ResourceNotFoundError: If dataset not found
            ValidationError: If access is denied
        """
        request_logger = get_request_logger()
        request_logger.info(f"Retrieving dataset {dataset_id} for user {user_id}")
        
        try:
            # Get dataset from Redis
            dataset_key = f"dataset:{dataset_id}"
            dataset_record = self.redis_client.get_json(dataset_key)
            
            if not dataset_record:
                raise ResourceNotFoundError(f"Dataset {dataset_id} not found")
            
            # Verify access
            if dataset_record['user_id'] != user_id:
                raise ValidationError("Access denied to dataset")
            
            request_logger.info(f"Dataset retrieved successfully: {dataset_id}")
            
            return dataset_record
            
        except (ResourceNotFoundError, ValidationError):
            raise
        except Exception as e:
            request_logger.error(f"Error retrieving dataset: {e}")
            raise StorageError(f"Failed to retrieve dataset: {str(e)}")
    
    def update_dataset(self, dataset_id: str, user_id: str, updates: Dict[str, Any]) -> Dict[str, Any]:
        """
        Update dataset information.
        
        Args:
            dataset_id: Dataset ID
            user_id: User ID for access verification
            updates: Fields to update
            
        Returns:
            Updated dataset information
            
        Raises:
            ResourceNotFoundError: If dataset not found
            ValidationError: If input data is invalid or access denied
        """
        request_logger = get_request_logger()
        request_logger.info(f"Updating dataset {dataset_id} for user {user_id}")
        
        try:
            # Get existing dataset
            dataset_record = self.get_dataset(dataset_id, user_id)
            
            # Validate updates
            allowed_fields = ['name', 'description', 'metadata', 'tags']
            for field in updates:
                if field not in allowed_fields:
                    raise ValidationError(f"Field '{field}' cannot be updated")
            
            # Sanitize updates
            sanitized_updates = {}
            for field, value in updates.items():
                if field in ['name', 'description']:
                    sanitized_updates[field] = sanitize_input(value.strip()) if value else ""
                elif field == 'metadata':
                    validate_dataset_metadata(value)
                    sanitized_updates[field] = sanitize_input(value)
                elif field == 'tags':
                    if isinstance(value, list):
                        sanitized_updates[field] = [sanitize_input(tag) for tag in value]
                    else:
                        raise ValidationError("Tags must be a list")
            
            # Apply updates
            dataset_record.update(sanitized_updates)
            dataset_record['updated_at'] = datetime.utcnow().isoformat()
            dataset_record['version'] += 1
            
            # Store updated record
            dataset_key = f"dataset:{dataset_id}"
            self.redis_client.set_json(dataset_key, dataset_record, ex=86400)
            
            request_logger.info(f"Dataset updated successfully: {dataset_id}")
            
            return dataset_record
            
        except (ResourceNotFoundError, ValidationError):
            raise
        except Exception as e:
            request_logger.error(f"Error updating dataset: {e}")
            raise StorageError(f"Failed to update dataset: {str(e)}")
    
    def delete_dataset(self, dataset_id: str, user_id: str) -> bool:
        """
        Delete a dataset.
        
        Args:
            dataset_id: Dataset ID
            user_id: User ID for access verification
            
        Returns:
            True if deleted successfully
            
        Raises:
            ResourceNotFoundError: If dataset not found
            ValidationError: If access is denied
        """
        request_logger = get_request_logger()
        request_logger.info(f"Deleting dataset {dataset_id} for user {user_id}")
        
        try:
            # Get dataset to verify access
            dataset_record = self.get_dataset(dataset_id, user_id)
            
            # Delete associated file if exists
            if dataset_record.get('file_info'):
                self._delete_dataset_file(dataset_record['file_info'])
            
            # Remove from Redis
            dataset_key = f"dataset:{dataset_id}"
            self.redis_client.delete(dataset_key)
            
            # Remove from user's dataset list
            user_datasets_key = f"user_datasets:{user_id}"
            self.redis_client.srem(user_datasets_key, dataset_id)
            
            request_logger.info(f"Dataset deleted successfully: {dataset_id}")
            
            return True
            
        except (ResourceNotFoundError, ValidationError):
            raise
        except Exception as e:
            request_logger.error(f"Error deleting dataset: {e}")
            raise StorageError(f"Failed to delete dataset: {str(e)}")
    
    def list_user_datasets(self, user_id: str, limit: int = 50, offset: int = 0,
                          status: Optional[str] = None) -> Dict[str, Any]:
        """
        List datasets for a user.
        
        Args:
            user_id: User ID
            limit: Maximum number of datasets to return
            offset: Number of datasets to skip
            status: Optional status filter
            
        Returns:
            Paginated list of datasets
            
        Raises:
            ValidationError: If pagination parameters are invalid
        """
        request_logger = get_request_logger()
        request_logger.info(f"Listing datasets for user {user_id}")
        
        try:
            # Validate pagination
            if limit < 1 or limit > 100:
                raise ValidationError("Limit must be between 1 and 100")
            if offset < 0:
                raise ValidationError("Offset must be non-negative")
            
            # Get user's dataset IDs
            user_datasets_key = f"user_datasets:{user_id}"
            dataset_ids = list(self.redis_client.smembers(user_datasets_key))
            
            # Apply status filter if provided
            if status:
                filtered_ids = []
                for dataset_id in dataset_ids:
                    dataset_key = f"dataset:{dataset_id}"
                    dataset_record = self.redis_client.get_json(dataset_key)
                    if dataset_record and dataset_record.get('status') == status:
                        filtered_ids.append(dataset_id)
                dataset_ids = filtered_ids
            
            # Apply pagination
            total_count = len(dataset_ids)
            paginated_ids = dataset_ids[offset:offset + limit]
            
            # Get dataset records
            datasets = []
            for dataset_id in paginated_ids:
                dataset_key = f"dataset:{dataset_id}"
                dataset_record = self.redis_client.get_json(dataset_key)
                if dataset_record:
                    datasets.append(dataset_record)
            
            # Calculate pagination metadata
            total_pages = (total_count + limit - 1) // limit
            current_page = (offset // limit) + 1
            
            request_logger.info(f"Retrieved {len(datasets)} datasets for user {user_id}")
            
            return {
                'datasets': datasets,
                'pagination': {
                    'total_count': total_count,
                    'limit': limit,
                    'offset': offset,
                    'total_pages': total_pages,
                    'current_page': current_page,
                    'has_next': offset + limit < total_count,
                    'has_previous': offset > 0
                }
            }
            
        except ValidationError:
            raise
        except Exception as e:
            request_logger.error(f"Error listing datasets: {e}")
            raise StorageError(f"Failed to list datasets: {str(e)}")
    
    def upload_dataset_file(self, dataset_id: str, user_id: str, file_data: FileStorage) -> Dict[str, Any]:
        """
        Upload a file to an existing dataset.
        
        Args:
            dataset_id: Dataset ID
            user_id: User ID for access verification
            file_data: File data to upload
            
        Returns:
            Updated dataset information
            
        Raises:
            ResourceNotFoundError: If dataset not found
            ValidationError: If access is denied or file is invalid
            FileProcessingError: If file processing fails
        """
        request_logger = get_request_logger()
        request_logger.info(f"Uploading file to dataset {dataset_id} for user {user_id}")
        
        try:
            # Get dataset to verify access
            dataset_record = self.get_dataset(dataset_id, user_id)
            
            # Process uploaded file
            file_info = self._process_uploaded_file(file_data, dataset_id, user_id)
            
            # Update dataset with file information
            dataset_record['file_info'] = file_info
            dataset_record['updated_at'] = datetime.utcnow().isoformat()
            dataset_record['version'] += 1
            
            # Store updated record
            dataset_key = f"dataset:{dataset_id}"
            self.redis_client.set_json(dataset_key, dataset_record, ex=86400)
            
            request_logger.info(f"File uploaded successfully to dataset {dataset_id}")
            
            return dataset_record
            
        except (ResourceNotFoundError, ValidationError, FileProcessingError):
            raise
        except Exception as e:
            request_logger.error(f"Error uploading file to dataset: {e}")
            raise StorageError(f"Failed to upload file to dataset: {str(e)}")
    
    def validate_dataset(self, dataset_id: str, user_id: str) -> Dict[str, Any]:
        """
        Validate dataset integrity and content.
        
        Args:
            dataset_id: Dataset ID
            user_id: User ID for access verification
            
        Returns:
            Validation results
            
        Raises:
            ResourceNotFoundError: If dataset not found
            ValidationError: If access is denied
        """
        request_logger = get_request_logger()
        request_logger.info(f"Validating dataset {dataset_id} for user {user_id}")
        
        try:
            # Get dataset
            dataset_record = self.get_dataset(dataset_id, user_id)
            
            validation_results = {
                'dataset_id': dataset_id,
                'validation_timestamp': datetime.utcnow().isoformat(),
                'checks': {}
            }
            
            # Check basic metadata
            validation_results['checks']['metadata'] = {
                'passed': bool(dataset_record.get('name') and dataset_record.get('user_id')),
                'message': 'Basic metadata validation'
            }
            
            # Check file integrity if file exists
            if dataset_record.get('file_info'):
                file_validation = self._validate_dataset_file(dataset_record['file_info'])
                validation_results['checks']['file_integrity'] = file_validation
            
            # Check permissions
            permissions = dataset_record.get('permissions', {})
            validation_results['checks']['permissions'] = {
                'passed': permissions.get('read', False),
                'message': 'Read permission check'
            }
            
            # Overall validation result
            all_checks_passed = all(check['passed'] for check in validation_results['checks'].values())
            validation_results['overall_status'] = 'valid' if all_checks_passed else 'invalid'
            validation_results['passed'] = all_checks_passed
            
            request_logger.info(f"Dataset validation completed: {validation_results['overall_status']}")
            
            return validation_results
            
        except (ResourceNotFoundError, ValidationError):
            raise
        except Exception as e:
            request_logger.error(f"Error validating dataset: {e}")
            raise ValidationError(f"Failed to validate dataset: {str(e)}")
    
    def get_dataset_statistics(self, dataset_id: str, user_id: str) -> Dict[str, Any]:
        """
        Get statistical information about a dataset.
        
        Args:
            dataset_id: Dataset ID
            user_id: User ID for access verification
            
        Returns:
            Dataset statistics
            
        Raises:
            ResourceNotFoundError: If dataset not found
            ValidationError: If access is denied
        """
        request_logger = get_request_logger()
        request_logger.info(f"Getting statistics for dataset {dataset_id} for user {user_id}")
        
        try:
            # Get dataset
            dataset_record = self.get_dataset(dataset_id, user_id)
            
            statistics = {
                'dataset_id': dataset_id,
                'basic_info': {
                    'name': dataset_record.get('name'),
                    'description': dataset_record.get('description'),
                    'created_at': dataset_record.get('created_at'),
                    'updated_at': dataset_record.get('updated_at'),
                    'version': dataset_record.get('version'),
                    'status': dataset_record.get('status')
                },
                'file_info': dataset_record.get('file_info'),
                'metadata': dataset_record.get('metadata', {}),
                'permissions': dataset_record.get('permissions', {}),
                'tags': dataset_record.get('tags', [])
            }
            
            # Add file-specific statistics if file exists
            if dataset_record.get('file_info'):
                file_stats = self._get_file_statistics(dataset_record['file_info'])
                statistics['file_statistics'] = file_stats
            
            request_logger.info(f"Dataset statistics retrieved successfully: {dataset_id}")
            
            return statistics
            
        except (ResourceNotFoundError, ValidationError):
            raise
        except Exception as e:
            request_logger.error(f"Error getting dataset statistics: {e}")
            raise ValidationError(f"Failed to get dataset statistics: {str(e)}")
    
    def search_datasets(self, user_id: str, query: str, limit: int = 20) -> List[Dict[str, Any]]:
        """
        Search datasets by name, description, or tags.
        
        Args:
            user_id: User ID for access verification
            query: Search query
            limit: Maximum number of results
            
        Returns:
            List of matching datasets
            
        Raises:
            ValidationError: If query is invalid
        """
        request_logger = get_request_logger()
        request_logger.info(f"Searching datasets for user {user_id} with query: {query}")
        
        try:
            # Validate query
            if not query or len(query.strip()) < 2:
                raise ValidationError("Search query must be at least 2 characters long")
            
            query = sanitize_input(query.strip().lower())
            
            # Get user's datasets
            user_datasets = self.list_user_datasets(user_id, limit=100)
            all_datasets = user_datasets['datasets']
            
            # Filter by search query
            matching_datasets = []
            for dataset in all_datasets:
                # Search in name, description, and tags
                name_match = query in dataset.get('name', '').lower()
                desc_match = query in dataset.get('description', '').lower()
                tags_match = any(query in tag.lower() for tag in dataset.get('tags', []))
                
                if name_match or desc_match or tags_match:
                    matching_datasets.append(dataset)
                
                if len(matching_datasets) >= limit:
                    break
            
            request_logger.info(f"Found {len(matching_datasets)} matching datasets")
            
            return matching_datasets
            
        except ValidationError:
            raise
        except Exception as e:
            request_logger.error(f"Error searching datasets: {e}")
            raise ValidationError(f"Failed to search datasets: {str(e)}")
    
    def _process_uploaded_file(self, file_data: FileStorage, dataset_id: str, user_id: str) -> Dict[str, Any]:
        """Process uploaded file and store it."""
        try:
            # Validate file
            if not file_data or file_data.filename == '':
                raise ValidationError("No file provided")
            
            # Get file information
            filename = file_data.filename
            file_size = len(file_data.read())
            file_data.seek(0)  # Reset file pointer
            
            # Validate file size (max 100MB)
            max_size = 100 * 1024 * 1024  # 100MB
            if file_size > max_size:
                raise ValidationError(f"File size exceeds maximum allowed size of {max_size // (1024*1024)}MB")
            
            # Validate file extension
            allowed_extensions = {'.csv', '.json', '.jsonl', '.parquet', '.txt'}
            file_extension = os.path.splitext(filename)[1].lower()
            if file_extension not in allowed_extensions:
                raise ValidationError(f"File type '{file_extension}' not supported. Allowed: {allowed_extensions}")
            
            # Generate file ID and path
            file_id = str(uuid.uuid4())
            file_hash = self._calculate_file_hash(file_data)
            file_data.seek(0)  # Reset file pointer
            
            # Store file using file handler
            storage_path = self.file_handler.store_file(file_data, file_id, user_id)
            
            # Create file information record
            file_info = {
                'file_id': file_id,
                'filename': filename,
                'file_size': file_size,
                'file_extension': file_extension,
                'file_hash': file_hash,
                'storage_path': storage_path,
                'upload_timestamp': datetime.utcnow().isoformat(),
                'mime_type': file_data.content_type or 'application/octet-stream'
            }
            
            self.logger.info(f"File processed successfully: {filename} ({file_size} bytes)")
            
            return file_info
            
        except ValidationError:
            raise
        except Exception as e:
            self.logger.error(f"Error processing uploaded file: {e}")
            raise FileProcessingError(f"Failed to process uploaded file: {str(e)}")
    
    def _calculate_file_hash(self, file_data: FileStorage) -> str:
        """Calculate SHA-256 hash of file content."""
        try:
            hash_sha256 = hashlib.sha256()
            for chunk in iter(lambda: file_data.read(4096), b""):
                hash_sha256.update(chunk)
            file_data.seek(0)  # Reset file pointer
            return hash_sha256.hexdigest()
        except Exception as e:
            self.logger.error(f"Error calculating file hash: {e}")
            raise FileProcessingError(f"Failed to calculate file hash: {str(e)}")
    
    def _delete_dataset_file(self, file_info: Dict[str, Any]) -> bool:
        """Delete dataset file from storage."""
        try:
            if file_info.get('storage_path'):
                return self.file_handler.delete_file(file_info['storage_path'])
            return True
        except Exception as e:
            self.logger.error(f"Error deleting dataset file: {e}")
            return False
    
    def _validate_dataset_file(self, file_info: Dict[str, Any]) -> Dict[str, Any]:
        """Validate dataset file integrity."""
        try:
            if not file_info.get('storage_path'):
                return {
                    'passed': False,
                    'message': 'No file associated with dataset'
                }
            
            # Check if file exists and is accessible
            file_exists = self.file_handler.file_exists(file_info['storage_path'])
            
            if not file_exists:
                return {
                    'passed': False,
                    'message': 'Dataset file not found in storage'
                }
            
            # Verify file size matches record
            actual_size = self.file_handler.get_file_size(file_info['storage_path'])
            expected_size = file_info.get('file_size', 0)
            
            if actual_size != expected_size:
                return {
                    'passed': False,
                    'message': f'File size mismatch: expected {expected_size}, got {actual_size}'
                }
            
            return {
                'passed': True,
                'message': 'File integrity validated successfully'
            }
            
        except Exception as e:
            self.logger.error(f"Error validating dataset file: {e}")
            return {
                'passed': False,
                'message': f'File validation error: {str(e)}'
            }
    
    def _get_file_statistics(self, file_info: Dict[str, Any]) -> Dict[str, Any]:
        """Get detailed file statistics."""
        try:
            if not file_info.get('storage_path'):
                return {}
            
            stats = {
                'file_size_bytes': file_info.get('file_size', 0),
                'file_size_human': self._format_file_size(file_info.get('file_size', 0)),
                'file_extension': file_info.get('file_extension', ''),
                'upload_timestamp': file_info.get('upload_timestamp', ''),
                'file_hash': file_info.get('file_hash', ''),
                'mime_type': file_info.get('mime_type', '')
            }
            
            # Add storage-specific information if available
            storage_stats = self.file_handler.get_file_statistics(file_info['storage_path'])
            stats.update(storage_stats)
            
            return stats
            
        except Exception as e:
            self.logger.error(f"Error getting file statistics: {e}")
            return {}
    
    def _format_file_size(self, size_bytes: int) -> str:
        """Format file size in human-readable format."""
        for unit in ['B', 'KB', 'MB', 'GB']:
            if size_bytes < 1024.0:
                return f"{size_bytes:.1f} {unit}"
            size_bytes /= 1024.0
        return f"{size_bytes:.1f} TB"