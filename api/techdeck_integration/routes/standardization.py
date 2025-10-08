"""
Standardization API routes for TechDeck-Python Pipeline Integration.

This module implements REST API endpoints for data standardization operations,
including schema validation, format conversion, and data normalization.
"""

import logging
from datetime import datetime
from typing import Dict, Any, Optional, List
from flask import Blueprint, request, jsonify, g
from werkzeug.exceptions import BadRequest, NotFound

from ..utils.validation import validate_standardization_request, sanitize_input
from ..utils.logger import get_request_logger
from ..error_handling.custom_errors import (
    ValidationError, StandardizationError, ResourceNotFoundError
)
from ..integration.redis_client import RedisClient
from ..auth.decorators import require_auth, require_role

# Initialize blueprint
standardization_bp = Blueprint('standardization', __name__)
logger = logging.getLogger(__name__)


@standardization_bp.route('/schemas', methods=['GET'])
@require_auth
def list_schemas():
    """
    List available standardization schemas.
    
    Returns:
        List of available schemas with metadata
        
    Raises:
        ResourceNotFoundError: If no schemas are available
    """
    request_logger = get_request_logger(g.request_id)
    request_logger.info("Listing available standardization schemas")
    
    try:
        # Get Redis client from app context
        redis_client = g.app.redis_client
        
        # Retrieve available schemas from cache or database
        schemas = _get_available_schemas(redis_client)
        
        if not schemas:
            raise ResourceNotFoundError("No standardization schemas available")
        
        # Log successful retrieval
        request_logger.info(f"Retrieved {len(schemas)} schemas")
        
        return jsonify({
            'success': True,
            'data': {
                'schemas': schemas,
                'count': len(schemas),
                'timestamp': datetime.utcnow().isoformat()
            }
        }), 200
        
    except ResourceNotFoundError:
        raise
    except Exception as e:
        request_logger.error(f"Error listing schemas: {e}")
        raise StandardizationError(f"Failed to retrieve schemas: {str(e)}")


@standardization_bp.route('/schemas/<schema_id>', methods=['GET'])
@require_auth
def get_schema(schema_id: str):
    """
    Get detailed information about a specific standardization schema.
    
    Args:
        schema_id: Unique identifier for the schema
        
    Returns:
        Schema details including fields, validation rules, and metadata
        
    Raises:
        ResourceNotFoundError: If schema is not found
        ValidationError: If schema_id is invalid
    """
    request_logger = get_request_logger(g.request_id)
    request_logger.info(f"Retrieving schema: {schema_id}")
    
    try:
        # Validate schema ID
        if not schema_id or not isinstance(schema_id, str):
            raise ValidationError("Invalid schema ID format")
        
        # Sanitize input
        schema_id = sanitize_input(schema_id)
        
        # Get Redis client from app context
        redis_client = g.app.redis_client
        
        # Retrieve schema details
        schema = _get_schema_details(redis_client, schema_id)
        
        if not schema:
            raise ResourceNotFoundError(f"Schema '{schema_id}' not found")
        
        # Log successful retrieval
        request_logger.info(f"Retrieved schema: {schema_id}")
        
        return jsonify({
            'success': True,
            'data': {
                'schema': schema,
                'timestamp': datetime.utcnow().isoformat()
            }
        }), 200
        
    except (ValidationError, ResourceNotFoundError):
        raise
    except Exception as e:
        request_logger.error(f"Error retrieving schema {schema_id}: {e}")
        raise StandardizationError(f"Failed to retrieve schema: {str(e)}")


@standardization_bp.route('/validate', methods=['POST'])
@require_auth
def validate_data():
    """
    Validate data against a standardization schema.
    
    Request Body:
        {
            "data": {...},          # Data to validate
            "schema_id": "string",  # Schema to validate against
            "options": {...}        # Optional validation options
        }
        
    Returns:
        Validation results including errors and warnings
        
    Raises:
        ValidationError: If request data is invalid
        ResourceNotFoundError: If schema is not found
    """
    request_logger = get_request_logger(g.request_id)
    request_logger.info("Validating data against schema")
    
    try:
        # Get request data
        request_data = request.get_json()
        if not request_data:
            raise ValidationError("Request body is required")
        
        # Validate request structure
        validation_result = validate_standardization_request(request_data)
        if not validation_result['valid']:
            raise ValidationError(validation_result['errors'])
        
        # Extract data and schema ID
        data = request_data.get('data')
        schema_id = request_data.get('schema_id')
        options = request_data.get('options', {})
        
        # Sanitize inputs
        data = sanitize_input(data)
        schema_id = sanitize_input(schema_id)
        
        request_logger.info(f"Validating data against schema: {schema_id}")
        
        # Get Redis client from app context
        redis_client = g.app.redis_client
        
        # Retrieve schema
        schema = _get_schema_details(redis_client, schema_id)
        if not schema:
            raise ResourceNotFoundError(f"Schema '{schema_id}' not found")
        
        # Perform validation
        validation_result = _validate_against_schema(data, schema, options)
        
        # Log validation results
        if validation_result['valid']:
            request_logger.info(f"Data validation passed for schema: {schema_id}")
        else:
            request_logger.warning(f"Data validation failed for schema: {schema_id}")
        
        return jsonify({
            'success': True,
            'data': {
                'valid': validation_result['valid'],
                'errors': validation_result.get('errors', []),
                'warnings': validation_result.get('warnings', []),
                'schema_id': schema_id,
                'timestamp': datetime.utcnow().isoformat()
            }
        }), 200
        
    except (ValidationError, ResourceNotFoundError):
        raise
    except Exception as e:
        request_logger.error(f"Error validating data: {e}")
        raise StandardizationError(f"Failed to validate data: {str(e)}")


@standardization_bp.route('/transform', methods=['POST'])
@require_auth
def transform_data():
    """
    Transform data according to a standardization schema.
    
    Request Body:
        {
            "data": {...},          # Data to transform
            "schema_id": "string",  # Schema to transform against
            "options": {...}        # Optional transformation options
        }
        
    Returns:
        Transformed data and transformation metadata
        
    Raises:
        ValidationError: If request data is invalid
        ResourceNotFoundError: If schema is not found
        StandardizationError: If transformation fails
    """
    request_logger = get_request_logger(g.request_id)
    request_logger.info("Transforming data according to schema")
    
    try:
        # Get request data
        request_data = request.get_json()
        if not request_data:
            raise ValidationError("Request body is required")
        
        # Validate request structure
        validation_result = validate_standardization_request(request_data)
        if not validation_result['valid']:
            raise ValidationError(validation_result['errors'])
        
        # Extract data and schema ID
        data = request_data.get('data')
        schema_id = request_data.get('schema_id')
        options = request_data.get('options', {})
        
        # Sanitize inputs
        data = sanitize_input(data)
        schema_id = sanitize_input(schema_id)
        
        request_logger.info(f"Transforming data using schema: {schema_id}")
        
        # Get Redis client from app context
        redis_client = g.app.redis_client
        
        # Retrieve schema
        schema = _get_schema_details(redis_client, schema_id)
        if not schema:
            raise ResourceNotFoundError(f"Schema '{schema_id}' not found")
        
        # Perform transformation
        transformation_result = _transform_data(data, schema, options)
        
        # Log transformation results
        request_logger.info(f"Data transformation completed for schema: {schema_id}")
        
        return jsonify({
            'success': True,
            'data': {
                'transformed_data': transformation_result['data'],
                'transformations_applied': transformation_result.get('transformations', []),
                'metadata': transformation_result.get('metadata', {}),
                'schema_id': schema_id,
                'timestamp': datetime.utcnow().isoformat()
            }
        }), 200
        
    except (ValidationError, ResourceNotFoundError):
        raise
    except Exception as e:
        request_logger.error(f"Error transforming data: {e}")
        raise StandardizationError(f"Failed to transform data: {str(e)}")


@standardization_bp.route('/batch-transform', methods=['POST'])
@require_auth
def batch_transform_data():
    """
    Transform multiple data items according to a standardization schema.
    
    Request Body:
        {
            "items": [...],         # List of data items to transform
            "schema_id": "string",  # Schema to transform against
            "options": {...}        # Optional transformation options
        }
        
    Returns:
        List of transformed items with transformation metadata
        
    Raises:
        ValidationError: If request data is invalid
        ResourceNotFoundError: If schema is not found
        StandardizationError: If transformation fails
    """
    request_logger = get_request_logger(g.request_id)
    request_logger.info("Batch transforming data according to schema")
    
    try:
        # Get request data
        request_data = request.get_json()
        if not request_data:
            raise ValidationError("Request body is required")
        
        # Validate request structure
        if 'items' not in request_data:
            raise ValidationError("Items field is required")
        
        items = request_data.get('items')
        schema_id = request_data.get('schema_id')
        options = request_data.get('options', {})
        
        # Validate items is a list
        if not isinstance(items, list):
            raise ValidationError("Items must be a list")
        
        if len(items) == 0:
            raise ValidationError("Items list cannot be empty")
        
        if len(items) > 1000:  # Batch size limit
            raise ValidationError("Batch size cannot exceed 1000 items")
        
        # Sanitize inputs
        items = [sanitize_input(item) for item in items]
        schema_id = sanitize_input(schema_id)
        
        request_logger.info(f"Batch transforming {len(items)} items using schema: {schema_id}")
        
        # Get Redis client from app context
        redis_client = g.app.redis_client
        
        # Retrieve schema
        schema = _get_schema_details(redis_client, schema_id)
        if not schema:
            raise ResourceNotFoundError(f"Schema '{schema_id}' not found")
        
        # Process items in batches for performance
        results = []
        batch_size = 100
        
        for i in range(0, len(items), batch_size):
            batch = items[i:i + batch_size]
            batch_results = []
            
            for item in batch:
                try:
                    result = _transform_data(item, schema, options)
                    batch_results.append({
                        'success': True,
                        'data': result['data'],
                        'transformations': result.get('transformations', []),
                        'metadata': result.get('metadata', {})
                    })
                except Exception as e:
                    batch_results.append({
                        'success': False,
                        'error': str(e),
                        'item_index': i + batch_results.index({
                            'success': False,
                            'error': str(e)
                        })
                    })
            
            results.extend(batch_results)
        
        # Calculate success rate
        successful = sum(1 for r in results if r['success'])
        success_rate = successful / len(results) * 100
        
        request_logger.info(f"Batch transformation completed: {successful}/{len(results)} successful")
        
        return jsonify({
            'success': True,
            'data': {
                'results': results,
                'total_items': len(results),
                'successful_items': successful,
                'failed_items': len(results) - successful,
                'success_rate': success_rate,
                'schema_id': schema_id,
                'timestamp': datetime.utcnow().isoformat()
            }
        }), 200
        
    except (ValidationError, ResourceNotFoundError):
        raise
    except Exception as e:
        request_logger.error(f"Error in batch transformation: {e}")
        raise StandardizationError(f"Failed to batch transform data: {str(e)}")


@standardization_bp.route('/schemas/<schema_id>/stats', methods=['GET'])
@require_auth
def get_schema_stats(schema_id: str):
    """
    Get usage statistics for a specific standardization schema.
    
    Args:
        schema_id: Unique identifier for the schema
        
    Returns:
        Schema usage statistics and performance metrics
        
    Raises:
        ResourceNotFoundError: If schema is not found
        ValidationError: If schema_id is invalid
    """
    request_logger = get_request_logger(g.request_id)
    request_logger.info(f"Retrieving schema statistics: {schema_id}")
    
    try:
        # Validate schema ID
        if not schema_id or not isinstance(schema_id, str):
            raise ValidationError("Invalid schema ID format")
        
        # Sanitize input
        schema_id = sanitize_input(schema_id)
        
        # Get Redis client from app context
        redis_client = g.app.redis_client
        
        # Retrieve schema
        schema = _get_schema_details(redis_client, schema_id)
        if not schema:
            raise ResourceNotFoundError(f"Schema '{schema_id}' not found")
        
        # Get statistics from Redis
        stats = _get_schema_statistics(redis_client, schema_id)
        
        # Log successful retrieval
        request_logger.info(f"Retrieved statistics for schema: {schema_id}")
        
        return jsonify({
            'success': True,
            'data': {
                'schema_id': schema_id,
                'statistics': stats,
                'timestamp': datetime.utcnow().isoformat()
            }
        }), 200
        
    except (ValidationError, ResourceNotFoundError):
        raise
    except Exception as e:
        request_logger.error(f"Error retrieving schema statistics {schema_id}: {e}")
        raise StandardizationError(f"Failed to retrieve schema statistics: {str(e)}")


# Helper Functions
def _get_available_schemas(redis_client: RedisClient) -> List[Dict[str, Any]]:
    """Retrieve list of available standardization schemas."""
    try:
        # Try to get from Redis cache first
        cached_schemas = redis_client.get('standardization:schemas:list')
        if cached_schemas:
            return cached_schemas
        
        # If not in cache, get from database (placeholder)
        schemas = [
            {
                'id': 'healthcare-patient',
                'name': 'Healthcare Patient Data',
                'description': 'Standard schema for patient demographic and medical data',
                'version': '1.0.0',
                'fields_count': 25,
                'last_updated': '2024-01-15T10:30:00Z'
            },
            {
                'id': 'clinical-trial',
                'name': 'Clinical Trial Data',
                'description': 'Schema for clinical trial participant data',
                'version': '2.1.0',
                'fields_count': 42,
                'last_updated': '2024-01-10T14:20:00Z'
            },
            {
                'id': 'mental-health-session',
                'name': 'Mental Health Session Data',
                'description': 'Schema for therapeutic session recordings and metadata',
                'version': '1.2.0',
                'fields_count': 18,
                'last_updated': '2024-01-12T09:15:00Z'
            }
        ]
        
        # Cache for 1 hour
        redis_client.setex('standardization:schemas:list', 3600, schemas)
        
        return schemas
        
    except Exception as e:
        logger.error(f"Error retrieving available schemas: {e}")
        return []


def _get_schema_details(redis_client: RedisClient, schema_id: str) -> Optional[Dict[str, Any]]:
    """Retrieve detailed information about a specific schema."""
    try:
        # Try to get from Redis cache first
        cache_key = f'standardization:schema:{schema_id}'
        cached_schema = redis_client.get(cache_key)
        if cached_schema:
            return cached_schema
        
        # If not in cache, get from database (placeholder)
        schemas = _get_available_schemas(redis_client)
        schema = next((s for s in schemas if s['id'] == schema_id), None)
        
        if schema:
            # Add detailed field information
            schema['fields'] = _get_schema_fields(schema_id)
            schema['validation_rules'] = _get_validation_rules(schema_id)
            schema['transformation_rules'] = _get_transformation_rules(schema_id)
            
            # Cache for 30 minutes
            redis_client.setex(cache_key, 1800, schema)
        
        return schema
        
    except Exception as e:
        logger.error(f"Error retrieving schema details for {schema_id}: {e}")
        return None


def _get_schema_fields(schema_id: str) -> List[Dict[str, Any]]:
    """Get field definitions for a schema."""
    # Placeholder field definitions
    fields_map = {
        'healthcare-patient': [
            {'name': 'patient_id', 'type': 'string', 'required': True, 'description': 'Unique patient identifier'},
            {'name': 'first_name', 'type': 'string', 'required': True, 'max_length': 50},
            {'name': 'last_name', 'type': 'string', 'required': True, 'max_length': 50},
            {'name': 'date_of_birth', 'type': 'date', 'required': True},
            {'name': 'gender', 'type': 'enum', 'required': True, 'values': ['M', 'F', 'O']},
            {'name': 'phone', 'type': 'string', 'required': False, 'pattern': '^\\+?[1-9]\\d{1,14}$'},
            {'name': 'email', 'type': 'email', 'required': False},
            {'name': 'address', 'type': 'object', 'required': False, 'fields': [
                {'name': 'street', 'type': 'string', 'required': True},
                {'name': 'city', 'type': 'string', 'required': True},
                {'name': 'state', 'type': 'string', 'required': True, 'max_length': 2},
                {'name': 'zip_code', 'type': 'string', 'required': True, 'pattern': '^\\d{5}(-\\d{4})?$'}
            ]}
        ],
        'clinical-trial': [
            {'name': 'trial_id', 'type': 'string', 'required': True},
            {'name': 'participant_id', 'type': 'string', 'required': True},
            {'name': 'enrollment_date', 'type': 'date', 'required': True},
            {'name': 'status', 'type': 'enum', 'required': True, 'values': ['enrolled', 'active', 'completed', 'withdrawn']},
            {'name': 'demographics', 'type': 'object', 'required': True, 'fields': [
                {'name': 'age', 'type': 'integer', 'required': True, 'min': 18, 'max': 100},
                {'name': 'gender', 'type': 'enum', 'required': True, 'values': ['M', 'F', 'O']},
                {'name': 'ethnicity', 'type': 'string', 'required': True}
            ]}
        ],
        'mental-health-session': [
            {'name': 'session_id', 'type': 'string', 'required': True},
            {'name': 'therapist_id', 'type': 'string', 'required': True},
            {'name': 'patient_id', 'type': 'string', 'required': True},
            {'name': 'session_date', 'type': 'datetime', 'required': True},
            {'name': 'duration_minutes', 'type': 'integer', 'required': True, 'min': 1, 'max': 240},
            {'name': 'session_type', 'type': 'enum', 'required': True, 'values': ['individual', 'group', 'family']},
            {'name': 'notes', 'type': 'string', 'required': False, 'max_length': 2000}
        ]
    }
    
    return fields_map.get(schema_id, [])


def _get_validation_rules(schema_id: str) -> Dict[str, Any]:
    """Get validation rules for a schema."""
    # Placeholder validation rules
    return {
        'strict_mode': True,
        'allow_extra_fields': False,
        'validate_types': True,
        'validate_formats': True,
        'custom_validators': []
    }


def _get_transformation_rules(schema_id: str) -> Dict[str, Any]:
    """Get transformation rules for a schema."""
    # Placeholder transformation rules
    return {
        'normalize_case': True,
        'trim_whitespace': True,
        'format_dates': True,
        'standardize_enums': True,
        'remove_pii': False
    }


def _validate_against_schema(data: Dict[str, Any], schema: Dict[str, Any], options: Dict[str, Any]) -> Dict[str, Any]:
    """Validate data against a schema."""
    errors = []
    warnings = []
    
    try:
        # Get field definitions
        fields = schema.get('fields', [])
        field_map = {field['name']: field for field in fields}
        
        # Check required fields
        required_fields = [field['name'] for field in fields if field.get('required', False)]
        missing_fields = [field for field in required_fields if field not in data]
        
        if missing_fields:
            errors.append(f"Missing required fields: {', '.join(missing_fields)}")
        
        # Validate each field
        for field_name, field_value in data.items():
            if field_name not in field_map:
                if not options.get('allow_extra_fields', False):
                    warnings.append(f"Extra field '{field_name}' not in schema")
                continue
            
            field_def = field_map[field_name]
            field_errors = _validate_field(field_name, field_value, field_def)
            errors.extend(field_errors)
        
        return {
            'valid': len(errors) == 0,
            'errors': errors,
            'warnings': warnings
        }
        
    except Exception as e:
        logger.error(f"Error validating data against schema: {e}")
        return {
            'valid': False,
            'errors': [f"Validation error: {str(e)}"],
            'warnings': []
        }


def _validate_field(field_name: str, field_value: Any, field_def: Dict[str, Any]) -> List[str]:
    """Validate a single field against its definition."""
    errors = []
    field_type = field_def.get('type', 'string')
    
    try:
        # Type validation
        if field_type == 'string' and not isinstance(field_value, str):
            errors.append(f"Field '{field_name}' must be a string")
        elif field_type == 'integer' and not isinstance(field_value, int):
            errors.append(f"Field '{field_name}' must be an integer")
        elif field_type == 'date' and not _is_valid_date(field_value):
            errors.append(f"Field '{field_name}' must be a valid date")
        elif field_type == 'email' and not _is_valid_email(field_value):
            errors.append(f"Field '{field_name}' must be a valid email address")
        elif field_type == 'enum' and field_value not in field_def.get('values', []):
            valid_values = ', '.join(field_def.get('values', []))
            errors.append(f"Field '{field_name}' must be one of: {valid_values}")
        
        # Additional validations
        if field_type == 'string':
            max_length = field_def.get('max_length')
            if max_length and len(field_value) > max_length:
                errors.append(f"Field '{field_name}' must not exceed {max_length} characters")
            
            pattern = field_def.get('pattern')
            if pattern and not _matches_pattern(field_value, pattern):
                errors.append(f"Field '{field_name}' does not match required pattern")
        
        elif field_type == 'integer':
            min_val = field_def.get('min')
            max_val = field_def.get('max')
            if min_val is not None and field_value < min_val:
                errors.append(f"Field '{field_name}' must be at least {min_val}")
            if max_val is not None and field_value > max_val:
                errors.append(f"Field '{field_name}' must not exceed {max_val}")
        
        elif field_type == 'object':
            # Recursively validate nested object
            nested_fields = field_def.get('fields', [])
            if nested_fields:
                nested_field_map = {f['name']: f for f in nested_fields}
                if isinstance(field_value, dict):
                    for nested_name, nested_value in field_value.items():
                        if nested_name in nested_field_map:
                            nested_errors = _validate_field(nested_name, nested_value, nested_field_map[nested_name])
                            errors.extend([f"{field_name}.{err}" for err in nested_errors])
                else:
                    errors.append(f"Field '{field_name}' must be an object")
        
        return errors
        
    except Exception as e:
        logger.error(f"Error validating field {field_name}: {e}")
        return [f"Error validating field '{field_name}': {str(e)}"]


def _transform_data(data: Dict[str, Any], schema: Dict[str, Any], options: Dict[str, Any]) -> Dict[str, Any]:
    """Transform data according to schema rules."""
    try:
        # Get transformation rules
        rules = _get_transformation_rules(schema['id'])
        rules.update(options)
        
        transformations = []
        transformed_data = {}
        
        # Apply transformations
        for field_name, field_value in data.items():
            transformed_value = field_value
            
            # String transformations
            if isinstance(field_value, str):
                if rules.get('normalize_case', False):
                    transformed_value = transformed_value.lower()
                    transformations.append(f"normalized_case:{field_name}")
                
                if rules.get('trim_whitespace', True):
                    transformed_value = transformed_value.strip()
                    transformations.append(f"trimmed:{field_name}")
            
            # Date transformations
            if rules.get('format_dates', True) and _is_valid_date(field_value):
                transformed_value = _format_date(field_value)
                transformations.append(f"formatted_date:{field_name}")
            
            transformed_data[field_name] = transformed_value
        
        return {
            'data': transformed_data,
            'transformations': transformations,
            'metadata': {
                'original_fields': len(data),
                'transformed_fields': len(transformed_data),
                'rules_applied': len(transformations)
            }
        }
        
    except Exception as e:
        logger.error(f"Error transforming data: {e}")
        raise StandardizationError(f"Failed to transform data: {str(e)}")


def _get_schema_statistics(redis_client: RedisClient, schema_id: str) -> Dict[str, Any]:
    """Get usage statistics for a schema."""
    try:
        # Generate some placeholder statistics
        stats = {
            'total_validations': 1250,
            'total_transformations': 890,
            'success_rate': 92.5,
            'average_validation_time_ms': 45,
            'average_transformation_time_ms': 120,
            'last_used': '2024-01-15T10:30:00Z',
            'usage_by_day': {
                '2024-01-14': 45,
                '2024-01-13': 38,
                '2024-01-12': 52,
                '2024-01-11': 41,
                '2024-01-10': 39
            }
        }
        
        return stats
        
    except Exception as e:
        logger.error(f"Error retrieving schema statistics for {schema_id}: {e}")
        return {}


# Utility functions
def _is_valid_date(date_string: str) -> bool:
    """Check if a string is a valid date."""
    try:
        datetime.fromisoformat(date_string.replace('Z', '+00:00'))
        return True
    except (ValueError, AttributeError):
        return False


def _is_valid_email(email: str) -> bool:
    """Check if a string is a valid email address."""
    import re
    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    return bool(re.match(pattern, email))


def _matches_pattern(value: str, pattern: str) -> bool:
    """Check if a string matches a regex pattern."""
    import re
    return bool(re.match(pattern, value))


def _format_date(date_string: str) -> str:
    """Format a date string to ISO format."""
    try:
        dt = datetime.fromisoformat(date_string.replace('Z', '+00:00'))
        return dt.isoformat()
    except (ValueError, AttributeError):
        return date_string