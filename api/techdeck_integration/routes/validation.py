"""
Validation API routes for TechDeck-Python Pipeline Integration.

This module implements REST API endpoints for data validation operations,
including quality checks, bias detection, and compliance validation.
"""

import logging
from datetime import datetime
from typing import Dict, Any, Optional, List
from flask import Blueprint, request, jsonify, g
from werkzeug.exceptions import BadRequest, NotFound

from ..utils.validation import validate_validation_request, sanitize_input
from ..utils.logger import get_request_logger
from ..error_handling.custom_errors import (
    ValidationError, ResourceNotFoundError, BiasDetectionError
)
from ..integration.redis_client import RedisClient
from ..auth.decorators import require_auth, require_role

# Initialize blueprint
validation_bp = Blueprint('validation', __name__)
logger = logging.getLogger(__name__)


@validation_bp.route('/quality-check', methods=['POST'])
@require_auth
def quality_check():
    """
    Perform comprehensive data quality validation.
    
    Request Body:
        {
            "data": {...},          # Data to validate
            "dataset_id": "string", # Optional dataset ID for context
            "checks": [...],        # List of quality checks to perform
            "options": {...}        # Optional validation options
        }
        
    Returns:
        Quality validation results including scores and recommendations
        
    Raises:
        ValidationError: If request data is invalid
        ResourceNotFoundError: If dataset is not found
    """
    request_logger = get_request_logger(g.request_id)
    request_logger.info("Performing data quality validation")
    
    try:
        # Get request data
        request_data = request.get_json()
        if not request_data:
            raise ValidationError("Request body is required")
        
        # Validate request structure
        validation_result = validate_validation_request(request_data, 'quality')
        if not validation_result['valid']:
            raise ValidationError(validation_result['errors'])
        
        # Extract data and parameters
        data = request_data.get('data')
        dataset_id = request_data.get('dataset_id')
        checks = request_data.get('checks', ['completeness', 'consistency', 'accuracy'])
        options = request_data.get('options', {})
        
        # Sanitize inputs
        data = sanitize_input(data)
        dataset_id = sanitize_input(dataset_id) if dataset_id else None
        
        request_logger.info(f"Performing quality checks: {checks}")
        
        # Get Redis client from app context
        redis_client = g.app.redis_client
        
        # Retrieve dataset context if provided
        dataset_context = None
        if dataset_id:
            dataset_context = _get_dataset_context(redis_client, dataset_id)
            if not dataset_context:
                raise ResourceNotFoundError(f"Dataset '{dataset_id}' not found")
        
        # Perform quality checks
        quality_results = _perform_quality_checks(data, checks, dataset_context, options)
        
        # Log validation results
        request_logger.info(f"Quality validation completed with score: {quality_results.get('overall_score', 0)}")
        
        return jsonify({
            'success': True,
            'data': {
                'quality_score': quality_results.get('overall_score', 0),
                'check_results': quality_results.get('check_results', {}),
                'recommendations': quality_results.get('recommendations', []),
                'dataset_id': dataset_id,
                'timestamp': datetime.utcnow().isoformat()
            }
        }), 200
        
    except (ValidationError, ResourceNotFoundError):
        raise
    except Exception as e:
        request_logger.error(f"Error performing quality check: {e}")
        raise ValidationError(f"Failed to perform quality check: {str(e)}")


@validation_bp.route('/bias-detection', methods=['POST'])
@require_auth
def bias_detection():
    """
    Perform bias detection analysis on data.
    
    Request Body:
        {
            "data": {...},          # Data to analyze for bias
            "context": {...},       # Context information for bias analysis
            "detection_type": "string", # Type of bias detection (demographic, content, etc.)
            "options": {...}        # Optional detection parameters
        }
        
    Returns:
        Bias detection results including scores and flagged items
        
    Raises:
        ValidationError: If request data is invalid
        BiasDetectionError: If bias detection fails
    """
    request_logger = get_request_logger(g.request_id)
    request_logger.info("Performing bias detection analysis")
    
    try:
        # Get request data
        request_data = request.get_json()
        if not request_data:
            raise ValidationError("Request body is required")
        
        # Validate request structure
        validation_result = validate_validation_request(request_data, 'bias')
        if not validation_result['valid']:
            raise ValidationError(validation_result['errors'])
        
        # Extract data and parameters
        data = request_data.get('data')
        context = request_data.get('context', {})
        detection_type = request_data.get('detection_type', 'demographic')
        options = request_data.get('options', {})
        
        # Sanitize inputs
        data = sanitize_input(data)
        context = sanitize_input(context)
        detection_type = sanitize_input(detection_type)
        
        request_logger.info(f"Performing {detection_type} bias detection")
        
        # Perform bias detection
        bias_results = _perform_bias_detection(data, context, detection_type, options)
        
        # Log detection results
        if bias_results.get('bias_detected', False):
            request_logger.warning(f"Bias detected with score: {bias_results.get('bias_score', 0)}")
        else:
            request_logger.info("No significant bias detected")
        
        return jsonify({
            'success': True,
            'data': {
                'bias_detected': bias_results.get('bias_detected', False),
                'bias_score': bias_results.get('bias_score', 0),
                'bias_types': bias_results.get('bias_types', []),
                'flagged_items': bias_results.get('flagged_items', []),
                'recommendations': bias_results.get('recommendations', []),
                'detection_type': detection_type,
                'timestamp': datetime.utcnow().isoformat()
            }
        }), 200
        
    except (ValidationError, BiasDetectionError):
        raise
    except Exception as e:
        request_logger.error(f"Error performing bias detection: {e}")
        raise BiasDetectionError(f"Failed to perform bias detection: {str(e)}")


@validation_bp.route('/compliance-check', methods=['POST'])
@require_auth
def compliance_check():
    """
    Perform compliance validation against regulatory requirements.
    
    Request Body:
        {
            "data": {...},          # Data to validate for compliance
            "compliance_type": "string", # Type of compliance (HIPAA, GDPR, etc.)
            "options": {...}        # Optional compliance parameters
        }
        
    Returns:
        Compliance validation results including violations and recommendations
        
    Raises:
        ValidationError: If request data is invalid
        ValidationError: If compliance check fails
    """
    request_logger = get_request_logger(g.request_id)
    request_logger.info("Performing compliance validation")
    
    try:
        # Get request data
        request_data = request.get_json()
        if not request_data:
            raise ValidationError("Request body is required")
        
        # Validate request structure
        validation_result = validate_validation_request(request_data, 'compliance')
        if not validation_result['valid']:
            raise ValidationError(validation_result['errors'])
        
        # Extract data and parameters
        data = request_data.get('data')
        compliance_type = request_data.get('compliance_type', 'HIPAA')
        options = request_data.get('options', {})
        
        # Sanitize inputs
        data = sanitize_input(data)
        compliance_type = sanitize_input(compliance_type)
        
        request_logger.info(f"Performing {compliance_type} compliance check")
        
        # Perform compliance validation
        compliance_results = _perform_compliance_check(data, compliance_type, options)
        
        # Log compliance results
        if compliance_results.get('compliant', False):
            request_logger.info(f"{compliance_type} compliance check passed")
        else:
            request_logger.warning(f"{compliance_type} compliance violations found")
        
        return jsonify({
            'success': True,
            'data': {
                'compliant': compliance_results.get('compliant', False),
                'violations': compliance_results.get('violations', []),
                'risk_score': compliance_results.get('risk_score', 0),
                'recommendations': compliance_results.get('recommendations', []),
                'compliance_type': compliance_type,
                'timestamp': datetime.utcnow().isoformat()
            }
        }), 200
        
    except ValidationError:
        raise
    except Exception as e:
        request_logger.error(f"Error performing compliance check: {e}")
        raise ValidationError(f"Failed to perform compliance check: {str(e)}")


@validation_bp.route('/schema-validation', methods=['POST'])
@require_auth
def schema_validation():
    """
    Validate data against a specific schema or data model.
    
    Request Body:
        {
            "data": {...},          # Data to validate
            "schema_id": "string",  # Schema ID to validate against
            "strict_mode": boolean, # Whether to use strict validation
            "options": {...}        # Optional validation parameters
        }
        
    Returns:
        Schema validation results including errors and warnings
        
    Raises:
        ValidationError: If request data is invalid
        ResourceNotFoundError: If schema is not found
    """
    request_logger = get_request_logger(g.request_id)
    request_logger.info("Performing schema validation")
    
    try:
        # Get request data
        request_data = request.get_json()
        if not request_data:
            raise ValidationError("Request body is required")
        
        # Validate request structure
        validation_result = validate_validation_request(request_data, 'schema')
        if not validation_result['valid']:
            raise ValidationError(validation_result['errors'])
        
        # Extract data and parameters
        data = request_data.get('data')
        schema_id = request_data.get('schema_id')
        strict_mode = request_data.get('strict_mode', True)
        options = request_data.get('options', {})
        
        # Sanitize inputs
        data = sanitize_input(data)
        schema_id = sanitize_input(schema_id)
        
        request_logger.info(f"Validating data against schema: {schema_id}")
        
        # Get Redis client from app context
        redis_client = g.app.redis_client
        
        # Retrieve schema
        schema = _get_validation_schema(redis_client, schema_id)
        if not schema:
            raise ResourceNotFoundError(f"Validation schema '{schema_id}' not found")
        
        # Perform schema validation
        validation_results = _validate_against_schema(data, schema, strict_mode, options)
        
        # Log validation results
        if validation_results.get('valid', False):
            request_logger.info(f"Schema validation passed for: {schema_id}")
        else:
            request_logger.warning(f"Schema validation failed for: {schema_id}")
        
        return jsonify({
            'success': True,
            'data': {
                'valid': validation_results.get('valid', False),
                'errors': validation_results.get('errors', []),
                'warnings': validation_results.get('warnings', []),
                'schema_id': schema_id,
                'validation_time_ms': validation_results.get('validation_time_ms', 0),
                'timestamp': datetime.utcnow().isoformat()
            }
        }), 200
        
    except (ValidationError, ResourceNotFoundError):
        raise
    except Exception as e:
        request_logger.error(f"Error performing schema validation: {e}")
        raise ValidationError(f"Failed to perform schema validation: {str(e)}")


@validation_bp.route('/batch-validate', methods=['POST'])
@require_auth
def batch_validate():
    """
    Validate multiple data items in batch.
    
    Request Body:
        {
            "items": [...],         # List of data items to validate
            "validation_type": "string", # Type of validation (quality, bias, schema)
            "options": {...}        # Optional validation parameters
        }
        
    Returns:
        Batch validation results for all items
        
    Raises:
        ValidationError: If request data is invalid
        ValidationError: If batch validation fails
    """
    request_logger = get_request_logger(g.request_id)
    request_logger.info("Performing batch validation")
    
    try:
        # Get request data
        request_data = request.get_json()
        if not request_data:
            raise ValidationError("Request body is required")
        
        # Validate request structure
        if 'items' not in request_data:
            raise ValidationError("Items field is required")
        
        items = request_data.get('items')
        validation_type = request_data.get('validation_type', 'quality')
        options = request_data.get('options', {})
        
        # Validate items is a list
        if not isinstance(items, list):
            raise ValidationError("Items must be a list")
        
        if len(items) == 0:
            raise ValidationError("Items list cannot be empty")
        
        if len(items) > 500:  # Batch size limit
            raise ValidationError("Batch size cannot exceed 500 items")
        
        # Sanitize inputs
        items = [sanitize_input(item) for item in items]
        validation_type = sanitize_input(validation_type)
        
        request_logger.info(f"Batch validating {len(items)} items with type: {validation_type}")
        
        # Process items in batches for performance
        results = []
        batch_size = 50
        
        for i in range(0, len(items), batch_size):
            batch = items[i:i + batch_size]
            batch_results = []
            
            for item in batch:
                try:
                    if validation_type == 'quality':
                        result = _perform_quality_checks(item, ['completeness', 'consistency'], None, options)
                    elif validation_type == 'bias':
                        result = _perform_bias_detection(item, {}, 'demographic', options)
                    elif validation_type == 'schema':
                        # Schema validation requires schema_id, use default
                        result = _validate_against_schema(item, {'fields': []}, True, options)
                    else:
                        raise ValidationError(f"Unsupported validation type: {validation_type}")
                    
                    batch_results.append({
                        'success': True,
                        'data': result,
                        'item_index': i + batch_results.index({
                            'success': True,
                            'data': result
                        })
                    })
                except Exception as e:
                    batch_results.append({
                        'success': False,
                        'error': str(e),
                        'item_index': i + len(batch_results)
                    })
            
            results.extend(batch_results)
        
        # Calculate success rate
        successful = sum(1 for r in results if r['success'])
        success_rate = successful / len(results) * 100
        
        request_logger.info(f"Batch validation completed: {successful}/{len(results)} successful")
        
        return jsonify({
            'success': True,
            'data': {
                'results': results,
                'total_items': len(results),
                'successful_items': successful,
                'failed_items': len(results) - successful,
                'success_rate': success_rate,
                'validation_type': validation_type,
                'timestamp': datetime.utcnow().isoformat()
            }
        }), 200
        
    except ValidationError:
        raise
    except Exception as e:
        request_logger.error(f"Error in batch validation: {e}")
        raise ValidationError(f"Failed to perform batch validation: {str(e)}")


@validation_bp.route('/validation-history', methods=['GET'])
@require_auth
def get_validation_history():
    """
    Get validation history and statistics.
    
    Query Parameters:
        dataset_id (str): Optional dataset ID to filter by
        validation_type (str): Optional validation type to filter by
        start_date (str): Optional start date for filtering
        end_date (str): Optional end date for filtering
        limit (int): Maximum number of results to return
        
    Returns:
        Validation history with statistics and trends
        
    Raises:
        ValidationError: If query parameters are invalid
    """
    request_logger = get_request_logger(g.request_id)
    request_logger.info("Retrieving validation history")
    
    try:
        # Get query parameters
        dataset_id = request.args.get('dataset_id')
        validation_type = request.args.get('validation_type')
        start_date = request.args.get('start_date')
        end_date = request.args.get('end_date')
        limit = min(int(request.args.get('limit', 100)), 1000)  # Max 1000
        
        # Sanitize inputs
        dataset_id = sanitize_input(dataset_id) if dataset_id else None
        validation_type = sanitize_input(validation_type) if validation_type else None
        
        request_logger.info(f"Retrieving validation history with filters: dataset={dataset_id}, type={validation_type}")
        
        # Get Redis client from app context
        redis_client = g.app.redis_client
        
        # Retrieve validation history
        history = _get_validation_history(redis_client, dataset_id, validation_type, start_date, end_date, limit)
        
        # Log successful retrieval
        request_logger.info(f"Retrieved {len(history.get('validations', []))} validation records")
        
        return jsonify({
            'success': True,
            'data': history,
            'timestamp': datetime.utcnow().isoformat()
        }), 200
        
    except ValueError as e:
        raise ValidationError(f"Invalid query parameter: {str(e)}")
    except Exception as e:
        request_logger.error(f"Error retrieving validation history: {e}")
        raise ValidationError(f"Failed to retrieve validation history: {str(e)}")


# Helper Functions
def _perform_quality_checks(data: Dict[str, Any], checks: List[str], dataset_context: Optional[Dict[str, Any]], options: Dict[str, Any]) -> Dict[str, Any]:
    """Perform comprehensive quality checks on data."""
    try:
        check_results = {}
        recommendations = []
        total_score = 0
        check_count = 0
        
        # Completeness check
        if 'completeness' in checks:
            completeness_score, completeness_issues = _check_completeness(data, dataset_context)
            check_results['completeness'] = {
                'score': completeness_score,
                'issues': completeness_issues,
                'passed': completeness_score >= 0.8
            }
            total_score += completeness_score
            check_count += 1
            
            if completeness_score < 0.8:
                recommendations.append("Improve data completeness by filling missing required fields")
        
        # Consistency check
        if 'consistency' in checks:
            consistency_score, consistency_issues = _check_consistency(data, dataset_context)
            check_results['consistency'] = {
                'score': consistency_score,
                'issues': consistency_issues,
                'passed': consistency_score >= 0.8
            }
            total_score += consistency_score
            check_count += 1
            
            if consistency_score < 0.8:
                recommendations.append("Review data consistency issues and standardize formats")
        
        # Accuracy check
        if 'accuracy' in checks:
            accuracy_score, accuracy_issues = _check_accuracy(data, dataset_context)
            check_results['accuracy'] = {
                'score': accuracy_score,
                'issues': accuracy_issues,
                'passed': accuracy_score >= 0.8
            }
            total_score += accuracy_score
            check_count += 1
            
            if accuracy_score < 0.8:
                recommendations.append("Validate data accuracy against known standards")
        
        # Calculate overall score
        overall_score = total_score / check_count if check_count > 0 else 0
        
        return {
            'overall_score': overall_score,
            'check_results': check_results,
            'recommendations': recommendations
        }
        
    except Exception as e:
        logger.error(f"Error performing quality checks: {e}")
        raise ValidationError(f"Quality check failed: {str(e)}")


def _perform_bias_detection(data: Dict[str, Any], context: Dict[str, Any], detection_type: str, options: Dict[str, Any]) -> Dict[str, Any]:
    """Perform bias detection analysis."""
    try:
        # Placeholder bias detection logic
        bias_detected = False
        bias_score = 0.0
        bias_types = []
        flagged_items = []
        recommendations = []
        
        if detection_type == 'demographic':
            # Check for demographic bias
            bias_score = 0.15  # Placeholder score
            if bias_score > 0.1:
                bias_detected = True
                bias_types.append('demographic_representation')
                flagged_items.append({
                    'field': 'demographics',
                    'issue': 'Potential underrepresentation of certain demographic groups',
                    'severity': 'medium'
                })
                recommendations.append("Ensure balanced representation across demographic groups")
        
        elif detection_type == 'content':
            # Check for content bias
            bias_score = 0.08  # Placeholder score
            if bias_score > 0.05:
                bias_detected = True
                bias_types.append('content_bias')
                flagged_items.append({
                    'field': 'content',
                    'issue': 'Potential content bias detected',
                    'severity': 'low'
                })
                recommendations.append("Review content for potential bias indicators")
        
        return {
            'bias_detected': bias_detected,
            'bias_score': bias_score,
            'bias_types': bias_types,
            'flagged_items': flagged_items,
            'recommendations': recommendations
        }
        
    except Exception as e:
        logger.error(f"Error performing bias detection: {e}")
        raise BiasDetectionError(f"Bias detection failed: {str(e)}")


def _perform_compliance_check(data: Dict[str, Any], compliance_type: str, options: Dict[str, Any]) -> Dict[str, Any]:
    """Perform compliance validation."""
    try:
        violations = []
        risk_score = 0
        recommendations = []
        
        if compliance_type == 'HIPAA':
            # HIPAA compliance checks
            if 'patient_name' in data or 'ssn' in data:
                violations.append({
                    'type': 'phi_exposure',
                    'severity': 'high',
                    'description': 'Protected Health Information (PHI) detected in data'
                })
                risk_score += 50
                recommendations.append("Remove or encrypt all PHI before processing")
            
            if 'email' in data and not options.get('encrypted_email', False):
                violations.append({
                    'type': 'unencrypted_pii',
                    'severity': 'medium',
                    'description': 'Email addresses should be encrypted'
                })
                risk_score += 20
                recommendations.append("Encrypt email addresses using approved methods")
        
        elif compliance_type == 'GDPR':
            # GDPR compliance checks
            if 'personal_data' in data and not options.get('consent_verified', False):
                violations.append({
                    'type': 'missing_consent',
                    'severity': 'high',
                    'description': 'Personal data processing without verified consent'
                })
                risk_score += 40
                recommendations.append("Verify data subject consent before processing")
        
        # Calculate compliance status
        compliant = len(violations) == 0 or risk_score < 30
        
        return {
            'compliant': compliant,
            'violations': violations,
            'risk_score': risk_score,
            'recommendations': recommendations
        }
        
    except Exception as e:
        logger.error(f"Error performing compliance check: {e}")
        raise ValidationError(f"Compliance check failed: {str(e)}")


def _get_validation_schema(redis_client: RedisClient, schema_id: str) -> Optional[Dict[str, Any]]:
    """Retrieve validation schema details."""
    try:
        # Try to get from Redis cache first
        cache_key = f'validation:schema:{schema_id}'
        cached_schema = redis_client.get(cache_key)
        if cached_schema:
            return cached_schema
        
        # Placeholder schema definitions
        schemas = {
            'healthcare-patient': {
                'id': 'healthcare-patient',
                'name': 'Healthcare Patient Data',
                'fields': [
                    {'name': 'patient_id', 'type': 'string', 'required': True},
                    {'name': 'age', 'type': 'integer', 'required': True, 'min': 0, 'max': 150},
                    {'name': 'gender', 'type': 'enum', 'required': True, 'values': ['M', 'F', 'O']}
                ]
            },
            'clinical-trial': {
                'id': 'clinical-trial',
                'name': 'Clinical Trial Data',
                'fields': [
                    {'name': 'trial_id', 'type': 'string', 'required': True},
                    {'name': 'participant_id', 'type': 'string', 'required': True},
                    {'name': 'enrollment_date', 'type': 'date', 'required': True}
                ]
            }
        }
        
        schema = schemas.get(schema_id)
        if schema:
            # Cache for 30 minutes
            redis_client.setex(cache_key, 1800, schema)
        
        return schema
        
    except Exception as e:
        logger.error(f"Error retrieving validation schema {schema_id}: {e}")
        return None


def _validate_against_schema(data: Dict[str, Any], schema: Dict[str, Any], strict_mode: bool, options: Dict[str, Any]) -> Dict[str, Any]:
    """Validate data against a specific schema."""
    start_time = datetime.utcnow()
    
    try:
        errors = []
        warnings = []
        
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
                if strict_mode:
                    errors.append(f"Unknown field '{field_name}' not in schema")
                else:
                    warnings.append(f"Extra field '{field_name}' not in schema")
                continue
            
            field_def = field_map[field_name]
            field_errors = _validate_field_against_schema(field_name, field_value, field_def)
            errors.extend(field_errors)
        
        validation_time = (datetime.utcnow() - start_time).total_seconds() * 1000
        
        return {
            'valid': len(errors) == 0,
            'errors': errors,
            'warnings': warnings,
            'validation_time_ms': validation_time
        }
        
    except Exception as e:
        logger.error(f"Error validating against schema: {e}")
        return {
            'valid': False,
            'errors': [f"Schema validation error: {str(e)}"],
            'warnings': [],
            'validation_time_ms': (datetime.utcnow() - start_time).total_seconds() * 1000
        }


def _validate_field_against_schema(field_name: str, field_value: Any, field_def: Dict[str, Any]) -> List[str]:
    """Validate a single field against schema definition."""
    errors = []
    field_type = field_def.get('type', 'string')
    
    try:
        # Type validation
        if field_type == 'string' and not isinstance(field_value, str):
            errors.append(f"Field '{field_name}' must be a string")
        elif field_type == 'integer' and not isinstance(field_value, int):
            errors.append(f"Field '{field_name}' must be an integer")
        elif field_type == 'date' and not _is_valid_date_format(field_value):
            errors.append(f"Field '{field_name}' must be a valid date")
        elif field_type == 'enum' and field_value not in field_def.get('values', []):
            valid_values = ', '.join(field_def.get('values', []))
            errors.append(f"Field '{field_name}' must be one of: {valid_values}")
        
        # Additional validations
        if field_type == 'string':
            max_length = field_def.get('max_length')
            if max_length and len(field_value) > max_length:
                errors.append(f"Field '{field_name}' must not exceed {max_length} characters")
        
        elif field_type == 'integer':
            min_val = field_def.get('min')
            max_val = field_def.get('max')
            if min_val is not None and field_value < min_val:
                errors.append(f"Field '{field_name}' must be at least {min_val}")
            if max_val is not None and field_value > max_val:
                errors.append(f"Field '{field_name}' must not exceed {max_val}")
        
        return errors
        
    except Exception as e:
        logger.error(f"Error validating field {field_name}: {e}")
        return [f"Error validating field '{field_name}': {str(e)}"]


def _get_validation_history(redis_client: RedisClient, dataset_id: Optional[str], validation_type: Optional[str], start_date: Optional[str], end_date: Optional[str], limit: int) -> Dict[str, Any]:
    """Retrieve validation history with optional filters."""
    try:
        # Placeholder validation history
        history = {
            'validations': [
                {
                    'id': 'val_001',
                    'dataset_id': 'dataset_123',
                    'validation_type': 'quality',
                    'status': 'passed',
                    'score': 0.92,
                    'timestamp': '2024-01-15T10:30:00Z'
                },
                {
                    'id': 'val_002',
                    'dataset_id': 'dataset_456',
                    'validation_type': 'bias',
                    'status': 'warning',
                    'score': 0.15,
                    'timestamp': '2024-01-14T14:20:00Z'
                }
            ],
            'statistics': {
                'total_validations': 245,
                'passed': 198,
                'failed': 32,
                'warnings': 15,
                'average_score': 0.87
            }
        }
        
        # Apply filters if provided
        if dataset_id:
            history['validations'] = [v for v in history['validations'] if v.get('dataset_id') == dataset_id]
        
        if validation_type:
            history['validations'] = [v for v in history['validations'] if v.get('validation_type') == validation_type]
        
        # Limit results
        history['validations'] = history['validations'][:limit]
        
        return history
        
    except Exception as e:
        logger.error(f"Error retrieving validation history: {e}")
        return {'validations': [], 'statistics': {}}


# Utility functions for quality checks
def _check_completeness(data: Dict[str, Any], dataset_context: Optional[Dict[str, Any]]) -> tuple:
    """Check data completeness."""
    if not data:
        return 0.0, ["No data provided"]
    
    total_fields = len(data)
    non_empty_fields = sum(1 for value in data.values() if value is not None and value != '')
    
    completeness_score = non_empty_fields / total_fields if total_fields > 0 else 0.0
    issues = []
    
    if completeness_score < 0.8:
        issues.append(f"Data completeness is {completeness_score:.1%}, below recommended 80%")
    
    return completeness_score, issues


def _check_consistency(data: Dict[str, Any], dataset_context: Optional[Dict[str, Any]]) -> tuple:
    """Check data consistency."""
    consistency_score = 0.95  # Placeholder score
    issues = []
    
    # Check for data type consistency
    for key, value in data.items():
        if isinstance(value, str) and value.isdigit():
            issues.append(f"Field '{key}' contains numeric string, consider using integer type")
            consistency_score -= 0.1
    
    return max(0.0, consistency_score), issues


def _check_accuracy(data: Dict[str, Any], dataset_context: Optional[Dict[str, Any]]) -> tuple:
    """Check data accuracy."""
    accuracy_score = 0.90  # Placeholder score
    issues = []
    
    # Basic accuracy checks
    if 'age' in data and isinstance(data['age'], (int, float)):
        if data['age'] < 0 or data['age'] > 150:
            issues.append("Age value appears to be outside reasonable range")
            accuracy_score -= 0.2
    
    return max(0.0, accuracy_score), issues


def _is_valid_date_format(date_string: str) -> bool:
    """Check if a string is in valid date format."""
    try:
        datetime.fromisoformat(date_string.replace('Z', '+00:00'))
        return True
    except (ValueError, AttributeError):
        return False