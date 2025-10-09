#!/usr/bin/env python3
"""
Distributed Quality Validation System for Pixelated Empathy AI
Parallelizes quality validation across multiple workers for high-performance processing
"""

import os
import sys
import json
import asyncio
import logging
from typing import Dict, List, Any, Optional, Callable, Tuple
from pathlib import Path
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
import multiprocessing as mp
from queue import Queue, Empty
import time
import hashlib
import pickle
from enum import Enum

# Redis for distributed task queue
try:
    import redis
    import redis.exceptions
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False
    logging.warning("Redis not available - using local processing only")

# Celery for distributed task processing
try:
    from celery import Celery, Task
    from celery.result import AsyncResult
    CELERY_AVAILABLE = True
except ImportError:
    CELERY_AVAILABLE = False
    logging.warning("Celery not available - using local processing only")

# Import caching system
try:
    from .quality_validation_cache import QualityValidationCache, CachedQualityValidator
    CACHING_AVAILABLE = True
    logging.info("Caching system available")
except ImportError as e:
    CACHING_AVAILABLE = False
    logging.warning(f"Caching system not available: {e}")
except Exception as e:
    CACHING_AVAILABLE = False
    logging.warning(f"Caching system not available due to error: {e}")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ValidationStatus(Enum):
    """Validation status enumeration"""
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class ValidationPriority(Enum):
    """Validation priority levels"""
    LOW = 1
    NORMAL = 2
    HIGH = 3
    CRITICAL = 4


@dataclass
class ValidationTask:
    """Represents a quality validation task"""
    task_id: str
    data_path: str
    validation_type: str
    priority: ValidationPriority
    metadata: Dict[str, Any]
    created_at: str
    status: ValidationStatus = ValidationStatus.PENDING
    worker_id: Optional[str] = None
    started_at: Optional[str] = None
    completed_at: Optional[str] = None
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        data = asdict(self)
        data['priority'] = self.priority.value
        data['status'] = self.status.value
        return data
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ValidationTask':
        """Create from dictionary"""
        data['priority'] = ValidationPriority(data['priority'])
        data['status'] = ValidationStatus(data['status'])
        return cls(**data)


@dataclass
class ValidationResult:
    """Represents validation result"""
    task_id: str
    success: bool
    quality_score: float
    metrics: Dict[str, float]
    issues: List[Dict[str, Any]]
    processing_time: float
    worker_id: str
    timestamp: str
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return asdict(self)


class QualityValidator:
    """Base quality validator class"""
    
    def __init__(self, validator_type: str):
        self.validator_type = validator_type
        self.worker_id = f"{validator_type}_{os.getpid()}_{int(time.time())}"
    
    def validate(self, data_path: str, metadata: Dict[str, Any]) -> ValidationResult:
        """Validate data quality - to be implemented by subclasses"""
        raise NotImplementedError("Subclasses must implement validate method")
    
    def _calculate_quality_score(self, metrics: Dict[str, float]) -> float:
        """Calculate overall quality score from metrics"""
        if not metrics:
            return 0.0
        
        # Weighted average of metrics
        weights = {
            'completeness': 0.3,
            'accuracy': 0.3,
            'consistency': 0.2,
            'relevance': 0.2
        }
        
        total_score = 0.0
        total_weight = 0.0
        
        for metric, value in metrics.items():
            weight = weights.get(metric, 0.1)
            total_score += value * weight
            total_weight += weight
        
        return total_score / total_weight if total_weight > 0 else 0.0


class ConversationQualityValidator(QualityValidator):
    """Validates conversation quality"""
    
    def __init__(self):
        super().__init__("conversation")
    
    def validate(self, data_path: str, metadata: Dict[str, Any]) -> ValidationResult:
        """Validate conversation quality"""
        start_time = time.time()
        
        try:
            # Load conversation data
            with open(data_path, 'r', encoding='utf-8') as f:
                if data_path.endswith('.json'):
                    data = json.load(f)
                else:
                    # Assume JSONL format
                    data = [json.loads(line) for line in f if line.strip()]
            
            # Perform quality checks
            metrics = {}
            issues = []
            
            # Check completeness
            completeness_score = self._check_completeness(data, issues)
            metrics['completeness'] = completeness_score
            
            # Check accuracy
            accuracy_score = self._check_accuracy(data, issues)
            metrics['accuracy'] = accuracy_score
            
            # Check consistency
            consistency_score = self._check_consistency(data, issues)
            metrics['consistency'] = consistency_score
            
            # Check relevance
            relevance_score = self._check_relevance(data, issues)
            metrics['relevance'] = relevance_score
            
            # Calculate overall quality score
            quality_score = self._calculate_quality_score(metrics)
            
            processing_time = time.time() - start_time
            
            return ValidationResult(
                task_id=metadata.get('task_id', ''),
                success=True,
                quality_score=quality_score,
                metrics=metrics,
                issues=issues,
                processing_time=processing_time,
                worker_id=self.worker_id,
                timestamp=datetime.now(timezone.utc).isoformat()
            )
            
        except Exception as e:
            processing_time = time.time() - start_time
            logger.error(f"Validation failed for {data_path}: {e}")
            
            return ValidationResult(
                task_id=metadata.get('task_id', ''),
                success=False,
                quality_score=0.0,
                metrics={},
                issues=[{'type': 'error', 'message': str(e)}],
                processing_time=processing_time,
                worker_id=self.worker_id,
                timestamp=datetime.now(timezone.utc).isoformat()
            )
    
    def _check_completeness(self, data: List[Dict], issues: List[Dict]) -> float:
        """Check data completeness"""
        if not data:
            issues.append({
                'type': 'completeness',
                'severity': 'critical',
                'message': 'No data found'
            })
            return 0.0
        
        required_fields = ['input', 'output', 'context']
        total_records = len(data)
        complete_records = 0
        
        for i, record in enumerate(data):
            missing_fields = [field for field in required_fields if not record.get(field)]
            
            if not missing_fields:
                complete_records += 1
            else:
                issues.append({
                    'type': 'completeness',
                    'severity': 'warning',
                    'message': f'Record {i}: Missing fields {missing_fields}',
                    'record_index': i
                })
        
        return complete_records / total_records if total_records > 0 else 0.0
    
    def _check_accuracy(self, data: List[Dict], issues: List[Dict]) -> float:
        """Check data accuracy"""
        if not data:
            return 0.0
        
        accurate_records = 0
        total_records = len(data)
        
        for i, record in enumerate(data):
            accuracy_issues = []
            
            # Check input length
            input_text = record.get('input', '')
            if len(input_text) < 10:
                accuracy_issues.append('Input too short')
            elif len(input_text) > 5000:
                accuracy_issues.append('Input too long')
            
            # Check output quality
            output_text = record.get('output', '')
            if len(output_text) < 5:
                accuracy_issues.append('Output too short')
            elif len(output_text) > 2000:
                accuracy_issues.append('Output too long')
            
            # Check for placeholder text
            placeholders = ['[PLACEHOLDER]', 'TODO', 'FIXME', 'XXX']
            for placeholder in placeholders:
                if placeholder in input_text or placeholder in output_text:
                    accuracy_issues.append(f'Contains placeholder: {placeholder}')
            
            if not accuracy_issues:
                accurate_records += 1
            else:
                issues.append({
                    'type': 'accuracy',
                    'severity': 'warning',
                    'message': f'Record {i}: {", ".join(accuracy_issues)}',
                    'record_index': i
                })
        
        return accurate_records / total_records if total_records > 0 else 0.0
    
    def _check_consistency(self, data: List[Dict], issues: List[Dict]) -> float:
        """Check data consistency"""
        if not data:
            return 0.0
        
        # Check format consistency
        format_consistency = self._check_format_consistency(data, issues)
        
        # Check content consistency
        content_consistency = self._check_content_consistency(data, issues)
        
        return (format_consistency + content_consistency) / 2
    
    def _check_format_consistency(self, data: List[Dict], issues: List[Dict]) -> float:
        """Check format consistency across records"""
        if not data:
            return 0.0
        
        # Get field structure from first record
        reference_fields = set(data[0].keys())
        consistent_records = 1  # First record is consistent by definition
        
        for i, record in enumerate(data[1:], 1):
            record_fields = set(record.keys())
            
            if record_fields != reference_fields:
                missing = reference_fields - record_fields
                extra = record_fields - reference_fields
                
                issue_parts = []
                if missing:
                    issue_parts.append(f'Missing: {missing}')
                if extra:
                    issue_parts.append(f'Extra: {extra}')
                
                issues.append({
                    'type': 'consistency',
                    'severity': 'warning',
                    'message': f'Record {i}: Field mismatch - {", ".join(issue_parts)}',
                    'record_index': i
                })
            else:
                consistent_records += 1
        
        return consistent_records / len(data)
    
    def _check_content_consistency(self, data: List[Dict], issues: List[Dict]) -> float:
        """Check content consistency"""
        if len(data) < 2:
            return 1.0
        
        # Check language consistency
        languages = set()
        for record in data:
            # Simple language detection based on character patterns
            text = record.get('input', '') + ' ' + record.get('output', '')
            if any(ord(c) > 127 for c in text):
                languages.add('non-english')
            else:
                languages.add('english')
        
        if len(languages) > 1:
            issues.append({
                'type': 'consistency',
                'severity': 'info',
                'message': f'Multiple languages detected: {languages}'
            })
        
        # For now, return high consistency if only one language
        return 0.9 if len(languages) == 1 else 0.7
    
    def _check_relevance(self, data: List[Dict], issues: List[Dict]) -> float:
        """Check data relevance"""
        if not data:
            return 0.0
        
        relevant_records = 0
        total_records = len(data)
        
        # Keywords that indicate mental health/empathy relevance
        relevant_keywords = [
            'feel', 'emotion', 'support', 'help', 'understand', 'listen',
            'anxiety', 'depression', 'stress', 'mental', 'health', 'therapy',
            'counseling', 'empathy', 'compassion', 'care', 'comfort'
        ]
        
        for i, record in enumerate(data):
            text = (record.get('input', '') + ' ' + record.get('output', '')).lower()
            
            # Count relevant keywords
            keyword_count = sum(1 for keyword in relevant_keywords if keyword in text)
            
            if keyword_count >= 2:  # At least 2 relevant keywords
                relevant_records += 1
            else:
                issues.append({
                    'type': 'relevance',
                    'severity': 'info',
                    'message': f'Record {i}: Low relevance (keywords: {keyword_count})',
                    'record_index': i
                })
        
        return relevant_records / total_records if total_records > 0 else 0.0


class DistributedQualityValidator:
    """Distributed quality validation coordinator"""
    
    def __init__(self, redis_url: str = None, celery_broker: str = None, enable_caching: bool = True):
        self.redis_url = redis_url or os.getenv('REDIS_URL', 'redis://localhost:6379/0')
        self.celery_broker = celery_broker or os.getenv('CELERY_BROKER', self.redis_url)
        self.enable_caching = enable_caching
        
        # Initialize Redis connection
        self.redis_client = None
        if REDIS_AVAILABLE:
            try:
                self.redis_client = redis.from_url(self.redis_url)
                self.redis_client.ping()
                logger.info("Connected to Redis for task queue")
            except Exception as e:
                logger.warning(f"Could not connect to Redis: {e}")
                self.redis_client = None
        
        # Initialize Celery app
        self.celery_app = None
        if CELERY_AVAILABLE:
            try:
                self.celery_app = Celery('quality_validator', broker=self.celery_broker)
                self.celery_app.conf.update(
                    task_serializer='pickle',
                    accept_content=['pickle'],
                    result_serializer='pickle',
                    timezone='UTC',
                    enable_utc=True,
                    task_routes={
                        'quality_validator.validate_task': {'queue': 'quality_validation'}
                    }
                )
                logger.info("Initialized Celery for distributed processing")
            except Exception as e:
                logger.warning(f"Could not initialize Celery: {e}")
                self.celery_app = None
        
        # Initialize caching system
        self.cache = None
        self.cached_validator = None
        if CACHING_AVAILABLE and self.enable_caching:
            try:
                self.cache = QualityValidationCache(self.redis_url)
                self.cached_validator = CachedQualityValidator(self.cache)
                logger.info("Initialized quality validation caching system")
            except Exception as e:
                logger.warning(f"Could not initialize caching system: {e}")
                self.cache = None
                self.cached_validator = None
        
        # Task storage
        self.tasks: Dict[str, ValidationTask] = {}
        self.results: Dict[str, ValidationResult] = {}
        
        # Local processing fallback
        self.local_executor = ProcessPoolExecutor(max_workers=mp.cpu_count())
        
        # Validators
        self.validators = {
            'conversation': ConversationQualityValidator()
        }
    
    def submit_validation_task(self, data_path: str, validation_type: str,
                             priority: ValidationPriority = ValidationPriority.NORMAL,
                             metadata: Dict[str, Any] = None) -> str:
        """Submit a validation task for processing"""
        task_id = self._generate_task_id(data_path)
        
        # Check cache if enabled
        if self.cached_validator:
            cache_hit, cached_result = self.cached_validator.validate_with_cache(
                data_path, validation_type, metadata
            )
            if cache_hit and cached_result:
                # Deserialize cached result
                try:
                    result_data = pickle.loads(cached_result)
                    result = ValidationResult.from_dict(result_data)
                    result.task_id = task_id
                    
                    # Store result
                    self.results[task_id] = result
                    
                    # Update task status
                    task = ValidationTask(
                        task_id=task_id,
                        data_path=data_path,
                        validation_type=validation_type,
                        priority=priority,
                        metadata=metadata or {},
                        created_at=datetime.now(timezone.utc).isoformat(),
                        status=ValidationStatus.COMPLETED,
                        completed_at=datetime.now(timezone.utc).isoformat(),
                        result=result_data
                    )
                    self.tasks[task_id] = task
                    
                    logger.info(f"Cached validation result found for task: {task_id}")
                    return task_id
                except Exception as e:
                    logger.warning(f"Failed to deserialize cached result: {e}")
        
        task = ValidationTask(
            task_id=task_id,
            data_path=data_path,
            validation_type=validation_type,
            priority=priority,
            metadata=metadata or {},
            created_at=datetime.now(timezone.utc).isoformat()
        )
        
        # Store task
        self.tasks[task_id] = task
        
        # Submit for processing
        if self.celery_app:
            # Use Celery for distributed processing
            celery_task = self.celery_app.send_task(
                'quality_validator.validate_task',
                args=[task.to_dict()],
                queue='quality_validation',
                priority=priority.value
            )
            task.worker_id = celery_task.id
        else:
            # Use local processing
            future = self.local_executor.submit(self._process_task_local, task)
            task.worker_id = str(id(future))
        
        task.status = ValidationStatus.PROCESSING
        task.started_at = datetime.now(timezone.utc).isoformat()
        
        logger.info(f"Submitted validation task: {task_id}")
        return task_id
    
    def get_task_status(self, task_id: str) -> Optional[ValidationTask]:
        """Get task status"""
        return self.tasks.get(task_id)
    
    def get_task_result(self, task_id: str) -> Optional[ValidationResult]:
        """Get task result"""
        return self.results.get(task_id)
    
    def wait_for_task(self, task_id: str, timeout: float = None) -> Optional[ValidationResult]:
        """Wait for task completion"""
        start_time = time.time()
        
        while True:
            result = self.get_task_result(task_id)
            if result:
                return result
            
            task = self.get_task_status(task_id)
            if task and task.status in [ValidationStatus.FAILED, ValidationStatus.CANCELLED]:
                return None
            
            if timeout and (time.time() - start_time) > timeout:
                logger.warning(f"Task {task_id} timed out after {timeout} seconds")
                return None
            
            time.sleep(0.1)
    
    def batch_validate(self, data_paths: List[str], validation_type: str,
                      batch_size: int = 10, max_workers: int = None) -> List[ValidationResult]:
        """Validate multiple files in batches"""
        if not max_workers:
            max_workers = min(len(data_paths), mp.cpu_count() * 2)
        
        logger.info(f"Starting batch validation of {len(data_paths)} files with {max_workers} workers")
        
        # Submit all tasks
        task_ids = []
        for i, data_path in enumerate(data_paths):
            priority = ValidationPriority.HIGH if i < batch_size else ValidationPriority.NORMAL
            task_id = self.submit_validation_task(data_path, validation_type, priority)
            task_ids.append(task_id)
        
        # Wait for all tasks to complete
        results = []
        completed = 0
        
        while completed < len(task_ids):
            for task_id in task_ids:
                if task_id not in [r.task_id for r in results]:
                    result = self.get_task_result(task_id)
                    if result:
                        results.append(result)
                        completed += 1
                        
                        if completed % 10 == 0:
                            logger.info(f"Completed {completed}/{len(task_ids)} validation tasks")
            
            time.sleep(0.5)
        
        logger.info(f"Batch validation completed: {len(results)} results")
        return results
    
    def get_validation_statistics(self) -> Dict[str, Any]:
        """Get validation statistics"""
        total_tasks = len(self.tasks)
        completed_tasks = len(self.results)
        
        if completed_tasks == 0:
            return {
                'total_tasks': total_tasks,
                'completed_tasks': 0,
                'success_rate': 0.0,
                'average_quality_score': 0.0,
                'average_processing_time': 0.0
            }
        
        successful_results = [r for r in self.results.values() if r.success]
        success_rate = len(successful_results) / completed_tasks
        
        if successful_results:
            avg_quality_score = sum(r.quality_score for r in successful_results) / len(successful_results)
            avg_processing_time = sum(r.processing_time for r in self.results.values()) / len(self.results.values())
        else:
            avg_quality_score = 0.0
            avg_processing_time = 0.0
        
        return {
            'total_tasks': total_tasks,
            'completed_tasks': completed_tasks,
            'success_rate': success_rate,
            'average_quality_score': avg_quality_score,
            'average_processing_time': avg_processing_time,
            'quality_distribution': self._get_quality_distribution()
        }
    
    def _get_quality_distribution(self) -> Dict[str, int]:
        """Get quality score distribution"""
        distribution = {
            'excellent': 0,  # 0.9-1.0
            'good': 0,       # 0.7-0.9
            'fair': 0,       # 0.5-0.7
            'poor': 0        # 0.0-0.5
        }
        
        for result in self.results.values():
            if not result.success:
                continue
                
            score = result.quality_score
            if score >= 0.9:
                distribution['excellent'] += 1
            elif score >= 0.7:
                distribution['good'] += 1
            elif score >= 0.5:
                distribution['fair'] += 1
            else:
                distribution['poor'] += 1
        
        return distribution
    
    def _process_task_local(self, task: ValidationTask) -> ValidationResult:
        """Process task locally"""
        try:
            validator = self.validators.get(task.validation_type)
            if not validator:
                raise ValueError(f"Unknown validation type: {task.validation_type}")
            
            result = validator.validate(task.data_path, task.metadata)
            result.task_id = task.task_id
            
            # Cache result if enabled
            if self.cached_validator and self.cache:
                try:
                    # Serialize result for caching
                    result_data = asdict(result)
                    serialized_result = pickle.dumps(result_data)
                    
                    # Cache the result
                    self.cached_validator.cache_validation_result(
                        task.data_path, 
                        task.validation_type, 
                        task.metadata, 
                        serialized_result
                    )
                    logger.debug(f"Cached validation result for task: {task.task_id}")
                except Exception as e:
                    logger.warning(f"Failed to cache validation result: {e}")
            
            # Store result
            self.results[task.task_id] = result
            
            # Update task status
            task.status = ValidationStatus.COMPLETED
            task.completed_at = datetime.now(timezone.utc).isoformat()
            
            return result
            
        except Exception as e:
            logger.error(f"Local task processing failed: {e}")
            
            # Create error result
            result = ValidationResult(
                task_id=task.task_id,
                success=False,
                quality_score=0.0,
                metrics={},
                issues=[{'type': 'error', 'message': str(e)}],
                processing_time=0.0,
                worker_id='local',
                timestamp=datetime.now(timezone.utc).isoformat()
            )
            
            self.results[task.task_id] = result
            task.status = ValidationStatus.FAILED
            task.error = str(e)
            
            return result
    
    def _generate_task_id(self, data_path: str) -> str:
        """Generate unique task ID"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        path_hash = hashlib.md5(data_path.encode()).hexdigest()[:8]
        return f"val_{timestamp}_{path_hash}"
    
    def cleanup(self):
        """Cleanup resources"""
        if self.local_executor:
            self.local_executor.shutdown(wait=True)
        
        if self.redis_client:
            self.redis_client.close()
        
        if self.cache:
            self.cache.close()


# Celery task for distributed processing
if CELERY_AVAILABLE:
    @Celery.task(bind=True)
    def validate_task(self, task_data: Dict[str, Any]) -> Dict[str, Any]:
        """Celery task for validation"""
        task = ValidationTask.from_dict(task_data)
        
        # Create validator
        validator_type = task.validation_type
        if validator_type == 'conversation':
            validator = ConversationQualityValidator()
        else:
            raise ValueError(f"Unknown validation type: {validator_type}")
        
        # Process validation
        result = validator.validate(task.data_path, task.metadata)
        result.task_id = task.task_id
        
        return result.to_dict()


def main():
    """Main CLI interface"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Distributed Quality Validation System")
    parser.add_argument('--redis-url', help="Redis URL for task queue")
    parser.add_argument('--celery-broker', help="Celery broker URL")
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Validate command
    validate_parser = subparsers.add_parser('validate', help='Validate a single file')
    validate_parser.add_argument('data_path', help='Path to data file')
    validate_parser.add_argument('--type', default='conversation', help='Validation type')
    validate_parser.add_argument('--priority', default='normal', choices=['low', 'normal', 'high', 'critical'])
    
    # Batch validate command
    batch_parser = subparsers.add_parser('batch', help='Validate multiple files')
    batch_parser.add_argument('data_dir', help='Directory containing data files')
    batch_parser.add_argument('--type', default='conversation', help='Validation type')
    batch_parser.add_argument('--pattern', default='*.json', help='File pattern to match')
    batch_parser.add_argument('--max-workers', type=int, help='Maximum number of workers')
    
    # Status command
    status_parser = subparsers.add_parser('status', help='Show validation statistics')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    # Create validator
    validator = DistributedQualityValidator(args.redis_url, args.celery_broker)
    
    try:
        if args.command == 'validate':
            priority_map = {
                'low': ValidationPriority.LOW,
                'normal': ValidationPriority.NORMAL,
                'high': ValidationPriority.HIGH,
                'critical': ValidationPriority.CRITICAL
            }
            
            task_id = validator.submit_validation_task(
                args.data_path,
                args.type,
                priority_map[args.priority]
            )
            
            print(f"Submitted task: {task_id}")
            
            # Wait for result
            result = validator.wait_for_task(task_id, timeout=300)
            if result:
                print(json.dumps(result.to_dict(), indent=2))
            else:
                print("Task failed or timed out")
        
        elif args.command == 'batch':
            from pathlib import Path
            import glob
            
            data_dir = Path(args.data_dir)
            if not data_dir.exists():
                print(f"Directory not found: {data_dir}")
                return
            
            # Find files matching pattern
            pattern = str(data_dir / args.pattern)
            data_files = glob.glob(pattern)
            
            if not data_files:
                print(f"No files found matching pattern: {pattern}")
                return
            
            print(f"Found {len(data_files)} files to validate")
            
            # Run batch validation
            results = validator.batch_validate(
                data_files,
                args.type,
                max_workers=args.max_workers
            )
            
            # Print summary
            successful = [r for r in results if r.success]
            print(f"\nValidation completed:")
            print(f"  Total files: {len(results)}")
            print(f"  Successful: {len(successful)}")
            print(f"  Failed: {len(results) - len(successful)}")
            
            if successful:
                avg_score = sum(r.quality_score for r in successful) / len(successful)
                print(f"  Average quality score: {avg_score:.3f}")
        
        elif args.command == 'status':
            stats = validator.get_validation_statistics()
            print(json.dumps(stats, indent=2))
    
    finally:
        validator.cleanup()


if __name__ == '__main__':
    main()
