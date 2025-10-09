#!/usr/bin/env python3
"""
Celery Configuration for Pixelated Empathy AI Distributed Processing
"""

import os
from celery import Celery
from kombu import Queue, Exchange

# Celery application configuration
def create_celery_app():
    """Create and configure Celery application"""
    
    # Get configuration from environment
    broker_url = os.getenv('CELERY_BROKER_URL', 'redis://localhost:6379/0')
    result_backend = os.getenv('CELERY_RESULT_BACKEND', 'redis://localhost:6379/0')
    
    # Create Celery app
    app = Celery('pixelated_empathy')
    
    # Configure Celery
    app.conf.update(
        # Broker settings
        broker_url=broker_url,
        result_backend=result_backend,
        
        # Task serialization
        task_serializer='pickle',
        accept_content=['pickle', 'json'],
        result_serializer='pickle',
        
        # Timezone settings
        timezone='UTC',
        enable_utc=True,
        
        # Task routing
        task_routes={
            'quality_validator.validate_task': {'queue': 'quality_validation'},
            'data_processor.process_task': {'queue': 'data_processing'},
            'model_trainer.train_task': {'queue': 'model_training'},
            'backup.backup_task': {'queue': 'backup'},
        },
        
        # Queue configuration
        task_default_queue='default',
        task_queues=(
            Queue('default', Exchange('default'), routing_key='default'),
            Queue('quality_validation', Exchange('quality'), routing_key='quality.validation'),
            Queue('data_processing', Exchange('data'), routing_key='data.processing'),
            Queue('model_training', Exchange('training'), routing_key='training.model'),
            Queue('backup', Exchange('backup'), routing_key='backup.task'),
            Queue('high_priority', Exchange('priority'), routing_key='priority.high'),
        ),
        
        # Worker settings
        worker_prefetch_multiplier=1,
        task_acks_late=True,
        worker_max_tasks_per_child=1000,
        
        # Task execution settings
        task_soft_time_limit=300,  # 5 minutes
        task_time_limit=600,       # 10 minutes
        task_reject_on_worker_lost=True,
        
        # Result settings
        result_expires=3600,  # 1 hour
        
        # Monitoring
        worker_send_task_events=True,
        task_send_sent_event=True,
        
        # Error handling
        task_annotations={
            '*': {'rate_limit': '100/m'},
            'quality_validator.validate_task': {'rate_limit': '50/m'},
            'model_trainer.train_task': {'rate_limit': '5/m'},
        },
        
        # Beat schedule (for periodic tasks)
        beat_schedule={
            'cleanup-old-results': {
                'task': 'maintenance.cleanup_old_results',
                'schedule': 3600.0,  # Every hour
            },
            'health-check': {
                'task': 'monitoring.health_check',
                'schedule': 300.0,   # Every 5 minutes
            },
            'backup-data': {
                'task': 'backup.backup_task',
                'schedule': 86400.0, # Daily
                'kwargs': {'backup_type': 'incremental'}
            },
        },
    )
    
    return app

# Create the Celery app instance
celery_app = create_celery_app()

# Task discovery
celery_app.autodiscover_tasks([
    'distributed_processing.quality_validator',
    'distributed_processing.data_processor',
    'distributed_processing.model_trainer',
    'distributed_processing.backup_manager',
    'distributed_processing.monitoring',
])

if __name__ == '__main__':
    celery_app.start()
