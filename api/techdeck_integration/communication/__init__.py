"""
Communication Module for TechDeck-Python Pipeline Integration.

This module provides comprehensive pipeline communication with Redis event bus,
six-stage coordination, WebSocket integration, and HIPAA++ compliant data handling.
"""

from .event_bus import EventBus, EventMessage, EventType, EventHandler
from .pipeline_coordinator import PipelineCoordinator, PipelineContext
from .state_manager import StateManager, PipelineState, StageState
from .progress_tracker import ProgressTracker, ProgressUpdate, WebSocketConnection
from .error_recovery import ErrorRecoveryManager, RecoveryStrategy, RecoveryResult, RecoveryConfig
from .bias_integration import BiasDetectionIntegration, BiasMetrics, BiasDetectionConfig
from .performance_monitor import PerformanceMonitor, PerformanceMetric, PerformanceSummary, PerformanceThreshold

__all__ = [
    # Event Bus
    'EventBus',
    'EventMessage', 
    'EventType',
    'EventHandler',
    
    # Pipeline Coordinator
    'PipelineCoordinator',
    'PipelineContext',
    
    # State Manager
    'StateManager',
    'PipelineState',
    'StageState',
    
    # Progress Tracker
    'ProgressTracker',
    'ProgressUpdate',
    'WebSocketConnection',
    
    # Error Recovery
    'ErrorRecoveryManager',
    'RecoveryStrategy',
    'RecoveryResult',
    'RecoveryConfig',
    
    # Bias Detection
    'BiasDetectionIntegration',
    'BiasMetrics',
    'BiasDetectionConfig',
    
    # Performance Monitor
    'PerformanceMonitor',
    'PerformanceMetric',
    'PerformanceSummary',
    'PerformanceThreshold'
]

# Module version
__version__ = '1.0.0'

# Module metadata
__author__ = 'Pixelated Empathy Team'
__description__ = 'Comprehensive pipeline communication for TechDeck-Python integration with HIPAA++ compliance'