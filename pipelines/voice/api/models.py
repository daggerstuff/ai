"""
Shared data models for Pixel Voice API and MCP server.
"""

from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Union
from pydantic import BaseModel, Field


class JobStatus(str, Enum):
    """Status of a pipeline job."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class PipelineStage(str, Enum):
    """Available pipeline stages."""

    AUDIO_QUALITY_CONTROL = "audio_quality_control"
    BATCH_TRANSCRIPTION = "batch_transcription"
    TRANSCRIPTION_FILTERING = "transcription_filtering"
    FEATURE_EXTRACTION = "feature_extraction"
    PERSONALITY_CLUSTERING = "personality_clustering"
    DIALOGUE_CONSTRUCTION = "dialogue_construction"
    DIALOGUE_VALIDATION = "dialogue_validation"
    THERAPEUTIC_GENERATION = "therapeutic_generation"
    VOICE_CONSISTENCY = "voice_consistency"
    VOICE_FILTERING = "voice_filtering"


class LogLevel(str, Enum):
    """Log levels."""

    DEBUG = "debug"
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


# Request Models
class TranscriptRequest(BaseModel):
    """Request model for transcription."""

    youtube_url: str = Field(..., description="YouTube URL to transcribe")
    language: str = Field(default="en", description="Language code for transcription")
    whisper_model: str = Field(default="large-v2", description="Whisper model to use")
    enable_diarization: bool = Field(default=True, description="Enable speaker diarization")


class PipelineJobRequest(BaseModel):
    """Request model for pipeline job execution."""

    job_name: str = Field(..., description="Name for the job")
    stages: List[PipelineStage] = Field(..., description="Pipeline stages to execute")
    input_data: Dict[str, Any] = Field(
        default_factory=dict, description="Input data for the pipeline"
    )
    config_overrides: Dict[str, Any] = Field(
        default_factory=dict, description="Configuration overrides"
    )


class StageExecutionRequest(BaseModel):
    """Request model for individual stage execution."""

    stage: PipelineStage = Field(..., description="Pipeline stage to execute")
    input_path: Optional[str] = Field(None, description="Input file or directory path")
    output_path: Optional[str] = Field(None, description="Output file or directory path")
    config_overrides: Dict[str, Any] = Field(
        default_factory=dict, description="Configuration overrides"
    )


# Response Models
class TranscriptResponse(BaseModel):
    """Response model for transcription."""

    transcript_path: str = Field(..., description="Path to the generated transcript")
    status: JobStatus = Field(..., description="Status of the transcription job")
    message: Optional[str] = Field(None, description="Additional message or error details")
    job_id: Optional[str] = Field(None, description="Job ID for tracking")


class JobInfo(BaseModel):
    """Information about a pipeline job."""

    job_id: str = Field(..., description="Unique job identifier")
    job_name: str = Field(..., description="Human-readable job name")
    status: JobStatus = Field(..., description="Current job status")
    stages: List[PipelineStage] = Field(..., description="Pipeline stages in the job")
    current_stage: Optional[PipelineStage] = Field(None, description="Currently executing stage")
    progress: float = Field(default=0.0, description="Job progress (0.0 to 1.0)")
    created_at: datetime = Field(..., description="Job creation timestamp")
    started_at: Optional[datetime] = Field(None, description="Job start timestamp")
    completed_at: Optional[datetime] = Field(None, description="Job completion timestamp")
    error_message: Optional[str] = Field(None, description="Error message if job failed")
    output_paths: Dict[str, str] = Field(
        default_factory=dict, description="Output file paths by stage"
    )


class StageResult(BaseModel):
    """Result of a pipeline stage execution."""

    stage: PipelineStage = Field(..., description="Pipeline stage that was executed")
    status: JobStatus = Field(..., description="Stage execution status")
    input_path: Optional[str] = Field(None, description="Input file or directory path")
    output_path: Optional[str] = Field(None, description="Output file or directory path")
    execution_time: float = Field(..., description="Execution time in seconds")
    error_message: Optional[str] = Field(None, description="Error message if stage failed")
    metrics: Dict[str, Any] = Field(default_factory=dict, description="Stage-specific metrics")


class PipelineStatus(BaseModel):
    """Overall pipeline status."""

    total_jobs: int = Field(..., description="Total number of jobs")
    running_jobs: int = Field(..., description="Number of currently running jobs")
    completed_jobs: int = Field(..., description="Number of completed jobs")
    failed_jobs: int = Field(..., description="Number of failed jobs")
    system_health: str = Field(..., description="Overall system health status")
    last_updated: datetime = Field(..., description="Last status update timestamp")


class LogEntry(BaseModel):
    """Log entry model."""

    timestamp: datetime = Field(..., description="Log entry timestamp")
    level: LogLevel = Field(..., description="Log level")
    message: str = Field(..., description="Log message")
    job_id: Optional[str] = Field(None, description="Associated job ID")
    stage: Optional[PipelineStage] = Field(None, description="Associated pipeline stage")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")


class DataExportRequest(BaseModel):
    """Request model for data export."""

    data_type: str = Field(..., description="Type of data to export (transcripts, features, etc.)")
    format: str = Field(default="json", description="Export format (json, csv, jsonl)")
    filter_criteria: Dict[str, Any] = Field(default_factory=dict, description="Filtering criteria")
    include_metadata: bool = Field(default=True, description="Include metadata in export")


class DataExportResponse(BaseModel):
    """Response model for data export."""

    export_path: str = Field(..., description="Path to the exported data file")
    record_count: int = Field(..., description="Number of records exported")
    file_size: int = Field(..., description="File size in bytes")
    format: str = Field(..., description="Export format used")
    created_at: datetime = Field(..., description="Export creation timestamp")


class HealthCheckResponse(BaseModel):
    """Health check response model."""

    status: str = Field(..., description="Service health status")
    version: str = Field(..., description="Service version")
    uptime: float = Field(..., description="Service uptime in seconds")
    dependencies: Dict[str, str] = Field(..., description="Status of external dependencies")
    system_info: Dict[str, Any] = Field(..., description="System information")


# MCP Tool Models
class MCPToolRequest(BaseModel):
    """Base model for MCP tool requests."""

    tool_name: str = Field(..., description="Name of the MCP tool to execute")
    parameters: Dict[str, Any] = Field(default_factory=dict, description="Tool parameters")


class MCPToolResponse(BaseModel):
    """Base model for MCP tool responses."""

    success: bool = Field(..., description="Whether the tool execution was successful")
    result: Any = Field(None, description="Tool execution result")
    error_message: Optional[str] = Field(None, description="Error message if execution failed")
    execution_time: float = Field(..., description="Tool execution time in seconds")
