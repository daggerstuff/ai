import asyncio
import logging
import time
from datetime import datetime
from typing import Optional

from fastapi import (
    FastAPI,
    HTTPException,
    WebSocket,
    WebSocketDisconnect,
    BackgroundTasks,
    Depends,
    Request,
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.responses import JSONResponse
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
import structlog

from .config import config
from .models import (
    TranscriptRequest,
    TranscriptResponse,
    PipelineJobRequest,
    StageExecutionRequest,
    JobInfo,
    JobStatus,
    PipelineStage,
    StageResult,
    PipelineStatus,
    HealthCheckResponse,
)
from .utils import pipeline_executor, data_manager
from .auth import get_current_user, User, RequireCreateJobs, RequireDeleteJobs, auth_manager
from .rate_limiting import init_rate_limiting, cleanup_rate_limiting, quota_manager, youtube_limiter
from .monitoring import setup_monitoring, metrics
from .database import init_database, get_db

# Configure structured logging
logger = structlog.get_logger(__name__)

# Initialize rate limiter
limiter = Limiter(key_func=get_remote_address)

# Initialize FastAPI app
app = FastAPI(
    title="Pixel Voice Pipeline API",
    description="Production-grade API for the Pixel Voice processing pipeline",
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
)

# Add security middleware
app.add_middleware(TrustedHostMiddleware, allowed_hosts=["*"])  # Configure for production

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=config.api.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Add rate limiting
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# WebSocket connections for real-time updates
websocket_connections: list[WebSocket] = []


# Startup and shutdown events
@app.on_event("startup")
async def startup_event():
    """Initialize services on startup."""
    logger.info("Starting Pixel Voice API server...")

    # Initialize database
    if hasattr(config, "database_url"):
        init_database(config.database_url)
        logger.info("Database initialized")

    # Initialize rate limiting
    redis_url = getattr(config, "redis_url", None)
    await init_rate_limiting(redis_url)
    logger.info("Rate limiting initialized")

    # Setup monitoring
    setup_monitoring(app)
    logger.info("Monitoring setup completed")

    # Ensure directories exist
    config.ensure_directories()

    logger.info("Pixel Voice API server startup completed")


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown."""
    logger.info("Shutting down Pixel Voice API server...")

    # Cleanup rate limiting
    await cleanup_rate_limiting()

    # Close database connections
    if hasattr(config, "database_url"):
        from .database import db_manager

        if db_manager:
            db_manager.close()

    logger.info("Pixel Voice API server shutdown completed")


# --- Utility Functions ---


async def broadcast_to_websockets(message: dict):
    """Broadcast a message to all connected WebSocket clients."""
    if websocket_connections:
        disconnected = []
        for websocket in websocket_connections:
            try:
                await websocket.send_json(message)
            except Exception:
                disconnected.append(websocket)

        # Remove disconnected clients
        for ws in disconnected:
            websocket_connections.remove(ws)


async def run_pipeline(youtube_url: str, language: str, whisper_model: str) -> str:
    """Download audio and transcribe, return path to latest transcript."""
    try:
        # 1. Download audio
        audio_cmd = ["python3", "audio_downloader.py", youtube_url]

        # 2. Batch transcribe
        transcribe_cmd = ["python3", "batch_transcribe.py"]

        # Download audio
        process = await asyncio.create_subprocess_exec(
            *audio_cmd,
            cwd=config.base_dir,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        await process.communicate()

        if process.returncode != 0:
            raise RuntimeError("Audio download failed")

        # Transcribe
        process = await asyncio.create_subprocess_exec(
            *transcribe_cmd,
            cwd=config.base_dir,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        await process.communicate()

        if process.returncode != 0:
            raise RuntimeError("Transcription failed")

        # Find latest transcript
        transcript_path = data_manager.get_latest_file(config.directories.voice_transcripts)
        if not transcript_path:
            raise RuntimeError("Transcript not found after processing.")

        return transcript_path

    except Exception as e:
        logger.error(f"Pipeline execution failed: {e}")
        raise


# --- API Endpoints ---


@app.get("/", response_model=HealthCheckResponse)
async def root():
    """Health check endpoint."""
    return HealthCheckResponse(
        status="healthy",
        version="1.0.0",
        uptime=0.0,  # TODO: Implement actual uptime tracking
        dependencies={"whisperx": "available", "yt-dlp": "available"},
        system_info={"environment": config.environment, "debug": config.debug},
    )


@app.post("/transcribe", response_model=TranscriptResponse)
@limiter.limit("10/minute")
async def transcribe(
    request: TranscriptRequest,
    background_tasks: BackgroundTasks,
    current_user: User = Depends(get_current_user),
):
    """API endpoint: YouTube URL → transcript path."""
    try:
        # Check user quota
        if quota_manager:
            can_download, error_msg = await quota_manager.check_quota(
                current_user.id, "youtube_download"
            )
            if not can_download:
                metrics.record_rate_limit_violation("youtube_download", current_user.id)
                raise HTTPException(status_code=429, detail=error_msg)

        # Record metrics
        start_time = time.time()

        # Create a simple transcription job
        job_id = await pipeline_executor.execute_pipeline_job(
            job_name=f"Transcribe {request.youtube_url}",
            stages=[PipelineStage.BATCH_TRANSCRIPTION],
            input_data={"youtube_url": request.youtube_url},
            config_overrides={
                "language": request.language,
                "whisper_model": request.whisper_model,
                "enable_diarization": request.enable_diarization,
            },
        )

        # Consume quota
        if quota_manager:
            await quota_manager.consume_quota(current_user.id, "youtube_download")
            await quota_manager.consume_quota(current_user.id, "api_call")

        # Record metrics
        metrics.record_youtube_download("started", time.time() - start_time)

        return TranscriptResponse(
            transcript_path="",  # Will be updated when job completes
            status=JobStatus.PENDING,
            message="Transcription job started",
            job_id=job_id,
        )
    except HTTPException:
        raise
    except Exception as e:
        metrics.record_error("transcription_error", "api")
        logger.error("Transcription request failed", error=str(e), user_id=current_user.id)
        raise HTTPException(status_code=500, detail=str(e)) from e


@app.post("/pipeline/jobs", response_model=str)
@limiter.limit("5/minute")
async def create_pipeline_job(request: PipelineJobRequest, current_user: User = RequireCreateJobs):
    """Create and start a new pipeline job."""
    try:
        # Check concurrent job quota
        if quota_manager:
            can_create, error_msg = await quota_manager.check_quota(
                current_user.id, "concurrent_job"
            )
            if not can_create:
                metrics.record_rate_limit_violation("concurrent_job", current_user.id)
                raise HTTPException(status_code=429, detail=error_msg)

        # Record metrics
        start_time = time.time()

        job_id = await pipeline_executor.execute_pipeline_job(
            job_name=request.job_name,
            stages=request.stages,
            input_data=request.input_data,
            config_overrides=request.config_overrides,
        )

        # Consume quota
        if quota_manager:
            await quota_manager.consume_quota(current_user.id, "job_start")
            await quota_manager.consume_quota(current_user.id, "api_call")

        # Record metrics
        metrics.record_pipeline_job("created", "success")

        # Broadcast job creation to WebSocket clients
        await broadcast_to_websockets(
            {
                "type": "job_created",
                "job_id": job_id,
                "job_name": request.job_name,
                "user_id": current_user.id,
            }
        )

        logger.info("Pipeline job created", job_id=job_id, user_id=current_user.id)
        return job_id
    except HTTPException:
        raise
    except Exception as e:
        metrics.record_error("job_creation_error", "api")
        logger.error("Pipeline job creation failed", error=str(e), user_id=current_user.id)
        raise HTTPException(status_code=500, detail=str(e)) from e


@app.get("/pipeline/jobs", response_model=list[JobInfo])
async def list_jobs(status: Optional[JobStatus] = None):
    """List all pipeline jobs, optionally filtered by status."""
    try:
        jobs = pipeline_executor.list_jobs(status_filter=status)
        return jobs
    except Exception as e:
        logger.error(f"Failed to list jobs: {e}")
        raise HTTPException(status_code=500, detail=str(e)) from e


@app.get("/pipeline/jobs/{job_id}", response_model=JobInfo)
async def get_job(job_id: str):
    """Get information about a specific job."""
    job_info = pipeline_executor.get_job_info(job_id)
    if not job_info:
        raise HTTPException(status_code=404, detail="Job not found")
    return job_info


@app.delete("/pipeline/jobs/{job_id}")
async def cancel_job(job_id: str):
    """Cancel a running job."""
    success = pipeline_executor.cancel_job(job_id)
    if not success:
        raise HTTPException(status_code=404, detail="Job not found or not cancellable")

    # Broadcast job cancellation to WebSocket clients
    await broadcast_to_websockets({"type": "job_cancelled", "job_id": job_id})

    return {"message": "Job cancelled successfully"}


@app.post("/pipeline/stages/execute", response_model=StageResult)
async def execute_stage(request: StageExecutionRequest):
    """Execute a single pipeline stage."""
    try:
        result = await pipeline_executor.execute_stage(
            stage=request.stage,
            input_path=request.input_path,
            output_path=request.output_path,
            config_overrides=request.config_overrides,
        )
        return result
    except Exception as e:
        logger.error(f"Stage execution failed: {e}")
        raise HTTPException(status_code=500, detail=str(e)) from e


@app.get("/pipeline/status", response_model=PipelineStatus)
async def get_pipeline_status():
    """Get overall pipeline status."""
    try:
        all_jobs = pipeline_executor.list_jobs()
        running_jobs = [j for j in all_jobs if j.status == JobStatus.RUNNING]
        completed_jobs = [j for j in all_jobs if j.status == JobStatus.COMPLETED]
        failed_jobs = [j for j in all_jobs if j.status == JobStatus.FAILED]

        return PipelineStatus(
            total_jobs=len(all_jobs),
            running_jobs=len(running_jobs),
            completed_jobs=len(completed_jobs),
            failed_jobs=len(failed_jobs),
            system_health="healthy" if len(failed_jobs) == 0 else "degraded",
            last_updated=datetime.now(),
        )
    except Exception as e:
        logger.error(f"Failed to get pipeline status: {e}")
        raise HTTPException(status_code=500, detail=str(e)) from e


@app.get("/data/{data_type}/latest")
async def get_latest_data(data_type: str):
    """Get the latest data file of a specific type."""
    try:
        # Map data types to directories
        data_type_mapping = {
            "transcripts": config.directories.voice_transcripts,
            "features": config.directories.voice_features,
            "dialogue_pairs": config.directories.dialogue_pairs,
            "therapeutic_pairs": config.directories.therapeutic_pairs,
            "consistency": config.directories.voice_consistency,
            "optimized": config.directories.voice_optimized,
        }

        if data_type not in data_type_mapping:
            raise HTTPException(status_code=400, detail=f"Unknown data type: {data_type}")

        directory = data_type_mapping[data_type]
        latest_file = data_manager.get_latest_file(directory)

        if not latest_file:
            raise HTTPException(status_code=404, detail=f"No {data_type} data found")

        data = data_manager.load_json_data(latest_file)
        return data
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get latest {data_type} data: {e}")
        raise HTTPException(status_code=500, detail=str(e)) from e


@app.get("/data/{data_type}/files")
async def list_data_files(data_type: str):
    """List all data files of a specific type."""
    try:
        data_type_mapping = {
            "transcripts": config.directories.voice_transcripts,
            "features": config.directories.voice_features,
            "dialogue_pairs": config.directories.dialogue_pairs,
            "therapeutic_pairs": config.directories.therapeutic_pairs,
            "consistency": config.directories.voice_consistency,
            "optimized": config.directories.voice_optimized,
        }

        if data_type not in data_type_mapping:
            raise HTTPException(status_code=400, detail=f"Unknown data type: {data_type}")

        directory = data_type_mapping[data_type]
        files = data_manager.list_files(directory, "*.json")

        return {"files": files, "count": len(files)}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to list {data_type} files: {e}")
        raise HTTPException(status_code=500, detail=str(e)) from e


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for real-time updates."""
    await websocket.accept()
    websocket_connections.append(websocket)

    try:
        while True:
            # Keep connection alive and handle incoming messages
            data = await websocket.receive_text()
            # Echo back for now - could implement specific commands
            await websocket.send_json({"type": "echo", "data": data})
    except WebSocketDisconnect:
        websocket_connections.remove(websocket)
