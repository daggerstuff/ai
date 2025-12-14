#!/usr/bin/env python3
"""
FastAPI Inference Service for Therapeutic MoE Model
Optimized for <2 second response time with high throughput
"""

import asyncio
import time
from typing import Dict, List, Optional
from datetime import datetime
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, BackgroundTasks, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
import uvicorn

from inference_optimizer import OptimizedInferenceEngine, InferenceConfig, create_optimized_engine
from models.therapeutic_progress_tracker import TherapeuticProgressTracker
import logging

logger = logging.getLogger(__name__)


# Global inference engine and progress tracker
inference_engine: Optional[OptimizedInferenceEngine] = None
progress_tracker: Optional[TherapeuticProgressTracker] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifecycle manager for FastAPI app"""
    global inference_engine, progress_tracker

    # Startup
    print("🚀 Starting Therapeutic AI Inference Service...")

    # Load model
    model_path = "./therapeutic_moe_model"
    inference_engine = create_optimized_engine(
        model_path=model_path,
        device="cuda",
        enable_cache=True,
        compile_model=True
    )

    # Initialize progress tracker
    progress_tracker = TherapeuticProgressTracker(db_path="therapeutic_progress.db")
    print("✅ Progress tracker initialized")

    print("✅ Service ready!")

    yield

    # Shutdown
    print("🛑 Shutting down service...")
    if progress_tracker:
        progress_tracker.close()


# Create FastAPI app
app = FastAPI(
    title="Therapeutic AI Inference API",
    description="Optimized inference service for therapeutic MoE model",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Request/Response models
class Message(BaseModel):
    """Conversation message"""
    role: str = Field(..., description="Role: 'user', 'assistant', or 'system'")
    content: str = Field(..., description="Message content")


class InferenceRequest(BaseModel):
    """Request for inference"""
    user_input: str = Field(..., description="User's input text", min_length=1)
    conversation_history: Optional[List[Message]] = Field(
        default=None,
        description="Previous conversation messages"
    )
    system_prompt: Optional[str] = Field(
        default=None,
        description="System prompt for context"
    )
    use_cache: bool = Field(
        default=True,
        description="Whether to use response cache"
    )
    max_new_tokens: Optional[int] = Field(
        default=None,
        description="Maximum tokens to generate"
    )
    temperature: Optional[float] = Field(
        default=None,
        description="Sampling temperature (0.0-2.0)"
    )
    # Progress tracking fields
    client_id: Optional[str] = Field(
        default=None,
        description="Client ID for progress tracking"
    )
    session_id: Optional[str] = Field(
        default=None,
        description="Session ID for progress tracking"
    )
    track_progress: bool = Field(
        default=True,
        description="Whether to track this interaction for progress"
    )


class InferenceResponse(BaseModel):
    """Response from inference"""
    response: str = Field(..., description="Generated response")
    latency: float = Field(..., description="Response latency in seconds")
    cache_hit: bool = Field(..., description="Whether response was cached")
    tokens_generated: int = Field(..., description="Number of tokens generated")
    metadata: Dict = Field(..., description="Additional metadata")


class HealthResponse(BaseModel):
    """Health check response"""
    status: str
    model_loaded: bool
    uptime_seconds: float
    total_requests: int
    avg_latency: float
    p95_latency: float
    cache_hit_rate: float
    meets_sla: bool


class MetricsResponse(BaseModel):
    """Metrics response"""
    total_requests: int
    avg_latency: float
    p50_latency: float
    p95_latency: float
    p99_latency: float
    cache_hit_rate: float
    cache_stats: Dict
    meets_sla: bool


# Track service start time
service_start_time = time.time()


@app.get("/", response_model=Dict)
async def root():
    """Root endpoint"""
    return {
        "service": "Therapeutic AI Inference API",
        "version": "1.0.0",
        "status": "running",
        "endpoints": {
            "inference": "/api/v1/inference",
            "health": "/health",
            "metrics": "/metrics"
        }
    }


@app.post("/api/v1/inference", response_model=InferenceResponse)
async def inference(request: InferenceRequest, background_tasks: BackgroundTasks):
    """
    Generate therapeutic response with progress tracking

    Target latency: <2 seconds (p95)
    """
    if inference_engine is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    try:
        # Convert conversation history
        conversation_history = None
        if request.conversation_history:
            conversation_history = [
                {"role": msg.role, "content": msg.content}
                for msg in request.conversation_history
            ]

        # Generate response
        response_text, metadata = await inference_engine.generate_async(
            user_input=request.user_input,
            conversation_history=conversation_history,
            system_prompt=request.system_prompt,
            use_cache=request.use_cache
        )

        # Track progress if enabled and client_id provided
        if request.track_progress and request.client_id and progress_tracker:
            background_tasks.add_task(
                _log_session_progress,
                client_id=request.client_id,
                session_id=request.session_id or f"session_{int(time.time())}",
                user_input=request.user_input,
                ai_response=response_text,
                conversation_history=conversation_history
            )

        return InferenceResponse(
            response=response_text,
            latency=metadata['latency'],
            cache_hit=metadata['cache_hit'],
            tokens_generated=metadata['tokens_generated'],
            metadata=metadata
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference failed: {str(e)}")


async def _log_session_progress(
    client_id: str,
    session_id: str,
    user_input: str,
    ai_response: str,
    conversation_history: Optional[List[Dict]] = None
):
    """Background task to log session progress"""
    try:
        if progress_tracker is None:
            return

        # Create conversation summary
        summary = f"User: {user_input[:100]}... | AI: {ai_response[:100]}..."

        # Analyze emotional state (simple heuristic for now)
        emotional_state = _analyze_emotional_state(user_input)

        # Log the session
        progress_tracker.log_session(
            client_id=client_id,
            session_id=session_id,
            conversation_summary=summary,
            emotional_state=emotional_state,
            therapeutic_goals=[],  # Could be extracted from conversation
            progress_notes=f"Session on {datetime.now().strftime('%Y-%m-%d')}",
            therapist_observations="",
            next_session_focus=""
        )

    except Exception as e:
        logger.error(f"Failed to log progress: {e}")


def _analyze_emotional_state(text: str) -> str:
    """Simple emotional state analysis"""
    from models.therapeutic_progress_tracker import EmotionalState

    text_lower = text.lower()

    # Very negative indicators
    if any(word in text_lower for word in ['suicidal', 'hopeless', 'worthless', 'hate myself']):
        return EmotionalState.VERY_NEGATIVE.value

    # Negative indicators
    if any(word in text_lower for word in ['anxious', 'depressed', 'sad', 'worried', 'stressed']):
        return EmotionalState.NEGATIVE.value

    # Positive indicators
    if any(word in text_lower for word in ['better', 'good', 'happy', 'hopeful', 'improving']):
        return EmotionalState.POSITIVE.value

    # Very positive indicators
    if any(word in text_lower for word in ['great', 'wonderful', 'amazing', 'fantastic', 'excellent']):
        return EmotionalState.VERY_POSITIVE.value

    # Default to neutral
    return EmotionalState.NEUTRAL.value


@app.get("/health", response_model=HealthResponse)
async def health():
    """Health check endpoint"""
    if inference_engine is None:
        return HealthResponse(
            status="unhealthy",
            model_loaded=False,
            uptime_seconds=time.time() - service_start_time,
            total_requests=0,
            avg_latency=0.0,
            p95_latency=0.0,
            cache_hit_rate=0.0,
            meets_sla=False
        )

    metrics = inference_engine.get_metrics()

    return HealthResponse(
        status="healthy",
        model_loaded=True,
        uptime_seconds=time.time() - service_start_time,
        total_requests=metrics['total_requests'],
        avg_latency=metrics['avg_latency'],
        p95_latency=metrics['p95_latency'],
        cache_hit_rate=metrics['cache_hit_rate'],
        meets_sla=metrics['meets_sla']
    )


@app.get("/metrics", response_model=MetricsResponse)
async def metrics():
    """Get detailed metrics"""
    if inference_engine is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    metrics_data = inference_engine.get_metrics()

    return MetricsResponse(**metrics_data)


@app.post("/cache/clear")
async def clear_cache():
    """Clear response cache"""
    if inference_engine is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    if inference_engine.response_cache:
        inference_engine.response_cache.clear()
        return {"status": "success", "message": "Cache cleared"}
    else:
        return {"status": "info", "message": "Cache not enabled"}


@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    """Add processing time header"""
    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time
    response.headers["X-Process-Time"] = str(process_time)
    return response


def start_service(
    host: str = "0.0.0.0",
    port: int = 8000,
    workers: int = 1,
    reload: bool = False
):
    """Start the inference service"""
    uvicorn.run(
        "inference_service:app",
        host=host,
        port=port,
        workers=workers,
        reload=reload,
        log_level="info"
    )


if __name__ == "__main__":
    start_service(
        host="0.0.0.0",
        port=8000,
        workers=1,
        reload=False
    )
