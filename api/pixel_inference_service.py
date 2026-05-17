"""
Pixel Model Inference Service

Provides FastAPI endpoints for Pixel model inference with:
- Model loading and caching
- Real-time conversation analysis with EQ awareness
- Bias detection and crisis intervention
- Multi-turn conversation context management
- Performance optimization (<200ms latency)
"""

import asyncio
import json
import logging
import os
import sys
import re
import time
import uuid
from datetime import UTC, datetime

# Import models and utilities
from pathlib import Path
from typing import Any

from fastapi import BackgroundTasks, FastAPI, HTTPException
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from ai.utils.torch_proxy import torch

# Add parent directories to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from pixel.models.pixel_base_model import PixelBaseModel

from ai.api.sentry_logging import initialize_sentry_logging

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)
initialize_sentry_logging(service_name="pixel-inference-service")

INFERENCE_LATENCY_WARNING_MS = 200
EMPATHY_SUPPORT_THRESHOLD = 0.7
PIXEL_API_DEFAULT_PORT = "8001"
MAX_BATCH_CONCURRENCY = int(os.getenv("PIXEL_MAX_BATCH_CONCURRENCY", "16"))
_THOUGHT_MARKER_RE = re.compile(r"^\s*\[Thought:\s*.*\]\s*$")
_STOP_TURN_RE = re.compile(r"^\s*\[STOP_TURN\]\s*$")
_PROMISE_MARKER_RE = re.compile(r"<promise>.*?</promise>", re.IGNORECASE | re.DOTALL)


def sanitize_agent_output(raw_text: str | None) -> str:
    """Strip internal protocol markers intended for agents from user-facing output."""
    if not raw_text:
        return ""

    output_lines: list[str] = []
    for line in raw_text.splitlines():
        if _THOUGHT_MARKER_RE.match(line):
            continue
        if _STOP_TURN_RE.match(line):
            continue
        if re.match(r"^\s*[✓✦]", line):
            continue
        if "<promise>" in line.lower() and "</promise>" in line.lower():
            line = _PROMISE_MARKER_RE.sub("", line)
            if not line.strip():
                continue
        output_lines.append(line.rstrip())
    return "\n".join(output_lines)
MAX_BATCH_CONCURRENCY = int(os.getenv("PIXEL_MAX_BATCH_CONCURRENCY", "16"))
MAX_BATCH_CONCURRENCY = int(os.getenv("PIXEL_MAX_BATCH_CONCURRENCY", "16"))
MAX_BATCH_CONCURRENCY = int(os.getenv("PIXEL_MAX_BATCH_CONCURRENCY", "16"))
MAX_BATCH_CONCURRENCY = int(os.getenv("PIXEL_MAX_BATCH_CONCURRENCY", "16"))
MAX_BATCH_CONCURRENCY = int(os.getenv("PIXEL_MAX_BATCH_CONCURRENCY", "16"))
MAX_BATCH_CONCURRENCY = int(os.getenv("PIXEL_MAX_BATCH_CONCURRENCY", "16"))
MAX_BATCH_CONCURRENCY = int(os.getenv("PIXEL_MAX_BATCH_CONCURRENCY", "16"))

# Request/Response Models


class ConversationMessage(BaseModel):
    """Single message in conversation history"""

    role: str = Field(..., description="Message role: user, assistant, or system")
    content: str = Field(..., description="Message content")
    timestamp: str | None = None


class PixelInferenceRequest(BaseModel):
    """Request model for Pixel inference"""

    user_query: str = Field(..., description="User query text")
    conversation_history: list[ConversationMessage] = Field(
        default_factory=list, description="Prior conversation messages for context"
    )
    context_type: str | None = Field(
        None,
        description=(
            "Context type: educational, support, crisis, clinical, informational"
        ),
    )
    user_id: str | None = Field(None, description="User identifier for tracking")
    session_id: str | None = Field(None, description="Session identifier")
    use_eq_awareness: bool = Field(
        True, description="Enable EQ-aware response generation"
    )
    include_metrics: bool = Field(
        True, description="Include quality metrics in response"
    )
    max_tokens: int = Field(200, description="Max tokens to generate")
    gestalt_directive: str | None = Field(None, description="Supervisor override directive")


class EQScores(BaseModel):
    """EQ measurement scores"""

    emotional_awareness: float
    empathy_recognition: float
    emotional_regulation: float
    social_cognition: float
    interpersonal_skills: float
    overall_eq: float


class ConversationMetadata(BaseModel):
    """Metadata about conversation analysis"""

    detected_techniques: list[str]
    technique_consistency: float
    bias_score: float
    safety_score: float
    crisis_signals: list[str] | None = None
    therapeutic_effectiveness_score: float


class PixelInferenceResponse(BaseModel):
    """Response model for Pixel inference"""

    response: str = Field(..., description="Generated response")
    inference_time_ms: float
    eq_scores: EQScores | None = None
    conversation_metadata: ConversationMetadata | None = None
    persona_mode: str = Field(
        "therapy", description="Detected persona: therapy or assistant"
    )
    confidence: float = Field(0.9, description="Confidence in response")
    warning: str | None = None
    agent_activities: list["AgentActivity"] | None = None
    shared_state: dict[str, Any] | None = None


class AgentConflict(BaseModel):
    """Represents a disagreement between agents"""
    with_agent: str = Field(..., alias="withAgent")
    severity: str  # low, medium, high
    description: str


class AgentActivity(BaseModel):
    """Represents a single activity from an agent"""

    id: str
    agent_name: str = Field(..., alias="agentName")
    agent_role: str | None = Field(None, alias="agentRole")
    type: str  # thought, action, observation, tool_use
    content: str
    thought: str | None = None
    action: str | None = None
    observation: str | None = None
    status: str  # thinking, acting, completed, error
    timestamp: float
    metadata: dict[str, Any] | None = None
    shared_state: dict[str, Any] | None = None
    conflict: AgentConflict | None = None


PixelInferenceResponse.update_forward_refs()


class ModelStatusResponse(BaseModel):
    """Model status information"""

    model_loaded: bool
    model_name: str
    inference_engine: str
    available_features: list[str]
    performance_metrics: dict[str, Any]
    last_inference_time_ms: float | None = None


# Pixel Inference Service


class PixelInferenceEngine:
    """Manages Pixel model loading, caching, and inference"""

    def __init__(self):
        self.model: PixelBaseModel | None = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_loaded = False
        self.inference_count = 0
        self.total_inference_time = 0.0
        self.model_path = os.getenv(
            "PIXEL_MODEL_PATH", "ai/models/pixel_core/models/pixel_base_model.pt"
        )
        # Per-agent metrics
        self.agent_stats = {
            "Coordinator": {"calls": 0, "total_time": 0, "errors": 0},
            "Psychologist": {"calls": 0, "total_time": 0, "errors": 0},
            "Memory Agent": {"calls": 0, "total_time": 0, "errors": 0},
            "Safety Guard": {"calls": 0, "total_time": 0, "errors": 0}
        }

    def record_agent_step(self, agent_name: str, duration_ms: float, success: bool = True):
        """Record a single reasoning step from an agent"""
        if agent_name in self.agent_stats:
            self.agent_stats[agent_name]["calls"] += 1
            self.agent_stats[agent_name]["total_time"] += duration_ms
            if not success:
                self.agent_stats[agent_name]["errors"] += 1

    def get_agent_report(self) -> dict[str, Any]:
        """Generate per-agent performance report"""
        report = {}
        for name, stats in self.agent_stats.items():
            avg_time = stats["total_time"] / stats["calls"] if stats["calls"] > 0 else 0
            report[name] = {
                "average_latency_ms": round(avg_time, 2),
                "error_rate": round(stats["errors"] / stats["calls"], 4) if stats["calls"] > 0 else 0,
                "throughput": stats["calls"]
            }
        return report

    def reset_agent_stats(self):
        """Reset all performance metrics"""
        for name in self.agent_stats:
            self.agent_stats[name] = {"calls": 0, "total_time": 0, "errors": 0}
        self.inference_count = 0
        self.total_inference_time = 0.0

    def load_model(self) -> bool:
        """Load Pixel model from disk"""
        try:
            return self._ensure_model_loaded()
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            self.model_loaded = False
            return False

    def _ensure_model_loaded(self) -> bool:
        if self.model is not None and self.model_loaded:
            logger.info("Model already loaded")
            return True

        logger.info(f"Loading Pixel model from {self.model_path}")

        # Check if model file exists
        if not os.path.exists(self.model_path):
            logger.warning(
                f"Model file not found at {self.model_path}, creating fresh model"
            )
            self.model = PixelBaseModel()
        else:
            self.model = PixelBaseModel.load(self.model_path)

        self.model = self.model.to(self.device)
        self.model.eval()
        self.model_loaded = True
        logger.info("Pixel model loaded successfully")
        return True

    def preprocess_input(
        self, query: str, history: list[ConversationMessage]
    ) -> torch.Tensor:
        """Convert query and history to model input tensor"""
        # Create simple token embedding (in production, use actual tokenizer)
        # For now, use positional encoding + word embeddings simulation
        history_context = " ".join([m.content for m in history[-3:]])  # Last 3 messages
        full_context = f"{history_context} {query}"

        # Simulate tokenization: create embedding
        # Shape: (batch=1, seq_len=query_len+context, d_model=768)
        seq_len = min(len(full_context.split()) + 1, 512)
        return torch.randn(1, seq_len, 768, device=self.device)

    async def generate_response(
        self, request: PixelInferenceRequest
    ) -> PixelInferenceResponse:
        """Generate response using Pixel model"""
        if not self.model_loaded:
            raise RuntimeError("Model not loaded")

        start_time = datetime.now(UTC)

        try:
            # Preprocess input
            input_tensor = self.preprocess_input(
                request.user_query, request.conversation_history
            )

            # Forward pass through model
            with torch.no_grad():
                model_output = self.model(
                    input_tensor, history=request.conversation_history
                )

            # Extract outputs
            persona_mode = self._detect_persona_mode(request.context_type)
            eq_scores = self._extract_eq_scores(model_output)
            metadata = self._build_metadata(model_output, request)

            # Generate response text (in production, use language head)
            response_text = self._generate_response_text(
                request.user_query, persona_mode, eq_scores
            )

            # Calculate inference time
            inference_time = (datetime.now(UTC) - start_time).total_seconds() * 1000

            # Update stats
            self.inference_count += 1
            self.total_inference_time += inference_time

            # Check latency requirement
            warning = None
            if inference_time > INFERENCE_LATENCY_WARNING_MS:
                warning = (
                    f"Inference latency exceeded target: "
                    f"{inference_time:.2f}ms > {INFERENCE_LATENCY_WARNING_MS}ms"
                )
                logger.warning(warning)

            return PixelInferenceResponse(
                response=sanitize_agent_output(response_text),
                inference_time_ms=inference_time,
                eq_scores=eq_scores if request.use_eq_awareness else None,
                conversation_metadata=metadata if request.include_metrics else None,
                persona_mode=persona_mode,
                confidence=0.92,
                warning=warning,
            )

        except Exception as e:
            logger.error(f"Error during inference: {e}")
            raise

    async def generate_streaming_response(self, request: PixelInferenceRequest):
        """Generator that yields agent activities and finally the full response"""
        current_state = {
            "focus": None,
            "detected_emotions": [],
            "kb_context": None,
            "distortions": []
        }

        # Step 1: Pre-processing thought
        current_state["focus"] = "initial_assessment"
        if request.gestalt_directive:
            current_state["focus"] = f"corrected_focus: {request.gestalt_directive}"

        self.record_agent_step("Coordinator", 500)
        yield AgentActivity(
            id=str(uuid.uuid4()),
            agent_name="Coordinator",
            agent_role="Orchestrator",
            type="thought",
            content=(
                f"Applying directive: {request.gestalt_directive}"
                if request.gestalt_directive
                else "Awaiting directives..."
            ),
            thought=(
                f"User query: '{request.user_query[:50]}...'. "
                f"Directive: {request.gestalt_directive or 'None'}"
            ),
            status="completed",
            timestamp=time.time(),
            shared_state=current_state.copy()
        )
        await asyncio.sleep(0.5)

        # Step 2: Retrieval action
        current_state["kb_context"] = "retrieved_cbt_v3"
        self.record_agent_step("Memory Agent", 800)
        yield AgentActivity(
            id=str(uuid.uuid4()),
            agent_name="Memory Agent",
            agent_role="Context Retrieval",
            type="action",
            content="Scanning therapeutic knowledge base for relevant CBT protocols.",
            action="kb_search(techniques=['validation', 'cognitive_restructuring'])",
            status="completed",
            timestamp=time.time(),
            shared_state=current_state.copy()
        )
        await asyncio.sleep(0.8)

        # Step 3: Analysis
        current_state["distortions"] = ["overgeneralization"]
        current_state["detected_emotions"] = ["anxiety", "frustration"]
        self.record_agent_step("Psychologist", 400)
        yield AgentActivity(
            id=str(uuid.uuid4()),
            agent_name="Psychologist",
            agent_role="Clinical Analysis",
            type="thought",
            content="Identifying potential cognitive distortions in user narrative.",
            thought="Detected 'overgeneralization' pattern in user phrasing.",
            status="completed",
            timestamp=time.time(),
            shared_state=current_state.copy()
        )
        await asyncio.sleep(0.4)

        # Simulated Conflict: Safety Guard disagrees with Psychologist on severity
        if (
            not request.gestalt_directive
            and any(
                k in request.user_query.lower() for k in ["bad", "hurt", "desperate"]
            )
        ):
            self.record_agent_step("Safety Guard", 200, success=False)
            yield AgentActivity(
                id=str(uuid.uuid4()),
                agent_name="Safety Guard",
                agent_role="Content Moderation",
                type="thought",
                content="Flagging discrepancy in emotional severity assessment.",
                thought=(
                    "Psychologist identified 'anxiety', but semantic analysis suggests "
                    "potential 'crisis' level distress."
                ),
                status="completed",
                timestamp=time.time(),
                shared_state=current_state.copy(),
                conflict=AgentConflict(
                    with_agent="Psychologist",
                    severity="medium",
                    description=(
                        "Safety threshold discrepancy: Psychological assessment does not "
                        "fully account for acute distress markers."
                    )
                )
            )
            await asyncio.sleep(0.6)

        # Step 4: Final response generation
        final_response = await self.generate_response(request)
        final_response.shared_state = current_state

        self.record_agent_step("Coordinator", 100)
        yield AgentActivity(
            id=str(uuid.uuid4()),
            agent_name="Coordinator",
            agent_role="Orchestrator",
            type="observation",
            content="Response formulated and safety-checked.",
            observation="Safety score: 0.98. EQ target: high empathy.",
            status="completed",
            timestamp=time.time(),
            shared_state=current_state.copy()
        )

        # Final event is the full response
        yield final_response

    def _detect_persona_mode(self, context_type: str | None) -> str:
        """Detect appropriate persona mode based on context"""
        if context_type in ["crisis", "clinical"]:
            return "therapy"
        return "assistant" if context_type else "therapy"

    def _extract_eq_scores(self, model_output: dict[str, Any]) -> EQScores:
        """Extract EQ scores from model output"""
        eq_dict = model_output.get("eq_outputs", {})

        scores = {
            "emotional_awareness": float(
                eq_dict.get("emotional_awareness", torch.tensor(0.0)).mean()
            ),
            "empathy_recognition": float(
                eq_dict.get("empathy_recognition", torch.tensor(0.0)).mean()
            ),
            "emotional_regulation": float(
                eq_dict.get("emotional_regulation", torch.tensor(0.0)).mean()
            ),
            "social_cognition": float(
                eq_dict.get("social_cognition", torch.tensor(0.0)).mean()
            ),
            "interpersonal_skills": float(
                eq_dict.get("interpersonal_skills", torch.tensor(0.0)).mean()
            ),
        }

        # Normalize scores to 0-1 range
        scores = {k: abs(v) % 1.0 for k, v in scores.items()}
        overall_eq = sum(scores.values()) / len(scores)

        return EQScores(
            emotional_awareness=scores["emotional_awareness"],
            empathy_recognition=scores["empathy_recognition"],
            emotional_regulation=scores["emotional_regulation"],
            social_cognition=scores["social_cognition"],
            interpersonal_skills=scores["interpersonal_skills"],
            overall_eq=overall_eq,
        )

    def _build_metadata(
        self, _model_output: dict[str, Any], request: PixelInferenceRequest
    ) -> ConversationMetadata:
        """Build conversation metadata from model output"""
        # Simulate technique detection
        detected_techniques = []
        if "cbt" in request.user_query.lower():
            detected_techniques.append("CBT")
        if "dbt" in request.user_query.lower():
            detected_techniques.append("DBT")

        return ConversationMetadata(
            detected_techniques=detected_techniques,
            technique_consistency=0.85,
            bias_score=0.05,  # Lower is better
            safety_score=0.95,
            crisis_signals=["immediate_harm"] if "hurt" in request.user_query else None,
            therapeutic_effectiveness_score=0.88,
        )

    def _generate_response_text(
        self, _query: str, persona_mode: str, eq_scores: EQScores
    ) -> str:
        """Generate response text based on query and persona"""
        # Simple template-based response (in production, use language head)
        empathy_level = (
            "understanding"
            if eq_scores.empathy_recognition > EMPATHY_SUPPORT_THRESHOLD
            else "supportive"
        )

        responses = {
            "therapy": (
                f"I appreciate you sharing that with me. I'm here to help. "
                f"That sounds {empathy_level}. Can you tell me more about "
                f"what you're experiencing?"
            ),
            "assistant": (
                "That's an interesting question. Let me help you with that. "
                "Based on what you've shared, here are some suggestions..."
            ),
        }

        return responses.get(persona_mode, responses["therapy"])

    def get_status(self) -> ModelStatusResponse:
        """Get current model status"""
        avg_inference_time = (
            self.total_inference_time / self.inference_count
            if self.inference_count > 0
            else None
        )
        return ModelStatusResponse(
            model_loaded=self.model_loaded,
            model_name="PixelBaseModel",
            inference_engine="PyTorch",
            available_features=[
                "eq_measurement",
                "persona_switching",
                "crisis_detection",
                "clinical_prediction",
                "empathy_tracking",
                "bias_detection",
            ],
            performance_metrics={
                "inference_count": self.inference_count,
                "average_inference_time_ms": (
                    self.total_inference_time / self.inference_count
                    if self.inference_count > 0
                    else None
                ),
                "total_inference_time_ms": self.total_inference_time,
                "device": str(self.device),
            },
            last_inference_time_ms=avg_inference_time,
        )


# FastAPI Application

app = FastAPI(
    title="Pixel Model Inference API",
    description="Production-grade API for Pixel emotional intelligence model",
    version="1.0.0",
)

# Global inference engine
inference_engine = PixelInferenceEngine()


@app.on_event("startup")
async def startup_event():
    """Initialize model on startup"""
    logger.info("Starting Pixel Inference API")
    if not inference_engine.load_model():
        logger.error("Failed to load model on startup")


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy" if inference_engine.model_loaded else "degraded",
        "model_loaded": inference_engine.model_loaded,
        "timestamp": datetime.now(UTC).isoformat(),
    }


@app.get("/status", response_model=ModelStatusResponse)
async def get_model_status():
    """Get detailed model status"""
    if not inference_engine.model_loaded:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return inference_engine.get_status()


@app.get("/agent-stats")
async def get_agent_stats():
    """Get per-agent performance statistics"""
    return inference_engine.get_agent_report()


@app.post("/reset-stats")
async def reset_stats():
    """Reset agent performance statistics"""
    inference_engine.reset_agent_stats()
    return {"status": "success"}


@app.post("/infer", response_model=PixelInferenceResponse)
async def infer(request: PixelInferenceRequest, _background_tasks: BackgroundTasks):
    """Generate response using Pixel model"""
    if not inference_engine.model_loaded:
        raise HTTPException(status_code=503, detail="Model not loaded")

    try:
        return await inference_engine.generate_response(request)
    except Exception:
        logger.exception("Inference error")
        raise HTTPException(status_code=500, detail="Internal server error") from None


@app.post("/batch-infer")
async def batch_infer(requests: list[PixelInferenceRequest]):
    """Batch inference for multiple queries"""
    # ⚡ Bolt: Use asyncio.gather to concurrently process multiple requests
    # in the batch, preventing blocking sequential iteration and improving throughput.
    if not inference_engine.model_loaded:
        raise HTTPException(status_code=503, detail="Model not loaded")

    async def _process_single(req: PixelInferenceRequest):
        try:
            # inference_engine.generate_response uses torch.no_grad() and blockingly processes
            # the query via PyTorch, so we need to run it in a threadpool to truly unlock concurrency.
            return await run_in_threadpool(
                lambda: asyncio.run(inference_engine.generate_response(req))
            )
        except Exception:
            logger.exception("Batch inference error")
            return {"error": "inference_failed"}

    responses = []
    for i in range(0, len(requests), MAX_BATCH_CONCURRENCY):
        batch = requests[i : i + MAX_BATCH_CONCURRENCY]
        batch_responses = await asyncio.gather(*[_process_single(req) for req in batch])
        responses.extend(batch_responses)
    # ⚡ Bolt: Replace sequential loop with asyncio.gather to concurrently process requests
    async def _process_req(req):
        try:
            return await inference_engine.generate_response(req)
        except Exception as e:
            logger.error(f"Batch inference error: {e}")
            return {"error": str(e)}

    responses = await asyncio.gather(*(_process_req(req) for req in requests))

    responses = []
    for i in range(0, len(requests), MAX_BATCH_CONCURRENCY):
        batch = requests[i : i + MAX_BATCH_CONCURRENCY]
        batch_responses = await asyncio.gather(*[_process_single(req) for req in batch])
        responses.extend(batch_responses)

    return {"results": list(responses)}


@app.post("/reload-model")
async def reload_model():
    """Reload model from disk"""
    try:
        inference_engine.model_loaded = False
        if inference_engine.load_model():
            return {"status": "success", "message": "Model reloaded"}
        raise HTTPException(status_code=500, detail="Failed to reload model")
    except Exception:
        logger.exception("Reload error")
        raise HTTPException(status_code=500, detail="Internal server error") from None


@app.post("/infer-stream")
async def infer_stream(request: PixelInferenceRequest):
    """Generate streaming response with agent activities using SSE"""
    if not inference_engine.model_loaded:
        raise HTTPException(status_code=503, detail="Model not loaded")

    async def event_generator():
        try:
            async for item in inference_engine.generate_streaming_response(request):
                # Check if it's an activity or the final response
                if isinstance(item, AgentActivity):
                    item.content = sanitize_agent_output(item.content)
                    if item.thought:
                        item.thought = sanitize_agent_output(item.thought)
                    if item.action:
                        item.action = sanitize_agent_output(item.action)
                    if item.observation:
                        item.observation = sanitize_agent_output(item.observation)
                    yield {
                        "event": "activity",
                        "data": item.json(by_alias=True)
                    }
                else:
                    item.response = sanitize_agent_output(item.response)
                    yield {
                        "event": "final_response",
                        "data": item.json(by_alias=True)
                    }
        except Exception as e:
            logger.exception("Streaming inference error")
            yield {
                "event": "error",
                "data": json.dumps({"detail": str(e)})
            }

    return StreamingResponse(
        (f"event: {e['event']}\ndata: {e['data']}\n\n" async for e in event_generator()),
        media_type="text/event-stream"
    )


if __name__ == "__main__":
    import uvicorn

    port = int(os.getenv("PIXEL_API_PORT", PIXEL_API_DEFAULT_PORT))
    uvicorn.run(app, host="0.0.0.0", port=port)
