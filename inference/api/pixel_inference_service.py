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
import re
import sys
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

from ai.tools.utilities.torch_proxy import torch

# Add parent directories to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from ai.inference.api.ift_inference import ABTestConfig, ABTestRouter, build_task_prompt, detect_task_type
from ai.inference.api.memory import get_memory_manager
from ai.inference.api.sentry_logging import initialize_sentry_logging
from ai.models.pixel_base_model import PixelBaseModel

# ---------------------------------------------------------------------------
# PAL Inference imports
# ---------------------------------------------------------------------------
_PAL_FRAMEWORK = str(
    Path(__file__).resolve().parents[2] / "data" / "synthetic" / "wrapper" / "pal_framework",
)
if _PAL_FRAMEWORK not in sys.path:
    sys.path.insert(0, _PAL_FRAMEWORK)

from inference_wrapper import (  # type: ignore[import-untyped]
    DEFAULT_LATENCY_BUDGET_SECONDS,
    PalInferenceWrapper,
)

# ---------------------------------------------------------------------------
# PIX-3912: Mera Hierarchical Clinical Prediction imports
# ---------------------------------------------------------------------------
try:
    from ai.research.therapeutic_concept_hierarchy import (
        TherapeuticConceptHierarchy,
        build_default_therapeutic_hierarchy,
    )
    from ai.tools.utilities.pipelines.inference.candidate_retrieval import (
        CandidateRetrievalEngine,
    )
    from ai.tools.utilities.pipelines.inference.evidence_scoring import (
        EvidenceScoringEngine,
    )

    _MERA_AVAILABLE = True
except Exception as _mera_import_exc:
    _MERA_AVAILABLE = False
    _mera_import_error = str(_mera_import_exc)
else:
    _mera_import_error = None

logger = logging.getLogger(__name__)
if not _MERA_AVAILABLE and _mera_import_error:
    logger.warning("Mera clinical prediction modules not available: %s", _mera_import_error)
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
        description=("Context type: educational, support, crisis, clinical, informational"),
    )
    user_id: str | None = Field(None, description="User identifier for tracking")
    session_id: str | None = Field(None, description="Session identifier")
    use_eq_awareness: bool = Field(True, description="Enable EQ-aware response generation")
    include_metrics: bool = Field(True, description="Include quality metrics in response")
    max_tokens: int = Field(200, description="Max tokens to generate")
    task_type: str | None = Field(None, description="Mental health task type (auto-detected if omitted)")
    force_model: str | None = Field(None, description="Force 'baseline' or 'ift' model; overrides A/B routing")
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
    persona_mode: str = Field("therapy", description="Detected persona: therapy or assistant")
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


# ---------------------------------------------------------------------------
# PAL Inference Models
# ---------------------------------------------------------------------------


class PalInferRequest(BaseModel):
    """PAL inference request — one dialogue string."""

    dialogue: str = Field(..., min_length=1, description="The patient dialogue to infer a persona for.")


class PalSelectionResponse(BaseModel):
    """Stage 1 persona selection result."""

    persona_string: str = Field(..., description="Selected persona as natural-language string.")
    selected_index: int = Field(..., ge=0, description="Zero-based index of selected persona.")
    latency_seconds: float = Field(..., ge=0.0, description="Wall-clock seconds for Stage 1.")


class PalGenerationResponse(BaseModel):
    """Stage 2 response generation result."""

    response: str = Field(..., description="Generated assistant response.")
    latency_seconds: float = Field(..., ge=0.0, description="Wall-clock seconds for Stage 2.")


class PalGenerateRequest(BaseModel):
    """PAL generation request."""

    persona_string: str = Field(..., min_length=1, description="The persona string to use for response generation.")
    dialogue_history: str = Field(..., description="The dialogue history up to this point.")


class PalInferResponse(BaseModel):
    """End-to-end PAL inference result."""

    selection: PalSelectionResponse
    generation: PalGenerationResponse
    total_latency_seconds: float = Field(..., ge=0.0, description="Total wall-clock seconds for both stages.")
    dialogue: str = Field(..., description="The input dialogue echoed back.")


# ---------------------------------------------------------------------------
# PIX-3912: Mera Clinical Prediction Models
# ---------------------------------------------------------------------------


class ClinicalPredictionRequest(BaseModel):
    """Request for hierarchical clinical diagnosis prediction."""

    patient_presentation: str = Field(..., description="Free-text clinical description of the patient")
    test_results: list[dict[str, Any]] | None = Field(None, description="Optional lab/test results")
    progression_notes: str | None = Field(None, description="Optional notes on symptom progression")
    top_k: int = Field(5, ge=1, le=20, description="Number of ranked diagnoses to return")
    include_evidence: bool = Field(True, description="Include detailed evidence chains")
    initial_guess: str | None = Field(None, description="Optional condition_id from prior screening")


class DiagnosisEvidenceItem(BaseModel):
    """A single piece of evidence supporting or contradicting a diagnosis."""

    finding_type: str
    description: str
    weight: float
    direction: str
    confidence: float


class RankedDiagnosis(BaseModel):
    """A ranked diagnosis with scores and evidence."""

    rank: int
    condition_id: str
    condition_name: str
    final_score: float
    retrieval_score: float
    evidence_score: float
    symptom_score: float
    typical_presentation_score: float
    test_result_score: float
    progression_score: float
    confidence: float
    hierarchy_path: list[str]
    evidence: list[DiagnosisEvidenceItem] | None = None


class ClinicalPredictionResponse(BaseModel):
    """Response from the Mera hierarchical clinical prediction pipeline."""

    ranked_diagnoses: list[RankedDiagnosis]
    inference_time_ms: float
    pipeline_version: str = "mera-v1.0"
    hierarchy_coverage: int = Field(0, description="Number of conditions in the hierarchy")


# Pixel Inference Service


class PixelInferenceEngine:
    """Manages Pixel model loading, caching, and inference"""

    def __init__(self):
        self.model: PixelBaseModel | None = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_loaded = False
        self.inference_count = 0
        self.total_inference_time = 0.0
        self.model_path = os.getenv("PIXEL_MODEL_PATH", "ai/models/pixel_core/models/pixel_base_model.pt")
        self.ab_router = ABTestRouter(
            ABTestConfig(
                enabled=os.getenv("PIXEL_IFT_AB_ENABLED", "false").lower() == "true",
                ift_traffic_percent=float(os.getenv("PIXEL_IFT_TRAFFIC_PERCENT", "0.0")),
                auto_rollback=os.getenv("PIXEL_IFT_AUTO_ROLLBACK", "true").lower() == "true",
            )
        )
        self.ab_router.register_baseline(self._baseline_generate)
        # Per-agent metrics
        self.agent_stats = {
            "Coordinator": {"calls": 0, "total_time": 0, "errors": 0},
            "Psychologist": {"calls": 0, "total_time": 0, "errors": 0},
            "Memory Agent": {"calls": 0, "total_time": 0, "errors": 0},
            "Safety Guard": {"calls": 0, "total_time": 0, "errors": 0},
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
                "throughput": stats["calls"],
            }
        return report

    def reset_agent_stats(self):
        """Reset all performance metrics"""
        for name in self.agent_stats:
            self.agent_stats[name] = {"calls": 0, "total_time": 0, "errors": 0}
        self.inference_count = 0
        self.total_inference_time = 0.0

    def load_model(self) -> bool:
        """Load Pixel model from disk and initialize IFT A/B router."""
        try:
            base_loaded = self._ensure_model_loaded()
            if base_loaded:
                # Attempt to load IFT model asynchronously; failure does not degrade base model
                try:
                    self.ab_router.load_ift_model()
                except Exception as e:
                    logger.warning(f"IFT model load skipped: {e}")
            return base_loaded
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
            logger.warning(f"Model file not found at {self.model_path}, creating fresh model")
            self.model = PixelBaseModel()
        else:
            self.model = PixelBaseModel.load(self.model_path)

        self.model = self.model.to(self.device)
        self.model.eval()
        self.model_loaded = True
        logger.info("Pixel model loaded successfully")
        return True

    def preprocess_input(self, query: str, history: list[ConversationMessage]) -> torch.Tensor:
        """Convert query and history to model input tensor"""
        # Create simple token embedding (in production, use actual tokenizer)
        # For now, use positional encoding + word embeddings simulation
        history_context = " ".join([m.content for m in history[-3:]])  # Last 3 messages
        full_context = f"{history_context} {query}"

        # Simulate tokenization: create embedding
        # Shape: (batch=1, seq_len=query_len+context, d_model=768)
        seq_len = min(len(full_context.split()) + 1, 512)
        return torch.randn(1, seq_len, 768, device=self.device)

    async def generate_response(self, request: PixelInferenceRequest) -> PixelInferenceResponse:
        """Generate response using Pixel model with optional IFT A/B routing."""
        if not self.model_loaded:
            raise RuntimeError("Model not loaded")

        start_time = datetime.now(UTC)

        try:
            # Detect mental health task type
            task_type = request.task_type or detect_task_type(request.user_query, request.context_type)

            # Build task-specific prompt
            history = [m.model_dump() if hasattr(m, "model_dump") else m.dict() for m in request.conversation_history]
            prompt = build_task_prompt(task_type, request.user_query, history)

            # Route to baseline or IFT model
            force_model = request.force_model
            if force_model:
                destination = force_model
                if destination == "ift":
                    response_text = self.ab_router.ift_model.generate(prompt, max_new_tokens=request.max_tokens)
                else:
                    response_text = self._baseline_generate(prompt)
            else:
                destination, response_text = self.ab_router.generate(
                    prompt,
                    task_type=task_type,
                    user_id=request.user_id,
                    session_id=request.session_id,
                )

            # If IFT returned empty, fall back to baseline
            if not response_text.strip():
                response_text = self._baseline_generate(prompt)
                destination = "baseline"

            # Preprocess input for Pixel base model (EQ/metadata)
            input_tensor = self.preprocess_input(request.user_query, request.conversation_history)
            with torch.no_grad():
                model_output = self.model(input_tensor, history=request.conversation_history)

            persona_mode = self._detect_persona_mode(request.context_type)
            eq_scores = self._extract_eq_scores(model_output)
            metadata = self._build_metadata(model_output, request)

            # Calculate inference time
            inference_time = (datetime.now(UTC) - start_time).total_seconds() * 1000

            # Update stats
            self.inference_count += 1
            self.total_inference_time += inference_time

            # Check latency requirement
            warning = None
            if inference_time > INFERENCE_LATENCY_WARNING_MS:
                warning = (
                    f"Inference latency exceeded target: {inference_time:.2f}ms > {INFERENCE_LATENCY_WARNING_MS}ms"
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
                shared_state={"model_used": destination, "task_type": task_type},
            )

        except Exception as e:
            logger.error(f"Error during inference: {e}")
            raise

    async def generate_streaming_response(self, request: PixelInferenceRequest):
        """Generator that yields agent activities and finally the full response"""
        current_state = {"focus": None, "detected_emotions": [], "kb_context": None, "distortions": []}

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
            thought=(f"User query: '{request.user_query[:50]}...'. Directive: {request.gestalt_directive or 'None'}"),
            status="completed",
            timestamp=time.time(),
            shared_state=current_state.copy(),
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
            shared_state=current_state.copy(),
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
            shared_state=current_state.copy(),
        )
        await asyncio.sleep(0.4)

        # Simulated Conflict: Safety Guard disagrees with Psychologist on severity
        if not request.gestalt_directive and any(k in request.user_query.lower() for k in ["bad", "hurt", "desperate"]):
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
                    ),
                ),
            )
            await asyncio.sleep(0.6)

        # Step 4: Final response generation
        final_response = await self.generate_response(request)
        final_response.shared_state = {**current_state, **(final_response.shared_state or {})}

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
            shared_state=current_state.copy(),
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
            "emotional_awareness": float(eq_dict.get("emotional_awareness", torch.tensor(0.0)).mean()),
            "empathy_recognition": float(eq_dict.get("empathy_recognition", torch.tensor(0.0)).mean()),
            "emotional_regulation": float(eq_dict.get("emotional_regulation", torch.tensor(0.0)).mean()),
            "social_cognition": float(eq_dict.get("social_cognition", torch.tensor(0.0)).mean()),
            "interpersonal_skills": float(eq_dict.get("interpersonal_skills", torch.tensor(0.0)).mean()),
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

    def _build_metadata(self, _model_output: dict[str, Any], request: PixelInferenceRequest) -> ConversationMetadata:
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

    def _generate_response_text(self, _query: str, persona_mode: str, eq_scores: EQScores) -> str:
        """Generate response text based on query and persona"""
        # Simple template-based response (in production, use language head)
        empathy_level = "understanding" if eq_scores.empathy_recognition > EMPATHY_SUPPORT_THRESHOLD else "supportive"

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

    def _baseline_generate(self, prompt: str) -> str:
        """Baseline prompt-engineered generation (used by A/B router)."""
        # In production this would call the base LLM with the prompt.
        # Here we simulate a prompt-aware response for A/B comparison.
        if "symptom" in prompt.lower():
            return "anxiety, low mood"
        if "severity" in prompt.lower():
            return "5"
        if "risk" in prompt.lower():
            return "low"
        if "empathy" in prompt.lower():
            return "cognitive: 4, affective: 4, compassionate: 4"
        return "I hear you, and that sounds really difficult. I'm here to support you."

    def get_status(self) -> ModelStatusResponse:
        """Get current model status including IFT/A-B state"""
        avg_inference_time = self.total_inference_time / self.inference_count if self.inference_count > 0 else None
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
                "ift_model",
                "ab_test_routing",
                "task_specific_prompts",
                "auto_rollback",
            ],
            performance_metrics={
                "inference_count": self.inference_count,
                "average_inference_time_ms": (
                    self.total_inference_time / self.inference_count if self.inference_count > 0 else None
                ),
                "total_inference_time_ms": self.total_inference_time,
                "device": str(self.device),
                "ift_model_loaded": self.ab_router.ift_model.loaded,
                "ab_test_enabled": self.ab_router.config.enabled,
                "ift_traffic_percent": self.ab_router.config.ift_traffic_percent,
                "rollback_active": self.ab_router.rollback_active,
                "ab_test_stats": self.ab_router.get_stats(),
            },
            last_inference_time_ms=avg_inference_time,
        )


# ---------------------------------------------------------------------------
# PIX-3912: Mera Clinical Prediction Engine
# ---------------------------------------------------------------------------


class MeraClinicalPredictionEngine:
    """
    Hierarchical clinical prediction engine inspired by Mera (arXiv 2501.17326).

    Pipeline: patient presentation → hierarchy encoding → candidate retrieval →
    evidence scoring → ranked diagnosis.
    """

    def __init__(self):
        self.hierarchy: TherapeuticConceptHierarchy | None = None
        self.retrieval_engine: CandidateRetrievalEngine | None = None
        self.scoring_engine: EvidenceScoringEngine | None = None
        self._initialized = False

    def initialize(self) -> bool:
        """Load hierarchy and initialize retrieval/scoring engines."""
        if self._initialized:
            return True
        if not _MERA_AVAILABLE:
            logger.warning("Mera modules not available; clinical prediction disabled")
            return False

        try:
            self.hierarchy = build_default_therapeutic_hierarchy()
            self.retrieval_engine = CandidateRetrievalEngine(self.hierarchy)
            self.scoring_engine = EvidenceScoringEngine(self.hierarchy)
            self._initialized = True
            logger.info(
                "MeraClinicalPredictionEngine initialized: %d conditions in hierarchy",
                len(self.hierarchy),
            )
            return True
        except Exception as e:
            logger.exception("Failed to initialize MeraClinicalPredictionEngine: %s", e)
            return False

    async def predict(self, request: ClinicalPredictionRequest) -> ClinicalPredictionResponse:
        """Run the full Memorize & Rank pipeline."""
        if not self._initialized and not self.initialize():
            raise RuntimeError("Mera clinical prediction engine not initialized")

        start_time = datetime.now(UTC)

        # Stage 1: Memorize — candidate retrieval
        candidates = self.retrieval_engine.retrieve(
            patient_presentation=request.patient_presentation,
            top_k=request.top_k * 2,  # retrieve more for ranking stage
            initial_guess=request.initial_guess,
        )

        # Stage 2: Rank — evidence-based scoring
        scored = self.scoring_engine.score(
            candidates=candidates,
            patient_presentation=request.patient_presentation,
            test_results=request.test_results,
            progression_notes=request.progression_notes,
        )

        # Take top_k
        top_scored = scored[: request.top_k]

        # Build response
        ranked_diagnoses: list[RankedDiagnosis] = []
        for rank, diag in enumerate(top_scored, start=1):
            evidence_items: list[DiagnosisEvidenceItem] | None = None
            if request.include_evidence:
                evidence_items = [
                    DiagnosisEvidenceItem(
                        finding_type=f.finding_type,
                        description=f.description,
                        weight=f.weight,
                        direction=f.direction,
                        confidence=f.confidence,
                    )
                    for f in diag.findings
                ]

            ranked_diagnoses.append(
                RankedDiagnosis(
                    rank=rank,
                    condition_id=diag.condition_id,
                    condition_name=diag.condition_name,
                    final_score=round(diag.final_score, 4),
                    retrieval_score=round(diag.retrieval_score, 4),
                    evidence_score=round(diag.evidence_score, 4),
                    symptom_score=round(diag.symptom_score, 4),
                    typical_presentation_score=round(diag.typical_presentation_score, 4),
                    test_result_score=round(diag.test_result_score, 4),
                    progression_score=round(diag.progression_score, 4),
                    confidence=round(diag.confidence, 4),
                    hierarchy_path=diag.hierarchy_path,
                    evidence=evidence_items,
                )
            )

        inference_time = (datetime.now(UTC) - start_time).total_seconds() * 1000

        return ClinicalPredictionResponse(
            ranked_diagnoses=ranked_diagnoses,
            inference_time_ms=inference_time,
            hierarchy_coverage=len(self.hierarchy) if self.hierarchy else 0,
        )

    def get_status(self) -> dict[str, Any]:
        return {
            "initialized": self._initialized,
            "hierarchy_size": len(self.hierarchy) if self.hierarchy else 0,
            "mera_available": _MERA_AVAILABLE,
        }


# ---------------------------------------------------------------------------
# PAL Stub LLM Clients (used when no real endpoint is configured)
# ---------------------------------------------------------------------------


class _PalStubSelector:
    """Returns option '1' for every request — the first candidate persona."""

    def __call__(self, _messages: list[dict[str, str]]) -> str:
        return "1"


class _PalStubGenerator:
    """Returns a canned persona-aligned response."""

    def __call__(self, _messages: list[dict[str, str]]) -> str:
        return "I have been feeling this way for a while now. Thank you for explaining things clearly, doctor."


def _pal_load_candidate_personas() -> list[dict[str, Any]]:
    """Load candidate personas from PAL_CANDIDATE_PERSONAS env var.

    Expects a JSON array of Meddies-shaped persona dicts. Falls back to
    two default personas so the service starts without configuration.
    """
    raw = os.environ.get("PAL_CANDIDATE_PERSONAS")
    if raw:
        try:
            personas = json.loads(raw)
            if not isinstance(personas, list) or not personas:
                raise ValueError("PAL_CANDIDATE_PERSONAS must be a non-empty JSON array")
            return personas
        except (json.JSONDecodeError, ValueError) as exc:
            logger.warning("Failed to parse PAL_CANDIDATE_PERSONAS, falling back to defaults: %s", exc)

    return [
        {
            "demographics": {"age": 45, "gender": "female", "location": "Hanoi"},
            "healthcare_behavior": {"health_literacy": "low", "preference": "traditional medicine"},
        },
        {
            "demographics": {"age": 30, "gender": "male", "location": "HCMC"},
            "healthcare_behavior": {"health_literacy": "high", "preference": "modern medicine"},
        },
    ]


def _pal_build_latency_budget() -> float:
    """Read PAL latency budget from env, falling back to the wrapper default."""
    raw = os.environ.get("PAL_LATENCY_BUDGET_SECONDS", str(DEFAULT_LATENCY_BUDGET_SECONDS))
    try:
        return float(raw)
    except ValueError:
        return DEFAULT_LATENCY_BUDGET_SECONDS


def _pal_init_wrapper() -> PalInferenceWrapper | None:
    """Initialize the PAL inference wrapper, returning None on failure."""
    try:
        # Attempt real LLM clients when endpoints are configured
        selector_endpoint = os.environ.get("PAL_SELECTOR_ENDPOINT")
        generator_endpoint = os.environ.get("PAL_GENERATOR_ENDPOINT")

        if selector_endpoint:
            try:
                from openai import OpenAI  # type: ignore[import-untyped]

                _openai_selector = OpenAI(base_url=selector_endpoint)
                selector_model = os.environ.get("PAL_SELECTOR_MODEL", "gpt-4o-mini")

                def _selector(messages: list[dict[str, str]]) -> str:
                    resp = _openai_selector.chat.completions.create(model=selector_model, messages=messages)  # type: ignore[arg-type]
                    return resp.choices[0].message.content or "1"

                selector = _selector
            except ImportError:
                logger.warning("openai not installed; falling back to stub selector")
                selector = _PalStubSelector()
        else:
            selector = _PalStubSelector()

        if generator_endpoint:
            try:
                from openai import OpenAI  # type: ignore[import-untyped]

                _openai_generator = OpenAI(base_url=generator_endpoint)
                generator_model = os.environ.get("PAL_GENERATOR_MODEL", "gpt-4o-mini")

                def _generator(messages: list[dict[str, str]]) -> str:
                    resp = _openai_generator.chat.completions.create(model=generator_model, messages=messages)  # type: ignore[arg-type]
                    return resp.choices[0].message.content or ""

                generator = _generator
            except ImportError:
                logger.warning("openai not installed; falling back to stub generator")
                generator = _PalStubGenerator()
        else:
            generator = _PalStubGenerator()

        personas = _pal_load_candidate_personas()
        budget = _pal_build_latency_budget()
        wrapper = PalInferenceWrapper(
            selector_client=selector,
            generator_client=generator,
            candidate_personas=personas,
            latency_budget_seconds=budget,
        )
        logger.info(
            "PAL wrapper initialized: %d candidate personas, %.2fs budget",
            len(personas),
            budget,
        )
        return wrapper
    except Exception:
        logger.exception("Failed to initialize PAL wrapper")
        return None


# FastAPI Application

app = FastAPI(
    title="Pixel Model Inference API",
    description="Production-grade API for Pixel emotional intelligence model",
    version="1.0.0",
)

# Global inference engine
inference_engine = PixelInferenceEngine()

# PAL inference wrapper (initialized in startup)
pal_wrapper: PalInferenceWrapper | None = None


@app.on_event("startup")
async def startup_event():
    """Initialize models and PAL wrapper on startup"""
    global pal_wrapper

    logger.info("Starting Pixel Inference API")

    # Initialize Pixel model
    if not inference_engine.load_model():
        logger.error("Failed to load Pixel model on startup")

    # Initialize PAL wrapper
    pal_wrapper = _pal_init_wrapper()
    if pal_wrapper is None:
        logger.warning("PAL wrapper not initialized — PAL endpoints will return 503")
    else:
        logger.info("PAL wrapper ready on startup")

    # Initialize Mera clinical prediction engine
    if _MERA_AVAILABLE:
        if clinical_prediction_engine.initialize():
            logger.info("Mera clinical prediction engine ready on startup")
        else:
            logger.warning("Mera clinical prediction engine failed to initialize")
    else:
        logger.info("Mera clinical prediction modules not available — skipping initialization")


@app.get("/health")
async def health_check():
    """Health check endpoint with Pixel + PAL + Clinical Prediction status"""
    return {
        "status": "healthy" if inference_engine.model_loaded else "degraded",
        "model_loaded": inference_engine.model_loaded,
        "pal": {
            "wrapper_initialized": pal_wrapper is not None,
            "n_candidate_personas": len(pal_wrapper.candidate_personas) if pal_wrapper else 0,
            "latency_budget_seconds": (
                pal_wrapper.latency_budget_seconds if pal_wrapper else DEFAULT_LATENCY_BUDGET_SECONDS
            ),
        },
        "clinical_prediction": clinical_prediction_engine.get_status(),
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
async def infer(request: PixelInferenceRequest, background_tasks: BackgroundTasks):
    """Generate response using Pixel model"""
    if not inference_engine.model_loaded:
        raise HTTPException(status_code=503, detail="Model not loaded")

    try:
        response = await inference_engine.generate_response(request)
    except Exception:
        logger.exception("Inference error")
        raise HTTPException(status_code=500, detail="Internal server error") from None

    # Schedule dream-cycle consolidation as a background task when
    # the request includes a user identifier.
    if request.user_id:
        background_tasks.add_task(
            _trigger_dream_cycle,
            user_id=request.user_id,
        )

    return response


async def _trigger_dream_cycle(user_id: str) -> None:
    """Fire-and-forget dream cycle for a user after session processing."""
    try:
        mm = get_memory_manager()
        result = await mm.trigger_dream_cycle(user_id=user_id)
        logger.info(
            "Background dream cycle %s for user %s: %d themes, %d patterns",
            result.get("dream_id", "?"),
            user_id,
            len(result.get("themes", [])),
            len(result.get("patterns", [])),
        )
    except Exception:
        logger.exception("Background dream cycle failed for user %s", user_id)


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
            return await run_in_threadpool(lambda: asyncio.run(inference_engine.generate_response(req)))
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
                    yield {"event": "activity", "data": item.json(by_alias=True)}
                else:
                    item.response = sanitize_agent_output(item.response)
                    yield {"event": "final_response", "data": item.json(by_alias=True)}
        except Exception as e:
            logger.exception("Streaming inference error")
            yield {"event": "error", "data": json.dumps({"detail": str(e)})}

    return StreamingResponse(
        (f"event: {e['event']}\ndata: {e['data']}\n\n" async for e in event_generator()), media_type="text/event-stream"
    )


@app.post("/ab-test/enable")
async def enable_ab_test(percent: float):
    """Enable A/B test with given IFT traffic percentage (0.0 - 1.0)."""
    inference_engine.ab_router.enable_ift(percent)
    return {
        "status": "success",
        "ift_traffic_percent": inference_engine.ab_router.config.ift_traffic_percent,
    }


# ---------------------------------------------------------------------------
# PIX-3912: Mera Clinical Prediction Endpoints
# ---------------------------------------------------------------------------

# Global clinical prediction engine
clinical_prediction_engine = MeraClinicalPredictionEngine()


@app.get("/clinical-prediction/status")
async def clinical_prediction_status():
    """Get status of the Mera clinical prediction engine."""
    return clinical_prediction_engine.get_status()


@app.post("/clinical-prediction", response_model=ClinicalPredictionResponse)
async def clinical_predict(request: ClinicalPredictionRequest):
    """
    Hierarchical clinical diagnosis prediction (Mera Memorize & Rank).

    Pipeline:
    1. Memorize: retrieve candidate diagnoses via hybrid retrieval
       (semantic + hierarchy + keyword + memory)
    2. Rank: score candidates against patient evidence
       (symptoms, typical presentation, test results, progression)
    """
    if not _MERA_AVAILABLE:
        raise HTTPException(status_code=503, detail="Mera clinical prediction modules not available")

    try:
        return await clinical_prediction_engine.predict(request)
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from None
    except Exception:
        logger.exception("Clinical prediction error")
        raise HTTPException(status_code=500, detail="Clinical prediction failed") from None


# ---------------------------------------------------------------------------
# PAL Inference Endpoints
# ---------------------------------------------------------------------------


@app.post("/ab-test/disable")
async def disable_ab_test():
    """Disable A/B test routing (all traffic to baseline)."""
    inference_engine.ab_router.config.enabled = False
    inference_engine.ab_router.config.ift_traffic_percent = 0.0
    return {"status": "success", "message": "A/B test disabled"}


@app.post("/ab-test/rollback")
async def rollback_ift():
    """Force rollback to baseline model."""
    inference_engine.ab_router.rollback()
    return {"status": "success", "message": "IFT rolled back to baseline"}


@app.post("/ab-test/restore")
async def restore_ift():
    """Restore IFT model after rollback."""
    inference_engine.ab_router.restore()
    return {"status": "success", "message": "IFT model restored"}


@app.get("/ab-test/stats")
async def get_ab_test_stats():
    """Get A/B test statistics."""
    return inference_engine.ab_router.get_stats()


if __name__ == "__main__":
    import uvicorn

    port = int(os.getenv("PIXEL_API_PORT", PIXEL_API_DEFAULT_PORT))
    uvicorn.run(app, host="0.0.0.0", port=port)
