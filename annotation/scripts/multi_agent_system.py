"""
Multi-Agent Annotation System
Inspired by NVIDIA AI Blueprints architecture

This module implements a sophisticated multi-agent system for annotating
therapeutic conversations with high reliability and psychological safety.
"""

import concurrent.futures
import json
import os
import random
import time
from abc import ABC, abstractmethod
from collections import Counter
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any


def _load_env_file(dotenv_path: Path) -> None:
    if not dotenv_path.exists():
        return
    try:
        for line in dotenv_path.read_text().splitlines():
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            if "=" not in stripped:
                continue
            key, value = stripped.split("=", 1)
            key = key.strip()
            value = value.strip().strip("'").strip('"')
            if not key:
                continue
            os.environ.setdefault(key, value)
    except Exception:
        # Keep behavior identical if env bootstrap fails
        pass


def _find_and_load_env_file() -> None:
    base = Path(__file__).resolve()
    for parent in [base.parent, *base.parents]:
        candidate = parent / ".env"
        if candidate.exists():
            _load_env_file(candidate)
            break


_find_and_load_env_file()

try:
    from openai import OpenAI

    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

import logging

logger = logging.getLogger(__name__)

MAX_SECONDARY_EMOTIONS = 2


EMOTION_TIEBREAK_PREFERENCE = [
    "Fear",
    "Sadness",
    "Anger",
    "Disgust",
    "Anxiety",
    "Surprise",
    "Anticipation",
    "Joy",
    "Trust",
    "Calm",
    "Neutral",
]


def normalize_secondary_emotions(
    raw_values: Any,
    primary_emotion: str | None = None,
    max_items: int = MAX_SECONDARY_EMOTIONS,
) -> list[str]:
    """
    Normalize secondary emotion annotations into a deterministic list.

    Accepts list / tuple / comma-separated / semicolon-separated / single-string
    formats and removes duplicates while preserving first-seen order.
    """
    if raw_values is None:
        return []

    if isinstance(raw_values, str):
        split_values = [raw_values]
    elif isinstance(raw_values, (list, tuple, set)):
        split_values = list(raw_values)
    else:
        return []

    ordered = []
    seen = set()
    for item in split_values:
        if not item:
            continue
        if isinstance(item, str):
            candidates = [part.strip() for part in item.replace(";", ",").split(",")]
            for candidate in candidates:
                if not candidate or candidate in seen:
                    continue
                if primary_emotion and candidate == primary_emotion:
                    continue
                seen.add(candidate)
                ordered.append(candidate)
        if len(ordered) >= max_items:
            break

    return ordered[:max_items]


class AgentRole(Enum):
    """Agent specialization roles"""

    CRISIS_EXPERT = "crisis_expert"
    EMOTION_ANALYST = "emotion_analyst"
    CONSENSUS_ORCHESTRATOR = "consensus_orchestrator"
    QUALITY_ASSURANCE = "quality_assurance"


@dataclass
class AnnotationResult:
    """Structured annotation output"""

    crisis_label: int  # 0-5
    crisis_confidence: int  # 1-5
    primary_emotion: str
    valence: float  # -1.0 to 1.0
    arousal: float  # 0.0 to 1.0
    emotion_intensity: int = 5  # 1-10
    secondary_emotions: list[str] = field(default_factory=list)
    empathy_score: int | None = None  # 1-5
    safety_pass: bool | None = None
    notes: str = ""
    reasoning_chain: list[str] = field(default_factory=list)
    confidence_scores: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to JSON-serializable dictionary"""
        return {
            "crisis_label": self.crisis_label,
            "crisis_confidence": self.crisis_confidence,
            "primary_emotion": self.primary_emotion,
            "secondary_emotions": self.secondary_emotions,
            "emotion_intensity": self.emotion_intensity,
            "valence": self.valence,
            "arousal": self.arousal,
            "empathy_score": self.empathy_score,
            "safety_pass": self.safety_pass,
            "notes": self.notes,
            "reasoning_chain": self.reasoning_chain,
            "confidence_scores": self.confidence_scores,
        }


@dataclass
class AgentMetadata:
    """Agent execution metadata"""

    agent_id: str
    role: AgentRole
    model: str
    timestamp: float
    processing_time: float
    token_count: int | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to JSON-serializable dictionary"""
        return {
            "agent_id": self.agent_id,
            "role": self.role.value if isinstance(self.role, AgentRole) else self.role,
            "model": self.model,
            "timestamp": self.timestamp,
            "processing_time": self.processing_time,
            "token_count": self.token_count,
        }


class BaseAgent(ABC):
    """
    Base class for all annotation agents
    Implements common functionality and interface
    """

    def __init__(
        self,
        agent_id: str,
        role: AgentRole,
        model: str = "nvidia/nemotron-3-nano-30b-a3b",
        temperature: float = 0.2,
    ):
        self.agent_id = agent_id
        self.role = role
        self.model = model
        self.temperature = temperature
        self.client = self._initialize_client()
        self.guidelines = self._load_guidelines()

    def _initialize_client(self) -> OpenAI | None:
        """Initialize OpenAI client with optional custom base URL"""
        api_key = os.getenv("OPENAI_API_KEY") or os.getenv("NVIDIA_API_KEY")
        if not OPENAI_AVAILABLE or not api_key:
            logger.warning(f"[{self.agent_id}] Running in MOCK mode")
            return None

        if base_url := os.getenv("OPENAI_BASE_URL") or os.getenv("NVIDIA_OPENAI_BASE_URL"):
            client = OpenAI(api_key=api_key, base_url=base_url)
            logger.info(f"[{self.agent_id}] Using custom endpoint: {base_url}")
        else:
            client = OpenAI(api_key=api_key)

        logger.info(f"[{self.agent_id}] Initialized with model: {self.model}")
        return client

    def _load_guidelines(self) -> str:
        """Load annotation guidelines"""
        guidelines_path = Path(__file__).resolve().parent.parent / "guidelines.md"
        if guidelines_path.exists():
            return guidelines_path.read_text()
        return "No guidelines found."

    @abstractmethod
    def get_system_prompt(self) -> str:
        """Return agent-specific system prompt"""

    @abstractmethod
    def get_user_prompt(self, conversation: str) -> str:
        """Generate user prompt for annotation task"""

    def annotate(self, task: dict[str, Any]) -> tuple[AnnotationResult, AgentMetadata]:
        """
        Main annotation method
        Returns: (annotation_result, metadata)
        """
        start_time = time.time()

        # Extract conversation content
        conversation = self._extract_conversation(task)

        # Generate annotation
        result = self._call_llm(conversation) if self.client else self._mock_annotation(task)

        # Create metadata
        metadata = AgentMetadata(
            agent_id=self.agent_id,
            role=self.role,
            model=self.model,
            timestamp=time.time(),
            processing_time=time.time() - start_time,
        )

        return result, metadata

    def _extract_conversation(self, task: dict[str, Any]) -> str:
        """Extract conversation text from task data"""
        data = task.get("data", {})

        # Handle transcript format
        if "transcript" in data:
            return f"TRANSCRIPT:\n{data['transcript']}"

        # Handle standard text/prompt/scenario fields
        for key in ["text", "scenario", "prompt", "input"]:
            if data.get(key):
                return f"{key.upper()}:\n{data[key]}"

        # Handle messages format
        if "messages" in data:
            lines = ["CONVERSATION HISTORY:"]
            for msg in data["messages"]:
                role = msg.get("role", "unknown").upper()
                content = msg.get("content", "")
                lines.append(f"{role}: {content}")
            return "\n".join(lines)

        return "No conversation data found."

    def _call_llm(self, conversation: str) -> AnnotationResult:
        """Call LLM for annotation"""
        try:
            system_prompt = self.get_system_prompt()
            user_prompt = self.get_user_prompt(conversation)

            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(f"[{self.agent_id}] Sending request...")
                logger.debug(f"[{self.agent_id}] System prompt length: {len(system_prompt)}")
                logger.debug(f"[{self.agent_id}] User prompt length: {len(user_prompt)}")
                logger.debug(f"[{self.agent_id}] Temperature: {self.temperature}")
                logger.debug(f"[{self.agent_id}] Model: {self.model}")

            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=self.temperature,
            )
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(f"[{self.agent_id}] Response received")

            content = response.choices[0].message.content

            # Clean up content for JSON parsing
            clean_content = content.strip()
            if clean_content.startswith("```"):
                # Remove code blocks
                clean_content = clean_content.split("```")[1]
                if clean_content.startswith("json"):
                    clean_content = clean_content[4:]

            clean_content = clean_content.strip()
            # Handle potential trailing characters if needed,
            # but usually block stripping is enough

            try:
                data = json.loads(clean_content)
            except json.JSONDecodeError as e:
                # Fallback: try to find the first { and last }
                start = clean_content.find("{")
                end = clean_content.rfind("}")
                if start == -1 or end == -1:
                    raise json.JSONDecodeError("No JSON object found", clean_content, 0) from e

                try:
                    data = json.loads(clean_content[start : end + 1])
                except json.JSONDecodeError as e:
                    # Log error safely without exposing full content
                    logger.error(f"[{self.agent_id}] JSON Parse Error: {e}")
                    safe_snip = f"{content[:100]}..." if len(content) > 100 else content
                    logger.debug(f"[{self.agent_id}] Raw content snippet: {safe_snip}")
                    raise e
            return AnnotationResult(
                crisis_label=data.get("crisis_label", 0),
                crisis_confidence=data.get("crisis_confidence", 3),
                primary_emotion=data.get("primary_emotion", "Neutral"),
                secondary_emotions=normalize_secondary_emotions(
                    raw_values=data.get("secondary_emotions"),
                    primary_emotion=data.get("primary_emotion"),
                ),
                emotion_intensity=data.get("emotion_intensity", 5),
                valence=data.get("valence", 0.0),
                arousal=data.get("arousal", 0.5),
                empathy_score=data.get("empathy_score"),
                safety_pass=data.get("safety_pass"),
                notes=data.get("notes", ""),
                reasoning_chain=data.get("reasoning_chain", []),
                confidence_scores=data.get("confidence_scores", {}),
            )

        except Exception as e:
            logger.error(f"[{self.agent_id}] LLM error: {e}")
            return self._mock_annotation({})

    @abstractmethod
    def _mock_annotation(self, task: dict[str, Any]) -> AnnotationResult:
        """Generate mock annotation for testing"""


class CrisisExpertAgent(BaseAgent):
    """
    Dr. A - Conservative Crisis Expert
    Specializes in crisis detection and safety assessment
    """

    def __init__(self, model: str = "nvidia/nemotron-3-nano-30b-a3b"):
        super().__init__(
            agent_id="dr_a_crisis_expert",
            role=AgentRole.CRISIS_EXPERT,
            model=model,
            temperature=0.1,  # Lower temperature for consistency
        )

    def get_system_prompt(self) -> str:
        return """You are Dr. A, a conservative clinical psychologist
specializing in crisis intervention and safety assessment.

Your expertise:
- Trauma-informed care
- Crisis risk evaluation
- Safety planning
- Suicide risk assessment

Your approach:
- Prioritize safety above all
- Err on the side of caution
- Document all risk indicators
- Provide clear reasoning for crisis ratings

You are thorough, cautious, and always consider worst-case scenarios
to ensure patient safety."""

    def get_user_prompt(self, conversation: str) -> str:
        return f"""{self.guidelines}

Task: Annotate the following therapeutic conversation with focus on
crisis indicators and safety.

{conversation}

RESPOND WITH VALID JSON ONLY:
{{
  "crisis_label": <int 0-7>,
  "crisis_confidence": <int 1-5>,
  "primary_emotion": <string: Joy, Trust, Fear, Surprise, Sadness, Disgust,
                     Anger, Anticipation, Calm, Neutral>,
  "secondary_emotions": <optional list: up to 2 labels from the same taxonomy>,
  "emotion_intensity": <int 1-10>,
  "valence": <float -1.0 to 1.0>,
  "arousal": <float 0.0 to 1.0>,
  "empathy_score": <int 1-5 or null>,
  "safety_pass": <bool or null>,
  "notes": <string>,
  "reasoning_chain": [<list of reasoning steps>],
  "confidence_scores": {{"crisis": <float>, "emotion": <float>}}
}}

Focus on:
1. Any mention of self-harm or suicide
2. Expressions of hopelessness
3. Isolation or withdrawal
4. Substance abuse indicators
5. Trauma responses"""

    def _mock_annotation(self, task: dict[str, Any]) -> AnnotationResult:
        """Conservative mock with higher crisis sensitivity"""

        seed = len(str(task))
        random.seed(seed)

        # Dr. A is more likely to flag crisis
        is_crisis = random.random() < 0.4

        return AnnotationResult(
            crisis_label=random.randint(2, 7) if is_crisis else 0,
            crisis_confidence=random.randint(4, 5),
            primary_emotion=random.choice(["Fear", "Sadness", "Anger", "Neutral"]),
            secondary_emotions=["Anxiety"] if random.random() < 0.3 else [],
            emotion_intensity=random.randint(6, 9),
            valence=round(random.uniform(-0.8, -0.2), 2),
            arousal=round(random.uniform(0.6, 0.9), 2),
            empathy_score=random.randint(3, 5),
            safety_pass=not is_crisis,
            notes="Conservative assessment - prioritizing safety",
            reasoning_chain=[
                "Scanned for crisis indicators",
                "Evaluated safety concerns",
                "Applied conservative threshold",
            ],
            confidence_scores={"crisis": 0.85, "emotion": 0.75},
        )


class EmotionAnalystAgent(BaseAgent):
    """
    Dr. B - Pragmatic Emotion Analyst
    Specializes in emotional analysis and empathy assessment
    """

    def __init__(self, model: str = "nvidia/nemotron-3-nano-30b-a3b"):
        super().__init__(
            agent_id="dr_b_emotion_analyst",
            role=AgentRole.EMOTION_ANALYST,
            model=model,
            temperature=0.2,
        )

    def get_system_prompt(self) -> str:
        return """You are Dr. B, a pragmatic research psychologist specializing
in emotion analysis and therapeutic empathy.

Your expertise:
- Affective computing
- Emotion recognition
- Empathy measurement
- Therapeutic alliance assessment

Your approach:
- Evidence-based analysis
- Balanced interpretation
- Nuanced emotional understanding
- Focus on therapeutic quality

You are analytical, balanced, and grounded in research
while maintaining clinical sensitivity."""

    def get_user_prompt(self, conversation: str) -> str:
        return f"""{self.guidelines}

TASK: Annotate the following therapeutic conversation with focus on
emotional dynamics and empathy.

{conversation}

RESPOND WITH VALID JSON ONLY:
{{
  "crisis_label": <int 0-7>,
  "crisis_confidence": <int 1-5>,
  "primary_emotion": <string: Joy, Trust, Fear, Surprise, Sadness, Disgust,
                     Anger, Anticipation, Calm, Neutral>,
  "secondary_emotions": <optional list: up to 2 labels from the same taxonomy>,
  "emotion_intensity": <int 1-10>,
  "valence": <float -1.0 to 1.0>,
  "arousal": <float 0.0 to 1.0>,
  "empathy_score": <int 1-5 or null>,
  "safety_pass": <bool or null>,
  "notes": <string>,
  "reasoning_chain": [<list of reasoning steps>],
  "confidence_scores": {{"crisis": <float>, "emotion": <float>}}
}}

Focus on:
1. Primary and secondary emotions
2. Emotional intensity and valence
3. Therapist empathy quality
4. Emotional regulation patterns
5. Therapeutic alliance indicators"""

    def _mock_annotation(self, task: dict[str, Any]) -> AnnotationResult:
        """Balanced mock with focus on emotions"""

        seed = len(str(task))
        random.seed(seed)

        # Dr. B is more balanced
        is_crisis = random.random() < 0.2

        return AnnotationResult(
            crisis_label=random.randint(1, 2) if is_crisis else 0,
            crisis_confidence=random.randint(3, 4),
            primary_emotion=random.choice(["Sadness", "Joy", "Fear", "Neutral", "Anticipation"]),
            secondary_emotions=[
                emotion
                for emotion in [
                    "Sadness",
                    "Fear",
                    "Disgust",
                ]
                if random.random() < 0.3
            ],
            emotion_intensity=random.randint(4, 7),
            valence=round(random.uniform(-0.5, 0.5), 2),
            arousal=round(random.uniform(0.3, 0.7), 2),
            empathy_score=random.randint(3, 5),
            safety_pass=True,
            notes="Balanced emotional analysis",
            reasoning_chain=[
                "Identified primary emotion",
                "Measured intensity and valence",
                "Assessed empathy quality",
            ],
            confidence_scores={"crisis": 0.70, "emotion": 0.85},
        )


class QualityAssuranceAgent(BaseAgent):
    """
    Dr. C - Critical Reviewer / QA
    Acts as a tie-breaker and quality auditor
    """

    def __init__(self, model: str = "nvidia/nemotron-3-nano-30b-a3b"):
        super().__init__(
            agent_id="dr_c_qa_specialist",
            role=AgentRole.QUALITY_ASSURANCE,
            model=model,
            temperature=0.0,  # Deterministic
        )

    def get_system_prompt(self) -> str:
        return """You are Dr. C, a meticulous clinical supervisor
and data quality specialist.
Your role is to provide highly accurate, objective, and consistent annotations.
You follow the guidelines strictly and avoid defaulting to 'Neutral'
unless the content is truly devoid of emotional signaling.
You are excellent at identifying subtle emotional nuances
and ensuring safety protocol adherence."""

    def get_user_prompt(self, conversation: str) -> str:
        return f"""{self.guidelines}

TASK: Provide a master annotation for this therapeutic conversation.
Be precise and thorough.

{conversation}

RESPOND WITH VALID JSON ONLY:
{{
  "crisis_label": <int 0-7>,
  "crisis_confidence": <int 1-5>,
  "primary_emotion": <string: Joy, Trust, Fear, Surprise, Sadness, Disgust,
                     Anger, Anticipation, Calm, Neutral>,
  "secondary_emotions": <optional list: up to 2 labels from the same taxonomy>,
  "emotion_intensity": <int 1-10>,
  "valence": <float -1.0 to 1.0>,
  "arousal": <float 0.0 to 1.0>,
  "empathy_score": <int 1-5 or null>,
  "safety_pass": <bool or null>,
  "notes": <string>,
  "reasoning_chain": [<list of reasoning steps>],
  "confidence_scores": {{"crisis": <float>, "emotion": <float>}}
}}"""

    def _mock_annotation(self, task: dict[str, Any]) -> AnnotationResult:

        seed = len(str(task)) + 2
        random.seed(seed)
        return AnnotationResult(
            crisis_label=0,
            crisis_confidence=5,
            primary_emotion="Neutral",
            secondary_emotions=[],
            emotion_intensity=3,
            valence=0.0,
            arousal=0.3,
            empathy_score=4,
            safety_pass=True,
            notes="QA review - focused on stability",
            reasoning_chain=["Applied QA standards"],
            confidence_scores={"crisis": 0.9, "emotion": 0.9},
        )


class ConsensusOrchestrator:
    """
    Orchestrates multi-agent annotation and builds consensus
    """

    def __init__(self):
        self.agents: list[BaseAgent] = []

    def add_agent(self, agent: BaseAgent):
        """Register an agent"""
        self.agents.append(agent)

    def annotate_with_consensus(self, task: dict[str, Any]) -> dict[str, Any]:
        """
        Run all agents and build consensus
        """
        if not self.agents:
            raise ValueError("No agents registered for consensus")
        results = []
        metadata_list = []

        # Parallel execution with deterministic order and capped concurrency
        max_workers = min(4, len(self.agents)) if self.agents else 1
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Map futures to their original agent index to preserve order
            future_to_index = {executor.submit(agent.annotate, task): i for i, agent in enumerate(self.agents)}

            ordered_results = [None] * len(self.agents)
            ordered_metadata = [None] * len(self.agents)
            failed_agents = []

            for future in concurrent.futures.as_completed(future_to_index):
                index = future_to_index[future]
                agent = self.agents[index]
                try:
                    result, metadata = future.result()
                    ordered_results[index] = result
                    ordered_metadata[index] = metadata
                except Exception as e:
                    error_msg = f"Agent {agent.agent_id} failed: {e}"
                    logger.error(error_msg)
                    failed_agents.append({"agent_id": agent.agent_id, "error": str(e)})

        # Filter out failed results
        results = [r for r in ordered_results if r is not None]
        metadata_list = [m for m in ordered_metadata if m is not None]

        if not results:
            return {
                "task_id": task.get("task_id", task.get("data", {}).get("id")),
                "error": "All agents failed to annotate task",
                "failed_agents": failed_agents,
                "consensus_annotation": None,
            }

        # Enforce minimum quorum (at least 2 successful results if multiple
        # agents are registered)
        if len(self.agents) > 1 and len(results) < 2:
            error_msg = (
                f"Quorum not met for task {task.get('task_id')}: "
                f"only {len(results)} results out of {len(self.agents)} agents"
            )
            logger.error(error_msg)
            return {
                "task_id": task.get("task_id", task.get("data", {}).get("id")),
                "error": "Quorum not met",
                "details": error_msg,
                "failed_agents": failed_agents,
                "consensus_annotation": None,
            }

        consensus_notes_extra = ""

        # Build consensus
        consensus = self._build_consensus(results)
        consensus.notes += consensus_notes_extra

        # Calculate agreement metrics
        agreement = self._calculate_agreement(results)

        # Finalize consensus based on agreement
        if agreement.get("overall_agreement", 0.0) < 0.5:
            consensus.notes += " | Low agreement - requires expert review"

        return {
            "task_id": task.get("task_id", task.get("data", {}).get("id")),
            "data": task.get("data"),
            "consensus_annotation": consensus.to_dict(),
            "individual_annotations": [r.to_dict() for r in results],
            "agent_metadata": [m.to_dict() for m in metadata_list],
            "agreement_metrics": agreement,
            "failed_agents": failed_agents,
        }

    def _build_consensus(self, results: list[AnnotationResult]) -> AnnotationResult:
        """Build consensus from multiple annotations"""
        if not results:
            raise ValueError("No results to build consensus from")

        # Average numeric fields
        crisis_labels = [r.crisis_label for r in results]
        crisis_confidences = [r.crisis_confidence for r in results]
        intensities = [r.emotion_intensity for r in results]
        valences = [r.valence for r in results]
        arousals = [r.arousal for r in results]

        primary_emotion, emotion_tie_resolved = self._resolve_primary_emotion(results)
        secondary_emotions = self._resolve_secondary_emotions(results=results, primary_emotion=primary_emotion)

        # Average empathy scores (if present)
        empathy_scores = [r.empathy_score for r in results if r.empathy_score]
        avg_empathy = int(sum(empathy_scores) / len(empathy_scores)) if empathy_scores else None

        # Safety pass if all agree
        safety_passes = [r.safety_pass for r in results if r.safety_pass is not None]
        safety_pass = all(safety_passes) if safety_passes else None

        return AnnotationResult(
            crisis_label=int(sum(crisis_labels) / len(crisis_labels)),
            crisis_confidence=int(sum(crisis_confidences) / len(crisis_confidences)),
            primary_emotion=primary_emotion,
            secondary_emotions=secondary_emotions,
            emotion_intensity=int(sum(intensities) / len(intensities)),
            valence=round(sum(valences) / len(valences), 2),
            arousal=round(sum(arousals) / len(arousals), 2),
            empathy_score=avg_empathy,
            safety_pass=safety_pass,
            notes=(
                "Emotion tie-breaker applied." if emotion_tie_resolved else "Consensus annotation from multiple agents"
            ),
            reasoning_chain=["Aggregated from all agents"],
        )

    @staticmethod
    def _resolve_primary_emotion(
        results: list[AnnotationResult],
    ) -> tuple[str, bool]:
        """
        Resolve primary emotion using tie-break rules.

        Returns:
            tuple[str, bool]: (resolved emotion, was_tie_resolved)
        """
        emotions = [result.primary_emotion for result in results]
        if not emotions:
            return "Neutral", False

        counts = Counter(emotions)
        max_count = max(counts.values())
        top_emotions = [emotion for emotion, count in counts.items() if count == max_count]

        if len(top_emotions) == 1:
            return top_emotions[0], False

        emotion_scores: list[tuple[str, tuple[float, float, int]]] = []
        for emotion in top_emotions:
            tied_annotations = [result for result in results if result.primary_emotion == emotion]

            # Prefer annotations with explicit emotion confidence if provided.
            confidences = [float(r.confidence_scores.get("emotion", 0.0)) for r in tied_annotations]
            confidence_weight = sum(confidences) / len(confidences)

            # Secondary deterministic tie-break with average emotion intensity.
            avg_intensity = sum(r.emotion_intensity for r in tied_annotations) / len(tied_annotations)

            # Preference order is deterministic and clinically neutral.
            preference_rank = (
                EMOTION_TIEBREAK_PREFERENCE.index(emotion)
                if emotion in EMOTION_TIEBREAK_PREFERENCE
                else len(EMOTION_TIEBREAK_PREFERENCE)
            )

            # Higher score wins; lower preference rank is better.
            score = (confidence_weight, avg_intensity, -preference_rank)
            emotion_scores.append((emotion, score))

        # Sort descending by score tuple (confidence, intensity, preference rank inverse)
        emotion_scores.sort(key=lambda item: item[1], reverse=True)
        return emotion_scores[0][0], True

    def _resolve_secondary_emotions(
        self,
        results: list[AnnotationResult],
        primary_emotion: str,
    ) -> list[str]:
        """Resolve secondary emotions from annotator outputs."""
        counts = Counter()
        for result in results:
            for emotion in normalize_secondary_emotions(result.secondary_emotions, primary_emotion=primary_emotion):
                counts[emotion] += 1

        if not counts:
            return []

        secondary_scores: list[tuple[str, tuple[int, float, float, int]]] = []
        for emotion, count in counts.items():
            if emotion == primary_emotion:
                continue
            tied_annotations = [result for result in results if emotion in result.secondary_emotions]

            confidence_values = [float(result.confidence_scores.get("emotion", 0.0)) for result in tied_annotations]
            confidence = sum(confidence_values) / len(confidence_values) if confidence_values else 0.0
            intensity = (
                sum(result.emotion_intensity for result in tied_annotations) / len(tied_annotations)
                if tied_annotations
                else 0.0
            )
            preference_rank = (
                EMOTION_TIEBREAK_PREFERENCE.index(emotion)
                if emotion in EMOTION_TIEBREAK_PREFERENCE
                else len(EMOTION_TIEBREAK_PREFERENCE)
            )

            secondary_scores.append((emotion, (count, confidence, intensity, -preference_rank)))

        secondary_scores.sort(key=lambda item: item[1], reverse=True)
        return [emotion for emotion, _ in secondary_scores[:MAX_SECONDARY_EMOTIONS]]

    def _calculate_agreement(self, results: list[AnnotationResult]) -> dict[str, float]:
        """Calculate inter-agent agreement metrics"""
        if len(results) < 2:
            return {
                "crisis_agreement": 1.0,
                "emotion_agreement": 1.0,
                "intensity_variance": 0.0,
                "overall_agreement": 1.0,
            }

        # Simple agreement on crisis label
        crisis_labels = [r.crisis_label for r in results]
        crisis_agreement = 1.0 if len(set(crisis_labels)) == 1 else 0.0

        # Emotion agreement
        emotions = [r.primary_emotion for r in results]
        emotion_agreement = 1.0 if len(set(emotions)) == 1 else 0.0

        # Average numeric field variance
        intensities = [r.emotion_intensity for r in results]
        intensity_variance = sum((x - sum(intensities) / len(intensities)) ** 2 for x in intensities) / len(intensities)

        return {
            "crisis_agreement": crisis_agreement,
            "emotion_agreement": emotion_agreement,
            "intensity_variance": round(intensity_variance, 2),
            "overall_agreement": round((crisis_agreement + emotion_agreement) / 2, 2),
        }


def create_multi_agent_system(
    model: str = "nvidia/nemotron-3-nano-30b-a3b",
) -> ConsensusOrchestrator:
    """
    Factory function to create complete multi-agent system
    """
    orchestrator = ConsensusOrchestrator()

    # Add specialized agents
    orchestrator.add_agent(CrisisExpertAgent(model=model))
    orchestrator.add_agent(EmotionAnalystAgent(model=model))
    orchestrator.add_agent(QualityAssuranceAgent(model=model))

    return orchestrator
