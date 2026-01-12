"""
Multi-Agent Annotation System
Inspired by NVIDIA AI Blueprints architecture

This module implements a sophisticated multi-agent system for annotating
therapeutic conversations with high reliability and psychological safety.
"""

import json
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional

try:
    from openai import OpenAI

    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False


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
    emotion_intensity: int  # 1-10
    valence: float  # -1.0 to 1.0
    arousal: float  # 0.0 to 1.0
    empathy_score: Optional[int] = None  # 1-5
    safety_pass: Optional[bool] = None
    notes: str = ""
    reasoning_chain: List[str] = field(default_factory=list)
    confidence_scores: Dict[str, float] = field(default_factory=dict)


@dataclass
class AgentMetadata:
    """Agent execution metadata"""

    agent_id: str
    role: AgentRole
    model: str
    timestamp: float
    processing_time: float
    token_count: Optional[int] = None


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

    def _initialize_client(self) -> Optional[OpenAI]:
        """Initialize OpenAI client with optional custom base URL"""
        import os

        if not OPENAI_AVAILABLE or not os.getenv("OPENAI_API_KEY"):
            print(f"[{self.agent_id}] Running in MOCK mode")
            return None

        base_url = os.getenv("OPENAI_BASE_URL")
        if base_url:
            client = OpenAI(base_url=base_url)
            print(f"[{self.agent_id}] Using custom endpoint: {base_url}")
        else:
            client = OpenAI()

        print(f"[{self.agent_id}] Initialized with model: {self.model}")
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
        pass

    @abstractmethod
    def get_user_prompt(self, conversation: str) -> str:
        """Generate user prompt for annotation task"""
        pass

    def annotate(self, task: Dict[str, Any]) -> tuple[AnnotationResult, AgentMetadata]:
        """
        Main annotation method
        Returns: (annotation_result, metadata)
        """
        start_time = time.time()

        # Extract conversation content
        conversation = self._extract_conversation(task)

        # Generate annotation
        if self.client:
            result = self._call_llm(conversation)
        else:
            result = self._mock_annotation(task)

        # Create metadata
        metadata = AgentMetadata(
            agent_id=self.agent_id,
            role=self.role,
            model=self.model,
            timestamp=time.time(),
            processing_time=time.time() - start_time,
        )

        return result, metadata

    def _extract_conversation(self, task: Dict[str, Any]) -> str:
        """Extract conversation text from task data"""
        data = task.get("data", {})

        # Handle transcript format
        if "transcript" in data:
            return f"TRANSCRIPT:\n{data['transcript']}"

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
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": self.get_system_prompt()},
                    {"role": "user", "content": self.get_user_prompt(conversation)},
                ],
                temperature=self.temperature,
            )

            content = response.choices[0].message.content
            data = json.loads(content)

            return AnnotationResult(
                crisis_label=data.get("crisis_label", 0),
                crisis_confidence=data.get("crisis_confidence", 3),
                primary_emotion=data.get("primary_emotion", "Neutral"),
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
            print(f"[{self.agent_id}] LLM error: {e}")
            return self._mock_annotation({})

    @abstractmethod
    def _mock_annotation(self, task: Dict[str, Any]) -> AnnotationResult:
        """Generate mock annotation for testing"""
        pass


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
        return """You are Dr. A, a conservative clinical psychologist specializing in crisis intervention and safety assessment.

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

You are thorough, cautious, and always consider worst-case scenarios to ensure patient safety."""

    def get_user_prompt(self, conversation: str) -> str:
        return f"""{self.guidelines}

TASK: Annotate the following therapeutic conversation with focus on crisis indicators and safety.

{conversation}

RESPOND WITH VALID JSON ONLY:
{{
  "crisis_label": <int 0-5>,
  "crisis_confidence": <int 1-5>,
  "primary_emotion": <string>,
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

    def _mock_annotation(self, task: Dict[str, Any]) -> AnnotationResult:
        """Conservative mock with higher crisis sensitivity"""
        import random

        seed = len(str(task))
        random.seed(seed)

        # Dr. A is more likely to flag crisis
        is_crisis = random.random() < 0.4

        return AnnotationResult(
            crisis_label=random.randint(2, 4) if is_crisis else 0,
            crisis_confidence=random.randint(4, 5),
            primary_emotion=random.choice(["Fear", "Sadness", "Anger", "Anxiety"]),
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
        return """You are Dr. B, a pragmatic research psychologist specializing in emotion analysis and therapeutic empathy.

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

You are analytical, balanced, and grounded in research while maintaining clinical sensitivity."""

    def get_user_prompt(self, conversation: str) -> str:
        return f"""{self.guidelines}

TASK: Annotate the following therapeutic conversation with focus on emotional dynamics and empathy.

{conversation}

RESPOND WITH VALID JSON ONLY:
{{
  "crisis_label": <int 0-5>,
  "crisis_confidence": <int 1-5>,
  "primary_emotion": <string>,
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

    def _mock_annotation(self, task: Dict[str, Any]) -> AnnotationResult:
        """Balanced mock with focus on emotions"""
        import random

        seed = len(str(task))
        random.seed(seed)

        # Dr. B is more balanced
        is_crisis = random.random() < 0.2

        return AnnotationResult(
            crisis_label=random.randint(1, 2) if is_crisis else 0,
            crisis_confidence=random.randint(3, 4),
            primary_emotion=random.choice(
                ["Sadness", "Joy", "Fear", "Neutral", "Hope"]
            ),
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


class ConsensusOrchestrator:
    """
    Orchestrates multi-agent annotation and builds consensus
    """

    def __init__(self):
        self.agents: List[BaseAgent] = []

    def add_agent(self, agent: BaseAgent):
        """Register an agent"""
        self.agents.append(agent)

    def annotate_with_consensus(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """
        Run all agents and build consensus
        """
        results = []
        metadata_list = []

        # Collect annotations from all agents
        for agent in self.agents:
            result, metadata = agent.annotate(task)
            results.append(result)
            metadata_list.append(metadata)

        # Build consensus
        consensus = self._build_consensus(results)

        # Calculate agreement metrics
        agreement = self._calculate_agreement(results)

        return {
            "task_id": task.get("task_id", task.get("data", {}).get("id")),
            "consensus_annotation": consensus.__dict__,
            "individual_annotations": [r.__dict__ for r in results],
            "agent_metadata": [m.__dict__ for m in metadata_list],
            "agreement_metrics": agreement,
        }

    def _build_consensus(self, results: List[AnnotationResult]) -> AnnotationResult:
        """Build consensus from multiple annotations"""
        if not results:
            raise ValueError("No results to build consensus from")

        # Average numeric fields
        crisis_labels = [r.crisis_label for r in results]
        crisis_confidences = [r.crisis_confidence for r in results]
        intensities = [r.emotion_intensity for r in results]
        valences = [r.valence for r in results]
        arousals = [r.arousal for r in results]

        # Most common emotion
        emotions = [r.primary_emotion for r in results]
        primary_emotion = max(set(emotions), key=emotions.count)

        # Average empathy scores (if present)
        empathy_scores = [r.empathy_score for r in results if r.empathy_score]
        avg_empathy = (
            int(sum(empathy_scores) / len(empathy_scores)) if empathy_scores else None
        )

        # Safety pass if all agree
        safety_passes = [r.safety_pass for r in results if r.safety_pass is not None]
        safety_pass = all(safety_passes) if safety_passes else None

        return AnnotationResult(
            crisis_label=int(sum(crisis_labels) / len(crisis_labels)),
            crisis_confidence=int(sum(crisis_confidences) / len(crisis_confidences)),
            primary_emotion=primary_emotion,
            emotion_intensity=int(sum(intensities) / len(intensities)),
            valence=round(sum(valences) / len(valences), 2),
            arousal=round(sum(arousals) / len(arousals), 2),
            empathy_score=avg_empathy,
            safety_pass=safety_pass,
            notes="Consensus annotation from multiple agents",
            reasoning_chain=["Aggregated from all agents"],
        )

    def _calculate_agreement(self, results: List[AnnotationResult]) -> Dict[str, float]:
        """Calculate inter-agent agreement metrics"""
        if len(results) < 2:
            return {"agreement": 1.0}

        # Simple agreement on crisis label
        crisis_labels = [r.crisis_label for r in results]
        crisis_agreement = 1.0 if len(set(crisis_labels)) == 1 else 0.0

        # Emotion agreement
        emotions = [r.primary_emotion for r in results]
        emotion_agreement = 1.0 if len(set(emotions)) == 1 else 0.0

        # Average numeric field variance
        intensities = [r.emotion_intensity for r in results]
        intensity_variance = sum(
            (x - sum(intensities) / len(intensities)) ** 2 for x in intensities
        ) / len(intensities)

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

    return orchestrator
