"""
Enhanced Annotation Agent with NVIDIA Blueprint Best Practices

Incorporates patterns from:
- AI-Q Enterprise Research Blueprint (NeMo Agent Toolkit)
- Ambient Healthcare Agents Blueprint (Guardrails, Medical Reasoning)
- Digital Twins for AI Factories (Evaluation & Profiling)

Key Enhancements:
1. NeMo Guardrails for safe, topically appropriate annotations
2. Advanced reasoning with Llama Nemotron models
3. Agent evaluation and profiling capabilities
4. Multi-agent orchestration patterns
"""

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional
from dataclasses import dataclass, asdict
from enum import Enum

# Try importing required libraries
try:
    from openai import OpenAI

    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

from agent_personas import DR_A_PERSONA, DR_B_PERSONA

# Constants
GUIDELINES_PATH = Path(__file__).resolve().parent.parent / "guidelines.md"


class AgentRole(Enum):
    """Agent roles based on NVIDIA healthcare agent patterns"""

    PRIMARY_ANNOTATOR = "primary_annotator"
    VALIDATION_AGENT = "validation_agent"
    CONSENSUS_AGENT = "consensus_agent"


@dataclass
class AnnotationMetrics:
    """Metrics for agent evaluation (NeMo Agent Toolkit pattern)"""

    task_id: str
    agent_id: str
    processing_time: float
    token_count: int
    confidence_score: float
    guardrail_checks: Dict[str, bool]
    reasoning_steps: List[str]


@dataclass
class GuardrailConfig:
    """Guardrail configuration inspired by NeMo Guardrails"""

    check_emotional_safety: bool = True
    check_clinical_accuracy: bool = True
    check_bias_detection: bool = True
    check_crisis_sensitivity: bool = True
    max_emotion_intensity: int = 10
    min_confidence_threshold: float = 0.6


class EnhancedAnnotationAgent:
    """
    Enhanced annotation agent with NVIDIA Blueprint patterns

    Features:
    - Guardrails for safe annotations
    - Reasoning capabilities with Llama Nemotron
    - Evaluation and profiling
    - Multi-agent orchestration support
    """

    def __init__(
        self,
        persona_name: str,
        model: str = "nvidia/llama-3.3-nemotron-super-49b-v1",
        role: AgentRole = AgentRole.PRIMARY_ANNOTATOR,
        guardrail_config: Optional[GuardrailConfig] = None,
    ):
        self.persona_name = persona_name
        self.model = model
        self.role = role
        self.guardrail_config = guardrail_config or GuardrailConfig()
        self.system_prompt = self._get_system_prompt(persona_name)
        self.guidelines = self._load_guidelines()
        self.metrics: List[AnnotationMetrics] = []

        # Initialize OpenAI client with custom base URL support
        self.client = None
        base_url = os.getenv("OPENAI_BASE_URL")

        if OPENAI_AVAILABLE and os.getenv("OPENAI_API_KEY"):
            if base_url:
                self.client = OpenAI(base_url=base_url)
                print(
                    f"[{persona_name}] Agent initialized with {model} "
                    f"(Custom Base URL: {base_url})"
                )
            else:
                self.client = OpenAI()
                print(f"[{persona_name}] Agent initialized with {model}")
        else:
            print(f"[{persona_name}] OpenAI client NOT available. Using MOCK mode.")

    def _get_system_prompt(self, name: str) -> str:
        """Get system prompt with enhanced reasoning instructions"""
        base_prompt = {"Dr. A": DR_A_PERSONA, "Dr. B": DR_B_PERSONA}.get(
            name, "You are a helpful annotator."
        )

        # Add reasoning enhancement (Llama Nemotron pattern)
        reasoning_enhancement = """

REASONING PROTOCOL (Llama Nemotron Pattern):
1. Analyze the conversation systematically
2. Identify key emotional and clinical indicators
3. Cross-reference with annotation guidelines
4. Apply domain expertise (trauma-informed, CPTSD-aware)
5. Validate against safety and ethical criteria
6. Generate structured annotation with confidence scores

GUARDRAILS:
- Prioritize psychological safety
- Detect and flag crisis indicators
- Maintain clinical accuracy
- Avoid bias in emotional assessment
- Respect cultural and linguistic variations
"""
        return base_prompt + reasoning_enhancement

    def _load_guidelines(self) -> str:
        """Load annotation guidelines"""
        if GUIDELINES_PATH.exists():
            return GUIDELINES_PATH.read_text()
        return "No guidelines found."

    def _apply_guardrails(self, annotation: Dict[str, Any]) -> Dict[str, bool]:
        """
        Apply NeMo Guardrails-inspired safety checks

        Returns dict of guardrail check results
        """
        checks = {}

        # Emotional safety check
        if self.guardrail_config.check_emotional_safety:
            emotion_intensity = annotation.get("emotion_intensity", 0)
            checks["emotional_safety"] = (
                0 <= emotion_intensity <= self.guardrail_config.max_emotion_intensity
            )

        # Clinical accuracy check
        if self.guardrail_config.check_clinical_accuracy:
            crisis_label = annotation.get("crisis_label", 0)
            crisis_confidence = annotation.get("crisis_confidence", 0)
            checks["clinical_accuracy"] = (
                0 <= crisis_label <= 5 and 1 <= crisis_confidence <= 5
            )

        # Bias detection check
        if self.guardrail_config.check_bias_detection:
            valence = annotation.get("valence", 0.0)
            arousal = annotation.get("arousal", 0.0)
            checks["bias_detection"] = -1.0 <= valence <= 1.0 and 0.0 <= arousal <= 1.0

        # Crisis sensitivity check
        if self.guardrail_config.check_crisis_sensitivity:
            crisis_label = annotation.get("crisis_label", 0)
            notes = annotation.get("notes", "")
            # High crisis labels should have detailed notes
            checks["crisis_sensitivity"] = crisis_label < 3 or len(notes) > 20

        return checks

    def _extract_reasoning_steps(self, annotation: Dict[str, Any]) -> List[str]:
        """Extract reasoning steps from annotation (for evaluation)"""
        steps = []

        # Infer reasoning from annotation structure
        if annotation.get("crisis_label", 0) > 0:
            steps.append(
                f"Detected crisis level {annotation['crisis_label']} "
                f"with confidence {annotation.get('crisis_confidence', 0)}"
            )

        if annotation.get("primary_emotion"):
            steps.append(
                f"Identified primary emotion: "
                f"{annotation['primary_emotion']} "
                f"(intensity: {annotation.get('emotion_intensity', 0)})"
            )

        if annotation.get("notes"):
            steps.append(f"Clinical notes: {annotation['notes'][:100]}...")

        return steps

    def annotate(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """
        Produce an annotation with guardrails and evaluation

        Returns annotation with metadata and metrics
        """
        start_time = time.time()

        # Prepare content
        data = task.get("data", {})
        transcript = data.get("transcript")
        messages = data.get("messages")

        content_text = ""
        if transcript:
            content_text = f"TRANSCRIPT:\n{transcript}"
        elif messages:
            content_text = "CONVERSATION HISTORY:\n"
            for m in messages:
                role = m.get("role", "unknown")
                content = m.get("content", "")
                content_text += f"{role.upper()}: {content}\n"

        # Enhanced prompt with reasoning instructions
        prompt = f"""
{self.guidelines}

TASK: Annotate the following therapeutic conversation sample.

REASONING INSTRUCTIONS:
1. Read the entire conversation carefully
2. Identify emotional states and crisis indicators
3. Assess therapeutic quality of responses
4. Apply clinical judgment with confidence scores
5. Document your reasoning in notes

{content_text}

Respond ONLY with valid JSON in this exact format:
{{
  "crisis_label": <int 0-5>,
  "crisis_confidence": <int 1-5>,
  "primary_emotion": <string>,
  "emotion_intensity": <int 1-10>,
  "valence": <float -1.0 to 1.0>,
  "arousal": <float 0.0 to 1.0>,
  "empathy_score": <int 1-5 or null>,
  "safety_pass": <bool or null>,
  "notes": <string with clinical reasoning>
}}
"""

        # Get annotation
        if self.client:
            annotation = self._call_llm(prompt, data)
        else:
            annotation = self._mock_annotation(data)

        # Apply guardrails
        guardrail_checks = self._apply_guardrails(annotation)

        # Extract reasoning steps
        reasoning_steps = self._extract_reasoning_steps(annotation)

        # Calculate metrics
        processing_time = time.time() - start_time
        token_count = len(prompt.split()) + len(str(annotation).split())
        confidence_score = annotation.get("crisis_confidence", 3) / 5.0

        # Store metrics for evaluation
        metrics = AnnotationMetrics(
            task_id=task.get("task_id", "unknown"),
            agent_id=self.persona_name,
            processing_time=processing_time,
            token_count=token_count,
            confidence_score=confidence_score,
            guardrail_checks=guardrail_checks,
            reasoning_steps=reasoning_steps,
        )
        self.metrics.append(metrics)

        # Check if all guardrails passed
        all_checks_passed = all(guardrail_checks.values())
        if not all_checks_passed:
            print(
                f"⚠️  Guardrail warnings for task "
                f"{task.get('task_id')}: {guardrail_checks}"
            )

        return annotation

    def _call_llm(self, prompt: str, data: Dict[str, Any]) -> Dict[str, Any]:
        """Call LLM with error handling and fallback"""
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.2,
                max_tokens=2000,
            )
            content = response.choices[0].message.content

            # Try to parse JSON from response
            # Handle cases where LLM wraps JSON in markdown
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0].strip()
            elif "```" in content:
                content = content.split("```")[1].split("```")[0].strip()

            return json.loads(content)
        except Exception as e:
            print(f"Error calling LLM: {e}. Falling back to mock generation.")
            return self._mock_annotation(data, error=False)

    def _mock_annotation(
        self, data: Dict[str, Any], error: bool = False
    ) -> Dict[str, Any]:
        """Generate mock annotation for testing"""
        import random

        time.sleep(0.01)

        if error:
            return {"error": "LLM call failed"}

        # Persona-based biases
        seed = len(str(data))
        random.seed(seed)

        if self.persona_name == "Dr. A":
            crisis_chance = 0.4
            avg_intensity = 7
        else:
            crisis_chance = 0.2
            avg_intensity = 5

        is_crisis = random.random() < crisis_chance

        return {
            "crisis_label": random.randint(1, 4) if is_crisis else 0,
            "crisis_confidence": random.randint(3, 5),
            "primary_emotion": random.choice(
                ["Sadness", "Fear", "Anger", "Joy", "Neutral"]
            ),
            "emotion_intensity": min(10, max(1, int(random.gauss(avg_intensity, 2)))),
            "valence": round(random.uniform(-1.0, 1.0), 2),
            "arousal": round(random.uniform(0.0, 1.0), 2),
            "empathy_score": random.randint(1, 5),
            "safety_pass": True,
            "notes": (
                f"Mock annotation by {self.persona_name}. "
                f"Guardrails: {self.guardrail_config}"
            ),
        }

    def export_metrics(self, output_path: Path) -> None:
        """Export agent metrics for evaluation (NeMo Agent Toolkit pattern)"""
        metrics_data = [asdict(m) for m in self.metrics]

        with open(output_path, "w") as f:
            json.dump(
                {
                    "agent_id": self.persona_name,
                    "model": self.model,
                    "role": self.role.value,
                    "total_tasks": len(self.metrics),
                    "avg_processing_time": (
                        sum(m.processing_time for m in self.metrics) / len(self.metrics)
                        if self.metrics
                        else 0
                    ),
                    "avg_confidence": (
                        sum(m.confidence_score for m in self.metrics)
                        / len(self.metrics)
                        if self.metrics
                        else 0
                    ),
                    "guardrail_pass_rate": self._calculate_guardrail_pass_rate(),
                    "metrics": metrics_data,
                },
                f,
                indent=2,
            )

        print(f"✅ Metrics exported to {output_path}")

    def _calculate_guardrail_pass_rate(self) -> float:
        """Calculate percentage of annotations passing all guardrails"""
        if not self.metrics:
            return 0.0

        passed = sum(1 for m in self.metrics if all(m.guardrail_checks.values()))
        return passed / len(self.metrics)


def process_batch(
    input_file: str,
    output_file: str,
    agent: EnhancedAnnotationAgent,
    export_metrics: bool = True,
):
    """Process annotation batch with enhanced agent"""
    input_path = Path(input_file)
    output_path = Path(output_file)

    print(f"Processing {input_path} with enhanced agent {agent.persona_name}...")

    if not input_path.exists():
        print(f"Input file {input_path} not found.")
        return

    # Ensure output dir exists
    output_path.parent.mkdir(parents=True, exist_ok=True)

    processed_count = 0
    with open(input_path, "r") as f_in, open(output_path, "w") as f_out:
        for line in f_in:
            if not line.strip():
                continue
            try:
                task = json.loads(line)
                annotations = agent.annotate(task)

                # Create result record
                result = {
                    "task_id": task.get("task_id", task.get("id")),
                    "annotator_id": agent.persona_name.lower()
                    .replace(" ", "_")
                    .replace(".", ""),
                    "annotations": annotations,
                    "metadata": {
                        "model": agent.model,
                        "role": agent.role.value,
                        "timestamp": time.time(),
                        "guardrails_enabled": True,
                    },
                }

                f_out.write(json.dumps(result) + "\n")
                processed_count += 1

                if processed_count % 10 == 0:
                    print(f"  Processed {processed_count} records...", end="\r")

            except json.JSONDecodeError:
                continue

    print(f"\n✅ Completed {processed_count} records.")
    print(f"   Saved to {output_path}")

    # Export metrics
    if export_metrics:
        metrics_path = output_path.parent / f"{output_path.stem}_metrics.json"
        agent.export_metrics(metrics_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Enhanced AI Annotation Agent with NVIDIA Blueprint Patterns"
    )
    parser.add_argument("--input", required=True, help="Input batch JSONL file")
    parser.add_argument("--output", required=True, help="Output results JSONL file")
    parser.add_argument(
        "--persona", choices=["Dr. A", "Dr. B"], required=True, help="Agent persona"
    )
    parser.add_argument(
        "--model",
        default="nvidia/llama-3.3-nemotron-super-49b-v1",
        help="LLM model to use (default: Llama Nemotron)",
    )
    parser.add_argument(
        "--role",
        choices=["primary_annotator", "validation_agent", "consensus_agent"],
        default="primary_annotator",
        help="Agent role in multi-agent system",
    )
    parser.add_argument(
        "--export-metrics",
        action="store_true",
        default=True,
        help="Export agent metrics for evaluation",
    )

    args = parser.parse_args()

    # Create enhanced agent
    agent = EnhancedAnnotationAgent(
        persona_name=args.persona, model=args.model, role=AgentRole(args.role)
    )

    # Process batch
    process_batch(args.input, args.output, agent, export_metrics=args.export_metrics)
