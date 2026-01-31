"""
Synthetic Data Distillation Pipeline

Implements advanced synthetic data generation techniques based on Confident AI's
"Using LLMs for Synthetic Data Generation: The Definitive Guide" including:
- Distillation: Generate training data from larger models to train smaller ones
- Self-Improvement: Use model outputs to iteratively improve training data quality
- Multi-step Prompting: TarGEN-style framework for enhanced dataset generation

Part of the Pixelated Empathy AI dataset pipeline.
"""

import hashlib
import json
import logging
import os
import random
import re
import sys
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Optional

# Handle imports
generation_path = Path(__file__).parent
pipeline_root = generation_path.parent.parent
sys.path.insert(0, str(pipeline_root))

try:
    from schemas.conversation_schema import Conversation, Message
except ImportError:
    try:
        from ai.pipelines.orchestrator.schemas.conversation_schema import Conversation, Message
    except ImportError:
        from conversation_schema import Conversation, Message

try:
    from logger import get_logger

    logger = get_logger("dataset_pipeline.synthetic_distillation")
except ImportError:
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)


class DistillationStrategy(Enum):
    """Synthetic data generation strategies."""

    DIRECT_DISTILLATION = "direct_distillation"
    SELF_IMPROVEMENT = "self_improvement"
    MULTI_STEP_PROMPTING = "multi_step_prompting"
    TARGEN_STYLE = "targen_style"
    CULTURAL_AUGMENTATION = "cultural_augmentation"
    EDGE_CASE_EXPANSION = "edge_case_expansion"


class PromptEvolutionType(Enum):
    """Types of prompt evolution for TarGEN-style generation."""

    COMPLEXITY_INCREASE = "complexity_increase"
    CONTEXT_ENRICHMENT = "context_enrichment"
    SPECIFICITY_ENHANCEMENT = "specificity_enhancement"
    CONSTRAINT_ADDITION = "constraint_addition"
    PERSPECTIVE_SHIFT = "perspective_shift"


@dataclass
class SyntheticGenerationConfig:
    """Configuration for synthetic data generation."""

    strategy: DistillationStrategy = DistillationStrategy.MULTI_STEP_PROMPTING
    model_provider: str = "ollama"  # ollama, openai, anthropic
    teacher_model: str = "llama3.2"  # Larger model for generation
    student_model: Optional[str] = None  # Target model (if different)
    temperature: float = 0.7
    max_tokens: int = 2048
    num_iterations: int = 3  # For self-improvement
    quality_threshold: float = 0.8
    batch_size: int = 10
    enable_self_evaluation: bool = True
    enable_diversity_scoring: bool = True
    cultural_contexts: list[str] = field(default_factory=lambda: [
        "western", "eastern_asian", "south_asian", "latin_american",
        "african", "middle_eastern", "indigenous"
    ])


@dataclass
class GeneratedSample:
    """A single generated synthetic sample."""

    conversation_id: str
    messages: list[Message]
    generation_strategy: DistillationStrategy
    iteration: int
    quality_score: float
    diversity_score: float
    metadata: dict[str, Any] = field(default_factory=dict)
    evolution_history: list[str] = field(default_factory=list)


@dataclass
class DistillationResult:
    """Result of a distillation run."""

    samples: list[GeneratedSample]
    total_generated: int
    quality_passed: int
    average_quality: float
    average_diversity: float
    generation_time_seconds: float
    strategy_used: DistillationStrategy
    metadata: dict[str, Any] = field(default_factory=dict)


class LLMProvider(ABC):
    """Abstract base class for LLM providers."""

    @abstractmethod
    async def generate(self, prompt: str, **kwargs: Any) -> str:
        """Generate a response from the LLM."""
        pass

    @abstractmethod
    def generate_sync(self, prompt: str, **kwargs: Any) -> str:
        """Synchronous generation."""
        pass


class OllamaProvider(LLMProvider):
    """Ollama LLM provider."""

    def __init__(self, model: str = "llama3.2", base_url: str = "http://localhost:11434"):
        self.model = model
        self.base_url = base_url

    async def generate(self, prompt: str, **kwargs: Any) -> str:
        """Async generate using Ollama."""
        import aiohttp

        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{self.base_url}/api/generate",
                json={
                    "model": self.model,
                    "prompt": prompt,
                    "stream": False,
                    **kwargs,
                },
            ) as resp:
                result = await resp.json()
                return result.get("response", "")

    def generate_sync(self, prompt: str, **kwargs: Any) -> str:
        """Synchronous generation using Ollama."""
        import requests

        response = requests.post(
            f"{self.base_url}/api/generate",
            json={
                "model": self.model,
                "prompt": prompt,
                "stream": False,
                **kwargs,
            },
            timeout=120,
        )
        result = response.json()
        return result.get("response", "")


class OpenAIProvider(LLMProvider):
    """OpenAI LLM provider."""

    def __init__(self, model: str = "gpt-4", api_key: Optional[str] = None):
        self.model = model
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")

    async def generate(self, prompt: str, **kwargs: Any) -> str:
        """Async generate using OpenAI."""
        import openai

        client = openai.AsyncOpenAI(api_key=self.api_key)
        response = await client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            **kwargs,
        )
        return response.choices[0].message.content or ""

    def generate_sync(self, prompt: str, **kwargs: Any) -> str:
        """Synchronous generation using OpenAI."""
        import openai

        client = openai.OpenAI(api_key=self.api_key)
        response = client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            **kwargs,
        )
        return response.choices[0].message.content or ""


class SyntheticDataDistillationPipeline:
    """
    Main pipeline for synthetic data distillation.

    Implements multiple strategies for generating high-quality synthetic
    therapeutic conversation data:

    1. Direct Distillation: Generate from teacher model directly
    2. Self-Improvement: Iteratively refine generated samples
    3. Multi-step Prompting: TarGEN-style progressive generation
    4. Cultural Augmentation: Diversify across cultural contexts
    5. Edge Case Expansion: Generate challenging scenarios
    """

    def __init__(self, config: Optional[SyntheticGenerationConfig] = None):
        """Initialize the distillation pipeline."""
        self.config = config or SyntheticGenerationConfig()
        self.provider = self._initialize_provider()

        # Prompt templates
        self._initialize_prompt_templates()

        # Quality evaluation prompts
        self._initialize_quality_prompts()

        logger.info(
            f"SyntheticDataDistillationPipeline initialized: "
            f"strategy={self.config.strategy.value}, "
            f"provider={self.config.model_provider}"
        )

    def _initialize_provider(self) -> LLMProvider:
        """Initialize the LLM provider based on config."""
        if self.config.model_provider == "ollama":
            return OllamaProvider(model=self.config.teacher_model)
        elif self.config.model_provider == "openai":
            return OpenAIProvider(model=self.config.teacher_model)
        else:
            raise ValueError(f"Unsupported provider: {self.config.model_provider}")

    def _initialize_prompt_templates(self) -> None:
        """Initialize prompt templates for different scenarios."""
        self.seed_prompts = {
            "therapeutic_conversation": """Generate a realistic therapeutic conversation between a therapist and a client.
The conversation should demonstrate evidence-based therapeutic techniques and empathetic responses.

Topic: {topic}
Therapeutic Approach: {approach}
Client Concern: {concern}

Requirements:
- 4-6 turns of natural dialogue
- Demonstrate active listening and validation
- Include appropriate therapeutic interventions
- Show progression in the conversation

Generate the conversation in this JSON format:
{{
  "conversation": [
    {{"role": "client", "content": "..."}},
    {{"role": "therapist", "content": "..."}}
  ],
  "metadata": {{
    "therapeutic_techniques": [...],
    "emotional_progression": "...",
    "key_insights": [...]
  }}
}}""",
            "crisis_intervention": """Generate a crisis intervention conversation between a counselor and someone in distress.

Crisis Type: {crisis_type}
Severity: {severity}
Cultural Context: {cultural_context}

Requirements:
- Demonstrate appropriate safety assessment
- Show de-escalation techniques
- Include appropriate referral or safety planning
- Maintain empathetic and non-judgmental tone

Generate the conversation in JSON format with conversation array and metadata.""",
            "cultural_adaptation": """Adapt the following therapeutic conversation to be culturally appropriate for {culture}.

Original Conversation:
{original_conversation}

Requirements:
- Maintain therapeutic integrity
- Adapt language, metaphors, and approaches
- Consider cultural norms around mental health
- Preserve core therapeutic techniques

Generate the adapted conversation in JSON format.""",
            "edge_case": """Generate a challenging therapeutic conversation that demonstrates handling:

Scenario: {scenario}
Difficulty Level: {difficulty}
Specific Challenges: {challenges}

Requirements:
- Show appropriate professional boundaries
- Demonstrate complex clinical reasoning
- Include realistic client resistance or complications
- Show appropriate therapist responses to challenges

Generate in JSON format with detailed metadata about challenges handled.""",
        }

        self.evolution_prompts = {
            PromptEvolutionType.COMPLEXITY_INCREASE: """Take this therapeutic conversation and make it more complex:

{conversation}

Add complexity by:
- Introducing comorbid conditions
- Adding family dynamics
- Including cultural factors
- Deepening emotional layers

Generate the enhanced conversation in JSON format.""",
            PromptEvolutionType.CONTEXT_ENRICHMENT: """Enrich this therapeutic conversation with additional context:

{conversation}

Add context including:
- Client history and background
- Environmental factors
- Relationship dynamics
- Treatment history

Generate the enriched conversation in JSON format.""",
            PromptEvolutionType.SPECIFICITY_ENHANCEMENT: """Make this therapeutic conversation more specific and detailed:

{conversation}

Enhance specificity by:
- Adding specific symptoms and behaviors
- Including concrete examples from client's life
- Using precise therapeutic terminology
- Adding measurable treatment goals

Generate the enhanced conversation in JSON format.""",
        }

    def _initialize_quality_prompts(self) -> None:
        """Initialize prompts for quality evaluation."""
        self.quality_prompt = """Evaluate this therapeutic conversation for quality on a scale of 0.0 to 1.0:

{conversation}

Evaluate these dimensions:
1. Therapeutic Appropriateness (0-1): Are responses clinically appropriate?
2. Empathy Quality (0-1): Does the therapist demonstrate genuine empathy?
3. Conversation Flow (0-1): Is the dialogue natural and coherent?
4. Safety Compliance (0-1): Are safety considerations properly addressed?
5. Educational Value (0-1): Would this be useful for training?

Respond with ONLY a JSON object:
{{
  "therapeutic_appropriateness": 0.0,
  "empathy_quality": 0.0,
  "conversation_flow": 0.0,
  "safety_compliance": 0.0,
  "educational_value": 0.0,
  "overall_score": 0.0,
  "feedback": "..."
}}"""

        self.diversity_prompt = """Evaluate the diversity of this conversation compared to typical therapeutic dialogues:

{conversation}

Consider:
1. Topic uniqueness
2. Approach variety
3. Client presentation diversity
4. Cultural sensitivity
5. Scenario novelty

Respond with ONLY a JSON object:
{{
  "diversity_score": 0.0,
  "unique_elements": [...],
  "improvement_suggestions": [...]
}}"""

    def generate_synthetic_data(
        self,
        num_samples: int,
        seed_topics: Optional[list[str]] = None,
        strategy_override: Optional[DistillationStrategy] = None,
    ) -> DistillationResult:
        """
        Generate synthetic therapeutic conversation data.

        Args:
            num_samples: Number of samples to generate
            seed_topics: Optional seed topics for generation
            strategy_override: Override the default strategy

        Returns:
            DistillationResult with generated samples and metrics
        """
        import time

        start_time = time.time()
        strategy = strategy_override or self.config.strategy

        logger.info(f"Starting synthetic generation: {num_samples} samples, strategy={strategy.value}")

        # Default seed topics if not provided
        if not seed_topics:
            seed_topics = [
                "anxiety management",
                "depression treatment",
                "relationship issues",
                "grief and loss",
                "trauma processing",
                "stress management",
                "self-esteem building",
                "anger management",
                "addiction recovery",
                "family therapy",
            ]

        samples: list[GeneratedSample] = []

        # Generate based on strategy
        if strategy == DistillationStrategy.DIRECT_DISTILLATION:
            samples = self._direct_distillation(num_samples, seed_topics)
        elif strategy == DistillationStrategy.SELF_IMPROVEMENT:
            samples = self._self_improvement_generation(num_samples, seed_topics)
        elif strategy == DistillationStrategy.MULTI_STEP_PROMPTING:
            samples = self._multi_step_generation(num_samples, seed_topics)
        elif strategy == DistillationStrategy.TARGEN_STYLE:
            samples = self._targen_style_generation(num_samples, seed_topics)
        elif strategy == DistillationStrategy.CULTURAL_AUGMENTATION:
            samples = self._cultural_augmentation(num_samples, seed_topics)
        elif strategy == DistillationStrategy.EDGE_CASE_EXPANSION:
            samples = self._edge_case_expansion(num_samples, seed_topics)

        # Filter by quality
        quality_samples = [s for s in samples if s.quality_score >= self.config.quality_threshold]

        elapsed_time = time.time() - start_time

        result = DistillationResult(
            samples=quality_samples,
            total_generated=len(samples),
            quality_passed=len(quality_samples),
            average_quality=sum(s.quality_score for s in samples) / len(samples) if samples else 0,
            average_diversity=sum(s.diversity_score for s in samples) / len(samples) if samples else 0,
            generation_time_seconds=elapsed_time,
            strategy_used=strategy,
            metadata={
                "seed_topics": seed_topics,
                "config": {
                    "teacher_model": self.config.teacher_model,
                    "temperature": self.config.temperature,
                    "quality_threshold": self.config.quality_threshold,
                },
            },
        )

        logger.info(
            f"Generation complete: {result.quality_passed}/{result.total_generated} passed quality, "
            f"avg_quality={result.average_quality:.2f}, time={elapsed_time:.1f}s"
        )

        return result

    def _direct_distillation(
        self, num_samples: int, seed_topics: list[str]
    ) -> list[GeneratedSample]:
        """Direct distillation from teacher model."""
        samples = []

        therapeutic_approaches = [
            "CBT", "DBT", "ACT", "Psychodynamic", "Humanistic",
            "Solution-Focused", "Narrative Therapy", "EMDR"
        ]

        client_concerns = [
            "persistent worry and anxiety",
            "feelings of sadness and hopelessness",
            "relationship difficulties",
            "work-related stress",
            "trauma symptoms",
            "self-esteem issues",
            "anger control problems",
            "substance use concerns",
        ]

        for i in range(num_samples):
            topic = random.choice(seed_topics)
            approach = random.choice(therapeutic_approaches)
            concern = random.choice(client_concerns)

            prompt = self.seed_prompts["therapeutic_conversation"].format(
                topic=topic,
                approach=approach,
                concern=concern,
            )

            try:
                response = self.provider.generate_sync(
                    prompt,
                    temperature=self.config.temperature,
                )

                sample = self._parse_and_create_sample(
                    response,
                    DistillationStrategy.DIRECT_DISTILLATION,
                    iteration=0,
                    metadata={"topic": topic, "approach": approach, "concern": concern},
                )

                if sample:
                    samples.append(sample)
                    logger.debug(f"Generated sample {i+1}/{num_samples}")

            except Exception as e:
                logger.warning(f"Error generating sample {i+1}: {e}")
                continue

        return samples

    def _self_improvement_generation(
        self, num_samples: int, seed_topics: list[str]
    ) -> list[GeneratedSample]:
        """Generate with self-improvement iterations."""
        samples = []

        # First, generate initial samples
        initial_samples = self._direct_distillation(num_samples, seed_topics)

        for sample in initial_samples:
            improved_sample = sample

            # Iterate to improve
            for iteration in range(1, self.config.num_iterations + 1):
                improvement_prompt = f"""Improve this therapeutic conversation while maintaining its core therapeutic value:

{self._sample_to_text(improved_sample)}

Current Quality Score: {improved_sample.quality_score:.2f}

Improvements to make:
1. Enhance empathetic responses
2. Add more specific therapeutic techniques
3. Improve conversation flow
4. Add realistic client emotional progression

Generate the improved conversation in JSON format."""

                try:
                    response = self.provider.generate_sync(
                        improvement_prompt,
                        temperature=self.config.temperature * 0.9,  # Slightly lower for refinement
                    )

                    new_sample = self._parse_and_create_sample(
                        response,
                        DistillationStrategy.SELF_IMPROVEMENT,
                        iteration=iteration,
                        metadata=improved_sample.metadata,
                    )

                    if new_sample and new_sample.quality_score > improved_sample.quality_score:
                        new_sample.evolution_history = [
                            *improved_sample.evolution_history,
                            f"iteration_{iteration}_quality_{new_sample.quality_score:.2f}",
                        ]
                        improved_sample = new_sample

                except Exception as e:
                    logger.warning(f"Error in improvement iteration {iteration}: {e}")
                    continue

            samples.append(improved_sample)

        return samples

    def _multi_step_generation(
        self, num_samples: int, seed_topics: list[str]
    ) -> list[GeneratedSample]:
        """Multi-step prompting generation."""
        samples = []

        for i in range(num_samples):
            topic = random.choice(seed_topics)

            # Step 1: Generate scenario context
            context_prompt = f"""Create a detailed therapy scenario context for the topic: {topic}

Include:
- Client background (brief)
- Presenting problem
- Relevant history
- Session context (first session, ongoing, etc.)

Respond with JSON: {{"context": "...", "client_profile": "...", "session_type": "..."}}"""

            try:
                context_response = self.provider.generate_sync(context_prompt)
                context_data = self._safe_json_parse(context_response)

                # Step 2: Generate conversation outline
                outline_prompt = f"""Based on this therapy context, create an outline for a therapeutic conversation:

Context: {json.dumps(context_data)}
Topic: {topic}

Create an outline with:
- Opening approach
- Key exploration points
- Therapeutic interventions to use
- Closing/homework

Respond with JSON: {{"outline": [...], "techniques": [...], "goals": [...]}}"""

                outline_response = self.provider.generate_sync(outline_prompt)
                outline_data = self._safe_json_parse(outline_response)

                # Step 3: Generate full conversation
                conversation_prompt = f"""Generate a complete therapeutic conversation based on:

Context: {json.dumps(context_data)}
Outline: {json.dumps(outline_data)}
Topic: {topic}

Create a natural, realistic conversation that follows the outline.

Generate in JSON format with conversation array and detailed metadata."""

                conversation_response = self.provider.generate_sync(
                    conversation_prompt,
                    temperature=self.config.temperature,
                )

                sample = self._parse_and_create_sample(
                    conversation_response,
                    DistillationStrategy.MULTI_STEP_PROMPTING,
                    iteration=0,
                    metadata={
                        "topic": topic,
                        "context": context_data,
                        "outline": outline_data,
                    },
                )

                if sample:
                    sample.evolution_history = ["context", "outline", "conversation"]
                    samples.append(sample)
                    logger.debug(f"Multi-step generated sample {i+1}/{num_samples}")

            except Exception as e:
                logger.warning(f"Error in multi-step generation {i+1}: {e}")
                continue

        return samples

    def _targen_style_generation(
        self, num_samples: int, seed_topics: list[str]
    ) -> list[GeneratedSample]:
        """TarGEN-style generation with query evolution."""
        samples = []

        evolution_types = list(PromptEvolutionType)

        for i in range(num_samples):
            topic = random.choice(seed_topics)

            # Start with basic generation
            basic_prompt = f"""Generate a basic therapeutic conversation about {topic}.
Keep it simple but clinically appropriate.
Generate in JSON format with conversation array."""

            try:
                basic_response = self.provider.generate_sync(basic_prompt)
                current_conv = basic_response

                # Evolve through multiple stages
                evolution_history = ["basic"]

                for j in range(min(3, len(evolution_types))):
                    evolution_type = random.choice(evolution_types)
                    evolution_prompt = self.evolution_prompts.get(
                        evolution_type,
                        self.evolution_prompts[PromptEvolutionType.COMPLEXITY_INCREASE],
                    ).format(conversation=current_conv)

                    evolved_response = self.provider.generate_sync(
                        evolution_prompt,
                        temperature=self.config.temperature,
                    )

                    current_conv = evolved_response
                    evolution_history.append(evolution_type.value)

                sample = self._parse_and_create_sample(
                    current_conv,
                    DistillationStrategy.TARGEN_STYLE,
                    iteration=len(evolution_history) - 1,
                    metadata={"topic": topic, "evolutions": evolution_history},
                )

                if sample:
                    sample.evolution_history = evolution_history
                    samples.append(sample)
                    logger.debug(f"TarGEN generated sample {i+1}/{num_samples}")

            except Exception as e:
                logger.warning(f"Error in TarGEN generation {i+1}: {e}")
                continue

        return samples

    def _cultural_augmentation(
        self, num_samples: int, seed_topics: list[str]
    ) -> list[GeneratedSample]:
        """Generate culturally diverse samples."""
        samples = []
        samples_per_culture = max(1, num_samples // len(self.config.cultural_contexts))

        for culture in self.config.cultural_contexts:
            for i in range(samples_per_culture):
                topic = random.choice(seed_topics)

                prompt = f"""Generate a therapeutic conversation that is culturally appropriate for {culture} culture.

Topic: {topic}
Cultural Context: {culture}

Consider:
- Cultural attitudes toward mental health
- Appropriate communication styles
- Family and community dynamics
- Cultural metaphors and expressions
- Stigma considerations

Generate in JSON format with conversation array and cultural adaptation notes in metadata."""

                try:
                    response = self.provider.generate_sync(
                        prompt,
                        temperature=self.config.temperature,
                    )

                    sample = self._parse_and_create_sample(
                        response,
                        DistillationStrategy.CULTURAL_AUGMENTATION,
                        iteration=0,
                        metadata={"topic": topic, "cultural_context": culture},
                    )

                    if sample:
                        samples.append(sample)

                except Exception as e:
                    logger.warning(f"Error in cultural augmentation for {culture}: {e}")
                    continue

        return samples

    def _edge_case_expansion(
        self, num_samples: int, seed_topics: list[str]
    ) -> list[GeneratedSample]:
        """Generate challenging edge case scenarios."""
        samples = []

        edge_cases = [
            {"scenario": "Client expresses suicidal ideation mid-session", "difficulty": "very_high"},
            {"scenario": "Client becomes angry and confrontational", "difficulty": "high"},
            {"scenario": "Client discloses abuse of a minor", "difficulty": "very_high"},
            {"scenario": "Client shows signs of psychosis", "difficulty": "high"},
            {"scenario": "Client requests inappropriate relationship", "difficulty": "high"},
            {"scenario": "Client refuses to engage in treatment", "difficulty": "moderate"},
            {"scenario": "Client reports severe dissociation", "difficulty": "high"},
            {"scenario": "Client presents with complex trauma history", "difficulty": "high"},
            {"scenario": "Client has comorbid substance use", "difficulty": "moderate"},
            {"scenario": "Therapeutic rupture occurs in session", "difficulty": "moderate"},
        ]

        for i in range(num_samples):
            edge_case = random.choice(edge_cases)
            topic = random.choice(seed_topics)

            prompt = self.seed_prompts["edge_case"].format(
                scenario=edge_case["scenario"],
                difficulty=edge_case["difficulty"],
                challenges=f"Related to {topic}; requires clinical judgment and safety considerations",
            )

            try:
                response = self.provider.generate_sync(
                    prompt,
                    temperature=self.config.temperature,
                )

                sample = self._parse_and_create_sample(
                    response,
                    DistillationStrategy.EDGE_CASE_EXPANSION,
                    iteration=0,
                    metadata={
                        "topic": topic,
                        "edge_case_scenario": edge_case["scenario"],
                        "difficulty": edge_case["difficulty"],
                    },
                )

                if sample:
                    samples.append(sample)
                    logger.debug(f"Edge case generated: {edge_case['scenario'][:50]}...")

            except Exception as e:
                logger.warning(f"Error in edge case generation: {e}")
                continue

        return samples

    def _parse_and_create_sample(
        self,
        response: str,
        strategy: DistillationStrategy,
        iteration: int,
        metadata: dict[str, Any],
    ) -> Optional[GeneratedSample]:
        """Parse LLM response and create a GeneratedSample."""
        try:
            # Parse JSON from response
            data = self._safe_json_parse(response)

            if not data:
                return None

            # Extract conversation
            conversation_data = data.get("conversation", data.get("messages", []))
            if not conversation_data:
                # Try to extract from response text
                conversation_data = self._extract_conversation_from_text(response)

            if not conversation_data:
                return None

            # Convert to Messages
            messages = []
            for turn in conversation_data:
                if isinstance(turn, dict):
                    role = turn.get("role", turn.get("speaker", "user"))
                    content = turn.get("content", turn.get("text", ""))

                    # Normalize role names
                    if role.lower() in ["client", "user", "patient"]:
                        role = "user"
                    elif role.lower() in ["therapist", "assistant", "counselor"]:
                        role = "assistant"

                    if content:
                        messages.append(Message(role=role, content=content))

            if len(messages) < 2:
                return None

            # Generate conversation ID
            conv_id = self._generate_conversation_id(strategy.value, iteration, messages[0].content)

            # Evaluate quality
            quality_score = self._evaluate_quality(messages)
            diversity_score = self._evaluate_diversity(messages) if self.config.enable_diversity_scoring else 0.5

            return GeneratedSample(
                conversation_id=conv_id,
                messages=messages,
                generation_strategy=strategy,
                iteration=iteration,
                quality_score=quality_score,
                diversity_score=diversity_score,
                metadata={
                    **metadata,
                    "response_metadata": data.get("metadata", {}),
                },
            )

        except Exception as e:
            logger.warning(f"Error parsing sample: {e}")
            return None

    def _safe_json_parse(self, text: str) -> dict[str, Any]:
        """Safely parse JSON from text that may contain extra content."""
        try:
            # Try direct parse first
            return json.loads(text)
        except json.JSONDecodeError:
            pass

        # Try to find JSON in the text
        json_patterns = [
            r'\{[\s\S]*\}',  # Match { ... }
            r'\[[\s\S]*\]',  # Match [ ... ]
        ]

        for pattern in json_patterns:
            matches = re.findall(pattern, text)
            for match in matches:
                try:
                    return json.loads(match)
                except json.JSONDecodeError:
                    continue

        return {}

    def _extract_conversation_from_text(self, text: str) -> list[dict[str, str]]:
        """Extract conversation turns from plain text response."""
        conversation = []

        # Try to find dialogue patterns
        patterns = [
            r'(?:Client|User|Patient):\s*(.+?)(?=(?:Therapist|Assistant|Counselor):|$)',
            r'(?:Therapist|Assistant|Counselor):\s*(.+?)(?=(?:Client|User|Patient):|$)',
        ]

        lines = text.split('\n')
        current_role = None
        current_content = []

        for line in lines:
            line = line.strip()
            if not line:
                continue

            # Check for role markers
            if any(marker in line.lower() for marker in ['client:', 'user:', 'patient:']):
                if current_role and current_content:
                    conversation.append({
                        "role": current_role,
                        "content": ' '.join(current_content),
                    })
                current_role = "user"
                content = re.sub(r'^(client|user|patient):\s*', '', line, flags=re.IGNORECASE)
                current_content = [content] if content else []

            elif any(marker in line.lower() for marker in ['therapist:', 'assistant:', 'counselor:']):
                if current_role and current_content:
                    conversation.append({
                        "role": current_role,
                        "content": ' '.join(current_content),
                    })
                current_role = "assistant"
                content = re.sub(r'^(therapist|assistant|counselor):\s*', '', line, flags=re.IGNORECASE)
                current_content = [content] if content else []

            elif current_role:
                current_content.append(line)

        # Add last turn
        if current_role and current_content:
            conversation.append({
                "role": current_role,
                "content": ' '.join(current_content),
            })

        return conversation

    def _evaluate_quality(self, messages: list[Message]) -> float:
        """Evaluate quality of generated conversation."""
        if not self.config.enable_self_evaluation:
            return 0.75  # Default score if evaluation disabled

        try:
            conv_text = "\n".join([f"{m.role}: {m.content}" for m in messages])
            prompt = self.quality_prompt.format(conversation=conv_text)

            response = self.provider.generate_sync(prompt, temperature=0.1)
            data = self._safe_json_parse(response)

            return data.get("overall_score", 0.5)

        except Exception as e:
            logger.warning(f"Error evaluating quality: {e}")
            return 0.5

    def _evaluate_diversity(self, messages: list[Message]) -> float:
        """Evaluate diversity of generated conversation."""
        try:
            conv_text = "\n".join([f"{m.role}: {m.content}" for m in messages])
            prompt = self.diversity_prompt.format(conversation=conv_text)

            response = self.provider.generate_sync(prompt, temperature=0.1)
            data = self._safe_json_parse(response)

            return data.get("diversity_score", 0.5)

        except Exception as e:
            logger.warning(f"Error evaluating diversity: {e}")
            return 0.5

    def _sample_to_text(self, sample: GeneratedSample) -> str:
        """Convert sample to text representation."""
        return "\n".join([f"{m.role}: {m.content}" for m in sample.messages])

    def _generate_conversation_id(
        self, strategy: str, iteration: int, content_preview: str
    ) -> str:
        """Generate unique conversation ID."""
        unique_string = f"{strategy}_{iteration}_{datetime.now().isoformat()}_{content_preview[:50]}"
        hash_value = hashlib.md5(unique_string.encode()).hexdigest()[:12]
        return f"syn_{hash_value}"

    def convert_to_conversations(
        self, result: DistillationResult
    ) -> list[Conversation]:
        """Convert DistillationResult samples to standard Conversation objects."""
        conversations = []

        for sample in result.samples:
            conversation = Conversation(
                conversation_id=sample.conversation_id,
                source=f"synthetic_{sample.generation_strategy.value}",
                messages=sample.messages,
                metadata={
                    "synthetic": True,
                    "generation_strategy": sample.generation_strategy.value,
                    "iteration": sample.iteration,
                    "quality_score": sample.quality_score,
                    "diversity_score": sample.diversity_score,
                    "evolution_history": sample.evolution_history,
                    **sample.metadata,
                },
            )
            conversations.append(conversation)

        return conversations

    def save_results(
        self, result: DistillationResult, output_path: Path
    ) -> None:
        """Save distillation results to file."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Convert to serializable format
        data = {
            "metadata": {
                "total_generated": result.total_generated,
                "quality_passed": result.quality_passed,
                "average_quality": result.average_quality,
                "average_diversity": result.average_diversity,
                "generation_time_seconds": result.generation_time_seconds,
                "strategy": result.strategy_used.value,
                "generated_at": datetime.now(timezone.utc).isoformat(),
                **result.metadata,
            },
            "samples": [
                {
                    "conversation_id": s.conversation_id,
                    "messages": [{"role": m.role, "content": m.content} for m in s.messages],
                    "generation_strategy": s.generation_strategy.value,
                    "iteration": s.iteration,
                    "quality_score": s.quality_score,
                    "diversity_score": s.diversity_score,
                    "evolution_history": s.evolution_history,
                    "metadata": s.metadata,
                }
                for s in result.samples
            ],
        }

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

        logger.info(f"Saved {len(result.samples)} samples to {output_path}")


# Factory function for easy instantiation
def create_distillation_pipeline(
    strategy: str = "multi_step_prompting",
    provider: str = "ollama",
    model: str = "llama3.2",
    quality_threshold: float = 0.8,
) -> SyntheticDataDistillationPipeline:
    """
    Create a distillation pipeline with common configurations.

    Args:
        strategy: Generation strategy (direct_distillation, self_improvement,
                 multi_step_prompting, targen_style, cultural_augmentation, edge_case_expansion)
        provider: LLM provider (ollama, openai)
        model: Model name
        quality_threshold: Minimum quality score for samples

    Returns:
        Configured SyntheticDataDistillationPipeline
    """
    strategy_enum = DistillationStrategy(strategy)

    config = SyntheticGenerationConfig(
        strategy=strategy_enum,
        model_provider=provider,
        teacher_model=model,
        quality_threshold=quality_threshold,
    )

    return SyntheticDataDistillationPipeline(config)


# Example usage and testing
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    print("Synthetic Data Distillation Pipeline")
    print("=" * 50)

    # Create pipeline
    pipeline = create_distillation_pipeline(
        strategy="direct_distillation",
        provider="ollama",
        model="llama3.2",
    )

    print("\nAvailable strategies:")
    for strategy in DistillationStrategy:
        print(f"  - {strategy.value}")

    print("\nTesting generation (this requires Ollama running)...")
    print("Run: ollama serve && ollama pull llama3.2")

    # Uncomment to test:
    # result = pipeline.generate_synthetic_data(
    #     num_samples=3,
    #     seed_topics=["anxiety management", "depression treatment"]
    # )
    # print(f"\nGenerated {result.quality_passed} quality samples")
    # conversations = pipeline.convert_to_conversations(result)
    # for conv in conversations[:2]:
    #     print(f"\n{conv.conversation_id}:")
    #     for msg in conv.messages[:2]:
    #         print(f"  [{msg.role}]: {msg.content[:100]}...")

