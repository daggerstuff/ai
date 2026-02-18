"""
LLM-based Taxonomy Classifier - Phase 2

This module provides LLM-powered classification for therapeutic conversations
that cannot be confidently classified using keyword-based methods.

Uses NVIDIA NIM with GLM4.7 for intelligent classification with reasoning.
"""

import json
import logging
import os
from typing import Any, Dict, List, Optional
from dataclasses import dataclass

from openai import OpenAI

from ai.pipelines.design.taxonomy_classifier import (
    TherapeuticCategory,
    CategoryClassification,
)

logger = logging.getLogger(__name__)


@dataclass
class LLMClassificationConfig:
    """Configuration for LLM classification."""

    model: str = "z-ai/glm4.7"  # NVIDIA NIM GLM4.7 model
    base_url: str = "https://integrate.api.nvidia.com/v1"
    temperature: float = 0.1  # Low temp for consistent classification
    max_tokens: int = 2000  # Increased for reasoning models
    confidence_threshold: float = 0.70
    enable_reasoning: bool = True


class LLMTaxonomyClassifier:
    """
    LLM-powered classifier for therapeutic conversations.

    Uses NVIDIA NIM with GLM4.7 to classify conversations that keyword-based methods
    struggle with, providing detailed reasoning for classifications.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        config: Optional[LLMClassificationConfig] = None,
    ):
        """
        Initialize the LLM classifier with NVIDIA NIM.

        Args:
            api_key: NVIDIA API key (defaults to OPENAI_API_KEY env var)
            config: Classification configuration
        """
        self.config = config or LLMClassificationConfig()
        self.client = OpenAI(
            api_key=api_key or os.getenv("OPENAI_API_KEY"),
            base_url=os.getenv("OPENAI_BASE_URL") or self.config.base_url,
        )

        # System prompt for classification
        self.system_prompt = self._build_system_prompt()

    def _build_system_prompt(self) -> str:
        """Build the system prompt for classification."""
        return """You are an expert clinical psychologist specializing in therapeutic conversation analysis.

Your task is to classify therapeutic conversations into ONE of these 6 categories:

1. **therapeutic_conversation** - Standard therapy sessions with general discussion, progress tracking, and therapeutic exploration
2. **crisis_support** - Active crisis intervention (suicide risk, self-harm, immediate danger)
3. **mental_health_support** - Mental health guidance (depression, anxiety, stress management, coping strategies)
4. **trauma_processing** - PTSD, abuse, assault, or trauma-focused therapy
5. **relationship_therapy** - Couples, family, or interpersonal relationship issues
6. **clinical_assessment** - Diagnosis, evaluation, intake sessions, or symptom screening

CLASSIFICATION GUIDELINES:
- **Priority Order**: If multiple categories apply, use this priority:
  1. crisis_support (if ANY crisis indicators)
  2. trauma_processing (if processing traumatic events)
  3. relationship_therapy (if primary focus is relationships)
  4. clinical_assessment (if diagnostic/evaluation focus)
  5. mental_health_support (if general mental health focus)
  6. therapeutic_conversation (default for general sessions)

- **Crisis Indicators**: suicidal thoughts, self-harm, immediate danger, wanting to die
- **Trauma Indicators**: PTSD symptoms, processing abuse/assault, flashbacks, trauma-specific therapy
- **Relationship Indicators**: couples therapy, family conflicts, relationship problems
- **Assessment Indicators**: diagnostic criteria, symptom checklists (PHQ-9, GAD-7), intake evaluations
- **Mental Health Indicators**: depression/anxiety management, coping strategies, wellness focus
- **Therapeutic Conversation**: General therapy work without specific crisis/trauma/relationship focus

Respond in JSON format:
{
    "category": "category_name",
    "confidence": 0.85,
    "reasoning": "Brief explanation of why this category was chosen",
    "key_indicators": ["indicator1", "indicator2", "indicator3"]
}

Be decisive but honest about confidence. Use confidence scores:
- 0.90-1.00: Very clear category indicators
- 0.75-0.89: Strong indicators with some ambiguity
- 0.60-0.74: Moderate confidence, could fit multiple categories
- Below 0.60: Unclear, use therapeutic_conversation as default"""

    def _build_user_prompt(self, conversation_text: str) -> str:
        """
        Build the user prompt with conversation text.

        Args:
            conversation_text: The conversation to classify

        Returns:
            Formatted user prompt
        """
        # Truncate if too long (keep first 4000 chars for context)
        if len(conversation_text) > 4000:
            conversation_text = (
                conversation_text[:4000] + "\n\n[... conversation truncated ...]"
            )

        return f"""Classify this therapeutic conversation:

{conversation_text}

Provide your classification in JSON format."""

    def classify(self, conversation_text: str) -> CategoryClassification:
        """
        Classify a conversation using LLM.

        Args:
            conversation_text: The conversation text to classify

        Returns:
            Classification result with category, confidence, and reasoning
        """
        try:
            response = self.client.chat.completions.create(
                model=self.config.model,
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {
                        "role": "user",
                        "content": self._build_user_prompt(conversation_text),
                    },
                ],
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens,
                # Note: Some models don't support response_format parameter
                # response_format={"type": "json_object"}
            )

            # Parse LLM response
            # Some reasoning models use reasoning_content instead of content
            message = response.choices[0].message
            content = message.content or getattr(message, "reasoning_content", None)

            if not content:
                raise ValueError(f"Empty response from LLM. Full response: {response}")

            # Try to extract JSON from response (handle markdown code blocks and reasoning text)
            content = content.strip()

            # Remove markdown code fences
            if content.startswith("```json"):
                content = content[7:]
            elif content.startswith("```"):
                content = content[3:]
            if content.endswith("```"):
                content = content[:-3]
            content = content.strip()

            # Try to find JSON object in the content
            # Some models may include reasoning text before/after JSON
            start_idx = content.find("{")
            end_idx = content.rfind("}")

            if start_idx != -1 and end_idx != -1 and start_idx < end_idx:
                json_str = content[start_idx : end_idx + 1]
                result = json.loads(json_str)
            else:
                # Fallback: try parsing whole content
                result = json.loads(content)

            # Validate category
            category_str = result.get("category", "therapeutic_conversation")
            try:
                category = TherapeuticCategory(category_str)
            except ValueError:
                logger.warning(
                    f"Invalid category from LLM: {category_str}, using default"
                )
                category = TherapeuticCategory.THERAPEUTIC_CONVERSATION

            confidence = float(result.get("confidence", 0.70))
            reasoning = result.get("reasoning", "LLM classification")
            key_indicators = result.get("key_indicators", [])

            return CategoryClassification(
                category=category,
                confidence=confidence,
                reasoning=f"LLM: {reasoning}",
                keywords_detected=key_indicators[:5],
            )

        except Exception as e:
            logger.error(f"LLM classification error: {e}")
            # Fallback to default
            return CategoryClassification(
                category=TherapeuticCategory.THERAPEUTIC_CONVERSATION,
                confidence=0.50,
                reasoning=f"LLM classification failed: {str(e)}",
                keywords_detected=[],
            )

    def classify_batch(
        self, conversations: List[str], show_progress: bool = True
    ) -> List[CategoryClassification]:
        """
        Classify multiple conversations.

        Args:
            conversations: List of conversation texts
            show_progress: Whether to show progress

        Returns:
            List of classification results
        """
        results = []
        total = len(conversations)

        for i, text in enumerate(conversations):
            if show_progress and (i + 1) % 10 == 0:
                logger.info(f"Classified {i + 1}/{total} conversations")

            result = self.classify(text)
            results.append(result)

        return results


def main():
    """Example usage of LLM classifier."""
    import argparse

    parser = argparse.ArgumentParser(
        description="LLM-based conversation classification with NVIDIA NIM"
    )
    parser.add_argument("--text", type=str, help="Text to classify")
    parser.add_argument(
        "--model", type=str, default="z-ai/glm4.7", help="NVIDIA NIM model"
    )

    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    config = LLMClassificationConfig(model=args.model)
    classifier = LLMTaxonomyClassifier(config=config)

    # Test classification
    test_text = (
        args.text
        or """
    Patient: I've been having intrusive thoughts about the car accident. 
    I keep seeing it happen over and over again.
    Therapist: Those sound like flashbacks. Can you tell me more about when these occur?
    Patient: Mostly when I'm driving or hear loud noises. My heart races and I feel like I'm back there.
    Therapist: That's a common PTSD symptom. Let's work on some grounding techniques.
    """
    )

    result = classifier.classify(test_text)

    print("\n" + "=" * 80)
    print("🤖 LLM CLASSIFICATION RESULT")
    print("=" * 80)
    print(f"Category: {result.category.value}")
    print(f"Confidence: {result.confidence:.2%}")
    print(f"Reasoning: {result.reasoning}")
    print(f"Key Indicators: {', '.join(result.keywords_detected)}")
    print("=" * 80)


if __name__ == "__main__":
    main()
