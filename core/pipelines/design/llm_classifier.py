"""
LLM-based Taxonomy Classifier - Phase 2

This module provides LLM-powered classification for therapeutic conversations
that cannot be confidently classified using keyword-based methods.

Uses NVIDIA NIM with GLM4.7 for intelligent classification with reasoning.
"""

import json
import logging
import os
from dataclasses import dataclass
from typing import List, Optional

from ai.core.pipelines.design.context_detector import ContextDetector
from ai.core.pipelines.design.reasoning_parser import ReasoningOutputParser
from ai.core.pipelines.design.situational_awareness import SituationalAwarenessAgent
from ai.core.pipelines.design.taxonomy_classifier import (
    CategoryClassification,
    TherapeuticCategory,
)
from openai import OpenAI

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
    timeout: int = 60  # Timeout for API calls in seconds


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
            timeout=self.config.timeout,
        )

        # System prompt for classification
        self.system_prompt = self._build_system_prompt()

        # Reasoning parser for GLM4.7 output
        self.reasoning_parser = ReasoningOutputParser()

        # Context detector for educational/theoretical detection
        self.context_detector = ContextDetector()

        # Situational awareness agent for deeper analysis
        self.situational_agent = SituationalAwarenessAgent()

    def _build_system_prompt(self) -> str:
        """Build the system prompt for classification."""
        return (
            "You are an expert clinical psychologist specializing in "
            "therapeutic conversation classification with CONTEXT "
            "AWARENESS.\n"
            "\n"
            "## CRITICAL CONTEXT RULES\n"
            "Distinguish between:\n"
            "1. ACTUAL THERAPY: Patient discussing their OWN issues "
            '(first-person: "I feel", "my trauma")\n'
            "2. EDUCATIONAL: Training, discussing therapy techniques "
            '(third-person: "clients who", "patients with")\n'
            '3. THEORETICAL: Hypothetical scenarios, "therapists '
            'might..."\n'
            "4. META: Talking ABOUT therapy, not doing actual therapy\n"
            "\n"
            "IF EDUCATIONAL/THEORETICAL/META → classify as "
            '"therapeutic_conversation" with confidence <0.60\n'
            "\n"
            "## CATEGORIES (strict definitions)\n"
            "\n"
            "1. **crisis_support** — ACTIVE, IMMEDIATE danger: "
            'suicidal ideation (including passive: "better off '
            'without me"), self-harm, acute crisis RIGHT NOW\n'
            "\n"
            "2. **trauma_processing** — Patient's OWN trauma "
            "experiences: PTSD, abuse, assault, flashbacks, "
            "nightmares from traumatic events\n"
            "\n"
            "3. **relationship_therapy** — Interpersonal conflicts: "
            "couples, family, partner, domestic issues, "
            "communication breakdown\n"
            "\n"
            "4. **clinical_assessment** — Formal diagnostic "
            "evaluation: screening tools (PHQ-9, GAD-7), DSM "
            "criteria, psychiatric intake, medication evaluation\n"
            "\n"
            "5. **mental_health_support** — CLINICAL mental health "
            "symptoms: diagnosed depression, clinical anxiety, "
            "panic attacks, sleep disorders, OCD, eating disorders, "
            "substance abuse, specific phobias. ALSO includes "
            "severe stress, burnout, and emotional overwhelm.\n"
            "\n"
            "6. **therapeutic_conversation** — GENERAL life "
            "challenges, personal growth, self-improvement, "
            "work-life balance, finding purpose, self-esteem, "
            "confidence building, career stress, existential "
            "questions, life transitions. Focus is on optimization "
            "and growth rather than symptom management.\n"
            "\n"
            "## KEY BOUNDARY: mental_health_support vs "
            "therapeutic_conversation\n"
            "\n"
            'Ask: "Is the person describing CLINICAL SYMPTOMS or '
            'SIGNIFICANT DISTRESS?"\n'
            '- YES → mental_health_support ("I\'m depressed", '
            '"panic attacks", "can\'t function", "stress/burnout")\n'
            '- NO → therapeutic_conversation ("want to improve", '
            '"finding purpose", "building confidence", '
            '"career planning")\n'
            "\n"
            "## OUTPUT FORMAT\n"
            "After your analysis, you MUST end with a JSON object:\n"
            '{"category": "category_name", "confidence": 0.85, '
            '"reasoning": "brief explanation", "key_indicators": '
            '["indicator1", "indicator2"]}'
        )

    def _build_user_prompt(self, conversation_text: str, situation=None) -> str:
        """
        Build the user prompt with conversation text and situational context.

        Args:
            conversation_text: The conversation to classify
            situation: Optional SituationalContext from awareness agent

        Returns:
            Formatted user prompt
        """
        # Truncate if too long (keep first 4000 chars for context)
        if len(conversation_text) > 4000:
            conversation_text = (
                conversation_text[:4000] + "\n\n[... conversation truncated ...]"
            )

        # Build context hints from situational analysis
        context_hints = ""
        if situation:
            hints = []
            if situation.is_growth_focused and situation.has_metaphorical_language:
                hints.append(
                    "⚠️ GROWTH/METAPHORICAL language detected - NOT literal crisis"
                )
            if situation.is_relationship_focused and not situation.is_crisis:
                hints.append(
                    "⚠️ RELATIONSHIP context - consider relationship_therapy over crisis"
                )
            if situation.has_processing_language:
                hints.append("⚠️ PROCESSING language - active coping, NOT acute crisis")
            if situation.is_past_tense:
                hints.append(
                    "⚠️ PAST TENSE - historical/processed, NOT current active issue"
                )
            if situation.is_assessment:
                hints.append("⚠️ ASSESSMENT markers - consider clinical_assessment")

            if hints:
                context_hints = "\n\n**CONTEXTUAL AWARENESS:**\n" + "\n".join(hints)
                if situation.reasoning:
                    context_hints += f"\n**Analysis**: {situation.reasoning}"

        return f"""Classify this therapeutic conversation:

{conversation_text}{context_hints}

Provide your analysis and end with the JSON classification object."""

    def classify(self, conversation_text: str) -> CategoryClassification:
        """
        Classify a conversation using LLM with context awareness.

        Args:
            conversation_text: The conversation text to classify

        Returns:
            Classification result with category, confidence, and reasoning
        """
        try:
            # Pre-check context to help LLM
            context = self.context_detector.detect_context(conversation_text)

            # Get situational awareness
            situation = self.situational_agent.analyze(conversation_text)

            # If strongly educational/theoretical, downgrade confidence
            if not context.is_therapeutic and context.confidence >= 0.8:
                logger.info(
                    f"Educational/theoretical context detected: {context.indicators}"
                )
                # Still send to LLM but it will be informed via prompt
            response = self.client.chat.completions.create(
                model=self.config.model,
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {
                        "role": "user",
                        "content": self._build_user_prompt(
                            conversation_text, situation
                        ),
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

            # Try to extract JSON from response
            # (handle markdown code blocks and reasoning text)
            content = content.strip()

            # Remove markdown code fences
            if content.startswith("```json"):
                content = content[7:]
            elif content.startswith("```"):
                content = content[3:]
            if content.endswith("```"):
                content = content[:-3]
            content = content.strip()

            # GLM4.7 is a reasoning model that provides analysis, not JSON
            # Try two strategies:
            # 1. Look for JSON in the response (if prompt worked)
            # 2. Parse the reasoning text using our dedicated parser

            import re

            # Strategy 1: Look for JSON objects
            json_pattern = r"\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}"
            json_matches = list(re.finditer(json_pattern, content))

            result = None
            for match in reversed(json_matches):
                json_str = match.group()
                try:
                    candidate = json.loads(json_str)
                    if "category" in candidate and "confidence" in candidate:
                        result = candidate
                        logger.debug(f"Found valid JSON at position {match.start()}")
                        break
                except json.JSONDecodeError:
                    continue

            # Strategy 2: If no JSON found, parse the reasoning text
            if not result:
                logger.info("No JSON found - using reasoning parser")
                parsed_result = self.reasoning_parser.parse_reasoning_output(content)
                # Convert CategoryClassification to dict format
                result = {
                    "category": parsed_result.category.value,
                    "confidence": parsed_result.confidence,
                    "reasoning": parsed_result.reasoning,
                    "key_indicators": parsed_result.keywords_detected,
                }

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

            # Post-process with context detector
            # If educational context detected, cap confidence
            # and ensure appropriate category
            if not context.is_therapeutic and context.confidence >= 0.7:
                # Force lower confidence for educational content
                if confidence > 0.65:
                    confidence = min(confidence, 0.55)
                    reasoning += (
                        " [Context: Educational/theoretical - confidence capped]"
                    )

                # Downgrade crisis/trauma to general therapeutic if educational
                if category in [
                    TherapeuticCategory.CRISIS_SUPPORT,
                    TherapeuticCategory.TRAUMA_PROCESSING,
                ]:
                    logger.warning(
                        "Educational context detected but classified as "
                        f"{category.value} - downgrading to "
                        "therapeutic_conversation"
                    )
                    category = TherapeuticCategory.THERAPEUTIC_CONVERSATION
                    reasoning += (
                        " [Downgraded from crisis/trauma due to educational context]"
                    )

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
    test_text = args.text or """
    Patient: I've been having intrusive thoughts about the car accident.
    I keep seeing it happen over and over again.
    Therapist: Those sound like flashbacks. Can you tell me more
        about when these occur?
    Patient: Mostly when I'm driving or hear loud noises. My heart
        races and I feel like I'm back there.
    Therapist: That's a common PTSD symptom. Let's work on some
        grounding techniques.
    """

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
