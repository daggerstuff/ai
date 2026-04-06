#!/usr/bin/env python3
"""
Reasoning Completeness Validator for Stage 2 Training Data.

Per MasterTrainingPlan.md:
- Validates Chain of Thought (CoT) structure
- Ensures reasoning completeness score ≥0.8
- Checks for evidence-based reasoning patterns

This validator checks:
1. Presence of reasoning structure (step-by-step thinking)
2. Evidence of differential diagnosis consideration
3. Clinical accuracy markers
4. Logical flow and completeness
"""

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class ReasoningAssessment:
    """Assessment result for reasoning completeness."""
    conversation_id: str
    completeness_score: float
    has_structured_reasoning: bool
    has_differential_diagnosis: bool
    has_evidence_based_markers: bool
    has_logical_flow: bool
    clinical_accuracy_score: float
    violations: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


class ReasoningCompletenessValidator:
    """
    Validates reasoning completeness for Stage 2 training data.

    Checks for:
    - Structured reasoning (step-by-step thinking)
    - Differential diagnosis consideration
    - Evidence-based reasoning markers
    - Logical flow and completeness
    - Clinical accuracy patterns
    """

    # Markers of structured reasoning
    REASONING_MARKERS = [
        r"\bfirst\b", r"\bsecond\b", r"\bthird\b",
        r"\bstep\b", r"\bphase\b", r"\bstage\b",
        r"\bapproach\b", r"\bmethod\b", r"\bprocess\b",
        r"\banalyze\b", r"\bassess\b", r"\bevaluate\b",
        r"\bconsider\b", r"\bexamine\b", r"\breview\b",
    ]

    # Differential diagnosis markers
    DIFFERENTIAL_MARKERS = [
        r"\bdifferential\b", r"\balternative\b", r"\bconsider\b",
        r"\brule out\b", r"\bexclude\b", r"\bpossibility\b",
        r"\bpotential\b", r"\bmay be\b", r"\bcould be\b",
        r"\bmight be\b", r"\bsuggests\b", r"\bindicates\b",
    ]

    # Evidence-based reasoning markers
    EVIDENCE_MARKERS = [
        r"\bresearch\b", r"\bstudy\b", r"\bevidence\b",
        r"\bdata\b", r"\bliterature\b", r"\bclinical\b",
        r"\bempirical\b", r"\bvalidated\b", r"\bproven\b",
        r"\bDSM\b", r"\bICD\b", r"\bguideline\b",
        r"\bprotocol\b", r"\bstandard\b", r"\bbest practice\b",
    ]

    # Clinical accuracy markers
    CLINICAL_MARKERS = [
        r"\bassessment\b", r"\bdiagnosis\b", r"\btreatment\b",
        r"\bintervention\b", r"\bsymptom\b", r"\bdisorder\b",
        r"\bcondition\b", r"\bpatient\b", r"\bclient\b",
        r"\btherapeutic\b", r"\bclinical\b", r"\bmental health\b",
    ]

    def __init__(self, completeness_threshold: float = 0.8):
        self.completeness_threshold = completeness_threshold

    def validate_conversation(self, conversation: dict) -> ReasoningAssessment:
        """
        Validate a single conversation for reasoning completeness.

        Handles multiple formats:
        1. Standard format: conversation with "messages" array
        2. CoT format: dict with "answer" field containing reasoning

        Args:
            conversation: Conversation dict with messages/answer and metadata

        Returns:
            ReasoningAssessment with completeness scores
        """
        conversation_id = conversation.get("conversation_id", conversation.get("id", "unknown"))

        # Handle CoT format (answer field)
        if "answer" in conversation:
            all_text = conversation.get("answer", "").lower()
        # Handle standard conversation format (messages array)
        elif "messages" in conversation:
            messages = conversation.get("messages", [])
            all_text = ""
            for msg in messages:
                if isinstance(msg, dict):
                    content = msg.get("content", "")
                    if isinstance(content, str):
                        all_text += " " + content.lower()
                elif isinstance(msg, str):
                    all_text += " " + msg.lower()
        else:
            # Unknown format
            all_text = ""

        # Check for reasoning structure
        has_structured_reasoning = self._check_markers(all_text, self.REASONING_MARKERS, min_count=2)

        # Check for differential diagnosis
        has_differential = self._check_markers(all_text, self.DIFFERENTIAL_MARKERS, min_count=1)

        # Check for evidence-based reasoning
        has_evidence = self._check_markers(all_text, self.EVIDENCE_MARKERS, min_count=1)

        # Check for clinical accuracy markers
        has_clinical = self._check_markers(all_text, self.CLINICAL_MARKERS, min_count=2)

        # Calculate completeness score
        completeness_score = self._calculate_completeness_score(
            has_structured_reasoning,
            has_differential,
            has_evidence,
            has_clinical,
            all_text
        )

        # Check logical flow (basic heuristic)
        has_logical_flow = self._check_logical_flow(all_text)

        # Clinical accuracy score
        clinical_accuracy_score = self._calculate_clinical_accuracy_score(all_text)

        # Build violations list
        violations = []
        if not has_structured_reasoning:
            violations.append("Missing structured reasoning markers")
        if not has_differential:
            violations.append("No differential diagnosis consideration")
        if not has_evidence:
            violations.append("No evidence-based reasoning markers")
        if not has_logical_flow:
            violations.append("Logical flow issues detected")
        if completeness_score < self.completeness_threshold:
            violations.append(f"Completeness score {completeness_score:.2f} below threshold {self.completeness_threshold}")

        return ReasoningAssessment(
            conversation_id=conversation_id,
            completeness_score=completeness_score,
            has_structured_reasoning=has_structured_reasoning,
            has_differential_diagnosis=has_differential,
            has_evidence_based_markers=has_evidence,
            has_logical_flow=has_logical_flow,
            clinical_accuracy_score=clinical_accuracy_score,
            violations=violations,
            metadata={
                "text_length": len(all_text),
                "reasoning_threshold": self.completeness_threshold,
            }
        )

    def _check_markers(self, text: str, markers: list[str], min_count: int = 1) -> bool:
        """Check if text contains minimum number of marker matches."""
        if not text:
            return False

        matches = 0
        for pattern in markers:
            if re.search(pattern, text, re.IGNORECASE):
                matches += 1
                if matches >= min_count:
                    return True
        return matches >= min_count

    def _calculate_completeness_score(
        self,
        has_structured: bool,
        has_differential: bool,
        has_evidence: bool,
        has_clinical: bool,
        text: str
    ) -> float:
        """
        Calculate overall reasoning completeness score.

        Scoring:
        - Structured reasoning: 30%
        - Differential diagnosis: 25%
        - Evidence-based: 25%
        - Clinical markers: 20%
        """
        score = 0.0

        # Component scores
        if has_structured:
            score += 0.30
        if has_differential:
            score += 0.25
        if has_evidence:
            score += 0.25
        if has_clinical:
            score += 0.20

        # Bonus for length (more detailed reasoning)
        word_count = len(text.split())
        if word_count > 200:
            score = min(1.0, score + 0.05)
        elif word_count > 100:
            score = min(1.0, score + 0.03)

        return round(score, 3)

    def _check_logical_flow(self, text: str) -> bool:
        """
        Check for basic logical flow indicators.

        Looks for:
        - Ordered reasoning (first, second, etc.)
        - Cause-effect language
        - Conditional reasoning
        """
        if not text:
            return False

        flow_patterns = [
            r"\bif.*then\b",
            r"\bbecause\b", r"\btherefore\b",
            r"\bthus\b", r"\bhence\b",
            r"\bconsequently\b", r"\bas a result\b",
            r"\bleading to\b", r"\bresults in\b",
        ]

        matches = 0
        for pattern in flow_patterns:
            if re.search(pattern, text, re.IGNORECASE):
                matches += 1

        # At least 2 flow indicators or very long text
        return matches >= 2 or len(text.split()) > 300

    def _calculate_clinical_accuracy_score(self, text: str) -> float:
        """
        Calculate clinical accuracy score based on marker density.

        Returns score 0-1 based on presence of clinical terminology
        and evidence-based language.
        """
        if not text:
            return 0.0

        clinical_count = 0
        for pattern in self.CLINICAL_MARKERS:
            if re.search(pattern, text, re.IGNORECASE):
                clinical_count += 1

        # Normalize to 0-1 scale (expecting ~10 clinical markers in good data)
        score = min(1.0, clinical_count / 10.0)

        return round(score, 3)

    def batch_validate(self, conversations: list[dict]) -> list[ReasoningAssessment]:
        """
        Validate multiple conversations.

        Returns list of ReasoningAssessment objects.
        """
        results = []
        for conv in conversations:
            results.append(self.validate_conversation(conv))
        return results


def validate_cot_dataset(file_path: Path, output_path: Path | None = None) -> dict:
    """
    Validate a CoT reasoning dataset file.

    Args:
        file_path: Path to JSON or JSONL file
        output_path: Optional path to write results

    Returns:
        Dictionary with validation summary
    """
    validator = ReasoningCompletenessValidator()

    # Load dataset
    conversations = []
    if file_path.suffix == '.jsonl':
        with open(file_path) as f:
            for line in f:
                try:
                    data = json.loads(line)
                    conversations.append(data)
                except json.JSONDecodeError:
                    continue
    elif file_path.suffix == '.json':
        with open(file_path) as f:
            data = json.load(f)
            if isinstance(data, list):
                conversations = data
            elif isinstance(data, dict) and 'conversations' in data:
                conversations = data['conversations']

    print(f"Loaded {len(conversations)} conversations from {file_path.name}")

    # Validate
    results = validator.batch_validate(conversations)

    # Summary
    passed = sum(1 for r in results if r.completeness_score >= validator.completeness_threshold)
    avg_score = sum(r.completeness_score for r in results) / len(results) if results else 0

    summary = {
        "file": str(file_path),
        "total_conversations": len(results),
        "passed": passed,
        "failed": len(results) - passed,
        "pass_rate": passed / len(results) if results else 0,
        "average_completeness_score": round(avg_score, 3),
        "threshold": validator.completeness_threshold,
    }

    print(f"\nValidation Summary for {file_path.name}:")
    print(f"  Total: {len(results)}")
    print(f"  Passed: {passed} ({summary['pass_rate']*100:.1f}%)")
    print(f"  Failed: {len(results) - passed}")
    print(f"  Average Score: {avg_score:.3f}")
    print(f"  Threshold: {validator.completeness_threshold}")

    # Write detailed results if output path specified
    if output_path:
        with open(output_path, 'w') as f:
            json.dump(summary, f, indent=2)
        print(f"\nResults written to: {output_path}")

    return summary


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python reasoning_completeness_validator.py <cot_file.jsonl> [output.json]")
        sys.exit(1)

    input_file = Path(sys.argv[1])
    output_file = Path(sys.argv[2]) if len(sys.argv) > 2 else None  # type: ignore

    if not input_file.exists():
        print(f"ERROR: File not found: {input_file}")
        sys.exit(1)

    validate_cot_dataset(input_file, output_file)
