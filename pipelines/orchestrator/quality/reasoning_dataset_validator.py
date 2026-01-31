"""
Reasoning Dataset Validation Scripts

Specialized validation for reasoning enhancement datasets with chain-of-thought
pattern detection, logical structure verification, and problem-solution analysis.
"""

import json
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from dataset_validator import DatasetValidator, ValidationResult
from logger import get_logger

logger = get_logger(__name__)


@dataclass
class ReasoningValidationResult(ValidationResult):
    """Extended validation result for reasoning datasets."""

    chain_of_thought_score: float = 0.0
    logical_structure_score: float = 0.0
    problem_solution_pairs: int = 0
    reasoning_patterns_found: list[str] = None
    complexity_distribution: dict[str, int] = None


class ReasoningDatasetValidator(DatasetValidator):
    """Specialized validator for reasoning enhancement datasets."""

    def __init__(self):
        super().__init__()
        self.logger = get_logger(__name__)

        # Reasoning specific validation rules
        self.reasoning_rules = {
            "min_chain_of_thought_score": 0.7,
            "min_logical_structure_score": 0.6,
            "min_problem_solution_pairs": 10,
            "required_reasoning_patterns": [
                "step_by_step",
                "cause_effect",
                "comparison",
            ],
            "complexity_levels": ["basic", "intermediate", "advanced"],
        }

        # Reasoning pattern detection
        self.reasoning_patterns = {
            "chain_of_thought": [
                r"\b(?:first|second|third|next|then|finally|therefore|thus|hence)\b",
                r"\b(?:step \d+|stage \d+|phase \d+)\b",
                r"\b(?:because|since|due to|as a result|consequently)\b",
            ],
            "logical_connectors": [
                r"\b(?:if|then|else|when|while|unless|although|however)\b",
                r"\b(?:and|or|but|yet|so|for|nor)\b",
                r"\b(?:moreover|furthermore|additionally|nevertheless)\b",
            ],
            "problem_indicators": [
                r"\b(?:problem|question|challenge|issue|puzzle)\b",
                r"\b(?:solve|find|determine|calculate|prove)\b",
                r"\b(?:what|how|why|when|where|which)\b",
            ],
            "solution_indicators": [
                r"\b(?:solution|answer|result|conclusion|outcome)\b",
                r"\b(?:therefore|thus|hence|so|consequently)\b",
                r"\b(?:the answer is|the result is|we conclude)\b",
            ],
            "reasoning_types": {
                "deductive": [r"\b(?:all|every|if.*then|therefore|thus)\b"],
                "inductive": [r"\b(?:some|many|often|usually|probably)\b"],
                "abductive": [r"\b(?:best explanation|most likely|probably because)\b"],
                "analogical": [r"\b(?:similar to|like|analogous|comparable)\b"],
            },
        }

        logger.info("ReasoningDatasetValidator initialized")

    def validate_reasoning_dataset(
        self, dataset_path: str
    ) -> ReasoningValidationResult:
        """Validate reasoning dataset with specialized checks."""
        logger.info(f"Validating reasoning dataset: {dataset_path}")

        # Run base validation
        base_result = self.validate_dataset(dataset_path, "reasoning")

        # Perform reasoning specific validation
        cot_score = self._calculate_chain_of_thought_score(dataset_path)
        logical_score = self._calculate_logical_structure_score(dataset_path)
        problem_solution_pairs = self._count_problem_solution_pairs(dataset_path)
        reasoning_patterns = self._detect_reasoning_patterns(dataset_path)
        complexity_dist = self._analyze_complexity_distribution(dataset_path)

        # Create extended result
        result = ReasoningValidationResult(
            dataset_name=base_result.dataset_name,
            is_valid=base_result.is_valid
            and self._meets_reasoning_requirements(
                cot_score, logical_score, problem_solution_pairs, reasoning_patterns
            ),
            file_count=base_result.file_count,
            total_size=base_result.total_size,
            format_compliance=base_result.format_compliance,
            integrity_check=base_result.integrity_check,
            quality_score=base_result.quality_score * ((cot_score + logical_score) / 2),
            issues=base_result.issues,
            validation_timestamp=base_result.validation_timestamp,
            chain_of_thought_score=cot_score,
            logical_structure_score=logical_score,
            problem_solution_pairs=problem_solution_pairs,
            reasoning_patterns_found=reasoning_patterns,
            complexity_distribution=complexity_dist,
        )

        # Add reasoning specific issues
        if cot_score < self.reasoning_rules["min_chain_of_thought_score"]:
            result.issues.append(f"Chain-of-thought score too low: {cot_score:.2f}")
        if logical_score < self.reasoning_rules["min_logical_structure_score"]:
            result.issues.append(
                f"Logical structure score too low: {logical_score:.2f}"
            )
        if problem_solution_pairs < self.reasoning_rules["min_problem_solution_pairs"]:
            result.issues.append(
                f"Insufficient problem-solution pairs: {problem_solution_pairs}"
            )

        logger.info(
            f"Reasoning validation completed: {'VALID' if result.is_valid else 'INVALID'}"
        )
        return result

    def _calculate_chain_of_thought_score(self, dataset_path: str) -> float:
        """Calculate chain-of-thought reasoning score."""
        try:
            cot_indicators = 0
            total_conversations = 0

            for file_path in Path(dataset_path).rglob("*.json"):
                if file_path.is_file():
                    try:
                        with open(file_path, encoding="utf-8") as f:
                            data = json.load(f)

                        conversations = self._extract_conversations(data)
                        for conversation in conversations:
                            total_conversations += 1

                            # Check for chain-of-thought patterns
                            text = self._extract_text_from_conversation(conversation)
                            if self._has_chain_of_thought_pattern(text):
                                cot_indicators += 1

                    except Exception:
                        continue

            return cot_indicators / max(total_conversations, 1)

        except Exception as e:
            logger.error(f"Chain-of-thought scoring failed: {e}")
            return 0.0

    def _calculate_logical_structure_score(self, dataset_path: str) -> float:
        """Calculate logical structure quality score."""
        try:
            logical_structures = 0
            total_conversations = 0

            for file_path in Path(dataset_path).rglob("*.json"):
                if file_path.is_file():
                    try:
                        with open(file_path, encoding="utf-8") as f:
                            data = json.load(f)

                        conversations = self._extract_conversations(data)
                        for conversation in conversations:
                            total_conversations += 1

                            # Check for logical structure
                            if self._has_logical_structure(conversation):
                                logical_structures += 1

                    except Exception:
                        continue

            return logical_structures / max(total_conversations, 1)

        except Exception as e:
            logger.error(f"Logical structure scoring failed: {e}")
            return 0.0

    def _count_problem_solution_pairs(self, dataset_path: str) -> int:
        """Count problem-solution pairs in the dataset."""
        try:
            pairs_count = 0

            for file_path in Path(dataset_path).rglob("*.json"):
                if file_path.is_file():
                    try:
                        with open(file_path, encoding="utf-8") as f:
                            data = json.load(f)

                        conversations = self._extract_conversations(data)
                        for conversation in conversations:
                            if self._is_problem_solution_pair(conversation):
                                pairs_count += 1

                    except Exception:
                        continue

            return pairs_count

        except Exception as e:
            logger.error(f"Problem-solution pair counting failed: {e}")
            return 0

    def _detect_reasoning_patterns(self, dataset_path: str) -> list[str]:
        """Detect various reasoning patterns in the dataset."""
        patterns_found = set()

        try:
            for file_path in Path(dataset_path).rglob("*.json"):
                if file_path.is_file():
                    try:
                        with open(file_path, encoding="utf-8") as f:
                            data = json.load(f)

                        conversations = self._extract_conversations(data)
                        for conversation in conversations:
                            text = self._extract_text_from_conversation(conversation)

                            # Check for different reasoning types
                            for reasoning_type, patterns in self.reasoning_patterns[
                                "reasoning_types"
                            ].items():
                                for pattern in patterns:
                                    if re.search(pattern, text, re.IGNORECASE):
                                        patterns_found.add(reasoning_type)

                    except Exception:
                        continue

        except Exception as e:
            logger.error(f"Reasoning pattern detection failed: {e}")

        return list(patterns_found)

    def _analyze_complexity_distribution(self, dataset_path: str) -> dict[str, int]:
        """Analyze complexity distribution of reasoning tasks."""
        complexity_counts = dict.fromkeys(self.reasoning_rules["complexity_levels"], 0)

        try:
            for file_path in Path(dataset_path).rglob("*.json"):
                if file_path.is_file():
                    try:
                        with open(file_path, encoding="utf-8") as f:
                            data = json.load(f)

                        conversations = self._extract_conversations(data)
                        for conversation in conversations:
                            complexity = self._assess_complexity(conversation)
                            if complexity in complexity_counts:
                                complexity_counts[complexity] += 1

                    except Exception:
                        continue

        except Exception as e:
            logger.error(f"Complexity analysis failed: {e}")

        return complexity_counts

    def _has_chain_of_thought_pattern(self, text: str) -> bool:
        """Check if text contains chain-of-thought reasoning patterns."""
        cot_pattern_count = 0

        for pattern in self.reasoning_patterns["chain_of_thought"]:
            if re.search(pattern, text, re.IGNORECASE):
                cot_pattern_count += 1

        # Require at least 2 different CoT patterns
        return cot_pattern_count >= 2

    def _has_logical_structure(self, conversation: dict) -> bool:
        """Check if conversation has logical structure."""
        text = self._extract_text_from_conversation(conversation)

        # Check for logical connectors
        connector_count = 0
        for pattern in self.reasoning_patterns["logical_connectors"]:
            if re.search(pattern, text, re.IGNORECASE):
                connector_count += 1

        # Check for structured reasoning (premises and conclusions)
        has_premises = bool(
            re.search(r"\b(?:given|assume|suppose|premise)\b", text, re.IGNORECASE)
        )
        has_conclusion = bool(
            re.search(r"\b(?:therefore|thus|conclude|result)\b", text, re.IGNORECASE)
        )

        return connector_count >= 2 or (has_premises and has_conclusion)

    def _is_problem_solution_pair(self, conversation: dict) -> bool:
        """Check if conversation represents a problem-solution pair."""
        text = self._extract_text_from_conversation(conversation)

        # Check for problem indicators
        has_problem = any(
            re.search(pattern, text, re.IGNORECASE)
            for pattern in self.reasoning_patterns["problem_indicators"]
        )

        # Check for solution indicators
        has_solution = any(
            re.search(pattern, text, re.IGNORECASE)
            for pattern in self.reasoning_patterns["solution_indicators"]
        )

        return has_problem and has_solution

    def _assess_complexity(self, conversation: dict) -> str:
        """Assess the complexity level of a reasoning task."""
        text = self._extract_text_from_conversation(conversation)

        # Simple heuristics for complexity assessment
        word_count = len(text.split())
        sentence_count = len(re.findall(r"[.!?]+", text))
        avg_sentence_length = word_count / max(sentence_count, 1)

        # Count reasoning indicators
        reasoning_indicators = sum(
            len(re.findall(pattern, text, re.IGNORECASE))
            for pattern_list in self.reasoning_patterns.values()
            if isinstance(pattern_list, list)
            for pattern in pattern_list
        )

        # Complexity scoring
        complexity_score = (
            (word_count / 100) * 0.3
            + (avg_sentence_length / 20) * 0.3
            + (reasoning_indicators / 10) * 0.4
        )

        if complexity_score < 0.5:
            return "basic"
        if complexity_score < 1.0:
            return "intermediate"
        return "advanced"

    def _extract_conversations(self, data) -> list[dict]:
        """Extract conversations from JSON data."""
        conversations = []

        if isinstance(data, list):
            conversations.extend(data)
        elif isinstance(data, dict):
            if "conversations" in data:
                conversations.extend(data["conversations"])
            elif "messages" in data:
                conversations.append(data)
            else:
                conversations.append(data)

        return conversations

    def _extract_text_from_conversation(self, conversation: dict) -> str:
        """Extract text content from a conversation."""
        text_parts = []

        if "messages" in conversation:
            for message in conversation["messages"]:
                if isinstance(message, dict) and "content" in message:
                    text_parts.append(message["content"])
        elif "content" in conversation:
            text_parts.append(conversation["content"])
        elif "text" in conversation:
            text_parts.append(conversation["text"])

        return " ".join(text_parts)

    def _meets_reasoning_requirements(
        self,
        cot_score: float,
        logical_score: float,
        pairs_count: int,
        patterns: list[str],
    ) -> bool:
        """Check if dataset meets reasoning requirements."""
        return (
            cot_score >= self.reasoning_rules["min_chain_of_thought_score"]
            and logical_score >= self.reasoning_rules["min_logical_structure_score"]
            and pairs_count >= self.reasoning_rules["min_problem_solution_pairs"]
            and len(
                set(patterns) & set(self.reasoning_rules["required_reasoning_patterns"])
            )
            >= 2
        )

    def generate_reasoning_report(
        self,
        results: list[ReasoningValidationResult],
        output_path: str = "reasoning_validation_report.json",
    ) -> str:
        """Generate specialized reasoning validation report."""
        report = {
            "report_type": "Reasoning Dataset Validation",
            "report_timestamp": datetime.now().isoformat(),
            "summary": {
                "total_datasets": len(results),
                "valid_datasets": sum(1 for r in results if r.is_valid),
                "average_cot_score": (
                    sum(r.chain_of_thought_score for r in results) / len(results)
                    if results
                    else 0
                ),
                "average_logical_score": (
                    sum(r.logical_structure_score for r in results) / len(results)
                    if results
                    else 0
                ),
                "total_problem_solution_pairs": sum(
                    r.problem_solution_pairs for r in results
                ),
                "reasoning_patterns_coverage": list(
                    set().union(
                        *[
                            r.reasoning_patterns_found
                            for r in results
                            if r.reasoning_patterns_found
                        ]
                    )
                ),
            },
            "detailed_results": [
                {
                    "dataset_name": r.dataset_name,
                    "is_valid": r.is_valid,
                    "chain_of_thought_score": r.chain_of_thought_score,
                    "logical_structure_score": r.logical_structure_score,
                    "problem_solution_pairs": r.problem_solution_pairs,
                    "reasoning_patterns_found": r.reasoning_patterns_found,
                    "complexity_distribution": r.complexity_distribution,
                    "issues": r.issues,
                }
                for r in results
            ],
        }

        with open(output_path, "w") as f:
            json.dump(report, f, indent=2)

        logger.info(f"Reasoning validation report generated: {output_path}")
        return output_path


# Example usage
if __name__ == "__main__":
    validator = ReasoningDatasetValidator()

    # Test validation
    result = validator.validate_reasoning_dataset("./test_reasoning_dataset")
