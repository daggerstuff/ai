"""
Personality Dataset Validation

Specialized validation for personality balancing datasets with trait analysis,
bias detection, and balanced representation verification.
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
class PersonalityValidationResult(ValidationResult):
    """Extended validation result for personality datasets."""

    trait_coverage_score: float = 0.0
    bias_detection_score: float = 0.0
    balance_score: float = 0.0
    personality_traits_found: dict[str, int] = None
    bias_indicators: list[str] = None
    demographic_balance: dict[str, float] = None


class PersonalityDatasetValidator(DatasetValidator):
    """Specialized validator for personality balancing datasets."""

    def __init__(self):
        super().__init__()
        self.logger = get_logger(__name__)

        # Personality validation rules
        self.personality_rules = {
            "min_trait_coverage": 0.8,  # 80% of major traits should be represented
            "max_bias_score": 0.3,  # Low bias threshold
            "min_balance_score": 0.7,  # Good balance across traits
            "required_traits": [
                "openness",
                "conscientiousness",
                "extraversion",
                "agreeableness",
                "neuroticism",
            ],
            "min_samples_per_trait": 10,
        }

        # Big Five personality trait indicators
        self.personality_indicators = {
            "openness": [
                r"\b(?:creative|imaginative|curious|artistic|innovative)\b",
                r"\b(?:open.minded|adventurous|intellectual|original)\b",
                r"\b(?:explore|discover|experiment|invent|create)\b",
            ],
            "conscientiousness": [
                r"\b(?:organized|disciplined|responsible|reliable|careful)\b",
                r"\b(?:planned|systematic|thorough|persistent|diligent)\b",
                r"\b(?:goal.oriented|focused|methodical|punctual)\b",
            ],
            "extraversion": [
                r"\b(?:outgoing|social|talkative|energetic|assertive)\b",
                r"\b(?:friendly|enthusiastic|active|gregarious)\b",
                r"\b(?:party|crowd|people|social|interact)\b",
            ],
            "agreeableness": [
                r"\b(?:kind|helpful|trusting|cooperative|sympathetic)\b",
                r"\b(?:compassionate|generous|forgiving|considerate)\b",
                r"\b(?:team|collaborate|support|care|empathy)\b",
            ],
            "neuroticism": [
                r"\b(?:anxious|worried|stressed|nervous|emotional)\b",
                r"\b(?:moody|sensitive|insecure|unstable|reactive)\b",
                r"\b(?:fear|anxiety|stress|worry|tension)\b",
            ],
        }

        # Bias detection patterns
        self.bias_patterns = {
            "gender_bias": [
                r"\b(?:men are|women are|males are|females are)\b",
                r"\b(?:boys should|girls should|guys always|ladies always)\b",
            ],
            "age_bias": [
                r"\b(?:young people are|old people are|millennials are|boomers are)\b",
                r"\b(?:teenagers always|elderly always|kids these days)\b",
            ],
            "cultural_bias": [
                r"\b(?:americans are|europeans are|asians are|africans are)\b",
                r"\b(?:people from.*are always|.*culture is)\b",
            ],
            "stereotype_indicators": [
                r"\b(?:all.*are|every.*is|typical.*behavior|naturally.*at)\b",
                r"\b(?:born.*way|genetic.*trait|inherently.*good)\b",
            ],
        }

        # Demographic indicators for balance analysis
        self.demographic_indicators = {
            "age_groups": [r"\b(?:young|middle.aged|elderly|teen|adult|senior)\b"],
            "gender_references": [r"\b(?:he|she|him|her|male|female|man|woman)\b"],
            "cultural_references": [
                r"\b(?:american|european|asian|african|hispanic|latino)\b"
            ],
        }

        logger.info("PersonalityDatasetValidator initialized")

    def validate_personality_dataset(
        self, dataset_path: str
    ) -> PersonalityValidationResult:
        """Validate personality dataset with specialized checks."""
        logger.info(f"Validating personality dataset: {dataset_path}")

        # Run base validation
        base_result = self.validate_dataset(dataset_path, "personality")

        # Perform personality specific validation
        trait_coverage = self._calculate_trait_coverage_score(dataset_path)
        bias_score = self._calculate_bias_detection_score(dataset_path)
        balance_score = self._calculate_balance_score(dataset_path)
        traits_found = self._analyze_personality_traits(dataset_path)
        bias_indicators = self._detect_bias_indicators(dataset_path)
        demographic_balance = self._analyze_demographic_balance(dataset_path)

        # Create extended result
        result = PersonalityValidationResult(
            dataset_name=base_result.dataset_name,
            is_valid=base_result.is_valid
            and self._meets_personality_requirements(
                trait_coverage, bias_score, balance_score, traits_found
            ),
            file_count=base_result.file_count,
            total_size=base_result.total_size,
            format_compliance=base_result.format_compliance,
            integrity_check=base_result.integrity_check,
            quality_score=base_result.quality_score * trait_coverage * (1 - bias_score),
            issues=base_result.issues,
            validation_timestamp=base_result.validation_timestamp,
            trait_coverage_score=trait_coverage,
            bias_detection_score=bias_score,
            balance_score=balance_score,
            personality_traits_found=traits_found,
            bias_indicators=bias_indicators,
            demographic_balance=demographic_balance,
        )

        # Add personality specific issues
        if trait_coverage < self.personality_rules["min_trait_coverage"]:
            result.issues.append(f"Trait coverage too low: {trait_coverage:.2f}")
        if bias_score > self.personality_rules["max_bias_score"]:
            result.issues.append(f"Bias score too high: {bias_score:.2f}")
        if balance_score < self.personality_rules["min_balance_score"]:
            result.issues.append(f"Balance score too low: {balance_score:.2f}")
        if bias_indicators:
            result.issues.extend(
                [f"Bias detected: {indicator}" for indicator in bias_indicators[:5]]
            )

        logger.info(
            f"Personality validation completed: {'VALID' if result.is_valid else 'INVALID'}"
        )
        return result

    def _calculate_trait_coverage_score(self, dataset_path: str) -> float:
        """Calculate personality trait coverage score."""
        try:
            trait_counts = dict.fromkeys(self.personality_rules["required_traits"], 0)
            total_conversations = 0

            for file_path in Path(dataset_path).rglob("*.json"):
                if file_path.is_file():
                    try:
                        with open(file_path, encoding="utf-8") as f:
                            data = json.load(f)

                        conversations = self._extract_conversations(data)
                        for conversation in conversations:
                            total_conversations += 1
                            text = self._extract_text_from_conversation(conversation)

                            # Check for each personality trait
                            for trait, patterns in self.personality_indicators.items():
                                if any(
                                    re.search(pattern, text, re.IGNORECASE)
                                    for pattern in patterns
                                ):
                                    trait_counts[trait] += 1

                    except Exception:
                        continue

            # Calculate coverage score
            covered_traits = sum(
                1
                for count in trait_counts.values()
                if count >= self.personality_rules["min_samples_per_trait"]
            )
            return covered_traits / len(
                self.personality_rules["required_traits"]
            )


        except Exception as e:
            logger.error(f"Trait coverage scoring failed: {e}")
            return 0.0

    def _calculate_bias_detection_score(self, dataset_path: str) -> float:
        """Calculate bias detection score (lower is better)."""
        try:
            bias_indicators = 0
            total_content_blocks = 0

            for file_path in Path(dataset_path).rglob("*.json"):
                if file_path.is_file():
                    try:
                        with open(file_path, encoding="utf-8") as f:
                            data = json.load(f)

                        conversations = self._extract_conversations(data)
                        for conversation in conversations:
                            total_content_blocks += 1
                            text = self._extract_text_from_conversation(conversation)

                            # Check for bias patterns
                            for _bias_type, patterns in self.bias_patterns.items():
                                for pattern in patterns:
                                    if re.search(pattern, text, re.IGNORECASE):
                                        bias_indicators += 1
                                        break  # Count once per content block per bias type

                    except Exception:
                        continue

            # Return bias ratio (0 = no bias, 1 = high bias)
            return bias_indicators / max(total_content_blocks, 1)

        except Exception as e:
            logger.error(f"Bias detection scoring failed: {e}")
            return 1.0  # Assume high bias on error

    def _calculate_balance_score(self, dataset_path: str) -> float:
        """Calculate balance score across personality traits."""
        try:
            trait_counts = dict.fromkeys(self.personality_indicators.keys(), 0)

            for file_path in Path(dataset_path).rglob("*.json"):
                if file_path.is_file():
                    try:
                        with open(file_path, encoding="utf-8") as f:
                            data = json.load(f)

                        conversations = self._extract_conversations(data)
                        for conversation in conversations:
                            text = self._extract_text_from_conversation(conversation)

                            # Count trait occurrences
                            for trait, patterns in self.personality_indicators.items():
                                if any(
                                    re.search(pattern, text, re.IGNORECASE)
                                    for pattern in patterns
                                ):
                                    trait_counts[trait] += 1

                    except Exception:
                        continue

            # Calculate balance using coefficient of variation (lower is more balanced)
            counts = list(trait_counts.values())
            if not counts or sum(counts) == 0:
                return 0.0

            mean_count = sum(counts) / len(counts)
            variance = sum((count - mean_count) ** 2 for count in counts) / len(counts)
            std_dev = variance**0.5

            # Convert coefficient of variation to balance score (1 = perfect balance, 0 = poor balance)
            cv = std_dev / mean_count if mean_count > 0 else 1.0
            return max(0.0, 1.0 - cv)


        except Exception as e:
            logger.error(f"Balance scoring failed: {e}")
            return 0.0

    def _analyze_personality_traits(self, dataset_path: str) -> dict[str, int]:
        """Analyze personality trait distribution."""
        trait_counts = dict.fromkeys(self.personality_indicators.keys(), 0)

        try:
            for file_path in Path(dataset_path).rglob("*.json"):
                if file_path.is_file():
                    try:
                        with open(file_path, encoding="utf-8") as f:
                            data = json.load(f)

                        conversations = self._extract_conversations(data)
                        for conversation in conversations:
                            text = self._extract_text_from_conversation(conversation)

                            # Count each trait
                            for trait, patterns in self.personality_indicators.items():
                                trait_matches = sum(
                                    len(re.findall(pattern, text, re.IGNORECASE))
                                    for pattern in patterns
                                )
                                trait_counts[trait] += trait_matches

                    except Exception:
                        continue

        except Exception as e:
            logger.error(f"Personality trait analysis failed: {e}")

        return trait_counts

    def _detect_bias_indicators(self, dataset_path: str) -> list[str]:
        """Detect specific bias indicators."""
        bias_found = []

        try:
            for file_path in Path(dataset_path).rglob("*.json"):
                if file_path.is_file():
                    try:
                        with open(file_path, encoding="utf-8") as f:
                            data = json.load(f)

                        conversations = self._extract_conversations(data)
                        for conversation in conversations:
                            text = self._extract_text_from_conversation(conversation)

                            # Check for specific bias types
                            for bias_type, patterns in self.bias_patterns.items():
                                for pattern in patterns:
                                    matches = re.findall(pattern, text, re.IGNORECASE)
                                    if matches:
                                        bias_found.append(
                                            f"{bias_type}: {matches[0][:50]}..."
                                        )

                    except Exception:
                        continue

        except Exception as e:
            logger.error(f"Bias indicator detection failed: {e}")

        return bias_found[:10]  # Limit to first 10 indicators

    def _analyze_demographic_balance(self, dataset_path: str) -> dict[str, float]:
        """Analyze demographic balance in the dataset."""
        demographic_counts = {}
        total_references = 0

        try:
            for file_path in Path(dataset_path).rglob("*.json"):
                if file_path.is_file():
                    try:
                        with open(file_path, encoding="utf-8") as f:
                            data = json.load(f)

                        conversations = self._extract_conversations(data)
                        for conversation in conversations:
                            text = self._extract_text_from_conversation(conversation)

                            # Count demographic references
                            for (
                                demo_type,
                                patterns,
                            ) in self.demographic_indicators.items():
                                for pattern in patterns:
                                    matches = re.findall(pattern, text, re.IGNORECASE)
                                    if matches:
                                        if demo_type not in demographic_counts:
                                            demographic_counts[demo_type] = 0
                                        demographic_counts[demo_type] += len(matches)
                                        total_references += len(matches)

                    except Exception:
                        continue

        except Exception as e:
            logger.error(f"Demographic balance analysis failed: {e}")

        # Convert to percentages
        demographic_balance = {}
        for demo_type, count in demographic_counts.items():
            demographic_balance[demo_type] = count / max(total_references, 1)

        return demographic_balance

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

    def _meets_personality_requirements(
        self,
        trait_coverage: float,
        bias_score: float,
        balance_score: float,
        traits_found: dict[str, int],
    ) -> bool:
        """Check if dataset meets personality requirements."""
        return (
            trait_coverage >= self.personality_rules["min_trait_coverage"]
            and bias_score <= self.personality_rules["max_bias_score"]
            and balance_score >= self.personality_rules["min_balance_score"]
            and all(
                count >= self.personality_rules["min_samples_per_trait"]
                for trait, count in traits_found.items()
                if trait in self.personality_rules["required_traits"]
            )
        )

    def generate_personality_report(
        self,
        results: list[PersonalityValidationResult],
        output_path: str = "personality_validation_report.json",
    ) -> str:
        """Generate specialized personality validation report."""
        report = {
            "report_type": "Personality Dataset Validation",
            "report_timestamp": datetime.now().isoformat(),
            "summary": {
                "total_datasets": len(results),
                "valid_datasets": sum(1 for r in results if r.is_valid),
                "average_trait_coverage": (
                    sum(r.trait_coverage_score for r in results) / len(results)
                    if results
                    else 0
                ),
                "average_bias_score": (
                    sum(r.bias_detection_score for r in results) / len(results)
                    if results
                    else 0
                ),
                "average_balance_score": (
                    sum(r.balance_score for r in results) / len(results)
                    if results
                    else 0
                ),
                "total_bias_indicators": sum(
                    len(r.bias_indicators) for r in results if r.bias_indicators
                ),
            },
            "detailed_results": [
                {
                    "dataset_name": r.dataset_name,
                    "is_valid": r.is_valid,
                    "trait_coverage_score": r.trait_coverage_score,
                    "bias_detection_score": r.bias_detection_score,
                    "balance_score": r.balance_score,
                    "personality_traits_found": r.personality_traits_found,
                    "bias_indicators": r.bias_indicators,
                    "demographic_balance": r.demographic_balance,
                    "issues": r.issues,
                }
                for r in results
            ],
        }

        with open(output_path, "w") as f:
            json.dump(report, f, indent=2)

        logger.info(f"Personality validation report generated: {output_path}")
        return output_path


# Example usage
if __name__ == "__main__":
    validator = PersonalityDatasetValidator()

    # Test validation
    result = validator.validate_personality_dataset("./test_personality_dataset")
