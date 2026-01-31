#!/usr/bin/env python3
"""
Task 5.13-5.24: Reasoning Enhancement Dataset Integration
Processes chain-of-thought reasoning datasets for clinical diagnosis,
neurodiversity awareness, and emotional intelligence training.
"""

import json
import os
import sys
from datetime import datetime
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from ai.pipelines.orchestrator.data_loader import load_hf_dataset
from ai.pipelines.orchestrator.logger import get_logger
from ai.pipelines.orchestrator.standardizer import from_input_output_pair, from_simple_message_list

logger = get_logger("task_5_reasoning_enhancement")


class ReasoningEnhancementProcessor:
    """
    Processor for chain-of-thought reasoning datasets.
    Handles clinical diagnostic reasoning, neurodiversity awareness, and emotional intelligence.
    """

    def __init__(self, output_dir: str, target_conversations: int = 5000):
        self.output_dir = output_dir
        self.target_conversations = target_conversations

        # Ensure output directory exists
        os.makedirs(output_dir, exist_ok=True)

        # Reasoning dataset configurations
        self.dataset_configs = [
            {
                "name": "clinical_diagnosis_cot",
                "hf_path": "moremilk/CoT_Reasoning_Clinical_Diagnosis_Mental_Health",
                "target": 2000,
                "priority": 1,
                "reasoning_type": "clinical_diagnosis"
            },
            {
                "name": "neurodivergent_cot",
                "hf_path": "moremilk/CoT_Neurodivergent_vs_Neurotypical_Interactions",
                "target": 1500,
                "priority": 2,
                "reasoning_type": "neurodiversity_awareness"
            },
            {
                "name": "heartbreak_cot",
                "hf_path": "moremilk/CoT_Heartbreak_and_Breakups",
                "target": 1000,
                "priority": 3,
                "reasoning_type": "emotional_intelligence"
            },
            {
                "name": "mens_mental_health_cot",
                "hf_path": "moremilk/CoT_Reasoning_Mens_Mental_Health",
                "target": 500,
                "priority": 4,
                "reasoning_type": "gender_specific_mental_health"
            }
        ]

        self.stats = {
            "datasets_processed": 0,
            "total_processed": 0,
            "total_accepted": 0,
            "format_errors": 0,
            "quality_filtered": 0,
            "processing_errors": 0
        }

    def process_all_datasets(self) -> dict:
        """Process all reasoning enhancement datasets."""
        logger.info("Starting reasoning enhancement dataset processing")
        start_time = datetime.now()

        all_conversations = []
        dataset_results = {}

        # Process datasets by priority
        sorted_configs = sorted(self.dataset_configs, key=lambda x: x["priority"])

        for config in sorted_configs:
            logger.info(f"Processing reasoning dataset: {config['name']} ({config['hf_path']})")

            try:
                conversations = self._process_single_dataset(config)
                all_conversations.extend(conversations)

                dataset_results[config["name"]] = {
                    "conversations_processed": len(conversations),
                    "target": config["target"],
                    "reasoning_type": config["reasoning_type"],
                    "success": True,
                    "hf_path": config["hf_path"]
                }

                self.stats["datasets_processed"] += 1
                logger.info(f"Successfully processed {len(conversations)} reasoning conversations from {config['name']}")

                # Check if we've reached overall target
                if len(all_conversations) >= self.target_conversations:
                    logger.info(f"Reached overall target of {self.target_conversations} conversations")
                    break

            except Exception as e:
                logger.error(f"Failed to process reasoning dataset {config['name']}: {e}")
                dataset_results[config["name"]] = {
                    "conversations_processed": 0,
                    "target": config["target"],
                    "reasoning_type": config["reasoning_type"],
                    "success": False,
                    "error": str(e),
                    "hf_path": config["hf_path"]
                }
                self.stats["processing_errors"] += 1

        # Save processed conversations
        output_path = os.path.join(self.output_dir, "reasoning_enhancement_conversations.jsonl")
        self._save_conversations(all_conversations, output_path)

        # Generate report
        processing_time = (datetime.now() - start_time).total_seconds()
        report = self._generate_report(dataset_results, all_conversations, processing_time)

        # Save report
        report_path = os.path.join(self.output_dir, "task_5_reasoning_processing_report.json")
        with open(report_path, "w") as f:
            json.dump(report, f, indent=2)

        logger.info(f"Reasoning processing completed. Processed {len(all_conversations)} conversations")
        return report

    def _process_single_dataset(self, config: dict) -> list:
        """Process a single reasoning dataset."""
        try:
            # Load dataset from HuggingFace
            dataset = load_hf_dataset(config["hf_path"])
            if dataset is None:
                raise ValueError(f"Failed to load reasoning dataset: {config['hf_path']}")

            processed_conversations = []

            # Convert to list for processing
            data_items = dataset.to_list() if hasattr(dataset, "to_list") else list(dataset)

            logger.info(f"Processing {len(data_items)} reasoning items from {config['name']}")

            for idx, item in enumerate(data_items):
                try:
                    # Process reasoning conversation
                    processed = self._process_reasoning_conversation(item, config, idx)
                    if processed:
                        processed_conversations.append(processed)
                        self.stats["total_accepted"] += 1

                        # Check if we've reached target for this dataset
                        if len(processed_conversations) >= config["target"]:
                            logger.info(f"Reached target of {config['target']} conversations for {config['name']}")
                            break

                    self.stats["total_processed"] += 1

                except Exception as e:
                    logger.warning(f"Error processing reasoning item {idx} from {config['name']}: {e}")
                    self.stats["processing_errors"] += 1
                    continue

            return processed_conversations

        except Exception as e:
            logger.error(f"Failed to process reasoning dataset {config['name']}: {e}")
            raise

    def _process_reasoning_conversation(self, item: dict, config: dict, idx: int) -> dict:
        """Process a single reasoning conversation with chain-of-thought extraction."""
        try:
            # Detect and convert format
            conversation = self._detect_and_convert_format(item, config["name"])
            if not conversation:
                self.stats["format_errors"] += 1
                return None

            # Basic quality checks
            if len(conversation.messages) < 2:
                self.stats["quality_filtered"] += 1
                return None

            # Check content length and reasoning complexity
            total_content = " ".join([msg.content for msg in conversation.messages])
            if len(total_content) < 50 or len(total_content) > 4000:
                self.stats["quality_filtered"] += 1
                return None

            # Extract reasoning patterns
            reasoning_patterns = self._extract_reasoning_patterns(total_content, config["reasoning_type"])
            cot_steps = self._extract_cot_steps(total_content)

            # Check reasoning complexity
            reasoning_complexity = self._calculate_reasoning_complexity(reasoning_patterns, cot_steps)
            if reasoning_complexity < 0.3:  # Minimum complexity threshold
                self.stats["quality_filtered"] += 1
                return None

            # Create standardized format
            return {
                "conversation": [
                    {"role": msg.role, "content": msg.content}
                    for msg in conversation.messages
                ],
                "metadata": {
                    "category": "reasoning_enhancement",
                    "subcategory": config["reasoning_type"],
                    "reasoning_patterns": reasoning_patterns,
                    "reasoning_complexity": reasoning_complexity,
                    "chain_of_thought_steps": cot_steps,
                    "clinical_relevance": self._assess_clinical_relevance(total_content, config["reasoning_type"]),
                    "source_dataset": config["name"],
                    "hf_path": config["hf_path"],
                    "processing_timestamp": datetime.now().isoformat(),
                    "original_index": idx,
                    "conversation_length": len(conversation.messages),
                    "total_content_length": len(total_content)
                }
            }


        except Exception as e:
            logger.warning(f"Failed to process reasoning conversation at index {idx}: {e}")
            return None

    def _detect_and_convert_format(self, item: dict, source: str):
        """Detect item format and convert to Conversation object."""
        try:
            # Try different format patterns
            if "input" in item and "output" in item:
                return from_input_output_pair(
                    item["input"], item["output"],
                    input_role="client", output_role="therapist", source=source
                )
            if "question" in item and "answer" in item:
                return from_input_output_pair(
                    item["question"], item["answer"],
                    input_role="client", output_role="therapist", source=source
                )
            if "conversations" in item:
                conversations = item["conversations"]
                if isinstance(conversations, list) and len(conversations) > 0:
                    return from_simple_message_list(conversations, source=source)
            elif "messages" in item:
                return from_simple_message_list(item["messages"], source=source)
            elif "text" in item:
                return from_input_output_pair(
                    "", item["text"],
                    input_role="client", output_role="therapist", source=source
                )
            else:
                # Try to extract any substantial text content
                content = str(item)
                if len(content) > 100:
                    return from_input_output_pair(
                        "", content,
                        input_role="client", output_role="therapist", source=source
                    )
                return None

        except Exception as e:
            logger.warning(f"Format detection failed for {source}: {e}")
            return None

    def _extract_reasoning_patterns(self, content: str, reasoning_type: str) -> list:
        """Extract reasoning patterns based on reasoning type."""
        content_lower = content.lower()
        patterns = []

        if reasoning_type == "clinical_diagnosis":
            clinical_patterns = [
                ("symptom_identification", ["symptom", "sign", "indicator", "manifestation", "presentation"]),
                ("differential_diagnosis", ["differential", "diagnosis", "condition", "disorder", "rule out"]),
                ("assessment_planning", ["assessment", "evaluation", "test", "measure", "screening"]),
                ("treatment_recommendation", ["treatment", "therapy", "intervention", "approach", "strategy"]),
                ("risk_assessment", ["risk", "danger", "safety", "harm", "crisis"]),
                ("prognosis_evaluation", ["prognosis", "outcome", "recovery", "improvement", "progress"])
            ]
            for pattern, keywords in clinical_patterns:
                if any(term in content_lower for term in keywords):
                    patterns.append(pattern)

        elif reasoning_type == "neurodiversity_awareness":
            neurodiversity_patterns = [
                ("accommodation_strategies", ["accommodation", "adaptation", "modification", "support"]),
                ("communication_adaptation", ["communication", "language", "expression", "understanding"]),
                ("sensory_considerations", ["sensory", "stimulation", "environment", "sensitivity"]),
                ("executive_function_support", ["executive", "planning", "organization", "attention"]),
                ("social_interaction_guidance", ["social", "interaction", "relationship", "communication"]),
                ("strength_identification", ["strength", "ability", "talent", "skill", "capacity"])
            ]
            for pattern, keywords in neurodiversity_patterns:
                if any(term in content_lower for term in keywords):
                    patterns.append(pattern)

        elif reasoning_type == "emotional_intelligence":
            emotional_patterns = [
                ("emotion_recognition", ["emotion", "feeling", "mood", "affect", "emotional state"]),
                ("empathy_demonstration", ["empathy", "understanding", "compassion", "perspective"]),
                ("emotional_regulation", ["regulation", "control", "management", "coping", "balance"]),
                ("social_awareness", ["social", "awareness", "context", "situation", "environment"]),
                ("relationship_management", ["relationship", "connection", "bond", "interaction"]),
                ("self_awareness", ["self-awareness", "insight", "reflection", "understanding"])
            ]
            for pattern, keywords in emotional_patterns:
                if any(term in content_lower for term in keywords):
                    patterns.append(pattern)

        elif reasoning_type == "gender_specific_mental_health":
            gender_patterns = [
                ("gender_role_awareness", ["gender", "role", "expectation", "stereotype", "identity"]),
                ("cultural_sensitivity", ["culture", "cultural", "background", "tradition", "values"]),
                ("identity_validation", ["identity", "validation", "acceptance", "affirmation"]),
                ("societal_pressure_recognition", ["pressure", "expectation", "society", "social", "norm"]),
                ("support_system_building", ["support", "network", "community", "connection"]),
                ("resource_identification", ["resource", "help", "assistance", "service", "tool"])
            ]
            for pattern, keywords in gender_patterns:
                if any(term in content_lower for term in keywords):
                    patterns.append(pattern)

        return patterns if patterns else ["general_reasoning"]

    def _extract_cot_steps(self, content: str) -> list:
        """Extract chain-of-thought reasoning steps from content."""
        # Look for common CoT indicators
        cot_indicators = [
            "first", "second", "third", "next", "then", "therefore", "because",
            "step 1", "step 2", "step 3", "initially", "subsequently", "finally",
            "let's think", "let me consider", "we need to", "it's important to"
        ]

        steps = []
        sentences = content.split(".")

        for sentence in sentences:
            sentence = sentence.strip().lower()
            if any(indicator in sentence for indicator in cot_indicators):
                steps.append(sentence[:150])  # Limit step length

        return steps[:8]  # Limit to 8 steps

    def _calculate_reasoning_complexity(self, patterns: list, cot_steps: list) -> float:
        """Calculate reasoning complexity score based on patterns and CoT steps."""
        if not patterns and not cot_steps:
            return 0.1

        # Base complexity from number of patterns
        pattern_complexity = min(len(patterns) * 0.15, 0.6)

        # CoT step complexity
        cot_complexity = min(len(cot_steps) * 0.1, 0.4)

        # Bonus for high-complexity patterns
        high_complexity_patterns = [
            "differential_diagnosis", "risk_assessment", "prognosis_evaluation",
            "executive_function_support", "emotional_regulation", "relationship_management"
        ]
        complexity_bonus = sum(0.1 for pattern in patterns if pattern in high_complexity_patterns)

        return min(pattern_complexity + cot_complexity + complexity_bonus, 1.0)

    def _assess_clinical_relevance(self, content: str, reasoning_type: str) -> float:
        """Assess clinical relevance of reasoning content."""
        content_lower = content.lower()

        clinical_terms = [
            "diagnosis", "treatment", "therapy", "symptom", "disorder", "condition",
            "assessment", "intervention", "clinical", "therapeutic", "mental health",
            "psychology", "psychiatry", "counseling", "patient", "client"
        ]

        term_count = sum(1 for term in clinical_terms if term in content_lower)

        # Adjust based on reasoning type
        if reasoning_type in ["clinical_diagnosis", "gender_specific_mental_health"]:
            return min(term_count * 0.08, 1.0)
        return min(term_count * 0.05 + 0.2, 1.0)

    def _save_conversations(self, conversations: list, output_path: str) -> None:
        """Save conversations to JSONL file."""
        with open(output_path, "w") as f:
            for conversation in conversations:
                f.write(json.dumps(conversation) + "\n")

        logger.info(f"Saved {len(conversations)} reasoning conversations to {output_path}")

    def _generate_report(self, dataset_results: dict, conversations: list, processing_time: float) -> dict:
        """Generate processing report."""
        return {
            "task": "5.13-5.24: Reasoning Enhancement Dataset Integration",
            "processing_summary": {
                "total_conversations": len(conversations),
                "target_conversations": self.target_conversations,
                "completion_percentage": (len(conversations) / self.target_conversations) * 100,
                "processing_time_seconds": processing_time,
                "datasets_processed": self.stats["datasets_processed"],
                "average_processing_rate": len(conversations) / processing_time if processing_time > 0 else 0
            },
            "quality_metrics": {
                "total_processed": self.stats["total_processed"],
                "total_accepted": self.stats["total_accepted"],
                "acceptance_rate": (self.stats["total_accepted"] / max(self.stats["total_processed"], 1)) * 100,
                "quality_filtered": self.stats["quality_filtered"],
                "format_errors": self.stats["format_errors"],
                "processing_errors": self.stats["processing_errors"]
            },
            "dataset_results": dataset_results,
            "conversation_analysis": {
                "reasoning_types": self._analyze_reasoning_types(conversations),
                "reasoning_patterns": self._analyze_reasoning_patterns(conversations),
                "complexity_distribution": self._analyze_complexity_distribution(conversations),
                "clinical_relevance_distribution": self._analyze_clinical_relevance(conversations)
            },
            "timestamp": datetime.now().isoformat()
        }

    def _analyze_reasoning_types(self, conversations: list) -> dict:
        """Analyze distribution of reasoning types."""
        types = {}
        for conv in conversations:
            reasoning_type = conv.get("metadata", {}).get("subcategory", "unknown")
            types[reasoning_type] = types.get(reasoning_type, 0) + 1
        return types

    def _analyze_reasoning_patterns(self, conversations: list) -> dict:
        """Analyze distribution of reasoning patterns."""
        pattern_counts = {}
        for conv in conversations:
            patterns = conv.get("metadata", {}).get("reasoning_patterns", [])
            for pattern in patterns:
                pattern_counts[pattern] = pattern_counts.get(pattern, 0) + 1
        return pattern_counts

    def _analyze_complexity_distribution(self, conversations: list) -> dict:
        """Analyze reasoning complexity distribution."""
        complexity_ranges = {"low": 0, "medium": 0, "high": 0}
        for conv in conversations:
            complexity = conv.get("metadata", {}).get("reasoning_complexity", 0.0)
            if complexity < 0.4:
                complexity_ranges["low"] += 1
            elif complexity < 0.7:
                complexity_ranges["medium"] += 1
            else:
                complexity_ranges["high"] += 1
        return complexity_ranges

    def _analyze_clinical_relevance(self, conversations: list) -> dict:
        """Analyze clinical relevance distribution."""
        relevance_ranges = {"low": 0, "medium": 0, "high": 0}
        for conv in conversations:
            relevance = conv.get("metadata", {}).get("clinical_relevance", 0.0)
            if relevance < 0.4:
                relevance_ranges["low"] += 1
            elif relevance < 0.7:
                relevance_ranges["medium"] += 1
            else:
                relevance_ranges["high"] += 1
        return relevance_ranges


def main():
    """Main execution function for reasoning enhancement processing."""

    # Configuration
    output_dir = "data/processed/reasoning"
    target_conversations = 5000  # 15% of 100K target


    try:
        # Create processor
        processor = ReasoningEnhancementProcessor(output_dir, target_conversations)

        # Process datasets
        result = processor.process_all_datasets()

        # Print results


        for _dataset_name, result_data in result["dataset_results"].items():
            "✅" if result_data["success"] else "❌"
            result_data.get("reasoning_type", "unknown")

        result["conversation_analysis"]["reasoning_types"]

        result["conversation_analysis"]["reasoning_patterns"]

        result["conversation_analysis"]["complexity_distribution"]

        result["conversation_analysis"]["clinical_relevance_distribution"]

        return True

    except Exception:
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
