#!/usr/bin/env python3
"""
Task 5.3: Chain-of-Thought Reasoning Integration (Tier 3)
Processes advanced therapeutic reasoning patterns from CoT datasets.
"""

import json
import os
import sys
from datetime import datetime
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from ai.dataset_pipeline.logger import get_logger

logger = get_logger("task_5_3_cot_reasoning")


class CoTReasoningProcessor:
    """
    Processor for Tier 3 Chain-of-Thought Reasoning Datasets.
    Handles advanced therapeutic reasoning patterns and clinical decision-making.
    """

    def __init__(self, output_dir: str = "data/processed/cot_reasoning"):
        self.output_dir = output_dir
        self.datasets_dir = "ai/datasets"

        # Ensure output directory exists
        os.makedirs(output_dir, exist_ok=True)

        # CoT reasoning dataset configurations (prioritized by therapeutic relevance)
        self.cot_configs = [
            {
                "name": "clinical_diagnosis_mental_health",
                "path": f"{self.datasets_dir}/CoT_Reasoning_Clinical_Diagnosis_Mental_Health/CoT_Reasoning_Clinical_Diagnosis_Mental_Health.json",
                "tier": 3,
                "reasoning_type": "clinical_diagnostic",
                "expected_quality": 0.90,
                "description": "Clinical diagnostic reasoning (38MB, 30K+ entries)"
            },
            {
                "name": "heartbreak_and_breakups",
                "path": f"{self.datasets_dir}/CoT_Heartbreak_and_Breakups/CoT-Breakups and heartbreak-9.8k.json",
                "tier": 3,
                "reasoning_type": "emotional_intelligence",
                "expected_quality": 0.85,
                "description": "Emotional intelligence & relationship therapy (38MB, 98K+ entries)"
            },
            {
                "name": "neurodivergent_vs_neurotypical",
                "path": f"{self.datasets_dir}/CoT_Neurodivergent_vs_Neurotypical_Interactions/CoT_Neurodivergent vs. Neurotypical Interactions.json",
                "tier": 3,
                "reasoning_type": "neurodiversity_aware",
                "expected_quality": 0.85,
                "description": "Neurodiversity-aware therapeutic approaches"
            },
            {
                "name": "mens_mental_health",
                "path": f"{self.datasets_dir}/CoT_Reasoning_Mens_Mental_Health/CoT_Reasoning_Mens_Mental_Health.json",
                "tier": 3,
                "reasoning_type": "gender_specific",
                "expected_quality": 0.80,
                "description": "Gender-specific therapeutic reasoning"
            },
            {
                "name": "philosophical_understanding",
                "path": f"{self.datasets_dir}/CoT_Philosophical_Understanding/CoT_Philosophical_Understanding.json",
                "tier": 3,
                "reasoning_type": "existential_philosophical",
                "expected_quality": 0.80,
                "description": "Existential/philosophical therapy (33MB, 60K entries)"
            },
            {
                "name": "rare_diseases_health_conditions",
                "path": f"{self.datasets_dir}/CoT_Rare-Diseases_And_Health-Conditions/CoT_Rare Disseases_Health Conditions_9.8k.json",
                "tier": 3,
                "reasoning_type": "medical_psychology",
                "expected_quality": 0.80,
                "description": "Medical psychology reasoning (68MB)"
            },
            {
                "name": "temporal_reasoning",
                "path": f"{self.datasets_dir}/CoT_Temporal_Reasoning_Dataset/CoT_Temporal_Reasoning_Dataset.json",
                "tier": 3,
                "reasoning_type": "temporal_therapeutic",
                "expected_quality": 0.75,
                "description": "Time-based therapeutic planning (15MB, 30K entries)"
            },
            {
                "name": "scientific_discovery_research",
                "path": f"{self.datasets_dir}/CoT_Reasoning_Scientific_Discovery_and_Research/CoT_Reasoning_Scientific Discovery and Research.json",
                "tier": 3,
                "reasoning_type": "evidence_based_practice",
                "expected_quality": 0.75,
                "description": "Evidence-based practice reasoning (38K+ entries)"
            },
            {
                "name": "cultural_nuances",
                "path": f"{self.datasets_dir}/CoT-Reasoning_Cultural_Nuances/CoT-Reasoning_Cultural_Nuances_Dataset.json",
                "tier": 3,
                "reasoning_type": "culturally_sensitive",
                "expected_quality": 0.75,
                "description": "Culturally-sensitive therapeutic approaches"
            },
            {
                "name": "legal_issues_laws",
                "path": f"{self.datasets_dir}/CoT_Legal_Issues_And_Laws/CoT_Legal_Issues_And_Laws.json",
                "tier": 3,
                "reasoning_type": "legal_ethical",
                "expected_quality": 0.70,
                "description": "Legal/ethical reasoning in therapy (25MB, 42K entries)"
            }
        ]

        self.stats = {
            "datasets_processed": 0,
            "total_conversations": 0,
            "total_accepted": 0,
            "quality_filtered": 0,
            "format_errors": 0,
            "processing_errors": 0,
            "reasoning_types": {},
            "reasoning_complexity": {}
        }

    def process_all_cot_datasets(self) -> dict:
        """Process all Chain-of-Thought reasoning datasets."""
        logger.info("Starting Task 5.3: Chain-of-Thought Reasoning Integration")
        start_time = datetime.now()

        all_conversations = []
        dataset_results = {}

        for config in self.cot_configs:
            logger.info(f"Processing {config['name']}: {config['description']}")

            try:
                conversations = self._process_single_cot_dataset(config)
                all_conversations.extend(conversations)

                dataset_results[config["name"]] = {
                    "conversations_processed": len(conversations),
                    "tier": config["tier"],
                    "reasoning_type": config["reasoning_type"],
                    "expected_quality": config["expected_quality"],
                    "description": config["description"],
                    "success": True
                }

                self.stats["datasets_processed"] += 1
                self.stats["reasoning_types"][config["reasoning_type"]] = self.stats["reasoning_types"].get(config["reasoning_type"], 0) + len(conversations)

                logger.info(f"Successfully processed {len(conversations)} conversations from {config['name']}")

            except Exception as e:
                logger.error(f"Failed to process {config['name']}: {e}")
                dataset_results[config["name"]] = {
                    "conversations_processed": 0,
                    "tier": config["tier"],
                    "reasoning_type": config["reasoning_type"],
                    "success": False,
                    "error": str(e)
                }
                self.stats["processing_errors"] += 1

        # Save consolidated CoT reasoning conversations
        output_path = os.path.join(self.output_dir, "cot_reasoning_conversations_consolidated.jsonl")
        self._save_conversations(all_conversations, output_path)

        # Generate comprehensive report
        processing_time = (datetime.now() - start_time).total_seconds()
        report = self._generate_report(dataset_results, all_conversations, processing_time)

        # Save report
        report_path = os.path.join(self.output_dir, "task_5_3_cot_reasoning_report.json")
        with open(report_path, "w") as f:
            json.dump(report, f, indent=2)

        logger.info(f"Task 5.3 completed. Processed {len(all_conversations)} CoT reasoning conversations")
        return report

    def _process_single_cot_dataset(self, config: dict) -> list:
        """Process a single CoT reasoning dataset."""
        dataset_path = config["path"]

        if not os.path.exists(dataset_path):
            logger.warning(f"Dataset file not found: {dataset_path}")
            return []

        processed_conversations = []

        try:
            with open(dataset_path) as f:
                data = json.load(f)

            if not isinstance(data, list):
                logger.error(f"Expected list format in {dataset_path}")
                return []

            for idx, item in enumerate(data):
                try:
                    self.stats["total_conversations"] += 1

                    # Process CoT reasoning conversation
                    processed = self._process_cot_conversation(item, config, idx)
                    if processed:
                        processed_conversations.append(processed)
                        self.stats["total_accepted"] += 1

                    # Progress logging
                    if idx % 10000 == 0 and idx > 0:
                        logger.info(f"Processed {idx} items from {config['name']}")

                except Exception as e:
                    logger.warning(f"Error processing item {idx} in {config['name']}: {e}")
                    self.stats["processing_errors"] += 1
                    continue

        except Exception as e:
            logger.error(f"Failed to read dataset {dataset_path}: {e}")
            raise

        return processed_conversations

    def _process_cot_conversation(self, item: dict, config: dict, item_id: int) -> dict:
        """Process a single CoT reasoning conversation."""
        try:
            # Extract conversation data from various CoT formats
            conversation = None
            reasoning_chain = None

            # Format 1: Direct conversation field
            if "conversation" in item:
                conversation = item["conversation"]
                reasoning_chain = item.get("reasoning", item.get("chain_of_thought", []))

            # Format 2: Messages field
            elif "messages" in item:
                conversation = item["messages"]
                reasoning_chain = item.get("reasoning", item.get("chain_of_thought", []))

            # Format 3: Input/Output with reasoning
            elif "input" in item and "output" in item:
                conversation = [
                    {"role": "client", "content": item["input"], "turn_id": 1},
                    {"role": "therapist", "content": item["output"], "turn_id": 2}
                ]
                reasoning_chain = item.get("reasoning", item.get("chain_of_thought", []))

            # Format 4: Question/Answer with reasoning
            elif "question" in item and "answer" in item:
                conversation = [
                    {"role": "client", "content": item["question"], "turn_id": 1},
                    {"role": "therapist", "content": item["answer"], "turn_id": 2}
                ]
                reasoning_chain = item.get("reasoning", item.get("chain_of_thought", []))

            # Format 5: Text field with embedded reasoning
            elif "text" in item:
                # Try to extract conversation and reasoning from text
                text_content = item["text"]
                if isinstance(text_content, str):
                    # Simple heuristic: split on reasoning indicators
                    if "reasoning:" in text_content.lower() or "chain of thought:" in text_content.lower():
                        parts = text_content.split("reasoning:")
                        if len(parts) == 1:
                            parts = text_content.split("chain of thought:")

                        if len(parts) >= 2:
                            conversation_part = parts[0].strip()
                            reasoning_part = parts[1].strip()

                            # Create basic conversation structure
                            conversation = [
                                {"role": "client", "content": conversation_part[:len(conversation_part)//2], "turn_id": 1},
                                {"role": "therapist", "content": conversation_part[len(conversation_part)//2:], "turn_id": 2}
                            ]
                            reasoning_chain = [reasoning_part]
                        else:
                            # No explicit reasoning, treat as conversation
                            conversation = [
                                {"role": "client", "content": text_content[:len(text_content)//2], "turn_id": 1},
                                {"role": "therapist", "content": text_content[len(text_content)//2:], "turn_id": 2}
                            ]
                            reasoning_chain = []
                    else:
                        # No reasoning indicators, treat as conversation
                        conversation = [
                            {"role": "client", "content": text_content[:len(text_content)//2], "turn_id": 1},
                            {"role": "therapist", "content": text_content[len(text_content)//2:], "turn_id": 2}
                        ]
                        reasoning_chain = []

            if not conversation or len(conversation) < 1:
                self.stats["format_errors"] += 1
                return None

            # Ensure conversation has proper structure
            if not isinstance(conversation, list):
                self.stats["format_errors"] += 1
                return None

            # Quality filtering
            total_content = " ".join([msg.get("content", "") for msg in conversation if isinstance(msg, dict)])
            if len(total_content) < 50:  # Too short for meaningful reasoning
                self.stats["quality_filtered"] += 1
                return None

            # Extract reasoning complexity
            reasoning_complexity = self._assess_reasoning_complexity(conversation, reasoning_chain)

            # Create standardized CoT conversation record
            record = {
                "conversation": conversation,
                "reasoning_chain": reasoning_chain if reasoning_chain else [],
                "metadata": {
                    "category": "cot_reasoning_therapeutic",
                    "tier": config["tier"],
                    "source_dataset": config["name"],
                    "reasoning_type": config["reasoning_type"],
                    "expected_quality": config["expected_quality"],
                    "description": config["description"],
                    "processing_timestamp": datetime.now().isoformat(),
                    "original_item_id": item_id,
                    "conversation_length": len(conversation),
                    "total_content_length": len(total_content),
                    "reasoning_chain_length": len(reasoning_chain) if reasoning_chain else 0,
                    "reasoning_complexity": reasoning_complexity,
                    "therapeutic_reasoning_indicators": self._extract_therapeutic_reasoning_indicators(total_content, reasoning_chain),
                    "clinical_decision_making": self._assess_clinical_decision_making(total_content, reasoning_chain),
                    "reasoning_quality_score": self._calculate_reasoning_quality_score(conversation, reasoning_chain)
                }
            }

            # Update reasoning complexity stats
            complexity_level = "high" if reasoning_complexity > 0.7 else "medium" if reasoning_complexity > 0.4 else "low"
            self.stats["reasoning_complexity"][complexity_level] = self.stats["reasoning_complexity"].get(complexity_level, 0) + 1

            return record

        except Exception as e:
            logger.warning(f"Failed to process CoT conversation at item {item_id}: {e}")
            return None

    def _assess_reasoning_complexity(self, conversation: list, reasoning_chain: list) -> float:
        """Assess the complexity of the reasoning in the conversation."""
        complexity_score = 0.0

        # Base complexity from conversation length and depth
        if len(conversation) >= 4:
            complexity_score += 0.2
        if len(conversation) >= 8:
            complexity_score += 0.1

        # Reasoning chain complexity
        if reasoning_chain:
            complexity_score += min(len(reasoning_chain) * 0.1, 0.3)

            # Check for complex reasoning patterns
            reasoning_text = " ".join(reasoning_chain) if isinstance(reasoning_chain, list) else str(reasoning_chain)
            complex_patterns = ["therefore", "because", "however", "although", "considering", "given that", "furthermore"]
            complexity_score += min(sum(0.05 for pattern in complex_patterns if pattern in reasoning_text.lower()), 0.2)

        # Content complexity indicators
        total_content = " ".join([msg.get("content", "") for msg in conversation])
        complexity_indicators = ["complex", "multifaceted", "nuanced", "intricate", "sophisticated", "comprehensive"]
        complexity_score += min(sum(0.05 for indicator in complexity_indicators if indicator in total_content.lower()), 0.2)

        return min(complexity_score, 1.0)

    def _extract_therapeutic_reasoning_indicators(self, content: str, reasoning_chain: list) -> list:
        """Extract therapeutic reasoning indicators."""
        content_lower = content.lower()
        reasoning_text = " ".join(reasoning_chain).lower() if reasoning_chain else ""
        combined_text = content_lower + " " + reasoning_text

        indicators = []

        reasoning_patterns = {
            "diagnostic_reasoning": ["diagnosis", "symptoms", "criteria", "assessment", "evaluation"],
            "treatment_planning": ["intervention", "strategy", "approach", "plan", "goals"],
            "risk_assessment": ["risk", "safety", "danger", "harm", "crisis"],
            "therapeutic_alliance": ["rapport", "relationship", "trust", "alliance", "connection"],
            "cognitive_restructuring": ["thoughts", "beliefs", "thinking", "cognitive", "reframe"],
            "emotional_processing": ["feelings", "emotions", "emotional", "process", "explore"],
            "behavioral_analysis": ["behavior", "actions", "patterns", "habits", "responses"],
            "systemic_thinking": ["family", "system", "relationships", "context", "environment"]
        }

        for indicator, keywords in reasoning_patterns.items():
            if any(keyword in combined_text for keyword in keywords):
                indicators.append(indicator)

        return indicators if indicators else ["general_therapeutic_reasoning"]

    def _assess_clinical_decision_making(self, content: str, reasoning_chain: list) -> float:
        """Assess the quality of clinical decision-making in the conversation."""
        decision_score = 0.0

        content_lower = content.lower()
        reasoning_text = " ".join(reasoning_chain).lower() if reasoning_chain else ""
        combined_text = content_lower + " " + reasoning_text

        # Decision-making indicators
        decision_indicators = ["decision", "choose", "recommend", "suggest", "propose", "consider"]
        decision_score += min(sum(0.1 for indicator in decision_indicators if indicator in combined_text), 0.3)

        # Evidence-based indicators
        evidence_indicators = ["research", "evidence", "studies", "literature", "proven", "effective"]
        decision_score += min(sum(0.1 for indicator in evidence_indicators if indicator in combined_text), 0.2)

        # Clinical reasoning indicators
        clinical_indicators = ["clinical", "therapeutic", "treatment", "intervention", "assessment"]
        decision_score += min(sum(0.05 for indicator in clinical_indicators if indicator in combined_text), 0.2)

        # Ethical considerations
        ethical_indicators = ["ethical", "appropriate", "boundaries", "professional", "responsible"]
        decision_score += min(sum(0.05 for indicator in ethical_indicators if indicator in combined_text), 0.15)

        # Reasoning chain quality
        if reasoning_chain and len(reasoning_chain) > 0:
            decision_score += 0.15

        return min(decision_score, 1.0)

    def _calculate_reasoning_quality_score(self, conversation: list, reasoning_chain: list) -> float:
        """Calculate overall reasoning quality score."""
        quality_score = 0.0

        # Conversation structure quality
        if len(conversation) >= 2:
            quality_score += 0.2
        if len(conversation) >= 4:
            quality_score += 0.1

        # Reasoning chain quality
        if reasoning_chain:
            quality_score += 0.2
            if len(reasoning_chain) >= 3:
                quality_score += 0.1

        # Content quality indicators
        total_content = " ".join([msg.get("content", "") for msg in conversation])
        quality_indicators = ["understand", "help", "support", "explore", "consider", "reflect"]
        quality_score += min(sum(0.05 for indicator in quality_indicators if indicator in total_content.lower()), 0.2)

        # Professional language
        professional_terms = ["therapy", "therapeutic", "counseling", "treatment", "intervention"]
        quality_score += min(sum(0.04 for term in professional_terms if term in total_content.lower()), 0.2)

        return min(quality_score, 1.0)

    def _save_conversations(self, conversations: list, output_path: str) -> None:
        """Save conversations to JSONL file."""
        with open(output_path, "w") as f:
            for conversation in conversations:
                f.write(json.dumps(conversation) + "\n")

        logger.info(f"Saved {len(conversations)} CoT reasoning conversations to {output_path}")

    def _generate_report(self, dataset_results: dict, conversations: list, processing_time: float) -> dict:
        """Generate comprehensive processing report."""
        return {
            "task": "5.3: Chain-of-Thought Reasoning Integration (Tier 3)",
            "processing_summary": {
                "total_conversations": len(conversations),
                "processing_time_seconds": processing_time,
                "datasets_processed": self.stats["datasets_processed"],
                "average_processing_rate": len(conversations) / processing_time if processing_time > 0 else 0
            },
            "quality_metrics": {
                "total_conversations_examined": self.stats["total_conversations"],
                "total_accepted": self.stats["total_accepted"],
                "acceptance_rate": (self.stats["total_accepted"] / max(self.stats["total_conversations"], 1)) * 100,
                "quality_filtered": self.stats["quality_filtered"],
                "format_errors": self.stats["format_errors"],
                "processing_errors": self.stats["processing_errors"]
            },
            "reasoning_types": self.stats["reasoning_types"],
            "reasoning_complexity": self.stats["reasoning_complexity"],
            "dataset_results": dataset_results,
            "conversation_analysis": {
                "therapeutic_reasoning_distribution": self._analyze_therapeutic_reasoning_distribution(conversations),
                "clinical_decision_making_distribution": self._analyze_clinical_decision_making_distribution(conversations),
                "reasoning_quality_distribution": self._analyze_reasoning_quality_distribution(conversations)
            },
            "timestamp": datetime.now().isoformat()
        }

    def _analyze_therapeutic_reasoning_distribution(self, conversations: list) -> dict:
        """Analyze therapeutic reasoning indicator distribution."""
        indicator_counts = {}
        for conv in conversations:
            indicators = conv.get("metadata", {}).get("therapeutic_reasoning_indicators", [])
            for indicator in indicators:
                indicator_counts[indicator] = indicator_counts.get(indicator, 0) + 1
        return indicator_counts

    def _analyze_clinical_decision_making_distribution(self, conversations: list) -> dict:
        """Analyze clinical decision-making quality distribution."""
        decision_ranges = {"low": 0, "medium": 0, "high": 0}
        for conv in conversations:
            decision_quality = conv.get("metadata", {}).get("clinical_decision_making", 0.0)
            if decision_quality < 0.3:
                decision_ranges["low"] += 1
            elif decision_quality < 0.7:
                decision_ranges["medium"] += 1
            else:
                decision_ranges["high"] += 1
        return decision_ranges

    def _analyze_reasoning_quality_distribution(self, conversations: list) -> dict:
        """Analyze reasoning quality score distribution."""
        quality_ranges = {"0.0-0.3": 0, "0.3-0.6": 0, "0.6-0.8": 0, "0.8-1.0": 0}
        for conv in conversations:
            quality = conv.get("metadata", {}).get("reasoning_quality_score", 0.0)
            if quality < 0.3:
                quality_ranges["0.0-0.3"] += 1
            elif quality < 0.6:
                quality_ranges["0.3-0.6"] += 1
            elif quality < 0.8:
                quality_ranges["0.6-0.8"] += 1
            else:
                quality_ranges["0.8-1.0"] += 1
        return quality_ranges


def main():
    """Main execution function for Task 5.3."""

    try:
        # Create processor
        processor = CoTReasoningProcessor()

        # Process all CoT reasoning datasets
        result = processor.process_all_cot_datasets()

        # Print results


        for _reasoning_type, _count in result["reasoning_types"].items():
            pass

        for _complexity, _count in result["reasoning_complexity"].items():
            pass

        return True

    except Exception:
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
