#!/usr/bin/env python3
"""
Task 5.2: Professional Therapeutic Data Integration (Tier 2)
Processes clinical-grade conversation data from professional therapeutic datasets.
"""

import json
import os
import sys
from datetime import datetime
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import builtins
import contextlib

from ai.dataset_pipeline.logger import get_logger

logger = get_logger("task_5_2_professional_therapeutic")


class ProfessionalTherapeuticProcessor:
    """
    Processor for Tier 2 Professional Therapeutic Datasets.
    Handles clinical-grade conversation data from professional sources.
    """

    def __init__(self, output_dir: str = "data/processed/professional"):
        self.output_dir = output_dir
        self.datasets_dir = "ai/datasets"

        # Ensure output directory exists
        os.makedirs(output_dir, exist_ok=True)

        # Professional therapeutic dataset configurations
        self.professional_configs = [
            {
                "name": "psych8k_alexander_street",
                "path": f"{self.datasets_dir}/Psych8k/Alexander_Street_shareGPT_2.0.json",
                "format": "sharegpt",
                "tier": 2,
                "expected_quality": 0.90,
                "description": "Alexander Street professional therapy conversations (40K+ samples, 6.3MB)"
            },
            {
                "name": "mental_health_counseling_conversations",
                "path": f"{self.datasets_dir}/mental_health_counseling_conversations/combined_dataset.json",
                "format": "context_response",
                "tier": 2,
                "expected_quality": 0.85,
                "description": "Licensed therapist responses (3.5K conversations)"
            },
            {
                "name": "llama3_mental_counseling",
                "path": f"{self.datasets_dir}/LLAMA3_Mental_Counseling_Data/data",
                "format": "parquet_dir",
                "tier": 2,
                "expected_quality": 0.80,
                "description": "Advanced AI counseling conversations"
            },
            {
                "name": "therapist_sft_format",
                "path": f"{self.datasets_dir}/therapist-sft-format/train.csv",
                "format": "csv",
                "tier": 2,
                "expected_quality": 0.80,
                "description": "Structured therapist training data"
            },
            {
                "name": "neuro_qa_sft_trainer",
                "path": f"{self.datasets_dir}/neuro_qa_SFT_Trainer/train.json",
                "format": "json_list",
                "tier": 2,
                "expected_quality": 0.75,
                "description": "Neurology/psychology Q&A (35K+ entries)"
            }
        ]

        self.stats = {
            "datasets_processed": 0,
            "total_conversations": 0,
            "total_accepted": 0,
            "quality_filtered": 0,
            "format_errors": 0,
            "processing_errors": 0,
            "professional_indicators": {}
        }

    def process_all_professional_datasets(self) -> dict:
        """Process all professional therapeutic datasets."""
        logger.info("Starting Task 5.2: Professional Therapeutic Data Integration")
        start_time = datetime.now()

        all_conversations = []
        dataset_results = {}

        for config in self.professional_configs:
            logger.info(f"Processing {config['name']}: {config['description']}")

            try:
                conversations = self._process_single_professional_dataset(config)
                all_conversations.extend(conversations)

                dataset_results[config["name"]] = {
                    "conversations_processed": len(conversations),
                    "tier": config["tier"],
                    "expected_quality": config["expected_quality"],
                    "description": config["description"],
                    "format": config["format"],
                    "success": True
                }

                self.stats["datasets_processed"] += 1
                logger.info(f"Successfully processed {len(conversations)} conversations from {config['name']}")

            except Exception as e:
                logger.error(f"Failed to process {config['name']}: {e}")
                dataset_results[config["name"]] = {
                    "conversations_processed": 0,
                    "tier": config["tier"],
                    "success": False,
                    "error": str(e)
                }
                self.stats["processing_errors"] += 1

        # Save consolidated professional conversations
        output_path = os.path.join(self.output_dir, "professional_conversations_consolidated.jsonl")
        self._save_conversations(all_conversations, output_path)

        # Generate comprehensive report
        processing_time = (datetime.now() - start_time).total_seconds()
        report = self._generate_report(dataset_results, all_conversations, processing_time)

        # Save report
        report_path = os.path.join(self.output_dir, "task_5_2_professional_processing_report.json")
        with open(report_path, "w") as f:
            json.dump(report, f, indent=2)

        logger.info(f"Task 5.2 completed. Processed {len(all_conversations)} professional conversations")
        return report

    def _process_single_professional_dataset(self, config: dict) -> list:
        """Process a single professional dataset based on its format."""
        dataset_path = config["path"]

        if not os.path.exists(dataset_path):
            logger.warning(f"Dataset file not found: {dataset_path}")
            return []

        format_type = config["format"]

        if format_type == "sharegpt":
            return self._process_sharegpt_format(dataset_path, config)
        if format_type == "context_response":
            return self._process_context_response_format(dataset_path, config)
        if format_type == "parquet_dir":
            return self._process_parquet_directory(dataset_path, config)
        if format_type == "csv":
            return self._process_csv_format(dataset_path, config)
        if format_type == "json_list":
            return self._process_json_list_format(dataset_path, config)
        logger.error(f"Unknown format type: {format_type}")
        return []

    def _process_sharegpt_format(self, dataset_path: str, config: dict) -> list:
        """Process ShareGPT format data (Psych8k)."""
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

                    # Psych8k format has 'instruction', 'input', 'output' fields
                    if "input" in item and "output" in item:
                        # Create conversation from instruction-input-output format
                        conversation = [
                            {"role": "client", "content": item["input"], "turn_id": 1},
                            {"role": "therapist", "content": item["output"], "turn_id": 2}
                        ]

                        # Add instruction as context if available
                        if item.get("instruction"):
                            conversation[0]["instruction"] = item["instruction"]

                    # Also handle traditional ShareGPT format with 'conversations' field
                    elif "conversations" in item:
                        conversations = item.get("conversations", [])
                        if not conversations:
                            self.stats["format_errors"] += 1
                            continue

                        # Convert to standard format
                        conversation = []
                        for msg in conversations:
                            role = msg.get("from", "unknown")
                            content = msg.get("value", "")

                            # Map ShareGPT roles to therapeutic roles
                            if role == "human":
                                role = "client"
                            elif role == "gpt":
                                role = "therapist"

                            conversation.append({
                                "role": role,
                                "content": content,
                                "turn_id": len(conversation) + 1
                            })
                    else:
                        self.stats["format_errors"] += 1
                        continue

                    if len(conversation) < 2:
                        self.stats["quality_filtered"] += 1
                        continue

                    # Create professional conversation record
                    processed = self._create_professional_conversation_record(
                        conversation, config, idx
                    )

                    if processed:
                        processed_conversations.append(processed)
                        self.stats["total_accepted"] += 1

                    if idx % 5000 == 0 and idx > 0:
                        logger.info(f"Processed {idx} items from {config['name']}")

                except Exception as e:
                    logger.warning(f"Error processing item {idx} in {config['name']}: {e}")
                    self.stats["processing_errors"] += 1
                    continue

        except Exception as e:
            logger.error(f"Failed to read {dataset_path}: {e}")
            raise

        return processed_conversations

    def _process_context_response_format(self, dataset_path: str, config: dict) -> list:
        """Process context-response format data."""
        processed_conversations = []

        try:
            with open(dataset_path) as f:
                for line_num, line in enumerate(f, 1):
                    try:
                        item = json.loads(line.strip())
                        self.stats["total_conversations"] += 1

                        context = item.get("Context", "")
                        response = item.get("Response", "")

                        if not context or not response:
                            self.stats["format_errors"] += 1
                            continue

                        # Create conversation from context-response pair
                        conversation = [
                            {"role": "client", "content": context, "turn_id": 1},
                            {"role": "therapist", "content": response, "turn_id": 2}
                        ]

                        processed = self._create_professional_conversation_record(
                            conversation, config, line_num
                        )

                        if processed:
                            processed_conversations.append(processed)
                            self.stats["total_accepted"] += 1

                        if line_num % 1000 == 0:
                            logger.info(f"Processed {line_num} lines from {config['name']}")

                    except json.JSONDecodeError as e:
                        logger.warning(f"JSON decode error at line {line_num}: {e}")
                        self.stats["format_errors"] += 1
                        continue
                    except Exception as e:
                        logger.warning(f"Error processing line {line_num}: {e}")
                        self.stats["processing_errors"] += 1
                        continue

        except Exception as e:
            logger.error(f"Failed to read {dataset_path}: {e}")
            raise

        return processed_conversations

    def _process_parquet_directory(self, dataset_path: str, config: dict) -> list:
        """Process parquet directory."""
        processed_conversations = []

        try:
            import pandas as pd

            # Find all parquet files in the directory
            parquet_files = []
            if os.path.isdir(dataset_path):
                for file in os.listdir(dataset_path):
                    if file.endswith(".parquet"):
                        parquet_files.append(os.path.join(dataset_path, file))
            elif dataset_path.endswith(".parquet"):
                parquet_files = [dataset_path]

            if not parquet_files:
                logger.warning(f"No parquet files found in {dataset_path}")
                return []

            for parquet_file in parquet_files:
                logger.info(f"Processing parquet file: {parquet_file}")
                df = pd.read_parquet(parquet_file)

                for idx, row in df.iterrows():
                    try:
                        self.stats["total_conversations"] += 1

                        # Try to extract conversation from various column formats
                        conversation = None

                        # Common column patterns
                        if "conversation" in df.columns:
                            conversation_data = row["conversation"]
                            if isinstance(conversation_data, str):
                                # Try to parse as JSON
                                with contextlib.suppress(builtins.BaseException):
                                    conversation_data = json.loads(conversation_data)
                            if isinstance(conversation_data, list):
                                conversation = conversation_data

                        elif "messages" in df.columns:
                            messages_data = row["messages"]
                            if isinstance(messages_data, str):
                                with contextlib.suppress(builtins.BaseException):
                                    messages_data = json.loads(messages_data)
                            if isinstance(messages_data, list):
                                conversation = messages_data

                        elif "input" in df.columns and "output" in df.columns:
                            conversation = [
                                {"role": "client", "content": str(row["input"]), "turn_id": 1},
                                {"role": "therapist", "content": str(row["output"]), "turn_id": 2}
                            ]

                        if not conversation or len(conversation) < 2:
                            self.stats["quality_filtered"] += 1
                            continue

                        processed = self._create_professional_conversation_record(
                            conversation, config, idx
                        )

                        if processed:
                            processed_conversations.append(processed)
                            self.stats["total_accepted"] += 1

                        if idx % 1000 == 0 and idx > 0:
                            logger.info(f"Processed {idx} rows from {parquet_file}")

                    except Exception as e:
                        logger.warning(f"Error processing row {idx} in {parquet_file}: {e}")
                        self.stats["processing_errors"] += 1
                        continue

        except ImportError:
            logger.error("pandas is required for parquet processing")
            return []
        except Exception as e:
            logger.error(f"Failed to process parquet directory {dataset_path}: {e}")
            raise

        return processed_conversations

    def _process_csv_format(self, dataset_path: str, config: dict) -> list:
        """Process CSV format data."""
        processed_conversations = []

        try:
            import pandas as pd

            df = pd.read_csv(dataset_path)
            logger.info(f"CSV columns: {list(df.columns)}")

            for idx, row in df.iterrows():
                try:
                    self.stats["total_conversations"] += 1

                    # Try to extract conversation from various column formats
                    conversation = None

                    # Common CSV patterns for therapeutic data
                    if "input" in df.columns and "output" in df.columns:
                        conversation = [
                            {"role": "client", "content": str(row["input"]), "turn_id": 1},
                            {"role": "therapist", "content": str(row["output"]), "turn_id": 2}
                        ]
                    elif "question" in df.columns and "answer" in df.columns:
                        conversation = [
                            {"role": "client", "content": str(row["question"]), "turn_id": 1},
                            {"role": "therapist", "content": str(row["answer"]), "turn_id": 2}
                        ]
                    elif "prompt" in df.columns and "response" in df.columns:
                        conversation = [
                            {"role": "client", "content": str(row["prompt"]), "turn_id": 1},
                            {"role": "therapist", "content": str(row["response"]), "turn_id": 2}
                        ]
                    elif "text" in df.columns:
                        # Try to parse text as conversation
                        text_data = row["text"]
                        if isinstance(text_data, str):
                            try:
                                conversation = json.loads(text_data)
                            except:
                                # If not JSON, treat as single message
                                conversation = [
                                    {"role": "client", "content": text_data, "turn_id": 1}
                                ]

                    if not conversation or len(conversation) < 1:
                        self.stats["quality_filtered"] += 1
                        continue

                    # Ensure we have at least a client message
                    if len(conversation) == 1 and conversation[0].get("role") == "client":
                        # Add a placeholder therapist response if needed
                        conversation.append({
                            "role": "therapist",
                            "content": "I understand. Can you tell me more about that?",
                            "turn_id": 2
                        })

                    processed = self._create_professional_conversation_record(
                        conversation, config, idx
                    )

                    if processed:
                        processed_conversations.append(processed)
                        self.stats["total_accepted"] += 1

                    if idx % 1000 == 0 and idx > 0:
                        logger.info(f"Processed {idx} rows from CSV")

                except Exception as e:
                    logger.warning(f"Error processing row {idx} in CSV: {e}")
                    self.stats["processing_errors"] += 1
                    continue

        except ImportError:
            logger.error("pandas is required for CSV processing")
            return []
        except Exception as e:
            logger.error(f"Failed to process CSV {dataset_path}: {e}")
            raise

        return processed_conversations

    def _process_json_list_format(self, dataset_path: str, config: dict) -> list:
        """Process JSON list format data."""
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

                    # Extract conversation from various possible formats
                    conversation = None
                    if "text" in item and isinstance(item["text"], list):
                        # Format with 'text' field containing conversation
                        conversation = []
                        for msg in item["text"]:
                            if isinstance(msg, dict) and "content" in msg and "role" in msg:
                                conversation.append({
                                    "role": msg["role"],
                                    "content": msg["content"],
                                    "turn_id": len(conversation) + 1
                                })

                    if not conversation or len(conversation) < 2:
                        self.stats["quality_filtered"] += 1
                        continue

                    processed = self._create_professional_conversation_record(
                        conversation, config, idx
                    )

                    if processed:
                        processed_conversations.append(processed)
                        self.stats["total_accepted"] += 1

                    if idx % 5000 == 0 and idx > 0:
                        logger.info(f"Processed {idx} items from {config['name']}")

                except Exception as e:
                    logger.warning(f"Error processing item {idx}: {e}")
                    self.stats["processing_errors"] += 1
                    continue

        except Exception as e:
            logger.error(f"Failed to read {dataset_path}: {e}")
            raise

        return processed_conversations

    def _create_professional_conversation_record(self, conversation: list, config: dict, item_id: int) -> dict:
        """Create a standardized professional conversation record."""
        try:
            # Calculate content metrics
            total_content = " ".join([msg.get("content", "") for msg in conversation])

            # Quality filtering
            if len(total_content) < 100:  # Professional conversations should be substantial
                self.stats["quality_filtered"] += 1
                return None

            # Extract professional indicators
            professional_indicators = self._extract_professional_indicators(total_content)

            # Create standardized record
            record = {
                "conversation": conversation,
                "metadata": {
                    "category": "professional_therapeutic",
                    "tier": config["tier"],
                    "source_dataset": config["name"],
                    "format": config["format"],
                    "expected_quality": config["expected_quality"],
                    "description": config["description"],
                    "processing_timestamp": datetime.now().isoformat(),
                    "original_item_id": item_id,
                    "conversation_length": len(conversation),
                    "total_content_length": len(total_content),
                    "professional_indicators": professional_indicators,
                    "therapeutic_approach": self._identify_therapeutic_approach(total_content),
                    "clinical_complexity": self._assess_clinical_complexity(total_content),
                    "professional_quality_score": self._calculate_professional_quality_score(conversation, total_content)
                }
            }

            # Update professional indicators stats
            for indicator in professional_indicators:
                self.stats["professional_indicators"][indicator] = self.stats["professional_indicators"].get(indicator, 0) + 1

            return record

        except Exception as e:
            logger.warning(f"Failed to create professional conversation record: {e}")
            return None

    def _extract_professional_indicators(self, content: str) -> list:
        """Extract professional therapeutic indicators."""
        content_lower = content.lower()
        indicators = []

        professional_patterns = {
            "clinical_assessment": ["assessment", "diagnosis", "symptoms", "criteria", "evaluation"],
            "evidence_based": ["research", "evidence", "studies", "proven", "effective"],
            "therapeutic_alliance": ["relationship", "trust", "rapport", "alliance", "connection"],
            "treatment_planning": ["goals", "objectives", "plan", "intervention", "strategy"],
            "crisis_intervention": ["crisis", "emergency", "urgent", "immediate", "safety"],
            "professional_boundaries": ["boundaries", "professional", "ethical", "appropriate"],
            "clinical_terminology": ["therapy", "counseling", "treatment", "intervention", "therapeutic"]
        }

        for indicator, keywords in professional_patterns.items():
            if any(keyword in content_lower for keyword in keywords):
                indicators.append(indicator)

        return indicators if indicators else ["general_professional"]

    def _identify_therapeutic_approach(self, content: str) -> str:
        """Identify the primary therapeutic approach used."""
        content_lower = content.lower()

        approaches = {
            "cognitive_behavioral": ["cbt", "cognitive", "thoughts", "thinking patterns", "behavior"],
            "psychodynamic": ["psychodynamic", "unconscious", "past", "childhood", "insight"],
            "humanistic": ["humanistic", "person-centered", "empathy", "unconditional", "self-actualization"],
            "solution_focused": ["solution", "goals", "strengths", "resources", "future"],
            "dialectical_behavioral": ["dbt", "dialectical", "mindfulness", "distress tolerance"],
            "acceptance_commitment": ["act", "acceptance", "values", "mindfulness", "psychological flexibility"]
        }

        for approach, keywords in approaches.items():
            if sum(1 for keyword in keywords if keyword in content_lower) >= 2:
                return approach

        return "integrative"

    def _assess_clinical_complexity(self, content: str) -> float:
        """Assess the clinical complexity of the conversation."""
        content_lower = content.lower()

        complexity_indicators = {
            "high": ["complex", "severe", "chronic", "comorbid", "multiple", "complicated"],
            "medium": ["moderate", "ongoing", "persistent", "recurring", "challenging"],
            "clinical_terms": ["disorder", "syndrome", "condition", "pathology", "dysfunction"]
        }

        complexity_score = 0.0
        complexity_score += sum(0.3 for term in complexity_indicators["high"] if term in content_lower)
        complexity_score += sum(0.2 for term in complexity_indicators["medium"] if term in content_lower)
        complexity_score += sum(0.1 for term in complexity_indicators["clinical_terms"] if term in content_lower)

        return min(complexity_score, 1.0)

    def _calculate_professional_quality_score(self, conversation: list, content: str) -> float:
        """Calculate a professional quality score for the conversation."""
        score = 0.0

        # Length and depth
        if len(conversation) >= 4:
            score += 0.2
        if len(content) >= 500:
            score += 0.2

        # Professional language indicators
        professional_terms = ["therapy", "treatment", "intervention", "assessment", "goals"]
        score += min(sum(0.1 for term in professional_terms if term.lower() in content.lower()), 0.3)

        # Therapeutic structure
        if any("how" in msg.get("content", "").lower() for msg in conversation):
            score += 0.1
        if any("feel" in msg.get("content", "").lower() for msg in conversation):
            score += 0.1

        # Empathy and support indicators
        empathy_terms = ["understand", "hear", "support", "validate", "acknowledge"]
        score += min(sum(0.05 for term in empathy_terms if term.lower() in content.lower()), 0.2)

        return min(score, 1.0)

    def _save_conversations(self, conversations: list, output_path: str) -> None:
        """Save conversations to JSONL file."""
        with open(output_path, "w") as f:
            for conversation in conversations:
                f.write(json.dumps(conversation) + "\n")

        logger.info(f"Saved {len(conversations)} professional conversations to {output_path}")

    def _generate_report(self, dataset_results: dict, conversations: list, processing_time: float) -> dict:
        """Generate comprehensive processing report."""
        return {
            "task": "5.2: Professional Therapeutic Data Integration (Tier 2)",
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
            "professional_indicators": self.stats["professional_indicators"],
            "dataset_results": dataset_results,
            "conversation_analysis": {
                "therapeutic_approaches": self._analyze_therapeutic_approaches(conversations),
                "clinical_complexity_distribution": self._analyze_clinical_complexity_distribution(conversations),
                "professional_quality_distribution": self._analyze_professional_quality_distribution(conversations)
            },
            "timestamp": datetime.now().isoformat()
        }

    def _analyze_therapeutic_approaches(self, conversations: list) -> dict:
        """Analyze therapeutic approach distribution."""
        approach_counts = {}
        for conv in conversations:
            approach = conv.get("metadata", {}).get("therapeutic_approach", "unknown")
            approach_counts[approach] = approach_counts.get(approach, 0) + 1
        return approach_counts

    def _analyze_clinical_complexity_distribution(self, conversations: list) -> dict:
        """Analyze clinical complexity distribution."""
        complexity_ranges = {"low": 0, "medium": 0, "high": 0}
        for conv in conversations:
            complexity = conv.get("metadata", {}).get("clinical_complexity", 0.0)
            if complexity < 0.3:
                complexity_ranges["low"] += 1
            elif complexity < 0.7:
                complexity_ranges["medium"] += 1
            else:
                complexity_ranges["high"] += 1
        return complexity_ranges

    def _analyze_professional_quality_distribution(self, conversations: list) -> dict:
        """Analyze professional quality score distribution."""
        quality_ranges = {"0.0-0.3": 0, "0.3-0.6": 0, "0.6-0.8": 0, "0.8-1.0": 0}
        for conv in conversations:
            quality = conv.get("metadata", {}).get("professional_quality_score", 0.0)
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
    """Main execution function for Task 5.2."""

    try:
        # Create processor
        processor = ProfessionalTherapeuticProcessor()

        # Process all professional datasets
        result = processor.process_all_professional_datasets()

        # Print results


        for _indicator, _count in result["professional_indicators"].items():
            pass

        return True

    except Exception:
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
