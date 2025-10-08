#!/usr/bin/env python3
"""
Task 5.33: TF-IDF Feature Vector Integration (Enterprise-Grade)
Memory-efficient, scalable integration of 256-dimensional TF-IDF features for ML applications.
"""

import csv
import json
import logging
import os
from collections import defaultdict
from datetime import datetime
from typing import Any

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

class TFIDFIntegrationProcessor:
    """Enterprise-grade TF-IDF feature integration processor with memory efficiency."""

    def __init__(self, output_dir: str = "ai/data/processed/phase_4_reddit_mental_health/task_5_33_tfidf_integration"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

        # Data sources with phase-based paths
        self.data_sources = [
            {
                "name": "condition_specific",
                "path": "data/processed/phase_4_reddit_mental_health/task_5_27_condition_specific/condition_specific_conversations.jsonl",
                "category": "mental_health_conditions"
            },
            {
                "name": "specialized_populations",
                "path": "data/processed/phase_4_reddit_mental_health/task_5_28_specialized_populations/specialized_populations_conversations.jsonl",
                "category": "specialized_populations"
            },
            {
                "name": "temporal_analysis",
                "path": "data/processed/phase_4_reddit_mental_health/task_5_29_temporal_analysis/temporal_analysis_conversations.jsonl",
                "category": "temporal_analysis"
            },
            {
                "name": "crisis_detection",
                "path": "data/processed/phase_4_reddit_mental_health/task_5_30_crisis_detection/crisis_detection_conversations.jsonl",
                "category": "crisis_intervention"
            },
            {
                "name": "additional_specialized",
                "path": "data/processed/phase_4_reddit_mental_health/task_5_31_additional_specialized/additional_specialized_conversations.jsonl",
                "category": "specialized_populations"
            },
            {
                "name": "control_groups",
                "path": "data/processed/phase_4_reddit_mental_health/task_5_32_control_groups/control_group_conversations.jsonl",
                "category": "control_baseline"
            }
        ]

        # Processing configuration
        self.batch_size = 1000  # Process 1000 conversations at a time
        self.feature_dimensions = 256

        # Statistics tracking
        self.stats = {
            "total_conversations": 0,
            "conversations_with_features": 0,
            "conversations_without_features": 0,
            "total_feature_vectors": 0,
            "source_breakdown": defaultdict(int),
            "category_breakdown": defaultdict(int),
            "processing_errors": 0
        }

        # Output file handles
        self.output_files = {}

    def process_integration(self) -> dict[str, Any]:
        """Main processing function with enterprise-grade error handling and logging."""
        logger.info("🔄 Starting Task 5.33: Enterprise TF-IDF Feature Integration")
        logger.info(f"💾 Memory-efficient batch processing (batch_size={self.batch_size})")
        logger.info(f"📊 Target feature dimensions: {self.feature_dimensions}")

        try:
            # Initialize output files
            self._initialize_output_files()

            # Process each data source
            for source in self.data_sources:
                logger.info(f"\n📄 Processing source: {source['name']}")
                self._process_source(source)

            # Finalize output files
            self._finalize_output_files()

            # Generate comprehensive report
            report = self._generate_report()

            # Save report
            report_path = os.path.join(self.output_dir, "integration_report.json")
            with open(report_path, "w", encoding="utf-8") as f:
                json.dump(report, f, indent=2, ensure_ascii=False)

            logger.info("\n✅ Task 5.33 completed successfully!")
            logger.info(f"   📊 Total conversations: {self.stats['total_conversations']:,}")
            logger.info(f"   🤖 ML-ready features: {self.stats['conversations_with_features']:,}")
            logger.info(f"   📁 Output directory: {self.output_dir}")
            logger.info(f"   📋 Report: {report_path}")

            return report

        except Exception as e:
            logger.error(f"❌ Critical error in TF-IDF integration: {e}")
            raise

    def _initialize_output_files(self):
        """Initialize all output files with proper headers."""
        logger.info("📁 Initializing output files...")

        # Feature matrix CSV
        feature_matrix_path = os.path.join(self.output_dir, "feature_matrix.csv")
        self.output_files["features"] = open(feature_matrix_path, "w", newline="")
        feature_writer = csv.writer(self.output_files["features"])
        feature_headers = ["conversation_id"] + [f"tfidf_{i}" for i in range(self.feature_dimensions)]
        feature_writer.writerow(feature_headers)

        # Labels CSV
        labels_path = os.path.join(self.output_dir, "labels.csv")
        self.output_files["labels"] = open(labels_path, "w", newline="")
        label_writer = csv.writer(self.output_files["labels"])
        label_headers = ["conversation_id", "source", "category", "condition", "risk_level",
                        "temporal_period", "population", "quality_score", "therapeutic_relevance"]
        label_writer.writerow(label_headers)

        # Metadata CSV
        metadata_path = os.path.join(self.output_dir, "metadata.csv")
        self.output_files["metadata"] = open(metadata_path, "w", newline="")
        metadata_writer = csv.writer(self.output_files["metadata"])
        metadata_headers = ["conversation_id", "text_length", "conversation_length",
                           "feature_count", "processing_timestamp"]
        metadata_writer.writerow(metadata_headers)

        # ML-ready conversations JSONL
        conversations_path = os.path.join(self.output_dir, "ml_conversations.jsonl")
        self.output_files["conversations"] = open(conversations_path, "w", encoding="utf-8")

        # Create CSV writers for reuse
        self.csv_writers = {
            "features": csv.writer(self.output_files["features"]),
            "labels": csv.writer(self.output_files["labels"]),
            "metadata": csv.writer(self.output_files["metadata"])
        }

    def _finalize_output_files(self):
        """Close all output files safely."""
        logger.info("📁 Finalizing output files...")
        for file_handle in self.output_files.values():
            file_handle.close()

    def _process_source(self, source: dict[str, str]):
        """Process a single data source with batch processing."""
        file_path = source["path"]

        if not os.path.exists(file_path):
            logger.warning(f"   ⚠️  Source file not found: {file_path}")
            return

        source_stats = {"processed": 0, "with_features": 0, "errors": 0}
        conversation_id = self.stats["total_conversations"]

        try:
            batch = []
            with open(file_path, encoding="utf-8") as f:
                for line_num, line in enumerate(f, 1):
                    try:
                        conversation = json.loads(line.strip())
                        batch.append(conversation)

                        # Process batch when full
                        if len(batch) >= self.batch_size:
                            processed_count = self._process_batch(batch, conversation_id, source)
                            conversation_id += len(batch)
                            source_stats["processed"] += len(batch)
                            source_stats["with_features"] += processed_count
                            batch = []  # Clear batch to free memory

                            # Progress logging
                            if source_stats["processed"] % 10000 == 0:
                                logger.info(f"      📈 Processed {source_stats['processed']:,} conversations...")

                    except json.JSONDecodeError as e:
                        source_stats["errors"] += 1
                        if source_stats["errors"] <= 5:  # Log first 5 errors only
                            logger.warning(f"   ⚠️  JSON decode error at line {line_num}: {e}")
                        continue

                # Process remaining batch
                if batch:
                    processed_count = self._process_batch(batch, conversation_id, source)
                    source_stats["processed"] += len(batch)
                    source_stats["with_features"] += processed_count

            # Update global statistics
            self.stats["total_conversations"] += source_stats["processed"]
            self.stats["conversations_with_features"] += source_stats["with_features"]
            self.stats["conversations_without_features"] += (source_stats["processed"] - source_stats["with_features"])
            self.stats["processing_errors"] += source_stats["errors"]
            self.stats["source_breakdown"][source["name"]] = source_stats["processed"]
            self.stats["category_breakdown"][source["category"]] += source_stats["processed"]

            logger.info(f"   ✅ {source['name']}: {source_stats['processed']:,} conversations, "
                       f"{source_stats['with_features']:,} with features")

        except Exception as e:
            logger.error(f"   ❌ Error processing {source['name']}: {e}")
            self.stats["processing_errors"] += 1

    def _process_batch(self, batch: list[dict], start_id: int, source: dict[str, str]) -> int:
        """Process a batch of conversations efficiently."""
        processed_with_features = 0

        for i, conversation in enumerate(batch):
            conversation_id = start_id + i
            metadata = conversation.get("metadata", {})
            tfidf_features = metadata.get("tfidf_features", {})

            if self._has_valid_features(tfidf_features):
                # Extract and validate feature vector
                feature_vector = self._extract_feature_vector(tfidf_features)

                if feature_vector is not None:
                    # Write feature vector
                    feature_row = [conversation_id, *feature_vector]
                    self.csv_writers["features"].writerow(feature_row)

                    # Write labels
                    label_row = [
                        conversation_id,
                        source["name"],
                        source["category"],
                        metadata.get("condition", "unknown"),
                        metadata.get("risk_level", "unknown"),
                        metadata.get("temporal_period", "unknown"),
                        metadata.get("population", "unknown"),
                        metadata.get("quality_score", 0.0),
                        metadata.get("therapeutic_relevance", 0.0)
                    ]
                    self.csv_writers["labels"].writerow(label_row)

                    # Write metadata
                    metadata_row = [
                        conversation_id,
                        metadata.get("text_length", 0),
                        metadata.get("conversation_length", 0),
                        len(tfidf_features),
                        datetime.now().isoformat()
                    ]
                    self.csv_writers["metadata"].writerow(metadata_row)

                    # Write ML-ready conversation
                    ml_conversation = {
                        "conversation_id": conversation_id,
                        "conversation": conversation.get("conversation", []),
                        "metadata": {
                            **metadata,
                            "ml_ready": True,
                            "feature_vector_id": conversation_id,
                            "processing_timestamp": datetime.now().isoformat()
                        }
                    }
                    self.output_files["conversations"].write(
                        json.dumps(ml_conversation, ensure_ascii=False) + "\n"
                    )

                    processed_with_features += 1
                    self.stats["total_feature_vectors"] += 1

        return processed_with_features

    def _has_valid_features(self, tfidf_features: dict) -> bool:
        """Check if TF-IDF features are valid and non-empty."""
        return (
            isinstance(tfidf_features, dict) and
            len(tfidf_features) > 0 and
            any(isinstance(v, (int, float)) and v != 0 for v in tfidf_features.values())
        )

    def _extract_feature_vector(self, tfidf_features: dict) -> list[float] | None:
        """Extract and validate 256-dimensional feature vector."""
        try:
            # Create feature vector with proper ordering
            feature_vector = [0.0] * self.feature_dimensions

            for feature_name, value in tfidf_features.items():
                if isinstance(value, (int, float)):
                    # Extract feature index from feature name
                    if feature_name.isdigit():
                        idx = int(feature_name)
                    elif "_" in feature_name:
                        # Handle features like 'feature_123' or 'tfidf_123'
                        idx = int(feature_name.split("_")[-1])
                    else:
                        continue

                    if 0 <= idx < self.feature_dimensions:
                        feature_vector[idx] = float(value)

            # Validate feature vector
            if any(v != 0 for v in feature_vector):
                return feature_vector
            return None

        except (ValueError, TypeError, IndexError):
            return None

    def _generate_report(self) -> dict[str, Any]:
        """Generate comprehensive integration report."""
        return {
            "task": "5.33: TF-IDF Feature Vector Integration",
            "version": "enterprise_v2",
            "processing_summary": {
                "total_conversations_processed": self.stats["total_conversations"],
                "conversations_with_features": self.stats["conversations_with_features"],
                "conversations_without_features": self.stats["conversations_without_features"],
                "total_feature_vectors_created": self.stats["total_feature_vectors"],
                "feature_dimensions": self.feature_dimensions,
                "processing_errors": self.stats["processing_errors"],
                "success_rate": (self.stats["conversations_with_features"] / max(self.stats["total_conversations"], 1)) * 100,
                "processing_timestamp": datetime.now().isoformat()
            },
            "source_breakdown": dict(self.stats["source_breakdown"]),
            "category_breakdown": dict(self.stats["category_breakdown"]),
            "output_files": {
                "feature_matrix": "feature_matrix.csv",
                "labels": "labels.csv",
                "metadata": "metadata.csv",
                "ml_conversations": "ml_conversations.jsonl",
                "integration_report": "integration_report.json"
            },
            "ml_applications": {
                "condition_classification": "Multi-class classification of mental health conditions",
                "crisis_detection": "Binary/multi-class crisis risk assessment",
                "population_analysis": "Specialized population identification",
                "temporal_analysis": "Time-series mental health pattern analysis",
                "therapeutic_response": "Automated therapeutic response generation"
            },
            "technical_specifications": {
                "memory_efficient_processing": True,
                "batch_size": self.batch_size,
                "streaming_output": True,
                "enterprise_grade": True,
                "scalable_architecture": True
            }
        }

def process_tfidf_integration():
    """Main function to run TF-IDF integration."""
    processor = TFIDFIntegrationProcessor()
    return processor.process_integration()

if __name__ == "__main__":
    process_tfidf_integration()
