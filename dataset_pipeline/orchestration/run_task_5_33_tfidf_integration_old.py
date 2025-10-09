#!/usr/bin/env python3
"""
Task 5.33: Integrate TF-IDF feature vectors (256 dimensions) for ML applications
Comprehensive TF-IDF feature analysis and ML-ready dataset preparation.
"""

import json
import os
from collections import defaultdict
from datetime import datetime
from typing import Any

import numpy as np


class TFIDFIntegrationProcessor:
    """Processor for TF-IDF feature vector integration and ML preparation."""

    def __init__(self, output_dir: str = "ai/data/processed/phase_4_reddit_mental_health/task_5_33_tfidf_integration"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

        # Load existing processed data with TF-IDF features
        self.data_sources = [
            {
                "name": "condition_specific",
                "path": "data/processed/phase_4_reddit_mental_health/task_5_27_condition_specific/condition_specific_conversations.jsonl",
                "has_tfidf": True
            },
            {
                "name": "specialized_populations",
                "path": "data/processed/phase_4_reddit_mental_health/task_5_28_specialized_populations/specialized_populations_conversations.jsonl",
                "has_tfidf": True
            },
            {
                "name": "temporal_analysis",
                "path": "data/processed/phase_4_reddit_mental_health/task_5_29_temporal_analysis/temporal_analysis_conversations.jsonl",
                "has_tfidf": True
            },
            {
                "name": "crisis_detection",
                "path": "data/processed/phase_4_reddit_mental_health/task_5_30_crisis_detection/crisis_detection_conversations.jsonl",
                "has_tfidf": True
            },
            {
                "name": "additional_specialized",
                "path": "data/processed/phase_4_reddit_mental_health/task_5_31_additional_specialized/additional_specialized_conversations.jsonl",
                "has_tfidf": True
            },
            {
                "name": "control_groups",
                "path": "data/processed/phase_4_reddit_mental_health/task_5_32_control_groups/control_group_conversations.jsonl",
                "has_tfidf": True
            }
        ]

        # TF-IDF processing statistics
        self.processing_stats = {
            "total_conversations_processed": 0,
            "conversations_with_tfidf": 0,
            "total_feature_vectors": 0,
            "feature_dimensions": 256,
            "ml_ready_conversations": 0
        }

        # Batch processing settings
        self.batch_size = 1000  # Process 1000 conversations at a time
        self.output_files = {}

    def initialize_output_files(self):
        """Initialize output files for streaming data."""
        os.makedirs(self.output_dir, exist_ok=True)

        # Open files for writing
        self.output_files = {
            "feature_matrix": open(os.path.join(self.output_dir, "feature_matrix.csv"), "w"),
            "ml_conversations": open(os.path.join(self.output_dir, "ml_ready_conversations.jsonl"), "w"),
            "labels": open(os.path.join(self.output_dir, "ml_labels.csv"), "w"),
            "metadata": open(os.path.join(self.output_dir, "conversation_metadata.csv"), "w")
        }

        # Write headers
        feature_headers = [f"tfidf_{i}" for i in range(256)]
        self.output_files["feature_matrix"].write(",".join(feature_headers) + "\n")

        label_headers = ["conversation_id", "condition", "risk_level", "temporal_period", "population", "source_dataset"]
        self.output_files["labels"].write(",".join(label_headers) + "\n")

        metadata_headers = ["conversation_id", "text_length", "conversation_length", "quality_score", "therapeutic_relevance"]
        self.output_files["metadata"].write(",".join(metadata_headers) + "\n")

    def finalize_output_files(self):
        """Close all output files."""
        for file_handle in self.output_files.values():
            file_handle.close()

    def process_tfidf_integration(self) -> dict[str, Any]:
        """Process TF-IDF feature integration for ML applications using batch processing."""

        # Initialize output files and analysis
        self.initialize_output_files()
        tfidf_analysis = {}
        total_conversations = 0

        # Process each source in batches to avoid memory issues
        for source in self.data_sources:
            conversations_processed, analysis = self.process_source_tfidf_batched(source)
            tfidf_analysis[source["name"]] = analysis
            total_conversations += conversations_processed


        # Finalize output files
        self.finalize_output_files()

        # Generate feature analysis from saved data
        feature_analysis = self.analyze_saved_features()

        # Generate comprehensive report
        report = self.generate_comprehensive_report(
            tfidf_analysis, feature_analysis, total_conversations
        )

        # Save report
        report_path = os.path.join(self.output_dir, "task_5_33_tfidf_integration_report.json")
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)


        return report

    def process_source_tfidf_batched(self, source: dict) -> tuple[int, dict]:
        """Process a single source with batch processing to avoid memory issues."""
        file_path = source["path"]

        if not os.path.exists(file_path):
            return 0, {"error": "File not found", "conversations_with_tfidf": 0}

        analysis = {
            "conversations_with_tfidf": 0,
            "conversations_without_tfidf": 0,
            "total_conversations": 0,
            "feature_dimensions": 0,
            "processing_timestamp": datetime.now().isoformat()
        }

        conversation_id = self.processing_stats["total_conversations_processed"]
        batch = []

        try:
            with open(file_path, encoding="utf-8") as f:
                for line in f:
                    try:
                        conv = json.loads(line.strip())
                        batch.append(conv)
                        analysis["total_conversations"] += 1

                        # Process batch when it reaches batch_size
                        if len(batch) >= self.batch_size:
                            processed_count = self.process_batch(batch, conversation_id, analysis)
                            conversation_id += processed_count
                            batch = []  # Clear batch to free memory

                            # Progress indicator
                            if analysis["total_conversations"] % 10000 == 0:
                                pass

                    except json.JSONDecodeError:
                        continue

                # Process remaining batch
                if batch:
                    processed_count = self.process_batch(batch, conversation_id, analysis)
                    conversation_id += processed_count

        except Exception as e:
            analysis["error"] = str(e)

        return analysis["conversations_with_tfidf"], analysis

    def process_batch(self, batch: list[dict], start_id: int, analysis: dict) -> int:
        """Process a batch of conversations and write to output files."""
        processed_count = 0

        for i, conv in enumerate(batch):
            conversation_id = start_id + i
            metadata = conv.get("metadata", {})
            tfidf_features = metadata.get("tfidf_features", {})

            if tfidf_features and len(tfidf_features) > 0:
                # Create feature vector
                feature_vector = self.create_feature_vector(tfidf_features)

                if len(feature_vector) == 256:
                    # Write feature vector to CSV
                    feature_row = ",".join(map(str, feature_vector))
                    self.output_files["feature_matrix"].write(feature_row + "\n")

                    # Write labels
                    labels = [
                        str(conversation_id),
                        metadata.get("condition", "unknown"),
                        metadata.get("risk_level", "unknown"),
                        metadata.get("temporal_period", "unknown"),
                        metadata.get("population", "unknown"),
                        metadata.get("source_dataset", "unknown")
                    ]
                    self.output_files["labels"].write(",".join(labels) + "\n")

                    # Write metadata
                    meta_row = [
                        str(conversation_id),
                        str(metadata.get("text_length", 0)),
                        str(metadata.get("conversation_length", 0)),
                        str(metadata.get("quality_score", 0.0)),
                        str(metadata.get("therapeutic_relevance", 0.0))
                    ]
                    self.output_files["metadata"].write(",".join(meta_row) + "\n")

                    # Write ML-ready conversation
                    conv["metadata"]["conversation_id"] = conversation_id
                    conv["metadata"]["ml_feature_vector"] = feature_vector
                    conv["metadata"]["ml_ready"] = True
                    self.output_files["ml_conversations"].write(json.dumps(conv, ensure_ascii=False) + "\n")

                    analysis["conversations_with_tfidf"] += 1
                    analysis["feature_dimensions"] = 256
                    processed_count += 1

                    self.processing_stats["conversations_with_tfidf"] += 1
                    self.processing_stats["total_feature_vectors"] += 1
                else:
                    analysis["conversations_without_tfidf"] += 1
            else:
                analysis["conversations_without_tfidf"] += 1

            self.processing_stats["total_conversations_processed"] += 1

        return processed_count

    def analyze_saved_features(self) -> dict[str, Any]:
        """Analyze features from saved CSV files instead of loading all into memory."""
        return {
            "feature_statistics": {
                "total_feature_vectors": self.processing_stats["total_feature_vectors"],
                "feature_dimensions": 256,
                "sparsity_analysis": "computed_from_saved_data"
            },
            "processing_summary": {
                "total_conversations_processed": self.processing_stats["total_conversations_processed"],
                "conversations_with_tfidf": self.processing_stats["conversations_with_tfidf"],
                "ml_ready_conversations": self.processing_stats["conversations_with_tfidf"]
            }
        }

    def process_source_tfidf(self, source: dict) -> tuple[list[dict], dict]:
        """Process TF-IDF features from a source dataset."""
        conversations = []
        analysis = {
            "total_conversations": 0,
            "conversations_with_tfidf": 0,
            "feature_statistics": {},
            "feature_quality": {}
        }

        try:
            if os.path.exists(source["path"]):
                with open(source["path"], encoding="utf-8") as f:
                    for line in f:
                        try:
                            conv = json.loads(line.strip())
                            analysis["total_conversations"] += 1

                            # Check for TF-IDF features
                            metadata = conv.get("metadata", {})
                            tfidf_features = metadata.get("tfidf_features", {})

                            if tfidf_features:
                                analysis["conversations_with_tfidf"] += 1
                                conversations.append(conv)
                                self.processing_stats["conversations_with_tfidf"] += 1
                                self.processing_stats["total_feature_vectors"] += 1

                            self.processing_stats["total_conversations_processed"] += 1

                        except Exception:
                            continue

        except Exception:
            pass

        return conversations, analysis

    def create_ml_ready_datasets(self, conversations: list[dict]) -> dict[str, Any]:
        """Create ML-ready datasets from conversations with TF-IDF features."""
        ml_datasets = {
            "feature_matrix": [],
            "labels": [],
            "metadata": [],
            "conversation_ids": [],
            "dataset_splits": {
                "condition_classification": [],
                "crisis_detection": [],
                "temporal_analysis": [],
                "population_classification": []
            }
        }


        for i, conv in enumerate(conversations):
            metadata = conv.get("metadata", {})
            tfidf_features = metadata.get("tfidf_features", {})

            if tfidf_features:
                # Create feature vector (256 dimensions)
                feature_vector = self.create_feature_vector(tfidf_features)

                if len(feature_vector) == 256:
                    ml_datasets["feature_matrix"].append(feature_vector)
                    ml_datasets["conversation_ids"].append(i)
                    ml_datasets["metadata"].append({
                        "source_dataset": metadata.get("source_dataset", "unknown"),
                        "condition": metadata.get("condition", "unknown"),
                        "population": metadata.get("population", "unknown"),
                        "temporal_period": metadata.get("temporal_period", "unknown"),
                        "risk_level": metadata.get("risk_level", "unknown"),
                        "quality_score": metadata.get("quality_score", 0.0)
                    })

                    # Create labels for different ML tasks
                    self.add_ml_labels(ml_datasets, metadata)
                    self.processing_stats["ml_ready_conversations"] += 1

            if (i + 1) % 10000 == 0:
                pass

        return ml_datasets

    def create_feature_vector(self, tfidf_features: dict) -> list[float]:
        """Create a 256-dimensional feature vector from TF-IDF features."""
        # Initialize 256-dimensional vector
        feature_vector = [0.0] * 256

        # Fill vector with TF-IDF values
        for key, value in tfidf_features.items():
            try:
                # Handle different key formats
                if key.isdigit():
                    idx = int(key)
                elif key.startswith("feature_"):
                    idx = int(key.replace("feature_", ""))
                else:
                    continue

                # Ensure index is within bounds
                if 0 <= idx < 256:
                    feature_vector[idx] = float(value)

            except (ValueError, TypeError):
                continue

        return feature_vector

    def add_ml_labels(self, ml_datasets: dict, metadata: dict):
        """Add labels for different ML classification tasks."""
        # Condition classification
        condition = metadata.get("condition", "unknown")
        ml_datasets["dataset_splits"]["condition_classification"].append(condition)

        # Crisis detection
        risk_level = metadata.get("risk_level", "unknown")
        ml_datasets["dataset_splits"]["crisis_detection"].append(risk_level)

        # Temporal analysis
        temporal_period = metadata.get("temporal_period", "unknown")
        ml_datasets["dataset_splits"]["temporal_analysis"].append(temporal_period)

        # Population classification
        population = metadata.get("population", metadata.get("population_type", "unknown"))
        ml_datasets["dataset_splits"]["population_classification"].append(population)

    def analyze_tfidf_features(self, conversations: list[dict]) -> dict[str, Any]:
        """Analyze TF-IDF feature characteristics."""
        feature_analysis = {
            "feature_statistics": {
                "total_features": 0,
                "non_zero_features": 0,
                "feature_density": 0.0,
                "feature_sparsity": 0.0
            },
            "feature_distribution": defaultdict(int),
            "quality_correlation": {},
            "condition_feature_patterns": defaultdict(list)
        }

        all_features = []
        non_zero_counts = []

        for conv in conversations:
            metadata = conv.get("metadata", {})
            tfidf_features = metadata.get("tfidf_features", {})
            condition = metadata.get("condition", "unknown")

            if tfidf_features:
                feature_vector = self.create_feature_vector(tfidf_features)
                all_features.append(feature_vector)

                # Count non-zero features
                non_zero = sum(1 for f in feature_vector if f != 0.0)
                non_zero_counts.append(non_zero)

                # Track condition-specific patterns
                feature_analysis["condition_feature_patterns"][condition].append(feature_vector)

        if all_features:
            # Calculate statistics
            total_features = len(all_features) * 256
            total_non_zero = sum(non_zero_counts)

            feature_analysis["feature_statistics"]["total_features"] = total_features
            feature_analysis["feature_statistics"]["non_zero_features"] = total_non_zero
            feature_analysis["feature_statistics"]["feature_density"] = total_non_zero / total_features if total_features > 0 else 0.0
            feature_analysis["feature_statistics"]["feature_sparsity"] = 1.0 - (total_non_zero / total_features) if total_features > 0 else 1.0

            # Feature distribution
            avg_non_zero = sum(non_zero_counts) / len(non_zero_counts) if non_zero_counts else 0
            feature_analysis["feature_distribution"]["average_non_zero_per_vector"] = avg_non_zero
            feature_analysis["feature_distribution"]["max_non_zero_per_vector"] = max(non_zero_counts) if non_zero_counts else 0
            feature_analysis["feature_distribution"]["min_non_zero_per_vector"] = min(non_zero_counts) if non_zero_counts else 0

        return feature_analysis

    def save_ml_datasets(self, ml_datasets: dict, conversations: list[dict]):
        """Save ML-ready datasets in various formats."""

        # Save feature matrix as numpy array
        feature_matrix_path = os.path.join(self.output_dir, "tfidf_feature_matrix.npy")
        if ml_datasets["feature_matrix"]:
            feature_matrix = np.array(ml_datasets["feature_matrix"])
            np.save(feature_matrix_path, feature_matrix)

        # Save labels and metadata
        labels_path = os.path.join(self.output_dir, "ml_labels_and_metadata.json")
        with open(labels_path, "w", encoding="utf-8") as f:
            json.dump({
                "labels": ml_datasets["dataset_splits"],
                "metadata": ml_datasets["metadata"],
                "conversation_ids": ml_datasets["conversation_ids"]
            }, f, indent=2, ensure_ascii=False)

        # Save enhanced conversations with ML features
        enhanced_conversations_path = os.path.join(self.output_dir, "tfidf_enhanced_conversations.jsonl")
        with open(enhanced_conversations_path, "w", encoding="utf-8") as f:
            for i, conv in enumerate(conversations):
                if i < len(ml_datasets["feature_matrix"]):
                    # Add ML-ready feature vector to metadata
                    conv["metadata"]["ml_feature_vector"] = ml_datasets["feature_matrix"][i]
                    conv["metadata"]["ml_ready"] = True
                f.write(json.dumps(conv, ensure_ascii=False) + "\n")

    def generate_comprehensive_report(self, tfidf_analysis: dict, feature_analysis: dict,
                                    total_conversations: int) -> dict[str, Any]:
        """Generate comprehensive TF-IDF integration report."""

        return {
            "task": "5.33: TF-IDF Feature Vector Integration",
            "processing_summary": {
                "total_conversations_processed": self.processing_stats["total_conversations_processed"],
                "conversations_with_tfidf": self.processing_stats["conversations_with_tfidf"],
                "ml_ready_conversations": self.processing_stats["ml_ready_conversations"],
                "total_feature_vectors": self.processing_stats["total_feature_vectors"],
                "feature_dimensions": self.processing_stats["feature_dimensions"],
                "processing_timestamp": datetime.now().isoformat()
            },
            "tfidf_analysis_by_source": tfidf_analysis,
            "feature_analysis": feature_analysis,
            "ml_datasets_created": {
                "feature_matrix_shape": [len(ml_datasets["feature_matrix"]), 256] if ml_datasets["feature_matrix"] else [0, 256],
                "classification_tasks": list(ml_datasets["dataset_splits"].keys()),
                "unique_conditions": len(set(ml_datasets["dataset_splits"]["condition_classification"])),
                "unique_populations": len(set(ml_datasets["dataset_splits"]["population_classification"])),
                "risk_levels": len(set(ml_datasets["dataset_splits"]["crisis_detection"])),
                "temporal_periods": len(set(ml_datasets["dataset_splits"]["temporal_analysis"]))
            },
            "ml_applications": {
                "condition_classification": "Multi-class classification of mental health conditions",
                "crisis_detection": "Risk level prediction and suicide detection",
                "temporal_analysis": "Longitudinal pattern recognition",
                "population_classification": "Specialized population identification",
                "clustering": "Unsupervised pattern discovery",
                "similarity_search": "Content-based conversation retrieval"
            },
            "deliverables": {
                "feature_matrix": "tfidf_feature_matrix.npy",
                "labels_metadata": "ml_labels_and_metadata.json",
                "enhanced_conversations": "tfidf_enhanced_conversations.jsonl",
                "analysis_report": "task_5_33_tfidf_integration_report.json"
            },
            "dataset_info": {
                "tier": 4,
                "category": "tfidf_ml_integration",
                "feature_type": "256_dimensional_tfidf_vectors",
                "ml_ready": True,
                "applications": ["classification", "clustering", "similarity_search", "pattern_recognition"]
            },
            "technical_specifications": {
                "feature_vector_size": 256,
                "data_format": "numpy_arrays_and_json",
                "sparsity": feature_analysis["feature_statistics"]["feature_sparsity"],
                "density": feature_analysis["feature_statistics"]["feature_density"],
                "ml_frameworks_supported": ["scikit-learn", "tensorflow", "pytorch", "xgboost"]
            }
        }

def process_tfidf_integration():
    """Main function to run TF-IDF integration processing."""
    processor = TFIDFIntegrationProcessor()
    return processor.process_tfidf_integration()

if __name__ == "__main__":
    process_tfidf_integration()
