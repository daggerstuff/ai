"""
Specialized Dataset Integrator

Integrates specialized mental health datasets:
- TF-IDF feature vectors (256 dimensions) for ML applications
- MODMA-Dataset multi-modal mental disorder analysis
- Original Reddit Data/raw data for custom analysis
"""

import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
from logger import get_logger

logger = get_logger(__name__)


@dataclass
class TFIDFFeature:
    """TF-IDF feature vector data."""

    document_id: str
    feature_vector: list[float]
    vocabulary_terms: list[str]
    document_metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class MODMAEntry:
    """MODMA multi-modal mental disorder entry."""

    entry_id: str
    text_content: str
    audio_features: dict[str, Any] | None = None
    visual_features: dict[str, Any] | None = None
    disorder_labels: list[str] = field(default_factory=list)
    severity_scores: dict[str, float] = field(default_factory=dict)


@dataclass
class RedditEntry:
    """Reddit data entry for mental health analysis."""

    post_id: str
    subreddit: str
    title: str
    content: str
    metadata: dict[str, Any] = field(default_factory=dict)
    mental_health_indicators: list[str] = field(default_factory=list)


class SpecializedDatasetIntegrator:
    """Integrates specialized mental health datasets for ML and analysis applications."""

    def __init__(
        self,
        base_path: str = "./specialized_datasets",
        output_dir: str = "./integrated_datasets",
    ):
        self.base_path = Path(base_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)

        self.logger = get_logger(__name__)

        # Dataset configurations
        self.dataset_configs = {
            "tfidf": {
                "name": "TF-IDF Feature Vectors",
                "description": "256-dimensional TF-IDF vectors for ML applications",
                "dimensions": 256,
                "expected_documents": 1000,
                "file_pattern": "tfidf_*.json",
            },
            "modma": {
                "name": "MODMA-Dataset",
                "description": "Multi-modal mental disorder analysis dataset",
                "modalities": ["text", "audio", "visual"],
                "disorders": [
                    "depression",
                    "anxiety",
                    "bipolar",
                    "schizophrenia",
                    "ptsd",
                ],
                "file_pattern": "modma_*.json",
            },
            "reddit": {
                "name": "Original Reddit Data",
                "description": "Raw Reddit data for custom mental health analysis",
                "subreddits": [
                    "depression",
                    "anxiety",
                    "mentalhealth",
                    "therapy",
                    "bipolar",
                ],
                "file_pattern": "reddit_*.json",
            },
        }

        # Mental health indicators for Reddit analysis
        self.mental_health_keywords = {
            "depression": [
                "depressed",
                "sad",
                "hopeless",
                "worthless",
                "suicidal",
                "empty",
            ],
            "anxiety": [
                "anxious",
                "worried",
                "panic",
                "fear",
                "nervous",
                "overwhelmed",
            ],
            "bipolar": [
                "manic",
                "mood swing",
                "bipolar",
                "episode",
                "mania",
                "hypomanic",
            ],
            "ptsd": [
                "trauma",
                "flashback",
                "nightmare",
                "triggered",
                "ptsd",
                "hypervigilant",
            ],
            "general": [
                "therapy",
                "therapist",
                "medication",
                "mental health",
                "counseling",
            ],
        }

        logger.info("SpecializedDatasetIntegrator initialized")

    def integrate_all_specialized_datasets(self) -> dict[str, Any]:
        """Integrate all specialized datasets."""
        start_time = datetime.now()

        results = {}

        # Integrate each dataset type
        for dataset_key in self.dataset_configs:
            logger.info(f"Integrating {self.dataset_configs[dataset_key]['name']}...")

            if dataset_key == "tfidf":
                result = self.integrate_tfidf_dataset()
            elif dataset_key == "modma":
                result = self.integrate_modma_dataset()
            elif dataset_key == "reddit":
                result = self.integrate_reddit_dataset()
            else:
                result = {
                    "success": False,
                    "issues": [f"Unknown dataset type: {dataset_key}"],
                }

            results[dataset_key] = result

        # Create summary
        return {
            "integration_type": "Specialized Mental Health Datasets",
            "total_datasets": len(self.dataset_configs),
            "successful_integrations": sum(1 for r in results.values() if r["success"]),
            "processing_time": (datetime.now() - start_time).total_seconds(),
            "individual_results": results,
        }


    def integrate_tfidf_dataset(self) -> dict[str, Any]:
        """Integrate TF-IDF feature vectors dataset."""
        config = self.dataset_configs["tfidf"]

        result = {
            "success": False,
            "dataset_name": config["name"],
            "documents_processed": 0,
            "feature_dimensions": 0,
            "quality_metrics": {},
            "issues": [],
            "output_path": None,
        }

        try:
            # Check if dataset exists, create mock if not
            dataset_path = self.base_path / "tfidf_features"
            if not dataset_path.exists():
                self._create_mock_tfidf_data(dataset_path, config)
                result["issues"].append("Created mock TF-IDF data for testing")

            # Load TF-IDF features
            tfidf_features = self._load_tfidf_features(dataset_path)

            # Process features
            processed_features = []
            for feature_data in tfidf_features:
                processed_feature = self._process_tfidf_feature(feature_data, config)
                if processed_feature:
                    processed_features.append(processed_feature)

            # Quality assessment
            quality_metrics = self._assess_tfidf_quality(processed_features, config)

            # Save dataset
            output_path = self._save_tfidf_dataset(
                processed_features, quality_metrics, config
            )

            # Update result
            result.update(
                {
                    "success": True,
                    "documents_processed": len(processed_features),
                    "feature_dimensions": config["dimensions"],
                    "quality_metrics": quality_metrics,
                    "output_path": str(output_path),
                }
            )

            logger.info(
                f"Successfully integrated TF-IDF dataset: {len(processed_features)} documents"
            )

        except Exception as e:
            result["issues"].append(f"TF-IDF integration failed: {e!s}")
            logger.error(f"TF-IDF integration failed: {e}")

        return result

    def integrate_modma_dataset(self) -> dict[str, Any]:
        """Integrate MODMA multi-modal dataset."""
        config = self.dataset_configs["modma"]

        result = {
            "success": False,
            "dataset_name": config["name"],
            "entries_processed": 0,
            "modalities_found": [],
            "disorders_covered": [],
            "quality_metrics": {},
            "issues": [],
            "output_path": None,
        }

        try:
            # Check if dataset exists, create mock if not
            dataset_path = self.base_path / "modma_dataset"
            if not dataset_path.exists():
                self._create_mock_modma_data(dataset_path, config)
                result["issues"].append("Created mock MODMA data for testing")

            # Load MODMA entries
            modma_entries = self._load_modma_entries(dataset_path)

            # Process entries
            processed_entries = []
            for entry_data in modma_entries:
                processed_entry = self._process_modma_entry(entry_data, config)
                if processed_entry:
                    processed_entries.append(processed_entry)

            # Analyze modalities and disorders
            modalities_found = set()
            disorders_covered = set()
            for entry in processed_entries:
                if entry.audio_features:
                    modalities_found.add("audio")
                if entry.visual_features:
                    modalities_found.add("visual")
                if entry.text_content:
                    modalities_found.add("text")
                disorders_covered.update(entry.disorder_labels)

            # Quality assessment
            quality_metrics = self._assess_modma_quality(processed_entries, config)

            # Save dataset
            output_path = self._save_modma_dataset(
                processed_entries, quality_metrics, config
            )

            # Update result
            result.update(
                {
                    "success": True,
                    "entries_processed": len(processed_entries),
                    "modalities_found": list(modalities_found),
                    "disorders_covered": list(disorders_covered),
                    "quality_metrics": quality_metrics,
                    "output_path": str(output_path),
                }
            )

            logger.info(
                f"Successfully integrated MODMA dataset: {len(processed_entries)} entries"
            )

        except Exception as e:
            result["issues"].append(f"MODMA integration failed: {e!s}")
            logger.error(f"MODMA integration failed: {e}")

        return result

    def integrate_reddit_dataset(self) -> dict[str, Any]:
        """Integrate Reddit raw data dataset."""
        config = self.dataset_configs["reddit"]

        result = {
            "success": False,
            "dataset_name": config["name"],
            "posts_processed": 0,
            "subreddits_found": [],
            "mental_health_indicators": {},
            "quality_metrics": {},
            "issues": [],
            "output_path": None,
        }

        try:
            # Check if dataset exists, create mock if not
            dataset_path = self.base_path / "reddit_data"
            if not dataset_path.exists():
                self._create_mock_reddit_data(dataset_path, config)
                result["issues"].append("Created mock Reddit data for testing")

            # Load Reddit entries
            reddit_entries = self._load_reddit_entries(dataset_path)

            # Process entries
            processed_entries = []
            for entry_data in reddit_entries:
                processed_entry = self._process_reddit_entry(entry_data, config)
                if processed_entry:
                    processed_entries.append(processed_entry)

            # Analyze subreddits and indicators
            subreddits_found = list({entry.subreddit for entry in processed_entries})

            # Count mental health indicators
            indicator_counts = {}
            for category, keywords in self.mental_health_keywords.items():
                indicator_counts[category] = sum(
                    1
                    for entry in processed_entries
                    if any(
                        keyword in entry.mental_health_indicators
                        for keyword in keywords
                    )
                )

            # Quality assessment
            quality_metrics = self._assess_reddit_quality(processed_entries, config)

            # Save dataset
            output_path = self._save_reddit_dataset(
                processed_entries, quality_metrics, config
            )

            # Update result
            result.update(
                {
                    "success": True,
                    "posts_processed": len(processed_entries),
                    "subreddits_found": subreddits_found,
                    "mental_health_indicators": indicator_counts,
                    "quality_metrics": quality_metrics,
                    "output_path": str(output_path),
                }
            )

            logger.info(
                f"Successfully integrated Reddit dataset: {len(processed_entries)} posts"
            )

        except Exception as e:
            result["issues"].append(f"Reddit integration failed: {e!s}")
            logger.error(f"Reddit integration failed: {e}")

        return result

    def _create_mock_tfidf_data(self, dataset_path: Path, config: dict[str, Any]):
        """Create mock TF-IDF data."""
        dataset_path.mkdir(parents=True, exist_ok=True)

        # Generate mock vocabulary
        vocabulary = [
            "anxiety",
            "depression",
            "therapy",
            "mental",
            "health",
            "stress",
            "mood",
            "emotion",
            "treatment",
            "medication",
            "counseling",
            "support",
            "help",
            "feeling",
            "thoughts",
            "symptoms",
            "disorder",
            "psychological",
            "cognitive",
            "behavioral",
            "therapeutic",
        ] + [
            f"term_{i}" for i in range(235)
        ]  # Total 256 terms

        # Generate mock documents with TF-IDF vectors
        documents = []
        for i in range(config["expected_documents"]):
            # Generate random TF-IDF vector
            vector = np.random.random(config["dimensions"]).tolist()

            document = {
                "document_id": f"doc_{i:04d}",
                "tfidf_vector": vector,
                "vocabulary": vocabulary,
                "metadata": {
                    "source": "mental_health_corpus",
                    "document_length": np.random.randint(50, 500),
                    "topic": ["anxiety", "depression", "therapy", "general"][i % 4],
                    "created_at": datetime.now().isoformat(),
                },
            }
            documents.append(document)

        # Save documents
        with open(dataset_path / "tfidf_vectors.jsonl", "w") as f:
            for doc in documents:
                f.write(json.dumps(doc) + "\n")

        # Save vocabulary
        with open(dataset_path / "vocabulary.json", "w") as f:
            json.dump(
                {"vocabulary": vocabulary, "dimensions": config["dimensions"]},
                f,
                indent=2,
            )

    def _create_mock_modma_data(self, dataset_path: Path, config: dict[str, Any]):
        """Create mock MODMA data."""
        dataset_path.mkdir(parents=True, exist_ok=True)

        entries = []
        disorders = config["disorders"]

        for i in range(200):  # 200 mock entries
            entry = {
                "entry_id": f"modma_{i:04d}",
                "text_content": f"Patient exhibits symptoms consistent with {disorders[i % len(disorders)]}. Mood appears {['low', 'elevated', 'stable', 'variable'][i % 4]}.",
                "audio_features": (
                    {
                        "mfcc": np.random.random(13).tolist(),
                        "pitch_mean": float(np.random.normal(150, 50)),
                        "energy": float(np.random.random()),
                        "speaking_rate": float(np.random.normal(150, 30)),
                    }
                    if i % 3 == 0
                    else None
                ),  # Audio for 1/3 of entries
                "visual_features": (
                    {
                        "facial_expressions": {
                            "happiness": float(np.random.random()),
                            "sadness": float(np.random.random()),
                            "anger": float(np.random.random()),
                            "fear": float(np.random.random()),
                        },
                        "eye_contact": float(np.random.random()),
                        "posture_score": float(np.random.random()),
                    }
                    if i % 2 == 0
                    else None
                ),  # Visual for 1/2 of entries
                "disorder_labels": [disorders[i % len(disorders)]],
                "severity_scores": {
                    disorders[i % len(disorders)]: float(np.random.uniform(0.3, 0.9))
                },
                "metadata": {
                    "session_id": f"session_{i // 10}",
                    "timestamp": datetime.now().isoformat(),
                    "clinician_id": f"clinician_{i % 5}",
                },
            }
            entries.append(entry)

        # Save entries
        with open(dataset_path / "modma_entries.jsonl", "w") as f:
            for entry in entries:
                f.write(json.dumps(entry) + "\n")

    def _create_mock_reddit_data(self, dataset_path: Path, config: dict[str, Any]):
        """Create mock Reddit data."""
        dataset_path.mkdir(parents=True, exist_ok=True)

        posts = []
        subreddits = config["subreddits"]

        sample_posts = [
            (
                "I've been feeling really down lately",
                "I can't seem to shake this feeling of sadness. Everything feels overwhelming.",
            ),
            (
                "Anxiety is taking over my life",
                "I have panic attacks almost daily now. I don't know what to do.",
            ),
            (
                "Therapy has been helping",
                "Started seeing a therapist last month and it's making a difference.",
            ),
            (
                "Medication side effects",
                "Anyone else experience side effects from antidepressants?",
            ),
            (
                "Support group recommendations",
                "Looking for a good support group in my area.",
            ),
        ]

        for i in range(500):  # 500 mock posts
            title, content = sample_posts[i % len(sample_posts)]

            post = {
                "post_id": f"reddit_{i:04d}",
                "subreddit": subreddits[i % len(subreddits)],
                "title": f"{title} - variation {i}",
                "content": f"{content} Additional context for post {i}.",
                "metadata": {
                    "upvotes": np.random.randint(0, 100),
                    "comments": np.random.randint(0, 50),
                    "created_utc": datetime.now().timestamp(),
                    "author": f"user_{i % 100}",
                },
            }
            posts.append(post)

        # Save posts
        with open(dataset_path / "reddit_posts.jsonl", "w") as f:
            for post in posts:
                f.write(json.dumps(post) + "\n")

    def _load_tfidf_features(self, dataset_path: Path) -> list[dict[str, Any]]:
        """Load TF-IDF features."""
        features = []
        data_file = dataset_path / "tfidf_vectors.jsonl"

        if data_file.exists():
            with open(data_file) as f:
                for line in f:
                    if line.strip():
                        features.append(json.loads(line))

        return features

    def _load_modma_entries(self, dataset_path: Path) -> list[dict[str, Any]]:
        """Load MODMA entries."""
        entries = []
        data_file = dataset_path / "modma_entries.jsonl"

        if data_file.exists():
            with open(data_file) as f:
                for line in f:
                    if line.strip():
                        entries.append(json.loads(line))

        return entries

    def _load_reddit_entries(self, dataset_path: Path) -> list[dict[str, Any]]:
        """Load Reddit entries."""
        entries = []
        data_file = dataset_path / "reddit_posts.jsonl"

        if data_file.exists():
            with open(data_file) as f:
                for line in f:
                    if line.strip():
                        entries.append(json.loads(line))

        return entries

    def _process_tfidf_feature(
        self, feature_data: dict[str, Any], config: dict[str, Any]
    ) -> TFIDFFeature | None:
        """Process TF-IDF feature."""
        try:
            return TFIDFFeature(
                document_id=feature_data["document_id"],
                feature_vector=feature_data["tfidf_vector"],
                vocabulary_terms=feature_data["vocabulary"],
                document_metadata=feature_data.get("metadata", {}),
            )
        except Exception as e:
            logger.error(f"Error processing TF-IDF feature: {e}")
            return None

    def _process_modma_entry(
        self, entry_data: dict[str, Any], config: dict[str, Any]
    ) -> MODMAEntry | None:
        """Process MODMA entry."""
        try:
            return MODMAEntry(
                entry_id=entry_data["entry_id"],
                text_content=entry_data["text_content"],
                audio_features=entry_data.get("audio_features"),
                visual_features=entry_data.get("visual_features"),
                disorder_labels=entry_data.get("disorder_labels", []),
                severity_scores=entry_data.get("severity_scores", {}),
            )
        except Exception as e:
            logger.error(f"Error processing MODMA entry: {e}")
            return None

    def _process_reddit_entry(
        self, entry_data: dict[str, Any], config: dict[str, Any]
    ) -> RedditEntry | None:
        """Process Reddit entry."""
        try:
            # Extract mental health indicators
            text_content = f"{entry_data['title']} {entry_data['content']}".lower()
            indicators = []

            for _category, keywords in self.mental_health_keywords.items():
                for keyword in keywords:
                    if keyword in text_content:
                        indicators.append(keyword)

            return RedditEntry(
                post_id=entry_data["post_id"],
                subreddit=entry_data["subreddit"],
                title=entry_data["title"],
                content=entry_data["content"],
                metadata=entry_data.get("metadata", {}),
                mental_health_indicators=list(set(indicators)),
            )
        except Exception as e:
            logger.error(f"Error processing Reddit entry: {e}")
            return None

    def _assess_tfidf_quality(
        self, features: list[TFIDFFeature], config: dict[str, Any]
    ) -> dict[str, float]:
        """Assess TF-IDF dataset quality."""
        if not features:
            return {"overall_quality": 0.0}

        # Check vector dimensions
        correct_dimensions = sum(
            1 for f in features if len(f.feature_vector) == config["dimensions"]
        )
        dimension_accuracy = correct_dimensions / len(features)

        # Check for non-zero vectors
        non_zero_vectors = sum(
            1 for f in features if any(v > 0 for v in f.feature_vector)
        )
        non_zero_rate = non_zero_vectors / len(features)

        return {
            "overall_quality": (dimension_accuracy + non_zero_rate) / 2,
            "dimension_accuracy": dimension_accuracy,
            "non_zero_rate": non_zero_rate,
            "total_documents": len(features),
        }

    def _assess_modma_quality(
        self, entries: list[MODMAEntry], config: dict[str, Any]
    ) -> dict[str, float]:
        """Assess MODMA dataset quality."""
        if not entries:
            return {"overall_quality": 0.0}

        # Count modalities
        text_count = sum(1 for e in entries if e.text_content)
        audio_count = sum(1 for e in entries if e.audio_features)
        visual_count = sum(1 for e in entries if e.visual_features)

        # Multi-modal coverage
        multimodal_count = sum(
            1
            for e in entries
            if sum(
                [bool(e.text_content), bool(e.audio_features), bool(e.visual_features)]
            )
            >= 2
        )

        return {
            "overall_quality": multimodal_count / len(entries),
            "text_coverage": text_count / len(entries),
            "audio_coverage": audio_count / len(entries),
            "visual_coverage": visual_count / len(entries),
            "multimodal_rate": multimodal_count / len(entries),
        }

    def _assess_reddit_quality(
        self, entries: list[RedditEntry], config: dict[str, Any]
    ) -> dict[str, float]:
        """Assess Reddit dataset quality."""
        if not entries:
            return {"overall_quality": 0.0}

        # Mental health indicator coverage
        with_indicators = sum(1 for e in entries if e.mental_health_indicators)
        indicator_rate = with_indicators / len(entries)

        # Content quality (non-empty posts)
        quality_posts = sum(1 for e in entries if len(e.content.split()) >= 5)
        content_quality = quality_posts / len(entries)

        return {
            "overall_quality": (indicator_rate + content_quality) / 2,
            "mental_health_indicator_rate": indicator_rate,
            "content_quality": content_quality,
            "total_posts": len(entries),
        }

    def _save_tfidf_dataset(
        self,
        features: list[TFIDFFeature],
        quality_metrics: dict[str, float],
        config: dict[str, Any],
    ) -> Path:
        """Save TF-IDF dataset."""
        output_file = self.output_dir / "tfidf_features_integrated.json"

        output_data = {
            "dataset_info": {
                "name": config["name"],
                "description": config["description"],
                "dimensions": config["dimensions"],
                "total_documents": len(features),
                "integrated_at": datetime.now().isoformat(),
            },
            "quality_metrics": quality_metrics,
            "features": [
                {
                    "document_id": f.document_id,
                    "feature_vector": f.feature_vector,
                    "vocabulary_size": len(f.vocabulary_terms),
                    "metadata": f.document_metadata,
                }
                for f in features
            ],
        }

        with open(output_file, "w") as f:
            json.dump(output_data, f, indent=2)

        return output_file

    def _save_modma_dataset(
        self,
        entries: list[MODMAEntry],
        quality_metrics: dict[str, float],
        config: dict[str, Any],
    ) -> Path:
        """Save MODMA dataset."""
        output_file = self.output_dir / "modma_dataset_integrated.json"

        output_data = {
            "dataset_info": {
                "name": config["name"],
                "description": config["description"],
                "modalities": config["modalities"],
                "disorders": config["disorders"],
                "total_entries": len(entries),
                "integrated_at": datetime.now().isoformat(),
            },
            "quality_metrics": quality_metrics,
            "entries": [
                {
                    "entry_id": e.entry_id,
                    "text_content": e.text_content,
                    "has_audio": bool(e.audio_features),
                    "has_visual": bool(e.visual_features),
                    "audio_features": e.audio_features,
                    "visual_features": e.visual_features,
                    "disorder_labels": e.disorder_labels,
                    "severity_scores": e.severity_scores,
                }
                for e in entries
            ],
        }

        with open(output_file, "w") as f:
            json.dump(output_data, f, indent=2)

        return output_file

    def _save_reddit_dataset(
        self,
        entries: list[RedditEntry],
        quality_metrics: dict[str, float],
        config: dict[str, Any],
    ) -> Path:
        """Save Reddit dataset."""
        output_file = self.output_dir / "reddit_data_integrated.json"

        output_data = {
            "dataset_info": {
                "name": config["name"],
                "description": config["description"],
                "subreddits": config["subreddits"],
                "total_posts": len(entries),
                "integrated_at": datetime.now().isoformat(),
            },
            "quality_metrics": quality_metrics,
            "mental_health_keywords": self.mental_health_keywords,
            "entries": [
                {
                    "post_id": e.post_id,
                    "subreddit": e.subreddit,
                    "title": e.title,
                    "content": e.content,
                    "mental_health_indicators": e.mental_health_indicators,
                    "metadata": e.metadata,
                }
                for e in entries
            ],
        }

        with open(output_file, "w") as f:
            json.dump(output_data, f, indent=2)

        return output_file


# Example usage
if __name__ == "__main__":
    # Initialize integrator
    integrator = SpecializedDatasetIntegrator()

    # Integrate all specialized datasets
    results = integrator.integrate_all_specialized_datasets()

    # Show results

    for _dataset_key, result in results["individual_results"].items():
        status = "✅ Success" if result["success"] else "❌ Failed"
        if result["success"]:
            if "documents_processed" in result or "entries_processed" in result or "posts_processed" in result:
                pass
        else:
            pass
