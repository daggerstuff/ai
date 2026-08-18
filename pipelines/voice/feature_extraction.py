import json
import logging
from pathlib import Path
from typing import Any, cast

import numpy as np
import torch
import transformers
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("FeatureExtraction")


def _redacted(text: str, max_chars: int = 24) -> str:
    """Return a short non-sensitive preview of therapeutic text for logs.

    Full transcript content is sensitive (PHI). Logs must never contain it.
    """
    stripped = text.strip().replace("\n", " ")
    if len(stripped) <= max_chars:
        return stripped
    return stripped[:max_chars] + "..."


class FeatureExtractor:
    def __init__(self, device: int = 0 if torch.cuda.is_available() else -1):
        logger.info(f"Loading emotion classification model on device {device}...")
        pipeline_func = cast(Any, transformers.pipeline)
        self.emotion_classifier = pipeline_func(
            "text-classification", model="SamLowe/roberta-base-go_emotions", device=device
        )

        logger.info(f"Loading empathy scoring model on device {device}...")
        # Using zero-shot classification for empathy as a proxy
        self.empathy_classifier = pipeline_func(
            "zero-shot-classification", model="facebook/bart-large-mnli", device=device
        )

    def extract_rhythm(self, segment: dict[str, Any]) -> float:
        """Calculate words per second as a proxy for speech rhythm."""
        duration = segment.get("end", 0.0) - segment.get("start", 0.0)
        if duration <= 0:
            return 0.0
        word_count = len(segment.get("text", "").split())
        return word_count / duration

    def process_segments(self, segments: list[dict[str, Any]]) -> list[dict[str, Any]]:
        enriched = []
        for seg in segments:
            text = seg.get("text", "")
            if not text.strip():
                continue

            # Emotion extraction
            try:
                emo_result = self.emotion_classifier(text)
                emotion = emo_result[0]["label"]
                emo_score = emo_result[0]["score"]
            except Exception as e:
                logger.warning(f"Emotion classification failed for segment (len={len(text)}): {e}")
                emotion = "neutral"
                emo_score = 0.0

            # Empathy scoring
            try:
                emp_result = cast(
                    dict[str, Any],
                    self.empathy_classifier(
                        text, candidate_labels=["empathetic", "neutral", "dismissive"]
                    ),
                )
                labels_list = cast(list[str], emp_result["labels"])
                scores_list = cast(list[float], emp_result["scores"])
                emp_score = scores_list[labels_list.index("empathetic")]
            except Exception as e:
                logger.warning(
                    f"Empathy classification failed for segment (len={len(text)}, preview={_redacted(text)}): {e}"
                )
                emp_score = 0.0

            # Rhythm extraction
            rhythm = self.extract_rhythm(seg)

            new_seg = dict(seg)
            new_seg["features"] = {
                "emotion": emotion,
                "emotion_score": float(emo_score),
                "empathy_score": float(emp_score),
                "rhythm_wps": float(rhythm),
            }
            enriched.append(new_seg)
        return enriched


class PersonalityClusterer:
    def __init__(self, n_clusters: int = 5):
        self.n_clusters = n_clusters
        self.scaler = StandardScaler()
        self.kmeans = KMeans(n_clusters=self.n_clusters, random_state=42, n_init="auto")
        self.reference_centroids: np.ndarray | None = None

    def _extract_feature_vector(self, segment: dict[str, Any]) -> list[float]:
        feats = segment.get("features", {})
        # We use numeric features for clustering: empathy score, rhythm (WPS), and emotion confidence score
        # In a more advanced version, we could encode categorical emotions using one-hot encoding.
        return [
            feats.get("empathy_score", 0.0),
            feats.get("rhythm_wps", 0.0),
            feats.get("emotion_score", 0.0),
        ]

    def fit_predict(self, segments: list[dict[str, Any]]) -> list[int]:
        if len(segments) < self.n_clusters:
            logger.warning(
                f"Not enough segments to cluster. Found {len(segments)}, need at least {self.n_clusters}."
            )
            return [-1] * len(segments)

        x_data = np.array([self._extract_feature_vector(s) for s in segments])
        x_scaled = self.scaler.fit_transform(x_data)

        labels = self.kmeans.fit_predict(x_scaled)

        if self.reference_centroids is None:
            self.reference_centroids = self.kmeans.cluster_centers_

        return labels.tolist()

    def detect_drift(self) -> float:
        if self.reference_centroids is None or not hasattr(self.kmeans, "cluster_centers_"):
            return 0.0

        current_centroids = self.kmeans.cluster_centers_
        # Compute drift as the norm of the difference between sorted centroids
        # Sorting is a naive way to align clusters if labels shifted, though bipartite matching is better
        drift = np.linalg.norm(
            np.sort(current_centroids, axis=0) - np.sort(self.reference_centroids, axis=0)
        )

        if drift > 1.0:  # Arbitrary threshold for flagging significant drift
            logger.warning(f"Significant cluster drift detected! Score: {drift:.2f}")

        self.reference_centroids = current_centroids
        return float(drift)


class FeatureExtractionPipeline:
    def __init__(
        self, input_dir: str = "ai/data/transcripts", output_dir: str = "ai/data/features"
    ):
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.extractor = FeatureExtractor()
        self.clusterer = PersonalityClusterer(n_clusters=5)

    def process_all(self) -> None:
        if not self.input_dir.exists():
            logger.error(f"Input directory {self.input_dir} does not exist.")
            return

        transcript_files = list(self.input_dir.glob("*.json"))
        logger.info(f"Found {len(transcript_files)} transcripts to process for features.")

        for file_path in transcript_files:
            out_file = self.output_dir / f"{file_path.stem}_features.json"
            if out_file.exists():
                logger.info(f"Skipping {file_path}, features already extracted.")
                continue

            try:
                with open(file_path, encoding="utf-8") as f:
                    segments = json.load(f)

                logger.info(f"Extracting features for {file_path.name}...")
                enriched_segments = self.extractor.process_segments(segments)

                logger.info(f"Clustering {len(enriched_segments)} segments...")
                labels = self.clusterer.fit_predict(enriched_segments)

                drift_score = self.clusterer.detect_drift()

                # Attach cluster labels back to segments
                for seg, label in zip(enriched_segments, labels, strict=False):
                    seg["features"]["cluster_id"] = int(label)

                output_data = {
                    "file": file_path.name,
                    "drift_score": drift_score,
                    "segments": enriched_segments,
                }

                with open(out_file, "w", encoding="utf-8") as f:
                    json.dump(output_data, f, indent=2)

                logger.info(f"Saved feature-enriched segments to {out_file}")

            except Exception as e:
                logger.error(f"Failed to process {file_path}: {e}")


if __name__ == "__main__":
    pipeline = FeatureExtractionPipeline()
    pipeline.process_all()
