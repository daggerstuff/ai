#!/usr/bin/env python3
"""
TF-IDF Feature-Based Conversation Clustering System for Task 6.16
Clusters conversations using TF-IDF features with 256 dimensions.
"""

import csv
import logging
from collections import defaultdict
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any

import numpy as np

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ClusteringMethod(Enum):
    """Clustering methods available."""
    KMEANS = "kmeans"
    HIERARCHICAL = "hierarchical"
    DBSCAN = "dbscan"
    GAUSSIAN_MIXTURE = "gaussian_mixture"


class SimilarityMetric(Enum):
    """Similarity metrics for clustering."""
    COSINE = "cosine"
    EUCLIDEAN = "euclidean"
    MANHATTAN = "manhattan"
    JACCARD = "jaccard"


@dataclass
class TFIDFFeatures:
    """TF-IDF feature representation."""
    conversation_id: str
    features: np.ndarray  # 256-dimensional TF-IDF features
    metadata: dict[str, Any]
    source_dataset: str
    mental_health_condition: str | None = None


@dataclass
class ConversationCluster:
    """Conversation cluster result."""
    cluster_id: int
    cluster_label: str
    conversations: list[str]
    centroid: np.ndarray
    cluster_size: int
    dominant_topics: list[str]
    mental_health_patterns: list[str]
    similarity_score: float
    cluster_quality_metrics: dict[str, float]


@dataclass
class ClusteringResult:
    """Complete clustering result."""
    method: ClusteringMethod
    num_clusters: int
    clusters: list[ConversationCluster]
    silhouette_score: float
    inertia: float | None
    cluster_quality: float
    feature_importance: dict[str, float]
    clustering_metadata: dict[str, Any]


class TFIDFClusterer:
    """
    TF-IDF feature-based conversation clustering system.
    """

    def __init__(self, feature_dim: int = 256):
        """Initialize the TF-IDF clusterer."""
        self.feature_dim = feature_dim
        self.tfidf_features = {}
        self.feature_vocabulary = {}
        self.mental_health_keywords = self._load_mental_health_keywords()
        self.topic_keywords = self._load_topic_keywords()

        logger.info(f"TFIDFClusterer initialized with {feature_dim} dimensions")

    def _load_mental_health_keywords(self) -> dict[str, list[str]]:
        """Load mental health condition keywords."""
        return {
            "depression": [
                "depressed", "sad", "hopeless", "worthless", "empty", "numb",
                "suicidal", "crying", "tired", "exhausted", "unmotivated"
            ],
            "anxiety": [
                "anxious", "worried", "panic", "fear", "nervous", "stress",
                "overwhelmed", "racing thoughts", "restless", "tense"
            ],
            "bipolar": [
                "manic", "mania", "mood swings", "euphoric", "grandiose",
                "impulsive", "racing", "elevated", "irritable"
            ],
            "ptsd": [
                "trauma", "flashbacks", "nightmares", "triggered", "hypervigilant",
                "avoidance", "intrusive thoughts", "dissociation"
            ],
            "adhd": [
                "attention", "focus", "hyperactive", "impulsive", "distracted",
                "procrastination", "disorganized", "restless"
            ],
            "eating_disorder": [
                "eating", "food", "weight", "body image", "purging", "binge",
                "restriction", "calories", "fat", "thin"
            ],
            "addiction": [
                "addiction", "substance", "alcohol", "drugs", "withdrawal",
                "craving", "relapse", "recovery", "sober", "clean"
            ],
            "ocd": [
                "obsessive", "compulsive", "rituals", "checking", "counting",
                "contamination", "intrusive", "repetitive"
            ]
        }

    def _load_topic_keywords(self) -> dict[str, list[str]]:
        """Load general topic keywords."""
        return {
            "relationships": [
                "relationship", "partner", "boyfriend", "girlfriend", "marriage",
                "divorce", "breakup", "dating", "love", "family"
            ],
            "work_career": [
                "work", "job", "career", "boss", "colleague", "office",
                "promotion", "salary", "unemployment", "interview"
            ],
            "health": [
                "health", "doctor", "medical", "hospital", "medication",
                "therapy", "treatment", "symptoms", "diagnosis"
            ],
            "social": [
                "friends", "social", "lonely", "isolation", "community",
                "support", "group", "people", "conversation"
            ],
            "financial": [
                "money", "financial", "debt", "budget", "income", "expenses",
                "savings", "bills", "poverty", "wealth"
            ],
            "education": [
                "school", "college", "university", "student", "study",
                "exam", "grade", "homework", "learning", "education"
            ]
        }

    def load_tfidf_features(self, feature_files: list[str]) -> int:
        """Load TF-IDF features from CSV files."""
        loaded_count = 0

        for file_path in feature_files:
            try:
                if not Path(file_path).exists():
                    logger.warning(f"Feature file not found: {file_path}")
                    continue

                # Extract metadata from filename
                filename = Path(file_path).stem
                parts = filename.split("_")
                condition = parts[0] if parts else "unknown"
                time_period = parts[1] if len(parts) > 1 else "unknown"

                # Load CSV file
                with open(file_path) as f:
                    reader = csv.reader(f)
                    next(reader)  # Skip header

                    for row_idx, row in enumerate(reader):
                        if len(row) >= self.feature_dim:
                            # Create conversation ID
                            conv_id = f"{condition}_{time_period}_{row_idx}"

                            # Extract features (assuming first 256 columns are TF-IDF features)
                            features = np.array([float(x) for x in row[:self.feature_dim]])

                            # Create TF-IDF feature object
                            tfidf_feature = TFIDFFeatures(
                                conversation_id=conv_id,
                                features=features,
                                metadata={
                                    "source_file": file_path,
                                    "row_index": row_idx,
                                    "time_period": time_period
                                },
                                source_dataset=f"reddit_{condition}",
                                mental_health_condition=condition
                            )

                            self.tfidf_features[conv_id] = tfidf_feature
                            loaded_count += 1

                logger.info(f"Loaded {loaded_count} features from {file_path}")

            except Exception as e:
                logger.error(f"Error loading features from {file_path}: {e}")

        logger.info(f"Total TF-IDF features loaded: {len(self.tfidf_features)}")
        return loaded_count

    def cluster_conversations(self,
                            method: ClusteringMethod = ClusteringMethod.KMEANS,
                            num_clusters: int = 10,
                            similarity_metric: SimilarityMetric = SimilarityMetric.COSINE) -> ClusteringResult:
        """Cluster conversations using TF-IDF features."""
        try:
            if not self.tfidf_features:
                raise ValueError("No TF-IDF features loaded. Call load_tfidf_features() first.")

            # Prepare feature matrix
            feature_matrix, conversation_ids = self._prepare_feature_matrix()

            # Perform clustering
            cluster_labels, clustering_metrics = self._perform_clustering(
                feature_matrix, method, num_clusters, similarity_metric
            )

            # Create cluster objects
            clusters = self._create_clusters(
                cluster_labels, conversation_ids, feature_matrix, method
            )

            # Calculate overall quality metrics
            quality_metrics = self._calculate_quality_metrics(
                feature_matrix, cluster_labels, method
            )

            result = ClusteringResult(
                method=method,
                num_clusters=len(clusters),
                clusters=clusters,
                silhouette_score=quality_metrics.get("silhouette_score", 0.0),
                inertia=quality_metrics.get("inertia"),
                cluster_quality=quality_metrics.get("overall_quality", 0.0),
                feature_importance=self._calculate_feature_importance(clusters),
                clustering_metadata={
                    "total_conversations": len(conversation_ids),
                    "feature_dimensions": self.feature_dim,
                    "similarity_metric": similarity_metric.value,
                    "clustering_parameters": clustering_metrics
                }
            )

            logger.info(f"Clustering completed: {len(clusters)} clusters, "
                       f"silhouette score: {result.silhouette_score:.3f}")

            return result

        except Exception as e:
            logger.error(f"Error in conversation clustering: {e}")
            raise

    def _prepare_feature_matrix(self) -> tuple[np.ndarray, list[str]]:
        """Prepare feature matrix for clustering."""
        conversation_ids = list(self.tfidf_features.keys())
        feature_matrix = np.array([
            self.tfidf_features[conv_id].features
            for conv_id in conversation_ids
        ])

        # Normalize features
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        feature_matrix = scaler.fit_transform(feature_matrix)


    def _perform_clustering(self, feature_matrix: np.ndarray,
                          method: ClusteringMethod,
                          num_clusters: int,
                          similarity_metric: SimilarityMetric) -> tuple[np.ndarray, dict[str, Any]]:
        """Perform the actual clustering."""
        try:
            if method == ClusteringMethod.KMEANS:
                from sklearn.cluster import KMeans
                clusterer = KMeans(n_clusters=num_clusters, random_state=42, n_init=10)
                cluster_labels = clusterer.fit_predict(feature_matrix)
                metrics = {"inertia": clusterer.inertia_, "n_iter": clusterer.n_iter_}

            elif method == ClusteringMethod.HIERARCHICAL:
                from sklearn.cluster import AgglomerativeClustering
                clusterer = AgglomerativeClustering(n_clusters=num_clusters)
                cluster_labels = clusterer.fit_predict(feature_matrix)
                metrics = {"n_clusters": clusterer.n_clusters_}

            elif method == ClusteringMethod.DBSCAN:
                from sklearn.cluster import DBSCAN
                clusterer = DBSCAN(eps=0.5, min_samples=5)
                cluster_labels = clusterer.fit_predict(feature_matrix)
                metrics = {"n_clusters": len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)}

            elif method == ClusteringMethod.GAUSSIAN_MIXTURE:
                from sklearn.mixture import GaussianMixture
                clusterer = GaussianMixture(n_components=num_clusters, random_state=42)
                cluster_labels = clusterer.fit_predict(feature_matrix)
                metrics = {"n_components": clusterer.n_components, "converged": clusterer.converged_}

            else:
                raise ValueError(f"Unsupported clustering method: {method}")

            return cluster_labels, metrics

        except ImportError as e:
            logger.error(f"Required sklearn module not available: {e}")
            # Fallback to simple clustering
            return self._simple_clustering(feature_matrix, num_clusters), {}

    def _simple_clustering(self, feature_matrix: np.ndarray, num_clusters: int) -> np.ndarray:
        """Simple clustering fallback when sklearn is not available."""
        # Simple k-means-like clustering using numpy
        n_samples = feature_matrix.shape[0]

        # Initialize centroids randomly
        centroids = feature_matrix[np.random.choice(n_samples, num_clusters, replace=False)]

        # Simple iterative clustering
        for _ in range(10):  # Max 10 iterations
            # Assign points to nearest centroid
            distances = np.array([
                [np.linalg.norm(point - centroid) for centroid in centroids]
                for point in feature_matrix
            ])
            cluster_labels = np.argmin(distances, axis=1)

            # Update centroids
            new_centroids = np.array([
                feature_matrix[cluster_labels == i].mean(axis=0) if np.any(cluster_labels == i)
                else centroids[i]
                for i in range(num_clusters)
            ])

            # Check convergence
            if np.allclose(centroids, new_centroids):
                break
            centroids = new_centroids

        return cluster_labels

    def _create_clusters(self, cluster_labels: np.ndarray,
                        conversation_ids: list[str],
                        feature_matrix: np.ndarray,
                        method: ClusteringMethod) -> list[ConversationCluster]:
        """Create cluster objects from clustering results."""
        clusters = []
        unique_labels = np.unique(cluster_labels)

        for cluster_id in unique_labels:
            if cluster_id == -1:  # Skip noise points in DBSCAN
                continue

            # Get conversations in this cluster
            cluster_mask = cluster_labels == cluster_id
            cluster_conversations = [conversation_ids[i] for i in np.where(cluster_mask)[0]]
            cluster_features = feature_matrix[cluster_mask]

            # Calculate centroid
            centroid = np.mean(cluster_features, axis=0)

            # Analyze cluster content
            dominant_topics = self._analyze_cluster_topics(cluster_conversations)
            mental_health_patterns = self._analyze_mental_health_patterns(cluster_conversations)

            # Calculate cluster quality
            similarity_score = self._calculate_cluster_similarity(cluster_features)
            quality_metrics = self._calculate_cluster_quality_metrics(cluster_features, centroid)

            cluster = ConversationCluster(
                cluster_id=int(cluster_id),
                cluster_label=f"Cluster_{cluster_id}_{dominant_topics[0] if dominant_topics else 'mixed'}",
                conversations=cluster_conversations,
                centroid=centroid,
                cluster_size=len(cluster_conversations),
                dominant_topics=dominant_topics,
                mental_health_patterns=mental_health_patterns,
                similarity_score=similarity_score,
                cluster_quality_metrics=quality_metrics
            )

            clusters.append(cluster)

        return sorted(clusters, key=lambda x: x.cluster_size, reverse=True)

    def _analyze_cluster_topics(self, conversation_ids: list[str]) -> list[str]:
        """Analyze dominant topics in a cluster."""
        topic_counts = defaultdict(int)

        for conv_id in conversation_ids:
            if conv_id in self.tfidf_features:
                condition = self.tfidf_features[conv_id].mental_health_condition
                if condition:
                    topic_counts[condition] += 1

        # Sort by frequency and return top topics
        sorted_topics = sorted(topic_counts.items(), key=lambda x: x[1], reverse=True)
        return [topic for topic, count in sorted_topics[:3]]

    def _analyze_mental_health_patterns(self, conversation_ids: list[str]) -> list[str]:
        """Analyze mental health patterns in a cluster."""
        pattern_counts = defaultdict(int)

        for conv_id in conversation_ids:
            if conv_id in self.tfidf_features:
                condition = self.tfidf_features[conv_id].mental_health_condition
                if condition and condition in self.mental_health_keywords:
                    for keyword in self.mental_health_keywords[condition]:
                        pattern_counts[keyword] += 1

        # Return top patterns
        sorted_patterns = sorted(pattern_counts.items(), key=lambda x: x[1], reverse=True)
        return [pattern for pattern, count in sorted_patterns[:5]]

    def _calculate_cluster_similarity(self, cluster_features: np.ndarray) -> float:
        """Calculate average similarity within cluster."""
        if len(cluster_features) < 2:
            return 1.0

        # Calculate pairwise cosine similarities
        similarities = []
        for i in range(len(cluster_features)):
            for j in range(i + 1, len(cluster_features)):
                # Cosine similarity
                dot_product = np.dot(cluster_features[i], cluster_features[j])
                norm_i = np.linalg.norm(cluster_features[i])
                norm_j = np.linalg.norm(cluster_features[j])

                if norm_i > 0 and norm_j > 0:
                    similarity = dot_product / (norm_i * norm_j)
                    similarities.append(similarity)

        return np.mean(similarities) if similarities else 0.0

    def _calculate_cluster_quality_metrics(self, cluster_features: np.ndarray,
                                         centroid: np.ndarray) -> dict[str, float]:
        """Calculate quality metrics for a cluster."""
        if len(cluster_features) == 0:
            return {"cohesion": 0.0, "separation": 0.0, "compactness": 0.0}

        # Cohesion: average distance to centroid
        distances_to_centroid = [
            np.linalg.norm(feature - centroid) for feature in cluster_features
        ]
        cohesion = 1.0 / (1.0 + np.mean(distances_to_centroid))  # Inverse distance

        # Compactness: standard deviation of distances
        compactness = 1.0 / (1.0 + np.std(distances_to_centroid))

        return {
            "cohesion": cohesion,
            "compactness": compactness,
            "size": len(cluster_features)
        }

    def _calculate_quality_metrics(self, feature_matrix: np.ndarray,
                                 cluster_labels: np.ndarray,
                                 method: ClusteringMethod) -> dict[str, float]:
        """Calculate overall clustering quality metrics."""
        try:
            from sklearn.metrics import silhouette_score
            silhouette = silhouette_score(feature_matrix, cluster_labels)
        except ImportError:
            # Fallback silhouette calculation
            silhouette = self._simple_silhouette_score(feature_matrix, cluster_labels)

        # Calculate inertia for k-means-like methods
        inertia = None
        if method in [ClusteringMethod.KMEANS, ClusteringMethod.GAUSSIAN_MIXTURE]:
            inertia = self._calculate_inertia(feature_matrix, cluster_labels)

        # Overall quality score
        unique_labels = len(np.unique(cluster_labels))
        cluster_balance = 1.0 - np.std([np.sum(cluster_labels == i) for i in np.unique(cluster_labels)]) / len(cluster_labels)
        overall_quality = (silhouette + cluster_balance) / 2.0

        return {
            "silhouette_score": silhouette,
            "inertia": inertia,
            "overall_quality": overall_quality,
            "num_clusters": unique_labels,
            "cluster_balance": cluster_balance
        }

    def _simple_silhouette_score(self, feature_matrix: np.ndarray,
                                cluster_labels: np.ndarray) -> float:
        """Simple silhouette score calculation."""
        if len(np.unique(cluster_labels)) < 2:
            return 0.0

        silhouette_scores = []

        for i, point in enumerate(feature_matrix):
            same_cluster = cluster_labels == cluster_labels[i]

            # Average distance to points in same cluster
            if np.sum(same_cluster) > 1:
                a = np.mean([np.linalg.norm(point - other)
                           for j, other in enumerate(feature_matrix)
                           if same_cluster[j] and i != j])
            else:
                a = 0

            # Average distance to points in nearest other cluster
            other_clusters = np.unique(cluster_labels[~same_cluster])
            if len(other_clusters) > 0:
                b_scores = []
                for cluster in other_clusters:
                    other_cluster_mask = cluster_labels == cluster
                    b_cluster = np.mean([np.linalg.norm(point - other)
                                       for j, other in enumerate(feature_matrix)
                                       if other_cluster_mask[j]])
                    b_scores.append(b_cluster)
                b = min(b_scores)
            else:
                b = 0

            # Silhouette score for this point
            if max(a, b) > 0:
                silhouette_scores.append((b - a) / max(a, b))
            else:
                silhouette_scores.append(0)

        return np.mean(silhouette_scores)

    def _calculate_inertia(self, feature_matrix: np.ndarray,
                          cluster_labels: np.ndarray) -> float:
        """Calculate inertia (within-cluster sum of squares)."""
        inertia = 0.0

        for cluster_id in np.unique(cluster_labels):
            cluster_mask = cluster_labels == cluster_id
            cluster_points = feature_matrix[cluster_mask]

            if len(cluster_points) > 0:
                centroid = np.mean(cluster_points, axis=0)
                inertia += np.sum([np.linalg.norm(point - centroid) ** 2
                                 for point in cluster_points])

        return inertia

    def _calculate_feature_importance(self, clusters: list[ConversationCluster]) -> dict[str, float]:
        """Calculate feature importance across clusters."""
        # This is a simplified feature importance calculation
        # In practice, you'd use more sophisticated methods

        feature_importance = {}

        # Analyze topic distribution across clusters
        all_topics = set()
        for cluster in clusters:
            all_topics.update(cluster.dominant_topics)

        for topic in all_topics:
            # Calculate how well this topic separates clusters
            topic_distribution = []
            for cluster in clusters:
                topic_count = cluster.dominant_topics.count(topic)
                topic_distribution.append(topic_count / len(cluster.dominant_topics) if cluster.dominant_topics else 0)

            # Higher variance means better separation
            importance = np.var(topic_distribution) if len(topic_distribution) > 1 else 0
            feature_importance[topic] = importance

        return feature_importance

    def find_similar_conversations(self, target_conversation_id: str,
                                 top_k: int = 10) -> list[tuple[str, float]]:
        """Find conversations most similar to target conversation."""
        if target_conversation_id not in self.tfidf_features:
            raise ValueError(f"Conversation {target_conversation_id} not found")

        target_features = self.tfidf_features[target_conversation_id].features
        similarities = []

        for conv_id, tfidf_feature in self.tfidf_features.items():
            if conv_id != target_conversation_id:
                # Calculate cosine similarity
                dot_product = np.dot(target_features, tfidf_feature.features)
                norm_target = np.linalg.norm(target_features)
                norm_other = np.linalg.norm(tfidf_feature.features)

                if norm_target > 0 and norm_other > 0:
                    similarity = dot_product / (norm_target * norm_other)
                    similarities.append((conv_id, similarity))

        # Sort by similarity and return top k
        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:top_k]

    def get_cluster_summary(self, clustering_result: ClusteringResult) -> dict[str, Any]:
        """Get summary of clustering results."""
        return {
            "method": clustering_result.method.value,
            "num_clusters": clustering_result.num_clusters,
            "silhouette_score": clustering_result.silhouette_score,
            "cluster_quality": clustering_result.cluster_quality,
            "clusters": [
                {
                    "cluster_id": cluster.cluster_id,
                    "cluster_label": cluster.cluster_label,
                    "size": cluster.cluster_size,
                    "dominant_topics": cluster.dominant_topics,
                    "mental_health_patterns": cluster.mental_health_patterns,
                    "similarity_score": cluster.similarity_score
                }
                for cluster in clustering_result.clusters
            ],
            "feature_importance": clustering_result.feature_importance,
            "metadata": clustering_result.clustering_metadata
        }


def main():
    """Test the TF-IDF clusterer."""
    clusterer = TFIDFClusterer()

    # Load sample TF-IDF features (you would provide actual file paths)
    sample_files = [
        "/home/vivi/pixelated/ai/datasets/reddit_mental_health/depression_2019_features_tfidf_256.csv",
        "/home/vivi/pixelated/ai/datasets/reddit_mental_health/anxiety_2019_features_tfidf_256.csv"
    ]

    # Load features
    loaded_count = clusterer.load_tfidf_features(sample_files)

    if loaded_count > 0:

        # Perform clustering
        clustering_result = clusterer.cluster_conversations(
            method=ClusteringMethod.KMEANS,
            num_clusters=5,
            similarity_metric=SimilarityMetric.COSINE
        )


        for _cluster in clustering_result.clusters[:3]:  # Show top 3 clusters
            pass

        # Get summary
        clusterer.get_cluster_summary(clustering_result)

    else:
        pass


if __name__ == "__main__":
    main()
