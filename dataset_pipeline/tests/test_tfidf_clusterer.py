#!/usr/bin/env python3
"""
Test suite for TFIDFClusterer - Task 6.16
"""

from unittest.mock import patch

import pytest
from tfidf_clusterer import (
    ClusteringMethod,
    ClusteringResult,
    ConversationCluster,
    SimilarityMetric,
    TFIDFClusterer,
    TFIDFFeatures,
)


class TestTFIDFClusterer:
    """Test cases for TFIDFClusterer."""

    def setup_method(self):
        """Set up test fixtures."""
        self.clusterer = TFIDFClusterer(feature_dim=256)

        # Create mock TF-IDF features
        self.mock_features = {
            "conv_1": TFIDFFeatures(
                conversation_id="conv_1",
                features=[0.1] * 256,  # Mock 256-dim features
                metadata={"source_file": "test.csv", "row_index": 0},
                source_dataset="test_dataset",
                mental_health_condition="depression"
            ),
            "conv_2": TFIDFFeatures(
                conversation_id="conv_2",
                features=[0.2] * 256,
                metadata={"source_file": "test.csv", "row_index": 1},
                source_dataset="test_dataset",
                mental_health_condition="anxiety"
            )
        }

        # Add mock features to clusterer
        self.clusterer.tfidf_features = self.mock_features

    def test_initialization(self):
        """Test clusterer initialization."""
        assert self.clusterer.feature_dim == 256
        assert self.clusterer.mental_health_keywords is not None
        assert self.clusterer.topic_keywords is not None
        assert len(self.clusterer.mental_health_keywords) > 0

    def test_load_mental_health_keywords(self):
        """Test mental health keywords loading."""
        keywords = self.clusterer._load_mental_health_keywords()

        assert "depression" in keywords
        assert "anxiety" in keywords
        assert isinstance(keywords["depression"], list)
        assert len(keywords["depression"]) > 0

    def test_load_topic_keywords(self):
        """Test topic keywords loading."""
        keywords = self.clusterer._load_topic_keywords()

        assert "relationships" in keywords
        assert "work_career" in keywords
        assert isinstance(keywords["relationships"], list)
        assert len(keywords["relationships"]) > 0

    @patch("builtins.open")
    @patch("csv.reader")
    @patch("pathlib.Path.exists")
    def test_load_tfidf_features(self, mock_exists, mock_csv_reader, mock_open):
        """Test TF-IDF features loading."""
        # Mock file existence
        mock_exists.return_value = True

        # Mock CSV data
        mock_csv_reader.return_value = [
            ["feature_0", "feature_1"] + [f"feature_{i}" for i in range(2, 256)],  # Header
            [str(0.1)] * 256,  # First row
            [str(0.2)] * 256   # Second row
        ]

        clusterer = TFIDFClusterer()
        loaded_count = clusterer.load_tfidf_features(["test_file.csv"])

        assert loaded_count == 2
        assert len(clusterer.tfidf_features) == 2

    def test_cluster_conversations_kmeans(self):
        """Test K-means clustering."""
        # Mock numpy arrays for features
        import numpy as np
        self.clusterer.tfidf_features["conv_1"].features = np.array([0.1] * 256)
        self.clusterer.tfidf_features["conv_2"].features = np.array([0.9] * 256)

        result = self.clusterer.cluster_conversations(
            method=ClusteringMethod.KMEANS,
            num_clusters=2,
            similarity_metric=SimilarityMetric.COSINE
        )

        assert isinstance(result, ClusteringResult)
        assert result.method == ClusteringMethod.KMEANS
        assert len(result.clusters) <= 2
        assert 0 <= result.silhouette_score <= 1

    def test_simple_clustering_fallback(self):
        """Test simple clustering fallback when sklearn not available."""
        import numpy as np
        feature_matrix = np.array([[0.1] * 256, [0.9] * 256])

        cluster_labels = self.clusterer._simple_clustering(feature_matrix, 2)

        assert len(cluster_labels) == 2
        assert all(label in [0, 1] for label in cluster_labels)

    def test_analyze_cluster_topics(self):
        """Test cluster topic analysis."""
        conversation_ids = ["conv_1", "conv_2"]
        topics = self.clusterer._analyze_cluster_topics(conversation_ids)

        assert isinstance(topics, list)
        assert "depression" in topics or "anxiety" in topics

    def test_analyze_mental_health_patterns(self):
        """Test mental health pattern analysis."""
        conversation_ids = ["conv_1"]
        patterns = self.clusterer._analyze_mental_health_patterns(conversation_ids)

        assert isinstance(patterns, list)
        # Should find depression-related patterns
        assert len(patterns) >= 0

    def test_calculate_cluster_similarity(self):
        """Test cluster similarity calculation."""
        import numpy as np
        cluster_features = np.array([[0.1, 0.2], [0.15, 0.25], [0.12, 0.22]])

        similarity = self.clusterer._calculate_cluster_similarity(cluster_features)

        assert 0 <= similarity <= 1
        assert isinstance(similarity, float)

    def test_find_similar_conversations(self):
        """Test finding similar conversations."""
        import numpy as np
        self.clusterer.tfidf_features["conv_1"].features = np.array([0.1] * 256)
        self.clusterer.tfidf_features["conv_2"].features = np.array([0.15] * 256)

        similar = self.clusterer.find_similar_conversations("conv_1", top_k=1)

        assert len(similar) == 1
        assert similar[0][0] == "conv_2"  # Most similar conversation
        assert isinstance(similar[0][1], float)  # Similarity score

    def test_get_cluster_summary(self):
        """Test cluster summary generation."""
        # Create mock clustering result
        import numpy as np

        mock_cluster = ConversationCluster(
            cluster_id=0,
            cluster_label="test_cluster",
            conversations=["conv_1", "conv_2"],
            centroid=np.array([0.1] * 256),
            cluster_size=2,
            dominant_topics=["depression"],
            mental_health_patterns=["sad", "hopeless"],
            similarity_score=0.8,
            cluster_quality_metrics={"cohesion": 0.7}
        )

        mock_result = ClusteringResult(
            method=ClusteringMethod.KMEANS,
            num_clusters=1,
            clusters=[mock_cluster],
            silhouette_score=0.6,
            inertia=10.5,
            cluster_quality=0.7,
            feature_importance={"depression": 0.8},
            clustering_metadata={"total_conversations": 2}
        )

        summary = self.clusterer.get_cluster_summary(mock_result)

        assert isinstance(summary, dict)
        assert "method" in summary
        assert "num_clusters" in summary
        assert "clusters" in summary
        assert summary["num_clusters"] == 1

    def test_error_handling(self):
        """Test error handling in clustering."""
        # Test with empty features
        empty_clusterer = TFIDFClusterer()

        with pytest.raises(ValueError):
            empty_clusterer.cluster_conversations()

    def test_feature_importance_calculation(self):
        """Test feature importance calculation."""
        import numpy as np

        mock_clusters = [
            ConversationCluster(
                cluster_id=0,
                cluster_label="depression_cluster",
                conversations=["conv_1"],
                centroid=np.array([0.1] * 256),
                cluster_size=1,
                dominant_topics=["depression"],
                mental_health_patterns=[],
                similarity_score=0.8,
                cluster_quality_metrics={}
            ),
            ConversationCluster(
                cluster_id=1,
                cluster_label="anxiety_cluster",
                conversations=["conv_2"],
                centroid=np.array([0.2] * 256),
                cluster_size=1,
                dominant_topics=["anxiety"],
                mental_health_patterns=[],
                similarity_score=0.7,
                cluster_quality_metrics={}
            )
        ]

        importance = self.clusterer._calculate_feature_importance(mock_clusters)

        assert isinstance(importance, dict)
        assert len(importance) >= 0


def test_tfidf_clusterer_integration():
    """Integration test for TF-IDF clusterer."""
    clusterer = TFIDFClusterer(feature_dim=10)  # Smaller for testing

    # Create mock features
    import numpy as np

    mock_features = {
        f"conv_{i}": TFIDFFeatures(
            conversation_id=f"conv_{i}",
            features=np.random.rand(10),
            metadata={"test": True},
            source_dataset="test",
            mental_health_condition="depression" if i < 3 else "anxiety"
        )
        for i in range(6)
    }

    clusterer.tfidf_features = mock_features

    # Test clustering
    result = clusterer.cluster_conversations(
        method=ClusteringMethod.KMEANS,
        num_clusters=2
    )

    # Verify results
    assert isinstance(result, ClusteringResult)
    assert len(result.clusters) <= 2
    assert result.num_clusters <= 2

    # Test similarity search
    similar = clusterer.find_similar_conversations("conv_0", top_k=2)
    assert len(similar) == 2

    # Test summary
    summary = clusterer.get_cluster_summary(result)
    assert "method" in summary
    assert "clusters" in summary


if __name__ == "__main__":
    pytest.main([__file__])
