import sys
import unittest
from pathlib import Path
from unittest.mock import patch

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from ai.pipelines.orchestrator.processing.feedback_loop_orchestrator import (  # noqa: E402
    FeedbackLoopOrchestrator,
)
from ai.pipelines.orchestrator.processing.nvidia_clients import (  # noqa: E402
    NemoCuratorClient,
    NemoCustomizerClient,
    NemoEvaluatorClient,
    NemoRetrieverClient,
)


class TestAdvancedNvidiaUpgrades(unittest.TestCase):
    """Verifies the 12 advanced 'outside the box' upgrades."""

    @patch("requests.post")
    def test_curator_advanced(self, mock_post):
        mock_post.return_value.json.return_value = {"status": "ok", "result": "test"}
        client = NemoCuratorClient()

        client.identify_emotional_gaps("path/to/data")
        client.detect_crisis_narratives("I feel hopeless")
        client.filter_cultural_bias("path/to/data")

        self.assertEqual(mock_post.call_count, 3)
        print("✅ Curator Advanced Methods Verified")

    @patch("requests.post")
    def test_retriever_advanced(self, mock_post):
        mock_post.return_value.json.return_value = {
            "data": [{"embedding": [0.1] * 1024}],
            "results": [],
        }
        client = NemoRetrieverClient()

        client.temporal_context_search("user_1", "query")
        client.safety_constrained_rerank("query", ["doc1"])
        client.tri_persona_search("query", ["doc1"])

        # tri_persona calls safety, dual_persona (which calls rerank twice)
        # So we expect multiple posts.
        self.assertTrue(mock_post.call_count > 3)
        print("✅ Retriever Advanced Methods Verified")

    @patch("requests.post")
    def test_customizer_advanced(self, mock_post):
        mock_post.return_value.json.return_value = {"id": "job_123"}
        client = NemoCustomizerClient()

        client.resonance_optimal_tuning("model_1", [0.9, 0.8])
        client.distill_therapeutic_essence("teacher_1")
        client.merge_persona_weights(["a1", "a2"], [0.5, 0.5])

        self.assertEqual(mock_post.call_count, 3)
        print("✅ Customizer Advanced Methods Verified")

    @patch("requests.post")
    def test_evaluator_advanced(self, mock_post):
        mock_post.return_value.json.return_value = {
            "score": 0.95,
            "drift_detected": False,
        }
        client = NemoEvaluatorClient()

        client.measure_empathic_resonance("user", "bot")
        client.detect_therapeutic_drift(["msg1"], "persona1")
        client.longitudinal_impact_score([0.8, 0.9])

        self.assertEqual(mock_post.call_count, 3)
        print("✅ Evaluator Advanced Methods Verified")

    @patch(
        "ai.pipelines.orchestrator.processing.nvidia_clients"
        ".NemoEvaluatorClient.measure_empathic_resonance"
    )
    @patch(
        "ai.pipelines.orchestrator.processing.nvidia_clients"
        ".NemoCustomizerClient.resonance_optimal_tuning"
    )
    def test_feedback_loop_orchestrator(self, mock_tune, mock_res):
        mock_res.return_value = {"score": 0.85}
        mock_tune.return_value = {"id": "opt_job_1"}

        orch = FeedbackLoopOrchestrator()
        samples = [{"user": "u", "bot": "b"}]
        result = orch.optimize_training_resonance("model_1", samples)

        self.assertEqual(result["id"], "opt_job_1")
        self.assertTrue(mock_res.called)
        self.assertTrue(mock_tune.called)
        print("✅ FeedbackLoopOrchestrator Verified")

    @patch("ai.pipelines.orchestrator.processing.nvidia_clients.NemoRetrieverClient")
    def test_youtube_rag_integration(self, mock_client_cls):
        """Verify YouTubeRAGSystem integrates advanced retriever methods."""
        import os
        from unittest.mock import MagicMock

        from ai.pipelines.orchestrator.youtube_rag_system import YouTubeRAGSystem

        with patch.dict(os.environ, {"USE_NVIDIA_RETRIEVER": "true"}):
            system = YouTubeRAGSystem()
            # Mock index and embeddings
            mock_entry = MagicMock()
            mock_entry.content = "Therapeutic content"
            mock_entry.embedding = [0.1] * 1024
            system.rag_index = [mock_entry]

            mock_instance = mock_client_cls.return_value
            mock_instance.tri_persona_search.return_value = {"emotional": []}
            mock_instance.safety_constrained_rerank.return_value = [
                {"text": "Therapeutic content", "relevance_score": 0.9}
            ]
            mock_instance.get_embedding.return_value = [0.1] * 1024

            results = system.search_transcripts("I need help", top_k=1)

            self.assertTrue(mock_instance.tri_persona_search.called)
            self.assertTrue(mock_instance.safety_constrained_rerank.called)
            self.assertEqual(len(results), 1)
            self.assertTrue(results[0]["metadata"]["safety_reranked"])
            print("✅ YouTubeRAGSystem Advanced Integration Verified")


if __name__ == "__main__":
    unittest.main()
