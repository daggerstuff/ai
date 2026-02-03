import logging
import os
import unittest
from unittest.mock import MagicMock, patch

import pandas as pd

# Set environment variables BEFORE importing modules that check them at module level
os.environ["USE_NVIDIA_CURATOR"] = "true"
os.environ["USE_NVIDIA_RETRIEVER"] = "true"
os.environ["USE_NEMO_CUSTOMIZER"] = "true"
os.environ["USE_NVIDIA_EVALUATOR"] = "true"
os.environ["NVIDIA_API_KEY"] = "test_key"

# Import the components
from ai.models.components.therapeutic_finetuning import (
    TherapeuticFinetunConfig,
    TherapeuticModelTrainer,
)
from ai.pipelines.orchestrator.evaluation_system import ComprehensiveEvaluator
from ai.pipelines.orchestrator.processing.clean import clean_and_deduplicate
from ai.pipelines.orchestrator.youtube_rag_system import YouTubeRAGSystem

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TestNvidiaMicroservicesIntegration(unittest.TestCase):
    """
    Master integration test for NVIDIA Microservices.
    """

    @patch("ai.pipelines.orchestrator.processing.clean.NemoCuratorClient")
    def test_curator_integration(self, mock_client):
        """Verify that clean_and_deduplicate attempts to use NeMo Curator."""
        mock_instance = mock_client.return_value
        df = pd.DataFrame({"text": ["Hello"]})
        clean_and_deduplicate(df)
        self.assertTrue(mock_instance.curate_therapeutic_data.called)
        logger.info("✅ NeMo Curator personalized integration verified.")

    @patch("ai.pipelines.orchestrator.youtube_rag_system.NemoRetrieverClient")
    @patch(
        "ai.pipelines.orchestrator.youtube_rag_system.SentenceTransformer", create=True
    )
    def test_retriever_integration(self, mock_st, mock_client):
        """Verify that YouTubeRAGSystem attempts to use NeMo Retriever."""
        mock_instance = mock_client.return_value
        # Mock Path to exist
        with patch("pathlib.Path.exists", return_value=True):
            with patch("pathlib.Path.mkdir"):
                rag = YouTubeRAGSystem(model_name="test-model")
                # Mock rag_index to prevent early return
                rag.rag_index = [MagicMock()]
                self.assertTrue(rag.use_nvidia)
                rag.search_transcripts("depression query")
                self.assertTrue(mock_instance.dual_persona_search.called)
        logger.info("✅ NeMo Retriever personalized integration verified.")

    @patch("ai.models.components.therapeutic_finetuning.NemoCustomizerClient")
    @patch(
        "ai.models.components.therapeutic_finetuning.TherapeuticModelTrainer._load_tokenizer"
    )
    @patch(
        "ai.models.components.therapeutic_finetuning.TherapeuticModelTrainer._load_model"
    )
    def test_customizer_integration(self, mock_model, mock_tok, mock_client):
        """Verify that TherapeuticModelTrainer attempts to use NeMo Customizer."""
        mock_instance = mock_client.return_value
        mock_instance.train_persona_adapter.return_value = {"id": "job_123"}

        # Test with persona
        config = TherapeuticFinetunConfig(personality_persona="dark_humor")
        trainer = TherapeuticModelTrainer(config=config)
        result = trainer.train(train_conversations=[{"messages": []}])

        self.assertEqual(result["status"], "dispatched")
        self.assertTrue(mock_instance.train_persona_adapter.called)
        logger.info("✅ NeMo Customizer personalized integration verified.")

    @patch("ai.pipelines.orchestrator.evaluation_system.NemoEvaluatorClient")
    @patch("ai.pipelines.orchestrator.evaluation_system.AccuracyEvaluator", create=True)
    @patch("ai.pipelines.orchestrator.evaluation_system.SafetyEvaluator", create=True)
    @patch("ai.pipelines.orchestrator.evaluation_system.FairnessEvaluator", create=True)
    @patch(
        "ai.pipelines.orchestrator.evaluation_system.TherapeuticResponseEvaluator",
        create=True,
    )
    def test_evaluator_integration(self, m1, m2, m3, m4, mock_client):
        """Verify that ComprehensiveEvaluator attempts to use NeMo Evaluator."""
        mock_instance = mock_client.return_value
        evaluator = ComprehensiveEvaluator()
        with patch(
            "ai.pipelines.orchestrator.evaluation_system.USE_NVIDIA_EVALUATOR", True
        ):
            evaluator.evaluate_model(model=None, tokenizer=None, dataset=MagicMock())
        self.assertTrue(mock_instance.evaluate_therapeutic_alignment.called)
        logger.info("✅ NeMo Evaluator personalized integration verified.")


if __name__ == "__main__":
    unittest.main()
