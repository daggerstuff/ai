import os
import sys
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

# Add project root to path
# ai/tests -> ai -> pixelated
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# Add ai/models/components to path specifically for emotion_classifier relative imports
sys.path.insert(0, str(project_root / "ai" / "models" / "components"))

# Mock emotion_classifier module to handle relative imports inside
# emotion_classifier_train
sys.modules["emotion_classifier"] = MagicMock()
# Mock classes inside it
sys.modules["emotion_classifier"].TherapeuticEmotionClassifier = MagicMock()
sys.modules["emotion_classifier"].EmotionClassifierConfig = MagicMock()


class TestRefactoredServices(unittest.TestCase):
    @patch("ai.pipelines.processing.nvidia_clients.NemoRetrieverClient")
    def test_clinical_embedder_uses_nemo(self, mock_client_cls):
        """Verify ClinicalKnowledgeEmbedder initializes NeMo Retriever."""
        from ai.models.pixel_core.data.clinical_knowledge_embedder import (
            ClinicalKnowledgeEmbedder,
        )

        mock_instance = mock_client_cls.return_value
        embedder = ClinicalKnowledgeEmbedder()

        # Check if client was initialized
        self.assertTrue(mock_client_cls.called)
        self.assertEqual(embedder.embedding_model, mock_instance)
        print("✅ ClinicalKnowledgeEmbedder uses NeMo Retriever")

    @patch("ai.pipelines.processing.nvidia_clients.NemoCuratorClient")
    def test_dataset_processor_uses_curator(self, mock_client_cls):
        """Verify DatasetProcessor uses NeMo Curator when enabled."""
        from ai.training.platforms.ovh.process_datasets_chatml import (
            DatasetProcessor,
        )

        # Set env var
        with patch.dict(os.environ, {"USE_NVIDIA_CURATOR": "true"}):
            processor = DatasetProcessor(base_dir="/tmp/test_datasets")
            # Mock internal methods to avoid file I/O
            processor.process_mental_health_counseling = MagicMock(return_value=[])
            processor.process_therapist_sft = MagicMock(return_value=[])
            processor.process_soulchat = MagicMock(return_value=[])
            processor.process_counsel_chat = MagicMock(return_value=[])
            processor.process_psych8k = MagicMock(return_value=[])
            processor.process_all_cot_datasets = MagicMock(return_value=[])
            processor.process_already_processed = MagicMock(return_value=[])
            processor.save_staged_data = MagicMock()
            processor.generate_report = MagicMock()
            processor.save_jsonl = MagicMock()

            processor.run_full_pipeline()

            # Check if client was initialized
            self.assertTrue(mock_client_cls.called)
            print("✅ DatasetProcessor uses NeMo Curator")

    @patch("ai.pipelines.processing.nvidia_clients.NemoCustomizerClient")
    def test_emotion_trainer_uses_customizer(self, mock_client_cls):
        """Verify EmotionClassifierTrainer offloads to NeMo Customizer."""
        from ai.models.components.training.emotion_classifier_train import (
            EmotionClassifierTrainer,
        )

        with patch.dict(os.environ, {"USE_NEMO_CUSTOMIZER": "true"}):
            model = MagicMock()  # Mock the model object directly
            train_loader = MagicMock()
            val_loader = MagicMock()

            mock_config = MagicMock()
            mock_config.learning_rate = 2e-5
            mock_config.num_epochs = 1
            mock_config.valence_weight = 1.0
            mock_config.arousal_weight = 1.0
            mock_config.emotion_weight = 1.0

            # Mock model parameters for optimizer
            import torch

            model.parameters.return_value = [torch.nn.Parameter(torch.randn(1))]

            trainer = EmotionClassifierTrainer(
                model, train_loader, val_loader, config=mock_config
            )

            mock_instance = mock_client_cls.return_value
            mock_instance.train_persona_adapter.return_value = {"id": "test-job"}

            result = trainer.train()

            self.assertTrue(mock_client_cls.called)
            self.assertTrue(mock_instance.train_persona_adapter.called)
            self.assertEqual(result["status"], "dispatched")
            print("✅ EmotionClassifierTrainer uses NeMo Customizer")


if __name__ == "__main__":
    unittest.main()
