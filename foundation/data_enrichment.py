"""
Data enrichment pipeline for therapeutic conversations.

Enriches conversations with:
- Emotion scores (valence/arousal, primary emotion)
- Crisis risk flags
- Bias measurements
- Quality metrics
- Therapeutic technique classification
"""

import json
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional

import torch
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))

from foundation.bias_detection import BiasDetector
from foundation.emotion_recognition import EmotionRecognizer
from models.bias_classifier import BiasDetectionClassifier, BiasDetectionTrainer
from models.conversation_evaluator import (
    ConversationQualityInferencer,
    ConversationQualityModel,
)
from models.emotion_classifier import (
    EmotionClassifierTrainer,
    TherapeuticEmotionClassifier,
)

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class ConversationEnricher:
    """Enriches conversations with multi-model predictions."""

    def __init__(
        self,
        emotion_model_path: Optional[str] = None,
        bias_model_path: Optional[str] = None,
        quality_model_path: Optional[str] = None,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        """
        Initialize enricher with trained models.

        Args:
            emotion_model_path: Path to emotion classifier checkpoint
            bias_model_path: Path to bias detector checkpoint
            quality_model_path: Path to quality evaluator checkpoint
            device: Device to run models on
        """
        self.device = device

        # Initialize emotion classifier
        logger.info("Loading emotion classifier...")
        emotion_model = TherapeuticEmotionClassifier()
        if emotion_model_path:
            checkpoint = torch.load(emotion_model_path, map_location=device)
            emotion_model.load_state_dict(checkpoint["model_state"])
        self.emotion_trainer = EmotionClassifierTrainer(emotion_model, device=device)
        self.emotion_baseline = EmotionRecognizer()

        # Initialize bias detector
        logger.info("Loading bias classifier...")
        bias_model = BiasDetectionClassifier()
        if bias_model_path:
            checkpoint = torch.load(bias_model_path, map_location=device)
            bias_model.load_state_dict(checkpoint["model_state"])
        self.bias_trainer = BiasDetectionTrainer(bias_model, device=device)
        self.bias_baseline = BiasDetector()

        # Initialize quality evaluator
        logger.info("Loading quality evaluator...")
        quality_model = ConversationQualityModel()
        if quality_model_path:
            checkpoint = torch.load(quality_model_path, map_location=device)
            quality_model.load_state_dict(checkpoint["model_state"])
        self.quality_inferencer = ConversationQualityInferencer(
            quality_model, device=device
        )

    def enrich_conversation(self, conversation: Dict) -> Dict:
        """
        Enrich single conversation with predictions.

        Args:
            conversation: Dictionary with 'text' and other fields

        Returns:
            Enriched conversation with new prediction fields
        """
        enriched = conversation.copy()
        text = conversation.get("text", "")

        if not text:
            logger.warning("Empty conversation text")
            return enriched

        # Emotion analysis
        try:
            emotion_pred = self.emotion_trainer.predict(text)
            enriched["emotion_predictions"] = {
                "valence": emotion_pred["valence"],
                "arousal": emotion_pred["arousal"],
                "primary_emotion": emotion_pred["primary_emotion"],
                "emotion_scores": emotion_pred["emotion_scores"],
            }

            # Baseline emotion for comparison
            emotion_baseline = self.emotion_baseline.analyze(text)
            enriched["emotion_baseline"] = emotion_baseline
        except Exception as e:
            logger.warning(f"Emotion analysis failed: {e}")
            enriched["emotion_predictions"] = None

        # Bias detection
        try:
            bias_pred = self.bias_trainer.predict(text)
            enriched["bias_predictions"] = {
                "gender_bias": bias_pred["gender_bias"],
                "gender_confidence": bias_pred["gender_confidence"],
                "racial_bias": bias_pred["racial_bias"],
                "racial_confidence": bias_pred["racial_confidence"],
                "cultural_bias": bias_pred["cultural_bias"],
                "cultural_confidence": bias_pred["cultural_confidence"],
                "overall_bias_score": bias_pred["overall_bias_score"],
            }

            # Baseline bias for comparison
            bias_baseline = self.bias_baseline.detect(text)
            enriched["bias_baseline"] = bias_baseline
        except Exception as e:
            logger.warning(f"Bias detection failed: {e}")
            enriched["bias_predictions"] = None

        # Quality assessment
        try:
            quality_scores = self.quality_inferencer.evaluate(text)
            overall_quality = self.quality_inferencer.compute_overall_score(
                quality_scores
            )
            enriched["quality_scores"] = quality_scores
            enriched["overall_quality"] = overall_quality
        except Exception as e:
            logger.warning(f"Quality assessment failed: {e}")
            enriched["quality_scores"] = None

        return enriched

    def enrich_batch(self, conversations: List[Dict]) -> List[Dict]:
        """Enrich multiple conversations."""
        return [
            self.enrich_conversation(conv)
            for conv in tqdm(conversations, desc="Enriching conversations")
        ]

    def enrich_from_file(
        self,
        input_path: str,
        output_path: str,
    ):
        """Enrich conversations from JSON file."""
        logger.info(f"Loading conversations from {input_path}")
        with open(input_path) as f:
            conversations = json.load(f)

        enriched = self.enrich_batch(conversations)

        logger.info(f"Saving enriched conversations to {output_path}")
        output_dir = Path(output_path).parent
        output_dir.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w") as f:
            json.dump(enriched, f, indent=2)

        logger.info(f"Enrichment complete! Processed {len(enriched)} conversations")


def generate_enrichment_statistics(
    enriched_conversations: List[Dict],
) -> Dict:
    """Generate statistics about enriched data."""
    stats = {
        "total_conversations": len(enriched_conversations),
        "emotion_predictions": {
            "valence": {"mean": 0, "std": 0, "min": 0, "max": 0},
            "arousal": {"mean": 0, "std": 0, "min": 0, "max": 0},
        },
        "bias_predictions": {
            "overall_bias_score": {"mean": 0, "std": 0},
        },
        "quality_scores": {
            dimension: {"mean": 0, "std": 0}
            for dimension in [
                "effectiveness",
                "safety",
                "cultural_competency",
                "coherence",
            ]
        },
        "quality_distribution": {
            "high": 0,  # > 0.7
            "medium": 0,  # 0.4-0.7
            "low": 0,  # < 0.4
        },
    }

    # Extract metrics
    valences = []
    arousals = []
    bias_scores = []
    quality_scores_by_dim = {
        "effectiveness": [],
        "safety": [],
        "cultural_competency": [],
        "coherence": [],
    }
    overall_qualities = []

    for conv in enriched_conversations:
        if conv.get("emotion_predictions"):
            valences.append(conv["emotion_predictions"]["valence"])
            arousals.append(conv["emotion_predictions"]["arousal"])

        if conv.get("bias_predictions"):
            bias_scores.append(conv["bias_predictions"]["overall_bias_score"])

        if conv.get("quality_scores"):
            for dim in quality_scores_by_dim:
                quality_scores_by_dim[dim].append(conv["quality_scores"].get(dim, 0))
            overall_qualities.append(conv.get("overall_quality", 0))

    # Compute statistics
    import numpy as np

    if valences:
        stats["emotion_predictions"]["valence"] = {
            "mean": float(np.mean(valences)),
            "std": float(np.std(valences)),
            "min": float(np.min(valences)),
            "max": float(np.max(valences)),
        }

    if arousals:
        stats["emotion_predictions"]["arousal"] = {
            "mean": float(np.mean(arousals)),
            "std": float(np.std(arousals)),
            "min": float(np.min(arousals)),
            "max": float(np.max(arousals)),
        }

    if bias_scores:
        stats["bias_predictions"]["overall_bias_score"] = {
            "mean": float(np.mean(bias_scores)),
            "std": float(np.std(bias_scores)),
        }

    for dim, scores in quality_scores_by_dim.items():
        if scores:
            stats["quality_scores"][dim] = {
                "mean": float(np.mean(scores)),
                "std": float(np.std(scores)),
            }

    if overall_qualities:
        high = len([q for q in overall_qualities if q > 0.7])
        medium = len([q for q in overall_qualities if 0.4 <= q <= 0.7])
        low = len([q for q in overall_qualities if q < 0.4])

        stats["quality_distribution"] = {
            "high": high,
            "medium": medium,
            "low": low,
        }

    return stats
