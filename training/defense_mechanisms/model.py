import json
import logging
import os
from dataclasses import dataclass, field
from typing import Dict, List, Any

import requests
from dotenv import load_dotenv

from ai.training.defense_mechanisms.constants import DEFENSE_LABELS, DEFENSE_MATURITY

logger = logging.getLogger(__name__)
load_dotenv()


@dataclass
class DefensePrediction:
    """Structured output for a defense mechanism prediction."""

    label: int
    label_name: str
    confidence: float
    probabilities: list[float]
    maturity_score: float | None
    raw_logits: list[float] = field(repr=False, default_factory=list)


class NIMDefenseClassifier:
    """
    NVIDIA NIM-based classifier for defense mechanism detection.
    Replaces local PyTorch/Transformers inference with remote API calls.
    """

    def __init__(
        self, model_name: str = "meta/llama-3.1-8b-instruct", temperature: float = 0.0
    ):
        self.model_name = model_name
        self.temperature = temperature

        self.api_key = (
            os.getenv("NIM_API_KEY")
            or os.getenv("NVIDIA_API_KEY")
            or os.getenv("OPENAI_API_KEY")
        )
        self.base_url = os.getenv(
            "OPENAI_BASE_URL", "https://integrate.api.nvidia.com/v1"
        )

        if not self.api_key:
            raise ValueError(
                "No NVIDIA NIM API key found. Set NIM_API_KEY or NVIDIA_API_KEY."
            )

        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "Accept": "application/json",
        }

    def _build_classification_prompt(self, text: str) -> str:
        """Constructs a strict classification prompt for the NIM LLM."""
        labels_str = "\n".join([f"{k}: {v}" for k, v in DEFENSE_LABELS.items()])
        return f"""You are a clinical psychologist expert in identifying psychological defense mechanisms according to the DMRS (Defense Mechanisms Rating Scales).

Analyze the following clinical utterance and classify it into EXACTLY ONE of the provided defense mechanism categories.

CATEGORIES:
{labels_str}

UTTERANCE:
"{text}"

Output strictly a JSON object with the following structure:
{{
  "label_id": <int>,
  "confidence": <float between 0.0 and 1.0>
}}
Do not output any additional text or markdown formatting. ONLY JSON.
"""

    def predict(self, texts: list[str]) -> list[DefensePrediction]:
        """
        Inference method returning structured prediction objects using NIM.
        """
        predictions = []

        for text in texts:
            prompt = self._build_classification_prompt(text)

            payload = {
                "model": self.model_name,
                "messages": [
                    {
                        "role": "system",
                        "content": "You are a clinical psychology API that outputs strict JSON.",
                    },
                    {"role": "user", "content": prompt},
                ],
                "temperature": self.temperature,
                "max_tokens": 128,
            }

            try:
                # Some embedding models in NIM might need /embeddings, but instruction models like nv-embedqa or nemotron-mini use /chat/completions
                response = requests.post(
                    f"{self.base_url}/chat/completions",
                    headers=self.headers,
                    json=payload,
                    timeout=10,
                )
                response.raise_for_status()
                result = response.json()

                content = result["choices"][0]["message"]["content"].strip()

                # Clean up potential markdown formatting
                if content.startswith("```json"):
                    content = content[7:-3]
                elif content.startswith("```"):
                    content = content[3:-3]

                parsed = json.loads(content)
                pred_label = int(parsed.get("label_id", 0))
                confidence = float(parsed.get("confidence", 0.5))

                # Mock probabilities since LLM doesn't natively output softmax over classes
                probs = [0.0] * len(DEFENSE_LABELS)
                if 0 <= pred_label < len(probs):
                    probs[pred_label] = confidence

                predictions.append(
                    DefensePrediction(
                        label=pred_label,
                        label_name=DEFENSE_LABELS.get(pred_label, "Unknown"),
                        confidence=confidence,
                        probabilities=probs,
                        maturity_score=DEFENSE_MATURITY.get(pred_label),
                        raw_logits=[],
                    )
                )

            except Exception as e:
                logger.error(
                    f"NIM classification failed for text: '{text[:50]}...'. Error: {e}"
                )
                # Fallback to Neutral (0)
                predictions.append(
                    DefensePrediction(
                        label=0,
                        label_name=DEFENSE_LABELS.get(0, "Neutral"),
                        confidence=0.0,
                        probabilities=[1.0] + [0.0] * (len(DEFENSE_LABELS) - 1),
                        maturity_score=DEFENSE_MATURITY.get(0),
                        raw_logits=[],
                    )
                )

        return predictions


# Keep torch dependencies stubbed for compatibility with scripts importing FocalLoss/DefenseClassifier
import torch
import torch.nn as nn


class FocalLoss(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()

    def forward(self, *args, **kwargs):
        return torch.tensor(0.0)


class DefenseClassifier(nn.Module):
    """Stubbed legacy class to prevent import errors in older scripts."""

    def __init__(self, *args, **kwargs):
        super().__init__()
        logger.warning(
            "Local PyTorch DefenseClassifier instantiated. Traffic should route to NIMDefenseClassifier."
        )
