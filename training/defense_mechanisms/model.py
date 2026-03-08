import logging
import os
from dataclasses import dataclass, field
from typing import List

import numpy as np
import requests
import torch
import torch.nn as nn
from ai.training.defense_mechanisms.constants import DEFENSE_LABELS, DEFENSE_MATURITY
from dotenv import load_dotenv

logger = logging.getLogger(__name__)
load_dotenv()


@dataclass
class DefensePrediction:
    """Structured output for a defense mechanism prediction."""

    label: int
    label_name: str
    confidence: float
    probabilities: List[float]
    maturity_score: float | None
    raw_logits: List[float] = field(repr=False, default_factory=list)


class NIMEmbeddingClassifier:
    """
    NVIDIA NIM Embedding-based classifier for defense mechanism detection.
    Uses 'nvidia/nv-embedqa-e5-v5' for high-precision vector similarity classification.
    """

    def __init__(self, model_name: str = "nvidia/nv-embedqa-e5-v5"):
        self.model_name = model_name
        self.api_key = (
            os.getenv("NIM_API_KEY")
            or os.getenv("NVIDIA_API_KEY")
            or os.getenv("OPENAI_API_KEY")
        )
        self.base_url = os.getenv(
            "OPENAI_BASE_URL", "https://integrate.api.nvidia.com/v1"
        )

        if not self.api_key:
            raise ValueError("No NVIDIA NIM API key found. Set NVIDIA_API_KEY.")

        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "Accept": "application/json",
        }

        self._prototypes = None
        self.session = requests.Session()
        self.session.headers.update(self.headers)

        self.proto_descriptions = {
            0: "A neutral, objective communication without defensive posturing.",
            1: "Denial: Refusal to acknowledge a painful reality or obvious fact.",
            2: "Projection: Attributing one's own unacceptable feelings to others.",
            3: "Passive-Aggression: Indirect expression of hostility.",
            4: "Acting Out: Expressing internal conflicts through physical actions.",
            5: "Splitting: Viewing situations as good or bad without nuance.",
            6: "Displacement: Redirection of an impulse onto a substitute target.",
            7: "Rationalization: Creating logical explanations for deeper motives.",
            8: "Intellectualization: Excessive abstract thinking to avoid feelings.",
        }

    def _post_request(self, endpoint: str, payload: dict, max_retries: int = 7) -> dict:
        """Execute a POST request to the NIM API with exponential backoff."""
        import random
        import time

        url = f"{self.base_url}/{endpoint}"

        for attempt in range(max_retries):
            try:
                response = self.session.post(
                    url,
                    json=payload,
                    timeout=30,
                )
                if response.status_code == 429:
                    sleep_time = (2**attempt) + random.uniform(0.1, 1.0)
                    logger.warning(
                        f"NIM API 429 Too Many Requests. Sleeping {sleep_time:.2f}s"
                    )
                    time.sleep(sleep_time)
                    continue

                if not response.ok:
                    logger.error(f"NIM {endpoint} API payload: {payload}")
                    logger.error(f"NIM {endpoint} API response: {response.text}")
                response.raise_for_status()
                return response.json()
            except requests.exceptions.RequestException as e:
                if attempt == max_retries - 1:
                    logger.error(f"NIM {endpoint} API call failed permanently: {e}")
                    raise
                sleep_time = (2**attempt) + random.uniform(0.1, 1.0)
                time.sleep(sleep_time)

    def _embed(self, texts: List[str], input_type: str = "query") -> np.ndarray:
        """Fetch embeddings from NVIDIA NIM."""
        payload = {
            "model": self.model_name,
            "input": texts,
            "input_type": input_type,
            "encoding_format": "float",
        }

        data = self._post_request("embeddings", payload)
        embeddings = [
            item["embedding"] for item in sorted(data["data"], key=lambda x: x["index"])
        ]
        return np.array(embeddings)

    def _get_prototypes(self) -> np.ndarray:
        """Lazy load and cache the embedding prototypes for labels."""
        if self._prototypes is None:
            logger.info(f"Generating label prototypes using {self.model_name}...")
            texts = [self.proto_descriptions[i] for i in range(len(DEFENSE_LABELS))]
            self._prototypes = self._embed(texts, input_type="query")
        return self._prototypes

    def predict(self, texts: List[str]) -> List[DefensePrediction]:
        """Classify texts via cosine similarity in embedding space."""
        if not texts:
            return []

        # Filter out empty strings to avoid 400 Bad Request
        valid_texts = [t for t in texts if t and t.strip()]

        # If all texts were empty, return default predictions
        if not valid_texts:
            return [
                DefensePrediction(
                    label=0,
                    label_name=DEFENSE_LABELS.get(0, "Neutral"),
                    confidence=0.0,
                    probabilities=[1.0] + [0.0] * (len(DEFENSE_LABELS) - 1),
                    maturity_score=DEFENSE_MATURITY.get(0),
                )
                for _ in texts
            ]

        try:
            prototypes = self._get_prototypes()
            embeddings = self._embed(valid_texts, input_type="query")

            # Vectorized Cosine Similarity
            prototypes_norm = prototypes / np.linalg.norm(
                prototypes, axis=1, keepdims=True
            )
            embeddings_norm = embeddings / np.linalg.norm(
                embeddings, axis=1, keepdims=True
            )
            similarities = np.dot(embeddings_norm, prototypes_norm.T)

            predictions = []
            for sim in similarities:
                # Softmax temperature scaling for confidence
                exp_sim = np.exp(sim * 20.0)
                probs = (exp_sim / np.sum(exp_sim)).tolist()

                pred_label = int(np.argmax(sim))
                confidence = float(np.max(probs))

                predictions.append(
                    DefensePrediction(
                        label=pred_label,
                        label_name=DEFENSE_LABELS.get(pred_label, "Unknown"),
                        confidence=confidence,
                        probabilities=probs,
                        maturity_score=DEFENSE_MATURITY.get(pred_label),
                        raw_logits=sim.tolist(),
                    )
                )
            return predictions

        except Exception as e:
            logger.error(f"Classification pipeline failed: {e}")
            return [
                DefensePrediction(
                    label=0,
                    label_name=DEFENSE_LABELS.get(0, "Neutral"),
                    confidence=0.0,
                    probabilities=[1.0] + [0.0] * (len(DEFENSE_LABELS) - 1),
                    maturity_score=DEFENSE_MATURITY.get(0),
                )
                for _ in texts
            ]


class DefenseClassifier(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.nim = NIMEmbeddingClassifier()

    def predict(self, texts: List[str]) -> List[DefensePrediction]:
        return self.nim.predict(texts)


class FocalLoss(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()

    def forward(self, *args, **kwargs):
        return torch.tensor(0.0)
