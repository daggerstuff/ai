import logging
import os
from datetime import datetime
from typing import Any, Dict, List, Optional

import requests

logger = logging.getLogger(__name__)


class NvidiaBaseClient:
    """Base client for NVIDIA NeMo Microservices."""

    def __init__(self, base_url: str, api_key: Optional[str] = None):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key or os.getenv("NVIDIA_API_KEY")
        self.headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}" if self.api_key else "",
        }

    def _post(self, endpoint: str, data: Dict[str, Any]) -> Dict[str, Any]:
        url = f"{self.base_url}/{endpoint.lstrip('/')}"
        try:
            response = requests.post(url, headers=self.headers, json=data)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            logger.error(f"Error calling {url}: {e}")
            if hasattr(e, "response") and e.response is not None:
                logger.error(f"Response: {e.response.text}")
            raise


class NemoCuratorClient(NvidiaBaseClient):
    """Client for NeMo Curator tailored for Pixelated Empathy."""

    def __init__(
        self, base_url: str = os.getenv("NEMO_CURATOR_URL", "http://localhost:8000")
    ):
        super().__init__(base_url)

    def process_dataset(
        self, dataset_path: str, operations: List[str]
    ) -> Dict[str, Any]:
        """Triggers a curation job on a dataset."""
        payload = {
            "dataset_path": dataset_path,
            "operations": operations,
            "config": {"dedup": {"threshold": 0.8}, "toxicity": {"min_score": 0.5}},
        }
        return self._post("/v1/curate", payload)

    def curate_therapeutic_data(self, dataset_path: str) -> Dict[str, Any]:
        """Specific curation for Pixelated Empathy therapeutic datasets."""
        # Custom operations for empathy and therapeutic alignment
        ops = ["dedup", "toxicity", "empathy_alignment", "crisis_safety_check"]
        payload = {
            "dataset_path": dataset_path,
            "operations": ops,
            "config": {
                "empathy_alignment": {"threshold": 0.75},
                "crisis_safety_check": {"sensitivity": "high"},
            },
        }
        return self._post("/v1/curate", payload)


class NemoRetrieverClient(NvidiaBaseClient):
    """Client for NVIDIA NIM Retriever with Dual Persona support."""

    def __init__(
        self,
        embedding_url: str = "http://localhost:8080",
        rerank_url: str = "http://localhost:8081",
    ):
        self.embedding_client = NvidiaBaseClient(embedding_url)
        self.rerank_client = NvidiaBaseClient(rerank_url)

    def get_embedding(self, text: str, model: str = "nv-embedqa-e5-v5") -> List[float]:
        payload = {"input": text, "model": model, "input_type": "query"}
        response = self.embedding_client._post("/v1/embeddings", payload)
        return response["data"][0]["embedding"]

    def rerank(
        self, query: str, documents: List[str], top_n: int = 5
    ) -> List[Dict[str, Any]]:
        payload = {"query": query, "documents": documents, "top_n": top_n}
        return self.rerank_client._post("/v1/reranking", payload)

    def dual_persona_search(
        self, query: str, documents: List[str], top_k: int = 5
    ) -> Dict[str, Any]:
        """
        Search optimized for Dual Persona (System 1: Emotional, System 2: Reasoning).
        Returns context for both emotional mirroring and evidence-based reasoning.
        """
        # System 1: Focus on emotional valence and empathy
        emotional_docs = self.rerank(
            f"emotional resonance for: {query}", documents=documents, top_n=top_k
        )
        # System 2: Focus on therapeutic evidence (CBT, DBT, etc.)
        reasoning_docs = self.rerank(
            f"therapeutic techniques and evidence for: {query}",
            documents=documents,
            top_n=top_k,
        )
        return {
            "emotional_context": emotional_docs,
            "reasoning_context": reasoning_docs,
        }


class NemoCustomizerClient(NvidiaBaseClient):
    """Client for NeMo Customizer with Persona-aware training."""

    def __init__(
        self, base_url: str = os.getenv("NEMO_CUSTOMIZER_URL", "http://localhost:8001")
    ):
        super().__init__(base_url)

    def start_finetuning(
        self, job_name: str, dataset_id: str, model_name: str
    ) -> Dict[str, Any]:
        payload = {
            "name": job_name,
            "dataset_id": dataset_id,
            "model": model_name,
            "training_type": "sft",
            "parameters": {"epochs": 3, "batch_size": 8, "learning_rate": 2e-4},
        }
        return self._post("/v1/jobs", payload)

    def train_persona_adapter(
        self, persona_name: str, dataset_id: str
    ) -> Dict[str, Any]:
        """Triggers training for a specific therapeutic persona (e.g. 'Dark Humor')."""
        job_name = f"persona_{persona_name}_{datetime.now().strftime('%Y%m%d')}"
        payload = {
            "name": job_name,
            "dataset_id": dataset_id,
            "training_type": "lora",
            "parameters": {
                "rank": 16,
                "alpha": 32,
                "target_modules": ["q_proj", "v_proj", "k_proj", "o_proj"],
            },
            "metadata": {"persona": persona_name, "project": "Pixelated Empathy"},
        }
        return self._post("/v1/jobs", payload)


class NemoEvaluatorClient(NvidiaBaseClient):
    """Client for NeMo Evaluator with Empathy Benchmark."""

    def __init__(
        self, base_url: str = os.getenv("NEMO_EVALUATOR_URL", "http://localhost:7331")
    ):
        super().__init__(base_url)

    def evaluate(
        self, predictions: List[str], references: List[str], metrics: List[str]
    ) -> Dict[str, Any]:
        payload = {
            "predictions": predictions,
            "references": references,
            "metrics": metrics,
        }
        return self._post("/v1/evaluate", payload)

    def evaluate_therapeutic_alignment(
        self, predictions: List[str], references: List[str]
    ) -> Dict[str, Any]:
        """Custom LLM-as-a-judge evaluation for empathy and therapeutic goals."""
        return self.evaluate(
            predictions,
            references,
            metrics=["empathy_scale", "safety_violation", "dark_humor_accuracy"],
        )
