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

    def identify_emotional_gaps(self, dataset_path: str) -> Dict[str, Any]:
        """Analyzes dataset to find under-represented emotional archetypes."""
        return self._post("/v1/analysis/emotional_coverage", {"path": dataset_path})

    def detect_crisis_narratives(self, text: str) -> Dict[str, Any]:
        """Detects subtle indicators of crisis beyond simple toxicity."""
        return self._post("/v1/detect/crisis", {"text": text, "sensitivity": "ultra"})

    def filter_cultural_bias(self, dataset_path: str) -> Dict[str, Any]:
        """Ensures diverse empathy expressions across cultural dimensions."""
        return self._post("/v1/filter/cultural_diversity", {"path": dataset_path})


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

    def temporal_context_search(
        self, user_id: str, query: str, top_k: int = 5
    ) -> Dict[str, Any]:
        """Retrieves context weighted by chronological therapeutic progression."""
        payload = {"user_id": user_id, "query": query, "strategy": "temporal_decay"}
        return self.embedding_client._post("/v1/temporal_search", payload)

    def safety_constrained_rerank(
        self, query: str, documents: List[str], top_n: int = 5
    ) -> List[Dict[str, Any]]:
        """Reranks while force-injecting clinical safety guidelines."""
        payload = {"query": query, "documents": documents, "force_safety": True}
        return self.rerank_client._post("/v1/reranking/safety", payload)

    def tri_persona_search(
        self, query: str, documents: List[str], top_k: int = 5
    ) -> Dict[str, Any]:
        """
        System 1 (Emotional), System 2 (Reasoning),
        and System 3 (Grounding/Safety).
        """
        base = self.dual_persona_search(query, documents, top_k)
        grounding_docs = self.rerank(
            f"physical grounding and clinical safety for: {query}",
            documents=documents,
            top_n=top_k,
        )
        base["grounding_context"] = grounding_docs
        return base


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

    def resonance_optimal_tuning(
        self, model_id: str, resonance_scores: List[float]
    ) -> Dict[str, Any]:
        """Auto-adjusts LoRA based on evaluator resonance feedback."""
        return self._post(
            "/v1/optimize/resonance", {"id": model_id, "scores": resonance_scores}
        )

    def distill_therapeutic_essence(self, teacher_id: str) -> Dict[str, Any]:
        """Distills clinical expert model into a fast micro-adapter."""
        return self._post(
            "/v1/distill", {"teacher": teacher_id, "target_type": "micro"}
        )

    def merge_persona_weights(
        self, adapter_ids: List[str], weights: List[float]
    ) -> Dict[str, Any]:
        """Dynamically merges persona adapters (e.g. Warmth + CBT)."""
        return self._post(
            "/v1/adapters/merge", {"ids": adapter_ids, "weights": weights}
        )


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

    def measure_empathic_resonance(
        self, user_utterance: str, bot_response: str
    ) -> Dict[str, Any]:
        """Quantifies emotional 'match' vs 'therapeutic detachment'."""
        return self._post(
            "/v1/measure/resonance", {"input": user_utterance, "output": bot_response}
        )

    def detect_therapeutic_drift(
        self, session_history: List[str], persona: str
    ) -> Dict[str, Any]:
        """Monitors if bot is drifting away from intended therapeutic framework."""
        return self._post(
            "/v1/detect/drift", {"history": session_history, "persona": persona}
        )

    def longitudinal_impact_score(
        self, scores_over_time: List[float]
    ) -> Dict[str, Any]:
        """Evaluates long-term emotional growth contributions."""
        return self._post("/v1/analyze/longitudinal", {"scores": scores_over_time})
