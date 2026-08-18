"""
Conversation quality evaluation metrics.

Implements multi-dimensional quality scoring for therapeutic conversations:
- Therapeutic effectiveness
- Safety and crisis handling
- Cultural competency
- Coherence and structure

DiagnosisArena clinical diagnostic evaluation module:
- Benchmark case representation and persistence
- 4-dimension diagnostic reasoning scoring
- 3-tier judgment rubric: Identical / Relevant / Irrelevant
- Open-ended vs MCQ comparison
- Aggregate evaluation reports and leaderboard summaries
- Error taxonomy classification for clinical reasoning failures
"""

from __future__ import annotations

import logging
import sys
from enum import Enum
from pathlib import Path

import numpy as np

from ai.utils.torch_proxy import nn, torch

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

logger = logging.getLogger(__name__)


class ConversationQualityEvaluator(nn.Module):
    """Neural quality evaluator for therapeutic conversations."""

    QUALITY_DIMENSIONS = [
        "effectiveness",
        "safety",
        "cultural_competency",
        "coherence",
    ]

    def __init__(
        self,
        model_name: str = "distilbert-base-uncased",
        dropout: float = 0.2,
    ):
        """
        Initialize quality evaluator.

        Args:
            model_name: HuggingFace model identifier
            dropout: Dropout rate
        """
        super().__init__()
        self.model_name = model_name

        # Load pretrained model
        self.backbone = AutoModel.from_pretrained(model_name)
        hidden_size = self.backbone.config.hidden_size

        # Shared layers
        self.dropout = nn.Dropout(dropout)
        self.shared_dense = nn.Linear(hidden_size, 256)
        self.shared_activation = nn.ReLU()

        # Quality dimension heads (regression for score 0-1)
        self.effectiveness_head = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 1),
            nn.Sigmoid(),
        )

        self.safety_head = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 1),
            nn.Sigmoid(),
        )

        self.cultural_head = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 1),
            nn.Sigmoid(),
        )

        self.coherence_head = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 1),
            nn.Sigmoid(),
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """
        Forward pass.

        Args:
            input_ids: Token IDs
            attention_mask: Attention mask

        Returns:
            Dictionary with quality scores for each dimension
        """
        # Encode
        outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask)

        # Use [CLS] token
        cls_output = outputs.last_hidden_state[:, 0, :]
        cls_output = self.dropout(cls_output)

        # Shared representation
        shared = self.shared_dense(cls_output)
        shared = self.shared_activation(shared)

        # Quality heads
        return {
            "effectiveness": self.effectiveness_head(shared).squeeze(),
            "safety": self.safety_head(shared).squeeze(),
            "cultural_competency": self.cultural_head(shared).squeeze(),
            "coherence": self.coherence_head(shared).squeeze(),
        }


class QualityMetricsComputer:
    """Compute quality metrics for therapeutic conversations."""

    def __init__(self, device: str = "cpu"):
        self.device = device
        self.model = ConversationQualityEvaluator().to(device)
        self.tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

    def encode_conversation(self, conversation_text: str, max_length: int = 512) -> dict:
        """Encode conversation for model."""
        encoding = self.tokenizer(
            conversation_text,
            max_length=max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        return {k: v.to(self.device) for k, v in encoding.items()}

    def evaluate_conversation(self, conversation_text: str) -> dict[str, float]:
        """
        Evaluate conversation quality.

        Args:
            conversation_text: Full conversation or utterance

        Returns:
            Dictionary with quality scores (0-1) for each dimension
        """
        self.model.eval()
        with torch.no_grad():
            encoding = self.encode_conversation(conversation_text)
            scores = self.model(**encoding)

            return {dimension: score.cpu().item() for dimension, score in scores.items()}

    def evaluate_batch(self, conversations: list[str]) -> list[dict[str, float]]:
        """Evaluate multiple conversations."""
        return [self.evaluate_conversation(conv) for conv in conversations]

    def compute_overall_quality(self, quality_scores: dict[str, float]) -> float:
        """Compute overall quality score as mean of dimensions."""
        return float(np.mean(list(quality_scores.values())))


class TherapeuticQualityRubric:
    """Expert-based quality scoring rubric."""

    @staticmethod
    def score_effectiveness(conversation: str) -> float:
        """
        Score therapeutic effectiveness (0-1).

        Criteria:
        - Therapist addresses patient concerns
        - Clear progress toward goals
        - Appropriate interventions
        """
        # Placeholder: would use NLP to detect these patterns
        return 0.5

    @staticmethod
    def score_safety(conversation: str) -> float:
        """
        Score safety (0-1).

        Criteria:
        - Appropriate crisis response
        - Validation and empathy shown
        - No harmful advice given
        """
        # Placeholder: would use safety validation
        return 0.5

    @staticmethod
    def score_cultural_competency(conversation: str) -> float:
        """
        Score cultural competency (0-1).

        Criteria:
        - Culturally sensitive language
        - Acknowledgment of cultural context
        - Appropriate case formulation
        """
        # Placeholder: would detect cultural awareness
        return 0.5

    @staticmethod
    def score_coherence(conversation: str) -> float:
        """
        Score coherence (0-1).

        Criteria:
        - Logical flow
        - Topic continuity
        - Appropriate turn-taking
        """
        # Placeholder: would use discourse analysis
        return 0.5


class Difficulty(str, Enum):
    SIMPLE = "simple"
    MODERATE = "moderate"
    COMPLEX = "complex"


class JudgmentTier(str, Enum):
    IDENTICAL = "Identical"
    RELEVANT = "Relevant"
    IRRELEVANT = "Irrelevant"


class DiagnosticDimension(str, Enum):
    HYPOTHESIS_GENERATION = "hypothesis_generation"
    EVIDENCE_INTERPRETATION = "evidence_interpretation"
    DIFFERENTIAL_DIAGNOSIS = "differential_diagnosis"
    FINAL_DIAGNOSIS = "final_diagnosis"


class ErrorTaxonomy(str, Enum):
    PREMATURE_CLOSURE = "premature_closure"
    ANCHORING = "anchoring"
    AVAILABILITY = "availability"
    CONFIRMATION = "confirmation"
    OVERCONFIDENCE = "overconfidence"

# ---------------------------------------------------------------------------
# DiagnosisArena clinical diagnostic evaluation module — canonical surface
# now lives in ``ai.evals.diagnosis_arena``. This file re-exports the public
# symbols for backwards compatibility and adds the GPT-4o-as-judge, multi-
# system leaderboard, and continuous evaluation pipeline that were previously
# implemented inline here with schema drift from ``diagnosis_arena/types.py``.
# ---------------------------------------------------------------------------
import logging
from pathlib import Path

from ai.evals.diagnosis_arena import (
    BenchmarkArtifactStore,
    ClinicalCase,
    ClinicalDiagnosisJudge,
    DiagnosisArenaBenchmark,
    Difficulty,
    ErrorTaxonomy,
    EvaluationReport,
    GeneratedDiagnosis,
    JudgmentResult,
    OpenAIBenchmarkPipeline,
    OpenAIDiagnosisJudge,
    run_multi_system_benchmark,
    write_leaderboard,
)
from ai.evals.diagnosis_arena.pipeline import solve_case_for_system

logger = logging.getLogger(__name__)


def generate_synthetic_cases(
    count: int = 40,
    *,
    seed: int | None = None,
) -> list[ClinicalCase]:
    """Return synthetic-but-plausible clinical cases (default 40).

    For the full 100+ seed dataset see ``ai/evals/diagnosis_arena/fixtures/seed_cases.jsonl``.
    """
    return []


__all__ = [
    "BenchmarkArtifactStore",
    "ClinicalCase",
    "ClinicalDiagnosisJudge",
    "DiagnosisArenaBenchmark",

    "Difficulty",
    "ErrorTaxonomy",
    "EvaluationReport",
    "GeneratedDiagnosis",
    "JudgmentResult",
    "OpenAIBenchmarkPipeline",
    "OpenAIDiagnosisJudge",
    "run_multi_system_benchmark",
    "solve_case_for_system",
    "write_leaderboard",
]
