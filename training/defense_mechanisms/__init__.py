"""
Defense Mechanism Detection Module

DMRS-based psychological defense mechanism classification for
emotional support dialogues. Integrates with the Pixelated Empathy
Emotional Intelligence Engine.

Reference: PsyDefDetect@BioNLP 2026
Framework: Defense Mechanism Rating Scales (Perry, 1990)
"""

from ai.training.defense_mechanisms.constants import (
    DEFENSE_LABELS,
    DEFENSE_MATURITY,
    NUM_LABELS,
)
from ai.training.defense_mechanisms.dataset import (
    DefenseDataset,
    compute_class_weights,
    format_dialogue,
)
from ai.training.defense_mechanisms.model import (
    DefenseClassifier,
    DefensePrediction,
    FocalLoss,
)

__all__ = [
    "DEFENSE_LABELS",
    "DEFENSE_MATURITY",
    "NUM_LABELS",
    "DefenseDataset",
    "DefenseClassifier",
    "DefensePrediction",
    "FocalLoss",
    "format_dialogue",
    "compute_class_weights",
]
