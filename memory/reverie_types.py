"""
Reverie Engine Types

Inspired by Westworld's "reveries" — Arnold Weber's mechanism where latent
memories from purged narrative loops surface as subconscious gestures that
influence behavior without conscious awareness.

In the memory system, reveries allow archived/latent memories to subtly
influence LLM behavior through system-level behavioral modifiers, without
being explicitly retrieved into the context window.

Python/TypeScript parity: src/types/reverie.ts mirrors this file.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


# ─── Reverie Phase ──────────────────────────────────────────────────────
# Tracks where a memory sits in the reverie lifecycle.
# dormant: not yet eligible for reveries
# seeded: flagged as reverie-eligible, waiting in latent pool
# surfacing: fishhook detected, reverie vector being formed
# active: reverie vector injected as behavioral modifier
# fading: resonance decaying, soon to return to dormant/seeded


class ReveriePhase(str, Enum):
    DORMANT = "dormant"
    SEEDED = "seeded"
    SURFACING = "surfacing"
    ACTIVE = "active"
    FADING = "fading"


# ─── Fishhook Match ─────────────────────────────────────────────────────
# A trigger that resonates between current context and a latent memory.
# "Fishhooks" pull from the deep sea of consciousness — subtle cues that
# access memories that were supposed to be purged.


class FishhookMatchType(str, Enum):
    LEXICAL = "lexical"
    EMOTIONAL = "emotional"
    PATTERN = "pattern"
    SURPRISE = "surprise"


@dataclass
class FishhookMatch:
    """A trigger that resonates between current context and a latent memory."""

    latent_memory_id: str
    trigger_memory_id: str
    match_type: FishhookMatchType
    resonance_score: float  # [0,1]
    matched_features: list[str] = field(default_factory=list)
    timestamp: int = 0


# ─── Reverie Vector ─────────────────────────────────────────────────────
# A subconscious influence derived from a latent memory. Does NOT contain
# raw memory content — only emotional tone and behavioral patterns.
# This is the "reverie" itself: a subtle gesture that influences behavior.


@dataclass
class EmotionalTone:
    """Emotional tone extracted from the source memory."""

    valence: float  # -1..1
    arousal: float  # 0..1
    categories: list[str] = field(default_factory=list)


@dataclass
class ReverieVector:
    """
    A subconscious influence derived from a latent memory.
    Does NOT contain raw memory content — only emotional tone and behavioral patterns.
    """

    id: str
    source_memory_id: str
    resonance_score: float  # [0,1]
    emotional_tone: EmotionalTone
    behavioral_nudge: str
    validation_pattern: str
    relational_pattern: Optional[str]
    phase: ReveriePhase
    created_at: int
    last_triggered_at: int
    trigger_count: int
    decay_half_life: int  # in messages


# ─── Reverie Config ─────────────────────────────────────────────────────


@dataclass
class ReverieConfig:
    """Configuration for the Reverie Engine."""

    fishhook_threshold: float = 0.3
    emotional_resonance_weight: float = 0.3
    lexical_resonance_weight: float = 0.25
    pattern_resonance_weight: float = 0.25
    surprise_resonance_weight: float = 0.2
    max_active_reveries: int = 3
    decay_half_life_messages: int = 10
    trigger_interval: int = 5
    latent_pool_min_importance: float = 0.1
    reverie_eligible_min_emotional_weight: float = 2.0
    fading_threshold: float = 0.05


DEFAULT_REVERIE_CONFIG = ReverieConfig()


# ─── Reverie Result ─────────────────────────────────────────────────────


@dataclass
class ReverieResult:
    """Result of a reverie engine processing cycle."""

    fishhooks: list[FishhookMatch] = field(default_factory=list)
    new_reveries: list[ReverieVector] = field(default_factory=list)
    active_reveries: list[ReverieVector] = field(default_factory=list)
    reverie_prompt: str = ""
    changed: bool = False
    elapsed_ms: int = 0


# ─── Reverie Seeds (from REM Dream) ─────────────────────────────────────


@dataclass
class ReverieSeed:
    """Memory seeded into latent pool."""

    memory_id: str
    reason: str
    potential: float  # [0,1]


@dataclass
class ReverieSeedResult:
    """Result of reverie seeding from REM Dream."""

    seeds: list[ReverieSeed] = field(default_factory=list)
    already_latent: list[str] = field(default_factory=list)
    latent_pool_size: int = 0
    elapsed_ms: int = 0
