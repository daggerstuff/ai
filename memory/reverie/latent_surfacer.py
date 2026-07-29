"""
LatentSurfacer — transforms FishhookMatch[] into ReverieVector[]

The second component of the Reverie Engine. Takes raw fishhook detections
and the source latent memories, then extracts emotional tone, derives
behavioral nudges, validation patterns, and relational patterns.

CRITICAL: Never returns raw memory content. Only subconscious-level
behavioral modifiers. The output is a "reverie" — a subtle influence
on generation behavior, not an explicit memory retrieval.

TS/Python parity: src/lib/memory/reverie/latent-surfacer.ts
"""

from __future__ import annotations

from typing import Optional

from memory.schema import MemoryBlock
from memory.reverie_types import (
    FishhookMatch,
    ReverieVector,
    ReverieConfig,
    ReveriePhase,
    EmotionalTone,
    DEFAULT_REVERIE_CONFIG,
)

# ─── Behavioral Nudge Templates ──────────────────────────────────────

BEHAVIORAL_NUDGES: dict[str, str] = {
    "anxiety": "acknowledge underlying anxiety without forcing resolution",
    "grief": "hold space for grief; do not rush to fix or minimize",
    "trauma": "trauma-informed pacing; prioritize felt safety before exploration",
    "fear": "normalize fear as protective; ground in present-moment safety",
    "anger": "validate anger; explore its protective function without judgment",
    "despair": "assess for crisis indicators; validate the weight without dismissing",
    "hopelessness": "counter hopelessness gently; anchor in small achievable steps",
    "hope": "reinforce emerging hope; build on positive momentum without inflating",
    "joy": "amplify joy authentically; connect to sustained meaning",
    "sadness": "honor sadness as natural; resist the urge to cheer up prematurely",
    "guilt": "distinguish healthy remorse from toxic guilt; explore repair actions",
    "shame": "externalize shame; reduce self-attack through compassion framing",
    "relief": "acknowledge relief; explore what shifted without assuming permanence",
    "pride": "recognize earned pride; connect effort to outcome",
    "confusion": "normalize confusion as part of integration; do not resolve prematurely",
    "loneliness": "validate loneliness; explore connection patterns without prescribing",
    "acceptance": "reinforce acceptance; link to behavioral congruence",
    "love": "honor love in its complexity; avoid reducing to sentiment",
    "trust": "recognize emerging trust; protect it through consistency",
    "curiosity": "nurture curiosity; connect to self-directed exploration",
}

VALIDATION_PATTERNS: dict[str, str] = {
    "anxiety": "Anxiety is a valid response to perceived uncertainty; validate the felt sense of threat before exploring alternatives",
    "grief": "Grief reflects the depth of attachment; validate the loss without offering closure narratives",
    "trauma": "Trauma responses are adaptive survival mechanisms; validate the body's protective intelligence",
    "fear": "Fear serves a protective function; validate the alert system before examining its accuracy",
    "anger": "Anger often protects more vulnerable emotions beneath; validate the protective layer",
    "despair": "Despair reflects accumulated weight; validate the burden without minimizing or catastrophizing",
    "hopelessness": "Hopelessness signals cognitive exhaustion; validate the fatigue without confirming the conclusion",
    "hope": "Hope emerging alongside difficulty is meaningful; validate without over-investing",
    "joy": "Joy in context of struggle is resilience; validate without dismissing the hard parts",
    "sadness": "Sadness is a natural response to loss or disappointment; validate without rushing repair",
    "guilt": "Guilt can signal values in tension; validate the moral sensitivity",
    "shame": "Shame thrives in isolation; validate through externalizing and contextualizing",
    "relief": "Relief indicates shifting internal conditions; validate without assuming permanence",
    "pride": "Pride in small steps builds self-efficacy; validate the process not just outcome",
    "confusion": "Confusion often precedes integration; validate the discomfort of not-knowing",
    "loneliness": "Loneliness signals unmet connection needs; validate without pathologizing",
    "acceptance": "Acceptance is not passivity; validate the active process of reckoning",
    "love": "Love coexists with difficulty; validate without romanticizing or minimizing",
    "trust": "Trust-building is gradual; validate the vulnerability involved",
    "curiosity": "Curiosity is self-directed healing; validate the inner drive toward growth",
}

DEFAULT_NUDGE = "respond with heightened emotional attunement; let the conversation breathe"
DEFAULT_VALIDATION = "validate the emotional experience without assuming its cause"


# ─── Helpers ──────────────────────────────────────────────────────────

def _generate_reverie_id(source_memory_id: str, timestamp: int) -> str:
    hash_val = 0
    for c in source_memory_id:
        hash_val = ((hash_val << 5) - hash_val + ord(c)) & 0xFFFFFFFF
    return f"rev_{abs(hash_val):x}_{timestamp:x}"


def _derive_behavioral_nudge(categories: list[str], resonance_score: float) -> str:
    if not categories:
        return DEFAULT_NUDGE
    sorted_cats = categories[:2]
    nudges = [BEHAVIORAL_NUDGES[c] for c in sorted_cats if c in BEHAVIORAL_NUDGES]
    if not nudges:
        return DEFAULT_NUDGE
    if len(nudges) == 1:
        return nudges[0]
    if resonance_score > 0.6:
        return f"{nudges[0]}; {nudges[1]}"
    return f"gently {nudges[0]}"


def _derive_validation_pattern(categories: list[str]) -> str:
    if not categories:
        return DEFAULT_VALIDATION
    patterns = [VALIDATION_PATTERNS[c] for c in categories[:2] if c in VALIDATION_PATTERNS]
    if not patterns:
        return DEFAULT_VALIDATION
    return patterns[0]


def _derive_relational_pattern(
    schema_references: list[str],
    categories: list[str],
) -> Optional[str]:
    if not schema_references:
        return None
    theme = categories[0] if categories else "emotional"
    schema_count = len(schema_references)
    if schema_count >= 3:
        return f"themes of {theme} recur across multiple consolidated schemas; likely a core pattern"
    if schema_count >= 1:
        return f"emerging {theme} pattern detected in consolidation; monitor for recurrence"
    return None


def _determine_initial_phase(resonance_score: float) -> ReveriePhase:
    if resonance_score > 0.7:
        return ReveriePhase.ACTIVE
    if resonance_score > 0.5:
        return ReveriePhase.SURFACING
    return ReveriePhase.SEEDED


# ─── LatentSurfacer ───────────────────────────────────────────────────

class LatentSurfacer:
    """Transforms FishhookMatch[] into ReverieVector[].

    Never returns raw memory content. Only subconscious-level behavioral
    modifiers: emotional tone, behavioral nudge, validation pattern,
    relational pattern.
    """

    def __init__(self, config: Optional[ReverieConfig] = None) -> None:
        self.config = config if config is not None else DEFAULT_REVERIE_CONFIG

    def surface(
        self,
        matches: list[FishhookMatch],
        latent_pool: list[MemoryBlock],
    ) -> list[ReverieVector]:
        if not matches:
            return []

        memory_map: dict[str, MemoryBlock] = {mem.id: mem for mem in latent_pool}
        reveries: list[ReverieVector] = []

        for match in matches:
            source_memory = memory_map.get(match.latent_memory_id)
            if source_memory is None:
                continue
            if not source_memory.consolidation.reverieEligible:
                continue
            if source_memory.gating.crisisFlag:
                continue

            emotions = source_memory.emotions
            categories = emotions.categories
            resonance = match.resonance_score

            reverie = ReverieVector(
                id=_generate_reverie_id(match.latent_memory_id, match.timestamp),
                source_memory_id=match.latent_memory_id,
                resonance_score=resonance,
                emotional_tone=EmotionalTone(
                    valence=emotions.valence,
                    arousal=emotions.arousal,
                    categories=list(categories),
                ),
                behavioral_nudge=_derive_behavioral_nudge(categories, resonance),
                validation_pattern=_derive_validation_pattern(categories),
                relational_pattern=_derive_relational_pattern(
                    source_memory.consolidation.schemaReferences,
                    categories,
                ),
                phase=_determine_initial_phase(resonance),
                created_at=match.timestamp,
                last_triggered_at=match.timestamp,
                trigger_count=1,
                decay_half_life=self.config.decay_half_life_messages,
            )
            reveries.append(reverie)

        reveries.sort(key=lambda r: r.resonance_score, reverse=True)

        max_active = self.config.max_active_reveries
        if len(reveries) > max_active:
            for i in range(max_active, len(reveries)):
                reveries[i].phase = ReveriePhase.DORMANT

        return reveries

    def retrigger(
        self,
        reverie: ReverieVector,
        new_resonance: float,
        timestamp: int,
    ) -> ReverieVector:
        trigger_count = reverie.trigger_count + 1
        blended = reverie.resonance_score * 0.4 + new_resonance * 0.6

        if blended > 0.7:
            phase = ReveriePhase.ACTIVE
        elif blended > 0.5:
            phase = ReveriePhase.SURFACING
        elif blended > self.config.fading_threshold:
            phase = ReveriePhase.SEEDED
        else:
            phase = ReveriePhase.FADING

        return ReverieVector(
            id=reverie.id,
            source_memory_id=reverie.source_memory_id,
            resonance_score=blended,
            emotional_tone=reverie.emotional_tone,
            behavioral_nudge=reverie.behavioral_nudge,
            validation_pattern=reverie.validation_pattern,
            relational_pattern=reverie.relational_pattern,
            phase=phase,
            created_at=reverie.created_at,
            last_triggered_at=timestamp,
            trigger_count=trigger_count,
            decay_half_life=reverie.decay_half_life,
        )

    def decay(
        self,
        reverie: ReverieVector,
        messages_since_trigger: int,
    ) -> ReverieVector:
        if messages_since_trigger <= 0:
            return reverie

        decay_factor = 0.5 ** (messages_since_trigger / reverie.decay_half_life)
        decayed = reverie.resonance_score * decay_factor

        if decayed <= self.config.fading_threshold:
            phase = ReveriePhase.FADING
        elif decayed > 0.7:
            phase = ReveriePhase.ACTIVE
        elif decayed > 0.5:
            phase = ReveriePhase.SURFACING
        else:
            phase = ReveriePhase.SEEDED

        return ReverieVector(
            id=reverie.id,
            source_memory_id=reverie.source_memory_id,
            resonance_score=decayed,
            emotional_tone=reverie.emotional_tone,
            behavioral_nudge=reverie.behavioral_nudge,
            validation_pattern=reverie.validation_pattern,
            relational_pattern=reverie.relational_pattern,
            phase=phase,
            created_at=reverie.created_at,
            last_triggered_at=reverie.last_triggered_at,
            trigger_count=reverie.trigger_count,
            decay_half_life=reverie.decay_half_life,
        )

    def is_faded(self, reverie: ReverieVector) -> bool:
        return (
            reverie.phase == ReveriePhase.FADING
            and reverie.resonance_score < self.config.fading_threshold
        )
