"""
SoftInjector — Applies reverie vectors as system-level behavioral modifiers.

The final stage of the Reverie Engine. Takes ReverieVectors produced by the
LatentSurfacer and applies them as subconscious behavioral influences.

CRITICAL: Never injects raw memory content. Only behavioral nudges, validation
patterns, and emotional tones derived from latent memories. These shape HOW
the model responds, not WHAT it says — mirroring Westworld's reveries where
subconscious gestures influence behavior without conscious awareness.

Python/TypeScript parity: src/lib/memory/reverie/soft-injector.ts mirrors this file.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from ..reverie_types import (
    DEFAULT_REVERIE_CONFIG,
    ReverieConfig,
    ReveriePhase,
    ReverieVector,
)

# ─── Phase-to-Influence Mapping ─────────────────────────────────────────
# Each reverie phase has a weight that scales its influence on the system
# prompt. Active reveries have full weight; dormant ones have zero.

PHASE_TO_INFLUENCE: dict[ReveriePhase, float] = {
    ReveriePhase.DORMANT: 0.0,
    ReveriePhase.SEEDED: 0.3,
    ReveriePhase.SURFACING: 0.6,
    ReveriePhase.ACTIVE: 1.0,
    ReveriePhase.FADING: 0.2,
}


# ─── Data Structures ─────────────────────────────────────────────────────


@dataclass
class InjectionResult:
    """Result of applying reverie vectors as behavioral modifiers."""

    prompt: str = ""
    active_reveries: list[ReverieVector] = field(default_factory=list)
    total_influence: float = 0.0
    empty: bool = True


@dataclass
class ConflictResolution:
    """Result of resolving conflicts between competing reveries."""

    winner: ReverieVector | None = None
    suppressed: list[ReverieVector] = field(default_factory=list)


# ─── SoftInjector ────────────────────────────────────────────────────────


class SoftInjector:
    """
    Applies reverie vectors as system-level behavioral modifiers.

    Never injects raw memory content — only behavioral nudges and validation
    patterns derived from latent memories. Decays exponentially over time.
    Maximum `max_active_reveries` simultaneous reveries.
    """

    def __init__(self, config: ReverieConfig = DEFAULT_REVERIE_CONFIG) -> None:
        self.config = config
        self._active_reveries: list[ReverieVector] = []
        self._message_counter = 0

    # ─── Public API ──────────────────────────────────────────────────────

    def apply(
        self,
        new_reveries: list[ReverieVector],
        message_count: int,
    ) -> InjectionResult:
        """
        Apply new reveries and return system-level behavioral modifier prompt.

        Args:
            new_reveries: New/surfaced reverie vectors from LatentSurfacer
            message_count: Current message count (for decay calculation)

        Returns:
            InjectionResult with assembled prompt and active reveries
        """
        self._message_counter = message_count

        # Step 1: Merge new reveries (retrigger if source already active)
        if new_reveries:
            self._merge_reveries(new_reveries)

        # Step 2: Apply exponential decay to all
        for rev in self._active_reveries:
            decayed = self._apply_decay(rev, message_count)
            rev.resonance_score = decayed.resonance_score
            rev.phase = decayed.phase

        # Step 3: Prune faded reveries
        self._prune_faded()

        # Step 4: Resolve conflicts if exceeding max
        if len(self._active_reveries) > self.config.max_active_reveries:
            self._resolve_conflicts()

        # Step 5: Assemble prompt
        prompt = self._assemble_prompt()

        total_influence = sum(self._current_influence(rev, message_count) for rev in self._active_reveries)

        return InjectionResult(
            prompt=prompt,
            active_reveries=list(self._active_reveries),
            total_influence=total_influence,
            empty=len(self._active_reveries) == 0,
        )

    def get_active(self) -> list[ReverieVector]:
        """Read-only access to currently active reveries."""
        return list(self._active_reveries)

    def get_current_prompt(self) -> str:
        """Get the current behavioral modifier prompt without applying changes."""
        if not self._active_reveries:
            return ""
        return self._assemble_prompt()

    def clear(self) -> None:
        """Clear all active reveries (e.g. on session end)."""
        self._active_reveries = []
        self._message_counter = 0

    # ─── Internal Methods ────────────────────────────────────────────────

    def _merge_reveries(self, new_reveries: list[ReverieVector]) -> None:
        """
        Merge new reveries with existing active reveries.
        If a reverie for the same source memory already exists, retrigger it
        instead of creating a duplicate.
        """
        existing_by_source: dict[str, ReverieVector] = {rev.source_memory_id: rev for rev in self._active_reveries}

        for new_rev in new_reveries:
            if new_rev.source_memory_id in existing_by_source:
                # Retrigger: blend resonance, increment trigger count
                existing = existing_by_source[new_rev.source_memory_id]
                blended = 0.4 * existing.resonance_score + 0.6 * new_rev.resonance_score
                existing.resonance_score = min(blended, 1.0)
                existing.trigger_count += 1
                existing.last_triggered_at = new_rev.created_at
                existing.phase = self._phase_from_resonance(existing.resonance_score)
            else:
                # New reverie
                self._active_reveries.append(new_rev)
                existing_by_source[new_rev.source_memory_id] = new_rev

    def _apply_decay(
        self,
        reverie: ReverieVector,
        message_count: int,
    ) -> ReverieVector:
        """
        Apply exponential decay to a reverie.
        Influence = resonance × 2^(-messages_since_trigger / half_life)
        """
        messages_since = message_count - reverie.last_triggered_at
        if messages_since <= 0:
            return reverie

        decay_factor = math.pow(2, -messages_since / reverie.decay_half_life)
        decayed_resonance = reverie.resonance_score * decay_factor
        new_phase = self._phase_from_resonance(decayed_resonance)

        reverie.resonance_score = decayed_resonance
        reverie.phase = new_phase
        return reverie

    def _prune_faded(self) -> None:
        """Remove reveries that have faded below the threshold."""
        self._active_reveries = [
            rev
            for rev in self._active_reveries
            if rev.phase != ReveriePhase.DORMANT and rev.resonance_score > self.config.fading_threshold
        ]

    def _resolve_conflicts(self) -> ConflictResolution:
        """
        Resolve conflicts when too many reveries are active.
        Highest influence wins; others are demoted to 'fading' phase.
        """
        sorted_revs = sorted(
            self._active_reveries,
            key=lambda r: self._current_influence(r, self._message_counter),
            reverse=True,
        )

        winners = sorted_revs[: self.config.max_active_reveries]
        suppressed = sorted_revs[self.config.max_active_reveries :]

        for rev in suppressed:
            rev.phase = ReveriePhase.FADING

        self._active_reveries = winners + suppressed

        return ConflictResolution(
            winner=winners[0] if winners else None,
            suppressed=suppressed,
        )

    def _assemble_prompt(self) -> str:
        """
        Assemble the system-level behavioral modifier prompt.
        Each reverie contributes behavioral nudge, validation pattern,
        relational awareness, and emotional tone — NOT raw memory content.
        """
        if not self._active_reveries:
            return ""

        sections: list[str] = []

        preamble = (
            "## Subconscious Behavioral Modifiers (Reverie Layer)\n\n"
            "The following are subtle behavioral influences derived from latent memory patterns.\n"
            "These are NOT memories to recall or reference directly. They shape HOW you respond,\n"
            "not WHAT you say. Integrate them as subtle currents in your therapeutic stance."
        )
        sections.append(preamble)

        for reverie in self._active_reveries:
            influence = self._current_influence(reverie, self._message_counter)
            if influence <= 0:
                continue

            percent = round(influence * 100)
            intensity = self._intensity_label(influence)

            lines: list[str] = []
            lines.append(f"\n[Reverie — {intensity} influence ({percent}%)]")
            lines.append(f"Behavioral nudge: {reverie.behavioral_nudge}")
            lines.append(f"Validation approach: {reverie.validation_pattern}")

            if reverie.relational_pattern:
                lines.append(f"Relational awareness: {reverie.relational_pattern}")

            tone = reverie.emotional_tone
            cats = ", ".join(tone.categories) if tone.categories else "none"
            lines.append(f"Emotional tone: valence={tone.valence:.2f}, arousal={tone.arousal:.2f}, categories=[{cats}]")

            sections.append("\n".join(lines))

        return "\n\n".join(sections)

    # ─── Helpers ─────────────────────────────────────────────────────────

    @staticmethod
    def _phase_from_resonance(resonance: float) -> ReveriePhase:
        """Determine reverie phase from resonance score."""
        if resonance > 0.7:
            return ReveriePhase.ACTIVE
        if resonance > 0.5:
            return ReveriePhase.SURFACING
        if resonance > 0.05:
            return ReveriePhase.SEEDED
        if resonance > 0:
            return ReveriePhase.FADING
        return ReveriePhase.DORMANT

    def _current_influence(
        self,
        reverie: ReverieVector,
        message_count: int,
    ) -> float:
        """Calculate current influence of a reverie at given message count."""
        messages_since = message_count - reverie.last_triggered_at
        if messages_since < 0:
            messages_since = 0

        decay_factor = math.pow(2, -messages_since / reverie.decay_half_life)
        phase_weight = PHASE_TO_INFLUENCE.get(reverie.phase, 0.0)

        return reverie.resonance_score * phase_weight * decay_factor

    @staticmethod
    def _intensity_label(influence: float) -> str:
        """Human-readable intensity label for prompt."""
        if influence > 0.5:
            return "strong"
        if influence > 0.25:
            return "moderate"
        return "subtle"
