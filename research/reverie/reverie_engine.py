"""
Reverie Engine — orchestrates the Reverie system.

Coordinates FishhookDetector, LatentSurfacer, and SoftInjector to
surface latent memories as subconscious behavioral modifiers.

Flow:
    1. Check if fishhook detection should run (every N messages)
    2. Run FishhookDetector on current message vs latent pool
    3. Pass matches to LatentSurfacer → ReverieVector[]
    4. Pass new reveries to SoftInjector → behavioral modifier prompt
    5. Return ReverieResult with all pieces

Also provides:
    - seed_reverie_candidates: mark memories as latent + reverie_eligible
    - get_latent_pool: filter memories by consolidation phase
    - get_active_reveries: read-only access to current active reveries

Python/TypeScript parity: src/lib/memory/reverie/reverie_engine.ts mirrors this file.
"""

from __future__ import annotations

import time

from ai.research.reverie_types import (
    DEFAULT_REVERIE_CONFIG,
    FishhookMatch,
    ReverieConfig,
    ReverieResult,
    ReverieSeed,
    ReverieSeedResult,
    ReverieVector,
)
from ai.research.schema import MemoryBlock

from .fishhook_detector import FishhookDetector
from .latent_surfacer import LatentSurfacer
from .soft_injector import SoftInjector


class ReverieEngine:
    """Orchestrates the Reverie system: fishhook detection → latent surfacing → soft injection."""

    def __init__(self, config: ReverieConfig = DEFAULT_REVERIE_CONFIG) -> None:
        self.config = config
        self.detector = FishhookDetector(config)
        self.surfacer = LatentSurfacer(config)
        self.injector = SoftInjector(config)
        self.message_count = 0
        self.latent_pool: list[MemoryBlock] = []
        self.last_detection_at = 0

    # ─── Public API ──────────────────────────────────────────────────────

    def process(
        self,
        current_message: str,
        current_emotions: tuple[float, float, list[str]],
        all_memories: list[MemoryBlock] | None = None,
    ) -> ReverieResult:
        """
        Process a new message through the reverie pipeline.

        Args:
            current_message: The user's current message text
            current_emotions: Dict with 'valence' (-1..1), 'arousal' (0..1), 'categories' (list[str])
            all_memories: All memories (used to extract latent pool if not set)

        Returns:
            ReverieResult with fishhooks, new_reveries, active_reveries, reverie_prompt
        """
        start_time = time.time() * 1000
        self.message_count += 1

        # Refresh latent pool if memories provided
        if all_memories is not None:
            self.latent_pool = self._extract_latent_pool(all_memories)

        # Check if detection should run this cycle
        if not self.detector.should_run(self.message_count):
            active = list(self.injector.get_active())
            prompt = self.injector.get_current_prompt() if active else ""
            return ReverieResult(
                fishhooks=[],
                new_reveries=[],
                active_reveries=active,
                reverie_prompt=prompt,
                changed=False,
                elapsed_ms=int(time.time() * 1000 - start_time),
            )

        # Build IDF index from latent pool
        self.detector.build_index(self.latent_pool)

        # Phase 1: Detect fishhooks
        fishhooks: list[FishhookMatch] = self.detector.detect(
            current_message,
            current_emotions,
            self.latent_pool,
        )

        if len(fishhooks) == 0:
            # No matches — still decay existing reveries
            injection_result = self.injector.apply([], self.message_count)
            return ReverieResult(
                fishhooks=[],
                new_reveries=[],
                active_reveries=injection_result.active_reveries,
                reverie_prompt=injection_result.prompt,
                changed=not injection_result.empty,
                elapsed_ms=int(time.time() * 1000 - start_time),
            )

        # Phase 2: Surface reverie vectors from matches
        new_reveries: list[ReverieVector] = self.surfacer.surface(fishhooks, self.latent_pool)

        # Phase 3: Inject into soft injector (merge + decay + resolve + prompt)
        injection_result = self.injector.apply(new_reveries, self.message_count)

        self.last_detection_at = self.message_count

        return ReverieResult(
            fishhooks=fishhooks,
            new_reveries=new_reveries,
            active_reveries=injection_result.active_reveries,
            reverie_prompt=injection_result.prompt,
            changed=True,
            elapsed_ms=int(time.time() * 1000 - start_time),
        )

    def seed_reverie_candidates(self, memories: list[MemoryBlock]) -> ReverieSeedResult:
        """
        Seed reverie candidates from a list of memories.

        Marks memories as latent + reverie_eligible if they meet criteria:
            - consolidation phase is 'archived' or 'forgotten'
            - emotional weight >= reverie_eligible_min_emotional_weight
            - NOT a crisis memory (crisis memories stay preserved)
            - importance.raw >= latent_pool_min_importance

        Returns:
            ReverieSeedResult with seeds, already_latent, latent_pool_size
        """
        start_time = time.time() * 1000
        seeds: list[ReverieSeed] = []
        already_latent: list[str] = []

        for mem in memories:
            # Skip crisis memories — they stay preserved, never latent
            if mem.gating.crisisFlag:
                continue

            # Skip if already latent
            if mem.consolidation.phase == "latent":
                already_latent.append(mem.id)
                continue

            # Only seed from archived or forgotten phase
            if mem.consolidation.phase not in ("archived", "forgotten"):
                continue

            # Check emotional weight threshold
            if mem.importance.emotionalWeight < self.config.reverie_eligible_min_emotional_weight:
                continue

            # Check minimum importance
            if mem.importance.raw < self.config.latent_pool_min_importance:
                continue

            # Calculate reverie potential
            potential = self._calculate_reverie_potential(mem)

            seeds.append(
                ReverieSeed(
                    memory_id=mem.id,
                    reason=self._derive_seed_reason(mem),
                    potential=potential,
                )
            )

        # Sort seeds by potential descending
        seeds.sort(key=lambda s: s.potential, reverse=True)

        return ReverieSeedResult(
            seeds=seeds,
            already_latent=already_latent,
            latent_pool_size=len(self.latent_pool),
            elapsed_ms=int(time.time() * 1000 - start_time),
        )

    def apply_seeds(self, memories: list[MemoryBlock], seeds: list[ReverieSeed]) -> list[MemoryBlock]:
        """
        Apply seeding to memories — returns updated MemoryBlocks with
        consolidation.phase = 'latent', consolidation.reverieEligible = True,
        and importance.reveriePotential set.
        """
        seed_map = {s.memory_id: s for s in seeds}
        updated: list[MemoryBlock] = []

        for mem in memories:
            seed = seed_map.get(mem.id)
            if seed:
                updated_mem = mem.model_copy(deep=True)
                updated_mem.consolidation.phase = "latent"
                updated_mem.consolidation.reverieEligible = True
                updated_mem.consolidation.reveriePhase = "seeded"
                updated_mem.importance.reveriePotential = seed.potential
                updated.append(updated_mem)
            else:
                updated.append(mem)

        # Update internal latent pool
        self.latent_pool = self._extract_latent_pool(updated)

        return updated

    def get_latent_pool(self) -> list[MemoryBlock]:
        """Get the current latent pool (memories with phase='latent' and reverie_eligible=True)."""
        return list(self.latent_pool)

    def set_latent_pool(self, memories: list[MemoryBlock]) -> None:
        """Set the latent pool directly (e.g., from external memory store)."""
        self.latent_pool = self._extract_latent_pool(memories)

    def get_active_reveries(self) -> list[ReverieVector]:
        """Get currently active reverie vectors."""
        return list(self.injector.get_active())

    def get_reverie_prompt(self) -> str:
        """Get the current reverie prompt (behavioral modifier)."""
        return self.injector.get_current_prompt()

    def clear(self) -> None:
        """Clear all active reveries (e.g., on session end)."""
        self.injector.clear()
        self.message_count = 0
        self.last_detection_at = 0

    def get_message_count(self) -> int:
        """Get message count for external monitoring."""
        return self.message_count

    # ─── Private helpers ──────────────────────────────────────────────────

    def _extract_latent_pool(self, memories: list[MemoryBlock]) -> list[MemoryBlock]:
        """
        Extract latent pool from a list of memories.
        Latent = phase is 'latent' AND reverie_eligible is True.
        """
        return [m for m in memories if m.consolidation.phase == "latent" and m.consolidation.reverieEligible]

    def _calculate_reverie_potential(self, mem: MemoryBlock) -> float:
        """
        Calculate reverie potential for a memory.

        Potential = weighted combination of:
            - emotional weight (0.4) — more emotional = more reverie potential
            - emotional categories diversity (0.2) — richer emotional content
            - schema references (0.2) — consolidated patterns
            - recency (0.2) — more recent latent memories have higher potential
        """
        emotional_component = min(mem.importance.emotionalWeight / 5.0, 1.0)
        category_diversity = min(len(mem.emotions.categories) / 5.0, 1.0)
        schema_richness = min(len(mem.consolidation.schemaReferences) / 5.0, 1.0)
        recency_component = mem.importance.recency

        return min(
            0.4 * emotional_component + 0.2 * category_diversity + 0.2 * schema_richness + 0.2 * recency_component,
            1.0,
        )

    def _derive_seed_reason(self, mem: MemoryBlock) -> str:
        """Derive a human-readable reason for why a memory was seeded."""
        reasons: list[str] = []

        if mem.importance.emotionalWeight >= 4.0:
            reasons.append("high emotional weight")
        if len(mem.emotions.categories) >= 3:
            reasons.append("rich emotional categories")
        if len(mem.consolidation.schemaReferences) >= 2:
            reasons.append("cross-linked in consolidation")
        if mem.importance.recency > 0.5:
            reasons.append("recently active")

        if len(reasons) == 0:
            return "eligible for reverie surfacing"

        return "; ".join(reasons)
