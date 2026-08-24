"""
Schema compatibility layer between TS and Python memory types.

Provides bidirectional conversion between:
  - ``foresight.schema.UnifiedMemory`` (canonical Python Pydantic, mirrors
    ``@pixelated/memory-schema`` TypeScript types)
  - ``ai.research.schema.MemoryBlock``         (legacy Python Pydantic — kept
    in sync via this module)

TS canonical:    packages/memory-schema/src/types.ts
Python MCP:       foresight/schema.py
Python legacy:    ai/research/schema.py
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from foresight.schema import (
    MEMORY_SCHEMA_VERSION,
    EmotionalContext,
    MemoryScope,
    RetentionPolicy,
    SourceService,
    UnifiedMemory,
)

from ai.research.schema import (
    ConsentGate as LegacyConsentGate,
    ConsolidationPhase as LegacyConsolidationPhase,
    MemoryBlock as LegacyMemoryBlock,
    MemoryConsolidation as LegacyMemoryConsolidation,
    MemoryEmotions as LegacyMemoryEmotions,
    MemoryGating as LegacyMemoryGating,
    MemoryImportance as LegacyMemoryImportance,
    PIIStatus as LegacyPIIStatus,
)

# Alias for consumers who don't want to pick a side
CanonicalMemory = UnifiedMemory

__all__ = [
    "MEMORY_SCHEMA_VERSION",
    "CanonicalMemory",
    "MemoryScope",
    "RetentionPolicy",
    "SourceService",
    "UnifiedMemory",
    "assert_schemas_in_sync",
    "from_legacy_block",
    "from_unified_memory",
    "to_legacy_block",
    "to_unified_memory",
]


# ─── Internal helpers ───────────────────────────────────────────────────────────


def _to_legacy_phase(
    phase: str | LegacyConsolidationPhase,
) -> LegacyConsolidationPhase:
    if isinstance(phase, LegacyConsolidationPhase):
        return phase
    mapping = {
        "raw": LegacyConsolidationPhase.RAW,
        "consolidated": LegacyConsolidationPhase.CONSOLIDATED,
        "archived": LegacyConsolidationPhase.ARCHIVED,
        "forgotten": LegacyConsolidationPhase.FORGOTTEN,
    }
    return mapping.get(str(phase).lower(), LegacyConsolidationPhase.RAW)


def _normalize_importance(imp: Any) -> float:
    """Extract a [0, 1] float importance from various input shapes."""
    if imp is None:
        return 0.5
    if isinstance(imp, (float, int)):
        return max(0.0, min(1.0, float(imp)))
    if isinstance(imp, dict):
        raw = imp.get("raw", imp.get("rawScore", 0.5))
        return max(0.0, min(1.0, float(raw)))
    if hasattr(imp, "raw"):
        return max(0.0, min(1.0, float(getattr(imp, "raw", 0.5))))
    return 0.5


# ─── Conversions: UnifiedMemory ↔ Legacy MemoryBlock ───────────────────────────


def to_legacy_block(
    unified: UnifiedMemory | dict[str, Any],
    tenant_id: str | None = None,
) -> LegacyMemoryBlock:
    """
    Convert a canonical UnifiedMemory into a legacy MemoryBlock.

    Used when passing memory to legacy ``ai/research/`` components that expect
    MemoryBlock. Supports both UnifiedMemory instances and dicts (e.g. from
    TS JSON serialization).
    """
    if isinstance(unified, dict):
        unified = UnifiedMemory(**unified)

    now_ms = int(datetime.now(UTC).timestamp() * 1000)
    effective_tenant = tenant_id or unified.tenant_id
    # Map UnifiedMemory.user_id → MemoryBlock.sessionId (the legacy field)
    effective_session = unified.user_id

    legacy_importance = LegacyMemoryImportance(
        raw=_normalize_importance(unified.importance),
        recency=0.0,
        relevance=0.0,
        emotionalWeight=1.0,
        actionability=0.0,
    )

    legacy_emotions = LegacyMemoryEmotions(
        valence=0.0,
        arousal=0.0,
        categories=[],
    )
    if unified.emotional_context:
        ec = unified.emotional_context
        legacy_emotions = LegacyMemoryEmotions(
            valence=ec.valence,
            arousal=ec.arousal,
            categories=[ec.primary_emotion] if ec.primary_emotion else [],
        )

    # Synthesize gating from crisis-like content signals
    crisis_keywords = ["suicide", "self-harm", "panic attack", "overdose"]
    content_lower = unified.content.lower() if unified.content else ""
    crisis_flag = any(kw in content_lower for kw in crisis_keywords)

    legacy_gating = LegacyMemoryGating(
        piiStatus=LegacyPIIStatus.ABSENT,
        crisisFlag=crisis_flag,
        traumaIndicators=[],
        consentGate=LegacyConsentGate.OPEN,
    )

    legacy_consolidation = LegacyMemoryConsolidation(
        phase=_to_legacy_phase("raw"),
        lastProcessed=now_ms,
        remCycles=0,
        schemaReferences=[],
    )

    return LegacyMemoryBlock(
        id=unified.id,
        tenantId=effective_tenant,
        sessionId=effective_session,
        content=unified.content,
        timestamp=now_ms,
        importance=legacy_importance,
        emotions=legacy_emotions,
        gating=legacy_gating,
        consolidation=legacy_consolidation,
    )


def from_legacy_block(legacy: LegacyMemoryBlock) -> UnifiedMemory:
    """
    Convert a legacy MemoryBlock into a canonical UnifiedMemory.

    This is the primary entry point for the MCP server's Python-side code
    (``ai/research/``) to interoperate with the unified schema used by
    ``foresight``.
    """
    imp = legacy.importance or LegacyMemoryImportance(
        raw=0.0, recency=0.0, relevance=0.0, emotionalWeight=1.0, actionability=0.0
    )
    emotions = legacy.emotions or LegacyMemoryEmotions(valence=0.0, arousal=0.0, categories=[])

    importance_float = max(0.0, min(1.0, float(imp.raw if hasattr(imp, "raw") else 0.0)))

    primary_emotion = ""
    if hasattr(emotions, "categories") and emotions.categories:
        primary_emotion = str(emotions.categories[0] or "")

    emotional_ctx = EmotionalContext(
        valence=getattr(emotions, "valence", 0.0),
        arousal=getattr(emotions, "arousal", 0.0),
        dominance=0.0,
        primary_emotion=primary_emotion,
        intensity=0.0,
    )

    scope_str = getattr(legacy, "scope", None)
    if scope_str:
        scope_map = {
            "session": MemoryScope.SESSION,
            "arc": MemoryScope.ARC,
            "trait": MemoryScope.TRAIT,
            "fact": MemoryScope.FACT,
        }
        unified_scope = scope_map.get(str(scope_str), MemoryScope.SESSION)
    else:
        unified_scope = MemoryScope.SESSION

    tags: list[str] = getattr(legacy, "tags", []) if hasattr(legacy, "tags") else []

    return UnifiedMemory(
        id=legacy.id,
        tenant_id=legacy.tenantId,
        user_id=legacy.sessionId,
        bank_id=getattr(legacy, "bank_id", legacy.sessionId),
        content=legacy.content,
        scope=unified_scope,
        retention=RetentionPolicy.SHORT_TERM,
        category=getattr(legacy, "category", "general"),
        tags=tags,
        importance=importance_float,
        emotional_context=emotional_ctx,
        created_at=str(datetime.fromtimestamp(legacy.timestamp / 1000, tz=UTC).isoformat()),
        updated_at=None,
        source_service=SourceService.AI_SERVICES,
    )


def to_unified_memory(data: dict[str, Any]) -> UnifiedMemory:
    """Parse any dict into a UnifiedMemory (supports camelCase TS JSON)."""
    return UnifiedMemory(**data)


def from_unified_memory(mem: UnifiedMemory) -> dict[str, Any]:
    """Serialize a UnifiedMemory to a dict matching the TS camelCase JSON contract."""
    return mem.model_dump(mode="json")


# ─── Schema sync assertion ──────────────────────────────────────────────────────


def assert_schemas_in_sync() -> None:
    """
    Runtime assertion that both Python schemas describe compatible fields.

    Run at module import or test setup to catch field-name drift before it
    causes runtime mismatches between the TS API contract and Python storage.

    Checks:
    1. ``MEMORY_SCHEMA_VERSION`` equals ``"1.0.0"``
    2. ``UnifiedMemory`` has all fields required for legacy conversion
    3. ``MemoryScope`` enum values are correct
    4. Bidirectional round-trip preserves id, content, tenant_id
    """
    # Check version constant
    assert MEMORY_SCHEMA_VERSION == "1.0.0", f"Schema version mismatch: foresight has {MEMORY_SCHEMA_VERSION!r}"

    # Check field overlap for to_legacy_block
    unified_fields = set(UnifiedMemory.model_fields.keys())
    required_for_legacy = {"id", "content", "user_id", "tenant_id"}
    missing = required_for_legacy - unified_fields
    assert not missing, f"UnifiedMemory missing fields needed for legacy conversion: {missing}"

    # Check MemoryScope enum values
    for variant, member in [
        ("session", MemoryScope.SESSION),
        ("arc", MemoryScope.ARC),
        ("trait", MemoryScope.TRAIT),
        ("fact", MemoryScope.FACT),
    ]:
        assert member.value == variant, f"MemoryScope.{variant.upper()} wrong value: {member.value!r}"

    # Round-trip: legacy → unified → legacy preserves id, content, tenant_id
    dummy_legacy = LegacyMemoryBlock(
        id="roundtrip_test",
        tenantId="test_tenant",
        sessionId="test_user",
        content="Patient described persistent low mood and sleep disruption",
        timestamp=0,
    )
    unified = from_legacy_block(dummy_legacy)
    assert unified.id == "roundtrip_test"
    assert unified.content == ("Patient described persistent low mood and sleep disruption")
    assert unified.tenant_id == "test_tenant"
    assert unified.user_id == "test_user"

    # Round-trip: unified → legacy → unified
    unified2 = UnifiedMemory(
        id="unified_roundtrip",
        tenant_id="tenant2",
        user_id="user2",
        bank_id="bank2",
        content="Therapist noted progress in CBT techniques",
        scope=MemoryScope.ARC,
        retention=RetentionPolicy.LONG_TERM,
        importance=0.72,
    )
    legacy2 = to_legacy_block(unified2)
    back = from_legacy_block(legacy2)
    assert back.id == "unified_roundtrip"
    assert back.content == "Therapist noted progress in CBT techniques"
    assert back.tenant_id == "tenant2"
