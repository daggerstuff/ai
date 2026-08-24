"""
PIX-510 Task 4: Cross-Language Memory API
FastAPI service for memory operations (CRUD + search + scoring).

Endpoints:
  POST   /memories              Create memory block
  GET    /memories              Search/list memories
  GET    /memories/{id}         Get single memory
  PATCH  /memories/{id}         Update memory
  DELETE /memories/{id}         Delete memory
  POST   /memories/score        Score existing memory by id
  GET    /memories/trajectory   Session emotional trajectory

Acceptance: all CRUD, filtering by importance/emotion/date, p95 < 100ms.
OpenAPI spec auto-generated. TypeScript client mirrors 1:1.
"""

from __future__ import annotations

import time
from typing import Annotated

from fastapi import Depends, FastAPI, HTTPException, Query, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from ai.research.emotion_classifier import EmotionClassifier
from ai.research.importance_scorer import ImportanceScorer
from ai.research.schema import (
    ConsentGate,
    ConsolidationPhase,
    MemoryBlock,
    MemoryConsolidation,
    MemoryEmotions,
    MemoryGating,
    MemoryImportance,
    MemorySearchFilters,
    MemoryWriteInput,
    PIIStatus,
)

# ─── In-memory store ───────────────────────────────────────────────────────────


class MemoryStore:
    """Simple in-memory store — replace with Redis/PostgreSQL backend for production."""

    def __init__(self) -> None:
        self._store: dict[str, MemoryBlock] = {}
        self._index: dict[str, list[str]] = {}  # tenantId → list of ids

    def put(self, block: MemoryBlock) -> MemoryBlock:
        self._store[block.id] = block
        if block.tenantId not in self._index:
            self._index[block.tenantId] = []
        if block.id not in self._index[block.tenantId]:
            self._index[block.tenantId].append(block.id)
        return block

    def get(self, memory_id: str, tenant_id: str) -> MemoryBlock | None:
        block = self._store.get(memory_id)
        if block is None or block.tenantId != tenant_id:
            return None
        return block

    def delete(self, memory_id: str, tenant_id: str) -> bool:
        block = self._store.get(memory_id)
        if block is None or block.tenantId != tenant_id:
            return False
        del self._store[memory_id]
        if memory_id in self._index.get(tenant_id, []):
            self._index[tenant_id].remove(memory_id)
        return True

    def search(self, filters: MemorySearchFilters) -> list[MemoryBlock]:
        results: list[MemoryBlock] = []
        ids = self._index.get(filters.tenantId, [])
        for mid in ids:
            block = self._store.get(mid)
            if block is None:
                continue

            if filters.sessionId and block.sessionId != filters.sessionId:
                continue
            if filters.minImportance is not None and block.importance.raw < filters.minImportance:
                continue
            if filters.maxImportance is not None and block.importance.raw > filters.maxImportance:
                continue
            if filters.emotions:
                if not any(c.lower() in [e.lower() for e in filters.emotions] for c in block.emotions.categories):
                    continue
            if filters.crisisOnly and not block.gating.crisisFlag:
                continue
            if filters.dateFrom and block.timestamp < filters.dateFrom:
                continue
            if filters.dateTo and block.timestamp > filters.dateTo:
                continue
            if filters.consolidationPhases and block.consolidation.phase not in filters.consolidationPhases:
                continue

            results.append(block)

        results.sort(key=lambda b: b.importance.raw, reverse=True)
        offset = filters.offset or 0
        limit = filters.limit or 50
        return results[offset : offset + limit]

    @property
    def total_count(self) -> int:
        return sum(len(ids) for ids in self._index.values())


# ─── Dependency factories (overridable in tests) ──────────────────────────────


def get_store() -> MemoryStore:
    """Factory for MemoryStore. Override via app.dependency_overrides in tests."""
    return MemoryStore()


def get_scorer() -> ImportanceScorer:
    """Factory for ImportanceScorer. Override via app.dependency_overrides in tests."""
    return ImportanceScorer.from_env()


def get_classifier() -> EmotionClassifier:
    """Factory for EmotionClassifier. Override via app.dependency_overrides in tests."""
    return EmotionClassifier(mode="lexicon")


# ─── FastAPI app ──────────────────────────────────────────────────────────────

app = FastAPI(
    title="Pixelated Memory API",
    description="Cross-language memory service for therapeutic AI — PIX-510 Sprint 1",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─── Helpers ──────────────────────────────────────────────────────────────────


def _id() -> str:
    return f"mem_{int(time.time_ns() / 1_000_000):x}"


def _now() -> int:
    return int(time.time_ns() / 1_000_000)


def _build_memory_block(
    input_data: MemoryWriteInput,
    emotions: MemoryEmotions | None,
    gating: MemoryGating | None,
    scorer: ImportanceScorer,
    classifier: EmotionClassifier,
) -> MemoryBlock:
    now = _now()

    if emotions is None:
        result = classifier.classify(input_data.content)
        emotions = MemoryEmotions(
            valence=result.valence,
            arousal=result.arousal,
            categories=result.categories,
        )
    if gating is None:
        crisis_keywords = ["suicide", "self-harm", "panic"]
        crisis = any(kw in input_data.content.lower() for kw in crisis_keywords)
        gating = MemoryGating(
            piiStatus=PIIStatus.ABSENT,
            crisisFlag=crisis,
            traumaIndicators=[],
            consentGate=ConsentGate.OPEN,
        )

    block = MemoryBlock(
        id=_id(),
        tenantId=input_data.tenantId,
        sessionId=input_data.sessionId,
        content=input_data.content,
        timestamp=now,
        importance=MemoryImportance(raw=0, recency=0, relevance=0, emotionalWeight=1, actionability=0.5),
        emotions=emotions,
        gating=gating,
        consolidation=MemoryConsolidation(
            phase=ConsolidationPhase.RAW,
            lastProcessed=now,
            remCycles=3,
            schemaReferences=[],
        ),
    )

    block.importance.raw = scorer.score(block)
    return block


# ─── Request/response models ──────────────────────────────────────────────────


class ScoreResponse(BaseModel):
    id: str
    importance: MemoryImportance
    components: dict[str, float]


class HealthResponse(BaseModel):
    status: str
    memory_count: int
    scorer_latency_ms: float
    classifier_latency_ms: float


# ─── Type aliases for injected dependencies ───────────────────────────────────

StoreDep = Annotated[MemoryStore, Depends(get_store)]
ScorerDep = Annotated[ImportanceScorer, Depends(get_scorer)]
ClassifierDep = Annotated[EmotionClassifier, Depends(get_classifier)]


# ─── Endpoints ────────────────────────────────────────────────────────────────


@app.get("/health", response_model=HealthResponse, tags=["health"])
async def health(store: StoreDep, scorer: ScorerDep, classifier: ClassifierDep):
    now = time.perf_counter()
    scorer.benchmark(100)
    scorer_ms = (time.perf_counter() - now) * 1000

    now = time.perf_counter()
    classifier.benchmark_latency("I feel anxious", n=100)
    classifier_ms = (time.perf_counter() - now) * 1000

    return HealthResponse(
        status="ok",
        memory_count=store.total_count,
        scorer_latency_ms=round(scorer_ms / 100, 4),
        classifier_latency_ms=round(classifier_ms / 100, 4),
    )


@app.post("/memories", response_model=MemoryBlock, status_code=status.HTTP_201_CREATED, tags=["memories"])
async def create_memory(
    input_data: MemoryWriteInput,
    store: StoreDep,
    scorer: ScorerDep,
    classifier: ClassifierDep,
    emotions: MemoryEmotions | None = None,
    gating: MemoryGating | None = None,
):
    """Create a new memory block. Auto-computes importance and emotion classification."""
    block = _build_memory_block(input_data, emotions, gating, scorer, classifier)
    return store.put(block)


@app.get("/memories", response_model=list[MemoryBlock], tags=["memories"])
async def search_memories(
    tenant_id: Annotated[str, Query(description="Tenant ID (required)")],
    store: StoreDep,
    session_id: Annotated[str | None, Query(description="Filter by session")] = None,
    min_importance: Annotated[float | None, Query(ge=0.0, le=1.0)] = None,
    max_importance: Annotated[float | None, Query(ge=0.0, le=1.0)] = None,
    emotions: Annotated[str | None, Query(description="Comma-separated emotion categories")] = None,
    crisis_only: Annotated[bool, Query()] = False,
    date_from: Annotated[int | None, Query(description="Unix ms")] = None,
    date_to: Annotated[int | None, Query(description="Unix ms")] = None,
    consolidation_phases: Annotated[
        str | None, Query(description="Comma-separated: raw,consolidated,archived,forgotten")
    ] = None,
    limit: Annotated[int, Query(ge=1, le=1000)] = 50,
    offset: Annotated[int, Query(ge=0)] = 0,
):
    filters = MemorySearchFilters(
        tenantId=tenant_id,
        sessionId=session_id,
        minImportance=min_importance,
        maxImportance=max_importance,
        emotions=[e.strip() for e in emotions.split(",")] if emotions else None,
        crisisOnly=crisis_only,
        dateFrom=date_from,
        dateTo=date_to,
        consolidationPhases=[ConsolidationPhase(p.strip()) for p in consolidation_phases.split(",")]
        if consolidation_phases
        else None,
        limit=limit,
        offset=offset,
    )
    return store.search(filters)


@app.get("/memories/{memory_id}", response_model=MemoryBlock, tags=["memories"])
async def get_memory(
    memory_id: str,
    tenant_id: Annotated[str, Query(description="Tenant ID for access control")],
    store: StoreDep,
):
    block = store.get(memory_id, tenant_id)
    if block is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Memory not found")
    return block


@app.patch("/memories/{memory_id}", response_model=MemoryBlock, tags=["memories"])
async def update_memory(
    memory_id: str,
    tenant_id: Annotated[str, Query(description="Tenant ID for access control")],
    store: StoreDep,
    scorer: ScorerDep,
    classifier: ClassifierDep,
    content: str | None = None,
    importance: float | None = None,
    consolidation_phase: ConsolidationPhase | None = None,
):
    block = store.get(memory_id, tenant_id)
    if block is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Memory not found")

    if content is not None:
        block.content = content
        result = classifier.classify(content)
        block.emotions = MemoryEmotions(
            valence=result.valence,
            arousal=result.arousal,
            categories=result.categories,
        )
        block.importance.raw = scorer.score(block)

    if importance is not None:
        block.importance.raw = min(max(importance, 0.0), 1.0)

    if consolidation_phase is not None:
        block.consolidation.phase = consolidation_phase

    block.consolidation.lastProcessed = _now()
    return store.put(block)


@app.delete("/memories/{memory_id}", status_code=status.HTTP_204_NO_CONTENT, tags=["memories"])
async def delete_memory(
    memory_id: str,
    tenant_id: Annotated[str, Query(description="Tenant ID for access control")],
    store: StoreDep,
):
    deleted = store.delete(memory_id, tenant_id)
    if not deleted:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Memory not found")


@app.post("/memories/score", response_model=ScoreResponse, tags=["scoring"])
async def score_memory(
    memory_id: str,
    tenant_id: Annotated[str, Query(description="Tenant ID for access control")],
    store: StoreDep,
    scorer: ScorerDep,
    context: Annotated[str, Query(description="Optional context for relevance scoring")] = "",
):
    block = store.get(memory_id, tenant_id)
    if block is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Memory not found")

    block.importance.raw = scorer.score(block, context)
    components = scorer.score_components(block, context)

    return ScoreResponse(
        id=block.id,
        importance=MemoryImportance(
            raw=components["raw"],
            recency=components["recency"],
            relevance=components["relevance"],
            emotionalWeight=components["emotionalWeight"],
            actionability=components["actionability"],
        ),
        components={
            "recency": components["recency"],
            "relevance": components["relevance"],
            "emotionalWeight": components["emotionalWeight"],
            "actionability": components["actionability"],
        },
    )


@app.get("/memories/trajectory/{session_id}", response_model=dict, tags=["trajectory"])
async def get_trajectory(
    session_id: str,
    tenant_id: Annotated[str, Query(description="Tenant ID for access control")],
    store: StoreDep,
    classifier: ClassifierDep,
    limit: Annotated[int, Query(ge=1, le=500)] = 50,
):
    results = store.search(
        MemorySearchFilters(
            tenantId=tenant_id,
            sessionId=session_id,
            limit=limit,
            offset=0,
        )
    )

    if not results:
        return {
            "sessionId": session_id,
            "trend": "stable",
            "crisisIndicators": [],
            "trajectory": [],
        }

    emotion_results = [classifier.classify(b.content) for b in results]

    traj = classifier.session_trajectory(emotion_results)

    return {
        "sessionId": session_id,
        "memoryCount": len(results),
        "trend": traj.trend,
        "crisisIndicators": traj.crisis_indicators,
        "maxIntensity": traj.max_intensity,
        "trajectory": [
            {
                "memoryId": b.id,
                "valence": r.valence,
                "arousal": r.arousal,
                "dominance": r.dominance,
            }
            for b, r in zip(results, emotion_results, strict=False)
        ],
    }


# ─── Entry point ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
