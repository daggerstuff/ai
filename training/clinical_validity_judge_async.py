"""Async wrapper around ClinicalValidityJudge.

Provides an asyncio-native evaluation path WITHOUT modifying the existing
synchronous `ClinicalValidityJudge.evaluate` / `.score` signatures, so the
existing test suite in `tests/test_clinical_validity_judge.py` keeps passing
unchanged.

Design:
- `AsyncClinicalJudge.evaluate(text, nemo_config)` delegates the existing
  synchronous `ClinicalValidityJudge.evaluate` to a thread executor. The sync
  judge internally calls `training.sdg_pipeline._call_nemo`, which is also
  synchronous — the executor keeps the call off the event loop and preserves
  the `training.sdg_pipeline._call_nemo` mock surface that the existing tests
  rely on.
- `AsyncJudgePipeline` is the producer/consumer queue architecture described
  by the Nightmare Fuel refactor brief: a producer puts candidates onto an
  `asyncio.Queue`, N configurable worker coroutines (default
  `NF_EVAL_CONCURRENCY=4`) drain the queue and call `AsyncClinicalJudge.evaluate`,
  and accepted results are written back onto a results queue. Throughput is
  tracked separately for generation (items enqueued/sec) and validation
  (items judged/sec) so the two stages can be monitored and tuned independently.

Public API (designed to be additive — no changes to ClinicalValidityJudge):

    class AsyncClinicalJudge:
        @classmethod
        async def evaluate(cls, text, nemo_config=None) -> dict
        @classmethod
        async def score(cls, text, nemo_config=None) -> float

    class AsyncJudgePipeline:
        def __init__(self, nemo_config, max_workers=4, accept_threshold=0.6)
        async def run(self, candidates: AsyncIterator[tuple[Any, str]]) -> PipelineResult
        @property
        def metrics(self) -> dict
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import Any, AsyncIterator, Optional

from training.clinical_validity_judge import ClinicalValidityJudge

logger = logging.getLogger("clinical_validity_judge_async")


def _default_concurrency() -> int:
    """Read NF_EVAL_CONCURRENCY from env, default 4. Bounded to [1, 64]."""
    import os

    raw = os.getenv("NF_EVAL_CONCURRENCY", "4")
    try:
        n = int(raw)
    except (TypeError, ValueError):
        return 4
    return max(1, min(64, n))


def _default_accept_threshold() -> float:
    """Read NF_ACCEPT_THRESHOLD from env, fall back to ClinicalValidityJudge.ACCEPT_THRESHOLD."""
    import os

    raw = os.getenv("NF_ACCEPT_THRESHOLD")
    if raw is None or raw == "":
        return ClinicalValidityJudge.ACCEPT_THRESHOLD
    try:
        return max(0.0, min(1.0, float(raw)))
    except (TypeError, ValueError):
        return ClinicalValidityJudge.ACCEPT_THRESHOLD


class AsyncClinicalJudge:
    """Async front for `ClinicalValidityJudge` that preserves sync signatures.

    The synchronous judge is invoked from inside a thread executor so the
    event loop stays free while `_call_nemo` does its blocking HTTP request.
    Any mocking of `training.sdg_pipeline._call_nemo` continues to work because
    the patched symbol is resolved lazily inside `_call_judge` at call time.
    """

    @classmethod
    async def evaluate(
        cls,
        text: str | None,
        nemo_config: Any | None = None,
    ) -> dict[str, Any]:
        """Async-identical to ClinicalValidityJudge.evaluate."""
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            None,
            lambda: ClinicalValidityJudge.evaluate(text, nemo_config),
        )

    @classmethod
    async def score(
        cls,
        text: str | None,
        nemo_config: Any | None = None,
    ) -> float:
        """Async-identical to ClinicalValidityJudge.score.

        ``text`` is normalized upstream of the sync ``ClinicalValidityJudge.score``
        call since that method's type hint disallows ``None`` (the underlying
        ``evaluate`` accepts ``None`` and returns a 0.0-score dict).
        """
        loop = asyncio.get_running_loop()
        normalized = text if text is not None else ""
        return await loop.run_in_executor(
            None,
            lambda: ClinicalValidityJudge.score(normalized, nemo_config),
        )


@dataclass
class PipelineMetrics:
    """Throughput counters split by stage.

    Fields:
        generated: Number of candidates enqueued by the producer.
        evaluated: Number of candidates that finished the judge.
        accepted: Number of candidates whose validity_score >= threshold.
        rejected: Number of candidates below threshold.
        errors: Number of candidates whose evaluation raised.
        gen_throughput: Items enqueued per second during the run.
        eval_throughput: Items judged per second during the run.
        wall_seconds: Total wall time.
    """

    generated: int = 0
    evaluated: int = 0
    accepted: int = 0
    rejected: int = 0
    errors: int = 0
    gen_throughput: float = 0.0
    eval_throughput: float = 0.0
    wall_seconds: float = 0.0

    def as_dict(self) -> dict[str, Any]:
        return {
            "generated": self.generated,
            "evaluated": self.evaluated,
            "accepted": self.accepted,
            "rejected": self.rejected,
            "errors": self.errors,
            "gen_throughput": round(self.gen_throughput, 3),
            "eval_throughput": round(self.eval_throughput, 3),
            "wall_seconds": round(self.wall_seconds, 3),
        }


@dataclass
class PipelineResult:
    """Output of `AsyncJudgePipeline.run`."""

    accepted: list[dict[str, Any]] = field(default_factory=list)
    rejected: list[dict[str, Any]] = field(default_factory=list)
    metrics: PipelineMetrics = field(default_factory=PipelineMetrics)


class AsyncJudgePipeline:
    """Producer/consumer queue that judges Nightmare Fuel candidates in parallel.

    The pipeline maintains two counters that allow callers and operators to
    distinguish generation throughput from validation throughput — two
    bottlenecks that previously collapsed behind a single sync call.

    Usage:

        pipeline = AsyncJudgePipeline(nemo_config, max_workers=4)
        result = await pipeline.run(candidate_iter)
        result.metrics.as_dict()  # generation vs validation throughput

    Concurrency is configurable per-pipeline (`max_workers` constructor arg)
    and per-environment (`NF_EVAL_CONCURRENCY`, default 4, bounded [1, 64]).
    Construction-time arg takes precedence over the env var.
    """

    def __init__(
        self,
        nemo_config: Any | None,
        max_workers: Optional[int] = None,
        accept_threshold: Optional[float] = None,
        queue_maxsize: int = 0,
    ):
        self._nemo_config = nemo_config
        self._max_workers = max_workers if max_workers is not None else _default_concurrency()
        if self._max_workers < 1:
            raise ValueError("max_workers must be >= 1")
        self._accept_threshold = accept_threshold if accept_threshold is not None else _default_accept_threshold()
        self._queue_maxsize = queue_maxsize
        # Live metrics mutated from multiple coroutines under GIL; asyncio
        # guarantees single-threaded scheduling so plain integers are safe.
        self.metrics = PipelineMetrics()

    async def run(
        self,
        candidates: AsyncIterator[tuple[Any, str]],
    ) -> PipelineResult:
        """Judge every candidate produced by `candidates`.

        `candidates` is an async generator yielding `(case_id, text)` tuples.
        Returns accepted/rejected cases split plus full stage metrics.
        """
        queue: asyncio.Queue[tuple[Any, str] | None] = asyncio.Queue(self._queue_maxsize)
        results: asyncio.Queue[dict[str, Any]] = asyncio.Queue()

        gen_started = time.monotonic()
        eval_started_or_first: float | None = None

        async def producer() -> None:
            nonlocal gen_started
            produced = 0
            try:
                async for case_id, text in candidates:
                    await queue.put((case_id, text))
                    produced += 1
            finally:
                # One sentinel per worker signals completion.
                for _ in range(self._max_workers):
                    await queue.put(None)
            gen_seconds = max(1e-9, time.monotonic() - gen_started)
            self.metrics.generated = produced
            self.metrics.gen_throughput = produced / gen_seconds

        async def worker() -> None:
            nonlocal eval_started_or_first
            while True:
                item = await queue.get()
                queue.task_done()
                if item is None:
                    break
                case_id, text = item
                if eval_started_or_first is None:
                    eval_started_or_first = time.monotonic()
                try:
                    result = await AsyncClinicalJudge.evaluate(text, self._nemo_config)
                    out = {"case_id": case_id, "text": text, "eval": result}
                    score = float(result.get("validity_score", 0.0))
                    self.metrics.evaluated += 1
                    if score >= self._accept_threshold:
                        self.metrics.accepted += 1
                        out["accepted"] = True
                    else:
                        self.metrics.rejected += 1
                        out["accepted"] = False
                    await results.put(out)
                except Exception:
                    self.metrics.errors += 1
                    logger.exception("AsyncJudgePipeline worker error judging case_id=%r", case_id)

        wall_start = time.monotonic()
        producer_task = asyncio.create_task(producer())
        worker_tasks = [asyncio.create_task(worker()) for _ in range(self._max_workers)]
        await producer_task
        # Drain the queue of all candidate items so workers can receive sentinels.
        await queue.join()
        await asyncio.gather(*worker_tasks, return_exceptions=False)

        accepted: list[dict[str, Any]] = []
        rejected: list[dict[str, Any]] = []
        while not results.empty():
            out = results.get_nowait()
            (accepted if out.get("accepted") else rejected).append(out)

        wall = time.monotonic() - wall_start
        self.metrics.wall_seconds = wall
        if eval_started_or_first is not None:
            eval_seconds = max(1e-9, time.monotonic() - eval_started_or_first)
            self.metrics.eval_throughput = self.metrics.evaluated / eval_seconds
        return PipelineResult(accepted=accepted, rejected=rejected, metrics=self.metrics)


__all__ = [
    "AsyncClinicalJudge",
    "AsyncJudgePipeline",
    "PipelineResult",
    "PipelineMetrics",
]
