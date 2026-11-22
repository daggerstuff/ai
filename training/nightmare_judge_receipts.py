"""Reuse completed model judgments using text-free, content-bound receipts."""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Awaitable, Callable
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field

from training.generation_backend import resolve_backend
from training.nightmare_records import GateAcceptedSession

Digest = Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]


class JudgeEvidenceError(RuntimeError):
    """Judgments are unavailable, incomplete or inconsistent with their inputs."""


class Candidate(GateAcceptedSession):
    model_config = ConfigDict(frozen=True, extra="allow", hide_input_in_errors=True)
    id: str = Field(min_length=1)
    raw_content: str = ""

    def digest(self) -> str:
        data = json.dumps(self.model_dump(exclude_unset=True), sort_keys=True, ensure_ascii=False).encode()
        return hashlib.sha256(data).hexdigest()


class JudgeRecipe(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", hide_input_in_errors=True)
    backend: str
    model: str
    endpoint_sha256: Digest
    prompt_sha256: Digest
    accept_threshold: float = Field(ge=0, le=1, allow_inf_nan=False)
    source_sha256: tuple[Digest, ...]

    def digest(self) -> str:
        return hashlib.sha256(self.model_dump_json().encode()).hexdigest()


class JudgeReceipt(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", strict=True, hide_input_in_errors=True)
    schema_version: Literal[1] = 1
    recipe_sha256: Digest
    candidate_sha256: Digest
    accepted: bool


def current_recipe(*, async_judge: bool, prompt: str) -> JudgeRecipe:
    """Bind effective model, threshold, prompt and implementation, excluding credentials."""
    if async_judge:
        backend = "nemo"
        model = os.environ.get("NEMO_MODEL", "mistral-nemo")
        endpoint = os.environ.get("NEMO_ENDPOINT", "") or os.environ.get("NVIDIA_BASE_URL", "")
        threshold = float(os.environ.get("NF_ACCEPT_THRESHOLD", "0.6"))
    else:
        config = resolve_backend()
        backend, model, endpoint = config.name, config.model, config.url
        threshold = 0.8
    sources = (
        "nightmare_judge_receipts.py",
        "nightmare_records.py",
        "nightmare_fuel_generator.py",
        "generation_backend.py",
        "clinical_validity_judge.py",
        "clinical_validity_scorer.py",
        "clinical_validity_judge_async.py",
        "sdg_pipeline.py",
    )
    return JudgeRecipe(
        backend=backend,
        model=model,
        endpoint_sha256=hashlib.sha256(endpoint.encode()).hexdigest(),
        prompt_sha256=hashlib.sha256(prompt.encode()).hexdigest(),
        accept_threshold=threshold,
        source_sha256=tuple(
            hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest() for name in sources
        ),
    )


class JudgeReceipts:
    """Atomic per-candidate decisions for a single judge recipe; no dialogue stored."""

    def __init__(self, root: Path, recipe: JudgeRecipe) -> None:
        self.recipe_sha256 = recipe.digest()
        self.root = root / self.recipe_sha256

    def read(self, candidate: Candidate) -> bool | None:
        digest = candidate.digest()
        path = self.root / f"{digest}.json"
        if not path.exists():
            return None
        receipt = JudgeReceipt.model_validate_json(path.read_bytes())
        if receipt.recipe_sha256 != self.recipe_sha256 or receipt.candidate_sha256 != digest:
            raise JudgeEvidenceError("judge receipt identity differs from its expected path")
        return receipt.accepted

    def store(self, candidate: Candidate, *, accepted: bool) -> None:
        digest = candidate.digest()
        receipt = JudgeReceipt(recipe_sha256=self.recipe_sha256, candidate_sha256=digest, accepted=accepted)
        self.root.mkdir(parents=True, exist_ok=True, mode=0o700)
        temporary: Path | None = None
        try:
            with NamedTemporaryFile(dir=self.root, mode="w", delete=False) as stream:
                temporary = Path(stream.name)
                stream.write(receipt.model_dump_json() + "\n")
                stream.flush()
                os.fsync(stream.fileno())
            temporary.replace(self.root / f"{digest}.json")
            directory = os.open(self.root, os.O_RDONLY | os.O_DIRECTORY)
            try:
                os.fsync(directory)
            finally:
                os.close(directory)
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)


async def judge_with_receipts(
    candidates: tuple[Candidate, ...],
    receipts: JudgeReceipts,
    judge: Callable[[tuple[Candidate, ...]], Awaitable[tuple[str, ...]]],
) -> tuple[Candidate, ...]:
    """Judge only uncached candidates; publish decisions only after a complete batch."""
    unique: dict[str, Candidate] = {}
    decisions: dict[str, bool] = {}
    pending: list[Candidate] = []
    for candidate in candidates:
        if candidate.id in unique:
            if candidate.digest() != unique[candidate.id].digest():
                raise JudgeEvidenceError("conflicting candidates share an ID")
            continue
        unique[candidate.id] = candidate
        decision = receipts.read(candidate)
        if decision is None:
            pending.append(candidate)
        else:
            decisions[candidate.id] = decision
    if pending:
        accepted_ids = await judge(tuple(pending))
        accepted = frozenset(accepted_ids)
        if len(accepted) != len(accepted_ids) or not accepted <= {candidate.id for candidate in pending}:
            raise JudgeEvidenceError("final judge returned duplicate or unknown candidate IDs")
        for candidate in pending:
            decision = candidate.id in accepted
            receipts.store(candidate, accepted=decision)
            decisions[candidate.id] = decision
    return tuple(candidate for identifier, candidate in unique.items() if decisions[identifier])
