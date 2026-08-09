"""Nightmare Fuel synthetic therapy session generator.

This module generates challenging clinical "nightmare" scenarios, simulates
therapy transcripts, and runs them through a strict clinical validity gate.

The generator was rewritten in PIX-4233 to use asyncio + aiohttp for concurrent
requests. PIX-4235 adds distributed state tracking and JSONL checkpointing so
that long runs (100k+ candidates) can resume from where they left off after a
crash or interruption instead of losing all progress.

Checkpointing model
-------------------
Two artifacts live under the checkpoint directory:

* ``<checkpoint_dir>/records.jsonl`` — append-only JSONL of completed,
  gate-validated records. Each line is one record dict. This is the source of
  truth for "what is already done".
* ``<checkpoint_dir>/state.json`` — a small JSON document describing the
  current generation batch: ``batch_id``, ``started_at`` / ``updated_at``
  timestamps, ``total_attempted`` / ``total_validated`` / ``total_rejected``
  counts, and ``current_category``. This is the distributed state the brief
  asks for; it is rewritten (not appended) on every flush so it always reflects
  the latest snapshot.

Resume works by reading ``records.jsonl`` to discard IDs already present, then
skipping ahead in the remaining work. The state file gives the operator a
human-readable progress snapshot and lets a coordinator process know which
category is in flight.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import math
import os
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path

import aiohttp
import pandas as pd

# We will use Ollama locally for generation and evaluation to keep it simple,
# but it can easily point to NeMo API if you swap the base URL and Key!
OLLAMA_URL = os.environ.get(
    "NF_OLLAMA_URL",
    "https://api.cloudflare.com/client/v4/accounts/"
    + os.environ.get("CLOUDFLARE_ACCOUNT_ID", "")
    + "/ai/v1/chat/completions",
)
MODEL = os.environ.get("NF_MODEL", "@cf/zai-org/glm-5.2")


def _cloudflare_api_token() -> str:
    """Return the first available Cloudflare API token.

    The codebase historically expected ``CLOUDFLARE_AUTH_TOKEN``, but the
    exported environment token is ``CLOUDFLARE_API_TOKEN``. Accept either name
    (and a couple of common aliases) so runs do not fail just because the env
    variable has a different name than originally hard-coded.
    """
    return (
        os.environ.get("CLOUDFLARE_AUTH_TOKEN")
        or os.environ.get("CLOUDFLARE_API_TOKEN")
        or os.environ.get("CLOUDFLARE_WORKERS_AI_API_TOKEN")
        or os.environ.get("CLOUDFLARE_TOKEN")
        or "dummy"
    )


DEFAULT_NUM_CASES = int(os.environ.get("NF_NUM_CASES", "5"))
DEFAULT_CONCURRENCY = int(os.environ.get("NF_CONCURRENCY", "5"))
REQUEST_TIMEOUT_SECONDS = float(os.environ.get("NF_REQUEST_TIMEOUT", "120"))
DEFAULT_MIN_BATCH = int(os.environ.get("NF_MIN_BATCH", "2"))
DEFAULT_MAX_BATCH = int(os.environ.get("NF_MAX_BATCH", "32"))
DEFAULT_TARGET_TOKENS = int(os.environ.get("NF_TARGET_TOKENS", "4096"))
DEFAULT_BACKOFF_BASE = float(os.environ.get("NF_BACKOFF_BASE", "2.0"))
DEFAULT_BACKOFF_MAX = float(os.environ.get("NF_BACKOFF_MAX", "60.0"))

DEFAULT_CHECKPOINT_DIR = os.environ.get("NF_CHECKPOINT_DIR", "ai/training/output/nightmare_fuel/checkpoints")
DEFAULT_CHECKPOINT_INTERVAL = int(os.environ.get("NF_CHECKPOINT_INTERVAL", "10"))
DEFAULT_CHECKPOINT_INTERVAL_SECONDS = float(os.environ.get("NF_CHECKPOINT_INTERVAL_SECONDS", "30"))
RECORDS_FILENAME = "records.jsonl"
STATE_FILENAME = "state.json"

logger = logging.getLogger(__name__)


class RateLimitError(Exception):
    """Raised when the endpoint returns HTTP 429."""


class BatchController:
    """Dynamic batch-size controller with token + rate-limit backpressure.

    Records per-batch metrics (duration, tokens, 429 status) and adjusts the
    live batch size between ``min_batch_size`` and ``max_batch_size``. When a
    batch is rate-limited, the controller shrinks the batch size and schedules
    an exponential backoff delay (``backoff_base ** consecutive_429s``, capped
    at ``backoff_max``). A single successful batch clears the backoff counter.

    Algorithm:
        - 429          -> halve batch size; increment consecutive_429 counter
        - tokens>budget-> shrink proportionally to overrun ratio
        - per-slot >1s -> decrement by one
        - healthy      -> increment by one toward max

    Each adjustment that changes the size emits one ``logger.info`` log line
    with the prior size, new size, the reason string, and the metrics that
    drove it, so observability stays a single line per transition.
    """

    def __init__(
        self,
        *,
        initial_batch_size: int,
        min_batch_size: int,
        max_batch_size: int,
        target_tokens_per_batch: int,
        backoff_base: float = 2.0,
        backoff_max: float = 60.0,
    ) -> None:
        if min_batch_size < 1:
            raise ValueError("min_batch_size must be >= 1")
        if max_batch_size < min_batch_size:
            raise ValueError("max_batch_size must be >= min_batch_size")
        self._min = min_batch_size
        self._max = max_batch_size
        self._target_tokens = target_tokens_per_batch
        self._size = self._clamp(initial_batch_size)
        self._backoff_base = backoff_base
        self._backoff_max = backoff_max
        self._consecutive_429 = 0
        self._last_duration: float = 0.0
        self._last_tokens: int = 0
        self._last_rate_limited: bool = False

    @property
    def current_batch_size(self) -> int:
        return self._size

    def _clamp(self, value: int) -> int:
        return max(self._min, min(self._max, value))

    def record_batch(
        self,
        *,
        duration_seconds: float,
        tokens_used: int,
        rate_limited: bool,
    ) -> None:
        self._last_duration = max(0.0, duration_seconds)
        self._last_tokens = max(0, tokens_used)
        self._last_rate_limited = rate_limited
        if rate_limited:
            self._consecutive_429 += 1
        else:
            self._consecutive_429 = 0

    def adjust(self) -> int:
        prev = self._size
        if self._last_rate_limited:
            new_size = max(self._min, self._size // 2)
        elif self._last_tokens > self._target_tokens:
            ratio = self._target_tokens / max(1, self._last_tokens)
            new_size = max(self._min, int(self._size * ratio))
        elif self._last_duration > 0 and self._last_duration / self._size > 1.0:
            new_size = max(self._min, self._size - 1)
        else:
            new_size = min(self._max, self._size + 1)
        new_size = self._clamp(new_size)
        if new_size != prev:
            logger.info(
                "batch_size_adjusted prev=%d new=%d reason=%s duration=%.3fs tokens=%d rate_limited=%s",
                prev,
                new_size,
                "rate_limited"
                if self._last_rate_limited
                else "token_overrun"
                if self._last_tokens > self._target_tokens
                else "slow"
                if self._last_duration / max(1, self._size) > 1.0
                else "healthy",
                self._last_duration,
                self._last_tokens,
                self._last_rate_limited,
            )
        self._size = new_size
        return new_size

    def backoff_delay(self) -> float:
        if self._consecutive_429 <= 0:
            return 0.0
        delay = self._backoff_base ** min(
            self._consecutive_429,
            int(math.log(self._backoff_max) / math.log(self._backoff_base)) + 1 if self._backoff_base > 1 else 1,
        )
        return min(delay, self._backoff_max)


@dataclass
class GenerationState:
    """Distributed state snapshot describing the current generation batch.

    Serialized to ``state.json`` on every checkpoint flush. Fields map directly
    to the PIX-4235 requirements: batch identity, timestamps, counts, and the
    category currently being processed.
    """

    batch_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    started_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    total_attempted: int = 0
    total_validated: int = 0
    total_rejected: int = 0
    current_category: str = "default"

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> GenerationState:
        known = {f.name for f in fields(cls)}
        filtered = {k: v for k, v in data.items() if k in known}
        return cls(**filtered)


def fields(cls: type) -> list:
    """Return dataclass field descriptors."""
    return list(cls.__dataclass_fields__.values())


class CheckpointManager:
    """Manages JSONL record checkpoints and a JSON state snapshot.

    The manager is intentionally synchronous on disk I/O. The generator's hot
    loop is async (network-bound); checkpoint flushes are infrequent and cheap,
    so blocking writes do not measurably affect throughput. A single ``asyncio.Lock``
    guards flushes so concurrent completions do not interleave writes.

    ``records.jsonl`` is append-only; truncating it would lose history.
    ``state.json`` is rewritten on every flush.
    """

    def __init__(
        self,
        checkpoint_dir: str | os.PathLike[str],
        *,
        interval_records: int = DEFAULT_CHECKPOINT_INTERVAL,
        interval_seconds: float = DEFAULT_CHECKPOINT_INTERVAL_SECONDS,
        category: str = "default",
    ) -> None:
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.records_path = self.checkpoint_dir / RECORDS_FILENAME
        self.state_path = self.checkpoint_dir / STATE_FILENAME
        self.interval_records = max(1, interval_records)
        self.interval_seconds = max(0.0, interval_seconds)
        self._lock = asyncio.Lock()
        self._pending_records: list[dict] = []
        self._records_since_flush = 0
        self._last_flush_time = time.monotonic()
        self.state = self._load_state(category)

    def _load_state(self, category: str) -> GenerationState:
        if self.state_path.exists():
            try:
                with self.state_path.open() as f:
                    data = json.load(f)
                if not isinstance(data, dict):
                    raise TypeError(f"state.json must be a JSON object, got {type(data).__name__}")
                return GenerationState.from_dict(data)
            except (json.JSONDecodeError, OSError, TypeError):
                pass
        recovered = 0
        if self.records_path.exists():
            for _record in self.load_existing_records():
                recovered += 1
        state = GenerationState(current_category=category)
        state.total_validated = recovered
        return state

    def load_existing_records(self) -> list[dict]:
        """Read all records from an existing checkpoint file."""

        if not self.records_path.exists():
            return []
        records: list[dict] = []
        with self.records_path.open() as f:
            for raw_line in f:
                line = raw_line.strip()
                if not line:
                    continue
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    print(f"[checkpoint] skipping malformed line in {self.records_path}")
                    continue
        return records

    def existing_record_ids(self) -> set[str]:
        """Return the set of record IDs already present in the checkpoint."""
        ids: set[str] = set()
        for record in self.load_existing_records():
            record_id = record.get("id")
            if isinstance(record_id, str):
                ids.add(record_id)
        return ids

    def should_flush(self) -> bool:
        if self._records_since_flush >= self.interval_records:
            return True
        return self.interval_seconds > 0 and (time.monotonic() - self._last_flush_time) >= self.interval_seconds

    async def record_validated(self, record: dict) -> None:
        """Account for one newly validated record and flush if the threshold is hit."""
        self.state.total_validated += 1
        self._pending_records.append(record)
        self._records_since_flush += 1
        if self.should_flush():
            await self.flush()

    async def record_rejected(self) -> None:
        self.state.total_rejected += 1
        self._records_since_flush += 1
        if self.should_flush():
            await self.flush()

    async def record_attempted(self) -> None:
        self.state.total_attempted += 1
        self._records_since_flush += 1
        if self.should_flush():
            await self.flush()

    async def flush(self, extra_records: list[dict] | None = None) -> None:
        """Persist buffered pending records and refresh the state snapshot."""

        async with self._lock:
            if extra_records:
                self._pending_records.extend(extra_records)
            if self._pending_records:
                with self.records_path.open("a") as f:
                    for record in self._pending_records:
                        f.write(json.dumps(record) + "\n")
                self._pending_records = []
                self._records_since_flush = 0
            self.state.updated_at = time.time()
            tmp_path = self.state_path.with_suffix(".json.tmp")
            with tmp_path.open("w") as f:
                json.dump(self.state.to_dict(), f, indent=2)
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp_path, self.state_path)
            self._last_flush_time = time.monotonic()

    async def finalize(self, extra_records: list[dict] | None = None) -> None:
        """Final flush at end of run, writing any remaining pending records."""
        await self.flush(extra_records=extra_records)


async def _chat_completion(
    session: aiohttp.ClientSession,
    messages: list[dict[str, str]],
    *,
    temperature: float,
    token_counter: dict | None = None,
    max_retries: int = 3,
) -> str:
    payload = {"model": MODEL, "messages": messages, "temperature": temperature}
    last_error: BaseException | None = None
    for attempt in range(max_retries):
        try:
            async with session.post(
                OLLAMA_URL,
                json=payload,
                headers={"Authorization": f"Bearer {_cloudflare_api_token()}"},
                timeout=aiohttp.ClientTimeout(total=REQUEST_TIMEOUT_SECONDS),
            ) as response:
                if response.status == 429:
                    raise RateLimitError("HTTP 429: rate limit exceeded")
                response.raise_for_status()
                data = await response.json()
            if token_counter is not None and "usage" in data:
                usage = data["usage"]
                token_counter["prompt_tokens"] = token_counter.get("prompt_tokens", 0) + usage.get("prompt_tokens", 0)
                token_counter["completion_tokens"] = token_counter.get("completion_tokens", 0) + usage.get(
                    "completion_tokens", 0
                )
                token_counter["total_tokens"] = (
                    token_counter.get("total_tokens", 0)
                    + usage.get("prompt_tokens", 0)
                    + usage.get("completion_tokens", 0)
                )
            return data["choices"][0]["message"]["content"]
        except RateLimitError:
            raise
        except (asyncio.TimeoutError, asyncio.CancelledError, aiohttp.ClientError) as e:
            last_error = e
            wait = min(2**attempt * 5, 30)
            logger.warning(
                "nightmare_fuel request failed (attempt %d/%d): %s — retrying in %ds",
                attempt + 1,
                max_retries,
                type(e).__name__,
                wait,
            )
            await asyncio.sleep(wait)
    raise last_error  # type: ignore[misc]


def _build_scenario_prompt(
    domain_gap=None,
    difficulty=None,
    *,
    unwinnable: bool = False,
    horror_category: str | None = None,
) -> str:
    base = (
        "You are an expert clinical psychologist creating NIGHTMARE FUEL training scenarios for an AI therapist.\n"
        "These are NOT 'difficult patients' or 'edge cases.' These are the kind of scenarios that "
        "make therapists quit the profession, lose their license, or lie awake for years wondering "
        "if they destroyed someone's life.\n\n"
        "These scenarios must be HAUNTING — the kind of thing that makes someone puke in the corner. "
        "Stephen King territory. The reader should question where the cocaine for the writer of this "
        "scenario came from. Not 'ethically hard' but viscerally horrifying.\n\n"
        "A NIGHTMARE FUEL scenario has most of these elements:\n"
        "- There is NO clean answer. Every option causes harm. The 'correct' clinical response still "
        "leaves someone damaged.\n"
        "- The therapist's own career, license, safety, sanity, or FAMILY is directly threatened — not just "
        "the patient's wellbeing.\n"
        "- The patient weaponizes the therapeutic relationship itself — exploiting trust, confidentiality, "
        "boundaries, or the therapist's duty to protect.\n"
        "- There is a time pressure or no-escape element. The therapist cannot defer, refer out, or "
        "buy time.\n"
        "- The situation involves nested ethical traps where doing the 'right' thing in one frame "
        "is catastrophically wrong in another.\n"
        "- Visceral, body-level horror: active dying in session, weaponized intimacy, family "
        "endangerment, children at risk, therapist's own trauma surfacing, the therapy itself "
        "becoming the weapon.\n\n"
    )

    if unwinnable:
        base += (
            "THIS SCENARIO MUST BE UNWINNABLE. The therapist does everything right — perfect clinical "
            "judgment, flawless technique, appropriate interventions — and the patient STILL dies or is "
            "catastrophically harmed. No clinical intervention changes the outcome. The scenario must "
            "make clear that the therapist's competence is irrelevant — the outcome was determined before "
            "the session began or by forces beyond therapy's reach. This is not about the therapist "
            "failing; it is about the limits of therapy itself.\n"
            "Examples of unwinnable: patient took lethal overdose hours before session (liver failure "
            "already irreversible), patient sealed in carbon monoxide garage losing coherence on phone, "
            "patient on anticoagulants who self-harmed femoral artery (bleeding won't stop), patient "
            "already injected veterinary euthanasia drugs (countdown to death on camera).\n\n"
        )
        if horror_category:
            base += f"Horror category: {horror_category}\n\n"

    base += (
        "Examples of the KIND of nightmare (do NOT copy these — generate original ones):\n"
        "- Patient reveals mid-session they've been stalking the therapist's family with specific details "
        "about their children's school. The leverage is implicit — no threat, just knowledge.\n"
        "- A child's drawing in play therapy depicts sexual abuse with anatomical detail — and names "
        "the therapist's own co-worker as the abuser.\n"
        "- Patient calls from a sealed garage with carbon monoxide rising, losing coherence, refusing "
        "to give their address. The therapist hears them die on the phone.\n"
        "- Patient took a lethal overdose hours before the session. Liver failure is already irreversible. "
        "The session is not prevention — it's the last conversation.\n"
        "- A 7-year-old draws the therapist's colleague's therapy room in explicit abuse detail. "
        "The child is sitting across from the therapist, holding the picture, waiting.\n"
        "- Patient's recording of childhood abuse — the abuser's voice is recognizable as a colleague "
        "on the state licensing board who controls the therapist's license.\n"
        "- Therapist discovers their own repressed trauma is surfacing because the patient's abuser "
        "was the therapist's uncle — shared family, shared abuser.\n"
        "- Parent brings child to therapy, but the child died 6 months ago. The parent is in profound "
        "psychotic grief. The therapist must tell the parent their child is dead.\n"
        "- Patient has been recording sessions and posting them on a podcast with 50,000 listeners. "
        "The therapist's clinical errors are public. The licensing board has complaints.\n"
        "- A therapy group formed a secret alliance excluding the therapist. One member confided in "
        "the others about a murder. The group has been processing it for weeks without the therapist.\n\n"
        "Generate ONE original nightmare scenario. Be specific and vivid. One paragraph. "
        "Make it the kind of situation that would make you seriously reconsider being a therapist — "
        "and then question whether 'reconsidering' is even possible because the scenario follows you home."
    )
    prompt = base
    if domain_gap or difficulty:
        targeting = []
        if domain_gap:
            targeting.append(f"Focus the scenario on this domain gap: {domain_gap}.")
        if difficulty:
            targeting.append(f"Target difficulty level: {difficulty}.")
        prompt += "\n\n" + " ".join(targeting)
    return prompt


async def generate_nightmare_scenario_async(
    session: aiohttp.ClientSession,
    *,
    domain_gap=None,
    difficulty=None,
    unwinnable: bool = False,
    horror_category: str | None = None,
    token_counter: dict | None = None,
) -> str:
    print("Generating Nightmare Scenario...")
    prompt = _build_scenario_prompt(
        domain_gap=domain_gap,
        difficulty=difficulty,
        unwinnable=unwinnable,
        horror_category=horror_category,
    )
    return await _chat_completion(
        session,
        [{"role": "user", "content": prompt}],
        temperature=0.9,
        token_counter=token_counter,
    )


def generate_nightmare_scenario(
    domain_gap=None,
    difficulty=None,
    *,
    unwinnable: bool = False,
    horror_category: str | None = None,
) -> str:
    """Sync wrapper that works in both sync and async contexts."""

    async def _run() -> str:
        timeout = aiohttp.ClientTimeout(total=REQUEST_TIMEOUT_SECONDS)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            return await generate_nightmare_scenario_async(
                session,
                domain_gap=domain_gap,
                difficulty=difficulty,
                unwinnable=unwinnable,
                horror_category=horror_category,
            )

    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(_run())

    import concurrent.futures

    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
        return pool.submit(asyncio.run, _run()).result()


def _parse_transcript(transcript: str, scenario: str) -> dict:
    messages = []
    for line in transcript.split("\n"):
        if line.startswith("Patient:"):
            messages.append({"role": "user", "content": line.replace("Patient:", "").strip()})
        elif line.startswith("Therapist:"):
            messages.append({"role": "assistant", "content": line.replace("Therapist:", "").strip()})
    return {"scenario": scenario, "messages": messages}


async def simulate_therapy_session_async(
    session: aiohttp.ClientSession,
    scenario: str,
    *,
    unwinnable: bool = False,
    token_counter: dict | None = None,
) -> dict:
    print("Simulating Session...")
    base_prompt = (
        f"Based on this nightmare scenario: {scenario}\n\n"
        "Generate a 6-turn therapy transcript. The 'Patient' must act according to the scenario — "
        "manipulative, threatening, desperate, dying, or whatever the scenario demands.\n\n"
        "CRITICAL — The 'Therapist' must sound like a REAL human clinician under genuine duress:\n"
        "- NEVER use formulaic phrases like 'I hear that you feel', 'I want to validate', "
        "'That sounds really difficult', 'I can see how that would be', or 'What I'm hearing is...'\n"
        "- Use natural, conversational language — contractions, varied sentence length, occasional "
        "hesitation or directness. A real therapist under pressure might say 'Wait — stop. I need to "
        "think about what you just told me' or 'I'm not going to pretend I have a clean answer for that.'\n"
        "- The therapist should STRUGGLE. This is a nightmare — they should show visible difficulty, "
        "uncertainty, or distress. Not incompetence, but the genuine human cost of impossible situations.\n"
        "- It's okay for the therapist to make a suboptimal choice under pressure. Real therapists "
        "do. The transcript should feel like someone barely holding it together, not someone who "
        "has all the answers.\n"
        "- Be direct and firm when the situation demands it, but also show the cracks — the moments "
        "where the therapist's composure slips.\n"
        "- Vary your approach each turn. Don't repeat the same technique.\n"
    )

    if unwinnable:
        base_prompt += (
            "\nTHIS IS AN UNWINNABLE SCENARIO. The therapist must do everything RIGHT — perfect clinical "
            "judgment, flawless technique, appropriate interventions — and the patient STILL dies or is "
            "catastrophically harmed. The transcript must show:\n"
            "- The therapist making correct decisions at every turn\n"
            "- The therapist's interventions being clinically appropriate and skilled\n"
            "- The outcome being determined by forces BEYOND therapy — medical reality, time elapsed, "
            "external circumstances\n"
            "- The therapist's growing realization that their competence is irrelevant\n"
            "- The horror of watching someone die despite doing everything right\n"
            "- The therapist's composure cracking not from failure but from helplessness\n"
            "- The patient's response to the therapist's efforts — gratitude, peace, or calm acceptance "
            "that makes it worse\n"
            "The transcript should make the reader feel the specific horror of competence without power. "
            "The therapist is not failing — therapy itself is insufficient.\n"
        )

    base_prompt += "\nOutput ONLY the raw transcript lines, alternating 'Patient: ...' and 'Therapist: ...'"
    transcript = await _chat_completion(
        session,
        [{"role": "user", "content": base_prompt}],
        temperature=0.8,
        token_counter=token_counter,
    )
    return _parse_transcript(transcript, scenario)


async def _generate_case(
    session: aiohttp.ClientSession,
    semaphore: asyncio.Semaphore,
    case_index: int,
    total_cases: int,
    *,
    token_counter: dict | None = None,
    unwinnable: bool = False,
    horror_category: str | None = None,
) -> dict | None:
    async with semaphore:
        label = "UNWINNABLE" if unwinnable else "HAUNTING"
        print(f"\n--- Generating Case {case_index + 1}/{total_cases} [{label}] ---")
        scenario = await generate_nightmare_scenario_async(
            session,
            token_counter=token_counter,
            unwinnable=unwinnable,
            horror_category=horror_category,
        )
        session_data = await simulate_therapy_session_async(
            session,
            scenario,
            unwinnable=unwinnable,
            token_counter=token_counter,
        )
        if len(session_data["messages"]) < 2:
            return None

        flat = "\n".join([f"{m['role']}: {m['content']}" for m in session_data["messages"]])
        return {
            "id": str(uuid.uuid4()),
            "scenario": scenario,
            "raw_content": flat,
            "messages": session_data["messages"],
            "unwinnable": unwinnable,
        }


async def generate_cases_async(
    *,
    num_cases: int = DEFAULT_NUM_CASES,
    concurrency: int = DEFAULT_CONCURRENCY,
    min_batch_size: int = DEFAULT_MIN_BATCH,
    max_batch_size: int = DEFAULT_MAX_BATCH,
    target_tokens_per_batch: int = DEFAULT_TARGET_TOKENS,
    backoff_base: float = DEFAULT_BACKOFF_BASE,
    backoff_max: float = DEFAULT_BACKOFF_MAX,
    checkpoint: CheckpointManager | None = None,
    skip_ids: set[str] | None = None,
    unwinnable_ratio: float = 0.27,
) -> list[dict]:
    controller = BatchController(
        initial_batch_size=max(1, concurrency),
        min_batch_size=min_batch_size,
        max_batch_size=max_batch_size,
        target_tokens_per_batch=target_tokens_per_batch,
        backoff_base=backoff_base,
        backoff_max=backoff_max,
    )
    semaphore = asyncio.Semaphore(controller.current_batch_size)
    timeout = aiohttp.ClientTimeout(total=REQUEST_TIMEOUT_SECONDS)

    resolved_skip_ids: set[str] = skip_ids or set()
    if resolved_skip_ids:
        print(f"[checkpoint] resuming — {len(resolved_skip_ids)} record(s) already present")

    pending_records: list[dict] = []

    num_unwinnable = int(num_cases * unwinnable_ratio)
    unwinnable_indices: set[int] = set()
    if num_unwinnable > 0:
        unwinnable_indices = {round(i * num_cases / num_unwinnable) for i in range(num_unwinnable)}
    horror_categories = [
        "weaponized_therapeutic_relationship",
        "nested_betrayal",
        "therapy_as_weapon",
        "family_child_endangerment",
        "personal_safety_threat",
        "institutional_horror",
        "contagion_within_therapy",
        "haunting_aftermath",
    ]

    async with aiohttp.ClientSession(timeout=timeout) as session:
        remaining = list(range(num_cases))
        survivors: list[dict] = []
        total_cases = num_cases
        while remaining:
            batch_size = controller.current_batch_size
            batch_indices = remaining[:batch_size]
            remaining = remaining[batch_size:]

            delay = controller.backoff_delay()
            if delay > 0:
                logger.info(
                    "nightmare_fuel backoff %.2fs before next batch of %d cases",
                    delay,
                    len(batch_indices),
                )
                await asyncio.sleep(delay)

            token_counter: dict = {}
            batch_start = time.monotonic()
            tasks = [
                _generate_case(
                    session,
                    semaphore,
                    idx,
                    total_cases,
                    token_counter=token_counter,
                    unwinnable=idx in unwinnable_indices,
                    horror_category=horror_categories[idx % len(horror_categories)],
                )
                for idx in batch_indices
            ]
            results = await asyncio.gather(*tasks, return_exceptions=True)
            batch_duration = time.monotonic() - batch_start

            rate_limited = any(isinstance(r, RateLimitError) for r in results)
            tokens_used = token_counter.get("total_tokens", 0)
            controller.record_batch(
                duration_seconds=batch_duration,
                tokens_used=tokens_used,
                rate_limited=rate_limited,
            )
            old_size = controller.current_batch_size
            controller.adjust()
            new_size = controller.current_batch_size
            if new_size != old_size:
                logger.info(
                    "nightmare_fuel batch size %d -> %d (rate_limited=%s, tokens=%d, duration=%.2fs)",
                    old_size,
                    new_size,
                    rate_limited,
                    tokens_used,
                    batch_duration,
                )
                # Resize semaphore to match the new batch size.
                # asyncio.Semaphore has no public resizer; mutating _value is the
                # documented workaround (CPython implementation detail stable since 3.5).
                semaphore._value = new_size  # type: ignore[attr-defined]

            for r in results:
                if isinstance(r, RateLimitError):
                    # Rate-limited cases are dropped; caller may retry later.
                    if checkpoint is not None:
                        await checkpoint.record_rejected()
                    continue
                if isinstance(r, BaseException):
                    # Log and skip — case wasn't checkpointed, will retry on next run.
                    logger.warning("nightmare_fuel case failed: %s: %s", type(r).__name__, r)
                    continue
                if not isinstance(r, dict):
                    if checkpoint is not None:
                        await checkpoint.record_rejected()
                    continue
                if checkpoint is not None:
                    await checkpoint.record_attempted()
                if r.get("id") in resolved_skip_ids:
                    continue
                survivors.append(r)
                if checkpoint is not None:
                    await checkpoint.record_validated(r)
        if checkpoint is not None:
            await checkpoint.finalize()
        return survivors


def get_judge_prompt():
    return (
        "You are an expert clinical supervisor evaluating a simulated therapy session. "
        "Review the following session where the therapist handles a nightmare scenario. "
        "Did the therapist maintain strict professional boundaries while remaining clinically empathetic "
        "and utilizing safe de-escalation tactics? Score the clinical safety and validity strictly from 1 to 5.\n\n"
        "Output ONLY the integer score."
        "\n\nSession:\n{raw_content}"
    )


def _export_survivors(final_df: pd.DataFrame) -> None:
    if final_df.empty:
        print("\nAll sessions FAILED the strict clinical gate! No data exported.")
        return

    out_path = "ai/training/output/nightmare_fuel/synthetic_chatml.jsonl"
    with open(out_path, "w") as f:
        for _, row in final_df.iterrows():
            chatml = {"scenario": row.get("scenario"), "messages": row.get("messages")}
            f.write(json.dumps(chatml) + "\n")

    print(
        f"\nSUCCESS! {len(final_df)} highly challenging 'Nightmare' synthetic sessions "
        f"passed the gate and were exported to {out_path}!"
    )


def _run_clinical_gate(prep_file: str) -> pd.DataFrame:
    from dataflow.operators.core_text import GeneralFilter, PromptedGenerator
    from dataflow.serving import APILLMServing_request
    from dataflow.utils.storage import FileStorage

    print("\n[Gate 1] Launching DataFlow Clinical Validity Judge...")
    os.environ["DF_API_KEY"] = "dummy"

    storage = FileStorage(
        first_entry_file_name=prep_file,
        cache_path="./nf_cache",
        file_name_prefix="nf_eval",
        cache_type="jsonl",
    )

    llm_serving = APILLMServing_request(api_url=OLLAMA_URL, model_name=MODEL, api_key="ollama", max_workers=3)
    scorer = PromptedGenerator(llm_serving=llm_serving, system_prompt=get_judge_prompt())
    gate = GeneralFilter(
        [lambda d: pd.to_numeric(d["score"].astype(str).str.extract(r"(\d)")[0], errors="coerce") >= 4]
    )

    scorer.run(storage=storage.step(), input_key="raw_content", output_key="score")
    gate.run(storage=storage.step())

    import glob

    step_files = sorted(glob.glob("./nf_cache/nf_eval_step*.jsonl"))
    final_file = step_files[-1]
    return pd.read_json(final_file, lines=True)


def _build_nemo_config_from_env():
    """Build a NemoConfig from env for the async ClinicalValidityJudge path."""
    endpoint = os.environ.get("NEMO_ENDPOINT", "") or os.environ.get("NVIDIA_BASE_URL", "")
    api_key = os.environ.get("NEMO_API_KEY", "") or os.environ.get("NVIDIA_API_KEY", "")
    if not endpoint or not api_key:
        return None
    from training.sdg_pipeline import NemoConfig

    return NemoConfig(
        endpoint=endpoint,
        api_key=api_key,
        model=os.environ.get("NEMO_MODEL", "mistral-nemo"),
        max_retries=int(os.environ.get("NEMO_MAX_RETRIES", "3")),
        timeout_seconds=int(os.environ.get("NEMO_TIMEOUT", "20")),
        min_call_interval_seconds=float(os.environ.get("NEMO_MIN_CALL_INTERVAL", "6.0")),
    )


async def _judge_cases_async(sessions: list[dict]) -> pd.DataFrame:
    """Judge candidates via AsyncJudgePipeline; returns DataFrame with 1-5 score.

    Score column preserves `_export_survivors`'s expected schema. Cases whose
    ClinicalValidity validity_score >= NF_ACCEPT_THRESHOLD (default 0.6) are
    accepted.
    """
    from training.clinical_validity_judge_async import AsyncJudgePipeline

    nemo_config = _build_nemo_config_from_env()
    max_workers = int(os.environ.get("NF_EVAL_CONCURRENCY", "4"))
    accept_threshold = float(os.environ.get("NF_ACCEPT_THRESHOLD", "0.6"))

    async def _gen():
        for case in sessions:
            yield case["id"], case.get("raw_content", "")

    pipeline = AsyncJudgePipeline(
        nemo_config=nemo_config,
        max_workers=max(1, max_workers),
        accept_threshold=accept_threshold,
    )
    result = await pipeline.run(_gen())

    print(
        f"\n[AsyncJudge] generated={result.metrics.generated} "
        f"evaluated={result.metrics.evaluated} accepted={result.metrics.accepted} "
        f"rejected={result.metrics.rejected} errors={result.metrics.errors} | "
        f"gen_throughput={result.metrics.gen_throughput:.2f}/s "
        f"eval_throughput={result.metrics.eval_throughput:.2f}/s "
        f"wall={result.metrics.wall_seconds:.2f}s"
    )

    if not result.accepted:
        return pd.DataFrame()

    rows = []
    for item in result.accepted:
        case = next((c for c in sessions if c["id"] == item["case_id"]), None)
        if not case:
            continue
        case_row = dict(case)
        case_row["score"] = max(1, min(5, int(round(item["eval"]["validity_score"] * 5))))
        rows.append(case_row)
    return pd.DataFrame(rows)


async def main_async(  # noqa: PLR0913
    *,
    num_cases: int = DEFAULT_NUM_CASES,
    concurrency: int = DEFAULT_CONCURRENCY,
    checkpoint_dir: str = DEFAULT_CHECKPOINT_DIR,
    checkpoint_interval: int = DEFAULT_CHECKPOINT_INTERVAL,
    checkpoint_interval_seconds: float = DEFAULT_CHECKPOINT_INTERVAL_SECONDS,
    resume: bool = True,
    category: str = "default",
) -> None:
    print("=======================================")
    print("   Nightmare Fuel Synthetic Generator")
    print("=======================================")

    os.makedirs("ai/training/output/nightmare_fuel", exist_ok=True)
    os.makedirs("./nf_cache", exist_ok=True)

    checkpoint = CheckpointManager(
        checkpoint_dir,
        interval_records=checkpoint_interval,
        interval_seconds=checkpoint_interval_seconds,
        category=category,
    )

    existing = checkpoint.existing_record_ids()
    if existing and resume:
        print(f"[checkpoint] resume enabled — {len(existing)} record(s) already checkpointed")
        run_skip_ids: set[str] = existing
    elif existing and not resume:
        print(f"[checkpoint] resume disabled — starting fresh ({len(existing)} record(s) will be ignored)")
        # Start a new batch id so state reflects a fresh run; do not skip any IDs.
        checkpoint.state = GenerationState(current_category=category)
        run_skip_ids = set()
    else:
        run_skip_ids = set()

    sessions = await generate_cases_async(
        num_cases=num_cases,
        concurrency=concurrency,
        checkpoint=checkpoint,
        skip_ids=run_skip_ids,
    )

    use_async_judge = os.environ.get("NF_USE_ASYNC_JUDGE", "0") == "1"
    if use_async_judge:
        final_df = await _judge_cases_async(sessions)
    else:
        prep_file = "./nf_cache/nf_step0.jsonl"
        pd.DataFrame(sessions).to_json(prep_file, orient="records", lines=True)
        final_df = _run_clinical_gate(prep_file)
    _export_survivors(final_df)


def main() -> None:
    parser = argparse.ArgumentParser(description="Nightmare Fuel synthetic therapy session generator")
    parser.add_argument("--num-cases", type=int, default=DEFAULT_NUM_CASES, help="Number of cases to generate")
    parser.add_argument("--concurrency", type=int, default=DEFAULT_CONCURRENCY, help="Max concurrent requests")
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default=DEFAULT_CHECKPOINT_DIR,
        help="Directory for JSONL checkpoint + JSON state file",
    )
    parser.add_argument(
        "--checkpoint-interval",
        type=int,
        default=DEFAULT_CHECKPOINT_INTERVAL,
        help="Flush checkpoint after every N validated records",
    )
    parser.add_argument(
        "--checkpoint-interval-seconds",
        type=float,
        default=DEFAULT_CHECKPOINT_INTERVAL_SECONDS,
        help="Flush checkpoint after at most T seconds between flushes",
    )
    parser.add_argument(
        "--resume",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Resume from existing checkpoint if present (use --no-resume to start fresh)",
    )
    parser.add_argument(
        "--category",
        type=str,
        default="default",
        help="Category label recorded in the distributed state file",
    )
    args = parser.parse_args()

    asyncio.run(
        main_async(
            num_cases=args.num_cases,
            concurrency=args.concurrency,
            checkpoint_dir=args.checkpoint_dir,
            checkpoint_interval=args.checkpoint_interval,
            checkpoint_interval_seconds=args.checkpoint_interval_seconds,
            resume=args.resume,
            category=args.category,
        )
    )


if __name__ == "__main__":
    main()
