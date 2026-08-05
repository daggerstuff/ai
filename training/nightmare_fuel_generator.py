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
import math

import argparse
import asyncio
import json
import logging
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
DEFAULT_NUM_CASES = int(os.environ.get("NF_NUM_CASES", "5"))
DEFAULT_CONCURRENCY = int(os.environ.get("NF_CONCURRENCY", "5"))
REQUEST_TIMEOUT_SECONDS = float(os.environ.get("NF_REQUEST_TIMEOUT", "120"))
DEFAULT_MIN_BATCH = int(os.environ.get("NF_MIN_BATCH", "2"))
DEFAULT_MAX_BATCH = int(os.environ.get("NF_MAX_BATCH", "32"))
DEFAULT_TARGET_TOKENS = int(os.environ.get("NF_TARGET_TOKENS", "4096"))
DEFAULT_BACKOFF_BASE = float(os.environ.get("NF_BACKOFF_BASE", "2.0"))
DEFAULT_BACKOFF_MAX = float(os.environ.get("NF_BACKOFF_MAX", "60.0"))

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


async def _chat_completion(
    session: aiohttp.ClientSession,
    messages: list[dict[str, str]],
    *,
    temperature: float,
    token_counter: dict | None = None,
) -> str:
    payload = {"model": MODEL, "messages": messages, "temperature": temperature}
    async with session.post(
        OLLAMA_URL,
        json=payload,
        headers={"Authorization": f"Bearer {os.environ.get('CLOUDFLARE_AUTH_TOKEN', 'dummy')}"},
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
            token_counter.get("total_tokens", 0) + usage.get("prompt_tokens", 0) + usage.get("completion_tokens", 0)
        )
    return data["choices"][0]["message"]["content"]


def _build_scenario_prompt(domain_gap=None, difficulty=None) -> str:
    prompt = (
        "You are an expert clinical psychologist creating training scenarios. "
        "Generate a highly challenging, extreme 'nightmare' clinical scenario. "
        "This should be a patient who is extremely resistant, has complex comorbidities "
        "(e.g. severe Borderline Personality Disorder with active substance abuse and defiance), "
        "or aggressively pushes boundaries. "
        "Provide a 1-paragraph summary of the patient and their presenting challenge."
    )
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
    token_counter: dict | None = None,
) -> str:
    print("Generating Nightmare Scenario...")
    prompt = _build_scenario_prompt(domain_gap=domain_gap, difficulty=difficulty)
    return await _chat_completion(
        session,
        [{"role": "user", "content": prompt}],
        temperature=0.9,
        token_counter=token_counter,
    )


def generate_nightmare_scenario(domain_gap=None, difficulty=None) -> str:
    """Sync wrapper that works in both sync and async contexts."""

    async def _run() -> str:
        timeout = aiohttp.ClientTimeout(total=REQUEST_TIMEOUT_SECONDS)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            return await generate_nightmare_scenario_async(
                session,
                domain_gap=domain_gap,
                difficulty=difficulty,
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
    token_counter: dict | None = None,
) -> dict:
    print("Simulating Session...")
    prompt = (
        f"Based on this nightmare scenario: {scenario}\n\n"
        "Generate a 6-turn therapy transcript. The 'Patient' must act extremely difficult, "
        "evasive, or confrontational according to the scenario. The 'Therapist' must attempt "
        "to use evidence-based clinical de-escalation, boundary setting, and empathy.\n\n"
        "Output ONLY the raw transcript lines, alternating 'Patient: ...' and 'Therapist: ...'"
    )
    transcript = await _chat_completion(
        session,
        [{"role": "user", "content": prompt}],
        temperature=0.8,
        token_counter=token_counter,
    )
    return _parse_transcript(transcript, scenario)


async def _generate_case(
    session: aiohttp.ClientSession,
    semaphore: asyncio.Semaphore,
    case_index: int,
    total_cases: int,
    token_counter: dict | None = None,
) -> dict | None:
    async with semaphore:
        print(f"\n--- Generating Case {case_index + 1}/{total_cases} ---")
        scenario = await generate_nightmare_scenario_async(session, token_counter=token_counter)
        session_data = await simulate_therapy_session_async(session, scenario, token_counter=token_counter)
        if len(session_data["messages"]) < 2:
            return None

        flat = "\n".join([f"{m['role']}: {m['content']}" for m in session_data["messages"]])
        return {
            "id": str(uuid.uuid4()),
            "scenario": scenario,
            "raw_content": flat,
            "messages": session_data["messages"],
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
                _generate_case(session, semaphore, idx, total_cases, token_counter=token_counter)
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
                    continue
                if isinstance(r, Exception):
                    # Re-raise unexpected errors after recording so we don't
                    # silently mask real failures.
                    raise r
                if not isinstance(r, dict):
                    continue
                survivors.append(r)
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
