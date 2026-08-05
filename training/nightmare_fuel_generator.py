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
import os
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path

import aiohttp
import pandas as pd

# We will use Ollama locally for generation and evaluation to keep it simple,
# but it can easily point to NeMo API if you swap the base URL and Key!
OLLAMA_URL = "https://ollama.pixelated.love/v1/chat/completions"
MODEL = "ornith:9b"
DEFAULT_NUM_CASES = int(os.environ.get("NF_NUM_CASES", "5"))
DEFAULT_CONCURRENCY = int(os.environ.get("NF_CONCURRENCY", "5"))
REQUEST_TIMEOUT_SECONDS = float(os.environ.get("NF_REQUEST_TIMEOUT", "120"))
DEFAULT_CHECKPOINT_DIR = os.environ.get("NF_CHECKPOINT_DIR", "ai/training/output/nightmare_fuel/checkpoints")
DEFAULT_CHECKPOINT_INTERVAL = int(os.environ.get("NF_CHECKPOINT_INTERVAL", "10"))
DEFAULT_CHECKPOINT_INTERVAL_SECONDS = float(os.environ.get("NF_CHECKPOINT_INTERVAL_SECONDS", "30"))
RECORDS_FILENAME = "records.jsonl"
STATE_FILENAME = "state.json"


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
    """Return dataclass field descriptors (avoids importing fields at module top)."""
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
                    return GenerationState.from_dict(json.load(f))
            except (json.JSONDecodeError, OSError):
                # Corrupt state file: start fresh but keep batch_id from a new uuid.
                pass
        return GenerationState(current_category=category)

    def load_existing_records(self) -> list[dict]:
        """Read all records from an existing checkpoint file.

        Returns an empty list if the file does not exist or is empty. Skips
        lines that fail to parse so a single corrupt line does not poison the
        resume.
        """

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
        """Account for one newly validated record and flush if the threshold is hit.

        The record is buffered internally and persisted on the next flush, so a
        crash between flushes can lose at most ``interval_records`` records. The
        caller does not need to track pending records separately.
        """
        self.state.total_validated += 1
        self._pending_records.append(record)
        self._records_since_flush += 1
        if self.should_flush():
            await self.flush()

    async def record_rejected(self) -> None:
        self.state.total_rejected += 1

    async def record_attempted(self) -> None:
        self.state.total_attempted += 1

    async def flush(self, extra_records: list[dict] | None = None) -> None:
        """Persist buffered pending records and refresh the state snapshot.

        ``extra_records`` (optional) are appended to the internal buffer before
        writing. ``records.jsonl`` is append-only; ``state.json`` is rewritten.
        """

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
            with self.state_path.open("w") as f:
                json.dump(self.state.to_dict(), f, indent=2)
            self._last_flush_time = time.monotonic()

    async def finalize(self, extra_records: list[dict] | None = None) -> None:
        """Final flush at end of run, writing any remaining pending records."""
        await self.flush(extra_records=extra_records)


async def _chat_completion(
    session: aiohttp.ClientSession,
    messages: list[dict[str, str]],
    *,
    temperature: float,
) -> str:
    payload = {"model": MODEL, "messages": messages, "temperature": temperature}
    async with session.post(
        OLLAMA_URL,
        json=payload,
        headers={"Authorization": "Bearer dummy"},
        timeout=aiohttp.ClientTimeout(total=REQUEST_TIMEOUT_SECONDS),
    ) as response:
        response.raise_for_status()
        data = await response.json()
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
) -> str:
    print("Generating Nightmare Scenario...")
    prompt = _build_scenario_prompt(domain_gap=domain_gap, difficulty=difficulty)
    return await _chat_completion(
        session,
        [{"role": "user", "content": prompt}],
        temperature=0.9,
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


async def simulate_therapy_session_async(session: aiohttp.ClientSession, scenario: str) -> dict:
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
    )
    return _parse_transcript(transcript, scenario)


async def _generate_case(
    session: aiohttp.ClientSession,
    semaphore: asyncio.Semaphore,
    case_index: int,
    total_cases: int,
) -> dict | None:
    async with semaphore:
        print(f"\n--- Generating Case {case_index + 1}/{total_cases} ---")
        scenario = await generate_nightmare_scenario_async(session)
        session_data = await simulate_therapy_session_async(session, scenario)
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
    checkpoint: CheckpointManager | None = None,
) -> list[dict]:
    """Generate ``num_cases`` cases concurrently, checkpointing as we go.

    When ``checkpoint`` is provided, the manager records attempted/validated/
    rejected counts and periodically flushes validated records to the JSONL
    checkpoint file. Any record whose ``id`` already exists in the checkpoint is
    skipped on resume so the caller does not pay to regenerate it.
    """

    semaphore = asyncio.Semaphore(max(1, concurrency))
    timeout = aiohttp.ClientTimeout(total=REQUEST_TIMEOUT_SECONDS)

    # Resume: do not regenerate cases already checkpointed.
    skip_ids: set[str] = set()
    if checkpoint is not None:
        skip_ids = checkpoint.existing_record_ids()
        if skip_ids:
            print(f"[checkpoint] resuming — {len(skip_ids)} record(s) already present")

    pending_records: list[dict] = []

    async with aiohttp.ClientSession(timeout=timeout) as session:
        tasks = [_generate_case(session, semaphore, case_index, num_cases) for case_index in range(num_cases)]
        for coro in asyncio.as_completed(tasks):
            result = await coro
            if checkpoint is not None:
                await checkpoint.record_attempted()
            if result is None:
                if checkpoint is not None:
                    await checkpoint.record_rejected()
                continue
            if result["id"] in skip_ids:
                continue
            pending_records.append(result)
            if checkpoint is not None:
                await checkpoint.record_validated(result)

    if checkpoint is not None:
        await checkpoint.finalize()

    return pending_records


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
    elif existing and not resume:
        print(f"[checkpoint] resume disabled — starting fresh ({len(existing)} record(s) will be ignored)")
        # Start a new batch id so state reflects a fresh run.
        checkpoint.state = GenerationState(current_category=category)

    sessions = await generate_cases_async(
        num_cases=num_cases,
        concurrency=concurrency,
        checkpoint=checkpoint,
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
