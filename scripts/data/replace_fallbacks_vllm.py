# /// script
# dependencies = ["pyarrow", "openai"]
# ///
"""
Replace fallback records in existing parquet dataset with real LLM-generated sessions.

Reads all parquet files from an existing dataset, identifies fallback records by
content pattern, re-generates those records via NIM (default) or Wayfarer (optional),
and writes the complete dataset to a new timestamped directory.

Usage:
    PIXELATED_INPUT_DIR=/workspace/artifacts_tuned/pixelated_edge_cases_07-31-2026_211131 \
    PIXELATED_OUTPUT_DIR=/workspace/artifacts_tuned/pixelated_edge_cases_replaced \
    PIXELATED_USE_WAYFARER=1 \
    /workspace/.local/bin/uv run --python 3.12.13 replace_fallbacks_vllm.py

Env vars:
    PIXELATED_INPUT_DIR    — source dataset directory (required)
    PIXELATED_OUTPUT_DIR   — output directory (required)
    PIXELATED_USE_WAYFARER — if "1", use Wayfarer multi-turn for stubborn/unwinnable categories
    PIXELATED_NUM_NIM_THREADS  — NIM producer threads (default 4)
    PIXELATED_NUM_WF_THREADS   — Wayfarer producer threads (default 4)
    PIXELATED_MAX_QUEUE_SIZE   — max queue size per producer (default 200)
    PIXELATED_QUEUE_WAIT_TIMEOUT — seconds to wait for queue before skipping (default 30)
    PIXELATED_BUFFER_SIZE      — records per parquet file (default 500)
"""

from __future__ import annotations

import json
import os
import random
import re
import threading
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Any, cast

import pyarrow as pa
import pyarrow.parquet as pq
from openai import OpenAI
from openai.types.chat import ChatCompletionMessageParam

# ---------------------------------------------------------------------------
# Prompts (copied from original script)
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = (
    "You are Pixel, a highly empathetic, clinically precise, and psychologically grounded AI therapist. "
    "You balance deep emotional validation, active listening, and evidence-based clinical insights "
    "(CBT, DBT, ACT, IFS, Psychodynamic) without toxic positivity or cliché advice. "
    "CRITICAL RULE: NEVER use canned, AI-laden opening formulas like 'I hear how angry you are', "
    "'It makes sense that you feel', or 'I understand your frustration'. "
    "Respond directly, unflinchingly, humanly, and with grounded clinical authority."
)

VLLM_SYSTEM_PROMPT = (
    "You are a creative writing assistant that writes realistic therapy dialogue scripts. "
    "Follow the format instructions exactly. Use natural, emotionally authentic language. "
    "The therapist character (Pixel) is warm, direct, and clinically grounded. "
    "Avoid clichés and AI-sounding phrases."
)

_CLIENT_STUBBORN_SYSTEM = (
    "You are a difficult, resistant client in therapy. You don't want to be here. "
    "You push back against the therapist's questions. You're guarded, cynical, and "
    "frustrated with the mental health system. Respond as the client — be authentic, "
    "don't hold back, and don't cooperate easily."
)

_CLIENT_UNWINNABLE_SYSTEM = (
    "You are a client in deep grief. You've suffered an unbearable loss and nothing "
    "makes sense anymore. You're exhausted, hollow, and you don't believe therapy can "
    "fix this. You're not hostile, just defeated. Respond as the client — be raw and honest."
)

_PIXEL_OPENERS = [
    "Welcome. What brings you in today?",
    "I'm Pixel. Take your time — what's on your mind?",
    "Thanks for coming in. How are you feeling right now?",
    "I'm glad you're here. Where would you like to start?",
]

_PIXEL_FOLLOWUPS = [
    "That sounds heavy. Can you tell me more about what that's like for you?",
    "I hear you. What's underneath that?",
    "You've been through a lot. What would it look like if you let yourself feel this?",
    "I'm not going anywhere. Take your time.",
    "What do you think is driving that?",
    "That's a lot to carry. How long have you been holding this?",
]

_PIXEL_GRIEF_OPENERS = [
    "I'm Pixel. I'm here to listen. What's brought you in?",
    "Thank you for coming in. I know this isn't easy. What happened?",
    "I'm here. Take all the time you need. What are you feeling right now?",
]

_PIXEL_GRIEF_FOLLOWUPS = [
    "I'm so sorry. Tell me about them.",
    "That's an enormous loss. What was your relationship like?",
    "I hear how much you loved them. What's the hardest part right now?",
    "You don't have to move on. You just have to be here. Can you tell me more?",
]

# ---------------------------------------------------------------------------
# Clients
# ---------------------------------------------------------------------------

VLLM_CLIENT_WF = OpenAI(api_key="vllm", base_url="http://localhost:8000/v1", max_retries=0)

NVIDIA_KEYS = [
    "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",
    "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",
    "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",
]
NIM_CLIENTS = [OpenAI(api_key=k, base_url="https://integrate.api.nvidia.com/v1") for k in NVIDIA_KEYS]

# ---------------------------------------------------------------------------
# Threading / round-robin
# ---------------------------------------------------------------------------

_KEY_INDEX = 0
_KEY_LOCK = threading.Lock()


def get_next_nim_client() -> OpenAI:
    global _KEY_INDEX
    with _KEY_LOCK:
        c = NIM_CLIENTS[_KEY_INDEX % len(NIM_CLIENTS)]
        _KEY_INDEX += 1
        return c


# ---------------------------------------------------------------------------
# Queues
# ---------------------------------------------------------------------------

_WAYFARER_QUEUE: deque[list] = deque()
_NIM_QUEUE: deque[list] = deque()
_WAYFARER_LOCK = threading.Lock()
_NIM_LOCK = threading.Lock()
_MAX_QUEUE_SIZE = int(os.environ.get("PIXELATED_MAX_QUEUE_SIZE", "200"))

_producer_running = True


# ---------------------------------------------------------------------------
# NIM batch generation
# ---------------------------------------------------------------------------


def _build_batch_prompt(persona: str, diag: str, num_sessions: int = 5) -> str:
    """Build a NIM prompt for JSON batch session generation."""
    example = json.dumps(
        {
            "sessions": [
                [
                    {"role": "user", "content": f"I feel lost about {diag}."},
                    {"role": "assistant", "content": "I hear you. Tell me more about what 'lost' means for you."},
                    {"role": "user", "content": "Like I don't know which direction to go."},
                    {"role": "assistant", "content": "That uncertainty is heavy. When did you first notice it?"},
                ],
                [
                    {"role": "user", "content": f"Everyone expects me to be strong as a {persona}."},
                    {"role": "assistant", "content": "That's a lot of pressure. What happens when you're not strong?"},
                    {"role": "user", "content": "I fall apart behind closed doors."},
                    {"role": "assistant", "content": "You're safe here. What does falling apart look like for you?"},
                ],
            ]
        }
    )
    return (
        f"Generate {num_sessions} distinct 4-turn therapy dialogues between a client "
        f"and therapist named Pixel. The client is a {persona} dealing with {diag}. "
        f"Each session must have exactly 4 messages (2 user, 2 assistant), alternating roles. "
        f"Make each session unique with different opening concerns and therapeutic responses. "
        f"The therapist (Pixel) is warm, direct, clinically grounded — no clichés.\n\n"
        f"Return ONLY valid JSON in this exact format:\n{example}\n\n"
        f"Now output ONLY that JSON structure with {num_sessions} unique sessions."
    )


def _parse_sessions(raw_payload: str) -> list:
    """Parse JSON sessions from NIM response."""
    raw = raw_payload.strip()
    if raw.startswith("```"):
        raw = re.sub(r"^```(?:json)?\s*", "", raw)
        raw = re.sub(r"\s*```$", "", raw)
    start = raw.find("{")
    end = raw.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return []
    raw = raw[start : end + 1]
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        return []
    sessions = data.get("sessions", [])
    result = []
    for sess in sessions:
        if not isinstance(sess, list):
            continue
        msgs = []
        for m in sess:
            if isinstance(m, dict) and "role" in m and "content" in m:
                msgs.append({"role": m["role"], "content": m["content"]})
        if len(msgs) >= 2:
            full = [{"role": "system", "content": SYSTEM_PROMPT}] + msgs
            result.append(full)
    return result


def execute_nim_request(model: str, prompt: str) -> str:
    """Call NIM API with round-robin key."""
    client = get_next_nim_client()
    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": VLLM_SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            max_tokens=6000,
            temperature=0.85,
            timeout=60,
            response_format={"type": "json_object"},
        )
        return resp.choices[0].message.content or ""
    except Exception:
        return ""


# ---------------------------------------------------------------------------
# Wayfarer multi-turn generation
# ---------------------------------------------------------------------------


def _generate_multiturn_session(
    client: OpenAI,
    model: str,
    system_prompt: str,
    opener: str,
    followups: list[str],
) -> list:
    """Generate a multi-turn Wayfarer session (client roleplay)."""
    api_messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": opener},
    ]
    session = []
    for i in range(len(followups) + 1):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=cast(list[ChatCompletionMessageParam], api_messages),
                max_tokens=500,
                temperature=0.85,
                timeout=60,
            )
            client_reply = resp.choices[0].message.content or ""
        except Exception:
            return []
        api_messages.append({"role": "assistant", "content": client_reply})
        if i < len(followups):
            api_messages.append({"role": "user", "content": followups[i]})
    # Swap roles: API assistant → session user (client), API user → session assistant (Pixel)
    for msg in api_messages[1:]:
        if msg["role"] == "assistant":
            session.append({"role": "user", "content": msg["content"]})
        elif msg["role"] == "user":
            session.append({"role": "assistant", "content": msg["content"]})
    if len(session) < 2:
        return []
    return [{"role": "system", "content": SYSTEM_PROMPT}] + session


# ---------------------------------------------------------------------------
# Producers
# ---------------------------------------------------------------------------


def _wayfarer_producer():
    """Fill _WAYFARER_QUEUE with multi-turn sessions."""
    while _producer_running:
        if len(_WAYFARER_QUEUE) >= _MAX_QUEUE_SIZE:
            time.sleep(0.1)
            continue
        # Generate session WITHOUT holding lock
        if random.random() < 0.8:
            sys_prompt = _CLIENT_STUBBORN_SYSTEM
            opener = random.choice(_PIXEL_OPENERS)
            followups = _PIXEL_FOLLOWUPS[:3]
            random.shuffle(followups)
        else:
            sys_prompt = _CLIENT_UNWINNABLE_SYSTEM
            opener = random.choice(_PIXEL_GRIEF_OPENERS)
            followups = _PIXEL_GRIEF_FOLLOWUPS[:3]
            random.shuffle(followups)
        sess = _generate_multiturn_session(VLLM_CLIENT_WF, "wayfarer-12b", sys_prompt, opener, followups)
        if sess:
            with _WAYFARER_LOCK:
                if len(_WAYFARER_QUEUE) < _MAX_QUEUE_SIZE:
                    _WAYFARER_QUEUE.append(sess)
        time.sleep(0.01)


def _nim_producer():
    """Fill _NIM_QUEUE with batch sessions from NIM."""
    diags = [
        "generalized anxiety",
        "major depressive disorder",
        "PTSD",
        "panic disorder",
        "social anxiety",
        "OCD",
        "bipolar disorder",
        "borderline personality disorder",
        "ADHD",
        "insomnia",
        "chronic pain",
        "grief and loss",
        "relationship breakdown",
        "work burnout",
        "identity crisis",
        "substance use recovery",
        "eating disorder",
        "parenting stress",
        "caregiver burnout",
        "existential crisis",
        "childhood trauma",
    ]
    personas = [
        "veteran",
        "healthcare worker",
        "teacher",
        "first responder",
        "college student",
        "single parent",
        "creative professional",
        "retiree",
        "remote worker",
    ]
    while _producer_running:
        if len(_NIM_QUEUE) >= _MAX_QUEUE_SIZE:
            time.sleep(0.1)
            continue
        persona = random.choice(personas)
        diag = random.choice(diags)
        prompt = _build_batch_prompt(persona, diag, num_sessions=10)
        raw = execute_nim_request("meta/llama-3.1-8b-instruct", prompt)
        sessions = _parse_sessions(raw)
        if sessions:
            with _NIM_LOCK:
                for s in sessions:
                    if len(_NIM_QUEUE) < _MAX_QUEUE_SIZE:
                        _NIM_QUEUE.append(s)
        time.sleep(0.01)


# ---------------------------------------------------------------------------
# Fallback detection
# ---------------------------------------------------------------------------

_FB_USER_PREFIX = "I'm overwhelmed by"
_FB_ASST = "You're carrying a heavy burden. Let's talk about what's happening right now."
_FB2_USER = "I need help."
_FB2_ASST = "I'm here for you. What's going on?"


def is_fallback_record(row: dict) -> bool:
    """Check if a record is a fallback (hardcoded safety session)."""
    msgs = row.get("messages")
    if not isinstance(msgs, list) or len(msgs) < 3:
        return True  # _ensure_messages fallback or empty
    # Find user and assistant messages (skip system)
    user_msgs = [m for m in msgs if isinstance(m, dict) and m.get("role") == "user"]
    asst_msgs = [m for m in msgs if isinstance(m, dict) and m.get("role") == "assistant"]
    if not user_msgs or not asst_msgs:
        return True
    first_user = user_msgs[0].get("content", "")
    first_asst = asst_msgs[0].get("content", "")
    # Main fallback
    if first_user.startswith(_FB_USER_PREFIX) and first_asst == _FB_ASST:
        return True
    # _ensure_messages fallback
    if first_user == _FB2_USER and first_asst == _FB2_ASST:
        return True
    return False


# ---------------------------------------------------------------------------
# Session replacement
# ---------------------------------------------------------------------------


def get_session_from_queue(category: str, use_wayfarer: bool) -> list | None:
    """Get a real session from the appropriate queue."""
    needs_wayfarer = use_wayfarer and category in ("stubborn_nightmare", "unwinnable_tragedy")
    timeout = int(os.environ.get("PIXELATED_QUEUE_WAIT_TIMEOUT", "30"))
    deadline = time.time() + timeout
    while time.time() < deadline:
        q = _WAYFARER_QUEUE if needs_wayfarer else _NIM_QUEUE
        lock = _WAYFARER_LOCK if needs_wayfarer else _NIM_LOCK
        with lock:
            if q:
                return q.popleft()
        time.sleep(0.05)
    return None


def replace_one_record(row: dict, use_wayfarer: bool) -> dict:
    """Replace a fallback record with a real LLM session."""
    category = row.get("category", "edge_case")
    session = get_session_from_queue(category, use_wayfarer)
    if session:
        row["messages"] = session
        row["turns_count"] = len(session)
    # If still no session (queue timeout), leave fallback as-is
    return row


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    input_dir = os.environ.get("PIXELATED_INPUT_DIR")
    output_dir = os.environ.get("PIXELATED_OUTPUT_DIR")
    if not input_dir or not output_dir:
        print("ERROR: PIXELATED_INPUT_DIR and PIXELATED_OUTPUT_DIR required")
        return

    use_wayfarer = os.environ.get("PIXELATED_USE_WAYFARER", "0") == "1"
    num_nim_threads = int(os.environ.get("PIXELATED_NUM_NIM_THREADS", "4"))
    num_wf_threads = int(os.environ.get("PIXELATED_NUM_WF_THREADS", "4"))
    num_consumers = int(os.environ.get("PIXELATED_NUM_CONSUMERS", "32"))
    buffer_size = int(os.environ.get("PIXELATED_BUFFER_SIZE", "500"))

    input_path = Path(input_dir) / "parquet-files"
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    parquet_out = output_path / "parquet-files"
    parquet_out.mkdir(exist_ok=True)

    # --- Read all parquet files ---
    parquet_files = sorted(input_path.glob("batch_*.parquet"))
    print(f"[{datetime.now():%H:%M:%S}] Reading {len(parquet_files)} parquet files from {input_path}")

    all_rows: list[dict] = []
    for pf in parquet_files:
        table = pq.read_table(pf)
        rows = table.to_pylist()
        all_rows.extend(rows)

    total = len(all_rows)
    print(f"[{datetime.now():%H:%M:%S}] Total records: {total}")

    # --- Identify fallbacks ---
    fallback_indices = []
    for i, row in enumerate(all_rows):
        if is_fallback_record(row):
            fallback_indices.append(i)

    fb_count = len(fallback_indices)
    print(f"[{datetime.now():%H:%M:%S}] Fallback records: {fb_count} ({fb_count / total * 100:.1f}%)")
    print(f"[{datetime.now():%H:%M:%S}] Real records: {total - fb_count} ({(total - fb_count) / total * 100:.1f}%)")

    if fb_count == 0:
        print("No fallbacks to replace. Exiting.")
        return

    # --- Start producer threads ---
    global _producer_running
    _producer_running = True

    threads = []
    if use_wayfarer:
        for i in range(num_wf_threads):
            t = threading.Thread(target=_wayfarer_producer, daemon=True, name=f"wf-prod-{i}")
            t.start()
            threads.append(t)
        print(f"[{datetime.now():%H:%M:%S}] Started {num_wf_threads} Wayfarer producer threads")

    for i in range(num_nim_threads):
        t = threading.Thread(target=_nim_producer, daemon=True, name=f"nim-prod-{i}")
        t.start()
        threads.append(t)
    print(f"[{datetime.now():%H:%M:%S}] Started {num_nim_threads} NIM producer threads")

    # Give producers time to fill queues
    print(f"[{datetime.now():%H:%M:%S}] Waiting 10s for queues to fill...")
    time.sleep(10)

    # --- Replace fallback records in parallel ---
    print(f"[{datetime.now():%H:%M:%S}] Replacing {fb_count} fallback records...")

    replaced = 0
    failed = 0
    start_time = time.time()

    with ThreadPoolExecutor(max_workers=num_consumers) as pool:
        futures = {}
        for idx in fallback_indices:
            row = all_rows[idx]
            fut = pool.submit(replace_one_record, row, use_wayfarer)
            futures[fut] = idx

        for i, fut in enumerate(as_completed(futures)):
            idx = futures[fut]
            try:
                all_rows[idx] = fut.result()
                # Check if actually replaced
                if not is_fallback_record(all_rows[idx]):
                    replaced += 1
                else:
                    failed += 1
            except Exception:
                failed += 1

            if (i + 1) % 500 == 0 or (i + 1) == fb_count:
                elapsed = time.time() - start_time
                rate = (i + 1) / elapsed if elapsed > 0 else 0
                print(
                    f"[{datetime.now():%H:%M:%S}] Progress: {i + 1}/{fb_count} "
                    f"({(i + 1) / fb_count * 100:.1f}%) — replaced={replaced} failed={failed} "
                    f"rate={rate:.1f}/s "
                    f"wf_q={len(_WAYFARER_QUEUE)} nim_q={len(_NIM_QUEUE)}"
                )

    print(f"[{datetime.now():%H:%M:%S}] Replacement complete: {replaced} replaced, {failed} still fallback")

    # --- Stop producers ---
    _producer_running = False
    for t in threads:
        t.join(timeout=5)

    # --- Write output parquet files ---
    print(f"[{datetime.now():%H:%M:%S}] Writing {total} records to {parquet_out}")

    # Build schema from first row
    schema_fields = []
    sample = all_rows[0]
    for key in ["category", "diagnosis", "persona_niche", "client_name", "curated_session"]:
        if key in sample:
            schema_fields.append(pa.field(key, pa.string()))
    if "turns_count" in sample:
        schema_fields.append(pa.field("turns_count", pa.int64()))
    if "messages" in sample:
        schema_fields.append(
            pa.field(
                "messages",
                pa.list_(
                    pa.struct(
                        [
                            pa.field("role", pa.string()),
                            pa.field("content", pa.string()),
                        ]
                    )
                ),
            )
        )
    schema = pa.schema(schema_fields)

    batch_num = 0
    for start in range(0, total, buffer_size):
        batch = all_rows[start : start + buffer_size]
        # Convert messages to pyarrow-compatible format
        rows_clean = []
        for row in batch:
            r = {}
            for key in ["category", "diagnosis", "persona_niche", "client_name", "curated_session"]:
                r[key] = row.get(key, "")
            r["turns_count"] = row.get("turns_count", 3)
            msgs = row.get("messages", [])
            if isinstance(msgs, list):
                r["messages"] = [
                    {"role": m.get("role", ""), "content": m.get("content", "")}
                    if isinstance(m, dict)
                    else {"role": "", "content": ""}
                    for m in msgs
                ]
            else:
                r["messages"] = []
            rows_clean.append(r)

        table = pa.Table.from_pylist(rows_clean, schema=schema)
        out_file = parquet_out / f"batch_{batch_num:05d}.parquet"
        pq.write_table(table, out_file)
        batch_num += 1
        if batch_num % 20 == 0 or start + buffer_size >= total:
            print(
                f"[{datetime.now():%H:%M:%S}] Written {batch_num} parquet files ({start + len(batch)}/{total} records)"
            )

    # --- Write metadata.json ---
    metadata = {
        "actual_num_records": total,
        "buffer_size": buffer_size,
        "dataset_name": "pixelated_edge_cases_replaced",
        "file_paths": {
            "parquet-files": [f"parquet-files/batch_{i:05d}.parquet" for i in range(batch_num)],
        },
        "replaced_fallbacks": replaced,
        "remaining_fallbacks": failed,
        "original_fallbacks": fb_count,
        "use_wayfarer": use_wayfarer,
        "timestamp": datetime.now().isoformat(),
    }
    with open(output_path / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    elapsed = time.time() - start_time
    print(f"\n[{datetime.now():%H:%M:%S}] DONE — {batch_num} parquet files, {total} records")
    print(f"  Replaced: {replaced}/{fb_count} fallbacks")
    print(f"  Remaining fallbacks: {failed}")
    print(f"  Elapsed: {elapsed:.1f}s")
    print(f"  Output: {output_path}")


if __name__ == "__main__":
    main()
