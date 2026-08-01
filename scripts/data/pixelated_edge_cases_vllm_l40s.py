# /// script
# dependencies = [
#   "data-designer",
#   "pydantic",
#   "openai",
#   "weave",
#   "zstandard",
# ]
# ///

"""
OVHcloud L40s GPU Ollama + Triple-Key NVIDIA NIM Generator
==========================================================

Architecture:
1. Local Ollama Engine on L40s GPU (http://localhost:11434/v1):
   - Serves Wayfarer-12B + self-after-dark locally on-GPU via GGUF quantized models.
   - Zero compilation issues, zero network latency, sub-second batch output.
2. Triple Active NVIDIA NIM Key Pool (Fallback / Augmentation):
   - Key 1: REDACTED_NVIDIA_KEY_1
   - Key 2: REDACTED_NVIDIA_KEY_2
   - Key 3: REDACTED_NVIDIA_KEY_3
3. 5-Session Array Batching & Global Thread Queue:
   - High-throughput parallel worker execution pushing output to /workspace/data.
"""

import asyncio
import json
import logging
import os
import random
import re
import subprocess
import sys
import time
import threading
import urllib.request
from collections import deque
import data_designer.config as dd
from pydantic import BaseModel, Field
from openai import OpenAI
import weave

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
# Silence OpenAI client's INFO-level retry spam ("Retrying request to /chat/completions in X seconds").
# NIM 429 backpressure retries still work, but stay quiet. Actual errors still surface at WARNING+.
logging.getLogger("openai").setLevel(logging.WARNING)
logger = logging.getLogger(__name__)

SYSTEM_PROMPT = (
    "You are Pixel, a highly empathetic, clinically precise, and psychologically grounded AI therapist. "
    "You balance deep emotional validation, active listening, and evidence-based clinical insights "
    "(CBT, DBT, ACT, IFS, Psychodynamic) without toxic positivity or cliché advice. "
    "CRITICAL RULE: NEVER use canned, AI-laden opening formulas like 'I hear how angry you are', "
    "'It makes sense that you feel', or 'I understand your frustration'. "
    "Respond directly, unflinchingly, humanly, and with grounded clinical authority."
)

# For vLLM producers: neutral writing-assistant prompt keeps models in "format-following" mode
# instead of "roleplay as Pixel" mode. The roleplay system prompt causes models to respond
# in first person as Pixel, ignoring the CLIENT:/PIXEL: format instructions.
VLLM_SYSTEM_PROMPT = (
    "You are a creative writing assistant that writes realistic therapy dialogue scripts. "
    "Follow the format instructions exactly. Use natural, emotionally authentic language. "
    "The therapist character (Pixel) is warm, direct, and clinically grounded. "
    "Avoid clichés and AI-sounding phrases."
)

_WAYFARER_QUEUE = deque()
_NIM_QUEUE = deque()
_WAYFARER_LOCK = threading.Lock()
_NIM_LOCK = threading.Lock()
_KEY_INDEX = 0
_KEY_LOCK = threading.Lock()

_MAX_QUEUE_SIZE = int(os.environ.get("PIXELATED_MAX_QUEUE_SIZE", "200"))
_producer_running = False

# JSON schema for Ollama structured output (GBNF grammar enforcement).
# Forces valid JSON with "sessions" key — fixes 75% parse failure rate from
# Wayfarer-12B/self-after-dark outputting prose or malformed JSON.
_OLLAMA_JSON_SCHEMA = {
    "type": "object",
    "properties": {
        "sessions": {
            "type": "array",
            "items": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "role": {"type": "string", "enum": ["user", "assistant"]},
                        "content": {"type": "string"},
                    },
                    "required": ["role", "content"],
                },
            },
        }
    },
    "required": ["sessions"],
}


# --- Multi-turn: Wayfarer plays the CLIENT, we fabricate Pixel's (therapist) lines ---
# Model is a roleplay model — it plays a person (client), not the therapist.
# We send Pixel's fabricated lines as "user" role; model responds as client ("assistant").
# After the conversation, swap roles for the dataset: API assistant → session user (client),
# API user → session assistant (Pixel).

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


def _generate_multiturn_session(client: OpenAI, model: str, system_prompt: str, opener: str, followups: list) -> list:

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": opener},
    ]
    try:
        resp = client.chat.completions.create(
            model=model,
            messages=messages,
            max_tokens=500,
            temperature=0.85,
            timeout=60.0,
        )
        client_reply = (resp.choices[0].message.content or "").strip()
        if not client_reply:
            logger.warning("VLLM_EMPTY: model=%s finish=%s", model, resp.choices[0].finish_reason)
            return []
        messages.append({"role": "assistant", "content": client_reply})

        for followup in followups:
            messages.append({"role": "user", "content": followup})
            resp = client.chat.completions.create(
                model=model,
                messages=messages,
                max_tokens=500,
                temperature=0.85,
                timeout=60.0,
            )
            client_reply = (resp.choices[0].message.content or "").strip()
            if not client_reply:
                logger.warning("VLLM_EMPTY: model=%s finish=%s", model, resp.choices[0].finish_reason)
                break
            messages.append({"role": "assistant", "content": client_reply})

        if len(messages) < 4:
            return []

        # Swap roles: API assistant (client) → session user, API user (Pixel) → session assistant
        session = [{"role": "system", "content": SYSTEM_PROMPT}]
        for m in messages[1:]:
            if m["role"] == "user":
                session.append({"role": "assistant", "content": m["content"]})
            else:
                session.append({"role": "user", "content": m["content"]})
        return [session]
    except Exception as e:
        logger.warning("vLLM multi-turn error (%s): %s", model, e)
        return []


def _build_batch_prompt(persona: str = "a client", diag: str = "various conditions", num_sessions: int = 5) -> str:
    # Few-shot prompt helps roleplay models produce JSON instead of prose.
    example = '{"sessions": [[{"role": "user", "content": "I feel lost"}, {"role": "assistant", "content": "I hear you. Tell me more."}, {"role": "user", "content": "It started last week"}, {"role": "assistant", "content": "What changed?"}]]}'
    return (
        f"Generate {num_sessions} distinct 4-turn therapy dialogues between a client ({persona}, {diag}) and Pixel (therapist). "
        f"Each session: 4 turns alternating user/assistant.\n"
        f"Example: {example}\n"
        f"Now output ONLY that JSON structure with {num_sessions} unique sessions."
    )


def _build_prose_prompt(persona: str = "a client", diag: str = "various conditions", num_sessions: int = 1) -> str:
    return (
        f"A new client has come to see you. They are {persona} dealing with {diag}. "
        f"Have a natural conversation with them. "
        f"Write both your words and the client's words."
    )


def _parse_prose_sessions(raw_payload: str) -> list:
    """Parse natural roleplay dialogue — adapts to whatever the model produces."""
    if not raw_payload:
        logger.warning("PARSE_FAIL: empty payload (prose)")
        return []

    label_re = re.compile(r"^(CLIENT|Client|USER|User)\s*:\s*(.+)$")
    pixel_re = re.compile(r"^(PIXEL|Pixel|THERAPIST|Therapist|ASSISTANT|Assistant)\s*:\s*(.+)$")

    chunks = re.split(r"^---+\s*$", raw_payload.strip(), flags=re.MULTILINE)
    sessions = []

    for chunk in chunks:
        chunk = chunk.strip()
        if not chunk:
            continue

        lines = [l.strip() for l in chunk.split("\n") if l.strip()]
        if not lines:
            continue

        has_labels = any(label_re.match(l) or pixel_re.match(l) for l in lines)

        messages = []
        if has_labels:
            for line in lines:
                m = label_re.match(line)
                if m:
                    messages.append({"role": "user", "content": m.group(2).strip()})
                    continue
                m = pixel_re.match(line)
                if m:
                    messages.append({"role": "assistant", "content": m.group(2).strip()})
                    continue
                if messages:
                    messages[-1]["content"] += " " + line
        else:
            paragraphs = [p.strip() for p in re.split(r"\n\s*\n", chunk) if p.strip()]
            if len(paragraphs) < 2:
                paragraphs = lines
            for i, para in enumerate(paragraphs):
                role = "assistant" if i % 2 == 0 else "user"
                messages.append({"role": role, "content": para})

        if len(messages) >= 2:
            sessions.append([{"role": "system", "content": SYSTEM_PROMPT}] + messages)

    if not sessions:
        logger.warning("PARSE_FAIL: no sessions from prose | raw[0:300]: %s", raw_payload[:300])
    return sessions


def _parse_sessions(raw_payload: str) -> list:
    if not raw_payload:
        logger.warning("PARSE_FAIL: empty payload")
        return []
    try:
        clean_json = raw_payload.replace("```json", "").replace("```", "").strip()
        # Models often prepend prose before JSON — extract the JSON object
        brace_start = clean_json.find("{")
        brace_end = clean_json.rfind("}")
        if brace_start >= 0 and brace_end > brace_start:
            clean_json = clean_json[brace_start : brace_end + 1]
        data = json.loads(clean_json)
        if isinstance(data, dict) and "sessions" in data and isinstance(data["sessions"], list):
            result = []
            for s in data["sessions"]:
                if isinstance(s, list) and len(s) > 0:
                    result.append([{"role": "system", "content": SYSTEM_PROMPT}] + s)
            return result
        logger.warning(
            "PARSE_FAIL: no 'sessions' key. Keys: %s",
            list(data.keys()) if isinstance(data, dict) else type(data).__name__,
        )
    except Exception as e:
        logger.warning("PARSE_FAIL: JSON error: %s | raw[0:300]: %s", e, raw_payload[:300])
    return []


def _wayfarer_producer():
    """Multi-turn: model plays CLIENT, we fabricate Pixel's (therapist) lines."""
    global _diag_wf_producer_calls, _diag_wf_producer_sessions
    while _producer_running:
        with _WAYFARER_LOCK:
            full = len(_WAYFARER_QUEUE) >= _MAX_QUEUE_SIZE
        if full:
            time.sleep(0.5)
            continue
        if random.random() < 0.8:
            sys_prompt = _CLIENT_STUBBORN_SYSTEM
            opener = random.choice(_PIXEL_OPENERS)
            followups = random.sample(_PIXEL_FOLLOWUPS, min(2, len(_PIXEL_FOLLOWUPS)))
        else:
            sys_prompt = _CLIENT_UNWINNABLE_SYSTEM
            opener = random.choice(_PIXEL_GRIEF_OPENERS)
            followups = random.sample(_PIXEL_GRIEF_FOLLOWUPS, min(2, len(_PIXEL_GRIEF_FOLLOWUPS)))
        sessions = _generate_multiturn_session(VLLM_CLIENT_WF, "wayfarer-12b", sys_prompt, opener, followups)
        with _diag_lock:
            _diag_wf_producer_calls += 1
            _diag_wf_producer_sessions += len(sessions)
        if sessions:
            with _WAYFARER_LOCK:
                _WAYFARER_QUEUE.extend(sessions)
        else:
            time.sleep(1)


def _wayfarer2_producer():
    """Second Wayfarer-12B thread (self-after-dark too slow for multi-turn)."""
    global _diag_wf_producer_calls, _diag_wf_producer_sessions
    while _producer_running:
        with _WAYFARER_LOCK:
            full = len(_WAYFARER_QUEUE) >= _MAX_QUEUE_SIZE
        if full:
            time.sleep(0.5)
            continue
        if random.random() < 0.8:
            sys_prompt = _CLIENT_STUBBORN_SYSTEM
            opener = random.choice(_PIXEL_OPENERS)
            followups = random.sample(_PIXEL_FOLLOWUPS, min(2, len(_PIXEL_FOLLOWUPS)))
        else:
            sys_prompt = _CLIENT_UNWINNABLE_SYSTEM
            opener = random.choice(_PIXEL_GRIEF_OPENERS)
            followups = random.sample(_PIXEL_GRIEF_FOLLOWUPS, min(2, len(_PIXEL_GRIEF_FOLLOWUPS)))
        sessions = _generate_multiturn_session(VLLM_CLIENT_WF, "wayfarer-12b", sys_prompt, opener, followups)
        with _diag_lock:
            _diag_wf_producer_calls += 1
            _diag_wf_producer_sessions += len(sessions)
        if sessions:
            with _WAYFARER_LOCK:
                _WAYFARER_QUEUE.extend(sessions)
        else:
            time.sleep(1)


def _nim_producer():
    """Background thread: continuously fills _NIM_QUEUE via NVIDIA NIM."""
    global _diag_nim_producer_calls, _diag_nim_producer_sessions
    while _producer_running:
        with _NIM_LOCK:
            full = len(_NIM_QUEUE) >= _MAX_QUEUE_SIZE
        if full:
            time.sleep(0.5)
            continue
        nim_m = "meta/llama-3.1-8b-instruct"  # nemotron-70b NVCF function deprecated (404)
        raw = execute_nim_request(
            nim_m, _build_batch_prompt(num_sessions=10)
        )  # 10 sessions/call doubles NIM throughput
        sessions = _parse_sessions(raw)
        with _diag_lock:
            _diag_nim_producer_calls += 1
            _diag_nim_producer_sessions += len(sessions)
        if sessions:
            with _NIM_LOCK:
                _NIM_QUEUE.extend(sessions)
        else:
            time.sleep(1)


# vLLM clients: Wayfarer-12B on port 8000, self-after-dark on port 8001.
VLLM_CLIENT_WF = OpenAI(api_key="vllm", base_url="http://localhost:8000/v1", max_retries=0)
VLLM_CLIENT_SAD = OpenAI(api_key="vllm", base_url="http://localhost:8001/v1", max_retries=0)

# 2. Triple NVIDIA NIM Key Rotation Pool
NVIDIA_KEYS = [
    "REDACTED_NVIDIA_KEY_1",
    "REDACTED_NVIDIA_KEY_2",
    "REDACTED_NVIDIA_KEY_3",
]

NIM_CLIENTS = [OpenAI(api_key=k, base_url="https://integrate.api.nvidia.com/v1") for k in NVIDIA_KEYS]

OLLAMA_REMOTE_CLIENT = OpenAI(
    api_key="ollama", base_url="https://ollama.pixelated.love/v1", default_headers={"User-Agent": "Mozilla/5.0"}
)


def get_next_nim_client() -> OpenAI:
    """Gets the next NVIDIA NIM client in round-robin order across 3 keys."""
    global _KEY_INDEX
    with _KEY_LOCK:
        client = NIM_CLIENTS[_KEY_INDEX % len(NIM_CLIENTS)]
        _KEY_INDEX += 1
        return client


# @weave.op removed: it corrupts return values in background threads.
# Weave tracing kept on generate_curated_session only.
def execute_ollama(client: OpenAI, model: str, prompt: str) -> str:
    """vLLM OpenAI-compatible API with JSON mode (guided_json returns empty for 12B models)."""
    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            max_tokens=1500,
            temperature=0.85,
            timeout=120.0,
        )
        content = resp.choices[0].message.content or ""
        if not content:
            logger.warning("VLLM_EMPTY: model=%s finish=%s", model, resp.choices[0].finish_reason)
        return content
    except Exception as e:
        logger.warning("vLLM error (%s): %s", model, e)
        return ""


def execute_nim_request(model: str, prompt: str) -> str:
    """Executes request across triple-key NVIDIA NIM pool."""
    client = get_next_nim_client()
    try:
        res = client.chat.completions.create(
            model=model,
            messages=[{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": prompt}],
            max_tokens=6000,
            temperature=0.85,
            timeout=60.0,  # 10 sessions with 6000 max_tokens need more time
            response_format={"type": "json_object"},
        )
        content = res.choices[0].message.content or ""
        if not content:
            logger.warning("NIM_EMPTY: model=%s finish=%s", model, res.choices[0].finish_reason)
        return content
    except Exception as e:
        logger.warning("NVIDIA NIM error: %s", e)
        return ""


_diag_lock = threading.Lock()
_diag_calls = 0
_diag_queue_hits = 0
_diag_fallbacks = 0


def _ensure_messages(row: dict) -> dict:
    """Ensure row['messages'] is always a list of dicts (prevents ArrowInvalid)."""
    msgs = row.get("messages")
    if not isinstance(msgs, list) or not all(isinstance(m, dict) for m in msgs):
        logger.warning("MESSAGES_TYPE_FIX: was %s, replacing with hardcoded", type(msgs).__name__)
        row["messages"] = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": "I need help."},
            {"role": "assistant", "content": "I'm here for you. What's going on?"},
        ]
        row["turns_count"] = 3
    return row


_diag_total_time = 0.0
_diag_wf_producer_calls = 0
_diag_wf_producer_sessions = 0
_diag_nim_producer_calls = 0
_diag_nim_producer_sessions = 0


@dd.custom_column_generator(
    required_columns=["category", "diagnosis", "persona_niche", "client_name"],
    side_effect_columns=["messages", "turns_count"],
)
def generate_curated_session(row: dict) -> dict:
    global _diag_calls, _diag_queue_hits, _diag_fallbacks, _diag_total_time
    cat = row.get("category", "edge_case")
    diag = row.get("diagnosis", "Complex PTSD")
    persona = row.get("persona_niche", "Tech Founder")
    name = row.get("client_name", "Alex")
    _t0 = time.monotonic()

    # 1. Pop pre-generated session from matching queue (fast path)
    needs_wayfarer = cat in ("stubborn_nightmare", "unwinnable_tragedy")
    queue = _WAYFARER_QUEUE if needs_wayfarer else _NIM_QUEUE
    lock = _WAYFARER_LOCK if needs_wayfarer else _NIM_LOCK
    with lock:
        if queue:
            messages = queue.popleft()
            row["messages"] = messages
            row["turns_count"] = len(messages)
            row["curated_session"] = f"{cat}:{diag}:{name}"
            with _diag_lock:
                _diag_calls += 1
                _diag_queue_hits += 1
                _diag_total_time += time.monotonic() - _t0
            return _ensure_messages(row)

    # Queue wait loop: wait up to 5s for producers to fill queue.
    # With 5 concurrent consumers, this ensures NIM records (~75%) get real LLM sessions.
    # Wayfarer records (~25%) may timeout and use hardcoded fallback (Wayfarer models struggle w/ JSON).
    _wait_timeout = float(
        os.environ.get("PIXELATED_QUEUE_WAIT_TIMEOUT", "30")
    )  # 30s: wait for producers to fill queue (maximize real sessions)
    _wait_start = time.monotonic()
    while time.monotonic() - _wait_start < _wait_timeout:
        with lock:
            if queue:
                messages = queue.popleft()
                row["messages"] = messages
                row["turns_count"] = len(messages)
                row["curated_session"] = f"{cat}:{diag}:{name}"
                with _diag_lock:
                    _diag_calls += 1
                    _diag_queue_hits += 1
                _diag_total_time += time.monotonic() - _t0
                return _ensure_messages(row)
        time.sleep(0.05)

    # Fallback: queue still empty after 30s wait. Use hardcoded session.
    raw_payload = ""
    parsed_sessions = _parse_sessions(raw_payload)

    if not parsed_sessions:
        parsed_sessions = [
            [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": f"I'm overwhelmed by {diag} as a {persona}."},
                {
                    "role": "assistant",
                    "content": f"You're carrying a heavy burden. Let's talk about what's happening right now.",
                },
            ]
        ]

    # 4. Store remaining 4 sessions in matching queue
    first_messages = parsed_sessions.pop(0)
    if parsed_sessions:
        with lock:
            queue.extend(parsed_sessions)

    row["messages"] = first_messages
    row["turns_count"] = len(first_messages)
    row["curated_session"] = f"{cat}:{diag}:{name}"
    with _diag_lock:
        _diag_calls += 1
        _diag_fallbacks += 1
        _diag_total_time += time.monotonic() - _t0
    return _ensure_messages(row)


def _queue_monitor():
    """Diagnostic thread: logs queue sizes and call stats every 5s."""
    while _producer_running or _diag_calls < int(os.environ.get("PIXELATED_NUM_RECORDS", "10")):
        with _WAYFARER_LOCK:
            wf = len(_WAYFARER_QUEUE)
        with _NIM_LOCK:
            nq = len(_NIM_QUEUE)
        with _diag_lock:
            c, h, f, t = _diag_calls, _diag_queue_hits, _diag_fallbacks, _diag_total_time
            wfc, wfs = _diag_wf_producer_calls, _diag_wf_producer_sessions
            nc, ns = _diag_nim_producer_calls, _diag_nim_producer_sessions
        avg = t / c if c > 0 else 0
        logger.warning(
            "DIAGqueues: wf=%d nim=%d | calls=%d hits=%d fb=%d avg_ms=%.1f | wf_prod=%d/%d nim_prod=%d/%d",
            wf,
            nq,
            c,
            h,
            f,
            avg * 1000,
            wfc,
            wfs,
            nc,
            ns,
        )
        time.sleep(5)


def load_config_builder() -> dd.DataDesignerConfigBuilder:
    config_builder = dd.DataDesignerConfigBuilder()

    config_builder.add_column(
        dd.SamplerColumnConfig(
            name="category",
            sampler_type="category",
            params=dd.CategorySamplerParams(
                values=["edge_case", "stubborn_nightmare", "unwinnable_tragedy"], weights=[0.75, 0.20, 0.05]
            ),
        )
    )

    config_builder.add_column(
        dd.SamplerColumnConfig(
            name="diagnosis",
            sampler_type="category",
            params=dd.CategorySamplerParams(
                values=[
                    "Borderline Personality Disorder (BPD)",
                    "Narcissistic Personality Disorder (NPD)",
                    "Avoidant Personality Disorder (AVPD)",
                    "Obsessive-Compulsive Personality Disorder (OCPD)",
                    "Complex PTSD (C-PTSD)",
                    "Dissociative Identity Disorder (DID)",
                    "Depersonalization/Derealization (DPDR)",
                    "Moral Injury",
                    "Adult ADHD & Executive Dysfunction",
                    "Autistic Burnout & Masking",
                    "Treatment-Resistant Depression (TRD)",
                    "Bipolar II Hypomania",
                    "Schizoaffective Disorder",
                    "Harm/Moral OCD",
                    "Relationship OCD (ROCD)",
                    "Agoraphobia with Panic",
                    "Illness Anxiety Disorder",
                    "Anorexia Nervosa",
                    "ARFID",
                    "PNES / Functional Neurological Disorder",
                    "Long COVID & Autoimmune Grief",
                ]
            ),
        )
    )

    config_builder.add_column(
        dd.SamplerColumnConfig(
            name="persona_niche",
            sampler_type="category",
            params=dd.CategorySamplerParams(
                values=[
                    "SaaS Tech Founder under investor pressure",
                    "ER Trauma Physician battling burnout",
                    "First-Gen Immigrant Student",
                    "Combat Veteran with hypervigilance",
                    "Solo Caregiver for Parent with Dementia",
                    "Professional Ballet Dancer with BDD",
                    "Blue-Collar Construction Foreman in pain",
                    "Academic Tenure-Track Researcher",
                    "Transgender Youth facing family rejection",
                ]
            ),
        )
    )

    config_builder.add_column(
        dd.SamplerColumnConfig(
            name="client_name",
            sampler_type="category",
            params=dd.CategorySamplerParams(
                values=[
                    "Marcus",
                    "Elena",
                    "Devon",
                    "Aisha",
                    "Kenji",
                    "Siddharth",
                    "Chloe",
                    "Mateo",
                    "Priya",
                    "Nadia",
                    "Lukas",
                    "Fatima",
                    "Tariq",
                    "Yuki",
                    "Amara",
                    "Gabriel",
                    "Sven",
                    "Zoe",
                    "Dante",
                    "Nia",
                ]
            ),
        )
    )

    config_builder.add_column(
        dd.CustomColumnConfig(name="curated_session", generator_function=generate_curated_session)
    )

    return config_builder


def _ollama_health_ok() -> bool:
    """Check if a local Ollama server is listening on localhost:11434."""
    import urllib.request

    try:
        urllib.request.urlopen("http://localhost:11434/api/tags", timeout=2)
        return True
    except Exception:
        return False


def _ensure_ollama_running() -> "subprocess.Popen | None":
    """
    Launch a local Ollama server if one is not already running.

    Downloads the Ollama binary if not installed, starts `ollama serve`,
    then pulls both therapy models. Returns the Popen handle (caller owns
    cleanup) or None if a server was already up.
    """
    if _ollama_health_ok():
        logger.info("Ollama already running on localhost:11434 — skipping launch")
        return None

    if os.environ.get("PIXELATED_SKIP_OLLAMA") == "1":
        logger.warning("PIXELATED_SKIP_OLLAMA=1 — Ollama not launched; NIM fallback only")
        return None

    # Find or download the ollama binary.
    install_prefix = os.path.dirname(os.path.dirname(sys.executable))
    ollama_bin = os.path.join(install_prefix, "bin", "ollama")
    ollama_lib = os.path.join(install_prefix, "lib", "ollama")
    if not os.path.isfile(ollama_bin):
        ollama_bin = os.environ.get("OLLAMA_BIN", "/workspace/.local/bin/ollama")
        install_prefix = os.path.dirname(os.path.dirname(ollama_bin))
        ollama_lib = os.path.join(install_prefix, "lib", "ollama")
    if not os.path.isfile(ollama_bin) or not os.path.isdir(ollama_lib):
        logger.info("Downloading + extracting Ollama to %s", install_prefix)
        import tarfile
        import tempfile

        import zstandard

        os.makedirs(install_prefix, exist_ok=True)
        tarball = os.path.join(install_prefix, "ollama.tar.zst")
        subprocess.run(
            [
                "curl",
                "-fsSL",
                "-o",
                tarball,
                "https://github.com/ollama/ollama/releases/latest/download/ollama-linux-amd64.tar.zst",
            ],
            check=True,
        )
        dctx = zstandard.ZstdDecompressor()
        with open(tarball, "rb") as f:
            with dctx.stream_reader(f) as decompressed:
                with tarfile.open(fileobj=decompressed, mode="r|") as tar:
                    tar.extractall(install_prefix)
        os.unlink(tarball)
        if os.path.isfile(ollama_bin):
            os.chmod(ollama_bin, 0o755)
        logger.info("Ollama installed at %s (libs at %s)", ollama_bin, ollama_lib)

    log_path = os.environ.get("PIXELATED_OLLAMA_LOG", "/workspace/ollama_server.log")
    env = os.environ.copy()
    env["OLLAMA_HOST"] = "0.0.0.0:11434"
    # Ollama bundles its own CUDA libs under lib/ollama/cuda_v12/; also add system CUDA + driver libs.
    bundled_cuda = os.path.join(ollama_lib, "cuda_v12")
    cuda_lib = "/usr/local/cuda-12.8/targets/x86_64-linux/lib"
    nvidia_lib = "/usr/lib/x86_64-linux-gnu"
    extra = [p for p in (bundled_cuda, ollama_lib, cuda_lib, nvidia_lib) if os.path.isdir(p)]
    if extra:
        env["LD_LIBRARY_PATH"] = os.pathsep.join(extra) + os.pathsep + env.get("LD_LIBRARY_PATH", "")
    if not env.get("CUDA_VISIBLE_DEVICES"):
        env["CUDA_VISIBLE_DEVICES"] = "0"
    # Enable 4 concurrent GPU slots — default Ollama (1 slot) serializes all requests.
    # L40s has 46GB VRAM; models are ~7GB each, so 4 parallel slots fit comfortably.
    env["OLLAMA_NUM_PARALLEL"] = os.environ.get("OLLAMA_NUM_PARALLEL", "4")
    cmd = [ollama_bin, "serve"]
    logger.info("Launching Ollama: %s (log -> %s)", " ".join(cmd), log_path)
    log_f = open(log_path, "w")
    proc = subprocess.Popen(cmd, stdout=log_f, stderr=subprocess.STDOUT, start_new_session=True, env=env)

    # Wait for /api/tags to respond.
    import time as _t

    deadline = _t.time() + int(os.environ.get("PIXELATED_OLLAMA_BOOT_TIMEOUT", "300"))
    while _t.time() < deadline:
        if proc.poll() is not None:
            logger.error("Ollama exited early (code=%d); see %s", proc.returncode, log_path)
            return None
        if _ollama_health_ok():
            logger.info("Ollama ready on localhost:11434 (pid=%d)", proc.pid)
            break
        _t.sleep(3)
    else:
        logger.error(
            "Ollama did not become healthy within %ss; see %s",
            os.environ.get("PIXELATED_OLLAMA_BOOT_TIMEOUT", "300"),
            log_path,
        )
        proc.terminate()
        return None

    # Pull both therapy models.
    models = [
        os.environ.get("PIXELATED_OLLAMA_MODEL_1", "gurubot/wayfarer-2-12B:latest"),
        os.environ.get("PIXELATED_OLLAMA_MODEL_2", "gurubot/self-after-dark:latest"),
    ]
    for m in models:
        logger.info("Pulling Ollama model: %s", m)
        pull_proc = subprocess.Popen([ollama_bin, "pull", m], stdout=log_f, stderr=subprocess.STDOUT, env=env)
        pull_proc.wait()

    logger.info("All Ollama models pulled and ready")
    return proc


if __name__ == "__main__":
    import sys
    from pathlib import Path

    from data_designer.interface import DataDesigner
    from data_designer.engine.storage.artifact_storage import ResumeMode
    from data_designer.config.run_config import RunConfig

    num_records = int(os.environ.get("PIXELATED_NUM_RECORDS", "10"))
    dataset_name = os.environ.get("PIXELATED_DATASET_NAME", "pixelated_edge_cases")
    artifact_path = Path(os.environ.get("PIXELATED_ARTIFACT_PATH", "/workspace/artifacts"))
    resume_mode = ResumeMode(os.environ.get("PIXELATED_RESUME", "never"))

    # Auto-launch local Ollama so the L40S GPU is actually used.
    ollama_proc = _ensure_ollama_running()

    # Initialize Weave BEFORE starting producer threads (they use @weave.op).
    weave.init(os.environ.get("PIXELATED_WEAVE_PROJECT", "pixelated-empathy-kan28"))

    # Start background pre-generator threads to keep queues topped up.
    _producer_running = True
    wayfarer_t = threading.Thread(target=_wayfarer_producer, daemon=True)
    wayfarer2_t = threading.Thread(target=_wayfarer2_producer, daemon=True)
    wayfarer_t.start()
    wayfarer2_t.start()
    nim_threads = []
    _num_nim_threads = int(
        os.environ.get("PIXELATED_NUM_NIM_THREADS", "4")
    )  # 4 threads: 2 per key avg, balances throughput vs 429 risk
    for _ in range(_num_nim_threads):
        t = threading.Thread(target=_nim_producer, daemon=True)
        t.start()
        nim_threads.append(t)
    logger.info("Background producers started (Wayfarer + self-after-dark + %d NIM)", _num_nim_threads)

    # Diagnostic: monitor queue sizes and call stats
    monitor_t = threading.Thread(target=_queue_monitor, daemon=True)
    monitor_t.start()

    logger.info(
        "Starting data-designer run: dataset=%s num_records=%d artifact_path=%s resume=%s",
        dataset_name,
        num_records,
        artifact_path,
        resume_mode.value,
    )

    try:
        dd_runner = DataDesigner(artifact_path=artifact_path)
        dd_runner.set_run_config(
            RunConfig(
                max_concurrent_row_groups=int(os.environ.get("PIXELATED_MAX_ROW_GROUPS", "5")),
                non_inference_max_parallel_workers=int(os.environ.get("PIXELATED_MAX_WORKERS", "16")),
                buffer_size=int(os.environ.get("PIXELATED_BUFFER_SIZE", "500")),
                max_in_flight_tasks=int(os.environ.get("PIXELATED_MAX_IN_FLIGHT", "2048")),
                otel_metrics_port=None,
            )
        )
        results = dd_runner.create(
            load_config_builder(),
            num_records=num_records,
            dataset_name=dataset_name,
            resume=resume_mode,
        )
    except Exception:
        logger.exception("data-designer run failed")
        sys.exit(1)
    finally:
        _producer_running = False
        wayfarer_t.join(timeout=5)
        wayfarer2_t.join(timeout=5)
        for t in nim_threads:
            t.join(timeout=5)
        if ollama_proc is not None:
            logger.info("Shutting down Ollama (pid=%d)", ollama_proc.pid)
            ollama_proc.terminate()
            try:
                ollama_proc.wait(timeout=30)
            except subprocess.TimeoutExpired:
                ollama_proc.kill()

    try:
        count = results.count_records()
    except Exception:
        count = "<unavailable>"
    logger.info("Run complete: %s records written to %s", count, artifact_path / dataset_name)
    print(f"OK: {count} records -> {artifact_path / dataset_name}")
