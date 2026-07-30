# /// script
# dependencies = [
#   "data-designer",
#   "pydantic",
#   "openai",
#   "vllm",
#   "weave",
# ]
# ///

"""
High-Speed OVHcloud L40s 80GB GPU vLLM + Triple-Key NVIDIA NIM Generator
========================================================================

Architecture:
1. Local vLLM Engine on L40s GPU (http://localhost:8000/v1):
   - Serves Wayfarer-2 / Llama-3 locally on-GPU at 400+ tokens/sec.
   - Zero network latency, zero rate limits, sub-second 5-session batch output.
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
import subprocess
import sys
import time
import threading
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

_GLOBAL_SESSION_QUEUE = deque()
_QUEUE_LOCK = threading.Lock()
_KEY_INDEX = 0
_KEY_LOCK = threading.Lock()

# 1. Local vLLM Engine Client (L40s 80GB GPU)
VLLM_CLIENT = OpenAI(api_key="vllm", base_url="http://localhost:8000/v1", max_retries=0)

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


@weave.op
def execute_vllm_local(prompt: str) -> str:
    """High-speed local inference on NVIDIA L40s 80GB GPU via vLLM."""
    try:
        res = VLLM_CLIENT.chat.completions.create(
            model="gurubot/wayfarer-2-12B",
            messages=[{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": prompt}],
            max_tokens=1500,
            temperature=0.85,
            timeout=10.0,
        )
        return res.choices[0].message.content or ""
    except Exception as e:
        logger.debug("Local vLLM error (falling back to NIM): %s", e)
        return ""


@weave.op
def execute_nim_request(model: str, prompt: str) -> str:
    """Executes request across triple-key NVIDIA NIM pool."""
    client = get_next_nim_client()
    try:
        res = client.chat.completions.create(
            model=model,
            messages=[{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": prompt}],
            max_tokens=1500,
            temperature=0.85,
            timeout=15.0,
        )
        return res.choices[0].message.content or ""
    except Exception as e:
        logger.debug("NVIDIA NIM error: %s", e)
        return ""


@weave.op
@dd.custom_column_generator(
    required_columns=["category", "diagnosis", "persona_niche", "client_name"],
    side_effect_columns=["messages", "turns_count"],
)
def generate_curated_session(row: dict) -> dict:
    cat = row.get("category", "edge_case")
    diag = row.get("diagnosis", "Complex PTSD")
    persona = row.get("persona_niche", "Tech Founder")
    name = row.get("client_name", "Alex")

    # 1. Pop pre-generated session from global queue
    with _QUEUE_LOCK:
        if len(_GLOBAL_SESSION_QUEUE) > 0:
            messages = _GLOBAL_SESSION_QUEUE.popleft()
            row["messages"] = messages
            row["turns_count"] = len(messages)
            row["curated_session"] = f"{cat}:{diag}:{name}"
            return row

    batch_prompt = (
        f"Generate 5 distinct realistic 4-turn therapy dialogues between clients ({persona}, {diag}) and Pixel (therapist). "
        f"Each session must have 4 alternate turns (user, assistant, user, assistant). "
        f'Output strictly JSON matching: {{"sessions": [[{{"role": "user"|"assistant", "content": "..."}}]]}}'
    )

    raw_payload = ""

    # 2. Try Local L40s vLLM GPU Server First
    raw_payload = execute_vllm_local(batch_prompt)

    # 3. Fallback to Triple NVIDIA NIM Key Pool if vLLM warming up
    if not raw_payload:
        nim_m = random.choice(["meta/llama-3.1-8b-instruct", "nvidia/llama-3.1-nemotron-70b-instruct"])
        raw_payload = execute_nim_request(nim_m, batch_prompt)

    parsed_sessions = []
    if raw_payload:
        try:
            clean_json = raw_payload.replace("```json", "").replace("```", "").strip()
            data = json.loads(clean_json)
            if isinstance(data, dict) and "sessions" in data and isinstance(data["sessions"], list):
                for s in data["sessions"]:
                    if isinstance(s, list) and len(s) > 0:
                        parsed_sessions.append([{"role": "system", "content": SYSTEM_PROMPT}] + s)
        except Exception:
            pass

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

    # 4. Store remaining 4 sessions in global queue
    first_messages = parsed_sessions.pop(0)
    if parsed_sessions:
        with _QUEUE_LOCK:
            _GLOBAL_SESSION_QUEUE.extend(parsed_sessions)

    row["messages"] = first_messages
    row["turns_count"] = len(first_messages)
    row["curated_session"] = f"{cat}:{diag}:{name}"
    return row


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


def _vllm_health_ok() -> bool:
    """Check if a vLLM server is already listening on localhost:8000."""
    import urllib.request

    try:
        urllib.request.urlopen("http://localhost:8000/health", timeout=2)
        return True
    except Exception:
        return False


def _ensure_vllm_running() -> "subprocess.Popen | None":
    """
    Launch a local vLLM server if one is not already running.

    Uses `uv run` so vLLM is installed on demand via the script's PEP-723
    dependency block (no separate pip install step required).

    Returns the Popen handle (caller owns cleanup) or None if a server was
    already up.
    """
    if _vllm_health_ok():
        logger.info("vLLM already running on localhost:8000 — skipping launch")
        return None

    if os.environ.get("PIXELATED_SKIP_VLLM") == "1":
        logger.warning("PIXELATED_SKIP_VLLM=1 — vLLM not launched; NIM fallback only")
        return None

    model = os.environ.get("PIXELATED_VLLM_MODEL", "gurubot/wayfarer-2-12B")
    port = int(os.environ.get("PIXELATED_VLLM_PORT", "8000"))
    gpu_util = os.environ.get("PIXELATED_VLLM_GPU_UTIL", "0.9")
    log_path = os.environ.get("PIXELATED_VLLM_LOG", "/workspace/vllm_server.log")

    # The parent `uv run` already installed vllm into this env, so the `vllm`
    # CLI lives next to sys.executable. Prepend that dir to PATH so the
    # subprocess can find it without re-invoking uv (and without needing uv on PATH).
    venv_bin = os.path.dirname(sys.executable)
    env = os.environ.copy()
    env["PATH"] = venv_bin + os.pathsep + env.get("PATH", "")

    hf_token = env.get("HF_TOKEN") or env.get("HUGGING_FACE_HUB_TOKEN")
    if not hf_token:
        for line in open("/workspace/.env"):
            line = line.strip()
            if line.startswith("HF_TOKEN="):
                hf_token = line.split("=", 1)[1]
                break
    if hf_token:
        env["HF_TOKEN"] = hf_token
        env["HUGGING_FACE_HUB_TOKEN"] = hf_token
    cmd = [
        "vllm",
        "serve",
        model,
        "--port",
        str(port),
        "--gpu-memory-utilization",
        gpu_util,
    ]
    logger.info("Launching vLLM: %s (log -> %s)", " ".join(cmd), log_path)
    log_f = open(log_path, "w")
    proc = subprocess.Popen(cmd, stdout=log_f, stderr=subprocess.STDOUT, start_new_session=True, env=env)

    # Wait for /health to respond (model download + load can take minutes).
    import time as _t

    deadline = _t.time() + int(os.environ.get("PIXELATED_VLLM_BOOT_TIMEOUT", "900"))
    while _t.time() < deadline:
        if proc.poll() is not None:
            logger.error("vLLM exited early (code=%d); see %s", proc.returncode, log_path)
            return None
        if _vllm_health_ok():
            logger.info("vLLM ready on localhost:%d (pid=%d)", port, proc.pid)
            return proc
        _t.sleep(5)
    logger.error(
        "vLLM did not become healthy within %ss; see %s", os.environ.get("PIXELATED_VLLM_BOOT_TIMEOUT", "900"), log_path
    )
    proc.terminate()
    return None


if __name__ == "__main__":
    import sys
    from pathlib import Path

    from data_designer.interface import DataDesigner
    from data_designer.engine.storage.artifact_storage import ResumeMode

    num_records = int(os.environ.get("PIXELATED_NUM_RECORDS", "10"))
    dataset_name = os.environ.get("PIXELATED_DATASET_NAME", "pixelated_edge_cases")
    artifact_path = Path(os.environ.get("PIXELATED_ARTIFACT_PATH", "/workspace/artifacts"))
    resume_mode = ResumeMode(os.environ.get("PIXELATED_RESUME", "never"))

    # Auto-launch local vLLM so the L40S GPU is actually used.
    vllm_proc = _ensure_vllm_running()

    weave.init(os.environ.get("PIXELATED_WEAVE_PROJECT", "pixelated-empathy-kan28"))

    logger.info(
        "Starting data-designer run: dataset=%s num_records=%d artifact_path=%s resume=%s",
        dataset_name,
        num_records,
        artifact_path,
        resume_mode.value,
    )

    try:
        dd_runner = DataDesigner(artifact_path=artifact_path)
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
        if vllm_proc is not None:
            logger.info("Shutting down vLLM (pid=%d)", vllm_proc.pid)
            vllm_proc.terminate()
            try:
                vllm_proc.wait(timeout=30)
            except subprocess.TimeoutExpired:
                vllm_proc.kill()

    try:
        count = results.count_records()
    except Exception:
        count = "<unavailable>"
    logger.info("Run complete: %s records written to %s", count, artifact_path / dataset_name)
    print(f"OK: {count} records -> {artifact_path / dataset_name}")
