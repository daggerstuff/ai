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

_WAYFARER_QUEUE = deque()
_NIM_QUEUE = deque()
_QUEUE_LOCK = threading.Lock()
_KEY_INDEX = 0
_KEY_LOCK = threading.Lock()

_MAX_QUEUE_SIZE = 50
_producer_running = False


def _build_batch_prompt(persona: str = "a client", diag: str = "various conditions") -> str:
    return (
        f"Generate 5 distinct realistic 4-turn therapy dialogues between clients ({persona}, {diag}) and Pixel (therapist). "
        f"Each session must have 4 alternate turns (user, assistant, user, assistant). "
        f'Output strictly JSON matching: {{"sessions": [[{{"role": "user"|"assistant", "content": "..."}}]]}}'
    )


def _parse_sessions(raw_payload: str) -> list:
    if not raw_payload:
        return []
    try:
        clean_json = raw_payload.replace("```json", "").replace("```", "").strip()
        data = json.loads(clean_json)
        if isinstance(data, dict) and "sessions" in data and isinstance(data["sessions"], list):
            result = []
            for s in data["sessions"]:
                if isinstance(s, list) and len(s) > 0:
                    result.append([{"role": "system", "content": SYSTEM_PROMPT}] + s)
            return result
    except Exception:
        pass
    return []


def _wayfarer_producer():
    """Background thread: fills _WAYFARER_QUEUE via local Ollama (Wayfarer-12B)."""
    while _producer_running:
        with _QUEUE_LOCK:
            if len(_WAYFARER_QUEUE) >= _MAX_QUEUE_SIZE:
                time.sleep(0.5)
                continue
        raw = execute_ollama("gurubot/wayfarer-2-12B:latest", _build_batch_prompt())
        sessions = _parse_sessions(raw)
        if sessions:
            with _QUEUE_LOCK:
                _WAYFARER_QUEUE.extend(sessions)
        else:
            time.sleep(1)


def _wayfarer2_producer():
    """Background thread: fills _WAYFARER_QUEUE via local Ollama (self-after-dark)."""
    while _producer_running:
        with _QUEUE_LOCK:
            if len(_WAYFARER_QUEUE) >= _MAX_QUEUE_SIZE:
                time.sleep(0.5)
                continue
        raw = execute_ollama("gurubot/self-after-dark:latest", _build_batch_prompt())
        sessions = _parse_sessions(raw)
        if sessions:
            with _QUEUE_LOCK:
                _WAYFARER_QUEUE.extend(sessions)
        else:
            time.sleep(1)


def _nim_producer():
    """Background thread: continuously fills _NIM_QUEUE via NVIDIA NIM."""
    while _producer_running:
        with _QUEUE_LOCK:
            if len(_NIM_QUEUE) >= _MAX_QUEUE_SIZE:
                time.sleep(0.5)
                continue
        nim_m = random.choice(["meta/llama-3.1-8b-instruct", "nvidia/llama-3.1-nemotron-70b-instruct"])
        raw = execute_nim_request(nim_m, _build_batch_prompt())
        sessions = _parse_sessions(raw)
        if sessions:
            with _QUEUE_LOCK:
                _NIM_QUEUE.extend(sessions)
        else:
            time.sleep(1)


# 1. Local Ollama Engine Client (L40s GPU, OpenAI-compatible API on port 11434)
OLLAMA_CLIENT = OpenAI(api_key="ollama", base_url="http://localhost:11434/v1", max_retries=0)

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
def execute_ollama(model: str, prompt: str) -> str:
    """Local inference on L40s GPU via Ollama (OpenAI-compatible API)."""
    try:
        res = OLLAMA_CLIENT.chat.completions.create(
            model=model,
            messages=[{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": prompt}],
            max_tokens=1500,
            temperature=0.85,
            timeout=30.0,
        )
        return res.choices[0].message.content or ""
    except Exception as e:
        logger.debug("Ollama error (%s): %s", model, e)
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

    # 1. Pop pre-generated session from matching queue
    needs_wayfarer = cat in ("stubborn_nightmare", "unwinnable_tragedy")
    queue = _WAYFARER_QUEUE if needs_wayfarer else _NIM_QUEUE
    with _QUEUE_LOCK:
        if queue:
            messages = queue.popleft()
            row["messages"] = messages
            row["turns_count"] = len(messages)
            row["curated_session"] = f"{cat}:{diag}:{name}"
            return row

    batch_prompt = _build_batch_prompt(persona, diag)

    raw_payload = ""

    # 2. Route: Wayfarer for stubborn/unwinnable, NIM for regular edge cases
    if needs_wayfarer:
        model = random.choice(["gurubot/wayfarer-2-12B:latest", "gurubot/self-after-dark:latest"])
        raw_payload = execute_ollama(model, batch_prompt)
        if not raw_payload:
            time.sleep(1)
            raw_payload = execute_ollama(model, batch_prompt)
    else:
        nim_m = random.choice(["meta/llama-3.1-8b-instruct", "nvidia/llama-3.1-nemotron-70b-instruct"])
        raw_payload = execute_nim_request(nim_m, batch_prompt)

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
        with _QUEUE_LOCK:
            queue.extend(parsed_sessions)

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
    ollama_bin = os.path.join(os.path.dirname(sys.executable), "ollama")
    if not os.path.isfile(ollama_bin):
        ollama_bin = os.environ.get("OLLAMA_BIN", "/workspace/.local/bin/ollama")
    if not os.path.isfile(ollama_bin):
        logger.info("Downloading + extracting Ollama to %s", ollama_bin)
        import tarfile
        import tempfile

        import zstandard

        os.makedirs(os.path.dirname(ollama_bin), exist_ok=True)
        tarball = ollama_bin + ".tar.zst"
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
                    with tempfile.TemporaryDirectory() as tmpdir:
                        tar.extractall(tmpdir)
                        extracted = os.path.join(tmpdir, "bin", "ollama")
                        if not os.path.isfile(extracted):
                            for root, _dirs, files in os.walk(tmpdir):
                                if "ollama" in files:
                                    extracted = os.path.join(root, "ollama")
                                    break
                        import shutil

                        shutil.copy2(extracted, ollama_bin)
        os.unlink(tarball)
        os.chmod(ollama_bin, 0o755)
        logger.info("Ollama binary installed at %s", ollama_bin)

    log_path = os.environ.get("PIXELATED_OLLAMA_LOG", "/workspace/ollama_server.log")
    env = os.environ.copy()
    env["OLLAMA_HOST"] = "0.0.0.0:11434"
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

    num_records = int(os.environ.get("PIXELATED_NUM_RECORDS", "10"))
    dataset_name = os.environ.get("PIXELATED_DATASET_NAME", "pixelated_edge_cases")
    artifact_path = Path(os.environ.get("PIXELATED_ARTIFACT_PATH", "/workspace/artifacts"))
    resume_mode = ResumeMode(os.environ.get("PIXELATED_RESUME", "never"))

    # Auto-launch local Ollama so the L40S GPU is actually used.
    ollama_proc = _ensure_ollama_running()

    # Start background pre-generator threads to keep queues topped up.
    _producer_running = True
    wayfarer_t = threading.Thread(target=_wayfarer_producer, daemon=True)
    wayfarer2_t = threading.Thread(target=_wayfarer2_producer, daemon=True)
    nim_t = threading.Thread(target=_nim_producer, daemon=True)
    wayfarer_t.start()
    wayfarer2_t.start()
    nim_t.start()
    logger.info("Background producers started (Wayfarer + self-after-dark + NIM)")

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
        _producer_running = False
        wayfarer_t.join(timeout=5)
        wayfarer2_t.join(timeout=5)
        nim_t.join(timeout=5)
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
