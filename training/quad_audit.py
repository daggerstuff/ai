import argparse
import logging
import os
import re
import shutil
import time

import requests

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

# Safely load from .env directly without requiring python-dotenv
env_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), ".env")
if os.path.exists(env_path):
    with open(env_path) as f:
        for line in f:
            if line.strip() and not line.startswith("#"):
                key, _, val = line.partition("=")
                key = key.strip()
                val = val.strip().strip("'").strip('"')
                if key and key not in os.environ:
                    os.environ[key] = val

NVIDIA_API_KEY = os.environ.get("NVIDIA_API_KEY", os.environ.get("NIM_API_KEY", ""))
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")

# Maximum characters of file content sent per LLM call.
MAX_CONTENT_CHARS = 16_000
HTTP_OK = 200
MIN_SYNC_FILES = 2
MAX_RETRIES = 3
RETRY_BASE_DELAY = 2  # seconds
BACKUP_SUFFIX = ".bak"


def _truncate(content, max_chars=MAX_CONTENT_CHARS):
    if len(content) <= max_chars:
        return content, False
    return content[:max_chars], True


def _retry_request(fn, max_retries=MAX_RETRIES, base_delay=RETRY_BASE_DELAY):
    """Retry a request function with exponential backoff on transient failures."""
    for attempt in range(1, max_retries + 1):
        try:
            return fn()
        except (requests.exceptions.ConnectionError, requests.exceptions.Timeout) as e:
            if attempt == max_retries:
                raise
            delay = base_delay * 2 ** (attempt - 1)
            logger.warning(
                f"  {C_YELLOW}⚠ Attempt {attempt}/{max_retries} failed: {e}. Retrying in {delay}s...{C_RESET}"
            )
            time.sleep(delay)
    return None


def _backup_file(path):
    """Create a .bak copy before overwriting. Returns the backup path or None."""
    backup_path = path + BACKUP_SUFFIX
    try:
        shutil.copy2(path, backup_path)
        logger.info(f"  {C_CYAN}📋 Backed up to {backup_path}{C_RESET}")
        return backup_path
    except Exception as e:
        logger.warning(f"  {C_YELLOW}⚠ Could not create backup for {path}: {e}{C_RESET}")
        return None


def _validate_remediated_code(original, remediated):
    """Basic sanity checks before accepting remediated code."""
    if not remediated or not remediated.strip():
        return False, "Empty code block"
    if len(remediated) < len(original) * 0.3:
        return False, (f"Remediated code is suspiciously short ({len(remediated)} vs {len(original)} original chars)")
    return True, "OK"


def _validate_keys():
    """Fail fast if no API keys are configured."""
    if not GEMINI_API_KEY and not NVIDIA_API_KEY:
        logger.error(f"{C_MAGENTA}❌ No API keys found. Set GEMINI_API_KEY and/or NVIDIA_API_KEY in .env{C_RESET}")
        raise SystemExit(1)
    available = []
    if GEMINI_API_KEY:
        available.append("Gemini")
    if NVIDIA_API_KEY:
        available.append("NIM")
    logger.info(f"{C_CYAN}🔑 Available providers: {', '.join(available)}{C_RESET}")


# ANSI formatting colors
C_GREEN = "\033[92m"
C_BLUE = "\033[94m"
C_YELLOW = "\033[93m"
C_MAGENTA = "\033[95m"
C_CYAN = "\033[96m"
C_RESET = "\033[0m"
C_BOLD = "\033[1m"


def get_auditor_prompts():
    return {
        "Gilfoyle": (
            "You are Gilfoyle. Provide a brutal, uncompromising, and highly technical critique of the"
            " architecture and code quality. "
            "Assume you are seeing this project for the very first time. You have zero prior context."
        ),
        "AI Engineer": (
            "You are the AI Engineer auditor. Focus on AI operations, telemetry, latency, and"
            " deployment pipelines. "
            "Assume you are seeing this project for the very first time. You have zero prior context."
        ),
        "LLM Engineer": (
            "You are the LLM Engineer auditor. Focus strictly on training optimization, dataset"
            " curation, precision, and VRAM efficiency. "
            "Assume you are seeing this project for the very first time. You have zero prior context."
        ),
        "Mental Health Expert": (
            "You are the Mental Health Expert. Analyze the approach for clinical safety, bias,"
            " hallucination risks, and empathy alignment. "
            "Assume you are seeing this project for the very first time. You have zero prior context."
        ),
    }


class _NIMConfig:
    """Bundles connection details for an OpenAI-compatible endpoint."""

    def __init__(self, label, base_url, api_key, model, timeout=30):
        self.label = label
        self.base_url = base_url
        self.api_key = api_key
        self.model = model
        self.timeout = timeout


def _openai_compat_call(cfg: _NIMConfig, system_prompt, user_content):
    """Generic OpenAI-compatible chat completions call with retry."""
    headers = {"Authorization": f"Bearer {cfg.api_key}", "Content-Type": "application/json"}
    payload = {
        "model": cfg.model,
        "messages": [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_content}],
        "temperature": 0.2,
    }

    def _do_call():
        response = requests.post(f"{cfg.base_url}/chat/completions", headers=headers, json=payload, timeout=cfg.timeout)
        if response.status_code == HTTP_OK:
            return response.json()["choices"][0]["message"]["content"]
        if response.status_code in (429, 502, 503):
            raise requests.exceptions.ConnectionError(f"Transient {response.status_code}")
        raise RuntimeError(f"{cfg.label} failed ({response.status_code}): {response.text[:300]}")

    return _retry_request(_do_call)


def query_llm(system_prompt, user_content):
    """
    Provider chain (fastest/most-reliable first):
    1. Gemini -- gemini-2.5-flash (stable GA, confirmed reachable)
    2. Gemini -- gemini-3-flash-preview (current gen, free tier, Apr 2026)
    3. NIM -- deepseek-ai/deepseek-r1 (671B, fallback)
    4. NIM -- deepseek-ai/deepseek-v3 (685B MoE, fallback)
    5. NIM -- moonshotai/kimi-k2-instruct (1T MoE, fallback)
    6. NIM -- meta/llama-3.3-70b-instruct (fallback)
    """

    # 1-2. Gemini (working from this host)
    if GEMINI_API_KEY:
        gemini_models = [
            ("gemini-2.5-flash", "Gemini 2.5 Flash"),
            ("gemini-3-flash-preview", "Gemini 3 Flash Preview"),
        ]
        for model_id, label in gemini_models:
            try:
                logger.info(f"  {C_YELLOW}⟳ Trying {label}...{C_RESET}")

                def _call_gemini(mid=model_id, lbl=label):
                    url = f"https://generativelanguage.googleapis.com/v1beta/models/{mid}:generateContent?key={GEMINI_API_KEY}"
                    payload = {
                        "systemInstruction": {"parts": [{"text": system_prompt}]},
                        "contents": [{"role": "user", "parts": [{"text": user_content}]}],
                        "generationConfig": {"temperature": 0.2},
                    }
                    response = requests.post(
                        url, headers={"Content-Type": "application/json"}, json=payload, timeout=180
                    )
                    if response.status_code == HTTP_OK:
                        return response.json()["candidates"][0]["content"]["parts"][0]["text"]
                    if response.status_code in (429, 502, 503):
                        raise requests.exceptions.ConnectionError(f"Transient {response.status_code}")
                    raise RuntimeError(f"{lbl} failed ({response.status_code}): {response.text[:300]}")

                return _retry_request(_call_gemini)
            except Exception as e:
                logger.warning(f"  {C_YELLOW}⚠ {label} exhausted retries: {e}{C_RESET}")

    # 3-6. NVIDIA NIM (fallback - may be unreachable depending on network)
    if NVIDIA_API_KEY:
        nim_models = [
            ("deepseek-ai/deepseek-r1", "DeepSeek R1 (671B)"),
            ("deepseek-ai/deepseek-v3", "DeepSeek V3 (685B MoE)"),
            ("moonshotai/kimi-k2-instruct", "Kimi K2 Instruct (1T MoE)"),
            ("meta/llama-3.3-70b-instruct", "Llama 3.3 70B Instruct"),
        ]
        for model_id, label in nim_models:
            try:
                logger.info(f"  {C_YELLOW}⟳ Trying NIM ({label})...{C_RESET}")
                return _openai_compat_call(
                    _NIMConfig(
                        f"NIM/{label}",
                        "https://integrate.api.nvidia.com/v1",
                        NVIDIA_API_KEY,
                        model_id,
                        timeout=120,
                    ),
                    system_prompt,
                    user_content,
                )
            except Exception as e:
                logger.warning(f"  {C_YELLOW}⚠ NIM/{label} exhausted retries: {e}{C_RESET}")

    logger.error(f"{C_MAGENTA}❌ ALL API PROVIDERS FAILED.{C_RESET}")
    return ""


def extract_code(text):
    # Match the first python/json block found
    match = re.search(r"```(?:python|json)?(.*?)```", text, flags=re.DOTALL)
    if match:
        return match.group(1).strip()
    return None


def _apply_orchestrator_remediation(target_path: str, content: str, findings: dict) -> None:
    """Run the orchestrator LLM call and write remediated code to disk."""
    logger.info(f"\n{C_CYAN}Quad-Audit for {target_path} completed.{C_RESET}")
    logger.info("=" * 60)
    logger.info(f"{C_BOLD}[Orchestrator Final Summary & Remediation]{C_RESET}")

    orchestrator_prompt = (
        "You are the Lead Orchestrator. Synthesize the findings from all four auditors into a"
        " cohesive report. "
        "Then, you MUST output the complete, fully remediated source code for the file that fixes"
        " all raised issues. "
        "You must output the ENTIRE updated file enclosed entirely within a single ```python"
        " (or ```json for notebooks) code block. "
        "Do NOT use framework monkey-patching and ensure you trace mathematical edge cases."
    )

    user_content = (
        f"Target File: {target_path}\n\nOriginal Code:\n```\n{_truncate(content)[0]}\n```\n\nAuditor Findings:\n"
    )
    for auditor, finding in findings.items():
        user_content += f"\n--- {auditor} ---\n{finding}\n"

    user_content, orch_truncated = _truncate(user_content)
    if orch_truncated:
        logger.info(f"  {C_YELLOW}⚠ Orchestrator payload truncated to {MAX_CONTENT_CHARS} chars{C_RESET}")

    logger.info(f"  {C_YELLOW}⟳ Awaiting Orchestrator Remediation from Providers...{C_RESET}")
    orchestrator_response = query_llm(orchestrator_prompt, user_content)

    if not orchestrator_response:
        logger.warning(f"  {C_YELLOW}⚠ No orchestrator response for {target_path}.{C_RESET}")
        return

    logger.info(f"  {C_GREEN}✓ Orchestrator Output Length: {len(orchestrator_response)} characters{C_RESET}")

    remediated_code = extract_code(orchestrator_response)
    if not remediated_code:
        logger.warning(
            f"  {C_YELLOW}⚠ Failed to extract valid code block from Orchestrator response for {target_path}{C_RESET}"
        )
        return

    valid, reason = _validate_remediated_code(content, remediated_code)
    if not valid:
        logger.warning(f"  {C_YELLOW}⚠ Rejected remediated code for {target_path}: {reason}{C_RESET}")
        return

    backup_path = _backup_file(target_path)
    logger.info(f"  {C_GREEN}{C_BOLD}✓ Applying Remediation to {target_path}{C_RESET}")
    try:
        with open(target_path, "w") as f:
            f.write(remediated_code)
    except Exception as e:
        logger.error(f"  {C_MAGENTA}❌ Failed to write {target_path}: {e}{C_RESET}")
        if backup_path and os.path.exists(backup_path):
            shutil.copy2(backup_path, target_path)
            logger.info(f"  {C_CYAN}📋 Restored from backup{C_RESET}")

    logger.info("=" * 60 + "\n")


def run_audit_for_file(target_path, prompts):
    if not os.path.exists(target_path):
        logger.info(f"{C_YELLOW}File not found: {target_path}. Skipping.{C_RESET}")
        return

    with open(target_path) as f:
        content = f.read()

    logger.info(f"\n{C_CYAN}► Initiating Zero-Trust Quad-Audit on {target_path}...{C_RESET}")
    logger.info("=" * 60)

    audit_content, was_truncated = _truncate(content)
    if was_truncated:
        logger.info(f"  {C_YELLOW}⚠ File truncated to {MAX_CONTENT_CHARS} chars for auditor calls{C_RESET}")

    findings = {}

    for auditor, prompt in prompts.items():
        logger.info(f"{C_BLUE}[{auditor} Audit Initiated]{C_RESET}")
        logger.info(f"  System Prompt: {prompt}")

        response = query_llm(prompt, audit_content)
        if not response:
            logger.warning(f"  {C_YELLOW}⚠ No response from {auditor}, skipping.{C_RESET}")
            continue
        findings[auditor] = response

        logger.info(f"  {C_GREEN}✓ Received Response{C_RESET}")
        logger.info("-" * 60)

    if not findings:
        logger.warning(f"{C_YELLOW}⚠ No auditor responses for {target_path}, skipping orchestrator.{C_RESET}")
        return

    _apply_orchestrator_remediation(target_path, content, findings)


def sync_files(file1, file2):
    try:
        with open(file1) as f1, open(file2) as f2:
            content1, content2 = f1.read(), f2.read()
    except Exception as e:
        logger.error(f"{C_YELLOW}⚠ Could not read files for sync: {e}{C_RESET}")
        return

    # Sync File 1 using File 2
    logger.info(f"  {C_BLUE}[Syncing {os.path.basename(file1)}]{C_RESET}")
    logger.info(f"  {C_YELLOW}⟳ Awaiting Sync Orchestrator from Providers...{C_RESET}")
    sync_prompt_1 = (
        f"You are the Sync Orchestrator. We have two versions of the same ML pipeline"
        f" (e.g. Python script vs Notebook). "
        f"Ensure that all core ML logic, security fixes, and improvements currently present"
        f" in '{os.path.basename(file2)}' "
        f"are ported over to '{os.path.basename(file1)}'."
        f" Keep the environment-specific aspects (like Colab vs Docker)"
        f" of '{os.path.basename(file1)}' intact. "
        f"You MUST output the ENTIRE updated content for '{os.path.basename(file1)}'"
        f" enclosed entirely within a single ```python (or ```json) code block."
    )
    user_content_1 = (
        f"File 1 ({os.path.basename(file1)}) Content:\n```\n{_truncate(content1)[0]}\n```\n\n"
        f"File 2 ({os.path.basename(file2)}) Content:\n```\n{_truncate(content2)[0]}\n```"
    )
    sync_response_1 = query_llm(sync_prompt_1, user_content_1)
    synced_code_1 = extract_code(sync_response_1)

    if synced_code_1:
        valid, reason = _validate_remediated_code(content1, synced_code_1)
        if valid:
            _backup_file(file1)
            logger.info(f"  {C_GREEN}{C_BOLD}✓ Applying Sync Remediation to {file1}{C_RESET}")
            try:
                with open(file1, "w") as f:
                    f.write(synced_code_1)
            except Exception as e:
                logger.error(f"  {C_MAGENTA}❌ Failed to write {file1}: {e}{C_RESET}")
        else:
            logger.warning(f"  {C_YELLOW}⚠ Rejected sync for {file1}: {reason}{C_RESET}")
    else:
        logger.warning(f"  {C_YELLOW}⚠ Failed to extract valid code block for {file1} sync{C_RESET}")

    # Re-read updated File 1
    with open(file1) as f:
        updated_content1 = f.read()

    # Sync File 2 using File 1
    logger.info(f"  {C_BLUE}[Syncing {os.path.basename(file2)}]{C_RESET}")
    logger.info(f"  {C_YELLOW}⟳ Awaiting Sync Orchestrator from Providers...{C_RESET}")
    sync_prompt_2 = (
        f"You are the Sync Orchestrator. We have two versions of the same ML pipeline. "
        f"Ensure that all core ML logic, security fixes, and improvements currently present"
        f" in '{os.path.basename(file1)}' "
        f"are ported over to '{os.path.basename(file2)}'."
        f" Keep the environment-specific aspects of '{os.path.basename(file2)}' intact. "
        f"You MUST output the ENTIRE updated content for '{os.path.basename(file2)}'"
        f" enclosed entirely within a single ```python (or ```json) code block."
    )
    user_content_2 = (
        f"File 1 ({os.path.basename(file1)}) Content:\n```\n{_truncate(updated_content1)[0]}\n```\n\n"
        f"File 2 ({os.path.basename(file2)}) Content:\n```\n{_truncate(content2)[0]}\n```"
    )
    sync_response_2 = query_llm(sync_prompt_2, user_content_2)
    synced_code_2 = extract_code(sync_response_2)

    if synced_code_2:
        valid, reason = _validate_remediated_code(content2, synced_code_2)
        if valid:
            _backup_file(file2)
            logger.info(f"  {C_GREEN}{C_BOLD}✓ Applying Sync Remediation to {file2}{C_RESET}")
            try:
                with open(file2, "w") as f:
                    f.write(synced_code_2)
            except Exception as e:
                logger.error(f"  {C_MAGENTA}❌ Failed to write {file2}: {e}{C_RESET}")
        else:
            logger.warning(f"  {C_YELLOW}⚠ Rejected sync for {file2}: {reason}{C_RESET}")
    else:
        logger.warning(f"  {C_YELLOW}⚠ Failed to extract valid code block for {file2} sync{C_RESET}")


def main():
    parser = argparse.ArgumentParser(description="Zero-Trust Quad-Audit workflow")
    parser.add_argument("target_paths", nargs="+", help="Paths to files to audit")
    parser.add_argument("--rounds", type=int, default=1, help="Number of times to run the audit loop")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run auditors and orchestrator but skip writing changes to disk",
    )
    args = parser.parse_args()

    _validate_keys()

    target_paths = args.target_paths
    total_rounds = args.rounds
    dry_run = args.dry_run

    if dry_run:
        logger.info(f"{C_YELLOW}🌵 DRY RUN — no files will be modified{C_RESET}")

    for current_round in range(1, total_rounds + 1):
        if total_rounds > 1:
            logger.info(f"\n{C_MAGENTA}{C_BOLD}============================================================")
            logger.info(f"🚀 INITIATING LOOP ROUND {current_round} OF {total_rounds}")
            logger.info(f"============================================================{C_RESET}")

        prompts = get_auditor_prompts()

        for target_path in target_paths:
            if dry_run:
                logger.info(f"\n{C_CYAN}► DRY RUN: Would audit {target_path}{C_RESET}")
                continue
            run_audit_for_file(target_path, prompts)

        # --- CROSS-FILE ALIGNMENT AND SYNC ---
        if len(target_paths) >= MIN_SYNC_FILES and not dry_run:
            logger.info(f"{C_MAGENTA}{C_BOLD}============================================================")
            logger.info(f"🔄 INITIATING ROUND {current_round} CROSS-FILE ALIGNMENT & SYNC")
            logger.info(f"============================================================{C_RESET}")
            sync_files(target_paths[0], target_paths[1])

        if total_rounds > 1:
            logger.info(f"{C_GREEN}{C_BOLD}✅ ALL {total_rounds} ROUNDS COMPLETED.{C_RESET}\n")


if __name__ == "__main__":
    main()
