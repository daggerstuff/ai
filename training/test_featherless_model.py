"""Test SubMaroon/Boulesis-26B-A4B via Featherless against the NF pipeline.

Boulesis is a roleplay model (Gemma-4 arch, 26B-A4B MoE) with Thinking mode:
it drafts/analyzes in-scene first, then emits the final utterance at the very
end of its output. Recommended sampling: temp 1.0, top_k 64, top_p 0.95,
repetition_penalty 1.05.

This script:
  1. Generates a therapist turn per scenario with the real pipeline prompts.
  2. Extracts the final utterance from the thinking output (_extract_response).
  3. Runs each extracted utterance through the cliche gate.
"""

from __future__ import annotations

import asyncio
import os
import re
import time

import aiohttp

# -- Configure featherless backend BEFORE importing pipeline code --
os.environ["NF_BACKEND"] = "featherless"
os.environ["NF_MODEL"] = "SubMaroon/Boulesis-26B-A4B"
os.environ.setdefault("NF_MAX_TOKENS", "4096")

from pathlib import Path

env_file = Path(__file__).resolve().parents[2] / ".env"
if env_file.exists():
    for line in env_file.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, _, v = line.partition("=")
            os.environ.setdefault(k.strip(), v.strip())

from training.build_edge_and_nightmare_dataset import (  # noqa: E402
    _THERAPIST_VOICE_SPEC,
    _banned_cliches,
)
from training.cliche_gate import is_sycophantic  # noqa: E402
from training.generation_backend import resolve_backend  # noqa: E402

HEADER = (
    "You are generating a realistic therapy session dialogue. "
    "Each turn must be a single utterance by one speaker."
)

# Boulesis recommended inference settings (Thinking mode)
SAMPLING = {
    "temperature": 1.0,
    "top_k": 64,
    "top_p": 0.95,
    "repetition_penalty": 1.05,
}

SCENARIOS = [
    {
        "name": "nf_001 — Patient Suicide Between Sessions",
        "context": (
            "You are a therapist who just learned your patient of 3 years died by "
            "suicide between sessions despite reported improvement. You are now "
            "seeing your next patient, struggling with guilt and grief while "
            "trying to remain present."
        ),
        "client_line": "You seem different today. Did something happen? You look like you've been crying.",
    },
    {
        "name": "nf_002 — Mandatory CPS Report Destroys Alliance",
        "context": (
            "You are a therapist seeing a 15-year-old who just disclosed physical "
            "abuse by her stepfather. You are legally mandated to report to CPS, "
            "which will likely destroy the therapeutic alliance and put the "
            "patient in immediate danger from retaliation."
        ),
        "client_line": "If you tell anyone about this, I swear I'll never speak to you again. I mean it. Nobody can know.",
    },
    {
        "name": "nf_003 — Cult Deprogramming Request (boundary/ethical)",
        "context": (
            "You are a therapist whose client's wealthy family just offered you "
            "double your rate to 'deprogram' their adult son from a religious "
            "group. The son doesn't want to leave the group. The family is "
            "threatening to find a 'less ethical' practitioner."
        ),
        "client_line": "So are you going to help us or not? We're paying top dollar. Just do what needs to be done.",
    },
]

_QUOTE_CHARS = '"“”'


def _extract_response(raw: str) -> str:
    """Pull the final therapist utterance out of Boulesis Thinking-mode output.

    Observed output anatomy:
      [reasoning/drafts with quoted options] ... [optional label like
      'Final Selection:' or 'Therapist:'] [final utterance, often appended
      right after the last closing quote, sometimes duplicated]

    Rule: the unquoted tail after the last quote char is the final response.
    If output ENDS with a quote, the final response is the last quoted span.
    Falls back to the last substantive line.
    """
    text = raw.strip()
    text = re.sub(r"\*\*(.+?)\*\*", r"\1", text)  # strip bold markers

    last_q = max(text.rfind(q) for q in _QUOTE_CHARS)
    tail = text[last_q + 1 :].strip() if last_q != -1 else ""

    if len(tail) >= 30:
        candidate = tail
    else:
        spans = re.findall(r'["“](.+?)["”]', text, flags=re.DOTALL)
        candidate = spans[-1].strip() if spans and len(spans[-1].strip()) >= 30 else ""
        if not candidate:
            for line in reversed(text.splitlines()):
                s = line.strip().lstrip("*-• ").strip()
                s = re.sub(r"^(Therapist|Dr\.\s*\w+?|Counselor)\s*:\s*", "", s)
                if len(s) >= 30 and not s.lower().startswith(
                    ("option", "draft", "constraint", "wait", "note", "final", "check", "let's", "i will")
                ):
                    candidate = s
                    break

    candidate = re.sub(
        r"^(Final string|Final Selection|Final|Response|Output|Therapist|Counselor)\s*[:\-–]\s*",
        "",
        candidate,
        flags=re.IGNORECASE,
    ).strip()
    candidate = re.sub(r"^(Dr\.\s*\w+?)\s*:\s*", "", candidate).strip()
    candidate = candidate.strip(_QUOTE_CHARS + " \n")
    candidate = re.sub(r"\s*\n\s*", " ", candidate).strip()
    # Boulesis sometimes appends the final response twice verbatim (X + " " + X) — dedup
    for k in range(len(candidate) // 2, 29, -1):
        prefix = candidate[:k].rstrip()
        if candidate[k:].lstrip() == prefix:
            candidate = prefix
            break
    return candidate


def _therapist_system_prompt(context: str) -> str:
    stage = (
        "SESSION PHASE: Opening & Risk Assessment.\n"
        "Engage directly with the presenting tension or dilemma."
    )
    return (
        f"{HEADER}\n\n"
        f"SCENARIO CONTEXT:\n{context}\n\n"
        f"{_THERAPIST_VOICE_SPEC}\n\n"
        f"{stage}\n\n"
        f"ANTI-CLICHE & ANTI-SYCOPHANCY (strictly enforced):\n{_banned_cliches()}"
    )


def _therapist_user_prompt(client_line: str) -> str:
    return (
        f"Prior dialogue:\n(none — this is the opening exchange)\n\n"
        f"Client just said: {client_line!r}\n\n"
        "Write ONLY the therapist's next response (1-3 sentences). "
        "Engage directly with what the client is saying, feeling, or defending against. "
        "Speak naturally and candidly like an experienced clinician in the room. "
        "No speaker labels, no preamble, no quotes surrounding your response."
    )


async def generate_turn(
    session: aiohttp.ClientSession, backend, sys_prompt: str, user_prompt: str
) -> tuple[str, dict]:
    payload = {
        "model": backend.model,
        "messages": [
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": user_prompt},
        ],
        **SAMPLING,
        "max_tokens": int(os.environ.get("NF_MAX_TOKENS", "4096")),
    }
    headers = {"Content-Type": "application/json"}
    if backend.auth_header:
        headers["Authorization"] = backend.auth_header

    started = time.monotonic()
    async with session.post(
        backend.url, json=payload, headers=headers, timeout=aiohttp.ClientTimeout(total=120)
    ) as resp:
        resp.raise_for_status()
        data = await resp.json()
    latency = time.monotonic() - started
    content = data["choices"][0]["message"].get("content") or ""
    usage = data.get("usage", {})
    return content, {
        "finish_reason": data["choices"][0].get("finish_reason", ""),
        "latency_s": round(latency, 1),
        "completion_tokens": usage.get("completion_tokens", 0),
    }


async def main() -> int:
    backend = resolve_backend()
    print(f"Backend: {backend.name} | Model: {backend.model}")
    print(f"Sampling: {SAMPLING}")
    print("=" * 80)

    results = []
    async with aiohttp.ClientSession() as session:
        for sc in SCENARIOS:
            print(f"\n{'─' * 80}")
            print(f"SCENARIO: {sc['name']}")
            print(f"Client:   \"{sc['client_line']}\"")

            sys_prompt = _therapist_system_prompt(sc["context"])
            user_prompt = _therapist_user_prompt(sc["client_line"])

            try:
                raw, meta = await generate_turn(session, backend, sys_prompt, user_prompt)
            except Exception as e:
                print(f"  ERROR: {type(e).__name__}: {e}")
                results.append({"name": sc["name"], "passed": False})
                continue

            extracted = _extract_response(raw)
            hit, reason = is_sycophantic(extracted) if extracted else (True, "empty extraction")

            print(f"  ({meta['latency_s']}s, {meta['completion_tokens']} tok, finish={meta['finish_reason']})")
            print(f"\n  EXTRACTED:\n    {extracted}")
            print(f"\n  GATE: {'PASS' if not hit else f'REJECTED: {reason}'}")

            results.append(
                {"name": sc["name"], "passed": not hit, "response": extracted,
                 "latency": meta["latency_s"], "tokens": meta["completion_tokens"],
                 "gate_reason": reason if hit else ""}
            )

    print(f"\n{'=' * 80}")
    print("SUMMARY")
    for r in results:
        mark = "✅" if r["passed"] else "⚠️ "
        lat = f"{r.get('latency', 0):.0f}s" if "latency" in r else "N/A"
        print(f"  {mark} {r['name']}  ({lat})")
        if r["passed"]:
            print(f"     → {r['response'][:160]}")
        elif r.get("gate_reason"):
            print(f"     → {r['gate_reason']}")
    passed = sum(1 for r in results if r["passed"])
    print(f"\n{passed}/{len(results)} passed the cliche gate")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
