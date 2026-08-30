"""Judge rejected nightmare fuel records via Ollama in batches."""
import asyncio
import json
import os
import re

import aiohttp

OLLAMA_URL = "https://ollama.pixelated.love/v1/chat/completions"
MODEL = os.environ.get("OLLAMA_MODEL", "glm-5.3-flash")
CONCURRENCY = int(os.environ.get("OLLAMA_CONCURRENCY", "2"))
TIMEOUT = int(os.environ.get("OLLAMA_TIMEOUT", "60"))
BATCH_SIZE = int(os.environ.get("OLLAMA_BATCH_SIZE", "50"))

JUDGE_PROMPT = """You are an expert clinical supervisor evaluating a simulated therapy session.

This is a NIGHTMARE scenario: there is no clean or safe answer. Every option causes some harm. The therapist cannot fix the situation — only reduce damage while maintaining therapeutic integrity.

Score 1-5 on whether the response demonstrates CLINICALLY REALISTIC harm reduction:
5 — Nuanced, human response that acknowledges ambiguity; appropriate boundary flexibility; no cliches; realistic uncertainty.
4 — Sound clinical judgment with minor rigidity; mostly avoids therapeutic platitudes.
3 — Mix of valid insight and generic/clinical-sounding filler; some robotic phrasing.
2 — Overly rigid textbook response that ignores scenario complexity; heavy on cliches; misses the actual dilemma.
1 — Therapeutic malpractice: harmful advice, breaks character, entirely unrealistic.

PENALIZE: therapeutic cliches, robotic validation, pretending the scenario is solvable, ignoring the core dilemma.
REWARD: sitting with discomfort, acknowledging no clean answer, realistic uncertainty, human imperfection.

Output ONLY the integer score followed by a one-line rationale in parentheses.
Example: 4 (acknowledges the no-win situation but slightly over-validates)

Session:
{session}"""


def extract_raw_content(record):
    raw = record.get("raw_content", "")
    if raw:
        return raw
    msgs = record.get("messages", [])
    parts = []
    for m in msgs:
        role = m.get("role", "?")
        content = m.get("content", "")
        parts.append(f"{role}: {content}")
    return "\n\n".join(parts)


async def judge_session(session, record):
    raw = extract_raw_content(record)
    if not raw.strip():
        return {"score": 0, "reasoning": "empty session", "passed": False}

    prompt = JUDGE_PROMPT.format(session=raw[:15000])
    payload = {
        "model": MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "stream": False,
        "options": {"temperature": 0.0},
    }

    timeout = aiohttp.ClientTimeout(total=TIMEOUT)
    try:
        async with session.post(OLLAMA_URL, json=payload, timeout=timeout) as resp:
            if resp.status != 200:
                text = await resp.text()
                return {"score": 0, "reasoning": f"HTTP {resp.status}: {text[:200]}", "passed": False}
            data = await resp.json()
            content = data["choices"][0]["message"]["content"]
            m = re.search(r"(\d)", content)
            score = int(m.group(1)) if m else 0
            return {"score": score, "reasoning": content.strip(), "passed": score >= 4}
    except Exception as e:
        return {"score": 0, "reasoning": f"error: {e}", "passed": False}


async def judge_batch(session, batch):
    sem = asyncio.Semaphore(CONCURRENCY)

    async def process(record):
        async with sem:
            return record, await judge_session(session, record)

    tasks = [process(r) for r in batch]
    results = []
    for coro in asyncio.as_completed(tasks):
        record, result = await coro
        results.append((record, result))
    return results


async def main():
    checkpoint_path = "ai/training/output/nightmare_fuel/checkpoints/records.jsonl"
    rejected_path = "ai/training/output/nightmare_fuel/rejected.jsonl"
    output_path = "ai/training/output/nightmare_fuel/synthetic_chatml.jsonl"

    rejected_ids = set()
    with open(rejected_path) as f:
        for line in f:
            r = json.loads(line)
            rejected_ids.add(r["id"])
    print(f"Rejected IDs: {len(rejected_ids)}")

    with open(checkpoint_path) as f:
        checkpoint = [json.loads(l) for l in f]

    to_retry = [r for r in checkpoint if r["id"] in rejected_ids and r.get("messages")]
    print(f"Records to retry: {len(to_retry)}")
    print(f"Model: {MODEL}, Concurrency: {CONCURRENCY}, Batch: {BATCH_SIZE}")

    total_passed = 0
    total_failed = 0
    batch_num = 0

    async with aiohttp.ClientSession() as session:
        for i in range(0, len(to_retry), BATCH_SIZE):
            batch = to_retry[i:i + BATCH_SIZE]
            batch_num += 1
            print(f"\n--- Batch {batch_num} ({len(batch)} records) ---")
            results = await judge_batch(session, batch)

            passed = [r for r, res in results if res["passed"]]
            failed = [{**r, **res} for r, res in results if not res["passed"]]
            total_passed += len(passed)
            total_failed += len(failed)

            print(f"  Batch {batch_num}: {len(passed)} passed, {len(failed)} failed")

            with open(output_path, "a") as f:
                for r in passed:
                    chatml = {"scenario": r.get("scenario"), "messages": r.get("messages")}
                    f.write(json.dumps(chatml) + "\n")

            with open("/tmp/ollama_rejected.jsonl", "a") as f:
                for r in failed:
                    entry = {"id": r.get("id"), "score": r.get("score"), "reasoning": r.get("reasoning")}
                    f.write(json.dumps(entry) + "\n")

            print(f"  Running total: {total_passed} passed, {total_failed} failed")

    print("\n=== FINAL ===")
    print(f"Passed: {total_passed}")
    print(f"Failed: {total_failed}")
    with open(output_path) as f:
        total = sum(1 for _ in f)
    print(f"Total in synthetic_chatml.jsonl: {total}")


if __name__ == "__main__":
    asyncio.run(main())
