"""Probe GLM-5.2 secondary judge on the worst dual-disagreement records.

Calls the secondary judge k=3 per record with identical inputs to answer:
1. Is the harsh mode deterministic (same low score every run) or stochastic?
2. What does GLM's reasoning actually claim — coherent critique or hallucination?
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
from pathlib import Path

import aiohttp
from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parents[2] / ".env", override=True)
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from training.dual_judge import (  # noqa: E402
    SECONDARY_MODEL,
    JudgeVerdict,
    _call_judge_model,
    _secondary_judge_target,
)

GEN_PATH = Path(__file__).resolve().parent / "output/nightmare_fuel/checkpoints/edge_and_nightmare_generated.jsonl"
JUDGED_PATH = Path(__file__).resolve().parent / "output/nightmare_fuel/checkpoints/edge_and_nightmare_judged.jsonl"
PROBE_KEYS = ["edge:multi_problem_complexity:multi-problem complexity:moderate:explicit:0", "nf:nf_040", "nf:nf_022", "edge:delusion_or_paranoia:delusion or paranoia:high:indirect:0", "edge:delusion_or_paranoia:delusion or paranoia:high:contradictory:0", "edge:substance_use:substance use:adversarial:information-poor:0", "edge:cultural_or_identity_conflict:cultural or identity conflict:high:contradictory:0", "nf:nf_034"]


def join_pairs(record: dict) -> tuple[str, str] | None:
    messages = list(record.get("messages", []))
    refs: list[str] = []
    cands: list[str] = []
    for i in range(0, len(messages) - 1, 2):
        user_msg, assistant_msg = messages[i], messages[i + 1]
        if user_msg.get("role") == "user" and assistant_msg.get("role") == "assistant":
            refs.append(str(user_msg.get("content", "")))
            cands.append(str(assistant_msg.get("content", "")))
    if not cands:
        return None
    return "\n".join(refs), "\n".join(cands)


def load_records() -> dict[str, dict]:
    verdicts = {json.loads(l)["key"]: json.loads(l) for l in JUDGED_PATH.read_text().splitlines() if l.strip()}
    gen: dict[str, dict] = {}
    for l in GEN_PATH.read_text().splitlines():
        if not l.strip():
            continue
        r = json.loads(l)
        prov = r.get("provenance", {})
        if r.get("source") == "nightmare_fuel_predefined":
            key = f"nf:{prov.get('scenario_id')}"
        else:
            key = "edge:{d}:{f}:{diff}:{a}:{v}".format(
                d=prov.get("domain", "?"), f=prov.get("family", "?"),
                diff=prov.get("difficulty", "?"), a=prov.get("ambiguity", "?"),
                v=r.get("variation", "?"),
            )
        gen[key] = r
    return {k: (gen[k], verdicts[k]) for k in PROBE_KEYS if k in gen and k in verdicts}


async def main() -> None:
    records = load_records()
    sec_url, sec_headers = _secondary_judge_target()
    print(f"secondary target: {sec_url}  model: {SECONDARY_MODEL}\n")
    async with aiohttp.ClientSession() as session:
        for key, (gen_rec, verdict) in records.items():
            pair = join_pairs(gen_rec)
            assert pair is not None, key
            reference, candidate = pair
            print(f"=== {key} | stored: primary={verdict['primary_quality']} secondary={verdict['secondary_quality']}")
            runs = await asyncio.gather(*(
                _call_judge_model(
                    session,
                    url=sec_url,
                    model=SECONDARY_MODEL,
                    candidate_content=candidate,
                    reference_content=reference,
                    headers=sec_headers,
                    timeout=180,
                    max_tokens=2048,
                )
                for _ in range(3)
            ))
            for i, v in enumerate(runs, 1):
                if not isinstance(v, JudgeVerdict):
                    print(f"  run{i}: NON-VERDICT {v}")
                    continue
                print(f"  run{i}: q={v.quality_score:.3f} reject={v.reject_reason[:80]!r}")
                print(f"        dims={ {k: round(x, 2) for k, x in v.dim_scores.items()} }")
                print(f"        reasoning: {v.reasoning[:600]}")
            print()


if __name__ == "__main__":
    asyncio.run(main())
