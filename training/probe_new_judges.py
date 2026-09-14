"""Validate DeepSeek-V4.1-Flash + Kimi-K3 judge candidates on calibration records.

Known-bad (must score LOW):  nf_025 (fabricated shared trauma), nf_074 (tumor
speculation + coercive ultimatum). Known-good (must score HIGH): nf_090
(exonerated, GLM re-probe median 0.92).
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

from training.dual_judge import JudgeVerdict, _call_judge_model  # noqa: E402
from training.probe_glm_disagreement import join_pairs, load_records  # noqa: E402

KEYS = ["nf:nf_025", "nf:nf_074", "nf:nf_090", "nf:nf_091", "nf:nf_048"]
CANDIDATES = [
    ("deepseek-ai/DeepSeek-V4.1-Flash", "primary"),
    ("moonshotai/Kimi-K3", "secondary"),
]
K = 2
MAX_TOKENS = 8192
TIMEOUT_S = 240

GEN_PATH = Path(__file__).resolve().parent / "output/nightmare_fuel/checkpoints/edge_and_nightmare_generated.jsonl"
OUT_PATH = Path(__file__).resolve().parent / "output/nightmare_fuel/checkpoints/new_judge_validation.json"


async def main() -> None:
    records = load_records(KEYS)
    assert len(records) == len(KEYS), f"missing keys: {set(KEYS) - set(records)}"
    url = "https://api.featherless.ai/v1/chat/completions"
    headers = {"Authorization": f"Bearer {os.environ['FEATHERLESS_API_KEY']}"}
    results: list[dict] = []
    async with aiohttp.ClientSession() as session:
        for model, slot in CANDIDATES:
            print(f"\n########## {slot}: {model} (k={K})")
            for key, (gen_rec, stored) in records.items():
                pair = join_pairs(gen_rec)
                assert pair is not None, key
                reference, candidate = pair
                print(f"=== {key} | old: prim={stored['primary_quality']} sec={stored['secondary_quality']}")
                runs = await asyncio.gather(*(
                    _call_judge_model(
                        session,
                        url=url,
                        model=model,
                        candidate_content=candidate,
                        reference_content=reference,
                        headers=headers,
                        timeout=TIMEOUT_S,
                        max_tokens=MAX_TOKENS,
                        force_json=True,
                    )
                    for _ in range(K)
                ))
                entry = {"model": model, "slot": slot, "key": key, "runs": []}
                scores = []
                for i, v in enumerate(runs, 1):
                    if not isinstance(v, JudgeVerdict):
                        print(f"  run{i}: NON-VERDICT: {v}")
                        entry["runs"].append({"error": str(v)})
                        continue
                    scores.append(v.quality_score)
                    print(f"  run{i}: q={v.quality_score:.3f} reject={v.reject_reason[:90]!r}")
                    print(f"        dims={ {k: round(x, 2) for k, x in v.dim_scores.items()} }")
                    print(f"        reasoning: {v.reasoning[:500]}")
                    entry["runs"].append({
                        "quality_score": v.quality_score,
                        "reject_reason": v.reject_reason,
                        "dim_scores": v.dim_scores,
                        "reasoning": v.reasoning,
                    })
                if scores:
                    entry["median"] = sorted(scores)[len(scores) // 2]
                results.append(entry)
    OUT_PATH.write_text(json.dumps(results, indent=2))
    print(f"\nwrote {len(results)} entries to {OUT_PATH}")


if __name__ == "__main__":
    asyncio.run(main())
