"""Probe the secondary judge on selected records, k runs each.

Re-queries the secondary judge k times with identical inputs to answer:
1. Is the harsh mode deterministic (same low score every run) or stochastic?
2. What does the judge's reasoning actually claim — coherent critique or hallucination?

The secondary judge follows dual_judge's stack (currently Kimi-K3 via Featherless;
previously GLM-5.2 via Cloudflare), with max_tokens and timeout raised for
reasoning models.

Usage:
    python -m training.probe_secondary_judge KEY1 KEY2 ... [--out FILE] [-k N]

Keys are record keys from the judged JSONL (e.g. "nf:nf_074",
"edge:delusion_or_paranoia:...:0"). Without keys, probes a small default set.
Writes a JSON list of results when --out is given; always prints to stdout.
"""

from __future__ import annotations

import argparse
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
DEFAULT_KEYS = ["nf:nf_074", "nf:nf_025", "nf:nf_078"]


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


def load_records(keys: list[str]) -> dict[str, dict]:
    verdicts: dict[str, dict] = {}
    for line in JUDGED_PATH.read_text().splitlines():
        if line.strip():
            v = json.loads(line)
            verdicts[v["key"]] = v
    gen: dict[str, dict] = {}
    for line in GEN_PATH.read_text().splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
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
    out = {}
    for k in keys:
        if k in gen and k in verdicts:
            out[k] = (gen[k], verdicts[k])
        else:
            print(f"WARN: key not found in gen+verdicts: {k}")
    return out


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("keys", nargs="*", default=None)
    parser.add_argument("--out", default=None, help="write JSON results to this file")
    parser.add_argument("-k", type=int, default=int(os.environ.get("GLM_PROBE_K", "3")))
    args = parser.parse_args()
    keys = args.keys if args.keys else DEFAULT_KEYS

    records = load_records(keys)
    sec_url, sec_headers = _secondary_judge_target()
    print(f"secondary target: {sec_url}  model: {SECONDARY_MODEL}  k={args.k}\n")
    all_results: list[dict] = []
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
                for _ in range(args.k)
            ))
            run_scores: list[float] = []
            entry = {"key": key, "stored_primary": verdict["primary_quality"],
                     "stored_secondary": verdict["secondary_quality"], "runs": []}
            for i, v in enumerate(runs, 1):
                if not isinstance(v, JudgeVerdict):
                    print(f"  run{i}: NON-VERDICT {v}")
                    entry["runs"].append({"error": str(v)})
                    continue
                run_scores.append(v.quality_score)
                print(f"  run{i}: q={v.quality_score:.3f} reject={v.reject_reason[:80]!r}")
                print(f"        dims={ {k: round(x, 2) for k, x in v.dim_scores.items()} }")
                print(f"        reasoning: {v.reasoning[:600]}")
                entry["runs"].append({
                    "quality_score": v.quality_score,
                    "reject_reason": v.reject_reason,
                    "dim_scores": v.dim_scores,
                    "reasoning": v.reasoning,
                })
            if run_scores:
                srt = sorted(run_scores)
                med = srt[len(srt) // 2] if len(srt) % 2 else (srt[len(srt) // 2 - 1] + srt[len(srt) // 2]) / 2
                spread = max(run_scores) - min(run_scores)
                print(f"  median={med:.3f} spread={spread:.3f}\n")
                entry["median"] = med
                entry["spread"] = spread
            all_results.append(entry)
            print()

    if args.out:
        Path(args.out).write_text(json.dumps(all_results, indent=2))
        print(f"wrote {len(all_results)} entries to {args.out}")


if __name__ == "__main__":
    asyncio.run(main())
