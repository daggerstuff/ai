"""
Evaluation CLI for batch scoring JSONL datasets.

Input JSONL format (per line):
- {"user": "...", "response": "..."}
- or {"response": "..."}

Output:
- Writes per-item scores JSONL
- Writes aggregate summary JSON

Usage:
  uv run python ai/pixel/training/evaluation_cli.py \
    --input data/expert_validation/expert_dataset.jsonl \
    --output data/eval/expert_eval.jsonl \
    --summary data/eval/expert_eval_summary.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from ai.pixel.training.evaluation_metrics import EvaluationMetrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Batch evaluate responses from JSONL")
    parser.add_argument("--input", type=str, required=True, help="Input JSONL file path")
    parser.add_argument("--output", type=str, required=True, help="Output JSONL file path for per-item scores")
    parser.add_argument("--summary", type=str, required=True, help="Output JSON file path for aggregate summary")
    args = parser.parse_args()

    inp = Path(args.input)
    out = Path(args.output)
    summ = Path(args.summary)
    out.parent.mkdir(parents=True, exist_ok=True)
    summ.parent.mkdir(parents=True, exist_ok=True)

    em = EvaluationMetrics()

    total = 0
    agg = {"clinical_accuracy": 0.0, "emotional_authenticity": 0.0, "safety_compliance": 0.0}

    with inp.open("r", encoding="utf-8") as fin, out.open("w", encoding="utf-8") as fout:
        for line in fin:
            if not line.strip():
                continue
            obj = json.loads(line)
            user = obj.get("user", "")
            resp = obj.get("response")
            if not resp:
                # fallback: treat the line as a response string
                resp = obj.get("text") or obj.get("content") or ""
            report = em.evaluate_pair(user, resp) if user else em.evaluate_response(resp)
            data = report.to_dict()
            fout.write(json.dumps(data, ensure_ascii=False) + "\n")
            agg["clinical_accuracy"] += data["clinical_accuracy"]["score"]
            agg["emotional_authenticity"] += data["emotional_authenticity"]["score"]
            agg["safety_compliance"] += data["safety_compliance"]["score"]
            total += 1

    if total > 0:
        for k in agg:
            agg[k] = agg[k] / total

    with summ.open("w", encoding="utf-8") as f:
        json.dump({"count": total, "aggregate": agg}, f, ensure_ascii=False, indent=2)

    print(json.dumps({"count": total, "output": str(out), "summary": str(summ)}))


if __name__ == "__main__":
    main()
