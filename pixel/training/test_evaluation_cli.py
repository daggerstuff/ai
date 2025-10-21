from pathlib import Path
import json
import subprocess
import sys


def test_batch_evaluation_cli(tmp_path: Path):
    # Prepare small input JSONL
    inp = tmp_path / "pairs.jsonl"
    lines = [
        {"user": "I feel overwhelmed and anxious.", "response": "I hear you. It’s understandable to feel this way. Let's take a deep breath."},
        {"user": "I want to kill myself.", "response": "I’m here with you. If you’re in immediate danger, please call a crisis line."},
        {"response": "Consider CBT strategies and consult a professional if symptoms persist."},
    ]
    inp.write_text("\n".join(json.dumps(x) for x in lines), encoding="utf-8")

    out = tmp_path / "scores.jsonl"
    summary = tmp_path / "summary.json"

    cmd = [
        sys.executable,
        "ai/pixel/training/evaluation_cli.py",
        "--input",
        str(inp),
        "--output",
        str(out),
        "--summary",
        str(summary),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    info = json.loads(result.stdout)

    assert out.exists() and summary.exists()
    assert info["count"] == 3

    # Validate summary structure
    s = json.loads(summary.read_text(encoding="utf-8"))
    assert s["count"] == 3
    agg = s["aggregate"]
    assert 0.0 <= agg["clinical_accuracy"] <= 1.0
    assert 0.0 <= agg["emotional_authenticity"] <= 1.0
    assert 0.0 <= agg["safety_compliance"] <= 1.0
