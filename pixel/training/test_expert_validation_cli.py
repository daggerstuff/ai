import json
import subprocess
import sys
from pathlib import Path


def run_cli(tmp_path: Path, target_count: int = 600) -> dict:
    out_file = tmp_path / "expert_dataset.jsonl"
    cmd = [
        sys.executable,
        "ai/pixel/training/expert_validation_cli.py",
        "--output",
        str(out_file),
        "--target-count",
        str(target_count),
        "--min-crisis",
        "0.08",
        "--max-crisis",
        "0.2",
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    return json.loads(result.stdout.strip())


def test_cli_generates_dataset_and_manifest(tmp_path: Path):
    info = run_cli(tmp_path, target_count=550)
    out_path = Path(info["output"])
    manifest_path = Path(info["manifest"]) 

    assert out_path.exists(), "JSONL dataset not created"
    assert manifest_path.exists(), "Training manifest not created"

    # Basic sanity: count is within requested bounds and file lines match
    lines = [l for l in out_path.read_text(encoding="utf-8").splitlines() if l.strip()]
    assert 500 <= info["count"] <= 1000
    assert info["count"] == len(lines)

    # Verify training manifest references dataset and record_count
    m = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert m["dataset"]["name"] == "pixel_expert_validation"
    assert m["dataset"]["record_count"] == info["count"]
