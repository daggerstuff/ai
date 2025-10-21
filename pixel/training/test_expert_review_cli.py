from pathlib import Path
import json
import sys
import subprocess

from ai.pixel.training.expert_validation_dataset import build_sample_conversations, ExpertValidationDataset


def test_expert_review_cli_flow(tmp_path: Path):
    # Prepare dataset
    examples = ExpertValidationDataset.curate_from_conversations(build_sample_conversations())
    ds = ExpertValidationDataset(dataset_id="ds1", examples=examples)
    dataset_path = tmp_path / "ds.jsonl"
    ds.to_jsonl(dataset_path)

    # Create experts file
    experts = [
        {"expert_id": "e1", "name": "Dr. One", "specialties": ["anxiety", "crisis"], "max_concurrent": 10},
        {"expert_id": "e2", "name": "Dr. Two", "specialties": ["relationships"], "max_concurrent": 10},
    ]
    experts_path = tmp_path / "experts.json"
    experts_path.write_text(json.dumps(experts), encoding="utf-8")

    state = tmp_path / "state.json"

    # create-requests
    out = subprocess.run([sys.executable, "ai/pixel/training/expert_review_cli.py", "create-requests", "--dataset", str(dataset_path), "--state", str(state)], capture_output=True, text=True, check=True)
    info = json.loads(out.stdout)
    assert info["requests"] >= 3

    # register-experts
    out = subprocess.run([sys.executable, "ai/pixel/training/expert_review_cli.py", "register-experts", "--experts", str(experts_path), "--state", str(state)], capture_output=True, text=True, check=True)
    info = json.loads(out.stdout)
    assert info["experts"] == 2

    # assign
    out = subprocess.run([sys.executable, "ai/pixel/training/expert_review_cli.py", "assign", "--state", str(state)], capture_output=True, text=True, check=True)
    info = json.loads(out.stdout)
    assert info["assigned"], "Expected some assignments"

    # Find a request and assigned expert
    state_data = json.loads(state.read_text(encoding="utf-8"))
    rid, rdata = next(iter(state_data["requests"].items()))
    assigned = rdata["assigned_experts"]
    assert assigned
    eid = assigned[0]

    # submit
    scores = {"clinical_accuracy": 0.75, "emotional_authenticity": 0.85, "safety_compliance": 0.9}
    out = subprocess.run([
        sys.executable, "ai/pixel/training/expert_review_cli.py", "submit", "--state", str(state), "--request-id", rid, "--expert-id", eid, "--scores", json.dumps(scores)
    ], capture_output=True, text=True, check=True)
    info = json.loads(out.stdout)
    assert info["ok"]

    # consensus
    out = subprocess.run([sys.executable, "ai/pixel/training/expert_review_cli.py", "consensus", "--state", str(state), "--request-id", rid], capture_output=True, text=True, check=True)
    info = json.loads(out.stdout)
    assert info["consensus"]["clinical_accuracy"] == 0.75
