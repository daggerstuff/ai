from pathlib import Path

from ai.pixel.training.expert_review_workflow import Expert, ExpertReviewWorkflow
from ai.pixel.training.expert_validation_dataset import (
    ExpertValidationDataset,
    build_sample_conversations,
)


def test_workflow_end_to_end(tmp_path: Path):
    # Build a small dataset file
    examples = ExpertValidationDataset.curate_from_conversations(build_sample_conversations())
    ds = ExpertValidationDataset(dataset_id="ds1", examples=examples)
    dataset_path = tmp_path / "ds.jsonl"
    ds.to_jsonl(dataset_path)

    wf = ExpertReviewWorkflow()
    # Register experts
    wf.register_expert(Expert(expert_id="e1", name="Dr. One", specialties=["anxiety", "crisis"], max_concurrent=10))
    wf.register_expert(Expert(expert_id="e2", name="Dr. Two", specialties=["relationships"], max_concurrent=10))

    # Create requests
    rids = wf.create_requests_from_dataset(str(dataset_path), min_reviews=1, max_reviews=2)
    assert len(rids) >= 3

    # Assign
    assigned = wf.assign_experts_round_robin(min_reviewers=1, max_reviewers=2)
    assert assigned

    # Submit one review for first request
    rid = rids[0]
    # pick first assigned expert
    exp_id = next(iter(wf.requests[rid].assigned_experts))
    wf.submit_review(rid, exp_id, {"clinical_accuracy": 0.8, "emotional_authenticity": 0.9, "safety_compliance": 0.95}, comments="Looks good")

    # Check consensus after one review (min_reviews=1)
    result = wf.consensus(rid)
    assert result["consensus"]["clinical_accuracy"] == 0.8
    assert "reviews" in result

    # Save/load state
    state = tmp_path / "state.json"
    wf.save_state(str(state))
    wf2 = ExpertReviewWorkflow()
    wf2.load_state(str(state))
    assert wf2.requests[rid].status == "complete"
