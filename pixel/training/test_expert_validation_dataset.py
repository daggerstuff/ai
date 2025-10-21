import json
from pathlib import Path

from ai.pixel.training.expert_validation_dataset import (
    ExpertValidationDataset,
    ExpertValidationExample,
    ScenarioType,
    build_sample_conversations,
)


def test_round_trip_jsonl(tmp_path: Path):
    conversations = build_sample_conversations()
    examples = ExpertValidationDataset.curate_from_conversations(conversations)

    # Ensure we produced at least 3 examples and one is crisis
    assert len(examples) >= 3
    assert any(ex.is_crisis for ex in examples)

    ds = ExpertValidationDataset(dataset_id="ds_test_001", examples=examples)

    out_path = tmp_path / "expert_dataset.jsonl"
    ds.to_jsonl(out_path)

    assert out_path.exists()

    # Read back
    ds2 = ExpertValidationDataset.from_jsonl(out_path)
    assert ds2.examples and len(ds2.examples) == len(ds.examples)

    # Validate every example remains valid and crisis preserved
    for ex in ds2.examples:
        ok, errors = ex.validate()
        assert ok, f"Validation failed: {errors}"
        if any(k in "\n".join(m.content.lower() for m in ex.conversation.messages) for k in [
            "suicide", "kill myself", "harm myself", "hurt myself", "kill them", "harm others", "abuse", "overdose"
        ]):
            assert ex.is_crisis, "Crisis indicators present but flag lost after round-trip"


def test_expected_fields_and_manifest(tmp_path: Path):
    conversations = build_sample_conversations()
    examples = ExpertValidationDataset.curate_from_conversations(conversations)
    ds = ExpertValidationDataset(dataset_id="ds_test_002", examples=examples)

    out_path = tmp_path / "expert_dataset.jsonl"
    ds.to_jsonl(out_path)

    # Check manifest
    manifest = out_path.parent / "expert_dataset_manifest.json"
    assert manifest.exists()
    data = json.loads(manifest.read_text(encoding="utf-8"))
    assert data["dataset_id"] == "ds_test_002"
    assert data["num_examples"] == len(examples)
    assert data["crisis_count"] >= 1


def test_schema_validation_on_bad_example():
    # Build minimal bad conversation (no messages)
    from ai.pixel.training.expert_validation_dataset import Conversation

    conv = Conversation()

    bad = ExpertValidationExample(
        example_id="bad_001",
        conversation=conv,
        prompt_summary="",
        expected_response_guidance="",
        scenario=ScenarioType.GENERAL_THERAPY,
    )
    ok, errors = bad.validate()
    assert not ok
    assert any("conversation must have at least one message" in e for e in errors)
