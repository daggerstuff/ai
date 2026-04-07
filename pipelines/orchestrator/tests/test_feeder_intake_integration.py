from ai.pipelines.orchestrator.configs.intake_routing import CONTINUITY_HOLDOUT_LANE
from ai.pipelines.orchestrator.knowledge_text_extractor import (
    ExtractedChunk,
    KnowledgeSourceMetadata,
    KnowledgeTextExtractor,
)
from ai.pipelines.orchestrator.ingestion.intake_gates import OrchestratorIntakeGates
from ai.pipelines.orchestrator.ingestion.intake_routing_adapter import (
    apply_intake_routing,
    split_records_with_preferences,
)
from ai.training.scripts.extract_long_running_therapy import _build_output_record


def test_knowledge_chunks_convert_to_stage1_training_records():
    extractor = KnowledgeTextExtractor(registry_path="missing-registry.json")
    source = KnowledgeSourceMetadata(
        source_id="book-1",
        title="Healing Through Example",
        author="A. Clinician",
        source_type="therapeutic_book",
    )
    chunk = ExtractedChunk(
        chunk_id="book-1_0",
        content="Attachment repair requires consistency and attunement.",
        source_id="book-1",
        chunk_index=0,
        metadata=source,
    )

    records = extractor.chunks_to_training_records([chunk])

    assert len(records) == 1
    metadata = records[0]["metadata"]
    assert metadata["source_family"] == "psych_book"
    assert metadata["stage"] == "stage1_foundation"
    assert metadata["intake_target_lane"] == "stage1_foundation"
    assert metadata["requires_human_review"] is False


def test_long_running_therapy_records_are_stamped_as_continuity_holdout():
    output = _build_output_record(
        messages=[
            {"role": "system", "content": "system"},
            {"role": "user", "content": "I'm overwhelmed."},
            {"role": "assistant", "content": "Let's slow it down together."},
        ],
        s3_path="s3://pixel-data/example.jsonl",
        turns=24,
    )

    metadata = output["metadata"]
    assert metadata["source_family"] == "long_running_therapy"
    assert metadata["stage"] == CONTINUITY_HOLDOUT_LANE
    assert metadata["intake_target_lane"] == CONTINUITY_HOLDOUT_LANE
    assert metadata["split"] == "test"


def test_pipeline_routing_and_split_preferences_preserve_continuity_holdouts():
    educational = apply_intake_routing(
        [
            {
                "text": "Explain transference in therapy.",
                "messages": [
                    {"role": "user", "content": "Explain transference in therapy."},
                    {
                        "role": "assistant",
                        "content": "Transference is when earlier relational patterns show up in the present.",
                    },
                ],
                "metadata": {},
            }
        ],
        intake_gates=OrchestratorIntakeGates(),
        source_family="psychology_knowledge",
    )
    holdout = apply_intake_routing(
        [
            {
                "text": "Long session",
                "messages": [
                    {"role": "user", "content": "turn one"},
                    {"role": "assistant", "content": "turn two"},
                ],
                "metadata": {"source_family": "long_running_therapy"},
            }
        ],
        intake_gates=OrchestratorIntakeGates(),
        source_family="long_running_therapy",
    )

    assert educational[0]["metadata"]["stage"] == "stage1_foundation"
    assert holdout[0]["metadata"]["stage"] == CONTINUITY_HOLDOUT_LANE
    assert holdout[0]["metadata"]["split"] == "test"

    aggregate_split = split_records_with_preferences(educational + holdout)

    assert aggregate_split.metadata["forced_counts"]["test"] == 1
    assert any(
        item["metadata"]["stage"] == CONTINUITY_HOLDOUT_LANE
        for item in aggregate_split.test
    )
