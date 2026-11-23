from pathlib import Path

import pytest

from ai.pipelines.data_processing.dataset_adapters.addiction_stories_adapter import (
    AddictionStoriesAdapter as LegacyAddictionAdapter,
)
from ai.pipelines.data_processing.dataset_adapters.personalitydbench_adapter import (
    PersonalityDBenchAdapter as LegacyPersonaAdapter,
)
from ai.sourcing.dataset_adapters.addiction_stories_adapter import AddictionStoriesAdapter
from ai.sourcing.dataset_adapters.personalitydbench_adapter import PersonalityDBenchAdapter
from training.curate_pipeline import (
    SYSTEM_PROMPTS,
    classify_tier,
    to_alpaca,
    to_chatml,
    to_chatml_from_alpaca,
)


@pytest.mark.parametrize("adapter_type", [AddictionStoriesAdapter, LegacyAddictionAdapter])
def test_source_classification_does_not_become_a_reviewed_therapy_target(
    tmp_path: Path, adapter_type: type[AddictionStoriesAdapter] | type[LegacyAddictionAdapter]
) -> None:
    adapter = adapter_type("addiction_stories", tmp_path)
    records = adapter.validate(
        adapter.convert_to_chatml([{"text": "A fictional source narrative.", "label": "Not Mentioned"}])
    )
    record = records[0]

    assert record["task_type"] == "text_classification"
    assert record["clinical_reviewed"] is False
    assert classify_tier(record) == "T3_BRONZE"
    curated = to_chatml(record, classify_tier(record), SYSTEM_PROMPTS[record["task_type"]])
    assert curated["messages"][-1]["content"] == "Trauma category: Not Mentioned"
    assert curated["clinical_reviewed"] is False
    assert curated["provenance"] == record["provenance"]


@pytest.mark.parametrize("adapter_type", [PersonalityDBenchAdapter, LegacyPersonaAdapter])
def test_patient_persona_survives_clinical_curation_and_alpaca_conversion(
    tmp_path: Path, adapter_type: type[PersonalityDBenchAdapter] | type[LegacyPersonaAdapter]
) -> None:
    adapter = adapter_type("personalitydbench", tmp_path)
    records = adapter.validate(
        adapter.convert_to_chatml(
            [
                {
                    "type": "steering",
                    "persona": "A fictional patient who avoids disagreement.",
                    "prompt": "What happened in that conversation?",
                    "response": "I went along with their plan.",
                    "pd_label": "avoidant",
                }
            ]
        )
    )
    record = records[0]
    assert record["task_type"] == "patient_simulation"
    assert record["clinical_reviewed"] is False
    assert classify_tier(record) == "T3_BRONZE"
    curated = to_chatml(record, classify_tier(record), SYSTEM_PROMPTS["therapy_response_generation"])
    assert curated["messages"] == record["messages"]
    assert curated["provenance"] == record["provenance"]
    alpaca = to_alpaca(record, classify_tier(record))
    assert alpaca is not None
    assert to_chatml_from_alpaca(alpaca)["messages"] == record["messages"]
    assert to_chatml_from_alpaca(alpaca)["provenance"] == record["provenance"]


@pytest.mark.parametrize("adapter_type", [PersonalityDBenchAdapter, LegacyPersonaAdapter])
def test_missing_patient_response_does_not_invent_a_diagnostic_target(
    tmp_path: Path, adapter_type: type[PersonalityDBenchAdapter] | type[LegacyPersonaAdapter]
) -> None:
    adapter = adapter_type("personalitydbench", tmp_path)

    assert adapter.convert_to_chatml([{"type": "steering", "prompt": "Hello", "pd_label": "avoidant"}]) == []


@pytest.mark.parametrize("adapter_type", [AddictionStoriesAdapter, LegacyAddictionAdapter])
def test_unlabelled_story_does_not_invent_a_classification_target(
    tmp_path: Path, adapter_type: type[AddictionStoriesAdapter] | type[LegacyAddictionAdapter]
) -> None:
    adapter = adapter_type("addiction_stories", tmp_path)

    assert adapter.convert_to_chatml([{"text": "An unlabelled fictional story."}]) == []


def test_patient_reply_without_role_context_is_not_curated_as_therapist() -> None:
    record = {
        "task_type": "patient_simulation",
        "messages": [
            {"role": "user", "content": "What happened?"},
            {"role": "assistant", "content": "A fictional patient reply."},
        ],
    }

    assert to_chatml(record, "T3_BRONZE", SYSTEM_PROMPTS["therapy_response_generation"]) == {}
    assert to_alpaca(record, "T3_BRONZE") is None


@pytest.mark.parametrize("task_type", ["text_classification", "symptom_classification"])
@pytest.mark.parametrize("has_source_instruction", [True, False])
def test_classification_instruction_survives_all_sft_formats(task_type: str, has_source_instruction: bool) -> None:
    source_instruction = "Classify the fictional text. Return exactly one category: category_a or category_b."
    messages = [
        {"role": "user", "content": "A fictional classification input."},
        {"role": "assistant", "content": "category_a"},
    ]
    if has_source_instruction:
        messages.insert(0, {"role": "system", "content": source_instruction})
    record = {"task_type": task_type, "messages": messages, "source": "fictional_classifier"}
    expected = source_instruction if has_source_instruction else SYSTEM_PROMPTS[task_type]
    curated = to_chatml(record, "T3_BRONZE", SYSTEM_PROMPTS[task_type])
    alpaca = to_alpaca(record, "T3_BRONZE")

    assert alpaca is not None
    assert curated["messages"][0] == {"role": "system", "content": expected}
    assert alpaca["instruction"] == expected
    assert to_chatml_from_alpaca(alpaca)["messages"] == curated["messages"]
    assert curated["messages"][-1] == {"role": "assistant", "content": "category_a"}


@pytest.mark.parametrize("task_type", ["text_classification", "symptom_classification"])
def test_classification_default_call_adds_task_context(task_type: str) -> None:
    record = {
        "task_type": task_type,
        "messages": [
            {"role": "user", "content": "A fictional classification input."},
            {"role": "assistant", "content": "category_a"},
        ],
    }

    curated = to_chatml(record, "T3_BRONZE")

    assert curated["messages"][0] == {"role": "system", "content": SYSTEM_PROMPTS[task_type]}
    assert curated["messages"][-1] == record["messages"][-1]
