from ai.pipelines.orchestrator.configs.intake_routing import (
    CONTINUITY_HOLDOUT_LANE,
    resolve_intake_route,
)
from ai.pipelines.orchestrator.configs.stages import (
    STAGE1_ID,
    STAGE2_ID,
    STAGE3_ID,
    STAGE4_ID,
)
from ai.pipelines.orchestrator.ingestion.intake_gates import OrchestratorIntakeGates


def _messages(user_text: str, assistant_text: str = "I hear you.") -> list[dict[str, str]]:
    return [
        {"role": "user", "content": user_text},
        {"role": "assistant", "content": assistant_text},
    ]


def test_resolve_intake_route_maps_known_feeders():
    assert resolve_intake_route("docs_manual").target_lane == STAGE1_ID
    assert resolve_intake_route("psych8k").target_lane == STAGE2_ID
    assert resolve_intake_route("nightmare_scenarios").target_lane == STAGE3_ID
    assert resolve_intake_route("youtube_transcript").target_lane == STAGE4_ID

    continuity_route = resolve_intake_route("long_running_therapy")
    assert continuity_route.target_lane == CONTINUITY_HOLDOUT_LANE
    assert continuity_route.split_preference == "test"


def test_intake_gates_reroute_session_crisis_to_stage3():
    gates = OrchestratorIntakeGates()

    decision = gates.evaluate(
        source_family="psych8k",
        messages=_messages("I want to kill myself and end my life tonight."),
    )

    assert decision.target_lane == STAGE3_ID
    assert decision.taxonomy_category == "crisis_support"
    assert any("crisis" in reason.lower() for reason in decision.reasons)


def test_intake_gates_reroute_educational_session_content_to_stage1():
    gates = OrchestratorIntakeGates()

    decision = gates.evaluate(
        source_family="psych8k",
        messages=_messages(
            "Can you explain cognitive behavioral therapy and describe how therapists might use it in treatment?"
        ),
    )

    assert decision.target_lane == STAGE1_ID
    assert decision.context_is_therapeutic is False
    assert any("educational" in reason.lower() for reason in decision.reasons)


def test_intake_gates_keep_persona_sources_in_stage4():
    gates = OrchestratorIntakeGates()

    decision = gates.evaluate(
        source_family="youtube_transcript",
        messages=_messages("I feel ashamed and disconnected from myself after what happened."),
    )

    assert decision.target_lane == STAGE4_ID


def test_intake_gates_force_long_running_therapy_into_continuity_holdout():
    gates = OrchestratorIntakeGates()

    decision = gates.evaluate(
        source_family="long_running_therapy",
        messages=_messages("I have been working through this with you for months now."),
    )

    assert decision.target_lane == CONTINUITY_HOLDOUT_LANE
    assert decision.split == "test"


def test_intake_gates_mark_low_confidence_items_for_human_review():
    gates = OrchestratorIntakeGates()

    decision = gates.evaluate(
        source_family="psych8k",
        messages=_messages("Hello.", "Hi."),
    )

    assert decision.target_lane == STAGE2_ID
    assert decision.requires_human_review is True
