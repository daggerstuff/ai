import importlib
from typing import Any, cast

import pytest


def convert(namespace: str, format_name: str, turns: list[dict[str, Any]], **options: Any) -> list[dict[str, Any]]:
    module = importlib.import_module(f"{namespace}.utils.converters")
    converter = getattr(module, f"{format_name}_to_chatml")
    if format_name == "json_conversation":
        rows = [{"dialog": turns}]
    else:
        rows = [{"conversations": [{"from": t.get("speaker"), "value": t["utterance"]} for t in turns]}]
    return cast("list[dict[str, Any]]", converter(rows, source="synthetic_role_boundary", **options))


@pytest.fixture(params=["ai.sourcing", "ai.pipelines.data_processing"])
def namespace(request: pytest.FixtureRequest) -> str:
    return str(request.param)


@pytest.fixture(params=["json_conversation", "sharegpt"])
def format_name(request: pytest.FixtureRequest) -> str:
    return str(request.param)


def test_only_explicit_assistant_role_becomes_target(namespace: str, format_name: str) -> None:
    records = convert(
        namespace,
        format_name,
        [
            {"speaker": " CLIENT ", "utterance": "A fictional question."},
            {"speaker": " Counselor ", "utterance": "A fictional answer."},
        ],
        user_role=" Client ",
        assistant_role="COUNSELOR",
    )
    assert records[0]["messages"] == [
        {"role": "user", "content": "A fictional question."},
        {"role": "assistant", "content": "A fictional answer."},
    ]


@pytest.mark.parametrize("unknown", ["tool", "system", "moderator", "", None])
def test_unknown_speaker_rejects_whole_conversation(namespace: str, format_name: str, unknown: str | None) -> None:
    records = convert(
        namespace,
        format_name,
        [
            {"speaker": "client", "utterance": "A fictional question."},
            {"speaker": unknown, "utterance": "Context that must not become a target."},
            {"speaker": "counselor", "utterance": "Do not detach this from its context."},
        ],
        user_role="client",
        assistant_role="counselor",
    )
    assert records == []


@pytest.mark.parametrize("speaker", ["client", "counselor"])
def test_system_prompt_does_not_make_a_one_sided_exchange_valid(namespace: str, format_name: str, speaker: str) -> None:
    assert (
        convert(
            namespace,
            format_name,
            [{"speaker": speaker, "utterance": "A fictional single turn."}],
            user_role="client",
            assistant_role="counselor",
            system_prompt="A fictional system instruction.",
        )
        == []
    )


@pytest.mark.parametrize(
    ("user_role", "assistant_role"), [("speaker", " SPEAKER "), ("", "counselor"), ("client", " ")]
)
def test_ambiguous_role_configuration_is_rejected(
    namespace: str, format_name: str, user_role: str, assistant_role: str
) -> None:
    with pytest.raises(ValueError, match="distinct non-empty"):
        convert(namespace, format_name, [], user_role=user_role, assistant_role=assistant_role)
