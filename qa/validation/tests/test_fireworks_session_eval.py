"""Offline contract tests for the in-session Fireworks evaluator bridge."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest
from tinker import types as tinker_types

from ai.training.fireworks_session_eval import FireworksSessionTransport, MuseGlimmerPromptTokenCounter

EXPECTED_RENDERED_PROMPT_TOKENS = 3


class FakeRenderer:
    def __init__(self) -> None:
        self.messages: list[dict[str, Any]] | None = None

    def build_generation_prompt(self, messages: list[dict[str, Any]]) -> Any:
        self.messages = messages
        return SimpleNamespace(to_ints=lambda: [10, 11, 12])

    def get_stop_sequences(self) -> list[int]:
        return [99]

    def parse_response(self, tokens: list[int]) -> tuple[object, Any]:
        assert tokens == [20, 21]
        return object(), SimpleNamespace(is_clean=True, value="stop_sequence")

    def to_openai_message(self, _parsed: object) -> dict[str, Any]:
        return {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": "search_resources:0",
                    "type": "function",
                    "function": {
                        "name": "search_resources",
                        "arguments": '{"query":"grounding"}',
                    },
                }
            ],
        }


class FakeSampler:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def sample(self, **kwargs: Any) -> Any:
        self.calls.append(kwargs)
        return SimpleNamespace(sequences=[tinker_types.SampledSequence(stop_reason="stop", _tokens_list=[20, 21])])


def test_transport_preserves_tool_context_and_returns_paired_eval_shape() -> None:
    renderer = FakeRenderer()
    base, tuned = FakeSampler(), FakeSampler()
    transport = FireworksSessionTransport({"base": base, "sft": tuned}, renderer)
    tools = [
        {
            "type": "function",
            "function": {
                "name": "search_resources",
                "description": "Search fictional resources",
                "parameters": {"type": "object", "properties": {"query": {"type": "string"}}},
            },
        }
    ]
    messages = [
        {"role": "user", "content": "Find grounding resources."},
        {
            "role": "assistant",
            "tool_calls": [
                {
                    "id": "prior-call",
                    "type": "function",
                    "function": {"name": "search_resources", "arguments": '{"query":"old"}'},
                }
            ],
        },
        {"role": "tool", "tool_call_id": "prior-call", "name": "search_resources", "content": "{}"},
        {"role": "user", "content": "Now find grounding resources."},
    ]

    result = transport.complete({"id": "sft"}, messages, tools, {"temperature": 0, "max_tokens": 32, "stop": []})

    assert renderer.messages is not None
    assert renderer.messages[0]["role"] == "_muse_glimmer_tools"
    assert renderer.messages[0]["_muse_glimmer_tool_specs"] == tools
    assert renderer.messages[2]["tool_calls"][0]["function"]["arguments"] == {"query": "old"}
    assert len(tuned.calls) == 1
    assert len(base.calls) == 0
    params = tuned.calls[0]["sampling_params"]
    assert isinstance(params, tinker_types.SamplingParams)
    assert params.stop == [99]
    assert result == {
        "response": "",
        "tool_calls": [
            {
                "id": "search_resources:0",
                "name": "search_resources",
                "arguments": '{"query":"grounding"}',
            }
        ],
        "finish_reason": "stop_sequence",
        "usage": {"prompt_tokens": 3, "completion_tokens": 2, "total_tokens": 5},
    }


def test_transport_rejects_invalid_tool_arguments_before_sampling() -> None:
    sampler = FakeSampler()
    transport = FireworksSessionTransport({"base": sampler}, FakeRenderer())

    with pytest.raises(ValueError, match="arguments must decode to an object"):
        transport.complete(
            {"id": "base"},
            [
                {
                    "role": "assistant",
                    "tool_calls": [
                        {"function": {"name": "x", "arguments": "[]"}},
                    ],
                }
            ],
            [],
            {"temperature": 0, "max_tokens": 32, "stop": []},
        )

    assert not sampler.calls


def test_token_counter_counts_the_exact_muse_glimmer_rendered_prompt() -> None:
    renderer = FakeRenderer()
    counter = MuseGlimmerPromptTokenCounter(renderer)
    tools = [{"type": "function", "function": {"name": "search_resources"}}]
    messages = [{"role": "user", "content": "find one"}]

    assert counter.count(messages, tools) == EXPECTED_RENDERED_PROMPT_TOKENS
    assert renderer.messages is not None
    assert renderer.messages[0]["role"] == "_muse_glimmer_tools"
    assert renderer.messages[0]["_muse_glimmer_tool_specs"] == tools
