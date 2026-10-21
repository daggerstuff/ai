"""Bridge Fireworks' in-session Muse Glimmer sampler to the paired evaluator.

The transport is intentionally injected with samplers and renderer created by
the active Firetitan session. It does not create sessions or make network
calls itself.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any

try:
    from fireworks.training import sdk as fireworks_sdk
except ImportError:
    fireworks_sdk = None

try:
    from tinker import types as tinker_types
except ImportError:
    tinker_types = None

from qa.validation.therapy_bench import paired_eval


def _muse_messages(messages: list[dict[str, Any]], tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Convert paired-eval OpenAI chat history to Muse Glimmer renderer input."""
    result: list[dict[str, Any]] = []
    if tools:
        result.append(
            {
                "role": "_muse_glimmer_tools",
                "_muse_glimmer_tool_specs": tools,
                "content": "",
            }
        )
    for source in messages:
        message = dict(source)
        calls = message.get("tool_calls") or []
        if calls:
            normalized_calls = []
            for source_call in calls:
                call = dict(source_call)
                function = dict(call.get("function") or {})
                arguments = function.get("arguments")
                if isinstance(arguments, str):
                    arguments = json.loads(arguments)
                if not isinstance(arguments, Mapping):
                    raise ValueError("tool call arguments must decode to an object")
                function["arguments"] = dict(arguments)
                call["function"] = function
                normalized_calls.append(call)
            message["tool_calls"] = normalized_calls
        result.append(message)
    return result


class FireworksSessionTransport:
    """Synchronous paired-eval transport backed by per-arm Fireworks samplers."""

    def __init__(self, samplers: Mapping[str, Any], renderer: Any) -> None:
        self.samplers = dict(samplers)
        self.renderer = renderer

    def complete(
        self,
        arm: dict[str, str],
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        decoding: dict[str, Any],
    ) -> dict[str, Any]:
        sampler = self.samplers.get(arm["id"])
        if sampler is None:
            raise ValueError(f"no Fireworks sampler configured for arm {arm['id']!r}")
        prompt = self.renderer.build_generation_prompt(_muse_messages(messages, tools))
        sampling_params_type = getattr(fireworks_sdk, "SamplingParams", None)
        if sampling_params_type is None and tinker_types is not None:
            sampling_params_type = tinker_types.SamplingParams
        if sampling_params_type is None:
            raise RuntimeError("Fireworks SDK SamplingParams type is unavailable")
        params = sampling_params_type(
            max_tokens=decoding["max_tokens"],
            temperature=decoding["temperature"],
            stop=self.renderer.get_stop_sequences(),
        )
        sampled = sampler.sample(prompt=prompt, num_samples=1, sampling_params=params)
        if hasattr(sampled, "result"):
            sampled = sampled.result()
        sequence = sampled.sequences[0]
        tokens = list(sequence.tokens)
        parsed, termination = self.renderer.parse_response(tokens)
        message = self.renderer.to_openai_message(parsed)
        return {
            "response": message.get("content") or "",
            "tool_calls": paired_eval._tool_calls(message),
            "finish_reason": getattr(termination, "value", str(termination)),
            "usage": {
                "prompt_tokens": len(prompt.to_ints()),
                "completion_tokens": len(tokens),
                "total_tokens": len(prompt.to_ints()) + len(tokens),
            },
        }


class MuseGlimmerPromptTokenCounter:
    """Count the exact rendered Muse Glimmer prompt used by the sampler."""

    def __init__(self, renderer: Any) -> None:
        self.renderer = renderer

    def count(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]]) -> int:
        prompt = self.renderer.build_generation_prompt(_muse_messages(messages, tools))
        return len(prompt.to_ints())
