"""Run the paired synthetic evaluator through W&B-compatible aiohttp transport."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path
from typing import Any

import aiohttp

from qa.validation.therapy_bench import paired_eval

HTTP_ERROR_STATUS = 400


class WandbAsyncTransport:
    """Use aiohttp for W&B's endpoints while retaining the shared eval runner."""

    def __init__(self, timeout_seconds: float = 120.0):
        self.timeout_seconds = timeout_seconds

    def complete(
        self,
        arm: dict[str, str],
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        decoding: dict[str, Any],
    ) -> dict[str, Any]:
        return asyncio.run(self._complete(arm, messages, tools, decoding))

    async def _complete(
        self,
        arm: dict[str, str],
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        decoding: dict[str, Any],
    ) -> dict[str, Any]:
        token = os.environ.get(arm["api_key_env"])
        if not token:
            raise RuntimeError(f"missing credential environment variable: {arm['api_key_env']}")
        body: dict[str, Any] = {
            "model": arm["model"],
            "messages": messages,
            "temperature": decoding["temperature"],
            "max_tokens": decoding["max_tokens"],
        }
        if decoding["stop"]:
            body["stop"] = decoding["stop"]
        if "chat_template_kwargs" in decoding:
            body["chat_template_kwargs"] = decoding["chat_template_kwargs"]
        if "response_format" in decoding:
            body["response_format"] = decoding["response_format"]
        if tools:
            body["tools"] = tools
        headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}
        if arm.get("project"):
            headers["OpenAI-Project"] = arm["project"]
        async with (
            aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=self.timeout_seconds)) as session,
            session.post(paired_eval._public_endpoint(arm["base_url"]), json=body, headers=headers) as response,
        ):
            raw = await response.read()
            if response.status >= HTTP_ERROR_STATUS:
                content_type = response.headers.get("Content-Type", "unknown").split(";", 1)[0]
                body_hash, top_level_keys, code, message, excerpt = paired_eval._safe_provider_error(
                    raw, arm, content_type
                )
                raise paired_eval.ProviderHTTPError(
                    paired_eval.ProviderErrorDetails(
                        response.status,
                        body_hash,
                        len(raw),
                        content_type,
                        top_level_keys,
                        code,
                        message,
                        excerpt,
                    )
                )
        payload = json.loads(raw)
        choice = payload["choices"][0]
        message = choice["message"]
        usage = payload.get("usage") or {}
        return {
            "response": message.get("content") or "",
            "tool_calls": paired_eval._tool_calls(message),
            "finish_reason": choice.get("finish_reason"),
            "usage": {key: usage.get(key) for key in ("prompt_tokens", "completion_tokens", "total_tokens")},
        }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--results-dir", type=Path, required=True)
    parser.add_argument("--arm", choices=("base", "sft", "both"), default="both")
    parser.add_argument("--max-cases", type=int, default=None)
    parser.add_argument("--max-input-tokens", type=int, default=None)
    parser.add_argument("--max-tool-rounds", type=int, default=0)
    parser.add_argument("--allow-sealed", action="store_true")
    parser.add_argument("--continue-on-transient-http-errors", action="store_true")
    args = parser.parse_args()
    request = paired_eval.RunRequest(
        args.cases,
        args.config,
        args.results_dir,
        paired_eval.RunnerOptions(
            arm_selection=args.arm,
            max_cases=args.max_cases,
            max_input_tokens=args.max_input_tokens,
            max_tool_rounds=args.max_tool_rounds,
            allow_sealed=args.allow_sealed,
            continue_on_transient_http_errors=args.continue_on_transient_http_errors,
        ),
    )
    result = paired_eval.run_paired(request, transport=WandbAsyncTransport())
    sys.stdout.write(json.dumps(result, indent=2) + "\n")
    return 0 if result.get("status") == "complete" else 1


if __name__ == "__main__":
    raise SystemExit(main())
