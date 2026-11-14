"""Score a local Hugging Face checkpoint on the paired crisis evaluation.

The shared runner in ``qa.validation.therapy_bench.paired_eval`` only needs a
``CompletionTransport``, so the same grading, resume and hashing code that scores
hosted arms can score an adapter sitting on disk. That matters here because the
v22 adapters have no serving path: W&B training is down, the Fireworks account is
suspended, and Adaption cannot serve a LoRA it did not train.

The base arm is served from the same weights with the adapter switched off, so a
run costs one model in memory and the two arms differ by the adapter alone.
"""

from __future__ import annotations

import argparse
import json
import sys
from contextlib import nullcontext
from pathlib import Path
from typing import Any

from qa.validation.therapy_bench import paired_eval

CONFIG_SCHEMA_VERSION = 1
LOCAL_BASE_URL = "http://localhost"
LOCAL_PROVIDER = "local-transformers"
LOCAL_API_KEY_ENV = "LOCAL_PAIRED_EVAL"
DEFAULT_MAX_NEW_TOKENS = 512
DEFAULT_STOPS = ["<|im_end|>", "<|endoftext|>"]


def build_config(
    run_id: str,
    base_model: str,
    adapter: str | None,
    max_new_tokens: int,
    enable_thinking: bool,
) -> dict[str, Any]:
    """Build the two-arm config the shared loader validates."""
    return {
        "schema_version": CONFIG_SCHEMA_VERSION,
        "run_id": run_id,
        "decoding": {
            "temperature": 0,
            "max_tokens": max_new_tokens,
            "stop": list(DEFAULT_STOPS),
            "chat_template_kwargs": {"enable_thinking": enable_thinking},
        },
        "pricing": {
            "input_usd_per_million_tokens": None,
            "output_usd_per_million_tokens": None,
            "hard_cap_usd": None,
            "require_usage": False,
            "require_finish_reason": True,
        },
        "arms": [
            {
                "id": "base",
                "provider": LOCAL_PROVIDER,
                "base_url": LOCAL_BASE_URL,
                "model": base_model,
                "api_key_env": LOCAL_API_KEY_ENV,
            },
            {
                "id": "sft",
                "provider": LOCAL_PROVIDER,
                "base_url": LOCAL_BASE_URL,
                "model": adapter or base_model,
                "api_key_env": LOCAL_API_KEY_ENV,
            },
        ],
    }


def strip_stops(text: str, stops: list[str]) -> str:
    """Cut the reply at the first stop marker, mirroring what a server would do."""
    cut = len(text)
    for stop in stops:
        index = text.find(stop)
        if index != -1:
            cut = min(cut, index)
    return text[:cut].strip()


class LocalChatTransport:
    """Generate with one model in memory, toggling the LoRA between arms."""

    def __init__(
        self,
        base_model: str,
        adapter: str | None,
        device: str,
        dtype: str,
        enable_thinking: bool,
    ) -> None:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(base_model)
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "left"
        self.enable_thinking = enable_thinking
        self.model = AutoModelForCausalLM.from_pretrained(
            base_model,
            device_map=None if device == "cpu" else {"": device},
            dtype=getattr(torch, dtype),
            attn_implementation="sdpa",
        )
        if device == "cpu":
            self.model.to("cpu")
        self.model.eval()
        self.has_adapter = adapter is not None
        if adapter is not None:
            from peft import PeftModel

            self.model = PeftModel.from_pretrained(self.model, adapter)
            self.model.eval()

    def _render(self, messages: list[dict[str, Any]], decoding: dict[str, Any]) -> str:
        kwargs = decoding.get("chat_template_kwargs", {})
        return self.tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=False,
            **kwargs,
        )

    def complete(
        self,
        arm: dict[str, str],
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        decoding: dict[str, Any],
    ) -> dict[str, Any]:
        if tools:
            raise RuntimeError("the local transport does not serve tool calls")
        prompt = self._render(messages, decoding)
        inputs = self.tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
        device = next(self.model.parameters()).device
        inputs = {key: value.to(device) for key, value in inputs.items()}
        serving_base = arm["id"] == "base" and self.has_adapter
        adapter_scope = self.model.disable_adapter() if serving_base else nullcontext()
        with self.torch.inference_mode(), adapter_scope:
            generated = self.model.generate(
                **inputs,
                max_new_tokens=decoding["max_tokens"],
                do_sample=False,
                pad_token_id=self.tokenizer.pad_token_id,
            )
        new_tokens = generated[0][inputs["input_ids"].shape[1] :]
        text = self.tokenizer.decode(new_tokens, skip_special_tokens=True)
        return {
            "response": strip_stops(text, decoding.get("stop", [])),
            "tool_calls": [],
            "finish_reason": "stop",
            "usage": None,
        }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-model", required=True)
    parser.add_argument("--adapter", type=Path, default=None)
    parser.add_argument("--cases", type=Path, required=True)
    parser.add_argument("--results-dir", type=Path, required=True)
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--max-new-tokens", type=int, default=DEFAULT_MAX_NEW_TOKENS)
    parser.add_argument("--max-cases", type=int, default=None)
    parser.add_argument("--enable-thinking", action="store_true")
    parser.add_argument("--allow-sealed", action="store_true")
    parser.add_argument("--arm", choices=("base", "sft", "both"), default="both")
    args = parser.parse_args()

    run_id = args.run_id or (args.adapter.name if args.adapter else Path(args.base_model).name)
    results_dir = args.results_dir
    results_dir.mkdir(parents=True, exist_ok=True)
    config = build_config(
        run_id,
        args.base_model,
        str(args.adapter) if args.adapter else None,
        args.max_new_tokens,
        args.enable_thinking,
    )
    config_path = results_dir / "config.json"
    config_path.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")

    transport = LocalChatTransport(
        args.base_model,
        str(args.adapter) if args.adapter else None,
        args.device,
        args.dtype,
        args.enable_thinking,
    )
    request = paired_eval.RunRequest(
        args.cases,
        config_path,
        results_dir,
        paired_eval.RunnerOptions(
            arm_selection=args.arm,
            max_cases=args.max_cases,
            allow_sealed=args.allow_sealed,
        ),
    )
    result = paired_eval.run_paired(request, transport=transport)
    sys.stdout.write(json.dumps(result, indent=2) + "\n")
    return 0 if result.get("status") == "complete" else 1


if __name__ == "__main__":
    raise SystemExit(main())
