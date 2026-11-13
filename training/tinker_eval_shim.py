"""Serve a Tinker sampler behind an OpenAI-compatible chat endpoint.

``paired_eval`` only knows how to talk to an OpenAI-compatible
``/chat/completions`` route, and Tinker only knows how to sample from a
``SamplingClient``. This module is the adapter between the two: it renders the
request with the model's own chat template, samples from the saved sampler, and
returns the reply in the OpenAI response shape.

The evaluation harness refuses non-https base URLs except on loopback, so the
shim listens on 127.0.0.1 and is pointed at with an arm like::

    {"id": "tinker", "provider": "tinker", "base_url": "http://127.0.0.1:8099/v1",
     "model": "tinker-v19", "api_key_env": "TINKER_SHIM_TOKEN"}

Run it with ``--run-record ai/training/output/tinker/tinker-run.json``; the
record names the sampler weights written by ``tinker_sft_release.py``.
"""

from __future__ import annotations

import argparse
import importlib
import json
import os
import sys
import time
from collections.abc import Mapping
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

from training.tinker_sft_release import (
    DEFAULT_BASE_MODEL,
    DEFAULT_BASE_URL,
    DEFAULT_OUT_DIR,
    RUN_NAME,
)

__all__ = ["RUN_NAME"]

DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8099
DEFAULT_MAX_TOKENS = 2048

# --check output is pasted into logs, so the bearer token never leaves the process.
REDACTED = "[redacted]"
DEFAULT_MODEL_NAME = "tinker-sft"
SHIM_TOKEN_ENV = "TINKER_SHIM_TOKEN"
CHAT_PATH_SUFFIX = "/chat/completions"
MODELS_PATH_SUFFIX = "/models"
HEALTH_PATH_SUFFIX = "/healthz"
MAX_BODY_BYTES = 4 * 1024 * 1024


class ShimError(RuntimeError):
    """The shim cannot answer a request as asked."""


@dataclass(frozen=True)
class Generation:
    """One sampled reply plus the accounting the OpenAI shape needs."""

    text: str
    prompt_tokens: int
    completion_tokens: int
    finish_reason: str


@dataclass(frozen=True)
class ShimConfig:
    """Everything the sampler and the HTTP layer need to know."""

    weights: str
    base_model: str = DEFAULT_BASE_MODEL
    model_name: str = DEFAULT_MODEL_NAME
    max_tokens: int = DEFAULT_MAX_TOKENS
    tokenizer_path: str | None = None
    token: str | None = None


def load_run_record(path: Path) -> dict[str, Any]:
    """Return the run record written by ``tinker_sft_release.py``."""
    if not path.exists():
        raise ShimError(f"{path} is missing; train a sampler with tinker_sft_release.py first")
    record = json.loads(path.read_text(encoding="utf-8"))
    if not record.get("sampling_path"):
        raise ShimError(f"{path} has no sampling_path; the run never saved sampler weights")
    return record


def config_from_run_record(
    path: Path,
    *,
    tokenizer_path: str | None = None,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    token: str | None = None,
) -> ShimConfig:
    """Return the shim config named by a run record."""
    record = load_run_record(path)
    return ShimConfig(
        weights=record["sampling_path"],
        base_model=record.get("base_model") or (record.get("plan") or {}).get("base_model") or DEFAULT_BASE_MODEL,
        model_name=record.get("run_name") or DEFAULT_MODEL_NAME,
        max_tokens=max_tokens,
        tokenizer_path=tokenizer_path,
        token=token,
    )


def render_prompt(
    tokenizer: Any,
    messages: list[dict[str, Any]],
    chat_template_kwargs: dict[str, Any] | None = None,
) -> list[int]:
    """Return the token ids the model should continue from."""
    if not messages:
        raise ShimError("messages must not be empty")
    try:
        ids = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            **(chat_template_kwargs or {}),
        )
    except (TypeError, ValueError) as error:
        raise ShimError(f"cannot render the chat template: {error}") from error
    # With tokenize=True a HF tokenizer returns a BatchEncoding (a Mapping), not
    # a plain list, so iterating it directly yields the "input_ids" key name.
    if isinstance(ids, Mapping):
        ids = ids["input_ids"]
    return [int(token) for token in ids]


class TinkerSampler:
    """Sample from a saved Tinker sampler and decode with its tokenizer."""

    def __init__(
        self,
        *,
        client: Any,
        tokenizer: Any,
        tinker: Any,
        default_max_tokens: int = DEFAULT_MAX_TOKENS,
    ) -> None:
        self._client = client
        self._tokenizer = tokenizer
        self._tinker = tinker
        self._default_max_tokens = default_max_tokens

    def generate(
        self,
        messages: list[dict[str, Any]],
        *,
        max_tokens: int | None = None,
        temperature: float = 1.0,
        stop: list[str] | None = None,
        chat_template_kwargs: dict[str, Any] | None = None,
    ) -> Generation:
        """Sample one reply for ``messages``."""
        prompt_ids = render_prompt(self._tokenizer, messages, chat_template_kwargs)
        params = self._tinker.SamplingParams(
            max_tokens=max_tokens or self._default_max_tokens,
            temperature=temperature,
            stop=stop or None,
        )
        response = self._client.sample(
            self._tinker.ModelInput.from_ints(prompt_ids),
            num_samples=1,
            sampling_params=params,
        ).result()
        if not response.sequences:
            raise ShimError("the sampler returned no sequences")
        sequence = response.sequences[0]
        tokens = [int(token) for token in sequence.tokens]
        text = self._tokenizer.decode(tokens, skip_special_tokens=True)
        reason = getattr(sequence, "stop_reason", "stop")
        return Generation(
            text=text,
            prompt_tokens=len(prompt_ids),
            completion_tokens=len(tokens),
            finish_reason="length" if reason == "length" else "stop",
        )


def build_sampler(config: ShimConfig) -> TinkerSampler:
    """Return a sampler for the weights named by ``config``."""
    tinker = importlib.import_module("tinker")
    service = tinker.ServiceClient(
        base_url=os.environ.get("TINKER_BASE_URL") or DEFAULT_BASE_URL,
        api_key=os.environ.get("TINKER_API_KEY"),
    )
    client = service.create_sampling_client(model_path=config.weights)
    if config.tokenizer_path:
        tokenizer = importlib.import_module("transformers").AutoTokenizer.from_pretrained(config.tokenizer_path)
    else:
        tokenizer = client.get_tokenizer()
    return TinkerSampler(
        client=client,
        tokenizer=tokenizer,
        tinker=tinker,
        default_max_tokens=config.max_tokens,
    )


def _completion_body(model: str, generation: Generation, created: int | None = None) -> dict[str, Any]:
    """Return ``generation`` in the OpenAI chat-completion response shape."""
    created = created if created is not None else int(time.time())
    return {
        "id": f"chatcmpl-tinker-{created}",
        "object": "chat.completion",
        "created": created,
        "model": model,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": generation.text},
                "finish_reason": generation.finish_reason,
            }
        ],
        "usage": {
            "prompt_tokens": generation.prompt_tokens,
            "completion_tokens": generation.completion_tokens,
            "total_tokens": generation.prompt_tokens + generation.completion_tokens,
        },
    }


def parse_chat_request(body: dict[str, Any], default_max_tokens: int) -> dict[str, Any]:
    """Return the sampling knobs carried by an OpenAI chat request."""
    messages = body.get("messages")
    if not isinstance(messages, list) or not messages:
        raise ShimError("messages must be a non-empty list")
    if body.get("tools"):
        raise ShimError("Tinker sampling cannot emit tool calls")
    if body.get("n") not in (None, 1):
        raise ShimError("only n=1 is supported")
    temperature = body.get("temperature")
    if temperature is None:
        temperature = 1.0
    if not isinstance(temperature, (int, float)) or isinstance(temperature, bool):
        raise ShimError("temperature must be a number")
    max_tokens = body.get("max_tokens") or body.get("max_completion_tokens") or default_max_tokens
    if not isinstance(max_tokens, int) or max_tokens <= 0:
        raise ShimError("max_tokens must be a positive integer")
    stop = body.get("stop")
    if isinstance(stop, str):
        stop = [stop]
    if stop is not None and not (isinstance(stop, list) and all(isinstance(entry, str) for entry in stop)):
        raise ShimError("stop must be a string or a list of strings")
    chat_template_kwargs = body.get("chat_template_kwargs")
    if chat_template_kwargs is not None and not isinstance(chat_template_kwargs, dict):
        raise ShimError("chat_template_kwargs must be an object")
    return {
        "messages": messages,
        "temperature": float(temperature),
        "max_tokens": max_tokens,
        "stop": stop,
        "chat_template_kwargs": chat_template_kwargs,
    }


def make_handler(
    sampler: TinkerSampler,
    config: ShimConfig,
) -> type[BaseHTTPRequestHandler]:
    """Return a request handler bound to ``sampler``."""

    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"
        server_version = "TinkerShim/1.0"

        def log_message(self, fmt: str, *args: Any) -> None:
            print(f"{self.address_string()} {fmt % args}", file=sys.stderr)

        def _send_json(self, status: int, payload: dict[str, Any]) -> None:
            encoded = json.dumps(payload).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(encoded)))
            self.end_headers()
            self.wfile.write(encoded)

        def _error(self, status: int, message: str) -> None:
            self._send_json(status, {"error": {"message": message, "type": "shim_error"}})

        def _authorized(self) -> bool:
            if not config.token:
                return True
            header = self.headers.get("Authorization", "")
            return header == f"Bearer {config.token}"

        def do_GET(self) -> None:
            path = self.path.split("?", 1)[0]
            if path.endswith(HEALTH_PATH_SUFFIX):
                self._send_json(200, {"status": "ok", "model": config.model_name})
                return
            if path.endswith(MODELS_PATH_SUFFIX):
                self._send_json(
                    200,
                    {
                        "object": "list",
                        "data": [
                            {
                                "id": config.model_name,
                                "object": "model",
                                "owned_by": "tinker",
                            }
                        ],
                    },
                )
                return
            self._error(404, f"unknown route: {path}")

        def do_POST(self) -> None:
            path = self.path.split("?", 1)[0]
            if not path.endswith(CHAT_PATH_SUFFIX):
                self._error(404, f"unknown route: {path}")
                return
            if not self._authorized():
                self._error(401, "missing or invalid bearer token")
                return
            length = int(self.headers.get("Content-Length") or 0)
            if length <= 0 or length > MAX_BODY_BYTES:
                self._error(400, "request body is missing or too large")
                return
            try:
                body = json.loads(self.rfile.read(length))
            except json.JSONDecodeError as error:
                self._error(400, f"request body is not JSON: {error}")
                return
            if not isinstance(body, dict):
                self._error(400, "request body must be a JSON object")
                return
            try:
                request = parse_chat_request(body, config.max_tokens)
                generation = sampler.generate(**request)
            except ShimError as error:
                self._error(400, str(error))
                return
            except Exception as error:
                self._error(502, f"{type(error).__name__}: {error}")
                return
            model = body.get("model") or config.model_name
            self._send_json(200, _completion_body(str(model), generation))

    return Handler


def serve(config: ShimConfig, *, host: str = DEFAULT_HOST, port: int = DEFAULT_PORT) -> None:
    """Serve ``config`` until interrupted."""
    sampler = build_sampler(config)
    server = ThreadingHTTPServer((host, port), make_handler(sampler, config))
    print(f"serving {config.model_name} on http://{host}:{port}/v1", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--run-record", type=Path, default=DEFAULT_OUT_DIR / RUN_NAME)
    parser.add_argument("--tokenizer", default=None, help="tokenizer path; defaults to the sampler's")
    parser.add_argument("--max-tokens", type=int, default=DEFAULT_MAX_TOKENS)
    parser.add_argument("--host", default=DEFAULT_HOST)
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument(
        "--token",
        default=os.environ.get(SHIM_TOKEN_ENV),
        help=f"bearer token clients must present (defaults to ${SHIM_TOKEN_ENV})",
    )
    parser.add_argument("--check", action="store_true", help="print the resolved config and exit")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Run the shim from the command line."""
    args = _parse_args(argv)
    try:
        config = config_from_run_record(
            args.run_record,
            tokenizer_path=args.tokenizer,
            max_tokens=args.max_tokens,
            token=args.token,
        )
    except ShimError as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    if args.check:
        resolved = dict(config.__dict__)
        if resolved.get("token"):
            resolved["token"] = REDACTED
        print(json.dumps(resolved, indent=2, sort_keys=True))
        return 0
    serve(config, host=args.host, port=args.port)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
