#!/usr/bin/env python3
"""Served-adapter chat client for W&B inference.

Every probe in ``ai/training`` talks to the same endpoint with the same retry
and thinking-disabled settings, so the transport lives here instead of being
copied per script. Callers own prompt construction and scoring.

Thinking is always disabled: the v13 adapter returns ``content=None`` with the
budget spent on reasoning tokens unless ``enable_thinking`` is false, which
silently turns every probe into a measurement of an empty string.
"""

from __future__ import annotations

import json
import os
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]

ENDPOINT = "https://api.inference.wandb.ai/v1/chat/completions"
SFT_MODEL = "wandb-artifact:///wutang/pixelated-empathy-kan28/arc-sft-baseline-ecf35d5e5384:v1"
BASE_MODEL = "Qwen/Qwen3.6-35B-A3B"

# Ledgers run 600-900 chars and occasionally much longer; a tight cap truncates
# mid-JSON and loses the spoken reply with it.
MAX_TOKENS = 1400
RETRYABLE_STATUSES = (408, 429)
SERVER_ERROR_MIN = 500
SERVER_ERROR_MAX = 600
MAX_BACKOFF_SECONDS = 30


def api_key() -> str:
    """WANDB_API_KEY from the environment, falling back to the repo .env."""
    key = os.environ.get("WANDB_API_KEY")
    if key:
        return key
    env_path = ROOT / ".env"
    if env_path.exists():
        for line in env_path.read_text(encoding="utf-8").splitlines():
            if line.startswith("WANDB_API_KEY="):
                return line.split("=", 1)[1].strip().strip('"').strip("'")
    raise SystemExit("WANDB_API_KEY not found in environment or .env")


@dataclass(slots=True)
class Sampler:
    """Served-adapter client: which model, how hot, and how hard to retry."""

    model: str
    api_key: str
    temperature: float = 0.9
    timeout: float = 180.0
    attempts: int = 4
    max_tokens: int = MAX_TOKENS

    def sample(self, messages: list[dict[str, Any]], samples: int = 1) -> list[str | None]:
        body = {
            "model": self.model,
            "messages": messages,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "n": samples,
            "chat_template_kwargs": {"enable_thinking": False},
        }
        payload = json.dumps(body).encode()
        headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}

        last_error = ""
        for attempt in range(self.attempts):
            request = urllib.request.Request(ENDPOINT, data=payload, headers=headers, method="POST")
            try:
                with urllib.request.urlopen(request, timeout=self.timeout) as response:
                    data = json.loads(response.read())
                return [choice.get("message", {}).get("content") for choice in data.get("choices", [])]
            except urllib.error.HTTPError as exc:
                last_error = f"HTTP {exc.code}: {exc.read()[:300].decode(errors='replace')}"
                if exc.code in RETRYABLE_STATUSES or SERVER_ERROR_MIN <= exc.code < SERVER_ERROR_MAX:
                    time.sleep(min(2**attempt * 2, MAX_BACKOFF_SECONDS))
                    continue
                break
            except Exception as exc:  # transport failure: retried with backoff below
                last_error = f"{type(exc).__name__}: {exc}"
                time.sleep(min(2**attempt * 2, MAX_BACKOFF_SECONDS))
        print(f"  ! sampling failed: {last_error}", file=sys.stderr)
        return []
