"""Unified client for CoreWeave Forge Post-Training, Inference, and Model Registry APIs."""

from __future__ import annotations

import asyncio
import json
import logging
import time
import urllib.error
import urllib.request
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import aiohttp

from training.forge.config import ForgeConfig

logger = logging.getLogger(__name__)

HTTP_ERROR_THRESHOLD = 400
USER_AGENT = "CoreWeave-Forge-Client/1.0 (+https://docs.coreweave.com/forge-home)"
TERMINAL_JOB_STATES = frozenset(
    {"succeeded", "completed", "failed", "cancelled", "canceled", "error"}
)


class ForgeAPIError(RuntimeError):
    """Raised when an API request to CoreWeave Forge fails."""

    def __init__(self, message: str, status_code: int | None = None):
        super().__init__(message)
        self.status_code = status_code


@dataclass
class ChatCompletionOptions:
    """Parameters for smart routed chat completions."""

    temperature: float = 0.7
    max_tokens: int = 2048
    tools: list[dict[str, Any]] | None = None
    extra_params: dict[str, Any] | None = None


class CoreWeaveForgeClient:
    """Client for communicating with CoreWeave Forge services."""

    def __init__(
        self,
        config: ForgeConfig | None = None,
        token: str | None = None,
        opener: Callable[..., Any] | None = None,
    ):
        self.config = config or ForgeConfig()
        self.token = token or self.config.api_key
        self._opener = opener

    def request(
        self,
        method: str,
        path: str,
        base_url: str | None = None,
        body: dict[str, Any] | None = None,
        timeout_seconds: float = 60.0,
    ) -> dict[str, Any] | list[Any]:
        """Perform an HTTP request against CoreWeave Forge API."""
        base = (base_url or self.config.training_api_base).rstrip("/")
        if not path:
            url = base
        else:
            clean_path = path if path.startswith("/") else f"/{path}"
            url = f"{base}{clean_path}"

        if self._opener is not None:
            return self._request_urllib(method, url, body, timeout_seconds)
        return asyncio.run(self._request_aiohttp(method, url, body, timeout_seconds))

    def _request_urllib(
        self,
        method: str,
        url: str,
        body: dict[str, Any] | None,
        timeout_seconds: float,
    ) -> dict[str, Any] | list[Any]:
        data = None if body is None else json.dumps(body).encode("utf-8")
        req = urllib.request.Request(
            url,
            data=data,
            method=method,
            headers={
                "Authorization": f"Bearer {self.token}",
                "Content-Type": "application/json",
                "Accept": "application/json",
                "User-Agent": USER_AGENT,
            },
        )
        try:
            with self._opener(req, timeout=timeout_seconds) as resp:  # type: ignore[misc]
                raw = resp.read()
        except urllib.error.HTTPError as exc:
            raise ForgeAPIError(
                f"CoreWeave Forge API returned HTTP {exc.code} for {method} {url}",
                status_code=exc.code,
            ) from None
        except Exception as exc:
            raise ForgeAPIError(
                f"CoreWeave Forge API request failed for {method} {url}: {type(exc).__name__}"
            ) from None

        try:
            parsed = json.loads(raw.decode("utf-8"))
        except Exception as exc:
            raise ForgeAPIError(
                f"CoreWeave Forge API returned invalid JSON from {url}"
            ) from exc

        if not isinstance(parsed, (dict, list)):
            raise ForgeAPIError(f"Unexpected response structure from {url}")
        return parsed

    async def _request_aiohttp(
        self,
        method: str,
        url: str,
        body: dict[str, Any] | None,
        timeout_seconds: float,
    ) -> dict[str, Any] | list[Any]:
        headers = {
            "Authorization": f"Bearer {self.token}",
            "Content-Type": "application/json",
            "Accept": "application/json",
            "User-Agent": USER_AGENT,
        }
        client_timeout = aiohttp.ClientTimeout(total=timeout_seconds)
        try:
            async with (
                aiohttp.ClientSession(timeout=client_timeout) as session,
                session.request(
                    method, url, json=body, headers=headers
                ) as response,
            ):
                if response.status >= HTTP_ERROR_THRESHOLD:
                    raise ForgeAPIError(
                        f"CoreWeave Forge API returned HTTP {response.status} for {method} {url}",
                        status_code=response.status,
                    )
                raw = await response.read()
        except ForgeAPIError:
            raise
        except Exception as exc:
            raise ForgeAPIError(
                f"CoreWeave Forge API request failed for {method} {url}: {type(exc).__name__}"
            ) from None

        try:
            parsed = json.loads(raw.decode("utf-8"))
        except Exception as exc:
            raise ForgeAPIError(
                f"CoreWeave Forge API returned invalid JSON from {url}"
            ) from exc

        if not isinstance(parsed, (dict, list)):
            raise ForgeAPIError(f"Unexpected response structure from {url}")
        return parsed

    # ─── Model Discovery ───────────────────────────────────────────────

    def get_base_models(self) -> list[dict[str, Any]]:
        """List trainable base models from the Post-Training API."""
        resp = self.request("GET", "/preview/base-models")
        if isinstance(resp, list):
            return [m for m in resp if isinstance(m, dict)]
        if isinstance(resp, dict):
            catalog = resp.get(
                "models", resp.get("data", {}).get("models", [])
            )
            if isinstance(catalog, list):
                return [m for m in catalog if isinstance(m, dict)]
        return []

    def get_inference_models(self) -> list[dict[str, Any]]:
        """List foundation models available on Serverless Inference."""
        resp = self.request(
            "GET", "/models", base_url=self.config.inference_api_base
        )
        if isinstance(resp, dict) and "data" in resp:
            data = resp["data"]
            if isinstance(data, list):
                return [m for m in data if isinstance(m, dict)]
        if isinstance(resp, list):
            return [m for m in resp if isinstance(m, dict)]
        return []

    # ─── Post-Training Lifecycle ───────────────────────────────────────

    def create_model(
        self,
        name: str,
        base_model: str,
        entity: str | None = None,
        project: str | None = None,
    ) -> dict[str, Any]:
        """Create or register a model entity under the specified project."""
        if base_model not in self.config.approved_models:
            raise ValueError(
                f"Unsupported base model {base_model!r}; approved models are: "
                f"{', '.join(sorted(self.config.approved_models))}"
            )
        payload = {
            "entity": entity or self.config.entity,
            "project": project or self.config.project,
            "name": name,
            "base_model": base_model,
            "return_existing": True,
        }
        resp = self.request("POST", "/preview/models", body=payload)
        if not isinstance(resp, dict):
            raise ForgeAPIError(
                "Model creation response is not a valid JSON object"
            )
        return resp

    def create_sft_job(
        self,
        model_id: str,
        training_data_url: str,
        config: dict[str, Any],
    ) -> dict[str, Any]:
        """Create a Serverless SFT training job."""
        payload = {
            "model_id": model_id,
            "training_data_url": training_data_url,
            "config": config,
        }
        resp = self.request("POST", "/preview/sft-training-jobs", body=payload)
        if not isinstance(resp, dict):
            raise ForgeAPIError("SFT job response is not a valid JSON object")
        return resp

    def create_rl_job(
        self,
        model_id: str,
        dataset_url: str,
        reward_function_spec: dict[str, Any],
        config: dict[str, Any],
    ) -> dict[str, Any]:
        """Create a Serverless RL post-training job."""
        payload = {
            "model_id": model_id,
            "dataset_url": dataset_url,
            "reward_spec": reward_function_spec,
            "config": config,
        }
        resp = self.request("POST", "/preview/rl-training-jobs", body=payload)
        if not isinstance(resp, dict):
            raise ForgeAPIError("RL job response is not a valid JSON object")
        return resp

    def get_job_status(self, job_id: str) -> dict[str, Any]:
        """Fetch the current status of a training job."""
        resp = self.request("GET", f"/preview/training-jobs/{job_id}")
        if not isinstance(resp, dict):
            raise ForgeAPIError("Status response is not a valid JSON object")
        return resp

    def poll_job_until_complete(
        self,
        job_id: str,
        poll_interval: float = 15.0,
        max_timeout: float = 7200.0,
        on_poll: Callable[[dict[str, Any]], None] | None = None,
    ) -> dict[str, Any]:
        """Poll job until it enters a terminal state or exceeds timeout."""
        start_time = time.time()
        while time.time() - start_time < max_timeout:
            status = self.get_job_status(job_id)
            if on_poll:
                on_poll(status)
            state = str(
                status.get("status", status.get("state", "unknown"))
            ).lower()
            if state in TERMINAL_JOB_STATES:
                return status
            time.sleep(poll_interval)

        raise TimeoutError(
            f"Job {job_id} did not complete within {max_timeout}s"
        )

    # ─── Smart Inference Router ────────────────────────────────────────

    def chat_completion(
        self,
        model: str,
        messages: list[dict[str, Any]],
        options: ChatCompletionOptions | None = None,
    ) -> dict[str, Any]:
        """Execute a chat completion with smart endpoint routing.

        Routes 'wandb-artifact:///' URIs to Post-Training API completions,
        and standard foundation model IDs to Serverless Inference.
        """
        opts = options or ChatCompletionOptions()
        endpoint = self.config.resolve_inference_endpoint(model)
        payload: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "temperature": opts.temperature,
            "max_tokens": opts.max_tokens,
        }
        if opts.tools:
            payload["tools"] = opts.tools
        if opts.extra_params:
            payload.update(opts.extra_params)

        t0 = time.perf_counter()
        # Call completion endpoint directly via custom base_url
        resp = self.request(
            method="POST",
            path="",
            base_url=endpoint,
            body=payload,
            timeout_seconds=90.0,
        )
        latency_ms = (time.perf_counter() - t0) * 1000.0

        if isinstance(resp, dict):
            resp["_latency_ms"] = round(latency_ms, 2)
            resp["_routed_endpoint"] = endpoint
            return resp

        raise ForgeAPIError(
            f"Unexpected non-object response from completion endpoint: {resp}"
        )
