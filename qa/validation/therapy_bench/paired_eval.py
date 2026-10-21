"""Provider-neutral, resumable paired inference for diagnostic case sets.

This runner records inference evidence only. It does not assign clinical or
promotion scores. Provider credentials are read from named environment
variables and are never written to output files.
"""

from __future__ import annotations

import argparse
import hashlib
import html
import importlib
import json
import logging
import math
import os
import re
import sys
import time
import urllib.error
import urllib.request
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Protocol
from urllib.parse import urlsplit

_CONFIG_KEYS = {"schema_version", "run_id", "decoding", "pricing", "arms"}
_DECODING_REQUIRED_KEYS = {"temperature", "max_tokens", "stop"}
_DECODING_OPTIONAL_KEYS = {"chat_template_kwargs", "response_format"}
_PRICING_KEYS = {
    "input_usd_per_million_tokens",
    "output_usd_per_million_tokens",
    "hard_cap_usd",
    "require_usage",
    "require_finish_reason",
}
_ARM_KEYS = {"id", "provider", "base_url", "model", "api_key_env"}
_OPTIONAL_ARM_KEYS = {"project"}
_CASE_KEYS = {
    "id",
    "scenario",
    "suite",
    "group_id",
    "split",
    "messages",
    "prompt",
    "tools",
    "tool_fixtures",
    "diagnostic_checks",
    "state_checks",
}
_ROLES = {"system", "user", "assistant"}
_GLIMMER_MARKERS = ("<|begin_of_text|>", "<|start|>", "<|message|>", "<|eot|>")
_MAX_OUTPUT_TOKENS = 4096
_MAX_ERROR_EXCERPT_CHARS = 1024
_MAX_TOOL_CALLS_PER_TURN = 4
_MAX_TOOL_ROUNDS = 8
_ARM_COUNT = 2
_TRANSIENT_HTTP_STATUSES = frozenset({500, 502, 503, 504})


class EvaluationInputError(ValueError):
    """Raised when inputs are ambiguous or unsafe to send to a chat endpoint."""


@dataclass(frozen=True)
class ProviderErrorDetails:
    status: int
    body_sha256: str
    body_bytes: int
    content_type: str
    top_level_keys: list[str]
    code: str | None
    message: str | None
    excerpt: str | None


class ProviderHTTPError(RuntimeError):
    """Sanitized HTTP failure; raw provider bodies are never exposed to callers."""

    def __init__(self, details: ProviderErrorDetails) -> None:
        super().__init__(f"http_{details.status}")
        self.details = details


def _sanitize_error_text(value: str, arm: dict[str, str]) -> str:
    message = value
    credential = os.environ.get(arm["api_key_env"], "")
    if credential:
        message = message.replace(credential, "[REDACTED]")
    message = re.sub(r"(?i)(bearer\s+)[A-Za-z0-9._~+/-]+=*", r"\1[REDACTED]", message)
    message = re.sub(r"(?i)(api[_-]?key|token|secret)(\s*[=:]\s*)[^\s,;<>\"]+", r"\1\2[REDACTED]", message)
    message = re.sub(r"(?is)<(script|style)\b[^>]*>.*?</\1\s*>", " ", message)
    message = re.sub(r"(?s)<[^>]+>", " ", message)
    message = re.sub(r"\s+", " ", html.unescape(message)).strip()
    return message[:_MAX_ERROR_EXCERPT_CHARS]


def _safe_provider_error(
    body: bytes, arm: dict[str, str], content_type: str
) -> tuple[str, list[str], str | None, str | None, str | None]:
    digest = sha256_bytes(body)
    top_level_keys: list[str] = []
    code = None
    message = None
    excerpt = None
    try:
        payload = json.loads(body)
    except json.JSONDecodeError, UnicodeDecodeError:
        text = body.decode("utf-8", errors="replace")
        title_match = re.search(r"(?is)<title\b[^>]*>(.*?)</title\s*>", text)
        excerpt = _sanitize_error_text(title_match.group(1) if title_match else text, arm)
        message = excerpt or None
    else:
        if isinstance(payload, dict):
            top_level_keys = sorted(
                key for key in payload if isinstance(key, str) and re.fullmatch(r"[A-Za-z0-9_.-]{1,64}", key)
            )[:32]
            error = payload.get("error")
            top_code = payload.get("code")
            if isinstance(top_code, str) and re.fullmatch(r"[A-Za-z0-9_.-]{1,80}", top_code):
                code = top_code
            candidate = payload.get("message")
            if not isinstance(candidate, str):
                candidate = payload.get("detail")
            if isinstance(error, dict):
                candidate_code = error.get("code")
                if (
                    code is None
                    and isinstance(candidate_code, str)
                    and re.fullmatch(r"[A-Za-z0-9_.-]{1,80}", candidate_code)
                ):
                    code = candidate_code
                if isinstance(error.get("message"), str):
                    candidate = error["message"]
            elif isinstance(error, str):
                candidate = error
            if isinstance(candidate, str):
                message = _sanitize_error_text(candidate, arm) or None
                excerpt = message
        elif content_type == "text/html":
            text = body.decode("utf-8", errors="replace")
            title_match = re.search(r"(?is)<title\b[^>]*>(.*?)</title\s*>", text)
            excerpt = _sanitize_error_text(title_match.group(1) if title_match else text, arm)
            message = excerpt or None
    return digest, top_level_keys, code, message, excerpt


class CompletionTransport(Protocol):
    """Minimal completion adapter used by the offline-testable runner."""

    def complete(
        self,
        arm: dict[str, str],
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        decoding: dict[str, Any],
    ) -> dict[str, Any]: ...


class PromptTokenCounter(Protocol):
    """Provider-compatible local tokenizer used before bounded smoke calls."""

    def count(self, messages: list[dict[str, str]], tools: list[dict[str, Any]]) -> int: ...


class ConservativeAsciiByteCounter:
    """Bound short prompts by UTF-8 bytes plus chat and tool framing headroom."""

    def __init__(self, framing_allowance: int = 128) -> None:
        self.framing_allowance = framing_allowance

    def count(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]]) -> int:
        message_bytes = sum(len(canonical_json(message).encode("utf-8")) + 8 for message in messages)
        system_allowance = len(b"Synthetic test only.") + 12
        tool_bytes = len(canonical_json(tools).encode("utf-8")) if tools else 0
        return message_bytes + tool_bytes + system_allowance + self.framing_allowance


@dataclass(frozen=True)
class DiagnosticMetric:
    """A deliberately non-clinical, unvalidated per-case diagnostic."""

    name: str
    passed: bool
    details: str
    validity: str = "diagnostic_unvalidated"


@dataclass(frozen=True)
class RunnerOptions:
    """Runtime selection and bounded-smoke controls."""

    arm_selection: str = "both"
    max_cases: int | None = None
    max_input_tokens: int | None = None
    max_tool_rounds: int = 0
    allow_sealed: bool = False
    continue_on_transient_http_errors: bool = False


@dataclass(frozen=True)
class RunRequest:
    """Files and bounded options for one local evaluation run."""

    cases_path: Path
    config_path: Path
    results_dir: Path
    options: RunnerOptions = RunnerOptions()


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def canonical_json(payload: Any) -> str:
    return json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        os.chmod(temporary, 0o600)
        json.dump(payload, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def _append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as stream:
        os.chmod(path, 0o600)
        stream.write(json.dumps(payload, ensure_ascii=False, separators=(",", ":")) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def load_config(path: Path) -> dict[str, Any]:
    config = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(config, dict) or set(config) != _CONFIG_KEYS:
        raise EvaluationInputError(f"Config keys must be exactly {sorted(_CONFIG_KEYS)}")
    if config["schema_version"] != 1:
        raise EvaluationInputError("Only config schema_version 1 is supported")
    if not isinstance(config["run_id"], str) or not config["run_id"].strip():
        raise EvaluationInputError("run_id must be a nonempty string")
    _validate_decoding(config["decoding"])
    _validate_pricing(config["pricing"])
    arms = config["arms"]
    if not isinstance(arms, list) or len(arms) != _ARM_COUNT:
        raise EvaluationInputError("exactly two arms are required: base and sft")
    if [arm.get("id") for arm in arms if isinstance(arm, dict)] != ["base", "sft"]:
        raise EvaluationInputError("arms must be ordered and named base, sft")
    for arm in arms:
        _validate_arm(arm)
    return config


def _validate_decoding(decoding: Any) -> None:
    if (
        not isinstance(decoding, dict)
        or not _DECODING_REQUIRED_KEYS.issubset(decoding)
        or set(decoding) - (_DECODING_REQUIRED_KEYS | _DECODING_OPTIONAL_KEYS)
    ):
        allowed = _DECODING_REQUIRED_KEYS | _DECODING_OPTIONAL_KEYS
        raise EvaluationInputError(f"decoding keys must be within {sorted(allowed)}")
    if isinstance(decoding["temperature"], bool) or decoding["temperature"] != 0:
        raise EvaluationInputError("paired diagnostic runner requires temperature 0")
    if (
        not isinstance(decoding["max_tokens"], int)
        or isinstance(decoding["max_tokens"], bool)
        or not 1 <= decoding["max_tokens"] <= _MAX_OUTPUT_TOKENS
    ):
        raise EvaluationInputError(f"decoding.max_tokens must be an integer in [1, {_MAX_OUTPUT_TOKENS}]")
    if not isinstance(decoding["stop"], list) or not all(isinstance(item, str) for item in decoding["stop"]):
        raise EvaluationInputError("decoding.stop must be a list of strings")
    if "chat_template_kwargs" in decoding:
        template_kwargs = decoding["chat_template_kwargs"]
        if (
            not isinstance(template_kwargs, dict)
            or set(template_kwargs) != {"enable_thinking"}
            or not isinstance(template_kwargs["enable_thinking"], bool)
        ):
            raise EvaluationInputError("decoding.chat_template_kwargs may only set boolean enable_thinking")
    if "response_format" in decoding:
        _validate_response_format(decoding["response_format"])


def _validate_response_format(value: Any) -> None:
    if value == {"type": "json_object"}:
        return
    if not isinstance(value, dict) or set(value) != {"type", "json_schema"} or value["type"] != "json_schema":
        raise EvaluationInputError("decoding.response_format requires json_object or a strict named JSON schema")
    schema = value["json_schema"]
    if (
        not isinstance(schema, dict)
        or set(schema) != {"name", "schema", "strict"}
        or not isinstance(schema["name"], str)
        or not re.fullmatch(r"[A-Za-z0-9_-]{1,64}", schema["name"])
        or schema["strict"] is not True
        or not isinstance(schema["schema"], dict)
        or not schema["schema"]
    ):
        raise EvaluationInputError("decoding.response_format requires a strict named JSON schema")


def _validate_arm(arm: Any) -> None:
    if not isinstance(arm, dict) or not _ARM_KEYS.issubset(arm) or set(arm) - _ARM_KEYS - _OPTIONAL_ARM_KEYS:
        raise EvaluationInputError(f"arm keys must be exactly {sorted(_ARM_KEYS)}")
    if any(not isinstance(arm[field], str) or not arm[field].strip() for field in _ARM_KEYS - {"id"}):
        raise EvaluationInputError("arm provider, endpoint, model, and credential env name must be strings")
    _validate_base_url(arm["base_url"])
    if not re.fullmatch(r"[A-Z][A-Z0-9_]*", arm["api_key_env"]):
        raise EvaluationInputError("api_key_env must be an uppercase environment variable name")
    if "project" in arm and (not isinstance(arm["project"], str) or not arm["project"].strip()):
        raise EvaluationInputError("project must be a nonempty string when supplied")


def _validate_pricing(pricing: Any) -> None:
    if not isinstance(pricing, dict) or set(pricing) != _PRICING_KEYS:
        raise EvaluationInputError(f"pricing keys must be exactly {sorted(_PRICING_KEYS)}")
    rates = (pricing["input_usd_per_million_tokens"], pricing["output_usd_per_million_tokens"])
    for rate in rates:
        if rate is not None and (not isinstance(rate, (int, float)) or isinstance(rate, bool) or rate < 0):
            raise EvaluationInputError("pricing rates must be nonnegative numbers or null")
    cap = pricing["hard_cap_usd"]
    if cap is not None and (not isinstance(cap, (int, float)) or isinstance(cap, bool) or cap <= 0):
        raise EvaluationInputError("pricing.hard_cap_usd must be a positive number or null")
    for flag in ("require_usage", "require_finish_reason"):
        if not isinstance(pricing[flag], bool):
            raise EvaluationInputError(f"pricing.{flag} must be boolean")
    if pricing["hard_cap_usd"] is not None and any(rate is None for rate in rates):
        raise EvaluationInputError("hard cost cap requires both token rates")
    if pricing["require_usage"] and any(rate is None for rate in rates):
        raise EvaluationInputError("required usage needs both token rates")


def _validate_base_url(value: str) -> None:
    parsed = urlsplit(value)
    if parsed.scheme != "https" or not parsed.netloc or parsed.username or parsed.password:
        raise EvaluationInputError("base_url must be an https URL without user info")
    if parsed.query or parsed.fragment:
        raise EvaluationInputError("base_url cannot contain query parameters or fragments")
    if parsed.path.endswith("/chat/completions"):
        raise EvaluationInputError("base_url must end before /chat/completions")


def _public_endpoint(base_url: str) -> str:
    return base_url.rstrip("/") + "/chat/completions"


def load_cases(path: Path) -> list[dict[str, Any]]:
    cases: list[dict[str, Any]] = []
    seen: set[str] = set()
    group_splits: dict[str, str] = {}
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            case = json.loads(line)
            if not isinstance(case, dict) or set(case) - _CASE_KEYS:
                raise EvaluationInputError(f"case line {line_number} has unsupported fields")
            _validate_case(case, line_number, seen)
            group_id = case.get("group_id")
            split = case.get("split")
            if group_id is not None or split is not None:
                if not isinstance(group_id, str) or not group_id.strip() or split not in {"dev", "sealed"}:
                    raise EvaluationInputError(f"case line {line_number} needs group_id and dev/sealed split together")
                previous_split = group_splits.setdefault(group_id, split)
                if previous_split != split:
                    raise EvaluationInputError(f"group {group_id} crosses evaluation splits")
            if not isinstance(case.get("tool_fixtures", {}), dict):
                raise EvaluationInputError(f"case line {line_number} tool_fixtures must be an object")
            state_checks = case.get("state_checks", [])
            if not isinstance(state_checks, list):
                raise EvaluationInputError(f"case line {line_number} state_checks must be a list")
            for check in state_checks:
                if not isinstance(check, dict):
                    raise EvaluationInputError(f"case line {line_number} state checks must be objects")
                _validate_state_check(check)
            cases.append(case)
    if not cases:
        raise EvaluationInputError("case set is empty")
    return cases


def _validate_case(case: dict[str, Any], line_number: int, seen: set[str]) -> str:
    case_id = case.get("id")
    if not isinstance(case_id, str) or not case_id.strip() or case_id in seen:
        raise EvaluationInputError(f"case line {line_number} has an empty or duplicate id")
    seen.add(case_id)
    if not isinstance(case.get("scenario"), str) or not case["scenario"].strip():
        raise EvaluationInputError(f"case {case_id} needs a scenario name")
    case["messages"] = _case_messages(case, case_id)
    case.pop("prompt", None)
    tools = case.get("tools", [])
    if not isinstance(tools, list) or any(not isinstance(tool, dict) for tool in tools):
        raise EvaluationInputError(f"case {case_id} tools must be a list of objects")
    checks = case.get("diagnostic_checks", [])
    if not isinstance(checks, list):
        raise EvaluationInputError(f"case {case_id} diagnostic_checks must be a list")
    for check in checks:
        if not isinstance(check, dict):
            raise EvaluationInputError(f"case {case_id} diagnostic checks must be objects")
        _validate_check(check)
    return case_id


def _case_messages(case: dict[str, Any], case_id: str) -> list[dict[str, str]]:
    messages = case.get("messages")
    prompt = case.get("prompt")
    if messages is not None and prompt is not None:
        raise EvaluationInputError(f"case {case_id} must provide messages or prompt, not both")
    if messages is None and isinstance(prompt, str):
        if any(marker in prompt for marker in _GLIMMER_MARKERS):
            raise EvaluationInputError(
                f"case {case_id} contains a provider-rendered prompt; chat endpoints need canonical messages"
            )
        messages = [{"role": "user", "content": prompt}]
    if not isinstance(messages, list) or not messages:
        raise EvaluationInputError(f"case {case_id} needs messages or a canonical prompt")
    normalized = []
    for message in messages:
        if not isinstance(message, dict) or set(message) != {"role", "content"}:
            raise EvaluationInputError(f"case {case_id} messages require only role and content")
        if not isinstance(message["role"], str) or message["role"] not in _ROLES:
            raise EvaluationInputError(f"case {case_id} has an unsupported message role")
        if not isinstance(message["content"], str) or not message["content"].strip():
            raise EvaluationInputError(f"case {case_id} has an unsupported role or non-text content")
        normalized.append({"role": message["role"], "content": message["content"]})
    if normalized[-1]["role"] != "user":
        raise EvaluationInputError(f"case {case_id} must end with a user message")
    return normalized


def _tool_calls(choice_message: dict[str, Any]) -> list[dict[str, str]]:
    result = []
    for call in choice_message.get("tool_calls") or []:
        function = call.get("function") or {}
        result.append(
            {
                "id": str(call.get("id") or ""),
                "name": str(function.get("name") or ""),
                "arguments": str(function.get("arguments") or ""),
            }
        )
    return result


class OpenAIChatCompletions:
    """Small stdlib-only client for OpenAI-compatible chat endpoints."""

    def __init__(self, timeout_seconds: float = 120.0) -> None:
        self.timeout_seconds = timeout_seconds

    def complete(
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
        if "chat_template_kwargs" in decoding:
            body["chat_template_kwargs"] = decoding["chat_template_kwargs"]
        if "response_format" in decoding:
            body["response_format"] = decoding["response_format"]
        if decoding["stop"]:
            body["stop"] = decoding["stop"]
        if tools:
            body["tools"] = tools
        request = urllib.request.Request(
            _public_endpoint(arm["base_url"]),
            data=json.dumps(body).encode("utf-8"),
            headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"},
            method="POST",
        )
        if arm.get("project"):
            request.add_header("OpenAI-Project", arm["project"])
        try:
            with urllib.request.urlopen(request, timeout=self.timeout_seconds) as response:
                payload = json.loads(response.read())
        except urllib.error.HTTPError as exc:
            body = exc.read(262_144)
            content_type = exc.headers.get_content_type() if exc.headers else "unknown"
            body_hash, top_level_keys, provider_code, provider_message, excerpt = _safe_provider_error(
                body, arm, content_type
            )
            raise ProviderHTTPError(
                ProviderErrorDetails(
                    exc.code,
                    body_hash,
                    len(body),
                    content_type,
                    top_level_keys,
                    provider_code,
                    provider_message,
                    excerpt,
                )
            ) from None
        except (urllib.error.URLError, TimeoutError) as exc:
            raise RuntimeError(type(exc).__name__) from None
        choice = payload["choices"][0]
        message = choice["message"]
        usage = payload.get("usage") or {}
        return {
            "response": message.get("content") or "",
            "tool_calls": _tool_calls(message),
            "finish_reason": choice.get("finish_reason"),
            "usage": {key: usage.get(key) for key in ("prompt_tokens", "completion_tokens", "total_tokens")},
        }


class HuggingFaceLocalTokenCounter:
    """Count with a locally available tokenizer and forbid model-file downloads."""

    def __init__(self, tokenizer_path: str) -> None:
        auto_tokenizer = importlib.import_module("transformers").AutoTokenizer
        self._tokenizer = auto_tokenizer.from_pretrained(tokenizer_path, local_files_only=True)

    def count(self, messages: list[dict[str, str]], tools: list[dict[str, Any]]) -> int:
        encoded = self._tokenizer.apply_chat_template(
            messages,
            tools=tools or None,
            tokenize=True,
            add_generation_prompt=True,
        )
        return len(encoded)


def score_diagnostics(case: dict[str, Any], completion: dict[str, Any]) -> list[DiagnosticMetric]:
    """Run small independent checks; every result remains diagnostic only."""
    metrics: list[DiagnosticMetric] = []
    response = completion.get("response", "")
    for check in case.get("diagnostic_checks", []):
        if not isinstance(check, dict) or set(check) not in (
            {"type", "required_keys"},
            {"type", "tool_name"},
            {"type", "contains_all"},
        ):
            raise EvaluationInputError(f"case {case['id']} has an invalid diagnostic check")
        if check["type"] == "json_required_keys":
            try:
                value = json.loads(response)
                passed = isinstance(value, dict) and set(check["required_keys"]).issubset(value)
                details = "valid_json_keys_present" if passed else "missing_required_json_keys"
            except json.JSONDecodeError, TypeError:
                passed, details = False, "invalid_json"
            name = "json_required_keys"
        elif check["type"] == "tool_called":
            passed = check["tool_name"] in {call["name"] for call in completion.get("tool_calls", [])}
            details = "expected_tool_called" if passed else "expected_tool_not_called"
            name = "tool_called"
        elif check["type"] == "contains_all":
            passed = all(term.casefold() in response.casefold() for term in check["contains_all"])
            details = "required_terms_present" if passed else "required_terms_missing"
            name = "contains_all"
        else:
            raise EvaluationInputError(f"case {case['id']} has unknown diagnostic check type")
        metrics.append(DiagnosticMetric(name=name, passed=passed, details=details))
    return metrics


def _validate_check(check: dict[str, Any]) -> None:
    expected = {
        "json_required_keys": {"type", "required_keys"},
        "tool_called": {"type", "tool_name"},
        "contains_all": {"type", "contains_all"},
    }
    check_type = check.get("type")
    if check_type not in expected or set(check) != expected[check_type]:
        raise EvaluationInputError("unsupported diagnostic check schema")
    values = check.get("required_keys", check.get("contains_all", [check.get("tool_name")]))
    if not isinstance(values, list) or not values or not all(isinstance(value, str) for value in values):
        raise EvaluationInputError("diagnostic check values must be nonempty lists of strings")


def _validate_state_check(check: dict[str, Any]) -> None:
    schemas = {
        "tool_status": {"type", "tool_name", "status"},
        "memory_value": {"type", "key", "value"},
        "state_value": {"type", "field", "value"},
        "supported_citations": {"type", "required_source_ids", "allowed_source_ids"},
        "response_enum": {"type", "field", "allowed_values"},
        "response_omits": {"type", "term"},
        "response_contains": {"type", "term"},
    }
    check_type = check.get("type")
    if check_type not in schemas or set(check) != schemas[check_type]:
        raise EvaluationInputError("unsupported deterministic state check schema")
    for value in check.values():
        if isinstance(value, str) and not value:
            raise EvaluationInputError("state check string values must be nonempty")
    for field in ("required_source_ids", "allowed_source_ids", "allowed_values"):
        if field in check and (
            not isinstance(check[field], list) or not all(isinstance(value, str) and value for value in check[field])
        ):
            raise EvaluationInputError(f"{field} must be a list of nonempty strings")
    if check_type == "supported_citations" and not set(check["required_source_ids"]).issubset(
        check["allowed_source_ids"]
    ):
        raise EvaluationInputError("required source ids must be in the allowed source set")


def _json_object(response: str) -> dict[str, Any]:
    try:
        parsed = json.loads(response)
    except json.JSONDecodeError:
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _check_tool_status(
    check: dict[str, Any], _response: str, _state: dict[str, Any], trace: list[dict[str, Any]]
) -> bool:
    return any(item["name"] == check["tool_name"] and item["status"] == check["status"] for item in trace)


def _check_memory(check: dict[str, Any], _response: str, state: dict[str, Any], _trace: list[dict[str, Any]]) -> bool:
    return state.get("memory", {}).get(check["key"]) == check["value"]


def _check_state(check: dict[str, Any], _response: str, state: dict[str, Any], _trace: list[dict[str, Any]]) -> bool:
    return state.get(check["field"]) == check["value"]


def _check_citations(
    check: dict[str, Any], response: str, _state: dict[str, Any], _trace: list[dict[str, Any]]
) -> bool:
    citations = _json_object(response).get("citations", [])
    return (
        isinstance(citations, list)
        and all(isinstance(item, str) for item in citations)
        and set(check["required_source_ids"]).issubset(citations)
        and set(citations).issubset(check["allowed_source_ids"])
    )


def _check_response_enum(
    check: dict[str, Any], response: str, _state: dict[str, Any], _trace: list[dict[str, Any]]
) -> bool:
    return _json_object(response).get(check["field"]) in check["allowed_values"]


def _check_response_term(
    check: dict[str, Any], response: str, _state: dict[str, Any], _trace: list[dict[str, Any]]
) -> bool:
    return check["term"].casefold() in response.casefold()


def _check_response_omits(
    check: dict[str, Any], response: str, _state: dict[str, Any], _trace: list[dict[str, Any]]
) -> bool:
    return not _check_response_term(check, response, _state, _trace)


_STATE_CHECKERS = {
    "tool_status": _check_tool_status,
    "memory_value": _check_memory,
    "state_value": _check_state,
    "supported_citations": _check_citations,
    "response_enum": _check_response_enum,
    "response_omits": _check_response_omits,
    "response_contains": _check_response_term,
}


def score_state_checks(
    case: dict[str, Any], response: str, state: dict[str, Any], tool_trace: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    """Check synthetic task state and evidence independently of prose similarity."""
    metrics = []
    for check in case.get("state_checks", []):
        checker = _STATE_CHECKERS.get(check["type"])
        if checker is None:
            raise EvaluationInputError(f"case {case['id']} has unknown deterministic state check")
        metrics.append(
            {
                "type": check["type"],
                "passed": checker(check, response, state, tool_trace),
                "validity": "deterministic_synthetic_state_unvalidated_for_quality_or_safety",
            }
        )
    return metrics


def _tool_lookup(args: dict[str, Any], state: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    item_id = args.get("item_id", args.get("query"))
    item = state.get("catalog", {}).get(item_id)
    return (
        ("ok", {"status": "ok", "item_id": item_id, "item": item})
        if item is not None
        else ("not_found", {"status": "not_found", "item_id": item_id})
    )


def _tool_search(_args: dict[str, Any], state: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    return "ok", {"status": "ok", "sources": state.get("sources", [])}


def _tool_read(args: dict[str, Any], state: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    document_id = args.get("document_id")
    document = state.get("documents", {}).get(document_id)
    return (
        ("ok", {"status": "ok", "document_id": document_id, "document": document})
        if document
        else ("not_found", {"status": "not_found", "document_id": document_id})
    )


def _tool_memory_read(args: dict[str, Any], state: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    key = args.get("key")
    value = state.setdefault("memory", {}).get(key)
    return (
        ("ok", {"status": "ok", "key": key, "value": value})
        if value is not None
        else ("not_found", {"status": "not_found", "key": key})
    )


def _tool_memory_write(args: dict[str, Any], state: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    if "memory_write_allowed" in state and state["memory_write_allowed"] is not True:
        return "consent_required", {"status": "consent_required", "saved": False}
    key, value = args.get("key"), args.get("value")
    if not isinstance(key, str) or not key or not isinstance(value, str):
        return "error", {"status": "error", "code": "invalid_arguments"}
    state.setdefault("memory", {})[key] = value
    return "ok", {"status": "ok", "key": key, "value": value}


def _tool_submit(args: dict[str, Any], state: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    if state.get("approval_granted") is not True:
        return "approval_required", {"status": "approval_required", "change_applied": False}
    state.setdefault("changes", []).append(args.get("change"))
    return "ok", {"status": "ok", "change_applied": True}


def _tool_search_resources(_args: dict[str, Any], state: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    return "ok", {"results": state.get("resource_search", [])}


def _tool_read_resource(args: dict[str, Any], state: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    resource_id = args.get("resource_id")
    resource = state.get("resources", {}).get(resource_id)
    return ("ok", resource) if resource else ("not_found", {"id": resource_id, "status": "not_found"})


def _tool_list_navigator_slots(args: dict[str, Any], state: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    date = args.get("date")
    slots = state.get("navigator_slots", {}).get(date, [])
    return "ok", {"slots": slots}


def _tool_book_navigator_slot(args: dict[str, Any], state: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    if args.get("consent") is not True or state.get("user_consent_granted") is not True:
        return "consent_required", {"status": "consent_required", "booked": False}
    slot_id = args.get("slot_id")
    slots = [slot for values in state.get("navigator_slots", {}).values() for slot in values]
    slot = next((candidate for candidate in slots if candidate.get("id") == slot_id), None)
    if slot is None:
        return "not_found", {"status": "not_found", "booked": False}
    # An empty optional note has one stored representation. An empty string is
    # permitted by the tool schema and satisfies "leave the note empty" just as
    # null does; literal strings such as "None" remain nonempty notes.
    note = args.get("note")
    appointment = {"slot_id": slot_id, "note": None if note == "" else note}
    state.setdefault("appointments", []).append(appointment)
    return "ok", {"status": "booked", "slot": slot}


def _tool_change_workspace_status(args: dict[str, Any], state: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    token = args.get("approval_token")
    if not token or token != state.get("approval_token"):
        return "approval_required", {"status": "approval_required", "change_applied": False}
    change = {"record_id": args.get("record_id"), "status": args.get("status")}
    state.setdefault("changes", []).append(change)
    return "ok", {"status": "ok", "change_applied": True}


def _tool_read_fictional_note(args: dict[str, Any], state: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    note_id = args.get("note_id")
    note = state.get("notes", {}).get(note_id)
    return ("ok", {"note_id": note_id, "text": note}) if note else ("not_found", {"note_id": note_id})


_MOCK_TOOLS = {
    "lookup_mock_catalog": _tool_lookup,
    "search_mock_sources": _tool_search,
    "read_mock_document": _tool_read,
    "memory_read": _tool_memory_read,
    "memory_write": _tool_memory_write,
    "submit_mock_change": _tool_submit,
    "search_resources": _tool_search_resources,
    "read_resource": _tool_read_resource,
    "list_navigator_slots": _tool_list_navigator_slots,
    "book_navigator_slot": _tool_book_navigator_slot,
    "change_workspace_status": _tool_change_workspace_status,
    "read_fictional_note": _tool_read_fictional_note,
}


def _run_mock_tool(name: str, arguments: str, state: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    try:
        parsed = json.loads(arguments or "{}")
    except json.JSONDecodeError:
        return "error", {"status": "error", "code": "invalid_arguments"}
    if not isinstance(parsed, dict):
        return "error", {"status": "error", "code": "arguments_must_be_object"}
    if name in state.get("fail_tools", []):
        return "error", {"status": "error", "code": "synthetic_tool_unavailable"}
    handler = _MOCK_TOOLS.get(name)
    return handler(parsed, state) if handler else ("error", {"status": "error", "code": "unsupported_tool"})


def _record_key(record: dict[str, Any]) -> tuple[str, str]:
    return record["arm_id"], record["case_id"]


def _preflight_tokens(
    cases: list[dict[str, Any]], counter: PromptTokenCounter | None, maximum: int | None
) -> dict[str, int]:
    if maximum is None:
        return {}
    if maximum < 1:
        raise EvaluationInputError("max_input_tokens requires a positive cap")
    counter = counter or ConservativeAsciiByteCounter()
    counts = {case["id"]: counter.count(case["messages"], case.get("tools", [])) for case in cases}
    oversized = {case_id: count for case_id, count in counts.items() if count > maximum}
    if oversized:
        raise EvaluationInputError(f"preflight token cap exceeded: {oversized}")
    return counts


def _estimate_max_cost(config: dict[str, Any], case_count: int, arm_count: int, options: RunnerOptions) -> float | None:
    pricing = config["pricing"]
    cap = pricing["hard_cap_usd"]
    if cap is None:
        return None
    if options.max_input_tokens is None:
        raise EvaluationInputError("a priced hard cap requires max_input_tokens preflight")
    input_cost = options.max_input_tokens * pricing["input_usd_per_million_tokens"]
    output_cost = config["decoding"]["max_tokens"] * pricing["output_usd_per_million_tokens"]
    completion_calls_per_case = options.max_tool_rounds + 1
    estimate = case_count * arm_count * completion_calls_per_case * (input_cost + output_cost) / 1_000_000
    if estimate > cap:
        raise EvaluationInputError(f"estimated worst-case cost ${estimate:.8f} exceeds hard cap ${cap:.2f}")
    return round(estimate, 8)


def _cost_from_usage(usage: Any, pricing: dict[str, Any]) -> float | None:
    if not isinstance(usage, dict):
        return None
    if pricing["input_usd_per_million_tokens"] is None or pricing["output_usd_per_million_tokens"] is None:
        return None
    input_tokens = usage.get("prompt_tokens")
    output_tokens = usage.get("completion_tokens")
    if not isinstance(input_tokens, int) or not isinstance(output_tokens, int):
        return None
    estimate = (
        input_tokens * pricing["input_usd_per_million_tokens"]
        + output_tokens * pricing["output_usd_per_million_tokens"]
    ) / 1_000_000
    return round(estimate, 10)


def _select_arms(config: dict[str, Any], selection: str) -> list[dict[str, str]]:
    if selection not in {"base", "sft", "both"}:
        raise EvaluationInputError("arm_selection must be base, sft, or both")
    return [arm for arm in config["arms"] if selection == "both" or arm["id"] == selection]


@dataclass
class ManifestContext:
    config: dict[str, Any]
    cases: list[dict[str, Any]]
    selected: list[dict[str, str]]
    cases_digest: str
    config_digest: str
    options: RunnerOptions
    token_counts: dict[str, int]


def _make_manifest(context: ManifestContext) -> dict[str, Any]:
    config = context.config
    cases = context.cases
    selected = context.selected
    options = context.options
    return {
        "schema_version": 1,
        "run_id": config["run_id"],
        "status": "running",
        "cases_sha256": context.cases_digest,
        "config_sha256": context.config_digest,
        "case_ids": [case["id"] for case in cases],
        "arm_ids": [arm["id"] for arm in selected],
        "comparison_mode": "paired" if len(selected) == _ARM_COUNT else "single_arm_diagnostic",
        "decoding": config["decoding"],
        "input_token_cap": options.max_input_tokens,
        "max_tool_rounds": options.max_tool_rounds,
        "allow_sealed": options.allow_sealed,
        "continue_on_transient_http_errors": options.continue_on_transient_http_errors,
        "preflight_token_upper_bounds": context.token_counts,
        "arms": [
            {
                "id": arm["id"],
                "provider": arm["provider"],
                "model": arm["model"],
                "endpoint": _public_endpoint(arm["base_url"]),
                "credential_env_name": arm["api_key_env"],
                "project": arm.get("project"),
            }
            for arm in selected
        ],
        "diagnostic_metric_validity": "diagnostic_unvalidated",
        "cost_usd": None,
        "estimated_max_cost_usd": _estimate_max_cost(config, len(cases), len(selected), options),
        "actual_billed_usd": None,
        "cost_status": "usage-based estimate only; actual billed amount not verified",
    }


def _check_resume(manifest_path: Path, manifest: dict[str, Any]) -> None:
    if not manifest_path.exists():
        _atomic_json(manifest_path, manifest)
        return
    existing = json.loads(manifest_path.read_text(encoding="utf-8"))
    existing.setdefault("continue_on_transient_http_errors", False)
    previous = {key: existing.get(key) for key in manifest if key != "status"}
    current = {key: value for key, value in manifest.items() if key != "status"}
    if previous != current:
        raise EvaluationInputError("resume rejected: case/config/arm identity changed")


def _nearest_rank(values: list[float], percentile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = max(0, math.ceil(percentile * len(ordered)) - 1)
    return round(ordered[index], 3)


def _attach_summary(
    manifest: dict[str, Any],
    selected: list[dict[str, str]],
    cases: list[dict[str, Any]],
    completed: dict[tuple[str, str], dict[str, Any]],
) -> None:
    estimates = []
    latency_summary = {}
    usage_summary = {}
    truncations = {}
    for arm in selected:
        records = [completed.get((arm["id"], case["id"])) for case in cases]
        successful = [record for record in records if record and record.get("status") == "success"]
        estimates.extend(
            record["estimated_cost_usd"]
            for record in records
            if record
            if isinstance(record.get("estimated_cost_usd"), (int, float))
        )
        latencies = [
            record["latency_ms"] for record in successful if isinstance(record.get("latency_ms"), (int, float))
        ]
        latency_summary[arm["id"]] = {
            "count": len(latencies),
            "p50_ms": _nearest_rank(latencies, 0.50),
            "p95_ms": _nearest_rank(latencies, 0.95),
        }
        usage_by_case = [
            [item for item in record.get("usage_by_completion", []) if _complete_usage(item)]
            for record in records
            if record
        ]
        usage = [item for case_usage in usage_by_case for item in case_usage]
        usage_summary[arm["id"]] = {
            "cases_with_usage": sum(bool(case_usage) for case_usage in usage_by_case),
            "prompt_tokens": sum(item.get("prompt_tokens", 0) for item in usage),
            "completion_tokens": sum(item.get("completion_tokens", 0) for item in usage),
            "total_tokens": sum(item.get("total_tokens", 0) for item in usage),
        }
        truncations[arm["id"]] = sum(record.get("finish_reason") == "length" for record in successful)
    manifest["estimated_cost_usd"] = round(sum(estimates), 10) if estimates else None
    manifest["latency_ms_by_arm"] = latency_summary
    manifest["usage_totals_by_arm"] = usage_summary
    manifest["finish_reason_length_count_by_arm"] = truncations


def _read_completed(path: Path) -> dict[tuple[str, str], dict[str, Any]]:
    completed: dict[tuple[str, str], dict[str, Any]] = {}
    if path.exists():
        with path.open(encoding="utf-8") as stream:
            for line in stream:
                if line.strip():
                    record = json.loads(line)
                    completed[_record_key(record)] = record
    return completed


@dataclass
class RunContext:
    config: dict[str, Any]
    config_digest: str
    token_counts: dict[str, int]
    input_token_cap: int | None
    max_tool_rounds: int
    token_counter: PromptTokenCounter
    transport: CompletionTransport


@dataclass
class CaseExecution:
    state: dict[str, Any]
    history: list[dict[str, Any]]
    tool_trace: list[dict[str, Any]]
    usages: list[dict[str, int] | None]
    costs: list[float | None]
    input_bounds: list[int | None]
    completion_calls: int = 0
    tool_rounds: int = 0


class CaseExecutionError(RuntimeError):
    def __init__(self, code: str, fatal: bool = False, missing: list[str] | None = None) -> None:
        super().__init__(code)
        self.code = code
        self.fatal = fatal
        self.missing = missing or []


def _case_identity(case: dict[str, Any], arm: dict[str, str], context: RunContext) -> dict[str, Any]:
    return {
        "arm_id": arm["id"],
        "provider": arm["provider"],
        "model": arm["model"],
        "case_id": case["id"],
        "group_id": case.get("group_id"),
        "split": case.get("split"),
        "pair_id": sha256_bytes(f"{context.config['run_id']}:{case['id']}".encode()),
        "model_sha256": sha256_bytes(arm["model"].encode()),
        "config_sha256": context.config_digest,
        "decoding_sha256": sha256_bytes(canonical_json(context.config["decoding"]).encode()),
        "case_sha256": sha256_bytes(canonical_json(case).encode()),
        "input_tokens_preflight": context.token_counts.get(case["id"]),
        "scenario": case["scenario"],
        "suite": case.get("suite"),
        "prompt_sha256": sha256_bytes(canonical_json(case["messages"]).encode()),
    }


def _record_tool_round(calls: list[dict[str, Any]], completion: dict[str, Any], progress: CaseExecution) -> None:
    progress.history.append(
        {
            "role": "assistant",
            "content": completion.get("response") or None,
            "tool_calls": [
                {
                    "id": call["id"],
                    "type": "function",
                    "function": {"name": call["name"], "arguments": call["arguments"]},
                }
                for call in calls
            ],
        }
    )
    for index, call in enumerate(calls):
        call_id = call["id"] or f"local-tool-{progress.tool_rounds}-{index}"
        status, result = _run_mock_tool(call["name"], call["arguments"], progress.state)
        progress.tool_trace.append(
            {
                "call_id": call_id,
                "name": call["name"],
                "arguments": call["arguments"],
                "status": status,
                "result": result,
            }
        )
        progress.history.append(
            {"role": "tool", "tool_call_id": call_id, "name": call["name"], "content": canonical_json(result)}
        )
    progress.tool_rounds += 1


def _validate_completion(
    completion: dict[str, Any], bound: int | None, context: RunContext, progress: CaseExecution
) -> list[dict[str, Any]]:
    usage = completion.get("usage")
    has_usage = _complete_usage(usage)
    progress.usages.append(usage if has_usage else None)
    progress.costs.append(_cost_from_usage(usage, context.config["pricing"]))
    pricing = context.config["pricing"]
    missing = ["usage"] if pricing["require_usage"] and not has_usage else []
    if pricing["require_finish_reason"] and not completion.get("finish_reason"):
        missing.append("finish_reason")
    if missing:
        raise CaseExecutionError("RequiredProviderMetadataMissing", True, missing)
    if has_usage and (
        (bound is not None and usage["prompt_tokens"] > bound)
        or usage["completion_tokens"] > context.config["decoding"]["max_tokens"]
    ):
        raise CaseExecutionError("ProviderUsageExceededPreflight", True)
    known_costs = [cost for cost in progress.costs if isinstance(cost, (int, float))]
    cap = pricing["hard_cap_usd"]
    if cap is not None and sum(known_costs) > cap:
        raise CaseExecutionError("ProviderUsageExceededHardCostCap", True)
    calls = completion.get("tool_calls", [])
    if not isinstance(calls, list):
        raise CaseExecutionError("MalformedToolCalls")
    if len(calls) > _MAX_TOOL_CALLS_PER_TURN:
        raise CaseExecutionError("ToolCallCountExceeded")
    return calls


def _run_completion_loop(
    case: dict[str, Any], arm: dict[str, str], context: RunContext, progress: CaseExecution
) -> dict[str, Any]:
    while True:
        bound = (
            context.token_counter.count(progress.history, case.get("tools", []))
            if context.input_token_cap is not None
            else None
        )
        if bound is not None and context.input_token_cap is not None and bound > context.input_token_cap:
            raise CaseExecutionError("ToolContextExceededPreflight")
        progress.input_bounds.append(bound)
        try:
            completion = context.transport.complete(
                arm, progress.history, case.get("tools", []), context.config["decoding"]
            )
        except Exception as exc:
            exc.case_progress = progress
            raise
        progress.completion_calls += 1
        calls = _validate_completion(completion, bound, context, progress)
        if not calls or context.max_tool_rounds == 0:
            return completion
        if progress.tool_rounds >= context.max_tool_rounds:
            raise CaseExecutionError("ToolRoundLimitExceeded")
        _record_tool_round(calls, completion, progress)


def _failure_record(
    base: dict[str, Any], exc: Exception, start: float, progress: CaseExecution | None
) -> dict[str, Any]:
    message = str(exc)
    http_code = message if re.fullmatch(r"http_[0-9]{3}", message) else None
    record = {
        **base,
        "status": "error",
        "error_type": getattr(exc, "code", type(exc).__name__),
        "error_code": getattr(exc, "code", http_code),
        "missing_metadata": getattr(exc, "missing", []),
        "fatal": isinstance(exc, ProviderHTTPError) or http_code is not None or bool(getattr(exc, "fatal", False)),
        "latency_ms": round((time.monotonic() - start) * 1000, 3),
        "retry_count": 0,
        "estimated_cost_usd": None,
        "actual_billed_usd": None,
    }
    if progress:
        known_costs = [cost for cost in progress.costs if isinstance(cost, (int, float))]
        record.update(
            {
                "usage_by_completion": progress.usages,
                "input_tokens_preflight_by_completion": progress.input_bounds,
                "tool_trace": progress.tool_trace,
                "tool_rounds": progress.tool_rounds,
                "completion_calls": progress.completion_calls,
                "estimated_cost_usd": round(sum(known_costs), 10) if known_costs else None,
            }
        )
    if isinstance(exc, ProviderHTTPError):
        detail = exc.details
        record.update(
            {
                "provider_http_status": detail.status,
                "provider_error_body_sha256": detail.body_sha256,
                "provider_error_body_bytes": detail.body_bytes,
                "provider_error_content_type": detail.content_type,
                "provider_error_top_level_keys": detail.top_level_keys,
                "provider_error_code": detail.code,
                "provider_error_message": detail.message,
                "provider_error_excerpt_redacted": detail.excerpt,
            }
        )
    return record


def _success_record(
    case: dict[str, Any], base: dict[str, Any], completion: dict[str, Any], progress: CaseExecution, start: float
) -> dict[str, Any]:
    complete_usages = [usage for usage in progress.usages if usage is not None]
    usage = {
        key: sum(item[key] for item in complete_usages)
        for key in ("prompt_tokens", "completion_tokens", "total_tokens")
    }
    if len(complete_usages) != len(progress.usages) or not complete_usages:
        usage = None
    known_costs = [cost for cost in progress.costs if isinstance(cost, (int, float))]
    estimate = round(sum(known_costs), 10) if len(known_costs) == len(progress.costs) and known_costs else None
    response = completion.get("response", "")
    return {
        **base,
        "status": "success",
        "response": response,
        "tool_calls": completion.get("tool_calls", []),
        "finish_reason": completion.get("finish_reason"),
        "usage": usage,
        "usage_by_completion": progress.usages,
        "input_tokens_preflight_by_completion": progress.input_bounds,
        "completion_calls": progress.completion_calls,
        "tool_rounds": progress.tool_rounds,
        "tool_trace": progress.tool_trace,
        "final_state_sha256": sha256_bytes(canonical_json(progress.state).encode()),
        "latency_ms": round((time.monotonic() - start) * 1000, 3),
        "retry_count": 0,
        "estimated_cost_usd": estimate,
        "actual_billed_usd": None,
        "diagnostic_metrics": [asdict(metric) for metric in score_diagnostics(case, completion)],
        "state_metrics": score_state_checks(case, response, progress.state, progress.tool_trace),
    }


def _execute_case(case: dict[str, Any], arm: dict[str, str], context: RunContext) -> dict[str, Any]:
    start = time.monotonic()
    base = _case_identity(case, arm, context)
    progress = CaseExecution(
        json.loads(canonical_json(case.get("tool_fixtures", {}))), list(case["messages"]), [], [], [], []
    )
    try:
        completion = _run_completion_loop(case, arm, context, progress)
        return _success_record(case, base, completion, progress, start)
    except Exception as exc:  # Persist partial evidence without provider bodies or secrets.
        progress = getattr(exc, "case_progress", progress)
        return _failure_record(base, exc, start, progress)


def _complete_usage(usage: Any) -> bool:
    if not isinstance(usage, dict):
        return False
    for key in ("prompt_tokens", "completion_tokens", "total_tokens"):
        value = usage.get(key)
        if not isinstance(value, int) or isinstance(value, bool) or value < 0:
            return False
    return True


def run_paired(
    request: RunRequest,
    *,
    transport: CompletionTransport | None = None,
    token_counter: PromptTokenCounter | None = None,
) -> dict[str, Any]:
    """Run selected arms, persisting each completion before continuing."""
    config = load_config(request.config_path)
    cases = load_cases(request.cases_path)
    options = request.options
    if not 0 <= options.max_tool_rounds <= _MAX_TOOL_ROUNDS:
        raise EvaluationInputError(f"max_tool_rounds must be an integer in [0, {_MAX_TOOL_ROUNDS}]")
    if any(case.get("split") == "sealed" for case in cases) and not options.allow_sealed:
        raise EvaluationInputError("sealed cases require explicit allow_sealed=True")
    if options.max_cases is not None:
        if options.max_cases < 1:
            raise EvaluationInputError("max_cases must be positive")
        cases = cases[: options.max_cases]
    effective_counter = token_counter or ConservativeAsciiByteCounter()
    counts = _preflight_tokens(cases, effective_counter, options.max_input_tokens)
    selected = _select_arms(config, options.arm_selection)
    results_dir = request.results_dir
    results_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
    os.chmod(results_dir, 0o700)
    cases_digest = sha256_bytes(request.cases_path.read_bytes())
    config_digest = sha256_bytes(canonical_json(config).encode())
    manifest_path = results_dir / "manifest.json"
    manifest = _make_manifest(ManifestContext(config, cases, selected, cases_digest, config_digest, options, counts))
    _check_resume(manifest_path, manifest)
    results_path = results_dir / "results.jsonl"
    completed = _read_completed(results_path)
    if any(record.get("fatal") for record in completed.values()):
        raise EvaluationInputError("resume blocked after fatal provider metadata failure; start a new run after review")
    context = RunContext(
        config,
        config_digest,
        counts,
        options.max_input_tokens,
        options.max_tool_rounds,
        effective_counter,
        transport or OpenAIChatCompletions(),
    )
    stop_after_fatal = False
    for case in cases:
        for arm in selected:
            key = (arm["id"], case["id"])
            if completed.get(key, {}).get("status") == "success":
                continue
            record = _execute_case(case, arm, context)
            _append_jsonl(results_path, record)
            completed[key] = record
            transient_case_failure = (
                options.continue_on_transient_http_errors
                and record.get("provider_http_status") in _TRANSIENT_HTTP_STATUSES
            )
            if record.get("fatal") and not transient_case_failure:
                stop_after_fatal = True
                break
        if stop_after_fatal:
            break
    success_count = sum(
        completed.get((arm["id"], case["id"]), {}).get("status") == "success" for arm in selected for case in cases
    )
    manifest["status"] = "complete" if success_count == len(selected) * len(cases) else "incomplete"
    manifest["success_count"] = success_count
    manifest["error_count"] = len(selected) * len(cases) - success_count
    _attach_summary(manifest, selected, cases, completed)
    _atomic_json(manifest_path, manifest)
    return manifest


def build_dry_run_report(request: RunRequest, token_counter: PromptTokenCounter | None = None) -> dict[str, Any]:
    """Validate identities, split policy, preflight, and budget without creating files or connecting."""
    config = load_config(request.config_path)
    cases = load_cases(request.cases_path)
    options = request.options
    if not 0 <= options.max_tool_rounds <= _MAX_TOOL_ROUNDS:
        raise EvaluationInputError(f"max_tool_rounds must be an integer in [0, {_MAX_TOOL_ROUNDS}]")
    if any(case.get("split") == "sealed" for case in cases) and not options.allow_sealed:
        raise EvaluationInputError("sealed cases require explicit allow_sealed=True")
    if options.max_cases is not None:
        if options.max_cases < 1:
            raise EvaluationInputError("max_cases must be positive")
        cases = cases[: options.max_cases]
    selected = _select_arms(config, options.arm_selection)
    counts = _preflight_tokens(cases, token_counter or ConservativeAsciiByteCounter(), options.max_input_tokens)
    return {
        "status": "dry_run_valid",
        "provider_calls": 0,
        "cases_sha256": sha256_bytes(request.cases_path.read_bytes()),
        "config_sha256": sha256_bytes(canonical_json(config).encode()),
        "case_ids": [case["id"] for case in cases],
        "group_ids": sorted({case["group_id"] for case in cases if case.get("group_id")}),
        "splits": sorted({case["split"] for case in cases if case.get("split")}),
        "arm_ids": [arm["id"] for arm in selected],
        "decoding": config["decoding"],
        "max_tool_rounds": options.max_tool_rounds,
        "input_token_cap": options.max_input_tokens,
        "continue_on_transient_http_errors": options.continue_on_transient_http_errors,
        "preflight_token_upper_bounds": counts,
        "estimated_max_cost_usd": _estimate_max_cost(config, len(cases), len(selected), options),
        "metrics_validity": "diagnostic_unvalidated",
        "clinical_quality_claims": False,
    }


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", type=Path, required=True, help="synthetic or approved cases JSONL")
    parser.add_argument("--config", type=Path, required=True, help="strict paired arm config JSON")
    parser.add_argument("--results-dir", type=Path, required=True, help="private local output directory")
    parser.add_argument("--arm", choices=("base", "sft", "both"), default="both")
    parser.add_argument("--max-cases", type=int, default=None)
    parser.add_argument("--max-input-tokens", type=int, default=None)
    parser.add_argument("--max-tool-rounds", type=int, default=0)
    parser.add_argument("--allow-sealed", action="store_true")
    parser.add_argument("--continue-on-transient-http-errors", action="store_true")
    parser.add_argument("--dry-run", action="store_true", help="validate local inputs and budget without inference")
    parser.add_argument(
        "--tokenizer-path",
        default=None,
        help="local HF tokenizer path; loaded with local_files_only for input-token preflight",
    )
    args = parser.parse_args(argv)
    token_counter = HuggingFaceLocalTokenCounter(args.tokenizer_path) if args.tokenizer_path else None
    request = RunRequest(
        args.cases,
        args.config,
        args.results_dir,
        RunnerOptions(
            args.arm,
            args.max_cases,
            args.max_input_tokens,
            args.max_tool_rounds,
            args.allow_sealed,
            args.continue_on_transient_http_errors,
        ),
    )
    if args.dry_run:
        sys.stdout.write(json.dumps(build_dry_run_report(request, token_counter), indent=2) + "\n")
        return 0
    result = run_paired(
        request,
        token_counter=token_counter,
    )
    logging.info(json.dumps({key: result[key] for key in ("run_id", "status", "success_count", "error_count")}))
    return 0 if result["status"] == "complete" else 1


if __name__ == "__main__":
    raise SystemExit(main())
