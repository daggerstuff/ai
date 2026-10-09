"""Thin, reusable async REST client for the Cloudflare Workers AI ``clef`` model.

Clef is a 27B multimodal *decision* model: it takes a ``state`` (text, JSON,
or images) plus a schema of typed questions (``noul`` / ``choice`` / ``score``)
and returns a calibrated probability distribution over every allowed option of
every question -- never free text.

The training pipeline uses it as a structured **co-auditor**, **tiebreaker**,
and **pre-screen** (see ``eval_clef.py`` for the arc-audit probe). This module
owns only the transport + credential resolution + request validation so every
consumer shares one implementation. All audit-task mapping, scoring, and
calibration logic lives in the caller.

Endpoint (Workers AI run):

    POST https://api.cloudflare.com/client/v4/accounts/{ACCOUNT_ID}/ai/run/@cf/cloudflare/clef
    Authorization: Bearer {API_TOKEN}
    body: {"model": "clef", "state": "...", "questions": {...}}
    -> {"model": "...", "answers": {...}, "usage": {"input_tokens": n, "output_tokens": n}}
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import time
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import aiohttp

logger = logging.getLogger(__name__)

CLEF_URL_BASE = "https://api.cloudflare.com/client/v4/accounts"
DEFAULT_CLEF_MODEL = "clef"
DEFAULT_TIMEOUT_SECONDS = float(os.environ.get("CLEF_TIMEOUT", "300"))
DEFAULT_MAX_RETRIES = 4
MAX_QUESTIONS = 64
MIN_QUESTIONS = 1

# Run-schema limits (clef input schema):
CHOICE_MIN_OPTIONS = 2
CHOICE_MAX_OPTIONS = 255
SCORE_MIN_LEVELS = 2
SCORE_MAX_LEVELS = 10
SUCCESS_STATUS = 200

# Valid question ids: letters, digits, '.', '_', '-' (max 100 chars) -- per the
# clef run schema.
_QUESTION_ID_RE = re.compile(r"^[A-Za-z0-9_.-]{1,100}$")

_CONFIG_ERROR_STATUS = frozenset({400, 401, 403, 404, 413, 422})


class ClefError(RuntimeError):
    """Non-retryable clef failure (config error, bad schema, bad request).

    Carries the HTTP status when one is available so callers can distinguish
    auth errors (401/403) from oversized/bad-request errors (400/413/422).
    """

    def __init__(self, message: str, *, status: int | None = None) -> None:
        super().__init__(message)
        self.status = status


@dataclass(frozen=True)
class ClefConfig:
    """Resolved account + token + model for the clef run endpoint."""

    account_id: str
    api_token: str
    model: str = DEFAULT_CLEF_MODEL

    @property
    def url(self) -> str:
        return f"{CLEF_URL_BASE}/{self.account_id}/ai/run/@cf/cloudflare/{self.model}"


def resolve_clef_config(env: Mapping[str, str] | None = None) -> ClefConfig:
    """Resolve clef credentials from the environment.

    Reads ``CLOUDFLARE_ACCOUNT_ID`` plus the same token precedence chain the
    rest of the pipeline uses (mirrors ``generation_backend._cloudflare_api_token``:
    the dedicated Workers AI key first, since the generic ``CLOUDFLARE_API_TOKEN``
    can be repurposed for R2 and would 401 on Workers AI). ``CLEF_MODEL``
    optionally selects ``clef`` (default) or ``clef-flash``.

    Raises ``ValueError`` when the account id or a token is missing, so a
    misconfigured run fails fast instead of 401-ing mid-batch.
    """
    env = dict(os.environ if env is None else env)
    account_id = (env.get("CLOUDFLARE_ACCOUNT_ID") or "").strip()
    if not account_id:
        raise ValueError("clef requires CLOUDFLARE_ACCOUNT_ID to be set")
    api_token = (
        env.get("CLOUDFLARE_WORKERS_AI_API_KEY")
        or env.get("CLOUDFLARE_AI_API_KEY")
        or env.get("CF_AIG_TOKEN")
        or env.get("CLOUDFLARE_WORKERS_AI_API")
        or env.get("CLOUDFLARE_WORKERS_AI_API_TOKEN")
        or env.get("CLOUDFLARE_AUTH_TOKEN")
        or env.get("CLOUDFLARE_API_TOKEN")
        or env.get("CLOUDFLARE_TOKEN")
        or ""
    ).strip()
    if not api_token:
        raise ValueError("clef requires a Workers AI token; set CLOUDFLARE_WORKERS_AI_API_KEY")
    model = (env.get("CLEF_MODEL") or DEFAULT_CLEF_MODEL).strip()
    if model not in ("clef", "clef-flash"):
        raise ValueError(f"CLEF_MODEL must be 'clef' or 'clef-flash', got {model!r}")
    return ClefConfig(account_id=account_id, api_token=api_token, model=model)


def validate_questions(questions: Mapping[str, Any]) -> None:
    """Validate a clef ``questions`` schema before a costly API call.

    Enforces the run-schema contract: 1-64 questions, valid ids, and each
    question a well-formed ``noul`` / ``choice`` / ``score``. Raises
    ``ValueError`` with a specific message on the first violation.
    """
    if not isinstance(questions, Mapping) or not (MIN_QUESTIONS <= len(questions) <= MAX_QUESTIONS):
        raise ValueError(
            f"questions must be a mapping with {MIN_QUESTIONS}-{MAX_QUESTIONS} entries, "
            f"got {len(questions) if isinstance(questions, Mapping) else type(questions).__name__}"
        )
    for qid, question in questions.items():
        if not _QUESTION_ID_RE.match(qid):
            raise ValueError(f"invalid question id {qid!r}")
        if not isinstance(question, Mapping):
            raise ValueError(f"question {qid!r} must be an object")
        qtype = question.get("type")
        instructions = question.get("instructions")
        if not (isinstance(instructions, str) and instructions.strip()):
            raise ValueError(f"question {qid!r} is missing a non-empty string 'instructions'")
        criteria = question.get("criteria")
        if qtype == "noul":
            continue  # criteria optional for noul
        if qtype == "choice":
            if not isinstance(criteria, Mapping) or not (CHOICE_MIN_OPTIONS <= len(criteria) <= CHOICE_MAX_OPTIONS):
                raise ValueError(
                    f"choice question {qid!r} needs a 'criteria' mapping of "
                    f"{CHOICE_MIN_OPTIONS}-{CHOICE_MAX_OPTIONS} options"
                )
        elif qtype == "score":
            if not isinstance(criteria, (list, tuple)) or not (SCORE_MIN_LEVELS <= len(criteria) <= SCORE_MAX_LEVELS):
                raise ValueError(
                    f"score question {qid!r} needs a 'criteria' list of {SCORE_MIN_LEVELS}-{SCORE_MAX_LEVELS} levels"
                )
        else:
            raise ValueError(f"question {qid!r} has invalid type {qtype!r}; expected noul|choice|score")


def noul_prob(answers: Mapping[str, Any], qid: str) -> float | None:
    """Return the probability the ``noul`` question is yes (None if absent)."""
    ans = answers.get(qid)
    if not isinstance(ans, Mapping):
        return None
    val = ans.get("noul")
    return float(val) if isinstance(val, (int, float)) else None


def choice_of(answers: Mapping[str, Any], qid: str) -> str | None:
    """Return the highest-probability option of a ``choice`` question (or None)."""
    ans = answers.get(qid)
    if not isinstance(ans, Mapping):
        return None
    val = ans.get("choice")
    return str(val) if isinstance(val, str) else None


def score_of(answers: Mapping[str, Any], qid: str) -> float | None:
    """Return the probability-weighted level of a ``score`` question (or None)."""
    ans = answers.get(qid)
    if not isinstance(ans, Mapping):
        return None
    val = ans.get("score")
    return float(val) if isinstance(val, (int, float)) else None


def _is_valid_answers(answers: Mapping[str, Any], required: Mapping[str, str]) -> bool:
    """True iff every ``qid -> expected_type`` in ``required`` is present and typed.

    A partial answer (e.g. the model declines one question) is treated as an
    infra miss by the caller so agreement/sensitivity metrics stay honest.
    """
    for qid, expected in required.items():
        ans = answers.get(qid)
        if not isinstance(ans, Mapping):
            return False
        if expected == "noul":
            if not isinstance(ans.get("noul"), (int, float)):
                return False
        elif expected == "choice":
            if not isinstance(ans.get("choice"), str):
                return False
        elif expected == "score":
            if not isinstance(ans.get("score"), (int, float)):
                return False
        else:  # pragma: no cover - defensive
            return False
    return True


async def clef_run(
    session: aiohttp.ClientSession,
    config: ClefConfig,
    state: Any,
    questions: Mapping[str, Any],
    *,
    images: list[Any] | None = None,
) -> dict[str, Any]:
    """Run one clef evaluation and return the parsed response body.

    ``state`` is the content to evaluate (a string, or structured object/array).
    ``questions`` is the validated typed-question schema. On success returns the
    full ``{"model", "answers", "usage"}`` dict.

    Timeout (``CLEF_TIMEOUT`` env) and retry count are module defaults so the
    signature stays narrow; mirror of ``generation_backend.chat_completion``.
    Retry policy: 429/5xx and transport/timeout errors retry with exponential
    backoff; 4xx config errors (401/403/404/400/413/422) raise
    :class:`ClefError` immediately (no retry), so an auth or oversize-body
    failure is surfaced rather than burned against the rate limit.
    """
    validate_questions(questions)
    payload: dict[str, Any] = {"model": config.model, "state": state, "questions": questions}
    if images:
        payload["images"] = images
    headers = {"Authorization": f"Bearer {config.api_token}", "Content-Type": "application/json"}

    last_error: BaseException | None = None
    started = time.monotonic()
    timeout = DEFAULT_TIMEOUT_SECONDS
    for attempt in range(DEFAULT_MAX_RETRIES):
        try:
            async with session.post(
                config.url,
                json=payload,
                headers=headers,
                timeout=aiohttp.ClientTimeout(total=timeout),
            ) as response:
                body = await response.text()
                if response.status == SUCCESS_STATUS:
                    try:
                        data = json.loads(body)
                    except json.JSONDecodeError:
                        data = None
                    # The Workers AI run endpoint wraps the model output in
                    # {"result": {...}} -- unwrap so callers always see the
                    # documented {"model", "answers", "usage"} shape.
                    if isinstance(data, dict) and isinstance(data.get("result"), Mapping):
                        data = data["result"]
                    if isinstance(data, dict) and isinstance(data.get("answers"), Mapping):
                        usage = data.get("usage") or {}
                        logger.debug(
                            "clef ok: model=%s in_tok=%s out_tok=%s lat=%.1fs",
                            data.get("model"),
                            usage.get("input_tokens"),
                            usage.get("output_tokens"),
                            time.monotonic() - started,
                        )
                        return data
                    last_error = ClefError(
                        f"{SUCCESS_STATUS} body missing 'answers' or unparseable: {body[:200]!r}",
                        status=SUCCESS_STATUS,
                    )
                elif response.status in _CONFIG_ERROR_STATUS:
                    raise ClefError(f"http_{response.status}: {body[:300]}", status=response.status)
                else:
                    last_error = ClefError(f"http_{response.status}: {body[:300]}", status=response.status)
        except ClefError as e:
            if e.status in _CONFIG_ERROR_STATUS:
                raise
            last_error = e
        except (TimeoutError, asyncio.CancelledError, aiohttp.ClientError) as e:
            last_error = ClefError(f"transport_error: {type(e).__name__}: {e}")

        wait = min(2**attempt * 2, 20)
        logger.warning(
            "clef request failed (attempt %d/%d): %s -- retrying in %ds",
            attempt + 1,
            DEFAULT_MAX_RETRIES,
            str(last_error)[:160],
            wait,
        )
        await asyncio.sleep(wait)
    assert last_error is not None
    raise last_error
