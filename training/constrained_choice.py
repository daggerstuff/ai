"""Read a model's probability over a closed set of answer labels.

Fireworks' System One serving stack scores a fixed menu of answers directly
instead of generating text ("Train Muse Glimmer for free", 2026-10-20): the
training loss there is K-way cross-entropy over the answer choices only, and
serving reads option probabilities rather than decoding.

OpenAI-compatible servers cannot be asked for the logprob of an arbitrary
continuation, but they do return ``top_logprobs`` for the first generated
position. Constraining the model to emit one letter per option turns that
single position into a K-way read-out: the probability mass landing on each
label token *is* the model's probability for that option.

The result is a cheap decision probe: one forward pass per question instead of
a full generation, and a confidence number that can be calibrated and
thresholded the way the Fireworks recipe calibrates its market predictions.
"""

from __future__ import annotations

import json
import math
import os
import time
import urllib.error
import urllib.request
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

DEFAULT_BASE_URL = "https://api.inference.wandb.ai/v1"
DEFAULT_TOP_LOGPROBS = 20
DEFAULT_TIMEOUT_SECONDS = 60.0
DEFAULT_MAX_ATTEMPTS = 3
DEFAULT_TEMPERATURE = 1.0
RETRYABLE_STATUSES = frozenset({408, 429})
SERVER_ERROR_MIN = 500
SERVER_ERROR_MAX = 599
BACKOFF_SECONDS = 1.5
LABEL_STRIP_CHARS = " \t\r\n.,:;!?()[]{}'\"`*_-#"
EXACTLY_ONE_TOKEN = 1


def normalize_label_token(token: str) -> str:
    """Fold a raw token down to the bare label it represents.

    Servers emit ``"A"``, ``" A"``, ``"a"`` and ``"A)"`` for the same choice
    depending on tokenizer and prompt shape, so matching has to be insensitive
    to surrounding whitespace, case and punctuation.
    """
    return token.strip(LABEL_STRIP_CHARS).lower()


@dataclass(frozen=True)
class ChoiceReadout:
    """Probability distribution over a closed label set at one position."""

    masses: Mapping[str, float]
    probs: Mapping[str, float]
    residual: float
    unscoreable: bool = False

    def best(self) -> str | None:
        """Return the most probable label, or ``None`` if nothing matched."""
        if self.unscoreable:
            return None
        # Highest probability wins; ties fall back to the lowest label so the
        # read-out never depends on dict insertion order.
        return min(self.probs.items(), key=lambda item: (-item[1], item[0]))[0]

    def probability_of(self, label: str) -> float:
        """Return the (option-set-normalised) probability of ``label``."""
        return self.probs.get(label, 0.0)

    def confidence(self) -> float:
        """Return the probability assigned to the winning label.

        This is the *share* within the option set: mass that landed on text
        outside the menu is renormalised away, matching how K-way
        cross-entropy over answer choices treats the options.
        """
        best = self.best()
        return 0.0 if best is None else self.probs[best]

    def raw_confidence(self) -> float:
        """Return the unrenormalised probability of the winning label.

        Unlike :meth:`confidence`, this keeps the residual: a read-out where
        the model spent most of its mass on text outside the menu reports a
        low number here instead of looking certain. Calibration needs this
        view, because ``confidence()`` cannot tell a confident decision apart
        from a refusal to answer at all.
        """
        best = self.best()
        return 0.0 if best is None else self.masses[best]


def read_choice_distribution(
    top_logprobs: Iterable[Mapping[str, Any]],
    labels: Sequence[str],
    *,
    normalize: bool = True,
) -> ChoiceReadout:
    """Aggregate first-token ``top_logprobs`` into a distribution over labels.

    ``normalize=True`` renormalises over the label set, which is what K-way
    cross-entropy over answer choices does: probability assigned to text
    outside the menu is reported separately as ``residual`` instead of being
    folded into the options.
    """
    if not labels:
        raise ValueError("labels must not be empty")

    canonical = {normalize_label_token(label): label for label in labels}
    if len(canonical) != len(labels):
        raise ValueError("labels must be distinct after normalisation")

    masses = dict.fromkeys(labels, 0.0)
    for entry in top_logprobs:
        token = entry.get("token")
        logprob = entry.get("logprob")
        if token is None or logprob is None:
            continue
        label = canonical.get(normalize_label_token(str(token)))
        if label is None:
            continue
        masses[label] += math.exp(float(logprob))

    total = sum(masses.values())
    residual = max(0.0, 1.0 - total)
    if total <= 0.0:
        return ChoiceReadout(
            masses=masses,
            probs=dict.fromkeys(labels, 0.0),
            residual=residual,
            unscoreable=True,
        )

    probs = {label: mass / total for label, mass in masses.items()} if normalize else dict(masses)
    return ChoiceReadout(masses=masses, probs=probs, residual=residual)


@dataclass(frozen=True)
class ChoiceProbe:
    """Minimal OpenAI-compatible client that reads one decision position."""

    model: str
    api_key: str
    base_url: str = DEFAULT_BASE_URL
    timeout: float = DEFAULT_TIMEOUT_SECONDS
    max_attempts: int = DEFAULT_MAX_ATTEMPTS
    top_logprobs: int = DEFAULT_TOP_LOGPROBS
    chat_template_kwargs: Mapping[str, Any] = field(default_factory=lambda: {"enable_thinking": False})

    @property
    def endpoint(self) -> str:
        return f"{self.base_url.rstrip('/')}/chat/completions"

    def read(
        self,
        messages: Sequence[Mapping[str, Any]],
        labels: Sequence[str],
        *,
        temperature: float = DEFAULT_TEMPERATURE,
    ) -> ChoiceReadout:
        """Ask the model to pick one label and read the distribution back."""
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": list(messages),
            "max_tokens": EXACTLY_ONE_TOKEN,
            "temperature": temperature,
            "logprobs": True,
            "top_logprobs": self.top_logprobs,
        }
        if self.chat_template_kwargs:
            payload["chat_template_kwargs"] = dict(self.chat_template_kwargs)

        body = self._post_with_retries(payload)
        choices = body.get("choices") or []
        if not choices:
            raise RuntimeError(f"no choices in probe response: {body!r}"[:400])
        logprobs = choices[0].get("logprobs") or {}
        return read_choice_distribution(logprobs.get("content") or [], labels)

    def _post_with_retries(self, payload: Mapping[str, Any]) -> Mapping[str, Any]:
        last_error: Exception | None = None
        for attempt in range(1, self.max_attempts + 1):
            try:
                return self._post(payload)
            except urllib.error.HTTPError as exc:
                last_error = exc
                if not _retryable(exc.code) or attempt == self.max_attempts:
                    raise
            except (urllib.error.URLError, TimeoutError) as exc:
                last_error = exc
                if attempt == self.max_attempts:
                    raise
            time.sleep(BACKOFF_SECONDS * attempt)
        raise RuntimeError(f"probe failed after {self.max_attempts} attempts: {last_error}")

    def _post(self, payload: Mapping[str, Any]) -> Mapping[str, Any]:
        request = urllib.request.Request(
            self.endpoint,
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=self.timeout) as response:
            body: object = json.loads(response.read().decode("utf-8"))
        if not isinstance(body, dict):
            raise ValueError("probe response must be a JSON object")
        return body


def _retryable(status: int) -> bool:
    return status in RETRYABLE_STATUSES or SERVER_ERROR_MIN <= status <= SERVER_ERROR_MAX


def load_api_key(env_name: str) -> str:
    """Read a credential from the process environment."""
    value = os.environ.get(env_name, "").strip()
    if not value:
        raise RuntimeError(f"{env_name} is not set")
    return value
