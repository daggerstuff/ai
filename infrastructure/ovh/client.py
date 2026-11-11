"""Minimal client for the OVHcloud public API (``api.us.ovhcloud.com``).

Scope note: OVHcloud AI Training is **not** served from this host. It lives on a
regional host (``us-east-va.ai.cloud.ovh.us``) that publishes no OpenAPI schema,
so training jobs are driven with the ``ovhai`` CLI. This client covers the
account / public-cloud / storage surfaces that *are* documented under
``/1.0/__schemas`` and ``/v2/__schemas``.

Authentication is a bearer token, resolved in this order:

1. ``OVH_TOKEN`` in the process environment
2. ``OVH_TOKEN`` in the nearest ``.env`` (walking up from the working directory)
3. the OAuth token cached by ``ovhai login`` in ``~/.config/ovhai/context.json``
"""

from __future__ import annotations

import json
import os
import time
from collections.abc import Iterator, Mapping
from pathlib import Path
from typing import Any

import httpx

DEFAULT_BASE_URL = "https://api.us.ovhcloud.com"
DEFAULT_API_VERSION = "1.0"
TOKEN_ENV = "OVH_TOKEN"
DEFAULT_REGION = "US-EAST-VA"

TRANSIENT_STATUS = frozenset({429, 500, 502, 503, 504})
MAX_ATTEMPTS = 3
BACKOFF_SECONDS = 1.5
ERROR_STATUS_FLOOR = 400

_QUOTES = "\"'"


class MissingTokenError(RuntimeError):
    """Raised when no OVH token can be found in the environment or config."""


class OvhApiError(RuntimeError):
    """An HTTP error returned by the OVHcloud API."""

    def __init__(
        self,
        status_code: int,
        klass: str,
        message: str,
        *,
        method: str,
        url: str,
    ) -> None:
        self.status_code = status_code
        self.klass = klass
        self.message = message
        self.method = method
        self.url = url
        super().__init__(f"{method} {url} -> HTTP {status_code} {klass}: {message}")

    @property
    def is_auth_error(self) -> bool:
        return self.status_code in (401, 403)


def read_env_file(path: Path) -> dict[str, str]:
    """Parse a dotenv-style file into a mapping. Never raises on bad lines."""
    values: dict[str, str] = {}
    try:
        raw = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return values
    for line in raw.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        key, _, value = stripped.partition("=")
        key = key.strip()
        if key.startswith("export "):
            key = key[len("export ") :].strip()
        if not key:
            continue
        values[key] = value.strip().strip(_QUOTES)
    return values


def find_dotenv(start: Path | None = None) -> Path | None:
    """Return the nearest ``.env`` at or above ``start`` (default: cwd)."""
    current = Path(start or Path.cwd()).resolve()
    for directory in (current, *current.parents):
        candidate = directory / ".env"
        if candidate.is_file():
            return candidate
    return None


def _ovhai_context_path() -> Path:
    return Path.home() / ".config" / "ovhai" / "context.json"


def _read_ovhai_token(context_path: Path, region: str) -> str | None:
    try:
        payload = json.loads(context_path.read_text(encoding="utf-8"))
    except OSError, json.JSONDecodeError:
        return None
    tokens = payload.get("access_tokens")
    if not isinstance(tokens, dict):
        return None
    token = tokens.get(region)
    return token if isinstance(token, str) and token else None


def resolve_token(
    *,
    env: Mapping[str, str] | None = None,
    dotenv_path: Path | None = None,
    ovhai_context: Path | None = None,
    region: str = DEFAULT_REGION,
) -> str:
    """Return the OVH bearer token, or raise :class:`MissingTokenError`."""
    token, _ = resolve_token_with_source(env=env, dotenv_path=dotenv_path, ovhai_context=ovhai_context, region=region)
    return token


def resolve_token_with_source(
    *,
    env: Mapping[str, str] | None = None,
    dotenv_path: Path | None = None,
    ovhai_context: Path | None = None,
    region: str = DEFAULT_REGION,
) -> tuple[str, str]:
    """Return ``(token, human-readable source)`` for the resolved token."""
    environ = os.environ if env is None else env
    from_env = environ.get(TOKEN_ENV)
    if from_env:
        return from_env, f"environment {TOKEN_ENV}"

    env_path = dotenv_path or find_dotenv()
    if env_path is not None:
        from_file = read_env_file(env_path).get(TOKEN_ENV)
        if from_file:
            return from_file, f"{env_path}:{TOKEN_ENV}"

    context_path = ovhai_context or _ovhai_context_path()
    from_ovhai = _read_ovhai_token(context_path, region)
    if from_ovhai:
        return from_ovhai, f"{context_path}:access_tokens[{region}]"

    raise MissingTokenError(
        f"no OVH token found: set {TOKEN_ENV} in the environment or in {find_dotenv() or '.env'}, "
        f"or run `ovhai login` to populate {context_path}"
    )


def _error_detail(response: httpx.Response) -> tuple[str, str]:
    """Extract ``(class, message)`` from an OVH error body."""
    try:
        payload = response.json()
    except ValueError:
        return "Client::Error", response.text.strip()[:500] or response.reason_phrase
    if isinstance(payload, dict):
        klass = str(payload.get("class") or "Client::Error")
        message = str(payload.get("message") or response.reason_phrase)
        return klass, message
    return "Client::Error", str(payload)[:500]


class OvhClient:
    """Synchronous OVHcloud API client.

    ``api_version`` selects the URL prefix (``1.0`` or ``v2``); individual calls
    may override it. ``transport`` is accepted so tests can inject an
    ``httpx.MockTransport``.
    """

    def __init__(
        self,
        token: str | None = None,
        *,
        base_url: str = DEFAULT_BASE_URL,
        api_version: str = DEFAULT_API_VERSION,
        timeout: float = 30.0,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        self._token = token
        self.base_url = base_url.rstrip("/")
        self.api_version = api_version.strip("/")
        self.max_attempts = MAX_ATTEMPTS
        self.sleep = time.sleep
        self._client = httpx.Client(timeout=timeout, transport=transport)

    # -- lifecycle ---------------------------------------------------------
    def close(self) -> None:
        self._client.close()

    def __enter__(self) -> OvhClient:
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()

    # -- token -------------------------------------------------------------
    @property
    def token(self) -> str:
        if self._token is None:
            self._token = resolve_token()
        return self._token

    def url_for(self, path: str, *, api_version: str | None = None) -> str:
        version = self.api_version if api_version is None else api_version.strip("/")
        suffix = "/" + path.lstrip("/")
        if not version:
            return f"{self.base_url}{suffix}"
        return f"{self.base_url}/{version}{suffix}"

    # -- transport ---------------------------------------------------------
    def request(
        self,
        method: str,
        path: str,
        *,
        params: Mapping[str, Any] | None = None,
        body: Any = None,
        api_version: str | None = None,
    ) -> Any:
        """Issue one API call and return the decoded JSON body (or ``None``)."""
        method = method.upper()
        url = self.url_for(path, api_version=api_version)
        headers = {"authorization": f"Bearer {self.token}", "accept": "application/json"}
        clean_params = {k: v for k, v in (params or {}).items() if v is not None}

        last_error: Exception | None = None
        attempts = max(1, self.max_attempts)
        for attempt in range(1, attempts + 1):
            try:
                response = self._client.request(method, url, params=clean_params or None, json=body, headers=headers)
            except httpx.TransportError as exc:  # connect/read/write/protocol
                last_error = exc
                if attempt == attempts:
                    raise RuntimeError(f"{method} {url} failed after {attempt} attempts: {exc}") from exc
                self.sleep(BACKOFF_SECONDS * attempt)
                continue

            if response.status_code in TRANSIENT_STATUS and attempt < attempts:
                self.sleep(BACKOFF_SECONDS * attempt)
                continue

            if response.status_code >= ERROR_STATUS_FLOOR:
                klass, message = _error_detail(response)
                raise OvhApiError(response.status_code, klass, message, method=method, url=url)

            if not response.content:
                return None
            try:
                return response.json()
            except ValueError:
                return response.text

        raise RuntimeError(f"{method} {url} failed: {last_error}")

    def get(self, path: str, *, params: Mapping[str, Any] | None = None, api_version: str | None = None) -> Any:
        return self.request("GET", path, params=params, api_version=api_version)

    def post(self, path: str, body: Any = None, *, api_version: str | None = None) -> Any:
        return self.request("POST", path, body=body, api_version=api_version)

    def put(self, path: str, body: Any = None, *, api_version: str | None = None) -> Any:
        return self.request("PUT", path, body=body, api_version=api_version)

    def delete(self, path: str, *, params: Mapping[str, Any] | None = None, api_version: str | None = None) -> Any:
        return self.request("DELETE", path, params=params, api_version=api_version)

    def paginate(
        self,
        path: str,
        *,
        params: Mapping[str, Any] | None = None,
        limit: int = 100,
        api_version: str | None = None,
        max_pages: int = 100,
    ) -> Iterator[Any]:
        """Walk ``?limit``/``?offset`` pages until a short page comes back.

        Only valid for endpoints that accept ``limit``/``offset``; the legacy
        ``/me/*`` collections return everything at once and simply yield one page.
        """
        offset = 0
        for _ in range(max_pages):
            page_params = {**(params or {}), "limit": limit, "offset": offset}
            page = self.get(path, params=page_params, api_version=api_version)
            if not isinstance(page, list) or not page:
                return
            yield from page
            if len(page) < limit:
                return
            offset += limit
