from __future__ import annotations

import hashlib
import hmac
import json
import os
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from functools import lru_cache
from threading import Lock
from time import time

from fastapi import HTTPException

_NONCE_TTL_SECONDS = 300
_MAX_CLOCK_SKEW_SECONDS = 300


class ReplayProtectionStore:
    def __init__(self) -> None:
        self._lock = Lock()
        self._nonces: dict[str, float] = {}

    def validate(self, *, actor_id: str, nonce: str, now_epoch: float) -> None:
        key = f"{actor_id}:{nonce}"
        with self._lock:
            self._purge(now_epoch)
            if key in self._nonces:
                raise HTTPException(status_code=409, detail="Replay detected for signed memory request")
            self._nonces[key] = now_epoch + _NONCE_TTL_SECONDS

    def _purge(self, now_epoch: float) -> None:
        expired = [key for key, expiry in self._nonces.items() if expiry <= now_epoch]
        for key in expired:
            self._nonces.pop(key, None)


_REPLAY_PROTECTION = ReplayProtectionStore()


@dataclass(frozen=True)
class MemoryActorPolicy:
    allow_any_user: bool = False
    allowed_users: frozenset[str] = frozenset()
    allowed_user_prefixes: tuple[str, ...] = ()

    def allows_user(self, user_id: str) -> bool:
        if self.allow_any_user:
            return True
        if user_id in self.allowed_users:
            return True
        return any(user_id.startswith(prefix) for prefix in self.allowed_user_prefixes)

    def summary(self) -> dict[str, object]:
        return {
            "allow_any_user": self.allow_any_user,
            "allowed_users": sorted(self.allowed_users),
            "allowed_user_prefixes": list(self.allowed_user_prefixes),
        }


@dataclass(frozen=True)
class MemoryAccessContext:
    actor_id: str
    auth_mode: str = "internal_service_hmac"
    actor_type: str = "service"
    policy: MemoryActorPolicy = MemoryActorPolicy()
    effective_user_id: str | None = None

    def audit_metadata(self) -> dict[str, str]:
        return {
            "memory_actor_id": self.actor_id,
            "memory_actor_type": self.actor_type,
            "memory_auth_mode": self.auth_mode,
        }

    def assert_user_scope(self, user_id: str) -> str:
        value = user_id.strip()
        if not value:
            raise HTTPException(status_code=400, detail="Missing X-Memory-User-Id header")
        if not self.policy.allows_user(value):
            raise HTTPException(
                status_code=403,
                detail=f"Actor '{self.actor_id}' is not allowed to act for user '{value}'",
            )
        return value


def _normalize_actor_env_key(actor_id: str) -> str:
    return "".join(ch if ch.isalnum() else "_" for ch in actor_id.upper())


def _actor_secret_for(actor_id: str, actor_tokens: Mapping[str, str]) -> str | None:
    secret = actor_tokens.get(actor_id)
    if secret:
        return secret
    normalized_actor = _normalize_actor_env_key(actor_id).lower()
    return actor_tokens.get(normalized_actor)


def _parse_request_timestamp(raw_timestamp: str | None) -> tuple[str, float]:
    value = (raw_timestamp or "").strip()
    if not value:
        raise HTTPException(status_code=400, detail="Missing X-Memory-Timestamp header")
    try:
        return value, float(int(value))
    except ValueError:
        try:
            parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError as exc:
            raise HTTPException(status_code=400, detail="Invalid X-Memory-Timestamp header") from exc
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=UTC)
        return value, parsed.timestamp()


def _required_header(value: str | None, header_name: str) -> str:
    normalized = (value or "").strip()
    if not normalized:
        raise HTTPException(status_code=400, detail=f"Missing {header_name} header")
    return normalized


def _env_flag(name: str, default: bool = False) -> bool:
    raw_value = os.environ.get(name)
    if raw_value is None:
        return default
    return raw_value.strip().lower() in {"1", "true", "yes", "on"}


def _canonical_request(
    *,
    actor_id: str,
    user_id: str,
    method: str,
    target: str,
    body: bytes,
    timestamp: str,
    nonce: str,
) -> str:
    body_hash = hashlib.sha256(body).hexdigest()
    return "\n".join(
        [
            actor_id,
            user_id,
            method.upper(),
            target,
            body_hash,
            timestamp,
            nonce,
        ]
    )


def _actor_tokens_from_json(raw_json: str | None) -> dict[str, str]:
    tokens: dict[str, str] = {}
    if not raw_json:
        return tokens
    try:
        payload = json.loads(raw_json)
    except json.JSONDecodeError as exc:
        raise RuntimeError("LOCAL_MEMORY_ACTOR_TOKENS_JSON must be valid JSON") from exc
    if not isinstance(payload, Mapping):
        raise RuntimeError("LOCAL_MEMORY_ACTOR_TOKENS_JSON must be a JSON object")
    for actor_id, token in payload.items():
        actor_value = str(actor_id).strip()
        token_value = str(token).strip()
        if actor_value and token_value:
            tokens[actor_value] = token_value
    return tokens


def _actor_tokens_from_prefix_env() -> dict[str, str]:
    tokens: dict[str, str] = {}
    prefix = "LOCAL_MEMORY_ACTOR_TOKEN_"
    for env_key, env_value in os.environ.items():
        if not env_key.startswith(prefix):
            continue
        actor_id = env_key[len(prefix) :].strip().lower()
        token_value = env_value.strip()
        if actor_id and token_value:
            tokens[actor_id] = token_value
    return tokens


def _parse_actor_policy(actor_id: str, raw_policy: object) -> MemoryActorPolicy:
    if raw_policy is None:
        return MemoryActorPolicy(allow_any_user=True)
    if raw_policy is True:
        return MemoryActorPolicy(allow_any_user=True)
    if raw_policy is False:
        return MemoryActorPolicy()
    if isinstance(raw_policy, list):
        return MemoryActorPolicy(allowed_users=frozenset(str(item).strip() for item in raw_policy if str(item).strip()))
    if not isinstance(raw_policy, Mapping):
        raise RuntimeError(f"Policy for actor '{actor_id}' must be an object, list, boolean, or null")
    allowed_users = frozenset(str(item).strip() for item in raw_policy.get("allowed_users", []) if str(item).strip())
    allowed_prefixes = tuple(
        str(item).strip() for item in raw_policy.get("allowed_user_prefixes", []) if str(item).strip()
    )
    allow_any_user = bool(raw_policy.get("allow_any_user", False))
    if not (allow_any_user or allowed_users or allowed_prefixes):
        raise RuntimeError(f"Policy for actor '{actor_id}' must allow at least one user scope")
    return MemoryActorPolicy(
        allow_any_user=allow_any_user,
        allowed_users=allowed_users,
        allowed_user_prefixes=allowed_prefixes,
    )


@lru_cache(maxsize=1)
def configured_actor_policies() -> dict[str, MemoryActorPolicy]:
    tokens = configured_actor_tokens()
    raw_json = os.environ.get("LOCAL_MEMORY_ACTOR_POLICIES_JSON")
    if not raw_json:
        if not tokens:
            return {}
        raise RuntimeError("LOCAL_MEMORY_ACTOR_POLICIES_JSON must be configured for every shared-memory actor")
    try:
        payload = json.loads(raw_json)
    except json.JSONDecodeError as exc:
        raise RuntimeError("LOCAL_MEMORY_ACTOR_POLICIES_JSON must be valid JSON") from exc
    if not isinstance(payload, Mapping):
        raise RuntimeError("LOCAL_MEMORY_ACTOR_POLICIES_JSON must be a JSON object")
    policies: dict[str, MemoryActorPolicy] = {}
    for actor_id, raw_policy in payload.items():
        key = str(actor_id).strip()
        if not key:
            continue
        policies[key] = _parse_actor_policy(key, raw_policy)
    missing_actors = sorted(actor_id for actor_id in tokens if actor_id not in policies)
    if missing_actors:
        raise RuntimeError(
            "LOCAL_MEMORY_ACTOR_POLICIES_JSON is missing policies for actors: " + ", ".join(missing_actors)
        )
    return policies


@lru_cache(maxsize=1)
def readiness_details() -> dict[str, object]:
    actor_tokens = configured_actor_tokens()
    policy_error: str | None = None
    try:
        actor_policies = configured_actor_policies()
    except RuntimeError as exc:
        actor_policies = {}
        policy_error = str(exc)
    return {
        "auth_configured": bool(actor_tokens),
        "configured_actors": sorted(actor_tokens.keys()),
        "db_path_configured": bool(os.environ.get("HINDSIGHT_LOCAL_DB_PATH")),
        "actor_policy_mode": "scoped" if actor_policies else "invalid",
        "actor_policies_configured": bool(actor_policies),
        "actor_policies_valid": policy_error is None,
        "foresight_bearer_compat_enabled": _env_flag("HINDSIGHT_COMPAT_ENABLE_BEARER", False),
        "configuration_error": policy_error,
        "signature_required": True,
        "nonce_ttl_seconds": _NONCE_TTL_SECONDS,
        "max_clock_skew_seconds": _MAX_CLOCK_SKEW_SECONDS,
    }


@lru_cache(maxsize=1)
def configured_actor_tokens() -> dict[str, str]:
    tokens = _actor_tokens_from_json(os.environ.get("LOCAL_MEMORY_ACTOR_TOKENS_JSON"))
    tokens.update(_actor_tokens_from_prefix_env())
    return tokens


def validate_memory_auth_configuration() -> None:
    """Fail closed when shared-service auth configuration is incomplete."""
    configured_actor_policies()


def _default_compat_user_id() -> str:
    candidates = (
        os.environ.get("HINDSIGHT_COMPAT_DEFAULT_USER_ID"),
        os.environ.get("SUBCONSCIOUS_USER_ID"),
        os.environ.get("USER"),
        os.environ.get("USERNAME"),
    )
    for candidate in candidates:
        value = (candidate or "").strip()
        if value:
            return value
    raise HTTPException(
        status_code=400,
        detail=(
            "Missing X-Memory-User-Id header. Set HINDSIGHT_COMPAT_DEFAULT_USER_ID for bearer-compatible local callers."
        ),
    )


def _resolve_compat_actor_id(actor_tokens: Mapping[str, str]) -> str:
    configured_actor_id = (os.environ.get("HINDSIGHT_COMPAT_BEARER_ACTOR_ID") or "").strip()
    if configured_actor_id:
        return configured_actor_id
    if len(actor_tokens) == 1:
        return next(iter(actor_tokens))
    raise HTTPException(
        status_code=503,
        detail=(
            "Multiple memory actors are configured. "
            "Set HINDSIGHT_COMPAT_BEARER_ACTOR_ID for bearer-compatible local callers."
        ),
    )


def _compat_policy_for_actor(
    actor_id: str,
    actor_policies: Mapping[str, MemoryActorPolicy],
) -> MemoryActorPolicy:
    policy = actor_policies.get(actor_id)
    if policy is None:
        normalized_actor = _normalize_actor_env_key(actor_id).lower()
        policy = actor_policies.get(normalized_actor)
    if policy is None:
        raise HTTPException(
            status_code=503,
            detail=f"Missing policy configuration for actor '{actor_id}'",
        )
    return policy


def _resolve_compat_user_id(user_id: str | None) -> str:
    return (user_id or "").strip() or _default_compat_user_id()


def _validate_compat_bearer_token(
    *,
    authorization: str | None,
    actor_id: str,
    actor_tokens: Mapping[str, str],
) -> None:
    value = (authorization or "").strip()
    if not value:
        raise HTTPException(status_code=400, detail="Missing Authorization header")
    scheme, _, token = value.partition(" ")
    if scheme.lower() != "bearer" or not token.strip():
        raise HTTPException(status_code=400, detail="Invalid Authorization header")

    expected_secret = _actor_secret_for(actor_id, actor_tokens)
    if not expected_secret:
        raise HTTPException(
            status_code=503,
            detail=f"Unknown compatibility actor '{actor_id}'",
        )
    if not hmac.compare_digest(token.strip(), expected_secret):
        raise HTTPException(status_code=403, detail="Invalid memory bearer token")


def _authorize_bearer_compat(
    *,
    authorization: str | None,
    user_id: str | None,
    actor_tokens: Mapping[str, str],
    actor_policies: Mapping[str, MemoryActorPolicy],
) -> MemoryAccessContext | None:
    if not _env_flag("HINDSIGHT_COMPAT_ENABLE_BEARER", False):
        return None

    if not (authorization or "").strip():
        return None

    compat_actor_id = _resolve_compat_actor_id(actor_tokens)
    _validate_compat_bearer_token(
        authorization=authorization,
        actor_id=compat_actor_id,
        actor_tokens=actor_tokens,
    )
    resolved_user_id = _resolve_compat_user_id(user_id)
    policy = _compat_policy_for_actor(compat_actor_id, actor_policies)
    if not policy.allows_user(resolved_user_id):
        raise HTTPException(
            status_code=403,
            detail=(f"Actor '{compat_actor_id}' is not allowed to act for user '{resolved_user_id}'"),
        )

    return MemoryAccessContext(
        actor_id=compat_actor_id,
        auth_mode="foresight_bearer_compat",
        actor_type="compat_cli",
        policy=policy,
        effective_user_id=resolved_user_id,
    )


def _authorize_signed_request(
    *,
    actor_id: str | None,
    user_id: str | None,
    actor_tokens: Mapping[str, str],
    actor_policies: Mapping[str, MemoryActorPolicy],
    request_method: str,
    request_target: str,
    request_body: bytes,
    timestamp: str | None,
    nonce: str | None,
    signature: str | None,
) -> MemoryAccessContext:
    value = (actor_id or "").strip()
    if not value:
        raise HTTPException(status_code=400, detail="Missing X-Memory-Actor-Id header")
    user_scope = required_user_id(user_id)
    expected_secret = _actor_secret_for(value, actor_tokens)
    if not expected_secret:
        raise HTTPException(status_code=403, detail="Unknown memory actor")
    nonce_value = _required_header(nonce, "X-Memory-Nonce")
    signature_value = _required_header(signature, "X-Memory-Signature")
    timestamp_value, timestamp_epoch = _parse_request_timestamp(timestamp)
    now_epoch = time()
    if abs(now_epoch - timestamp_epoch) > _MAX_CLOCK_SKEW_SECONDS:
        raise HTTPException(status_code=401, detail="Signed memory request timestamp is outside the allowed window")
    canonical = _canonical_request(
        actor_id=value,
        user_id=user_scope,
        method=request_method,
        target=request_target,
        body=request_body,
        timestamp=timestamp_value,
        nonce=nonce_value,
    )
    expected_signature = hmac.new(
        expected_secret.encode("utf-8"),
        canonical.encode("utf-8"),
        hashlib.sha256,
    ).hexdigest()
    if not hmac.compare_digest(signature_value, expected_signature):
        raise HTTPException(status_code=403, detail="Invalid memory request signature")
    _REPLAY_PROTECTION.validate(actor_id=value, nonce=nonce_value, now_epoch=now_epoch)
    policy_actor_id = value
    if policy_actor_id not in actor_policies:
        policy_actor_id = _normalize_actor_env_key(value).lower()
    policy = _compat_policy_for_actor(policy_actor_id, actor_policies)
    if not policy.allows_user(user_scope):
        raise HTTPException(
            status_code=403,
            detail=f"Actor '{value}' is not allowed to act for user '{user_scope}'",
        )
    return MemoryAccessContext(actor_id=value, policy=policy, effective_user_id=user_scope)


def authorize_memory_access(
    *,
    actor_id: str | None,
    user_id: str | None,
    request_method: str,
    request_target: str,
    request_body: bytes,
    timestamp: str | None,
    nonce: str | None,
    signature: str | None,
    authorization: str | None = None,
) -> MemoryAccessContext:
    actor_tokens = configured_actor_tokens()
    if not actor_tokens:
        raise HTTPException(status_code=503, detail="Memory actor credentials are not configured")
    actor_policies = configured_actor_policies()
    compat_access = _authorize_bearer_compat(
        authorization=authorization,
        user_id=user_id,
        actor_tokens=actor_tokens,
        actor_policies=actor_policies,
    )
    if compat_access is not None:
        return compat_access
    return _authorize_signed_request(
        actor_id=actor_id,
        user_id=user_id,
        actor_tokens=actor_tokens,
        actor_policies=actor_policies,
        request_method=request_method,
        request_target=request_target,
        request_body=request_body,
        timestamp=timestamp,
        nonce=nonce,
        signature=signature,
    )


def required_user_id(user_id: str | None) -> str:
    value = (user_id or "").strip()
    if not value:
        raise HTTPException(status_code=400, detail="Missing X-Memory-User-Id header")
    return value


def resolve_authorized_user_id(
    access: MemoryAccessContext,
    user_id: str | None,
) -> str:
    resolved_user_id = (user_id or "").strip() or access.effective_user_id or ""
    if not resolved_user_id and access.auth_mode == "foresight_bearer_compat":
        resolved_user_id = _default_compat_user_id()
    resolved_user_id = required_user_id(resolved_user_id)
    return access.assert_user_scope(resolved_user_id)
