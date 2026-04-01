from __future__ import annotations

import hashlib
import hmac
import json
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from functools import lru_cache
from threading import Lock
from time import time
from typing import Dict, Mapping, Optional

from fastapi import HTTPException


_NONCE_TTL_SECONDS = 300
_MAX_CLOCK_SKEW_SECONDS = 300


class ReplayProtectionStore:
    def __init__(self) -> None:
        self._lock = Lock()
        self._nonces: Dict[str, float] = {}

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

    def summary(self) -> Dict[str, object]:
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

    def audit_metadata(self) -> Dict[str, str]:
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


def _actor_secret_for(actor_id: str, actor_tokens: Mapping[str, str]) -> Optional[str]:
    secret = actor_tokens.get(actor_id)
    if secret:
        return secret
    normalized_actor = _normalize_actor_env_key(actor_id).lower()
    return actor_tokens.get(normalized_actor)


def _parse_request_timestamp(raw_timestamp: Optional[str]) -> tuple[str, float]:
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
            parsed = parsed.replace(tzinfo=timezone.utc)
        return value, parsed.timestamp()


def _required_header(value: Optional[str], header_name: str) -> str:
    normalized = (value or "").strip()
    if not normalized:
        raise HTTPException(status_code=400, detail=f"Missing {header_name} header")
    return normalized


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


def _actor_tokens_from_json(raw_json: Optional[str]) -> Dict[str, str]:
    tokens: Dict[str, str] = {}
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


def _actor_tokens_from_prefix_env() -> Dict[str, str]:
    tokens: Dict[str, str] = {}
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
        return MemoryActorPolicy(
            allowed_users=frozenset(str(item).strip() for item in raw_policy if str(item).strip())
        )
    if not isinstance(raw_policy, Mapping):
        raise RuntimeError(
            f"Policy for actor '{actor_id}' must be an object, list, boolean, or null"
        )
    allowed_users = frozenset(
        str(item).strip()
        for item in raw_policy.get("allowed_users", [])
        if str(item).strip()
    )
    allowed_prefixes = tuple(
        str(item).strip()
        for item in raw_policy.get("allowed_user_prefixes", [])
        if str(item).strip()
    )
    allow_any_user = bool(raw_policy.get("allow_any_user", False))
    if not (allow_any_user or allowed_users or allowed_prefixes):
        raise RuntimeError(
            f"Policy for actor '{actor_id}' must allow at least one user scope"
        )
    return MemoryActorPolicy(
        allow_any_user=allow_any_user,
        allowed_users=allowed_users,
        allowed_user_prefixes=allowed_prefixes,
    )


@lru_cache(maxsize=1)
def configured_actor_policies() -> Dict[str, MemoryActorPolicy]:
    tokens = configured_actor_tokens()
    raw_json = os.environ.get("LOCAL_MEMORY_ACTOR_POLICIES_JSON")
    if not raw_json:
        if not tokens:
            return {}
        raise RuntimeError(
            "LOCAL_MEMORY_ACTOR_POLICIES_JSON must be configured for every shared-memory actor"
        )
    try:
        payload = json.loads(raw_json)
    except json.JSONDecodeError as exc:
        raise RuntimeError("LOCAL_MEMORY_ACTOR_POLICIES_JSON must be valid JSON") from exc
    if not isinstance(payload, Mapping):
        raise RuntimeError("LOCAL_MEMORY_ACTOR_POLICIES_JSON must be a JSON object")
    policies: Dict[str, MemoryActorPolicy] = {}
    for actor_id, raw_policy in payload.items():
        key = str(actor_id).strip()
        if not key:
            continue
        policies[key] = _parse_actor_policy(key, raw_policy)
    missing_actors = sorted(
        actor_id for actor_id in tokens if actor_id not in policies
    )
    if missing_actors:
        raise RuntimeError(
            "LOCAL_MEMORY_ACTOR_POLICIES_JSON is missing policies for actors: "
            + ", ".join(missing_actors)
        )
    return policies


@lru_cache(maxsize=1)
def readiness_details() -> Dict[str, object]:
    actor_tokens = configured_actor_tokens()
    policy_error: Optional[str] = None
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
        "configuration_error": policy_error,
        "signature_required": True,
        "nonce_ttl_seconds": _NONCE_TTL_SECONDS,
        "max_clock_skew_seconds": _MAX_CLOCK_SKEW_SECONDS,
    }


@lru_cache(maxsize=1)
def configured_actor_tokens() -> Dict[str, str]:
    tokens = _actor_tokens_from_json(os.environ.get("LOCAL_MEMORY_ACTOR_TOKENS_JSON"))
    tokens.update(_actor_tokens_from_prefix_env())
    return tokens


def validate_memory_auth_configuration() -> None:
    """Fail closed when shared-service auth configuration is incomplete."""
    configured_actor_policies()


def authorize_memory_access(
    *,
    actor_id: Optional[str],
    user_id: Optional[str],
    request_method: str,
    request_target: str,
    request_body: bytes,
    timestamp: Optional[str],
    nonce: Optional[str],
    signature: Optional[str],
) -> MemoryAccessContext:
    actor_tokens = configured_actor_tokens()
    if not actor_tokens:
        raise HTTPException(status_code=503, detail="Memory actor credentials are not configured")
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
    actor_policies = configured_actor_policies()
    policy = actor_policies.get(value)
    if policy is None:
        normalized_actor = _normalize_actor_env_key(value).lower()
        policy = actor_policies.get(normalized_actor)
    if policy is None:
        raise HTTPException(
            status_code=503,
            detail=f"Missing policy configuration for actor '{value}'",
        )
    if not policy.allows_user(user_scope):
        raise HTTPException(
            status_code=403,
            detail=f"Actor '{value}' is not allowed to act for user '{user_scope}'",
        )
    return MemoryAccessContext(actor_id=value, policy=policy)


def required_user_id(user_id: Optional[str]) -> str:
    value = (user_id or "").strip()
    if not value:
        raise HTTPException(status_code=400, detail="Missing X-Memory-User-Id header")
    return value
