from __future__ import annotations

import json
import os
from functools import lru_cache
from dataclasses import dataclass
from typing import Dict, Mapping, Optional

from fastapi import HTTPException


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
    auth_mode: str = "internal_service"
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
    raw_json = os.environ.get("LOCAL_MEMORY_ACTOR_POLICIES_JSON")
    if not raw_json:
        tokens = configured_actor_tokens()
        return {actor_id: MemoryActorPolicy(allow_any_user=True) for actor_id in tokens}
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
    return policies


@lru_cache(maxsize=1)
def readiness_details() -> Dict[str, object]:
    actor_tokens = configured_actor_tokens()
    actor_policies = configured_actor_policies()
    return {
        "auth_configured": bool(actor_tokens),
        "configured_actors": sorted(actor_tokens.keys()),
        "db_path_configured": bool(os.environ.get("HINDSIGHT_LOCAL_DB_PATH")),
        "actor_policy_mode": "scoped" if actor_policies else "none",
        "actor_policies_configured": bool(actor_policies),
    }


@lru_cache(maxsize=1)
def configured_actor_tokens() -> Dict[str, str]:
    tokens = _actor_tokens_from_json(os.environ.get("LOCAL_MEMORY_ACTOR_TOKENS_JSON"))
    tokens.update(_actor_tokens_from_prefix_env())
    return tokens


def authorize_memory_access(
    *,
    authorization: Optional[str],
    actor_id: Optional[str],
) -> MemoryAccessContext:
    actor_tokens = configured_actor_tokens()
    if not actor_tokens:
        raise HTTPException(status_code=503, detail="Memory actor credentials are not configured")
    value = (actor_id or "").strip()
    if not value:
        raise HTTPException(status_code=400, detail="Missing X-Memory-Actor-Id header")
    expected_token = actor_tokens.get(value)
    if not expected_token:
        normalized_actor = _normalize_actor_env_key(value).lower()
        expected_token = actor_tokens.get(normalized_actor)
    if not expected_token:
        raise HTTPException(status_code=403, detail="Unknown memory actor")
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing bearer token")
    provided = authorization.split(" ", 1)[1].strip()
    if provided != expected_token:
        raise HTTPException(status_code=403, detail="Invalid bearer token")
    policy = configured_actor_policies().get(value)
    if policy is None:
        normalized_actor = _normalize_actor_env_key(value).lower()
        policy = configured_actor_policies().get(normalized_actor)
    if policy is None:
        policy = MemoryActorPolicy(allow_any_user=True)
    return MemoryAccessContext(actor_id=value, policy=policy)


def required_user_id(user_id: Optional[str]) -> str:
    value = (user_id or "").strip()
    if not value:
        raise HTTPException(status_code=400, detail="Missing X-Memory-User-Id header")
    return value
