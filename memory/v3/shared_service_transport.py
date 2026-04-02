"""HTTP transport for the shared local memory service."""

from __future__ import annotations

import hashlib
import hmac
import json
import time
import uuid
from typing import Any, Optional
from urllib.parse import urlencode

import httpx

from ai.api.mcp_server.memory_auth import _canonical_request


class SharedMemoryServiceError(RuntimeError):
    """Raised when the shared memory service returns a non-success response."""

    def __init__(self, *, status_code: int, message: str) -> None:
        super().__init__(message)
        self.status_code = status_code


class SharedMemoryServiceTransport:
    """Owns signing and HTTP transport for the shared memory service."""

    def __init__(
        self,
        *,
        base_url: str,
        actor_id: str,
        actor_secret: str,
        timeout_ms: int = 5000,
        client: Optional[httpx.AsyncClient] = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.actor_id = actor_id.strip()
        self.actor_secret = actor_secret.strip()
        self.timeout_ms = timeout_ms
        self._client = client
        self._owns_client = client is None

    async def request_json(
        self,
        *,
        method: str,
        path: str,
        user_id: str,
        json_body: Optional[dict[str, Any]] = None,
        params: Optional[dict[str, Any]] = None,
        expected_status: int = 200,
    ) -> Any:
        client = await self._get_client()
        request_path = path if path.startswith("/") else f"/{path}"
        encoded_body = (
            json.dumps(json_body, separators=(",", ":")).encode("utf-8")
            if json_body is not None
            else b""
        )
        target = request_path
        if params:
            target = f"{request_path}?{urlencode(params, doseq=True)}"
        response = await client.request(
            method=method,
            url=request_path,
            params=params,
            content=encoded_body or None,
            headers=self._signed_headers(
                method=method,
                target=target,
                user_id=user_id,
                body=encoded_body,
            ),
        )
        if response.status_code != expected_status:
            try:
                error_payload = response.json()
            except ValueError:
                error_payload = response.text
            raise SharedMemoryServiceError(
                status_code=response.status_code,
                message=(
                    f"Shared memory service request failed "
                    f"({response.status_code} {method} {path}): {error_payload}"
                ),
            )
        if expected_status == 204:
            return None
        return response.json()

    async def health_check(self) -> bool:
        client = await self._get_client()
        response = await client.get("/health")
        if response.status_code != 200:
            return False
        payload = response.json()
        return payload.get("status") in {"healthy", "degraded"}

    async def close(self) -> None:
        if self._client is not None and self._owns_client:
            await self._client.aclose()
        self._client = None

    async def _get_client(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(
                base_url=self.base_url,
                timeout=self.timeout_ms / 1000,
            )
        return self._client

    def _signed_headers(
        self,
        *,
        method: str,
        target: str,
        user_id: str,
        body: bytes,
    ) -> dict[str, str]:
        timestamp = str(int(time.time()))
        nonce = uuid.uuid4().hex
        signature = hmac.new(
            self.actor_secret.encode("utf-8"),
            _canonical_request(
                actor_id=self.actor_id,
                user_id=user_id,
                method=method,
                target=target,
                body=body,
                timestamp=timestamp,
                nonce=nonce,
            ).encode("utf-8"),
            hashlib.sha256,
        ).hexdigest()
        return {
            "Content-Type": "application/json",
            "X-Memory-Actor-Id": self.actor_id,
            "X-Memory-User-Id": user_id,
            "X-Memory-Timestamp": timestamp,
            "X-Memory-Nonce": nonce,
            "X-Memory-Signature": signature,
        }
