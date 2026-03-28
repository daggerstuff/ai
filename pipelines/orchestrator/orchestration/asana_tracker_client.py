"""
Asana API client for tracker synchronization flows.
"""

from __future__ import annotations

import json
import logging
import os
import time
import urllib.error
import urllib.parse
import urllib.request
from typing import Any

logger = logging.getLogger(__name__)


class AsanaTrackerClient:
    """Dedicated Asana client for tracker sync and OAuth token management."""

    def __init__(self) -> None:
        self._access_token_cache: str | None = None
        self._access_token_expiry_epoch: float = 0.0

    @staticmethod
    def _format_http_error(exc: urllib.error.HTTPError, message: str) -> str:
        """Summarize Asana HTTP errors without exposing raw provider payloads."""
        status = getattr(exc, "code", "unknown")
        try:
            payload = exc.read().decode("utf-8", errors="ignore")
        except Exception:
            payload = ""

        provider_message = ""
        if payload:
            try:
                parsed = json.loads(payload)
            except json.JSONDecodeError:
                parsed = None

            if isinstance(parsed, dict):
                errors = parsed.get("errors")
                if isinstance(errors, list) and errors:
                    first_error = errors[0]
                    if isinstance(first_error, dict):
                        provider_message = str(first_error.get("message", "")).strip()

        if provider_message:
            return f"{message} (status={status}): {provider_message}"
        return f"{message} (status={status})"

    def get_access_token(self) -> str:
        """Get Asana bearer token from direct access token or OAuth refresh flow."""
        now = time.time()
        if self._access_token_cache and self._access_token_expiry_epoch > now + 30:
            return self._access_token_cache

        direct_token = os.getenv("ASANA_ACCESS_TOKEN", "").strip()
        if direct_token:
            self._access_token_cache = direct_token
            self._access_token_expiry_epoch = now + 300
            return direct_token

        client_id = os.getenv("ASANA_CLIENT_ID", os.getenv("ASANA_CID", "")).strip()
        client_secret = os.getenv(
            "ASANA_CLIENT_SECRET", os.getenv("ASANA_CS", "")
        ).strip()
        refresh_token = os.getenv("ASANA_REFRESH_TOKEN", "").strip()

        if client_id and client_secret and not refresh_token:
            auth_code = os.getenv("ASANA_AUTH_CODE", "").strip()
            redirect_uri = os.getenv("ASANA_REDIRECT_URI", "").strip()
            if auth_code and redirect_uri:
                token_request_body = urllib.parse.urlencode(
                    {
                        "grant_type": "authorization_code",
                        "client_id": client_id,
                        "client_secret": client_secret,
                        "redirect_uri": redirect_uri,
                        "code": auth_code,
                    }
                ).encode("utf-8")

                request = urllib.request.Request(
                    "https://app.asana.com/-/oauth_token",
                    data=token_request_body,
                    headers={"Content-Type": "application/x-www-form-urlencoded"},
                    method="POST",
                )

                try:
                    with urllib.request.urlopen(request, timeout=15) as response:
                        token_response = json.loads(response.read().decode("utf-8"))
                except urllib.error.HTTPError as exc:
                    raise RuntimeError(
                        self._format_http_error(
                            exc, "Asana OAuth auth-code exchange failed"
                        )
                    ) from exc
                except urllib.error.URLError as exc:
                    raise RuntimeError(
                        f"Asana OAuth auth-code exchange failed: {exc}"
                    ) from exc

                if not isinstance(token_response, dict):
                    raise RuntimeError(
                        "Asana OAuth auth-code exchange returned invalid payload"
                    )

                exchanged_access = str(token_response.get("access_token", "")).strip()
                exchanged_refresh = str(token_response.get("refresh_token", "")).strip()
                expires_in_raw = token_response.get("expires_in", 3600)

                if not exchanged_access:
                    raise RuntimeError(
                        "Asana OAuth auth-code exchange returned no access_token"
                    )

                if exchanged_refresh:
                    logger.warning(
                        "Asana OAuth auth-code exchange returned a refresh token. "
                        "Persist it through ASANA_REFRESH_TOKEN in your secret store "
                        "and clear ASANA_AUTH_CODE before the current access token expires."
                    )

                try:
                    expires_in = float(expires_in_raw)
                except (TypeError, ValueError):
                    expires_in = 3600.0

                self._access_token_cache = exchanged_access
                self._access_token_expiry_epoch = now + max(expires_in, 60.0)
                return exchanged_access

        if not (client_id and client_secret and refresh_token):
            raise RuntimeError(
                "Set ASANA_ACCESS_TOKEN or OAuth vars "
                "(ASANA_CLIENT_ID/ASANA_CLIENT_SECRET/ASANA_REFRESH_TOKEN), "
                "or bootstrap with ASANA_AUTH_CODE + ASANA_REDIRECT_URI."
            )

        token_request_body = urllib.parse.urlencode(
            {
                "grant_type": "refresh_token",
                "client_id": client_id,
                "client_secret": client_secret,
                "refresh_token": refresh_token,
            }
        ).encode("utf-8")

        request = urllib.request.Request(
            "https://app.asana.com/-/oauth_token",
            data=token_request_body,
            headers={"Content-Type": "application/x-www-form-urlencoded"},
            method="POST",
        )

        try:
            with urllib.request.urlopen(request, timeout=15) as response:
                token_response = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            raise RuntimeError(
                self._format_http_error(exc, "Asana OAuth token refresh failed")
            ) from exc
        except urllib.error.URLError as exc:
            raise RuntimeError(f"Asana OAuth token refresh failed: {exc}") from exc

        if not isinstance(token_response, dict):
            raise RuntimeError("Asana OAuth token refresh returned invalid payload")

        access_token = str(token_response.get("access_token", "")).strip()
        if not access_token:
            raise RuntimeError("Asana OAuth token refresh returned no access_token")

        expires_in_raw = token_response.get("expires_in", 3600)
        try:
            expires_in = float(expires_in_raw)
        except (TypeError, ValueError):
            expires_in = 3600.0

        self._access_token_cache = access_token
        self._access_token_expiry_epoch = now + max(expires_in, 60.0)
        return access_token

    def request(
        self,
        method: str,
        path: str,
        payload: dict[str, Any] | None = None,
        query_params: dict[str, Any] | None = None,
    ) -> Any:
        """Issue a request to Asana API v1.0 and return parsed response data."""
        token = self.get_access_token()

        url = f"https://app.asana.com/api/1.0{path}"
        if query_params:
            encoded_params = urllib.parse.urlencode(query_params, doseq=True)
            url = f"{url}?{encoded_params}"

        body = None
        if payload is not None:
            body = json.dumps({"data": payload}).encode("utf-8")
        request = urllib.request.Request(
            url,
            data=body,
            headers={
                "Authorization": f"Bearer {token}",
                "Content-Type": "application/json",
            },
            method=method,
        )
        with urllib.request.urlopen(request, timeout=15) as response:
            response_data = json.loads(response.read().decode("utf-8"))
            if not isinstance(response_data, dict) or "data" not in response_data:
                raise RuntimeError("Invalid Asana response payload")
            return response_data["data"]

    @staticmethod
    def has_auth_context() -> bool:
        """Return True when any valid auth path is configured for Asana API use."""
        if os.getenv("ASANA_ACCESS_TOKEN", "").strip():
            return True

        client_id = os.getenv("ASANA_CLIENT_ID", os.getenv("ASANA_CID", "")).strip()
        client_secret = os.getenv(
            "ASANA_CLIENT_SECRET", os.getenv("ASANA_CS", "")
        ).strip()
        refresh_token = os.getenv("ASANA_REFRESH_TOKEN", "").strip()

        return bool(client_id and client_secret and refresh_token)


__all__ = ["AsanaTrackerClient"]
