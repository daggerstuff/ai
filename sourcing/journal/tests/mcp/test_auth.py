import pytest
import jwt as pyjwt

from ai.sourcing.journal.mcp.auth.authentication import (
    APIKeyAuth,
    JWTAuth,
    CompositeAuth,
)
from ai.sourcing.journal.mcp.auth.authorization import RBAC
from ai.sourcing.journal.mcp.config import AuthConfig
from ai.sourcing.journal.mcp.protocol import MCPRequest, MCPError, MCPErrorCode


def _build_request(auth_header: str) -> MCPRequest:
    return MCPRequest(
        method="tools/call",
        params={"headers": {"Authorization": auth_header}},
    )


@pytest.mark.asyncio
async def test_api_key_auth_accepts_valid_key() -> None:
    config = AuthConfig(enabled=True, api_key_required=True, allowed_api_keys=["secret"])
    auth = APIKeyAuth(config)

    user = await auth.authenticate(_build_request("Bearer secret"))

    assert user["role"] == "admin"
    assert user["permissions"] == ["*"]


@pytest.mark.asyncio
async def test_api_key_auth_rejects_invalid_key() -> None:
    config = AuthConfig(enabled=True, api_key_required=True, allowed_api_keys=["secret"])
    auth = APIKeyAuth(config)

    with pytest.raises(MCPError) as exc:
        await auth.authenticate(_build_request("Bearer nope"))

    assert exc.value.code == MCPErrorCode.AUTHENTICATION_ERROR


@pytest.mark.asyncio
async def test_jwt_auth_valid_token() -> None:
    secret = "top-secret"
    token = pyjwt.encode({"sub": "user-1", "email": "user@example.com"}, secret, algorithm="HS256")

    config = AuthConfig(
        enabled=True,
        api_key_required=False,
        jwt_secret=secret,
        allowed_api_keys=[],
    )
    auth = JWTAuth(config)

    user = await auth.authenticate(_build_request(f"Bearer {token}"))

    assert user["user_id"] == "user-1"
    assert user["email"] == "user@example.com"


@pytest.mark.asyncio
async def test_jwt_auth_missing_subject_raises() -> None:
    secret = "top-secret"
    token = pyjwt.encode({"email": "user@example.com"}, secret, algorithm="HS256")
    config = AuthConfig(enabled=True, api_key_required=False, jwt_secret=secret)
    auth = JWTAuth(config)

    with pytest.raises(MCPError) as exc:
        await auth.authenticate(_build_request(f"Bearer {token}"))

    assert exc.value.code == MCPErrorCode.AUTHENTICATION_ERROR


@pytest.mark.asyncio
async def test_composite_auth_falls_back_to_jwt() -> None:
    secret = "top-secret"
    token = pyjwt.encode({"sub": "user-42"}, secret, algorithm="HS256")
    config = AuthConfig(
        enabled=True,
        api_key_required=True,
        allowed_api_keys=["admin-key"],
        jwt_secret=secret,
    )
    auth = CompositeAuth(config)

    user = await auth.authenticate(_build_request(f"Bearer {token}"))

    assert user["user_id"] == "user-42"


@pytest.mark.asyncio
async def test_rbac_authorize_based_on_role() -> None:
    rbac = RBAC()
    user = {"role": "research_coordinator"}

    allowed = await rbac.authorize(user, "create_session", "execute")
    denied = await rbac.authorize({"role": "viewer"}, "create_session", "execute")

    assert allowed is True
    assert denied is False


@pytest.mark.asyncio
async def test_rbac_require_authorization_raises_for_missing_permission() -> None:
    rbac = RBAC()
    user = {"role": "viewer"}

    with pytest.raises(MCPError) as exc:
        await rbac.require_authorization(user, "create_session", "execute")

    assert exc.value.code == MCPErrorCode.AUTHORIZATION_ERROR

