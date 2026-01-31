import json

import pytest

from ai.sourcing.journal.mcp.protocol import (
    MCPProtocolHandler,
    MCPRequest,
    MCPResponse,
    MCPError,
    JSONRPCErrorCode,
)


def test_parse_request_from_json_string() -> None:
    request_json = json.dumps({"jsonrpc": "2.0", "method": "tools/list", "id": 1})

    request = MCPProtocolHandler.parse_request(request_json)

    assert isinstance(request, MCPRequest)
    assert request.method == "tools/list"
    assert request.id == 1


def test_parse_request_invalid_json_raises() -> None:
    with pytest.raises(MCPError) as exc:
        MCPProtocolHandler.parse_request("{invalid json")

    assert exc.value.code == JSONRPCErrorCode.PARSE_ERROR


def test_format_response_success() -> None:
    response = MCPResponse.success({"ok": True}, id=1)

    payload = MCPProtocolHandler.format_response(response)
    data = json.loads(payload)

    assert data["result"]["ok"] is True
    assert data["id"] == 1


def test_format_error_generates_expected_structure() -> None:
    payload = MCPProtocolHandler.format_error(
        JSONRPCErrorCode.METHOD_NOT_FOUND,
        "Nope",
        data={"method": "unknown"},
        request_id="abc",
    )
    data = json.loads(payload)

    assert data["error"]["code"] == JSONRPCErrorCode.METHOD_NOT_FOUND
    assert data["error"]["data"]["method"] == "unknown"
    assert data["id"] == "abc"

