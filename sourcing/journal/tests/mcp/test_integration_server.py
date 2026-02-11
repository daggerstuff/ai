import html
import json
from datetime import datetime
from typing import Any, Dict, Tuple

import pytest

from ai.sourcing.journal.mcp.config import MCPConfig
from ai.sourcing.journal.mcp.server import MCPServer
from ai.sourcing.journal.mcp.utils.progress_streaming import (
    ProgressStatus,
    ProgressUpdate,
)
from ai.sourcing.journal.tests.mcp._stubs import (
    AllowAllAuthorization,
    FakeCommandHandlerService,
)


@pytest.fixture
def integration_server(
    monkeypatch,
    sample_research_session,
    sample_dataset_source,
    sample_evaluation,
    sample_acquired_dataset,
    sample_integration_plan,
) -> Tuple[MCPServer, FakeCommandHandlerService]:
    """Provide an MCPServer instance wired with the fake command handler service."""
    report = {
        "report_id": f"report_{sample_research_session.session_id}",
        "session_id": sample_research_session.session_id,
        "report_type": "summary_report",
        "format": "json",
        "generated_date": datetime.utcnow().isoformat(),
        "content": {"summary": "Test summary"},
        "file_path": None,
    }
    fake_service = FakeCommandHandlerService(
        session=sample_research_session,
        sources=[sample_dataset_source],
        evaluations=[sample_evaluation],
        acquisitions=[sample_acquired_dataset],
        integration_plans=[sample_integration_plan],
        report=report,
    )

    from ai.sourcing.journal.mcp import server as server_module

    monkeypatch.setattr(
        server_module,
        "CommandHandlerService",
        lambda *args, **kwargs: fake_service,
    )

    config = MCPConfig()
    config.auth.enabled = False
    config.auth.api_key_required = False
    config.rate_limits.enabled = False
    config.logging.enable_audit_logging = False

    server = MCPServer(config)
    server.authorization_handler = AllowAllAuthorization()
    server.current_user = {"user_id": "tester", "role": "admin", "permissions": ["*"]}
    server.audit_logger.enabled = False
    return server, fake_service


def _decode_response(payload: str) -> Dict[str, Any]:
    """Parse JSON-RPC response string."""
    return json.loads(payload)


def _load_text_content(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Extract and decode JSON payload stored as text content."""
    text = payload["result"]["content"][0]["text"]
    return json.loads(html.unescape(text))


def _load_resource_content(payload: Dict[str, Any]) -> Dict[str, Any]:
    text = payload["result"]["contents"][0]["text"]
    return json.loads(html.unescape(text))


@pytest.mark.asyncio
async def test_initialize_returns_capabilities(integration_server):
    server, _ = integration_server

    response = await server.handle_request(
        json.dumps(
            {
                "jsonrpc": "2.0",
                "id": "init-1",
                "method": "initialize",
            }
        )
    )
    data = _decode_response(response)

    assert data["result"]["protocolVersion"] == server.config.protocol_version
    assert "tools" in data["result"]["capabilities"]


@pytest.mark.asyncio
async def test_list_sessions_via_tool_call(integration_server):
    server, fake_service = integration_server

    response = await server.handle_request(
        json.dumps(
            {
                "jsonrpc": "2.0",
                "id": "tool-1",
                "method": "tools/call",
                "params": {"name": "list_sessions", "arguments": {}},
            }
        )
    )
    payload = _decode_response(response)
    content = _load_text_content(payload)

    assert content["sessions"][0]["session_id"] == fake_service.session.session_id
    assert content["count"] == 1


@pytest.mark.asyncio
async def test_resources_read_progress_metrics(integration_server):
    server, fake_service = integration_server

    response = await server.handle_request(
        json.dumps(
            {
                "jsonrpc": "2.0",
                "id": "resource-1",
                "method": "resources/read",
                "params": {
                    "uri": f"research://progress/metrics/{fake_service.session.session_id}",
                    "arguments": {"session_id": fake_service.session.session_id},
                },
            }
        )
    )
    payload = _decode_response(response)
    content = _load_resource_content(payload)

    assert content["sources_identified"] >= 1
    assert content["integration_plans_created"] >= 1


@pytest.mark.asyncio
async def test_progress_streaming_broadcast(integration_server):
    server, fake_service = integration_server
    updates = []

    async def listener(update: ProgressUpdate) -> None:
        updates.append(update.to_dict())

    operation_id = "op-test"
    session_id = fake_service.session.session_id
    await server.progress_streamer.subscribe(operation_id, listener, session_id=session_id)

    await server.progress_streamer.broadcast_update(
        ProgressUpdate(
            operation_id=operation_id,
            session_id=session_id,
            status=ProgressStatus.RUNNING,
            progress_percent=50.0,
            message="Half way",
            timestamp=datetime.utcnow(),
            metadata={"step": "testing"},
        )
    )

    assert updates
    assert updates[0]["operation_id"] == operation_id
    assert updates[0]["status"] == ProgressStatus.RUNNING.value


@pytest.mark.asyncio
async def test_tool_call_missing_name_returns_error(integration_server):
    server, _ = integration_server

    response = await server.handle_request(
        json.dumps(
            {
                "jsonrpc": "2.0",
                "id": "tool-error",
                "method": "tools/call",
                "params": {"arguments": {}},
            }
        )
    )
    payload = _decode_response(response)

    assert payload["error"]["code"] == -32602  # INVALID_PARAMS

