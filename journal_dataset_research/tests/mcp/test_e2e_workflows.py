import html
import json
from typing import Dict, Tuple

import pytest

from ai.journal_dataset_research.mcp.config import MCPConfig
from ai.journal_dataset_research.mcp.server import MCPServer
from ai.journal_dataset_research.tests.mcp._stubs import (
    AllowAllAuthorization,
    FakeCommandHandlerService,
)


def _parse_content(response: str) -> Dict:
    payload = json.loads(response)
    result = payload["result"]
    if "content" in result:
        content_value = result["content"]
        if isinstance(content_value, list) and content_value:
            text = content_value[0]["text"]
            return json.loads(html.unescape(text))
    return result


@pytest.fixture
def e2e_server(
    monkeypatch,
    sample_research_session,
    sample_dataset_source,
    sample_evaluation,
    sample_acquired_dataset,
    sample_integration_plan,
) -> Tuple[MCPServer, FakeCommandHandlerService]:
    report = {
        "report_id": f"report_{sample_research_session.session_id}",
        "session_id": sample_research_session.session_id,
        "report_type": "session_report",
        "format": "json",
        "generated_date": "2025-01-01T00:00:00Z",
        "content": {"summary": "Initial report"},
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

    from ai.journal_dataset_research.mcp import server as server_module

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
    server.current_user = {"user_id": "agent", "role": "admin", "permissions": ["*"]}
    server.audit_logger.enabled = False
    discover_tool = server.tools.get("discover_sources")
    if discover_tool:
        discover_tool.validate_parameters = lambda params: None  # type: ignore[assignment]
    return server, fake_service


@pytest.mark.asyncio
async def test_agent_workflow_end_to_end(
    e2e_server,
) -> None:
    server, fake_service = e2e_server
    session_id = fake_service.session.session_id

    # Create session
    response = await server.handle_request(
        json.dumps(
            {
                "jsonrpc": "2.0",
                "id": "create",
                "method": "tools/call",
                "params": {
                    "name": "create_session",
                    "arguments": {
                        "target_sources": ["pubmed", "zenodo"],
                        "search_keywords": {"therapy": ["cbt", "dbt"]},
                        "weekly_targets": {"sources_identified": 5},
                        "session_id": session_id,
                    },
                },
            }
        )
    )
    create_payload = _parse_content(response)
    assert create_payload["session_id"] == session_id
    assert "target_sources" in create_payload

    # Discover sources
    response = await server.handle_request(
        json.dumps(
            {
                "jsonrpc": "2.0",
                "id": "discover",
                "method": "tools/call",
                "params": {
                        "name": "discover_sources",
                        "arguments": {
                            "session_id": session_id,
                            "keywords": ["therapy"],
                            "sources": ["pubmed"],
                        },
                },
            }
        )
    )
    discover_payload = _parse_content(response)
    assert discover_payload["total_sources"] >= 1

    # Evaluate sources
    response = await server.handle_request(
        json.dumps(
            {
                "jsonrpc": "2.0",
                "id": "evaluate",
                "method": "tools/call",
                "params": {
                    "name": "evaluate_sources",
                    "arguments": {
                        "session_id": session_id,
                    },
                },
            }
        )
    )
    evaluation_payload = _parse_content(response)
    assert evaluation_payload["total_evaluated"] >= 1

    # Acquire datasets
    response = await server.handle_request(
        json.dumps(
            {
                "jsonrpc": "2.0",
                "id": "acquire",
                "method": "tools/call",
                "params": {
                    "name": "acquire_datasets",
                    "arguments": {
                        "session_id": session_id,
                    },
                },
            }
        )
    )
    acquisition_payload = _parse_content(response)
    assert acquisition_payload["total_acquired"] >= 1

    # Create integration plans
    response = await server.handle_request(
        json.dumps(
            {
                "jsonrpc": "2.0",
                "id": "integrate",
                "method": "tools/call",
                "params": {
                    "name": "create_integration_plans",
                    "arguments": {
                        "session_id": session_id,
                        "target_format": "chatml",
                    },
                },
            }
        )
    )
    integration_payload = _parse_content(response)
    assert integration_payload["total_plans"] >= 1

    # Generate report
    response = await server.handle_request(
        json.dumps(
            {
                "jsonrpc": "2.0",
                "id": "report",
                "method": "tools/call",
                "params": {
                    "name": "generate_report",
                    "arguments": {
                        "session_id": session_id,
                        "report_type": "summary_report",
                        "format": "json",
                    },
                },
            }
        )
    )
    report_payload = _parse_content(response)
    assert report_payload["report_type"] == "summary_report"

    # Read progress metrics resource to confirm state updates
    progress_response = await server.handle_request(
        json.dumps(
            {
                "jsonrpc": "2.0",
                "id": "progress",
                "method": "resources/read",
                "params": {
                    "uri": f"research://progress/metrics/{session_id}",
                    "arguments": {"session_id": session_id},
                },
            }
        )
    )
    progress_data = json.loads(progress_response)
    progress_payload = json.loads(
        html.unescape(progress_data["result"]["contents"][0]["text"])
    )
    assert progress_payload["sources_identified"] >= 1
    assert progress_payload["integration_plans_created"] >= 1

    # Unknown method yields error (agent handles gracefully)
    error_response = await server.handle_request(
        json.dumps(
            {
                "jsonrpc": "2.0",
                "id": "unknown",
                "method": "tools/call",
                "params": {"name": "nonexistent_tool", "arguments": {}},
            }
        )
    )
    payload = json.loads(error_response)
    assert payload["error"]["code"] == -32000  # Tool execution error

