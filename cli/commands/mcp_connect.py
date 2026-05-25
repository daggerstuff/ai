"""
MCP Connect Commands for Pixelated AI CLI

This module provides commands for connecting to and managing MCP (Model Context Protocol) agents,
including agent discovery, connection management, and agent-based operations.
"""

import json
import time
from typing import Any

import click

from cli.auth import AuthManager
from cli.config import get_config
from cli.utils import get_logger, setup_logging

logger = get_logger(__name__)


@click.group(name="mcp-connect")
@click.pass_context
def mcp_connect_group(ctx):
    """MCP agent connection and management commands."""
    setup_logging(ctx.obj.get("verbose", False))
    logger.info("MCP connect command group initialized")


@mcp_connect_group.command()
@click.option("--host", default="localhost", help="MCP server host")
@click.option("--port", default=8080, help="MCP server port")
@click.option("--timeout", default=30, help="Connection timeout in seconds")
@click.option("--retry", default=3, help="Number of retry attempts")
@click.pass_context
def connect(_ctx, host: str, port: int, timeout: int, retry: int):
    """Connect to an MCP agent server."""
    try:
        config_obj = get_config()
        auth_manager = AuthManager(config_obj)

        click.echo(f"🔗 Connecting to MCP agent at {host}:{port}...")

        # Validate authentication
        if not auth_manager.is_authenticated():
            click.echo("❌ Authentication required. Please login first.", err=True)
            return

        # Attempt connection with retries
        for attempt in range(retry):
            try:
                click.echo(f"🔄 Connection attempt {attempt + 1}/{retry}...")

                connection_result = _connect_to_mcp_agent(auth_manager, host, port, timeout)

                if connection_result["success"]:
                    click.echo("✅ Successfully connected to MCP agent!")
                    click.echo(f"🆔 Agent ID: {connection_result['agent_id']}")
                    click.echo(f"🤖 Agent name: {connection_result['agent_name']}")
                    click.echo(f"📋 Available tools: {len(connection_result.get('tools', []))}")

                    # Store connection info
                    _save_connection_info(host, port, connection_result["agent_id"])

                    # List available tools
                    if connection_result.get("tools"):
                        click.echo("\n🔧 Available tools:")
                        for tool in connection_result["tools"][:5]:  # Show first 5
                            click.echo(f"  • {tool['name']}: {tool.get('description', 'No description')}")

                        if len(connection_result["tools"]) > 5:
                            click.echo(f"  ... and {len(connection_result['tools']) - 5} more")

                    return
                click.echo(f"⚠️  Connection attempt failed: {connection_result.get('error', 'Unknown error')}")

            except Exception as e:
                logger.warning(f"Connection attempt {attempt + 1} failed: {e}")
                if attempt < retry - 1:
                    click.echo("⏳ Retrying in 2 seconds...")
                    time.sleep(2)
                else:
                    click.echo("❌ All connection attempts failed", err=True)
                    raise click.Abort() from e

    except Exception as e:
        logger.error(f"MCP connection failed: {e}")
        click.echo(f"❌ MCP connection failed: {e}", err=True)
        raise click.Abort() from e


@mcp_connect_group.command()
@click.option(
    "--status",
    type=click.Choice(["active", "inactive", "all"]),
    default="all",
    help="Filter by connection status",
)
@click.option("--detailed", is_flag=True, help="Show detailed connection information")
@click.pass_context
def list_agents(_ctx, status: str, detailed: bool):
    """List available MCP agents and their status."""
    try:
        config_obj = get_config()
        auth_manager = AuthManager(config_obj)

        click.echo("📋 MCP Agents Status:")

        agents = _get_mcp_agents(auth_manager, status)

        if not agents:
            click.echo("❌ No MCP agents found")
            return

        for agent in agents:
            status_icon = "🟢" if agent["status"] == "active" else "🔴" if agent["status"] == "inactive" else "🟡"

            click.echo(f"{status_icon} {agent['name']} ({agent['id']})")
            click.echo(f"   Host: {agent['host']}:{agent['port']}")
            click.echo(f"   Status: {agent['status']}")
            click.echo(f"   Tools: {agent.get('tool_count', 0)}")

            if detailed:
                click.echo(f"   Version: {agent.get('version', 'unknown')}")
                click.echo(f"   Last seen: {agent.get('last_seen', 'never')}")
                click.echo(f"   Capabilities: {', '.join(agent.get('capabilities', []))}")

            click.echo()

        # Summary
        active_count = sum(1 for agent in agents if agent["status"] == "active")
        click.echo(f"📊 Summary: {active_count}/{len(agents)} agents active")

    except Exception as e:
        logger.error(f"Failed to list MCP agents: {e}")
        click.echo(f"❌ Failed to list MCP agents: {e}", err=True)
        raise click.Abort() from e


@mcp_connect_group.command()
@click.option("--agent-id", required=True, help="MCP agent ID to disconnect")
@click.option("--force", is_flag=True, help="Force disconnect without confirmation")
@click.pass_context
def disconnect(_ctx, agent_id: str, force: bool):
    """Disconnect from an MCP agent."""
    try:
        config_obj = get_config()
        auth_manager = AuthManager(config_obj)

        # Get agent info for confirmation
        agent_info = _get_agent_info(auth_manager, agent_id)

        if not agent_info:
            click.echo(f"❌ Agent {agent_id} not found", err=True)
            return

        if not force:
            # Ask for confirmation
            click.echo("⚠️  You are about to disconnect from agent:")
            click.echo(f"   Name: {agent_info['name']}")
            click.echo(f"   Host: {agent_info['host']}:{agent_info['port']}")

            if not click.confirm("Do you want to continue?"):
                click.echo("❌ Disconnect cancelled")
                return

        # Perform disconnect
        result = _disconnect_from_agent(auth_manager, agent_id)

        if result["success"]:
            click.echo(f"✅ Successfully disconnected from agent {agent_id}")
            _remove_connection_info(agent_id)
        else:
            click.echo(
                f"❌ Failed to disconnect: {result.get('error', 'Unknown error')}",
                err=True,
            )

    except Exception as e:
        logger.error(f"Disconnect failed: {e}")
        click.echo(f"❌ Disconnect failed: {e}", err=True)
        raise click.Abort() from e


@mcp_connect_group.command()
@click.option("--agent-id", required=True, help="MCP agent ID")
@click.option("--tool", required=True, help="Tool name to execute")
@click.option("--params", help="Tool parameters as JSON string")
@click.option("--params-file", type=click.Path(exists=True), help="Tool parameters from JSON file")
@click.option("--async", "async_mode", is_flag=True, help="Execute asynchronously")
@click.pass_context
def execute(
    _ctx,
    agent_id: str,
    tool: str,
    params: str | None,
    params_file: str | None,
    async_mode: bool,
):
    """Execute a tool on a connected MCP agent."""
    try:
        config_obj = get_config()
        auth_manager = AuthManager(config_obj)

        # Validate that we have parameters
        if not params and not params_file:
            click.echo("❌ Either --params or --params-file must be provided", err=True)
            return

        # Load parameters
        tool_params = {}
        if params:
            try:
                tool_params = json.loads(params)
            except json.JSONDecodeError as e:
                click.echo(f"❌ Invalid JSON in --params: {e}", err=True)
                return

        if params_file:
            try:
                with open(params_file) as f:
                    file_params = json.load(f)
                    tool_params.update(file_params)
            except (json.JSONDecodeError, FileNotFoundError) as e:
                click.echo(f"❌ Error loading params file: {e}", err=True)
                return

        click.echo(f"🔧 Executing tool '{tool}' on agent {agent_id}...")

        if async_mode:
            # Async execution
            execution_id = _execute_tool_async(auth_manager, agent_id, tool, tool_params)
            click.echo("🚀 Async execution started!")
            click.echo(f"🆔 Execution ID: {execution_id}")
            click.echo(f"📊 Monitor with: pixelated mcp-connect status --execution-id {execution_id}")
        else:
            # Sync execution
            click.echo("⏳ Executing tool (this may take a while)...")

            result = _execute_tool_sync(auth_manager, agent_id, tool, tool_params)

            if result["success"]:
                click.echo("✅ Tool execution completed successfully!")

                # Display results
                if "result" in result:
                    click.echo("📊 Results:")
                    if isinstance(result["result"], dict):
                        click.echo(json.dumps(result["result"], indent=2))
                    else:
                        click.echo(str(result["result"]))

                if "metadata" in result:
                    click.echo("📋 Metadata:")
                    click.echo(json.dumps(result["metadata"], indent=2))

            else:
                click.echo(
                    f"❌ Tool execution failed: {result.get('error', 'Unknown error')}",
                    err=True,
                )

    except Exception as e:
        logger.error(f"Tool execution failed: {e}")
        click.echo(f"❌ Tool execution failed: {e}", err=True)
        raise click.Abort() from e


@mcp_connect_group.command()
@click.option("--execution-id", help="Specific execution to check")
@click.option("--agent-id", help="Filter by agent ID")
@click.option("--limit", default=10, help="Maximum number of results")
@click.pass_context
def status(_ctx, execution_id: str | None, agent_id: str | None, limit: int):
    """Check execution status of MCP agent tools."""
    try:
        config_obj = get_config()
        auth_manager = AuthManager(config_obj)

        if execution_id:
            # Get specific execution status
            execution_info = _get_execution_status(auth_manager, execution_id)
            if execution_info:
                _display_execution_details(execution_info)
            else:
                click.echo(f"❌ Execution {execution_id} not found", err=True)

        else:
            # List recent executions
            executions = _list_executions(auth_manager, agent_id, limit)
            _display_executions_list(executions)

    except Exception as e:
        logger.error(f"Status check failed: {e}")
        click.echo(f"❌ Status check failed: {e}", err=True)
        raise click.Abort() from e


@mcp_connect_group.command()
@click.option("--agent-id", required=True, help="MCP agent ID")
@click.option("--capability", required=True, help="Capability to test")
@click.pass_context
def test(_ctx, agent_id: str, capability: str):
    """Test MCP agent capabilities."""
    try:
        config_obj = get_config()
        auth_manager = AuthManager(config_obj)

        click.echo(f"🧪 Testing capability '{capability}' on agent {agent_id}...")

        test_result = _test_agent_capability(auth_manager, agent_id, capability)

        if test_result["success"]:
            click.echo("✅ Capability test passed!")

            if "details" in test_result:
                click.echo("📊 Test details:")
                click.echo(json.dumps(test_result["details"], indent=2))

        else:
            click.echo(
                f"❌ Capability test failed: {test_result.get('error', 'Unknown error')}",
                err=True,
            )

            if "suggestions" in test_result:
                click.echo("💡 Suggestions:")
                for suggestion in test_result["suggestions"]:
                    click.echo(f"  • {suggestion}")

    except Exception as e:
        logger.error(f"Capability test failed: {e}")
        click.echo(f"❌ Capability test failed: {e}", err=True)
        raise click.Abort() from e


@mcp_connect_group.command()
@click.pass_context
def interactive(_ctx):
    """Start interactive MCP agent session."""
    try:
        config_obj = get_config()
        auth_manager = AuthManager(config_obj)

        click.echo("🎯 Starting interactive MCP agent session...")
        click.echo("Type 'help' for available commands, 'exit' to quit")
        click.echo("-" * 50)

        while True:
            try:
                user_input = click.prompt("mcp>", type=str)

                if user_input.lower() in ["exit", "quit"]:
                    click.echo("👋 Interactive session ended")
                    break

                if user_input.lower() == "help":
                    _display_interactive_help()
                    continue

                # Parse command
                parts = user_input.split()
                if not parts:
                    continue

                command = parts[0]

                if command == "agents":
                    agents = _get_mcp_agents(auth_manager, "all")
                    _display_agents_summary(agents)

                elif command == "connect":
                    if len(parts) < 3:
                        click.echo("❌ Usage: connect <host> <port>")
                        continue
                    host, port = parts[1], int(parts[2])
                    # Handle connect command
                    click.echo(f"Connecting to {host}:{port}...")

                elif command == "execute":
                    if len(parts) < 3:
                        click.echo("❌ Usage: execute <agent-id> <tool>")
                        continue
                    agent_id, tool = parts[1], parts[2]
                    # Handle execute command
                    click.echo(f"Executing {tool} on {agent_id}...")

                else:
                    click.echo(f"❌ Unknown command: {command}")
                    click.echo("Type 'help' for available commands")

            except KeyboardInterrupt:
                click.echo("\n👋 Interactive session ended")
                break
            except EOFError:
                click.echo("\n👋 Interactive session ended")
                break

    except Exception as e:
        logger.error(f"Interactive session failed: {e}")
        click.echo(f"❌ Interactive session failed: {e}", err=True)
        raise click.Abort() from e


# Helper functions


def _connect_to_mcp_agent(_auth_manager: AuthManager, host: str, port: int, _timeout: int) -> dict[str, Any]:
    """Connect to MCP agent server."""
    logger.info(f"Connecting to MCP agent at {host}:{port}")

    # This would call the actual API endpoint
    # For now, return mock connection result
    return {
        "success": True,
        "agent_id": f"agent_{host}_{port}",
        "agent_name": f"MCP Agent {host}",
        "tools": [
            {"name": "bias_detection", "description": "Detect bias in text"},
            {"name": "sentiment_analysis", "description": "Analyze sentiment"},
            {"name": "text_summarization", "description": "Summarize text"},
        ],
        "capabilities": ["text_processing", "bias_detection", "sentiment_analysis"],
    }


def _get_mcp_agents(_auth_manager: AuthManager, status: str) -> list[dict[str, Any]]:
    """Get list of MCP agents."""
    # This would call the actual API endpoint
    # For now, return mock data
    agents = [
        {
            "id": "agent_localhost_8080",
            "name": "Local MCP Agent",
            "host": "localhost",
            "port": 8080,
            "status": "active",
            "tool_count": 5,
            "version": "1.0.0",
            "last_seen": "2024-01-01T12:00:00Z",
            "capabilities": ["text_processing", "bias_detection"],
        },
        {
            "id": "agent_remote_8081",
            "name": "Remote MCP Agent",
            "host": "remote.example.com",
            "port": 8081,
            "status": "inactive",
            "tool_count": 3,
            "version": "1.0.0",
            "last_seen": "2024-01-01T11:00:00Z",
            "capabilities": ["sentiment_analysis"],
        },
    ]

    if status != "all":
        agents = [agent for agent in agents if agent["status"] == status]

    return agents


def _disconnect_from_agent(_auth_manager: AuthManager, agent_id: str) -> dict[str, Any]:
    """Disconnect from MCP agent."""
    logger.info(f"Disconnecting from agent: {agent_id}")

    # This would call the actual API endpoint
    return {"success": True, "message": f"Successfully disconnected from {agent_id}"}


def _get_agent_info(auth_manager: AuthManager, agent_id: str) -> dict[str, Any] | None:
    """Get information about a specific agent."""
    # This would call the actual API endpoint
    agents = _get_mcp_agents(auth_manager, "all")

    for agent in agents:
        if agent["id"] == agent_id:
            return agent

    return None


def _save_connection_info(_host: str, _port: int, agent_id: str) -> None:
    """Save connection information for future use."""
    # This would save to a local cache or config file
    logger.info(f"Saved connection info for {agent_id}")


def _remove_connection_info(agent_id: str) -> None:
    """Remove connection information."""
    # This would remove from local cache or config file
    logger.info(f"Removed connection info for {agent_id}")


def _execute_tool_sync(_auth_manager: AuthManager, agent_id: str, tool: str, _params: dict[str, Any]) -> dict[str, Any]:
    """Execute tool synchronously."""
    logger.info(f"Executing tool {tool} on agent {agent_id}")

    # This would call the actual API endpoint
    # Simulate some processing time
    time.sleep(1)

    return {
        "success": True,
        "result": {
            "analysis": f"Results from {tool}",
            "confidence": 0.95,
            "processing_time": 1.2,
        },
        "metadata": {"agent_id": agent_id, "tool": tool, "execution_time": 1.2},
    }


def _execute_tool_async(_auth_manager: AuthManager, agent_id: str, tool: str, _params: dict[str, Any]) -> str:
    """Execute tool asynchronously."""
    logger.info(f"Async execution of tool {tool} on agent {agent_id}")

    # This would call the actual API endpoint
    return f"exec_{int(time.time())}"


def _get_execution_status(_auth_manager: AuthManager, execution_id: str) -> dict[str, Any] | None:
    """Get execution status."""
    # This would call the actual API endpoint
    return {
        "id": execution_id,
        "status": "completed",
        "progress": 100,
        "result": {"analysis": "Completed analysis", "confidence": 0.95},
        "created_at": "2024-01-01T12:00:00Z",
        "completed_at": "2024-01-01T12:01:00Z",
    }


def _list_executions(_auth_manager: AuthManager, agent_id: str | None, limit: int) -> list[dict[str, Any]]:
    """List recent executions."""
    # This would call the actual API endpoint
    executions = [
        {
            "id": "exec_1",
            "agent_id": "agent_localhost_8080",
            "tool": "bias_detection",
            "status": "completed",
            "created_at": "2024-01-01T12:00:00Z",
        },
        {
            "id": "exec_2",
            "agent_id": "agent_localhost_8080",
            "tool": "sentiment_analysis",
            "status": "running",
            "created_at": "2024-01-01T12:05:00Z",
        },
    ]

    if agent_id:
        executions = [exec for exec in executions if exec["agent_id"] == agent_id]

    return executions[:limit]


def _display_execution_details(execution_info: dict[str, Any]) -> None:
    """Display execution details."""
    click.echo(f"Execution ID: {execution_info['id']}")
    click.echo(f"Status: {execution_info['status']}")
    click.echo(f"Progress: {execution_info['progress']}%")
    click.echo(f"Created: {execution_info['created_at']}")

    if execution_info.get("completed_at"):
        click.echo(f"Completed: {execution_info['completed_at']}")

    if execution_info.get("result"):
        click.echo("Results:")
        click.echo(json.dumps(execution_info["result"], indent=2))


def _display_executions_list(executions: list[dict[str, Any]]) -> None:
    """Display list of executions."""
    if not executions:
        click.echo("No executions found")
        return

    click.echo("📋 Recent Executions:")
    for execution in executions:
        status_icon = "🟢" if execution["status"] == "completed" else "🟡" if execution["status"] == "running" else "🔴"
        click.echo(f"  {status_icon} {execution['id']} - {execution['tool']} - {execution['status']}")


def _test_agent_capability(_auth_manager: AuthManager, agent_id: str, capability: str) -> dict[str, Any]:
    """Test agent capability."""
    logger.info(f"Testing capability {capability} on agent {agent_id}")

    # This would call the actual API endpoint
    # Simulate capability test
    if capability == "bias_detection":
        return {
            "success": True,
            "details": {
                "test_cases_run": 10,
                "accuracy": 0.95,
                "average_response_time": 0.05,  # seconds
            },
        }
    return {
        "success": False,
        "error": f"Capability {capability} not supported by this agent",
        "suggestions": [
            "Check available capabilities with: pixelated mcp-connect list-agents --detailed",
            "Try a different capability test",
            "Connect to a different agent",
        ],
    }


def _display_interactive_help() -> None:
    """Display help for interactive mode."""
    click.echo("🔧 Interactive MCP Commands:")
    click.echo("  agents                    - List available agents")
    click.echo("  connect <host> <port>     - Connect to agent")
    click.echo("  execute <agent-id> <tool> - Execute tool on agent")
    click.echo("  help                      - Show this help")
    click.echo("  exit/quit                 - Exit interactive mode")


def _display_agents_summary(agents: list[dict[str, Any]]) -> None:
    """Display agents summary."""
    active_agents = [agent for agent in agents if agent["status"] == "active"]
    click.echo(f"📊 MCP Agents: {len(active_agents)}/{len(agents)} active")

    for agent in agents:
        status_icon = "🟢" if agent["status"] == "active" else "🔴"
        click.echo(f"  {status_icon} {agent['name']}")
