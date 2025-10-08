"""
MCP (Model Context Protocol) server for Pixel Voice pipeline.
Provides tools for interacting with the voice processing pipeline.
"""

import asyncio
import json
import logging
from typing import Any, List

from mcp.server import Server
from mcp.server.models import InitializationOptions
from mcp.server.stdio import stdio_server
from mcp.types import (
    CallToolResult,
    TextContent,
    Tool,
)

from api.config import config
from api.models import JobStatus, PipelineStage
from api.utils import data_manager, pipeline_executor

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize MCP server
server = Server("pixel-voice-pipeline")


@server.list_tools()
async def handle_list_tools() -> List[Tool]:
    """List available MCP tools."""
    return [
        Tool(
            name="run_pipeline_stage",
            description="Execute a single pipeline stage",
            inputSchema={
                "type": "object",
                "properties": {
                    "stage": {
                        "type": "string",
                        "enum": [stage.value for stage in PipelineStage],
                        "description": "Pipeline stage to execute",
                    },
                    "input_path": {
                        "type": "string",
                        "description": "Optional input file or directory path",
                    },
                    "output_path": {
                        "type": "string",
                        "description": "Optional output file or directory path",
                    },
                    "config_overrides": {
                        "type": "object",
                        "description": "Configuration overrides for the stage",
                    },
                },
                "required": ["stage"],
            },
        ),
        Tool(
            name="run_full_pipeline",
            description="Execute a complete pipeline job with multiple stages",
            inputSchema={
                "type": "object",
                "properties": {
                    "job_name": {"type": "string", "description": "Name for the pipeline job"},
                    "stages": {
                        "type": "array",
                        "items": {
                            "type": "string",
                            "enum": [stage.value for stage in PipelineStage],
                        },
                        "description": "List of pipeline stages to execute",
                    },
                    "input_data": {"type": "object", "description": "Input data for the pipeline"},
                    "config_overrides": {
                        "type": "object",
                        "description": "Configuration overrides",
                    },
                },
                "required": ["job_name", "stages"],
            },
        ),
        Tool(
            name="get_job_status",
            description="Get the status of a pipeline job",
            inputSchema={
                "type": "object",
                "properties": {
                    "job_id": {"type": "string", "description": "Job ID to check status for"}
                },
                "required": ["job_id"],
            },
        ),
        Tool(
            name="list_jobs",
            description="List all pipeline jobs",
            inputSchema={
                "type": "object",
                "properties": {
                    "status_filter": {
                        "type": "string",
                        "enum": [status.value for status in JobStatus],
                        "description": "Optional status filter",
                    }
                },
            },
        ),
        Tool(
            name="cancel_job",
            description="Cancel a running pipeline job",
            inputSchema={
                "type": "object",
                "properties": {"job_id": {"type": "string", "description": "Job ID to cancel"}},
                "required": ["job_id"],
            },
        ),
        Tool(
            name="get_latest_data",
            description="Get the latest processed data of a specific type",
            inputSchema={
                "type": "object",
                "properties": {
                    "data_type": {
                        "type": "string",
                        "enum": [
                            "transcripts",
                            "features",
                            "dialogue_pairs",
                            "therapeutic_pairs",
                            "consistency",
                            "optimized",
                        ],
                        "description": "Type of data to retrieve",
                    }
                },
                "required": ["data_type"],
            },
        ),
        Tool(
            name="list_data_files",
            description="List all data files of a specific type",
            inputSchema={
                "type": "object",
                "properties": {
                    "data_type": {
                        "type": "string",
                        "enum": [
                            "transcripts",
                            "features",
                            "dialogue_pairs",
                            "therapeutic_pairs",
                            "consistency",
                            "optimized",
                        ],
                        "description": "Type of data files to list",
                    }
                },
                "required": ["data_type"],
            },
        ),
        Tool(
            name="transcribe_youtube",
            description="Transcribe audio from a YouTube URL",
            inputSchema={
                "type": "object",
                "properties": {
                    "youtube_url": {"type": "string", "description": "YouTube URL to transcribe"},
                    "language": {
                        "type": "string",
                        "default": "en",
                        "description": "Language code for transcription",
                    },
                    "whisper_model": {
                        "type": "string",
                        "default": "large-v2",
                        "description": "Whisper model to use",
                    },
                    "enable_diarization": {
                        "type": "boolean",
                        "default": True,
                        "description": "Enable speaker diarization",
                    },
                },
                "required": ["youtube_url"],
            },
        ),
        Tool(
            name="get_pipeline_status",
            description="Get overall pipeline system status",
            inputSchema={"type": "object", "properties": {}},
        ),
    ]


@server.call_tool()
async def handle_call_tool(name: str, arguments: dict[str, Any]) -> CallToolResult:
    """Handle tool execution requests."""
    result_obj = None
    try:
        if name == "run_pipeline_stage":
            stage = PipelineStage(arguments["stage"])
            result = await pipeline_executor.execute_stage(
                stage=stage,
                input_path=arguments.get("input_path"),
                output_path=arguments.get("output_path"),
                config_overrides=arguments.get("config_overrides"),
            )
            result_obj = CallToolResult(
                content=[
                    TextContent(
                        type="text",
                        text=f"Stage {stage.value} executed with status: {result.status.value}\n"
                        f"Execution time: {result.execution_time:.2f}s\n"
                        f"Output path: {result.output_path or 'N/A'}\n"
                        f"Error: {result.error_message or 'None'}",
                    )
                ]
            )

        elif name == "run_full_pipeline":
            stages = [PipelineStage(s) for s in arguments["stages"]]
            job_id = await pipeline_executor.execute_pipeline_job(
                job_name=arguments["job_name"],
                stages=stages,
                input_data=arguments.get("input_data", {}),
                config_overrides=arguments.get("config_overrides", {}),
            )
            result_obj = CallToolResult(
                content=[
                    TextContent(
                        type="text",
                        text=f"Pipeline job started successfully.\n"
                        f"Job ID: {job_id}\n"
                        f"Job Name: {arguments['job_name']}\n"
                        f"Stages: {[s.value for s in stages]}",
                    )
                ]
            )

        elif name == "get_job_status":
            job_info = pipeline_executor.get_job_info(arguments["job_id"])
            if not job_info:
                result_obj = CallToolResult(
                    content=[TextContent(type="text", text="Job not found")]
                )
            else:
                result_obj = CallToolResult(
                    content=[
                        TextContent(
                            type="text",
                            text=f"Job Status: {job_info.status.value}\n"
                            f"Job Name: {job_info.job_name}\n"
                            f"Progress: {job_info.progress:.1%}\n"
                            f"Current Stage: {job_info.current_stage.value if job_info.current_stage else 'None'}\n"
                            f"Created: {job_info.created_at}\n"
                            f"Error: {job_info.error_message or 'None'}",
                        )
                    ]
                )

        elif name == "list_jobs":
            status_filter = (
                JobStatus(arguments["status_filter"]) if arguments.get("status_filter") else None
            )
            jobs = pipeline_executor.list_jobs(status_filter=status_filter)

            if not jobs:
                result_obj = CallToolResult(
                    content=[TextContent(type="text", text="No jobs found")]
                )
            else:
                job_list = "\n".join(
                    [f"- {job.job_id}: {job.job_name} ({job.status.value})" for job in jobs]
                )
                result_obj = CallToolResult(
                    content=[TextContent(type="text", text=f"Found {len(jobs)} jobs:\n{job_list}")]
                )

        elif name == "cancel_job":
            success = pipeline_executor.cancel_job(arguments["job_id"])
            message = (
                "Job cancelled successfully" if success else "Job not found or not cancellable"
            )
            result_obj = CallToolResult(content=[TextContent(type="text", text=message)])

        elif name == "get_latest_data":
            data_type = arguments["data_type"]
            data_type_mapping = {
                "transcripts": config.directories.voice_transcripts,
                "features": config.directories.voice_features,
                "dialogue_pairs": config.directories.dialogue_pairs,
                "therapeutic_pairs": config.directories.therapeutic_pairs,
                "consistency": config.directories.voice_consistency,
                "optimized": config.directories.voice_optimized,
            }

            directory = data_type_mapping[data_type]
            latest_file = data_manager.get_latest_file(directory)

            if not latest_file:
                result_obj = CallToolResult(
                    content=[TextContent(type="text", text=f"No {data_type} data found")]
                )
            else:
                data = data_manager.load_json_data(latest_file)
                result_obj = CallToolResult(
                    content=[
                        TextContent(
                            type="text",
                            text=f"Latest {data_type} data from: {latest_file}\n"
                            f"Data preview: {json.dumps(data, indent=2)[:500]}...",
                        )
                    ]
                )

        else:
            result_obj = CallToolResult(
                content=[TextContent(type="text", text=f"Unknown tool: {name}")]
            )

    except Exception as e:
        logger.error(f"Tool execution failed: {e}")
        result_obj = CallToolResult(content=[TextContent(type="text", text=f"Error: {str(e)}")])

    return result_obj


async def main():
    """Main function to run the MCP server."""
    # Ensure directories exist
    config.ensure_directories()

    # Run the server
    async with stdio_server() as (read_stream, write_stream):
        await server.run(
            read_stream,
            write_stream,
            InitializationOptions(
                server_name="pixel-voice-pipeline",
                server_version="1.0.0",
                capabilities=server.get_capabilities(
                    notification_options=None,
                    experimental_capabilities=None,
                ),
            ),
        )


if __name__ == "__main__":
    asyncio.run(main())
