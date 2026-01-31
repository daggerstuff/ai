"""
Acquisition workflow prompts for MCP Server.

This module provides prompt templates for dataset acquisition workflows.
"""

import logging
from typing import Any, Dict, List, Optional

from ai.sourcing.journal.mcp.prompts.base import MCPPrompt

logger = logging.getLogger(__name__)


class AcquireDatasetsPrompt(MCPPrompt):
    """Prompt for acquiring dataset sources workflow."""

    def __init__(self) -> None:
        """Initialize AcquireDatasetsPrompt."""
        super().__init__(
            name="acquire_datasets_workflow",
            description="Guide for acquiring datasets from identified sources. This prompt provides step-by-step instructions for using the acquisition workflow to download and store datasets.",
            arguments=[
                {
                    "name": "session_id",
                    "type": "string",
                    "description": "The research session ID",
                    "required": True,
                },
                {
                    "name": "source_ids",
                    "type": "array",
                    "description": "Optional list of specific source IDs to acquire. If not provided, all evaluated sources in the session will be acquired.",
                    "required": False,
                },
            ],
        )

    def render(self, params: Optional[Dict[str, Any]] = None) -> str:
        """
        Render acquisition workflow prompt.

        Args:
            params: Prompt parameters (session_id, source_ids)

        Returns:
            Rendered prompt text
        """
        if not params:
            params = {}

        # Validate arguments
        self.validate_arguments(params)

        session_id = params.get("session_id", "{session_id}")
        source_ids = params.get("source_ids")

        # Format source_ids for display
        if source_ids:
            source_ids_str = ", ".join(f'"{sid}"' for sid in source_ids)
            source_ids_json = str(source_ids)
        else:
            source_ids_str = "all evaluated sources in session"
            source_ids_json = "null"

        template = f"""# Dataset Acquisition Workflow

## Overview
This workflow guides you through acquiring datasets from identified sources for research session `{session_id}`.

## Acquisition Process

The acquisition process involves:
1. **Access Method Determination**: Analyzing source to determine access method
2. **Access Request Submission**: Submitting requests if required
3. **Dataset Download**: Downloading datasets via direct download or API
4. **Storage and Validation**: Storing datasets securely and validating integrity

## Access Methods

Datasets may be accessed through:
- **Direct Download**: Publicly available datasets
- **API Access**: Programmatic access via API
- **Access Request**: Form-based or email-based requests
- **Collaboration**: Research collaboration agreements
- **Registration**: Registration with data provider

## Step 1: Prepare Acquisition Parameters

**Session ID**: `{session_id}`

**Source IDs**: {source_ids_str}
- If not specified, all evaluated sources in the session will be acquired
- Specify source IDs to acquire only selected sources
- Prioritize high-scoring sources from evaluation phase

## Step 2: Execute Acquisition

Use the `acquire_datasets` tool with the following parameters:

```json
{{
  "session_id": "{session_id}",
  "source_ids": {source_ids_json}
}}
```

## Step 3: Review Acquisition Results

After execution, the tool will return:
- **Total acquired**: Number of datasets acquired
- **Acquisitions list**: Array of acquisition objects with:
  - `acquisition_id`: Unique identifier
  - `source_id`: Source identifier
  - `status`: Acquisition status (pending, in_progress, completed, failed)
  - `access_method`: Method used to access dataset
  - `download_progress`: Download progress percentage (0-100)
  - `file_path`: Local file path where dataset is stored
  - `file_size`: Dataset file size in bytes
  - `file_format`: Dataset format (CSV, JSON, JSONL, Parquet, XML)
  - `checksum`: File integrity checksum
  - `error_message`: Error message if acquisition failed

## Step 4: Monitor Acquisition Progress

Use the `get_acquisitions` tool to monitor progress:
- Check acquisition status
- Monitor download progress
- Review error messages for failed acquisitions

## Step 5: Handle Failed Acquisitions

If acquisitions fail:
1. Review error messages using `get_acquisition` tool
2. Check access requirements and permissions
3. Retry failed acquisitions using `update_acquisition` tool
4. Cancel in-progress acquisitions if needed

## Step 6: Next Steps

After acquisition:
1. Verify dataset integrity using checksums
2. Review dataset formats and structures
3. Proceed to integration planning using `create_integration_plans` tool
4. Monitor progress using progress resources

## Best Practices

1. **Prioritization**: Acquire high-priority sources first
2. **Batch Processing**: Acquire multiple sources in parallel for efficiency
3. **Error Handling**: Monitor and retry failed acquisitions
4. **Storage Management**: Ensure sufficient storage space before acquisition
5. **Compliance**: Verify access permissions and license compliance before acquisition

## Example Usage

```python
# Acquire all evaluated sources
acquire_datasets(
    session_id="session_123"
)

# Acquire specific sources
acquire_datasets(
    session_id="session_123",
    source_ids=["source_1", "source_2", "source_3"]
)
```

## Notes

- Acquisition operations are asynchronous and may take several minutes to hours
- Progress updates are available via progress resources
- Results are automatically stored in the session state
- Access requests may require manual approval
- Download progress is tracked and reported in real-time
- You can cancel in-progress acquisitions if needed
"""

        return template

