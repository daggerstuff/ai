"""
Integration workflow prompts for MCP Server.

This module provides prompt templates for dataset integration planning workflows.
"""

import logging
from typing import Any, Dict, List, Optional

from ai.sourcing.journal.mcp.prompts.base import MCPPrompt

logger = logging.getLogger(__name__)


class CreateIntegrationPlansPrompt(MCPPrompt):
    """Prompt for creating integration plans workflow."""

    def __init__(self) -> None:
        """Initialize CreateIntegrationPlansPrompt."""
        super().__init__(
            name="create_integration_plans_workflow",
            description="Guide for creating integration plans for acquired datasets. This prompt provides step-by-step instructions for using the integration planning workflow to prepare datasets for training pipeline integration.",
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
                    "description": "Optional list of specific source IDs to create integration plans for. If not provided, all acquired datasets in the session will be used.",
                    "required": False,
                },
                {
                    "name": "target_format",
                    "type": "string",
                    "description": "Target format for integration (chatml or conversation_record)",
                    "required": False,
                },
            ],
        )

    def render(self, params: Optional[Dict[str, Any]] = None) -> str:
        """
        Render integration planning workflow prompt.

        Args:
            params: Prompt parameters (session_id, source_ids, target_format)

        Returns:
            Rendered prompt text
        """
        if not params:
            params = {}

        # Validate arguments
        self.validate_arguments(params)

        session_id = params.get("session_id", "{session_id}")
        source_ids = params.get("source_ids")
        target_format = params.get("target_format", "chatml")

        # Format source_ids for display
        if source_ids:
            source_ids_str = ", ".join(f'"{sid}"' for sid in source_ids)
            source_ids_json = str(source_ids)
        else:
            source_ids_str = "all acquired datasets in session"
            source_ids_json = "null"

        template = f"""# Integration Planning Workflow

## Overview
This workflow guides you through creating integration plans for acquired datasets for research session `{session_id}`. Integration plans specify how datasets will be transformed and integrated into the training pipeline.

## Integration Planning Process

The integration planning process involves:
1. **Dataset Structure Analysis**: Analyzing dataset format and structure
2. **Schema Mapping**: Mapping dataset fields to training pipeline format
3. **Transformation Specification**: Defining required transformations
4. **Complexity Estimation**: Estimating integration complexity and effort
5. **Preprocessing Script Generation**: Generating Python scripts for preprocessing

## Target Formats

Integration plans support two target formats:

1. **ChatML Format** (default)
   - Message-based conversation format
   - Structure: `{{"role": "user|assistant|system", "content": "text"}}`
   - Compatible with most LLM training pipelines

2. **ConversationRecord Format**
   - Structured conversation record format
   - Includes metadata and context
   - Suitable for specialized training pipelines

## Complexity Levels

Integration complexity is estimated as:
- **Low Complexity**: Simple transformations, <2 hours
- **Medium Complexity**: Moderate transformations, 2-8 hours
- **High Complexity**: Complex transformations, >8 hours

## Step 1: Prepare Integration Parameters

**Session ID**: `{session_id}`

**Source IDs**: {source_ids_str}
- If not specified, all acquired datasets in the session will be used
- Specify source IDs to create plans for selected datasets

**Target Format**: `{target_format}`
- Options: `chatml` or `conversation_record`
- Default: `chatml`

## Step 2: Execute Integration Planning

Use the `create_integration_plans` tool with the following parameters:

```json
{{
  "session_id": "{session_id}",
  "source_ids": {source_ids_json},
  "target_format": "{target_format}"
}}
```

## Step 3: Review Integration Plans

After execution, the tool will return:
- **Total plans created**: Number of integration plans created
- **Plans list**: Array of integration plan objects with:
  - `plan_id`: Unique identifier
  - `source_id`: Source identifier
  - `target_format`: Target format for integration
  - `complexity_level`: Low, Medium, or High
  - `estimated_hours`: Estimated integration time in hours
  - `schema_mapping`: Field mappings from source to target format
  - `transformation_spec`: Required transformations
  - `required_preprocessing`: List of preprocessing steps
  - `notes`: Integration notes and considerations

## Step 4: Review Individual Plans

Use the `get_integration_plan` tool to get detailed plan:
- Review schema mappings
- Check transformation specifications
- Assess complexity estimates
- Review preprocessing requirements

## Step 5: Generate Preprocessing Scripts

Use the `generate_preprocessing_script` tool to generate Python scripts:
- Scripts include all required transformations
- Scripts are ready to execute
- Scripts include error handling and validation

## Step 6: Next Steps

After integration planning:
1. Review integration plans and complexity estimates
2. Generate preprocessing scripts for selected plans
3. Execute preprocessing scripts to transform datasets
4. Integrate transformed datasets into training pipeline
5. Monitor progress using progress resources

## Best Practices

1. **Format Selection**: Choose target format based on training pipeline requirements
2. **Complexity Assessment**: Review complexity estimates before proceeding
3. **Batch Planning**: Create plans for multiple datasets in parallel
4. **Script Generation**: Generate preprocessing scripts for all plans
5. **Validation**: Validate schema mappings before executing transformations

## Example Usage

```python
# Create integration plans for all acquired datasets
create_integration_plans(
    session_id="session_123",
    target_format="chatml"
)

# Create integration plans for specific sources
create_integration_plans(
    session_id="session_123",
    source_ids=["source_1", "source_2"],
    target_format="conversation_record"
)

# Generate preprocessing script
generate_preprocessing_script(
    session_id="session_123",
    plan_id="plan_1"
)
```

## Notes

- Integration planning operations are asynchronous and may take several minutes
- Progress updates are available via progress resources
- Results are automatically stored in the session state
- Preprocessing scripts are generated in Python
- Schema mappings are validated before plan creation
- You can review and modify integration plans if needed
"""

        return template

