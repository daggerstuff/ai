"""
Evaluation workflow prompts for MCP Server.

This module provides prompt templates for dataset evaluation workflows.
"""

import logging
from typing import Any, Dict, List, Optional

from ai.sourcing.journal.mcp.prompts.base import MCPPrompt

logger = logging.getLogger(__name__)


class EvaluateSourcesPrompt(MCPPrompt):
    """Prompt for evaluating dataset sources workflow."""

    def __init__(self) -> None:
        """Initialize EvaluateSourcesPrompt."""
        super().__init__(
            name="evaluate_sources_workflow",
            description="Guide for evaluating dataset sources across multiple quality dimensions. This prompt provides step-by-step instructions for using the evaluation workflow to assess dataset quality and prioritize sources.",
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
                    "description": "Optional list of specific source IDs to evaluate. If not provided, all sources in the session will be evaluated.",
                    "required": False,
                },
            ],
        )

    def render(self, params: Optional[Dict[str, Any]] = None) -> str:
        """
        Render evaluation workflow prompt.

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
            source_ids_str = "all sources in session"
            source_ids_json = "null"

        template = f"""# Dataset Evaluation Workflow

## Overview
This workflow guides you through evaluating dataset sources across four quality dimensions for research session `{session_id}`.

## Evaluation Dimensions

The evaluation engine assesses each dataset across four dimensions:

1. **Therapeutic Relevance** (35% weight)
   - Relevance to therapeutic applications
   - Clinical applicability
   - Mental health domain alignment

2. **Data Structure Quality** (25% weight)
   - Organization and completeness
   - Data format consistency
   - Metadata quality

3. **Training Integration Potential** (20% weight)
   - Ease of integration into training pipeline
   - Format compatibility
   - Preprocessing requirements

4. **Ethical Accessibility** (20% weight)
   - License compatibility
   - Privacy compliance (HIPAA, GDPR)
   - Ethical considerations

## Scoring System

- Each dimension scored 1-10
- Weighted average calculates overall score (1-10)
- Priority tiers:
  - **High Priority**: ≥7.5 overall score
  - **Medium Priority**: ≥5.0 overall score
  - **Low Priority**: <5.0 overall score

## Step 1: Prepare Evaluation Parameters

**Session ID**: `{session_id}`

**Source IDs**: {source_ids_str}
- If not specified, all sources in the session will be evaluated
- Specify source IDs to evaluate only selected sources

## Step 2: Execute Evaluation

Use the `evaluate_sources` tool with the following parameters:

```json
{{
  "session_id": "{session_id}",
  "source_ids": {source_ids_json}
}}
```

## Step 3: Review Evaluation Results

After execution, the tool will return:
- **Total evaluated**: Number of sources evaluated
- **Evaluations list**: Array of evaluation objects with:
  - `evaluation_id`: Unique identifier
  - `source_id`: Source identifier
  - `therapeutic_relevance_score`: Score (1-10)
  - `data_structure_score`: Score (1-10)
  - `training_integration_score`: Score (1-10)
  - `ethical_accessibility_score`: Score (1-10)
  - `overall_score`: Weighted average (1-10)
  - `priority_tier`: High, Medium, or Low
  - `evaluation_notes`: Detailed assessment notes
  - `compliance_checks`: License, privacy, HIPAA compliance status

## Step 4: Filter and Prioritize

Use the `get_evaluations` tool to filter results:
- Filter by priority tier (high, medium, low)
- Filter by score ranges
- Filter by evaluation status
- Sort by overall score

## Step 5: Review Individual Evaluations

Use the `get_evaluation` tool to get detailed evaluation:
- Review evaluation notes
- Check compliance status
- Review competitive advantages
- Assess integration complexity

## Step 6: Next Steps

After evaluation:
1. Prioritize high-scoring sources for acquisition
2. Review compliance requirements for selected sources
3. Proceed to acquisition phase using `acquire_datasets` tool
4. Monitor progress using progress resources

## Best Practices

1. **Prioritization**: Focus on high-priority sources (≥7.5) first
2. **Compliance**: Review ethical accessibility scores before acquisition
3. **Integration**: Consider training integration potential for pipeline compatibility
4. **Batch Evaluation**: Evaluate all sources at once for efficiency
5. **Manual Override**: Use `update_evaluation` tool to adjust scores if needed

## Example Usage

```python
# Evaluate all sources in session
evaluate_sources(
    session_id="session_123"
)

# Evaluate specific sources
evaluate_sources(
    session_id="session_123",
    source_ids=["source_1", "source_2", "source_3"]
)
```

## Notes

- Evaluation operations are asynchronous and may take several minutes
- Progress updates are available via progress resources
- Results are automatically stored in the session state
- Compliance checks are performed automatically during evaluation
- You can manually override evaluation scores if needed
"""

        return template

