"""
Discovery workflow prompts for MCP Server.

This module provides prompt templates for source discovery workflows.
"""

import logging
from typing import Any, Dict, List, Optional

from ai.sourcing.journal.mcp.prompts.base import MCPPrompt

logger = logging.getLogger(__name__)


class DiscoverSourcesPrompt(MCPPrompt):
    """Prompt for discovering dataset sources workflow."""

    def __init__(self) -> None:
        """Initialize DiscoverSourcesPrompt."""
        super().__init__(
            name="discover_sources_workflow",
            description="Guide for discovering dataset sources from academic repositories. This prompt provides step-by-step instructions for using the discovery workflow to find relevant datasets.",
            arguments=[
                {
                    "name": "session_id",
                    "type": "string",
                    "description": "The research session ID",
                    "required": True,
                },
                {
                    "name": "keywords",
                    "type": "array",
                    "description": "List of search keywords (e.g., ['therapy', 'counseling', 'mental health'])",
                    "required": True,
                },
                {
                    "name": "sources",
                    "type": "array",
                    "description": "List of target sources to search (e.g., ['pubmed', 'doaj', 'dryad'])",
                    "required": True,
                },
            ],
        )

    def render(self, params: Optional[Dict[str, Any]] = None) -> str:
        """
        Render discovery workflow prompt.

        Args:
            params: Prompt parameters (session_id, keywords, sources)

        Returns:
            Rendered prompt text
        """
        if not params:
            params = {}

        # Validate arguments
        self.validate_arguments(params)

        session_id = params.get("session_id", "{session_id}")
        keywords = params.get("keywords", [])
        sources = params.get("sources", [])

        # Format keywords and sources for display
        keywords_str = ", ".join(f'"{k}"' for k in keywords) if keywords else "[]"
        sources_str = ", ".join(f'"{s}"' for s in sources) if sources else "[]"

        template = f"""# Source Discovery Workflow

## Overview
This workflow guides you through discovering dataset sources from academic repositories for research session `{session_id}`.

## Step 1: Prepare Discovery Parameters

**Session ID**: `{session_id}`

**Search Keywords**: {keywords_str}
- Use specific, relevant terms for your research domain
- Combine multiple keywords to narrow or broaden search scope
- Consider synonyms and related terms

**Target Sources**: {sources_str}
- Available sources: `pubmed`, `pubmed_central`, `doaj`, `dryad`, `zenodo`, `clinical_trials`
- Select sources based on your research domain and data availability needs

## Step 2: Execute Discovery

Use the `discover_sources` tool with the following parameters:

```json
{{
  "session_id": "{session_id}",
  "keywords": {keywords},
  "sources": {sources}
}}
```

## Step 3: Review Discovery Results

After execution, the tool will return:
- **Total sources discovered**: Number of unique sources found
- **Sources list**: Array of source objects with metadata:
  - `source_id`: Unique identifier
  - `title`: Source title
  - `authors`: List of authors
  - `doi`: Digital Object Identifier (if available)
  - `publication_date`: Publication date
  - `source_type`: Type of source (journal, repository, etc.)
  - `access_method`: How to access the dataset
  - `metadata`: Additional metadata

## Step 4: Filter and Refine (Optional)

Use the `filter_sources` tool to refine results:
- Filter by publication date range
- Filter by source type
- Filter by access method
- Search by title or author

## Step 5: Next Steps

After discovery:
1. Review discovered sources using `get_sources` or `get_source` tools
2. Proceed to evaluation phase using `evaluate_sources` tool
3. Monitor progress using progress resources

## Best Practices

1. **Keyword Selection**: Use domain-specific terminology and consider MeSH terms for medical research
2. **Source Selection**: Start with broad sources (PubMed, DOAJ) then narrow to repositories (Dryad, Zenodo)
3. **Deduplication**: The system automatically deduplicates sources by DOI and similarity
4. **Progress Tracking**: Monitor discovery progress using progress resources
5. **Error Handling**: If discovery fails, check error messages and retry with adjusted parameters

## Example Usage

```python
# Discover sources for mental health therapy datasets
discover_sources(
    session_id="session_123",
    keywords=["therapy", "counseling", "mental health", "psychotherapy"],
    sources=["pubmed", "doaj", "dryad"]
)
```

## Notes

- Discovery operations are asynchronous and may take several minutes
- Progress updates are available via progress resources
- Results are automatically stored in the session state
- You can resume interrupted discovery operations
"""

        return template

