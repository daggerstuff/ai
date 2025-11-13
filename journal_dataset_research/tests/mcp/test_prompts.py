import pytest

from ai.journal_dataset_research.mcp.prompts.discovery import DiscoverSourcesPrompt
from ai.journal_dataset_research.mcp.prompts.evaluation import EvaluateSourcesPrompt
from ai.journal_dataset_research.mcp.utils.validation import ValidationError


def test_discover_sources_prompt_render_includes_values() -> None:
    prompt = DiscoverSourcesPrompt()

    rendered = prompt.render(
        {
            "session_id": "session-123",
            "keywords": ["therapy", "cbt"],
            "sources": ["pubmed", "zenodo"],
        }
    )

    assert "discover_sources" in rendered
    assert '"therapy"' in rendered
    assert '"pubmed"' in rendered


def test_discover_sources_prompt_missing_session_id() -> None:
    prompt = DiscoverSourcesPrompt()

    with pytest.raises(ValidationError):
        prompt.render({"keywords": ["therapy"], "sources": ["pubmed"]})


def test_evaluate_sources_prompt_schema_includes_arguments() -> None:
    prompt = EvaluateSourcesPrompt()

    argument_names = {arg["name"] for arg in prompt.prompt_schema["arguments"]}
    assert {"session_id", "source_ids"}.issubset(argument_names)


def test_evaluate_sources_prompt_optional_source_ids() -> None:
    prompt = EvaluateSourcesPrompt()

    rendered = prompt.render({"session_id": "session-456"})
    assert "all sources in session" in rendered

    rendered_with_sources = prompt.render(
        {"session_id": "session-456", "source_ids": ["source-1", "source-2"]}
    )
    assert '"source-1"' in rendered_with_sources

