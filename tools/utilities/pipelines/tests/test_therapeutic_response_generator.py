"""Tests for the therapeutic response generator."""

import pytest

from ai.tools.utilities.pipelines.therapeutic_response_generator import TherapeuticResponseGenerator


@pytest.fixture
def generator() -> TherapeuticResponseGenerator:
    return TherapeuticResponseGenerator()


def test_a_response_is_produced_for_plain_text(generator: TherapeuticResponseGenerator) -> None:
    """Every request yields a non-empty response with a tone."""
    package = generator.process({"text": "I feel sad"})

    assert isinstance(package.response, str)
    assert package.response.strip()
    assert package.tone


def test_the_response_reflects_the_input(generator: TherapeuticResponseGenerator) -> None:
    """The generator acknowledges the content it was given."""
    package = generator.process({"text": "I feel sad"})

    assert "sad" in package.response.lower() or "sharing" in package.response.lower()


def test_a_non_crisis_request_takes_standard_support(generator: TherapeuticResponseGenerator) -> None:
    """Absent a crisis flag the package carries the standard action."""
    package = generator.process({"text": "I feel sad", "crisis": False})

    assert package.crisis_action == "standard_support"


def test_the_package_carries_metadata(generator: TherapeuticResponseGenerator) -> None:
    """The package exposes metadata alongside the response."""
    package = generator.process({"text": "I feel sad"})

    assert isinstance(package.metadata, dict)
