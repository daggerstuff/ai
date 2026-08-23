from unittest.mock import MagicMock, patch

import pytest

from ai.memory.mem0_gemini.agent_memory_tools import AgentContext
from ai.memory.mem0_gemini.manager import GeminiMem0Config, GeminiMem0Manager
from ai.memory.mem0_gemini.memory_ingestion_config import (
    CrisisDetector,
    PIIFilter,
    SpeculationFilter,
)
from ai.memory.mem0_gemini.multi_agent_memory import (
    AgentIdentity,
    AgentRole,
    CollaborationContext,
    MultiAgentMemory,
)


@pytest.fixture
def mock_genai():
    with patch("google.genai.Client") as mock:
        yield mock


@pytest.fixture
def mock_mem0_client():
    with patch("mem0.MemoryClient") as mock:
        mock_instance = MagicMock()
        mock.return_value = mock_instance
        yield mock_instance


def test_pii_filter():
    pii_filter = PIIFilter()
    text = "Call me at 650-555-0199 or visit 123 Main St. My SSN is 000-11-2222."
    filtered = pii_filter.filter_for_storage(text)
    assert "[REDACTED]" in filtered
    assert "000-11-2222" not in filtered


def test_speculation_filter():
    assert SpeculationFilter.is_speculative("I guess it might be okay") is True
    assert SpeculationFilter.is_speculative("The patient is diagnosed with ADHD") is False
    assert SpeculationFilter.get_confidence_adjustment("I think maybe") <= 0.5
    assert SpeculationFilter.get_confidence_adjustment("I definitely see") > 0.8


def test_crisis_detector():
    detector = CrisisDetector()
    assert detector.get_crisis_severity("I want to kill myself") == "medium"
    assert detector.get_crisis_severity("I feel a bit sad today") == "none"
    assert detector.get_crisis_severity("I have a plan to hurt myself tonight") == "high"
    assert detector.get_crisis_severity("This is my final goodbye forever") == "critical"


def test_gemini_manager_initialization(_mock_genai):
    config = GeminiMem0Config(gemini_api_key="test", mem0_api_key="test")
    # Patching mem0.MemoryClient so GeminiMem0Manager can instantiate it
    with patch("mem0.MemoryClient"):
        manager = GeminiMem0Manager(config)
        assert manager.pii_filter is not None
        assert manager.crisis_detector is not None


def test_agent_context_to_metadata():
    ctx = AgentContext(user_id="u1", session_id="s1", agent_id="a1", scope="shared")
    metadata = ctx.to_metadata()
    assert metadata["user_id"] == "u1"
    assert metadata["session_id"] == "s1"
    assert metadata["agent_id"] == "a1"
    assert metadata["scope"] == "shared"


@pytest.mark.asyncio
async def test_multi_agent_memory_handoff():
    mock_client = MagicMock()
    mock_client.add.return_value = {"results": [{"id": "m1"}]}
    mock_client.get_all.return_value = {"results": []}

    # Inject mock client directly
    memory = MultiAgentMemory(memory_client=mock_client)

    ctx = CollaborationContext(user_id="u1", session_id="s1")
    target = AgentIdentity(agent_id="agent2", role=AgentRole.PRACTICE)

    # Test handoff
    result = await memory.handoff_to_agent(ctx, target_agent=target, summary="Handoff notes")
    assert result["success"] is True
    assert result["handoff_memory_id"] is not None
    assert mock_client.add.called
