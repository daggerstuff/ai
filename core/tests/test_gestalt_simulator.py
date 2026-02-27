import json
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest

from core.gestalt_simulator import GestaltSimulator


@pytest.fixture
def mock_genai():
    with patch("google.genai.Client") as mock_client:
        yield mock_client


@pytest.fixture
def simulator(mock_genai):
    # Use a dummy path for defense model to trigger mock mode in GestaltEngine
    return GestaltSimulator(defense_model_path="dummy_path.ckpt")


def test_simulator_initialization(simulator):
    assert simulator.gestalt_engine is not None
    assert simulator.persona_manager is not None
    # Verify that it's in mock mode since dummy_path.ckpt doesn't exist
    assert not simulator.gestalt_engine.defense_model_loaded


def test_simulate_turn_mock_mode(simulator):
    dialogue = [{"speaker": "therapist", "text": "How are you today?"}]
    target_utterance = "I'm feeling a bit overwhelmed."

    with patch.object(
        simulator, "_call_llm", return_value="I don't want to talk about it."
    ) as mock_llm:
        result = simulator.simulate_turn(dialogue, target_utterance)

        assert result["original_utterance"] == target_utterance
        assert result["new_response"] == "I don't want to talk about it."
        assert "persona_id" in result
        assert "directive_used" in result

        # Verify LLM was called with expected arguments
        args, kwargs = mock_llm.call_args
        system_prompt = args[0]
        history = args[1]

        assert "therapy patient" in system_prompt.lower()
        assert len(history) == 2
        assert history[0]["content"] == "How are you today?"
        assert history[1]["content"] == "I'm feeling a bit overwhelmed."


def test_process_batch(simulator):
    # Create a temporary input JSONL file
    dataset = [
        {
            "messages": [
                {"role": "system", "content": "System prompt"},
                {"role": "user", "content": "User prompt 1"},
                {"role": "assistant", "content": "Assistant prompt 1"},
            ]
        },
        {
            "messages": [
                {"role": "system", "content": "System prompt"},
                {"role": "user", "content": "User prompt 2"},
                {"role": "assistant", "content": "Assistant prompt 2"},
            ]
        },
    ]

    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".jsonl", delete=False
    ) as temp_in:
        # Sourcery - Avoid loops in tests.
        temp_in.write("".join(f"{json.dumps(record)}\n" for record in dataset))
        temp_in_path = temp_in.name

    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".jsonl", delete=False
    ) as temp_out:
        temp_out_path = temp_out.name

    try:
        with patch.object(simulator, "simulate_turn") as mock_simulate:
            _execute_batch_simulation(
                mock_simulate, simulator, temp_in_path, temp_out_path
            )
    finally:
        # Sourcery - Avoid conditionals in tests.
        Path(temp_in_path).unlink(missing_ok=True)
        Path(temp_out_path).unlink(missing_ok=True)


def _execute_batch_simulation(mock_simulate, simulator, temp_in_path, temp_out_path):
    mock_simulate.return_value = {
        "new_response": "Synthesized response",
        "persona_id": "anxious",
        "directive_used": "directive",
        "gestalt_state": {},
    }

    written_count = simulator.process_batch(temp_in_path, temp_out_path)

    assert written_count == 2

    # Verify output content
    with open(temp_out_path, "r") as f:
        output_records = [json.loads(line) for line in f]

    assert len(output_records) == 2
    for record in output_records:
        assert record["messages"][-1]["content"] == "Synthesized response"
        assert "gestalt_simulation" in record["metadata"]
        assert record["metadata"]["gestalt_simulation"]["persona_id"] == "anxious"
