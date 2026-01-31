from ai.pipelines.orchestrator.conversation_schema import Conversation
from ai.pipelines.orchestrator.standardizer import from_input_output_pair, from_simple_message_list


def test_from_simple_message_list():
    raw = [
        {"role": "user", "content": "Hi"},
        {"role": "assistant", "content": "Hello! How can I help you?"},
    ]
    conv = from_simple_message_list(raw, conversation_id="c1", source="testset")
    assert isinstance(conv, Conversation)
    assert conv.id == "c1"
    assert conv.source == "testset"
    assert len(conv.messages) == 2
    assert conv.messages[0].role == "user"
    assert conv.messages[1].role == "assistant"


def test_from_input_output_pair():
    conv = from_input_output_pair(
        "What is CBT?", "Cognitive Behavioral Therapy", conversation_id="c2", source="testset"
    )
    assert isinstance(conv, Conversation)
    assert conv.id == "c2"
    assert conv.source == "testset"
    assert len(conv.messages) == 2
    assert conv.messages[0].role == "user"
    assert conv.messages[0].content == "What is CBT?"
    assert conv.messages[1].role == "assistant"
    assert conv.messages[1].content == "Cognitive Behavioral Therapy"
