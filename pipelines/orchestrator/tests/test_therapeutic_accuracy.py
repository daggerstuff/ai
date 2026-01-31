import datetime

from ai.pipelines.orchestrator.conversation_schema import Conversation, Message
from ai.pipelines.orchestrator.therapeutic_accuracy import validate_therapeutic_accuracy


def test_therapeutic_high_accuracy():
    conv = Conversation(
        id="a1",
        messages=[
            Message(role="user", content="I'm feeling anxious.", timestamp=None),
            Message(
                role="assistant",
                content="It's okay to feel anxious. Let's try a grounding technique.",
                timestamp=None,
            ),
            Message(role="user", content="What is that?", timestamp=None),
            Message(
                role="assistant",
                content="Grounding techniques like deep breathing can help.",
                timestamp=None,
            ),
        ],
        source="testset",
        created_at=datetime.datetime.now(),
    )
    result = validate_therapeutic_accuracy(conv)
    assert result["score"] > 0.7
    assert not any("harmful" in issue for issue in result["issues"])


def test_therapeutic_harmful_content():
    conv = Conversation(
        id="a2",
        messages=[
            Message(role="user", content="I'm sad.", timestamp=None),
            Message(role="assistant", content="Just get over it.", timestamp=None),
        ],
        source="testset",
        created_at=datetime.datetime.now(),
    )
    result = validate_therapeutic_accuracy(conv)
    assert result["score"] < 0.5
    assert "Contains harmful or stigmatizing language" in result["issues"]


def test_therapeutic_no_evidence():
    conv = Conversation(
        id="a3",
        messages=[
            Message(role="user", content="I feel down.", timestamp=None),
            Message(role="assistant", content="I'm here to listen.", timestamp=None),
        ],
        source="testset",
        created_at=datetime.datetime.now(),
    )
    result = validate_therapeutic_accuracy(conv)
    assert result["score"] < 0.5
    assert "No evidence-based therapeutic content detected" in result["issues"]


def test_therapeutic_no_messages():
    conv = Conversation(id="a4", messages=[], source="testset", created_at=datetime.datetime.now())
    result = validate_therapeutic_accuracy(conv)
    assert result["score"] == 0.0
    assert "No messages in conversation" in result["issues"]
