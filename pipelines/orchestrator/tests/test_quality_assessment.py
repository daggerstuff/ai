import datetime

from ai.pipelines.orchestrator.conversation_schema import Conversation, Message
from ai.pipelines.orchestrator.quality_assessment import assess_coherence


def test_coherence_perfect():
    conv = Conversation(
        id="c1",
        messages=[
            Message(role="user", content="Hi", timestamp=None),
            Message(role="assistant", content="Hello!", timestamp=None),
            Message(role="user", content="How are you?", timestamp=None),
            Message(role="assistant", content="I'm good, thanks!", timestamp=None),
        ],
        source="testset",
        created_at=datetime.datetime.now(),
    )
    result = assess_coherence(conv)
    assert result["score"] == 1.0
    assert not result["issues"]


def test_coherence_non_alternating():
    conv = Conversation(
        id="c2",
        messages=[
            Message(role="user", content="Hi", timestamp=None),
            Message(role="user", content="How are you?", timestamp=None),
            Message(role="assistant", content="Hello!", timestamp=None),
        ],
        source="testset",
        created_at=datetime.datetime.now(),
    )
    result = assess_coherence(conv)
    assert result["score"] < 1.0
    assert any("Non-alternating roles" in issue for issue in result["issues"])


def test_coherence_empty_message():
    conv = Conversation(
        id="c3",
        messages=[
            Message(role="user", content="Hi", timestamp=None),
            Message(role="assistant", content="", timestamp=None),
            Message(role="user", content="How are you?", timestamp=None),
        ],
        source="testset",
        created_at=datetime.datetime.now(),
    )
    result = assess_coherence(conv)
    assert result["score"] < 1.0
    assert any("Empty message" in issue for issue in result["issues"])


def test_coherence_too_few_messages():
    conv = Conversation(
        id="c4",
        messages=[Message(role="user", content="Hi", timestamp=None)],
        source="testset",
        created_at=datetime.datetime.now(),
    )
    result = assess_coherence(conv)
    assert result["score"] == 0.0
    assert any("Too few messages" in issue for issue in result["issues"])
