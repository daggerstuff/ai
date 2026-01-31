import datetime

from ai.pipelines.orchestrator.conversation_schema import Conversation, Message
from ai.pipelines.orchestrator.emotional_authenticity import score_emotional_authenticity


def test_authenticity_high():
    conv = Conversation(
        id="t1",
        messages=[
            Message(role="user", content="I feel happy today!", timestamp=None),
            Message(
                role="assistant", content="That's wonderful! Joy is important.", timestamp=None
            ),
            Message(role="user", content="But sometimes I get anxious.", timestamp=None),
            Message(
                role="assistant", content="It's normal to feel anxious at times.", timestamp=None
            ),
            Message(role="user", content="Now I feel relieved after talking.", timestamp=None),
        ],
        source="testset",
        created_at=datetime.datetime.now(),
    )
    result = score_emotional_authenticity(conv)
    assert result["score"] > 0.7
    assert not result["issues"]


def test_authenticity_flat():
    conv = Conversation(
        id="t2",
        messages=[
            Message(role="user", content="Hello.", timestamp=None),
            Message(role="assistant", content="Hi.", timestamp=None),
            Message(role="user", content="How are you?", timestamp=None),
            Message(role="assistant", content="I'm fine.", timestamp=None),
        ],
        source="testset",
        created_at=datetime.datetime.now(),
    )
    result = score_emotional_authenticity(conv)
    assert result["score"] < 0.5
    assert "Most messages lack emotional content" in result["issues"]


def test_authenticity_low_diversity():
    conv = Conversation(
        id="t3",
        messages=[
            Message(role="user", content="I am sad.", timestamp=None),
            Message(role="assistant", content="Why are you sad?", timestamp=None),
            Message(role="user", content="Just sad.", timestamp=None),
            Message(role="assistant", content="I hope you feel better.", timestamp=None),
        ],
        source="testset",
        created_at=datetime.datetime.now(),
    )
    result = score_emotional_authenticity(conv)
    assert result["score"] < 0.7
    assert "Low diversity of emotional expression" in result["issues"]


def test_authenticity_no_messages():
    conv = Conversation(id="t4", messages=[], source="testset", created_at=datetime.datetime.now())
    result = score_emotional_authenticity(conv)
    assert result["score"] == 0.0
    assert "No messages in conversation" in result["issues"]
