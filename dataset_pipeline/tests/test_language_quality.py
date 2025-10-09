import datetime

from ai.dataset_pipeline.conversation_schema import Conversation, Message
from ai.dataset_pipeline.language_quality import assess_language_quality

LANGUAGE_QUALITY_HIGH_THRESHOLD = 0.7
LANGUAGE_QUALITY_LOW_THRESHOLD = 0.5
LANGUAGE_QUALITY_EMPTY_SCORE = 0.0


def test_language_quality_high():
    conv = Conversation(
        id="lq1",
        messages=[
            Message(
                role="user",
                content="I appreciate your thoughtful response. It was insightful and helpful.",
                timestamp=None,
            ),
            Message(
                role="assistant",
                content="Thank you! I'm glad my explanation provided clarity and support.",
                timestamp=None,
            ),
            Message(
                role="user",
                content="Absolutely. The vocabulary you used was quite advanced.",
                timestamp=None,
            ),
        ],
        source="testset",
        created_at=datetime.datetime.now(datetime.UTC),
    )
    result = assess_language_quality(conv)
    assert result["score"] > LANGUAGE_QUALITY_HIGH_THRESHOLD
    assert not result["issues"]


def test_language_quality_low():
    conv = Conversation(
        id="lq2",
        messages=[
            Message(role="user", content="thx uuuuu", timestamp=None),
            Message(role="assistant", content="np", timestamp=None),
            Message(role="user", content="ok", timestamp=None),
        ],
        source="testset",
        created_at=datetime.datetime.now(datetime.UTC),
    )
    result = assess_language_quality(conv)
    assert result["score"] < LANGUAGE_QUALITY_LOW_THRESHOLD
    assert (
        "Frequent spelling errors detected" in result["issues"]
        or "Sentences are too simple" in result["issues"]
    )


def test_language_quality_no_messages():
    conv = Conversation(
        id="lq3",
        messages=[],
        source="testset",
        created_at=datetime.datetime.now(datetime.timezone.utc),
    )
    result = assess_language_quality(conv)
    assert result["score"] == LANGUAGE_QUALITY_EMPTY_SCORE
    assert "No messages in conversation" in result["issues"]
