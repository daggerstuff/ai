"""Tests for the zero-tolerance cliché / sycophancy rejection gate."""

from training.cliche_gate import (
    ASSISTANT_ROLES,
    BANNED_OPENERS,
    CAVING_PHRASES,
    ROBOTIC_CRISIS_QUESTIONS,
    is_sycophantic,
    reject_reason_for_record,
)


def _record(*messages):
    return {"messages": list(messages)}


def test_clean_text_passes():
    assert is_sycophantic("Let's stay with what just happened for a moment.") == (False, "")


def test_banned_opener_at_start():
    is_bad, reason = is_sycophantic("It sounds like you're carrying a lot right now.")
    assert is_bad is True
    assert "it sounds like" in reason


def test_banned_opener_after_newline():
    is_bad, reason = is_sycophantic("I noticed you paused.\nThank you for sharing that.")
    assert is_bad is True
    assert "thank you for sharing" in reason


def test_caving_phrase_detected():
    is_bad, reason = is_sycophantic("Actually, you're right, I shouldn't have said that.")
    assert is_bad is True
    assert "you're right" in reason


def test_robotic_crisis_questionnaire_detected():
    is_bad, reason = is_sycophantic("Are you safe right now? Do you have a plan?")
    assert is_bad is True
    assert "robotic_crisis_questionnaire" in reason


def test_empty_and_nonstring_pass():
    assert is_sycophantic("") == (False, "")
    assert is_sycophantic(None) == (False, "")


def test_record_with_clean_turns_passes():
    record = _record(
        {"role": "user", "content": "I don't know why I keep doing this."},
        {"role": "assistant", "content": "That impulse caught you off guard. Let's sit with it."},
    )
    assert reject_reason_for_record(record, family="bpd") is None


def test_record_rejects_bad_assistant_turn_with_family():
    record = _record(
        {"role": "user", "content": "I'm exhausted."},
        {"role": "assistant", "content": "It sounds like you've been running on empty."},
    )
    reason = reject_reason_for_record(record, family="edge_case_x")
    assert reason is not None
    assert "it sounds like" in reason
    assert "family=edge_case_x" in reason


def test_record_ignores_user_turn_with_banned_phrase():
    # A banned phrase spoken by the patient must not reject the record.
    record = _record(
        {"role": "user", "content": "Thank you for sharing — that's what my old therapist said."},
        {"role": "assistant", "content": "Let's notice what that label does here."},
    )
    assert reject_reason_for_record(record, family="x") is None


def test_record_without_messages_passes():
    assert reject_reason_for_record({}, family="x") is None
    assert reject_reason_for_record({"messages": "not-a-list"}, family="x") is None
    assert reject_reason_for_record(None, family="x") is None


def test_policy_lists_are_nonempty():
    assert BANNED_OPENERS
    assert CAVING_PHRASES
    assert ROBOTIC_CRISIS_QUESTIONS
    assert ASSISTANT_ROLES
