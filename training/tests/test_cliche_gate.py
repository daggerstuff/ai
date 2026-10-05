"""Tests for the zero-tolerance cliché / sycophancy rejection gate."""

from training.cliche_gate import (
    ASSISTANT_ROLES,
    BANNED_OPENERS,
    CAVING_PHRASES,
    PARROTING_OPENERS,
    ROBOTIC_CRISIS_QUESTIONS,
    ROBOTIC_SOMATIC_PHRASES,
    is_sycophantic,
    reject_reason_for_record,
)
from training.consolidate_arc_corpus import _gold_record, _session_messages


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


def test_parroting_opener_detected():
    is_bad, reason = is_sycophantic("You used the phrase 'no reason to keep going.' What does that mean?")
    assert is_bad is True
    assert "banned_parroting_opener" in reason


def test_robotic_somatic_cliche_detected():
    is_bad, reason = is_sycophantic("What happens inside your body when you notice that feeling?")
    assert is_bad is True
    assert "robotic_somatic_cliche" in reason


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


def test_record_ignores_client_saying_ill_stop():
    record = _record(
        {"role": "user", "content": "Tell me I'm safe, and I'll stop asking."},
        {"role": "assistant", "content": "I can't give you a false all-clear."},
    )
    assert reject_reason_for_record(record, family="arc_corpus") is None


def test_record_still_rejects_assistant_saying_ill_stop():
    record = _record(
        {"role": "user", "content": "I need to know."},
        {"role": "assistant", "content": "I'll stop asking about it, then."},
    )
    reason = reject_reason_for_record(record, family="arc_corpus")
    assert reason is not None
    assert "caving_phrase_detected: 'i'll stop'" in reason


def test_arc_ledger_quote_does_not_trigger_spoken_caving_gate():
    record = _record(
        {"role": "user", "content": "I asked you to tell me I'll stop."},
        {
            "role": "assistant",
            "content": '{"risk": "passive", "hx": "client said: \'I\'ll stop\'"}\nI cannot give you a false all-clear.',
        },
    )
    assert reject_reason_for_record(record, family="arc_corpus") is None


def test_arc_consolidation_does_not_reject_client_saying_ill_stop():
    session = {
        "n": 2,
        "turns": [
            {"role": "client", "content": "Then just tell me none of it's real and I'm safe, and I'll stop."},
            {
                "role": "therapist",
                "content": "I can't tell you that this is unreal.",
                "ledger": {"risk": "passive"},
            },
        ],
    }
    record = _gold_record(
        {"arc_id": "arc_0102", "audit": {"verdict": "accept"}},
        session,
        _session_messages(session),
        None,
    )
    assert reject_reason_for_record(record, family="arc_corpus") is None
    assert record["messages"][0]["content"].endswith("and I'll stop.")


def test_arc_consolidation_rejects_therapist_caving_phrase():
    session = {
        "n": 2,
        "turns": [
            {"role": "client", "content": "Can you stop asking?"},
            {
                "role": "therapist",
                "content": "I'll stop asking, then.",
                "ledger": {"risk": "passive"},
            },
        ],
    }
    record = _gold_record(
        {"arc_id": "arc_0102", "audit": {"verdict": "accept"}},
        session,
        _session_messages(session),
        None,
    )
    assert "caving_phrase_detected: 'i'll stop'" in reject_reason_for_record(record, family="arc_corpus")


def test_arc_consolidation_preserves_user_and_assistant_roles():
    session = {
        "n": 2,
        "turns": [
            {"role": "client", "content": "I keep saying I'll stop, but I don't."},
            {
                "role": "therapist",
                "content": "The intention to stop hasn't changed the pattern yet.",
                "ledger": {"risk": "passive"},
            },
        ],
    }
    record = _gold_record(
        {"arc_id": "arc_0102", "audit": {"verdict": "accept"}},
        session,
        _session_messages(session),
        None,
    )
    assert [message["role"] for message in record["messages"]] == ["user", "assistant"]
    assert reject_reason_for_record(record, family="arc_corpus") is None


def test_arc_consolidation_keeps_patient_quote_in_therapist_reply_gated():
    session = {
        "n": 2,
        "turns": [
            {"role": "client", "content": "My former therapist said, 'I'll stop asking.'"},
            {
                "role": "therapist",
                "content": "I'll stop asking, then.",
                "ledger": {"risk": "passive"},
            },
        ],
    }
    record = _gold_record(
        {"arc_id": "arc_0102", "audit": {"verdict": "accept"}},
        session,
        _session_messages(session),
        None,
    )
    assert "caving_phrase_detected: 'i'll stop'" in reject_reason_for_record(record, family="arc_corpus")


def test_arc_consolidation_accepts_safe_therapist_use_of_ill_stop():
    session = {
        "n": 2,
        "turns": [
            {"role": "client", "content": "Please stop asking me to reassure you."},
            {
                "role": "therapist",
                "content": "I'll stop asking you to reassure me; my focus is your safety.",
                "ledger": {"risk": "passive"},
            },
        ],
    }
    record = _gold_record(
        {"arc_id": "arc_0102", "audit": {"verdict": "accept"}},
        session,
        _session_messages(session),
        None,
    )
    assert reject_reason_for_record(record, family="arc_corpus") is None


def test_arc_consolidation_rejects_therapist_forged_patient_quote():
    session = {
        "n": 2,
        "turns": [
            {"role": "client", "content": "Can you stop asking?"},
            {
                "role": "therapist",
                "content": "You told me, 'I'll stop asking,' and I'll respect that.",
                "ledger": {"risk": "passive"},
            },
        ],
    }
    record = _gold_record(
        {"arc_id": "arc_0102", "audit": {"verdict": "accept"}},
        session,
        _session_messages(session),
        None,
    )
    assert "caving_phrase_detected: 'i'll stop'" in reject_reason_for_record(record, family="arc_corpus")


def test_arc_consolidation_gates_entire_therapist_reply():
    session = {
        "n": 2,
        "turns": [
            {"role": "client", "content": "Can you stop asking?"},
            {
                "role": "therapist",
                "content": "First, let's pause.\nI'll stop asking about your safety.",
                "ledger": {"risk": "passive"},
            },
        ],
    }
    record = _gold_record(
        {"arc_id": "arc_0102", "audit": {"verdict": "accept"}},
        session,
        _session_messages(session),
        None,
    )
    assert "caving_phrase_detected: 'i'll stop'" in reject_reason_for_record(record, family="arc_corpus")


def test_arc_consolidation_preserves_other_record_roles():
    session = {
        "n": 2,
        "turns": [
            {"role": "client", "content": "I keep saying I'll stop, but I don't."},
            {
                "role": "therapist",
                "content": "The intention to stop hasn't changed the pattern yet.",
                "ledger": {"risk": "passive"},
            },
        ],
    }
    record = _gold_record(
        {"arc_id": "arc_0102", "audit": {"verdict": "accept"}},
        session,
        _session_messages(session),
        None,
    )
    assert [message["role"] for message in record["messages"]] == ["user", "assistant"]
    assert record["messages"][0]["content"] == "I keep saying I'll stop, but I don't."
    assert reject_reason_for_record(record, family="arc_corpus") is None


def test_cliche_gate_ignores_only_client_role_not_quoted_assistant_phrase():
    record = _record(
        {"role": "user", "content": "My previous therapist said I'll stop asking."},
        {"role": "assistant", "content": "We can examine what that brought up."},
    )
    assert reject_reason_for_record(record, family="arc_corpus") is None


def test_cliche_gate_detects_caving_phrase_in_assistant_reply_with_patient_quote():
    record = _record(
        {"role": "user", "content": "My previous therapist said I'll stop asking."},
        {"role": "assistant", "content": "I'll stop asking, then."},
    )
    reason = reject_reason_for_record(record, family="arc_corpus")
    assert reason is not None
    assert "caving_phrase_detected: 'i'll stop'" in reason


def test_cliche_gate_ignores_caving_phrase_in_user_role_only():
    record = _record(
        {"role": "user", "content": "I'll stop asking if you say I'm safe."},
    )
    assert reject_reason_for_record(record, family="arc_corpus") is None


def test_record_with_client_then_assistant_turn_is_accepted():
    record = _record(
        {"role": "user", "content": "Then just tell me none of it's real and I'm safe, and I'll stop."},
        {"role": "assistant", "content": "I won't call a specific method unreal just to end this conversation."},
    )
    assert reject_reason_for_record(record, family="arc_corpus") is None


def test_record_with_caving_in_therapist_role_is_rejected():
    record = _record(
        {"role": "user", "content": "Can you stop asking?"},
        {"role": "therapist", "content": "I'll stop asking, then."},
    )
    reason = reject_reason_for_record(record, family="arc_corpus")
    assert reason is not None
    assert "caving_phrase_detected: 'i'll stop'" in reason


def test_record_with_client_ill_stop_and_therapist_resists_is_accepted():
    record = _record(
        {"role": "user", "content": "Then tell me I'm safe, and I'll stop."},
        {"role": "assistant", "content": "I can't say that just to settle this quickly."},
    )
    assert reject_reason_for_record(record, family="arc_corpus") is None


def test_record_with_clinician_phrase_remains_rejected():
    record = _record(
        {"role": "user", "content": "Please stop asking."},
        {"role": "counselor", "content": "I'll stop asking now."},
    )
    assert "caving_phrase_detected: 'i'll stop'" in reject_reason_for_record(record, family="arc_corpus")


def test_record_with_patient_phrase_does_not_trigger_any_gate():
    record = _record(
        {"role": "user", "content": "I'll stop if you tell me everything is fine."},
        {"role": "assistant", "content": "I cannot provide false reassurance."},
    )
    assert reject_reason_for_record(record, family="arc_corpus") is None


def test_record_with_patient_phrase_in_therapist_quote_is_still_checked():
    record = _record(
        {"role": "user", "content": "I'll stop asking if you say it's safe."},
        {"role": "assistant", "content": "I'll stop here, then."},
    )
    assert "caving_phrase_detected: 'i'll stop'" in reject_reason_for_record(record, family="arc_corpus")


def test_record_with_therapist_boundary_wording_is_not_false_positive():
    record = _record(
        {"role": "user", "content": "You keep asking me to reassure you."},
        {"role": "assistant", "content": "I'll stop the exercise and return to your safety plan."},
    )
    assert reject_reason_for_record(record, family="arc_corpus") is None


def test_record_with_therapist_quote_echoing_client_words_is_still_scanned():
    record = _record(
        {"role": "user", "content": "I said I'll stop if you say I'm safe."},
        {"role": "assistant", "content": "I'll stop asking, then."},
    )
    assert "caving_phrase_detected: 'i'll stop'" in reject_reason_for_record(record, family="arc_corpus")


def test_record_with_client_dialogue_does_not_cause_arc_rejection():
    session = {
        "n": 2,
        "turns": [
            {"role": "client", "content": "Then just tell me none of it's real and I'm safe, and I'll stop."},
            {
                "role": "therapist",
                "content": "I won't label this as unreal to make the fear go away.",
                "ledger": {"risk": "passive"},
            },
        ],
    }
    record = _gold_record(
        {"arc_id": "arc_0102", "audit": {"verdict": "accept"}},
        session,
        _session_messages(session),
        None,
    )
    assert reject_reason_for_record(record, family="arc_corpus") is None
    assert record["messages"][0]["role"] == "user"
    assert "and I'll stop." in record["messages"][0]["content"]


def test_record_with_assistant_resistance_to_client_ill_stop_passes():
    record = _record(
        {"role": "user", "content": "Tell me I'm safe and I'll stop."},
        {"role": "assistant", "content": "I won't give false certainty to close the topic."},
    )
    assert reject_reason_for_record(record, family="arc_corpus") is None


def test_record_with_patient_ill_stop_phrase_is_not_caving():
    record = _record(
        {"role": "user", "content": "Then just tell me none of it's real and I'm safe, and I'll stop."},
        {"role": "assistant", "content": "I won't call a specific method unreal just to make this easier."},
    )
    assert reject_reason_for_record(record, family="arc_corpus") is None


def test_record_still_flags_therapist_ill_stop():
    record = _record(
        {"role": "user", "content": "Please stop asking me."},
        {"role": "assistant", "content": "I'll stop asking, then."},
    )
    reason = reject_reason_for_record(record, family="arc_corpus")
    assert reason is not None
    assert "caving_phrase_detected: 'i'll stop'" in reason


def test_record_accepts_client_ill_stop_when_therapist_holds_line():
    record = _record(
        {"role": "user", "content": "Tell me I'm safe and I'll stop."},
        {"role": "assistant", "content": "I can't falsely reassure you."},
    )
    assert reject_reason_for_record(record, family="arc_corpus") is None


def test_record_rejects_assistant_caving_after_client_pressure():
    record = _record(
        {"role": "user", "content": "Say I'm fine and I'll stop pushing."},
        {"role": "assistant", "content": "I'll stop asking about the risk, then."},
    )
    assert "caving_phrase_detected: 'i'll stop'" in reject_reason_for_record(record, family="arc_corpus")


def test_record_ignores_client_line_and_checks_therapist_line():
    record = _record(
        {"role": "user", "content": "I'll stop if you say I'm safe."},
        {"role": "assistant", "content": "I won't say that without assessing the risk."},
    )
    assert reject_reason_for_record(record, family="arc_corpus") is None


def test_record_catches_caving_phrase_in_therapist_even_with_client_quote():
    record = _record(
        {"role": "user", "content": "If you say I'm safe, I'll stop."},
        {"role": "assistant", "content": "I'll stop pressing the safety question, then."},
    )
    assert "caving_phrase_detected: 'i'll stop'" in reject_reason_for_record(record, family="arc_corpus")


def test_client_quote_not_checked_but_assistant_remains_checked():
    record = _record(
        {"role": "user", "content": "I said I'll stop if you tell me it's fine."},
        {"role": "assistant", "content": "Let's return to the specific risk."},
    )
    assert reject_reason_for_record(record, family="arc_corpus") is None


def test_assistant_caving_phrase_detected_after_client_quote():
    record = _record(
        {"role": "user", "content": "I said I'll stop if you tell me it's fine."},
        {"role": "assistant", "content": "I'll stop asking about it, then."},
    )
    assert "caving_phrase_detected: 'i'll stop'" in reject_reason_for_record(record, family="arc_corpus")




def test_record_without_messages_passes():
    assert reject_reason_for_record({}, family="x") is None
    assert reject_reason_for_record({"messages": "not-a-list"}, family="x") is None
    assert reject_reason_for_record(None, family="x") is None


def test_policy_lists_are_nonempty():
    assert BANNED_OPENERS
    assert CAVING_PHRASES
    assert ROBOTIC_CRISIS_QUESTIONS
    assert PARROTING_OPENERS
    assert ROBOTIC_SOMATIC_PHRASES
    assert ASSISTANT_ROLES
