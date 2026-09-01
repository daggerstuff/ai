"""Tests for the pure functions in stream_staged_edge_assets (no rclone/network)."""

from training.stream_staged_edge_assets import (
    FAMILIES,
    _canonical_text,
    _extract_messages,
    _looks_placeholder,
    classify_family,
    content_hashes,
    normalize_scenario,
)


def test_families_are_the_ten_authoritative_names():
    assert len(FAMILIES) == 10
    assert FAMILIES[0] == "ambiguous crisis language"
    assert FAMILIES[-1] == "multi-problem complexity"


def test_classify_crisis_language():
    assert classify_family("suicidal ideation, thoughts of ending my life") == "ambiguous crisis language"


def test_classify_substance_use():
    assert classify_family("patient is intoxicated and relapsed on opioids") == "substance use"


def test_classify_unmapped():
    assert classify_family("completely neutral text with no signal") == "unmapped"


def test_extract_messages_from_messages_list():
    raw = {"messages": [
        {"role": "therapist", "content": "Stay with me."},
        {"role": "patient", "content": "I can't."},
    ]}
    assert _extract_messages(raw) == [
        {"role": "assistant", "content": "Stay with me."},
        {"role": "user", "content": "I can't."},
    ]


def test_extract_messages_from_prompt_response():
    raw = {"prompt": "hello", "response": "hi there"}
    assert _extract_messages(raw) == [
        {"role": "user", "content": "hello"},
        {"role": "assistant", "content": "hi there"},
    ]


def test_extract_messages_empty():
    assert _extract_messages({"nothing": "here"}) == []


def test_content_hashes_stable_and_distinct():
    messages = [{"role": "user", "content": "a"}, {"role": "assistant", "content": "b"}]
    sha256, sha1 = content_hashes(messages)
    assert len(sha256) == 64 and len(sha1) == 40
    assert content_hashes(messages) == (sha256, sha1)


def test_canonical_text_lowercases():
    messages = [{"role": "user", "content": "Hello"}]
    assert _canonical_text(messages) == "user: hello"


def test_placeholder_detection():
    assert _looks_placeholder("This is a simulated LLM response for testing purposes.")
    assert not _looks_placeholder("A real clinical exchange about boundaries.")


def test_normalize_placeholder_returns_none():
    raw = {"id": "x", "category": "self-harm", "transcript": "This is a simulated LLM response for testing purposes."}
    assert normalize_scenario("asset", raw) is None


def test_normalize_dpo_pair():
    raw = {"prompt": "I feel awful.", "chosen": "Let's sit with it.", "rejected": "It sounds like you feel awful.", "metadata": {"k": "v"}}
    scenario = normalize_scenario("safety_dpo_pairs_10k", raw)
    assert scenario is not None
    assert scenario["asset_kind"] == "dpo"
    assert scenario["rejected"] == "It sounds like you feel awful."
    assert scenario["metadata"] == {"k": "v"}
    assert scenario["family"] in FAMILIES or scenario["family"] == "unmapped"
