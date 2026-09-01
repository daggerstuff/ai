"""Tests for the pure functions in build_clinical_backed_examples (no network)."""

from pathlib import Path

from training.build_clinical_backed_examples import (
    STAGE1,
    STAGE2,
    _EmitState,
    _book_stage,
    _is_dpo,
    _process,
    normalize_benchmark_record,
    normalize_book_record,
)


def test_book_stage_routes_clinical_to_therapeutic():
    assert _book_stage("Complex PTSD_ From Surviving to Thriving by Pete Walker") == STAGE2
    assert _book_stage("Internal Family Systems Therapy by Richard C. Schwartz") == STAGE2
    assert _book_stage("DSMV") == STAGE2


def test_book_stage_routes_self_help_to_foundation():
    assert _book_stage("Art of Speed Reading Body Language") == STAGE1
    assert _book_stage("How to Analyze People with Dark Psychology") == STAGE1


def test_is_dpo():
    assert _is_dpo({"prompt": "p", "chosen": "c", "rejected": "r"})
    assert not _is_dpo({"messages": []})


def test_normalize_book_record_builds_chatml(tmp_path):
    item = {
        "instruction": "I feel stuck.",
        "output": "Let's name what 'stuck' is doing for you right now.",
        "metadata": {"source_book": "Complex PTSD", "source_type": "clinical_literature"},
    }
    record = normalize_book_record(item, tmp_path / "Complex PTSD.jsonl")
    assert record is not None
    assert record["stage"] == STAGE2
    assert record["tier"] == "T1_GOLD"
    assert record["source"] == "clinical_book"
    assert [m["role"] for m in record["messages"]] == ["system", "user", "assistant"]
    assert record["sha256"] and record["sha1"]


def test_normalize_book_record_skips_empty(tmp_path):
    assert normalize_book_record({"instruction": "", "output": ""}, tmp_path / "x.jsonl") is None


def test_normalize_dpo_record_promotes_chosen():
    raw = {"prompt": "I'm scared.", "chosen": "What's one thing you can control right now?",
           "rejected": "It sounds like you're scared.", "metadata": {"domain": "identity"}}
    record = normalize_benchmark_record("safety_dpo_pairs_10k", raw)
    assert record is not None
    assert record["stage"] == STAGE1
    assert record["diagnostic_tag"] == "identity"
    assert [m["role"] for m in record["messages"]] == ["system", "user", "assistant"]
    assert record["messages"][-1]["content"] == raw["chosen"]


def test_normalize_chatml_record_preserves_metadata():
    raw = {
        "messages": [
            {"role": "system", "content": "DSM-5 AUD persona."},
            {"role": "user", "content": "I don't have a problem."},
            {"role": "assistant", "content": "Tell me what a typical week of drinking looks like."},
        ],
        "source": "clinical_redteam",
        "persona_name": "Liam",
        "provenance": {"k": "v"},
    }
    record = normalize_benchmark_record("clinical_redteam", raw)
    assert record is not None
    assert record["stage"] == STAGE2
    assert record["tier"] == "benchmark"
    assert record["persona_name"] == "Liam"
    assert record["provenance"] == {"k": "v"}
    assert record["sha256"] and record["sha1"]


def test_process_gates_sycophantic_assistant(tmp_path):
    out = tmp_path / "staging.jsonl"
    reject = tmp_path / "rejections.jsonl"
    state = _EmitState(seen=set(), out_path=out, reject_path=reject, summary={
        "books": 0, "benchmark": 0, "emitted": 0, "rejected": 0,
        "duplicates": 0, "placeholders": 0, "stage1": 0, "stage2": 0,
    })
    record = {
        "messages": [
            {"role": "user", "content": "I feel awful."},
            {"role": "assistant", "content": "It sounds like you feel awful."},
        ],
        "source": "clinical_book",
        "stage": STAGE2,
        "sha256": "abc",
        "sha1": "def",
    }
    _process(record, "clinical_book", state)
    assert state.summary["rejected"] == 1
    assert state.summary["emitted"] == 0
    assert not out.exists()
    assert reject.exists()


def test_process_dedups_and_emits(tmp_path):
    out = tmp_path / "staging.jsonl"
    reject = tmp_path / "rejections.jsonl"
    state = _EmitState(seen={"abc"}, out_path=out, reject_path=reject, summary={
        "books": 0, "benchmark": 0, "emitted": 0, "rejected": 0,
        "duplicates": 0, "placeholders": 0, "stage1": 0, "stage2": 0,
    })
    record = {
        "messages": [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "What brings you here today?"},
        ],
        "source": "clinical_book",
        "stage": STAGE1,
        "sha256": "abc",
        "sha1": "def",
    }
    _process(record, "clinical_book", state)
    assert state.summary["duplicates"] == 1
    assert not out.exists()

    record["sha256"] = "new"
    record["sha1"] = "new1"
    _process(record, "clinical_book", state)
    assert state.summary["emitted"] == 1
    assert state.summary["stage1"] == 1
    assert out.exists()
    lines = out.read_text().strip().splitlines()
    assert len(lines) == 1