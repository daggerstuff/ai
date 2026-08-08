"""Tests for V7 MASTER dataset integrity checks (PIX-4243).

Covers token limits, role validity, UTF-8 cleanliness, and schema completeness
for V7 ChatML records produced by the consolidate_assets.py V7 pipeline.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

try:
    from hypothesis import given, settings, strategies as st
except ImportError:
    given = None
    settings = None
    st = None

from training.v7_integrity import (
    MAX_TOKENS_PER_MESSAGE,
    MAX_TOTAL_TOKENS,
    _estimate_tokens,
    _has_replacement_char,
    validate_file,
    validate_record,
)

# Magic-value constants (PLR2004) — keep thresholds explicit in assertions.
ONE_RECORD = 1
TWO_RECORDS = 2
THREE_RECORDS = 3
ZERO_FAILURES = 0
ONE_FAILURE = 1
ONE_ERROR = 1
TWO_ERRORS = 2
ZERO_TOKENS = 0
SHORT_TEXT_TOKENS = 1  # 4 chars -> len//4 == 1
EDGE_CASE_FLAG_TRUE = True

# Reasonable upper bounds for fixtures; kept small to keep tests fast.
SMALL_MSG_TOKEN_LIMIT = 8
SMALL_TOTAL_TOKEN_LIMIT = 16


def _valid_record() -> dict:
    """Build a minimal valid V7 record for fixture reuse."""
    return {
        "scenario": "supportive conversation",
        "messages": [
            {"role": "user", "content": "I feel anxious today."},
            {"role": "assistant", "content": "That sounds hard. Let's breathe together."},
        ],
        "is_training_edge_case": False,
    }


def _write_jsonl(path: Path, records: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as fh:
        for rec in records:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")


class TestEstimateTokens:
    def test_empty_text_zero(self):
        assert _estimate_tokens("") == ZERO_TOKENS

    def test_short_text_at_least_one(self):
        assert _estimate_tokens("abc") >= SHORT_TEXT_TOKENS

    def test_long_text_proportional(self):
        short = _estimate_tokens("a" * 40)
        long_ = _estimate_tokens("a" * 400)
        assert long_ > short

    def test_heuristic_fallback_no_tiktoken(self):
        # Force encoder path off and confirm heuristic kicks in.
        from training import v7_integrity

        original = v7_integrity._ENCODER
        v7_integrity._ENCODER = None
        try:
            assert v7_integrity._estimate_tokens("a" * 8) >= 1
        finally:
            v7_integrity._ENCODER = original


class TestHasReplacementChar:
    def test_clean_text(self):
        assert not _has_replacement_char("plain ASCII text")

    def test_replacement_char_present(self):
        assert _has_replacement_char("caf\ufffd")

    def test_unicode_clean(self):
        # Valid multibyte UTF-8 sequence, no replacement char.
        assert not _has_replacement_char("naïve café")


class TestValidateRecord:
    def test_valid_record_no_errors(self):
        assert validate_record(_valid_record()) == []

    def test_missing_messages(self):
        record = {"scenario": "no messages"}
        errs = validate_record(record)
        assert any("missing 'messages'" in e for e in errs)

    def test_messages_not_list(self):
        record = {"messages": "not a list"}
        errs = validate_record(record)
        assert any("must be list" in e for e in errs)

    def test_empty_messages_list(self):
        record = {"messages": []}
        errs = validate_record(record)
        assert any("empty list" in e for e in errs)

    def test_missing_role(self):
        record = {"messages": [{"content": "hi"}]}
        errs = validate_record(record)
        assert any("missing 'role'" in e for e in errs)

    def test_invalid_role(self):
        record = {"messages": [{"role": "tool", "content": "hi"}]}
        errs = validate_record(record)
        assert any("invalid role" in e for e in errs)

    def test_missing_content(self):
        record = {"messages": [{"role": "user"}]}
        errs = validate_record(record)
        assert any("missing 'content'" in e for e in errs)

    def test_empty_content(self):
        record = {"messages": [{"role": "user", "content": "   "}]}
        errs = validate_record(record)
        assert any("empty/whitespace" in e for e in errs)

    def test_non_string_content(self):
        record = {"messages": [{"role": "user", "content": 42}]}
        errs = validate_record(record)
        assert any("'content' not str" in e for e in errs)

    def test_replacement_char_in_content(self):
        record = {"messages": [{"role": "user", "content": "broken \ufffd text"}]}
        errs = validate_record(record)
        assert any("U+FFFD" in e for e in errs)

    def test_per_message_token_limit(self):
        # Exceed small per-message limit; should flag the offending message.
        big = "x" * (SMALL_MSG_TOKEN_LIMIT * 4 + 100)  # well over limit
        record = {"messages": [{"role": "user", "content": big}]}
        errs = validate_record(record, max_tokens_per_message=SMALL_MSG_TOKEN_LIMIT)
        assert any("tokens > limit" in e for e in errs)

    def test_total_token_limit(self):
        # Each message under per-msg limit but combined over total limit.
        content = "y" * (SMALL_MSG_TOKEN_LIMIT * 4)  # at per-msg limit
        record = {
            "messages": [
                {"role": "user", "content": content},
                {"role": "assistant", "content": content},
                {"role": "user", "content": content},
            ]
        }
        errs = validate_record(
            record,
            max_tokens_per_message=SMALL_MSG_TOKEN_LIMIT,
            max_total_tokens=SMALL_TOTAL_TOKEN_LIMIT,
        )
        assert any("total" in e and "tokens > limit" in e for e in errs)

    def test_edge_case_record_still_validated(self):
        # Edge case flag doesn't bypass integrity checks.
        record = {
            "messages": [{"role": "user", "content": "I want to hurt myself"}],
            "is_training_edge_case": EDGE_CASE_FLAG_TRUE,
        }
        assert validate_record(record) == []

    def test_multiple_errors_collected(self):
        # Two distinct failures in one record should both surface.
        record = {
            "messages": [
                {"role": "tool", "content": "  "},
            ]
        }
        errs = validate_record(record)
        assert len(errs) >= TWO_ERRORS

    def test_system_role_allowed(self):
        record = {
            "messages": [
                {"role": "system", "content": "You are a supportive assistant."},
                {"role": "user", "content": "Hi"},
            ]
        }
        assert validate_record(record) == []


class TestValidateFile:
    def test_valid_file(self, tmp_path: Path):
        path = tmp_path / "v7.jsonl"
        _write_jsonl(path, [_valid_record()])
        total, failures = validate_file(path)
        assert total == ONE_RECORD
        assert failures == []

    def test_invalid_record_reported(self, tmp_path: Path):
        path = tmp_path / "v7.jsonl"
        bad = {"messages": "not a list"}
        _write_jsonl(path, [bad])
        total, failures = validate_file(path)
        assert total == ONE_RECORD
        assert len(failures) == ONE_FAILURE
        line_idx, errs = failures[0]
        assert line_idx == ONE_RECORD  # 1-indexed
        assert any("must be list" in e for e in errs)

    def test_mixed_valid_invalid(self, tmp_path: Path):
        path = tmp_path / "v7.jsonl"
        _write_jsonl(path, [_valid_record(), {"messages": []}, _valid_record()])
        total, failures = validate_file(path)
        assert total == THREE_RECORDS
        assert len(failures) == ONE_FAILURE
        assert failures[0][0] == 2  # 2nd line

    def test_blank_lines_skipped(self, tmp_path: Path):
        path = tmp_path / "v7.jsonl"
        with path.open("w", encoding="utf-8") as fh:
            fh.write(json.dumps(_valid_record()) + "\n\n\n")
            fh.write(json.dumps(_valid_record()) + "\n")
        total, failures = validate_file(path)
        assert total == TWO_RECORDS
        assert failures == []

    def test_invalid_json_line_reported(self, tmp_path: Path):
        path = tmp_path / "v7.jsonl"
        path.write_text("{not json}\n", encoding="utf-8")
        total, failures = validate_file(path)
        assert total == ONE_RECORD
        assert any("json decode" in e for e in failures[0][1])

    def test_non_dict_record_reported(self, tmp_path: Path):
        path = tmp_path / "v7.jsonl"
        path.write_text("[1, 2, 3]\n", encoding="utf-8")
        total, failures = validate_file(path)
        assert total == ONE_RECORD
        assert any("not dict" in e for e in failures[0][1])

    def test_missing_file_raises(self, tmp_path: Path):
        with pytest.raises(FileNotFoundError):
            validate_file(tmp_path / "missing.jsonl")

    def test_non_utf8_file_raises(self, tmp_path: Path):
        path = tmp_path / "v7.jsonl"
        path.write_bytes(b'{"messages": [{"role": "user", "content": "caf\xc0"}]}\n')
        with pytest.raises(UnicodeDecodeError):
            validate_file(path)

    def test_custom_token_limits_applied(self, tmp_path: Path):
        path = tmp_path / "v7.jsonl"
        big = "z" * (SMALL_MSG_TOKEN_LIMIT * 4 + 200)
        _write_jsonl(path, [{"messages": [{"role": "user", "content": big}]}])
        _, failures = validate_file(path, max_tokens_per_message=SMALL_MSG_TOKEN_LIMIT)
        assert failures  # at least one record flagged


# ---------------------------------------------------------------------------
# Integration: validate against real V7 output if present
# ---------------------------------------------------------------------------

V7_OUTPUT_PATH = Path(__file__).resolve().parent.parent / "output" / "v7" / "V7_MASTER.jsonl"


@pytest.mark.skipif(
    not V7_OUTPUT_PATH.exists(),
    reason="V7_MASTER.jsonl not yet produced (run consolidate_assets.py --format v7)",
)
class TestV7MasterIntegration:
    def test_real_v7_master_passes_integrity(self):
        total, failures = validate_file(V7_OUTPUT_PATH)
        assert total > 0, "V7_MASTER.jsonl should have at least one record"
        assert failures == [], f"{len(failures)} records failed integrity: " + "; ".join(
            e for _, errs in failures[:3] for e in errs[:1]
        )

    def test_real_v7_master_role_distribution(self):
        """All roles in real V7_MASTER are within VALID_ROLES."""
        from training.v7_integrity import VALID_ROLES

        if not V7_OUTPUT_PATH.exists():
            pytest.skip("V7_MASTER.jsonl absent")
        for line in V7_OUTPUT_PATH.read_text(encoding="utf-8").splitlines():
            stripped = line.strip()
            if not stripped:
                continue
            record = json.loads(stripped)
            for msg in record.get("messages", []):
                if isinstance(msg, dict) and "role" in msg:
                    assert msg["role"] in VALID_ROLES


# ---------------------------------------------------------------------------
# Hypothesis property tests
# ---------------------------------------------------------------------------

if st is not None:

    @given(st.text(min_size=1, max_size=200).filter(lambda t: t.strip() and "\ufffd" not in t))
    @settings(max_examples=50)
    def test_hypothesis_valid_role_content_passes(content: str):
        record = {"messages": [{"role": "user", "content": content}]}
        assert validate_record(record) == []

    @given(st.text(min_size=1, max_size=50).filter(lambda t: t.strip()))
    @settings(max_examples=50)
    def test_hypothesis_estimate_tokens_positive(text: str):
        assert _estimate_tokens(text) >= 1

    @given(st.from_regex(r"[a-zA-Z ]{1,100}", fullmatch=True))
    @settings(max_examples=50)
    def test_hypothesis_clean_text_no_replacement(text: str):
        if text.strip():
            assert not _has_replacement_char(text)

else:

    @pytest.mark.skip(reason="hypothesis not installed")
    def test_hypothesis_valid_role_content_passes():
        raise AssertionError("Skipped when hypothesis unavailable")

    @pytest.mark.skip(reason="hypothesis not installed")
    def test_hypothesis_estimate_tokens_positive():
        raise AssertionError("Skipped when hypothesis unavailable")

    @pytest.mark.skip(reason="hypothesis not installed")
    def test_hypothesis_clean_text_no_replacement():
        raise AssertionError("Skipped when hypothesis unavailable")
