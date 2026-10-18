#!/usr/bin/env python3
"""Tests for V7 dataset integrity tester module."""

from __future__ import annotations

import json
import os
import sys

# Ensure ai/ is on sys.path
_ai_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ai_root not in sys.path:
    sys.path.insert(0, _ai_root)

from dataset_pipeline.processors.integrity_test import (
    IntegrityReport,
    IntegrityViolation,
    check_record,
    format_report,
    run_integrity_test,
)


# ---------------------------------------------------------------------------
# Helper: build valid ChatML record
# ---------------------------------------------------------------------------
def _make_valid_record(**overrides: object) -> dict:
    base: dict = {
        "messages": [
            {"role": "system", "content": "You are a therapist."},
            {"role": "user", "content": "I feel anxious."},
            {"role": "assistant", "content": "Let's explore that."},
        ],
        "source": "test_dataset",
        "task_type": "therapy_response_generation",
    }
    base.update(overrides)
    return base


# ---------------------------------------------------------------------------
# check_record tests
# ---------------------------------------------------------------------------
class TestCheckRecord:
    def test_valid_record_no_violations(self):
        record = _make_valid_record()
        violations = check_record(record, 0)
        assert violations == []

    def test_missing_messages_field(self):
        record = {"source": "test", "task_type": "test"}
        violations = check_record(record, 0)
        assert len(violations) == 1
        assert violations[0].check == "chatml_structure"
        assert violations[0].severity == "error"

    def test_messages_not_list(self):
        record = _make_valid_record()
        record["messages"] = "not a list"
        violations = check_record(record, 0)
        assert any(v.check == "chatml_structure" for v in violations)

    def test_too_few_messages(self):
        record = _make_valid_record()
        record["messages"] = [{"role": "user", "content": "hello"}]
        violations = check_record(record, 0)
        assert any(v.check == "chatml_structure" for v in violations)

    def test_empty_content(self):
        record = _make_valid_record()
        record["messages"][1]["content"] = "   "
        violations = check_record(record, 0)
        assert any(v.check == "chatml_structure" for v in violations)

    def test_invalid_role(self):
        record = _make_valid_record()
        record["messages"][1]["role"] = "admin"
        violations = check_record(record, 0)
        assert any(v.check == "role_validity" and "Invalid role" in v.message for v in violations)

    def test_consecutive_same_role(self):
        record = _make_valid_record()
        record["messages"] = [
            {"role": "user", "content": "hello"},
            {"role": "user", "content": "again"},
            {"role": "assistant", "content": "hi"},
        ]
        violations = check_record(record, 0)
        assert any(v.check == "role_validity" and "Consecutive" in v.message for v in violations)

    def test_no_user_message(self):
        record = _make_valid_record()
        record["messages"] = [
            {"role": "system", "content": "You are a therapist."},
            {"role": "assistant", "content": "Hello!"},
        ]
        violations = check_record(record, 0)
        assert any(v.check == "role_validity" and "No user" in v.message for v in violations)

    def test_no_assistant_message(self):
        record = _make_valid_record()
        record["messages"] = [
            {"role": "system", "content": "You are a therapist."},
            {"role": "user", "content": "Hello!"},
        ]
        violations = check_record(record, 0)
        assert any(v.check == "role_validity" and "No assistant" in v.message for v in violations)

    def test_message_too_long_chars(self):
        record = _make_valid_record()
        record["messages"][1]["content"] = "x" * 20001
        violations = check_record(record, 0, max_message_chars=20000)
        assert any(v.check == "token_limits" for v in violations)

    def test_message_exceeds_token_limit(self):
        record = _make_valid_record()
        # 5000 chars / 4 = 1250 tokens > 1000 limit
        record["messages"][1]["content"] = "x" * 5000
        violations = check_record(record, 0, max_tokens_per_message=1000)
        assert any(v.check == "token_limits" and "token" in v.message for v in violations)

    def test_conversation_exceeds_token_limit(self):
        record = _make_valid_record()
        # Each message ~10000 chars → ~2500 tokens, total ~7500 > 5000
        for msg in record["messages"]:
            msg["content"] = "y" * 10000
        violations = check_record(record, 0, max_tokens_per_conversation=5000)
        assert any(v.check == "token_limits" and "Conversation" in v.message for v in violations)

    def test_mojibake_replacement_char(self):
        record = _make_valid_record()
        record["messages"][1]["content"] = "Hello\ufffdWorld"
        violations = check_record(record, 0)
        assert any(v.check == "utf8_encoding" for v in violations)

    def test_control_chars_detected(self):
        record = _make_valid_record()
        record["messages"][1]["content"] = "Hello\x00\x01World"
        violations = check_record(record, 0)
        assert any(v.check == "utf8_encoding" and "Control" in v.message for v in violations)

    def test_clean_utf8_no_violation(self):
        record = _make_valid_record()
        record["messages"][1]["content"] = "Héllo, wörld! 你好世界"
        violations = check_record(record, 0)
        assert not any(v.check == "utf8_encoding" for v in violations)

    def test_missing_v7_metadata(self):
        record = _make_valid_record()
        del record["source"]
        violations = check_record(record, 0)
        assert any(v.check == "v7_metadata" and "source" in v.message for v in violations)

    def test_skip_metadata_check(self):
        record = _make_valid_record()
        del record["source"]
        del record["task_type"]
        violations = check_record(record, 0, check_v7_metadata=False)
        assert not any(v.check == "v7_metadata" for v in violations)

    def test_v7_metadata_is_warning_not_error(self):
        record = _make_valid_record()
        del record["source"]
        violations = check_record(record, 0)
        meta_violations = [v for v in violations if v.check == "v7_metadata"]
        assert len(meta_violations) == 1
        assert meta_violations[0].severity == "warning"

    def test_structure_failure_short_circuits(self):
        """If structure is broken, other checks are skipped."""
        record = {"messages": "invalid"}
        violations = check_record(record, 0)
        assert len(violations) == 1
        assert violations[0].check == "chatml_structure"

    def test_shard_file_in_violation(self):
        record = _make_valid_record()
        record["messages"] = "invalid"
        violations = check_record(record, 0, shard_file="shard_0000.jsonl")
        assert violations[0].shard_file == "shard_0000.jsonl"


# ---------------------------------------------------------------------------
# run_integrity_test tests
# ---------------------------------------------------------------------------
class TestRunIntegrityTest:
    def _write_jsonl(self, path, records):
        with open(path, "w", encoding="utf-8") as f:
            for r in records:
                f.write(json.dumps(r) + "\n")

    def test_all_valid_records_pass(self, tmp_path):
        jsonl = tmp_path / "shard_0000.jsonl"
        self._write_jsonl(jsonl, [_make_valid_record() for _ in range(5)])
        report = run_integrity_test(tmp_path)
        assert report.passed
        assert report.total_records == 5
        assert report.errors == 0

    def test_invalid_record_fails(self, tmp_path):
        jsonl = tmp_path / "shard_0000.jsonl"
        records = [_make_valid_record(), {"messages": "bad"}]
        self._write_jsonl(jsonl, records)
        report = run_integrity_test(tmp_path)
        assert not report.passed
        assert report.errors == 1
        assert report.violations[0].check == "chatml_structure"

    def test_invalid_json_line(self, tmp_path):
        jsonl = tmp_path / "shard_0000.jsonl"
        with jsonl.open("w", encoding="utf-8") as f:
            f.write(json.dumps(_make_valid_record()) + "\n")
            f.write("{invalid json}\n")
        report = run_integrity_test(tmp_path)
        assert not report.passed
        assert any(v.check == "json_parse" for v in report.violations)

    def test_multiple_shards(self, tmp_path):
        for i in range(3):
            jsonl = tmp_path / f"shard_{i:04d}.jsonl"
            self._write_jsonl(jsonl, [_make_valid_record()])
        report = run_integrity_test(tmp_path)
        assert report.total_records == 3
        assert report.passed

    def test_excludes_report_files(self, tmp_path):
        shard = tmp_path / "shard_0000.jsonl"
        report_file = tmp_path / "report.jsonl"
        self._write_jsonl(shard, [_make_valid_record()])
        self._write_jsonl(report_file, [{"some": "report"}])
        report = run_integrity_test(tmp_path)
        assert report.total_records == 1  # report.jsonl excluded

    def test_single_file_input(self, tmp_path):
        jsonl = tmp_path / "data.jsonl"
        self._write_jsonl(jsonl, [_make_valid_record()])
        report = run_integrity_test(jsonl)
        assert report.total_records == 1
        assert report.passed

    def test_path_not_found(self):
        report = run_integrity_test("/nonexistent/path")
        assert not report.passed
        assert any(v.check == "file_access" for v in report.violations)

    def test_empty_directory(self, tmp_path):
        report = run_integrity_test(tmp_path)
        assert not report.passed
        assert any(v.check == "file_access" and "No JSONL" in v.message for v in report.violations)

    def test_warnings_dont_fail_report(self, tmp_path):
        jsonl = tmp_path / "shard_0000.jsonl"
        record = _make_valid_record()
        del record["source"]
        self._write_jsonl(jsonl, [record])
        report = run_integrity_test(tmp_path)
        assert report.passed  # warnings don't fail
        assert report.warnings == 1
        assert report.errors == 0

    def test_empty_lines_skipped(self, tmp_path):
        jsonl = tmp_path / "shard_0000.jsonl"
        with jsonl.open("w", encoding="utf-8") as f:
            f.write(json.dumps(_make_valid_record()) + "\n")
            f.write("\n")
            f.write(json.dumps(_make_valid_record()) + "\n")
        report = run_integrity_test(tmp_path)
        assert report.total_records == 2

    def test_record_not_dict(self, tmp_path):
        jsonl = tmp_path / "shard_0000.jsonl"
        with jsonl.open("w", encoding="utf-8") as f:
            f.write(json.dumps(_make_valid_record()) + "\n")
            f.write(json.dumps([1, 2, 3]) + "\n")  # JSON array, not dict
        report = run_integrity_test(tmp_path)
        assert not report.passed
        assert any(v.check == "json_parse" and "not a JSON object" in v.message for v in report.violations)


# ---------------------------------------------------------------------------
# format_report tests
# ---------------------------------------------------------------------------
class TestFormatReport:
    def test_empty_report_pass(self):
        report = IntegrityReport(total_records=10, passed=True)
        text = format_report(report)
        assert "PASS" in text
        assert "10" in text

    def test_report_with_violations(self):
        report = IntegrityReport(total_records=5, passed=False, errors=2, warnings=1)
        report.violations = [
            IntegrityViolation(
                severity="error", check="role_validity",
                message="Invalid role 'admin'", record_index=3,
                shard_file="shard_0000.jsonl",
            ),
            IntegrityViolation(
                severity="error", check="token_limits",
                message="Message exceeds 4096 tokens", record_index=1,
            ),
            IntegrityViolation(
                severity="warning", check="v7_metadata",
                message="Missing source", record_index=2,
            ),
        ]
        text = format_report(report)
        assert "FAIL" in text
        assert "2 errors" in text
        assert "1 warnings" in text
        assert "role_validity" in text
        assert "token_limits" in text
        assert "v7_metadata" in text


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------
class TestEdgeCases:
    def test_system_message_can_be_first(self):
        """System message at index 0 should not violate alternation."""
        record = _make_valid_record()
        violations = check_record(record, 0)
        assert not any(v.check == "role_validity" for v in violations)

    def test_multiple_system_messages_allowed(self):
        """Multiple system messages should not trigger alternation violation."""
        record = _make_valid_record()
        record["messages"] = [
            {"role": "system", "content": "You are a therapist."},
            {"role": "system", "content": "Additional context."},
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there"},
        ]
        violations = check_record(record, 0)
        assert not any(v.check == "role_validity" for v in violations)

    def test_non_string_content_rejected(self):
        record = _make_valid_record()
        record["messages"][1]["content"] = 123
        violations = check_record(record, 0)
        assert any(v.check == "chatml_structure" for v in violations)

    def test_tab_newline_not_control_chars(self):
        record = _make_valid_record()
        record["messages"][1]["content"] = "Hello\tWorld\nNext line"
        violations = check_record(record, 0)
        assert not any(v.check == "utf8_encoding" for v in violations)

    def test_unicode_emoji_passes(self):
        record = _make_valid_record()
        record["messages"][1]["content"] = "I feel 😢 about this situation"
        violations = check_record(record, 0)
        assert not any(v.check == "utf8_encoding" for v in violations)
