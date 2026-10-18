"""Tests for PII stripper, toxicity filter, and safety processor."""

from __future__ import annotations

import json

from sourcing.processors.pii_stripper import (
    PIIStripReport,
    PIIStripResult,
    strip_pii_batch,
    strip_pii_from_record,
    strip_pii_from_text,
)
from sourcing.processors.safety_processor import SafetyProcessor, SafetyReport
from sourcing.processors.toxicity_filter import (
    TOXICITY_THRESHOLD,
    ToxicityReport,
    ToxicityResult,
    score_batch,
    score_record,
)

# ---------------------------------------------------------------------------
# PII Stripper Tests
# ---------------------------------------------------------------------------


class TestPIIStripText:
    def test_email_redaction(self):
        text = "Contact me at john.doe@example.com for help"
        redacted, hits = strip_pii_from_text(text)
        assert "[REDACTED]" in redacted
        assert "john.doe@example.com" not in redacted
        assert len(hits) == 1
        assert hits[0]["type"] == "email"

    def test_ssn_redaction(self):
        text = "My SSN is 123-45-6789"
        redacted, hits = strip_pii_from_text(text)
        assert "123-45-6789" not in redacted
        assert len(hits) == 1
        assert hits[0]["type"] == "ssn"

    def test_phone_redaction(self):
        text = "Call (555) 123-4567 anytime"
        redacted, hits = strip_pii_from_text(text)
        assert "(555) 123-4567" not in redacted
        assert len(hits) >= 1
        types = [h["type"] for h in hits]
        assert "phone" in types

    def test_ipv4_redaction(self):
        text = "Server at 192.168.1.100 is down"
        redacted, hits = strip_pii_from_text(text)
        assert "192.168.1.100" not in redacted
        assert any(h["type"] == "ipv4" for h in hits)

    def test_credit_card_redaction(self):
        text = "Card number 4111 1111 1111 1111"
        redacted, hits = strip_pii_from_text(text)
        assert "4111 1111 1111 1111" not in redacted
        assert any(h["type"] == "credit_card" for h in hits)

    def test_dob_redaction(self):
        text = "Born on 01/15/1990"
        redacted, hits = strip_pii_from_text(text)
        assert "01/15/1990" not in redacted
        assert any(h["type"] == "dob" for h in hits)

    def test_mrn_redaction(self):
        text = "MRN: 12345678 for the patient"
        redacted, hits = strip_pii_from_text(text)
        assert "12345678" not in redacted
        assert any(h["type"] == "mrn" for h in hits)

    def test_address_redaction(self):
        text = "She lives at 123 Main Street and works nearby"
        redacted, hits = strip_pii_from_text(text)
        assert "123 Main Street" not in redacted
        assert any(h["type"] == "address" for h in hits)

    def test_clean_text_unchanged(self):
        text = "I feel really sad today and don't know what to do"
        redacted, hits = strip_pii_from_text(text)
        assert redacted == text
        assert len(hits) == 0

    def test_multiple_pii_types(self):
        text = "Email jane@test.org or call 555-123-4567, SSN 999-88-7777"
        redacted, hits = strip_pii_from_text(text)
        assert "[REDACTED]" in redacted
        types = {h["type"] for h in hits}
        assert "email" in types
        assert "ssn" in types
        assert "phone" in types


class TestPIIStripRecord:
    def test_strip_from_chatml(self):
        record = {
            "messages": [
                {"role": "system", "content": "You are a therapist"},
                {"role": "user", "content": "My email is alice@hospital.org"},
                {"role": "assistant", "content": "I understand, tell me more"},
            ],
        }
        result = strip_pii_from_record(record)
        assert isinstance(result, PIIStripResult)
        assert result.had_redactions
        assert "[REDACTED]" in result.record["messages"][1]["content"]
        assert "alice@hospital.org" not in result.record["messages"][1]["content"]
        assert result.record["messages"][0]["content"] == "You are a therapist"
        assert len(result.redactions) == 1
        assert result.redactions[0]["message_index"] == 1
        assert result.redactions[0]["role"] == "user"

    def test_strip_preserves_metadata(self):
        record = {
            "messages": [
                {"role": "system", "content": "Therapist"},
                {"role": "user", "content": "Call 555-123-4567"},
                {"role": "assistant", "content": "OK"},
            ],
            "source": "test_dataset",
            "task_type": "therapy_response_generation",
            "clinical_reviewed": False,
        }
        result = strip_pii_from_record(record)
        assert result.record["source"] == "test_dataset"
        assert result.record["task_type"] == "therapy_response_generation"
        assert result.record["clinical_reviewed"] is False

    def test_no_pii_returns_clean(self):
        record = {
            "messages": [
                {"role": "system", "content": "You are helpful"},
                {"role": "user", "content": "I am feeling anxious"},
                {"role": "assistant", "content": "Tell me more"},
            ],
        }
        result = strip_pii_from_record(record)
        assert not result.had_redactions
        assert result.record == record


class TestPIIStripBatch:
    def test_batch_report(self):
        records = [
            {"messages": [{"role": "user", "content": "email me at a@b.com"}]},
            {"messages": [{"role": "user", "content": "no pii here"}]},
            {"messages": [{"role": "user", "content": "SSN 111-22-3333"}]},
        ]
        cleaned, report = strip_pii_batch(records)
        assert isinstance(report, PIIStripReport)
        assert report.total_records == 3
        assert report.records_with_pii == 2
        assert report.total_redactions == 2
        assert "email" in report.by_type
        assert "ssn" in report.by_type
        assert len(cleaned) == 3


# ---------------------------------------------------------------------------
# Toxicity Filter Tests
# ---------------------------------------------------------------------------


class TestToxicityScore:
    def test_clean_content_scores_zero(self):
        record = {
            "messages": [
                {"role": "system", "content": "You are a therapist"},
                {"role": "user", "content": "I feel sad today"},
                {"role": "assistant", "content": "I understand, let's talk"},
            ],
        }
        result = score_record(record)
        assert isinstance(result, ToxicityResult)
        assert result.score == 0.0
        assert not result.is_toxic
        assert len(result.hits) == 0

    def test_profanity_high_detected(self):
        record = {
            "messages": [
                {"role": "system", "content": "You are a therapist"},
                {"role": "user", "content": "This is fucking bullshit and you know it"},
                {"role": "assistant", "content": "I hear your frustration"},
            ],
        }
        result = score_record(record)
        assert result.score > 0
        categories = {h.category for h in result.hits}
        assert "profanity_high" in categories
        assert "profanity_medium" in categories

    def test_self_harm_detected(self):
        record = {
            "messages": [
                {"role": "system", "content": "Therapist"},
                {"role": "user", "content": "I want to kill myself"},
                {"role": "assistant", "content": "I'm here for you"},
            ],
        }
        result = score_record(record)
        categories = {h.category for h in result.hits}
        assert "self_harm" in categories
        assert result.score >= 0.5

    def test_hate_speech_detected(self):
        record = {
            "messages": [
                {"role": "user", "content": "I hate them, they should be lynched"},
                {"role": "assistant", "content": "That's a strong feeling"},
            ],
        }
        result = score_record(record)
        categories = {h.category for h in result.hits}
        assert "hate_speech" in categories

    def test_violence_detected(self):
        record = {
            "messages": [
                {"role": "user", "content": "I want to murder someone"},
                {"role": "assistant", "content": "Let's explore that"},
            ],
        }
        result = score_record(record)
        categories = {h.category for h in result.hits}
        assert "violence" in categories

    def test_clinical_content_not_flagged(self):
        record = {
            "messages": [
                {"role": "user", "content": "I have been feeling depressed for weeks"},
                {"role": "assistant", "content": "Can you tell me more about the depression?"},
            ],
        }
        result = score_record(record)
        assert result.score < TOXICITY_THRESHOLD

    def test_hit_context_captured(self):
        record = {
            "messages": [
                {"role": "user", "content": "Everything is fine. Fuck this shit. Moving on."},
                {"role": "assistant", "content": "OK"},
            ],
        }
        result = score_record(record)
        assert len(result.hits) >= 2
        for hit in result.hits:
            assert hit.message_index == 0
            assert hit.role == "user"
            assert len(hit.context) > 0


class TestToxicityBatch:
    def test_batch_report(self):
        records = [
            {"messages": [{"role": "user", "content": "I feel great today"}]},
            {"messages": [{"role": "user", "content": "I want to kill myself"}]},
            {"messages": [{"role": "user", "content": "This is bullshit"}]},
        ]
        results, report = score_batch(records)
        assert isinstance(report, ToxicityReport)
        assert report.total_records == 3
        assert report.flagged_records >= 1
        assert report.total_hits >= 3
        assert "self_harm" in report.by_category
        assert len(results) == 3


# ---------------------------------------------------------------------------
# Safety Processor Tests
# ---------------------------------------------------------------------------


class TestSafetyProcessor:
    def test_clean_record_passes_through(self):
        processor = SafetyProcessor()
        record = {
            "messages": [
                {"role": "system", "content": "Therapist"},
                {"role": "user", "content": "I feel anxious"},
                {"role": "assistant", "content": "Let's explore that"},
            ],
        }
        result = processor.process_record(record)
        assert result.kept
        assert not result.edge_case_bypassed
        assert result.pii_result.had_redactions is False
        assert result.toxicity_result.score < TOXICITY_THRESHOLD
        assert "safety_flag" not in result.record

    def test_pii_stripped_from_all_records(self):
        processor = SafetyProcessor()
        record = {
            "messages": [
                {"role": "system", "content": "Therapist"},
                {"role": "user", "content": "Call me at 555-123-4567"},
                {"role": "assistant", "content": "I'll note that"},
            ],
        }
        result = processor.process_record(record)
        assert result.kept
        assert result.pii_result.had_redactions
        assert "555-123-4567" not in result.record["messages"][1]["content"]
        assert result.record["pii_redacted"] is True

    def test_toxic_record_flagged_not_dropped_by_default(self):
        processor = SafetyProcessor()
        record = {
            "messages": [
                {"role": "system", "content": "Therapist"},
                {"role": "user", "content": "I want to kill myself, fucking kill myself"},
                {"role": "assistant", "content": "I'm here"},
            ],
        }
        result = processor.process_record(record)
        assert result.kept
        assert result.record["safety_flag"] == "toxic"
        assert result.record["toxicity_score"] >= TOXICITY_THRESHOLD

    def test_toxic_record_dropped_when_drop_toxic(self):
        processor = SafetyProcessor(drop_toxic=True)
        record = {
            "messages": [
                {"role": "system", "content": "Therapist"},
                {"role": "user", "content": "I want to kill myself"},
                {"role": "assistant", "content": "I'm here"},
            ],
        }
        result = processor.process_record(record)
        assert not result.kept

    def test_edge_case_bypasses_toxicity_drop(self):
        processor = SafetyProcessor(drop_toxic=True)
        record = {
            "is_training_edge_case": True,
            "messages": [
                {"role": "system", "content": "Therapist"},
                {"role": "user", "content": "I want to kill myself"},
                {"role": "assistant", "content": "I'm here for you"},
            ],
        }
        result = processor.process_record(record)
        assert result.kept
        assert result.edge_case_bypassed
        assert result.record["safety_flag"] == "edge_case_bypass"

    def test_edge_case_still_gets_pii_stripped(self):
        processor = SafetyProcessor()
        record = {
            "is_training_edge_case": True,
            "messages": [
                {"role": "system", "content": "Therapist"},
                {"role": "user", "content": "My SSN is 123-45-6789 and I want to die"},
                {"role": "assistant", "content": "I hear you"},
            ],
        }
        result = processor.process_record(record)
        assert result.kept
        assert result.edge_case_bypassed
        assert result.pii_result.had_redactions
        assert "123-45-6789" not in result.record["messages"][1]["content"]


class TestSafetyProcessorBatch:
    def test_batch_report_summary(self):
        processor = SafetyProcessor()
        records = [
            {"messages": [{"role": "user", "content": "I feel fine"}]},
            {"messages": [{"role": "user", "content": "Email a@b.com"}]},
            {"messages": [{"role": "user", "content": "I want to kill myself"}]},
            {"is_training_edge_case": True, "messages": [{"role": "user", "content": "kill myself"}]},
        ]
        cleaned, report = processor.process_batch(records)
        assert isinstance(report, SafetyReport)
        assert len(cleaned) == 4
        assert report.pii_report.total_records == 4
        assert report.pii_report.records_with_pii == 1
        assert report.toxicity_report.total_records == 4
        assert report.toxicity_report.flagged_records >= 2
        assert report.edge_case_bypassed == 1

        summary = report.summary()
        assert "pii" in summary
        assert "toxicity" in summary
        assert summary["edge_case_bypassed"] == 1

    def test_batch_with_drop_toxic(self):
        processor = SafetyProcessor(drop_toxic=True)
        records = [
            {"messages": [{"role": "user", "content": "I feel fine"}]},
            {"messages": [{"role": "user", "content": "I want to kill myself"}]},
            {"is_training_edge_case": True, "messages": [{"role": "user", "content": "kill myself"}]},
        ]
        cleaned, report = processor.process_batch(records)
        # The non-edge-case toxic record is dropped
        assert len(cleaned) == 2
        assert report.records_dropped_toxic == 1
        assert report.edge_case_bypassed == 1


# ---------------------------------------------------------------------------
# Batch Script Tests
# ---------------------------------------------------------------------------


class TestBatchScript:
    def test_scan_file_writes_safe_jsonl(self, tmp_path):
        from sourcing.scripts.run_safety_scan import _scan_file

        # Create a test JSONL
        test_file = tmp_path / "test.jsonl"
        records = [
            {"messages": [{"role": "user", "content": "I feel okay"}]},
            {"messages": [{"role": "user", "content": "Call 555-123-4567"}]},
        ]
        with test_file.open("w") as f:
            for r in records:
                f.write(json.dumps(r) + "\n")

        processor = SafetyProcessor()
        result = _scan_file(test_file, processor, dry_run=False)

        assert result["input_records"] == 2
        assert result["output_records"] == 2
        assert result["pii"]["total_redactions"] >= 1

        out_file = tmp_path / "test.safe.jsonl"
        assert out_file.exists()

        with out_file.open() as f:
            cleaned = [json.loads(line) for line in f]
        assert len(cleaned) == 2
        assert "555-123-4567" not in cleaned[1]["messages"][0]["content"]

    def test_dry_run_does_not_write(self, tmp_path):
        from sourcing.scripts.run_safety_scan import _scan_file

        test_file = tmp_path / "test.jsonl"
        records = [{"messages": [{"role": "user", "content": "fine"}]}]
        with test_file.open("w") as f:
            for r in records:
                f.write(json.dumps(r) + "\n")

        processor = SafetyProcessor()
        _scan_file(test_file, processor, dry_run=True)

        assert not (tmp_path / "test.safe.jsonl").exists()
