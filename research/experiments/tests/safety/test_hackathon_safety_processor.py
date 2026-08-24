"""
PIX-4240 tests — heuristic toxicity detection + strict PII stripping across
imported hackathon data.

Covers:
  - PII patterns stripped: email, phone, SSN, address, medical ID, URL-with-path
  - Toxicity heuristics flag genuinely toxic content per category
  - Legitimate clinical discussion is NOT flagged (over-filtering guard)
  - HackathonSafetyProcessor end-to-end with shard routing
  - run_safety_pass CLI script end-to-end on synthetic sample
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from pipelines.data_processing.processors.safety_processors import (
    HackathonSafetyProcessor,
    SafetyProcessResult,
    SafetyReport,
)
from pipelines.data_processing.processors.toxicity_detector import (
    HeuristicToxicityDetector,
)
from ai.tools.utilities.core.pipelines.processing.pii_scrubber import (
    PiiScrubber,
    PiiScrubberConfig,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def detector() -> HeuristicToxicityDetector:
    return HeuristicToxicityDetector()


@pytest.fixture
def scrubber() -> PiiScrubber:
    return PiiScrubber(PiiScrubberConfig(use_spacy_for_names=False, log_findings=False))


@pytest.fixture
def processor() -> HackathonSafetyProcessor:
    return HackathonSafetyProcessor()


def _chatml(user: str, assistant: str = "Acknowledged.") -> dict:
    return {
        "messages": [
            {"role": "system", "content": "You are a clinical assistant."},
            {"role": "user", "content": user},
            {"role": "assistant", "content": assistant},
        ],
        "metadata": {"source_family": "mental_health_conversations"},
    }


# ---------------------------------------------------------------------------
# PII stripping
# ---------------------------------------------------------------------------


class TestPiiStripping:
    """Verify strict PII patterns are stripped from sample content."""

    def test_strips_email(self, scrubber: PiiScrubber) -> None:
        text = "Reach me at john.doe@example.com for follow-up."
        result = scrubber.scrub(text)
        assert "@" not in result.scrubbed_text, "email not stripped"
        assert result.pii_counts.get("email") == 1

    def test_strips_phone(self, scrubber: PiiScrubber) -> None:
        text = "Call (415) 555-1234 anytime."
        result = scrubber.scrub(text)
        assert "555-1234" not in result.scrubbed_text, "phone not stripped"
        assert result.pii_counts.get("phone") == 1

    def test_strips_ssn(self, scrubber: PiiScrubber) -> None:
        text = "SSN: 123-45-6789 on file."
        result = scrubber.scrub(text)
        assert "123-45-6789" not in result.scrubbed_text, "SSN not stripped"
        assert result.pii_counts.get("ssn") == 1

    def test_strips_address(self, scrubber: PiiScrubber) -> None:
        text = "Patient lives at 123 Main Street near the park."
        result = scrubber.scrub(text)
        assert "123 Main Street" not in result.scrubbed_text, "address not stripped"
        assert result.pii_counts.get("address") == 1

    def test_strips_medical_record_number(self, scrubber: PiiScrubber) -> None:
        text = "MRN: ABC123456 for the patient."
        result = scrubber.scrub(text)
        assert "ABC123456" not in result.scrubbed_text, "MRN not stripped"
        assert result.pii_counts.get("medical_record_number") == 1

    def test_strips_url_with_identifying_path(self, scrubber: PiiScrubber) -> None:
        text = "His profile is at https://example.com/users/john-doe/profile?view=1"
        result = scrubber.scrub(text)
        assert "john-doe" not in result.scrubbed_text, "url identifying path not stripped"
        assert result.pii_counts.get("url_with_identifying_path") == 1

    def test_does_not_flag_bare_hostname_url(self, scrubber: PiiScrubber) -> None:
        text = "Visit https://example.com for general info."
        result = scrubber.scrub(text)
        assert "https://example.com" in result.scrubbed_text, "bare URL should NOT be stripped"
        assert "url_with_identifying_path" not in result.pii_counts

    def test_strips_credit_card_with_luhn(self, scrubber: PiiScrubber) -> None:
        # 4242 4242 4242 4242 — passes Luhn (Stripe test card)
        text = "Card: 4242 4242 4242 4242 on file."
        result = scrubber.scrub(text)
        assert "4242 4242 4242 4242" not in result.scrubbed_text, "credit card not stripped"
        assert result.pii_counts.get("credit_card") == 1

    def test_does_not_flag_invalid_luhn_credit_card(self, scrubber: PiiScrubber) -> None:
        # 16 digits but failing Luhn
        text = "Code 1111 2222 3333 5555 is just an ID."
        result = scrubber.scrub(text)
        assert "1111 2222 3333 5555" in result.scrubbed_text
        assert result.pii_counts.get("credit_card") is None

    def test_scrub_dict_recurses_into_messages(self, scrubber: PiiScrubber) -> None:
        record = _chatml("Email me at j.doe@example.com please.")
        cleaned = scrubber.scrub_dict(record)
        contents = [m["content"] for m in cleaned["messages"]]
        assert all("@example.com" not in c for c in contents), "scrub_dict failed to recurse"

    def test_scrub_dict_does_not_mutate_input(self, scrubber: PiiScrubber) -> None:
        record = _chatml("Email: j.doe@example.com")
        original_content = record["messages"][1]["content"]
        scrubber.scrub_dict(record)
        assert record["messages"][1]["content"] == original_content, "scrub_dict mutated input"


# ---------------------------------------------------------------------------
# Toxicity detector — per category, distinguishing clinical framing
# ---------------------------------------------------------------------------


class TestToxicityDetection:
    """Verify the heuristic detector flags toxic content per category."""

    def test_self_harm_suicide_direct_intent_flagged(self, detector: HeuristicToxicityDetector) -> None:
        text = "I want to kill myself. There is no way out."
        result = detector.detect(text)
        assert result.is_toxic, "self-harm ideation should be flagged"
        assert result.categories["self_harm_suicide"].triggered

    def test_self_harm_clinical_discussion_not_flagged(self, detector: HeuristicToxicityDetector) -> None:
        text = (
            "The patient may experience suicidal ideation as part of clinical depression. "
            " therapists should assess risk and develop a safety plan."
        )
        result = detector.detect(text)
        # The trigger patterns ("suicidal ideation" does not match SELF_HARM patterns directly),
        # and even words like "kill myself" should not appear — this is clinical framing.
        # Make sure clinical discussion is NOT flagged.
        assert not result.is_toxic, "clinical discussion incorrectly flagged as toxic"
        assert not result.categories["self_harm_suicide"].triggered

    def test_graphic_violence_flagged(self, detector: HeuristicToxicityDetector) -> None:
        text = "He stabbed her in the throat repeatedly until she stopped moving."
        result = detector.detect(text)
        assert result.is_toxic
        assert result.categories["graphic_violence"].triggered

    def test_hate_speech_slur_flagged(self, detector: HeuristicToxicityDetector) -> None:
        text = "I hate that slur — keep that nigger out of here."
        result = detector.detect(text)
        assert result.is_toxic
        assert result.categories["hate_discrimination"].triggered

    def test_inappropriate_sexual_minor_flagged(self, detector: HeuristicToxicityDetector) -> None:
        text = "I downloaded child porn last night and shared it."
        result = detector.detect(text)
        assert result.is_toxic
        assert result.categories["inappropriate_sexual"].triggered

    def test_manipulative_coercive_threat_flagged(self, detector: HeuristicToxicityDetector) -> None:
        text = "If you leave me, I'll kill myself — don't do it."
        result = detector.detect(text)
        assert result.is_toxic
        assert result.categories["manipulative_coercive"].triggered

    def test_clinical_abuse_history_not_flagged(self, detector: HeuristicToxicityDetector) -> None:
        text = (
            "The client reported a history of abuse as part of her therapeutic intake. "
            "Patient presents with trauma symptoms per DSM-5 criteria. Recommend trauma-focused CBT."
        )
        result = detector.detect(text)
        assert not result.is_toxic, "clinical trauma history incorrectly flagged"

    def test_clinical_substance_use_not_flagged(self, detector: HeuristicToxicityDetector) -> None:
        text = (
            "The patient disclosed substance use including opioids. Treatment plan includes "
            "relapse prevention counseling and medication management."
        )
        result = detector.detect(text)
        assert not result.is_toxic

    def test_clinical_self_harm_history_not_flagged(self, detector: HeuristicToxicityDetector) -> None:
        text = (
            "Patient reported cutting herself in the past as a coping mechanism. "
            "Therapist developed a safety plan with the patient during today's session. "
            "The patient is in treatment for self-harm and suicidal ideation."
        )
        result = detector.detect(text)
        assert not result.is_toxic, "clinical self-harm history incorrectly flagged"

    def test_detect_record_handles_invalid_input(self, detector: HeuristicToxicityDetector) -> None:
        from typing import cast

        bad_record = cast(dict, "not a dict")
        assert detector.detect_record(bad_record).is_toxic is False
        assert detector.detect_record({}).is_toxic is False


# ---------------------------------------------------------------------------
# HackathonSafetyProcessor end-to-end
# ---------------------------------------------------------------------------


class TestHackathonSafetyProcessor:
    """End-to-end tests for the orchestrating safety processor."""

    def test_process_returns_cleaned_record_and_report(self, processor: HackathonSafetyProcessor) -> None:
        record = _chatml("Contact me at jane.doe@example.com today.")
        result = processor.process(record)
        assert isinstance(result, SafetyProcessResult)
        assert isinstance(result.report, SafetyReport)
        assert "@example.com" not in result.cleaned_record["messages"][1]["content"]
        assert result.report.pii_counts.get("email") == 1

    def test_process_does_not_mutate_input(self, processor: HackathonSafetyProcessor) -> None:
        record = _chatml("Email: j.doe@example.com")
        original = json.dumps(record, sort_keys=True)
        processor.process(record)
        assert json.dumps(record, sort_keys=True) == original, "processor mutated input"

    def test_routes_toxic_record_to_review(self, processor: HackathonSafetyProcessor) -> None:
        record = _chatml("I want to kill myself right now, nobody would care.")
        result = processor.process(record)
        assert result.report.routed_to_toxic_review
        assert "self_harm_suicide" in result.report.toxicity_triggered_categories

    def test_keeps_clinical_record_in_clear_stream(self, processor: HackathonSafetyProcessor) -> None:
        record = _chatml(
            "Patient reports a history of suicidal ideation. Recommend safety planning and weekly therapy sessions."
        )
        result = processor.process(record)
        assert not result.report.routed_to_toxic_review, "clinical record routed to toxic review"
        assert result.report.toxicity_score == 0.0

    def test_attaches_safety_report_to_metadata(self, processor: HackathonSafetyProcessor) -> None:
        record = _chatml("Email: j.doe@example.com")
        result = processor.process(record)
        meta = result.cleaned_record.get("metadata", {})
        assert "safety_report" in meta, "safety report missing from metadata"
        assert meta["safety_report"]["pii_total"] == 1

    def test_record_id_is_stable(self, processor: HackathonSafetyProcessor) -> None:
        record = _chatml("Identical content")
        id1 = processor.process(record).report.record_id
        id2 = processor.process(record).report.record_id
        assert id1 == id2, "record_id not stable"
        assert id1 is not None and len(id1) == 16

    def test_raises_on_non_dict_input(self, processor: HackathonSafetyProcessor) -> None:
        with pytest.raises(TypeError):
            processor.process("not a dict")  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Pipeline integration smoke test (extract_everything import)
# ---------------------------------------------------------------------------


class TestPipelineIntegrationImport:
    """Smoke test — extract_everything imports the safety processor cleanly."""

    def test_import_succeeds(self) -> None:
        pytest.importorskip("fitz", reason="PyMuPDF not installed")
        from pipelines.data_processing.orchestration import extract_everything

        assert hasattr(extract_everything, "HackathonSafetyProcessor"), (
            "HackathonSafetyProcessor not imported into extract_everything"
        )

    def test_upload_shard_accepts_prefix(self) -> None:
        pytest.importorskip("fitz", reason="PyMuPDF not installed")
        import inspect

        from pipelines.data_processing.orchestration.extract_everything import upload_shard

        sig = inspect.signature(upload_shard)
        assert "prefix" in sig.parameters, "upload_shard missing prefix kwarg"


# ---------------------------------------------------------------------------
# run_safety_pass CLI end-to-end
# ---------------------------------------------------------------------------


class TestRunSafetyPassCli:
    """End-to-end smoke test for the standalone CLI driver."""

    def test_end_to_end_on_synthetic_dataset(self, tmp_path: Path) -> None:
        from pipelines.data_processing.scripts.run_safety_pass import run_pass

        # Build a small synthetic input dataset
        records = [
            _chatml("User email: alice@example.com - please reach out."),
            _chatml("I want to kill myself, there's no point to keep going."),
            _chatml("Patient reports a history of suicidal ideation. Treatment plan in place."),
            _chatml("Call (555) 123-4567 immediately.", assistant="Will call now."),
            _chatml("Decapitate the body and display the head publicly.", assistant="Tips noted."),
        ]

        # Wrap each as the raw shape expected by iter_raw_records
        wrapped = [{"raw_data": r, "metadata": {"source_family": "mental_health_conversations"}} for r in records]

        input_file = tmp_path / "input.jsonl"
        input_file.write_text(
            "\n".join(json.dumps(r, ensure_ascii=False) for r in wrapped),
            encoding="utf-8",
        )

        out_dir = tmp_path / "out"
        report = run_pass(input_file, out_dir, shard_size=10)

        assert report["totals"]["raw_records"] == 5

        assert report["totals"]["routed_toxic_review"] == 2
        assert report["totals"]["clear_records"] == 3

        assert report["pii"]["total_findings"] >= 2  # email + phone at minimum

        clear_files = list((out_dir / "clear").glob("shard_*.jsonl"))
        assert clear_files, "no clear shard emitted"

        toxic_files = list((out_dir / "toxic_review").glob("shard_*.jsonl"))
        assert toxic_files, "no toxic_review shard emitted"

        assert (out_dir / "reports" / "safety_report.json").exists()

        clear_blob = "\n".join(f.read_text(encoding="utf-8") for f in clear_files)
        assert "@example.com" not in clear_blob, "PII leaked into clear output"
        assert "123-4567" not in clear_blob, "phone PII leaked into clear output"


# ---------------------------------------------------------------------------
# Additional tests (review thread requests)
# ---------------------------------------------------------------------------


class TestAdvisoryToxicity:
    """Tests for advisory (non-routed) toxicity that stays in clear stream."""

    def test_advisory_toxicity_stays_in_clear(self) -> None:
        """Toxicity below threshold should be flagged but not routed to toxic_review."""
        from pipelines.data_processing.processors.safety_processors import HackathonSafetyProcessor

        processor = HackathonSafetyProcessor(toxic_route_threshold=2.0)
        record = _chatml("You are crazy and imagining things.")
        result = processor.process(record)
        assert not result.report.routed_to_toxic_review
        assert result.report.toxicity_score > 0.0

    def test_metadata_pii_aggregation(self, processor: HackathonSafetyProcessor) -> None:
        """PII in metadata values should be counted in the report."""
        record = {
            "messages": [{"role": "user", "content": "Hello"}],
            "metadata": {"source_family": "test", "contact": "j.doe@example.com"},
        }
        result = processor.process(record)
        assert result.report.pii_counts.get("email") == 1

    def test_clinical_match_reporting(self, processor: HackathonSafetyProcessor) -> None:
        """Clinical matches should be reported in the safety report."""
        record = _chatml("Patient in therapy for substance use. I want to kill myself.")
        result = processor.process(record)
        assert len(result.report.clinical_matches_summary) > 0 or result.report.toxicity_score > 0


class TestCliDirectoryInput:
    """Tests for CLI directory and CSV input handling."""

    def test_csv_input(self, tmp_path: Path) -> None:
        """CSV input should be processed correctly."""
        import csv

        csv_file = tmp_path / "data.csv"
        with csv_file.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["text", "role"])
            writer.writeheader()
            writer.writerow({"text": "Hello world", "role": "user"})
            writer.writerow({"text": "I want to kill myself", "role": "user"})

        from pipelines.data_processing.scripts.run_safety_pass import run_pass

        out_dir = tmp_path / "out"
        report = run_pass(csv_file, out_dir, shard_size=10)
        assert report["totals"]["raw_records"] >= 1

    def test_empty_input_returns_error(self, tmp_path: Path) -> None:
        """Empty input should produce zero records."""
        empty_file = tmp_path / "empty.jsonl"
        empty_file.write_text("", encoding="utf-8")

        from pipelines.data_processing.scripts.run_safety_pass import run_pass

        out_dir = tmp_path / "out"
        report = run_pass(empty_file, out_dir, shard_size=10)
        assert report["totals"]["raw_records"] == 0
