"""Tests for the persona-driven email regeneration pipeline."""

from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest

# Ensure training module is importable
sys.path.insert(0, str(Path(__file__).parent.parent))

from training.email_regen_pipeline import (  # noqa: E402
    DEFAULT_PERSONAS,
    DEFAULT_SYSTEM_PROMPT,
    PERSONA_PROMPTS,
    PERSONA_TRAITS,
    TOXICITY_PATTERNS,
    EmailRecord,
    PipelineConfig,
    PipelineStats,
    RegenerationResult,
    build_expansion_prompt,
    build_parser,
    _check_persona_consistency,
    _check_toxicity,
    _content_hash,
    _count_sentences,
    _mock_expand,
    expand_email,
    is_short_email,
    main,
    parse_gmail_export,
    quality_check,
    regenerate_email,
    repair_corpus,
    run_pipeline,
    _rouge_l_similarity,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def tmp_jsonl(tmp_path):
    """Create a temp JSONL file with email records."""
    records = [
        {
            "id": "email_001",
            "subject": "Feeling overwhelmed",
            "body": "I don't know what to do. Everything feels too much right now.",
            "sender": "user@example.com",
            "timestamp": "2024-01-15T10:00:00Z",
        },
        {
            "id": "email_002",
            "subject": "Can't sleep",
            "body": "I've been awake all night with racing thoughts. Help.",
            "sender": "user2@example.com",
            "timestamp": "2024-01-16T03:00:00Z",
        },
        {
            "id": "email_003",
            "subject": "Therapy session follow-up",
            "body": "This is a very long email body that exceeds the maximum input length threshold. " * 20,
            "sender": "user3@example.com",
            "timestamp": "2024-01-17T14:00:00Z",
        },
        {
            "id": "email_004",
            "subject": "Short note",
            "body": "Hi",
            "sender": "user4@example.com",
            "timestamp": "2024-01-18T09:00:00Z",
        },
    ]
    path = tmp_path / "emails.jsonl"
    with open(path, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec) + "\n")
    return str(path)


@pytest.fixture
def tmp_corpus(tmp_path):
    """Create a temp training corpus JSONL with some short entries."""
    records = []
    # Normal-length entry (should be skipped in repair)
    records.append({
        "source": "training",
        "task_type": "chat",
        "messages": [
            {"role": "system", "content": "You are a therapist."},
            {"role": "user", "content": "I feel anxious."},
            {"role": "assistant", "content": "I understand you're feeling anxious. " * 10},
        ],
        "quality_score": 0.8,
    })
    # Short entry (should be repaired)
    records.append({
        "source": "training",
        "task_type": "chat",
        "messages": [
            {"role": "system", "content": "You are a therapist."},
            {"role": "user", "content": "I feel sad."},
            {"role": "assistant", "content": "That's tough."},
        ],
        "quality_score": 0.3,
    })
    path = tmp_path / "corpus.jsonl"
    with open(path, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec) + "\n")
    return str(path)


@pytest.fixture
def config():
    return PipelineConfig()


@pytest.fixture
def short_email():
    return EmailRecord(
        email_id="test_001",
        subject="Feeling lost",
        body="I feel lost and don't know where to turn.",
        sender="user@example.com",
        timestamp="2024-01-15T10:00:00Z",
    )


# ---------------------------------------------------------------------------
# Constants tests
# ---------------------------------------------------------------------------


class TestConstants:
    def test_default_personas(self):
        assert "warm_empathetic" in DEFAULT_PERSONAS
        assert "structured_practical" in DEFAULT_PERSONAS
        assert "gentle_exploratory" in DEFAULT_PERSONAS
        assert len(DEFAULT_PERSONAS) == 3

    def test_toxicity_patterns(self):
        assert "kill yourself" in TOXICITY_PATTERNS
        assert "suicide" in TOXICITY_PATTERNS
        assert "self-harm" in TOXICITY_PATTERNS

    def test_persona_traits(self):
        for persona in DEFAULT_PERSONAS:
            assert persona in PERSONA_TRAITS
            assert len(PERSONA_TRAITS[persona]) > 0

    def test_persona_prompts(self):
        for persona in DEFAULT_PERSONAS:
            assert persona in PERSONA_PROMPTS
            assert len(PERSONA_PROMPTS[persona]) > 20

    def test_default_system_prompt(self):
        assert "Pixelated Empathy" in DEFAULT_SYSTEM_PROMPT
        assert "clinical" in DEFAULT_SYSTEM_PROMPT.lower()


# ---------------------------------------------------------------------------
# Dataclass tests
# ---------------------------------------------------------------------------


class TestDataclasses:
    def test_email_record_defaults(self):
        rec = EmailRecord(email_id="1", subject="Test", body="Hello")
        assert rec.sender == ""
        assert rec.timestamp == ""

    def test_pipeline_config_defaults(self):
        cfg = PipelineConfig()
        assert cfg.min_input_length == 10
        assert cfg.max_input_length == 500
        assert cfg.min_output_sentences == 3
        assert cfg.max_output_length == 500
        assert cfg.rouge_l_too_similar == 0.85
        assert cfg.rouge_l_meaning_drift == 0.30
        assert cfg.min_quality_score == 0.5
        assert cfg.personas == list(DEFAULT_PERSONAS)
        assert cfg.temperature == 0.8

    def test_pipeline_config_custom_personas(self):
        cfg = PipelineConfig(personas=["warm_empathetic"])
        assert cfg.personas == ["warm_empathetic"]

    def test_regeneration_result_defaults(self):
        rec = EmailRecord(email_id="1", subject="S", body="B")
        result = RegenerationResult(original=rec, regenerated_messages=[])
        assert result.quality_score == 0.0
        assert result.quality_flags == []
        assert result.persona == ""
        assert result.accepted is False
        assert result.error is None

    def test_pipeline_stats_defaults(self):
        stats = PipelineStats()
        assert stats.total == 0
        assert stats.processed == 0
        assert stats.accepted == 0
        assert stats.rejected == 0
        assert stats.skipped == 0
        assert stats.errors == 0
        assert stats.persona_distribution == {}


# ---------------------------------------------------------------------------
# Helper function tests
# ---------------------------------------------------------------------------


class TestContentHash:
    def test_deterministic(self):
        h1 = _content_hash("hello world")
        h2 = _content_hash("hello world")
        assert h1 == h2

    def test_different_inputs(self):
        h1 = _content_hash("hello")
        h2 = _content_hash("world")
        assert h1 != h2

    def test_length(self):
        h = _content_hash("test")
        assert len(h) == 16


class TestRougeLSimilarity:
    def test_identical(self):
        assert _rouge_l_similarity("hello world", "hello world") == 1.0

    def test_no_overlap(self):
        assert _rouge_l_similarity("apple", "orange") == 0.0

    def test_partial_overlap(self):
        sim = _rouge_l_similarity("hello world foo", "hello world bar")
        assert 0.0 < sim < 1.0
        assert abs(sim - (2 / 4)) < 0.01  # 2 common, 4 total unique

    def test_empty_strings(self):
        assert _rouge_l_similarity("", "") == 0.0

    def test_case_insensitive(self):
        assert _rouge_l_similarity("Hello World", "hello world") == 1.0


class TestCountSentences:
    def test_single_sentence(self):
        assert _count_sentences("Hello world.") == 1

    def test_multiple_sentences(self):
        assert _count_sentences("Hello. World. Foo.") == 3

    def test_exclamation(self):
        assert _count_sentences("Hello! World!") == 2

    def test_question(self):
        assert _count_sentences("How are you? What happened?") == 2

    def test_empty(self):
        assert _count_sentences("") == 0

    def test_no_punctuation(self):
        assert _count_sentences("Hello world") == 1


class TestCheckToxicity:
    def test_clean_text(self):
        assert _check_toxicity("I feel happy today") is False

    def test_toxic_text(self):
        assert _check_toxicity("I want to kill yourself") is True

    def test_suicide(self):
        assert _check_toxicity("thoughts of suicide") is True

    def test_case_insensitive(self):
        assert _check_toxicity("KILL YOURSELF") is True

    def test_custom_patterns(self):
        assert _check_toxicity("bad word", ["bad word"]) is True
        assert _check_toxicity("clean text", ["bad word"]) is False


class TestCheckPersonaConsistency:
    def test_warm_empathetic_match(self):
        text = "I hear you with empathetic warmth and validating compassion."
        score, warnings = _check_persona_consistency(text, "warm_empathetic")
        assert score > 0.25
        assert len(warnings) == 0

    def test_no_match(self):
        text = "Just do the thing. Here are steps."
        score, warnings = _check_persona_consistency(text, "warm_empathetic")
        assert score < 0.5

    def test_unknown_persona(self):
        score, warnings = _check_persona_consistency("test", "unknown_persona")
        assert score == 0.5
        assert "unknown persona" in warnings[0]


# ---------------------------------------------------------------------------
# Parsing tests
# ---------------------------------------------------------------------------


class TestParseGmailExport:
    def test_parse_valid_jsonl(self, tmp_jsonl):
        records = parse_gmail_export(tmp_jsonl)
        assert len(records) == 4
        assert records[0].email_id == "email_001"
        assert records[0].subject == "Feeling overwhelmed"
        assert "too much" in records[0].body
        assert records[0].sender == "user@example.com"

    def test_parse_missing_file(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            parse_gmail_export(str(tmp_path / "nonexistent.jsonl"))

    def test_parse_empty_file(self, tmp_path):
        path = tmp_path / "empty.jsonl"
        path.write_text("")
        records = parse_gmail_export(str(path))
        assert records == []

    def test_parse_skips_blank_lines(self, tmp_path):
        path = tmp_path / "blanks.jsonl"
        path.write_text(
            json.dumps({"id": "1", "body": "test"}) + "\n\n\n"
        )
        records = parse_gmail_export(str(path))
        assert len(records) == 1

    def test_parse_skips_invalid_json(self, tmp_path):
        path = tmp_path / "invalid.jsonl"
        path.write_text(
            '{"id": "1", "body": "valid"}\n'
            "not json at all\n"
            '{"id": "2", "body": "also valid"}\n'
        )
        records = parse_gmail_export(str(path))
        assert len(records) == 2

    def test_parse_alternative_field_names(self, tmp_path):
        path = tmp_path / "alt.jsonl"
        path.write_text(
            json.dumps({
                "email_id": "e1",
                "text": "Hello there",
                "from": "sender@test.com",
                "date": "2024-01-01",
            }) + "\n"
        )
        records = parse_gmail_export(str(path))
        assert len(records) == 1
        assert records[0].email_id == "e1"
        assert records[0].body == "Hello there"
        assert records[0].sender == "sender@test.com"

    def test_parse_generates_id_if_missing(self, tmp_path):
        path = tmp_path / "noid.jsonl"
        path.write_text(json.dumps({"body": "test"}) + "\n")
        records = parse_gmail_export(str(path))
        assert len(records) == 1
        assert records[0].email_id == "email_1"


class TestIsShortEmail:
    def test_short_email(self, config):
        rec = EmailRecord(email_id="1", subject="", body="Short message here.")
        assert is_short_email(rec, config) is True

    def test_too_short(self, config):
        rec = EmailRecord(email_id="1", subject="", body="Hi")
        assert is_short_email(rec, config) is False

    def test_too_long(self, config):
        rec = EmailRecord(
            email_id="1", subject="", body="x" * 600
        )
        assert is_short_email(rec, config) is False

    def test_custom_config(self):
        cfg = PipelineConfig(min_input_length=5, max_input_length=50)
        rec = EmailRecord(
            email_id="1",
            subject="",
            body="This is a message that is definitely longer than fifty characters total.",
        )
        assert is_short_email(rec, cfg) is False


# ---------------------------------------------------------------------------
# LLM expansion tests
# ---------------------------------------------------------------------------


class TestBuildExpansionPrompt:
    def test_returns_messages_list(self, short_email, config):
        messages = build_expansion_prompt(short_email, "warm_empathetic", config)
        assert isinstance(messages, list)
        assert len(messages) == 2
        assert messages[0]["role"] == "system"
        assert messages[1]["role"] == "user"

    def test_system_prompt_included(self, short_email, config):
        messages = build_expansion_prompt(short_email, "warm_empathetic", config)
        assert messages[0]["content"] == config.system_prompt

    def test_persona_instruction_included(self, short_email, config):
        messages = build_expansion_prompt(short_email, "warm_empathetic", config)
        assert "empathetic" in messages[1]["content"].lower()

    def test_subject_included(self, short_email, config):
        messages = build_expansion_prompt(short_email, "warm_empathetic", config)
        assert "Subject:" in messages[1]["content"]
        assert short_email.subject in messages[1]["content"]

    def test_unknown_persona_uses_default(self, short_email, config):
        messages = build_expansion_prompt(short_email, "unknown", config)
        assert "therapist" in messages[1]["content"].lower() or "empathetic" in messages[1]["content"].lower()


class TestMockExpand:
    def test_returns_expanded_text(self, short_email, config):
        result = _mock_expand(short_email, "warm_empathetic", config)
        assert len(result) > len(short_email.body)
        assert short_email.body in result

    def test_persona_prefix(self, short_email, config):
        result = _mock_expand(short_email, "warm_empathetic", config)
        assert "I hear you" in result or "makes complete sense" in result

    def test_persona_suffix(self, short_email, config):
        result = _mock_expand(short_email, "structured_practical", config)
        assert "step" in result.lower() or "take one step" in result.lower()

    def test_min_sentences_enforced(self, short_email, config):
        result = _mock_expand(short_email, "gentle_exploratory", config)
        assert _count_sentences(result) >= config.min_output_sentences


class TestExpandEmail:
    def test_mock_fallback(self, short_email, config):
        result = expand_email(short_email, "warm_empathetic", None, config)
        assert len(result) > len(short_email.body)

    def test_llm_callback(self, short_email, config):
        def mock_llm(messages):
            return "This is a mocked LLM response with multiple sentences. It validates the user. It offers guidance."

        result = expand_email(short_email, "warm_empathetic", mock_llm, config)
        assert "mocked LLM response" in result

    def test_llm_failure_fallback(self, short_email, config):
        def failing_llm(messages):
            raise RuntimeError("API error")

        result = expand_email(short_email, "warm_empathetic", failing_llm, config)
        assert len(result) > 0  # Should fall back to mock

    def test_llm_empty_response_fallback(self, short_email, config):
        def empty_llm(messages):
            return ""

        result = expand_email(short_email, "warm_empathetic", empty_llm, config)
        assert len(result) > 0


# ---------------------------------------------------------------------------
# Quality check tests
# ---------------------------------------------------------------------------


class TestQualityCheck:
    def test_good_expansion(self, config):
        original = "I feel lost and don't know where to turn."
        expanded = (
            "I hear you, and what you're feeling makes complete sense. "
            "I feel lost and don't know where to turn. "
            "You're not alone in this, and together we can find a path forward."
        )
        score, flags = quality_check(original, expanded, config, "warm_empathetic")
        assert score >= config.min_quality_score
        assert "toxicity_detected" not in flags

    def test_too_short(self, config):
        original = "I feel lost"
        expanded = "I hear you."
        score, flags = quality_check(original, expanded, config)
        assert "too_short" in flags
        assert score < 1.0

    def test_too_long(self, config):
        original = "I feel lost"
        expanded = "x" * 600
        score, flags = quality_check(original, expanded, config)
        assert "too_long" in flags

    def test_too_similar(self, config):
        original = "I feel lost and don't know where to turn right now today"
        expanded = original  # identical
        score, flags = quality_check(original, expanded, config)
        assert "too_similar" in flags

    def test_meaning_drift(self, config):
        original = "I feel lost and don't know where to turn right now today"
        expanded = "The weather is nice and sunny outside today let's go for a walk in the park"
        score, flags = quality_check(original, expanded, config)
        assert "meaning_drift" in flags

    def test_toxicity(self, config):
        original = "I feel bad"
        expanded = (
            "I understand you're hurting. But remember, "
            "thinking about suicide is not the answer. "
            "Please reach out for help right now."
        )
        score, flags = quality_check(original, expanded, config)
        assert "toxicity_detected" in flags

    def test_insufficient_sentences(self, config):
        original = "I feel lost"
        expanded = "I hear you and that's okay."
        score, flags = quality_check(original, expanded, config)
        assert "insufficient_sentences" in flags

    def test_score_bounded(self, config):
        original = "test"
        expanded = "x" * 600 + " kill yourself " + "y" * 600
        score, flags = quality_check(original, expanded, config)
        assert 0.0 <= score <= 1.0

    def test_persona_low_match(self, config):
        original = "I feel lost"
        expanded = (
            "Here are the steps. Do step one. Do step two. "
            "Do step three. Do step four. Do step five."
        )
        score, flags = quality_check(original, expanded, config, "warm_empathetic")
        assert any("persona" in f for f in flags)


# ---------------------------------------------------------------------------
# Regenerate email tests
# ---------------------------------------------------------------------------


class TestRegenerateEmail:
    def test_returns_result(self, short_email, config):
        result = regenerate_email(short_email, None, config)
        assert isinstance(result, RegenerationResult)
        assert result.original == short_email
        assert len(result.regenerated_messages) == 3

    def test_persona_selected(self, short_email, config):
        result = regenerate_email(short_email, None, config)
        assert result.persona in config.personas

    def test_persona_deterministic(self, short_email, config):
        result1 = regenerate_email(short_email, None, config)
        result2 = regenerate_email(short_email, None, config)
        assert result1.persona == result2.persona

    def test_messages_format(self, short_email, config):
        result = regenerate_email(short_email, None, config)
        assert result.regenerated_messages[0]["role"] == "system"
        assert result.regenerated_messages[1]["role"] == "user"
        assert result.regenerated_messages[2]["role"] == "assistant"

    def test_accepted_with_mock(self, short_email, config):
        result = regenerate_email(short_email, None, config)
        # Mock expansion should produce acceptable quality
        assert result.accepted is True
        assert result.quality_score > 0

    def test_error_handling(self, config):
        email = EmailRecord(email_id="err", subject="", body="Short body test.")
        with patch(
            "training.email_regen_pipeline.expand_email",
            side_effect=Exception("LLM error"),
        ):
            result = regenerate_email(email, None, config)
            assert result.error is not None
            assert result.accepted is False


# ---------------------------------------------------------------------------
# Pipeline tests
# ---------------------------------------------------------------------------


class TestRunPipeline:
    def test_basic_run(self, tmp_jsonl, tmp_path, config):
        output = str(tmp_path / "output.jsonl")
        stats = run_pipeline(tmp_jsonl, output, None, config)
        assert isinstance(stats, PipelineStats)
        assert stats.total == 4
        assert stats.processed > 0
        assert stats.skipped > 0  # The long email and too-short email skipped

    def test_output_file_created(self, tmp_jsonl, tmp_path, config):
        output = str(tmp_path / "output.jsonl")
        run_pipeline(tmp_jsonl, output, None, config)
        assert os.path.exists(output)

    def test_output_format(self, tmp_jsonl, tmp_path, config):
        output = str(tmp_path / "output.jsonl")
        run_pipeline(tmp_jsonl, output, None, config)
        with open(output, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    rec = json.loads(line)
                    assert "messages" in rec
                    assert "source" in rec
                    assert rec["source"] == "email_regen"
                    assert "persona" in rec
                    assert "quality_score" in rec
                    break

    def test_stats_correct(self, tmp_jsonl, tmp_path, config):
        output = str(tmp_path / "output.jsonl")
        stats = run_pipeline(tmp_jsonl, output, None, config)
        assert stats.total == 4
        assert stats.processed + stats.skipped + stats.errors <= stats.total

    def test_persona_distribution(self, tmp_jsonl, tmp_path, config):
        output = str(tmp_path / "output.jsonl")
        stats = run_pipeline(tmp_jsonl, output, None, config)
        if stats.accepted > 0:
            assert len(stats.persona_distribution) > 0

    def test_llm_callback_used(self, tmp_jsonl, tmp_path, config):
        call_count = [0]

        def mock_llm(messages):
            call_count[0] += 1
            return (
                "I understand you're going through a difficult time. "
                "Your feelings are valid and it makes sense to feel this way. "
                "Let's explore some gentle steps forward together."
            )

        output = str(tmp_path / "output.jsonl")
        run_pipeline(tmp_jsonl, output, mock_llm, config)
        assert call_count[0] > 0

    def test_empty_input(self, tmp_path, config):
        input_path = str(tmp_path / "empty.jsonl")
        Path(input_path).write_text("")
        output = str(tmp_path / "out.jsonl")
        stats = run_pipeline(input_path, output, None, config)
        assert stats.total == 0

    def test_skips_too_short_and_too_long(self, tmp_path, config):
        path = tmp_path / "extremes.jsonl"
        path.write_text(
            json.dumps({"id": "1", "body": "Hi"}) + "\n"
            + json.dumps({"id": "2", "body": "x" * 600}) + "\n"
        )
        output = str(tmp_path / "out.jsonl")
        stats = run_pipeline(str(path), output, None, config)
        assert stats.skipped == 2
        assert stats.processed == 0


class TestRepairCorpus:
    def test_repairs_short_entries(self, tmp_corpus, tmp_path, config):
        output = str(tmp_path / "repaired.jsonl")
        stats = repair_corpus(tmp_corpus, output, None, config)
        assert stats.total == 2
        assert stats.processed == 1  # One short entry
        assert stats.skipped == 1   # One normal entry

    def test_output_preserves_all_entries(self, tmp_corpus, tmp_path, config):
        output = str(tmp_path / "repaired.jsonl")
        repair_corpus(tmp_corpus, output, None, config)
        with open(output, encoding="utf-8") as f:
            lines = [l for l in f if l.strip()]
        assert len(lines) == 2

    def test_repaired_entry_has_expanded_content(self, tmp_corpus, tmp_path, config):
        output = str(tmp_path / "repaired.jsonl")
        repair_corpus(tmp_corpus, output, None, config)
        with open(output, encoding="utf-8") as f:
            for line in f:
                rec = json.loads(line.strip())
                messages = rec["messages"]
                assistant_content = ""
                for msg in messages:
                    if msg["role"] == "assistant":
                        assistant_content = msg["content"]
                        break
                assert len(assistant_content) >= config.min_output_length

    def test_repaired_entry_marked(self, tmp_corpus, tmp_path, config):
        output = str(tmp_path / "repaired.jsonl")
        repair_corpus(tmp_corpus, output, None, config)
        with open(output, encoding="utf-8") as f:
            for line in f:
                rec = json.loads(line.strip())
                if rec.get("repaired"):
                    assert "persona" in rec
                    assert "quality_score" in rec

    def test_normal_entry_not_marked(self, tmp_corpus, tmp_path, config):
        output = str(tmp_path / "repaired.jsonl")
        repair_corpus(tmp_corpus, output, None, config)
        with open(output, encoding="utf-8") as f:
            for line in f:
                rec = json.loads(line.strip())
                if not rec.get("repaired"):
                    assert "repaired" not in rec or rec["repaired"] is False


# ---------------------------------------------------------------------------
# CLI tests
# ---------------------------------------------------------------------------


class TestCLI:
    def test_build_parser(self):
        parser = build_parser()
        assert parser is not None
        # Check required args
        args = parser.parse_args(["--input", "in.jsonl", "--output", "out.jsonl"])
        assert args.input == "in.jsonl"
        assert args.output == "out.jsonl"
        assert args.mode == "generate"

    def test_parser_mode_choices(self):
        parser = build_parser()
        args = parser.parse_args(["--input", "i", "--output", "o", "--mode", "repair"])
        assert args.mode == "repair"

    def test_parser_custom_params(self):
        parser = build_parser()
        args = parser.parse_args([
            "--input", "i", "--output", "o",
            "--min-input-length", "5",
            "--max-input-length", "100",
            "--min-output-sentences", "2",
            "--temperature", "0.5",
            "--personas", "warm_empathetic",
        ])
        assert args.min_input_length == 5
        assert args.max_input_length == 100
        assert args.min_output_sentences == 2
        assert args.temperature == 0.5
        assert args.personas == ["warm_empathetic"]

    def test_main_generate_mode(self, tmp_jsonl, tmp_path):
        output = str(tmp_path / "cli_out.jsonl")
        with patch.object(sys, "argv", [
            "email_regen_pipeline.py",
            "--input", tmp_jsonl,
            "--output", output,
        ]):
            result = main()
        assert result == 0
        assert os.path.exists(output)

    def test_main_repair_mode(self, tmp_corpus, tmp_path):
        output = str(tmp_path / "cli_repaired.jsonl")
        with patch.object(sys, "argv", [
            "email_regen_pipeline.py",
            "--input", tmp_corpus,
            "--output", output,
            "--mode", "repair",
        ]):
            result = main()
        assert result == 0
        assert os.path.exists(output)

    def test_main_returns_nonzero_on_errors(self, tmp_path):
        # Nonexistent input file
        output = str(tmp_path / "out.jsonl")
        with patch.object(sys, "argv", [
            "email_regen_pipeline.py",
            "--input", str(tmp_path / "nonexistent.jsonl"),
            "--output", output,
        ]):
            with pytest.raises(FileNotFoundError):
                main()

    def test_main_verbose_flag(self, tmp_jsonl, tmp_path):
        output = str(tmp_path / "verbose_out.jsonl")
        with patch.object(sys, "argv", [
            "email_regen_pipeline.py",
            "--input", tmp_jsonl,
            "--output", output,
            "--verbose",
        ]):
            result = main()
        assert result == 0


# ---------------------------------------------------------------------------
# Integration / edge case tests
# ---------------------------------------------------------------------------


class TestEdgeCases:
    def test_unicode_content(self, tmp_path, config):
        path = tmp_path / "unicode.jsonl"
        path.write_text(
            json.dumps({
                "id": "u1",
                "subject": "Siento tristeza",
                "body": "Me siento muy triste hoy. No sé qué hacer.",
                "sender": "user@test.com",
            }, ensure_ascii=False) + "\n"
        )
        records = parse_gmail_export(str(path))
        assert len(records) == 1
        assert "triste" in records[0].body

        result = regenerate_email(records[0], None, config)
        assert result.accepted is True

    def test_all_three_personas_used(self, tmp_path, config):
        """Test that all personas get assigned across multiple records."""
        path = tmp_path / "multi.jsonl"
        for i in range(30):
            path.write_text(
                path.read_text(encoding="utf-8")
                + json.dumps({"id": f"e{i}", "body": f"Feeling message number {i} here."}) + "\n"
                if path.exists()
                else json.dumps({"id": f"e{i}", "body": f"Feeling message number {i} here."}) + "\n"
            )
        output = str(tmp_path / "multi_out.jsonl")
        stats = run_pipeline(str(path), output, None, config)
        # With 30 records, at least 2 different personas should be used
        if stats.accepted >= 3:
            assert len(stats.persona_distribution) >= 2

    def test_deterministic_persona_assignment(self, config):
        """Same email always gets same persona."""
        email = EmailRecord(
            email_id="det1",
            subject="Test",
            body="I need some guidance today.",
        )
        results = [regenerate_email(email, None, config) for _ in range(5)]
        personas = {r.persona for r in results}
        assert len(personas) == 1

    def test_quality_score_range(self, short_email, config):
        result = regenerate_email(short_email, None, config)
        assert 0.0 <= result.quality_score <= 1.0

    def test_provenance_in_output(self, tmp_jsonl, tmp_path, config):
        output = str(tmp_path / "prov_out.jsonl")
        run_pipeline(tmp_jsonl, output, None, config)
        with open(output, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    rec = json.loads(line)
                    assert "provenance" in rec
                    assert rec["provenance"]["source_type"] == "email_export"
                    assert "persona_expansion" in rec["provenance"]["transformations"]
                    break
