"""
Tests for the therapist voice extraction refactored modules.

Covers:
  - extraction_config : channel config lookup, key resolution, constants
  - extraction_models : ChannelResult dataclass, computed properties
  - synthetic_templates : template structure and content
  - extraction_io     : profile derivation helpers, report generation
  - extract_therapist_voice : conversation generation, scoring, validation, CLI
"""

from __future__ import annotations

import csv
import json
import tempfile
from pathlib import Path

from scripts.extract_therapist_voice import (
    _CLIENT_QUESTION_TEMPLATES,
    _build_synthetic_dialogue,
    _get_topics_for_expertise,
    _strip_transcript_metadata,
    annotate_conversations,
    generate_conversation_from_transcript,
    generate_synthetic_conversations,
    parse_args,
    score_conversations,
    validate_conversation_quality,
)

# ---------------------------------------------------------------------------
# Module imports
# ---------------------------------------------------------------------------
from scripts.extraction_config import (
    CHANNEL_CONFIGS,
    DEFAULT_CONVERSATIONS,
    MAX_MARKED_SENTENCE_LENGTH,
    MIN_COMMON_PHRASE_COUNT,
    MIN_SENTENCE_WORDS,
    TOPIC_BANK,
    get_config,
    resolve_channel_key,
)
from scripts.extraction_io import (
    _derive_communication_patterns,
    _derive_tone_characteristics,
    generate_quality_report,
    save_channel_output,
)
from scripts.extraction_models import ChannelResult
from scripts.synthetic_templates import SYNTHETIC_TEMPLATES

# ===================================================================
#  extraction_config
# ===================================================================


class TestGetConfig:
    def test_exact_match(self):
        """Exact key returns the right config."""
        cfg = get_config("DocSnipes")
        assert cfg is not None
        assert cfg["name"] == "Doc Snipes"

    def test_case_insensitive(self):
        """Lookup is case-insensitive."""
        for variant in ["docsnipes", "DOCSNIPES", "DoCsNiPeS"]:
            cfg = get_config(variant)
            assert cfg is not None, f"Failed for {variant!r}"
            assert cfg["name"] == "Doc Snipes"

    def test_unknown_returns_none(self):
        """Unknown key returns None."""
        assert get_config("NonExistentChannel") is None
        assert get_config("") is None

    def test_all_configs_have_required_keys(self):
        """Every channel config has the expected schema."""
        required = {"name", "signature", "description", "style", "expertise", "approach"}
        for key, cfg in CHANNEL_CONFIGS.items():
            missing = required - set(cfg.keys())
            assert not missing, f"Channel {key!r} missing keys: {missing}"
            assert isinstance(cfg["expertise"], list), f"Channel {key!r} expertise is not a list"
            assert len(cfg["expertise"]) > 0, f"Channel {key!r} has empty expertise"

    def test_signature_dedup(self):
        """No two channels share the same signature."""
        sigs = [cfg["signature"] for cfg in CHANNEL_CONFIGS.values()]
        assert len(sigs) == len(set(sigs)), "Duplicate signatures detected"


class TestResolveChannelKey:
    def test_exact_resolution(self):
        assert resolve_channel_key("DocSnipes") == "DocSnipes"

    def test_case_insensitive_resolution(self):
        assert resolve_channel_key("docsnipes") == "DocSnipes"
        assert resolve_channel_key("DOCTORRAMANI") == "DoctorRamani"

    def test_unknown_returns_none(self):
        assert resolve_channel_key("") is None
        assert resolve_channel_key("NotAChannel") is None


class TestConfigConstants:
    def test_constants_are_positive_ints(self):
        for val in [MIN_SENTENCE_WORDS, MAX_MARKED_SENTENCE_LENGTH, MIN_COMMON_PHRASE_COUNT, DEFAULT_CONVERSATIONS]:
            assert isinstance(val, int), f"{val} is not int"
            assert val > 0, f"{val} is not positive"

    def test_topic_bank_has_all_categories(self):
        expected = {"general", "trauma", "personality_disorders", "cbt_dbt", "attachment"}
        assert set(TOPIC_BANK.keys()) == expected

    def test_topic_bank_entries_nonempty(self):
        for category, topics in TOPIC_BANK.items():
            assert len(topics) > 0, f"Category {category!r} has no topics"
            for topic in topics:
                assert isinstance(topic, str) and len(topic) > 10, f"Short topic in {category}: {topic!r}"


# ===================================================================
#  extraction_models
# ===================================================================


class TestChannelResult:
    def test_default_construction(self):
        r = ChannelResult(name="Test")
        assert r.name == "Test"
        assert r.transcripts == []
        assert r.conversations == []
        assert r.scores == []
        assert r.score_detail == []
        assert r.validation_report == {}

    def test_mean_score_empty(self):
        assert ChannelResult(name="X").mean_score == 0.0

    def test_mean_score_computed(self):
        r = ChannelResult(name="X", scores=[0.5, 1.0, 1.5])
        assert r.mean_score == 1.0

    def test_pass_rate_empty(self):
        assert ChannelResult(name="X").pass_rate == 0.0

    def test_pass_rate_threshold(self):
        # ≥ 0.5 passes
        r = ChannelResult(name="X", scores=[0.0, 0.49, 0.5, 0.7, 1.0])
        assert r.pass_rate == 3 / 5

    def test_high_quality_rate_empty(self):
        assert ChannelResult(name="X").high_quality_rate == 0.0

    def test_high_quality_rate_threshold(self):
        # ≥ 0.7 is high quality
        r = ChannelResult(name="X", scores=[0.5, 0.69, 0.7, 0.99])
        assert r.high_quality_rate == 2 / 4

    def test_all_properties_with_mixed_scores(self):
        r = ChannelResult(
            name="Test",
            scores=[0.1, 0.3, 0.5, 0.7, 0.9],
            transcripts=["a", "b"],
            conversations=[{"id": 1}],
        )
        assert r.mean_score == 0.5
        assert r.pass_rate == 3 / 5  # 0.5, 0.7, 0.9
        assert r.high_quality_rate == 2 / 5  # 0.7, 0.9
        assert len(r.transcripts) == 2
        assert len(r.conversations) == 1


# ===================================================================
#  synthetic_templates
# ===================================================================


class TestSyntheticTemplates:
    def test_has_50_templates(self):
        assert len(SYNTHETIC_TEMPLATES) == 50

    def test_every_template_has_client_and_therapist(self):
        for i, tpl in enumerate(SYNTHETIC_TEMPLATES):
            assert "client" in tpl, f"Template {i} missing 'client'"
            assert "therapist" in tpl, f"Template {i} missing 'therapist'"
            assert isinstance(tpl["client"], str)
            assert len(tpl["client"]) > 10
            assert isinstance(tpl["therapist"], str)
            assert len(tpl["therapist"]) > 10


# ===================================================================
#  extraction_io
# ===================================================================


class TestDeriveCommunicationPatterns:
    def test_empty_profile_returns_style_approach_only(self):
        patterns = _derive_communication_patterns({}, None)
        # Always at least empathy + style + approach
        assert len(patterns) >= 3

    def test_high_empathy_detected(self):
        profile = {"empathy_markers": {"I understand": 5, "I hear you": 4, "That's hard": 3}}
        patterns = _derive_communication_patterns(profile, None)
        emp = [p for p in patterns if "High empathy" in p]
        assert len(emp) >= 1

    def test_low_empathy_detected(self):
        profile = {"empathy_markers": {"I know": 1}}
        patterns = _derive_communication_patterns(profile, None)
        emp = [p for p in patterns if "Empathy expressed" in p]
        assert len(emp) >= 1

    def test_analytical_transitions(self):
        profile = {
            "transition_phrases": {"But": 10, "What happens": 5, "The reality is": 3, "So": 1},
            "empathy_markers": {"ok": 1},
        }
        patterns = _derive_communication_patterns(profile, None)
        analytical = [p for p in patterns if "Analytical" in p]
        assert len(analytical) >= 1

    def test_config_values_appear_in_output(self):
        profile = {"empathy_markers": {"ok": 1}}
        config = {"style": "warm_clinical_hopeful", "approach": "clinical_warm"}
        patterns = _derive_communication_patterns(profile, config)
        style_lines = [p for p in patterns if "Style:" in p]
        approach_lines = [p for p in patterns if "Approach:" in p]
        assert len(style_lines) == 1
        assert "warm clinical hopeful" in style_lines[0]
        assert len(approach_lines) == 1
        assert "clinical warm" in approach_lines[0]

    def test_analogies_and_examples_listed(self):
        profile = {
            "empathy_markers": {"ok": 1},
            "analogies": ["like a fish", "like a tree", "as if running", "imagine a river"],
            "examples": ["for example one", "for example two", "think back to", "let's say"],
        }
        patterns = _derive_communication_patterns(profile, None)
        assert any("analogies" in p.lower() or "metaphors" in p.lower() for p in patterns)
        assert any("examples" in p.lower() for p in patterns)


class TestDeriveToneCharacteristics:
    def test_default_values(self):
        result = _derive_tone_characteristics({"empathy_markers": {}, "transition_phrases": {}}, None)
        assert "empathy_level" in result
        assert "formality" in result
        assert "pacing" in result
        assert "emotional_temperature" in result

    def test_clinical_style_formality(self):
        result = _derive_tone_characteristics(
            {"empathy_markers": {"ok": 1}, "transition_phrases": {"Now": 1}},
            {"style": "professional_educational_supportive"},
        )
        assert result["formality"] == "professional_structured"

    def test_conversational_style(self):
        result = _derive_tone_characteristics(
            {"empathy_markers": {"ok": 1}, "transition_phrases": {"So": 1}},
            {"style": "conversational_expert_interview"},
        )
        assert result["formality"] == "conversational_accessible"

    def test_high_empathy_level(self):
        result = _derive_tone_characteristics(
            {"empathy_markers": {"I understand": 5, "I hear you": 4, "That's hard": 3}},
            {"style": "warm"},
        )
        assert result["empathy_level"] == "high"

    def test_warm_temperature(self):
        result = _derive_tone_characteristics(
            {"empathy_markers": {"ok": 1}, "transition_phrases": {}},
            {"style": "compassionate_warm"},
        )
        assert result["emotional_temperature"] == "warm_supportive"

    def test_authoritative_temperature(self):
        result = _derive_tone_characteristics(
            {"empathy_markers": {"ok": 1}, "transition_phrases": {}},
            {"style": "authoritative_clinical"},
        )
        assert result["emotional_temperature"] == "authoritative_grounded"


class TestGenerateQualityReport:
    def test_empty_results(self):
        report = generate_quality_report([])
        assert report.startswith("# Clinical Validity Quality Report")
        assert "No validation issues" in report

    def test_single_result(self):
        r = ChannelResult(name="Test", scores=[0.5, 0.7], score_detail=[{"technique": 0.6, "alliance": 0.5}])
        report = generate_quality_report([r])
        assert "Test" in report
        assert "0.5000" in report or "0.500" in report
        assert "2" in report  # score count

    def test_dimension_averages_with_multiple_details(self):
        r = ChannelResult(
            name="A",
            scores=[0.6, 0.7],
            score_detail=[
                {"technique": 0.5, "alliance": 0.6, "structure": 0.7, "cultural": 0.3, "ebp": 0.4},
                {"technique": 0.7, "alliance": 0.4, "structure": 0.5, "cultural": 0.5, "ebp": 0.6},
            ],
        )
        report = generate_quality_report([r])
        assert "Technique" in report
        assert "Alliance" in report
        assert "0.600" in report  # avg technique = 0.6

    def test_validation_issues_appear(self):
        r = ChannelResult(
            name="Test",
            scores=[0.5],
            validation_report={"issues": ["Low message count"]},
        )
        report = generate_quality_report([r])
        assert "Low message count" in report

    def test_no_issues_message(self):
        r = ChannelResult(name="Test", scores=[0.5], validation_report={})
        report = generate_quality_report([r])
        assert "No validation issues detected" in report

    def test_overall_row_present(self):
        r = ChannelResult(name="Test", scores=[0.3, 0.7])
        report = generate_quality_report([r])
        assert "**Overall**" in report


class TestSaveChannelOutput:
    def test_saves_profile_json_and_scores_csv(self):
        """Verify files created and their content."""
        r = ChannelResult(
            name="Doc Snipes",
            transcripts=["hello world"],
            voice_profile={
                "profile_raw": {
                    "sentence_starters": {"I think": 2},
                    "empathy_markers": {"I understand": 1},
                    "common_phrases": {"the key is": 3},
                    "transition_phrases": {"So": 1},
                    "analogies": [],
                    "examples": [],
                    "teaching_patterns": [],
                    "common_phrases": {},
                },
                "report": "# Analysis Report",
            },
            conversations=[{"conversation_id": "test_001"}],
            scores=[0.75],
            score_detail=[{"technique": 0.7, "alliance": 0.6, "structure": 0.5, "cultural": 0.4, "ebp": 0.8}],
        )

        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            save_channel_output("DocSnipes", r, out)

            sig = "doc_snipes"
            channel_dir = out / f"{sig}_voice"
            exports_dir = channel_dir / "exports"

            # Profile JSON
            profile_file = channel_dir / "docsnipes_voice_profile.json"
            assert profile_file.exists(), f"Missing {profile_file}"
            data = json.loads(profile_file.read_text())
            assert data["name"] == "Doc Snipes"
            assert data["clinical_validity_score"] == 0.75

            # Analysis report MD
            report_file = channel_dir / "docsnipes_voice_analysis.md"
            assert report_file.exists()

            # Conversations JSONL
            conv_file = exports_dir / "docsnipes_conversations.jsonl"
            assert conv_file.exists()
            lines = conv_file.read_text().strip().split("\n")
            assert len(lines) == 1
            assert '"conversation_id": "test_001"' in lines[0]

            # Scores CSV
            scores_file = channel_dir / "docsnipes_clinical_scores.csv"
            assert scores_file.exists()
            reader = csv.DictReader(scores_file.read_text().splitlines())
            rows = list(reader)
            assert len(rows) == 1
            assert round(float(rows[0]["score"]), 4) == 0.75

    def test_no_profile_does_not_crash(self):
        r = ChannelResult(name="Empty", scores=[])
        with tempfile.TemporaryDirectory() as tmp:
            save_channel_output("Empty", r, Path(tmp))
            # Should not raise


# ===================================================================
#  extract_therapist_voice — helpers
# ===================================================================


class TestStripTranscriptMetadata:
    def test_strips_metadata_before_transcript_section(self):
        content = """# Title
Some frontmatter
## Transcript
This is a sufficiently long paragraph of transcript content that exceeds the one hundred character threshold required by the stripping function to be considered a valid paragraph for inclusion in the output.
"""
        result = _strip_transcript_metadata(content)
        cleaned = "\n".join(result)
        assert "Title" not in cleaned
        assert "frontmatter" not in cleaned
        assert "transcript content" in cleaned

    def test_empty_when_no_transcript_section(self):
        content = "# Just metadata\nNo transcript section.\n"
        result = _strip_transcript_metadata(content)
        assert result == []

    def test_short_paragraphs_excluded(self):
        content = "## Transcript\n\nShort.\n\nThis is a long enough paragraph that clearly exceeds the one hundred character minimum threshold required by the stripping logic for inclusion in the output data.\n"
        result = _strip_transcript_metadata(content)
        assert "Short." not in "\n".join(result)
        assert "long enough paragraph" in "\n".join(result)


class TestGetTopicsForExpertise:
    def test_trauma_expertise_returns_trauma_topics(self):
        topics = _get_topics_for_expertise(["complex_trauma", "PTSD"])
        for t in topics:
            assert any("trauma" in t.lower() or "nervous" in t.lower() for t in topics), "No trauma topics found"

    def test_personality_expertise(self):
        topics = _get_topics_for_expertise(["narcissistic_abuse", "BPD"])
        personality = any("narcissistic" in t.lower() or "boundar" in t.lower() for t in topics)
        assert personality, "No personality-disorder topics found"

    def test_attachment_expertise(self):
        topics = _get_topics_for_expertise(["attachment_theory", "attachment_wounds"])
        assert any("attachment" in t.lower() for t in topics)

    def test_cbt_dbt_expertise(self):
        topics = _get_topics_for_expertise(["CBT", "emotion_regulation"])
        assert any("cognitive" in t.lower() or "distress" in t.lower() for t in topics)

    def test_fallback_to_general(self):
        topics = _get_topics_for_expertise(["something_obscure"])
        assert len(topics) > 0
        # Should come from general
        assert any("Managing anxiety" in t for t in topics)


class TestBuildSyntheticDialogue:
    def test_basic_structure(self):
        config = CHANNEL_CONFIGS["DocSnipes"]
        dialog = _build_synthetic_dialogue(0, "Managing anxiety", config)
        assert dialog["conversation_id"] == "doc_snipes_synthetic_0000"
        assert dialog["stage"] == "stage4_voice_persona"
        assert len(dialog["messages"]) == 2
        assert dialog["messages"][0]["role"] == "client"
        assert dialog["messages"][1]["role"] == "therapist"
        assert dialog["metadata"]["topic"] == "Managing anxiety"
        assert dialog["metadata"]["index"] == 0

    def test_conversation_id_format(self):
        config = CHANNEL_CONFIGS["DoctorRamani"]
        for i in [0, 1, 99]:
            dialog = _build_synthetic_dialogue(i, "Test", config)
            expected = f"doctor_ramani_synthetic_{i:04d}"
            assert dialog["conversation_id"] == expected

    def test_random_template_choice(self):
        config = CHANNEL_CONFIGS["DocSnipes"]
        dialogs = [_build_synthetic_dialogue(i, "Topic", config) for i in range(100)]
        # With random choice from 50 templates, we should see variation
        contents = {d["messages"][0]["content"] for d in dialogs}
        assert len(contents) > 1, "No template variation detected"

    def test_config_values_in_metadata(self):
        config = CHANNEL_CONFIGS["TimFletcher"]
        dialog = _build_synthetic_dialogue(0, "Trauma", config)
        markers = dialog["metadata"]["personality_markers"]
        assert markers["style"] == "compassionate_educational_step_by_step"
        assert markers["approach"] == "trauma_informed_psychoeducation"


# ===================================================================
#  extract_therapist_voice — conversation generation
# ===================================================================


class TestGenerateConversationFromTranscript:
    def test_basic_structure(self):
        config = CHANNEL_CONFIGS["DocSnipes"]
        content = "## Transcript\n\nThis is a substantial paragraph of transcript data that is long enough to pass the minimum threshold of one hundred characters as required by the stripping logic.\n"
        conv = generate_conversation_from_transcript("test_title", content, config)
        assert conv["conversation_id"] == "doc_snipes_test_title"
        assert conv["stage"] == "stage4_voice_persona"
        assert "messages" in conv
        # system + client + therapist
        assert len(conv["messages"]) >= 3
        assert conv["messages"][0]["role"] == "system"
        assert conv["messages"][1]["role"] == "client"
        assert conv["messages"][2]["role"] == "therapist"

    def test_system_prompt_contains_channel_description(self):
        config = CHANNEL_CONFIGS["DocSnipes"]
        content = "## Transcript\n\nDummy paragraph that is quite long indeed to meet the length threshold for a valid paragraph.\n"
        conv = generate_conversation_from_transcript("title", content, config)
        system_msg = conv["messages"][0]["content"]
        assert "Doc Snipes" in system_msg
        assert "clinical_educational" in system_msg

    def test_picks_client_questions_cyclically(self):
        config = CHANNEL_CONFIGS["DocSnipes"]
        # Multiple paragraphs to test cycling through question templates
        paras = "\n\n".join(
            [
                f"This is paragraph number {i} that is definitely long enough to pass the hundred character threshold for content extraction."
                for i in range(5)
            ]
        )
        content = f"## Transcript\n\n{paras}\n"
        conv = generate_conversation_from_transcript("multi", content, config)
        client_msgs = [m for m in conv["messages"] if m["role"] == "client"]
        assert len(client_msgs) == 5
        # All client messages should be from the template set
        for msg in client_msgs:
            assert msg["content"] in _CLIENT_QUESTION_TEMPLATES

    def test_metadata_contains_channel_info(self):
        config = CHANNEL_CONFIGS["PatrickTeahan"]
        content = "## Transcript\n\nA sufficiently long paragraph that meets the one hundred character threshold for inclusion in the training conversation.\n"
        conv = generate_conversation_from_transcript("meta_test", content, config)
        meta = conv["metadata"]
        assert meta["persona_id"] == "patrick_teahan"
        assert meta["voice_signature"] == "patrick_teahan_v1"
        assert meta["source_family"] == "stage4_voice_persona"


class TestGenerateSyntheticConversations:
    def test_returns_expected_count(self):
        config = CHANNEL_CONFIGS["DocSnipes"]
        convs = generate_synthetic_conversations(config, 5)
        assert len(convs) == 5

    def test_each_has_system_message(self):
        config = CHANNEL_CONFIGS["DocSnipes"]
        convs = generate_synthetic_conversations(config, 10)
        for conv in convs:
            assert conv["messages"][0]["role"] == "system"

    def test_unique_ids(self):
        config = CHANNEL_CONFIGS["DocSnipes"]
        convs = generate_synthetic_conversations(config, 20)
        ids = [c["conversation_id"] for c in convs]
        assert len(ids) == len(set(ids))

    def test_topics_distributed_from_expertise(self):
        config = CHANNEL_CONFIGS["CrappyChildhoodFairy"]
        convs = generate_synthetic_conversations(config, 5)
        for conv in convs:
            assert "topic" in conv["metadata"]
            assert len(conv["metadata"]["topic"]) > 0


# ===================================================================
#  extract_therapist_voice — scoring
# ===================================================================


class TestScoreConversations:
    def test_empty_conversations(self):
        scores, details = score_conversations([])
        assert scores == []
        assert details == []

    def test_scoring_skipped_when_scorer_unavailable(self, monkeypatch):
        monkeypatch.setattr("scripts.extract_therapist_voice.SCORER_AVAILABLE", False)
        conv = [{"messages": [{"role": "therapist", "content": "Test response"}]}]
        scores, details = score_conversations(conv)
        assert scores == []
        assert details == []

    def test_skips_non_therapist_messages(self):
        conv = [
            {
                "messages": [
                    {"role": "system", "content": "You are a therapist."},
                    {"role": "client", "content": "I need help."},
                    {"role": "therapist", "content": "That sounds difficult."},
                    {"role": "client", "content": "Yes it is."},
                ]
            }
        ]
        scores, details = score_conversations(conv)
        if scores:  # scorer may be available
            assert len(scores) == 1
            # Only therapist messages contribute
            assert len(details) == 1

    def test_empty_therapist_content_gives_zero_score(self):
        conv = [{"messages": [{"role": "therapist", "content": ""}]}]
        scores, _details = score_conversations(conv)
        if scores:
            assert scores[0] == 0.0


class TestAnnotateConversations:
    def test_adds_clinical_validity_metadata(self):
        convs = [{"messages": [{"role": "therapist", "content": "Test"}], "metadata": {}}]
        annotate_conversations(
            convs, [0.75], [{"technique": 0.7, "alliance": 0.6, "structure": 0.5, "cultural": 0.4, "ebp": 0.8}]
        )
        cv = convs[0]["metadata"]["clinical_validity"]
        assert cv["score"] == 0.75
        assert cv["dimensions"]["technique"] == 0.7

    def test_missing_metadata_creates_it(self):
        convs = [{"messages": []}]
        annotate_conversations(convs, [0.5], [{"technique": 0.5}])
        assert convs[0]["metadata"]["clinical_validity"]["score"] == 0.5


# ===================================================================
#  extract_therapist_voice — validation
# ===================================================================


class TestValidateConversationQuality:
    def test_passes_valid_conversations(self):
        convs = [
            {
                "messages": [
                    {"role": "system", "content": "You are a therapist."},
                    {"role": "client", "content": "Help."},
                    {"role": "therapist", "content": "I hear you."},
                ],
                "metadata": {"clinical_validity": {"score": 0.5}},
            }
        ]
        result = validate_conversation_quality(convs)
        assert result["pass"] is True, f"Issues found: {result['issues']}"

    def test_flags_fewer_than_3_messages(self):
        convs = [
            {
                "messages": [
                    {"role": "system", "content": "You are a therapist."},
                    {"role": "client", "content": "Hi."},
                ],
                "metadata": {},
            }
        ]
        result = validate_conversation_quality(convs)
        assert not result["pass"]
        assert any("fewer than 3" in i for i in result["issues"])

    def test_flags_invalid_roles(self):
        convs = [
            {
                "messages": [
                    {"role": "system", "content": "You are X."},
                    {"role": "client", "content": "Hi."},
                    {"role": "hacker", "content": "pwnd"},
                ],
                "metadata": {},
            }
        ]
        result = validate_conversation_quality(convs)
        assert any("invalid roles" in i for i in result["issues"])

    def test_flags_empty_content(self):
        convs = [
            {
                "messages": [
                    {"role": "system", "content": "You are X."},
                    {"role": "client", "content": ""},
                    {"role": "therapist", "content": "  "},
                ],
                "metadata": {},
            }
        ]
        result = validate_conversation_quality(convs)
        assert any("empty message" in i for i in result["issues"])

    def test_warns_high_average_score(self):
        convs = [
            {
                "messages": [{"role": "therapist", "content": "good"}],
                "metadata": {"clinical_validity": {"score": 0.95}},
            }
        ]
        result = validate_conversation_quality(convs)
        assert any("High average" in i for i in result["issues"])

    def test_notes_no_scores(self):
        convs = [
            {
                "messages": [{"role": "therapist", "content": "good"}],
                "metadata": {},
            }
        ]
        result = validate_conversation_quality(convs)
        assert any("No scores found" in i for i in result["issues"])

    def test_validates_multiple_conversations(self):
        convs = [
            {
                "messages": [
                    {"role": "system", "content": "A"},
                    {"role": "client", "content": "B"},
                    {"role": "therapist", "content": "C"},
                ],
                "metadata": {"clinical_validity": {"score": 0.5}},
            },
            {
                "messages": [{"role": "invalid", "content": "X"}],
                "metadata": {"clinical_validity": {"score": 0.3}},
            },
        ]
        result = validate_conversation_quality(convs)
        assert result["total"] == 2
        assert result["details"]["role_violations"] >= 1


# ===================================================================
#  extract_therapist_voice — CLI arg parsing
# ===================================================================


class TestParseArgs:
    def test_defaults(self):
        args = parse_args([])
        assert args.num_conversations == 50
        assert args.save is True
        assert args.no_score is False
        assert args.force_synthetic is False
        assert args.quiet is False

    def test_channel_arg(self):
        args = parse_args(["--channel", "DocSnipes"])
        assert args.channel == "DocSnipes"

    def test_all_flag(self):
        args = parse_args(["--all"])
        assert args.all is True

    def test_list_flag(self):
        args = parse_args(["--list"])
        assert args.list is True

    def test_custom_conversations(self):
        args = parse_args(["--num-conversations", "25"])
        assert args.num_conversations == 25

    def test_no_score(self):
        args = parse_args(["--no-score"])
        assert args.no_score is True

    def test_force_synthetic(self):
        args = parse_args(["--force-synthetic"])
        assert args.force_synthetic is True

    def test_quiet(self):
        args = parse_args(["--quiet"])
        assert args.quiet is True

    def test_short_flags(self):
        args = parse_args(["-c", "Test", "-n", "10", "-s"])
        assert args.channel == "Test"
        assert args.num_conversations == 10
        assert args.force_synthetic is True

    def test_list_takes_precedence(self):
        """--list should not require other args."""
        args = parse_args(["--list"])
        assert args.list is True
