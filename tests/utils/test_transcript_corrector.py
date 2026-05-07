from unittest.mock import mock_open, patch

from utils.transcript_corrector import TranscriptCorrector


@patch("utils.transcript_corrector.Path.exists", return_value=True)
@patch("builtins.open", new_callable=mock_open, read_data='{"common_misinterpretations": {"bad": "good"}}')
def test_clean_structure_edge_cases(_mock_file, _mock_exists):
    corrector = TranscriptCorrector("mock.json")

    assert corrector.correct_transcript("") == ""
    assert corrector.correct_transcript("   ") == ""
    assert corrector.correct_transcript("um, uh, like you know") == ""
    assert corrector.correct_transcript("this is bad") == "This is good"

@patch("utils.transcript_corrector.Path.exists", return_value=True)
@patch("builtins.open", new_callable=mock_open, read_data='{"cptsd_terms": ["trauma"], "medical_terms": ["brain"]}')
def test_validate_term_coverage(_mock_file, _mock_exists):
    corrector = TranscriptCorrector("mock.json")

    metrics = corrector.validate_term_coverage("Trauma affects the brain.")
    assert metrics["cptsd_term_count"] == 1
    assert metrics["medical_term_count"] == 1
    assert metrics["domain_coverage_score"] == 1.0

    metrics = corrector.validate_term_coverage("Nothing here.")
    assert metrics["domain_coverage_score"] == 0.0


@patch("utils.transcript_corrector.Path.exists", return_value=True)
@patch("builtins.open", new_callable=mock_open, read_data='{"common_misinterpretations": {"bad term": "good term"}}')
def test_apply_terminology_fixes_edge_cases(_mock_file, _mock_exists):
    corrector = TranscriptCorrector("mock.json")

    assert corrector._apply_terminology_fixes("This is a BAD TERM.") == "This is a good term."
    assert corrector._apply_terminology_fixes("bad term and Bad Term") == "good term and good term"


@patch("utils.transcript_corrector.Path.exists", return_value=True)
@patch("builtins.open", new_callable=mock_open, read_data='{}')
def test_correct_transcript_with_punctuation_and_capitalization(_mock_file, _mock_exists):
    corrector = TranscriptCorrector("mock.json")

    # Tests repeated punctuation collapse and sentence capitalization
    assert corrector.correct_transcript("hello world!!  how are you??") == "Hello world! How are you?"

    # Tests automatic capitalization of "i"
    assert corrector.correct_transcript("i think i am fine.") == "I think I am fine."

    # Tests specific CPTSD safety net replacements like im and i'd
    assert corrector.correct_transcript("im going home.") == "I'm going home."
    assert corrector.correct_transcript("i'd want to talk.") == "I'd want to talk."

    # Tests spacing around punctuation
    assert corrector.correct_transcript("wait , what ?") == "Wait, what?"

from unittest.mock import MagicMock

def test_missing_config_path_fallback():
    with patch("utils.transcript_corrector.Path.exists", side_effect=[False, True]):
        with patch("builtins.open", new_callable=mock_open, read_data='{"cptsd_terms": ["trauma"]}'):
            corrector = TranscriptCorrector("mock.json")
            assert corrector.terms["cptsd_terms"] == ["trauma"]

def test_missing_config_path_fallback_both_missing():
    with patch("utils.transcript_corrector.Path.exists", side_effect=[False, False]):
        corrector = TranscriptCorrector("mock.json")
        assert corrector.terms["cptsd_terms"] == []
        assert corrector.terms["medical_terms"] == []
        assert corrector.terms["common_misinterpretations"] == {}

def test_load_terminology_exception():
    with patch("utils.transcript_corrector.Path.exists", return_value=True):
        with patch("builtins.open", side_effect=Exception("Test Exception")):
            corrector = TranscriptCorrector("mock.json")
            assert corrector.terms["cptsd_terms"] == []

def test_contextual_correction_client_fallback():
    with patch("utils.transcript_corrector.Path.exists", return_value=True):
        with patch("builtins.open", new_callable=mock_open, read_data='{}'):
            corrector = TranscriptCorrector("mock.json")

            # Setup a client that returns valid output
            corrector._contextual_correction_client = MagicMock(return_value="Corrected sentence.")
            result = corrector._llm_contextual_correction("original sentence", "context")
            assert result == "Corrected sentence."

            # Setup a client that returns empty/invalid output, should fallback
            corrector._contextual_correction_client = MagicMock(return_value="   ")
            result = corrector._llm_contextual_correction("original sentence.", "context")
            # The fallback processing happens in llm_contextual_correction
            assert result == "Original sentence."
