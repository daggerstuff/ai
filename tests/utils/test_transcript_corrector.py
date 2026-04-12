from unittest.mock import mock_open, patch

from utils.transcript_corrector import TranscriptCorrector


@patch("utils.transcript_corrector.Path.exists", return_value=True)
@patch("builtins.open", new_callable=mock_open, read_data='{"common_misinterpretations": {"bad": "good"}}')
def test_clean_structure_edge_cases(mock_file, mock_exists):
    corrector = TranscriptCorrector("mock.json")

    assert corrector.correct_transcript("") == ""
    assert corrector.correct_transcript("   ") == ""
    assert corrector.correct_transcript("um, uh, like you know") == ""
    assert corrector.correct_transcript("this is bad") == "this is good"

@patch("utils.transcript_corrector.Path.exists", return_value=True)
@patch("builtins.open", new_callable=mock_open, read_data='{"cptsd_terms": ["trauma"], "medical_terms": ["brain"]}')
def test_validate_term_coverage(mock_file, mock_exists):
    corrector = TranscriptCorrector("mock.json")

    metrics = corrector.validate_term_coverage("Trauma affects the brain.")
    assert metrics["cptsd_term_count"] == 1
    assert metrics["medical_term_count"] == 1
    assert metrics["domain_coverage_score"] == 1.0

    metrics = corrector.validate_term_coverage("Nothing here.")
    assert metrics["domain_coverage_score"] == 0.0


@patch("utils.transcript_corrector.Path.exists", return_value=True)
@patch("builtins.open", new_callable=mock_open, read_data='{"common_misinterpretations": {"bad term": "good term"}}')
def test_apply_terminology_fixes_edge_cases(mock_file, mock_exists):
    corrector = TranscriptCorrector("mock.json")

    assert corrector._apply_terminology_fixes("This is a BAD TERM.") == "This is a good term."
    assert corrector._apply_terminology_fixes("bad term and Bad Term") == "good term and good term"
