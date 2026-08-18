from unittest.mock import mock_open, patch

from utils.transcript_corrector import TranscriptCorrector


@patch("utils.transcript_corrector.Path.exists", return_value=True)
@patch(
    "builtins.open",
    new_callable=mock_open,
    read_data=('{"cptsd_terms": [], "medical_terms": [], "common_misinterpretations": {}}'),
)
def test_transcript_corrector_empty(_mock_file, _mock_exists):
    corrector = TranscriptCorrector(config_path="dummy.json")
    result = corrector.validate_term_coverage("")
    assert result["cptsd_term_count"] == 0
    assert result["medical_term_count"] == 0
    assert result["domain_coverage_score"] == 0.0


@patch("utils.transcript_corrector.Path.exists", return_value=True)
@patch(
    "builtins.open",
    new_callable=mock_open,
    read_data=('{"cptsd_terms": ["trauma"], "medical_terms": ["flashback"], "common_misinterpretations": {}}'),
)
def test_transcript_corrector_fillers(_mock_file, _mock_exists):
    corrector = TranscriptCorrector(config_path="dummy.json")
    # Full match
    result = corrector.validate_term_coverage("Trauma can cause a flashback.")
    assert result["cptsd_term_count"] == 1
    assert result["medical_term_count"] == 1
    assert result["domain_coverage_score"] == 1.0

    # Case insensitive test
    result = corrector.validate_term_coverage("FLASHBACK and TRAUMA")
    assert result["cptsd_term_count"] == 1
    assert result["medical_term_count"] == 1
    assert result["domain_coverage_score"] == 1.0
