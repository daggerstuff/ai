import pytest
from unittest.mock import patch, mock_open
from utils.transcript_corrector import TranscriptCorrector

@patch("utils.transcript_corrector.Path.exists", return_value=True)
@patch("builtins.open", new_callable=mock_open, read_data='{"cptsd_terms": [], "medical_terms": [], "common_misinterpretations": {}}')
def test_transcript_corrector_empty(mock_file, mock_exists):
    corrector = TranscriptCorrector(config_path="dummy.json")

    assert corrector.correct_transcript("") == ""
    assert corrector.correct_transcript("   ") == ""
    assert corrector.correct_transcript(None) == ""

@patch("utils.transcript_corrector.Path.exists", return_value=True)
@patch("builtins.open", new_callable=mock_open, read_data='{"cptsd_terms": [], "medical_terms": [], "common_misinterpretations": {}}')
def test_transcript_corrector_fillers(mock_file, mock_exists):
    corrector = TranscriptCorrector(config_path="dummy.json")

    # "well" is not a filler in the regex, "um,", "you know,", "like,", "uh" are
    assert corrector.correct_transcript("um, well you know, I think, like, this is uh good.") == "well I think, this is good."
