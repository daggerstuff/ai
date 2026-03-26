import pytest
from unittest.mock import patch, mock_open
from utils.transcript_corrector import TranscriptCorrector

def test_correct_transcript_empty():
    """Test that correct_transcript returns empty string for empty input."""
    with patch('utils.transcript_corrector.Path.exists', return_value=True), \
         patch('builtins.open', new_callable=mock_open, read_data='{}'):
        corrector = TranscriptCorrector("dummy_path")

        # Test empty string
        assert corrector.correct_transcript("") == ""

        # Test whitespace string
        assert corrector.correct_transcript("   ") == ""

        # Test None - typehint says string but we handle falsy implicitly, let's verify
        assert corrector.correct_transcript(None) == ""
