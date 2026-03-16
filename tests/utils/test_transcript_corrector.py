import pytest
import os
import tempfile
import json
from utils.transcript_corrector import TranscriptCorrector

def test_transcript_corrector_handles_empty_and_null_inputs():
    """
    Test that TranscriptCorrector correctly handles empty strings,
    whitespace-only strings, and null inputs without throwing errors.
    """
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json.dump({
            "cptsd_terms": [],
            "medical_terms": [],
            "common_misinterpretations": {}
        }, f)
        temp_config = f.name

    try:
        corrector = TranscriptCorrector(config_path=temp_config)

        # Assert empty string is handled
        assert corrector.correct_transcript("") == ""

        # Assert whitespace-only string is handled
        assert corrector.correct_transcript("   \n   \t  ") == ""

        # Assert None is handled
        assert corrector.correct_transcript(None) == ""
    finally:
        os.unlink(temp_config)
