import pytest
from utils.transcript_corrector import TranscriptCorrector

def test_transcript_corrector_edge_cases():
    corrector = TranscriptCorrector(config_path="config/therapeutic_terminology.json")

    # Test empty string
    assert corrector.correct_transcript("") == ""

    # Test string with only whitespace
    assert corrector.correct_transcript("   \n  \t  ") == ""

    # Test structure cleaning (filler words)
    assert corrector._clean_structure("um, well, like, you know, I feel bad") == "well, I feel bad"

    # Test terminology replacement
    assert corrector._apply_terminology_fixes("I had EMD R therapy") == "I had EMDR therapy"

    # Test full pass with filler words and bad terminology
    result = corrector.correct_transcript("um, I think I need C B T or D B T, you know")
    assert result == "I think I need CBT or DBT,"

    # Test coverage calculations
    coverage = corrector.validate_term_coverage("EMDR and CBT are very helpful for C-PTSD.")
    assert coverage["cptsd_term_count"] == 0
    assert coverage["medical_term_count"] == 2
    assert coverage["domain_coverage_score"] > 0
