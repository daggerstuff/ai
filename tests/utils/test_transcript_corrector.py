from utils.transcript_corrector import TranscriptCorrector


def test_clean_structure_removes_fillers():
    """Verify that _clean_structure removes filler words correctly."""
    corrector = TranscriptCorrector(config_path="dummy_path.json")
    text = "So, um, like, I went to the store, you know."
    result = corrector._clean_structure(text)
    assert result == "So, I went to the store, ."
