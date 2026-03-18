from utils.transcript_corrector import TranscriptCorrector


def test_clean_structure_removes_fillers():
    corrector = TranscriptCorrector(config_path="dummy.json")
    text = "I mean, it was, uh, really difficult."
    cleaned = corrector._clean_structure(text)
    assert cleaned == "it was, really difficult."


def test_correct_transcript_empty():
    corrector = TranscriptCorrector(config_path="dummy.json")
    assert corrector.correct_transcript("") == ""
    assert corrector.correct_transcript("   ") == ""
    assert corrector.correct_transcript(None) == ""


def test_clean_structure_complex():
    corrector = TranscriptCorrector(config_path="dummy.json")
    text = "um,   this is    like,   really hard, you know?"
    cleaned = corrector._clean_structure(text)
    assert cleaned == "this is really hard, ?"
