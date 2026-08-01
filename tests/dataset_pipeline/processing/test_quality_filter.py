import pytest
from ai.dataset_pipeline.processors.quality_filter import QualityFilter


def test_quality_filter_valid_conversation():
    qf = QualityFilter()
    valid_record = {
        "messages": [
            {"role": "user", "content": "How can I handle anxiety when working from home?"},
            {"role": "assistant", "content": "Working from home can blur boundaries. Setting clear work hours and taking short regular breaks can help ground your routine."},
        ]
    }
    assert qf.passes_filter(valid_record) is True


def test_quality_filter_deduplicates_exact_matches():
    qf = QualityFilter()
    record = {
        "messages": [
            {"role": "user", "content": "What is cognitive reframing?"},
            {"role": "assistant", "content": "Cognitive reframing involves identifying negative thoughts and replacing them with balanced perspectives."},
        ]
    }
    assert qf.passes_filter(record) is True
    # Second time should fail exact hash check
    assert qf.passes_filter(record) is False


def test_quality_filter_fingerprint_deduplication():
    qf = QualityFilter()
    record1 = {
        "messages": [
            {"role": "user", "content": "What is cognitive reframing?   "},
            {"role": "assistant", "content": "Cognitive reframing involves identifying negative thoughts and replacing them with balanced perspectives."},
        ]
    }
    record2 = {
        "messages": [
            {"role": "user", "content": "What is COGNITIVE reframing?"},
            {"role": "assistant", "content": "Cognitive reframing involves identifying negative thoughts and replacing them with balanced perspectives. Extra tail!"},
        ]
    }
    assert qf.passes_filter(record1) is True
    # Second record shares same prompt/response prefix fingerprint, so it should be rejected
    assert qf.passes_filter(record2) is False


def test_quality_filter_ngram_repetition():
    qf = QualityFilter()
    record_with_repetition = {
        "messages": [
            {"role": "user", "content": "Tell me about panic attacks."},
            {
                "role": "assistant",
                "content": "Panic attacks are intense. Panic attacks are intense. Panic attacks are intense. They cause rapid heartbeats.",
            },
        ]
    }
    assert qf.passes_filter(record_with_repetition) is False


def test_quality_filter_word_density():
    qf = QualityFilter()
    record_high_density = {
        "messages": [
            {"role": "user", "content": "I feel lost."},
            {
                "role": "assistant",
                "content": "lost lost lost lost lost lost lost lost lost lost lost lost lost lost lost",
            },
        ]
    }
    assert qf.passes_filter(record_high_density) is False
