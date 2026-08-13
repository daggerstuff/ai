"""Tests for Stage 1 QA filter chain (PIX-4342).

All heavy ML models (fasttext, presidio, detoxify, datasketch) are mocked
so tests run without GPU, model downloads, or optional dependencies.
"""

from __future__ import annotations

from unittest.mock import MagicMock

from training.stage1_filters import (
    ChainStats,
    DedupFilter,
    FilterResult,
    LanguageFilter,
    PIIFilter,
    Stage1FilterChain,
    ToxicityFilter,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_record(text: str = "Hello, how are you feeling today?", source: str = "test") -> dict:
    return {
        "messages": [
            {"role": "user", "content": text},
            {"role": "assistant", "content": "I'm here to support you."},
        ],
        "source": source,
        "task_type": "therapy_response_generation",
    }


def make_lang_filter(lang: str = "en") -> LanguageFilter:
    mock_model = MagicMock()
    mock_model.predict.return_value = ([f"__label__{lang}"], [0.99])
    return LanguageFilter(model=mock_model)


def make_pii_filter(
    results: list | None = None, llm_borderline: bool = False, llm_client: MagicMock | None = None
) -> PIIFilter:
    mock_analyzer = MagicMock()
    mock_analyzer.analyze.return_value = results or []
    return PIIFilter(analyzer=mock_analyzer, llm_borderline=llm_borderline, llm_client=llm_client)


def make_toxicity_filter(scores: dict[str, float] | None = None) -> ToxicityFilter:
    scores = scores or {"severe_toxicity": 0.01, "threat": 0.01}
    mock_model = MagicMock()
    mock_model.predict.return_value = scores
    models = dict.fromkeys(ToxicityFilter.MODEL_NAMES, mock_model)
    return ToxicityFilter(models=models)


class TestLanguageFilter:
    def test_english_passes(self):
        f = make_lang_filter("en")
        result = f(make_record("I feel sad today."))
        assert result.passed is True
        assert result.metadata["lang"] == "en"

    def test_non_english_dropped(self):
        f = make_lang_filter("zh")
        result = f(make_record("I feel sad today."))
        assert result.passed is False
        assert "non_english" in result.reason
        assert result.metadata["lang"] == "zh"

    def test_regex_overrides_fasttext(self):
        f = make_lang_filter("en")
        cjk_text = "你好你好你好你好你好你好你好你好你好你好"
        record = make_record(cjk_text)
        result = f(record)
        assert result.passed is False
        assert result.reason == "non_english_regex_override"

    def test_empty_text_dropped(self):
        f = make_lang_filter("en")
        result = f({"messages": [], "source": "test"})
        assert result.passed is False
        assert result.reason == "empty_text"


class TestPIIFilter:
    def test_clean_record_passes(self):
        f = make_pii_filter(results=[])
        result = f(make_record("I am feeling sad."))
        assert result.passed is True
        assert result.metadata["pii_found"] is False

    def test_confirmed_pii_dropped(self):
        pii_result = MagicMock()
        pii_result.entity_type = "EMAIL_ADDRESS"
        pii_result.score = 0.95
        f = make_pii_filter(results=[pii_result])
        result = f(make_record("Contact me at john@example.com"))
        assert result.passed is False
        assert "pii_confirmed" in result.reason
        assert "EMAIL_ADDRESS" in result.metadata["entity_types"]

    def test_borderline_without_llm_passes(self):
        borderline = MagicMock()
        borderline.entity_type = "PHONE_NUMBER"
        borderline.score = 0.5
        f = make_pii_filter(results=[borderline], llm_borderline=False)
        result = f(make_record("Call me at 555-1234"))
        assert result.passed is True
        assert result.reason == "pii_borderline_no_llm"
        assert result.metadata["pii_found"] is False

    def test_borderline_with_llm_confirmed(self):
        borderline = MagicMock()
        borderline.entity_type = "PHONE_NUMBER"
        borderline.score = 0.5
        mock_llm = MagicMock()
        mock_llm.generate.return_value = "YES"
        f = make_pii_filter(results=[borderline], llm_borderline=True, llm_client=mock_llm)
        result = f(make_record("Call me at 555-1234"))
        assert result.passed is False
        assert "pii_llm_confirmed" in result.reason

    def test_borderline_with_llm_rejected(self):
        borderline = MagicMock()
        borderline.entity_type = "PHONE_NUMBER"
        borderline.score = 0.5
        mock_llm = MagicMock()
        mock_llm.generate.return_value = "NO"
        f = make_pii_filter(results=[borderline], llm_borderline=True, llm_client=mock_llm)
        result = f(make_record("Call me at 555-1234"))
        assert result.passed is True
        assert result.reason == "pii_borderline_llm_rejected"


class TestToxicityFilter:
    def test_clean_passes(self):
        f = make_toxicity_filter({"severe_toxicity": 0.01, "threat": 0.01})
        result = f(make_record("I feel a bit down today."))
        assert result.passed is True

    def test_severe_toxicity_dropped(self):
        f = make_toxicity_filter({"severe_toxicity": 0.50, "threat": 0.01})
        result = f(make_record("You should die."))
        assert result.passed is False
        assert "severe_toxicity" in result.reason

    def test_threat_dropped(self):
        f = make_toxicity_filter({"severe_toxicity": 0.01, "threat": 0.20})
        result = f(make_record("I will hurt you."))
        assert result.passed is False
        assert "threat" in result.reason

    def test_models_unavailable_passes(self):
        mock_model = MagicMock()
        mock_model.predict.side_effect = Exception("model error")
        models = dict.fromkeys(ToxicityFilter.MODEL_NAMES, mock_model)
        f = ToxicityFilter(models=models)
        result = f(make_record("Some text."))
        assert result.passed is True
        assert result.reason == "toxicity_models_unavailable"


class TestDedupFilter:
    def test_first_occurrence_passes(self, tmp_path):
        f = DedupFilter(dedup_store_path=str(tmp_path / "test_dedup.db"))
        result = f(make_record("Unique content here."))
        assert result.passed is True
        assert result.metadata["dedup"] == "unique"
        f.close()

    def test_exact_duplicate_dropped(self, tmp_path):
        f = DedupFilter(dedup_store_path=str(tmp_path / "test_dedup.db"))
        record = make_record("Duplicate content here.")
        f(record)
        result = f(record)
        assert result.passed is False
        assert result.reason == "exact_duplicate"
        f.close()

    def test_near_duplicate_dropped(self, tmp_path):
        f = DedupFilter(dedup_store_path=str(tmp_path / "test_dedup.db"))

        text1 = "the quick brown fox jumps over the lazy dog near the river bank today and yesterday we went swimming"
        text2 = "the quick brown fox jumps over the lazy dog near the river bank today and yesterday we went fishing"
        record1 = make_record(text1)
        record2 = make_record(text2)

        f(record1)
        result = f(record2)
        assert result.passed is False
        assert "near_duplicate" in result.reason
        f.close()

    def test_different_records_pass(self, tmp_path):
        f = DedupFilter(dedup_store_path=str(tmp_path / "test_dedup.db"))
        f(make_record("Completely different content one."))
        result = f(make_record("Totally unrelated other content."))
        assert result.passed is True
        f.close()


class TestStage1FilterChain:
    def test_chain_runs_filters_in_order(self):
        call_order: list[str] = []

        def make_tracked_filter(name: str):
            def filter_fn(record: dict) -> FilterResult:
                call_order.append(name)
                return FilterResult(passed=True, reason="", metadata={})

            return filter_fn

        chain = Stage1FilterChain(
            language_filter=make_tracked_filter("language"),
            pii_filter=make_tracked_filter("pii"),
            toxicity_filter=make_tracked_filter("toxicity"),
            dedup_filter=make_tracked_filter("dedup"),
        )

        records = [make_record("test")]
        list(chain.filter(records))

        assert call_order == ["language", "pii", "toxicity", "dedup"]

    def test_chain_stops_at_first_drop(self):
        pii_filter = MagicMock(return_value=FilterResult(passed=False, reason="pii_confirmed", metadata={}))
        toxicity_filter = MagicMock(return_value=FilterResult(passed=True, reason="", metadata={}))
        chain = Stage1FilterChain(
            language_filter=MagicMock(return_value=FilterResult(passed=True, reason="", metadata={})),
            pii_filter=pii_filter,
            toxicity_filter=toxicity_filter,
            dedup_filter=MagicMock(return_value=FilterResult(passed=True, reason="", metadata={})),
        )

        records = [make_record("test")]
        results = list(chain.filter(records))

        assert len(results) == 0
        toxicity_filter.assert_not_called()

    def test_chain_passes_clean_records(self, tmp_path):
        chain = Stage1FilterChain(
            language_filter=make_lang_filter("en"),
            pii_filter=make_pii_filter(results=[]),
            toxicity_filter=make_toxicity_filter({"severe_toxicity": 0.01, "threat": 0.01}),
            dedup_filter=DedupFilter(dedup_store_path=str(tmp_path / "chain_dedup.db")),
        )

        records = [make_record("I feel sad today."), make_record("I need help with anxiety.")]
        results = list(chain.filter(records))

        assert len(results) == 2
        chain.close()

    def test_chain_summary_logged(self, tmp_path, caplog):
        chain = Stage1FilterChain(
            language_filter=make_lang_filter("en"),
            pii_filter=make_pii_filter(results=[]),
            toxicity_filter=make_toxicity_filter({"severe_toxicity": 0.01, "threat": 0.01}),
            dedup_filter=DedupFilter(dedup_store_path=str(tmp_path / "summary_dedup.db")),
        )

        records = [make_record("I feel sad.")]
        list(chain.filter(records))

        summary = chain.summary()
        assert "Stage1 Filter Chain Summary" in summary
        assert "Pass-through" in summary
        chain.close()

    def test_pass_through_rate(self, tmp_path):
        lang_filter = MagicMock(
            side_effect=[
                FilterResult(passed=True, reason="", metadata={}),
                FilterResult(passed=False, reason="non_english:zh", metadata={}),
            ]
        )
        chain = Stage1FilterChain(
            language_filter=lang_filter,
            pii_filter=make_pii_filter(results=[]),
            toxicity_filter=make_toxicity_filter(),
            dedup_filter=DedupFilter(dedup_store_path=str(tmp_path / "rate_dedup.db")),
        )

        records = [make_record("Good English."), make_record("Non-English.")]
        results = list(chain.filter(records))

        assert len(results) == 1
        assert chain.stats.total_input == 2
        assert chain.stats.total_output == 1
        assert chain.stats.pass_through_rate == 50.0
        chain.close()

    def test_stats_dict_structure(self, tmp_path):
        chain = Stage1FilterChain(
            language_filter=make_lang_filter("en"),
            pii_filter=make_pii_filter(results=[]),
            toxicity_filter=make_toxicity_filter(),
            dedup_filter=DedupFilter(dedup_store_path=str(tmp_path / "stats_dedup.db")),
        )

        records = [make_record("test")]
        list(chain.filter(records))

        stats = chain.stats_dict()
        assert "total_input" in stats
        assert "total_output" in stats
        assert "pass_through_rate" in stats
        assert "filters" in stats
        assert "language" in stats["filters"]
        assert "pii" in stats["filters"]
        assert "toxicity" in stats["filters"]
        assert "dedup" in stats["filters"]
        chain.close()


class TestChainStats:
    def test_pass_through_rate_calculation(self):
        stats = ChainStats(total_input=100, total_output=30)
        assert stats.pass_through_rate == 30.0

    def test_zero_input_safe(self):
        stats = ChainStats()
        assert stats.pass_through_rate == 0.0

    def test_to_dict_structure(self):
        stats = ChainStats(total_input=10, total_output=5)
        d = stats.to_dict()
        assert d["total_input"] == 10
        assert d["total_output"] == 5
        assert "pass_through_rate" in d
        assert "filters" in d
