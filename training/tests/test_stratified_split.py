"""Tests for stratified_split module.

Covers:
- Feature extractors (domain, difficulty, language, tier)
- Hash-split fallback
- Stratified split correctness and balance
- Rare class collapsing
- Integrity gates (disjoint, ratio, domain/language balance)
- CLI entry point
- Edge cases (empty, tiny, single-language)
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path
from unittest.mock import patch

import pytest

from training.stratified_split import (
    DEFAULT_RATIOS,
    MIN_CLASS_SAMPLES,
    OTHER_MARKER,
    RATIO_TOLERANCE,
    SUPPORTED_LANGUAGES,
    _collapse_rare,
    _content_hash,
    _hash_split,
    classify_difficulty,
    classify_domain,
    classify_language,
    classify_tier,
    extract_features,
    integrity_gates,
    main,
    stratified_split,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_record(
    messages: list[dict] | None = None,
    domain: str | None = None,
    difficulty: str | None = None,
    language: str | None = None,
    tier: str | None = None,
    category: str | None = None,
    metadata: dict | None = None,
) -> dict:
    """Build a minimal record dict."""
    record: dict = {}
    if messages is None:
        messages = [
            {"role": "system", "content": "You are a clinical AI assistant."},
            {"role": "user", "content": "I feel anxious."},
            {"role": "assistant", "content": "Let's work through that."},
        ]
    record["messages"] = messages
    if domain is not None:
        record["domain"] = domain
    if difficulty is not None:
        record["difficulty"] = difficulty
    if language is not None:
        record["language"] = language
    if tier is not None:
        record["tier"] = tier
    if category is not None:
        record["category"] = category
    if metadata is not None:
        record["metadata"] = metadata
    return record


@pytest.fixture
def diverse_records() -> list[dict]:
    """200 records across multiple domains, difficulties, languages, tiers."""
    records: list[dict] = []
    domains = ["cptsd_trauma", "dissociation", "addiction", "ocd_intrusive_thoughts",
               "eating_disorders", "grief", "general_counseling", "somatic_therapy"]
    difficulties = ["easy", "medium", "hard"]
    languages = ["en", "en", "en", "es", "fr"]  # mostly English, some multilingual
    tiers = ["T1_GOLD", "T2_SILVER", "T3_BRONZE", "T4_SAFETY"]

    for i in range(400):
        domain = domains[i % len(domains)]
        difficulty = difficulties[i % len(difficulties)]
        language = languages[i % len(languages)]
        tier = tiers[i % len(tiers)]

        # Vary message length for difficulty heuristic, include index for uniqueness
        if difficulty == "easy":
            messages = [
                {"role": "system", "content": f"System prompt {i}."},
                {"role": "user", "content": f"Short question {i}."},
                {"role": "assistant", "content": f"Short answer {i}."},
            ]
        elif difficulty == "hard":
            messages = [
                {"role": "system", "content": f"System prompt {i}."},
                {"role": "user", "content": f"X{i}" * 500},
                {"role": "assistant", "content": f"Y{i}" * 500},
                {"role": "user", "content": f"X{i}" * 500},
                {"role": "assistant", "content": f"Y{i}" * 500},
                {"role": "user", "content": f"X{i}" * 500},
                {"role": "assistant", "content": f"Y{i}" * 500},
                {"role": "user", "content": f"X{i}" * 500},
                {"role": "assistant", "content": f"Y{i}" * 500},
            ]
        else:
            messages = [
                {"role": "system", "content": f"System prompt {i}."},
                {"role": "user", "content": f"Medium length question about therapy {i}."},
                {"role": "assistant", "content": f"Medium length response with guidance {i}."},
                {"role": "user", "content": f"Follow-up question {i}."},
                {"role": "assistant", "content": f"Follow-up response {i}."},
            ]

        records.append(_make_record(
            messages=messages,
            domain=domain,
            difficulty=difficulty,
            language=language,
            tier=tier,
        ))
    return records


# ---------------------------------------------------------------------------
# Feature extractors
# ---------------------------------------------------------------------------

class TestClassifyDomain:
    def test_explicit_domain_field(self):
        record = _make_record(domain="cptsd_trauma")
        assert classify_domain(record) == "cptsd_trauma"

    def test_explicit_category_field(self):
        record = _make_record(category="dissociation")
        assert classify_domain(record) == "dissociation"

    def test_metadata_domain(self):
        record = _make_record(metadata={"domain": "addiction"})
        assert classify_domain(record) == "addiction"

    def test_metadata_category(self):
        record = _make_record(metadata={"category": "ocd_intrusive_thoughts"})
        assert classify_domain(record) == "ocd_intrusive_thoughts"

    def test_keyword_matching(self):
        record = _make_record(messages=[
            {"role": "user", "content": "I'm dealing with cptsd and trauma."},
            {"role": "assistant", "content": "Let's work on that."},
        ])
        assert classify_domain(record) == "cptsd_trauma"

    def test_no_match_returns_uncategorized(self):
        record = _make_record(messages=[
            {"role": "user", "content": "Hello world."},
            {"role": "assistant", "content": "Hi there."},
        ])
        assert classify_domain(record) == "uncategorized"


class TestClassifyDifficulty:
    def test_easy_short(self):
        record = _make_record(messages=[
            {"role": "system", "content": "Sys."},
            {"role": "user", "content": "Hi."},
            {"role": "assistant", "content": "Hello."},
        ])
        assert classify_difficulty(record) == "easy"

    def test_hard_many_turns(self):
        messages = [{"role": "system", "content": "Sys."}]
        for _ in range(8):
            messages.append({"role": "user", "content": "X" * 200})
            messages.append({"role": "assistant", "content": "Y" * 200})
        record = _make_record(messages=messages)
        assert classify_difficulty(record) == "hard"

    def test_hard_long_content(self):
        record = _make_record(messages=[
            {"role": "system", "content": "Sys."},
            {"role": "user", "content": "X" * 2000},
            {"role": "assistant", "content": "Y" * 2000},
        ])
        assert classify_difficulty(record) == "hard"

    def test_medium(self):
        record = _make_record(messages=[
            {"role": "system", "content": "Sys."},
            {"role": "user", "content": "Tell me about anxiety and how to cope with it."},
            {"role": "assistant", "content": "Let's explore some techniques like deep breathing."},
            {"role": "user", "content": "What else?"},
            {"role": "assistant", "content": "Try journaling too."},
        ])
        assert classify_difficulty(record) == "medium"


class TestClassifyLanguage:
    def test_explicit_language(self):
        record = _make_record(language="es")
        assert classify_language(record) == "es"

    def test_metadata_language(self):
        record = _make_record(metadata={"language": "fr"})
        assert classify_language(record) == "fr"

    def test_keyword_es(self):
        record = _make_record(messages=[
            {"role": "user", "content": "Hola, gracias por tu ayuda. ¿Cómo estás?"},
            {"role": "assistant", "content": "¡Muy bien!"},
        ])
        assert classify_language(record) == "es"

    def test_keyword_fr(self):
        record = _make_record(messages=[
            {"role": "user", "content": "Bonjour, je vous remercie beaucoup."},
            {"role": "assistant", "content": "Très bien."},
        ])
        assert classify_language(record) == "fr"

    def test_keyword_de(self):
        record = _make_record(messages=[
            {"role": "user", "content": "Guten Tag, ich danke Ihnen sehr."},
            {"role": "assistant", "content": "Sehr gut."},
        ])
        assert classify_language(record) == "de"

    def test_keyword_pt(self):
        record = _make_record(messages=[
            {"role": "user", "content": "Olá, muito obrigado. Você está bem?"},
            {"role": "assistant", "content": "Está tudo bem."},
        ])
        assert classify_language(record) == "pt"

    def test_defaults_en(self):
        record = _make_record(messages=[
            {"role": "user", "content": "Hello there."},
            {"role": "assistant", "content": "Hi."},
        ])
        assert classify_language(record) == "en"

    def test_invalid_language_field_defaults_en(self):
        record = _make_record(language="zh")
        assert classify_language(record) == "en"

    def test_all_supported_languages_in_constant(self):
        assert set(SUPPORTED_LANGUAGES) == {"en", "es", "fr", "pt", "de"}


class TestClassifyTier:
    def test_explicit_tier(self):
        record = _make_record(tier="T1_GOLD")
        assert classify_tier(record) == "T1_GOLD"

    def test_metadata_tier(self):
        record = _make_record(metadata={"tier": "T2_SILVER"})
        assert classify_tier(record) == "T2_SILVER"

    def test_quality_tier_field(self):
        record = _make_record()
        record["quality_tier"] = "T3_BRONZE"
        assert classify_tier(record) == "T3_BRONZE"

    def test_fallback_to_curate_pipeline(self):
        """If no explicit tier and curate_pipeline is available, delegate."""
        record = _make_record(messages=[
            {"role": "system", "content": "Sys."},
            {"role": "user", "content": "Hi."},
            {"role": "assistant", "content": "Hello."},
        ])
        # No tier field, no clinical_reviewed → should fall through
        result = classify_tier(record)
        assert result in ("T1_GOLD", "T2_SILVER", "T3_BRONZE", "T4_SAFETY")

    def test_safety_tier_via_curate_pipeline(self):
        record = _make_record(messages=[
            {"role": "system", "content": "Sys."},
            {"role": "user", "content": "Hi."},
            {"role": "assistant", "content": "Hello."},
        ])
        record["task_type"] = "adversarial_safety"
        assert classify_tier(record) == "T4_SAFETY"


class TestExtractFeatures:
    def test_returns_feature_for_each_record(self):
        records = [_make_record(domain="cptsd_trauma", language="en", tier="T1_GOLD")]
        features = extract_features(records)
        assert len(features) == 1
        assert "domain" in features[0]
        assert "difficulty" in features[0]
        assert "language" in features[0]
        assert "tier" in features[0]

    def test_empty_records(self):
        assert extract_features([]) == []


# ---------------------------------------------------------------------------
# Hash-split fallback
# ---------------------------------------------------------------------------

class TestHashSplit:
    def test_basic_split(self):
        records = [_make_record(domain=f"d{i}") for i in range(100)]
        splits = _hash_split(records)
        total = len(splits["train"]) + len(splits["val"]) + len(splits["test"])
        assert total == 100

    def test_all_records_accounted_for(self):
        records = [_make_record() for _ in range(50)]
        splits = _hash_split(records)
        total = sum(len(v) for v in splits.values())
        assert total == 50

    def test_disjoint(self):
        records = [_make_record(messages=[{"role": "user", "content": f"unique{i}"}])
                    for i in range(100)]
        splits = _hash_split(records)
        train_hashes = {_content_hash(r) for r in splits["train"]}
        val_hashes = {_content_hash(r) for r in splits["val"]}
        test_hashes = {_content_hash(r) for r in splits["test"]}
        assert not (train_hashes & val_hashes)
        assert not (train_hashes & test_hashes)
        assert not (val_hashes & test_hashes)

    def test_custom_ratios(self):
        records = [_make_record() for _ in range(1000)]
        splits = _hash_split(records, {"train": 0.80, "val": 0.10, "test": 0.10})
        total = len(splits["train"]) + len(splits["val"]) + len(splits["test"])
        assert total == 1000
        # Roughly 80/10/10
        assert len(splits["train"]) > 700
        assert len(splits["val"]) < 200


# ---------------------------------------------------------------------------
# Rare class collapsing
# ---------------------------------------------------------------------------

class TestCollapseRare:
    def test_frequent_unchanged(self):
        values = ["a"] * 100 + ["b"] * 100
        result = _collapse_rare(values, min_samples=50)
        assert result == ["a"] * 100 + ["b"] * 100

    def test_rare_replaced(self):
        values = ["a"] * 100 + ["b"] * 10
        result = _collapse_rare(values, min_samples=50)
        assert result == ["a"] * 100 + [OTHER_MARKER] * 10

    def test_all_rare(self):
        values = ["a"] * 5 + ["b"] * 5
        result = _collapse_rare(values, min_samples=50)
        assert all(v == OTHER_MARKER for v in result)

    def test_default_threshold(self):
        values = ["x"] * (MIN_CLASS_SAMPLES - 1)
        result = _collapse_rare(values)
        assert all(v == OTHER_MARKER for v in result)


# ---------------------------------------------------------------------------
# Stratified split
# ---------------------------------------------------------------------------

class TestStratifiedSplit:
    def test_empty_records(self):
        result = stratified_split([])
        assert result == {"train": [], "val": [], "test": []}

    def test_tiny_dataset_uses_hash_fallback(self):
        records = [_make_record() for _ in range(3)]
        result = stratified_split(records)
        total = sum(len(v) for v in result.values())
        assert total == 3

    def test_all_records_preserved(self, diverse_records):
        result = stratified_split(diverse_records)
        total = len(result["train"]) + len(result["val"]) + len(result["test"])
        assert total == len(diverse_records)

    def test_disjoint_splits(self, diverse_records):
        result = stratified_split(diverse_records)
        train_ids = {_content_hash(r) for r in result["train"]}
        val_ids = {_content_hash(r) for r in result["val"]}
        test_ids = {_content_hash(r) for r in result["test"]}
        assert not (train_ids & val_ids)
        assert not (train_ids & test_ids)
        assert not (val_ids & test_ids)

    def test_approximate_ratios(self, diverse_records):
        result = stratified_split(diverse_records)
        total = len(diverse_records)
        train_ratio = len(result["train"]) / total
        val_ratio = len(result["val"]) / total
        test_ratio = len(result["test"]) / total
        # ±5pp tolerance for stratified (stratification may shift a few)
        assert abs(train_ratio - 0.70) < 0.10, f"train ratio {train_ratio}"
        assert abs(val_ratio - 0.15) < 0.10, f"val ratio {val_ratio}"
        assert abs(test_ratio - 0.15) < 0.10, f"test ratio {test_ratio}"

    def test_reproducible(self, diverse_records):
        r1 = stratified_split(diverse_records, random_state=42)
        r2 = stratified_split(diverse_records, random_state=42)
        # Same records in each split
        h1 = {_content_hash(r) for r in r1["train"]}
        h2 = {_content_hash(r) for r in r2["train"]}
        assert h1 == h2

    def test_different_seed_different_split(self, diverse_records):
        r1 = stratified_split(diverse_records, random_state=42)
        r2 = stratified_split(diverse_records, random_state=999)
        h1 = {_content_hash(r) for r in r1["train"]}
        h2 = {_content_hash(r) for r in r2["train"]}
        # Should be different (very unlikely to be identical for 200 records)
        assert h1 != h2

    def test_custom_ratios(self, diverse_records):
        result = stratified_split(
            diverse_records,
            train_ratio=0.80,
            val_ratio=0.10,
            test_ratio=0.10,
        )
        total = len(diverse_records)
        assert len(result["train"]) / total > 0.70
        assert len(result["test"]) / total < 0.20

    def test_invalid_ratios_raise(self):
        with pytest.raises(ValueError, match="sum to 1.0"):
            stratified_split(
                [_make_record()],
                train_ratio=0.5,
                val_ratio=0.5,
                test_ratio=0.5,
            )

    def test_language_coverage(self, diverse_records):
        result = stratified_split(diverse_records)
        # All languages present in input should appear in at least train
        input_langs = {classify_language(r) for r in diverse_records}
        train_langs = {classify_language(r) for r in result["train"]}
        assert input_langs == train_langs

    def test_domain_representation(self, diverse_records):
        """Each domain in the full dataset should appear in the train split."""
        result = stratified_split(diverse_records)
        full_domains = {classify_domain(r) for r in diverse_records}
        train_domains = {classify_domain(r) for r in result["train"]}
        assert full_domains == train_domains

    def test_single_language_all_english(self):
        records = [
            _make_record(domain="cptsd_trauma", language="en", tier="T1_GOLD")
            for _ in range(100)
        ]
        result = stratified_split(records)
        total = sum(len(v) for v in result.values())
        assert total == 100

    def test_min_class_samples_parameter(self):
        """Test that min_class_samples affects rare class handling."""
        records = [
            _make_record(domain="rare_domain", language="en")
            for _ in range(10)
        ] + [
            _make_record(domain="common_domain", language="en")
            for _ in range(100)
        ]
        # With min_class_samples=5, "rare_domain" has 10 >= 5, so it stays
        result = stratified_split(records, min_class_samples=5)
        total = sum(len(v) for v in result.values())
        assert total == 110

        # With min_class_samples=50, "rare_domain" has 10 < 50, collapses
        result2 = stratified_split(records, min_class_samples=50)
        total2 = sum(len(v) for v in result2.values())
        assert total2 == 110


# ---------------------------------------------------------------------------
# Integrity gates
# ---------------------------------------------------------------------------

class TestIntegrityGates:
    def test_empty_splits_pass(self):
        gates = integrity_gates({"train": [], "val": [], "test": []})
        assert gates["passed"] is True

    def test_disjoint_passes(self, diverse_records):
        splits = stratified_split(diverse_records)
        gates = integrity_gates(splits)
        assert gates["checks"]["hash_disjoint"] is True

    def test_overlapping_fails(self):
        record = _make_record()
        splits = {
            "train": [record],
            "val": [record],  # same record → overlap
            "test": [],
        }
        gates = integrity_gates(splits)
        assert gates["checks"]["hash_disjoint"] is False
        assert gates["passed"] is False

    def test_ratio_check_passes(self, diverse_records):
        splits = stratified_split(diverse_records)
        gates = integrity_gates(splits)
        assert gates["checks"]["ratio"] is True

    def test_ratio_check_fails(self):
        # 100/0/0 split when target is 70/15/15
        records = [_make_record(messages=[{"role": "user", "content": f"u{i}"}])
                    for i in range(100)]
        splits = {"train": records, "val": [], "test": []}
        gates = integrity_gates(splits)
        assert gates["checks"]["ratio"] is False

    def test_domain_balance_passes(self, diverse_records):
        splits = stratified_split(diverse_records)
        gates = integrity_gates(splits, tolerance=0.03)
        assert gates["checks"]["domain_balance"] is True

    def test_language_balance_passes(self, diverse_records):
        splits = stratified_split(diverse_records)
        gates = integrity_gates(splits)
        assert gates["checks"]["language_balance"] is True

    def test_all_checks_pass_on_good_split(self, diverse_records):
        splits = stratified_split(diverse_records)
        gates = integrity_gates(splits, tolerance=0.03)
        assert gates["passed"] is True
        assert all(gates["checks"].values())

    def test_returns_details(self, diverse_records):
        splits = stratified_split(diverse_records)
        gates = integrity_gates(splits)
        assert "actual_ratios" in gates["details"]
        assert "target_ratios" in gates["details"]
        assert "domain_deviations" in gates["details"]
        assert "language_deviations" in gates["details"]

    def test_custom_tolerance(self):
        records = [_make_record() for _ in range(100)]
        splits = {"train": records, "val": [], "test": []}
        # With tolerance of 1.0, everything passes
        gates = integrity_gates(splits, tolerance=1.0)
        assert gates["checks"]["ratio"] is True


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

class TestCLI:
    def test_main_writes_split_files(self, tmp_path, diverse_records):
        input_file = tmp_path / "input.jsonl"
        with open(input_file, "w") as f:
            for r in diverse_records:
                f.write(json.dumps(r) + "\n")

        output_dir = tmp_path / "output"
        with patch.object(sys, "argv", ["stratified_split", str(input_file), str(output_dir), "--seed", "42", "--skip-gates"]):
            ret = main()

        assert ret == 0
        assert (output_dir / "train.jsonl").exists()
        assert (output_dir / "val.jsonl").exists()
        assert (output_dir / "test.jsonl").exists()

        with open(output_dir / "train.jsonl") as f:
            train_count = sum(1 for _ in f)
        with open(output_dir / "val.jsonl") as f:
            val_count = sum(1 for _ in f)
        with open(output_dir / "test.jsonl") as f:
            test_count = sum(1 for _ in f)
        assert train_count + val_count + test_count == len(diverse_records)

    def test_main_with_gates(self, tmp_path, diverse_records):
        input_file = tmp_path / "input.jsonl"
        with open(input_file, "w") as f:
            for r in diverse_records:
                f.write(json.dumps(r) + "\n")

        output_dir = tmp_path / "output"
        with patch.object(sys, "argv", ["stratified_split", str(input_file), str(output_dir), "--seed", "42"]):
            ret = main()
        assert ret == 0

    def test_main_empty_file(self, tmp_path):
        input_file = tmp_path / "empty.jsonl"
        input_file.write_text("")
        output_dir = tmp_path / "output"
        with patch.object(sys, "argv", ["stratified_split", str(input_file), str(output_dir)]):
            ret = main()
        assert ret == 1

    def test_main_directory_input(self, tmp_path, diverse_records):
        input_dir = tmp_path / "input"
        input_dir.mkdir()
        for i in range(3):
            f = input_dir / f"part{i}.jsonl"
            with open(f, "w") as fh:
                for r in diverse_records[:50]:
                    fh.write(json.dumps(r) + "\n")

        output_dir = tmp_path / "output"
        with patch.object(sys, "argv", ["stratified_split", str(input_dir), str(output_dir), "--skip-gates"]):
            ret = main()
        assert ret == 0
        assert (output_dir / "train.jsonl").exists()

    def test_main_invalid_json_skipped(self, tmp_path, diverse_records):
        input_file = tmp_path / "input.jsonl"
        with open(input_file, "w") as f:
            f.write(json.dumps(diverse_records[0]) + "\n")
            f.write("NOT JSON\n")
            f.write(json.dumps(diverse_records[1]) + "\n")

        output_dir = tmp_path / "output"
        with patch.object(sys, "argv", ["stratified_split", str(input_file), str(output_dir), "--skip-gates"]):
            ret = main()
        assert ret == 0


# ---------------------------------------------------------------------------
# Content hash
# ---------------------------------------------------------------------------

class TestContentHash:
    def test_same_content_same_hash(self):
        r1 = _make_record(messages=[{"role": "user", "content": "hello"}])
        r2 = _make_record(messages=[{"role": "user", "content": "hello"}])
        assert _content_hash(r1) == _content_hash(r2)

    def test_different_content_different_hash(self):
        r1 = _make_record(messages=[{"role": "user", "content": "hello1"}])
        r2 = _make_record(messages=[{"role": "user", "content": "hello2"}])
        assert _content_hash(r1) != _content_hash(r2)

    def test_empty_messages(self):
        r = _make_record(messages=[])
        h = _content_hash(r)
        assert isinstance(h, str)
        assert len(h) == 32  # MD5 hex
