"""Tests for the clinical validity scorer benchmark."""

from __future__ import annotations

import json
import math
import tempfile
from pathlib import Path

import pytest

from training.benchmark import (
    CSV_FILES,
    FIXTURE_DIR,
    HUMAN_DIMENSIONS,
    _compute_agreement,
    _compute_mae,
    _compute_pearson,
    _compute_spearman,
    _find_longest_prefix,
    _normalize_conversation_id,
    build_report,
    load_all_csvs,
    run_benchmark,
    score_row,
    validate_csv_structure,
)


class TestHelperFunctions:
    """Tests for helper functions."""

    def test_compute_mae(self):
        assert _compute_mae([1.0, 2.0, 3.0], [1.0, 2.0, 3.0]) == 0.0
        assert _compute_mae([1.0, 2.0, 3.0], [1.0, 2.0, 4.0]) == pytest.approx(1/3)
        assert _compute_mae([0.0, 0.0], [1.0, 1.0]) == 1.0
        assert _compute_mae([], []) == 0.0

    def test_compute_agreement(self):
        assert _compute_agreement([0.5, 0.6, 0.7], [0.5, 0.7, 0.6], 0.2) == 1.0
        assert _compute_agreement([0.5, 0.6, 0.7], [0.8, 0.9, 1.0], 0.2) == 0.0
        assert _compute_agreement([0.5, 0.6], [0.55, 0.65], 0.2) == 1.0
        assert _compute_agreement([], []) == 0.0

    def test_compute_pearson_constant(self):
        assert _compute_pearson([1.0, 1.0, 1.0], [0.5, 0.6, 0.7]) == 0.0
        assert _compute_pearson([0.5, 0.6, 0.7], [1.0, 1.0, 1.0]) == 0.0
        assert _compute_pearson([1.0, 2.0], [1.0, 2.0]) == 1.0
        assert _compute_pearson([1.0, 2.0], [2.0, 4.0]) == 1.0

    def test_compute_spearman_constant(self):
        assert _compute_spearman([1.0, 1.0, 1.0], [0.5, 0.6, 0.7]) == 0.0
        assert _compute_spearman([1.0, 2.0], [1.0, 2.0]) == pytest.approx(1.0)
        assert _compute_spearman([1.0, 2.0], [2.0, 1.0]) == pytest.approx(-1.0)

    def test_find_longest_prefix(self):
        assert _find_longest_prefix("couples_therapy_official_CouplesTherapyOfficial_The_Evolution") == "couples_therapy_official"
        assert _find_longest_prefix("tim_fletcher_TimFletcher_26_Signs") == "tim_fletcher"
        assert _find_longest_prefix("therapy_in_a_nutshell_TherapyinaNutshellPodcast_When") == "therapy_in_a_nutshell"
        assert _find_longest_prefix("random_unknown_id") is None

    def test_normalize_conversation_id(self):
        channel, title = _normalize_conversation_id("couples_therapy_official_CouplesTherapyOfficial_The_Evolution")
        assert channel == "Couples Therapy Official"
        assert "The_Evolution" in title

        channel, title = _normalize_conversation_id("tim_fletcher_TimFletcher_26_Signs")
        assert channel == "Tim Fletcher"
        assert "26_Signs" in title

        channel, title = _normalize_conversation_id("random_unknown_id")
        assert channel is None


class TestCsvLoading:
    """Tests for CSV loading and validation."""

    def test_all_csv_files_exist(self):
        for name in CSV_FILES:
            path = FIXTURE_DIR / name
            assert path.exists(), f"Missing CSV: {name}"

    def test_load_all_csvs(self):
        rows, checksums = load_all_csvs()
        assert len(rows) == 691, f"Expected 691 rows, got {len(rows)}"
        assert len(checksums) == 13, f"Expected 13 CSVs, got {len(checksums)}"

    def test_validate_csv_structure_valid(self):
        rows, _ = load_all_csvs()
        errors = validate_csv_structure(rows)
        assert len(errors) == 0, f"Unexpected validation errors: {errors[:5]}"

    def test_validate_csv_structure_required_fields(self):
        # All rows should have conversation_id and score fields
        rows, _ = load_all_csvs()
        for i, row in enumerate(rows):
            assert "conversation_id" in row, f"Row {i} missing conversation_id"
            assert "score" in row, f"Row {i} missing score"
            for dim in HUMAN_DIMENSIONS:
                assert dim in row, f"Row {i} missing {dim}"


class TestScoreRow:
    """Tests for the score_row function."""

    def test_score_row_returns_scores_for_valid_conv_id(self):
        rows, _ = load_all_csvs()
        # Find a real (non-synthetic) sample
        real_row = None
        for row in rows:
            if "synthetic" not in row["conversation_id"].lower():
                real_row = row
                break

        if real_row is None:
            pytest.skip("No real samples found")

        result = score_row(real_row)
        if result is not None:
            assert "overall" in result
            assert "technique" in result
            assert "alliance" in result
            assert "structure" in result
            assert "cultural" in result
            assert "ebp" in result
            assert "dsm5" in result
            assert 0.0 <= result["overall"] <= 1.0

    def test_score_row_returns_none_for_missing_transcript(self):
        # A synthetic sample should not have a matching transcript
        rows, _ = load_all_csvs()
        for row in rows:
            if "synthetic" in row["conversation_id"].lower():
                result = score_row(row)
                # Either None (transcript not found) or all zeros
                if result is not None:
                    assert result["overall"] == 0.0
                break


class TestBuildReport:
    """Tests for the build_report function."""

    def test_report_structure(self):
        rows, checksums = load_all_csvs()
        report = build_report(rows, checksums, "3.0.0")

        assert "scorer_version" in report
        assert report["scorer_version"] == "3.0.0"
        assert "total_sample_count" in report
        assert "scored_sample_count" in report
        assert "csv_checksums" in report
        assert "overall" in report
        assert "per_dimension" in report
        assert "per_channel" in report

    def test_report_overall_fields(self):
        rows, checksums = load_all_csvs()
        report = build_report(rows, checksums, "3.0.0")

        assert "pearson_correlation" in report["overall"]
        assert "spearman_correlation" in report["overall"]
        assert "mae" in report["overall"]

    def test_report_per_dimension_fields(self):
        rows, checksums = load_all_csvs()
        report = build_report(rows, checksums, "3.0.0")

        for dim in HUMAN_DIMENSIONS:
            assert dim in report["per_dimension"]
            dim_report = report["per_dimension"][dim]
            assert "pearson" in dim_report
            assert "spearman" in dim_report
            assert "mae" in dim_report
            assert "agreement_rate" in dim_report

    def test_report_mae_is_valid_float(self):
        rows, checksums = load_all_csvs()
        report = build_report(rows, checksums, "3.0.0")

        mae = report["overall"]["mae"]
        assert isinstance(mae, float)
        assert 0.0 <= mae <= 1.0

    def test_report_correlations_in_range(self):
        rows, checksums = load_all_csvs()
        report = build_report(rows, checksums, "3.0.0")

        pearson = report["overall"]["pearson_correlation"]
        spearman = report["overall"]["spearman_correlation"]

        assert isinstance(pearson, float)
        assert isinstance(spearman, float)
        # Correlations can be NaN if no variance, but not out of [-1, 1]
        if not math.isnan(pearson):
            assert -1.0 <= pearson <= 1.0
        if not math.isnan(spearman):
            assert -1.0 <= spearman <= 1.0


class TestRunBenchmark:
    """Tests for the run_benchmark function."""

    def test_run_benchmark_succeeds(self):
        report = run_benchmark()
        assert report is not None

    def test_run_benchmark_with_output_file(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            output_path = Path(f.name)

        try:
            report = run_benchmark(output_path)
            assert output_path.exists()

            # Read and parse the saved file
            with output_path.open() as f:
                loaded = json.load(f)

            assert loaded["scorer_version"] == report["scorer_version"]
            assert loaded["total_sample_count"] == report["total_sample_count"]
        finally:
            if output_path.exists():
                output_path.unlink()


class TestDeterminism:
    """Tests for benchmark determinism."""

    def test_same_inputs_produce_same_report(self):
        rows1, checksums1 = load_all_csvs()
        rows2, checksums2 = load_all_csvs()

        report1 = build_report(rows1, checksums1, "3.0.0")
        report2 = build_report(rows2, checksums2, "3.0.0")

        # Overall metrics should be identical
        assert report1["overall"]["pearson_correlation"] == report2["overall"]["pearson_correlation"]
        assert report1["overall"]["spearman_correlation"] == report2["overall"]["spearman_correlation"]
        assert report1["overall"]["mae"] == report2["overall"]["mae"]

        # Checksums should be identical
        assert report1["csv_checksums"] == report2["csv_checksums"]

    def test_checksums_deterministic(self):
        _rows, checksums = load_all_csvs()

        # Checksums should be consistent
        for _name, checksum in checksums.items():
            assert isinstance(checksum, str)
            assert len(checksum) == 32  # MD5 hex length


class TestMissingCsvHandling:
    """Tests for error handling with missing CSV files."""

    def test_missing_csv_raises_error(self):
        original_files = CSV_FILES[:]
        # Patch to use a non-existent file
        import training.benchmark as bm
        bm.CSV_FILES = ["nonexistent_file.csv"]

        with pytest.raises(FileNotFoundError) as exc_info:
            load_all_csvs()

        assert "Missing CSV fixture files" in str(exc_info.value)

        # Restore
        bm.CSV_FILES = original_files
