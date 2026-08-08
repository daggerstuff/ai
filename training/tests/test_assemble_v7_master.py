"""Tests for assemble_v7_master orchestrator (PIX-4232)."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pytest

from training.scripts.assemble_v7_master import (
    StageResult,
    _find_jsonl,
    build_parser,
    main,
    run_pipeline,
)


def _write_jsonl(path: Path, records: list[dict]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    return path


def _valid_record(instruction: str = "Hello", output: str = "Hi there") -> dict:
    return {"instruction": instruction, "output": output}


ONE_RECORD = 1
TWO_RECORDS = 2
ZERO_RESULTS = 0
ONE_STAGE = 1
TWO_STAGES = 2
THREE_STAGES = 3
FOUR_STAGES = 4
EXIT_OK = 0
EXIT_FAIL = 1
SMALL_THRESHOLD = 0.5


class TestStageResult:
    def test_defaults(self) -> None:
        r = StageResult("test", EXIT_OK, 1.0)
        assert r.name == "test"
        assert r.exit_code == EXIT_OK
        assert r.duration_s == 1.0
        assert r.output_lines == []

    def test_with_lines(self) -> None:
        r = StageResult("test", EXIT_FAIL, 0.5, ["line1", "line2"])
        assert r.output_lines == ["line1", "line2"]


class TestFindJsonl:
    def test_master_preferred(self, tmp_path: Path) -> None:
        _write_jsonl(tmp_path / "V7_MASTER.jsonl", [_valid_record()])
        _write_jsonl(tmp_path / "shard_0000.jsonl", [_valid_record()])
        result = _find_jsonl(tmp_path)
        assert result is not None
        assert result.name == "V7_MASTER.jsonl"

    def test_shard_fallback(self, tmp_path: Path) -> None:
        _write_jsonl(tmp_path / "shard_0000.jsonl", [_valid_record()])
        result = _find_jsonl(tmp_path)
        assert result is not None
        assert result.name == "shard_0000.jsonl"

    def test_any_jsonl_fallback(self, tmp_path: Path) -> None:
        _write_jsonl(tmp_path / "weird.jsonl", [_valid_record()])
        result = _find_jsonl(tmp_path)
        assert result is not None
        assert result.name == "weird.jsonl"

    def test_no_dir(self, tmp_path: Path) -> None:
        assert _find_jsonl(tmp_path / "nonexistent") is None

    def test_empty_dir(self, tmp_path: Path) -> None:
        (tmp_path / "subdir").mkdir()
        assert _find_jsonl(tmp_path) is None


class TestBuildParser:
    def test_required_input_dirs(self) -> None:
        parser = build_parser()
        with pytest.raises(SystemExit):
            parser.parse_args([])

    def test_defaults(self) -> None:
        parser = build_parser()
        args = parser.parse_args(["--input_dirs", "data/"])
        assert args.input_dirs == ["data/"]
        assert args.skip_dedup is False
        assert args.skip_integrity is False
        assert args.skip_s3 is False
        assert args.shard_size == 0
        assert args.jaccard_threshold == 0.92
        assert args.lsh_threshold == 0.85
        assert args.s3_bucket == "pixeldata"
        assert args.s3_prefix == "datasets/v7/"
        assert args.s3_region == "US-EAST-VA"
        assert args.s3_dry_run is False

    def test_skip_flags(self) -> None:
        parser = build_parser()
        args = parser.parse_args(
            [
                "--input_dirs",
                "data/",
                "--skip_dedup",
                "--skip_integrity",
                "--skip_s3",
            ]
        )
        assert args.skip_dedup is True
        assert args.skip_integrity is True
        assert args.skip_s3 is True


class TestRunPipelineIntegration:
    """Integration tests that run the real subprocess stages."""

    def test_skip_all_optional_stages(self, tmp_path: Path) -> None:
        input_dir = tmp_path / "input"
        _write_jsonl(
            input_dir / "fixture.jsonl",
            [
                _valid_record("I feel hopeless", "Call 988."),
                _valid_record("I feel hopeless", "Call 988."),
            ],
        )
        work_dir = tmp_path / "v7"
        results = run_pipeline(
            input_dirs=[str(input_dir)],
            work_dir=work_dir,
            skip_dedup=True,
            skip_integrity=True,
            skip_s3=True,
            jaccard_threshold=SMALL_THRESHOLD,
        )
        assert len(results) == ONE_STAGE
        assert all(r.exit_code == EXIT_OK for r in results)
        output = _find_jsonl(work_dir)
        assert output is not None

    def test_consolidate_dedup_integrity(self, tmp_path: Path) -> None:
        input_dir = tmp_path / "input"
        _write_jsonl(
            input_dir / "fixture.jsonl",
            [
                _valid_record("I feel hopeless", "Call 988."),
                _valid_record("I feel hopeless", "Call 988."),
                _valid_record("Explain CBT.", "Cognitive behavioral therapy helps reframe thoughts."),
                _valid_record("Explain CBT.", "Cognitive behavioral therapy reframes negative thoughts."),
            ],
        )
        work_dir = tmp_path / "v7"
        results = run_pipeline(
            input_dirs=[str(input_dir)],
            work_dir=work_dir,
            skip_s3=True,
            jaccard_threshold=SMALL_THRESHOLD,
            lsh_threshold=SMALL_THRESHOLD,
        )
        assert len(results) == THREE_STAGES
        assert all(r.exit_code == EXIT_OK for r in results)
        final = _find_jsonl(work_dir / "lsh")
        assert final is not None

    def test_missing_input_graceful(self, tmp_path: Path) -> None:
        work_dir = tmp_path / "v7"
        results = run_pipeline(
            input_dirs=[str(tmp_path / "nonexistent")],
            work_dir=work_dir,
            skip_dedup=True,
            skip_integrity=True,
            skip_s3=True,
        )
        assert len(results) == ONE_STAGE
        assert results[0].exit_code == EXIT_OK
        output = _find_jsonl(work_dir)
        assert output is not None
        assert output.stat().st_size == ZERO_RESULTS or sum(1 for _ in output.open()) == ZERO_RESULTS

    def test_s3_dry_run_stage(self, tmp_path: Path) -> None:
        input_dir = tmp_path / "input"
        _write_jsonl(input_dir / "fixture.jsonl", [_valid_record()])
        work_dir = tmp_path / "v7"
        results = run_pipeline(
            input_dirs=[str(input_dir)],
            work_dir=work_dir,
            skip_dedup=True,
            skip_integrity=True,
            skip_s3=False,
            s3_dry_run=True,
            s3_bucket="pixeldata",
            jaccard_threshold=SMALL_THRESHOLD,
        )
        assert len(results) == TWO_STAGES
        assert all(r.exit_code == EXIT_OK for r in results)


class TestMainCLI:
    def test_main_success_skip_all(self, tmp_path: Path) -> None:
        input_dir = tmp_path / "input"
        _write_jsonl(
            input_dir / "fixture.jsonl",
            [
                _valid_record("Hello", "Hi"),
                _valid_record("Hello", "Hi"),
            ],
        )
        rc = main(
            [
                "--input_dirs",
                str(input_dir),
                "--work_dir",
                str(tmp_path / "v7"),
                "--skip_dedup",
                "--skip_integrity",
                "--skip_s3",
                "--jaccard_threshold",
                str(SMALL_THRESHOLD),
            ]
        )
        assert rc == EXIT_OK

    def test_main_missing_input_graceful(self, tmp_path: Path) -> None:
        rc = main(
            [
                "--input_dirs",
                str(tmp_path / "nonexistent"),
                "--work_dir",
                str(tmp_path / "v7"),
                "--skip_dedup",
                "--skip_integrity",
                "--skip_s3",
            ]
        )
        assert rc == EXIT_OK

    def test_main_verbose_flag(self, tmp_path: Path) -> None:
        input_dir = tmp_path / "input"
        _write_jsonl(input_dir / "fixture.jsonl", [_valid_record()])
        rc = main(
            [
                "--input_dirs",
                str(input_dir),
                "--work_dir",
                str(tmp_path / "v7"),
                "--skip_dedup",
                "--skip_integrity",
                "--skip_s3",
                "-v",
            ]
        )
        assert rc == EXIT_OK
