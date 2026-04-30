"""Tests for the data audit script."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from training.data_audit import _classify_file, build_parser, run_audit


class TestClassifyFile:

    @pytest.mark.parametrize(
        "filename,expected",
        [
            ("stage1_foundation_counseling.jsonl", "general_counseling"),
            ("addiction_therapy.jsonl", "addiction"),
            ("narcissistic_abuse.jsonl", "personality_disorders"),
            ("cptsd_content.jsonl", "cptsd_trauma"),
            ("nightmare_fuel.jsonl", "crisis_edge_cases"),
            ("long_sessions.jsonl", "long_running_therapy"),
            ("voice_persona.jsonl", "voice_persona"),
            ("simulation.jsonl", "roleplay_simulation"),
            ("dpo_pairs.jsonl", "dpo_preference"),
            ("cot_reasoning.jsonl", "cot_reasoning"),
            ("therapeutic_expertise.jsonl", "therapeutic_expertise"),
            ("benchmark_core.jsonl", "safety_guardrails"),
            ("clinical_literature.jsonl", "clinical_literature"),
            ("youtube_transcripts.jsonl", "video_transcripts"),
            ("dissociation_data.jsonl", "dissociation"),
            ("somatic_therapy.jsonl", "somatic_therapy"),
            ("attachment_disorders.jsonl", "attachment_disorders"),
            ("grief_counseling.jsonl", "complicated_grief"),
            ("eating_disorders.jsonl", "eating_disorders"),
            ("ocd_intrusive.jsonl", "ocd_intrusive_thoughts"),
            ("neurodivergent.jsonl", "neurodivergent"),
            ("cultural_religious.jsonl", "cultural_religious"),
            ("random_data.jsonl", "uncategorized"),
        ],
    )
    def test_classification(self, filename: str, expected: str):
        assert _classify_file(filename) == expected


class TestRunAudit:

    def test_empty_directory(self, tmp_path: Path):
        input_dir = tmp_path / "empty"
        input_dir.mkdir()
        output = tmp_path / "report.json"

        args = build_parser().parse_args([
            "--input_dirs", str(input_dir),
            "--output", str(output),
        ])
        run_audit(args)

        report = json.loads(output.read_text(encoding="utf-8"))
        assert report["total_samples"] == 0
        assert report["total_files_scanned"] == 0

    def test_known_dataset_counts(self, tmp_path: Path):
        input_dir = tmp_path / "data"
        input_dir.mkdir()

        (input_dir / "stage1_foundation.jsonl").write_text(
            "\n".join(["{}"] * 50),
            encoding="utf-8",
        )
        (input_dir / "nightmare_fuel.jsonl").write_text(
            "\n".join(["{}"] * 100),
            encoding="utf-8",
        )

        output = tmp_path / "report.json"
        args = build_parser().parse_args([
            "--input_dirs", str(input_dir),
            "--output", str(output),
        ])
        run_audit(args)

        report = json.loads(output.read_text(encoding="utf-8"))
        assert report["categories"]["general_counseling"]["sample_count"] == 50
        assert report["categories"]["crisis_edge_cases"]["sample_count"] == 100

    def test_threshold_flagging(self, tmp_path: Path):
        input_dir = tmp_path / "data"
        input_dir.mkdir()

        (input_dir / "addiction_small.jsonl").write_text(
            "\n".join(["{}"] * 10),
            encoding="utf-8",
        )

        output = tmp_path / "report.json"
        args = build_parser().parse_args([
            "--input_dirs", str(input_dir),
            "--output", str(output),
            "--threshold", "500",
        ])
        run_audit(args)

        report = json.loads(output.read_text(encoding="utf-8"))
        addiction = report["categories"]["addiction"]
        assert addiction["status"] == "partial"
        assert addiction["sample_count"] == 10
        assert addiction["threshold"] == 500

    def test_missing_directory_logged(self, tmp_path: Path):
        output = tmp_path / "report.json"
        args = build_parser().parse_args([
            "--input_dirs", str(tmp_path / "nonexistent"),
            "--output", str(output),
        ])
        run_audit(args)

        report = json.loads(output.read_text(encoding="utf-8"))
        assert report["total_samples"] == 0

    def test_custom_threshold(self, tmp_path: Path):
        input_dir = tmp_path / "data"
        input_dir.mkdir()

        (input_dir / "voice_persona.jsonl").write_text(
            "\n".join(["{}"] * 600),
            encoding="utf-8",
        )

        output = tmp_path / "report.json"
        args = build_parser().parse_args([
            "--input_dirs", str(input_dir),
            "--output", str(output),
            "--threshold", "500",
        ])
        run_audit(args)

        report = json.loads(output.read_text(encoding="utf-8"))
        assert report["categories"]["voice_persona"]["status"] == "covered"
        assert report["threshold"] == 500
