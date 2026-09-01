"""Tests for the 10-family edge-case matrix + Moderate-guard wiring in
build_edge_and_nightmare_dataset.py (plan step 7)."""

from __future__ import annotations

import ast
import io
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from training import generation_backend as gb
from training.build_edge_and_nightmare_dataset import (
    AMBIGUITY_TYPES,
    DIFFICULTY_LEVELS,
    EDGE_CASE_DOMAINS,
    _parse_args,
    _process_record,
    _variations_per_combo,
    build_edge_case_matrix,
)

_DESIGNER_EDGE_CASES = (
    Path(__file__).resolve().parents[3]
    / "scripts"
    / "data"
    / "designer"
    / "configs"
    / "edge_cases.py"
)


def _designer_families() -> list[str]:
    """Extract the authoritative edge_family list from the Data Designer config."""
    tree = ast.parse(_DESIGNER_EDGE_CASES.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if not isinstance(node, ast.Dict):
            continue
        for key, value in zip(node.keys, node.values):
            if (
                isinstance(key, ast.Constant)
                and key.value == "edge_family"
                and isinstance(value, ast.List)
            ):
                return [elt.value for elt in value.elts if isinstance(elt, ast.Constant)]
    raise AssertionError("edge_family list not found in Data Designer config")


class TestTaxonomyMatchesDesigner:
    def test_ten_families_match_data_designer(self):
        designer = _designer_families()
        build = [d["family"] for d in EDGE_CASE_DOMAINS]
        assert set(build) == set(designer)
        assert len(build) == 10
        assert len(set(build)) == 10

    def test_matrix_is_full_cartesian_product(self):
        matrix = build_edge_case_matrix()
        assert len(matrix) == len(EDGE_CASE_DOMAINS) * len(DIFFICULTY_LEVELS) * len(AMBIGUITY_TYPES)
        assert len(matrix) == 120
        combo = matrix[0]
        assert {"family", "domain", "description", "difficulty", "ambiguity"} <= set(combo)


class TestVariationsPerCombo:
    def test_default_is_one(self):
        assert _variations_per_combo(_parse_args([]), 120) == 1

    def test_target_derives_variations(self):
        assert _variations_per_combo(_parse_args(["--target", "50000"]), 120) == 417

    def test_explicit_variations(self):
        assert _variations_per_combo(_parse_args(["--variations-per-combo", "5"]), 120) == 5

    def test_target_floor_is_one(self):
        assert _variations_per_combo(_parse_args(["--target", "3"]), 120) == 1


class TestProcessRecord:
    def _guard(self):
        guard = MagicMock(spec=gb.ModerateGuard)
        guard.record = MagicMock()
        return guard

    def test_none_skips(self):
        guard = self._guard()
        fout = io.StringIO()
        generated, rejected = _process_record(None, guard, fout, 0, 0)
        assert (generated, rejected) == (0, 0)
        assert guard.record.call_count == 0
        assert fout.getvalue() == ""

    def test_valid_record_counted_and_written(self):
        guard = self._guard()
        fout = io.StringIO()
        rec = {"family": "substance use", "messages": [{"role": "user", "content": "x"}]}
        generated, rejected = _process_record(rec, guard, fout, 0, 0)
        assert (generated, rejected) == (1, 0)
        guard.record.assert_called_once_with()
        assert fout.getvalue().strip()

    def test_cliche_record_counted_but_not_written(self):
        guard = self._guard()
        fout = io.StringIO()
        rec = {
            "family": "ambiguous crisis language",
            "messages": [
                {"role": "assistant", "content": "It sounds like you're hurting yourself."},
            ],
        }
        generated, rejected = _process_record(rec, guard, fout, 0, 0)
        assert (generated, rejected) == (0, 1)
        guard.record.assert_called_once_with()
        assert fout.getvalue() == ""