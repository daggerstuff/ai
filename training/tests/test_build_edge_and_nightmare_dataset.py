"""Tests for the 10-family edge-case matrix + Moderate-guard wiring in
build_edge_and_nightmare_dataset.py (plan step 7)."""

from __future__ import annotations

import ast
import io
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from training import build_edge_and_nightmare_dataset as bmod, dual_judge, generation_backend as gb
from training.build_edge_and_nightmare_dataset import (
    AMBIGUITY_TYPES,
    DIFFICULTY_LEVELS,
    EDGE_CASE_DOMAINS,
    _generate_transcript_turns,
    _parse_args,
    _process_record,
    _render_transcript,
    _roles_alternate,
    _strip_utterance,
    _variations_per_combo,
    build_edge_case_matrix,
)

_DESIGNER_EDGE_CASES = (
    Path(__file__).resolve().parents[3] / "scripts" / "data" / "designer" / "configs" / "edge_cases.py"
)

_NUM_EDGE_FAMILIES = len(EDGE_CASE_DOMAINS)
_MATRIX_SIZE = _NUM_EDGE_FAMILIES * len(DIFFICULTY_LEVELS) * len(AMBIGUITY_TYPES)


def _designer_families() -> list[str]:
    """Extract the authoritative edge_family list from the Data Designer config."""
    tree = ast.parse(_DESIGNER_EDGE_CASES.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if not isinstance(node, ast.Dict):
            continue
        for key, value in zip(node.keys, node.values, strict=False):
            if isinstance(key, ast.Constant) and key.value == "edge_family" and isinstance(value, ast.List):
                return [elt.value for elt in value.elts if isinstance(elt, ast.Constant)]
    raise AssertionError("edge_family list not found in Data Designer config")


class TestTaxonomyMatchesDesigner:
    def test_ten_families_match_data_designer(self):
        designer = _designer_families()
        build = [d["family"] for d in EDGE_CASE_DOMAINS]
        assert set(build) == set(designer)
        assert len(build) == _NUM_EDGE_FAMILIES
        assert len(set(build)) == _NUM_EDGE_FAMILIES

    def test_matrix_is_full_cartesian_product(self):
        matrix = build_edge_case_matrix()
        assert len(matrix) == _MATRIX_SIZE
        combo = matrix[0]
        assert {"family", "domain", "description", "difficulty", "ambiguity"} <= set(combo)


class TestVariationsPerCombo:
    def test_default_is_one(self):
        assert _variations_per_combo(_parse_args([]), _MATRIX_SIZE) == 1

    def test_target_derives_variations(self):
        target = 50000
        assert (
            _variations_per_combo(_parse_args(["--target", str(target)]), _MATRIX_SIZE)
            == (target + _MATRIX_SIZE - 1) // _MATRIX_SIZE
        )

    def test_explicit_variations(self):
        explicit = 5
        assert _variations_per_combo(_parse_args(["--variations-per-combo", str(explicit)]), _MATRIX_SIZE) == explicit

    def test_target_floor_is_one(self):
        assert _variations_per_combo(_parse_args(["--target", "3"]), _MATRIX_SIZE) == 1


class TestProcessRecord:
    def _guard(self):
        guard = MagicMock(spec=gb.ModerateGuard)
        guard.record = MagicMock()
        return guard

    def _verdict(self, accepted: bool, reason: str = ""):
        verdict = MagicMock()
        verdict.accepted = accepted
        verdict.primary.reject_reason = ""
        verdict.reason = reason
        return verdict

    @pytest.mark.asyncio
    async def test_none_skips(self, monkeypatch):
        guard = self._guard()
        fout = io.StringIO()
        session = MagicMock()

        async def _judge(_rec, **_kwargs):
            return self._verdict(True)

        monkeypatch.setattr(dual_judge, "judge_record_turns", _judge)
        dg, dr = await _process_record(None, guard, fout, session)
        assert (dg, dr) == (0, 0)
        assert guard.record.call_count == 0
        assert fout.getvalue() == ""

    @pytest.mark.asyncio
    async def test_valid_record_counted_and_written(self, monkeypatch, tmp_path):
        guard = self._guard()
        fout_path = tmp_path / "out.jsonl"
        fout = fout_path.open("a", encoding="utf-8")
        session = MagicMock()

        async def _judge(_rec, **_kwargs):
            return self._verdict(True)

        monkeypatch.setattr(dual_judge, "judge_record_turns", _judge)
        rec = {"family": "substance use", "messages": [{"role": "user", "content": "x"}]}
        dg, dr = await _process_record(rec, guard, fout, session)
        fout.close()
        assert (dg, dr) == (1, 0)
        guard.record.assert_called_once_with()
        assert fout_path.read_text(encoding="utf-8").strip()

    @pytest.mark.asyncio
    async def test_rejected_record_counted_but_not_written(self, monkeypatch):
        guard = self._guard()
        fout = io.StringIO()
        session = MagicMock()

        async def _judge(_rec, **_kwargs):
            return self._verdict(False, "banned_sycophantic_opener")

        monkeypatch.setattr(dual_judge, "judge_record_turns", _judge)
        rec = {
            "family": "ambiguous crisis language",
            "messages": [
                {"role": "assistant", "content": "It sounds like you're hurting yourself."},
            ],
        }
        dg, dr = await _process_record(rec, guard, fout, session)
        assert (dg, dr) == (0, 1)
        guard.record.assert_called_once_with()
        assert fout.getvalue() == ""


class TestRolesAlternate:
    def test_alternating_passes(self):
        msgs = [
            {"role": "user", "content": "a"},
            {"role": "assistant", "content": "b"},
            {"role": "user", "content": "c"},
            {"role": "assistant", "content": "d"},
        ]
        assert _roles_alternate(msgs)

    def test_must_start_with_user(self):
        msgs = [{"role": "assistant", "content": "b"}, {"role": "user", "content": "a"}]
        assert not _roles_alternate(msgs)

    def test_consecutive_same_role_fails(self):
        msgs = [
            {"role": "user", "content": "a"},
            {"role": "assistant", "content": "b"},
            {"role": "assistant", "content": "c"},
        ]
        assert not _roles_alternate(msgs)

    def test_empty_passes(self):
        assert _roles_alternate([])


class TestStripUtterance:
    def test_strips_role_label(self):
        assert _strip_utterance("Therapist: You said 'no reason to keep going.'") == (
            "You said 'no reason to keep going.'"
        )

    def test_strips_surrounding_quotes(self):
        assert _strip_utterance('"I don\'t know."') == "I don't know."

    def test_passthrough_plain_line(self):
        assert _strip_utterance("Just a plain line.") == "Just a plain line."


class TestRenderTranscript:
    def test_renders_speaker_lines(self):
        msgs = [
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "hi"},
        ]
        assert _render_transcript(msgs) == "Client: hello\nTherapist: hi"


class TestGenerateTranscriptTurns:
    @pytest.mark.asyncio
    async def test_alternates_then_stops_on_failed_turn(self, monkeypatch):
        replies = iter(["c1", "t1", "c2", "t2", "c3", "t3"])

        async def _fake_call(_session, _system_prompt, _user_prompt):
            return next(replies, "")

        monkeypatch.setattr(bmod, "_call_llm", _fake_call)
        msgs = await _generate_transcript_turns(None, "sys", "ctx", target=15)
        assert [m["role"] for m in msgs] == ["user", "assistant"] * 3
        assert [m["content"] for m in msgs] == ["c1", "t1", "c2", "t2", "c3", "t3"]
